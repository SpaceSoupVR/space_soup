pub use space_soup_sky::{
    decode_radiance, project_irradiance, sky_lighting, uv_to_direction, Panorama, SkyIrradiance,
    SkySun,
};
pub use space_soup_sky::time_of_day::{AtlasLayers, LayerWeights, SkySnapshot, TimeOfDayParams, TimeOfDaySky};

use wgpu::*;

use super::lights::wgsl_lights_block;

pub const AMBIENT: f32 = 0.6;

pub fn direction_to_uv(d: [f32; 3]) -> [f32; 2] {
    let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt().max(1e-6);
    let (x, y, z) = (d[0] / len, d[1] / len, d[2] / len);
    let u = (x.atan2(z) + std::f32::consts::PI) / (2.0 * std::f32::consts::PI);
    let v = y.clamp(-1.0, 1.0).acos() / std::f32::consts::PI;
    [u, v]
}

pub fn f32_to_f16(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let mut exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mant = bits & 0x7f_ffff;
    if exp >= 0x1f {
        return sign | 0x7c00;
    }
    if exp <= 0 {
        return sign;
    }
    if exp > 0x1e {
        exp = 0x1e;
    }
    sign | ((exp as u16) << 10) | ((mant >> 13) as u16)
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SkyExtra {

    pub mode: [f32; 4],

    pub sun: [f32; 4],

    pub sun_radiance: [f32; 4],

    pub moon: [f32; 4],

    pub moon_radiance: [f32; 4],

    pub moon_sunward: [f32; 4],

    pub stars: [f32; 4],

    pub extinction: [f32; 4],

    pub celestial: [[f32; 4]; 3],

    pub cloud_scale: [f32; 4],

    pub cloud_offset: [f32; 4],

    pub cloud_misc: [f32; 4],

    pub cloud_light_dir: [f32; 4],

    pub cloud_light: [f32; 4],

    pub cloud_sky: [f32; 4],

    pub cloud_ground: [f32; 4],
}

impl SkyExtra {
    pub fn set_clouds(&mut self, f: Option<&CloudFrame>) {
        let Some(f) = f else {
            self.cloud_scale = [0.0; 4];
            return;
        };
        let p = &f.params;
        self.cloud_scale = [f.tile_inv[0], f.tile_inv[1], f.tile_inv[2], p.coverage];
        self.set_cloud_offsets(f);
        self.cloud_misc[2] = f.sigma_column;
        self.cloud_misc[3] = p.altitude_m + 0.5 * p.thickness_m;
        self.cloud_light_dir = [f.light_dir[0], f.light_dir[1], f.light_dir[2], p.thickness_m];
        let v4 = |v: [f32; 3]| [v[0], v[1], v[2], 0.0];
        self.cloud_light = v4(f.light_irradiance);
        self.cloud_sky = v4(f.ambient_sky);
        self.cloud_ground = v4(f.ambient_ground);
    }

    pub fn set_cloud_offsets(&mut self, f: &CloudFrame) {
        let o = f.offsets;
        self.cloud_offset = [o[0][0], o[0][1], o[1][0], o[1][1]];
        self.cloud_misc[0] = o[2][0];
        self.cloud_misc[1] = o[2][1];
    }


    pub fn from_snapshot(snap: &SkySnapshot, seconds: f32, zenith_depth: [f32; 3]) -> Self {
        use space_soup_sky::time_of_day as tod;
        let d = &snap.discs;
        let v4 = |v: [f32; 3], w: f32| [v[0], v[1], v[2], w];
        // The moon's illuminance at the eye per unit of `moon_radiance` (the
        // full disc's): its solid angle times its phase's share (the halo).
        let moon_halo = std::f32::consts::PI * tod::MOON_ANGULAR_RADIUS.powi(2) * tod::moon_phase_brightness(snap.astro.moon_phase_angle);

        let mean_sky = 1.0 / snap.panorama_scale.max(1e-30);
        let px = (0.05f32).to_radians();
        let brightest = d.star_zero_point * 10f32.powf(0.6) / (2.0 * std::f32::consts::PI * (0.5 * px).powi(2));
        let stars_show = d.star_count > 0.0 && brightest > 1e-3 * mean_sky;
        Self {
            mode: [1.0, 1.0 / snap.panorama_scale.max(1e-30), seconds, if stars_show { 1.0 } else { 0.0 }],
            sun: v4(d.sun_dir, space_soup_sky::time_of_day::SUN_ANGULAR_RADIUS),
            sun_radiance: v4(d.sun_radiance, apparent_sine(d.sun_dir[1])),
            moon: v4(d.moon_dir, space_soup_sky::time_of_day::MOON_ANGULAR_RADIUS),
            moon_radiance: v4(d.moon_radiance, d.earthshine),
            moon_sunward: v4(d.moon_sunward, 0.0),
            stars: [d.star_zero_point, d.limiting_magnitude, d.star_count, 1.0],
            extinction: v4(zenith_depth, moon_halo),
            celestial: [v4(d.celestial_rows[0], 0.0), v4(d.celestial_rows[1], 0.0), v4(d.celestial_rows[2], 0.0)],
            ..Default::default()
        }
    }
}

/// Linear limb darkening of the sun, I(mu)/I(1) = 1 - u (1 - mu), per
/// channel: Neckel and Labs' u near 650, 550 and 450 nm. One grey u gave a
/// flat white disc; real suns are dimmer and redder at the edge.
pub const SUN_LIMB_U: [f32; 3] = [0.52, 0.62, 0.75];
/// The limb coefficient `TimeOfDaySky` normalises `sun_radiance` by.
pub const SUN_LIMB_BAKED: f32 = 0.6;
/// How far from a sun or moon its glare reaches, degrees.
pub const HALO_DEGREES: f32 = 20.0;
/// How far outside its true radius a disc is looked for: refraction lifts a
/// disc on the horizon by up to 35' (0.6 deg).
pub const DISC_REACH: f32 = 0.012;

/// The apparent elevation's sine for a true one (`sky_refraction`'s twin):
/// Bennett's refraction for the true altitude, which is close enough to
/// where the disc is drawn to anchor its reddening.
pub fn apparent_sine(true_sine: f32) -> f32 {
    let el = true_sine.clamp(-1.0, 1.0).asin();
    let h = el.to_degrees().max(-1.5);
    let r = (1.0 / 60.0f32).to_radians() / (h + 7.31 / (h + 4.4)).to_radians().tan();
    (el + r).sin()
}

use space_soup_sky::clouds::{CloudField, CloudFrame};

pub struct SkyClouds {
    pub field: std::sync::Arc<CloudField>,
    pub frame: CloudFrame,
    pub sun_clear: Option<SkySun>,
}

/// Metres a side of the clouds' shadow map round the eye: 16 m texels at
/// `CLOUD_SHADOW_SIZE` 64, finer than a cloud's soft edge.
pub const CLOUD_SHADOW_EXTENT_M: f32 = 1024.0;
/// The map's centre moves in steps this long, so it need not be rebuilt
/// every metre walked.
pub const CLOUD_SHADOW_SNAP_M: f32 = 128.0;
/// Rows built a frame: the whole map every 8 frames (~0.1 s at 72 Hz),
/// 512 cloud lookups a frame instead of 4096 at once.
pub const CLOUD_SHADOW_ROWS_PER_FRAME: u32 = 8;
/// The share of the sun's flux on level ground a cloud that hides it whole
/// scatters down from its base. Under a cumulus the ground takes some
/// 25-35% of the clear sky's global light (direct gone, diffuse up); the
/// clear sky's own share is already in the ambient, and this lifts the shade
/// from 1/5.2 of full sun (sky alone, measured at 14 h) to about 1/3.6
/// (0.12 gave 1/4.2: the fill is occluded as the sky is).
pub const CLOUD_FILL: f32 = 0.16;

/// `Uniforms::cloud_map` for a map centred on `centre` (world x, z) under a
/// clear light `sun`, read from player space: world = R(yaw) p + offset,
/// R taking (x, z) to (c x + s z, -s x + c z).
pub fn cloud_map_for(centre: [f32; 2], sun: &SkySun, yaw: f32, offset: [f32; 3]) -> [[f32; 4]; 3] {
    let e = 1.0 / CLOUD_SHADOW_EXTENT_M;
    let h = 0.5 * CLOUD_SHADOW_EXTENT_M;
    let (s, c) = yaw.sin_cos();
    let o = [(offset[0] - (centre[0] - h)) * e, (offset[2] - (centre[1] - h)) * e];
    let d = sun.direction;
    let up = d[1].max(0.0) / (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt().max(1e-6);
    let f = sun.light_rgb.map(|v| v * up * CLOUD_FILL);
    [[c * e, s * e, -s * e, c * e], [o[0], o[1], 0.0, 0.0], [f[0], f[1], f[2], 1.0]]
}

/// The map's outer ring set to its average, so ground past the map (which
/// the clamped sampler reads at the edge) takes the average light, never
/// one row's patch smeared to the horizon.
pub fn cloud_map_border(texels: &mut [u8], size: u32) -> f32 {
    let mean = texels.iter().map(|&v| v as f32).sum::<f32>() / (255.0 * texels.len().max(1) as f32);
    let m = (mean * 255.0).round() as u8;
    for j in 0..size {
        for i in 0..size {
            if i == 0 || j == 0 || i + 1 == size || j + 1 == size {
                texels[(j * size + i) as usize] = m;
            }
        }
    }
    mean
}

/// The clouds' shadow on the ground, built a slice of rows a frame
/// (`Sky::step_cloud_shadow`) and shown when whole.
#[derive(Default)]
pub struct CloudShadow {
    shown: Vec<u8>,
    building: Vec<u8>,
    shown_centre: Option<[f32; 2]>,
    shown_mean: f32,
    build_centre: Option<[f32; 2]>,
    next_row: u32,
    /// The shown map changed since `take_cloud_shadow_upload`.
    dirty: bool,
    /// Read this frame: the clouds are up, the sun is above them, and the
    /// lever is on.
    active: bool,
}

pub struct Sky {
    pub bind_group: BindGroup,

    pub irradiance: SkyIrradiance,

    pub sun: Option<SkySun>,

    pub reflection: Option<ReflectionSky>,

    pub extra: SkyExtra,
    extra_buffer: Buffer,
    texture: Texture,
    sampler: Sampler,

    pub clouds: Option<SkyClouds>,
    pub cloud_shadow: CloudShadow,
    cloud_texture: Texture,
    cloud_sampler: Sampler,

    /// The time of day's way from the last snapshot to the newest: drawn
    /// and lit between them, frame by frame (`Sky::tick`), so the sky's
    /// colour and the light it gives never step at a snapshot.
    blend: Option<SkyBlend>,
}

/// Two time-of-day snapshots and how far the sky is from one to the other.
/// Panoramas in engine units (the snapshot's normalised panorama over its
/// scale), so two of very different brightness mix in the light they hold.
struct SkyBlend {
    width: u32,
    height: u32,
    from_pano: Vec<f32>,
    to_pano: Vec<f32>,
    from_scale: f32,
    to_scale: f32,
    from_irradiance: SkyIrradiance,
    to_irradiance: SkyIrradiance,
    from_sun: Option<[f32; 3]>,
    to_sun: Option<SkySun>,
    /// 0 at the last snapshot, 1 at the newest.
    t: f32,
    /// Real seconds the blend takes: the clock's time between snapshots.
    seconds: f32,
}

impl SkyBlend {
    fn scale(&self, t: f32) -> f32 {
        (self.from_scale.max(1e-30).ln() * (1.0 - t) + self.to_scale.max(1e-30).ln() * t).exp()
    }

    /// The panorama at `t`, normalised by the blended scale as a snapshot's is.
    fn panorama(&self, t: f32) -> Panorama {
        let s = self.scale(t);
        let rgb = self.from_pano.iter().zip(&self.to_pano).map(|(a, b)| (a + (b - a) * t) * s).collect();
        Panorama { width: self.width, height: self.height, rgb }
    }

    fn irradiance(&self, t: f32) -> SkyIrradiance {
        let (a, b) = (&self.from_irradiance.sh, &self.to_irradiance.sh);
        SkyIrradiance { sh: std::array::from_fn(|i| std::array::from_fn(|c| a[i][c] + (b[i][c] - a[i][c]) * t)) }
    }

    /// The light (clear of clouds) at `t`: the newest snapshot's direction,
    /// the colour between the two. The direction steps with the snapshots, as
    /// the static sun shadow re-records only then.
    fn sun(&self, t: f32) -> Option<SkySun> {
        let to = self.to_sun?;
        let from = self.from_sun.unwrap_or([0.0; 3]);
        Some(SkySun { light_rgb: std::array::from_fn(|c| from[c] + (to.light_rgb[c] - from[c]) * t), ..to })
    }
}

#[derive(Clone)]
pub struct ReflectionSky {
    pub pano: Panorama,
    pub rotation_deg: f32,
    pub intensity: f32,
}

impl ReflectionSky {

    pub fn new(pano: &Panorama, rotation_deg: f32, intensity: f32) -> Self {
        let pano = match space_soup_sky::extract_sun(pano, rotation_deg, intensity) {
            Some((_, rest)) => rest,
            None => pano.clone(),
        };
        Self { pano, rotation_deg, intensity }
    }

    pub fn radiance(&self, d: glam::Vec3) -> [f32; 3] {
        let (rc, rs) = (self.rotation_deg.to_radians().cos(), self.rotation_deg.to_radians().sin());

        let d = d.normalize_or_zero();
        let p = [rc * d.x - rs * d.z, d.y, rs * d.x + rc * d.z];
        let [u, v] = direction_to_uv(p);
        let (w, h) = (self.pano.width, self.pano.height);
        let fx = u * w as f32 - 0.5;
        let fy = (v * h as f32 - 0.5).clamp(0.0, (h - 1) as f32);
        let x0 = fx.floor();
        let tx = fx - x0;
        let y0 = fy.floor();
        let ty = fy - y0;
        let wrap = |x: f32| (x.rem_euclid(w as f32) as u32).min(w - 1);
        let (xa, xb) = (wrap(x0), wrap(x0 + 1.0));
        let (ya, yb) = (y0 as u32, (y0 as u32 + 1).min(h - 1));
        let mut out = [0.0f32; 3];
        let (a, b, c, e) = (self.pano.texel(xa, ya), self.pano.texel(xb, ya), self.pano.texel(xa, yb), self.pano.texel(xb, yb));
        for k in 0..3 {
            let top = a[k] + (b[k] - a[k]) * tx;
            let bottom = c[k] + (e[k] - c[k]) * tx;
            out[k] = (top + (bottom - top) * ty) * self.intensity;
        }
        out
    }

    pub fn cube_faces(&self, res: u32) -> Vec<u8> {
        let res = res.max(1);
        let mut out = Vec::with_capacity((res * res * 6 * 8) as usize);
        for face in 0..6usize {
            for y in 0..res {
                for x in 0..res {
                    let mut acc = [0.0f32; 3];
                    for (sx, sy) in [(0.25f32, 0.25f32), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)] {
                        let d = crate::renderer::probe_prefilter::texel_direction(
                            face,
                            (x as f32 + sx) / res as f32,
                            (y as f32 + sy) / res as f32,
                        );
                        let r = self.radiance(d);
                        for k in 0..3 {
                            acc[k] += r[k] * 0.25;
                        }
                    }
                    for v in [acc[0], acc[1], acc[2], 1.0] {
                        out.extend_from_slice(&f32_to_f16(v).to_le_bytes());
                    }
                }
            }
        }
        out
    }
}

pub fn sky_bind_group_layout(device: &Device) -> BindGroupLayout {
    device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("sky_bgl"),
        entries: &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Sampler(SamplerBindingType::Filtering),
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 3,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 4,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Sampler(SamplerBindingType::Filtering),
                count: None,
            },
        ],
    })
}

fn cloud_texture(device: &Device, queue: &Queue, field: Option<&CloudField>) -> Texture {
    let mips = field.map(|f| f.mips()).unwrap_or_else(|| vec![(1, vec![0, 0, 0, 255])]);
    let size = mips[0].0;
    let texture = device.create_texture(&TextureDescriptor {
        label: Some("sky_clouds"),
        size: Extent3d { width: size, height: size, depth_or_array_layers: 1 },
        mip_level_count: mips.len() as u32,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
        view_formats: &[],
    });
    for (level, (s, bytes)) in mips.iter().enumerate() {
        queue.write_texture(
            TexelCopyTextureInfo { texture: &texture, mip_level: level as u32, origin: Origin3d::ZERO, aspect: TextureAspect::All },
            bytes,
            TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(s * 4), rows_per_image: Some(*s) },
            Extent3d { width: *s, height: *s, depth_or_array_layers: 1 },
        );
    }
    texture
}

impl Sky {
    pub fn new(
        device: &Device,
        queue: &Queue,
        layout: &BindGroupLayout,
        pano: &Panorama,
        rotation_deg: f32,
        intensity: f32,
    ) -> Self {
        let texture = device.create_texture(&TextureDescriptor {
            label: Some("sky_panorama"),
            size: Extent3d {
                width: pano.width,
                height: pano.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });

        write_panorama(queue, &texture, pano);

        let sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("sky_sampler"),

            address_mode_u: AddressMode::Repeat,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            ..Default::default()
        });

        let light_pano = pano.for_lighting(1024);
        let (irradiance, sun, without_sun) = match space_soup_sky::extract_sun(&light_pano, rotation_deg, intensity) {
            Some((sun, rest)) => (project_irradiance(&rest, rotation_deg, intensity), Some(sun), rest),
            None => (project_irradiance(&light_pano, rotation_deg, intensity), None, light_pano.clone()),
        };
        let extra = SkyExtra::default();
        let extra_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("sky_extra"),
            size: std::mem::size_of::<SkyExtra>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&extra_buffer, 0, bytemuck::bytes_of(&extra));
        let cloud_texture = cloud_texture(device, queue, None);
        let cloud_sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("sky_cloud_sampler"),
            address_mode_u: AddressMode::Repeat,
            address_mode_v: AddressMode::Repeat,
            address_mode_w: AddressMode::Repeat,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: MipmapFilterMode::Linear,
            ..Default::default()
        });
        let bind_group = Self::bind(device, layout, &texture, &sampler, &extra_buffer, &cloud_texture, &cloud_sampler);

        Self {
            bind_group,
            irradiance,
            sun,
            reflection: Some(ReflectionSky { pano: without_sun, rotation_deg, intensity }),
            extra,
            extra_buffer,
            texture,
            sampler,
            clouds: None,
            cloud_shadow: CloudShadow::default(),
            cloud_texture,
            cloud_sampler,
            blend: None,
        }
    }

    /// How far the sky is through its blend to the newest snapshot, 0..1.
    pub fn blend_progress(&self) -> f32 {
        self.blend.as_ref().map_or(1.0, |b| b.t)
    }

    pub fn has_clouds(&self) -> bool {
        self.clouds.is_some()
    }

    fn bind(device: &Device, layout: &BindGroupLayout, texture: &Texture, sampler: &Sampler, extra: &Buffer, clouds: &Texture, cloud_sampler: &Sampler) -> BindGroup {
        let view = texture.create_view(&TextureViewDescriptor::default());
        let cloud_view = clouds.create_view(&TextureViewDescriptor::default());
        device.create_bind_group(&BindGroupDescriptor {
            label: Some("sky_bg"),
            layout,
            entries: &[
                BindGroupEntry { binding: 0, resource: BindingResource::TextureView(&view) },
                BindGroupEntry { binding: 1, resource: BindingResource::Sampler(sampler) },
                BindGroupEntry { binding: 2, resource: extra.as_entire_binding() },
                BindGroupEntry { binding: 3, resource: BindingResource::TextureView(&cloud_view) },
                BindGroupEntry { binding: 4, resource: BindingResource::Sampler(cloud_sampler) },
            ],
        })
    }

    /// The newest time-of-day snapshot, blended into from where the sky is
    /// now over `blend_seconds` (`Sky::tick`); 0 shows it at once.
    pub fn apply_snapshot(&mut self, device: &Device, queue: &Queue, layout: &BindGroupLayout, snap: &SkySnapshot, seconds: f32, zenith_depth: [f32; 3]) {
        self.apply_snapshot_blended(device, queue, layout, snap, seconds, zenith_depth, 0.0);
    }

    pub fn apply_snapshot_blended(
        &mut self,
        device: &Device,
        queue: &Queue,
        layout: &BindGroupLayout,
        snap: &SkySnapshot,
        seconds: f32,
        zenith_depth: [f32; 3],
        blend_seconds: f32,
    ) {
        let engine = |p: &Panorama, scale: f32| p.rgb.iter().map(|v| v / scale.max(1e-30)).collect::<Vec<f32>>();
        let to_pano = engine(&snap.panorama, snap.panorama_scale);
        let from = self.blend.as_ref().filter(|b| b.width == snap.panorama.width && b.height == snap.panorama.height && blend_seconds > 0.0).map(|b| {
            let t = b.t.min(1.0);
            let s = b.scale(t);
            (engine(&b.panorama(t), s), s, b.irradiance(t), b.sun(t).map(|l| l.light_rgb))
        });
        let (from_pano, from_scale, from_irradiance, from_sun) = match from {
            Some(f) => f,
            None => (to_pano.clone(), snap.panorama_scale, snap.irradiance, snap.light_clear.map(|l| l.light_rgb)),
        };
        self.blend = Some(SkyBlend {
            width: snap.panorama.width,
            height: snap.panorama.height,
            from_pano,
            to_pano,
            from_scale,
            to_scale: snap.panorama_scale,
            from_irradiance,
            to_irradiance: snap.irradiance,
            from_sun,
            to_sun: snap.light_clear,
            t: if blend_seconds > 0.0 { 0.0 } else { 1.0 },
            seconds: blend_seconds,
        });
        let shown = self.blend.as_ref().map(|b| b.panorama(b.t)).unwrap_or_else(|| snap.panorama.clone());
        let pano = &shown;
        let size = self.texture.size();
        if size.width != pano.width || size.height != pano.height {
            self.texture = device.create_texture(&TextureDescriptor {
                label: Some("sky_panorama"),
                size: Extent3d { width: pano.width, height: pano.height, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba16Float,
                usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
                view_formats: &[],
            });
            self.bind_group = Self::bind(device, layout, &self.texture, &self.sampler, &self.extra_buffer, &self.cloud_texture, &self.cloud_sampler);
        }
        write_panorama(queue, &self.texture, pano);
        let new_field = match (&snap.cloud_field, &self.clouds) {
            (Some(f), Some(c)) => !std::sync::Arc::ptr_eq(f, &c.field),
            (Some(_), None) => true,
            _ => false,
        };
        if new_field {
            self.cloud_texture = cloud_texture(device, queue, snap.cloud_field.as_deref());
            self.bind_group = Self::bind(device, layout, &self.texture, &self.sampler, &self.extra_buffer, &self.cloud_texture, &self.cloud_sampler);
        }
        self.clouds = match (snap.clouds, &snap.cloud_field) {
            (Some(frame), Some(field)) => Some(SkyClouds { field: field.clone(), frame: frame.at(seconds as f64), sun_clear: snap.light_clear }),
            _ => None,
        };
        self.sun = snap.light;
        self.reflection = Some(ReflectionSky { pano: snap.reflection_engine(), rotation_deg: 0.0, intensity: 1.0 });
        self.extra = SkyExtra::from_snapshot(snap, seconds, zenith_depth);
        self.extra.set_clouds(self.clouds.as_ref().map(|c| &c.frame));
        self.take_blend();
        self.dim_sun_by_clouds();
        queue.write_buffer(&self.extra_buffer, 0, bytemuck::bytes_of(&self.extra));
    }

    /// The light and the panorama scale where the blend stands now.
    fn take_blend(&mut self) {
        let Some(b) = self.blend.as_ref() else { return };
        let t = b.t.min(1.0);
        self.irradiance = b.irradiance(t);
        if let Some(c) = self.clouds.as_mut() {
            c.sun_clear = b.sun(t);
        }
        if self.clouds.is_none() {
            self.sun = b.sun(t).or(self.sun);
        }
        self.extra.mode[1] = 1.0 / b.scale(t).max(1e-30);
    }

    /// ONE FRAME of the time of day's sky: the clock (cloud drift, twinkle),
    /// the blend toward the newest snapshot, and -- when `astro` is given --
    /// the sun, moon and stars where they are THIS frame, so they glide
    /// instead of stepping at each snapshot (0.3 deg, ~7 headset pixels,
    /// once a second at a 20-minute day; user, 2026-10-20).
    pub fn tick(&mut self, queue: &Queue, seconds: f32, dt: f32, astro: Option<&space_soup_sky::time_of_day::Astronomy>) {
        if self.extra.mode[0] <= 0.5 {
            return;
        }
        self.extra.mode[2] = seconds;
        if let Some(c) = self.clouds.as_mut() {
            c.frame = c.frame.at(seconds as f64);
            self.extra.set_cloud_offsets(&c.frame);
        }
        let blending = self.blend.as_mut().is_some_and(|b| {
            let was = b.t;
            b.t = if b.seconds > 0.0 { (b.t + dt.max(0.0) / b.seconds).min(1.0) } else { 1.0 };
            b.t != was
        });
        if blending {
            if let Some(b) = self.blend.as_ref() {
                write_panorama(queue, &self.texture, &b.panorama(b.t));
            }
            self.take_blend();
        }
        if let Some(a) = astro {
            let v4 = |v: [f32; 3], w: f32| [v[0], v[1], v[2], w];
            self.extra.sun = v4(a.sun, self.extra.sun[3]);
            self.extra.moon = v4(a.moon, self.extra.moon[3]);
            self.extra.moon_sunward = v4(a.moon_sunward, 0.0);
            let m = &a.celestial_to_world;
            self.extra.celestial = [v4(m[0], 0.0), v4(m[1], 0.0), v4(m[2], 0.0)];
        }
        self.dim_sun_by_clouds();
        queue.write_buffer(&self.extra_buffer, 0, bytemuck::bytes_of(&self.extra));
    }

    fn dim_sun_by_clouds(&mut self) {
        let Some(c) = &self.clouds else { return };
        let through = c.frame.light_transmittance_over(&c.field, [0.0, 0.0], space_soup_sky::clouds::LEVEL_SUN_HALF_M, space_soup_sky::clouds::LEVEL_SUN_SAMPLES);
        self.sun = c.sun_clear.map(|l| SkySun { light_rgb: l.light_rgb.map(|v| v * through), ..l });
        // The glare round the sun and the moon goes as the clouds cover them
        // (`sky_halo_through` in the cloud shader).
        let toward = |v: [f32; 4]| c.frame.transmittance_along(&c.field, [0.0, 0.0], [v[0], v[1], v[2]]);
        self.extra.cloud_sky[3] = toward(self.extra.sun);
        self.extra.cloud_ground[3] = toward(self.extra.moon);
    }

    /// One frame of the clouds' shadow map round `eye_xz` (world x, z):
    /// the next `CLOUD_SHADOW_ROWS_PER_FRAME` rows toward the clear sun, the
    /// whole map at once when none is shown yet. `on` false (the lever, or
    /// no auto light) leaves the level's sun dimmed as a whole.
    pub fn step_cloud_shadow(&mut self, eye_xz: [f32; 2], on: bool) {
        let size = crate::renderer::uniforms::CLOUD_SHADOW_SIZE;
        let n = (size * size) as usize;
        let cs = &mut self.cloud_shadow;
        let light = self.clouds.as_ref().and_then(|c| c.sun_clear.map(|s| (c, s)));
        let Some((c, sun)) = light.filter(|(c, s)| on && c.frame.params.coverage > 0.0 && s.direction[1] > 0.0) else {
            cs.active = false;
            return;
        };
        let snap = |v: f32| (v / CLOUD_SHADOW_SNAP_M).round() * CLOUD_SHADOW_SNAP_M;
        let centre = [snap(eye_xz[0]), snap(eye_xz[1])];
        if cs.build_centre != Some(centre) || cs.building.len() != n {
            cs.build_centre = Some(centre);
            cs.building.resize(n, 255);
            cs.next_row = 0;
        }
        let rows = if cs.shown_centre.is_none() || !cs.active { size } else { CLOUD_SHADOW_ROWS_PER_FRAME };
        let end = (cs.next_row + rows).min(size);
        let d = sun.direction;
        let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt().max(1e-6);
        c.frame.shadow_rows(&c.field, centre, CLOUD_SHADOW_EXTENT_M, size, d.map(|v| v / len), cs.next_row..end, &mut cs.building);
        cs.next_row = end;
        if end >= size {
            std::mem::swap(&mut cs.shown, &mut cs.building);
            cs.shown_centre = Some(centre);
            cs.shown_mean = cloud_map_border(&mut cs.shown, size);
            cs.dirty = true;
            cs.next_row = 0;
            cs.building.resize(n, 255);
        }
        cs.active = cs.shown_centre.is_some();
    }

    /// The shown map's texels when they changed since the last call.
    pub fn take_cloud_shadow_upload(&mut self) -> Option<&[u8]> {
        let cs = &mut self.cloud_shadow;
        (cs.active && std::mem::take(&mut cs.dirty)).then_some(&cs.shown[..])
    }

    /// Where the shown map lies (`Uniforms::cloud_map`) for the player's
    /// `yaw` and `offset`; zeros when it is not read.
    pub fn cloud_map(&self, yaw: f32, offset: [f32; 3]) -> [[f32; 4]; 3] {
        let cs = &self.cloud_shadow;
        let sun = self.clouds.as_ref().and_then(|c| c.sun_clear);
        match (cs.shown_centre.filter(|_| cs.active), sun) {
            (Some(c), Some(sun)) => cloud_map_for(c, &sun, yaw, offset),
            _ => [[0.0; 4]; 3],
        }
    }

    /// The sun as the clouds over `xz` (world) let it through: what the
    /// falling weather round the head is lit by while the shadow map is
    /// read, so snow in a cloud's shadow does not glow in the level's sun.
    pub fn sun_at(&self, xz: [f32; 2]) -> Option<SkySun> {
        let clear = self.clouds.as_ref().filter(|_| self.cloud_shadow.active).and_then(|c| c.sun_clear.map(|s| (c, s)));
        let Some((c, s)) = clear else { return self.sun };
        let d = s.direction;
        let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt().max(1e-6);
        let t = c.frame.transmittance_along(&c.field, xz, d.map(|v| v / len));
        Some(SkySun { light_rgb: s.light_rgb.map(|v| v * t), ..s })
    }

    /// The sun the lit shaders take: the clear one while the shadow map
    /// dims it per pixel, else the one dimmed by the clouds over the level.
    pub fn sun_direct(&self) -> Option<SkySun> {
        if self.cloud_shadow.active {
            if let Some(c) = &self.clouds {
                return c.sun_clear;
            }
        }
        self.sun
    }

    pub fn none(device: &Device, queue: &Queue, layout: &BindGroupLayout, ambient: f32) -> Self {
        let mut s = Self::new(
            device,
            queue,
            layout,
            &Panorama::solid([0.0, 0.0, 0.0], 1, 1),
            0.0,
            1.0,
        );
        s.irradiance = SkyIrradiance::flat(ambient);
        s.sun = None;

        s.reflection = None;
        s
    }
}

fn write_panorama(queue: &Queue, texture: &Texture, pano: &Panorama) {
    let mut half = Vec::with_capacity((pano.width * pano.height * 4) as usize);
    for i in 0..(pano.width * pano.height) as usize {
        half.push(f32_to_f16(pano.rgb[i * 3]));
        half.push(f32_to_f16(pano.rgb[i * 3 + 1]));
        half.push(f32_to_f16(pano.rgb[i * 3 + 2]));
        half.push(f32_to_f16(1.0));
    }
    queue.write_texture(
        TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: Origin3d::ZERO,
            aspect: TextureAspect::All,
        },
        bytemuck::cast_slice(&half),
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(pano.width * 8),
            rows_per_image: Some(pano.height),
        },
        Extent3d {
            width: pano.width,
            height: pano.height,
            depth_or_array_layers: 1,
        },
    );
}

pub struct SkyPipeline {
    pub pipeline: RenderPipeline,

    pub clouded: RenderPipeline,
    pub layout: BindGroupLayout,
}

impl SkyPipeline {
    pub fn new(

        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        samples: u32,
    ) -> Self {
        Self::new_with_view(device, format, uniform_layout, samples, crate::renderer::multiview::ViewMode::Mono)
    }

    pub fn new_stereo(

        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        samples: u32,
    ) -> Self {
        Self::new_with_view(device, format, uniform_layout, samples, crate::renderer::multiview::ViewMode::Stereo)
    }

    fn new_with_view(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        samples: u32,
        view: crate::renderer::multiview::ViewMode,
    ) -> Self {
        let layout = sky_bind_group_layout(device);
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("sky_layout"),
            bind_group_layouts: &[Some(uniform_layout), Some(&layout)],
            immediate_size: 0,
        });

        let build = |label: &str, src: String| {
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some(label),
            source: ShaderSource::Wgsl(view.shader(src).into()),
        });
        device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(&pipeline_layout),
            vertex: VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: PipelineCompilationOptions::default(),

                buffers: &[],
            },
            fragment: Some(FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: PipelineCompilationOptions::default(),
                targets: &[Some(ColorTargetState {
                    format,
                    blend: None,
                    write_mask: ColorWrites::COLOR,
                })],
            }),
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                cull_mode: None,
                front_face: FrontFace::Ccw,
                polygon_mode: PolygonMode::Fill,
                ..Default::default()
            },
            depth_stencil: Some(DepthStencilState {
                format: TextureFormat::Depth32Float,

                depth_write_enabled: Some(false),
                depth_compare: Some(CompareFunction::LessEqual),
                stencil: StencilState::default(),
                bias: DepthBiasState::default(),
            }),
            multisample: MultisampleState { count: samples, ..Default::default() },
            multiview_mask: view.mask(),
            cache: None,
        })
        };
        let pipeline = build("sky_pipeline", sky_shader());
        let clouded = build("sky_clouds_pipeline", sky_cloud_shader());
        Self { pipeline, clouded, layout }
    }

    pub fn pick(&self, sky: &Sky) -> &RenderPipeline {
        if sky.has_clouds() { &self.clouded } else { &self.pipeline }
    }
}

pub const SKY_CRACK_DEBUG: bool = false;

fn sky_shader() -> String {
    format!(
        r#"
{lights_block}

@group(1) @binding(0) var sky_tex: texture_2d<f32>;
@group(1) @binding(1) var sky_samp: sampler;

struct SkyExtra {{
    mode: vec4<f32>,
    sun: vec4<f32>,
    sun_radiance: vec4<f32>,
    moon: vec4<f32>,
    moon_radiance: vec4<f32>,
    moon_sunward: vec4<f32>,
    stars: vec4<f32>,
    extinction: vec4<f32>,
    celestial: array<vec4<f32>, 3>,
}}
@group(1) @binding(2) var<uniform> sky_extra: SkyExtra;

struct VOut {{
    @builtin(position) clip: vec4<f32>,
    @location(0) ndc: vec2<f32>,
}}

@vertex fn vs_main(@builtin(vertex_index) vi: u32) -> VOut {{
    var p = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    var out: VOut;

    out.clip = vec4<f32>(p[vi], 1.0, 1.0);
    out.ndc = p[vi];
    return out;
}}

const SKY_PI: f32 = 3.14159265;

const STAR_CELLS: f32 = {star_cells:.1};

const MOON_SEEN: f32 = 2.0;

fn star_hash(x: u32, y: u32, z: u32) -> vec4<f32> {{
    var h = x * 1597334673u ^ y * 3812015801u ^ z * 2798796415u;
    var o = vec4<u32>(0u);
    for (var k = 0u; k < 4u; k = k + 1u) {{
        h = h ^ (h >> 16u);
        h = h * 2246822519u;
        h = h ^ (h >> 13u);
        h = h * 3266489917u;
        h = h ^ (h >> 16u);
        o[k] = h;
        h = h + 0x9e3779b9u;
    }}
    return vec4<f32>(o >> vec4<u32>(8u)) / 16777216.0;
}}

fn star_face_dir(face: u32, uv: vec2<f32>) -> vec3<f32> {{
    switch face {{
        case 0u: {{ return normalize(vec3<f32>(1.0, uv.x, uv.y)); }}
        case 1u: {{ return normalize(vec3<f32>(-1.0, uv.x, uv.y)); }}
        case 2u: {{ return normalize(vec3<f32>(uv.x, 1.0, uv.y)); }}
        case 3u: {{ return normalize(vec3<f32>(uv.x, -1.0, uv.y)); }}
        case 4u: {{ return normalize(vec3<f32>(uv.x, uv.y, 1.0)); }}
        default: {{ return normalize(vec3<f32>(uv.x, uv.y, -1.0)); }}
    }}
}}

fn sky_stars(d: vec3<f32>, px: f32) -> vec3<f32> {{
    let q = vec3<f32>(dot(sky_extra.celestial[0].xyz, d), dot(sky_extra.celestial[1].xyz, d), dot(sky_extra.celestial[2].xyz, d));
    let a = abs(q);
    var face = 0u;
    var uv = vec2<f32>(0.0);
    if (a.x >= a.y && a.x >= a.z) {{
        face = select(1u, 0u, q.x > 0.0);
        uv = q.yz / a.x;
    }} else if (a.y >= a.z) {{
        face = select(3u, 2u, q.y > 0.0);
        uv = q.xz / a.y;
    }} else {{
        face = select(5u, 4u, q.z > 0.0);
        uv = q.xy / a.z;
    }}
    let w = atan(uv) * (4.0 / SKY_PI);
    let f = (w * 0.5 + 0.5) * STAR_CELLS;
    let base = floor(f - 0.5);
    let sigma = max(0.5 * px, 1.5e-4);
    let airmass = 1.0 / (d.y + 0.025 * exp(-11.0 * d.y));
    let dimmed = exp(-sky_extra.extinction.xyz * airmass);
    let twinkle_depth = clamp((airmass - 2.5) / 10.0, 0.0, 0.6) * sky_extra.stars.w;
    let cell_angle = (SKY_PI * 0.5) / STAR_CELLS;
    var sum = vec3<f32>(0.0);
    for (var k = 0; k < 4; k = k + 1) {{
        let cell = base + vec2<f32>(f32(k & 1), f32(k >> 1u));
        if (any(cell < vec2<f32>(0.0)) || any(cell >= vec2<f32>(STAR_CELLS))) {{
            continue;
        }}
        let h = star_hash(u32(cell.x), u32(cell.y), face);

        let c = tan(((cell + 0.5) / STAR_CELLS * 2.0 - 1.0) * (SKY_PI * 0.25));
        let r2 = 1.0 + dot(c, c);
        let omega = cell_angle * cell_angle * (1.0 + c.x * c.x) * (1.0 + c.y * c.y) / (r2 * sqrt(r2));
        let p = sky_extra.stars.z * omega / (4.0 * SKY_PI);
        if (h.x >= p) {{
            continue;
        }}

        let m = max(sky_extra.stars.y + log2(max(h.x / p, 1e-6)) * (0.30103 / 0.47), -1.5);
        let at = star_face_dir(face, tan(((cell + h.yz) / STAR_CELLS * 2.0 - 1.0) * (SKY_PI * 0.25)));
        let off = length(cross(q, at));
        let flux = sky_extra.stars.x * exp2(-1.3287712 * m);
        let spread = flux / (2.0 * SKY_PI * sigma * sigma) * exp(-off * off / (2.0 * sigma * sigma));

        let t = h.w;
        let colour = mix(vec3<f32>(1.18, 0.95, 0.72), vec3<f32>(0.82, 0.92, 1.25), t * t * (3.0 - 2.0 * t));
        let tw = 1.0 + twinkle_depth * (0.6 * sin(sky_extra.mode.z * (11.0 + 7.0 * h.y) + 40.0 * h.z) + 0.4 * sin(sky_extra.mode.z * (23.0 + 9.0 * h.z) + 17.0 * h.y));
        sum = sum + spread * colour * tw;
    }}
    return sum * dimmed;
}}

fn moon_albedo(p: vec3<f32>) -> f32 {{
    var maria = array<vec4<f32>, 11>(
        vec4<f32>(33.0, -16.0, 11.0, 0.5),
        vec4<f32>(18.0, -57.0, 19.0, 0.45),
        vec4<f32>(28.0, 17.0, 7.0, 0.5),
        vec4<f32>(8.5, 31.0, 8.0, 0.5),
        vec4<f32>(17.0, 59.0, 5.0, 0.55),
        vec4<f32>(-8.0, 51.0, 6.5, 0.45),
        vec4<f32>(-21.0, -17.0, 7.0, 0.4),
        vec4<f32>(-24.0, -39.0, 4.0, 0.45),
        vec4<f32>(56.0, 1.0, 6.0, 0.35),
        vec4<f32>(-15.0, 35.0, 4.0, 0.45),
        vec4<f32>(0.0, -30.0, 9.0, 0.3),
    );
    var a = 1.0;
    for (var i = 0; i < 11; i = i + 1) {{
        let m = maria[i];
        let la = m.x * (SKY_PI / 180.0);
        let lo = m.y * (SKY_PI / 180.0);
        let c = vec3<f32>(cos(la) * sin(lo), sin(la), cos(la) * cos(lo));
        let ang = acos(clamp(dot(p, c), -1.0, 1.0)) * (180.0 / SKY_PI);
        a = a * (1.0 - m.w * (1.0 - smoothstep(m.z * 0.55, m.z, ang)));
    }}

    let ty = vec3<f32>(cos(-0.75) * sin(-0.19), sin(-0.75), cos(-0.75) * cos(-0.19));
    a = a + 0.25 * (1.0 - smoothstep(1.0, 4.0, acos(clamp(dot(p, ty), -1.0, 1.0)) * (180.0 / SKY_PI)));

    return a * 1.18;
}}

const SUN_LIMB_U: vec3<f32> = vec3<f32>({limb_r:?}, {limb_g:?}, {limb_b:?});
const SUN_LIMB_BAKED: f32 = {limb_baked:?};
const HALO_SHARE: f32 = {halo_share:?};
const HALO_DEGREES: f32 = {halo_degrees:?};
const HALO_COS: f32 = {halo_cos:?};
const HALO_CIE: vec3<f32> = vec3<f32>({cie_1:?}, {cie_2:?}, {cie_3:?});
const DISC_REACH: f32 = {disc_reach:?};

fn sky_cie(th2: f32) -> f32 {{
    let inv = inverseSqrt(max(th2, 1e-6));
    return inv * (HALO_CIE.x + inv * (HALO_CIE.y + inv * HALO_CIE.z));
}}

// The glare round a sun or moon: the eye's veil (CIE 146, the lamps' curve
// and share, `glare::cie_veil`) out to HALO_DEGREES, windowed to zero there.
// Per unit of the source's illuminance at the eye.
fn sky_halo(d: vec3<f32>, toward: vec3<f32>, core_deg: f32) -> f32 {{
    let c = dot(d, toward);
    if (c < HALO_COS) {{
        return 0.0;
    }}
    let th = acos(clamp(c, -1.0, 1.0)) * (180.0 / SKY_PI);
    let core2 = core_deg * core_deg;
    return HALO_SHARE * max(sky_cie(th * th + core2) - sky_cie(HALO_DEGREES * HALO_DEGREES + core2), 0.0);
}}

// How much of the sun (x) and the moon (y) the clouds let through; the cloud
// shader reads it from its uniforms.
fn sky_halo_through() -> vec2<f32> {{
    return vec2<f32>(1.0, 1.0);
}}

// Air mass along an apparent elevation's sine, as the stars use it.
fn sky_air(y: f32) -> f32 {{
    let h = max(y, -0.02);
    return 1.0 / (h + 0.025 * exp(-11.0 * h));
}}

// Refraction at an apparent elevation (radians), Bennett 1982: 34' at the
// horizon, 5' at 10 deg.
fn sky_refraction(el: f32) -> f32 {{
    let h = max(el * (180.0 / SKY_PI), -1.5);
    return (SKY_PI / 180.0 / 60.0) / tan((h + 7.31 / (h + 4.4)) * (SKY_PI / 180.0));
}}

// The true direction a pixel at apparent direction `d` sees: refraction
// lifts everything near the horizon, the lower limb of a setting sun most,
// so the disc comes out flattened.
fn sky_true_dir(d: vec3<f32>) -> vec3<f32> {{
    let el = asin(clamp(d.y, -1.0, 1.0));
    let e2 = el - sky_refraction(el);
    let k = cos(e2) / max(cos(el), 1e-4);
    return vec3<f32>(d.x * k, sin(e2), d.z * k);
}}

fn sky_apparent(s: vec3<f32>) -> vec3<f32> {{
    let el = asin(clamp(s.y, -1.0, 1.0));
    let e2 = el + sky_refraction(el);
    let k = cos(e2) / max(cos(el), 1e-4);
    return vec3<f32>(s.x * k, sin(e2), s.z * k);
}}

fn sky_moon(d: vec3<f32>, px: f32) -> vec4<f32> {{
    let m = sky_extra.moon.xyz;
    let rho = sky_extra.moon.w;
    if (dot(d, m) < HALO_COS || m.y < -0.05) {{
        return vec4<f32>(0.0);
    }}
    let seen_at = sky_apparent(m);
    let risen = smoothstep(-rho, rho, seen_at.y);
    let halo = sky_extra.moon_radiance.rgb * sky_extra.extinction.w * risen * sky_halo_through().y
        * sky_halo(d, seen_at, max(rho * (180.0 / SKY_PI), 0.5));
    if (dot(d, m) < cos(rho + DISC_REACH + 3.0 * px)) {{
        return vec4<f32>(halo, 0.0);
    }}
    let dt = sky_true_dir(d);
    let pole = sky_extra.celestial[2].xyz;
    let up = normalize(pole - m * dot(pole, m));
    let right = cross(m, up);
    let xy = vec2<f32>(dot(dt, right), dot(dt, up)) / sin(rho);
    if (dot(dt, m) < 0.0) {{
        return vec4<f32>(halo, 0.0);
    }}
    let r = length(xy);
    let e = px / rho;
    let cover = 1.0 - smoothstep(1.0 - e, 1.0 + e, r);
    let mu = sqrt(max(1.0 - min(r * r, 1.0), 0.0));
    let n = right * xy.x + up * xy.y - m * mu;
    let mu0 = dot(n, sky_extra.moon_sunward.xyz);
    let lit = smoothstep(-e, e, mu0) * 2.0 * max(mu0, 0.0) / (max(mu0, 0.0) + mu + 1e-4);
    let albedo = moon_albedo(vec3<f32>(xy.x, xy.y, mu));
    let surface = sky_extra.moon_radiance.rgb * albedo * (lit + sky_extra.moon_radiance.w);

    let rgb = sky_extra.moon_radiance.rgb;
    let peak = max(max(rgb.r, rgb.g), rgb.b) * 1.3 * max(camera.post_params.x, 0.0);
    let seen = min(1.0, MOON_SEEN / max(peak, 1e-12));
    return vec4<f32>(surface * cover * seen + halo, cover);
}}

// The sun: its disc -- limb-darkened per channel (Neckel and Labs: the edge
// dimmer and redder), reddened across itself by the air mass near the
// horizon, lifted and flattened by refraction -- and its glare.
// `sun_radiance` is the disc's mean radiance through the air at the
// snapshot's apparent elevation, whose sine is its `.w`.
fn sky_sun_disc(d: vec3<f32>, px: f32) -> vec3<f32> {{
    let s = sky_extra.sun.xyz;
    let rho = sky_extra.sun.w;
    if (dot(d, s) < HALO_COS) {{
        return vec3<f32>(0.0);
    }}
    let seen_at = sky_apparent(s);
    let risen = smoothstep(-rho, rho, seen_at.y);
    let e_sun = sky_extra.sun_radiance.rgb * (SKY_PI * rho * rho * (1.0 - SUN_LIMB_BAKED / 3.0));
    var out = e_sun * (risen * sky_halo_through().x * sky_halo(d, seen_at, max(rho * (180.0 / SKY_PI), 0.5)));
    if (dot(d, s) < cos(rho + DISC_REACH + 3.0 * px)) {{
        return out;
    }}
    let dt = sky_true_dir(d);
    if (dot(dt, s) < 0.0) {{
        return out;
    }}
    let r = length(cross(dt, s)) / sin(rho);
    let e = px / rho;
    let cover = 1.0 - smoothstep(1.0 - e, 1.0 + e, r);
    let mu = sqrt(max(1.0 - min(r * r, 1.0), 0.0));
    let limb = (1.0 - SUN_LIMB_U * (1.0 - mu)) * ((1.0 - SUN_LIMB_BAKED / 3.0) / (1.0 - SUN_LIMB_U / 3.0));
    let redden = exp(clamp(-sky_extra.extinction.xyz * (sky_air(d.y) - sky_air(sky_extra.sun_radiance.w)), vec3<f32>(-4.0), vec3<f32>(4.0)));
    return out + sky_extra.sun_radiance.rgb * limb * redden * cover;
}}

@fragment fn fs_main(in: VOut) -> @location(0) vec4<f32> {{

    let near = cam_inv_view_proj() * vec4<f32>(in.ndc, 0.0, 1.0);
    let far  = cam_inv_view_proj() * vec4<f32>(in.ndc, 1.0, 1.0);
    let dir = normalize(far.xyz / far.w - near.xyz / near.w);

    let world_dir = to_world_direction(dir);

    let px = max(length(dpdx(world_dir)), length(dpdy(world_dir)));
    let uv = sky_uv(world_dir);
    var radiance = textureSample(sky_tex, sky_samp, uv).rgb * camera.sky_params.x;

    if (sky_extra.mode.x > 0.5) {{
        radiance = radiance * sky_extra.mode.y + sky_sun_disc(world_dir, px);
        let moon = sky_moon(world_dir, px);
        radiance = radiance + moon.rgb;
        if (sky_extra.mode.w > 0.5 && world_dir.y > 0.0) {{
            radiance = radiance + sky_stars(world_dir, px) * (1.0 - moon.a);
        }}
    }}

    {sky_tail}
}}
"#,
        lights_block = wgsl_lights_block(0, 1),
        star_cells = STAR_CELLS,
        limb_r = SUN_LIMB_U[0],
        limb_g = SUN_LIMB_U[1],
        limb_b = SUN_LIMB_U[2],
        limb_baked = SUN_LIMB_BAKED,
        halo_share = super::glare::VEIL_SHARE,
        halo_degrees = HALO_DEGREES,
        halo_cos = HALO_DEGREES.to_radians().cos(),
        cie_1 = super::glare::CIE_TERMS[0],
        cie_2 = super::glare::CIE_TERMS[1],
        cie_3 = super::glare::CIE_TERMS[2],
        disc_reach = DISC_REACH,
        sky_tail = if SKY_CRACK_DEBUG {
            "return vec4<f32>(1.0, 0.0, 1.0, 1.0);"
        } else {
            "return vec4<f32>(tonemap(radiance), 1.0);"
        },
    )
}

pub fn sky_cloud_shader() -> String {
    use space_soup_sky::clouds as c;
    let src = sky_shader();
    let edits: [(&str, String); 5] = [
        (
            "fn sky_halo_through() -> vec2<f32> {\n    return vec2<f32>(1.0, 1.0);\n}",
            "fn sky_halo_through() -> vec2<f32> {\n    return vec2<f32>(sky_extra.cloud_sky.w, sky_extra.cloud_ground.w);\n}".to_string(),
        ),
        (
            "    celestial: array<vec4<f32>, 3>,\n}",
            "    celestial: array<vec4<f32>, 3>,\n    cloud_scale: vec4<f32>,\n    cloud_offset: vec4<f32>,\n    cloud_misc: vec4<f32>,\n    cloud_light_dir: vec4<f32>,\n    cloud_light: vec4<f32>,\n    cloud_sky: vec4<f32>,\n    cloud_ground: vec4<f32>,\n}".to_string(),
        ),
        (
            "@group(1) @binding(2) var<uniform> sky_extra: SkyExtra;\n",
            "@group(1) @binding(2) var<uniform> sky_extra: SkyExtra;\n@group(1) @binding(3) var cloud_tex: texture_2d<f32>;\n@group(1) @binding(4) var cloud_samp: sampler;\n".to_string(),
        ),
        (
            "@fragment fn fs_main(",
            format!(
                r#"const CLOUD_EARTH: f32 = {earth:.1};
const CLOUD_EROSION: f32 = {erosion:?};
const CLOUD_MAX_SLANT: f32 = {slant:?};
const CLOUD_TAP_REACH: f32 = {reach:?};
const CLOUD_MIN_LIGHT_Y: f32 = {min_ly:?};
const CLOUD_OCT_A: f32 = {oa:?};
const CLOUD_OCT_B: f32 = {ob:?};
const CLOUD_OCT_C: f32 = {oc:?};
const CLOUD_FORWARD_G: f32 = {fg:?};
const CLOUD_BACK_G: f32 = {bg:?};
const CLOUD_BACK_SHARE: f32 = {bs:?};
const CLOUD_DIFFUSE_K: f32 = {dk:?};
const CLOUD_HAZE: f32 = {haze:e};
const CLOUD_EDGE_SOFT: f32 = {edge:?};
const CLOUD_TAP_NEAR: f32 = {near:?};
const CLOUD_LIGHT_W0: f32 = {w0:?};
const CLOUD_LIGHT_W1: f32 = {w1:?};
const CLOUD_LIGHT_W2: f32 = {w2:?};
const CLOUD_POWDER_K: f32 = {powder:?};

fn cloud_hg(g: f32, c: f32) -> f32 {{
    let g2 = g * g;
    return (1.0 - g2) / (4.0 * SKY_PI * pow(max(1.0 + g2 - 2.0 * g * c, 1e-6), 1.5));
}}

fn cloud_phase(s: f32, c: f32) -> f32 {{
    return (1.0 - CLOUD_BACK_SHARE) * cloud_hg(CLOUD_FORWARD_G * s, c) + CLOUD_BACK_SHARE * cloud_hg(CLOUD_BACK_G * s, c);
}}

fn cloud_density(xz: vec2<f32>) -> f32 {{
    let cover = sky_extra.cloud_scale.w;
    let a = textureSample(cloud_tex, cloud_samp, xz * sky_extra.cloud_scale.x + sky_extra.cloud_offset.xy).r;
    let b = textureSample(cloud_tex, cloud_samp, xz * sky_extra.cloud_scale.y + sky_extra.cloud_offset.zw).g;
    let detail = textureSample(cloud_tex, cloud_samp, xz * sky_extra.cloud_scale.z + sky_extra.cloud_misc.xy).b;
    let s = 0.5 * (a + b);
    let base = select(1.0 - 2.0 * (1.0 - s) * (1.0 - s), 2.0 * s * s, s < 0.5);
    let d = clamp((base - (1.0 - cover)) / max(cover, 1e-4), 0.0, 1.0);
    return d * (1.0 - CLOUD_EROSION * detail * (1.0 - d));
}}

fn cloud_edge(d: f32) -> f32 {{
    let t = clamp(d / CLOUD_EDGE_SOFT, 0.0, 1.0);
    return d * t * t * (3.0 - 2.0 * t);
}}

fn sky_clouds(dir: vec3<f32>, sky: vec3<f32>) -> vec4<f32> {{
    let mu = max(dir.y, 1e-4);
    let h = sky_extra.cloud_misc.w;
    let cc = h * (2.0 * CLOUD_EARTH + h);
    let rm = CLOUD_EARTH * mu;
    let t = cc / (sqrt(rm * rm + cc) + rm);
    let eye = to_world_space(cam_pos());
    let xz = eye.xz + dir.xz * t;
    let sigma = sky_extra.cloud_misc.z;
    let d = cloud_density(xz);
    let de = cloud_edge(d);
    let tau_v = sigma * de * min(1.0 / mu, CLOUD_MAX_SLANT);
    let l = sky_extra.cloud_light_dir.xyz;
    let ly = max(l.y, CLOUD_MIN_LIGHT_Y);
    let thick = sky_extra.cloud_light_dir.w;
    let off = l.xz / ly * (0.5 * thick);
    let tap = off * min(1.0, CLOUD_TAP_REACH * thick / max(length(off), 1e-6));
    let path = CLOUD_LIGHT_W0 * de + CLOUD_LIGHT_W1 * cloud_edge(cloud_density(xz + CLOUD_TAP_NEAR * tap)) + CLOUD_LIGHT_W2 * cloud_edge(cloud_density(xz + tap));
    let tau_l = sigma * path * min(0.5 / ly, CLOUD_MAX_SLANT);
    let cv = dot(dir, l);
    let tau_c = sigma * de;
    let powder = 1.0 - exp(-CLOUD_POWDER_K * tau_c);
    let oct = (cloud_phase(1.0, cv) * exp(-tau_l)
        + CLOUD_OCT_A * cloud_phase(CLOUD_OCT_C, cv) * exp(-CLOUD_OCT_B * tau_l)
        + CLOUD_OCT_A * CLOUD_OCT_A * cloud_phase(CLOUD_OCT_C * CLOUD_OCT_C, cv) * exp(-CLOUD_OCT_B * CLOUD_OCT_B * tau_l))
        * (powder + (1.0 - powder) * (0.5 + 0.5 * cv));
    let td_l = 1.0 / (1.0 + CLOUD_DIFFUSE_K * 2.0 * tau_l);
    let td_c = 1.0 / (1.0 + CLOUD_DIFFUSE_K * tau_c);
    let light = sky_extra.cloud_light.rgb * (max(l.y, 0.0) * td_l / SKY_PI + oct)
        + sky_extra.cloud_sky.rgb * td_c + sky_extra.cloud_ground.rgb * (1.0 - td_c);
    let alpha = (1.0 - exp(-tau_v)) * select(0.0, 1.0, dir.y > 0.0);
    let a = alpha * exp(-t * CLOUD_HAZE);
    return vec4<f32>(sky * (1.0 - a) + a * light, 1.0 - alpha);
}}

@fragment fn fs_main("#,
                earth = c::EARTH_RADIUS_M,
                erosion = c::EROSION,
                slant = c::MAX_SLANT,
                reach = c::LIGHT_TAP_REACH,
                min_ly = c::MIN_LIGHT_Y,
                oa = c::OCTAVE_A,
                ob = c::OCTAVE_B,
                oc = c::OCTAVE_C,
                fg = c::PHASE_FORWARD_G,
                bg = c::PHASE_BACK_G,
                bs = c::PHASE_BACK_SHARE,
                dk = 0.75 * (1.0 - c::DIFFUSE_G),
                haze = c::HAZE_PER_M,
                edge = c::EDGE_SOFT,
                near = c::LIGHT_TAP_NEAR,
                w0 = c::LIGHT_WEIGHTS[0],
                w1 = c::LIGHT_WEIGHTS[1],
                w2 = c::LIGHT_WEIGHTS[2],
                powder = c::POWDER_K,
            ),
        ),
        (
            "        radiance = radiance * sky_extra.mode.y + sky_sun_disc(world_dir, px);\n        let moon = sky_moon(world_dir, px);\n        radiance = radiance + moon.rgb;\n        if (sky_extra.mode.w > 0.5 && world_dir.y > 0.0) {\n            radiance = radiance + sky_stars(world_dir, px) * (1.0 - moon.a);\n        }",
            "        let cloud = sky_clouds(world_dir, radiance * sky_extra.mode.y);\n        radiance = cloud.rgb + sky_sun_disc(world_dir, px) * cloud.a;\n        let moon = sky_moon(world_dir, px);\n        radiance = radiance + moon.rgb * cloud.a;\n        if (sky_extra.mode.w > 0.5 && world_dir.y > 0.0) {\n            radiance = radiance + sky_stars(world_dir, px) * (1.0 - moon.a) * cloud.a;\n        }".to_string(),
        ),
    ];
    let mut out = src;
    for (from, to) in edits {
        assert_eq!(out.matches(from).count(), 1, "sky_cloud_shader: the sky shader no longer holds `{}`", from.lines().next().unwrap_or(""));
        out = out.replacen(from, &to, 1);
    }
    out
}

pub const STAR_CELLS: f32 = 160.0;

pub fn star_count_in_cells(star_count: f32) -> f32 {
    let n = STAR_CELLS as u32;
    let cell_angle = std::f32::consts::FRAC_PI_2 / STAR_CELLS;
    let mut sum = 0.0f64;
    for y in 0..n {
        for x in 0..n {
            let c = |i: u32| ((i as f32 + 0.5) / STAR_CELLS * 2.0 - 1.0) * std::f32::consts::FRAC_PI_4;
            let (cx, cy) = (c(x).tan(), c(y).tan());
            let r2 = 1.0 + cx * cx + cy * cy;
            let omega = cell_angle * cell_angle * (1.0 + cx * cx) * (1.0 + cy * cy) / (r2 * r2.sqrt());
            sum += (star_count * omega / (4.0 * std::f32::consts::PI)).min(1.0) as f64;
        }
    }
    (sum * 6.0) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_star_field_holds_the_catalogues_count() {
        for n in [100.0f32, 2_000.0, 9_100.0, 20_000.0] {
            let got = star_count_in_cells(n);
            assert!((got / n - 1.0).abs() < 0.01, "{n} stars asked, the field holds {got}");
        }

        let crowded = star_count_in_cells(1.0e7);
        assert!(crowded < 6.0 * STAR_CELLS * STAR_CELLS + 1.0, "{crowded}");
        assert!(sky_shader().contains(&format!("const STAR_CELLS: f32 = {:.1};", STAR_CELLS)));
    }

    #[test]
    fn the_crack_diagnostic_is_off() {
        assert!(!SKY_CRACK_DEBUG, "SKY_CRACK_DEBUG is on: the sky is painted magenta");
    }

    fn workspace_root() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..")
    }

    fn lobby_sky() -> Option<Panorama> {
        let p = workspace_root()
            .join("game/skies/kloofendal_48d_partly_cloudy_puresky/sky.hdr");
        let bytes = std::fs::read(p).ok()?;
        decode_radiance(&bytes).ok()
    }

    #[test]
    fn the_shipped_panorama_decodes() {
        let Some(pano) = lobby_sky() else {
            eprintln!("skipping: the lobby's sky is not installed");
            return;
        };
        assert_eq!(pano.width, pano.height * 2);
        assert!(pano.width >= 1024);
        assert_eq!(pano.rgb.len(), (pano.width * pano.height * 3) as usize);
        assert!(pano.rgb.iter().all(|v| v.is_finite() && *v >= 0.0));
    }

    #[test]
    fn a_panorama_carries_values_above_one() {

        let Some(pano) = lobby_sky() else {
            eprintln!("skipping: the lobby's sky is not installed");
            return;
        };
        let peak = pano.rgb.iter().cloned().fold(0.0f32, f32::max);
        assert!(peak > 100.0, "peak radiance {peak} is too low to be HDR");
    }

    #[test]
    fn the_suns_texel_is_where_the_direction_mapping_says_it_is() {

        let Some(pano) = lobby_sky() else {
            eprintln!("skipping: the lobby's sky is not installed");
            return;
        };
        let mut best = (f32::MIN, 0u32, 0u32);
        for y in 0..pano.height {
            for x in 0..pano.width {
                let t = pano.texel(x, y);
                let l = 0.2126 * t[0] + 0.7152 * t[1] + 0.0722 * t[2];
                if l > best.0 {
                    best = (l, x, y);
                }
            }
        }

        let (elev, azim) = (48.0f32.to_radians(), 34.3f32.to_radians());
        let d = [
            elev.cos() * azim.sin(),
            elev.sin(),
            elev.cos() * azim.cos(),
        ];
        let uv = direction_to_uv(d);
        let (px, py) = (
            (uv[0] * pano.width as f32) as u32,
            (uv[1] * pano.height as f32) as u32,
        );
        assert!(
            px.abs_diff(best.1) <= 2 && py.abs_diff(best.2) <= 2,
            "the sun's direction maps to ({px}, {py}) but the brightest texel is at ({}, {})",
            best.1,
            best.2,
        );
    }

    #[test]
    fn the_shipped_sky_lights_the_ground_from_above_and_from_its_sun() {

        let Some(pano) = lobby_sky() else {
            eprintln!("skipping: the lobby's sky is not installed");
            return;
        };
        let sh = project_irradiance(&pano, 0.0, 1.0);

        let up = sh.evaluate([0.0, 1.0, 0.0]);
        let down = sh.evaluate([0.0, -1.0, 0.0]);
        assert!(
            up[0] > down[0] * 1.5,
            "the sky is not brighter above than below: up {up:?}, down {down:?}",
        );

        let a = 34.3f32.to_radians();
        let toward = sh.evaluate([a.sin(), 0.0, a.cos()]);
        let away = sh.evaluate([-a.sin(), 0.0, -a.cos()]);
        assert!(
            toward[0] > away[0],
            "facing the sun ({toward:?}) is not brighter than facing away ({away:?})",
        );

        assert!(
            up[0] > 0.05 && up[0] < 20.0,
            "upward irradiance {up:?} is not a plausible daylight value",
        );
    }

    #[test]
    fn the_uv_mapping_round_trips() {
        for &d in &[
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.3, 0.5, -0.8],
        ] {
            let uv = direction_to_uv(d);
            let back = uv_to_direction(uv[0], uv[1]);
            let n = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            for c in 0..3 {
                assert!(
                    (back[c] - d[c] / n).abs() < 1e-3,
                    "{d:?} -> {uv:?} -> {back:?}",
                );
            }
        }
    }

    #[test]
    fn a_uniform_sky_gives_uniform_irradiance() {

        let pano = Panorama::solid([0.5, 0.5, 0.5], 64, 32);
        let sh = project_irradiance(&pano, 0.0, 1.0);
        for n in [[0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.3, -0.4, 0.9]] {
            let e = sh.evaluate(n);
            for c in 0..3 {
                assert!(
                    (e[c] - 0.5).abs() < 0.02,
                    "normal {n:?} got {e:?}, expected a flat 0.5",
                );
            }
        }
    }

    #[test]
    fn the_solid_angle_weighting_is_not_optional() {

        let mut pano = Panorama::solid([0.0, 0.0, 0.0], 64, 32);
        for y in 0..2 {
            for x in 0..64 {
                let i = ((y * 64 + x) * 3) as usize;
                pano.rgb[i] = 10.0;
                pano.rgb[i + 1] = 10.0;
                pano.rgb[i + 2] = 10.0;
            }
        }
        let up = project_irradiance(&pano, 0.0, 1.0).evaluate([0.0, 1.0, 0.0]);
        assert!(
            up[0] < 1.0,
            "a thin cap at the pole contributed {up:?}, so it is not being \
             weighted by its solid angle",
        );
        assert!(up[0] > 0.0, "the cap contributed nothing at all: {up:?}");
    }

    #[test]
    fn a_sky_bright_on_one_side_lights_that_side() {

        let mut pano = Panorama::solid([0.0, 0.0, 0.0], 64, 32);
        for y in 0..32 {
            for x in 0..32 {
                let i = ((y * 64 + x) * 3) as usize;
                pano.rgb[i] = 4.0;
                pano.rgb[i + 1] = 4.0;
                pano.rgb[i + 2] = 4.0;
            }
        }
        let sh = project_irradiance(&pano, 0.0, 1.0);

        let lit = sh.evaluate([-1.0, 0.0, 0.0]);
        let dark = sh.evaluate([1.0, 0.0, 0.0]);
        assert!(
            lit[0] > dark[0] * 2.0,
            "the lit side got {lit:?} and the dark side {dark:?}",
        );
    }

    #[test]
    fn rotating_the_sky_moves_the_light_with_it() {

        let mut pano = Panorama::solid([0.0, 0.0, 0.0], 64, 32);
        for y in 0..32 {
            for x in 0..16 {
                let i = ((y * 64 + x) * 3) as usize;
                pano.rgb[i] = 8.0;
                pano.rgb[i + 1] = 8.0;
                pano.rgb[i + 2] = 8.0;
            }
        }
        let probe = [-1.0, 0.0, 0.0];
        let straight = project_irradiance(&pano, 0.0, 1.0).evaluate(probe);
        let turned = project_irradiance(&pano, 180.0, 1.0).evaluate(probe);
        assert!(
            (straight[0] - turned[0]).abs() > straight[0] * 0.3,
            "turning the sky by 180 degrees barely changed the light: \
             {straight:?} vs {turned:?}",
        );
    }

    #[test]
    fn intensity_scales_the_light_it_casts() {
        let pano = Panorama::solid([0.4, 0.4, 0.4], 32, 16);
        let one = project_irradiance(&pano, 0.0, 1.0).evaluate([0.0, 1.0, 0.0]);
        let two = project_irradiance(&pano, 0.0, 2.0).evaluate([0.0, 1.0, 0.0]);
        assert!((two[0] - one[0] * 2.0).abs() < 0.01, "{one:?} vs {two:?}");
    }

    #[test]
    fn no_sky_evaluates_to_exactly_the_old_flat_ambient() {

        let flat = SkyIrradiance::flat(0.6);
        for n in [[0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [-0.5, 0.2, 0.8]] {
            let e = flat.evaluate(n);
            for c in 0..3 {
                assert!((e[c] - 0.6).abs() < 1e-4, "normal {n:?} got {e:?}, wanted 0.6");
            }
        }
    }

    #[test]
    fn half_precision_survives_the_range_a_sky_uses() {
        for v in [0.0f32, 0.5, 1.0, 12.5, 250.0, 6000.0] {
            let h = f32_to_f16(v);

            let sign = (h >> 15) as u32;
            let exp = ((h >> 10) & 0x1f) as i32;
            let mant = (h & 0x3ff) as u32;
            let back = if exp == 0 {
                0.0
            } else {
                f32::from_bits((sign << 31) | (((exp - 15 + 127) as u32) << 23) | (mant << 13))
            };
            let err = if v == 0.0 { back } else { (back - v).abs() / v };
            assert!(err < 0.01, "{v} became {back}");
        }
    }

    #[test]
    fn a_malformed_file_is_refused_rather_than_guessed_at() {
        assert!(decode_radiance(b"not an hdr at all").is_err());
        assert!(decode_radiance(b"#?RADIANCE\n\n+Y 4 +X 4\n").is_err());
    }
}

#[cfg(test)]
mod render_tests {
    use super::*;
    use crate::renderer::lights::LightsUniform;
    use crate::renderer::uniforms::{PlayerUpload, PostUpload, ShadowUpload, SkyUpload, UniformBuffer};

    const SIZE: u32 = 9;

    fn split_panorama() -> Panorama {
        let (w, h) = (64u32, 32u32);
        let mut rgb = Vec::with_capacity((w * h * 3) as usize);
        for _y in 0..h {
            for x in 0..w {

                if x < w / 2 {
                    rgb.extend_from_slice(&[3.0, 0.0, 0.0]);
                } else {
                    rgb.extend_from_slice(&[0.0, 0.0, 3.0]);
                }
            }
        }
        Panorama { width: w, height: h, rgb }
    }

    fn look(dir: [f32; 3], pano: &Panorama) -> Option<[u8; 4]> {
        look_turned(dir, pano, 0.0)
    }

    fn look_turned(dir: [f32; 3], pano: &Panorama, yaw: f32) -> Option<[u8; 4]> {
        render_centre(dir, yaw, 1.0, PostUpload::default(), |device, queue, layout| Sky::new(device, queue, layout, pano, 0.0, 1.0))
    }

    fn look_at_snapshot(dir: [f32; 3], snap: &SkySnapshot, zenith: [f32; 3], fov: f32, exposure: f32) -> Option<[u8; 4]> {
        render_centre(dir, 0.0, fov, PostUpload { exposure, ..Default::default() }, |device, queue, layout| {
            let mut sky = Sky::none(device, queue, layout, AMBIENT);
            sky.apply_snapshot(device, queue, layout, snap, 0.0, zenith);
            sky
        })
    }

    fn render_centre(
        dir: [f32; 3],
        yaw: f32,
        fov: f32,
        post: PostUpload,
        make_sky: impl FnOnce(&Device, &Queue, &BindGroupLayout) -> Sky,
    ) -> Option<[u8; 4]> {
        render_centre_with(dir, yaw, fov, post, make_sky, false)
    }

    fn render_centre_with(
        dir: [f32; 3],
        yaw: f32,
        fov: f32,
        post: PostUpload,
        make_sky: impl FnOnce(&Device, &Queue, &BindGroupLayout) -> Sky,
        force_cloud_twin: bool,
    ) -> Option<[u8; 4]> {
        let (device, queue) = crate::renderer::terrain_pipeline::tests::headless_gpu()?;
        let format = TextureFormat::Rgba8Unorm;

        let lights = LightsUniform::new(&device);
        let (_shadows, uniforms) =
            crate::renderer::uniforms::test_support::scene_uniforms(&device, &lights);
        lights.upload(&queue, &[]);

        let eye = glam::Vec3::ZERO;
        let d = glam::Vec3::from(dir).normalize();
        let up = if d.y.abs() > 0.99 { glam::Vec3::Z } else { glam::Vec3::Y };
        let view_proj = glam::Mat4::perspective_rh(fov, 1.0, 0.1, 100.0)
            * glam::Mat4::look_at_rh(eye, eye + d, up);
        uniforms.upload_scene(
            &queue,
            view_proj,
            eye,
            &ShadowUpload::disabled(),
            &SkyUpload { intensity: 1.0, sh: [[0.0; 4]; 9] },
            &post,
            &PlayerUpload { yaw, ..Default::default() },
        );

        let pipeline = SkyPipeline::new(&device, format, &uniforms.layout, 1);
        let sky = make_sky(&device, &queue, &pipeline.layout);

        let desc = |fmt, usage| TextureDescriptor {
            label: None,
            size: Extent3d { width: SIZE, height: SIZE, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: fmt,
            usage,
            view_formats: &[],
        };
        let target = device.create_texture(&desc(
            format,
            TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
        ));
        let depth = device.create_texture(&desc(
            TextureFormat::Depth32Float,
            TextureUsages::RENDER_ATTACHMENT,
        ));
        let target_view = target.create_view(&Default::default());
        let depth_view = depth.create_view(&Default::default());
        let readback = device.create_buffer(&BufferDescriptor {
            label: None,
            size: (256 * SIZE) as u64,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("sky_test_pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &target_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: Operations { load: LoadOp::Clear(Color::GREEN), store: StoreOp::Store },
                })],
                depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(Operations { load: LoadOp::Clear(1.0), store: StoreOp::Store }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                multiview_mask: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(if force_cloud_twin { &pipeline.clouded } else { pipeline.pick(&sky) });
            pass.set_bind_group(0, &uniforms.bind_group, &[]);
            pass.set_bind_group(1, &sky.bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        encoder.copy_texture_to_buffer(
            TexelCopyTextureInfo {
                texture: &target,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            TexelCopyBufferInfo {
                buffer: &readback,
                layout: TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(256),
                    rows_per_image: Some(SIZE),
                },
            },
            Extent3d { width: SIZE, height: SIZE, depth_or_array_layers: 1 },
        );
        queue.submit(Some(encoder.finish()));

        let slice = readback.slice(..);
        slice.map_async(MapMode::Read, |_| {});
        device.poll(PollType::Wait { submission_index: None, timeout: None }).ok();
        let data = slice.get_mapped_range().unwrap();
        let at = (SIZE / 2) as usize * 256 + (SIZE / 2) as usize * 4;
        Some([data[at], data[at + 1], data[at + 2], data[at + 3]])
    }

    macro_rules! shot {
        ($e:expr) => {
            match $e {
                Some(px) => px,
                None => {
                    eprintln!("skipping: no GPU adapter available");
                    return;
                }
            }
        };
    }

    #[test]
    fn the_shader_samples_the_direction_the_cpu_mapping_says_it_should() {

        let pano = split_panorama();

        for dir in [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.3, 0.0, 1.0], [0.3, 0.0, -1.0]] {
            let px = shot!(look(dir, &pano));
            let uv = direction_to_uv(dir);
            let expected = pano.texel(
                ((uv[0] * pano.width as f32) as u32).min(pano.width - 1),
                ((uv[1] * pano.height as f32) as u32).min(pano.height - 1),
            );
            let want_red = expected[0] > expected[2];
            assert!(
                (px[0] > px[2]) == want_red,
                "looking {dir:?}: the CPU mapping says {expected:?} but the \
                 shader drew {px:?}",
            );
        }
    }

    #[test]
    fn the_sky_stays_with_the_world_when_the_player_turns() {
        let pano = split_panorama();
        for yaw in [std::f32::consts::FRAC_PI_2, std::f32::consts::PI, -std::f32::consts::FRAC_PI_4] {
            for world in [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.3, 0.0, 1.0], [0.3, 0.0, -1.0]] {
                let player = glam::Quat::from_rotation_y(-yaw) * glam::Vec3::from(world);
                let px = shot!(look_turned(player.to_array(), &pano, yaw));
                let uv = direction_to_uv(world);
                let expected = pano.texel(
                    ((uv[0] * pano.width as f32) as u32).min(pano.width - 1),
                    ((uv[1] * pano.height as f32) as u32).min(pano.height - 1),
                );
                assert!(
                    (px[0] > px[2]) == (expected[0] > expected[2]),
                    "turned {yaw} rad, looking {world:?} in the world: the sky there is \
                     {expected:?} but the shader drew {px:?}",
                );
            }
        }
    }

    #[test]
    fn the_sky_is_tonemapped_rather_than_clipped() {

        let px = shot!(look([-1.0, 0.0, 0.0], &split_panorama()));
        assert!(px[0] > 240, "red should be near the top of the range, got {px:?}");
        assert!(
            px[1] > 14 && px[1] < 32,
            "expected the ACES cross-channel green (~22), got {px:?}",
        );
        assert!(
            px[2] > 1 && px[2] < 14,
            "expected the ACES cross-channel blue (~6), got {px:?}",
        );
    }

    #[test]
    fn the_sky_only_fills_where_nothing_was_drawn() {

        let Some((device, queue)) = crate::renderer::terrain_pipeline::tests::headless_gpu() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let format = TextureFormat::Rgba8Unorm;
        let lights = LightsUniform::new(&device);
        let (_shadows, uniforms) =
            crate::renderer::uniforms::test_support::scene_uniforms(&device, &lights);
        lights.upload(&queue, &[]);
        uniforms.upload_with_sky(
            &queue,
            glam::Mat4::IDENTITY,
            glam::Vec3::ZERO,
            &ShadowUpload::disabled(),
            &SkyUpload { intensity: 1.0, sh: [[0.0; 4]; 9] },
        );
        let pipeline = SkyPipeline::new(&device, format, &uniforms.layout, 1);
        let pano = split_panorama();
        let sky = Sky::new(&device, &queue, &pipeline.layout, &pano, 0.0, 1.0);

        let desc = |fmt, usage| TextureDescriptor {
            label: None,
            size: Extent3d { width: SIZE, height: SIZE, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: fmt,
            usage,
            view_formats: &[],
        };
        let target = device.create_texture(&desc(
            format,
            TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
        ));
        let depth = device.create_texture(&desc(
            TextureFormat::Depth32Float,
            TextureUsages::RENDER_ATTACHMENT,
        ));
        let tv = target.create_view(&Default::default());
        let dv = depth.create_view(&Default::default());
        let readback = device.create_buffer(&BufferDescriptor {
            label: None,
            size: (256 * SIZE) as u64,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("occluded_sky"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &tv,
                    depth_slice: None,
                    resolve_target: None,
                    ops: Operations { load: LoadOp::Clear(Color::GREEN), store: StoreOp::Store },
                })],
                depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                    view: &dv,

                    depth_ops: Some(Operations { load: LoadOp::Clear(0.0), store: StoreOp::Store }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                multiview_mask: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&pipeline.pipeline);
            pass.set_bind_group(0, &uniforms.bind_group, &[]);
            pass.set_bind_group(1, &sky.bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        encoder.copy_texture_to_buffer(
            TexelCopyTextureInfo {
                texture: &target, mip_level: 0, origin: Origin3d::ZERO, aspect: TextureAspect::All,
            },
            TexelCopyBufferInfo {
                buffer: &readback,
                layout: TexelCopyBufferLayout {
                    offset: 0, bytes_per_row: Some(256), rows_per_image: Some(SIZE),
                },
            },
            Extent3d { width: SIZE, height: SIZE, depth_or_array_layers: 1 },
        );
        queue.submit(Some(encoder.finish()));
        let slice = readback.slice(..);
        slice.map_async(MapMode::Read, |_| {});
        device.poll(PollType::Wait { submission_index: None, timeout: None }).ok();
        let data = slice.get_mapped_range().unwrap();
        let at = (SIZE / 2) as usize * 256 + (SIZE / 2) as usize * 4;
        let px = [data[at], data[at + 1], data[at + 2], data[at + 3]];
        assert!(
            px[1] > 200 && px[0] < 40 && px[2] < 40,
            "the sky drew over geometry that was already in front of it: {px:?}",
        );
    }

    #[test]
    fn the_time_of_day_sky_draws_noon_and_the_moon() {
        let tod = TimeOfDaySky::new(TimeOfDayParams::default());
        let zenith = tod.atmosphere.eye_transmittance([0.0, 1.0, 0.0]).map(|t| -t.max(1e-6).ln());
        let noon = tod.snapshot(12.0, 0.0);

        let px = shot!(look_at_snapshot([0.0, 1.0, 0.0], &noon, zenith, 0.5, 4.0));
        assert!(px[2] > px[0] + 30 && px[2] > px[1] && px[2] > 60, "noon's zenith is not blue: {px:?}");
        let night = tod.snapshot(23.5, 0.0);
        let moon = night.moon.direction;
        assert!(moon[1] > 0.2, "the moon should be up at 23:30 in the defaults: {moon:?}");

        let disc = shot!(look_at_snapshot(moon, &night, zenith, 0.02, 4.0));
        let m = glam::Vec3::from(moon);
        let beside = (m + m.cross(glam::Vec3::Y).normalize() * 0.05).normalize().to_array();
        let sky = shot!(look_at_snapshot(beside, &night, zenith, 0.02, 4.0));
        assert!(disc[0].max(disc[1]).max(disc[2]) > 60, "the moon's disc is dark: {disc:?}");
        assert!(sky[0].max(sky[1]).max(sky[2]) < 10, "the sky beside the moon is lit: {sky:?}");
    }

    fn cloud_snapshot(coverage: f32) -> (SkySnapshot, [f32; 3]) {
        let clouds = space_soup_sky::clouds::CloudParams { coverage, ..Default::default() };
        let tod = TimeOfDaySky::new(TimeOfDayParams { start_hour: 13.0, clouds, ..Default::default() });
        let zenith = tod.atmosphere.eye_transmittance([0.0, 1.0, 0.0]).map(|t| -t.max(1e-6).ln());
        (tod.snapshot(13.0, 0.0), zenith)
    }

    fn snap_px(dir: [f32; 3], snap: &SkySnapshot, zenith: [f32; 3], fov: f32, post: PostUpload, twin: bool) -> Option<[u8; 4]> {
        let snap = snap.clone();
        render_centre_with(dir, 0.0, fov, post, move |device, queue, layout| {
            let mut sky = Sky::none(device, queue, layout, AMBIENT);
            sky.apply_snapshot(device, queue, layout, &snap, 0.0, zenith);
            sky
        }, twin)
    }

    #[test]
    fn the_cloud_twin_with_no_clouds_draws_the_clear_skys_pixels() {
        let (clear, zenith) = cloud_snapshot(0.0);
        for dir in [[0.0, 1.0, 0.0], [0.6, 0.3, 0.2], [-0.2, 0.05, 1.0], [0.0, -0.5, 1.0]] {
            let post = PostUpload { exposure: 4.0, ..Default::default() };
            let a = shot!(snap_px(dir, &clear, zenith, 0.5, post, false));
            let b = shot!(snap_px(dir, &clear, zenith, 0.5, post, true));
            assert_eq!(a, b, "the cloud twin changes a clear sky at {dir:?}");
        }
    }

    #[test]
    fn the_cloud_shader_draws_what_the_cpu_twin_computes() {
        let (snap, zenith) = cloud_snapshot(0.7);
        let frame = snap.clouds.expect("clouds in the snapshot").at(0.0);
        let field = snap.cloud_field.clone().unwrap();
        let mut clear_snap = snap.clone();
        clear_snap.clouds = None;
        clear_snap.cloud_field = None;
        let pano = ReflectionSky { pano: snap.panorama_engine(), rotation_deg: 0.0, intensity: 1.0 };
        let linear = |e: f32| PostUpload { exposure: e, tonemap: crate::renderer::tonemap::ToneMapping::None, ..Default::default() };
        let (mut compared, mut through_cloud) = (0, 0);
        for i in 0..24 {
            let a = i as f32 * 0.61;
            let up = 0.45 + 0.5 * ((i * 7) % 11) as f32 / 11.0;
            let h = (1.0 - up * up).sqrt();
            let d = glam::Vec3::new(h * a.cos(), up, h * a.sin()).normalize();
            if d.dot(glam::Vec3::from(snap.discs.sun_dir)) > 0.995 {
                continue;
            }
            let (guess, tau) = frame.shade(&field, [0.0, 0.0], d.to_array(), pano.radiance(d));
            let e = 0.6 / guess.iter().cloned().fold(1e-9f32, f32::max);
            let sky8 = shot!(snap_px(d.to_array(), &clear_snap, zenith, 0.01, linear(e), false));
            let got = shot!(snap_px(d.to_array(), &snap, zenith, 0.01, linear(e), false));
            if sky8.iter().take(3).any(|&v| v >= 250) {
                continue;
            }
            let sky_lin = [0, 1, 2].map(|c| sky8[c] as f32 / 255.0 / e);
            let (want, _) = frame.shade(&field, [0.0, 0.0], d.to_array(), sky_lin);
            for c in 0..3 {
                let w = (want[c] * e * 255.0).clamp(0.0, 255.0);
                assert!((w - got[c] as f32).abs() <= 6.0, "{d:?} channel {c}: CPU {w:.1}, GPU {} (tau {tau:.2})", got[c]);
            }
            compared += 1;
            if tau > 0.5 {
                through_cloud += 1;
            }
        }
        assert!(compared >= 12, "only {compared} directions compared");
        assert!(through_cloud >= 3, "only {through_cloud} of them looked through cloud");
    }

    #[test]
    fn an_overcast_noon_zenith_is_grey_not_blue() {
        let (clear, zenith) = cloud_snapshot(0.0);
        let (over, _) = cloud_snapshot(1.0);
        let post = PostUpload { exposure: 4.0, ..Default::default() };
        let blue = shot!(snap_px([0.0, 1.0, 0.0], &clear, zenith, 0.5, post, false));
        let grey = shot!(snap_px([0.0, 1.0, 0.0], &over, zenith, 0.5, post, false));
        let tint = |p: [u8; 4]| p[2] as f32 - p[0] as f32;
        assert!(tint(blue) > 30.0, "the clear zenith is not blue: {blue:?}");
        assert!(tint(grey) < 0.4 * tint(blue), "the overcast zenith stays blue: {grey:?} vs {blue:?}");
    }

    #[test]
    fn the_pipeline_validates_multisampled_too() {

        let Some((device, queue)) = crate::renderer::terrain_pipeline::tests::headless_gpu() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let _ = &queue;
        let lights = LightsUniform::new(&device);
        let (_shadows, uniforms) =
            crate::renderer::uniforms::test_support::scene_uniforms(&device, &lights);
        let err_scope_1 = device.push_error_scope(ErrorFilter::Validation);
        let _one = SkyPipeline::new(&device, TextureFormat::Rgba8UnormSrgb, &uniforms.layout, 1);
        let _four = SkyPipeline::new(&device, TextureFormat::Rgba8UnormSrgb, &uniforms.layout, 4);
        if let Some(e) = pollster::block_on(err_scope_1.pop()) {
            panic!("the sky pipeline failed validation: {e}");
        }
    }
}

#[cfg(test)]
pub fn sky_shader_src() -> String {
    sky_shader()
}

#[cfg(test)]
mod cloud_size_tests {
    #[test]
    fn the_cloud_twin_stays_small() {
        let clear: usize = crate::renderer::shader_inlining::inlined_sizes(&super::sky_shader(), "fs_main").iter().map(|r| r.own * r.copies).sum();
        let clouded: usize = crate::renderer::shader_inlining::inlined_sizes(&super::sky_cloud_shader(), "fs_main").iter().map(|r| r.own * r.copies).sum();
        eprintln!("sky fs_main by the proxy: clear {clear}, with clouds {clouded}");
        assert!(clouded < clear + 400, "the cloud code adds {} (283 when written, 2026-10-19)", clouded - clear);
    }
}
