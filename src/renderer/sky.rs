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
    // Keeps the subnormals (down to 6e-8): this flushed everything under
    // f16's smallest normal, 6.1e-5, to zero, so moonlit and starlit texels
    // of the relit atlas and the ground map stored black (2026-10-22).
    crate::renderer::uniforms::f32_to_f16(v)
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
    /// How the two shown cloud maps are carried to now (`CloudSky::display_flow`):
    /// the last whole map's, the newest's.
    pub cloud_flow: [[f32; 4]; 2],
    /// Each cloud layer's height and wind, lowest first (`cloud_sky::flow_winds`).
    pub cloud_winds: [[f32; 4]; 4],
    /// x: the eye's altitude the maps are carried for.
    pub cloud_eye: [f32; 4],
}

impl SkyExtra {
    pub fn set_clouds(&mut self, f: Option<&CloudFrame>) {
        // `cloud_scale` is the cloud maps' (`Sky::step_clouds`).
        let Some(f) = f else {
            return;
        };
        let p = &f.params;
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
use space_soup_sky::cloud_layers::{self as cloud_layers, CloudLayer, CloudModel, CloudNoise, CloudWeather, LAYERS};
use space_soup_sky::cloud_population::CloudPopulation;
use crate::renderer::cloud_sky::{CloudSky, CloudSkyUniform};

pub struct SkyClouds {
    pub field: std::sync::Arc<CloudField>,
    pub frame: CloudFrame,
    pub sun_clear: Option<SkySun>,
    /// The layers' coverage made to mean area (`CloudModel`), kept while
    /// the scene's cloud parameters stay the same.
    pub model: std::sync::Arc<CloudModel>,
    pub seconds: f64,
    /// The layers where they are now (`Sky::tick`).
    pub layers: [CloudLayer; LAYERS],
    /// The light is the moon's (the sun is down).
    pub light_is_moon: bool,
    /// Every cloud of the populated layers (`cloud_population`), carried
    /// from snapshot to snapshot while the weather field stays.
    pub population: CloudPopulation,
}

/// The clouds' clock at `real` seconds of the day: real time times the
/// scene's `time_scale`, as the snapshot's cloud frame is built
/// (`TimeOfDayParams::cloud_seconds`). Until 2026-10-22 nothing applied it
/// here, so the headset's clouds ran at real speed whatever was authored,
/// and every snapshot's frame was set back to real time at the next tick
/// (user: the clouds "do not noticeably move").
fn cloud_clock(frame: &CloudFrame, real: f64) -> f64 {
    real * frame.params.clamped().time_scale as f64
}

impl SkyClouds {
    /// What the CPU's rays read the clouds from.
    pub fn weather(&self) -> CloudWeather<'_> {
        CloudWeather::new(&self.field, Some(&self.population))
    }

    /// The population at `seconds`, and the layers placed over its maps:
    /// the worker's when there is one -- the first too, which took 1.7 s on
    /// the headset's render thread (2026-10-21); until it comes the layers
    /// read the weather's mean, as past the map -- and `warm` the rays whose
    /// tiles it should fill.
    fn settle(&mut self, worker: Option<&mut crate::renderer::cloud_worker::CloudWorker>, warm: Vec<([f32; 2], f32, [f32; 3])>) {
        match worker {
            Some(w) => {
                let (params, field, model) = (self.model.params, self.field.clone(), self.model.clone());
                let ask = |seconds| crate::renderer::cloud_worker::Request { params, field, model, seconds, warm };
                if let Some(frame) = w.poll(self.seconds, ask) {
                    self.population = frame;
                }
            }
            _ => {
                self.population.update(&self.model.params, &self.field, self.seconds);
            }
        }
        self.population.place(&mut self.layers, self.seconds);
    }
}

/// The level's share of the clear light through the clouds: their direct
/// transmittance averaged over a 3 x 3 grid `LEVEL_SUN_HALF_M` round the
/// level's middle, for what is lit as a whole (probes, lightmaps, effects).
pub fn level_transmittance(layers: &[CloudLayer; LAYERS], weather: CloudWeather, dir: [f32; 3]) -> f32 {
    let noise = CloudNoise::shared();
    let h = space_soup_sky::clouds::LEVEL_SUN_HALF_M;
    let d = unit3(dir);
    let mut sum = 0.0;
    for j in 0..3 {
        for i in 0..3 {
            let xz = [(i as f32 - 1.0) * h, (j as f32 - 1.0) * h];
            sum += cloud_layers::transmittance_toward(layers, &noise, weather, xz, 0.0, d);
        }
    }
    sum / 9.0
}

fn unit3(d: [f32; 3]) -> [f32; 3] {
    let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt().max(1e-6);
    d.map(|v| v / len)
}

fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Metres a side of the clouds' shadow map round the eye: 16 m texels at
/// `CLOUD_SHADOW_SIZE` 64, finer than a cloud's soft edge.
pub const CLOUD_SHADOW_EXTENT_M: f32 = 1024.0;
/// The map's centre moves in steps this long, so it need not be rebuilt
/// every metre walked.
pub const CLOUD_SHADOW_SNAP_M: f32 = 128.0;
/// Frames over which the CPU renews its estimate of the shadow map's mean
/// (the light on ground past its edge): its 16 columns of the twin, one a
/// frame (`step_clouds`).
pub const CLOUD_SHADOW_MEAN_FRAMES: u32 = 16;
/// Frames to one draw of the clouds' shadow on the ground (`step_clouds`).
pub const CLOUD_SHADOW_EVERY: u32 = 4;
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

/// The clouds' shadow on the ground, drawn on the GPU every
/// `CLOUD_SHADOW_EVERY`th frame from the clouds overhead (`Sky::step_clouds`).
#[derive(Default)]
pub struct CloudShadow {
    centre: Option<[f32; 2]>,
    mean: f32,
    /// The light through at the 16 points the mean is of, one renewed a
    /// frame (`step_clouds`); `None` until all are first drawn.
    samples: Option<[f32; 16]>,
    frames: u32,
    /// Read this frame: the clouds are up, the light is above them, and the
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
    /// The volumetric clouds' sky map and shadow, made with the first
    /// clouded snapshot.
    cloud_sky: Option<CloudSky>,
    night: NightMaps,
    /// Bound in the map's place while there is none: clear, all sky.
    cloud_texture: Texture,
    /// And in the heights' place.
    alt_texture: Texture,
    cloud_sampler: Sampler,
    /// Counts the bind groups made, so a user of the maps knows to bind again.
    bind_serial: u64,
    /// The moment and the eye the map being built holds the sky at.
    cloud_build: Option<(f64, [f32; 3])>,
    /// A slice of the sky map every this many `step_clouds` (1: every
    /// frame), and the frames counted toward it (`set_cloud_build_every`).
    cloud_every: (u32, u32),
    /// The cloud clock's step at the last tick, seconds.
    cloud_dt: f32,
    /// The last `step_clouds`' CPU stages, ms: shadow mean, layers placed,
    /// population upload, encode, submit (CLOUDTIME on the headset).
    cloud_laps: [f32; 5],
    /// The haze the clouds are seen through (`SkySnapshot::aerosol_per_m`).
    aerosol_per_m: f32,
    /// Where the eye was last frame (world x, z), for the halos' clouds.
    eye_xz: [f32; 2],
    /// Toward the clouds' light (the sun, or the moon) THIS frame, from the
    /// astronomy `Sky::tick` runs every frame. The snapshot's direction steps
    /// 0.6 deg every 2 s, and at 19 deg up that slid every cloud's shadow
    /// tens of metres in one frame: the beach lost its sun at once
    /// (headset, 2026-10-20 16:55:57).
    light_now: Option<[f32; 3]>,
    /// Toward the light the scene is lit by (the snapshot's sun, or its
    /// moon) THIS frame, clouds or not: what the lit shaders take
    /// (`sun_direct`). The snapshot's own direction steps once a snapshot,
    /// and a sharp glint on the water jumped with it, a pulse across the
    /// sea every second (headset, 2026-10-20 19:24).
    light_glide: Option<[f32; 3]>,
    /// Whether the snapshot's light is the moon.
    light_is_moon: bool,

    /// The time of day's way from the last snapshot to the newest: drawn
    /// and lit between them, frame by frame (`Sky::tick`), so the sky's
    /// colour and the light it gives never step at a snapshot.
    blend: Option<SkyBlend>,
    /// The clouds' population drawn a second ahead off the render thread,
    /// when the app asked for it (`cloud_worker::enable`).
    cloud_worker: Option<crate::renderer::cloud_worker::CloudWorker>,
    /// Whether the clouds have had a population of their own yet: the first
    /// is made here, before the first map, not on the worker.
    clouds_populated: bool,
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
                    view_dimension: TextureViewDimension::D2Array,
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
            night_entry(5),
            night_entry(6),
            night_entry(7),
            // The cloud maps' heights (`CloudSky::alts_view`).
            BindGroupLayoutEntry {
                binding: 8,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::D2Array,
                    multisampled: false,
                },
                count: None,
            },
        ],
    })
}

fn night_entry(binding: u32) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::FRAGMENT,
        ty: BindingType::Texture { sample_type: TextureSampleType::Float { filterable: true }, view_dimension: TextureViewDimension::D2, multisampled: false },
        count: None,
    }
}

/// THE NIGHT SKY'S MAPS (`space_soup_sky::night_sky`): the catalogue's
/// stars in their cells, the moon's face, the Milky Way.
struct NightMaps {
    views: [TextureView; 3],
}

impl NightMaps {
    fn new(device: &Device, queue: &Queue) -> Self {
        use space_soup_sky::night_sky as ns;
        static CELLS: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
        static MILKY: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
        let cells = CELLS.get_or_init(|| ns::star_cells(STAR_CELLS as u32).into_iter().flatten().collect());
        let milky = MILKY.get_or_init(|| ns::MILKY_WAY_MAP.chunks_exact(3).flat_map(|c| [c[0], c[1], c[2], 255]).collect());
        let n = STAR_CELLS as u32;
        let make = |label: &str, w: u32, h: u32, bytes: &[u8]| {
            let t = device.create_texture(&TextureDescriptor {
                label: Some(label),
                size: Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba8Unorm,
                usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
                view_formats: &[],
            });
            queue.write_texture(
                TexelCopyTextureInfo { texture: &t, mip_level: 0, origin: Origin3d::ZERO, aspect: TextureAspect::All },
                bytes,
                TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w * 4), rows_per_image: Some(h) },
                Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            );
            t.create_view(&TextureViewDescriptor::default())
        };
        Self {
            views: [
                make("sky_star_cells", n, n * 12, cells),
                make("sky_moon", ns::MOON_SIZE.0, ns::MOON_SIZE.1, ns::MOON_MAP),
                make("sky_milky_way", ns::MILKY_WAY_SIZE.0, ns::MILKY_WAY_SIZE.1, milky),
            ],
        }
    }
}

fn cloud_dummy(device: &Device, queue: &Queue) -> Texture {
    let texture = device.create_texture(&TextureDescriptor {
        label: Some("sky_clouds_none"),
        size: Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba16Float,
        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let clear: [u16; 4] = [0, 0, 0, f32_to_f16(1.0)];
    queue.write_texture(
        TexelCopyTextureInfo { texture: &texture, mip_level: 0, origin: Origin3d::ZERO, aspect: TextureAspect::All },
        bytemuck::cast_slice(&clear),
        TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(8), rows_per_image: Some(1) },
        Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
    );
    texture
}

fn clouds_view(cloud_sky: &Option<CloudSky>, dummy: &Texture) -> TextureView {
    match cloud_sky {
        Some(c) => c.maps_view().clone(),
        None => dummy.create_view(&TextureViewDescriptor { dimension: Some(TextureViewDimension::D2Array), ..Default::default() }),
    }
}

/// Bound in the heights' place while there is no map: 0, nothing to carry,
/// and nothing in the way (transmittance 1).
fn alt_dummy(device: &Device, queue: &Queue) -> Texture {
    let texture = device.create_texture(&TextureDescriptor {
        label: Some("sky_cloud_alts_none"),
        size: Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba16Float,
        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        TexelCopyTextureInfo { texture: &texture, mip_level: 0, origin: Origin3d::ZERO, aspect: TextureAspect::All },
        &[0u8, 0, 0x00, 0x3c, 0, 0, 0, 0],
        TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(8), rows_per_image: Some(1) },
        Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
    );
    texture
}

fn alts_view(cloud_sky: &Option<CloudSky>, dummy: &Texture) -> TextureView {
    match cloud_sky {
        Some(c) => c.alts_view().clone(),
        None => dummy.create_view(&TextureViewDescriptor { dimension: Some(TextureViewDimension::D2Array), ..Default::default() }),
    }
}

/// What SpaceWarp's sky pass reads the clouds' motion from (`sky_motion`).
pub struct SkyMotionSource {
    /// Changes whenever the maps bound change.
    pub key: u64,
    pub maps: TextureView,
    pub alts: TextureView,
    /// The newest map's layer; the cloud clock's seconds since last frame;
    /// the eye's altitude; 1 with clouds up.
    pub misc: [f32; 4],
    pub winds: [[f32; 4]; 4],
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
        let cloud_texture = cloud_dummy(device, queue);
        let alt_texture = alt_dummy(device, queue);
        let night = NightMaps::new(device, queue);
        let cloud_sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("sky_cloud_sampler"),
            address_mode_u: AddressMode::Repeat,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: MipmapFilterMode::Linear,
            ..Default::default()
        });
        let bind_group = Self::bind(device, layout, &texture, &sampler, &extra_buffer, &clouds_view(&None, &cloud_texture), &alts_view(&None, &alt_texture), &cloud_sampler, &night);

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
            cloud_sky: None,
            light_now: None,
            light_glide: None,
            light_is_moon: false,
            night,
            cloud_texture,
            alt_texture,
            cloud_sampler,
            bind_serial: 0,
            cloud_build: None,
            cloud_every: (1, 0),
            cloud_dt: 0.0,
            cloud_laps: [0.0; 5],
            aerosol_per_m: space_soup_sky::time_of_day::AEROSOL_PER_M,
            eye_xz: [0.0, 0.0],
            blend: None,
            cloud_worker: None,
            clouds_populated: false,
        }
    }

    /// How far the sky is through its blend to the newest snapshot, 0..1.
    pub fn blend_progress(&self) -> f32 {
        self.blend.as_ref().map_or(1.0, |b| b.t)
    }

    pub fn has_clouds(&self) -> bool {
        self.clouds.is_some() && self.cloud_sky.is_some()
    }

    /// What SpaceWarp's sky pass moves the clouds' pixels by: the newest map,
    /// its heights and the winds, and the cloud clock's last step.
    pub fn motion_source(&self) -> SkyMotionSource {
        SkyMotionSource {
            key: self.bind_serial,
            maps: clouds_view(&self.cloud_sky, &self.cloud_texture),
            alts: alts_view(&self.cloud_sky, &self.alt_texture),
            misc: [self.extra.cloud_misc[1], self.cloud_dt, self.extra.cloud_eye[0], if self.has_clouds() { 1.0 } else { 0.0 }],
            winds: self.extra.cloud_winds,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn bind(device: &Device, layout: &BindGroupLayout, texture: &Texture, sampler: &Sampler, extra: &Buffer, cloud_view: &TextureView, alt_view: &TextureView, cloud_sampler: &Sampler, night: &NightMaps) -> BindGroup {
        let view = texture.create_view(&TextureViewDescriptor::default());
        device.create_bind_group(&BindGroupDescriptor {
            label: Some("sky_bg"),
            layout,
            entries: &[
                BindGroupEntry { binding: 0, resource: BindingResource::TextureView(&view) },
                BindGroupEntry { binding: 1, resource: BindingResource::Sampler(sampler) },
                BindGroupEntry { binding: 2, resource: extra.as_entire_binding() },
                BindGroupEntry { binding: 3, resource: BindingResource::TextureView(&cloud_view) },
                BindGroupEntry { binding: 4, resource: BindingResource::Sampler(cloud_sampler) },
                BindGroupEntry { binding: 5, resource: BindingResource::TextureView(&night.views[0]) },
                BindGroupEntry { binding: 6, resource: BindingResource::TextureView(&night.views[1]) },
                BindGroupEntry { binding: 7, resource: BindingResource::TextureView(&night.views[2]) },
                BindGroupEntry { binding: 8, resource: BindingResource::TextureView(alt_view) },
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
            self.bind_group = Self::bind(device, layout, &self.texture, &self.sampler, &self.extra_buffer, &clouds_view(&self.cloud_sky, &self.cloud_texture), &alts_view(&self.cloud_sky, &self.alt_texture), &self.cloud_sampler, &self.night);
            self.bind_serial += 1;
        }
        write_panorama(queue, &self.texture, pano);
        let new_field = match (&snap.cloud_field, &self.clouds) {
            (Some(f), Some(c)) => !std::sync::Arc::ptr_eq(f, &c.field),
            (Some(_), None) => true,
            _ => false,
        };
        if new_field {
            if let Some(field) = snap.cloud_field.as_deref() {
                match self.cloud_sky.as_mut() {
                    Some(c) => c.set_weather(device, queue, field),
                    None => self.cloud_sky = Some(CloudSky::new(device, queue, field, &CloudNoise::shared())),
                }
                self.cloud_build = None;
            }
            self.bind_group = Self::bind(device, layout, &self.texture, &self.sampler, &self.extra_buffer, &clouds_view(&self.cloud_sky, &self.cloud_texture), &alts_view(&self.cloud_sky, &self.alt_texture), &self.cloud_sampler, &self.night);
            self.bind_serial += 1;
        }
        // A new weather field starts the worker's births over.
        if new_field {
            self.cloud_worker = None;
        }
        if crate::renderer::cloud_worker::enabled() && self.cloud_worker.is_none() {
            self.cloud_worker = Some(crate::renderer::cloud_worker::CloudWorker::new());
        }
        let warm = self.cloud_warm_rays();
        let kept = self.clouds.as_ref().map(|c| c.model.clone());
        let population = self.clouds.as_mut().filter(|_| !new_field).map(|c| std::mem::take(&mut c.population)).unwrap_or_default();
        self.clouds = match (snap.clouds, &snap.cloud_field) {
            (Some(frame), Some(field)) => {
                let model = kept
                    .filter(|m| m.params == frame.params)
                    .unwrap_or_else(|| std::sync::Arc::new(CloudModel::new(&frame.params, &CloudNoise::shared())));
                let secs = cloud_clock(&frame, seconds as f64);
                let mut c = SkyClouds {
                    field: field.clone(),
                    frame: frame.at(secs),
                    sun_clear: snap.light_clear,
                    layers: model.layers(secs),
                    model,
                    seconds: secs,
                    // The clouds' own light, which at dusk is still the sun
                    // after the ground's has become the moon
                    // (`TimeOfDaySky::snapshot`, the clouds' light at their
                    // height): its direction glides on with this one.
                    light_is_moon: dot3(frame.light_dir, snap.astro.moon) > dot3(frame.light_dir, snap.astro.sun),
                    population,
                };
                // THE FIRST POPULATION BEFORE THE FIRST MAP (2026-10-23, user:
                // "the sky was much less dense, and then all of the sudden it was
                // filled"). Left to the worker, the first maps were drawn from
                // the weather's mean, a sparser sky, and the population's own
                // cross-faded in seconds later. This is the level's load, where
                // its 1.7 s on this thread cost nothing played.
                if self.clouds_populated {
                    c.settle(self.cloud_worker.as_mut(), warm);
                } else {
                    c.settle(None, warm);
                    self.clouds_populated = true;
                }
                Some(c)
            }
            _ => None,
        };
        self.sun = snap.light;
        self.light_is_moon = snap.light_is_moon;
        self.aerosol_per_m = snap.aerosol_per_m;
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
            self.light_glide = None;
            return;
        }
        self.extra.mode[2] = seconds;
        let warm = self.cloud_warm_rays();
        if let Some(c) = self.clouds.as_mut() {
            let now = cloud_clock(&c.frame, seconds as f64);
            self.cloud_dt = (now - c.seconds).clamp(0.0, 0.25 * c.frame.params.clamped().time_scale.max(1.0) as f64) as f32;
            let before = c.seconds;
            c.seconds = now;
            c.frame = c.frame.at(c.seconds);
            c.layers = c.model.layers(c.seconds);
            c.settle(self.cloud_worker.as_mut(), warm);
            // Whether the clouds' clock and population move on the headset,
            // every ten seconds of it (user: "the clouds don't seem to move
            // at all", 2026-10-21).
            if (before / 10.0).floor() != (c.seconds / 10.0).floor() {
                let n: Vec<usize> = (0..space_soup_sky::cloud_population::POPULATED).map(|k| c.population.splats(k).0.len()).collect();
                log::info!(
                    "CLOUDCLOCK {:.1} s: population of {:.1} s, generation {}, splats {:?}, worker {}, layer 1 shift {:?}",
                    c.seconds,
                    c.population.seconds(),
                    c.population.generation(),
                    n,
                    self.cloud_worker.is_some(),
                    c.layers.get(1).map(|l| l.offset),
                );
            }
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
            self.light_now = self.clouds.as_ref().map(|c| if c.light_is_moon { a.moon } else { a.sun });
            self.light_glide = Some(if self.light_is_moon { a.moon } else { a.sun });
        }
        self.dim_sun_by_clouds();
        queue.write_buffer(&self.extra_buffer, 0, bytemuck::bytes_of(&self.extra));
    }

    fn dim_sun_by_clouds(&mut self) {
        let Some(dir) = self.cloud_light_dir() else { return };
        let Some(c) = &self.clouds else { return };
        let through = level_transmittance(&c.layers, c.weather(), dir);
        self.sun = c.sun_clear.map(|l| SkySun { light_rgb: l.light_rgb.map(|v| v * through), ..l });
        // The glare round the sun and the moon goes as the clouds cover them
        // from the eye (`sky_halo_through` in the cloud shader).
        let noise = CloudNoise::shared();
        let eye = self.eye_xz;
        let toward = |v: [f32; 4]| cloud_layers::transmittance_toward(&c.layers, &noise, c.weather(), eye, 0.0, unit3([v[0], v[1], v[2]]));
        self.extra.cloud_sky[3] = toward(self.extra.sun);
        self.extra.cloud_ground[3] = toward(self.extra.moon);
    }

    /// Toward the clouds' light now: this frame's astronomy when it has
    /// run, else the snapshot's.
    /// The rays toward the light the render thread casts through the clouds
    /// (`dim_sun_by_clouds`, the shadow's mean in `step_clouds`), for the
    /// worker to warm the population's tiles along.
    fn cloud_warm_rays(&self) -> Vec<([f32; 2], f32, [f32; 3])> {
        let Some(dir) = self.cloud_light_dir() else { return Vec::new() };
        let h = space_soup_sky::clouds::LEVEL_SUN_HALF_M;
        let mut rays: Vec<([f32; 2], f32, [f32; 3])> = (0..9).map(|k| ([((k % 3) as f32 - 1.0) * h, ((k / 3) as f32 - 1.0) * h], 0.0, dir)).collect();
        let v3 = |v: [f32; 4]| unit3([v[0], v[1], v[2]]);
        rays.push((self.eye_xz, 0.0, v3(self.extra.sun)));
        rays.push((self.eye_xz, 0.0, v3(self.extra.moon)));
        if let Some(centre) = self.cloud_shadow.centre {
            let at = |k: i32| (k as f32 - 1.5) / 4.0 * CLOUD_SHADOW_EXTENT_M;
            for k in 0..16 {
                rays.push(([centre[0] + at(k % 4), centre[1] + at(k / 4)], self.extra.cloud_eye[0], dir));
            }
        }
        rays
    }

    pub fn cloud_light_dir(&self) -> Option<[f32; 3]> {
        let c = self.clouds.as_ref()?;
        Some(unit3(self.light_now.unwrap_or(c.frame.light_dir)))
    }

    /// The last `step_clouds`' CPU stages, for the headset's log.
    pub fn cloud_step_laps(&self) -> String {
        let l = self.cloud_laps;
        format!("mean {:.1} layers {:.1} population {:.1} encode {:.1} submit {:.1}", l[0], l[1], l[2], l[3], l[4])
    }

    /// Where the cloud maps' history is let go (`CloudSky::set_history_reject`).
    pub fn set_cloud_history_reject(&mut self, above: [f32; 2], horizon: [f32; 2], share: bool) {
        if let Some(gpu) = self.cloud_sky.as_mut() {
            gpu.set_history_reject(above, horizon);
            gpu.set_history_share(share);
        }
    }

    /// Build a slice of the sky map every `n`th frame only. Inside a closed
    /// room the sky shows through a window or a door at most, and a map
    /// carried on its winds for four times as long still holds the clouds
    /// where they are; the hall's frames were the ones over budget with the
    /// clouds on (headset, 2026-10-22).
    pub fn set_cloud_build_every(&mut self, n: u32) {
        let n = n.max(1);
        if n != self.cloud_every.0 {
            log::info!("CLOUDRATE a slice of the sky map every {n} frame(s)");
        }
        self.cloud_every.0 = n;
    }

    /// ONE FRAME OF THE CLOUDS (`CloudSky`): the next slice of the sky map
    /// seen from `eye` (world), and -- when `shadow` is given and `on` (the
    /// lever, the clouds up, the light above them) -- their shadow on the
    /// ground round it. `whole` draws the whole map at once (a still).
    pub fn step_clouds(&mut self, device: &Device, queue: &Queue, eye: [f32; 3], on: bool, shadow: Option<&TextureView>, whole: bool) {
        self.eye_xz = [eye[0], eye[2]];
        let mut lap = std::time::Instant::now();
        let mut laps = [0.0f32; 5];
        let mut mark = |i: usize| {
            let now = std::time::Instant::now();
            laps[i] = now.duration_since(lap).as_secs_f32() * 1000.0;
            lap = now;
        };
        let dir = self.cloud_light_dir();
        let (Some(c), Some(gpu), Some(dir)) = (self.clouds.as_ref(), self.cloud_sky.as_mut(), dir) else {
            self.cloud_shadow.active = false;
            return;
        };
        let cs = &mut self.cloud_shadow;
        let light = c.sun_clear.filter(|_| on && c.frame.params.coverage > 0.0 && dir[1] > 0.0);
        let snap = |v: f32| (v / CLOUD_SHADOW_SNAP_M).round() * CLOUD_SHADOW_SNAP_M;
        let centre = [snap(eye[0]), snap(eye[2])];
        // The ground's shadow is drawn about this centre: one that moves
        // builds this frame, so the shadow never stands a snap off.
        let shadow_moved = cs.centre != Some(centre);
        if light.is_some() {
            // THE MEAN LIGHT THROUGH, ONE POINT A FRAME. All 16 rays at
            // once took 3.0-3.6 ms of the Quest's render thread every 16th
            // frame and whenever the centre moved (`time_the_clouds_on_the_
            // render_thread`): a hitch five times a second. One a frame is
            // ~0.2 ms, and the mean trails the sky by 16 frames.
            let noise = CloudNoise::shared();
            let ray = |n: usize| {
                let at = |k: usize| (k as f32 - 1.5) / 4.0 * CLOUD_SHADOW_EXTENT_M;
                cloud_layers::transmittance_toward(&c.layers, &noise, c.weather(), [centre[0] + at(n % 4), centre[1] + at(n / 4)], eye[1], dir)
            };
            let samples = cs.samples.get_or_insert_with(|| std::array::from_fn(ray));
            let n = cs.frames as usize % CLOUD_SHADOW_MEAN_FRAMES as usize;
            samples[n] = ray(n);
            cs.mean = samples.iter().sum::<f32>() / 16.0;
            cs.frames = cs.frames.wrapping_add(1);
            cs.centre = Some(centre);
        }
        let was_active = cs.active;
        cs.active = light.is_some() && shadow.is_some();
        mark(0);
        let f = &c.frame;
        let u = CloudSkyUniform::new(&c.layers, eye, dir, f.light_irradiance, f.ambient_sky, f.ambient_ground, centre, CLOUD_SHADOW_EXTENT_M, cs.mean).with_aerosol(self.aerosol_per_m);
        // A map holds the sky at one moment, seen from one eye, through all
        // its slices; the sky shader carries it on from there. A map begun
        // now is shown most two builds from now, so it is of then.
        let now = c.seconds;
        // A frame that builds no slice still carries the shown maps on (the
        // uniforms below) and keeps the ground's shadow it last drew.
        let (every, count) = self.cloud_every;
        self.cloud_every.1 = count.wrapping_add(1);
        let build = whole || gpu.is_fresh() || every <= 1 || count % every == 0 || (shadow_moved && cs.active);
        if build && (gpu.starts_map(whole) || self.cloud_build.is_none()) {
            let at = if whole || gpu.is_fresh() { now } else { now + gpu.map_lead() };
            self.cloud_build = Some((at, eye));
        }
        let winds = crate::renderer::cloud_sky::flow_winds(&c.model.params, &c.layers);
        if build {
            let (at, build_eye) = self.cloud_build.unwrap_or((now, eye));
            // The layers at the map's moment, placed over the envelopes as
            // they are now (their origin can move at any update).
            let mut layers = c.model.layers(at);
            c.population.place(&mut layers, at);
            let m = CloudSkyUniform::new(&layers, build_eye, dir, f.light_irradiance, f.ambient_sky, f.ambient_ground, centre, CLOUD_SHADOW_EXTENT_M, cs.mean).with_aerosol(self.aerosol_per_m);
            mark(1);
            gpu.set_population(device, queue, &c.population);
            mark(2);
            gpu.set_winds(winds);
            let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor { label: Some("clouds") });
            // The ground's shadow every CLOUD_SHADOW_EVERY-th frame, or when
            // its centre moves: 16 m texels, and the clouds cross a metre in
            // those frames (0.33 ms a draw on the Quest).
            let draw_shadow = shadow_moved || !was_active || cs.frames % CLOUD_SHADOW_EVERY == 0;
            let began = gpu.starts_map(whole);
            gpu.encode_at(&mut encoder, queue, &m, &u, shadow.filter(|_| cs.active && draw_shadow), whole, now, at);
            mark(3);
            let t = std::time::Instant::now();
            let commands = encoder.finish();
            let finish_ms = t.elapsed().as_secs_f32() * 1000.0;
            queue.submit(Some(commands));
            // The headset showed this submit at 10-28 ms every ~20 s
            // (CLOUDTIME, 2026-10-23), a dropped frame each: named here.
            let submit_ms = t.elapsed().as_secs_f32() * 1000.0 - finish_ms;
            if finish_ms + submit_ms > 8.0 {
                let splats: usize = (0..crate::renderer::cloud_sky::cloud_population::POPULATED).map(|k| c.population.splats(k).0.len()).sum();
                log::info!(
                    "CLOUDSUBMIT finish {finish_ms:.1} ms, submit {submit_ms:.1} ms; envelopes splatted {}, ground shadow {}, map begun {}, {splats} splats",
                    gpu.splatted(),
                    draw_shadow && cs.active,
                    began,
                );
            }
        }
        mark(4);
        self.cloud_laps = laps;
        let d = gpu.display();
        self.extra.cloud_misc[0] = d[0];
        self.extra.cloud_misc[1] = d[1];
        self.extra.cloud_misc[2] = d[2];
        let k = gpu.display_scales();
        self.extra.cloud_scale = [k[0], k[1], 0.0, 0.0];
        self.extra.cloud_flow = gpu.display_flow(now, eye);
        self.extra.cloud_winds = winds;
        self.extra.cloud_eye = [eye[1], 0.0, 0.0, 0.0];
        queue.write_buffer(&self.extra_buffer, 0, bytemuck::bytes_of(&self.extra));
    }

    /// How hard the clouds over `xz` (world) rain, 0..1
    /// (`space_soup_sky::precipitation`); None with no clouds in the sky.
    pub fn precipitation_at(&self, xz: [f32; 2]) -> Option<f32> {
        self.clouds.as_ref().map(|c| space_soup_sky::precipitation::precipitation_at(&c.layers, c.weather(), xz))
    }

    /// Where the shown map lies (`Uniforms::cloud_map`) for the player's
    /// `yaw` and `offset`; zeros when it is not read.
    pub fn cloud_map(&self, yaw: f32, offset: [f32; 3]) -> [[f32; 4]; 3] {
        let cs = &self.cloud_shadow;
        let sun = self.clouds.as_ref().and_then(|c| c.sun_clear);
        match (cs.centre.filter(|_| cs.active), sun) {
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
        let dir = self.cloud_light_dir().unwrap_or(s.direction);
        let t = cloud_layers::transmittance_penumbra(&c.layers, &CloudNoise::shared(), c.weather(), xz, 0.0, dir);
        Some(SkySun { light_rgb: s.light_rgb.map(|v| v * t), ..s })
    }

    /// The sun the lit shaders take: the clear one while the shadow map
    /// dims it per pixel, else the one dimmed by the clouds over the level.
    /// It points where the light is this frame (`light_glide`).
    pub fn sun_direct(&self) -> Option<SkySun> {
        let glide = |s: SkySun| SkySun { direction: self.light_glide.unwrap_or(s.direction), ..s };
        self.sun_stepped().map(glide)
    }

    /// The same light pointing where the newest snapshot put it: what the
    /// static sun map is drawn along, so it is drawn again once a snapshot
    /// rather than every frame.
    pub fn sun_stepped(&self) -> Option<SkySun> {
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
@group(1) @binding(5) var star_cells: texture_2d<f32>;
@group(1) @binding(6) var moon_map: texture_2d<f32>;
@group(1) @binding(7) var milky_way: texture_2d<f32>;

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

// A star's colour from its B - V (Ballesteros 2012's temperature, then
// Planck's law at 610, 550 and 465 nm over 550), held to unit luminance and
// taken halfway to white: at night the eye sees star colours weakly.
fn star_colour(bv: f32) -> vec3<f32> {{
    let temp = 4600.0 * (1.0 / (0.92 * bv + 1.7) + 1.0 / (0.92 * bv + 0.62));
    let k = 1.4388e7 / temp;
    let g = exp(k / 550.0) - 1.0;
    let p = vec3<f32>(pow(550.0 / 610.0, 5.0) * g / (exp(k / 610.0) - 1.0), 1.0, pow(550.0 / 465.0, 5.0) * g / (exp(k / 465.0) - 1.0));
    let c = p / dot(p, vec3<f32>(0.2126, 0.7152, 0.0722));
    return mix(vec3<f32>(1.0), c, 0.5);
}}

const MILKY_WAY_SCALE: f32 = {milky_way_scale:?};

// The Milky Way at celestial direction q (NASA Deep Star Maps, its bright
// stars the catalogue's): radiance per a magnitude-0 star's illuminance.
fn sky_milky_way(q: vec3<f32>) -> vec3<f32> {{
    let ra = atan2(q.y, q.x);
    let dec = asin(clamp(q.z, -1.0, 1.0));
    let uv = vec2<f32>(fract(0.5 - ra / (2.0 * SKY_PI)), 0.5 - dec / SKY_PI);
    let c = textureSampleLevel(milky_way, sky_samp, uv, 0.0).rgb;
    return pow(c, vec3<f32>(2.2)) * (sky_extra.stars.x * MILKY_WAY_SCALE);
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
    var sum = vec3<f32>(0.0);
    let lim = sky_extra.stars.y;
    for (var k = 0; k < 4; k = k + 1) {{
        let cell = base + vec2<f32>(f32(k & 1), f32(k >> 1u));
        if (any(cell < vec2<f32>(0.0)) || any(cell >= vec2<f32>(STAR_CELLS))) {{
            continue;
        }}
        for (var second = 0u; second < 2u; second = second + 1u) {{
            let t = textureLoad(star_cells, vec2<i32>(i32(cell.x), i32(cell.y) + i32((face + 6u * second) * u32(STAR_CELLS))), 0);
            if (t.z <= 0.0) {{
                break;
            }}
            // The catalogue's star: its magnitude, faded out round the
            // night's limiting magnitude (the light pollution's).
            let m = (t.z * 255.0 - 1.0) / 25.0 - 2.0;
            let fade = 1.0 - smoothstep(lim - 0.5, lim + 0.5, m);
            if (fade <= 0.0) {{
                continue;
            }}
            let place = (t.xy * 255.0 + 0.5) / 256.0;
            let at = star_face_dir(face, tan(((cell + place) / STAR_CELLS * 2.0 - 1.0) * (SKY_PI * 0.25)));
            let off = length(cross(q, at));
            let flux = sky_extra.stars.x * exp2(-1.3287712 * m) * fade;
            let spread = flux / (2.0 * SKY_PI * sigma * sigma) * exp(-off * off / (2.0 * sigma * sigma));
            let h = star_hash(u32(cell.x), u32(cell.y), face + 6u * second);
            let tw = 1.0 + twinkle_depth * (0.6 * sin(sky_extra.mode.z * (11.0 + 7.0 * h.y) + 40.0 * h.z) + 0.4 * sin(sky_extra.mode.z * (23.0 + 9.0 * h.z) + 17.0 * h.y));
            sum = sum + spread * star_colour(t.w * 2.55 - 0.5) * tw;
        }}
    }}
    sum = sum + sky_milky_way(q);
    return sum * dimmed;
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

// The moon's halo against the sun's (`sky_halo`): at the sun's share the
// full moon wore a glare near the sun's, 15 degrees across in the night eye
// (headset, 2026-10-22).
const MOON_HALO: f32 = 0.3;

// `k`: the share of the disc (not its halo) that comes through (clouds).
fn sky_moon(d: vec3<f32>, px: f32, k: f32) -> vec4<f32> {{
    let m = sky_extra.moon.xyz;
    let rho = sky_extra.moon.w;
    if (dot(d, m) < HALO_COS || m.y < -0.05) {{
        return vec4<f32>(0.0);
    }}
    let seen_at = sky_apparent(m);
    let risen = smoothstep(-rho, rho, seen_at.y);
    let halo = MOON_HALO * sky_extra.moon_radiance.rgb * sky_extra.extinction.w * risen * sky_halo_through().y
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
    // The moon's own face (LRO): x east, y north, z toward the eye.
    let p = vec3<f32>(xy.x, xy.y, mu);
    let lat = asin(clamp(p.y, -1.0, 1.0));
    let lon = atan2(p.x, p.z);
    let face = textureSampleLevel(moon_map, sky_samp, vec2<f32>(lon / (2.0 * SKY_PI) + 0.5, 0.5 - lat / SKY_PI), 0.0);
    // Its relief (LOLA) tilts the ground toward the sun or away, which the
    // terminator shows.
    let slope = (face.gb - 0.5) * 0.25;
    let east = vec3<f32>(cos(lon), 0.0, -sin(lon));
    let north = vec3<f32>(-sin(lat) * sin(lon), cos(lat), -sin(lat) * cos(lon));
    let pn = normalize(p - slope.x * east - slope.y * north);
    let n = right * pn.x + up * pn.y - m * pn.z;
    let mu0 = dot(n, sky_extra.moon_sunward.xyz);
    let lit = smoothstep(-e, e, mu0) * 2.0 * max(mu0, 0.0) / (max(mu0, 0.0) + mu + 1e-4);
    let tint = (face.a - 0.5) * 0.5;
    let albedo = face.r * 2.0 * vec3<f32>(1.0 - 0.5 * tint, 1.0, 1.0 + 0.5 * tint);
    let surface = sky_extra.moon_radiance.rgb * albedo * (lit + sky_extra.moon_radiance.w);

    let rgb = sky_extra.moon_radiance.rgb;
    let peak = max(max(rgb.r, rgb.g), rgb.b) * 1.3 * max(camera.post_params.x, 0.0);
    let seen = min(1.0, MOON_SEEN / max(peak, 1e-12));
    return vec4<f32>(surface * cover * seen * k + halo, cover);
}}

// The sun: its disc -- limb-darkened per channel (Neckel and Labs: the edge
// dimmer and redder), reddened across itself by the air mass near the
// horizon, lifted and flattened by refraction -- and its glare.
// `sun_radiance` is the disc's mean radiance through the air at the
// snapshot's apparent elevation, whose sine is its `.w`.
fn sky_sun_disc(d: vec3<f32>, px: f32, k: f32) -> vec3<f32> {{
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
    return out + sky_extra.sun_radiance.rgb * limb * redden * cover * k;
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
        radiance = radiance * sky_extra.mode.y + sky_sun_disc(world_dir, px, 1.0);
        let moon = sky_moon(world_dir, px, 1.0);
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
        milky_way_scale = space_soup_sky::night_sky::milky_way_scale(),
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
    let src = sky_shader();
    let edits: [(&str, String); 5] = [
        (
            "fn sky_halo_through() -> vec2<f32> {\n    return vec2<f32>(1.0, 1.0);\n}",
            "fn sky_halo_through() -> vec2<f32> {\n    return vec2<f32>(sky_extra.cloud_sky.w, sky_extra.cloud_ground.w);\n}".to_string(),
        ),
        (
            "    celestial: array<vec4<f32>, 3>,\n}",
            "    celestial: array<vec4<f32>, 3>,\n    cloud_scale: vec4<f32>,\n    cloud_offset: vec4<f32>,\n    cloud_misc: vec4<f32>,\n    cloud_light_dir: vec4<f32>,\n    cloud_light: vec4<f32>,\n    cloud_sky: vec4<f32>,\n    cloud_ground: vec4<f32>,\n    cloud_flow: array<vec4<f32>, 2>,\n    cloud_winds: array<vec4<f32>, 4>,\n    cloud_eye: vec4<f32>,\n}".to_string(),
        ),
        (
            "@group(1) @binding(2) var<uniform> sky_extra: SkyExtra;\n",
            "@group(1) @binding(2) var<uniform> sky_extra: SkyExtra;\n@group(1) @binding(3) var cloud_maps: texture_2d_array<f32>;\n@group(1) @binding(4) var cloud_samp: sampler;\n@group(1) @binding(8) var cloud_alts: texture_2d_array<f32>;\n".to_string(),
        ),
        (
            "@fragment fn fs_main(",
            format!(
                r#"{flow}

// A sky map read through a cubic B-spline, in four bilinear taps (Sigg and
// Hadwiger 2005): smooth to its second derivative, so no texel's square
// shows at a cloud's edge however far the headset magnifies the map.
fn sky_cloud_map(uv: vec2<f32>, layer: i32) -> vec4<f32> {{
    let size = vec2<f32>(textureDimensions(cloud_maps, 0));
    let p = uv * size - 0.5;
    let i = floor(p);
    let f = p - i;
    let f2 = f * f;
    let f3 = f2 * f;
    let w0 = (1.0 - 3.0 * f + 3.0 * f2 - f3) / 6.0;
    let w1 = (4.0 - 6.0 * f2 + 3.0 * f3) / 6.0;
    let w2 = (1.0 + 3.0 * f + 3.0 * f2 - 3.0 * f3) / 6.0;
    let w3 = f3 / 6.0;
    let g0 = w0 + w1;
    let g1 = w2 + w3;
    let c0 = (i + 0.5 - 1.0 + w1 / g0) / size;
    let c1 = (i + 0.5 + 1.0 + w3 / g1) / size;
    let s00 = textureSampleLevel(cloud_maps, cloud_samp, vec2<f32>(c0.x, c0.y), layer, 0.0);
    let s10 = textureSampleLevel(cloud_maps, cloud_samp, vec2<f32>(c1.x, c0.y), layer, 0.0);
    let s01 = textureSampleLevel(cloud_maps, cloud_samp, vec2<f32>(c0.x, c1.y), layer, 0.0);
    let s11 = textureSampleLevel(cloud_maps, cloud_samp, vec2<f32>(c1.x, c1.y), layer, 0.0);
    return g0.y * (g0.x * s00 + g1.x * s10) + g1.y * (g0.x * s01 + g1.x * s11);
}}

// The volumetric clouds (`cloud_sky.rs`): the sky map's last whole build
// blending into its newest, rgb their light and a how much sky shows past
// them, haze and all. Each map is read where it holds what is here now:
// the cloud seen along `dir`, at the newest map's height for it, carried
// back on that height's wind to the map's moment and seen from the map's
// eye (`cloud_sky::advected_uv`).
// How much of a disc behind the clouds comes through: the map's alpha is
// haze-weighted (a far cloud fades into the air, and the air's own glow is
// in the sky behind), but a cloud blocks the sun however far off it is.
// Near the horizon the alpha read 1 behind thick cloud and the sun shone
// through it (headset, 2026-10-21). Read only near a disc (`alts` .g), as
// the share of the alpha a disc gets.
var<private> cloud_through: f32 = 1.0;

fn sky_clouds(dir: vec3<f32>, sky: vec3<f32>) -> vec4<f32> {{
    if (dir.y <= 0.0) {{
        return vec4<f32>(sky, 1.0);
    }}
    let uv = cloud_map_uv(dir);
    var uva = uv;
    var uvb = uv;
    // Each map carried on its OWN height and wind for what it shows here
    // (`fs_map`, alts .r and .ba). Both carried on the newest map's, the
    // older map's carry changed at every swap, when the newest became a
    // different map: its clouds jumped up to 0.4 degrees, back and forth,
    // once a build (user, 2026-10-23: "a small backstep every 1 second";
    // `the_shown_clouds_ride_their_wind_frame_by_frame`).
    let here = textureSampleLevel(cloud_alts, cloud_samp, uv, i32(sky_extra.cloud_misc.y), 0.0);
    let older = textureSampleLevel(cloud_alts, cloud_samp, uv, i32(sky_extra.cloud_misc.x), 0.0);
    if (older.r > 0.0) {{
        let fa = sky_extra.cloud_flow[0];
        let p = dir * cloud_flow_distance(sky_extra.cloud_eye.x, older.r, dir.y);
        uva = cloud_map_uv(p + fa.xyz - vec3<f32>(older.b, 0.0, older.a) * fa.w);
    }}
    if (here.r > 0.0) {{
        let fb = sky_extra.cloud_flow[1];
        let p = dir * cloud_flow_distance(sky_extra.cloud_eye.x, here.r, dir.y);
        uvb = cloud_map_uv(p + fb.xyz - vec3<f32>(here.b, 0.0, here.a) * fb.w);
    }}
    var a = sky_cloud_map(uva, i32(sky_extra.cloud_misc.x));
    var b = sky_cloud_map(uvb, i32(sky_extra.cloud_misc.y));
    a = vec4<f32>(a.rgb * sky_extra.cloud_scale.x, a.a);
    b = vec4<f32>(b.rgb * sky_extra.cloud_scale.y, b.a);
    let c = mix(a, b, sky_extra.cloud_misc.z);
    if (max(dot(dir, sky_extra.sun.xyz), dot(dir, sky_extra.moon.xyz)) > 0.9995) {{
        let ta = textureSampleLevel(cloud_alts, cloud_samp, uva, i32(sky_extra.cloud_misc.x), 0.0).g;
        let tb = textureSampleLevel(cloud_alts, cloud_samp, uvb, i32(sky_extra.cloud_misc.y), 0.0).g;
        // As a share of the alpha, which it never exceeds: both are read at
        // the same uv (test `a_far_cloud_the_haze_hides_still_blocks_the_sun`).
        cloud_through = mix(ta, tb, sky_extra.cloud_misc.z) / max(c.a, 1e-4);
    }}
    return vec4<f32>(sky * c.a + c.rgb, c.a);
}}

@fragment fn fs_main("#,
                flow = crate::renderer::cloud_sky::cloud_flow_wgsl(),
            ),
        ),
        (
            "        radiance = radiance * sky_extra.mode.y + sky_sun_disc(world_dir, px, 1.0);\n        let moon = sky_moon(world_dir, px, 1.0);\n        radiance = radiance + moon.rgb;\n        if (sky_extra.mode.w > 0.5 && world_dir.y > 0.0) {\n            radiance = radiance + sky_stars(world_dir, px) * (1.0 - moon.a);\n        }",
            "        let cloud = sky_clouds(world_dir, radiance * sky_extra.mode.y);\n        radiance = cloud.rgb + sky_sun_disc(world_dir, px, cloud_through) * cloud.a;\n        let moon = sky_moon(world_dir, px, cloud_through);\n        radiance = radiance + moon.rgb * cloud.a;\n        if (sky_extra.mode.w > 0.5 && world_dir.y > 0.0) {\n            radiance = radiance + sky_stars(world_dir, px) * (1.0 - moon.a) * cloud.a;\n        }".to_string(),
        ),
    ];
    let mut out = src;
    for (from, to) in edits {
        assert_eq!(out.matches(from).count(), 1, "sky_cloud_shader: the sky shader no longer holds `{}`", from.lines().next().unwrap_or(""));
        out = out.replacen(from, &to, 1);
    }
    out
}

pub const STAR_CELLS: f32 = 256.0;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_shader_reads_the_star_cells_at_their_size() {
        assert!(sky_shader().contains(&format!("const STAR_CELLS: f32 = {:.1};", STAR_CELLS)));
    }

    /// WHAT THE CLOUDS COST THE RENDER THREAD a frame, as `Sky::tick` and
    /// `step_clouds` spend it on the CPU: the layers, the population's update
    /// and the rays toward the light, over two minutes of 72 Hz frames.
    /// The headset logged ticks of 5-55 ms about once a second and 120-160
    /// ms now and then (TODTIME, 2026-10-21): a dropped frame each second.
    /// `--ignored`.
    #[test]
    #[ignore]
    fn time_the_clouds_on_the_render_thread() {
        use space_soup_sky::cloud_layers::CloudModel;
        use space_soup_sky::cloud_population::CloudPopulation;
        let field = CloudField::build(7);
        let noise = CloudNoise::shared();
        let params = space_soup_sky::clouds::CloudParams { coverage: 0.45, wind_speed: 12.0, ..Default::default() };
        let model = CloudModel::new(&params, &noise);
        let built = std::time::Instant::now();
        let _again = CloudModel::new(&params, &noise);
        eprintln!("CLOUDTICK a CloudModel takes {:.1} ms", built.elapsed().as_secs_f64() * 1000.0);
        let mut population = CloudPopulation::new();
        let t0 = 20_000.0;
        let started = std::time::Instant::now();
        population.update(&params, &field, t0);
        let first = started.elapsed().as_secs_f64() * 1000.0;
        let (mut worst, mut worst_update, mut worst_rays, mut over) = (0.0f64, 0.0f64, 0.0f64, 0u32);
        let mut worst_mean = 0.0f64;
        let (mut sum_level, mut sum_halo) = (0.0f64, 0.0f64);
        let frames = 72 * 120;
        for f in 0..frames {
            let t = t0 + f as f64 / 72.0;
            let a = std::time::Instant::now();
            let mut layers = model.layers(t);
            population.update(&params, &field, t);
            population.place(&mut layers, t);
            let b = std::time::Instant::now();
            let w = CloudWeather::new(&field, Some(&population));
            let dir = [0.4, 0.5, 0.3];
            let l0 = std::time::Instant::now();
            let _ = level_transmittance(&layers, w, dir);
            let l1 = std::time::Instant::now();
            let _ = cloud_layers::transmittance_toward(&layers, &noise, w, [3.0, 4.0], 0.0, unit3(dir));
            let _ = cloud_layers::transmittance_toward(&layers, &noise, w, [3.0, 4.0], 0.0, unit3([-0.4, 0.5, -0.3]));
            sum_level += (l1 - l0).as_secs_f64() * 1000.0;
            sum_halo += l1.elapsed().as_secs_f64() * 1000.0;
            // The shadow's mean, one of its 16 points a frame (`step_clouds`).
            let m = std::time::Instant::now();
            let k = f % CLOUD_SHADOW_MEAN_FRAMES as usize;
            let _ = cloud_layers::transmittance_toward(&layers, &noise, w, [(k % 4) as f32 * 1000.0, (k / 4) as f32 * 1000.0], 0.0, unit3(dir));
            worst_mean = worst_mean.max(m.elapsed().as_secs_f64() * 1000.0);
            let c = std::time::Instant::now();
            let (u, r, all) = ((b - a).as_secs_f64() * 1000.0, (c - b).as_secs_f64() * 1000.0, (c - a).as_secs_f64() * 1000.0);
            worst = worst.max(all);
            worst_update = worst_update.max(u);
            worst_rays = worst_rays.max(r);
            over += (all > 2.0) as u32;
        }
        eprintln!("CLOUDTICK first update {first:.1} ms; over {frames} frames the worst {worst:.2} ms (update {worst_update:.2}, rays {worst_rays:.2}, of which the mean's ray {worst_mean:.2}), {over} frames over 2 ms; a frame's level light {:.3} ms, halo rays {:.3} ms on average", sum_level / frames as f64, sum_halo / frames as f64);
        // The same with the worker drawing each second ahead (`cloud_worker`),
        // in real time so it has the second it is given.
        let field = std::sync::Arc::new(field);
        let model = std::sync::Arc::new(model);
        let mut worker = crate::renderer::cloud_worker::CloudWorker::new();
        let (mut worst, mut over, mut swaps) = (0.0f64, 0u32, 0u32);
        let mut worst_at = (0usize, 0.0f64, 0.0f64, 0.0f64);
        let start = std::time::Instant::now();
        let real = 72 * 20;
        for f in 0..real {
            while start.elapsed().as_secs_f64() < f as f64 / 72.0 {
                std::thread::yield_now();
            }
            let t = t0 + 200.0 + f as f64 / 72.0;
            let a = std::time::Instant::now();
            let mut layers = model.layers(t);
            let dir = [0.4, 0.5, 0.3];
            let warm: Vec<([f32; 2], f32, [f32; 3])> = (0..9).map(|k| ([((k % 3) as f32 - 1.0) * 150.0, ((k / 3) as f32 - 1.0) * 150.0], 0.0, unit3(dir))).collect();
            let ask = |seconds| crate::renderer::cloud_worker::Request { params, field: field.clone(), model: model.clone(), seconds, warm };
            let p0 = std::time::Instant::now();
            if let Some(frame) = worker.poll(t, ask) {
                population = frame;
                swaps += 1;
            }
            let p1 = std::time::Instant::now();
            population.place(&mut layers, t);
            let w = CloudWeather::new(&field, Some(&population));
            let _ = level_transmittance(&layers, w, dir);
            let all = a.elapsed().as_secs_f64() * 1000.0;
            if all > worst {
                worst_at = (f, (p0 - a).as_secs_f64() * 1000.0, (p1 - p0).as_secs_f64() * 1000.0, p1.elapsed().as_secs_f64() * 1000.0);
            }
            worst = worst.max(all);
            over += (all > 2.0) as u32;
        }
        eprintln!(
            "CLOUDTICK with the worker: {real} frames in real time, {swaps} populations swapped in, the worst {worst:.2} ms (frame {}: layers {:.2}, poll and swap {:.2}, place and light {:.2}), {over} frames over 2 ms",
            worst_at.0, worst_at.1, worst_at.2, worst_at.3
        );
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

    /// THE SKY'S CLOUDS THROUGH TEN MINUTES of test_room's: each map built
    /// whole at that moment, its share of cloud above 10 deg printed (the
    /// offline frames lost every cumulus past ~200 s, 2026-10-22).
    #[test]
    #[ignore]
    fn print_the_sky_maps_cloud_through_ten_minutes() {
        use crate::renderer::cloud_sky::tests::{half, read_back};
        use crate::renderer::cloud_sky::{MAP_HEIGHT, MAP_WIDTH};
        let Some((device, queue)) = crate::renderer::terrain_pipeline::tests::headless_gpu() else { return };
        let layout = sky_bind_group_layout(&device);
        let clouds = space_soup_sky::clouds::CloudParams { coverage: 0.45, altitude_m: 1200.0, wind_speed: 16.0, weather_minutes: 15.0, storms: 0.35, time_scale: 2.5, kinds: [1.0, 1.0, 0.0, 1.0], ..Default::default() };
        let params = TimeOfDayParams { start_hour: 15.0, day_minutes: 0.0, clouds, ..Default::default() };
        let tod = TimeOfDaySky::new(params);
        let zenith = tod.atmosphere.eye_transmittance([0.0, 1.0, 0.0]).map(|t| -t.max(1e-6).ln());
        let snap = tod.snapshot(15.0, 0.0);
        for i in 0..16 {
            let secs = i as f32 * 40.0;
            let mut sky = Sky::none(&device, &queue, &layout, AMBIENT);
            sky.apply_snapshot(&device, &queue, &layout, &snap, secs, zenith);
            sky.step_clouds(&device, &queue, [0.0, 1.6, 12.0], false, None, true);
            let gpu = sky.cloud_sky.as_ref().unwrap();
            let map = read_back(&device, &queue, gpu.textures().0, gpu.display()[1] as u32, MAP_WIDTH, MAP_HEIGHT, 8);
            let rows = (MAP_HEIGHT / 9)..MAP_HEIGHT;
            let (mut cloud, mut n) = (0usize, 0usize);
            for y in rows {
                for x in (0..MAP_WIDTH).step_by(4) {
                    let i = (y * MAP_WIDTH + x) as usize;
                    n += 1;
                    if half(&map[i * 8 + 6..i * 8 + 8]) < 0.5 {
                        cloud += 1;
                    }
                }
            }
            let c = sky.clouds.as_ref().unwrap();
            eprintln!("{secs:4.0} s (cloud clock {:.0} s): {:.1}% cloud, {} cumulus, envelope {:?}", c.seconds, 100.0 * cloud as f32 / n as f32, c.population.splats(1).0.len(), c.layers[1].envelope);
        }
    }

    /// THE AUTHORED CLOUD SPEED REACHES THEIR CLOCK: at `time_scale` 3 the
    /// clouds are three times as far through their day as at 1 after the
    /// same real seconds, at a tick and at a snapshot landing alike. Nothing
    /// applied it before 2026-10-22: the headset logged "population of
    /// 80.0 s" after 80 s with 2.5 authored.
    #[test]
    fn the_clouds_run_at_their_authored_speed() {
        let Some((device, queue)) = crate::renderer::terrain_pipeline::tests::headless_gpu() else { return };
        let layout = sky_bind_group_layout(&device);
        let run = |time_scale: f32| {
            let clouds = space_soup_sky::clouds::CloudParams { coverage: 0.45, time_scale, ..Default::default() };
            let params = TimeOfDayParams { start_hour: 13.0, day_minutes: 20.0, clouds, ..Default::default() };
            let tod = TimeOfDaySky::new(params);
            let zenith = tod.atmosphere.eye_transmittance([0.0, 1.0, 0.0]).map(|t| -t.max(1e-6).ln());
            let mut sky = Sky::none(&device, &queue, &layout, AMBIENT);
            sky.apply_snapshot(&device, &queue, &layout, &tod.snapshot(13.0, 0.0), 0.0, zenith);
            sky.tick(&queue, 10.0, 0.1, None);
            let ticked = sky.clouds.as_ref().unwrap().seconds;
            sky.apply_snapshot(&device, &queue, &layout, &tod.snapshot(13.2, params.days_at(10.0)), 10.0, zenith);
            (ticked, sky.clouds.as_ref().unwrap().seconds)
        };
        assert_eq!(run(1.0), (10.0, 10.0));
        assert_eq!(run(3.0), (30.0, 30.0));
    }

    /// THE HEADSET'S CLOUDS OVER TWENTY REAL SECONDS: the worker on, a
    /// snapshot landing every second as the headset's clock lands them, and
    /// a tick and a map slice every 72nd of a second. Prints the population's
    /// clock and generation and where layer 1's first clouds stand, to tell a
    /// still sky from a slow one (user, 2026-10-21: "the clouds don't seem
    /// to move at all"). Turns the worker on for the whole process, so run it
    /// alone: `--ignored --exact`.
    #[test]
    #[ignore]
    fn the_headsets_clouds_move_with_the_worker() {
        let Some((device, queue)) = crate::renderer::terrain_pipeline::tests::headless_gpu() else { return };
        crate::renderer::cloud_worker::enable();
        let layout = sky_bind_group_layout(&device);
        let clouds = space_soup_sky::clouds::CloudParams { coverage: 0.45, wind_speed: 12.0, weather_minutes: 15.0, storms: 0.35, ..Default::default() };
        let params = TimeOfDayParams { start_hour: 16.5, day_minutes: 20.0, clouds, ..Default::default() };
        let tod = TimeOfDaySky::new(params);
        let zenith = tod.atmosphere.eye_transmittance([0.0, 1.0, 0.0]).map(|t| -t.max(1e-6).ln());
        let mut sky = Sky::none(&device, &queue, &layout, AMBIENT);
        let eye = [0.0f32, 1.6, 12.0];
        let hour = |s: f32| 16.5 + s * 24.0 / (20.0 * 60.0);
        let start = std::time::Instant::now();
        let (mut last_apply, mut secs) = (-1.0f32, 0.0f32);
        let mut gens = std::collections::BTreeSet::new();
        let mut first: Option<[f32; 2]> = None;
        for frame in 0..72 * 20 {
            while start.elapsed().as_secs_f32() < frame as f32 / 72.0 {
                std::thread::yield_now();
            }
            let dt = 1.0 / 72.0;
            secs += dt;
            if secs - last_apply >= 1.0 {
                let blend = if last_apply < 0.0 { 0.0 } else { 1.0 };
                sky.apply_snapshot_blended(&device, &queue, &layout, &tod.snapshot(hour(secs), 0.0), secs, zenith, blend);
                last_apply = secs;
            }
            let astro = space_soup_sky::time_of_day::Astronomy::at(&params, hour(secs), 0.0);
            sky.tick(&queue, secs, dt, Some(&astro));
            sky.step_clouds(&device, &queue, eye, true, None, false);
            let c = sky.clouds.as_ref().unwrap();
            gens.insert(c.population.generation());
            let (splats, _) = c.population.splats(1);
            let at = splats.first().map(|s| [s.x, s.z]);
            if frame % 144 == 0 {
                eprintln!(
                    "{secs:5.1} s: population of {:.1} s, generation {}, {} cumulus, first at {:?}",
                    c.population.seconds(),
                    c.population.generation(),
                    splats.len(),
                    at
                );
            }
            if first.is_none() {
                first = at;
            }
        }
        eprintln!("{} populations shown over 20 s", gens.len());
        assert!(gens.len() >= 10, "the sky took only {} populations in 20 s", gens.len());
    }

    /// WHAT THE SKY SHADER SHOWS, FRAME BY FRAME, against the wind: the
    /// headset's loop (the worker on, a snapshot each second, test_room's
    /// clouds on its 2.5x clock at SpaceWarp's 36 Hz), and after each frame
    /// the sky shader's own lookup (`sky_clouds`: both shown maps carried,
    /// read bicubically, blended) over a band of the sky. Per edge patch,
    /// the shift along the wind that best lays this frame's clouds on the
    /// last frame's carried one frame on: 0 when the shown clouds ride their
    /// wind, below 0 where they fall back (user, 2026-10-23, after the carry
    /// fixes: "still had the back step"). Prints `SHOWN` per frame and a
    /// summary. Turns the worker on for the process: `--ignored --exact`.
    #[test]
    #[ignore]
    fn the_shown_clouds_ride_their_wind_frame_by_frame() {
        let Some((device, queue)) = crate::renderer::terrain_pipeline::tests::headless_gpu() else { return };
        crate::renderer::cloud_worker::enable();
        let layout = sky_bind_group_layout(&device);
        let mut clouds = space_soup_sky::clouds::CloudParams { coverage: 0.35, wind_speed: 16.0, weather_minutes: 15.0, storms: 0.35, time_scale: 2.5, altitude_m: 1200.0, ..Default::default() };
        clouds.kinds = [1.0, 1.0, 0.0, 1.0];
        // `SHOWN_KINDS=0,0,0,1`: one layer alone (the cirrus).
        if let Ok(k) = std::env::var("SHOWN_KINDS") {
            for (i, v) in k.split(',').filter_map(|v| v.trim().parse::<f32>().ok()).take(4).enumerate() {
                clouds.kinds[i] = v;
            }
        }
        // `SHOWN_START_HOUR=h`: the run from that hour (19.3 runs through dusk).
        let start_hour: f32 = std::env::var("SHOWN_START_HOUR").ok().and_then(|v| v.parse().ok()).unwrap_or(16.5);
        let params = TimeOfDayParams { start_hour, day_minutes: 20.0, turbidity: 12.0, clouds, ..Default::default() };
        let tod = TimeOfDaySky::new(params);
        let zenith = tod.atmosphere.eye_transmittance([0.0, 1.0, 0.0]).map(|t| -t.max(1e-6).ln());
        let mut sky = Sky::none(&device, &queue, &layout, AMBIENT);
        let eye = [0.0f32, 1.6, 12.0];
        let hz: f32 = std::env::var("SHOWN_HZ").ok().and_then(|v| v.parse().ok()).unwrap_or(36.0);
        let seconds: f32 = std::env::var("SHOWN_SECONDS").ok().and_then(|v| v.parse().ok()).unwrap_or(16.0);
        let hour = |s: f32| start_hour + s * 24.0 / (20.0 * 60.0);
        // `SHOWN_PNG_AT=s,s,...`: the shown band at those seconds, the maps'
        // light over a flat sky of their mean, to `$OUT/shown_<s>.png`.
        let png_at: Vec<f32> = std::env::var("SHOWN_PNG_AT").map(|v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect()).unwrap_or_default();

        // The band shown: every STEP-th column, rows from 8 to 50 degrees up.
        use crate::renderer::cloud_sky::{advected_uv, map_dir, map_uv, MAP_HEIGHT, MAP_WIDTH};
        const STEP: u32 = 2;
        let row_of = |deg: f32| ((deg.to_radians() / std::f32::consts::FRAC_PI_2).powf(1.0 / crate::renderer::cloud_sky::MAP_PACK) * MAP_HEIGHT as f32) as u32;
        let (row0, row1) = (row_of(8.0), row_of(50.0));
        let (cols, rows) = (MAP_WIDTH / STEP, row1 - row0);
        let shader = sky_cloud_shader();
        let cubic = {
            let start = shader.find("fn sky_cloud_map(").unwrap();
            let end = start + shader[start..].find("\n}\n").unwrap() + 2;
            shader[start..end].to_string()
        };
        let source = format!(
            "@group(0) @binding(0) var cloud_maps: texture_2d_array<f32>;
             @group(0) @binding(1) var cloud_samp: sampler;
             @group(0) @binding(2) var cloud_alts: texture_2d_array<f32>;
             struct Shown {{ flow: array<vec4<f32>, 2>, misc: vec4<f32>, eye: vec4<f32>, grid: vec4<f32>, scale: vec4<f32> }}
             @group(0) @binding(3) var<uniform> shown: Shown;
             @group(0) @binding(4) var<storage, read_write> out: array<vec4<f32>>;
             @group(0) @binding(5) var<storage, read_write> out_rgb: array<vec4<f32>>;
             {flow}
             {cubic}
             const SHARED_ALTS: bool = {shared};
             @compute @workgroup_size(64)
             fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
                 let cols = u32(shown.grid.x);
                 if (id.x >= cols * u32(shown.grid.z)) {{ return; }}
                 let uv = vec2<f32>((f32(id.x % cols) * shown.grid.w + 0.5) / {w:.1}, (shown.grid.y + f32(id.x / cols) + 0.5) / {h:.1});
                 let el = 1.57079633 * pow(uv.y, CLOUD_FLOW_PACK);
                 let az = (uv.x - 0.5) * 6.28318531;
                 let dir = vec3<f32>(sin(az) * cos(el), sin(el), cos(az) * cos(el));
                 var uva = uv;
                 var uvb = uv;
                 // As `sky_clouds`; `SHOWN_SHARED_ALTS=1` carries both maps on
                 // the newest's heights and winds, as before 2026-10-23.
                 let here = textureSampleLevel(cloud_alts, cloud_samp, uv, i32(shown.misc.y), 0.0);
                 let older = select(textureSampleLevel(cloud_alts, cloud_samp, uv, i32(shown.misc.x), 0.0), here, SHARED_ALTS);
                 if (older.r > 0.0) {{
                     let p = dir * cloud_flow_distance(shown.eye.x, older.r, dir.y);
                     uva = cloud_map_uv(p + shown.flow[0].xyz - vec3<f32>(older.b, 0.0, older.a) * shown.flow[0].w);
                 }}
                 if (here.r > 0.0) {{
                     let p = dir * cloud_flow_distance(shown.eye.x, here.r, dir.y);
                     uvb = cloud_map_uv(p + shown.flow[1].xyz - vec3<f32>(here.b, 0.0, here.a) * shown.flow[1].w);
                 }}
                 let ma = sky_cloud_map(uva, i32(shown.misc.x));
                 let mb = sky_cloud_map(uvb, i32(shown.misc.y));
                 let a = ma.a;
                 let b = mb.a;
                 out[id.x] = vec4<f32>(mix(a, b, shown.misc.z), here.r, here.b, here.a);
                 out_rgb[id.x] = vec4<f32>(mix(ma.rgb * shown.scale.x, mb.rgb * shown.scale.y, shown.misc.z), mix(a, b, shown.misc.z));
             }}",
            flow = crate::renderer::cloud_sky::cloud_flow_wgsl(),
            shared = std::env::var("SHOWN_SHARED_ALTS").is_ok(),
            w = MAP_WIDTH as f32,
            h = MAP_HEIGHT as f32,
        );
        let module = device.create_shader_module(ShaderModuleDescriptor { label: Some("shown_clouds"), source: ShaderSource::Wgsl(source.into()) });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor { label: None, layout: None, module: &module, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
        let n = (cols * rows) as usize;
        let size = (n * 16) as u64;
        let out_buf = device.create_buffer(&BufferDescriptor { label: None, size, usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC, mapped_at_creation: false });
        let read_buf = device.create_buffer(&BufferDescriptor { label: None, size, usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST, mapped_at_creation: false });
        let rgb_buf = device.create_buffer(&BufferDescriptor { label: None, size, usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC, mapped_at_creation: false });
        let rgb_read = device.create_buffer(&BufferDescriptor { label: None, size, usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST, mapped_at_creation: false });
        let shown_rgb: std::cell::RefCell<Vec<[f32; 4]>> = Default::default();
        let uni = device.create_buffer(&BufferDescriptor { label: None, size: 96, usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST, mapped_at_creation: false });
        let samp = device.create_sampler(&SamplerDescriptor {
            address_mode_u: AddressMode::Repeat,
            address_mode_v: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            ..Default::default()
        });
        let show = |sky: &Sky| -> Vec<[f32; 4]> {
            let gpu = sky.cloud_sky.as_ref().unwrap();
            let e = &sky.extra;
            let mut u = [0.0f32; 24];
            u[20..24].copy_from_slice(&e.cloud_scale);
            u[0..4].copy_from_slice(&e.cloud_flow[0]);
            u[4..8].copy_from_slice(&e.cloud_flow[1]);
            u[8..12].copy_from_slice(&e.cloud_misc);
            u[12..16].copy_from_slice(&e.cloud_eye);
            u[16..20].copy_from_slice(&[cols as f32, row0 as f32, rows as f32, STEP as f32]);
            queue.write_buffer(&uni, 0, bytemuck::cast_slice(&u));
            let bind = device.create_bind_group(&BindGroupDescriptor {
                label: None,
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    BindGroupEntry { binding: 0, resource: BindingResource::TextureView(gpu.maps_view()) },
                    BindGroupEntry { binding: 1, resource: BindingResource::Sampler(&samp) },
                    BindGroupEntry { binding: 2, resource: BindingResource::TextureView(gpu.alts_view()) },
                    BindGroupEntry { binding: 3, resource: uni.as_entire_binding() },
                    BindGroupEntry { binding: 4, resource: out_buf.as_entire_binding() },
                    BindGroupEntry { binding: 5, resource: rgb_buf.as_entire_binding() },
                ],
            });
            let mut enc = device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bind, &[]);
                pass.dispatch_workgroups((n as u32).div_ceil(64), 1, 1);
            }
            enc.copy_buffer_to_buffer(&out_buf, 0, &read_buf, 0, size);
            enc.copy_buffer_to_buffer(&rgb_buf, 0, &rgb_read, 0, size);
            queue.submit(Some(enc.finish()));
            read_buf.slice(..).map_async(MapMode::Read, |_| {});
            rgb_read.slice(..).map_async(MapMode::Read, |_| {});
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            let v: Vec<[f32; 4]> = bytemuck::cast_slice(&read_buf.slice(..).get_mapped_range().unwrap()).to_vec();
            read_buf.unmap();
            *shown_rgb.borrow_mut() = bytemuck::cast_slice(&rgb_read.slice(..).get_mapped_range().unwrap()).to_vec();
            rgb_read.unmap();
            v
        };
        // The shown grid's coordinate of a map uv, and a bilinear read of it.
        let grid_of = |uv: [f32; 2]| [(uv[0] * MAP_WIDTH as f32 - 0.5) / STEP as f32, uv[1] * MAP_HEIGHT as f32 - row0 as f32 - 0.5];
        let read = |g: &[[f32; 4]], p: [f32; 2]| -> f32 {
            let (x0, y0) = (p[0].floor(), p[1].floor().clamp(0.0, (rows - 2) as f32));
            let (fx, fy) = (p[0] - x0, (p[1] - y0).clamp(0.0, 1.0));
            let at = |i: i32, j: i32| g[(j.clamp(0, rows as i32 - 1) as u32 * cols + i.rem_euclid(cols as i32) as u32) as usize][0];
            let (i, j) = (x0 as i32, y0 as i32);
            let top = at(i, j) + (at(i + 1, j) - at(i, j)) * fx;
            let bottom = at(i, j + 1) + (at(i + 1, j + 1) - at(i, j + 1)) * fx;
            top + (bottom - top) * fy
        };

        let start = std::time::Instant::now();
        let (mut last_apply, mut secs) = (-1.0f32, 0.0f32);
        let mut last: Option<(Vec<[f32; 4]>, f64)> = None;
        let (mut medians, mut backs, mut worst_spread) = (Vec::new(), 0usize, 0.0f32);
        let mut pops: Vec<(f32, usize, usize, f32)> = Vec::new();
        for frame in 0..(hz * seconds) as usize {
            while start.elapsed().as_secs_f32() < frame as f32 / hz {
                std::thread::yield_now();
            }
            let dt = 1.0 / hz;
            secs += dt;
            if secs - last_apply >= 1.0 {
                let blend = if last_apply < 0.0 { 0.0 } else { 1.0 };
                sky.apply_snapshot_blended(&device, &queue, &layout, &tod.snapshot(hour(secs), 0.0), secs, zenith, blend);
                last_apply = secs;
            }
            let astro = space_soup_sky::time_of_day::Astronomy::at(&params, hour(secs), 0.0);
            sky.tick(&queue, secs, dt, Some(&astro));
            sky.step_clouds(&device, &queue, eye, true, None, false);
            let now = sky.clouds.as_ref().unwrap().seconds;
            let g = show(&sky);
            if png_at.iter().any(|&t| t > secs - dt && t <= secs) {
                let rgb = shown_rgb.borrow();
                let cloudy: Vec<&[f32; 4]> = rgb.iter().filter(|p| p[3] < 0.5).collect();
                let mean = cloudy.iter().fold([0.0f32; 3], |m, p| [m[0] + p[0], m[1] + p[1], m[2] + p[2]]).map(|v| v / cloudy.len().max(1) as f32);
                let sky_rgb = mean.map(|v| v * 0.8);
                let mut lums: Vec<f32> = rgb.iter().map(|p| p[0] + p[1] + p[2] + p[3] * (sky_rgb[0] + sky_rgb[1] + sky_rgb[2])).collect();
                lums.sort_by(|a, b| a.total_cmp(b));
                let white = lums[(lums.len() as f32 * 0.995) as usize].max(1e-12) / 3.0;
                let mut img = image::RgbImage::new(cols, rows);
                for j in 0..rows {
                    for i in 0..cols {
                        let p = rgb[(j * cols + i) as usize];
                        let c = [0, 1, 2].map(|k| ((p[k] + p[3] * sky_rgb[k]) / white).clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0);
                        img.put_pixel(i, rows - 1 - j, image::Rgb(c.map(|v| v as u8)));
                    }
                }
                let out = std::path::PathBuf::from(std::env::var("OUT").unwrap_or_else(|_| "/tmp".into()));
                img.save(out.join(format!("shown_{secs:05.1}.png"))).unwrap();
                eprintln!("SHOWN png at {secs:.2} s, hour {:.3}", hour(secs));
            }
            if let Some((prev, then)) = last.take().filter(|_| secs > 4.0) {
                let age = (now - then) as f32;
                // POPS: points whose cloud changes by more than 5% of the sky
                // in one frame, followed on their wind (a cloud growing over
                // two minutes changes a thousandth a frame).
                let (mut appear, mut vanish) = (0usize, 0usize);
                for j in 0..rows {
                    for i in 0..cols {
                        let here = g[(j * cols + i) as usize];
                        if here[1] <= 0.0 {
                            continue;
                        }
                        let uv = [((i * STEP) as f32 + 0.5) / MAP_WIDTH as f32, ((row0 + j) as f32 + 0.5) / MAP_HEIGHT as f32];
                        let back = grid_of(advected_uv(map_dir(uv), here[1], eye[1], [here[2], here[3]], [0.0; 3], age));
                        let change = here[0] - read(&prev, back);
                        appear += usize::from(change < -0.05);
                        vanish += usize::from(change > 0.05);
                    }
                }
                pops.push((secs, appear, vanish, sky.cloud_sky.as_ref().unwrap().display()[2]));
                let mut shifts = Vec::new();
                for pj in (0..rows.saturating_sub(16)).step_by(16) {
                    for pi in (0..cols).step_by(32) {
                        let mut texels = Vec::new();
                        let (mut lo, mut hi) = (1.0f32, 0.0f32);
                        for j in pj..pj + 16 {
                            for i in pi..pi + 32 {
                                let here = g[(j * cols + i) as usize];
                                let uv = [((i * STEP) as f32 + 0.5) / MAP_WIDTH as f32, ((row0 + j) as f32 + 0.5) / MAP_HEIGHT as f32];
                                let back = grid_of(advected_uv(map_dir(uv), here[1], eye[1], [here[2], here[3]], [0.0; 3], age));
                                let p = [i as f32, j as f32];
                                let mut fwd = [p[0] - back[0], p[1] - back[1]];
                                if fwd[0] > cols as f32 / 2.0 {
                                    fwd[0] -= cols as f32;
                                } else if fwd[0] < -(cols as f32) / 2.0 {
                                    fwd[0] += cols as f32;
                                }
                                let len = (fwd[0] * fwd[0] + fwd[1] * fwd[1]).sqrt();
                                if here[1] <= 0.0 || len < 1e-4 {
                                    continue;
                                }
                                lo = lo.min(here[0]);
                                hi = hi.max(here[0]);
                                texels.push((here[0], back, [fwd[0] / len, fwd[1] / len]));
                            }
                        }
                        if hi - lo < 0.3 || texels.len() < 256 {
                            continue;
                        }
                        let cost = |s: f32| -> f32 { texels.iter().map(|&(w, b, f)| (read(&prev, [b[0] - s * f[0], b[1] - s * f[1]]) - w).abs()).sum() };
                        let mut best = (cost(0.0), 0.0f32);
                        for k in -40..=40 {
                            let s = k as f32 * 0.1;
                            let c = cost(s);
                            if c < best.0 {
                                best = (c, s);
                            }
                        }
                        shifts.push(best.1);
                    }
                }
                if !shifts.is_empty() {
                    shifts.sort_by(f32::total_cmp);
                    let q = |f: f32| shifts[((shifts.len() - 1) as f32 * f) as usize];
                    let med = q(0.5);
                    backs += usize::from(med < -0.15);
                    worst_spread = worst_spread.max(q(0.9) - q(0.1));
                    medians.push(med);
                    let d = sky.cloud_sky.as_ref().unwrap().display();
                    eprintln!(
                        "SHOWN {secs:6.2} s  map {:.0}->{:.0} at {:.2}  gen {}  {} patches: shift p10 {:+.1} median {:+.2} p90 {:+.1}",
                        d[0], d[1], d[2], sky.clouds.as_ref().unwrap().population.generation(), shifts.len(), q(0.1), med, q(0.9)
                    );
                }
            }
            last = Some((g, now));
        }
        for (t, a, v, z) in pops.iter().filter(|p| p.1 + p.2 > 0) {
            eprintln!("POP {t:6.2} s  build {z:.2}: {a} points gained over 5% cloud in a frame, {v} lost it");
        }
        let (appear, vanish): (usize, usize) = pops.iter().fold((0, 0), |s, p| (s.0 + p.1, s.1 + p.2));
        eprintln!("POP summary: {appear} sudden gains, {vanish} sudden losses over {} frames of {} points", pops.len(), cols * rows);
        let mean = medians.iter().sum::<f32>() / medians.len().max(1) as f32;
        eprintln!("SHOWN summary: {} frames, median shift mean {mean:+.3} texels, {backs} frames with the median below -0.15; worst patch spread {worst_spread:.1}", medians.len());
        assert!(worst_spread <= 0.25, "the shown clouds jump (p90 - p10 of {worst_spread:.1} texels in one frame)");
    }

    /// A new time-of-day snapshot must not change the light the lit shaders
    /// take: on the headset (2026-10-20, 16:55:57) the whole beach lost its
    /// sun in the frame a snapshot landed. Prints the light per frame
    /// either side of one.
    #[test]
    fn a_snapshot_landing_does_not_step_the_suns_light() {
        let Some((device, queue)) = crate::renderer::terrain_pipeline::tests::headless_gpu() else { return };
        let layout = sky_bind_group_layout(&device);
        let clouds = space_soup_sky::clouds::CloudParams { coverage: 0.45, ..Default::default() };
        let tod = TimeOfDaySky::new(TimeOfDayParams { start_hour: 17.6, clouds, ..Default::default() });
        let zenith = tod.atmosphere.eye_transmittance([0.0, 1.0, 0.0]).map(|t| -t.max(1e-6).ln());
        let mut sky = Sky::none(&device, &queue, &layout, AMBIENT);
        let size = crate::renderer::uniforms::CLOUD_SHADOW_SIZE;
        let shadow = device.create_texture(&TextureDescriptor {
            label: None,
            size: Extent3d { width: size, height: size, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::R8Unorm,
            usage: TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let view = shadow.create_view(&Default::default());
        let eye = [3.0f32, 1.6, 22.0];
        let mut secs = 600.0f32;
        let hour = |s: f32| 17.6 + (s - 600.0) * 0.02;
        sky.apply_snapshot_blended(&device, &queue, &layout, &tod.snapshot(hour(secs), 0.0), secs, zenith, 0.0);
        let mut seen: Vec<(f32, f32, f32, f32)> = Vec::new();
        for frame in 0..300 {
            let dt = 1.0 / 72.0;
            secs += dt;
            if frame == 150 {
                sky.apply_snapshot_blended(&device, &queue, &layout, &tod.snapshot(hour(secs), 0.0), secs, zenith, 2.0);
            }
            let params = TimeOfDayParams { start_hour: 17.6, clouds, ..Default::default() };
            let astro = space_soup_sky::time_of_day::Astronomy::at(&params, hour(secs), 0.0);
            sky.tick(&queue, secs, dt, Some(&astro));
            sky.step_clouds(&device, &queue, eye, true, Some(&view), false);
            let direct = sky.sun_direct().map_or(0.0, |s| s.light_rgb[1]);
            let level = sky.sun.map_or(0.0, |s| s.light_rgb[1]);
            let c = sky.clouds.as_ref().unwrap();
            let t = space_soup_sky::cloud_layers::transmittance_penumbra(&c.layers, &CloudNoise::shared(), c.weather(), [eye[0], eye[2]], eye[1], sky.cloud_light_dir().unwrap());
            seen.push((secs, direct, level, t));
        }
        // The change in the light reaching the eye as a share of the clear
        // sun's, frame to frame (a relative change blows up in deep shade).
        let worst = seen.windows(2).map(|w| (w[1].1 * w[1].3 - w[0].1 * w[0].3).abs() / w[0].1.max(1e-9)).enumerate().fold((0, 0.0f32), |a, (i, v)| if v > a.1 { (i, v) } else { a });
        let full: f32 = seen.windows(2).map(|w| (w[1].3 - w[0].3).abs()).sum();
        eprintln!("the light at the eye travelled {full:.2} (sum of |change|) over {} frames", seen.len());
        for (i, s) in seen.iter().enumerate().filter(|(i, _)| (*i as i32 - 150).abs() <= 2 || (*i as i32 - worst.0 as i32).abs() <= 4) {
            eprintln!("frame {i}: direct {:.4} level {:.4} through at eye {:.4}", s.1, s.2, s.3);
        }
        eprintln!("worst frame-to-frame change of the light on the eye: {:.3} at frame {}", worst.1, worst.0);
        // A shade sweeping in over 2 s at 72 Hz moves 0.007 a frame.
        assert!(worst.1 < 0.02, "the light on the eye steps by {:.3} at frame {}", worst.1, worst.0);
    }

    fn snap_px(dir: [f32; 3], snap: &SkySnapshot, zenith: [f32; 3], fov: f32, post: PostUpload, twin: bool) -> Option<[u8; 4]> {
        let snap = snap.clone();
        render_centre_with(dir, 0.0, fov, post, move |device, queue, layout| {
            let mut sky = Sky::none(device, queue, layout, AMBIENT);
            sky.apply_snapshot(device, queue, layout, &snap, 0.0, zenith);
            sky.step_clouds(device, queue, [0.0, 0.0, 0.0], false, None, true);
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

    /// The sky shader shows the cloud map: the clear sky's own pixels in
    /// the gaps, and cloud -- not sky -- where the map holds one. (The
    /// clouds' density is held to its CPU twin by `cloud_sky`'s test.)
    #[test]
    fn the_cloud_shader_shows_cloud_where_the_map_has_it_and_sky_between() {
        let (snap, zenith) = cloud_snapshot(0.5);
        let mut clear_snap = snap.clone();
        clear_snap.clouds = None;
        clear_snap.cloud_field = None;
        let post = PostUpload { exposure: 4.0, ..Default::default() };
        let (mut gaps, mut clouds) = (0, 0);
        for i in 0..24 {
            let a = i as f32 * 0.61;
            let up = 0.35 + 0.6 * ((i * 7) % 11) as f32 / 11.0;
            let h = (1.0 - up * up).sqrt();
            let d = glam::Vec3::new(h * a.cos(), up, h * a.sin()).normalize();
            if d.dot(glam::Vec3::from(snap.discs.sun_dir)) > 0.99 {
                continue;
            }
            let sky = shot!(snap_px(d.to_array(), &clear_snap, zenith, 0.01, post, false));
            let got = shot!(snap_px(d.to_array(), &snap, zenith, 0.01, post, false));
            let diff = (0..3).map(|c| (sky[c] as i32 - got[c] as i32).abs()).max().unwrap();
            if diff <= 2 {
                gaps += 1;
            } else if diff >= 20 {
                clouds += 1;
            }
        }
        assert!(gaps >= 3 && clouds >= 3, "{gaps} gaps and {clouds} clouds in 24 directions");
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
        // A budget of ours, not the cliff: the I-cache cliff measured on this
        // proxy is near 2424 (water), and the clouded sky is ~1316 in all.
        // 400 -> 420 on 2026-10-21 for the discs' own cloud transmittance
        // (+17, read only near a disc), which put the sun back behind far
        // cloud on the horizon.
        assert!(clouded < clear + 420, "the cloud code adds {} (283 when written, 2026-10-19; 401 on 2026-10-21)", clouded - clear);
    }
}
