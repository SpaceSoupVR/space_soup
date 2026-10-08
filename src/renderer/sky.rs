//! The sky: the background behind a level, and the ambient light in it.
//!
//! WHY BOTH JOBS BELONG TO ONE OBJECT
//!
//! An HDRI is one equirectangular panorama carrying real intensity rather than
//! clamped colour, and it does the background and the lighting at once. That is
//! the whole reason the format replaced a six-sided skybox: there, the sky was a
//! picture behind the level and lit nothing, so "the lighting does not match the
//! sky" was a permanent class of bug. Here they are the same data and cannot
//! disagree.
//!
//! The editor has had this for a while. Nothing in the renderer, the client, the
//! protocol or the server knew what a sky was, so a level authored against a
//! panorama arrived on the headset with a dark blue clear colour behind it and a
//! flat 0.6 ambient inside it.
//!
//! LIGHTING IS SPHERICAL HARMONICS, NOT A TEXTURE
//!
//! Diffuse irradiance from an environment is a very smooth function of the
//! surface normal -- so smooth that nine coefficients reproduce it to within a
//! percent or so, which is the standard result. Projecting once at load and
//! evaluating in the shader costs about twenty instructions and NO texture
//! fetch, against a cube sample per pixel for the obvious alternative.
//!
//! On a tile GPU that distinction is the whole argument. Bandwidth is the scarce
//! resource; arithmetic is not. Nine `vec4`s in the uniform buffer we already
//! bind cost nothing per pixel at all.
//!
//! Specular from the environment is deliberately absent. Doing it properly needs
//! a prefiltered mip chain and a BRDF lookup, which is a real amount of memory
//! and two more fetches per pixel, and the engine's Blinn-Phong highlight from
//! actual lights already covers the case people notice.
//!
//! THE DIRECTION MAPPING IS MEASURED, NOT ASSUMED
//!
//! `direction_to_uv` below is pinned to the editor's: a sun at azimuth 34.3 and
//! elevation 48 in the lobby's panorama lands on texel (609, 119), which is
//! exactly the brightest texel in that file. Getting this wrong puts the light
//! somewhere other than the visible sun, and a level lit from the wrong side
//! looks like a lighting bug rather than a mapping one.

//! Sky panorama loading and the environment it lights the scene with.
//!
//! The MODEL -- the panorama, the harmonics, and the projection between them --
//! now lives in `space_soup_sky` so the baker evaluates exactly the same sky
//! this shader does. Two implementations of one physical quantity is how a
//! baked reflection ended up brighter than the surface it reflected.

pub use space_soup_sky::{
    decode_radiance, project_irradiance, sky_lighting, uv_to_direction, Panorama, SkyIrradiance,
    SkySun,
};
pub use space_soup_sky::time_of_day::{AtlasLayers, LayerWeights, SkySnapshot, TimeOfDayParams, TimeOfDaySky};


use anyhow::{bail, Result};
use wgpu::*;

use super::lights::wgsl_lights_block;

/// The flat ambient a scene without a sky gets, matching the shader constant.
pub const AMBIENT: f32 = 0.6;






/// Where a world direction lands in an equirectangular panorama.
///
/// MEASURED AGAINST THE EDITOR, not derived from a convention. The lobby's
/// panorama has its sun at azimuth 34.3 / elevation 48 in world terms -- found
/// by scanning the rendered sky for its brightest direction -- and this maps
/// that to texel (609, 119), which is the brightest texel in the file itself.
/// Two independent measurements of the same thing.
pub fn direction_to_uv(d: [f32; 3]) -> [f32; 2] {
    let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt().max(1e-6);
    let (x, y, z) = (d[0] / len, d[1] / len, d[2] / len);
    let u = (x.atan2(z) + std::f32::consts::PI) / (2.0 * std::f32::consts::PI);
    let v = y.clamp(-1.0, 1.0).acos() / std::f32::consts::PI;
    [u, v]
}






/// Half-precision, for the panorama texture.
///
/// `Rgba16Float` rather than `Rgba8UnormSrgb`: a panorama carries values well
/// above 1 and that is the entire point of it. Storing the background as sRGB
/// bytes would clamp the sun to white and make the buffer disagree with the
/// coefficients projected from the same file.
pub fn f32_to_f16(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let mut exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mant = bits & 0x7f_ffff;
    if exp >= 0x1f {
        return sign | 0x7c00; // infinity, or a NaN flushed to one
    }
    if exp <= 0 {
        return sign; // subnormal, flushed to zero
    }
    if exp > 0x1e {
        exp = 0x1e;
    }
    sign | ((exp as u16) << 10) | ((mant >> 13) as u16)
}

/// WHAT THE SKY SHADER DRAWS OVER THE PANORAMA in a time-of-day sky: the
/// sun's and the moon's discs and the stars. All zero for a photographed sky,
/// whose sun is in the picture and whose night never comes -- `mode.x` 0 is
/// the shader's old path, exactly. Group 1, binding 2. World directions.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SkyExtra {
    /// x: 1 = a time-of-day sky; y: what the panorama is multiplied by to read
    /// engine units (it is stored normalised, see `SkySnapshot::panorama_scale`);
    /// z: seconds, for the stars' twinkle; w: 1 = stars could show at all.
    pub mode: [f32; 4],
    /// xyz toward the sun, w its angular radius.
    pub sun: [f32; 4],
    /// The disc's centre radiance after the air (engine), w the limb darkening.
    pub sun_radiance: [f32; 4],
    /// xyz toward the moon, w its angular radius.
    pub moon: [f32; 4],
    /// The FULL moon's surface radiance after the air, w earthshine (a
    /// fraction of it).
    pub moon_radiance: [f32; 4],
    /// xyz the way the moon's lit side faces (toward the sun from the moon).
    pub moon_sunward: [f32; 4],
    /// x a magnitude-0 star's illuminance (engine), y the faintest magnitude,
    /// z how many stars to it on the whole sphere, w twinkle strength.
    pub stars: [f32; 4],
    /// xyz the zenith's optical depth (what dims a star by `exp(-tau X)`).
    pub extinction: [f32; 4],
    /// World -> celestial: rows, the celestial axes in the world. Row 2 is
    /// the north celestial pole, which the moon's north follows.
    pub celestial: [[f32; 4]; 3],
}

impl SkyExtra {
    /// What `snap` draws, `seconds` into the level (the twinkle's clock).
    pub fn from_snapshot(snap: &SkySnapshot, seconds: f32, zenith_depth: [f32; 3]) -> Self {
        let d = &snap.discs;
        let v4 = |v: [f32; 3], w: f32| [v[0], v[1], v[2], w];
        // STARS ONLY WHEN THEY COULD SHOW: the brightest (magnitude -1.5)
        // against the sky's mean, spread over a 0.05-degree pixel. By day the
        // sky outshines it a thousandfold and the shader skips the field
        // uniformly -- its whole cost.
        let mean_sky = 1.0 / snap.panorama_scale.max(1e-30);
        let px = (0.05f32).to_radians();
        let brightest = d.star_zero_point * 10f32.powf(0.6) / (2.0 * std::f32::consts::PI * (0.5 * px).powi(2));
        let stars_show = d.star_count > 0.0 && brightest > 1e-3 * mean_sky;
        Self {
            mode: [1.0, 1.0 / snap.panorama_scale.max(1e-30), seconds, if stars_show { 1.0 } else { 0.0 }],
            sun: v4(d.sun_dir, space_soup_sky::time_of_day::SUN_ANGULAR_RADIUS),
            sun_radiance: v4(d.sun_radiance, 0.6),
            moon: v4(d.moon_dir, space_soup_sky::time_of_day::MOON_ANGULAR_RADIUS),
            moon_radiance: v4(d.moon_radiance, d.earthshine),
            moon_sunward: v4(d.moon_sunward, 0.0),
            stars: [d.star_zero_point, d.limiting_magnitude, d.star_count, 1.0],
            extinction: v4(zenith_depth, 0.0),
            celestial: [v4(d.celestial_rows[0], 0.0), v4(d.celestial_rows[1], 0.0), v4(d.celestial_rows[2], 0.0)],
        }
    }
}

/// The panorama on the GPU, plus the coefficients projected from it.
pub struct Sky {
    pub bind_group: BindGroup,
    /// The ambient, WITHOUT the sun when the sky has one -- see `sun`.
    pub irradiance: SkyIrradiance,
    /// The sky's sun, taken out of `irradiance` by `space_soup_sky::sky_lighting`
    /// and handed back as one directional light, in WORLD space.
    ///
    /// Left in the harmonics, half this sky's light was a glow over the whole
    /// sun-facing hemisphere: no shadow, no sunlit patch through a door, and
    /// a surface facing away from it lit by the harmonics' ringing. The baker
    /// splits the sky with the same function and bakes this sun into the
    /// level's lightmaps, so the renderer shades it only on what has none.
    /// The panorama the player SEES keeps its sun; only the lighting moves.
    pub sun: Option<SkySun>,
    /// The sky REFLECTIONS show: the panorama without its sun, turned and
    /// scaled as the scene shows it. `None` for no sky. See [`ReflectionSky`].
    pub reflection: Option<ReflectionSky>,
    /// What the shader draws over the panorama; zero for a photographed sky.
    /// See [`SkyExtra`].
    pub extra: SkyExtra,
    extra_buffer: Buffer,
    texture: Texture,
    sampler: Sampler,
}

/// THE SKY A REFLECTION SEES: the panorama with its sun taken out (the sun
/// reaches every surface as a light, and its highlight is that light's --
/// reflected here as well it would be counted twice), with the scene's
/// rotation and intensity.
///
/// Reflections read it from a layer of the probe array, built at the probes'
/// size by [`Self::cube_faces`] and blurred by the same GGX chain, so a rough
/// surface blurs the sky exactly as much as a photograph. Until 2026-09-28 a
/// reflection that left the building read the sky's nine harmonics instead: the
/// floor mirrored the front door as a flat pale patch where the real sky has
/// clouds (headset 21:46:20).
#[derive(Clone)]
pub struct ReflectionSky {
    pub pano: Panorama,
    pub rotation_deg: f32,
    pub intensity: f32,
}

impl ReflectionSky {
    /// The sky reflections see for this panorama: its sun taken out exactly as
    /// `sky_lighting` takes it out, so the two agree about where the sun was.
    pub fn new(pano: &Panorama, rotation_deg: f32, intensity: f32) -> Self {
        let pano = match space_soup_sky::extract_sun(pano, rotation_deg, intensity) {
            Some((_, rest)) => rest,
            None => pano.clone(),
        };
        Self { pano, rotation_deg, intensity }
    }

    /// Radiance toward WORLD direction `d`: bilinear, wrapping in longitude.
    pub fn radiance(&self, d: glam::Vec3) -> [f32; 3] {
        let (rc, rs) = (self.rotation_deg.to_radians().cos(), self.rotation_deg.to_radians().sin());
        // The inverse of the rotation `project_irradiance` gives the picture:
        // world back to panorama.
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

    /// Six cube faces at `res`, RGBA half floats in the probe cube's layout
    /// (`probe_prefilter::texel_direction`), ready for `prefilter_probe`. Each
    /// texel averages a 2x2 of samples, so a panorama finer than the cube is
    /// not aliased into it.
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
        ],
    })
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
            // REPEAT across u, clamp down v. A panorama wraps in longitude and
            // emphatically does not in latitude -- wrapping there samples the
            // sky when looking at the ground.
            address_mode_u: AddressMode::Repeat,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            ..Default::default()
        });

        // `sky_lighting`, with the sun-free panorama kept for reflections: one
        // extraction, so the two cannot disagree about where the sun was.
        let (irradiance, sun, without_sun) = match space_soup_sky::extract_sun(pano, rotation_deg, intensity) {
            Some((sun, rest)) => (project_irradiance(&rest, rotation_deg, intensity), Some(sun), rest),
            None => (project_irradiance(pano, rotation_deg, intensity), None, pano.clone()),
        };
        let extra = SkyExtra::default();
        let extra_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("sky_extra"),
            size: std::mem::size_of::<SkyExtra>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&extra_buffer, 0, bytemuck::bytes_of(&extra));
        let bind_group = Self::bind(device, layout, &texture, &sampler, &extra_buffer);

        Self {
            bind_group,
            irradiance,
            sun,
            reflection: Some(ReflectionSky { pano: without_sun, rotation_deg, intensity }),
            extra,
            extra_buffer,
            texture,
            sampler,
        }
    }

    fn bind(device: &Device, layout: &BindGroupLayout, texture: &Texture, sampler: &Sampler, extra: &Buffer) -> BindGroup {
        let view = texture.create_view(&TextureViewDescriptor::default());
        device.create_bind_group(&BindGroupDescriptor {
            label: Some("sky_bg"),
            layout,
            entries: &[
                BindGroupEntry { binding: 0, resource: BindingResource::TextureView(&view) },
                BindGroupEntry { binding: 1, resource: BindingResource::Sampler(sampler) },
                BindGroupEntry { binding: 2, resource: extra.as_entire_binding() },
            ],
        })
    }

    /// THE TIME OF DAY'S SKY AT ONE MOMENT, in place of whatever was shown:
    /// the panorama (normalised, as `snap` stores it), the ambient, the ONE
    /// directional light (the sun, or the moon after it), the sky reflections
    /// see, and the discs and stars. Everything downstream -- the lights block,
    /// the static sun map (redrawn when the direction moves), the meter, the
    /// water's and the effects' sun -- reads these same fields, so nothing
    /// else has to know the sky moved.
    ///
    /// `zenith_depth` is the atmosphere's optical depth straight up, which
    /// dims the stars. Cheap: a 256 x 128 half-float upload and 176 bytes.
    pub fn apply_snapshot(&mut self, device: &Device, queue: &Queue, layout: &BindGroupLayout, snap: &SkySnapshot, seconds: f32, zenith_depth: [f32; 3]) {
        let pano = &snap.panorama;
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
            self.bind_group = Self::bind(device, layout, &self.texture, &self.sampler, &self.extra_buffer);
        }
        write_panorama(queue, &self.texture, pano);
        self.irradiance = snap.irradiance;
        self.sun = snap.light;
        self.reflection = Some(ReflectionSky { pano: snap.panorama_engine(), rotation_deg: 0.0, intensity: 1.0 });
        self.extra = SkyExtra::from_snapshot(snap, seconds, zenith_depth);
        queue.write_buffer(&self.extra_buffer, 0, bytemuck::bytes_of(&self.extra));
    }

    /// The stars' twinkle clock, every frame: four bytes.
    pub fn set_clock(&mut self, queue: &Queue, seconds: f32) {
        if self.extra.mode[0] > 0.5 {
            self.extra.mode[2] = seconds;
            queue.write_buffer(&self.extra_buffer, 0, bytemuck::bytes_of(&self.extra.mode));
        }
    }

    /// A scene with no sky: one black texel, and the flat ambient as before.
    ///
    /// Always bound, like terrain's placeholder layers. An optional binding
    /// would mean two bind group layouts and therefore two pipelines, and this
    /// costs one texel.
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
        // No sky to reflect: the flat ambient's harmonics answer instead.
        s.reflection = None;
        s
    }
}

/// A panorama into its half-float texture, the same size.
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

    /// The same, drawing BOTH EYES in one pass. See `multiview::ViewMode`.
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
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("sky_shader"),
            source: ShaderSource::Wgsl(view.shader(sky_shader()).into()),
        });
        let layout = sky_bind_group_layout(device);
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("sky_layout"),
            bind_group_layouts: &[Some(uniform_layout), Some(&layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("sky_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: PipelineCompilationOptions::default(),
                // No vertex buffer: three vertices generated from their index.
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
                // DRAWN LAST, AND WRITES NO DEPTH.
                //
                // The sky sits at the far plane and is only visible where
                // nothing else was drawn, so it goes after the opaques with
                // `LessEqual` and early-Z rejects every covered pixel. Drawing
                // it FIRST would shade every one of those pixels and then throw
                // the work away, which on a fill-limited tile GPU is the whole
                // cost of the pass for nothing.
                depth_write_enabled: Some(false),
                depth_compare: Some(CompareFunction::LessEqual),
                stencil: StencilState::default(),
                bias: DepthBiasState::default(),
            }),
            multisample: MultisampleState { count: samples, ..Default::default() },
            multiview_mask: view.mask(),
            cache: None,
        });

        Self { pipeline, layout }
    }
}

/// Paint the sky SOLID MAGENTA, to find gaps in level geometry.
///
/// The seams along the front ceiling/wall junctions of test_room are dotted
/// specks 5x brighter than the dark interior on either side of them (35/39/31
/// against 7/6/4, normal view, 2026-09-22) -- brighter than anything lighting
/// that corner. Either the sky is showing through pixel-sized gaps where the
/// geometry does not quite meet, or something in the shading (the reflection
/// probe, which photographed the bright doorway) is lighting those pixels.
///
/// The probe is baked offline and cannot turn magenta; the sky can. So a
/// magenta speck on the junction is a HOLE, and a bright non-magenta one is
/// shading. One look decides which of two very different fixes is needed.
///
/// Must be `false` in anything shipped; `the_crack_diagnostic_is_off` asserts
/// it.
pub const SKY_CRACK_DEBUG: bool = false;

fn sky_shader() -> String {
    format!(
        r#"
{lights_block}

// Group 1: the panorama. Group 0 comes from the lights block above, which is
// where `sky_uv` and the irradiance coefficients live.
@group(1) @binding(0) var sky_tex: texture_2d<f32>;
@group(1) @binding(1) var sky_samp: sampler;

// What a time-of-day sky draws over its panorama. See `SkyExtra`.
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

// One oversized triangle covering the target, from the vertex index alone.
@vertex fn vs_main(@builtin(vertex_index) vi: u32) -> VOut {{
    var p = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    var out: VOut;
    // z = 1: the far plane. With depth_compare LessEqual and no depth write,
    // this passes only where nothing nearer was drawn.
    out.clip = vec4<f32>(p[vi], 1.0, 1.0);
    out.ndc = p[vi];
    return out;
}}

const SKY_PI: f32 = 3.14159265;
// Star cells along a cube face's side, angle-even. A cell is ~0.6 degrees:
// at the headset's ~0.05-degree pixels a star's footprint stays well inside
// the 2 x 2 cells read round it.
const STAR_CELLS: f32 = {star_cells:.1};
// The moon's brightest part after exposure when the eye is on it: ACES puts
// 2.0 at ~0.8, so the seas (0.6 of the highlands) still read. See `sky_moon`.
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

// A point on face `face` at face coordinates `uv` (-1..1), as a direction.
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

// THE STARS: a field fixed on the celestial sphere, as many to each magnitude
// as the real sky has (x ~3 a magnitude to the limit), each drawn as a
// Gaussian no narrower than half a pixel and carrying its whole light at any
// size -- a star narrower than a pixel falls between pixel centres and
// vanishes otherwise. Dimmed by the air (Kasten-Young airmass) and twinkling
// only where the eye looks through a lot of it, near the horizon.
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
        // The cell's solid angle: an angle-even cell over a cube face.
        let c = tan(((cell + 0.5) / STAR_CELLS * 2.0 - 1.0) * (SKY_PI * 0.25));
        let r2 = 1.0 + dot(c, c);
        let omega = cell_angle * cell_angle * (1.0 + c.x * c.x) * (1.0 + c.y * c.y) / (r2 * sqrt(r2));
        let p = sky_extra.stars.z * omega / (4.0 * SKY_PI);
        if (h.x >= p) {{
            continue;
        }}
        // Given a star, h.x / p is uniform: the magnitude from the counts,
        // brighter ones rarer, none brighter than Sirius.
        let m = max(sky_extra.stars.y + log2(max(h.x / p, 1e-6)) * (0.30103 / 0.47), -1.5);
        let at = star_face_dir(face, tan(((cell + h.yz) / STAR_CELLS * 2.0 - 1.0) * (SKY_PI * 0.25)));
        let off = length(cross(q, at));
        let flux = sky_extra.stars.x * exp2(-1.3287712 * m);
        let spread = flux / (2.0 * SKY_PI * sigma * sigma) * exp(-off * off / (2.0 * sigma * sigma));
        // Colour from a temperature drawn toward the sun-like middle.
        let t = h.w;
        let colour = mix(vec3<f32>(1.18, 0.95, 0.72), vec3<f32>(0.82, 0.92, 1.25), t * t * (3.0 - 2.0 * t));
        let tw = 1.0 + twinkle_depth * (0.6 * sin(sky_extra.mode.z * (11.0 + 7.0 * h.y) + 40.0 * h.z) + 0.4 * sin(sky_extra.mode.z * (23.0 + 9.0 * h.z) + 17.0 * h.y));
        sum = sum + spread * colour * tw;
    }}
    return sum * dimmed;
}}

// The moon's dark seas, as the near side shows them (selenographic latitude,
// longitude, radius in degrees; lunar north up, east to the right).
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
    // Tycho's bright rays, in the south.
    let ty = vec3<f32>(cos(-0.75) * sin(-0.19), sin(-0.75), cos(-0.75) * cos(-0.19));
    a = a + 0.25 * (1.0 - smoothstep(1.0, 4.0, acos(clamp(dot(p, ty), -1.0, 1.0)) * (180.0 / SKY_PI)));
    // The disc's mean comes back to 1, so the light it sends is the
    // measured light.
    return a * 1.18;
}}

// THE MOON: a sphere lit from `moon_sunward`, shaded Lommel-Seeliger (a dusty
// surface: the full moon is a flat disc, not a ball), its seas, earthshine on
// the dark part, both edges anti-aliased to the pixel. Returns (radiance,
// coverage): what is behind it -- the stars -- is hidden by coverage.
fn sky_moon(d: vec3<f32>, px: f32) -> vec4<f32> {{
    let m = sky_extra.moon.xyz;
    let rho = sky_extra.moon.w;
    let along = dot(d, m);
    if (along < cos(rho + 3.0 * px) || m.y < -0.02) {{
        return vec4<f32>(0.0);
    }}
    let pole = sky_extra.celestial[2].xyz;
    let up = normalize(pole - m * dot(pole, m));
    let right = cross(m, up);
    let xy = vec2<f32>(dot(d, right), dot(d, up)) / sin(rho);
    let r = length(xy);
    let e = px / rho;
    let cover = 1.0 - smoothstep(1.0 - e, 1.0 + e, r);
    let mu = sqrt(max(1.0 - min(r * r, 1.0), 0.0));
    let n = right * xy.x + up * xy.y - m * mu;
    let mu0 = dot(n, sky_extra.moon_sunward.xyz);
    let lit = smoothstep(-e, e, mu0) * 2.0 * max(mu0, 0.0) / (max(mu0, 0.0) + mu + 1e-4);
    let albedo = moon_albedo(vec3<f32>(xy.x, xy.y, mu));
    let surface = sky_extra.moon_radiance.rgb * albedo * (lit + sky_extra.moon_radiance.w);
    // THE EYE ON THE MOON: a small bright thing looked at is seen in its own
    // light -- the eye adapts to it as to a lamp's bulb
    // (`tonemap::bulb_adaptation`) -- so at night its seas show instead of a
    // white dot ~2,000 times the adapted sky. Its brightest part is brought to
    // `MOON_SEEN` after exposure, never brighter than it is. Only the disc:
    // the moon's light on the scene is the directional light, unscaled.
    let rgb = sky_extra.moon_radiance.rgb;
    let peak = max(max(rgb.r, rgb.g), rgb.b) * 1.3 * max(camera.post_params.x, 0.0);
    let seen = min(1.0, MOON_SEEN / max(peak, 1e-12));
    return vec4<f32>(surface * cover * seen, cover);
}}

// THE SUN'S DISC, limb-darkened, after the air.
fn sky_sun_disc(d: vec3<f32>, px: f32) -> vec3<f32> {{
    let s = sky_extra.sun.xyz;
    let rho = sky_extra.sun.w;
    let along = dot(d, s);
    if (along < cos(rho + 3.0 * px)) {{
        return vec3<f32>(0.0);
    }}
    let r = length(cross(d, s)) / sin(rho);
    let e = px / rho;
    let cover = 1.0 - smoothstep(1.0 - e, 1.0 + e, r);
    let mu = sqrt(max(1.0 - min(r * r, 1.0), 0.0));
    return sky_extra.sun_radiance.rgb * (1.0 - sky_extra.sun_radiance.w * (1.0 - mu)) * cover;
}}

@fragment fn fs_main(in: VOut) -> @location(0) vec4<f32> {{
    // The view ray, recovered by putting the pixel back through the inverse of
    // view_proj. Two points on the ray rather than one, because the near point
    // is where the eye is and the difference is the direction -- which is what
    // makes this correct for an off-centre projection, and every headset's is.
    let near = cam_inv_view_proj() * vec4<f32>(in.ndc, 0.0, 1.0);
    let far  = cam_inv_view_proj() * vec4<f32>(in.ndc, 1.0, 1.0);
    let dir = normalize(far.xyz / far.w - near.xyz / near.w);

    // That ray is in the PLAYER's frame, the one the camera's matrices are
    // in; the panorama is pinned to the world. Turned back by the rig's yaw
    // first, or a snap or stick turn carried the sky round with the player
    // while the level and its lighting stayed put (headset, 2026-10-01).
    let world_dir = to_world_direction(dir);
    // One pixel's angle, taken here, in uniform control flow.
    let px = max(length(dpdx(world_dir)), length(dpdy(world_dir)));
    let uv = sky_uv(world_dir);
    var radiance = textureSample(sky_tex, sky_samp, uv).rgb * camera.sky_params.x;
    // A TIME-OF-DAY SKY: the panorama is stored normalised; the sun's and the
    // moon's discs and the stars go over it. A photographed sky (mode 0) is
    // exactly the old path.
    if (sky_extra.mode.x > 0.5) {{
        radiance = radiance * sky_extra.mode.y + sky_sun_disc(world_dir, px);
        let moon = sky_moon(world_dir, px);
        radiance = radiance + moon.rgb;
        if (sky_extra.mode.w > 0.5 && world_dir.y > 0.0) {{
            radiance = radiance + sky_stars(world_dir, px) * (1.0 - moon.a);
        }}
    }}

    // The shared curve, the same one every lit surface uses. This pass used to
    // run its own local Reinhard because nothing downstream tone mapped -- which
    // meant the sky and the geometry in front of it disagreed about how
    // highlights roll off, and the seam showed at the horizon.
    {sky_tail}
}}
"#,
        lights_block = wgsl_lights_block(0, 1),
        star_cells = STAR_CELLS,
        sky_tail = if SKY_CRACK_DEBUG {
            "return vec4<f32>(1.0, 0.0, 1.0, 1.0);"
        } else {
            "return vec4<f32>(tonemap(radiance), 1.0);"
        },
    )
}

/// Star cells along each cube face of the celestial sphere. See the shader's
/// `sky_stars`; [`star_count_in_cells`] counts what it draws.
pub const STAR_CELLS: f32 = 160.0;

/// How many stars the shader's field holds over the whole sphere for a
/// `star_count` -- its cells' chances summed, the CPU twin of `sky_stars`'s
/// `p`, so the count the sky draws is the count the catalogue says.
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

    /// The shader's star field holds as many stars as the catalogue law asks
    /// for: no cell is ever asked for more than one at the counts the sky uses
    /// (9,100 naked-eye stars over ~154k cells), so none is lost.
    #[test]
    fn the_star_field_holds_the_catalogues_count() {
        for n in [100.0f32, 2_000.0, 9_100.0, 20_000.0] {
            let got = star_count_in_cells(n);
            assert!((got / n - 1.0).abs() < 0.01, "{n} stars asked, the field holds {got}");
        }
        // Far past one a cell, the field saturates rather than inventing stars.
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
        assert_eq!((pano.width, pano.height), (1024, 512));
        assert_eq!(pano.rgb.len(), (1024 * 512 * 3) as usize);
        assert!(pano.rgb.iter().all(|v| v.is_finite() && *v >= 0.0));
    }

    #[test]
    fn a_panorama_carries_values_above_one() {
        // The entire reason for the format. If a decode clamped -- or if this
        // were loaded as 8-bit -- the sun would be the same brightness as a
        // white cloud and the projection below would be meaningless.
        let Some(pano) = lobby_sky() else {
            eprintln!("skipping: the lobby's sky is not installed");
            return;
        };
        let peak = pano.rgb.iter().cloned().fold(0.0f32, f32::max);
        assert!(peak > 100.0, "peak radiance {peak} is too low to be HDR");
    }

    #[test]
    fn the_suns_texel_is_where_the_direction_mapping_says_it_is() {
        // THE PIN. The lobby's sun was measured in the running editor at
        // azimuth 34.3, elevation 48 -- by scanning the rendered sky for its
        // brightest direction. This asserts the mapping agrees, so the light
        // the engine computes comes from where the sky visibly shows it.
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
        // END TO END on the file a level actually uses. The synthetic tests
        // above prove the maths; this proves the maths applied to real data
        // gives a sane answer, which is the part that would otherwise only be
        // discovered by putting a headset on.
        let Some(pano) = lobby_sky() else {
            eprintln!("skipping: the lobby's sky is not installed");
            return;
        };
        let sh = project_irradiance(&pano, 0.0, 1.0);

        // An outdoor panorama is far brighter above than below: a surface
        // facing up must receive more than one facing down. If the latitude
        // mapping were flipped this is the assertion that notices.
        let up = sh.evaluate([0.0, 1.0, 0.0]);
        let down = sh.evaluate([0.0, -1.0, 0.0]);
        assert!(
            up[0] > down[0] * 1.5,
            "the sky is not brighter above than below: up {up:?}, down {down:?}",
        );

        // And it is brighter toward the sun than away from it. The lobby's sun
        // is at azimuth 34.3, so a surface facing that way gets more.
        let a = 34.3f32.to_radians();
        let toward = sh.evaluate([a.sin(), 0.0, a.cos()]);
        let away = sh.evaluate([-a.sin(), 0.0, -a.cos()]);
        assert!(
            toward[0] > away[0],
            "facing the sun ({toward:?}) is not brighter than facing away ({away:?})",
        );

        // Sane magnitudes. A daylight HDRI should land the ambient somewhere
        // usable rather than at 0.001 or at 40 -- both of which are what a
        // missing or doubled normalisation looks like, and both of which read
        // as "the sky does nothing" or "everything is white".
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
        // A sphere of constant radiance L lights every normal identically, and
        // the value is L. Anything else means the solid-angle weighting or the
        // normalisation is wrong -- both of which are easy to get subtly wrong
        // and impossible to see in a screenshot.
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
        // An equirectangular image gives the pole as many texels as the
        // equator. Without the sin(theta) weight, a small bright cap overhead
        // dominates the result; with it, it contributes in proportion to the
        // solid angle it actually covers. So: a sky black everywhere except a
        // narrow band at the very top must contribute only a little.
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
        // The whole point of using nine coefficients rather than one: the
        // ambient term becomes DIRECTIONAL. A normal facing the bright half
        // must receive more than one facing away, which a flat ambient cannot
        // express at all.
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
        // u < 0.5 is the lit half, which by direction_to_uv is -x.
        let lit = sh.evaluate([-1.0, 0.0, 0.0]);
        let dark = sh.evaluate([1.0, 0.0, 0.0]);
        assert!(
            lit[0] > dark[0] * 2.0,
            "the lit side got {lit:?} and the dark side {dark:?}",
        );
    }

    #[test]
    fn rotating_the_sky_moves_the_light_with_it() {
        // `rotation_deg` exists so an author can put the sun where the level
        // needs it. If the coefficients did not turn with the picture, the
        // control would move the background and leave the lighting behind --
        // exactly the mismatch an HDRI exists to prevent.
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
        // The compatibility claim. A level with no sky must render EXACTLY as
        // it did before this existed, in every direction -- otherwise adding
        // the feature silently re-lights every scene that does not use it.
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
            // Reconstruct, the way the GPU will.
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

/// Rendering tests for the background pass.
///
/// The projection and the coefficients above are pure and tested without a GPU.
/// What those cannot see is the shader: whether the view ray is reconstructed
/// correctly, whether `sky_uv` in WGSL agrees with `direction_to_uv` in Rust,
/// and whether the depth setup keeps the sky behind the level instead of over
/// it. All three fail silently and look like art problems.
#[cfg(test)]
mod render_tests {
    use super::*;
    use crate::renderer::lights::LightsUniform;
    use crate::renderer::uniforms::{PlayerUpload, PostUpload, ShadowUpload, SkyUpload, UniformBuffer};

    const SIZE: u32 = 9; // odd, so the centre texel is exactly at NDC (0, 0)

    /// Red over one half of the longitude, blue over the other.
    ///
    /// A split rather than a gradient: it turns "is the shader looking the right
    /// way" into a question with a categorical answer, which a smooth panorama
    /// would blur into a judgement call.
    fn split_panorama() -> Panorama {
        let (w, h) = (64u32, 32u32);
        let mut rgb = Vec::with_capacity((w * h * 3) as usize);
        for _y in 0..h {
            for x in 0..w {
                // Values above 1 so the Reinhard curve in the shader has
                // something to do and the test notices if it is removed.
                if x < w / 2 {
                    rgb.extend_from_slice(&[3.0, 0.0, 0.0]);
                } else {
                    rgb.extend_from_slice(&[0.0, 0.0, 3.0]);
                }
            }
        }
        Panorama { width: w, height: h, rgb }
    }

    /// Renders the sky looking along `dir` and returns the centre pixel.
    fn look(dir: [f32; 3], pano: &Panorama) -> Option<[u8; 4]> {
        look_turned(dir, pano, 0.0)
    }

    /// The same, with the rig turned by `yaw`: `dir` is then the view in the
    /// PLAYER's frame, as the headset's camera matrices give it.
    fn look_turned(dir: [f32; 3], pano: &Panorama, yaw: f32) -> Option<[u8; 4]> {
        render_centre(dir, yaw, 1.0, PostUpload::default(), |device, queue, layout| Sky::new(device, queue, layout, pano, 0.0, 1.0))
    }

    /// The time-of-day sky of `snap` looking along `dir` through a lens
    /// `fov` radians wide, at `exposure`: the centre pixel.
    fn look_at_snapshot(dir: [f32; 3], snap: &SkySnapshot, zenith: [f32; 3], fov: f32, exposure: f32) -> Option<[u8; 4]> {
        render_centre(dir, 0.0, fov, PostUpload { exposure, ..Default::default() }, |device, queue, layout| {
            let mut sky = Sky::none(device, queue, layout, AMBIENT);
            sky.apply_snapshot(device, queue, layout, snap, 0.0, zenith);
            sky
        })
    }

    /// Renders a sky looking along `dir` and returns the centre pixel.
    fn render_centre(
        dir: [f32; 3],
        yaw: f32,
        fov: f32,
        post: PostUpload,
        make_sky: impl FnOnce(&Device, &Queue, &BindGroupLayout) -> Sky,
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
            pass.set_pipeline(&pipeline.pipeline);
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
        // THE PIN BETWEEN THE TWO MAPPINGS. `sky_uv` in WGSL and
        // `direction_to_uv` in Rust are separate implementations of one
        // convention, and the coefficients are projected with the Rust one
        // while the background is drawn with the WGSL one. If they disagree,
        // a level is lit from somewhere other than where its sun is drawn --
        // which reads as a lighting bug and is a mapping bug.
        let pano = split_panorama();
        // Nudged off the axes on purpose. [0,0,1] lands on u=0.5 and [0,0,-1]
        // on the u=0/1 wrap -- both exactly on a boundary of this two-colour
        // fixture, where the sampler returns a 50/50 blend and "which channel
        // dominates" has no defined answer. That went unnoticed for as long as
        // the sky used a per-channel Reinhard, which maps an even red/blue
        // blend to exactly equal bytes so the comparison came out false and the
        // assertion passed by coincidence. ACES mixes channels, the tie broke,
        // and the accident surfaced. These offsets put every sample about three
        // texels inside a region.
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

    /// THE SKY STAYS WITH THE WORLD WHEN THE PLAYER TURNS. A snap or stick
    /// turn changes only the rig's yaw: the camera then looks along the
    /// player-frame direction, and the background must still show the sky
    /// that lies that way in the world. On the headset the sun swung round
    /// with every snap turn while the level stayed put (2026-10-01); the
    /// offline frames that pin turning draw no sky, so nothing caught it.
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
        // The panorama carries pure red at 3.0, and ACES mixes channels: the
        // green and blue that come back are made entirely by the colour
        // matrices. That is a stronger signature than the red channel, which
        // saturates for a fully-saturated primary this bright and so looks the
        // same tone mapped as clipped.
        //
        // A hard clamp leaves G and B at exactly 0. The old local Reinhard,
        // being per-channel, also left them at 0 and put red at ~191. Only a
        // matrix-based curve puts light in the other two, so this now pins
        // WHICH curve is running and not merely that one is.
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
        // It is drawn LAST at the far plane with no depth write, so early-Z
        // rejects covered pixels rather than shading them and throwing the work
        // away. Asserted by clearing depth to 0 -- as though the whole frame
        // were already covered by nearer geometry -- and checking the clear
        // colour survives.
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
                    // Everything already at the near plane: nothing of the sky
                    // may pass the LessEqual test.
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

    /// THE TIME OF DAY'S SKY ON THE GPU (smoke test): the panorama, the
    /// sun's and the moon's discs and the stars' branch run and land where
    /// the CPU says. Noon's zenith is blue; at night the moon's disc is lit
    /// against a black sky a few of its widths away.
    #[test]
    fn the_time_of_day_sky_draws_noon_and_the_moon() {
        let tod = TimeOfDaySky::new(TimeOfDayParams::default());
        let zenith = tod.atmosphere.eye_transmittance([0.0, 1.0, 0.0]).map(|t| -t.max(1e-6).ln());
        let noon = tod.snapshot(12.0, 0.0);
        // Linear target (Rgba8Unorm): the zenith is ~0.1 at exposure 1.
        let px = shot!(look_at_snapshot([0.0, 1.0, 0.0], &noon, zenith, 0.5, 4.0));
        assert!(px[2] > px[0] + 30 && px[2] > px[1] && px[2] > 60, "noon's zenith is not blue: {px:?}");
        let night = tod.snapshot(23.5, 0.0);
        let moon = night.moon.direction;
        assert!(moon[1] > 0.2, "the moon should be up at 23:30 in the defaults: {moon:?}");
        // A 1.2 degree lens: the 0.52 degree disc fills the middle.
        let disc = shot!(look_at_snapshot(moon, &night, zenith, 0.02, 4.0));
        let m = glam::Vec3::from(moon);
        let beside = (m + m.cross(glam::Vec3::Y).normalize() * 0.05).normalize().to_array();
        let sky = shot!(look_at_snapshot(beside, &night, zenith, 0.02, 4.0));
        assert!(disc[0].max(disc[1]).max(disc[2]) > 60, "the moon's disc is dark: {disc:?}");
        assert!(sky[0].max(sky[1]).max(sky[2]) < 10, "the sky beside the moon is lit: {sky:?}");
    }

    #[test]
    fn the_pipeline_validates_multisampled_too() {
        // It runs in the scene pass, which is 4x on the headset.
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

// Scene shader sources, for `multiview::every_scene_shader_survives_the_multiview_transform`.
// Test-only: the gate has to see exactly the text each pipeline is built from,
// and nothing on a development machine can build a multiview pipeline to check.
#[cfg(test)]
pub fn sky_shader_src() -> String {
    sky_shader()
}
