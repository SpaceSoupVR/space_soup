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
    _texture: Texture,
    _sampler: Sampler,
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

        let mut half = Vec::with_capacity((pano.width * pano.height * 4) as usize);
        for i in 0..(pano.width * pano.height) as usize {
            half.push(f32_to_f16(pano.rgb[i * 3]));
            half.push(f32_to_f16(pano.rgb[i * 3 + 1]));
            half.push(f32_to_f16(pano.rgb[i * 3 + 2]));
            half.push(f32_to_f16(1.0));
        }
        queue.write_texture(
            TexelCopyTextureInfo {
                texture: &texture,
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
        let view = texture.create_view(&TextureViewDescriptor::default());
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("sky_bg"),
            layout,
            entries: &[
                BindGroupEntry { binding: 0, resource: BindingResource::TextureView(&view) },
                BindGroupEntry { binding: 1, resource: BindingResource::Sampler(&sampler) },
            ],
        });

        Self {
            bind_group,
            irradiance,
            sun,
            reflection: Some(ReflectionSky { pano: without_sun, rotation_deg, intensity }),
            _texture: texture,
            _sampler: sampler,
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
    let uv = sky_uv(to_world_direction(dir));
    let radiance = textureSample(sky_tex, sky_samp, uv).rgb * camera.sky_params.x;

    // The shared curve, the same one every lit surface uses. This pass used to
    // run its own local Reinhard because nothing downstream tone mapped -- which
    // meant the sky and the geometry in front of it disagreed about how
    // highlights roll off, and the seam showed at the horizon.
    {sky_tail}
}}
"#,
        lights_block = wgsl_lights_block(0, 1),
        sky_tail = if SKY_CRACK_DEBUG {
            "return vec4<f32>(1.0, 0.0, 1.0, 1.0);"
        } else {
            "return vec4<f32>(tonemap(radiance), 1.0);"
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let (device, queue) = crate::renderer::terrain_pipeline::tests::headless_gpu()?;
        let format = TextureFormat::Rgba8Unorm;

        let lights = LightsUniform::new(&device);
        let (_shadows, uniforms) =
            crate::renderer::uniforms::test_support::scene_uniforms(&device, &lights);
        lights.upload(&queue, &[]);

        let eye = glam::Vec3::ZERO;
        let d = glam::Vec3::from(dir).normalize();
        let up = if d.y.abs() > 0.99 { glam::Vec3::Z } else { glam::Vec3::Y };
        let view_proj = glam::Mat4::perspective_rh(1.0, 1.0, 0.1, 100.0)
            * glam::Mat4::look_at_rh(eye, eye + d, up);
        uniforms.upload_scene(
            &queue,
            view_proj,
            eye,
            &ShadowUpload::disabled(),
            &SkyUpload { intensity: 1.0, sh: [[0.0; 4]; 9] },
            &PostUpload::default(),
            &PlayerUpload { yaw, ..Default::default() },
        );

        let pipeline = SkyPipeline::new(&device, format, &uniforms.layout, 1);
        let sky = Sky::new(&device, &queue, &pipeline.layout, pano, 0.0, 1.0);

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
