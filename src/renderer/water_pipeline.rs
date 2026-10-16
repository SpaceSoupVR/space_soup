use wgpu::*;

use crate::renderer::lights::wgsl_lights_block;
use crate::renderer::water_waves::{WaveField, WaveParams, MIPS, N as WAVE_N};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct WaterVertex {
    pub position: [f32; 3],

    pub depth: f32,
}

pub const LIGHT_PATH: f32 = 1.15;

pub const SWASH_PERIOD: f32 = 8.0;

const SWASH_LEAD: f32 = 0.45;

const BREAKER_PER_SWASH: f32 = 1.8;
const BREAK_RATIO: f32 = 0.78;

const SWASH_ALONG: [(f32, f32, f32, f32, f32); 3] =
    [(0.047, 0.031, 0.14, 0.0, 1.0), (-0.083, 0.059, 0.09, 1.7, -2.0), (0.151, -0.112, 0.08, 4.1, 3.0)];

const SWASH_SET_TURNS: f32 = 4.0;
const SWASH_SECTION: (f32, f32, f32, f32) = (0.21, -0.13, -0.11, 0.17);

fn swashes_a_loop() -> f32 {
    crate::renderer::water_waves::WaveParams::default().loop_seconds / SWASH_PERIOD
}

fn swash_count(t: f32, still_depth: f32, xz: glam::Vec2) -> f32 {
    let w0 = 2.0 * std::f32::consts::PI / crate::renderer::water_waves::WaveParams::default().loop_seconds;
    let along: f32 = SWASH_ALONG.iter().map(|&(kx, kz, a, o, n)| a * (xz.dot(glam::Vec2::new(kx, kz)) + o + n * w0 * t).sin()).sum();
    t / SWASH_PERIOD + still_depth * SWASH_LEAD + along
}

fn breaker_amp(t: f32, xz: glam::Vec2, count: f32) -> f32 {
    let w0 = 2.0 * std::f32::consts::PI / crate::renderer::water_waves::WaveParams::default().loop_seconds;
    let set = 0.6 + 0.4 * (2.0 * std::f32::consts::PI * SWASH_SET_TURNS / swashes_a_loop() * count + 0.7).sin();
    let (ax, az, bx, bz) = SWASH_SECTION;
    let inner = 1.1 * (xz.dot(glam::Vec2::new(bx, bz)) + 2.0 * w0 * t).sin();
    let section = 0.5 + 0.5 * (xz.dot(glam::Vec2::new(ax, az)) + inner + 3.0 * w0 * t).sin();
    set * (0.45 + 0.55 * section)
}

const BREAKER_FRONT: f32 = 0.04;
const BREAKER_FRONT_ROUND: f32 = 0.08;
const BREAKER_LIP_FROM: f32 = 0.8;
const BREAKER_BACK: f32 = 0.1;

fn breaker_front(amp: f32) -> f32 {
    let t = ((amp - BREAKER_LIP_FROM) / (1.0 - BREAKER_LIP_FROM)).clamp(0.0, 1.0);
    BREAKER_FRONT_ROUND + (BREAKER_FRONT - BREAKER_FRONT_ROUND) * t * t * (3.0 - 2.0 * t)
}

const HORIZON_LIFT: (f32, f32) = (4000.0, 15000.0);

fn paws_calm(wind_speed: f32) -> f32 {
    let t = ((wind_speed - 2.0) / 4.0).clamp(0.0, 1.0);
    0.15 + 0.85 * t * t * (3.0 - 2.0 * t)
}

const SWELL: [(f32, f32, f32); 3] = [(4.3, 0.0, 0.5), (2.9, 0.55, 0.3), (2.1, -0.7, 0.2)];

fn swell_wgsl(i: usize) -> String {
    let w0 = 2.0 * std::f32::consts::PI / crate::renderer::water_waves::WaveParams::default().loop_seconds;
    let (length, turn, share) = SWELL[i];
    let k = 2.0 * std::f32::consts::PI / length;
    let omega = ((9.81 * k).sqrt() / w0).round() * w0;
    format!("vec4<f32>({:?}, {:?}, {:?}, {:?})", k * turn.cos(), k * turn.sin(), omega, share)
}

const FOAM_TILES: f32 = 1.0 / 3.0;

const BED_ALBEDO: f32 = 0.45;

pub const WAVE_REACH: f32 = 40.0;

pub const RINGS: usize = 4;
pub const RING_SECONDS: f32 = 3.0;
const RING_SPEED: f32 = 0.3;
const RING_WAVELENGTH: f32 = 0.08;
const RING_HEIGHT: f32 = 0.006;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct WaterUniform {

    pub extinction: [f32; 4],

    pub scatter: [f32; 4],

    pub tiles: [f32; 4],

    pub waves: [f32; 4],

    pub lost: [[f32; 4]; MIPS as usize],

    pub rings: [[f32; 4]; RINGS],

    pub air: [f32; 4],

    pub look: [f32; 4],

    pub prints: [f32; 4],
}

#[derive(Copy, Clone, Debug)]
pub struct WaterOptics {

    pub height: f32,

    pub shallow: [f32; 3],

    pub deep: [f32; 3],
    pub depth_scale: f32,
    pub shore_fade: f32,

    pub swash: f32,

    pub swell: f32,

    pub gloss: f32,

    pub mud: f32,
}

impl WaterUniform {
    pub fn new(o: &WaterOptics, waves: &WaveParams) -> Self {

        let k = |c: f32| -c.clamp(1e-4, 1.0).ln() / (o.depth_scale.max(0.01) * (1.0 + LIGHT_PATH));
        Self {
            extinction: [k(o.shallow[0]), k(o.shallow[1]), k(o.shallow[2]), o.height],
            scatter: [o.deep[0], o.deep[1], o.deep[2], o.shore_fade.max(1e-3)],
            tiles: [waves.tile[0], waves.tile[1], waves.tile[2], 0.0],
            waves: [waves.unresolved_slope_variance(), o.swash.max(0.0), WAVE_REACH, SWASH_PERIOD],
            lost: waves.lost_slope_variance(),
            rings: [[0.0; 4]; RINGS],
            air: [paws_calm(waves.wind_speed), o.swell.max(0.0), waves.wind_dir.cos(), waves.wind_dir.sin()],
            look: [o.gloss.clamp(0.0, 1.0), o.mud.clamp(0.0, 1.0), 0.0, 0.0],
            prints: [0.0; 4],
        }
    }

    pub fn set_time(&mut self, seconds: f64, loop_seconds: f32) {
        self.tiles[3] = seconds.rem_euclid(loop_seconds.max(1.0) as f64) as f32;
    }
}

pub struct WaterPipeline {
    pub pipeline: RenderPipeline,

    pub ringless: RenderPipeline,

    pub waterline: RenderPipeline,
    pub material_layout: BindGroupLayout,
    pub default_mud: TextureView,

    pub sampler: Sampler,

    pub dual_source: bool,

    inputs: WaterInputs,
}

impl WaterPipeline {
    pub fn new(
        device: &Device,
        format: TextureFormat,
        camera_layout: &BindGroupLayout,
        samples: u32,
    ) -> Self {
        let dual = Self::device_blends_two(device);
        Self::new_with(device, format, camera_layout, samples, crate::renderer::multiview::ViewMode::Mono, dual)
    }

    pub fn new_stereo(
        device: &Device,
        format: TextureFormat,
        camera_layout: &BindGroupLayout,
        samples: u32,
    ) -> Self {
        let dual = Self::device_blends_two(device);
        Self::new_with(device, format, camera_layout, samples, crate::renderer::multiview::ViewMode::Stereo, dual)
    }

    pub fn device_blends_two(device: &Device) -> bool {
        device.features().contains(Features::DUAL_SOURCE_BLENDING)
    }

    pub fn new_with(
        device: &Device,
        format: TextureFormat,
        camera_layout: &BindGroupLayout,
        samples: u32,
        view: crate::renderer::multiview::ViewMode,
        dual_source: bool,
    ) -> Self {
        let texture = |binding, visibility, view_dimension| BindGroupLayoutEntry {
            binding,
            visibility,
            ty: BindingType::Texture {
                sample_type: TextureSampleType::Float { filterable: true },
                view_dimension,
                multisampled: false,
            },
            count: None,
        };
        let material_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("water_material_bgl"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::VERTEX_FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },

                texture(1, ShaderStages::VERTEX_FRAGMENT, TextureViewDimension::D2Array),
                texture(2, ShaderStages::FRAGMENT, TextureViewDimension::D2Array),
                texture(3, ShaderStages::FRAGMENT, TextureViewDimension::D2Array),
                texture(4, ShaderStages::FRAGMENT, TextureViewDimension::D2),
                BindGroupLayoutEntry {
                    binding: 5,
                    visibility: ShaderStages::VERTEX_FRAGMENT,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
                texture(6, ShaderStages::FRAGMENT, TextureViewDimension::D2),
            ],
        });
        let default_mud = device
            .create_texture(&TextureDescriptor {
                label: Some("water_mud_none"),
                size: Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rg8Unorm,
                usage: TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
            .create_view(&TextureViewDescriptor::default());
        let sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("water_waves_sampler"),
            address_mode_u: AddressMode::Repeat,
            address_mode_v: AddressMode::Repeat,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: MipmapFilterMode::Linear,
            ..Default::default()
        });

        let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("water_pipeline_layout"),
            bind_group_layouts: &[Some(camera_layout), Some(&material_layout)],
            immediate_size: 0,
        });

        let blend = if dual_source {
            BlendState {
                color: BlendComponent { src_factor: BlendFactor::One, dst_factor: BlendFactor::Src1, operation: BlendOperation::Add },
                alpha: BlendComponent { src_factor: BlendFactor::Zero, dst_factor: BlendFactor::One, operation: BlendOperation::Add },
            }
        } else {
            BlendState {
                color: BlendComponent {
                    src_factor: BlendFactor::One,
                    dst_factor: BlendFactor::OneMinusSrcAlpha,
                    operation: BlendOperation::Add,
                },
                alpha: BlendComponent { src_factor: BlendFactor::Zero, dst_factor: BlendFactor::One, operation: BlendOperation::Add },
            }
        };

        let pipeline = build_water(device, &layout, format, blend, samples, view, water_wgsl_with(dual_source, true), "water_pipeline");
        let ringless = build_water(device, &layout, format, blend, samples, view, water_wgsl_with(dual_source, false), "water_pipeline_ringless");
        let waterline = build_water(device, &layout, format, blend, samples, view, water_line_wgsl(dual_source), "water_pipeline_waterline");
        let inputs = (layout, format, blend, samples, view);
        if crate::renderer::shader_checks::PIPELINE_STATISTICS.load(std::sync::atomic::Ordering::Relaxed) {

            for (cut, _) in WATER_CUTS {
                let _ = water_with_cut(device, &inputs, dual_source, cut);
            }
        }

        Self { pipeline, ringless, waterline, material_layout, sampler, dual_source, inputs, default_mud }
    }

    pub fn with_cut(&self, device: &Device, cut: &str) -> Option<RenderPipeline> {
        water_with_cut(device, &self.inputs, self.dual_source, cut)
    }

    pub fn bind_groups(&self, device: &Device, uniform: &Buffer, waves: &WaveField) -> [BindGroup; 2] {
        self.bind_groups_with(device, uniform, waves, &self.default_mud)
    }

    pub fn bind_groups_with(&self, device: &Device, uniform: &Buffer, waves: &WaveField, mud: &TextureView) -> [BindGroup; 2] {
        std::array::from_fn(|set| {
            device.create_bind_group(&BindGroupDescriptor {
                label: Some("water_material"),
                layout: &self.material_layout,
                entries: &[
                    BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
                    BindGroupEntry { binding: 1, resource: BindingResource::TextureView(&waves.displacement_views[set]) },
                    BindGroupEntry { binding: 2, resource: BindingResource::TextureView(&waves.derivatives_view) },
                    BindGroupEntry { binding: 3, resource: BindingResource::TextureView(&waves.curvature_views[set]) },
                    BindGroupEntry { binding: 4, resource: BindingResource::TextureView(&waves.foam_view) },
                    BindGroupEntry { binding: 5, resource: BindingResource::Sampler(&self.sampler) },
                    BindGroupEntry { binding: 6, resource: BindingResource::TextureView(mud) },
                ],
            })
        })
    }
}

type WaterInputs = (PipelineLayout, TextureFormat, BlendState, u32, crate::renderer::multiview::ViewMode);

pub(crate) const WATER_CUTS: &[(&str, &[(&str, &str)])] = &[
    ("none", &[]),

    (
        "breaker_face",
        &[("    if (water.waves.y > 0.0 && still_depth > 0.0 && still_depth < 2.8) {\n        let breaker_count", "    if (false) {\n        let breaker_count")],
    ),

    ("swell", &[("        tilt = tilt + water.air.y * smoothstep(0.15, 0.6, still_depth) * swell_at(q, water.tiles.w).yz;\n", "")]),

    ("paws", &[("    let paws = mix(water.air.x, 1.0, smoothstep(-0.6, 0.7, lull));", "    let paws = 1.0;")]),

    ("c2", &[("    let c2 = textureSampleBias(wave_curve, wave_samp, q / water.tiles.z, 2, 2.0) * paws;", "    let c2 = vec4<f32>(0.0);")]),

    ("caustics", &[("    let caustic = mix(1.0, gather, CAUSTIC_CONTRAST * murk * (1.0 - smoothstep(1.5, 5.0, thickness)));", "    let caustic = 1.0;")]),

    ("surf", &[("    if (water.waves.y > 0.0) {\n        let count = swash_count(still_depth, q, water.tiles.w);", "    if (false) {\n        let count = swash_count(still_depth, q, water.tiles.w);")]),

    ("walk", &[("    if (on_map && bed_shows) {\n        let flat", "    if (false) {\n        let flat")]),

    ("no_deep_skip", &[("    let bed_shows = max(max(t_view.x * t_light.x, t_view.y * t_light.y), t_view.z * t_light.z) >= DEEP_SKIP
        || smoothstep(0.0, water.scatter.w, thickness) < 1.0;", "    let bed_shows = true;")]),

    ("beyond", &[("    if (drawn < 1.0 && bed_shows) {\n        beyond", "    if (false) {\n        beyond")]),

    ("refract", &[("    if (has_map) {\n        let closing", "    if (false) {\n        let closing")]),

    ("env", &[("    let env = outdoor_radiance(", "    let env = vec3<f32>(0.0);\n    let env_unused = outdoor_radiance(")]),

    ("glint", &[("        glint = glint + radiance * WATER_PI * water_ggx(n, v, to_l, a2);\n", "")]),

    ("prints", &[("    let colour = mix(shade, foam_light, froth_cover);", "    let colour = vec3<f32>(print.x, print.y, 0.15);")]),
];

fn water_with_cut(device: &Device, inputs: &WaterInputs, dual_source: bool, cut: &str) -> Option<RenderPipeline> {
    let (_, edits) = WATER_CUTS.iter().find(|(name, _)| *name == cut)?;
    let mut src = water_wgsl_with(dual_source, false);
    for (from, to) in edits.iter() {
        if src.matches(from).count() != 1 {
            log::warn!("water cut {cut}: `{from}` is not in the shader once");
            return None;
        }
        src = src.replacen(from, to, 1);
    }
    let (layout, format, blend, samples, view) = inputs;
    Some(build_water(device, layout, *format, *blend, *samples, *view, src, &format!("water_cut_{cut}")))
}

#[allow(clippy::too_many_arguments)]
fn build_water(
    device: &Device,
    layout: &PipelineLayout,
    format: TextureFormat,
    blend: BlendState,
    samples: u32,
    view: crate::renderer::multiview::ViewMode,
    source: String,
    label: &str,
) -> RenderPipeline {
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some(label),
            source: ShaderSource::Wgsl(view.shader(source).into()),
        });
        device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(layout),
            vertex: VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Some(VertexBufferLayout {
                    array_stride: std::mem::size_of::<WaterVertex>() as BufferAddress,
                    step_mode: VertexStepMode::Vertex,
                    attributes: &vertex_attr_array![0 => Float32x3, 1 => Float32],
                })],
                compilation_options: Default::default(),
            },
            fragment: Some(FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(ColorTargetState { format, blend: Some(blend), write_mask: ColorWrites::ALL })],
                compilation_options: Default::default(),
            }),
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,

                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(DepthStencilState {
                format: TextureFormat::Depth32Float,

                depth_write_enabled: Some(true),
                depth_compare: Some(CompareFunction::Less),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: MultisampleState { count: samples, ..Default::default() },
            multiview_mask: view.mask(),
            cache: None,
        })
}

const WAVE_ALLOWANCE: f32 = 2.0;

pub fn water_bounds(verts: &[WaterVertex], waves: &WaveParams) -> (glam::Vec3, glam::Vec3) {
    let (lo, hi) = verts.iter().fold(
        (glam::Vec3::splat(f32::INFINITY), glam::Vec3::splat(f32::NEG_INFINITY)),
        |(lo, hi), v| {
            let p = glam::Vec3::from(v.position);
            (lo.min(p), hi.max(p))
        },
    );
    let reach = glam::Vec3::splat(WAVE_ALLOWANCE + 2.0 * waves.significant_height());
    (lo - reach, hi + reach)
}

pub fn water_seen(
    bounds: (glam::Vec3, glam::Vec3),
    views: &[[glam::Vec4; 6]],
    doorways: Option<&[[glam::Vec4; 6]]>,
) -> bool {
    let meets = |planes: &[glam::Vec4; 6]| {
        let mut unbounded = *planes;
        unbounded[5] = glam::Vec4::new(0.0, 0.0, 0.0, 1.0);
        crate::renderer::shadow::aabb_in_frustum(&unbounded, bounds.0, bounds.1)
    };
    views.iter().any(meets) && doorways.map_or(true, |d| d.iter().any(meets))
}

pub fn swash_lift(u: &WaterUniform, still_depth: f32, xz: glam::Vec2) -> f32 {
    let swash = u.waves[1];
    if swash <= 0.0 {
        return swell_lift(u, still_depth, xz);
    }
    let smooth = |e0: f32, e1: f32, x: f32| {
        let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    };
    let count = swash_count(u.tiles[3], still_depth, xz);
    let phase = count.rem_euclid(1.0);
    let amp = breaker_amp(u.tiles[3], xz, count);
    let rise = smooth(0.0, 0.16, phase) * (1.0 - smooth(0.16, 0.92, phase));

    let grown = BREAKER_PER_SWASH * swash * amp * (1.0 - smooth(1.2, 2.8, still_depth));
    let height = grown.min(BREAK_RATIO * still_depth.max(0.0));
    let s = phase - phase.round();
    let b = s / BREAKER_BACK;
    let front = breaker_front(amp);
    let profile = if s > 0.0 { (1.0 + b) * (-b).exp() } else { (-(s * s) / (front * front)).exp() };
    swash * (1.0 - smooth(0.3, 1.5, still_depth)) * rise * (0.5 + 0.5 * amp) + height * profile + swell_lift(u, still_depth, xz)
}

fn swell_lift(u: &WaterUniform, still_depth: f32, xz: glam::Vec2) -> f32 {
    if u.air[1] <= 0.0 {
        return 0.0;
    }
    let w0 = 2.0 * std::f32::consts::PI / crate::renderer::water_waves::WaveParams::default().loop_seconds;
    let along = glam::Vec2::new(u.air[2], u.air[3]);
    let across = glam::Vec2::new(-along.y, along.x);
    let height: f32 = SWELL
        .iter()
        .enumerate()
        .map(|(i, &(length, turn, share))| {
            let k = 2.0 * std::f32::consts::PI / length;
            let omega = ((9.81 * k).sqrt() / w0).round() * w0;
            let kv = along * (k * turn.cos()) + across * (k * turn.sin());
            share * (kv.dot(xz) - omega * u.tiles[3] + i as f32 * 2.1).sin()
        })
        .sum();

    let t = ((still_depth - 0.15) / 0.45).clamp(0.0, 1.0);
    u.air[1] * height * t * t * (3.0 - 2.0 * t)
}

fn loop_rate() -> f32 {
    2.0 * std::f32::consts::PI / crate::renderer::water_waves::WaveParams::default().loop_seconds
}

fn along_wgsl(i: usize) -> String {
    let (kx, kz, a, o, n) = SWASH_ALONG[i];
    format!("{a:?} * sin(dot(xz, vec2<f32>({kx:?}, {kz:?})) + {o:?} + {:?} * t)", n * loop_rate())
}

pub fn surface_wgsl() -> String {
    format!(
        r#"
const SWASH_LEAD: f32 = {swash_lead};
const BREAKER_PER_SWASH: f32 = {breaker_per_swash};
const BREAK_RATIO: f32 = {break_ratio};
const BREAKER_FRONT: f32 = {breaker_front};
const BREAKER_FRONT_ROUND: f32 = {breaker_front_round};
const BREAKER_LIP_FROM: f32 = {breaker_lip_from};
const BREAKER_BACK: f32 = {breaker_back};

struct WaterMat {{
    extinction: vec4<f32>,
    scatter: vec4<f32>,
    tiles: vec4<f32>,
    waves: vec4<f32>,
    lost: array<vec4<f32>, {mips}>,
    rings: array<vec4<f32>, {rings}>,
    air: vec4<f32>,
    look: vec4<f32>,
    prints: vec4<f32>,
}}

fn swash_count(still_depth: f32, xz: vec2<f32>, t: f32) -> f32 {{
    let along = {along0} + {along1} + {along2};
    return t / water.waves.w + still_depth * SWASH_LEAD + along;
}}
fn swash_phase(still_depth: f32, xz: vec2<f32>, t: f32) -> f32 {{
    return fract(swash_count(still_depth, xz, t));
}}

fn breaker_amp(xz: vec2<f32>, t: f32, count: f32) -> f32 {{
    let sets = 0.6 + 0.4 * sin({set_rate:?} * count + 0.7);
    let inner = 1.1 * sin(dot(xz, vec2<f32>({bx:?}, {bz:?})) + {w2:?} * t);
    let section = 0.5 + 0.5 * sin(dot(xz, vec2<f32>({ax:?}, {az:?})) + inner + {w3:?} * t);
    return sets * (0.45 + 0.55 * section);
}}

fn swash_rise(phase: f32) -> f32 {{
    return smoothstep(0.0, 0.16, phase) * (1.0 - smoothstep(0.16, 0.92, phase));
}}

fn breaker_height(still_depth: f32, amp: f32) -> f32 {{
    let grown = BREAKER_PER_SWASH * water.waves.y * amp * (1.0 - smoothstep(1.2, 2.8, still_depth));
    return min(grown, BREAK_RATIO * max(still_depth, 0.0));
}}

fn breaker_profile(phase: f32, amp: f32) -> f32 {{
    let s = phase - round(phase);
    let b = s / BREAKER_BACK;
    let f = breaker_front(amp);
    return select(exp(-(s * s) / (f * f)), (1.0 + b) * exp(-b), s > 0.0);
}}

fn breaker_front(amp: f32) -> f32 {{
    return mix(BREAKER_FRONT_ROUND, BREAKER_FRONT, smoothstep(BREAKER_LIP_FROM, 1.0, amp));
}}

fn breaker_profile_slope(phase: f32, amp: f32) -> f32 {{
    let s = phase - round(phase);
    let b = s / BREAKER_BACK;
    let f = breaker_front(amp);
    return select(-2.0 * s / (f * f) * exp(-(s * s) / (f * f)), -b / BREAKER_BACK * exp(-b), s > 0.0);
}}

fn swell_wave(w: vec4<f32>, xz: vec2<f32>, t: f32, offset: f32) -> vec3<f32> {{
    let along = water.air.zw;
    let k = along * w.x + vec2<f32>(-along.y, along.x) * w.y;
    let phase = dot(k, xz) - w.z * t + offset;
    return w.w * vec3<f32>(sin(phase), k * cos(phase));
}}
fn swell_at(xz: vec2<f32>, t: f32) -> vec3<f32> {{
    return swell_wave({swell0}, xz, t, 0.0) + swell_wave({swell1}, xz, t, 2.1) + swell_wave({swell2}, xz, t, 4.2);
}}
"#,
        swash_lead = format!("{SWASH_LEAD:?}"),
        along0 = along_wgsl(0),
        along1 = along_wgsl(1),
        along2 = along_wgsl(2),
        set_rate = 2.0 * std::f32::consts::PI * SWASH_SET_TURNS / swashes_a_loop(),
        ax = SWASH_SECTION.0,
        az = SWASH_SECTION.1,
        bx = SWASH_SECTION.2,
        bz = SWASH_SECTION.3,
        w2 = 2.0 * loop_rate(),
        w3 = 3.0 * loop_rate(),
        breaker_per_swash = format!("{BREAKER_PER_SWASH:?}"),
        break_ratio = format!("{BREAK_RATIO:?}"),
        breaker_front = format!("{BREAKER_FRONT:?}"),
        breaker_back = format!("{BREAKER_BACK:?}"),
        breaker_front_round = format!("{BREAKER_FRONT_ROUND:?}"),
        breaker_lip_from = format!("{BREAKER_LIP_FROM:?}"),
        swell0 = swell_wgsl(0),
        swell1 = swell_wgsl(1),
        swell2 = swell_wgsl(2),
        mips = MIPS,
        rings = RINGS,
    )
}

pub fn surface_fn_wgsl(name: &str, disp: &str) -> String {
    format!(
        r#"
fn {name}(pos: vec3<f32>, depth: f32, eye: vec3<f32>, t: f32) -> vec3<f32> {{
    var p = pos;

    let reach = water.waves.z;
    let near = select(0.0, 1.0 - smoothstep(reach * 0.6, reach, distance(pos.xz, eye.xz)), reach > 0.0);
    let open = near * smoothstep(0.0, 1.2, depth);
    if (open > 0.0) {{
        let d0 = textureSampleLevel({disp}, wave_samp, pos.xz / water.tiles.x, 0, 1.0).xyz;
        let d1 = textureSampleLevel({disp}, wave_samp, pos.xz / water.tiles.y, 1, 3.0).xyz;
        p = p + (d0 + d1) * open;
    }}
    if (water.waves.y > 0.0) {{
        let shore = 1.0 - smoothstep(0.3, 1.5, depth);
        let count = swash_count(depth, pos.xz, t);
        let phase = fract(count);
        let amp = breaker_amp(pos.xz, t, count);
        p.y = p.y + water.waves.y * shore * swash_rise(phase) * (0.5 + 0.5 * amp) + breaker_height(depth, amp) * breaker_profile(phase, amp);
    }}
    if (water.air.y > 0.0) {{
        p.y = p.y + water.air.y * swell_at(pos.xz, t).x * near * smoothstep(0.15, 0.6, depth);
    }}

    let far = distance(pos.xz, eye.xz);
    p.y = mix(p.y, max(p.y, eye.y), smoothstep({lift0}, {lift1}, far));
    return p;
}}
"#,
        lift0 = format!("{:?}", HORIZON_LIFT.0),
        lift1 = format!("{:?}", HORIZON_LIFT.1),
    )
}

const RING_BLOCK: &str = r#"    if (water.rings[0].w > 0.0) {
        let resolved = 1.0 - smoothstep(0.25, 0.5, span / RING_WAVELENGTH);
        for (var i = 0; i < RINGS_N; i = i + 1) {
            let r = water.rings[i];
            if (r.w <= 0.0) {
                break;
            }
            let d = q - r.xy;
            let dist = length(d);
            let age = r.z;
            let spread = 0.03 + 0.08 * age;
            let x = dist - RING_SPEED * age;
            let k = 2.0 * WATER_PI / RING_WAVELENGTH;
            let amp = RING_HEIGHT * sqrt(r.w) * exp(-age / 1.1) / sqrt(1.0 + dist * 10.0);
            let train = exp(-x * x / (2.0 * spread * spread));
            let rise = amp * resolved * train * -k * sin(k * x);
            tilt = tilt + d / max(dist, 1e-4) * rise;

            ring_rough = ring_rough + 0.5 * (amp * k) * (amp * k) * train * (1.0 - resolved);
            let foam_reach = 0.12 + 0.1 * sqrt(r.w) + 0.15 * age;
            ring_foam = ring_foam + min(0.35 * sqrt(r.w), 0.6) * exp(-dist * dist / (foam_reach * foam_reach)) * (1.0 - smoothstep(0.4, RING_SECONDS, age));
        }
    }
"#;

fn water_wgsl(dual: bool) -> String {
    water_wgsl_with(dual, true)
}

const WATER_LINE_VIEW: &str = "    let t_view = exp(-k * path);";
const WATER_LINE_RISING: &str = "    let t_view = select(exp(-k * path), vec3<f32>(0.0), down.y > 0.0);";

fn water_line_wgsl(dual: bool) -> String {
    const TEST: &str = r#"
fn water_line_y(xz: vec2<f32>, eye: vec3<f32>) -> f32 {
    let surface = water.extinction.w;
    var depth = 30.0;
    if (camera.sky_params.w > 0.5) {
        depth = surface - bed_at(vec3<f32>(xz.x, 0.0, xz.y)).w;
    }
    let t = water.tiles.w;
    let first = water_surface(vec3<f32>(xz.x, surface, xz.y), depth, eye, t);
    let rest = xz - (first.xz - xz);
    return water_surface(vec3<f32>(rest.x, surface, rest.y), depth, eye, t).y;
}
fn water_line_under(player: vec3<f32>) -> bool {
    let c = cam_view_proj() * vec4<f32>(player, 1.0);
    let h = cam_inv_view_proj() * vec4<f32>(c.xy / c.w, 0.0, 1.0);
    let start = to_world_space(h.xyz / h.w);
    let eye = to_world_space(cam_pos());
    let h0 = water_line_y(eye.xz, eye);
    let hx = water_line_y(eye.xz + vec2<f32>(0.1, 0.0), eye);
    let hz = water_line_y(eye.xz + vec2<f32>(0.0, 0.1), eye);
    let plane = vec3<f32>(h0, (hx - h0) * 10.0, (hz - h0) * 10.0);
    return plane.x + dot(start.xz - eye.xz, plane.yz) - start.y > 0.0;
}
"#;
    let src = water_wgsl_with(dual, true);
    assert_eq!(src.matches("@fragment fn fs_main").count(), 1);
    assert_eq!(src.matches("    return FsOut(").count(), 1);
    assert_eq!(src.matches(WATER_LINE_VIEW).count(), 1);

    src.replacen("@fragment fn fs_main", &format!("{TEST}@fragment fn fs_main"), 1)
        .replacen(WATER_LINE_VIEW, WATER_LINE_RISING, 1)
        .replacen("    return FsOut(", "    if (water_line_under(in.world_pos)) {\n        discard;\n    }\n    return FsOut(", 1)
}

fn water_wgsl_with(dual: bool, rings: bool) -> String {
    let ring_block = if rings { RING_BLOCK.replace("RINGS_N", &RINGS.to_string()) } else { String::new() };
    let output = if dual {
        "struct FsOut {\n    @location(0) @blend_src(0) colour: vec4<f32>,\n    @location(0) @blend_src(1) through: vec4<f32>,\n}"
    } else {
        "struct FsOut {\n    @location(0) colour: vec4<f32>,\n}"
    };
    let ret = if dual {
        "return FsOut(vec4<f32>(tonemap(colour) + beyond, 1.0), vec4<f32>(through, 1.0));"
    } else {

        "return FsOut(vec4<f32>(tonemap(colour) + beyond, 1.0 - dot(through, vec3<f32>(1.0 / 3.0))));"
    };
    format!(
        r#"{enable}
{lights_block}

const WATER_PI: f32 = 3.14159265;
const WATER_MIPS: i32 = {mips};
const WAVE_N: f32 = {wave_n};
const LIGHT_PATH: f32 = {light_path};
const IOR_WATER: f32 = 1.333;

const OPEN_PATH: f32 = 12.0;

const WALK_FIRST: f32 = 0.6;
const WALK_GROWTH: f32 = 1.55;
const WALK_STEPS: i32 = 6;
const LIP_BURIED: f32 = {lip_buried};
const LIP_RAMP: f32 = {lip_ramp};
const MUD_FLATTEN: f32 = 0.9;
const MUD_GRAIN_TILES: f32 = 2.5;
const MUD_GRAIN_TILT: f32 = 0.15;
const MUD_GRAIN_SHADE: f32 = 0.6;
const MUD_ROUGH: f32 = 0.35;
const MUD_SKY_SHARE: f32 = 0.5;
const MUD_PRINT_STEP: f32 = 1.0;
const MUD_PRINT_TILT: f32 = 0.6;
const MUD_PRINT_DARK: f32 = 0.7;
const MUD_PRINT_RIM: f32 = 0.9;
const MUD_WET_GLOSS: f32 = 0.9;
const MUD_POOL_GLOSS: f32 = 0.9;
const MUD_SLUSH_RIPPLE: f32 = 0.8;
const MUD_WET_DARK: f32 = 0.35;
const MUD_POOL_FROM: f32 = 0.05;
const MUD_POOL_TO: f32 = 0.12;

const SHELF_SLOPE: f32 = 0.15;
const FOAM_TILES: f32 = {foam_tiles};
const BED_ALBEDO: f32 = {bed_albedo};
const RING_SPEED: f32 = {ring_speed};
const WATER_TRACE_COARSEN: i32 = 2;

const DEEP_SKIP: f32 = {deep_skip:?};
const CAUSTIC_CONTRAST: f32 = 0.6;
const CAUSTIC_MURK: f32 = 1.0;
const PAWS_RATE: vec2<f32> = vec2<f32>(0.1308997, 0.0785398);
const RING_WAVELENGTH: f32 = {ring_wavelength};
const RING_HEIGHT: f32 = {ring_height};
const RING_SECONDS: f32 = {ring_seconds};
{surface}
@group(1) @binding(0) var<uniform> water: WaterMat;
@group(1) @binding(1) var wave_disp: texture_2d_array<f32>;
@group(1) @binding(2) var wave_slope: texture_2d_array<f32>;
@group(1) @binding(3) var wave_curve: texture_2d_array<f32>;
@group(1) @binding(4) var foam_tex: texture_2d<f32>;
@group(1) @binding(5) var wave_samp: sampler;
@group(1) @binding(6) var mud_tex: texture_2d<f32>;

struct VIn {{
    @location(0) pos: vec3<f32>,
    @location(1) depth: f32,
}}
struct VOut {{
    @builtin(position) clip: vec4<f32>,

    @location(0) world_pos: vec3<f32>,

    @location(1) depth: f32,

    @location(2) tex_pos: vec3<f32>,

    @location(3) rest: vec2<f32>,
}}

{surface_fn}
@vertex fn vs_main(v: VIn) -> VOut {{
    var out: VOut;
    let p = water_surface(v.pos, v.depth, to_world_space(cam_pos()), water.tiles.w);
    let player = to_player_space(p);
    out.clip = cam_view_proj() * vec4<f32>(player, 1.0);

    if (out.clip.w > 0.0) {{
        out.clip.z = min(out.clip.z, out.clip.w * 0.999999);
    }}
    out.world_pos = player;
    out.depth = v.depth;
    out.tex_pos = p;
    out.rest = v.pos.xz;
    return out;
}}

fn bed_at(at: vec3<f32>) -> vec4<f32> {{
    let uv = ground_uv(at);
    let half_texel = 0.5 / vec2<f32>(textureDimensions(ground_map));
    let edge = clamp(uv, half_texel, vec2<f32>(1.0) - half_texel);
    let g = textureSampleLevel(ground_map, probe_samp, edge, 0.0);
    return vec4<f32>(g.rgb, g.a - SHELF_SLOPE * length((uv - edge) / camera.ground_params.zw));
}}

fn water_lost(lod: f32) -> vec4<f32> {{
    let l = clamp(lod, 0.0, f32(WATER_MIPS - 1));
    let i = i32(floor(l));
    let j = min(i + 1, WATER_MIPS - 1);
    return mix(water.lost[i], water.lost[j], l - f32(i));
}}

fn water_ggx(n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, a2: f32) -> f32 {{
    let nl = max(dot(n, l), 0.0);
    let h = normalize(l + v);
    let nh = max(dot(n, h), 0.0);
    let nv = max(dot(n, v), 1e-4);
    let dd = nh * nh * (a2 - 1.0) + 1.0;
    let d = a2 / (WATER_PI * dd * dd);
    let vis = 0.5 / (nl * sqrt(nv * nv * (1.0 - a2) + a2) + nv * sqrt(nl * nl * (1.0 - a2) + a2) + 1e-5);
    let f = 0.02 + 0.98 * pow(1.0 - max(dot(v, h), 0.0), 5.0);
    return d * vis * f * nl;
}}

{output}

@fragment fn fs_main(in: VOut) -> FsOut {{

    let q = in.rest;
    let span = max(length(dpdx(q)), length(dpdy(q)));
    let s0 = textureSample(wave_slope, wave_samp, q / water.tiles.x, 0);
    let s1 = textureSample(wave_slope, wave_samp, q / water.tiles.y, 1);

    let lull = sin(dot(q, vec2<f32>(0.47, 0.31)) + water.tiles.w * PAWS_RATE.x + 1.6 * sin(dot(q, vec2<f32>(-0.27, 0.58)) - water.tiles.w * PAWS_RATE.y));
    let paws = mix(water.air.x, 1.0, smoothstep(-0.6, 0.7, lull));
    let s2 = textureSample(wave_slope, wave_samp, q / water.tiles.z, 2) * paws;
    let c0 = textureSample(wave_curve, wave_samp, q / water.tiles.x, 0);
    let c1 = textureSample(wave_curve, wave_samp, q / water.tiles.y, 1);

    let c2 = textureSampleBias(wave_curve, wave_samp, q / water.tiles.z, 2, 2.0) * paws;
    let bubbles = textureSample(foam_tex, wave_samp, q * FOAM_TILES).r;

    let p = in.tex_pos;
    let surface = water.extinction.w;

    let has_map = camera.sky_params.w > 0.5;
    let uv = ground_uv(p);
    let on_map = has_map && all(uv > vec2<f32>(0.0)) && all(uv < vec2<f32>(1.0));
    let bed = select(surface - in.depth, bed_at(p).w, has_map);
    let still_depth = surface - bed;
    let thickness = max(p.y - bed, 0.0);

    let shoal = smoothstep(0.0, 1.2, still_depth);
    let slope = (s0.xy + s1.xy) * shoal + s2.xy;
    let push = vec2<f32>(1.0) + (s0.zw + s1.zw) * shoal + s2.zw;
    var tilt = slope / max(push, vec2<f32>(0.25));

    if (water.air.y > 0.0) {{
        tilt = tilt + water.air.y * smoothstep(0.15, 0.6, still_depth) * swell_at(q, water.tiles.w).yz;
    }}
    var breaker_phase = 0.0;
    if (water.waves.y > 0.0 && still_depth > 0.0 && still_depth < 2.8) {{
        let breaker_count = swash_count(still_depth, q, water.tiles.w);
        breaker_phase = fract(breaker_count);
        let ax = p + vec3<f32>(0.5, 0.0, 0.0);
        let az = p + vec3<f32>(0.0, 0.0, 0.5);
        let px = swash_phase(surface - select(bed, bed_at(ax).w, has_map), q + vec2<f32>(0.5, 0.0), water.tiles.w);
        let pz = swash_phase(surface - select(bed, bed_at(az).w, has_map), q + vec2<f32>(0.0, 0.5), water.tiles.w);
        let dx = px - breaker_phase;
        let dz = pz - breaker_phase;
        let across = vec2<f32>(dx - round(dx), dz - round(dz)) / 0.5;
        let breaker_size = breaker_amp(q, water.tiles.w, breaker_count);
        tilt = tilt + breaker_height(still_depth, breaker_size) * breaker_profile_slope(breaker_phase, breaker_size) * across;
    }}

    var ring_foam = 0.0;
    var ring_rough = 0.0;
{ring_block}    let mud = water.look.y * (1.0 - smoothstep(MUD_POOL_FROM, MUD_POOL_TO, thickness));
    let grain = textureSample(foam_tex, wave_samp, q * MUD_GRAIN_TILES).r - 0.5;
    let grain_b = textureSample(foam_tex, wave_samp, q * MUD_GRAIN_TILES * 0.37 + vec2<f32>(0.5)).r - 0.5;
    let wave_tilt = tilt;
    var print = vec2<f32>(0.0);
    var print_rim = 0.0;
    var print_tilt = vec2<f32>(0.0);
    let print_uv = (q - water.prints.xy) * water.prints.zw;
    if (water.prints.z > 0.0 && all(print_uv > vec2<f32>(0.0)) && all(print_uv < vec2<f32>(1.0))) {{
        print = textureSampleLevel(mud_tex, wave_samp, print_uv, 0.0).rg;
        let e = water.prints.zw * MUD_PRINT_STEP;
        let dx = textureSampleLevel(mud_tex, wave_samp, print_uv + vec2<f32>(e.x, 0.0), 0.0).r - textureSampleLevel(mud_tex, wave_samp, print_uv - vec2<f32>(e.x, 0.0), 0.0).r;
        let dz = textureSampleLevel(mud_tex, wave_samp, print_uv + vec2<f32>(0.0, e.y), 0.0).r - textureSampleLevel(mud_tex, wave_samp, print_uv - vec2<f32>(0.0, e.y), 0.0).r;
        print_tilt = vec2<f32>(dx, dz);
        print_rim = smoothstep(0.05, 0.5, length(vec2<f32>(dx, dz))) * (1.0 - print.x);
    }}
    let fresh = print.y * mud;
    let pool = print.x * mud;
    let wet = max(fresh * MUD_WET_GLOSS, pool * MUD_POOL_GLOSS);
    let wet_gloss = mix(water.look.x, 1.0, wet);
    tilt = wave_tilt * (1.0 - MUD_FLATTEN * mud * (1.0 - wet * MUD_SLUSH_RIPPLE)) + vec2<f32>(grain, grain_b) * MUD_GRAIN_TILT * mud * (1.0 - wet) + print_tilt * MUD_PRINT_TILT * mud;
    let n = normalize(to_player_direction(vec3<f32>(-tilt.x, 1.0, -tilt.y)));

    let lost0 = water_lost(log2(max(span * WAVE_N / water.tiles.x, 1e-6))).x;
    let lost1 = water_lost(log2(max(span * WAVE_N / water.tiles.y, 1e-6))).y;
    let lost2 = water_lost(log2(max(span * WAVE_N / water.tiles.z, 1e-6))).z;
    let a2 = clamp(max((lost0 + lost1) * shoal * shoal + (lost2 + water.waves.x) * paws * paws + ring_rough, MUD_ROUGH * mud), 4e-4, 1.0);
    let rough = sqrt(sqrt(a2));

    let v = normalize(cam_pos() - in.world_pos);
    let fresnel = 0.02 + 0.98 * pow(1.0 - max(dot(n, v), 0.0), 5.0);

    var r = reflect(-v, n);
    r.y = max(r.y, 0.002);
    r = normalize(r);
    ground_trace_coarsen = WATER_TRACE_COARSEN;
    let env = outdoor_radiance(p, normalize(to_world_direction(r)), r, rough, clamp(rough * PROBE_ROUGHNESS_MIPS, PROBE_MIN_LOD, PROBE_MAX_LOD));
    let env_seen = env * wet_gloss * select(1.0, 1.0 / WATER_PI, camera.sky_params.y < 0.5);

    var glint = vec3<f32>(0.0);
    var sunlight = vec3<f32>(0.0);
    for (var i = 0u; i < lights.count.x; i = i + 1u) {{
        let l = lights.lights[i];
        let kind = l.params.z;
        if (kind < 1.5 && l.position.w > -0.5) {{
            continue;
        }}
        var to_l = normalize(-l.direction.xyz);
        var atten = 1.0;
        if (kind > 1.5) {{
            atten = sun_visibility(l, in.world_pos);
        }} else {{
            let dl = l.position.xyz - in.world_pos;
            let dist = length(dl);
            to_l = dl / max(dist, 1e-4);
            let x2 = dist * dist / max(l.params.x * l.params.x, 1e-8);
            let fade = clamp(1.0 - x2 * x2, 0.0, 1.0);
            atten = fade * fade / max(dist * dist, LAMP_RADIUS * LAMP_RADIUS);
            if (kind > 0.5) {{
                atten = atten * smoothstep(l.params.y, l.direction.w, dot(-to_l, l.direction.xyz));
            }}
        }}
        let radiance = l.color_intensity.rgb * l.color_intensity.a * atten;
        glint = glint + radiance * WATER_PI * water_ggx(n, v, to_l, a2);
        sunlight = sunlight + radiance * max(to_l.y, 0.0);
    }}

    let down = normalize(to_world_direction(-v));
    let cos_i = max(-down.y, 0.0);
    let sin_t = sqrt(max(1.0 - cos_i * cos_i, 0.0)) / IOR_WATER;
    let cos_t = sqrt(1.0 - sin_t * sin_t);
    let level = vec2<f32>(down.x, down.z);
    let across = level * (sin_t / max(length(level), 1e-6));
    let refracted = vec3<f32>(across.x, -cos_t, across.y);
    let first = thickness / cos_t;
    var path = first;
    if (has_map) {{
        let closing = thickness - (bed - bed_at(p + refracted * first).w);
        path = select(first * OPEN_PATH, first * thickness / max(closing, 1e-4), closing * OPEN_PATH > thickness);
    }}
    let bed_depth = max(surface - p.y + cos_t * path, 0.0);

    let k = water.extinction.xyz;
    let t_view = exp(-k * path);
    let t_light = exp(-k * (bed_depth * LIGHT_PATH));
    let bed_shows = max(max(t_view.x * t_light.x, t_view.y * t_light.y), t_view.z * t_light.z) >= DEEP_SKIP
        || smoothstep(0.0, water.scatter.w, thickness) < 1.0;
    var drawn = select(0.0, 1.0, !has_map);
    if (on_map && bed_shows) {{
        let flat = thickness / max(cos_i, 0.01);
        var seen = flat * OPEN_PATH;
        var walked = 0.0;
        var gap = thickness;
        var met = false;
        var clears = 0.0;
        var buried = 0.0;
        var step = WALK_FIRST;
        for (var i = 0; i < WALK_STEPS; i = i + 1) {{
            let t = flat * step;
            let at = p + down * t;
            let next_gap = at.y - bed_at(at).w;
            if (!met && next_gap <= 0.0) {{
                met = true;
                seen = walked + (t - walked) * gap / max(gap - next_gap, 1e-5);
            }}
            if (met) {{
                clears = max(clears, next_gap);
            }}
            buried = min(buried, next_gap);
            walked = t;
            gap = next_gap;
            step = step * WALK_GROWTH;
        }}
        let seen_uv = ground_uv(p + down * seen);
        let from_edge = min(min(seen_uv.x, 1.0 - seen_uv.x) / camera.ground_params.z, min(seen_uv.y, 1.0 - seen_uv.y) / camera.ground_params.w);
        let lip = smoothstep(0.0, 0.5, clears) * saturate((LIP_BURIED + buried) * LIP_RAMP);
        let map_span = min(1.0 / max(camera.ground_params.z, 1e-6), 1.0 / max(camera.ground_params.w, 1e-6));
        let rim_in = min(3.0, 0.1 * map_span);
        let rim_out = min(6.0, 0.2 * map_span);
        drawn = (1.0 - lip) * (1.0 - smoothstep(2.0, 4.0, seen / max(flat, 1e-4))) * smoothstep(rim_in, rim_out, from_edge);
    }}

    let sky_down = sky_irradiance(vec3<f32>(0.0, 1.0, 0.0));
    let print_shade = (1.0 - MUD_PRINT_DARK * print.x * mud) * (1.0 + MUD_PRINT_RIM * print_rim * mud) * (1.0 - MUD_WET_DARK * fresh);
    let silt = mix(1.0, 1.0 + MUD_GRAIN_SHADE * (grain + 0.5 * grain_b), mud) * print_shade;
    let body = water.scatter.rgb * silt * (sky_irradiance(n) * mix(1.0, MUD_SKY_SHARE, mud) + sunlight * max(dot(n, vec3<f32>(0.0, 1.0, 0.0)), 0.0)) * (vec3<f32>(1.0) - t_view);

    let fine = 0.5 * (1.0 - smoothstep(0.08, 0.3, span * WAVE_N * 0.25 / water.tiles.z));
    let mid = max(shoal, 0.6) * (1.0 - smoothstep(0.08, 0.3, span * WAVE_N / water.tiles.y));
    let curve = c0.xyz * (0.5 * shoal) + c1.xyz * mid + c2.xyz * fine;
    let focus = thickness * (1.0 - 1.0 / IOR_WATER);
    let jx = 1.0 - focus * curve.x;
    let jz = 1.0 - focus * curve.y;
    let jxz = focus * curve.z;

    let det = jx * jz - jxz * jxz;
    let gather = inverseSqrt(det * det + 0.0625);

    let murk = exp(-(water.extinction.x + water.extinction.y + water.extinction.z) * (1.0 / 3.0) * thickness * CAUSTIC_MURK);
    let caustic = mix(1.0, gather, CAUSTIC_CONTRAST * murk * (1.0 - smoothstep(1.5, 5.0, thickness)));
    let sun_on_bed = BED_ALBEDO * sunlight * t_light;
    let sky_on_bed = BED_ALBEDO * sky_down * t_light;
    let sun_share = dot(sun_on_bed, vec3<f32>(1.0)) / max(dot(sun_on_bed + sky_on_bed, vec3<f32>(1.0)), 1e-4);
    let brighter = max(caustic - 1.0, 0.0) * sun_on_bed * t_view;
    let dimmer = 1.0 + min(caustic - 1.0, 0.0) * sun_share;

    let caps = max(c0.w, c1.w) * shoal;
    var surf = 0.0;
    if (water.waves.y > 0.0) {{
        let count = swash_count(still_depth, q, water.tiles.w);
        let phase = fract(count);

        let patchy = 0.5 + 0.5 * sin(dot(q, vec2<f32>(0.41, 0.17)) + 1.3 * sin(dot(q, vec2<f32>(-0.23, 0.37)) + water.tiles.w * 0.0785398));

        let s = phase - round(phase);
        let amp = breaker_amp(q, water.tiles.w, count);
        let grown = BREAKER_PER_SWASH * water.waves.y * amp * (1.0 - smoothstep(1.2, 2.8, still_depth));
        let holds = grown / max(BREAK_RATIO * still_depth, 1e-3);
        let spilling = smoothstep(0.75, 1.0, holds) * exp(-(s - 0.008) * (s - 0.008) / (0.012 * 0.012));
        let broken = smoothstep(1.0, 1.4, holds) * smoothstep(-0.02, 0.1, still_depth);
        let face = exp(-(s + 0.012) * (s + 0.012) / (0.03 * 0.03));
        let trail = select(0.0, exp(-s / 0.05), s > 0.0);
        let breaker = max(spilling, broken * max(face, 0.65 * trail)) * mix(0.3, 1.0, patchy);
        let wash = smoothstep(0.0, 0.04, phase) * (1.0 - smoothstep(0.08, 0.4, phase))
            * (1.0 - smoothstep(0.0, 0.25, thickness)) * (1.0 - smoothstep(0.0, 0.6, still_depth))

            * smoothstep(0.1, 0.75, amp);
        surf = max(breaker, wash) + 0.18 * patchy * (1.0 - smoothstep(0.0, 0.2, thickness));
    }}
    let froth = clamp(caps + surf + ring_foam, 0.0, 1.0);
    let cover = clamp((bubbles - (1.0 - froth)) * 4.0, 0.0, 1.0) * smoothstep(0.0, 0.05, froth);
    let foam_light = 0.8 * (sky_irradiance(n) + sunlight);

    let edge = smoothstep(0.0, water.scatter.w, thickness);
    let lip = smoothstep(0.0, 0.015, thickness);
    let shade = (env_seen * fresnel + glint * wet_gloss + (body + brighter) * (1.0 - fresnel)) * edge;
    let under = mix(vec3<f32>(1.0), t_view * t_light * dimmer * (1.0 - fresnel), edge) * print_shade;
    let froth_cover = cover * lip;
    let colour = mix(shade, foam_light, froth_cover);

    var beyond = vec3<f32>(0.0);
    if (drawn < 1.0 && bed_shows) {{
        beyond = under * tonemap(bed_at(p + refracted * path).xyz) * ((1.0 - drawn) * (1.0 - froth_cover));
    }}
    let through = under * (drawn * (1.0 - froth_cover));
    {ret}
}}
"#,
        enable = if dual { "enable dual_source_blending;" } else { "" },
        lights_block = wgsl_lights_block(0, 1),
        mips = MIPS,
        wave_n = format!("{:.1}", WAVE_N as f32),
        light_path = format!("{LIGHT_PATH:?}"),
        foam_tiles = format!("{FOAM_TILES:?}"),
        bed_albedo = format!("{BED_ALBEDO:?}"),
        lip_buried = format!("{:?}", spacesoup_graphics::ground::LIP_BURIED),
        lip_ramp = format!("{:?}", spacesoup_graphics::ground::LIP_RAMP),
        ring_block = ring_block,
        ring_speed = format!("{RING_SPEED:?}"),
        deep_skip = 0.5f32 / 255.0,
        ring_wavelength = format!("{RING_WAVELENGTH:?}"),
        ring_height = format!("{RING_HEIGHT:?}"),
        ring_seconds = format!("{RING_SECONDS:?}"),
        output = output,
        ret = ret,
        surface = surface_wgsl(),
        surface_fn = surface_fn_wgsl("water_surface", "wave_disp"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::renderer::lights::{Light, LightKind, LightsUniform};
    use crate::renderer::uniforms::test_support::scene_uniforms;
    use crate::renderer::uniforms::{PlayerUpload, ShadowUpload};
    use crate::renderer::Color3;
    use glam::Vec3;
    use wgpu::util::DeviceExt;

    fn gpu(dual: bool) -> Option<(Device, Queue)> {
        let instance = Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&RequestAdapterOptions {
            apply_limit_buckets: false,
            power_preference: PowerPreference::default(),
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok()?;
        if dual && !adapter.features().contains(Features::DUAL_SOURCE_BLENDING) {
            return None;
        }
        pollster::block_on(adapter.request_device(&DeviceDescriptor {
            required_features: if dual { Features::DUAL_SOURCE_BLENDING } else { Features::empty() },
            required_limits: crate::renderer::uniforms::scene_limits(Limits::default()),
            ..Default::default()
        }))
        .ok()
    }

    #[derive(Clone, Copy)]
    struct Case {
        dual: bool,

        depth: f32,
        eye: Vec3,

        walked: Vec3,
        wind: f32,

        bed: [f64; 3],

        sun: Option<Vec3>,
        shallow: [f32; 3],
        deep: [f32; 3],
    }

    fn case(dual: bool) -> Case {
        Case {
            dual,
            depth: 1.0,
            eye: OVERHEAD,
            walked: Vec3::ZERO,
            wind: 0.0,
            bed: [0.0; 3],
            sun: None,
            shallow: [0.30, 0.70, 0.75],
            deep: [0.01, 0.04, 0.07],
        }
    }

    const OVERHEAD: Vec3 = Vec3::new(0.0, 8.0, 0.5);

    const GRAZING: Vec3 = Vec3::new(0.0, 0.05, 40.0);

    const DOWN_THE_SLOPE: Vec3 = Vec3::new(0.0, 3.0, 5.7);

    fn render(c: &Case) -> Option<[u8; 4]> {
        let image = render_image(c)?;
        Some(image[(SIZE / 2 * SIZE + SIZE / 2) as usize])
    }

    const SIZE: u32 = 8;

    fn render_image(c: &Case) -> Option<Vec<[u8; 4]>> {
        let (device, queue) = gpu(c.dual)?;
        let format = TextureFormat::Rgba8Unorm;

        let lights = LightsUniform::new(&device);
        let sun = c.sun.map(|d| Light {
            mask_channel: None,
            shadow_near: None,
            source_radius: 0.0,
            in_level_bake: true,
            position: Vec3::ZERO,
            direction: d.normalize(),
            kind: LightKind::Directional,
            color: Color3(255, 255, 255, 255),
            intensity: 2.0,
            range: 0.0,
            cone_angle_deg: 0.0,
            inner_cone_angle_deg: 0.0,
        });
        lights.upload(&queue, &sun.into_iter().collect::<Vec<_>>());
        let (_shadows, uniforms) = scene_uniforms(&device, &lights);

        uniforms.upload_scene(
            &queue,
            glam::Mat4::IDENTITY,
            c.eye,
            &ShadowUpload::disabled(),
            &crate::renderer::uniforms::SkyUpload::none(),
            &crate::renderer::uniforms::PostUpload::default(),
            &PlayerUpload { offset: c.walked, yaw: 0.0, ..Default::default() },
        );

        let pipeline = WaterPipeline::new_with(
            &device,
            format,
            &uniforms.layout,
            1,
            crate::renderer::multiview::ViewMode::Mono,
            c.dual,
        );
        let params = WaveParams { wind_speed: c.wind, fetch: 2000.0, ..Default::default() };
        let waves = WaveField::new(&device, &queue, params);
        let mut u = WaterUniform::new(
            &WaterOptics { height: 0.0, shallow: c.shallow, deep: c.deep, depth_scale: 1.0, shore_fade: 0.25, swash: 0.0, swell: 0.0, gloss: 1.0, mud: 0.0 },
            &params,
        );
        u.waves[2] = 0.0;
        let ubuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("water_test_uniform"),
            contents: bytemuck::bytes_of(&u),
            usage: BufferUsages::UNIFORM,
        });
        let groups = pipeline.bind_groups(&device, &ubuf, &waves);

        let o = c.walked;
        let v = |x: f32, y: f32| WaterVertex { position: [x + o.x, y + o.y, 0.5 + o.z], depth: c.depth };
        let verts = [v(-1.0, -1.0), v(3.0, -1.0), v(-1.0, 3.0)];
        let vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("water_test_vb"),
            contents: bytemuck::cast_slice(&verts),
            usage: BufferUsages::VERTEX,
        });

        let target = device.create_texture(&TextureDescriptor {
            label: Some("water_test_target"),
            size: Extent3d { width: SIZE, height: SIZE, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let depth_tex = device.create_texture(&TextureDescriptor {
            label: Some("water_test_depth"),
            size: Extent3d { width: SIZE, height: SIZE, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Depth32Float,
            usage: TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let tv = target.create_view(&Default::default());
        let dv = depth_tex.create_view(&Default::default());
        let readback = device.create_buffer(&BufferDescriptor {
            label: Some("water_test_readback"),
            size: (256 * SIZE) as u64,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&Default::default());
        waves.update(&queue, &mut encoder, 12.0);
        {
            let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("water_test_pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &tv,
                    depth_slice: None,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(Color { r: c.bed[0], g: c.bed[1], b: c.bed[2], a: 1.0 }),
                        store: StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                    view: &dv,
                    depth_ops: Some(Operations { load: LoadOp::Clear(1.0), store: StoreOp::Store }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                multiview_mask: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&pipeline.pipeline);
            pass.set_bind_group(0, &uniforms.bind_group, &[]);
            pass.set_bind_group(1, &groups[waves.current()], &[]);
            pass.set_vertex_buffer(0, vb.slice(..));
            pass.draw(0..3, 0..1);
        }
        encoder.copy_texture_to_buffer(
            TexelCopyTextureInfo { texture: &target, mip_level: 0, origin: Origin3d::ZERO, aspect: TextureAspect::All },
            TexelCopyBufferInfo {
                buffer: &readback,
                layout: TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(256), rows_per_image: Some(SIZE) },
            },
            Extent3d { width: SIZE, height: SIZE, depth_or_array_layers: 1 },
        );
        queue.submit(Some(encoder.finish()));
        let slice = readback.slice(..);
        slice.map_async(MapMode::Read, |_| {});
        device.poll(PollType::Wait { submission_index: None, timeout: None }).ok();
        let data = slice.get_mapped_range().unwrap();
        Some(
            (0..SIZE * SIZE)
                .map(|i| {
                    let o = (i / SIZE * 256 + i % SIZE * 4) as usize;
                    [data[o], data[o + 1], data[o + 2], data[o + 3]]
                })
                .collect(),
        )
    }

    fn shows_through(c: Case) -> Option<[i32; 3]> {
        let white = render(&Case { bed: [1.0; 3], ..c })?;
        let black = render(&Case { bed: [0.0; 3], ..c })?;
        Some([0, 1, 2].map(|i| white[i] as i32 - black[i] as i32))
    }

    fn sum(c: [i32; 3]) -> i32 {
        c[0] + c[1] + c[2]
    }

    fn each_path(test: impl Fn(bool)) {
        for dual in [false, true] {
            if gpu(dual).is_none() {
                eprintln!("skipping the {} path: no GPU for it", if dual { "dual-source" } else { "one-alpha" });
                continue;
            }
            test(dual);
        }
    }

    #[test]
    fn the_water_shader_compiles_and_draws() {

        each_path(|dual| {
            let px = render(&Case { eye: GRAZING, ..case(dual) }).unwrap();
            assert!(px[0] as u32 + px[1] as u32 + px[2] as u32 > 0, "water drew nothing over a black bed ({dual}): {px:?}");
        });
    }

    #[test]
    fn deep_water_hides_the_bed() {

        each_path(|dual| {
            let shallow = shows_through(Case { depth: 0.2, ..case(dual) }).unwrap();
            let deep = shows_through(Case { depth: 8.0, ..case(dual) }).unwrap();
            assert!(sum(shallow) > 3 * sum(deep) + 30, "deep water must hide the bed ({dual}): {deep:?} vs {shallow:?}");
        });
    }

    #[test]
    fn the_bed_takes_the_waters_colour_channel_by_channel() {

        let Some(through) = gpu(true).and_then(|_| shows_through(case(true))) else {
            eprintln!("skipping: no GPU with dual-source blending");
            return;
        };
        assert!(
            through[0] + 20 < through[1] && through[0] + 20 < through[2],
            "the bed must lose its red before its green and blue: {through:?}",
        );
    }

    #[test]
    fn a_grazing_view_shows_less_of_the_bed() {

        each_path(|dual| {
            let down = shows_through(Case { depth: 0.3, ..case(dual) }).unwrap();
            let across = shows_through(Case { depth: 0.3, eye: GRAZING, ..case(dual) }).unwrap();
            assert!(sum(down) > sum(across) + 60, "a grazing view must mirror rather than show the bed ({dual}): {across:?} vs {down:?}");
        });
    }

    #[test]
    fn the_shoreline_fades_out_rather_than_ending_in_a_line() {

        each_path(|dual| {
            let brink = render(&Case { depth: 0.001, bed: [0.6; 3], ..case(dual) }).unwrap();
            let body = render(&Case { depth: 2.0, bed: [0.6; 3], ..case(dual) }).unwrap();
            let off = |px: [u8; 4]| (0..3).map(|i| (px[i] as i32 - 153).abs()).max().unwrap();
            assert!(off(brink) <= 3, "the shoreline must leave the bed untouched ({dual}): {brink:?}");
            assert!(off(body) > 20, "two metres of water must change what is under it ({dual}): {body:?}");
        });
    }

    #[test]
    fn the_waves_belong_to_the_world_not_the_player() {

        each_path(|dual| {
            let c = Case { wind: 9.0, eye: DOWN_THE_SLOPE, bed: [1.0; 3], ..case(dual) };
            let here = render_image(&c).unwrap();
            let walked = render_image(&Case { walked: Vec3::new(7.3, 0.0, 3.1), ..c }).unwrap();
            assert_ne!(here, walked, "the waves did not change with the world under the pixels ({dual})");
        });
    }

    #[test]
    fn waves_change_the_surface() {

        each_path(|dual| {
            let c = Case { eye: DOWN_THE_SLOPE, bed: [1.0; 3], ..case(dual) };
            let calm = render_image(&c).unwrap();
            let windy = render_image(&Case { wind: 9.0, ..c }).unwrap();
            assert_ne!(calm, windy, "a 9 m/s wind changed nothing ({dual})");
        });
    }

    #[test]
    fn the_sun_glints_where_the_water_mirrors_it() {

        each_path(|dual| {
            let c = Case { depth: 20.0, ..case(dual) };
            let mirrored = render(&Case { sun: Some(Vec3::NEG_Y), ..c }).unwrap();
            let aside = render(&Case { sun: Some(Vec3::new(1.0, -0.3, 0.0)), ..c }).unwrap();
            let lum = |p: [u8; 4]| p[0] as u32 + p[1] as u32 + p[2] as u32;
            assert!(lum(mirrored) > lum(aside) + 60, "no glint from a mirrored sun ({dual}): {mirrored:?} vs {aside:?}");
        });
    }

    #[test]
    fn the_breakers_come_in_sets_and_sections_and_loop_seamlessly() {
        let loop_s = crate::renderer::water_waves::WaveParams::default().loop_seconds;

        for x in [-20.0f32, 0.0, 13.0, 41.0] {
            let xz = glam::Vec2::new(x, 30.0);
            let a = (swash_count(17.0, 1.1, xz).rem_euclid(1.0), breaker_amp(17.0, xz, swash_count(17.0, 1.1, xz)));
            let b = (swash_count(17.0 + loop_s, 1.1, xz).rem_euclid(1.0), breaker_amp(17.0 + loop_s, xz, swash_count(17.0 + loop_s, 1.1, xz)));
            assert!((a.0 - b.0).abs() < 1e-3 || (a.0 - b.0).abs() > 0.999, "{a:?} {b:?}");
            assert!((a.1 - b.1).abs() < 2e-3, "{a:?} {b:?}");
        }

        let (mut lo, mut hi, mut amp_lo, mut amp_hi) = (f32::MAX, f32::MIN, f32::MAX, f32::MIN);
        for i in 0..61 {
            let xz = glam::Vec2::new(-30.0 + i as f32, 30.0);
            let c = swash_count(50.0, 1.0, xz);
            let a = breaker_amp(50.0, xz, c);
            lo = lo.min(c);
            hi = hi.max(c);
            amp_lo = amp_lo.min(a);
            amp_hi = amp_hi.max(a);
        }
        assert!(hi - lo > 0.15, "the crest arrives together: {lo}..{hi}");
        assert!(amp_hi - amp_lo > 0.3, "the crest is one height: {amp_lo}..{amp_hi}");

        let src = surface_wgsl();
        for i in 0..3 {
            assert!(src.contains(&along_wgsl(i)));
        }
    }

    #[test]
    fn every_water_cut_matches_and_validates() {
        for (cut, edits) in WATER_CUTS {
            for dual in [false, true] {
                let mut src = water_wgsl_with(dual, false);
                for (from, to) in edits.iter() {
                    assert_eq!(src.matches(from).count(), 1, "water cut {cut}: {from}");
                    src = src.replacen(from, to, 1);
                }
                let module = wgpu::naga::front::wgsl::parse_str(&src).unwrap_or_else(|e| panic!("{cut}: {}", e.emit_to_string(&src)));
                wgpu::naga::valid::Validator::new(wgpu::naga::valid::ValidationFlags::all(), wgpu::naga::valid::Capabilities::all())
                    .validate(&module)
                    .unwrap_or_else(|e| panic!("{cut}: {}", e.emit_to_string(&src)));
            }
        }
    }

    #[test]
    fn the_waterline_twin_is_the_water_plus_its_test() {
        for dual in [true, false] {
            let plain = water_wgsl_with(dual, true);
            let line = water_line_wgsl(dual);
            assert!(!plain.contains("water_line_under"));
            let without = line
                .replacen("    if (water_line_under(in.world_pos)) {\n        discard;\n    }\n", "", 1)
                .replacen(WATER_LINE_RISING, WATER_LINE_VIEW, 1);
            let start = without.find("\nfn water_line_y(").unwrap();
            let end = without.find("@fragment fn fs_main").unwrap();
            assert_eq!(format!("{}{}", &without[..start], &without[end..]), plain);
        }
    }

    #[test]
    fn the_water_lip_counts_only_where_the_line_grazed_the_ground() {
        let src = water_wgsl(false);
        assert!(src.contains("let lip = smoothstep(0.0, 0.5, clears) * saturate((LIP_BURIED + buried) * LIP_RAMP);"));
        assert!(src.contains("buried = min(buried, next_gap);"));
        assert!(src.contains(&format!("const LIP_BURIED: f32 = {:?};", spacesoup_graphics::ground::LIP_BURIED)));
        assert!(src.contains(&format!("const LIP_RAMP: f32 = {:?};", spacesoup_graphics::ground::LIP_RAMP)));
        assert!((spacesoup_graphics::ground::lip(1.0, -0.2) - 1.0).abs() < 1e-6);
        assert_eq!(spacesoup_graphics::ground::lip(1.0, -1.0), 0.0);
    }

    #[test]
    fn white_seen_through_depth_scale_comes_out_shallow() {

        let o = WaterOptics { height: 0.0, shallow: [0.3, 0.7, 0.75], deep: [0.0; 3], depth_scale: 2.0, shore_fade: 0.3, swash: 0.0, swell: 0.0, gloss: 1.0, mud: 0.0 };
        let u = WaterUniform::new(&o, &WaveParams::default());
        for i in 0..3 {
            let seen = (-u.extinction[i] * o.depth_scale * (1.0 + LIGHT_PATH)).exp();
            assert!((seen - o.shallow[i]).abs() < 1e-5, "channel {i}: {seen} vs {}", o.shallow[i]);
        }
    }
}

#[cfg(test)]
pub fn water_shader_src() -> String {
    water_wgsl(false)
}

#[cfg(test)]
mod seen_tests {
    use super::*;
    use crate::renderer::shadow::frustum_planes;
    use glam::{Mat4, Vec3};

    fn view() -> [glam::Vec4; 6] {
        frustum_planes(
            Mat4::perspective_rh_gl(90f32.to_radians(), 1.0, 0.1, 100.0)
                * Mat4::look_at_rh(Vec3::ZERO, Vec3::NEG_Z, Vec3::Y),
        )
    }

    fn pond(centre: Vec3) -> (Vec3, Vec3) {
        (centre - Vec3::new(5.0, 0.5, 5.0), centre + Vec3::new(5.0, 0.5, 5.0))
    }

    #[test]
    fn water_ahead_is_seen_and_water_behind_or_aside_is_not() {
        let v = [view()];
        assert!(water_seen(pond(Vec3::new(0.0, -1.0, -20.0)), &v, None), "ahead");
        assert!(!water_seen(pond(Vec3::new(0.0, -1.0, 20.0)), &v, None), "behind");
        assert!(!water_seen(pond(Vec3::new(60.0, -1.0, -20.0)), &v, None), "far to the side");
    }

    #[test]
    fn water_past_the_far_plane_is_still_seen() {

        assert!(water_seen(pond(Vec3::new(0.0, -1.0, -2000.0)), &[view()], None));
    }

    #[test]
    fn either_eye_is_enough() {
        let other = frustum_planes(
            Mat4::perspective_rh_gl(90f32.to_radians(), 1.0, 0.1, 100.0)
                * Mat4::look_at_rh(Vec3::ZERO, Vec3::Z, Vec3::Y),
        );
        assert!(water_seen(pond(Vec3::new(0.0, -1.0, 20.0)), &[view(), other], None));
    }

    #[test]
    fn from_a_closed_room_only_a_doorway_shows_it() {
        let v = [view()];
        let ahead = pond(Vec3::new(0.0, -1.0, -20.0));
        assert!(!water_seen(ahead, &v, Some(&[])), "no doorway in view: hidden by the walls");
        assert!(water_seen(ahead, &v, Some(&[view()])), "a doorway looking at it");
        assert!(water_seen(ahead, &v, None), "not in a closed room: no culling");
    }

    #[test]
    fn the_bounds_reach_as_far_as_the_waves_can() {
        let verts = [
            WaterVertex { position: [0.0, 1.0, 0.0], depth: 1.0 },
            WaterVertex { position: [10.0, 1.0, 4.0], depth: 2.0 },
        ];
        let calm = WaveParams { wind_speed: 1.0, fetch: 10.0, ..WaveParams::default() };
        let storm = WaveParams { wind_speed: 20.0, fetch: 50_000.0, ..WaveParams::default() };
        let (lo, hi) = water_bounds(&verts, &calm);
        assert!(lo.x <= -WAVE_ALLOWANCE && hi.z >= 4.0 + WAVE_ALLOWANCE && lo.y < 1.0 && hi.y > 1.0);
        let (slo, shi) = water_bounds(&verts, &storm);
        assert!(shi.y - slo.y > hi.y - lo.y + 1.0, "a storm's waves reach further: {slo} {shi} vs {lo} {hi}");
    }
}
