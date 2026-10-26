use wgpu::*;

use super::cuboid::SolidVertex;
use super::material_wgsl::{wgsl_biplanar_block, wgsl_whiteout_block};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TerrainMaterialUniform {

    pub repeat: [f32; 4],

    pub slope_start_deg: f32,
    pub slope_end_deg: f32,

    pub height_start: f32,
    pub height_end: f32,

    pub macro_repeat: f32,
    pub macro_strength: f32,

    pub biplanar_start_deg: f32,

    pub use_splat: f32,

    pub normal_strength: f32,

    pub wet_line: f32,
    pub wet_band: f32,

    pub parallax: f32,

    pub mud_layer: f32,
    pub deform_metres: f32,
    pub deform_texel: f32,
    pub wet_dark: f32,
}

impl Default for TerrainMaterialUniform {
    fn default() -> Self {
        Self {
            repeat: [8.0, 4.0, 10.0, 6.0],
            slope_start_deg: 22.0,
            slope_end_deg: 40.0,
            height_start: 1e9,
            height_end: 1e9,
            macro_repeat: 140.0,
            macro_strength: 0.35,
            biplanar_start_deg: 18.0,
            use_splat: 0.0,
            normal_strength: 1.0,
            wet_line: -1e9,
            wet_band: 1.0,
            parallax: 0.0,
            mud_layer: -1.0,
            deform_metres: 0.1,
            deform_texel: 0.05,
            wet_dark: 0.5,
        }
    }
}

pub struct TerrainPipeline {
    pub pipeline: RenderPipeline,
    pub material_layout: BindGroupLayout,
}

impl TerrainPipeline {
    pub fn new(device: &Device, format: TextureFormat, uniform_layout: &BindGroupLayout) -> Self {
        Self::new_multisampled(device, format, uniform_layout, 1)
    }

    pub fn new_multisampled(

        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        samples: u32,
    ) -> Self {
        Self::new_with_view(device, format, uniform_layout, samples, crate::renderer::multiview::ViewMode::Mono)
    }

    pub fn new_multisampled_stereo(

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
        Self::build(device, format, uniform_layout, samples, view, TerrainRole::Scene, None)
    }

    pub fn new_probe_reader(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        samples: u32,
        probe_layout: &BindGroupLayout,
    ) -> Self {
        Self::build(
            device,
            format,
            uniform_layout,
            samples,
            crate::renderer::multiview::ViewMode::Mono,
            TerrainRole::Read,
            Some(probe_layout),
        )
    }

    pub fn new_probe_reader_twins(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        samples: u32,
        probe_layout: &BindGroupLayout,
    ) -> [Self; 3] {
        [(true, false), (false, true), (true, true)].map(|(spotless, sun_baked)| {
            Self::new_probe_reader_twin(device, format, uniform_layout, samples, probe_layout, spotless, sun_baked)
        })
    }

    pub fn new_edited_probe_readers(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        samples: u32,
        probe_layout: &BindGroupLayout,
        edit: &str,
    ) -> Option<[Self; 4]> {
        let (edit, only) = edit.split_once('@').map_or((edit, None), |(e, c)| (e, Some(c)));
        let shipped = terrain_shader_for(TerrainRole::Read);
        let edited = crate::renderer::brush_pipeline::with_reader_edit(shipped.clone(), edit)?;
        let class = |name: &str| if only.is_none_or(|c| c == name) { edited.clone() } else { shipped.clone() };
        let baked = |s: String| crate::renderer::brush_pipeline::sun_reader_shader(s, crate::renderer::brush_pipeline::FaceSun::Baked);
        let spotless = crate::renderer::lights::without_spot_shadows;
        let build = |source: String, label: &str| {
            Self::build_from(
                device,
                format,
                uniform_layout,
                samples,
                crate::renderer::multiview::ViewMode::Mono,
                TerrainRole::Read,
                Some(probe_layout),
                source,
                label,
            )
        };
        Some([
            build(class("full"), "terrain_pipeline_read_edited"),
            build(spotless(class("full")), "terrain_pipeline_read_spotless_edited"),
            build(baked(class("baked")), "terrain_pipeline_read_baked_edited"),
            build(spotless(baked(class("baked"))), "terrain_pipeline_read_baked_spotless_edited"),
        ])
    }

    pub fn new_probe_reader_twin(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        samples: u32,
        probe_layout: &BindGroupLayout,
        spotless: bool,
        sun_baked: bool,
    ) -> Self {
        let mut source = terrain_shader_for(TerrainRole::Read);
        if sun_baked {
            source = crate::renderer::brush_pipeline::sun_reader_shader(source, crate::renderer::brush_pipeline::FaceSun::Baked);
        }
        if spotless {
            source = crate::renderer::lights::without_spot_shadows(source);
        }
        let label = match (spotless, sun_baked) {
            (false, false) => "terrain_pipeline_read",
            (true, false) => "terrain_pipeline_read_spotless",
            (false, true) => "terrain_pipeline_read_baked",
            (true, true) => "terrain_pipeline_read_baked_spotless",
        };
        Self::build_from(
            device,
            format,
            uniform_layout,
            samples,
            crate::renderer::multiview::ViewMode::Mono,
            TerrainRole::Read,
            Some(probe_layout),
            source,
            label,
        )
    }

    pub fn new_inlined(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        samples: u32,
        probe_layout: &BindGroupLayout,
        fixups: &crate::renderer::probe_fixup::ProbeFixups,
    ) -> [Self; 3] {
        let pass = with_inlined_layer_reads(terrain_shader_for(TerrainRole::ProbePass));
        let in_pass = |source: String, label: &str| {
            Self::build_from(
                device,
                crate::renderer::brush_pipeline::probe_pass::FORMAT,
                uniform_layout,
                1,
                crate::renderer::multiview::ViewMode::Mono,
                TerrainRole::ProbePass,
                Some(fixups.pass_layout()),
                source,
                label,
            )
        };
        [
            Self::build_from(
                device,
                format,
                uniform_layout,
                samples,
                crate::renderer::multiview::ViewMode::Mono,
                TerrainRole::Read,
                Some(probe_layout),
                with_inlined_layer_reads(terrain_shader_for(TerrainRole::Read)),
                "terrain_pipeline_read_inlined",
            ),
            in_pass(pass.clone(), "terrain_probe_pass_inlined"),
            in_pass(crate::renderer::lights::without_pool_maps(pass), "terrain_probe_pass_poolless_inlined"),
        ]
    }

    pub fn new_probe_pass(
        device: &Device,
        uniform_layout: &BindGroupLayout,
        fixups: &crate::renderer::probe_fixup::ProbeFixups,
    ) -> Self {
        Self::build(
            device,
            crate::renderer::brush_pipeline::probe_pass::FORMAT,
            uniform_layout,
            1,
            crate::renderer::multiview::ViewMode::Mono,
            TerrainRole::ProbePass,
            Some(fixups.pass_layout()),
        )
    }

    pub fn new_probe_pass_poolless(
        device: &Device,
        uniform_layout: &BindGroupLayout,
        fixups: &crate::renderer::probe_fixup::ProbeFixups,
    ) -> Self {
        Self::build_from(
            device,
            crate::renderer::brush_pipeline::probe_pass::FORMAT,
            uniform_layout,
            1,
            crate::renderer::multiview::ViewMode::Mono,
            TerrainRole::ProbePass,
            Some(fixups.pass_layout()),
            crate::renderer::lights::without_pool_maps(terrain_shader_for(TerrainRole::ProbePass)),
            "terrain_probe_pass_poolless",
        )
    }

    pub fn new_probe_pass_with_cut(
        device: &Device,
        uniform_layout: &BindGroupLayout,
        fixups: &crate::renderer::probe_fixup::ProbeFixups,
        cut: &str,
    ) -> Option<Self> {
        let (label, edits) = PROBE_PASS_REGISTER_CUTS.iter().find(|(label, _)| *label == cut)?;
        let mut src = terrain_shader_for(TerrainRole::ProbePass);
        for (from, to) in *edits {
            if !src.contains(from) {
                return None;
            }
            src = src.replacen(from, to, 1);
        }
        Some(Self::build_from(
            device,
            crate::renderer::brush_pipeline::probe_pass::FORMAT,
            uniform_layout,
            1,
            crate::renderer::multiview::ViewMode::Mono,
            TerrainRole::ProbePass,
            Some(fixups.pass_layout()),
            src,
            label,
        ))
    }

    pub fn log_probe_pass_register_cuts(
        device: &Device,
        uniform_layout: &BindGroupLayout,
        fixups: &crate::renderer::probe_fixup::ProbeFixups,
    ) {
        let base = terrain_shader_for(TerrainRole::ProbePass);
        for (label, edits) in PROBE_PASS_REGISTER_CUTS {
            let mut src = base.clone();
            if let Some((from, _)) = edits.iter().find(|(from, _)| !src.contains(from)) {
                log::warn!("register cut {label}: `{from}` is not in the shader");
                continue;
            }
            for (from, to) in *edits {
                src = src.replacen(from, to, 1);
            }
            let _ = Self::build_from(
                device,
                crate::renderer::brush_pipeline::probe_pass::FORMAT,
                uniform_layout,
                1,
                crate::renderer::multiview::ViewMode::Mono,
                TerrainRole::ProbePass,
                Some(fixups.pass_layout()),
                src,
                label,
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_weather_twins(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        samples: u32,
        probe_layout: &BindGroupLayout,
        fixups: &crate::renderer::probe_fixup::ProbeFixups,
        weather_layout: &BindGroupLayout,
    ) -> TerrainWeatherTwins {
        let twin = |src: String| crate::renderer::weather::with_weather(&src).expect("the ground's weather twin no longer matches its shader");
        let read = terrain_shader_for(TerrainRole::Read);
        let baked = crate::renderer::brush_pipeline::sun_reader_shader(read.clone(), crate::renderer::brush_pipeline::FaceSun::Baked);
        let spotless = crate::renderer::lights::without_spot_shadows;
        let reader = |src: String, label: &str| {
            Self::build_from_with(
                device,
                format,
                uniform_layout,
                samples,
                crate::renderer::multiview::ViewMode::Mono,
                TerrainRole::Read,
                Some(weather_layout),
                Some(probe_layout),
                twin(src),
                label,
            )
        };
        let pass_src = terrain_shader_for(TerrainRole::ProbePass);
        let pass = |src: String, label: &str| {
            Self::build_from_with(
                device,
                crate::renderer::brush_pipeline::probe_pass::FORMAT,
                uniform_layout,
                1,
                crate::renderer::multiview::ViewMode::Mono,
                TerrainRole::ProbePass,
                Some(weather_layout),
                Some(fixups.pass_layout()),
                twin(src),
                label,
            )
        };
        TerrainWeatherTwins {
            readers: [
                reader(read.clone(), "terrain_weather_read"),
                reader(spotless(read), "terrain_weather_read_spotless"),
                reader(baked.clone(), "terrain_weather_read_baked"),
                reader(spotless(baked), "terrain_weather_read_baked_spotless"),
            ],
            pass: pass(pass_src.clone(), "terrain_weather_probe_pass"),
            pass_poolless: pass(crate::renderer::lights::without_pool_maps(pass_src), "terrain_weather_probe_pass_poolless"),
        }
    }

    fn build(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        samples: u32,
        view: crate::renderer::multiview::ViewMode,
        role: TerrainRole,
        group3: Option<&BindGroupLayout>,
    ) -> Self {
        let label = match role {
            TerrainRole::Scene => "terrain_pipeline",
            TerrainRole::Read => "terrain_pipeline_read",
            TerrainRole::ProbePass => "terrain_probe_pass",
        };
        Self::build_from(device, format, uniform_layout, samples, view, role, group3, terrain_shader_for(role), label)
    }

    #[allow(clippy::too_many_arguments)]
    fn build_from(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        samples: u32,
        view: crate::renderer::multiview::ViewMode,
        role: TerrainRole,
        group3: Option<&BindGroupLayout>,
        source: String,
        label: &str,
    ) -> Self {
        Self::build_from_with(device, format, uniform_layout, samples, view, role, None, group3, source, label)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn build_from_with(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        samples: u32,
        view: crate::renderer::multiview::ViewMode,
        role: TerrainRole,
        group2: Option<&BindGroupLayout>,
        group3: Option<&BindGroupLayout>,
        mut source: String,
        label: &str,
    ) -> Self {
        if role == TerrainRole::ProbePass && device.features().contains(wgpu::Features::SHADER_EARLY_DEPTH_TEST) {

            source = source.replacen("@fragment fn fs_main(", "@fragment @early_depth_test(force) fn fs_main(", 1);
        }

        let shader = crate::renderer::shader_checks::audited_shader_module(device, ShaderModuleDescriptor {
            label: Some("terrain_shader"),
            source: ShaderSource::Wgsl(crate::renderer::shader_precision::for_device(device, view.shader(source)).into()),
        });
        let material_layout = material_bind_group_layout(device);
        let mut layouts = vec![Some(uniform_layout), Some(&material_layout)];
        if group2.is_some() || group3.is_some() {
            layouts.push(group2);
        }
        if let Some(l) = group3 {
            layouts.push(Some(l));
        }
        let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("terrain_layout"),
            bind_group_layouts: &layouts,
            immediate_size: 0,
        });

        let probe_targets = crate::renderer::brush_pipeline::probe_pass::targets(false);
        let scene_target = [Some(ColorTargetState { format, blend: Some(BlendState::ALPHA_BLENDING), write_mask: ColorWrites::ALL })];
        let targets: &[Option<ColorTargetState>] =
            if role == TerrainRole::ProbePass { &probe_targets } else { &scene_target };
        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(&layout),
            vertex: VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: PipelineCompilationOptions::default(),

                buffers: &[Some(SolidVertex::layout())],
            },
            fragment: Some(FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: PipelineCompilationOptions::default(),

                targets,
            }),
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                cull_mode: Some(Face::Back),
                front_face: FrontFace::Ccw,
                polygon_mode: PolygonMode::Fill,
                ..Default::default()
            },
            depth_stencil: Some(DepthStencilState {
                format: TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(CompareFunction::Less),
                stencil: StencilState::default(),
                bias: DepthBiasState::default(),
            }),
            multisample: MultisampleState { count: samples, ..Default::default() },
            multiview_mask: view.mask(),
            cache: None,
        });

        Self { pipeline, material_layout }
    }
}

pub struct TerrainWeatherTwins {

    pub readers: [TerrainPipeline; 4],
    pub pass: TerrainPipeline,
    pub pass_poolless: TerrainPipeline,
}

const PROBE_PASS_REGISTER_CUTS: &[(&str, &[(&str, &str)])] = &[
    ("terrain_cut_none", &[]),

    (
        "terrain_cut_gate",
        &[(
            "rgb * seen / a\n            * capsule_glass_beam(p, camera.capsules[glow_i * 2 + 1].xyz);",
            "rgb * seen / a;",
        )],
    ),

    (
        "terrain_cut_relight",
        &[(crate::renderer::lights::SURFACE_RELIT_READ, ""), (crate::renderer::lights::SURFACE_RELIT_CALL, "")],
    ),

    ("terrain_cut_dedup_all", crate::renderer::brush_pipeline::DEDUP_EDITS),

    ("terrain_cut_dedup_sky", &[crate::renderer::brush_pipeline::DEDUP_OUTDOOR_SKY]),
    ("terrain_cut_dedup_portals", &[crate::renderer::brush_pipeline::DEDUP_PORTAL_SAMPLES]),
    ("terrain_cut_dedup_near_far", &[crate::renderer::brush_pipeline::DEDUP_NEAR_FAR]),

    (
        "terrain_cut_dedup_sky_near_far",
        &[crate::renderer::brush_pipeline::DEDUP_OUTDOOR_SKY, crate::renderer::brush_pipeline::DEDUP_NEAR_FAR],
    ),

    ("terrain_cut_escape_colour", &[("    if (h.escaped) {\n        col = probe_escape_colour(", "    if (false) {\n        col = probe_escape_colour(")]),
];

fn terrain_lights_block(options: crate::renderer::lights::LightsBlockOptions) -> String {
    crate::renderer::lights::wgsl_lights_block_with(
        0,
        1,
        crate::renderer::lights::LightsBlockOptions { cull_range_first: true, long_step_from_plane: true, ..options },
    )
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum TerrainRole {

    Scene,

    Read,

    ProbePass,
}

pub fn terrain_shader_for(role: TerrainRole) -> String {
    let src = terrain_shader();
    if role == TerrainRole::Scene {
        return src;
    }
    let plain_lights = terrain_lights_block(crate::renderer::lights::LightsBlockOptions::default());
    assert!(src.contains(&plain_lights), "the terrain shader no longer embeds the lights block as generated");
    let shade = "    let lit = shade_material_env(\n";
    assert!(src.contains(shade), "the terrain shader no longer shades through `shade_material_env`");
    match role {
        TerrainRole::Scene => unreachable!(),
        TerrainRole::Read => {
            let read_lights = terrain_lights_block(crate::renderer::lights::LightsBlockOptions {
                probe_from_pass: true,
                ..Default::default()
            });
            let src = src.replacen(&plain_lights, &read_lights, 1).replacen(
                shade,
                &format!(
                    "    // From the half-resolution probe pass, as the brushes read it. See\n    // `brush_pipeline::probe_pass`.\n    let probe_pass_tolerance = max(4.0 * fwidth(in.clip.z), 1e-6);\n    probe_env_given = probe_pass_upsample(in.clip.xy, in.clip.z, probe_pass_tolerance, 0.0);\n{shade}"
                ),
                1,
            );

            format!(
                "{src}{}{}",
                crate::renderer::brush_pipeline::probe_pass::READER_WGSL,
                crate::renderer::brush_pipeline::probe_pass::FACE_CODE_WGSL,
            )
        }
        TerrainRole::ProbePass => {
            let pass_lights = terrain_lights_block(crate::renderer::lights::LightsBlockOptions {
                defer_secondary: true,
                ..Default::default()
            });
            let src = src.replacen(&plain_lights, &pass_lights, 1);
            let start = src.find(shade).expect("checked above");
            let end = start + src[start..].find("\n}\n").expect("the fragment stage ends");

            format!(
                "{}    probe_fragment = in.clip;\n    return probe_env_for_pass(\n        in.world_pos, shaded_n, rough, clamp(ao_map, 0.0, 1.0), sky_vis, vec3<f32>(0.0), in.world_pos, n,\n    );{}",
                &src[..start],
                &src[end..],
            )
        }
    }
}

pub fn material_bind_group_layout(device: &Device) -> BindGroupLayout {
    device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("terrain_material_layout"),
        entries: &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::D2Array,
                    multisampled: false,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::VERTEX_FRAGMENT,
                ty: BindingType::Sampler(SamplerBindingType::Filtering),
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 9,
                visibility: ShaderStages::VERTEX_FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 3,
                visibility: ShaderStages::VERTEX_FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },

            BindGroupLayoutEntry {
                binding: 5,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::D2Array,
                    multisampled: false,
                },
                count: None,
            },

            BindGroupLayoutEntry {
                binding: 6,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::D2Array,
                    multisampled: false,
                },
                count: None,
            },

            BindGroupLayoutEntry {
                binding: 7,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::D2Array,
                    multisampled: false,
                },
                count: None,
            },
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

            BindGroupLayoutEntry {
                binding: 4,
                visibility: ShaderStages::VERTEX_FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
        ],
    })
}

fn terrain_shader() -> String {
    format!(
        r"
@group(1) @binding(0) var layer_tex: texture_2d_array<f32>;
@group(1) @binding(1) var layer_samp: sampler;
@group(1) @binding(2) var macro_tex: texture_2d<f32>;

struct Material {{
    repeat: vec4<f32>,
    slope_start_deg: f32,
    slope_end_deg: f32,
    height_start: f32,
    height_end: f32,
    macro_repeat: f32,
    macro_strength: f32,
    biplanar_start_deg: f32,
    use_splat: f32,
    normal_strength: f32,
    wet_line: f32,
    wet_band: f32,
    parallax: f32,
    mud_layer: f32,
    deform_metres: f32,
    deform_texel: f32,
    wet_dark: f32,
}}
@group(1) @binding(3) var<uniform> mat: Material;
@group(1) @binding(4) var splat_tex: texture_2d<f32>;
@group(1) @binding(9) var deform_tex: texture_2d<f32>;

const DEFORM_FLAT: f32 = 128.0 / 255.0;
const DEFORM_POOL_FROM: f32 = 0.35;
const DEFORM_POOL_TO: f32 = 0.7;
const DEFORM_NORMAL: f32 = 1.0;

fn mud_weight_v(uv: vec2<f32>) -> f32 {{
    if (mat.mud_layer < -0.5) {{
        return 0.0;
    }}
    if (mat.use_splat > 0.5) {{
        let s = textureSampleLevel(splat_tex, layer_samp, uv, 0.0);
        let total = max(s.x + s.y + s.z + s.w, 0.001);
        let w = select(select(select(s.x, s.y, mat.mud_layer > 0.5), s.z, mat.mud_layer > 1.5), s.w, mat.mud_layer > 2.5);
        return clamp(w / total, 0.0, 1.0);
    }}
    return select(0.0, 1.0, mat.mud_layer < 0.5);
}}

fn mud_weight_f(w: vec4<f32>) -> f32 {{
    if (mat.mud_layer < -0.5) {{
        return 0.0;
    }}
    return select(select(select(w.x, w.y, mat.mud_layer > 0.5), w.z, mat.mud_layer > 1.5), w.w, mat.mud_layer > 2.5);
}}
@group(1) @binding(5) var normal_tex: texture_2d_array<f32>;
@group(1) @binding(7) var rough_tex: texture_2d_array<f32>;
@group(1) @binding(8) var ao_tex: texture_2d_array<f32>;

@group(1) @binding(6) var sky_occ_tex: texture_2d_array<f32>;

{lights_block}
{biplanar_block}
{whiteout_block}

struct VIn  {{ @location(0) pos: vec3<f32>, @location(1) norm: vec3<f32>, @location(2) col: vec4<f32>, @location(3) uv2: vec2<f32> }}
struct VOut {{
    @builtin(position) clip: vec4<f32>,
    @location(0) col: vec4<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) world_pos: vec3<f32>,

    @location(4) tex_pos: vec3<f32>,

    @location(3) uv: vec2<f32>,
}}

@vertex fn vs_main(v: VIn) -> VOut {{
    var out: VOut;
    var pos = v.pos;
    let deform = textureSampleLevel(deform_tex, layer_samp, v.uv2, 0.0).r;
    pos.y = pos.y + (deform - DEFORM_FLAT) * 2.0 * mat.deform_metres * mud_weight_v(v.uv2);
    out.clip      = cam_view_proj() * vec4<f32>(pos, 1.0);
    out.col       = v.col;
    out.normal    = v.norm;
    out.world_pos = pos;
    out.tex_pos   = to_world_space(pos);
    out.uv        = v.uv2;
    return out;
}}

fn sample_planar(layer: i32, world: vec3<f32>, repeat: f32) -> vec3<f32> {{
    let uv = world.xz / max(repeat, 0.001);
    return textureSample(layer_tex, layer_samp, uv, layer).rgb;
}}

fn sample_biplanar(layer: i32, world: vec3<f32>, n: vec3<f32>, repeat: f32) -> vec3<f32> {{
    let b = biplanar_axes(world, n, repeat);
    let c_major = textureSample(layer_tex, layer_samp, b.uv_major, layer).rgb;
    let c_minor = textureSample(layer_tex, layer_samp, b.uv_minor, layer).rgb;
    return mix(c_minor, c_major, b.w);
}}

fn layer_normal(layer: i32, world: vec3<f32>, n: vec3<f32>, repeat: f32) -> vec3<f32> {{
    let uv = world.xz / max(repeat, 0.001);
    let packed = textureSample(normal_tex, layer_samp, uv, layer).rgb;

    var tn = packed * 2.0 - 1.0;
    tn = vec3<f32>(tn.xy * mat.normal_strength, tn.z);

    return normalize(whiteout(1u, world_planar_bend(tn), n));
}}

fn world_planar_bend(tn: vec3<f32>) -> vec3<f32> {{
    let p = to_player_direction(vec3<f32>(tn.x, 0.0, -tn.y));
    return vec3<f32>(p.x, -p.z, tn.z);
}}

fn layer_colour(layer: i32, world: vec3<f32>, n: vec3<f32>, slope_deg: f32) -> vec3<f32> {{
    let repeat = mat.repeat[layer];
    if (slope_deg < mat.biplanar_start_deg) {{
        return sample_planar(layer, world, repeat);
    }}
    return sample_biplanar(layer, world, n, repeat);
}}

struct SampleFrame {{
    planar_uv: vec2<f32>,
    planar_ddx: vec2<f32>,
    planar_ddy: vec2<f32>,
    major_uv: vec2<f32>,
    major_ddx: vec2<f32>,
    major_ddy: vec2<f32>,
    minor_uv: vec2<f32>,
    minor_ddx: vec2<f32>,
    minor_ddy: vec2<f32>,
    blend: f32,
    biplanar: f32,

    detail_uv: vec2<f32>,
    detail_ddx: vec2<f32>,
    detail_ddy: vec2<f32>,
    detail_axis: u32,

    macro_value: f32,
    minor_axis: u32,
}}

const CLIFF_WARP_M: f32 = 2.5;

const CLIFF_BAND_WARP_M: f32 = 3.0;
const CLIFF_BAND_SCALE_M: f32 = 7.0;

fn layers_height(uv: vec2<f32>, ddx: vec2<f32>, ddy: vec2<f32>, layer: i32) -> f32 {{
    let r = max(mat.repeat[layer], 0.001);
    return textureSampleGrad(ao_tex, layer_samp, uv / r, layer, ddx / r, ddy / r).g;
}}

fn dominant_layer(w: vec4<f32>) -> i32 {{
    var best = 0;
    var bw = w.x;
    if (w.y > bw) {{ best = 1; bw = w.y; }}
    if (w.z > bw) {{ best = 2; bw = w.z; }}
    if (w.w > bw) {{ best = 3; }}
    return best;
}}

fn parallax_shift(uv: vec2<f32>, ddx: vec2<f32>, ddy: vec2<f32>, w: vec4<f32>, eye: vec3<f32>, relief: f32) -> vec2<f32> {{
    let up = max(eye.y, 0.12);
    let slide = -eye.xz / up;
    let steps = mix(6.0, 14.0, 1.0 - up);
    let layer = dominant_layer(w);
    let layer_d = relief / steps;
    let step_uv = slide * layer_d;
    var cur_uv = uv;
    var cur_depth = 0.0;
    var surf = (1.0 - layers_height(cur_uv, ddx, ddy, layer)) * relief;
    var prev_uv = uv;
    var prev_depth = 0.0;
    var prev_surf = surf;
    for (var i = 0; i < 14; i++) {{
        if (cur_depth >= surf || f32(i) >= steps) {{
            break;
        }}
        prev_uv = cur_uv;
        prev_depth = cur_depth;
        prev_surf = surf;
        cur_uv = cur_uv + step_uv;
        cur_depth = cur_depth + layer_d;
        surf = (1.0 - layers_height(cur_uv, ddx, ddy, layer)) * relief;
    }}

    let after = surf - cur_depth;
    let before = prev_surf - prev_depth;
    let t = clamp(before / max(before - after, 1e-5), 0.0, 1.0);
    return mix(prev_uv, cur_uv, t) - uv;
}}

const PARALLAX_NEAR_M: f32 = 18.0;
const PARALLAX_FAR_M: f32 = 30.0;

fn sample_frame(world: vec3<f32>, n: vec3<f32>, slope_deg: f32, w: vec4<f32>, eye: vec3<f32>, eye_dist: f32) -> SampleFrame {{
    var f: SampleFrame;
    f.planar_uv  = world.xz;
    f.planar_ddx = dpdx(world.xz);
    f.planar_ddy = dpdy(world.xz);

    let relief = mat.parallax * (1.0 - smoothstep(PARALLAX_NEAR_M, PARALLAX_FAR_M, eye_dist))
        * select(1.0, 0.0, slope_deg >= mat.biplanar_start_deg);
    if (relief > 0.0005) {{
        f.planar_uv = f.planar_uv + parallax_shift(f.planar_uv, f.planar_ddx, f.planar_ddy, w, eye, relief);
    }}

    let b = biplanar_axes(world, n, 1.0);
    f.major_uv  = b.uv_major;
    f.major_ddx = dpdx(b.uv_major);
    f.major_ddy = dpdy(b.uv_major);
    f.minor_uv  = b.uv_minor;
    f.minor_ddx = dpdx(b.uv_minor);
    f.minor_ddy = dpdy(b.uv_minor);
    f.blend     = b.w;
    f.biplanar  = select(0.0, 1.0, slope_deg >= mat.biplanar_start_deg);
    let steep = slope_deg >= mat.biplanar_start_deg;
    f.detail_uv   = select(f.planar_uv, f.major_uv, steep);
    f.detail_ddx  = select(f.planar_ddx, f.major_ddx, steep);
    f.detail_ddy  = select(f.planar_ddy, f.major_ddy, steep);
    f.detail_axis = select(1u, b.axis_major, steep);
    f.minor_axis  = b.axis_minor;
    let mr = max(mat.macro_repeat, 0.001);
    f.macro_value = textureSampleGrad(macro_tex, layer_samp, f.detail_uv / mr, f.detail_ddx / mr, f.detail_ddy / mr).r;
    var warp = select(vec2<f32>(0.0), vec2<f32>(f.macro_value - 0.5, 0.4 - 0.8 * f.macro_value) * CLIFF_WARP_M, steep);

    if (steep) {{
        let ms = 1.0 / CLIFF_BAND_SCALE_M;
        let m2 = textureSampleGrad(macro_tex, layer_samp, f.detail_uv * ms + vec2<f32>(0.31, 0.57), f.detail_ddx * ms, f.detail_ddy * ms).r;
        warp = warp + vec2<f32>(0.35, 1.0) * (m2 - 0.5) * CLIFF_BAND_WARP_M;
    }}
    f.major_uv  = f.major_uv + warp;
    f.minor_uv  = f.minor_uv + warp;
    f.detail_uv = f.detail_uv + warp;
    return f;
}}

fn layer_colour_at(layer: i32, f: SampleFrame) -> vec3<f32> {{
    let r = max(mat.repeat[layer], 0.001);
    if (f.biplanar < 0.5) {{
        return textureSampleGrad(
            layer_tex, layer_samp, f.planar_uv / r, layer,
            f.planar_ddx / r, f.planar_ddy / r,
        ).rgb;
    }}

    let rot = mat2x2<f32>(0.8, 0.6, -0.6, 0.8);
    let s2 = 1.0 / (r * 2.3);
    let c_major = mix(
        textureSampleGrad(layer_tex, layer_samp, f.major_uv / r, layer, f.major_ddx / r, f.major_ddy / r).rgb,
        textureSampleGrad(
            layer_tex, layer_samp, rot * f.major_uv * s2 + vec2<f32>(0.37, 0.11), layer,
            rot * f.major_ddx * s2, rot * f.major_ddy * s2,
        ).rgb,
        0.3 + 0.3 * f.macro_value,
    );
    let c_minor = textureSampleGrad(
        layer_tex, layer_samp, f.minor_uv / r, layer, f.minor_ddx / r, f.minor_ddy / r,
    ).rgb;
    return mix(c_minor, c_major, f.blend);
}}

fn layer_rough_at(layer: i32, f: SampleFrame) -> f32 {{
    let r = max(mat.repeat[layer], 0.001);
    return textureSampleGrad(
        rough_tex, layer_samp, f.detail_uv / r, layer, f.detail_ddx / r, f.detail_ddy / r,
    ).r;
}}

fn layer_ao_at(layer: i32, f: SampleFrame) -> f32 {{
    let r = max(mat.repeat[layer], 0.001);
    return textureSampleGrad(
        ao_tex, layer_samp, f.detail_uv / r, layer, f.detail_ddx / r, f.detail_ddy / r,
    ).r;
}}

fn layer_normal_at(layer: i32, n: vec3<f32>, f: SampleFrame) -> vec3<f32> {{
    let r = max(mat.repeat[layer], 0.001);
    let packed = textureSampleGrad(
        normal_tex, layer_samp, f.detail_uv / r, layer, f.detail_ddx / r, f.detail_ddy / r,
    ).rgb;

    var tn = packed * 2.0 - 1.0;
    tn = vec3<f32>(tn.xy * mat.normal_strength, tn.z);

    if (f.detail_axis != 1u) {{

        let rot = mat2x2<f32>(0.8, 0.6, -0.6, 0.8);
        let s2 = 1.0 / (r * 2.3);
        let p2 = textureSampleGrad(
            normal_tex, layer_samp, rot * f.detail_uv * s2 + vec2<f32>(0.37, 0.11), layer,
            rot * f.detail_ddx * s2, rot * f.detail_ddy * s2,
        ).rgb;
        var t2 = p2 * 2.0 - 1.0;
        t2 = vec3<f32>(transpose(rot) * (t2.xy * mat.normal_strength), t2.z);
        let k = 0.3 + 0.3 * f.macro_value;
        tn = vec3<f32>(mix(tn.xy, t2.xy, k) * 1.3, mix(tn.z, t2.z, k));
        let nw = to_world_direction(n);
        var bent = whiteout(f.detail_axis, tn, nw);

        let w_major = 0.5 + 0.5 * smoothstep(0.5, 0.62, f.blend);
        if (w_major < 0.999 && f.biplanar > 0.5) {{
            let pm = textureSampleGrad(
                normal_tex, layer_samp, f.minor_uv / r, layer, f.minor_ddx / r, f.minor_ddy / r,
            ).rgb;
            var tm = pm * 2.0 - 1.0;
            tm = vec3<f32>(tm.xy * mat.normal_strength, tm.z);
            bent = bent * w_major + whiteout(f.minor_axis, tm, nw) * (1.0 - w_major);
        }}
        return to_player_direction(bent);
    }}
    return whiteout(1u, world_planar_bend(tn), n);
}}

const WEIGHT_EPS: f32 = 0.004;

const TERRAIN_DETAIL_FADE: f32 = 8.0;

fn layer_weights(uv: vec2<f32>, world_y: f32, slope_deg: f32) -> vec4<f32> {{

    let authored_raw = textureSample(splat_tex, layer_samp, uv);
    let total = authored_raw.r + authored_raw.g + authored_raw.b + authored_raw.a;
    let authored = authored_raw / max(total, 0.001);

    let rock_w = smoothstep(mat.slope_start_deg, mat.slope_end_deg, slope_deg);
    var derived = vec4<f32>(1.0 - rock_w, rock_w, 0.0, 0.0);
    if (mat.height_end > mat.height_start) {{
        let high_w = smoothstep(mat.height_start, mat.height_end, world_y);
        derived = vec4<f32>(derived.x * (1.0 - high_w), derived.y * (1.0 - high_w), high_w, 0.0);
    }}

    return select(derived, authored, mat.use_splat > 0.5);
}}

@fragment fn fs_main(in: VOut) -> @location(0) vec4<f32> {{

    let step_x = dpdx(in.world_pos);
    let step_y = dpdy(in.world_pos);
    pixel_footprint = sqrt(length(step_x) * length(step_y));
    let n = normalize(in.normal);
    set_pixel_long_step(n, normalize(cam_pos() - in.world_pos));

    let slope_deg = degrees(acos(clamp(n.y, -1.0, 1.0)));

    let w = layer_weights(in.uv, in.tex_pos.y, slope_deg);

    let f = sample_frame(
        in.tex_pos, to_world_direction(n), slope_deg, w,
        to_world_direction(normalize(cam_pos() - in.world_pos)), distance(in.world_pos, cam_pos()),
    );

    let detail_far = camera.post_params.z;
    let detail = select(1.0, 1.0 - smoothstep(detail_far, detail_far + TERRAIN_DETAIL_FADE, distance(in.world_pos, cam_pos())), detail_far > 0.0);

{layer_reads}
    shaded_n = normalize(select(n, shaded_n, length(shaded_n) > 0.0001));

    let mudw = mud_weight_f(w);
    let deform = textureSampleLevel(deform_tex, layer_samp, in.uv, 0.0);
    let sunk = max(DEFORM_FLAT - deform.r, 0.0) / DEFORM_FLAT * mudw;
    let wet_zone = deform.g * mudw;
    if (mudw > 0.001) {{
        let e = 1.0 / vec2<f32>(textureDimensions(deform_tex));
        let hx = textureSampleLevel(deform_tex, layer_samp, in.uv + vec2<f32>(e.x, 0.0), 0.0).r - textureSampleLevel(deform_tex, layer_samp, in.uv - vec2<f32>(e.x, 0.0), 0.0).r;
        let hz = textureSampleLevel(deform_tex, layer_samp, in.uv + vec2<f32>(0.0, e.y), 0.0).r - textureSampleLevel(deform_tex, layer_samp, in.uv - vec2<f32>(0.0, e.y), 0.0).r;
        let slope = vec2<f32>(hx, hz) * (2.0 * mat.deform_metres / max(2.0 * mat.deform_texel, 1e-4)) * mudw * DEFORM_NORMAL;
        let bent = to_player_direction(normalize(vec3<f32>(-slope.x, 1.0, -slope.y)));
        shaded_n = normalize(shaded_n + bent - to_player_direction(vec3<f32>(0.0, 1.0, 0.0)));
    }}
    let wet = clamp(max(wet_zone, sunk), 0.0, 1.0);
    let pool = smoothstep(DEFORM_POOL_FROM, DEFORM_POOL_TO, sunk);
    albedo = albedo * (1.0 - mat.wet_dark * wet) * (1.0 - 0.35 * pool);
    rough_map = mix(rough_map, rough_map * (1.0 - 0.6 * wet), 1.0);
    rough_map = mix(rough_map, 0.08, pool);

    let rough = specular_aa_roughness(clamp(rough_map, 0.0, 1.0), dpdx(shaded_n), dpdy(shaded_n));

    let m = f.macro_value;
    albedo = albedo * (1.0 + (m - 0.5) * 2.0 * mat.macro_strength);

    albedo = albedo * (1.0 - 0.45 * (1.0 - smoothstep(mat.wet_line, mat.wet_line + mat.wet_band, in.tex_pos.y)));

    let ground_map = textureSample(sky_occ_tex, layer_samp, in.uv, 0);
    let sky_vis = ground_map.r;

    let ground_sun_d = (ground_map.g - 0.5) * (2.0 * {sun_range:?});
    let ground_sun_w = max(max(ground_map.b * {sun_range:?}, 0.5 * fwidth(ground_sun_d)), 0.02);
    receiver_sun_mask = select(-1.0, smoothstep(-ground_sun_w, ground_sun_w, ground_sun_d), ground_map.a < 0.5);

    let st_layers = textureNumLayers(sky_occ_tex) - 1u;
    var st_0 = vec4<f32>(1.0);
    var st_1 = vec4<f32>(1.0);
    var st_2 = vec4<f32>(1.0);
    var st_3 = vec4<f32>(1.0);
    if (st_layers > 0u) {{
        st_0 = textureSample(sky_occ_tex, layer_samp, in.uv, 1);
    }}
    if (st_layers > 1u) {{
        st_1 = textureSample(sky_occ_tex, layer_samp, in.uv, 2);
    }}
    if (st_layers > 2u) {{
        st_2 = textureSample(sky_occ_tex, layer_samp, in.uv, 3);
    }}
    if (st_layers > 3u) {{
        st_3 = textureSample(sky_occ_tex, layer_samp, in.uv, 4);
    }}
    set_stationary_masks(st_0, st_1, st_2, st_3, {stationary_range:?});

    let lit = shade_material_env(
        in.world_pos,
        shaded_n,
        rough,
        clamp(ao_map, 0.0, 1.0),
        sky_vis,
        vec3<f32>(0.0),
        vec4<f32>(0.5, 0.5, 0.5, 0.0),
        albedo,

        in.world_pos,

        n,
    );
    return vec4<f32>(tonemap(in.col.rgb * lit), 1.0);
}}
",
        lights_block = terrain_lights_block(crate::renderer::lights::LightsBlockOptions::default()),
        sun_range = super::brush_pipeline::SUN_MASK_DISTANCE_TEXELS,
        stationary_range = super::brush_pipeline::STATIONARY_MASK_DISTANCE_TEXELS,
        biplanar_block = wgsl_biplanar_block(),
        whiteout_block = wgsl_whiteout_block(),
        layer_reads = LAYER_READS,
    )
}

const LAYER_READS: &str = "    var albedo = vec3<f32>(0.0);
    var shaded_n = vec3<f32>(0.0);
    var rough_map = 0.0;
    var ao_map = 0.0;
    var layers_left = select(0u, 1u, w.x > WEIGHT_EPS) | select(0u, 2u, w.y > WEIGHT_EPS)
        | select(0u, 4u, w.z > WEIGHT_EPS) | select(0u, 8u, w.w > WEIGHT_EPS);
    loop {
        if (layers_left == 0u) {
            break;
        }
        let layer = i32(countTrailingZeros(layers_left));
        layers_left = layers_left & (layers_left - 1u);
        let lw = select(select(select(w.x, w.y, layer == 1), w.z, layer == 2), w.w, layer == 3);
        albedo = albedo + layer_colour_at(layer, f) * lw;
        if (detail > 0.0) {
            shaded_n = shaded_n + layer_normal_at(layer, n, f) * lw;
        }
        rough_map = rough_map + layer_rough_at(layer, f) * lw;
        ao_map = ao_map + layer_ao_at(layer, f) * lw;
    }
    if (detail > 0.0) {
        shaded_n = mix(n, shaded_n, detail);
    }
";

const LAYER_READS_INLINED: &str = "    var albedo = vec3<f32>(0.0);
    if (w.x > WEIGHT_EPS) { albedo = albedo + layer_colour_at(0, f) * w.x; }
    if (w.y > WEIGHT_EPS) { albedo = albedo + layer_colour_at(1, f) * w.y; }
    if (w.z > WEIGHT_EPS) { albedo = albedo + layer_colour_at(2, f) * w.z; }
    if (w.w > WEIGHT_EPS) { albedo = albedo + layer_colour_at(3, f) * w.w; }
    var shaded_n = vec3<f32>(0.0);
    if (detail > 0.0) {
        if (w.x > WEIGHT_EPS) { shaded_n = shaded_n + layer_normal_at(0, n, f) * w.x; }
        if (w.y > WEIGHT_EPS) { shaded_n = shaded_n + layer_normal_at(1, n, f) * w.y; }
        if (w.z > WEIGHT_EPS) { shaded_n = shaded_n + layer_normal_at(2, n, f) * w.z; }
        if (w.w > WEIGHT_EPS) { shaded_n = shaded_n + layer_normal_at(3, n, f) * w.w; }
        shaded_n = mix(n, shaded_n, detail);
    }
    var rough_map = 0.0;
    var ao_map = 0.0;
    if (w.x > WEIGHT_EPS) { rough_map = rough_map + layer_rough_at(0, f) * w.x; ao_map = ao_map + layer_ao_at(0, f) * w.x; }
    if (w.y > WEIGHT_EPS) { rough_map = rough_map + layer_rough_at(1, f) * w.y; ao_map = ao_map + layer_ao_at(1, f) * w.y; }
    if (w.z > WEIGHT_EPS) { rough_map = rough_map + layer_rough_at(2, f) * w.z; ao_map = ao_map + layer_ao_at(2, f) * w.z; }
    if (w.w > WEIGHT_EPS) { rough_map = rough_map + layer_rough_at(3, f) * w.w; ao_map = ao_map + layer_ao_at(3, f) * w.w; }
";

pub fn with_inlined_layer_reads(src: String) -> String {
    assert_eq!(src.matches(LAYER_READS).count(), 1, "the ground shader's layer reads moved");
    src.replacen(LAYER_READS, LAYER_READS_INLINED, 1)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::renderer::cuboid::SolidVertex;
    use crate::renderer::lights::LightsUniform;
    use wgpu::util::DeviceExt;

    pub fn headless_gpu() -> Option<(Device, Queue)> {
        let instance = Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&RequestAdapterOptions {
            apply_limit_buckets: false,
            power_preference: PowerPreference::default(),
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok()?;
        pollster::block_on(adapter.request_device(&DeviceDescriptor {
            required_features: Features::empty(),
            required_limits: crate::renderer::uniforms::scene_limits(Limits::default()),
            ..Default::default()
        }))
        .ok()
    }

    fn vertex(pos: [f32; 3], normal: [f32; 3]) -> SolidVertex {
        SolidVertex {
            position: pos,
            normal,
            color: [1.0, 1.0, 1.0, 1.0],
            uv2: [0.0, 0.0],
            reflectivity: 0.0,
        }
    }

    #[derive(Copy, Clone)]
    pub enum Palette {
        Test,
        Fallback,

        Ramp,
    }

    pub fn render_quad_with_normal(normal: [f32; 3]) -> Option<[u8; 4]> {
        render_quad(normal, Palette::Test, None, None)
    }

    pub fn render_quad_with_splat(normal: [f32; 3], splat: &TerrainImage) -> Option<[u8; 4]> {
        render_quad(normal, Palette::Test, Some(splat), None)
    }

    pub fn render_quad_with_fallback_material(normal: [f32; 3]) -> Option<[u8; 4]> {
        render_quad(normal, Palette::Fallback, None, None)
    }

    fn render_quad(
        normal: [f32; 3],
        palette: Palette,
        splat: Option<&TerrainImage>,
        sky_occlusion: Option<&TerrainImage>,
    ) -> Option<[u8; 4]> {
        render_quad_full(normal, palette, splat, sky_occlusion, &[None, None, None, None], false)
    }

    pub fn render_quad_full(
        normal: [f32; 3],
        palette: Palette,
        splat: Option<&TerrainImage>,
        sky_occlusion: Option<&TerrainImage>,
        normals: &[Option<TerrainImage>],
        lit: bool,
    ) -> Option<[u8; 4]> {
        render_quad_lit_by(normal, palette, splat, sky_occlusion, normals, lit.then_some(None))
    }

    pub fn render_quad_lit_by(
        normal: [f32; 3],
        palette: Palette,
        splat: Option<&TerrainImage>,
        sky_occlusion: Option<&TerrainImage>,
        normals: &[Option<TerrainImage>],
        light: Option<Option<u8>>,
    ) -> Option<[u8; 4]> {
        render_quad_light_at(normal, palette, splat, sky_occlusion, normals, light, glam::Vec3::new(3.0, 2.0, 0.0))
    }

    pub fn render_quad_light_at(
        normal: [f32; 3],
        palette: Palette,
        splat: Option<&TerrainImage>,
        sky_occlusion: Option<&TerrainImage>,
        normals: &[Option<TerrainImage>],
        light: Option<Option<u8>>,
        light_pos: glam::Vec3,
    ) -> Option<[u8; 4]> {
        render_quad_turned(normal, palette, splat, sky_occlusion, normals, light, light_pos, 0.0)
    }

    pub fn render_quad_turned(
        normal: [f32; 3],
        palette: Palette,
        splat: Option<&TerrainImage>,
        sky_occlusion: Option<&TerrainImage>,
        normals: &[Option<TerrainImage>],
        light: Option<Option<u8>>,
        light_pos: glam::Vec3,
        yaw: f32,
    ) -> Option<[u8; 4]> {

        let read = glam::Vec3::new(0.125, -0.125, 0.5);
        let yaw_inv = glam::Quat::from_rotation_y(-yaw);
        let to_player = |p: glam::Vec3| read + yaw_inv * (p - read);
        let light_pos = to_player(light_pos);

        const WORLD_SHIFT: glam::Vec3 = glam::Vec3::new(2.375, 0.625, 4.0);
        let offset = read + WORLD_SHIFT - glam::Quat::from_rotation_y(yaw) * read;
        let normal = (yaw_inv * glam::Vec3::from(normal)).to_array();
        let lit = light.is_some();
        let (device, queue) = headless_gpu()?;
        let format = TextureFormat::Rgba8Unorm;

        let lights = LightsUniform::new(&device);
        let (_shadows, uniforms) =
            crate::renderer::uniforms::test_support::scene_uniforms(&device, &lights);

        uniforms.upload_scene(
            &queue,
            glam::Mat4::IDENTITY,

            to_player(glam::Vec3::new(0.0, 0.0, 5.0)),
            &crate::renderer::uniforms::ShadowUpload::disabled(),
            &crate::renderer::uniforms::SkyUpload::none(),
            &crate::renderer::uniforms::PostUpload::default(),
            &crate::renderer::uniforms::PlayerUpload { offset, yaw, ..Default::default() },
        );
        if lit {

            lights.upload(&queue, &[crate::renderer::lights::Light {
                mask_channel: light.flatten(),
                shadow_near: None,
                source_radius: 0.0,
                in_level_bake: true,
                position: light_pos,
                direction: glam::Vec3::new(0.0, -1.0, 0.0),
                kind: crate::renderer::lights::LightKind::Point,
                color: crate::renderer::Color3(255, 255, 255, 255),

                intensity: 3.0,
                range: 50.0,
                cone_angle_deg: 180.0,
                inner_cone_angle_deg: 0.0,
            }]);
        } else {
            lights.upload(&queue, &[]);
        }

        let pipeline = TerrainPipeline::new(&device, format, &uniforms.layout);

        let solid = |rgb: [u8; 4]| TerrainImage { width: 1, height: 1, rgba: rgb.to_vec() };
        let test_layers = [
            solid([255, 0, 0, 255]),
            solid([0, 0, 255, 255]),
            solid([0, 255, 0, 255]),
            solid([255, 255, 255, 255]),
        ];
        let neutral = solid([128, 128, 128, 255]);

        let material = match palette {
            Palette::Test => TerrainMaterial::new(
                &device, &queue, &pipeline.material_layout,
                &test_layers, &neutral, splat, sky_occlusion, normals,
                &[], &[], TerrainMaterialUniform::default(),
            ),
            Palette::Fallback => {
                TerrainMaterial::fallback(&device, &queue, &pipeline.material_layout)
            }
            Palette::Ramp => {
                let ramp = TerrainImage {
                    width: 64,
                    height: 64,
                    rgba: (0..64 * 64).flat_map(|i| [(i % 64) as u8 * 4, 0, 0, 255]).collect(),
                };

                let layers = [ramp, solid_image([0, 0, 255], 64, 64), solid_image([0, 255, 0], 64, 64), solid_image([255, 255, 255], 64, 64)];
                TerrainMaterial::new(
                    &device, &queue, &pipeline.material_layout,
                    &layers, &neutral, splat, sky_occlusion, normals,
                    &[], &[], TerrainMaterialUniform::default(),
                )
            }
        };

        let verts = [
            vertex([-1.0, -1.0, 0.5], normal),
            vertex([3.0, -1.0, 0.5], normal),
            vertex([-1.0, 3.0, 0.5], normal),
        ];
        let vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("test_vb"),
            contents: bytemuck::cast_slice(&verts),
            usage: BufferUsages::VERTEX,
        });
        let ib = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("test_ib"),
            contents: bytemuck::cast_slice(&[0u32, 1, 2]),
            usage: BufferUsages::INDEX,
        });

        const SIZE: u32 = 8;
        let target = device.create_texture(&TextureDescriptor {
            label: Some("test_target"),
            size: Extent3d { width: SIZE, height: SIZE, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1,
            dimension: TextureDimension::D2,
            format,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let depth = device.create_texture(&TextureDescriptor {
            label: Some("test_depth"),
            size: Extent3d { width: SIZE, height: SIZE, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Depth32Float,
            usage: TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let target_view = target.create_view(&Default::default());
        let depth_view = depth.create_view(&Default::default());

        let readback = device.create_buffer(&BufferDescriptor {
            label: Some("test_readback"),
            size: u64::from(256 * SIZE),
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("test_pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &target_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: Operations { load: LoadOp::Clear(Color::BLACK), store: StoreOp::Store },
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
            pass.set_bind_group(1, &material.bind_group, &[]);
            pass.set_vertex_buffer(0, vb.slice(..));
            pass.set_index_buffer(ib.slice(..), IndexFormat::Uint32);
            pass.draw_indexed(0..3, 0, 0..1);
        }
        encoder.copy_texture_to_buffer(
            TexelCopyTextureInfo {
                texture: &target, mip_level: 0, origin: Origin3d::ZERO, aspect: TextureAspect::All,
            },
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
        let centre = (SIZE / 2) as usize * 256 + (SIZE / 2) as usize * 4;
        Some([data[centre], data[centre + 1], data[centre + 2], data[centre + 3]])
    }

    #[test]

    #[test]
    fn the_probe_pass_terrain_shaders_validate_and_build() {
        use wgpu::naga;

        let poolless = crate::renderer::lights::without_pool_maps(terrain_shader_for(TerrainRole::ProbePass));
        let read = terrain_shader_for(TerrainRole::Read);
        let baked = crate::renderer::brush_pipeline::sun_reader_shader(read.clone(), crate::renderer::brush_pipeline::FaceSun::Baked);
        let spotless = crate::renderer::lights::without_spot_shadows;
        for (role, src) in [
            ("read", read.clone()),
            ("pass", terrain_shader_for(TerrainRole::ProbePass)),
            ("poolless pass", poolless),
            ("spotless read", spotless(read.clone())),
            ("baked read", baked.clone()),
            ("baked spotless read", spotless(baked)),
            ("inlined read", with_inlined_layer_reads(read)),
            ("inlined pass", with_inlined_layer_reads(terrain_shader_for(TerrainRole::ProbePass))),
        ] {
            let module = naga::front::wgsl::parse_str(&src)
                .unwrap_or_else(|e| panic!("{role}: {}", e.emit_to_string(&src)));
            naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
                .validate(&module)
                .unwrap_or_else(|e| panic!("{role}: {e:?}"));
        }
        assert!(terrain_shader_for(TerrainRole::Read).contains("const PROBE_ENV_FROM_PASS: bool = true;"));
        let pass = terrain_shader_for(TerrainRole::ProbePass);
        assert!(pass.contains("const PROBE_SECONDARY_DEFERRED: bool = true;") && pass.contains("return probe_env_for_pass("));
        let Some((device, _queue)) = headless_gpu() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let lights = LightsUniform::new(&device);
        let (_shadows, uniforms) = crate::renderer::uniforms::test_support::scene_uniforms(&device, &lights);
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let probe_layout = crate::renderer::brush_pipeline::probe_pass::bind_group_layout(&device);
        let _read = TerrainPipeline::new_probe_reader(&device, TextureFormat::Rgba8UnormSrgb, &uniforms.layout, 4, &probe_layout);
        let fixups = crate::renderer::probe_fixup::ProbeFixups::new(&device, &uniforms.layout, 1024);
        let _pass = TerrainPipeline::new_probe_pass(&device, &uniforms.layout, &fixups);
        let _poolless = TerrainPipeline::new_probe_pass_poolless(&device, &uniforms.layout, &fixups);
        let _twins =
            TerrainPipeline::new_probe_reader_twins(&device, TextureFormat::Rgba8UnormSrgb, &uniforms.layout, 4, &probe_layout);
        let _inlined =
            TerrainPipeline::new_inlined(&device, TextureFormat::Rgba8UnormSrgb, &uniforms.layout, 4, &probe_layout, &fixups);
        let err = pollster::block_on(scope.pop());
        assert!(err.is_none(), "the terrain probe pipelines failed to build: {err:?}");
    }

    #[test]
    fn the_ground_readers_twins_and_inlined_reads_change_only_their_switches() {
        let read = terrain_shader_for(TerrainRole::Read);
        let baked = crate::renderer::brush_pipeline::sun_reader_shader(read.clone(), crate::renderer::brush_pipeline::FaceSun::Baked);
        assert_eq!(baked.replacen("const SUN_MASK_EVERYWHERE: bool = true;", "const SUN_MASK_EVERYWHERE: bool = false;", 1), read);
        let spotless = crate::renderer::lights::without_spot_shadows(read.clone());
        assert_ne!(spotless, read, "the spotless twin switched nothing off");
        let inlined = with_inlined_layer_reads(read.clone());
        assert_eq!(inlined.replacen(LAYER_READS_INLINED, LAYER_READS, 1), read);

        for call in ["layer_colour_at(", "layer_normal_at(", "layer_rough_at(", "layer_ao_at("] {
            let body = |src: &str| src.split("@fragment").nth(1).map_or(0, |entry| entry.matches(call).count());
            assert_eq!((body(&read), body(&inlined)), (1, 4), "{call}");
        }
    }

    #[test]
    fn a_ground_map_is_baked_everywhere_only_when_every_texel_is() {
        let map = |alphas: &[u8]| TerrainImage {
            width: alphas.len() as u32,
            height: 1,
            rgba: alphas.iter().flat_map(|&a| [200, 128, 0, a]).collect(),
        };
        assert!(map(&[0, 0, 0, 0]).sun_baked_everywhere());
        assert!(map(&[0, 127]).sun_baked_everywhere(), "127 reads as 0.498, under the shader's half");
        assert!(!map(&[0, 128]).sun_baked_everywhere(), "128 reads as 0.502: not baked there");
        assert!(!map(&[255]).sun_baked_everywhere());
        assert!(!FULL_SKY.sun_baked_everywhere(), "no map is no bake");
        assert!(!TerrainImage { width: 0, height: 0, rgba: Vec::new() }.sun_baked_everywhere());
    }

    #[test]
    fn every_terrain_register_cut_applies_and_validates() {
        use wgpu::naga;
        let base = terrain_shader_for(TerrainRole::ProbePass);
        for (label, edits) in PROBE_PASS_REGISTER_CUTS {
            let mut src = base.clone();
            for (from, to) in *edits {
                assert!(src.contains(from), "{label}: `{from}` is not in the ground's probe pass");

                assert_eq!(src.matches(from).count(), 1, "{label}: `{from}` is in the ground's probe pass more than once");
                src = src.replacen(from, to, 1);
            }
            let module = naga::front::wgsl::parse_str(&src)
                .unwrap_or_else(|e| panic!("{label}: {}", e.emit_to_string(&src)));
            naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
                .validate(&module)
                .unwrap_or_else(|e| panic!("{label}: {e:?}"));
        }
    }

    #[test]
    fn weightless_layers_are_skipped_rather_than_sampled() {
        let src = super::terrain_shader();
        assert!(
            src.contains("textureSampleGrad"),
            "explicit gradients are what make a per-fragment branch legal at all",
        );
        for w in ["w.x > WEIGHT_EPS", "w.y > WEIGHT_EPS", "w.z > WEIGHT_EPS", "w.w > WEIGHT_EPS"] {
            assert!(src.contains(w), "missing the skip for {w}");
        }
    }

    #[test]
    fn the_gradients_are_taken_outside_every_branch() {

        let src = super::terrain_shader();
        let frame = src
            .split("fn sample_frame")
            .nth(1)
            .and_then(|s| s.split("\nfn ").next())
            .expect("sample_frame present");
        let entry = src
            .split("@fragment")
            .nth(1)
            .expect("fragment entry present");
        let allowed = |tok: &str| frame.matches(tok).count() + entry.matches(tok).count();
        assert_eq!(
            src.matches("dpdx(").count(),
            allowed("dpdx("),
            "a derivative escaped both sample_frame and the fragment entry",
        );
        assert_eq!(src.matches("dpdy(").count(), allowed("dpdy("));

        let footprint = "let step_x = dpdx(in.world_pos);\n    let step_y = dpdy(in.world_pos);";
        let extra = usize::from(entry.contains(footprint));
        assert!(
            entry.matches("dpdx(").count() <= 1 + extra && entry.matches("dpdy(").count() <= 1 + extra,
            "more than the two named derivative pairs in the fragment entry; \
             sampling gradients belong in sample_frame",
        );
    }

    #[test]
    fn baked_sky_occlusion_darkens_the_terrain() {

        let sealed = TerrainImage { width: 1, height: 1, rgba: vec![0, 0, 0, 255] };
        let Some(open) = render_quad([0.0, 1.0, 0.0], Palette::Test, None, None) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let dark = render_quad([0.0, 1.0, 0.0], Palette::Test, None, Some(&sealed)).unwrap();
        assert!(
            dark[0] < open[0],
            "zero sky visibility must darken the ground: {dark:?} vs open {open:?}",
        );
    }

    #[test]
    fn unbaked_terrain_shades_exactly_as_it_did_before() {

        let full = TerrainImage { width: 1, height: 1, rgba: vec![255, 255, 255, 255] };
        let Some(absent) = render_quad([0.0, 1.0, 0.0], Palette::Test, None, None) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let explicit = render_quad([0.0, 1.0, 0.0], Palette::Test, None, Some(&full)).unwrap();
        assert_eq!(
            absent, explicit,
            "no occlusion map must shade identically to a fully-open one: \
             {absent:?} vs {explicit:?}",
        );
    }

    #[test]
    fn a_mip_chain_runs_all_the_way_down_to_one_texel() {

        assert_eq!(mip_levels_for(1024, 1024), 11);
        assert_eq!(mip_levels_for(1, 1), 1);
        assert_eq!(mip_levels_for(256, 64), 9, "levels follow the LONGER side");

        let levels = mip_levels_for(8, 8);
        let chain = mip_chain(&[128u8; 8 * 8 * 4], 8, 8, levels);
        assert_eq!(chain.len() as u32, levels);
        assert_eq!((chain[0].1, chain[0].2), (8, 8));
        assert_eq!(chain.last().unwrap().1, 1, "chain must reach 1x1");
        for (data, w, h) in &chain {
            assert_eq!(data.len(), (w * h * 4) as usize, "level {w}x{h} is the wrong size");
        }
    }

    #[test]
    fn mips_are_averaged_in_linear_space_not_srgb() {

        let mut px = vec![0u8; 2 * 2 * 4];
        for i in 0..4 {
            let v = if i % 2 == 0 { 0u8 } else { 255u8 };
            for c in 0..3 {
                px[i * 4 + c] = v;
            }
            px[i * 4 + 3] = 255;
        }
        let chain = mip_chain(&px, 2, 2, 2);
        let r = chain[1].0[0];
        assert!(
            (185..=191).contains(&r),
            "black+white must average to sRGB ~188 (linear 0.5), got {r} -- \
             {} suggests the average was taken on the encoded bytes",
            if r < 140 { "which" } else { "this" },
        );
        assert_eq!(chain[1].0[3], 255, "alpha is already linear and must not be transferred");
    }

    #[test]
    fn a_flat_colour_survives_every_mip_level() {

        let chain = mip_chain(&[200u8; 16 * 16 * 4], 16, 16, mip_levels_for(16, 16));
        for (data, w, h) in &chain {
            for (i, b) in data.iter().enumerate() {
                assert!(
                    b.abs_diff(200) <= 1,
                    "level {w}x{h} byte {i} drifted to {b} from 200",
                );
            }
        }
    }

    fn flat_ground_shades_from_the_ground_layer() {
        let Some(px) = render_quad_with_normal([0.0, 1.0, 0.0]) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };

        assert!(px[0] > px[2], "flat ground should take the ground layer (red), got {px:?}");
    }

    #[test]
    fn a_cliff_shades_from_the_rock_layer() {

        let Some(px) = render_quad_with_normal([1.0, 0.0, 0.0]) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        assert!(px[2] > px[0], "a cliff should take the rock layer (blue), got {px:?}");
    }

    #[test]
    fn the_slope_blend_is_gradual_rather_than_a_hard_switch() {

        let a = 31.0_f32.to_radians();
        let Some(px) = render_quad_with_normal([a.sin(), a.cos(), 0.0]) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        assert!(px[0] > 8, "ground layer vanished mid-blend: {px:?}");
        assert!(px[2] > 8, "rock layer absent mid-blend: {px:?}");
    }
}

pub struct TerrainMaterial {
    pub bind_group: BindGroup,
    pub uniform: Buffer,

    pub splat_texture: Texture,

    pub splat_size: [u32; 2],
}

pub fn mip_levels_for(w: u32, h: u32) -> u32 {
    32 - w.max(h).max(1).leading_zeros()
}

pub const DEFAULT_TERRAIN_ROUGHNESS: u8 = 200;

const TERRAIN_NORMAL_VARIANCE_CLAMP: f32 = 0.18;

pub fn roughness_chain_with_normal_variance(
    rough: &[u8],
    normal: &[u8],
    w: u32,
    h: u32,
    levels: u32,
) -> Vec<(Vec<u8>, u32, u32)> {
    let mut chain = mip_chain_linear(rough, w, h, levels);

    let mut vecs: Vec<[f32; 3]> = (0..(w * h) as usize)
        .map(|i| {
            [
                f32::from(normal[i * 4]) / 255.0 * 2.0 - 1.0,
                f32::from(normal[i * 4 + 1]) / 255.0 * 2.0 - 1.0,
                f32::from(normal[i * 4 + 2]) / 255.0 * 2.0 - 1.0,
            ]
        })
        .collect();
    let (mut cw, mut ch) = (w, h);

    for level in 1..chain.len() {
        let (nw, nh) = ((cw / 2).max(1), (ch / 2).max(1));
        let mut next = vec![[0.0f32; 3]; (nw * nh) as usize];
        for y in 0..nh {
            for x in 0..nw {
                let mut acc = [0.0f32; 3];
                for (dx, dy) in [(0u32, 0u32), (1, 0), (0, 1), (1, 1)] {
                    let sx = (x * 2 + dx).min(cw - 1);
                    let sy = (y * 2 + dy).min(ch - 1);
                    let v = vecs[(sy * cw + sx) as usize];
                    acc[0] += v[0];
                    acc[1] += v[1];
                    acc[2] += v[2];
                }
                next[(y * nw + x) as usize] = [acc[0] / 4.0, acc[1] / 4.0, acc[2] / 4.0];
            }
        }
        vecs = next;
        cw = nw;
        ch = nh;

        let (data, lw, lh) = &mut chain[level];
        if *lw != cw || *lh != ch {

            break;
        }
        for i in 0..(cw * ch) as usize {
            let v = vecs[i];
            let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            let sigma2 = if len > 1e-4 { (1.0 - len) / len } else { 1.0 };
            let kernel = (2.0 * sigma2).min(TERRAIN_NORMAL_VARIANCE_CLAMP);
            let base = f32::from(data[i * 4]) / 255.0;
            let filtered = (base * base + kernel).clamp(0.0, 1.0).sqrt();
            let byte = (filtered * 255.0).round().clamp(0.0, 255.0) as u8;
            data[i * 4] = byte;
            data[i * 4 + 1] = byte;
            data[i * 4 + 2] = byte;
        }
    }
    chain
}

pub fn mip_chain_linear(rgba: &[u8], w: u32, h: u32, levels: u32) -> Vec<(Vec<u8>, u32, u32)> {
    let mut out = vec![(rgba.to_vec(), w, h)];
    for _ in 1..levels {
        let (src, sw, sh) = out.last().unwrap();
        let (dw, dh) = ((sw / 2).max(1), (sh / 2).max(1));
        let mut dst = vec![0u8; (dw * dh * 4) as usize];
        for y in 0..dh {
            for x in 0..dw {
                for c in 0..4 {
                    let fetch = |sx: u32, sy: u32| -> f32 {
                        let i = ((sy.min(sh - 1) * sw + sx.min(sw - 1)) * 4 + c) as usize;
                        f32::from(src[i]) / 255.0
                    };
                    let (x0, y0) = (x * 2, y * 2);
                    let avg = (fetch(x0, y0) + fetch(x0 + 1, y0) + fetch(x0, y0 + 1)
                        + fetch(x0 + 1, y0 + 1))
                        * 0.25;
                    dst[((y * dw + x) * 4 + c) as usize] =
                        (avg.clamp(0.0, 1.0) * 255.0).round() as u8;
                }
            }
        }
        out.push((dst, dw, dh));
    }
    out
}

pub fn mip_chain(rgba: &[u8], w: u32, h: u32, levels: u32) -> Vec<(Vec<u8>, u32, u32)> {
    fn to_linear(c: u8) -> f32 {
        let s = f32::from(c) / 255.0;
        if s <= 0.040_45 { s / 12.92 } else { ((s + 0.055) / 1.055).powf(2.4) }
    }
    fn to_srgb(l: f32) -> u8 {
        let s = if l <= 0.003_130_8 { l * 12.92 } else { 1.055 * l.powf(1.0 / 2.4) - 0.055 };
        (s.clamp(0.0, 1.0) * 255.0).round() as u8
    }

    let mut out = vec![(rgba.to_vec(), w, h)];
    for _ in 1..levels {
        let (src, sw, sh) = out.last().unwrap();
        let (dw, dh) = ((sw / 2).max(1), (sh / 2).max(1));
        let mut dst = vec![0u8; (dw * dh * 4) as usize];
        for y in 0..dh {
            for x in 0..dw {
                for c in 0..4 {

                    let fetch = |sx: u32, sy: u32| -> f32 {
                        let i = ((sy.min(sh - 1) * sw + sx.min(sw - 1)) * 4 + c) as usize;
                        if c == 3 { f32::from(src[i]) / 255.0 } else { to_linear(src[i]) }
                    };
                    let (x0, y0) = (x * 2, y * 2);
                    let avg = (fetch(x0, y0) + fetch(x0 + 1, y0) + fetch(x0, y0 + 1)
                        + fetch(x0 + 1, y0 + 1))
                        * 0.25;
                    dst[((y * dw + x) * 4 + c) as usize] = if c == 3 {
                        (avg.clamp(0.0, 1.0) * 255.0).round() as u8
                    } else {
                        to_srgb(avg)
                    };
                }
            }
        }
        out.push((dst, dw, dh));
    }
    out
}

impl TerrainMaterial {

    pub fn new(
        device: &Device,
        queue: &Queue,
        layout: &BindGroupLayout,
        layers: &[TerrainImage],
        macro_image: &TerrainImage,

        splat: Option<&TerrainImage>,
        sky_occlusion: Option<&TerrainImage>,

        normals: &[Option<TerrainImage>],

        rough: &[Option<TerrainImage>],
        ao: &[Option<TerrainImage>],
        settings: TerrainMaterialUniform,
    ) -> Self {
        Self::new_deformed(device, queue, layout, layers, macro_image, splat, sky_occlusion, normals, rough, ao, settings, None)
    }

    pub fn flat_deform_view(device: &Device, queue: &Queue) -> TextureView {
        let tex = device.create_texture(&TextureDescriptor {
            label: Some("terrain_deform_flat"),
            size: Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rg8Unorm,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            TexelCopyTextureInfo { texture: &tex, mip_level: 0, origin: Origin3d::ZERO, aspect: TextureAspect::All },
            &[128, 0],
            TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(2), rows_per_image: Some(1) },
            Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
        );
        tex.create_view(&TextureViewDescriptor::default())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_deformed(
        device: &Device,
        queue: &Queue,
        layout: &BindGroupLayout,
        layers: &[TerrainImage],
        macro_image: &TerrainImage,
        splat: Option<&TerrainImage>,
        sky_occlusion: Option<&TerrainImage>,
        normals: &[Option<TerrainImage>],
        rough: &[Option<TerrainImage>],
        ao: &[Option<TerrainImage>],
        settings: TerrainMaterialUniform,
        deform: Option<&TextureView>,
    ) -> Self {
        assert!(!layers.is_empty(), "terrain material needs at least one layer");
        let (w, h) = (layers[0].width, layers[0].height);
        assert!(
            layers.iter().all(|l| l.width == w && l.height == h),
            "all terrain layers must share one size to live in a D2Array",
        );

        let mip_level_count = mip_levels_for(w, h);
        let array = device.create_texture(&TextureDescriptor {
            label: Some("terrain_layers"),
            size: Extent3d { width: w, height: h, depth_or_array_layers: 4 },
            mip_level_count,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8UnormSrgb,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });
        for slot in 0..4 {
            let src = layers.get(slot).unwrap_or_else(|| layers.last().unwrap());
            let chain = mip_chain(&src.rgba, w, h, mip_level_count);
            for (level, (data, lw, lh)) in chain.iter().enumerate() {
                queue.write_texture(
                    TexelCopyTextureInfo {
                        texture: &array,
                        mip_level: level as u32,
                        origin: Origin3d { x: 0, y: 0, z: slot as u32 },
                        aspect: TextureAspect::All,
                    },
                    data,
                    TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(4 * lw),
                        rows_per_image: Some(*lh),
                    },
                    Extent3d { width: *lw, height: *lh, depth_or_array_layers: 1 },
                );
            }
        }

        let macro_tex = device.create_texture(&TextureDescriptor {
            label: Some("terrain_macro"),
            size: Extent3d { width: macro_image.width, height: macro_image.height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,

            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            TexelCopyTextureInfo {
                texture: &macro_tex,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            &macro_image.rgba,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * macro_image.width),
                rows_per_image: Some(macro_image.height),
            },
            Extent3d { width: macro_image.width, height: macro_image.height, depth_or_array_layers: 1 },
        );

        let normal_array = device.create_texture(&TextureDescriptor {
            label: Some("terrain_normals"),
            size: Extent3d { width: w, height: h, depth_or_array_layers: 4 },
            mip_level_count,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });
        for slot in 0..4 {
            let flat = solid_image(FLAT_NORMAL, w, h);
            let src = match normals.get(slot).and_then(|n| n.as_ref()) {
                Some(img) if img.width == w && img.height == h => img.clone_image(),
                Some(img) => resample(img, w, h),
                None => flat,
            };

            let chain = mip_chain_linear(&src.rgba, w, h, mip_level_count);
            for (level, (data, lw, lh)) in chain.iter().enumerate() {
                queue.write_texture(
                    TexelCopyTextureInfo {
                        texture: &normal_array,
                        mip_level: level as u32,
                        origin: Origin3d { x: 0, y: 0, z: slot as u32 },
                        aspect: TextureAspect::All,
                    },
                    data,
                    TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(4 * lw),
                        rows_per_image: Some(*lh),
                    },
                    Extent3d { width: *lw, height: *lh, depth_or_array_layers: 1 },
                );
            }
        }

        let mut rough_arr = Vec::with_capacity(4);
        let mut ao_arr = Vec::with_capacity(4);
        for slot in 0..4 {
            let pick = |src: &[Option<TerrainImage>], fallback: u8| -> TerrainImage {
                match src.get(slot).and_then(|x| x.as_ref()) {
                    Some(img) if img.width == w && img.height == h => img.clone_image(),
                    Some(img) => resample(img, w, h),
                    None => solid_image([fallback; 3], w, h),
                }
            };
            rough_arr.push(pick(rough, DEFAULT_TERRAIN_ROUGHNESS));
            ao_arr.push(pick(ao, 255));
        }

        let data_array = |label: &str| {
            device.create_texture(&TextureDescriptor {
                label: Some(label),
                size: Extent3d { width: w, height: h, depth_or_array_layers: 4 },
                mip_level_count,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba8Unorm,
                usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
                view_formats: &[],
            })
        };
        let rough_array = data_array("terrain_rough");
        let ao_array = data_array("terrain_ao");

        for slot in 0..4 {
            let normal_src = match normals.get(slot).and_then(|n| n.as_ref()) {
                Some(img) if img.width == w && img.height == h => img.clone_image(),
                Some(img) => resample(img, w, h),
                None => solid_image(FLAT_NORMAL, w, h),
            };
            let chains = [
                (
                    &rough_array,
                    roughness_chain_with_normal_variance(
                        &rough_arr[slot].rgba,
                        &normal_src.rgba,
                        w,
                        h,
                        mip_level_count,
                    ),
                ),
                (&ao_array, mip_chain_linear(&ao_arr[slot].rgba, w, h, mip_level_count)),
            ];
            for (tex, chain) in chains {
                for (level, (data, lw, lh)) in chain.iter().enumerate() {
                    queue.write_texture(
                        TexelCopyTextureInfo {
                            texture: tex,
                            mip_level: level as u32,
                            origin: Origin3d { x: 0, y: 0, z: slot as u32 },
                            aspect: TextureAspect::All,
                        },
                        data,
                        TexelCopyBufferLayout {
                            offset: 0,
                            bytes_per_row: Some(4 * lw),
                            rows_per_image: Some(*lh),
                        },
                        Extent3d { width: *lw, height: *lh, depth_or_array_layers: 1 },
                    );
                }
            }
        }

        let sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("terrain_sampler"),
            address_mode_u: AddressMode::Repeat,
            address_mode_v: AddressMode::Repeat,
            address_mode_w: AddressMode::Repeat,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: MipmapFilterMode::Linear,

            anisotropy_clamp: 8,
            ..Default::default()
        });

        let splat_image = splat.unwrap_or(&NO_SPLAT);
        let splat_tex = device.create_texture(&TextureDescriptor {
            label: Some("terrain_splat"),
            size: Extent3d {
                width: splat_image.width,
                height: splat_image.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            TexelCopyTextureInfo {
                texture: &splat_tex,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            &splat_image.rgba,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * splat_image.width),
                rows_per_image: Some(splat_image.height),
            },
            Extent3d {
                width: splat_image.width,
                height: splat_image.height,
                depth_or_array_layers: 1,
            },
        );

        let occ_image = sky_occlusion.unwrap_or(&FULL_SKY);
        let occ_layers = (occ_image.rgba.len() / (4 * occ_image.width as usize * occ_image.height as usize).max(1)).max(1) as u32;
        let sky_occ_tex = device.create_texture(&TextureDescriptor {
            label: Some("terrain_sky_occlusion"),
            size: Extent3d {
                width: occ_image.width,
                height: occ_image.height,
                depth_or_array_layers: occ_layers,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,

            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            TexelCopyTextureInfo {
                texture: &sky_occ_tex,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            &occ_image.rgba[..(4 * occ_image.width * occ_image.height * occ_layers) as usize],
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * occ_image.width),
                rows_per_image: Some(occ_image.height),
            },
            Extent3d {
                width: occ_image.width,
                height: occ_image.height,
                depth_or_array_layers: occ_layers,
            },
        );

        let mut settings = settings;
        settings.use_splat = if splat.is_some() { 1.0 } else { 0.0 };

        let uniform = device.create_buffer(&BufferDescriptor {
            label: Some("terrain_material_uniform"),
            size: std::mem::size_of::<TerrainMaterialUniform>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform, 0, bytemuck::bytes_of(&settings));

        let flat_deform;
        let deform_view = if let Some(v) = deform { v } else {
            flat_deform = Self::flat_deform_view(device, queue);
            &flat_deform
        };
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("terrain_material"),
            layout,
            entries: &[
                BindGroupEntry { binding: 9, resource: BindingResource::TextureView(deform_view) },
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(
                        &array.create_view(&TextureViewDescriptor {
                            dimension: Some(TextureViewDimension::D2Array),
                            ..Default::default()
                        }),
                    ),
                },
                BindGroupEntry { binding: 1, resource: BindingResource::Sampler(&sampler) },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(
                        &macro_tex.create_view(&TextureViewDescriptor::default()),
                    ),
                },
                BindGroupEntry { binding: 3, resource: uniform.as_entire_binding() },
                BindGroupEntry {
                    binding: 4,
                    resource: BindingResource::TextureView(
                        &splat_tex.create_view(&TextureViewDescriptor::default()),
                    ),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: BindingResource::TextureView(
                        &normal_array.create_view(&TextureViewDescriptor {
                            dimension: Some(TextureViewDimension::D2Array),
                            ..Default::default()
                        }),
                    ),
                },
                BindGroupEntry {
                    binding: 6,
                    resource: BindingResource::TextureView(&sky_occ_tex.create_view(&TextureViewDescriptor {
                        dimension: Some(TextureViewDimension::D2Array),
                        ..Default::default()
                    })),
                },
                BindGroupEntry {
                    binding: 7,
                    resource: BindingResource::TextureView(
                        &rough_array.create_view(&TextureViewDescriptor {
                            dimension: Some(TextureViewDimension::D2Array),
                            ..Default::default()
                        }),
                    ),
                },
                BindGroupEntry {
                    binding: 8,
                    resource: BindingResource::TextureView(
                        &ao_array.create_view(&TextureViewDescriptor {
                            dimension: Some(TextureViewDimension::D2Array),
                            ..Default::default()
                        }),
                    ),
                },
            ],
        });

        let splat_size = [splat_image.width, splat_image.height];
        Self { bind_group, uniform, splat_texture: splat_tex, splat_size }
    }

    pub fn update_splat(&self, queue: &Queue, image: &TerrainImage) -> bool {
        if [image.width, image.height] != self.splat_size {
            return false;
        }
        queue.write_texture(
            TexelCopyTextureInfo {
                texture: &self.splat_texture,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            &image.rgba,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * image.width),
                rows_per_image: Some(image.height),
            },
            Extent3d { width: image.width, height: image.height, depth_or_array_layers: 1 },
        );
        true
    }

    pub fn fallback(device: &Device, queue: &Queue, layout: &BindGroupLayout) -> Self {
        Self::fallback_with_splat(device, queue, layout, None, None)
    }

    pub fn fallback_with_splat(
        device: &Device,
        queue: &Queue,
        layout: &BindGroupLayout,
        splat: Option<&TerrainImage>,
        sky_occlusion: Option<&TerrainImage>,
    ) -> Self {
        Self::from_layers(
            device, queue, layout, &[None, None, None, None], &[None, None, None, None], splat,
            sky_occlusion,
        )
    }

    pub fn from_layers(
        device: &Device,
        queue: &Queue,
        layout: &BindGroupLayout,
        layers: &[Option<TerrainImage>],
        normals: &[Option<TerrainImage>],
        splat: Option<&TerrainImage>,
        sky_occlusion: Option<&TerrainImage>,
    ) -> Self {
        Self::from_layers_with(
            device,
            queue,
            layout,
            layers,
            normals,

            &[],
            &[],
            splat,
            sky_occlusion,
            TerrainMaterialUniform::default(),
        )
    }

    pub fn from_layers_with(
        device: &Device,
        queue: &Queue,
        layout: &BindGroupLayout,
        layers: &[Option<TerrainImage>],
        normals: &[Option<TerrainImage>],
        rough: &[Option<TerrainImage>],
        ao: &[Option<TerrainImage>],
        splat: Option<&TerrainImage>,
        sky_occlusion: Option<&TerrainImage>,
        settings: TerrainMaterialUniform,
    ) -> Self {
        Self::from_layers_deformed(device, queue, layout, layers, normals, rough, ao, splat, sky_occlusion, settings, None)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_layers_deformed(
        device: &Device,
        queue: &Queue,
        layout: &BindGroupLayout,
        layers: &[Option<TerrainImage>],
        normals: &[Option<TerrainImage>],
        rough: &[Option<TerrainImage>],
        ao: &[Option<TerrainImage>],
        splat: Option<&TerrainImage>,
        sky_occlusion: Option<&TerrainImage>,
        settings: TerrainMaterialUniform,
        deform: Option<&TextureView>,
    ) -> Self {
        let (w, h) = layers
            .iter()
            .flatten()
            .map(|l| (l.width, l.height))
            .next()
            .unwrap_or((1, 1));

        let filled: Vec<TerrainImage> = FALLBACK_LAYER_COLOURS
            .iter()
            .enumerate()
            .map(|(i, rgb)| match layers.get(i).and_then(|l| l.as_ref()) {

                Some(img) if img.width == w && img.height == h => TerrainImage {
                    width: img.width,
                    height: img.height,
                    rgba: img.rgba.clone(),
                },
                Some(img) => resample(img, w, h),
                None => solid_image(*rgb, w, h),
            })
            .collect();

        Self::new_deformed(
            device,
            queue,
            layout,
            &filled,
            &solid_image([128, 128, 128], 1, 1),
            splat,
            sky_occlusion,
            normals,
            rough,
            ao,
            settings,
            deform,
        )
    }

}

#[derive(Clone)]
pub struct TerrainImage {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

impl TerrainImage {

    pub fn with_stationary_masks(&self, masks: &[TerrainImage]) -> Option<TerrainImage> {
        if masks.iter().any(|m| m.width != self.width || m.height != self.height || m.rgba.len() != self.rgba.len()) {
            return None;
        }
        let mut rgba = self.rgba.clone();
        for m in masks {
            rgba.extend_from_slice(&m.rgba);
        }
        Some(TerrainImage { width: self.width, height: self.height, rgba })
    }

    pub fn sun_baked_everywhere(&self) -> bool {
        !self.rgba.is_empty() && self.rgba.chunks_exact(4).all(|texel| texel[3] < 128)
    }
}

#[cfg(test)]
mod material_tests {
    use super::tests::*;

    #[test]
    fn a_stationary_lamp_on_the_ground_reads_its_mask() {
        let map = super::TerrainImage { width: 1, height: 1, rgba: vec![255, 128, 0, 255] };
        let mask = |d: u8| super::TerrainImage { width: 1, height: 1, rgba: vec![d, 0, 255, 255] };
        let layered = |d: u8| map.with_stationary_masks(&[mask(d)]).unwrap();

        let up = [0.0, 1.0, 0.0];
        let render = |sky: &super::TerrainImage, light: Option<Option<u8>>| {
            render_quad_lit_by(up, Palette::Test, None, Some(sky), &[None, None, None, None], light)
        };
        let Some(dark) = render(&map, None) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let lamp = render(&map, Some(None)).unwrap();
        let shadowed = render(&layered(0), Some(Some(0))).unwrap();
        let lit = render(&layered(255), Some(Some(0))).unwrap();
        assert!(lamp[0] > dark[0] + 10, "the test lamp does not light the ground: {lamp:?} vs {dark:?}");
        assert_eq!(shadowed, dark, "a lamp its mask hides still lit the ground");
        assert_eq!(lit, lamp, "a lamp its mask shows is not the unmasked lamp");

        let other_channel = render(&layered(0), Some(Some(1))).unwrap();
        assert_eq!(other_channel, lamp, "a lamp on another channel read this one's shadow");
    }
    use super::*;

    #[test]
    fn the_fallback_material_selects_layers_the_same_way() {
        let Some(flat) = render_quad_with_fallback_material([0.0, 1.0, 0.0]) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let cliff = render_quad_with_fallback_material([1.0, 0.05, 0.0]).unwrap();

        let greenness = |p: [u8; 4]| i32::from(p[1]) - i32::from(p[2]);
        assert!(
            greenness(flat) > greenness(cliff) + 4 && greenness(flat) > 4,
            "flat ground should read greener than a cliff: flat {flat:?} cliff {cliff:?}",
        );
        assert!(flat[3] == 255 && cliff[3] == 255, "terrain must be opaque");
    }

    #[test]
    fn a_short_layer_list_pads_to_a_full_array() {
        let Some((device, queue)) = headless_gpu() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let layout = material_bind_group_layout(&device);
        let one = TerrainImage { width: 1, height: 1, rgba: vec![10, 20, 30, 255] };
        let err_scope_1 = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let _m = TerrainMaterial::new(
            &device, &queue, &layout,
            std::slice::from_ref(&one),
            &one,
            None,
            None,
            &[None, None, None, None],
            &[],
            &[],
            TerrainMaterialUniform::default(),
        );
        assert!(
            pollster::block_on(err_scope_1.pop()).is_none(),
            "a single-layer terrain material must still bind validly",
        );
    }
}

static NO_SPLAT: std::sync::LazyLock<TerrainImage> = std::sync::LazyLock::new(|| TerrainImage {
    width: 1,
    height: 1,
    rgba: vec![255, 0, 0, 0],
});

static FULL_SKY: std::sync::LazyLock<TerrainImage> = std::sync::LazyLock::new(|| TerrainImage {
    width: 1,
    height: 1,
    rgba: vec![255, 255, 255, 255],
});

#[cfg(test)]
mod authored_splat_tests {
    use super::tests::*;
    use super::*;

    fn splat(rgba: [u8; 4]) -> TerrainImage {
        TerrainImage { width: 1, height: 1, rgba: rgba.to_vec() }
    }

    const FLAT: [f32; 3] = [0.0, 1.0, 0.0];
    const CLIFF: [f32; 3] = [1.0, 0.05, 0.0];

    #[test]
    fn authored_weights_override_the_slope_blend() {
        let Some(unauthored) = render_quad_with_normal(FLAT) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        assert!(unauthored[0] > unauthored[2], "flat ground should be red without a map: {unauthored:?}");

        let painted = render_quad_with_splat(FLAT, &splat([0, 255, 0, 0])).unwrap();
        assert!(
            painted[2] > painted[0],
            "a map demanding layer 1 must beat the slope rule on flat ground: {painted:?}",
        );
    }

    #[test]
    fn authored_weights_override_a_cliff_too() {
        let Some(unauthored) = render_quad_with_normal(CLIFF) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        assert!(unauthored[2] > unauthored[0], "a cliff should be blue without a map: {unauthored:?}");

        let painted = render_quad_with_splat(CLIFF, &splat([255, 0, 0, 0])).unwrap();
        assert!(
            painted[0] > painted[2],
            "a map demanding layer 0 must beat the slope rule on a cliff: {painted:?}",
        );
    }

    #[test]
    fn a_mixed_texel_blends_rather_than_picking_a_winner() {
        let Some(mixed) = render_quad_with_splat(FLAT, &splat([128, 0, 128, 0])) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        assert!(mixed[0] > 30, "layer 0 (red) should contribute: {mixed:?}");
        assert!(mixed[1] > 30, "layer 2 (green) should contribute: {mixed:?}");
        assert!(mixed[2] < mixed[0], "layer 1 (blue) was not painted: {mixed:?}");
    }

    #[test]
    fn unnormalised_weights_are_normalised_not_amplified() {

        let Some(normal_sum) = render_quad_with_splat(FLAT, &splat([128, 0, 128, 0])) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let double_sum = render_quad_with_splat(FLAT, &splat([255, 0, 255, 0])).unwrap();

        for channel in 0..3 {
            let delta = normal_sum[channel].abs_diff(double_sum[channel]);
            assert!(
                delta <= 4,
                "weights summing to 510 must shade like weights summing to 255 \
                 (channel {channel}: {normal_sum:?} vs {double_sum:?})",
            );
        }
    }

    #[test]
    fn an_empty_texel_does_not_produce_a_divide_by_zero() {
        let Some(px) = render_quad_with_splat(FLAT, &splat([0, 0, 0, 0])) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        assert_eq!(px[3], 255, "terrain must stay opaque: {px:?}");
        for channel in 0..3 {
            assert!(px[channel] < 250, "an empty texel must not blow out: {px:?}");
        }
    }
}

impl TerrainImage {

    pub fn load(path: &std::path::Path) -> Option<Self> {
        let decoded = match ::image::open(path) {
            Ok(d) => d.to_rgba8(),
            Err(e) => {
                log::warn!("terrain texture {}: {e}", path.display());
                return None;
            }
        };
        Some(Self {
            width: decoded.width(),
            height: decoded.height(),
            rgba: decoded.into_raw(),
        })
    }
}

pub const TERRAIN_LAYER_FILES: [&str; 4] = ["ground.jpg", "rock.jpg", "high.jpg", "sediment.jpg"];

pub const TERRAIN_NORMAL_FILES: [&str; 4] =
    ["ground_n.jpg", "rock_n.jpg", "high_n.jpg", "sediment_n.jpg"];

pub const TERRAIN_ROUGH_FILES: [&str; 4] =
    ["ground_r.jpg", "rock_r.jpg", "high_r.jpg", "sediment_r.jpg"];

pub const TERRAIN_AO_FILES: [&str; 4] =
    ["ground_ao.jpg", "rock_ao.jpg", "high_ao.jpg", "sediment_ao.jpg"];

pub fn load_terrain_layers(dir: &std::path::Path) -> Vec<Option<TerrainImage>> {
    TERRAIN_LAYER_FILES
        .iter()
        .map(|name| TerrainImage::load(&dir.join(name)))
        .collect()
}

pub fn load_terrain_rough(dir: &std::path::Path) -> Vec<Option<TerrainImage>> {
    TERRAIN_ROUGH_FILES
        .iter()
        .map(|name| TerrainImage::load(&dir.join(name)))
        .collect()
}

pub fn load_terrain_ao(dir: &std::path::Path) -> Vec<Option<TerrainImage>> {
    TERRAIN_AO_FILES
        .iter()
        .zip(TERRAIN_DISPLACEMENT_FILES.iter())
        .map(|(ao_name, d_name)| {
            let ao = TerrainImage::load(&dir.join(ao_name));
            let disp = TerrainImage::load(&dir.join(d_name));
            with_height_in_green(ao, disp)
        })
        .collect()
}

pub const TERRAIN_DISPLACEMENT_FILES: [&str; 4] =
    ["ground_d.jpg", "rock_d.jpg", "high_d.jpg", "sediment_d.jpg"];

pub fn with_height_in_green(ao: Option<TerrainImage>, disp: Option<TerrainImage>) -> Option<TerrainImage> {
    let mut out = match (ao, &disp) {
        (Some(ao), _) => ao,
        (None, Some(d)) => solid_image([255, 255, 255], d.width, d.height),
        (None, None) => return None,
    };
    let height = match disp {
        Some(d) if d.width == out.width && d.height == out.height => d,
        Some(d) => resample(&d, out.width, out.height),
        None => solid_image([255, 255, 255], out.width, out.height),
    };
    for (px, h) in out.rgba.chunks_exact_mut(4).zip(height.rgba.chunks_exact(4)) {
        px[1] = h[0];
    }
    Some(out)
}

pub fn load_terrain_settings(dir: &std::path::Path) -> TerrainMaterialUniform {
    let mut out = TerrainMaterialUniform::default();
    let path = dir.join(TERRAIN_SETTINGS_FILE);
    let text = match std::fs::read_to_string(&path) {
        Ok(t) => t,
        Err(_) => return out,
    };
    let raw: serde_json::Value = match serde_json::from_str(&text) {
        Ok(v) => v,
        Err(e) => {
            log::warn!("{}: {e} -- using default terrain settings", path.display());
            return out;
        }
    };
    if let Some(values) = raw.get("repeat").and_then(|r| r.as_array()) {
        for (i, slot) in out.repeat.iter_mut().enumerate() {
            if let Some(v) = values.get(i).and_then(serde_json::Value::as_f64) {

                *slot = (v as f32).max(MIN_REPEAT);
            }
        }
    }
    if let Some(v) = raw.get("normal_strength").and_then(serde_json::Value::as_f64) {
        out.normal_strength = (v as f32).clamp(0.0, 4.0);
    }
    if let Some(v) = raw.get("parallax").and_then(serde_json::Value::as_f64) {
        out.parallax = (v as f32).clamp(0.0, 0.5);
    }
    if let Some(v) = raw.get("mud_layer").and_then(serde_json::Value::as_i64) {
        out.mud_layer = if (0..4).contains(&v) { v as f32 } else { -1.0 };
    }
    if let Some(v) = raw.get("deform").and_then(serde_json::Value::as_f64) {
        out.deform_metres = (v as f32).clamp(0.0, 0.5);
    }
    if let Some(v) = raw.get("wet_dark").and_then(serde_json::Value::as_f64) {
        out.wet_dark = (v as f32).clamp(0.0, 1.0);
    }
    out
}

pub const TERRAIN_SETTINGS_FILE: &str = "settings.json";

pub const MIN_REPEAT: f32 = 0.05;

pub fn load_terrain_normals(dir: &std::path::Path) -> Vec<Option<TerrainImage>> {
    TERRAIN_NORMAL_FILES
        .iter()
        .map(|name| TerrainImage::load(&dir.join(name)))
        .collect()
}

pub const FALLBACK_LAYER_COLOURS: [[u8; 3]; 4] = [
    [86, 112, 62],
    [104, 100, 94],
    [132, 128, 120],
    [92, 84, 70],
];

pub(crate) fn solid_image(rgb: [u8; 3], width: u32, height: u32) -> TerrainImage {
    let mut rgba = Vec::with_capacity((width * height * 4) as usize);
    for _ in 0..(width * height) {
        rgba.extend_from_slice(&[rgb[0], rgb[1], rgb[2], 255]);
    }
    TerrainImage { width, height, rgba }
}

pub(crate) fn resample(src: &TerrainImage, width: u32, height: u32) -> TerrainImage {
    let mut rgba = Vec::with_capacity((width * height * 4) as usize);
    for y in 0..height {
        let sy = (u64::from(y) * u64::from(src.height) / u64::from(height.max(1))) as u32;
        for x in 0..width {
            let sx = (u64::from(x) * u64::from(src.width) / u64::from(width.max(1))) as u32;
            let at = ((sy.min(src.height - 1) * src.width + sx.min(src.width - 1)) * 4) as usize;
            rgba.extend_from_slice(&src.rgba[at..at + 4]);
        }
    }
    TerrainImage { width, height, rgba }
}

#[cfg(test)]
mod layer_loading_tests {
    use super::tests::headless_gpu;
    use super::*;

    fn img(rgb: [u8; 3], w: u32, h: u32) -> TerrainImage {
        solid_image(rgb, w, h)
    }

    #[test]
    fn a_missing_layer_falls_back_without_taking_the_others_with_it() {
        let Some((device, queue)) = headless_gpu() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let layout = material_bind_group_layout(&device);

        let err_scope_2 = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let _m = TerrainMaterial::from_layers(
            &device,
            &queue,
            &layout,
            &[Some(img([10, 20, 30], 64, 64)), None, Some(img([1, 2, 3], 64, 64)), None],
            &[None, None, None, None],
            None,
            None,
        );
        assert!(
            pollster::block_on(err_scope_2.pop()).is_none(),
            "a partial layer set must still bind validly",
        );
    }

    #[test]
    fn mixed_sizes_are_normalised_rather_than_failing_validation() {
        let Some((device, queue)) = headless_gpu() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let layout = material_bind_group_layout(&device);

        let err_scope_3 = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let _m = TerrainMaterial::from_layers(
            &device,
            &queue,
            &layout,
            &[
                Some(img([10, 20, 30], 64, 64)),
                Some(img([40, 50, 60], 16, 16)),
                None,
                Some(img([70, 80, 90], 128, 32)),
            ],
            &[None, None, None, None],
            None,
            None,
        );
        assert!(
            pollster::block_on(err_scope_3.pop()).is_none(),
            "mixed layer sizes must be normalised, not rejected",
        );
    }

    #[test]
    fn no_layers_at_all_is_the_old_flat_palette() {
        let Some((device, queue)) = headless_gpu() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let layout = material_bind_group_layout(&device);
        let err_scope_4 = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let _m = TerrainMaterial::from_layers(
            &device, &queue, &layout, &[None, None, None, None], &[None, None, None, None], None, None,
        );
        assert!(pollster::block_on(err_scope_4.pop()).is_none());
    }

    #[test]
    fn a_missing_file_reports_none_rather_than_failing() {
        assert!(TerrainImage::load(std::path::Path::new("/nonexistent/ground.jpg")).is_none());
    }

    #[test]
    fn the_layer_file_names_match_the_shader_slots() {
        assert_eq!(TERRAIN_LAYER_FILES.len(), FALLBACK_LAYER_COLOURS.len());
        assert_eq!(TERRAIN_LAYER_FILES[0], "ground.jpg");
        assert_eq!(TERRAIN_LAYER_FILES[1], "rock.jpg");
    }

    fn settings_dir(label: &str, body: Option<&str>) -> std::path::PathBuf {
        let dir = std::env::temp_dir()
            .join(format!("terrain_settings_{label}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        if let Some(text) = body {
            std::fs::write(dir.join(TERRAIN_SETTINGS_FILE), text).unwrap();
        }
        dir
    }

    #[test]
    fn absent_settings_are_the_built_in_defaults() {
        let dir = settings_dir("absent", None);
        let loaded = load_terrain_settings(&dir);
        assert_eq!(loaded.repeat, TerrainMaterialUniform::default().repeat);
        assert_eq!(loaded.normal_strength, 1.0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn repeat_comes_from_the_file() {
        let dir = settings_dir("repeat", Some(r#"{"repeat": [2.0, 3.5, 12.0, 6.0]}"#));
        assert_eq!(load_terrain_settings(&dir).repeat, [2.0, 3.5, 12.0, 6.0]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_unmentioned_key_keeps_its_default() {
        let dir = settings_dir("partial", Some(r#"{"repeat": [1.0, 1.0, 1.0, 1.0]}"#));
        let loaded = load_terrain_settings(&dir);
        assert_eq!(loaded.repeat, [1.0; 4]);
        assert_eq!(loaded.normal_strength, 1.0);
        assert_eq!(loaded.slope_start_deg, TerrainMaterialUniform::default().slope_start_deg);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_short_repeat_array_only_sets_what_it_names() {
        let dir = settings_dir("short", Some(r#"{"repeat": [2.0]}"#));
        let loaded = load_terrain_settings(&dir);
        let default = TerrainMaterialUniform::default().repeat;
        assert_eq!(loaded.repeat, [2.0, default[1], default[2], default[3]]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_zero_tile_size_is_clamped_not_taken() {
        let dir = settings_dir("zero", Some(r#"{"repeat": [0.0, -4.0, 10.0, 6.0]}"#));
        let loaded = load_terrain_settings(&dir);
        assert_eq!(loaded.repeat[0], MIN_REPEAT);
        assert_eq!(loaded.repeat[1], MIN_REPEAT);
        assert_eq!(loaded.repeat[2], 10.0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn malformed_settings_fall_back_rather_than_failing() {
        let dir = settings_dir("broken", Some("{ this is not json"));
        assert_eq!(
            load_terrain_settings(&dir).repeat,
            TerrainMaterialUniform::default().repeat,
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn resampling_preserves_the_requested_size() {
        let out = resample(&img([9, 9, 9], 7, 3), 16, 16);
        assert_eq!((out.width, out.height), (16, 16));
        assert_eq!(out.rgba.len(), 16 * 16 * 4);
    }
}

pub const FLAT_NORMAL: [u8; 3] = [128, 128, 255];

impl TerrainImage {
    fn clone_image(&self) -> TerrainImage {
        TerrainImage { width: self.width, height: self.height, rgba: self.rgba.clone() }
    }
}

#[cfg(test)]
mod normal_map_tests {
    use super::tests::{render_quad_full, Palette};
    use super::*;

    const FLAT: [f32; 3] = [0.0, 1.0, 0.0];

    fn normal_map(rgb: [u8; 3]) -> TerrainImage {
        solid_image(rgb, 1, 1)
    }

    fn only_layer0(map: TerrainImage) -> [Option<TerrainImage>; 4] {
        [Some(map), None, None, None]
    }

    fn brightness(px: [u8; 4]) -> u32 {
        u32::from(px[0]) + u32::from(px[1]) + u32::from(px[2])
    }

    #[test]
    fn a_tilted_normal_changes_the_lighting() {
        let Some(flat) = render_quad_full(FLAT, Palette::Test, None, None, &only_layer0(normal_map(FLAT_NORMAL)), true)
        else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };

        let toward = render_quad_full(FLAT, Palette::Test, None, None, &only_layer0(normal_map([230, 128, 160])), true).unwrap();

        let away = render_quad_full(FLAT, Palette::Test, None, None, &only_layer0(normal_map([25, 128, 160])), true).unwrap();

        assert_ne!(brightness(toward), brightness(flat), "a tilted normal must change shading: {toward:?} vs {flat:?}");
        assert!(
            brightness(toward) > brightness(away),
            "tilting toward the light must be brighter than tilting away: {toward:?} vs {away:?}",
        );
    }

    #[test]
    fn green_tilts_the_ground_toward_the_top_of_its_picture() {
        let north = glam::Vec3::new(0.4, 2.0, -3.0);
        let at = |green: u8| {
            super::tests::render_quad_light_at(
                FLAT, Palette::Test, None, None, &only_layer0(normal_map([128, green, 180])), Some(None), north,
            )
        };
        let Some(faces_top) = at(230) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let faces_bottom = at(26).unwrap();
        assert!(
            brightness(faces_top) > brightness(faces_bottom) + 6,
            "a texel facing the top of the picture (-z) must be lit more by a lamp toward -z: \
             {faces_top:?} vs {faces_bottom:?}",
        );
    }

    #[test]
    fn a_normal_maps_bumps_do_not_turn_with_the_player() {
        let lamp = glam::Vec3::new(0.4, 2.0, -3.0);
        let at = |rgb: [u8; 3], yaw: f32| {
            super::tests::render_quad_turned(
                FLAT, Palette::Test, None, None, &only_layer0(normal_map(rgb)), Some(None), lamp, yaw,
            )
        };
        let tilted = [210, 60, 170];
        let Some(still) = at(tilted, 0.0) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };

        let opposite = at([46, 196, 170], 0.0).unwrap();
        assert!(
            brightness(still).abs_diff(brightness(opposite)) > 6,
            "the test's lamp must tell the tilts apart: {still:?} vs {opposite:?}",
        );
        use std::f32::consts::{FRAC_PI_2, FRAC_PI_4, PI};
        for yaw in [FRAC_PI_2, PI, -FRAC_PI_4, 1.0] {
            let turned = at(tilted, yaw).unwrap();
            let off = still.iter().zip(&turned).map(|(a, b)| a.abs_diff(*b)).max().unwrap();
            assert!(off <= 1, "turned {yaw} radians: {turned:?}, unturned {still:?}");
        }
    }

    #[test]
    fn steep_ground_is_projected_along_the_worlds_axes_at_any_turn() {
        let lamp = glam::Vec3::new(0.4, 2.0, -3.0);
        let tilt = 20f32.to_radians();
        let slope = [tilt.sin(), tilt.cos(), 0.0];
        let at = |yaw: f32| {
            super::tests::render_quad_turned(
                slope, Palette::Ramp, None, None, &[None, None, None, None], Some(None), lamp, yaw,
            )
        };
        let Some(still) = at(0.0) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        assert!(still[0] > 10, "the ramp's red shows: {still:?}");
        use std::f32::consts::{FRAC_PI_2, FRAC_PI_4, PI};
        for yaw in [FRAC_PI_2, PI, -FRAC_PI_4, 1.0] {
            let turned = at(yaw).unwrap();
            let off = still.iter().zip(&turned).map(|(a, b)| a.abs_diff(*b)).max().unwrap();
            assert!(off <= 1, "turned {yaw} radians: {turned:?}, unturned {still:?}");
        }
    }

    #[test]
    fn a_flat_normal_map_is_a_true_no_op() {
        let Some(without) = render_quad_full(FLAT, Palette::Test, None, None, &[None, None, None, None], true)
        else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let with_flat = render_quad_full(FLAT, Palette::Test, None, None, &only_layer0(normal_map(FLAT_NORMAL)), true).unwrap();
        assert_eq!(with_flat, without, "a flat normal map must shade identically to none");
    }

    #[test]
    fn zero_strength_disables_the_maps() {
        let Some((device, queue)) = super::tests::headless_gpu() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let layout = material_bind_group_layout(&device);
        let mut settings = TerrainMaterialUniform::default();
        settings.normal_strength = 0.0;

        let err_scope_5 = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let _m = TerrainMaterial::new(
            &device, &queue, &layout,
            &[solid_image([200, 200, 200], 4, 4)],
            &solid_image([128, 128, 128], 1, 1),
            None,
            None,
            &only_layer0(solid_image([230, 128, 160], 4, 4)),
            &[],
            &[],
            settings,
        );
        assert!(pollster::block_on(err_scope_5.pop()).is_none());
    }

    #[test]
    fn a_normal_map_of_a_different_size_is_resampled() {
        let Some((device, queue)) = super::tests::headless_gpu() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let layout = material_bind_group_layout(&device);
        let err_scope_6 = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let _m = TerrainMaterial::from_layers(
            &device, &queue, &layout,
            &[Some(solid_image([10, 20, 30], 64, 64)), None, None, None],
            &[Some(solid_image(FLAT_NORMAL, 16, 16)), None, None, None],
            None,
            None,
        );
        assert!(
            pollster::block_on(err_scope_6.pop()).is_none(),
            "a mismatched normal map must be resampled, not rejected",
        );
    }

    #[test]
    fn the_flat_normal_texel_unpacks_to_straight_out() {

        let unpack = |v: u8| (f32::from(v) / 255.0) * 2.0 - 1.0;
        assert!(unpack(FLAT_NORMAL[0]).abs() < 0.01);
        assert!(unpack(FLAT_NORMAL[1]).abs() < 0.01);
        assert!((unpack(FLAT_NORMAL[2]) - 1.0).abs() < 0.01);
    }
}

#[cfg(test)]
pub fn terrain_shader_src() -> String {
    terrain_shader()
}

#[cfg(test)]
mod terrain_material_parity_tests {

    #[test]
    fn terrain_uses_the_material_path_with_albedo_passed_in() {
        let src = super::terrain_shader_src();
        assert!(
            src.contains("let lit = shade_material_env("),
            "terrain no longer shades through the material path",
        );
        assert!(
            !src.contains("shade_with_sky(in.world_pos"),
            "terrain went back to the sky-only path",
        );
        assert!(
            src.contains("tonemap(in.col.rgb * lit)"),
            "the albedo is being multiplied over the lit result again; it must \
             go INTO the shading so highlights are not tinted by it",
        );
        assert!(
            !src.contains("tonemap(albedo * in.col.rgb * lit)"),
            "the old multiply-over form is back",
        );
    }

    #[test]
    fn the_normal_variance_lands_in_roughness() {
        let src = super::terrain_shader_src();

        assert!(
            src.contains("layer_rough_at(layer, f)"),
            "terrain no longer samples a roughness map",
        );
        assert!(
            src.contains("layer_ao_at(layer, f)"),
            "terrain no longer samples an occlusion map",
        );
        assert!(
            !src.contains("min(2.0 * sigma2"),
            "the runtime variance term is back alongside the baked mips; that \
             counts the same measurement twice",
        );

        assert!(
            src.contains("specular_aa_roughness(clamp(rough_map, 0.0, 1.0), dpdx(shaded_n), dpdy(shaded_n))"),
            "the geometric specular-AA term is gone; the bake alone cannot see \
             curvature or the anisotropy gap",
        );
    }
}

#[cfg(test)]
mod terrain_mip_tests {

    #[test]
    fn the_normal_array_is_created_with_a_mip_chain() {
        let src = include_str!("terrain_pipeline.rs");
        let i = src.find(r#"label: Some("terrain_normals")"#).expect("normal array is gone");
        let window = &src[i..i + 400];
        assert!(
            window.contains("mip_level_count,"),
            "terrain_normals is back to a single mip level; distant ground will \
             point-sample full-resolution normals and shimmer",
        );
        assert!(
            !window.contains("mip_level_count: 1"),
            "terrain_normals explicitly requests one mip level",
        );
    }

    #[test]
    fn the_data_arrays_are_mipped_and_linear() {
        let src = include_str!("terrain_pipeline.rs");
        let i = src.find("let data_array = |label: &str|").expect("data arrays are gone");
        let window = &src[i..i + 600];
        assert!(
            window.contains("mip_level_count,") && !window.contains("mip_level_count: 1"),
            "the roughness/occlusion arrays lost their mip chain",
        );
        assert!(
            window.contains("TextureFormat::Rgba8Unorm") && !window.contains("Rgba8UnormSrgb"),
            "a measurement array is being created as sRGB",
        );

        assert!(
            src.contains("roughness_chain_with_normal_variance(")
                && src.contains("mip_chain_linear(&ao_arr[slot].rgba"),
            "roughness is no longer baked with the normal's variance, or AO is \
             no longer a plain linear chain",
        );
    }

    #[test]
    fn every_layer_has_a_roughness_and_occlusion_file_name() {
        assert_eq!(super::TERRAIN_ROUGH_FILES.len(), super::TERRAIN_LAYER_FILES.len());
        assert_eq!(super::TERRAIN_AO_FILES.len(), super::TERRAIN_LAYER_FILES.len());
        for (i, colour) in super::TERRAIN_LAYER_FILES.iter().enumerate() {
            let stem = colour.trim_end_matches(".jpg");
            assert_eq!(super::TERRAIN_ROUGH_FILES[i], format!("{stem}_r.jpg"));
            assert_eq!(super::TERRAIN_AO_FILES[i], format!("{stem}_ao.jpg"));
        }
    }

    #[test]
    fn the_normal_chain_is_built_linear() {
        let src = include_str!("terrain_pipeline.rs");
        let i = src.find(r#"label: Some("terrain_normals")"#).expect("normal array is gone");
        let window = &src[i..i + 1600];
        assert!(
            window.contains("mip_chain_linear(&src.rgba, w, h, mip_level_count)"),
            "the normal mip chain is not the linear one",
        );
    }

    #[test]
    fn the_linear_chain_preserves_a_flat_value() {
        let px = vec![128u8; 8 * 8 * 4];
        let chain = super::mip_chain_linear(&px, 8, 8, super::mip_levels_for(8, 8));
        assert!(chain.len() > 1, "no chain was built");
        for (data, w, h) in &chain {
            assert_eq!(data.len(), (w * h * 4) as usize);
            for b in data {
                assert!(
                    (i32::from(*b) - 128).abs() <= 1,
                    "a flat 128 became {b}; the linear chain is bending values",
                );
            }
        }
    }
}
