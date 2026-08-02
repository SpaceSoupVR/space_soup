use super::cuboid::{SolidVertex, WireVertex};
use super::lights::wgsl_lights_block;
use super::ssr::wgsl_ssr_block;
use wgpu::*;

pub fn lightmap_bind_group_layout(device: &Device) -> BindGroupLayout {
    device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("lightmap_bgl"),
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

pub struct SolidPipeline {
    pub pipeline: RenderPipeline,
    pub lightmap_layout: BindGroupLayout,
}
pub struct WirePipeline {
    pub pipeline: RenderPipeline,
}

impl SolidPipeline {
    /// Single-view. Used by the offscreen passes (scope world view, mirror
    /// reflection) which render into single-layer targets.
    pub fn new(device: &Device, format: TextureFormat, uniform_layout: &BindGroupLayout) -> Self {
        Self::new_with_front_face(device, format, uniform_layout, FrontFace::Ccw, 1)
    }

    /// Single-pass stereo, for the eye pass. Only valid against a 2-layer
    /// attachment; pointing it at a single-layer target is a validation error.
    pub fn new_multiview(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
    ) -> Self {
        Self::new_with_front_face(device, format, uniform_layout, FrontFace::Ccw, 2)
    }

    pub fn new_mirror(device: &Device, format: TextureFormat, uniform_layout: &BindGroupLayout) -> Self {
        Self::new_with_front_face(device, format, uniform_layout, FrontFace::Cw, 1)
    }

    fn new_with_front_face(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        front_face: FrontFace,
        views: u32,
    ) -> Self {
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("solid_shader"),
            source: ShaderSource::Wgsl(solid_shader(views).into()),
        });
        let lightmap_layout = lightmap_bind_group_layout(device);
        let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("solid_layout"),
            bind_group_layouts: &[uniform_layout, &lightmap_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("solid_pipeline"),
            layout: Some(&layout),
            vertex: VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: PipelineCompilationOptions::default(),
                buffers: &[SolidVertex::layout()],
            },
            fragment: Some(FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: PipelineCompilationOptions::default(),
                targets: &[Some(ColorTargetState {
                    format,
                    blend: Some(BlendState::ALPHA_BLENDING),
                    write_mask: ColorWrites::ALL,
                })],
            }),
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                cull_mode: Some(Face::Back),
                front_face,
                polygon_mode: PolygonMode::Fill,
                ..Default::default()
            },
            depth_stencil: Some(DepthStencilState {
                format: TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: CompareFunction::Less,
                stencil: StencilState::default(),
                bias: DepthBiasState::default(),
            }),
            multisample: MultisampleState::default(),
            multiview: std::num::NonZeroU32::new(views).filter(|v| v.get() > 1),
            cache: None,
        });
        Self {
            pipeline,
            lightmap_layout,
        }
    }

    pub fn new_ssr(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        ssr_camera_layout: &BindGroupLayout,
        ssr_scene_layout: &BindGroupLayout,
    ) -> Self {
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("solid_ssr_shader"),
            source: ShaderSource::Wgsl(solid_ssr_shader().into()),
        });
        let lightmap_layout = lightmap_bind_group_layout(device);
        let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("solid_ssr_layout"),
            bind_group_layouts: &[uniform_layout, &lightmap_layout, ssr_camera_layout, ssr_scene_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("solid_ssr_pipeline"),
            layout: Some(&layout),
            vertex: VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: PipelineCompilationOptions::default(),
                buffers: &[SolidVertex::layout()],
            },
            fragment: Some(FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: PipelineCompilationOptions::default(),
                targets: &[Some(ColorTargetState {
                    format,
                    blend: Some(BlendState::ALPHA_BLENDING),
                    write_mask: ColorWrites::ALL,
                })],
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
                depth_write_enabled: true,
                depth_compare: CompareFunction::LessEqual,
                stencil: StencilState::default(),
                bias: DepthBiasState::default(),
            }),
            multisample: MultisampleState::default(),
            multiview: None,
            cache: None,
        });
        Self {
            pipeline,
            lightmap_layout,
        }
    }
}

impl WirePipeline {
    /// Single-view, for offscreen passes into single-layer targets.
    pub fn new(device: &Device, format: TextureFormat, uniform_layout: &BindGroupLayout) -> Self {
        Self::new_with_views(device, format, uniform_layout, 1)
    }

    /// Single-pass stereo, for the eye pass against a 2-layer attachment.
    pub fn new_multiview(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
    ) -> Self {
        Self::new_with_views(device, format, uniform_layout, 2)
    }

    fn new_with_views(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        views: u32,
    ) -> Self {
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("wire_shader"),
            source: ShaderSource::Wgsl(wire_shader(views).into()),
        });
        let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("wire_layout"),
            bind_group_layouts: &[uniform_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("wire_pipeline"),
            layout: Some(&layout),
            vertex: VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: PipelineCompilationOptions::default(),
                buffers: &[WireVertex::layout()],
            },
            fragment: Some(FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: PipelineCompilationOptions::default(),
                targets: &[Some(ColorTargetState {
                    format,
                    blend: Some(BlendState::ALPHA_BLENDING),
                    write_mask: ColorWrites::ALL,
                })],
            }),
            primitive: PrimitiveState {
                topology: PrimitiveTopology::LineList,
                cull_mode: None,
                polygon_mode: PolygonMode::Fill,
                ..Default::default()
            },
            depth_stencil: Some(DepthStencilState {
                format: TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: CompareFunction::LessEqual,
                stencil: StencilState::default(),
                bias: DepthBiasState::default(),
            }),
            multisample: MultisampleState::default(),
            multiview: std::num::NonZeroU32::new(views).filter(|v| v.get() > 1),
            cache: None,
        });
        Self { pipeline }
    }
}

fn solid_shader(views: u32) -> String {
    let camera_block = super::uniforms::camera_uniform_wgsl(0, 0);
    let view_param = super::uniforms::view_index_param(views);
    let view_proj = super::uniforms::view_proj_expr(views);
    format!(
        r#"
{camera_block}
@group(1) @binding(0) var lm_tex: texture_2d<f32>;
@group(1) @binding(1) var lm_samp: sampler;

{lights_block}

struct VIn  {{ @location(0) pos: vec3<f32>, @location(1) norm: vec3<f32>, @location(2) col: vec4<f32>, @location(3) uv2: vec2<f32> }}
struct VOut {{
    @builtin(position) clip: vec4<f32>,
    @location(0) col: vec4<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) world_pos: vec3<f32>,
    @location(3) uv2: vec2<f32>,
}}

@vertex fn vs_main(v: VIn{view_param}) -> VOut {{
    var out: VOut;
    out.clip      = {view_proj} * vec4<f32>(v.pos, 1.0);
    out.col       = v.col;
    out.normal    = v.norm;
    out.world_pos = v.pos;
    out.uv2       = v.uv2;
    return out;
}}

@fragment fn fs_main(in: VOut) -> @location(0) vec4<f32> {{
    let n = normalize(in.normal);
    let lit = shade(in.world_pos, n);
    let lightmap = textureSample(lm_tex, lm_samp, in.uv2).rgb;
    return vec4<f32>(in.col.rgb * lit * lightmap, in.col.a);
}}
"#,
        lights_block = wgsl_lights_block(0, 1)
    )
}

fn solid_ssr_shader() -> String {
    format!(
        r#"
struct Uniforms {{ view_proj: mat4x4<f32> }}
@group(0) @binding(0) var<uniform> u: Uniforms;

@group(1) @binding(0) var lm_tex: texture_2d<f32>;
@group(1) @binding(1) var lm_samp: sampler;

{lights_block}

{ssr_block}

struct VIn  {{
    @location(0) pos: vec3<f32>,
    @location(1) norm: vec3<f32>,
    @location(2) col: vec4<f32>,
    @location(3) uv2: vec2<f32>,
    @location(4) reflectivity: f32,
}}
struct VOut {{
    @builtin(position) clip: vec4<f32>,
    @location(0) col: vec4<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) world_pos: vec3<f32>,
    @location(3) uv2: vec2<f32>,
    @location(4) reflectivity: f32,
}}

@vertex fn vs_main(v: VIn) -> VOut {{
    var out: VOut;
    out.clip         = u.view_proj * vec4<f32>(v.pos, 1.0);
    out.col          = v.col;
    out.normal       = v.norm;
    out.world_pos    = v.pos;
    out.uv2          = v.uv2;
    out.reflectivity = v.reflectivity;
    return out;
}}

@fragment fn fs_main(in: VOut) -> @location(0) vec4<f32> {{
    let n = normalize(in.normal);
    let lit = shade(in.world_pos, n);
    let lightmap = textureSample(lm_tex, lm_samp, in.uv2).rgb;
    let base = in.col.rgb * lit * lightmap;
    let reflected = ssr_reflect(in.world_pos, n, base, in.reflectivity);
    return vec4<f32>(reflected, in.col.a);
}}
"#,
        lights_block = wgsl_lights_block(0, 1),
        ssr_block = wgsl_ssr_block(2, 3)
    )
}

fn wire_shader(views: u32) -> String {
    let camera_block = super::uniforms::camera_uniform_wgsl(0, 0);
    let view_param = super::uniforms::view_index_param(views);
    let view_proj = super::uniforms::view_proj_expr(views);
    format!(
        r#"
{camera_block}
struct VIn  {{ @location(0) pos: vec3<f32>, @location(1) col: vec4<f32> }}
struct VOut {{ @builtin(position) clip: vec4<f32>, @location(0) col: vec4<f32> }}

@vertex fn vs_main(v: VIn{view_param}) -> VOut {{
    var p = {view_proj} * vec4<f32>(v.pos, 1.0);
    p.z -= 0.0001 * p.w;
    return VOut(p, v.col);
}}
@fragment fn fs_main(in: VOut) -> @location(0) vec4<f32> {{ return in.col; }}
"#
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::renderer::lights::LightsUniform;
    use crate::renderer::ssr::SsrPipelines;
    use crate::renderer::uniforms::UniformBuffer;

    fn headless_gpu() -> Option<(Device, Queue)> {
        let instance = Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&RequestAdapterOptions {
            power_preference: PowerPreference::default(),
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok()?;
        pollster::block_on(adapter.request_device(&DeviceDescriptor {
            required_features: Features::empty(),
            required_limits: Limits::default(),
            ..Default::default()
        }))
        .ok()
    }

    #[test]
    fn solid_ssr_pipeline_builds_on_a_real_device() {
        let Some((device, _queue)) = headless_gpu() else {
            eprintln!("skipping: no GPU adapter available in this environment");
            return;
        };
        let format = TextureFormat::Rgba8UnormSrgb;

        let lights_uniform = LightsUniform::new(&device);
        let uniform_buf = UniformBuffer::new(&device, &lights_uniform);
        let ssr_pipelines = SsrPipelines::new(&device, format);

        let _solid_ssr = SolidPipeline::new_ssr(
            &device,
            format,
            &uniform_buf.layout,
            ssr_pipelines.camera_layout(),
            ssr_pipelines.scene_texture_layout(),
        );
    }
}

