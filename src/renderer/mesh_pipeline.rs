use wgpu::*;
use super::mesh::MeshVertex;

pub struct MeshPipeline {
    pub pipeline:      RenderPipeline,
    pub texture_layout: BindGroupLayout,
    pub model_layout:   BindGroupLayout,
}

impl MeshPipeline {
    pub fn new(device: &Device, format: TextureFormat, camera_layout: &BindGroupLayout) -> Self {
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label:  Some("mesh_shader"),
            source: ShaderSource::Wgsl(MESH_SHADER.into()),
        });

        let texture_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("mesh_texture_bgl"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample_type:    TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                        multisampled:   false,
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
        });

        let model_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("mesh_model_bgl"),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::VERTEX,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("mesh_layout"),
            bind_group_layouts: &[camera_layout, &model_layout, &texture_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("mesh_pipeline"),
            layout: Some(&layout),
            vertex: VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: PipelineCompilationOptions::default(),
                buffers: &[MeshVertex::layout()],
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
                depth_compare: CompareFunction::Less,
                stencil: StencilState::default(),
                bias: DepthBiasState::default(),
            }),
            multisample: MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        Self { pipeline, texture_layout, model_layout }
    }

    pub fn create_model_uniform(&self, device: &Device) -> ModelUniform {
        let buffer = device.create_buffer(&BufferDescriptor {
            label: Some("mesh_model_uniform"),
            size: 64, // mat4x4<f32>
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("mesh_model_bg"),
            layout: &self.model_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        });

        ModelUniform { buffer, bind_group }
    }
}

pub struct ModelUniform {
    pub buffer:     Buffer,
    pub bind_group: BindGroup,
}

impl ModelUniform {
    pub fn upload(&self, queue: &Queue, model: glam::Mat4) {
        queue.write_buffer(&self.buffer, 0, bytemuck::cast_slice(&model.to_cols_array()));
    }
}

const MESH_SHADER: &str = r#"
struct CameraUniform { view_proj: mat4x4<f32> }
@group(0) @binding(0) var<uniform> camera: CameraUniform;

struct ModelUniform { model: mat4x4<f32> }
@group(1) @binding(0) var<uniform> model_u: ModelUniform;

@group(2) @binding(0) var tex: texture_2d<f32>;
@group(2) @binding(1) var samp: sampler;

struct VIn {
    @location(0) position: vec3<f32>,
    @location(1) normal:   vec3<f32>,
    @location(2) uv:       vec2<f32>,
}

struct VOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) uv:     vec2<f32>,
}

@vertex
fn vs_main(v: VIn) -> VOut {
    let world_pos = model_u.model * vec4<f32>(v.position, 1.0);
    var out: VOut;
    out.clip   = camera.view_proj * world_pos;
    out.normal = (model_u.model * vec4<f32>(v.normal, 0.0)).xyz;
    out.uv     = v.uv;
    return out;
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4<f32> {
    let light_dir = normalize(vec3<f32>(0.5, 1.0, 0.8));
    let n = normalize(in.normal);
    let diffuse = max(dot(n, light_dir), 0.0) * 0.6 + 0.4;

    let tex_color = textureSample(tex, samp, in.uv);
    return vec4<f32>(tex_color.rgb * diffuse, tex_color.a);
}
"#;
