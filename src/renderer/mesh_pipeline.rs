use super::lights::wgsl_lights_block;
use super::mesh::{MeshVertex, SkinnedMeshVertex, MAX_SKIN_JOINTS};
use super::pipeline::lightmap_bind_group_layout;
use wgpu::*;

pub struct MeshPipeline {
    pub pipeline: RenderPipeline,
    pub texture_layout: BindGroupLayout,
    pub model_layout: BindGroupLayout,
    pub lightmap_layout: BindGroupLayout,
}

impl MeshPipeline {
    pub fn new(device: &Device, format: TextureFormat, camera_layout: &BindGroupLayout) -> Self {
        Self::new_with_front_face(device, format, camera_layout, FrontFace::Ccw, 1)
    }

    /// See `pipeline::SolidPipeline::new_multisampled`.
    pub fn new_multisampled(
        device: &Device,
        format: TextureFormat,
        camera_layout: &BindGroupLayout,
        samples: u32,
    ) -> Self {
        Self::new_with_front_face(device, format, camera_layout, FrontFace::Ccw, samples)
    }

    pub fn new_mirror(device: &Device, format: TextureFormat, camera_layout: &BindGroupLayout) -> Self {
        Self::new_with_front_face(device, format, camera_layout, FrontFace::Cw, 1)
    }

    fn new_with_front_face(
        device: &Device,
        format: TextureFormat,
        camera_layout: &BindGroupLayout,
        front_face: FrontFace,
        samples: u32,
    ) -> Self {
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("mesh_shader"),
            source: ShaderSource::Wgsl(mesh_shader().into()),
        });

        let texture_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("mesh_texture_bgl"),
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

        let lightmap_layout = lightmap_bind_group_layout(device);

        let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("mesh_layout"),
            bind_group_layouts: &[camera_layout, &model_layout, &texture_layout, &lightmap_layout],
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
            multisample: MultisampleState { count: samples, ..Default::default() },
            multiview: None,
            cache: None,
        });

        Self {
            pipeline,
            texture_layout,
            model_layout,
            lightmap_layout,
        }
    }

    pub fn create_model_uniform(&self, device: &Device) -> ModelUniform {
        let buffer = device.create_buffer(&BufferDescriptor {
            label: Some("mesh_model_uniform"),
            size: 64,
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
    pub buffer: Buffer,
    pub bind_group: BindGroup,
}

impl ModelUniform {
    pub fn upload(&self, queue: &Queue, model: glam::Mat4) {
        queue.write_buffer(
            &self.buffer,
            0,
            bytemuck::cast_slice(&model.to_cols_array()),
        );
    }
}

pub struct SkinnedMeshPipeline {
    pub pipeline: RenderPipeline,
    pub texture_layout: BindGroupLayout,
    pub model_layout: BindGroupLayout,
    pub skin_joint_layout: BindGroupLayout,
}

impl SkinnedMeshPipeline {
    pub fn new(device: &Device, format: TextureFormat, camera_layout: &BindGroupLayout) -> Self {
        Self::new_multisampled(device, format, camera_layout, 1)
    }

    /// See `pipeline::SolidPipeline::new_multisampled`.
    pub fn new_multisampled(
        device: &Device,
        format: TextureFormat,
        camera_layout: &BindGroupLayout,
        samples: u32,
    ) -> Self {
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("skinned_mesh_shader"),
            source: ShaderSource::Wgsl(skinned_mesh_shader().into()),
        });

        let texture_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("skinned_mesh_texture_bgl"),
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
        });

        let model_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("skinned_mesh_model_bgl"),
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

        let skin_joint_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("skin_joint_bgl"),
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
            label: Some("skinned_mesh_layout"),
            bind_group_layouts: &[
                camera_layout,
                &model_layout,
                &texture_layout,
                &skin_joint_layout,
            ],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("skinned_mesh_pipeline"),
            layout: Some(&layout),
            vertex: VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: PipelineCompilationOptions::default(),
                buffers: &[SkinnedMeshVertex::layout()],
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
                cull_mode: None,
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
            multisample: MultisampleState { count: samples, ..Default::default() },
            multiview: None,
            cache: None,
        });

        Self {
            pipeline,
            texture_layout,
            model_layout,
            skin_joint_layout,
        }
    }

    pub fn create_model_uniform(&self, device: &Device) -> ModelUniform {
        let buffer = device.create_buffer(&BufferDescriptor {
            label: Some("skinned_mesh_model_uniform"),
            size: 64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("skinned_mesh_model_bg"),
            layout: &self.model_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        });
        ModelUniform { buffer, bind_group }
    }
}

fn skinned_mesh_shader() -> String {
    format!(
        r#"
// Group 0 -- the camera, the lights and both shadow maps -- is declared by
// `wgsl_lights_block` below, so there is one description of that layout rather
// than one per shader.

struct ModelUniform {{ model: mat4x4<f32> }}
@group(1) @binding(0) var<uniform> model_u: ModelUniform;

@group(2) @binding(0) var tex: texture_2d<f32>;
@group(2) @binding(1) var samp: sampler;

struct JointMatrices {{ mats: array<mat4x4<f32>, {max_skin_joints}> }}
@group(3) @binding(0) var<uniform> joints: JointMatrices;

{lights_block}

struct VIn {{
    @location(0) position:      vec3<f32>,
    @location(1) normal:        vec3<f32>,
    @location(2) uv:            vec2<f32>,
    @location(3) joint_ids:     vec4<u32>,
    @location(4) joint_weights: vec4<f32>,
}}

struct VOut {{
    @builtin(position) clip: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) uv:     vec2<f32>,
    @location(2) world_pos: vec3<f32>,
}}

@vertex
fn vs_main(v: VIn) -> VOut {{
    let p = vec4<f32>(v.position, 1.0);
    let n = vec4<f32>(v.normal, 0.0);

    let skinned_p =
        (joints.mats[v.joint_ids.x] * p) * v.joint_weights.x +
        (joints.mats[v.joint_ids.y] * p) * v.joint_weights.y +
        (joints.mats[v.joint_ids.z] * p) * v.joint_weights.z +
        (joints.mats[v.joint_ids.w] * p) * v.joint_weights.w;

    let skinned_n =
        (joints.mats[v.joint_ids.x] * n) * v.joint_weights.x +
        (joints.mats[v.joint_ids.y] * n) * v.joint_weights.y +
        (joints.mats[v.joint_ids.z] * n) * v.joint_weights.z +
        (joints.mats[v.joint_ids.w] * n) * v.joint_weights.w;

    let world_pos = model_u.model * skinned_p;
    var out: VOut;
    out.clip      = camera.view_proj * world_pos;
    out.normal    = (model_u.model * skinned_n).xyz;
    out.uv        = v.uv;
    out.world_pos = world_pos.xyz;
    return out;
}}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4<f32> {{
    let n = normalize(in.normal);
    let lit = shade(in.world_pos, n);
    let tex_color = textureSample(tex, samp, in.uv);
    return vec4<f32>(tex_color.rgb * lit, tex_color.a);
}}
"#,
        lights_block = wgsl_lights_block(0, 1),
        max_skin_joints = MAX_SKIN_JOINTS
    )
}

fn mesh_shader() -> String {
    format!(
        r#"
// Group 0 -- the camera, the lights and both shadow maps -- is declared by
// `wgsl_lights_block` below, so there is one description of that layout rather
// than one per shader.

struct ModelUniform {{ model: mat4x4<f32> }}
@group(1) @binding(0) var<uniform> model_u: ModelUniform;

@group(2) @binding(0) var tex: texture_2d<f32>;
@group(2) @binding(1) var samp: sampler;

@group(3) @binding(0) var lm_tex: texture_2d<f32>;
@group(3) @binding(1) var lm_samp: sampler;

{lights_block}

struct VIn {{
    @location(0) position: vec3<f32>,
    @location(1) normal:   vec3<f32>,
    @location(2) uv:       vec2<f32>,
    @location(3) uv2:      vec2<f32>,
}}

struct VOut {{
    @builtin(position) clip: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) uv:     vec2<f32>,
    @location(2) world_pos: vec3<f32>,
    @location(3) uv2:    vec2<f32>,
}}

@vertex
fn vs_main(v: VIn) -> VOut {{
    let world_pos = model_u.model * vec4<f32>(v.position, 1.0);
    var out: VOut;
    out.clip      = camera.view_proj * world_pos;
    out.normal    = (model_u.model * vec4<f32>(v.normal, 0.0)).xyz;
    out.uv        = v.uv;
    out.world_pos = world_pos.xyz;
    out.uv2       = v.uv2;
    return out;
}}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4<f32> {{
    let n = normalize(in.normal);
    let lit = shade(in.world_pos, n);
    let tex_color = textureSample(tex, samp, in.uv);
    let lightmap = textureSample(lm_tex, lm_samp, in.uv2).rgb;
    return vec4<f32>(tex_color.rgb * lit * lightmap, tex_color.a);
}}
"#,
        lights_block = wgsl_lights_block(0, 1)
    )
}

