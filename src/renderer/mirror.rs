
use glam::{Quat, Vec3};
use wgpu::*;

use super::mesh_pipeline::ModelUniform;

mod math;
pub use math::{build_mirror_quad, oblique_near_clip, plane_to_eye_space, reflection_matrix, world_plane_equation};

#[derive(Debug, Clone, Copy)]
pub struct MirrorSurface {
    pub position: Vec3,
    pub rotation: Quat,
    pub half_size: Vec3,
}

impl MirrorSurface {
    pub fn normal(&self) -> Vec3 {
        self.rotation * Vec3::NEG_Z
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MirrorVertex {
    position: [f32; 3],
}

impl MirrorVertex {
    fn layout() -> VertexBufferLayout<'static> {
        VertexBufferLayout {
            array_stride: std::mem::size_of::<MirrorVertex>() as BufferAddress,
            step_mode: VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![0 => Float32x3],
        }
    }
}

pub struct MirrorTarget {
    _color_texture: Texture,
    pub color_view: TextureView,
    _depth_texture: Texture,
    pub depth_view: TextureView,
    pub texture_bind_group: BindGroup,
}

pub struct MirrorPipeline {
    pub pipeline: RenderPipeline,
    texture_layout: BindGroupLayout,
    model_layout: BindGroupLayout,
    reflected_vp_layout: BindGroupLayout,
    sampler: Sampler,
}

impl MirrorPipeline {
    pub fn new(device: &Device, format: TextureFormat, camera_layout: &BindGroupLayout) -> Self {
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("mirror_shader"),
            source: ShaderSource::Wgsl(mirror_shader().into()),
        });

        let texture_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("mirror_texture_bgl"),
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
            label: Some("mirror_model_bgl"),
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

        let reflected_vp_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("mirror_reflected_vp_bgl"),
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
            label: Some("mirror_layout"),
            bind_group_layouts: &[camera_layout, &model_layout, &texture_layout, &reflected_vp_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("mirror_pipeline"),
            layout: Some(&layout),
            vertex: VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: PipelineCompilationOptions::default(),
                buffers: &[MirrorVertex::layout()],
            },
            fragment: Some(FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: PipelineCompilationOptions::default(),
                targets: &[Some(ColorTargetState {
                    format,
                    blend: Some(BlendState::REPLACE),
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
            multisample: MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("mirror_sampler"),
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            ..Default::default()
        });

        Self {
            pipeline,
            texture_layout,
            model_layout,
            reflected_vp_layout,
            sampler,
        }
    }

    pub fn create_model_uniform(&self, device: &Device) -> ModelUniform {
        let buffer = device.create_buffer(&BufferDescriptor {
            label: Some("mirror_model_uniform"),
            size: 64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("mirror_model_bg"),
            layout: &self.model_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        });
        ModelUniform { buffer, bind_group }
    }

    pub fn create_reflected_vp_uniform(&self, device: &Device) -> ModelUniform {
        let buffer = device.create_buffer(&BufferDescriptor {
            label: Some("mirror_reflected_vp_uniform"),
            size: 64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("mirror_reflected_vp_bg"),
            layout: &self.reflected_vp_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        });
        ModelUniform { buffer, bind_group }
    }

    pub fn create_target(&self, device: &Device, format: TextureFormat, width: u32, height: u32) -> MirrorTarget {
        let color_texture = device.create_texture(&TextureDescriptor {
            label: Some("mirror_color"),
            size: Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let color_view = color_texture.create_view(&TextureViewDescriptor::default());

        let depth_texture = device.create_texture(&TextureDescriptor {
            label: Some("mirror_depth"),
            size: Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Depth32Float,
            usage: TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth_view = depth_texture.create_view(&TextureViewDescriptor::default());

        let texture_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("mirror_texture_bg"),
            layout: &self.texture_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&color_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Sampler(&self.sampler),
                },
            ],
        });

        MirrorTarget {
            _color_texture: color_texture,
            color_view,
            _depth_texture: depth_texture,
            depth_view,
            texture_bind_group,
        }
    }
}

fn mirror_shader() -> String {
    r#"
struct CameraUniform { view_proj: mat4x4<f32> }
@group(0) @binding(0) var<uniform> camera: CameraUniform;

struct ModelUniform { model: mat4x4<f32> }
@group(1) @binding(0) var<uniform> model_u: ModelUniform;

@group(2) @binding(0) var mirror_tex: texture_2d<f32>;
@group(2) @binding(1) var mirror_samp: sampler;

struct ReflectedVp { view_proj: mat4x4<f32> }
@group(3) @binding(0) var<uniform> reflected: ReflectedVp;

struct VIn { @location(0) position: vec3<f32> }
struct VOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) reflected_clip: vec4<f32>,
}

@vertex
fn vs_main(v: VIn) -> VOut {
    var out: VOut;
    let world_pos = model_u.model * vec4<f32>(v.position, 1.0);
    out.clip = camera.view_proj * world_pos;
    out.clip.z -= 0.0001 * out.clip.w;
    out.reflected_clip = reflected.view_proj * world_pos;
    return out;
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4<f32> {
    let ndc = in.reflected_clip.xy / in.reflected_clip.w;
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    return textureSample(mirror_tex, mirror_samp, uv);
}
"#
    .to_string()
}
