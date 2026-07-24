
use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use wgpu::*;

use super::Color3;

pub struct Particle {
    pub position: Vec3,
    pub size: f32,
    pub color: Color3,
}

pub struct Beam {
    pub start: Vec3,
    pub end: Vec3,
    pub width: f32,
    pub color: Color3,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct ParticleVertex {
    pub position: [f32; 3],
    pub color: [f32; 4],
    pub uv: [f32; 2],
}

impl ParticleVertex {
    pub const ATTRIBS: [VertexAttribute; 3] =
        vertex_attr_array![0 => Float32x3, 1 => Float32x4, 2 => Float32x2];

    pub fn layout() -> VertexBufferLayout<'static> {
        VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as BufferAddress,
            step_mode: VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

pub fn build_particle_quad_one(
    p: &Particle,
    cam_right: Vec3,
    cam_up: Vec3,
) -> (Vec<ParticleVertex>, Vec<u32>) {
    let half = p.size * 0.5;
    let color = p.color.to_linear();
    let corners = [
        (p.position - cam_right * half - cam_up * half, [-1.0, -1.0]),
        (p.position + cam_right * half - cam_up * half, [1.0, -1.0]),
        (p.position + cam_right * half + cam_up * half, [1.0, 1.0]),
        (p.position - cam_right * half + cam_up * half, [-1.0, 1.0]),
    ];
    let verts = corners
        .into_iter()
        .map(|(pos, uv)| ParticleVertex {
            position: pos.into(),
            color,
            uv,
        })
        .collect();
    (verts, vec![0, 1, 2, 0, 2, 3])
}

pub fn build_beam_quad_one(b: &Beam, view_dir: Vec3) -> (Vec<ParticleVertex>, Vec<u32>) {
    let axis = b.end - b.start;
    let width_axis = axis
        .cross(view_dir)
        .try_normalize()
        .unwrap_or(Vec3::X)
        * (b.width * 0.5);
    let color = b.color.to_linear();
    let corners = [
        (b.start - width_axis, [-1.0, 0.0]),
        (b.start + width_axis, [1.0, 0.0]),
        (b.end + width_axis, [1.0, 0.0]),
        (b.end - width_axis, [-1.0, 0.0]),
    ];
    let verts = corners
        .into_iter()
        .map(|(pos, uv)| ParticleVertex {
            position: pos.into(),
            color,
            uv,
        })
        .collect();
    (verts, vec![0, 1, 2, 0, 2, 3])
}

pub fn build_particle_mesh(
    particles: &[Particle],
    beams: &[Beam],
    cam_right: Vec3,
    cam_up: Vec3,
    view_dir: Vec3,
) -> (Vec<ParticleVertex>, Vec<u32>) {
    let mut verts: Vec<ParticleVertex> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    for p in particles {
        let (v, i) = build_particle_quad_one(p, cam_right, cam_up);
        let base = verts.len() as u32;
        verts.extend(v);
        indices.extend(i.into_iter().map(|x| x + base));
    }
    for b in beams {
        let (v, i) = build_beam_quad_one(b, view_dir);
        let base = verts.len() as u32;
        verts.extend(v);
        indices.extend(i.into_iter().map(|x| x + base));
    }

    (verts, indices)
}

pub struct ParticlePipeline {
    pub pipeline: RenderPipeline,
}

impl ParticlePipeline {
    pub fn new(device: &Device, format: TextureFormat, uniform_layout: &BindGroupLayout) -> Self {
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("particle_shader"),
            source: ShaderSource::Wgsl(particle_shader().into()),
        });
        let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("particle_layout"),
            bind_group_layouts: &[uniform_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("particle_pipeline"),
            layout: Some(&layout),
            vertex: VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: PipelineCompilationOptions::default(),
                buffers: &[ParticleVertex::layout()],
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
                depth_write_enabled: false,
                depth_compare: CompareFunction::Less,
                stencil: StencilState::default(),
                bias: DepthBiasState::default(),
            }),
            multisample: MultisampleState::default(),
            multiview: None,
            cache: None,
        });
        Self { pipeline }
    }
}

fn particle_shader() -> String {
    r#"
struct Uniforms { view_proj: mat4x4<f32> }
@group(0) @binding(0) var<uniform> u: Uniforms;

struct VIn  { @location(0) pos: vec3<f32>, @location(1) col: vec4<f32>, @location(2) uv: vec2<f32> }
struct VOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) col: vec4<f32>,
    @location(1) uv: vec2<f32>,
}

@vertex fn vs_main(v: VIn) -> VOut {
    var out: VOut;
    out.clip = u.view_proj * vec4<f32>(v.pos, 1.0);
    out.col = v.col;
    out.uv = v.uv;
    return out;
}

@fragment fn fs_main(in: VOut) -> @location(0) vec4<f32> {
    let fall = clamp(1.0 - length(in.uv), 0.0, 1.0);
    return vec4<f32>(in.col.rgb, in.col.a * fall * fall);
}
"#
    .to_string()
}

