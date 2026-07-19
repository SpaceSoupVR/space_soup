//! Shared rendering for two otherwise-unrelated effects — particle
//! billboards and laser beams — that happen to want the exact same thing:
//! simple, unlit, alpha-blended, depth-write-disabled quads with a soft
//! radial falloff. One pipeline, one vertex type, one draw call per frame,
//! same "list of structs -> one flat buffer -> one draw call" shape
//! `cuboid.rs`'s `build_solid_mesh`/`build_wire_mesh` already use.

use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use wgpu::*;

use super::Color3;

/// One particle instance for this frame — a camera-facing quad centered at
/// `position`, `size` wide/tall, colored/faded by `color`.
pub struct Particle {
    pub position: Vec3,
    pub size: f32,
    pub color: Color3,
}

/// A laser beam for this frame, from `start` (the emitter) to `end` (wherever
/// the server's PhysX raycast said it terminates).
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
    /// Local quad-space coordinate, `(±1, ±1)` at the corners — the fragment
    /// shader turns `length(uv)` into a soft radial falloff. A beam quad
    /// only varies `uv.x` across its width and holds `uv.y` at 0, so the
    /// exact same falloff formula gives a soft-edged strip instead of a dot.
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

/// One particle's camera-facing quad — corners offset from `p.position`
/// along the camera's own right/up basis vectors, not the particle's own
/// rotation (there isn't one), so it always faces the viewer.
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

/// A beam's quad — width axis perpendicular to both the beam direction and
/// the view direction, so it presents its full width toward the camera
/// regardless of viewing angle. `uv.y` is 0 for every vertex (no along-beam
/// falloff — solid brightness down the whole length), only `uv.x` varies.
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

/// Concatenates every particle's and every beam's quad into one flat buffer,
/// rebasing indices — exactly `build_solid_mesh`'s shape, just merging two
/// source kinds instead of one.
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
