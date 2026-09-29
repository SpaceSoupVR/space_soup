//! APPLICATION SPACEWARP (`XR_FB_space_warp`): the app renders every other
//! frame, and the compositor makes the ones between from each eye's motion
//! vectors and depth.
//!
//! Every frame the renderer draws the static world a second time, small --
//! at the motion-vector size the runtime recommends -- into two more
//! swapchains: how far each pixel moved on screen since the previous frame
//! (`CurrNDC - PrevNDC`, the spec's definition, as Meta's own sample computes
//! it: this frame's view-projection against the previous frame's), and its
//! depth. Both ride on each eye's projection view as
//! `XrCompositionLayerSpaceWarpInfoFB`.
//!
//! # What moves
//!
//! Vertices reach the GPU already in the player's frame, so a world point's
//! previous clip position is `prev_view_proj * prev_world_to_player *
//! player_to_world * v`: the head's motion AND locomotion are in the vectors,
//! and `appSpaceDeltaPose` stays the identity.
//!
//! Meshes carry their own motion too: a rigid one by its previous model
//! matrix, a skinned one -- the player's hands and body, other avatars -- by
//! its previous joint palette (`GltfSkin::prev_joint_buffer`), skinned twice
//! in the vertex shader. Not yet: layered meshes (caves) and effects.
//!
//! # Conventions
//!
//! The images are Vulkan images, row 0 at the top, so the vectors are in
//! Vulkan's normalised device coordinates: y DOWN, z 0..1 -- wgpu's clip space
//! has y up, so y is negated. Depth is the scene's: 0 at the near plane
//! (3 cm), 1 at the far (1 km).

use glam::Mat4;
#[cfg(target_os = "android")]
use openxr as xr;

/// Near and far planes of every eye projection (`Camera::xr_projection`).
pub const NEAR_Z: f32 = 0.03;
pub const FAR_Z: f32 = 1000.0;

/// The motion vectors' format: signed half floats, as the spec recommends.
pub const MOTION_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// One draw's two clip transforms, as the shader reads them: this frame's
/// and the previous frame's, each already carrying the draw's model matrix.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MotionCamera {
    pub curr: [[f32; 4]; 4],
    pub prev: [[f32; 4]; 4],
}

/// Bytes between two draws' slots in the ring of [`MotionCamera`]s: the
/// dynamic-offset alignment every device allows.
pub const SLOT_STRIDE: u64 = 256;

/// Slots in the ring, both eyes: the world and every mesh, per eye.
pub const MAX_SLOTS: u32 = 256;

/// The shader: this frame's clip position against the previous frame's, as
/// NDC, y flipped to Vulkan's. See the module docs. A skinned vertex is
/// skinned twice, by this frame's joints and by the previous frame's, the
/// same weighted sum the eye pass takes (`mesh_pipeline`).
pub fn shader() -> String {
    format!(
        r#"
struct MotionCamera {{
    curr: mat4x4<f32>,
    prev: mat4x4<f32>,
}}
@group(0) @binding(0) var<uniform> cam: MotionCamera;

struct Joints {{
    m: array<mat4x4<f32>, {joints}>,
}}
@group(1) @binding(0) var<uniform> joints: Joints;
@group(1) @binding(1) var<uniform> prev_joints: Joints;

struct VOut {{
    @builtin(position) clip: vec4<f32>,
    @location(0) curr: vec4<f32>,
    @location(1) prev: vec4<f32>,
}}

@vertex fn vs_main(@location(0) pos: vec3<f32>) -> VOut {{
    var out: VOut;
    let p = vec4<f32>(pos, 1.0);
    out.clip = cam.curr * p;
    out.curr = out.clip;
    out.prev = cam.prev * p;
    return out;
}}

@vertex fn vs_skinned(
    @location(0) pos: vec3<f32>,
    @location(3) ids: vec4<u32>,
    @location(4) w: vec4<f32>,
) -> VOut {{
    var out: VOut;
    let p = vec4<f32>(pos, 1.0);
    let now = (joints.m[ids.x] * p) * w.x + (joints.m[ids.y] * p) * w.y
        + (joints.m[ids.z] * p) * w.z + (joints.m[ids.w] * p) * w.w;
    let before = (prev_joints.m[ids.x] * p) * w.x + (prev_joints.m[ids.y] * p) * w.y
        + (prev_joints.m[ids.z] * p) * w.z + (prev_joints.m[ids.w] * p) * w.w;
    out.clip = cam.curr * now;
    out.curr = out.clip;
    out.prev = cam.prev * before;
    return out;
}}

@fragment fn fs_main(in: VOut) -> @location(0) vec4<f32> {{
    let d = in.curr.xyz / in.curr.w - in.prev.xyz / in.prev.w;
    return vec4<f32>(d.x, -d.y, d.z, 0.0);
}}
"#,
        joints = crate::renderer::mesh::MAX_SKIN_JOINTS,
    )
}

/// The previous frame's clip transform for THIS frame's player-frame
/// vertices: back to the world through this frame's player transform, into
/// the previous frame's player frame, through the previous frame's camera.
/// (A mesh needs none of this: its model matrix is already in each frame's
/// own player frame, so its previous clip is `prev_view_proj * prev_model`.)
pub fn previous_clip(prev_view_proj: Mat4, prev_world_to_player: Mat4, world_to_player: Mat4) -> Mat4 {
    prev_view_proj * prev_world_to_player * world_to_player.inverse()
}

/// The motion vector for a point at player-frame `p`, as the shader computes
/// it: the CPU reference the tests hold the shader's arithmetic to.
pub fn motion_vector(curr: Mat4, prev: Mat4, p: glam::Vec3) -> glam::Vec3 {
    let c = curr * p.extend(1.0);
    let q = prev * p.extend(1.0);
    let d = c.truncate() / c.w - q.truncate() / q.w;
    glam::Vec3::new(d.x, -d.y, d.z)
}

/// Which vertex layout a motion draw reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MotionKind {
    Brush,
    Solid,
    Mesh,
    Skinned,
}

/// The motion-vector pipelines, one per vertex layout the eye pass draws
/// with, each reading only what places a vertex; and the ring of per-draw
/// cameras they share.
pub struct MotionPipelines {
    pub camera_layout: wgpu::BindGroupLayout,
    pub joints_layout: wgpu::BindGroupLayout,
    brush: wgpu::RenderPipeline,
    solid: wgpu::RenderPipeline,
    mesh: wgpu::RenderPipeline,
    skinned: wgpu::RenderPipeline,
}

impl MotionPipelines {
    pub fn new(device: &wgpu::Device, depth_format: wgpu::TextureFormat) -> Self {
        let camera_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("space_warp_camera"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<MotionCamera>() as u64),
                },
                count: None,
            }],
        });
        let joint_entry = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let joints_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("space_warp_joints"),
            entries: &[joint_entry(0), joint_entry(1)],
        });
        let plain_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("space_warp"),
            bind_group_layouts: &[Some(&camera_layout)],
            immediate_size: 0,
        });
        let skinned_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("space_warp_skinned"),
            bind_group_layouts: &[Some(&camera_layout), Some(&joints_layout)],
            immediate_size: 0,
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("space_warp"),
            source: wgpu::ShaderSource::Wgsl(shader().into()),
        });
        let make = |label: &str, layout: &wgpu::PipelineLayout, entry: &str, stride: u64, attributes: &[wgpu::VertexAttribute]| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    buffers: &[Some(wgpu::VertexBufferLayout {
                        array_stride: stride,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes,
                    })],
                },
                primitive: wgpu::PrimitiveState {
                    // Both sides: the world's own winding differs by source,
                    // and a motion vector has no front.
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: depth_format,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::Less),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: Default::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: MOTION_FORMAT,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            })
        };
        let position = [wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 0, shader_location: 0 }];
        let size = |n: usize| n as u64;
        let brush = make("space_warp_brush", &plain_layout, "vs_main", size(std::mem::size_of::<crate::renderer::brush_pipeline::BrushVertex>()), &position);
        let solid = make("space_warp_solid", &plain_layout, "vs_main", size(std::mem::size_of::<crate::renderer::cuboid::SolidVertex>()), &position);
        let mesh = make("space_warp_mesh", &plain_layout, "vs_main", size(std::mem::size_of::<crate::renderer::mesh::MeshVertex>()), &position);
        let skinned_attributes = [
            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 0, shader_location: 0 },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Uint32x4,
                offset: std::mem::offset_of!(crate::renderer::mesh::SkinnedMeshVertex, joint_ids) as u64,
                shader_location: 3,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x4,
                offset: std::mem::offset_of!(crate::renderer::mesh::SkinnedMeshVertex, joint_weights) as u64,
                shader_location: 4,
            },
        ];
        let skinned = make(
            "space_warp_skinned",
            &skinned_layout,
            "vs_skinned",
            size(std::mem::size_of::<crate::renderer::mesh::SkinnedMeshVertex>()),
            &skinned_attributes,
        );
        Self { camera_layout, joints_layout, brush, solid, mesh, skinned }
    }

    fn pipeline(&self, kind: MotionKind) -> &wgpu::RenderPipeline {
        match kind {
            MotionKind::Brush => &self.brush,
            MotionKind::Solid => &self.solid,
            MotionKind::Mesh => &self.mesh,
            MotionKind::Skinned => &self.skinned,
        }
    }
}

/// One draw of the motion pass: its geometry, the ring slot of its two
/// cameras, and -- skinned -- its joints.
pub struct MotionDraw<'a> {
    pub kind: MotionKind,
    pub vertices: &'a wgpu::Buffer,
    pub indices: &'a wgpu::Buffer,
    pub count: u32,
    pub slot: u32,
    pub joints: Option<&'a wgpu::BindGroup>,
}

/// Draw one eye's motion vectors and depth into `motion` and `depth`.
pub fn record(
    encoder: &mut wgpu::CommandEncoder,
    pipelines: &MotionPipelines,
    cameras: &wgpu::BindGroup,
    motion: &wgpu::TextureView,
    depth: &wgpu::TextureView,
    draws: &[MotionDraw],
) {
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("space_warp"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: motion,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store },
        })],
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
            view: depth,
            depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
            stencil_ops: None,
        }),
        ..Default::default()
    });
    for d in draws.iter().filter(|d| d.count > 0 && d.slot < MAX_SLOTS) {
        if d.kind == MotionKind::Skinned && d.joints.is_none() {
            continue;
        }
        pass.set_pipeline(pipelines.pipeline(d.kind));
        pass.set_bind_group(0, cameras, &[(d.slot as u64 * SLOT_STRIDE) as u32]);
        if let Some(j) = d.joints {
            pass.set_bind_group(1, j, &[]);
        }
        pass.set_vertex_buffer(0, d.vertices.slice(..));
        pass.set_index_buffer(d.indices.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..d.count, 0, 0..1);
    }
}

/// The space warp info for one eye, as `XrCompositionLayerSpaceWarpInfoFB`.
#[cfg(target_os = "android")]
pub fn layer_info(
    motion: &xr::Swapchain<xr::Vulkan>,
    depth: &xr::Swapchain<xr::Vulkan>,
    eye: u32,
    size: (u32, u32),
) -> xr::sys::CompositionLayerSpaceWarpInfoFB {
    let rect = xr::Rect2Di {
        offset: xr::Offset2Di { x: 0, y: 0 },
        extent: xr::Extent2Di { width: size.0 as i32, height: size.1 as i32 },
    };
    xr::sys::CompositionLayerSpaceWarpInfoFB {
        ty: xr::sys::CompositionLayerSpaceWarpInfoFB::TYPE,
        next: std::ptr::null(),
        layer_flags: xr::sys::CompositionLayerSpaceWarpInfoFlagsFB::EMPTY,
        motion_vector_sub_image: xr::sys::SwapchainSubImage { swapchain: motion.as_raw(), image_rect: rect, image_array_index: eye },
        app_space_delta_pose: xr::Posef::IDENTITY,
        depth_sub_image: xr::sys::SwapchainSubImage { swapchain: depth.as_raw(), image_rect: rect, image_array_index: eye },
        min_depth: 0.0,
        max_depth: 1.0,
        near_z: NEAR_Z,
        far_z: FAR_Z,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Quat, Vec3};

    fn view_proj(eye: Vec3, yaw: f32) -> Mat4 {
        let proj = Mat4::perspective_rh(1.5, 1.0, NEAR_Z, FAR_Z);
        proj * Mat4::from_rotation_translation(Quat::from_rotation_y(yaw), eye).inverse()
    }

    /// The pipelines build: naga validates the shader only when a device
    /// creates them.
    #[test]
    fn the_pipelines_build() {
        let Some((device, _queue)) = crate::renderer::terrain_pipeline::tests::headless_gpu() else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let _p = MotionPipelines::new(&device, wgpu::TextureFormat::Depth32Float);
        assert!(SLOT_STRIDE >= device.limits().min_uniform_buffer_offset_alignment as u64);
        let err = pollster::block_on(scope.pop());
        assert!(err.is_none(), "{err:?}");
    }

    /// A hand moving right in front of a still head: its pixels move right,
    /// through its own model matrix -- a mesh's previous clip is the previous
    /// camera times its previous model, both in the previous player frame.
    #[test]
    fn a_moving_mesh_carries_its_own_motion() {
        let vp = view_proj(Vec3::new(0.0, 1.6, 0.0), 0.0);
        let before = Mat4::from_translation(Vec3::new(0.2, 1.2, -0.5));
        let now = Mat4::from_translation(Vec3::new(0.25, 1.2, -0.5));
        let local = Vec3::new(0.0, 0.0, 0.0);
        let mv = motion_vector(vp * now, vp * before, local);
        assert!(mv.x > 0.05 && mv.y.abs() < 1e-4, "{mv}");
    }

    /// A still head and a still player: nothing moves.
    #[test]
    fn nothing_moves_when_nothing_moves() {
        let vp = view_proj(Vec3::new(0.0, 1.6, 0.0), 0.0);
        let w2p = Mat4::from_translation(Vec3::new(-3.0, 0.0, 2.0));
        let prev = previous_clip(vp, w2p, w2p);
        let mv = motion_vector(vp, prev, Vec3::new(0.3, 1.2, -4.0));
        assert!(mv.length() < 1e-6, "{mv}");
    }

    /// The head turning left: the world slides right on screen, +x.
    #[test]
    fn turning_left_slides_the_world_right() {
        let prev_vp = view_proj(Vec3::new(0.0, 1.6, 0.0), 0.0);
        let vp = view_proj(Vec3::new(0.0, 1.6, 0.0), 0.05);
        let mv = motion_vector(vp, previous_clip(prev_vp, Mat4::IDENTITY, Mat4::IDENTITY), Vec3::new(0.0, 1.6, -5.0));
        assert!(mv.x > 0.01 && mv.y.abs() < 1e-4, "{mv}");
    }

    /// The head rising: the world drops on screen, which in Vulkan's
    /// coordinates -- y down -- is +y.
    #[test]
    fn rising_drops_the_world_down_the_image() {
        let prev_vp = view_proj(Vec3::new(0.0, 1.6, 0.0), 0.0);
        let vp = view_proj(Vec3::new(0.0, 1.7, 0.0), 0.0);
        let mv = motion_vector(vp, previous_clip(prev_vp, Mat4::IDENTITY, Mat4::IDENTITY), Vec3::new(0.0, 1.6, -3.0));
        assert!(mv.y > 0.01, "{mv}");
    }

    /// Walking forward with the thumbstick (locomotion, not the head): the
    /// vertices arrive in the NEW player frame, and the vector still says the
    /// world came toward the eye -- a point ahead and to the right moves
    /// further right.
    #[test]
    fn locomotion_is_in_the_vectors() {
        let vp = view_proj(Vec3::new(0.0, 1.6, 0.0), 0.0);
        let prev_w2p = Mat4::IDENTITY;
        let w2p = Mat4::from_translation(Vec3::new(0.0, 0.0, 0.5)); // the player moved 0.5 m forward (-z)
        let world = Vec3::new(1.0, 1.6, -4.0);
        let p = w2p.transform_point3(world);
        let mv = motion_vector(vp, previous_clip(vp, prev_w2p, w2p), p);
        assert!(mv.x > 0.005, "{mv}");
    }
}
