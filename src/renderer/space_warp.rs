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
//! Not yet: objects that move by themselves -- props, avatars, hands. They get
//! the camera's motion only, so between rendered frames they hold still
//! where they were, and at half rate they would judder. Behind a lever, off
//! by default, until they carry their own previous transforms.
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

/// One eye's two cameras, as the shader reads them.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MotionCamera {
    pub curr: [[f32; 4]; 4],
    pub prev: [[f32; 4]; 4],
}

/// The shader: this frame's clip position against the previous frame's, as
/// NDC, y flipped to Vulkan's. See the module docs.
pub const SHADER: &str = r#"
struct MotionCamera {
    curr: mat4x4<f32>,
    prev: mat4x4<f32>,
}
@group(0) @binding(0) var<uniform> cam: MotionCamera;

struct VOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) curr: vec4<f32>,
    @location(1) prev: vec4<f32>,
}

@vertex fn vs_main(@location(0) pos: vec3<f32>) -> VOut {
    var out: VOut;
    let p = vec4<f32>(pos, 1.0);
    out.clip = cam.curr * p;
    out.curr = out.clip;
    out.prev = cam.prev * p;
    return out;
}

@fragment fn fs_main(in: VOut) -> @location(0) vec4<f32> {
    let d = in.curr.xyz / in.curr.w - in.prev.xyz / in.prev.w;
    return vec4<f32>(d.x, -d.y, d.z, 0.0);
}
"#;

/// The previous frame's clip transform for THIS frame's player-frame
/// vertices: back to the world through this frame's player transform, into
/// the previous frame's player frame, through the previous frame's camera.
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

/// The motion-vector pipelines: one per vertex stride the world is drawn
/// with (brushes, and the solid buffer the terrain shares), each reading only
/// the position at offset 0.
pub struct MotionPipelines {
    pub layout: wgpu::BindGroupLayout,
    pub brush: wgpu::RenderPipeline,
    pub solid: wgpu::RenderPipeline,
}

impl MotionPipelines {
    pub fn new(device: &wgpu::Device, depth_format: wgpu::TextureFormat) -> Self {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("space_warp_camera"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("space_warp"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("space_warp"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let make = |label: &str, stride: u64| {
            let attributes = [wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 0, shader_location: 0 }];
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[Some(wgpu::VertexBufferLayout {
                        array_stride: stride,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &attributes,
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
        let brush = make("space_warp_brush", std::mem::size_of::<crate::renderer::brush_pipeline::BrushVertex>() as u64);
        let solid = make("space_warp_solid", std::mem::size_of::<crate::renderer::cuboid::SolidVertex>() as u64);
        Self { layout, brush, solid }
    }
}

/// The world's geometry for one motion-vector pass: `(vertices, indices,
/// index count)` per stride.
pub struct MotionGeometry<'a> {
    pub brush: Option<(&'a wgpu::Buffer, &'a wgpu::Buffer, u32)>,
    pub solid: Option<(&'a wgpu::Buffer, &'a wgpu::Buffer, u32)>,
}

/// Draw one eye's motion vectors and depth into `motion` and `depth`.
pub fn record(
    encoder: &mut wgpu::CommandEncoder,
    pipelines: &MotionPipelines,
    camera: &wgpu::BindGroup,
    motion: &wgpu::TextureView,
    depth: &wgpu::TextureView,
    geometry: &MotionGeometry,
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
    pass.set_bind_group(0, camera, &[]);
    for (pipeline, geo) in [(&pipelines.brush, geometry.brush), (&pipelines.solid, geometry.solid)] {
        if let Some((vb, ib, count)) = geo {
            if count > 0 {
                pass.set_pipeline(pipeline);
                pass.set_vertex_buffer(0, vb.slice(..));
                pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..count, 0, 0..1);
            }
        }
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
        let err = pollster::block_on(scope.pop());
        assert!(err.is_none(), "{err:?}");
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
