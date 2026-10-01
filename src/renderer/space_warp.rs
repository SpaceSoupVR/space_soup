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
//! # Reflections
//!
//! A reflection is not painted on the surface that shows it: in a mirror the
//! room behind you is an image as far behind the glass as the room is in
//! front of it, and it moves as that far-off point would. Moved with the wall,
//! every reflection sat still on the wall for the synthesised frame and
//! jumped on the next rendered one -- jitter on every polished floor and wall,
//! and a doubled image while walking (headset, 2026-09-29).
//!
//! So the brushes are drawn with [`MotionKind::BrushReflect`] when the frame
//! allows it: each pixel moves as a point along its view ray, beyond the
//! surface by as much as the pixel's reflected image counts
//! ([`reflection_motion_weight`], read from the eye image's alpha,
//! `lights::reflection_alpha`) of how far the reflection reached (read from
//! the probe pass, `probe_pass::REACH_FORMAT`). See [`reflected_point`]. The
//! depth stays the surface's.
//!
//! # Conventions
//!
//! The vectors are in normalised device coordinates with y UP -- wgpu's own
//! clip space, written as it comes -- z 0..1. Not Vulkan's y-down NDC, though
//! these are Vulkan images: Meta's runtime takes Quest vectors "left-handed"
//! (Unity's setting; only Android XR headsets take right-handed, y-down ones),
//! and Godot's Vulkan renderer turns its y-down clip space back up before it
//! writes them. The first build negated y; the pass never ran in normal play
//! then (see `render_frame`), so nothing ever showed which way was right.
//! Depth is the scene's: 0 at the near plane (3 cm), 1 at the far (1 km).

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
    /// x = the sign the vector's y is written with (+1: y up, see the module
    /// docs; -1 only to diagnose); y = a scale on the whole vector (1; 0 only
    /// to diagnose). zw unused.
    pub params: [f32; 4],
    /// The eye, in the space the draw's vertices are in, for
    /// [`MotionKind::BrushReflect`]. w unused.
    pub eye: [f32; 4],
    /// For [`MotionKind::BrushReflect`]: x, y = eye-image pixels per motion
    /// pixel along each axis; z = 1 to move reflected images with what they
    /// show, 0 to move every pixel with its surface; w = 1 to write, instead
    /// of motion, what it read -- the reflected share, the reach in metres and
    /// the distance ratio (DIAGNOSIS ONLY, `space_warp_debug` 32768).
    pub reflect: [f32; 4],
}

/// The farthest a pixel's content is taken to be, in metres: the sky's
/// reflection (`lights::PROBE_REACH_SKY`) moves as a point this far away.
pub const MAX_REFLECTED_DISTANCE: f32 = 10000.0;

/// HOW MUCH MORE CONTRAST A REFLECTED IMAGE CARRIES than the polished surface
/// showing it: a room or a sky reflected varies by about half its brightness
/// from place to place, lit wall to dark corner, doorway to frame; polished
/// marble's own pattern by about a tenth (Marble020's colour map: 0.08 over a
/// few texels, 0.11 over the whole). See [`reflection_motion_weight`].
pub const REFLECTION_CONTRAST_RATIO: f32 = 5.0;

/// HOW MUCH A PIXEL'S REFLECTED IMAGE COUNTS IN ITS MOTION, from its share of
/// the pixel's brightness (`lights::reflected_image` over the whole).
///
/// One vector moves both layers, so one of them is always left behind by
/// some of the difference between their motions. What shows is a layer's
/// CONTRAST moved to the wrong place: weighting each layer's error by its
/// contrast and minimising the sum of squares gives
/// `(k s)^2 / ((k s)^2 + (1 - s)^2)`, with `s` the share and `k`
/// [`REFLECTION_CONTRAST_RATIO`]. Weighting by brightness alone gave the
/// doorway's reflection on the marble floor a fifth of the correction -- the
/// sky there is a fifth of the pixel's light and nearly all of its detail
/// (headset, 2026-09-29).
pub fn reflection_motion_weight(share: f32) -> f32 {
    let s = share.clamp(0.0, 1.0);
    let image = REFLECTION_CONTRAST_RATIO * s;
    let surface = 1.0 - s;
    image * image / (image * image + surface * surface).max(1e-12)
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
    params: vec4<f32>,
    eye: vec4<f32>,
    reflect: vec4<f32>,
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

fn motion_of(curr: vec4<f32>, prev: vec4<f32>) -> vec4<f32> {{
    // A point behind the previous frame's eye had no place on its screen:
    // no motion is the least wrong answer, where dividing by a w near 0
    // would write infinities for the compositor to sample.
    if (prev.w <= {min_w}) {{
        return vec4<f32>(0.0);
    }}
    let d = clamp(curr.xyz / curr.w - prev.xyz / prev.w, vec3<f32>(-{max_d}), vec3<f32>({max_d}));
    return vec4<f32>(d.x, d.y * cam.params.x, d.z, 0.0) * cam.params.y;
}}

@fragment fn fs_main(in: VOut) -> @location(0) vec4<f32> {{
    return motion_of(in.curr, in.prev);
}}

// REFLECTIONS: the brushes again, each pixel moved as what it shows. See the
// module docs and `reflected_point`, which this is held to.
@group(1) @binding(0) var eye_image: texture_2d<f32>;
@group(1) @binding(1) var reach_image: texture_2d<f32>;

struct VOutReflect {{
    @builtin(position) clip: vec4<f32>,
    @location(0) pos: vec3<f32>,
}}

@vertex fn vs_reflect(@location(0) pos: vec3<f32>) -> VOutReflect {{
    var out: VOutReflect;
    out.clip = cam.curr * vec4<f32>(pos, 1.0);
    out.pos = pos;
    return out;
}}

// What one eye-image pixel at `texel` says: how much its reflected image
// counts in its motion (`lights::reflection_alpha`) and how far that image's
// ray reached. The probe pass is half the eye image each way.
fn reflect_inputs(texel: vec2<f32>) -> vec2<f32> {{
    let size = vec2<i32>(textureDimensions(eye_image));
    let e = clamp(vec2<i32>(texel), vec2<i32>(0), size - vec2<i32>(1));
    let weight = clamp(1.0 - textureLoad(eye_image, e, 0).a, 0.0, 1.0);
    let reach_size = vec2<i32>(textureDimensions(reach_image));
    let r = clamp(e / 2, vec2<i32>(0), reach_size - vec2<i32>(1));
    return vec2<f32>(weight, max(textureLoad(reach_image, r, 0).r, 0.0));
}}

// The ratio of a surface point's distance `d` from the eye to the distance
// its pixel's content moves as: 1 for the surface, less as the image counts
// for more and reaches further.

fn reflect_ratio(texel: vec2<f32>, d: f32) -> f32 {{
    let i = reflect_inputs(texel);
    return 1.0 - i.x * i.y / (d + i.y);
}}

@fragment fn fs_reflect(in: VOutReflect) -> @location(0) vec4<f32> {{
    var p = in.pos;
    if (cam.reflect.z > 0.5) {{
        let to = in.pos - cam.eye.xyz;
        let d = max(length(to), 1e-3);
        // FOUR of the eye pixels under this motion pixel, one in each
        // quarter of its block: a highlight a few pixels wide still counts.
        // Averaged as ratios -- inverse distances -- which is averaging the
        // motions they make.
        let base = floor(in.clip.xy);
        let s = cam.reflect.xy;
        let ratio = 0.25 * (
            reflect_ratio((base + vec2<f32>(0.25, 0.25)) * s, d)
            + reflect_ratio((base + vec2<f32>(0.75, 0.25)) * s, d)
            + reflect_ratio((base + vec2<f32>(0.25, 0.75)) * s, d)
            + reflect_ratio((base + vec2<f32>(0.75, 0.75)) * s, d)
        );
        p = cam.eye.xyz + to / max(ratio, d / {max_reflected});
        if (cam.reflect.w > 0.5) {{
            let seen = reflect_inputs((base + vec2<f32>(0.5)) * s);
            return vec4<f32>(seen.x, seen.y, ratio, 1.0);
        }}
    }}
    return motion_of(cam.curr * vec4<f32>(p, 1.0), cam.prev * vec4<f32>(p, 1.0));
}}
"#,
        joints = crate::renderer::mesh::MAX_SKIN_JOINTS,
        min_w = format!("{:?}", MIN_PREV_W),
        max_d = format!("{:?}", MAX_NDC_MOTION),
        max_reflected = format!("{:?}", MAX_REFLECTED_DISTANCE),
    )
}

/// WHERE A PIXEL'S CONTENT IS, for its motion: the CPU reference the shader's
/// `fs_reflect` is held to.
///
/// `p` is the surface the eye at `eye` sees; `weight` how much the pixel's
/// reflected image counts in its motion ([`reflection_motion_weight`], read
/// back from the eye image's alpha), `reach` how far past the surface the
/// reflected ray went to what the image shows. A planar mirror's image of
/// that thing is on the view ray, `reach` beyond `p` -- which moves on screen
/// exactly as the thing's reflection does.
///
/// A pixel that is part surface and part image has one motion for both. Moving
/// the eye moves a point on screen by an amount that goes as one over its
/// distance, so blending the two layers' inverse distances by `weight`
/// blends their motions by it.
pub fn reflected_point(eye: glam::Vec3, p: glam::Vec3, weight: f32, reach: f32) -> glam::Vec3 {
    let to = p - eye;
    let d = to.length().max(1e-3);
    let reach = reach.max(0.0);
    let ratio = 1.0 - weight.clamp(0.0, 1.0) * reach / (d + reach);
    eye + to / ratio.max(d / MAX_REFLECTED_DISTANCE)
}

/// Below this previous clip w a point counts as having been behind the eye.
pub const MIN_PREV_W: f32 = 1e-4;

/// The largest motion written, in NDC per axis: a point that crossed the
/// whole screen in one frame gives the compositor nothing to extrapolate,
/// and a half float saturates to infinity not far above.
pub const MAX_NDC_MOTION: f32 = 2.0;

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
    if q.w <= MIN_PREV_W {
        return glam::Vec3::ZERO;
    }
    (c.truncate() / c.w - q.truncate() / q.w).clamp(glam::Vec3::splat(-MAX_NDC_MOTION), glam::Vec3::splat(MAX_NDC_MOTION))
}

/// `appSpaceDeltaPose` as a rotation and a translation: the tracking space's
/// pose in the world last frame, inverted, times this frame's --
/// `Inv(prev) * curr`, as Meta's XrSpaceWarp sample builds it. The tracking
/// space is where the player stands, its pose in the world
/// `inverse(world_to_player)`, so the delta is
/// `prev_world_to_player * inverse(world_to_player)`: walking forward a metre
/// is a metre forward, turning is the turn.
pub fn app_space_delta_rigid(prev_world_to_player: Mat4, world_to_player: Mat4) -> (glam::Quat, glam::Vec3) {
    let (_, rotation, translation) = (prev_world_to_player * world_to_player.inverse()).to_scale_rotation_translation();
    (rotation, translation)
}

/// A LOCOMOTION JUMP the compositor must not carry on: a snap turn or a
/// teleport. Its frame's motion vectors hold the whole jump -- every pixel of
/// a 45 degree snap thrown to [`MAX_NDC_MOTION`] -- and the frame synthesised
/// from them threw the world on by half of that again, once at every snap,
/// for players who turn by stick rather than by turning round (user,
/// 2026-10-01: snap turns, smooth turns and physical turns must all hold).
/// The spec's answer is `XR_COMPOSITION_LAYER_SPACE_WARP_INFO_FRAME_SKIP_BIT_FB`
/// on that frame: no synthesis from it. Smooth turning and walking stay far
/// under the bounds -- 180 degrees a second is 5 a frame at 36 frames a
/// second, a sprint 0.1 m -- so only a jump skips.
pub fn locomotion_jumped(prev_world_to_player: Mat4, world_to_player: Mat4) -> bool {
    let (r, t) = app_space_delta_rigid(prev_world_to_player, world_to_player);
    r.angle_between(glam::Quat::IDENTITY) > JUMP_TURN_DEGREES.to_radians() || t.length() > JUMP_METRES
}

/// How far locomotion may turn the player in one frame and still be motion:
/// see [`locomotion_jumped`].
pub const JUMP_TURN_DEGREES: f32 = 15.0;

/// How far it may carry them: see [`locomotion_jumped`].
pub const JUMP_METRES: f32 = 0.5;

/// [`app_space_delta_rigid`] as the `XrPosef` the layer info carries.
#[cfg(target_os = "android")]
pub fn app_space_delta(prev_world_to_player: Mat4, world_to_player: Mat4) -> xr::Posef {
    let (r, t) = app_space_delta_rigid(prev_world_to_player, world_to_player);
    xr::Posef {
        orientation: xr::Quaternionf { x: r.x, y: r.y, z: r.z, w: r.w },
        position: xr::Vector3f { x: t.x, y: t.y, z: t.z },
    }
}

/// Which vertex layout a motion draw reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MotionKind {
    Brush,
    /// The brushes, moving reflected images with what they show: group 1 is
    /// the eye image and the probe pass's reach (`reflect_layout`). See the
    /// module docs.
    BrushReflect,
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
    /// Group 1 of [`MotionKind::BrushReflect`]: the eye image this frame's
    /// scene pass drew, and the probe pass's reach, each one eye's.
    pub reflect_layout: wgpu::BindGroupLayout,
    brush: wgpu::RenderPipeline,
    brush_reflect: wgpu::RenderPipeline,
    solid: wgpu::RenderPipeline,
    mesh: wgpu::RenderPipeline,
    skinned: wgpu::RenderPipeline,
    /// Whether the depth format carries stencil (D24S8), which the pass then
    /// clears too rather than leave for wgpu to zero on first use.
    stencil: bool,
}

impl MotionPipelines {
    pub fn new(device: &wgpu::Device, depth_format: wgpu::TextureFormat) -> Self {
        let camera_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("space_warp_camera"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
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
        let image_entry = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let reflect_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("space_warp_reflect"),
            entries: &[image_entry(0), image_entry(1)],
        });
        let reflect_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("space_warp_reflect"),
            bind_group_layouts: &[Some(&camera_layout), Some(&reflect_layout)],
            immediate_size: 0,
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("space_warp"),
            source: wgpu::ShaderSource::Wgsl(shader().into()),
        });
        let make = |label: &str, layout: &wgpu::PipelineLayout, (entry, fragment): (&str, &str), stride: u64, attributes: &[wgpu::VertexAttribute]| {
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
                    entry_point: Some(fragment),
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
        let plain = ("vs_main", "fs_main");
        let brush_stride = size(std::mem::size_of::<crate::renderer::brush_pipeline::BrushVertex>());
        let brush = make("space_warp_brush", &plain_layout, plain, brush_stride, &position);
        let brush_reflect = make("space_warp_brush_reflect", &reflect_pipeline_layout, ("vs_reflect", "fs_reflect"), brush_stride, &position);
        let solid = make("space_warp_solid", &plain_layout, plain, size(std::mem::size_of::<crate::renderer::cuboid::SolidVertex>()), &position);
        let mesh = make("space_warp_mesh", &plain_layout, plain, size(std::mem::size_of::<crate::renderer::mesh::MeshVertex>()), &position);
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
            ("vs_skinned", "fs_main"),
            size(std::mem::size_of::<crate::renderer::mesh::SkinnedMeshVertex>()),
            &skinned_attributes,
        );
        Self {
            camera_layout,
            joints_layout,
            reflect_layout,
            brush,
            brush_reflect,
            solid,
            mesh,
            skinned,
            stencil: depth_format.has_stencil_aspect(),
        }
    }

    /// Group 1 of [`MotionKind::BrushReflect`] for one eye: `eye_image`, a
    /// single-layer view of what the scene pass drew, and `reach`, the probe
    /// pass's reach target for the same eye.
    pub fn reflect_bind_group(&self, device: &wgpu::Device, eye_image: &wgpu::TextureView, reach: &wgpu::TextureView) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("space_warp_reflect"),
            layout: &self.reflect_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(eye_image) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(reach) },
            ],
        })
    }

    fn pipeline(&self, kind: MotionKind) -> &wgpu::RenderPipeline {
        match kind {
            MotionKind::Brush => &self.brush,
            MotionKind::BrushReflect => &self.brush_reflect,
            MotionKind::Solid => &self.solid,
            MotionKind::Mesh => &self.mesh,
            MotionKind::Skinned => &self.skinned,
        }
    }
}

/// One draw of the motion pass: its geometry, the ring slot of its two
/// cameras, and its group 1 -- a skinned mesh's joints, or the brushes'
/// reflection inputs (`MotionPipelines::reflect_bind_group`).
pub struct MotionDraw<'a> {
    pub kind: MotionKind,
    pub vertices: &'a wgpu::Buffer,
    pub indices: &'a wgpu::Buffer,
    /// The first index and how many, of `indices`.
    pub first: u32,
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
    stencil_store: bool,
    depth_clear: f32,
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
            depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(depth_clear), store: wgpu::StoreOp::Store }),
            stencil_ops: pipelines
                .stencil
                .then_some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(0),
                    store: if stencil_store { wgpu::StoreOp::Store } else { wgpu::StoreOp::Discard },
                }),
        }),
        ..Default::default()
    });
    for d in draws.iter().filter(|d| d.count > 0 && d.slot < MAX_SLOTS) {
        if matches!(d.kind, MotionKind::Skinned | MotionKind::BrushReflect) && d.joints.is_none() {
            continue;
        }
        pass.set_pipeline(pipelines.pipeline(d.kind));
        pass.set_bind_group(0, cameras, &[(d.slot as u64 * SLOT_STRIDE) as u32]);
        if let Some(j) = d.joints {
            pass.set_bind_group(1, j, &[]);
        }
        pass.set_vertex_buffer(0, d.vertices.slice(..));
        pass.set_index_buffer(d.indices.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(d.first..d.first + d.count, 0, 0..1);
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

    /// The head rising: the world drops on screen, which with y up is -y.
    #[test]
    fn rising_drops_the_world_down_the_image() {
        let prev_vp = view_proj(Vec3::new(0.0, 1.6, 0.0), 0.0);
        let vp = view_proj(Vec3::new(0.0, 1.7, 0.0), 0.0);
        let mv = motion_vector(vp, previous_clip(prev_vp, Mat4::IDENTITY, Mat4::IDENTITY), Vec3::new(0.0, 1.6, -3.0));
        assert!(mv.y < -0.01, "{mv}");
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

    /// `appSpaceDeltaPose` is where this frame's tracking space sits in last
    /// frame's: half a metre forward is (0, 0, -0.5); a turn is the turn.
    #[test]
    fn the_delta_pose_is_the_locomotion() {
        let w2p = |offset: Vec3, yaw: f32| Mat4::from_quat(Quat::from_rotation_y(yaw).inverse()) * Mat4::from_translation(-offset);
        let (r, t) = app_space_delta_rigid(w2p(Vec3::new(2.0, 0.0, 1.0), 0.0), w2p(Vec3::new(2.0, 0.0, 0.5), 0.0));
        assert!(r.angle_between(Quat::IDENTITY) < 1e-5, "{r}");
        assert!((t - Vec3::new(0.0, 0.0, -0.5)).length() < 1e-5, "{t}");
        let (r, t) = app_space_delta_rigid(w2p(Vec3::ZERO, 0.3), w2p(Vec3::ZERO, 0.5));
        assert!(r.angle_between(Quat::from_rotation_y(0.2)) < 1e-5, "{r}");
        assert!(t.length() < 1e-5, "{t}");
        let (r, t) = app_space_delta_rigid(w2p(Vec3::ONE, 0.4), w2p(Vec3::ONE, 0.4));
        assert!(r.angle_between(Quat::IDENTITY) < 1e-5 && t.length() < 1e-5, "{r} {t}");
    }

    /// A SNAP TURN OR A TELEPORT IS A JUMP, and no extrapolation may cross it;
    /// smooth turning and walking at any sensible speed are motion. Each from
    /// anywhere, facing anywhere: the jump is the change, not the pose.
    #[test]
    fn snap_turns_and_teleports_skip_synthesis_and_smooth_motion_does_not() {
        let w2p = |offset: Vec3, yaw: f32| Mat4::from_quat(Quat::from_rotation_y(yaw).inverse()) * Mat4::from_translation(-offset);
        for (at, facing) in [(Vec3::ZERO, 0.0f32), (Vec3::new(3.0, 0.0, -7.0), 2.5), (Vec3::new(-1.0, 0.2, 4.0), -1.0)] {
            let before = w2p(at, facing);
            let jump = |offset: Vec3, turn_degrees: f32| locomotion_jumped(before, w2p(at + offset, facing + turn_degrees.to_radians()));
            for snap in [30.0, 45.0, -45.0, 90.0, 180.0] {
                assert!(jump(Vec3::ZERO, snap), "a {snap} degree snap at {at} facing {facing}");
            }
            assert!(jump(Vec3::new(2.0, 0.0, -1.0), 0.0), "a teleport");
            // Smooth turning: 180 degrees a second at 36 frames a second.
            assert!(!jump(Vec3::ZERO, 5.0) && !jump(Vec3::ZERO, -5.0), "a smooth turn");
            // Walking and sprinting, turning as they go.
            assert!(!jump(Vec3::new(0.04, 0.0, -0.03), 2.0) && !jump(Vec3::new(0.1, 0.0, 0.0), -4.0), "walking");
            assert!(!jump(Vec3::ZERO, 0.0), "standing still");
        }
    }

    /// A floor mirror: the image of a point it reflects lies on the view ray,
    /// as far below the floor as the point is above it -- and that image, not
    /// the floor, is what the reflection moves as.
    #[test]
    fn a_reflection_moves_as_the_image_of_what_it_shows() {
        let eye = Vec3::new(0.0, 1.6, 0.0);
        let thing = Vec3::new(0.5, 1.0, -6.0);
        let image = Vec3::new(thing.x, -thing.y, thing.z);
        // Where the floor (y = 0) shows it: the view ray to the image.
        let p = eye + (image - eye) * (eye.y / (eye.y - image.y));
        let v = reflected_point(eye, p, 1.0, (thing - p).length());
        assert!((v - image).length() < 1e-4, "{v} vs {image}");
        // Stepping 5 cm right, from where the frame was drawn: the reflection
        // moves exactly as the image does, and measurably unlike the floor.
        let prev = view_proj(eye - Vec3::new(0.05, 0.0, 0.0), 0.0);
        let curr = view_proj(eye, 0.0);
        let truth = motion_vector(curr, prev, image);
        let moved = motion_vector(curr, prev, v);
        let with_floor = motion_vector(curr, prev, p);
        assert!((moved - truth).truncate().length() < 1e-5, "{moved} vs {truth}");
        assert!((with_floor - truth).truncate().length() > 0.002, "the floor alone was right: {with_floor} vs {truth}");
    }

    /// The image's weight in a pixel's motion: none without a reflection, all
    /// of it for a mirror, rising steadily between -- and already most of it
    /// where a sky is a fifth of the light on polished marble, as the doorway's
    /// reflection on the floor measured (headset, 2026-09-29).
    #[test]
    fn a_reflection_counts_by_its_contrast_not_its_brightness() {
        assert_eq!(reflection_motion_weight(0.0), 0.0);
        assert!((reflection_motion_weight(1.0) - 1.0).abs() < 1e-6);
        let mut last = 0.0;
        for i in 1..=100 {
            let w = reflection_motion_weight(i as f32 / 100.0);
            assert!(w > last, "not rising at {i}%: {w} after {last}");
            last = w;
        }
        let at_a_fifth = reflection_motion_weight(0.2);
        assert!(at_a_fifth > 0.55 && at_a_fifth < 0.7, "{at_a_fifth}");
        assert!(reflection_motion_weight(0.05) < 0.1, "a faint reflection took over its surface");
    }

    /// No reflected share, or a reflection that reaches nowhere: the surface.
    /// And the sky's reach is far but finite.
    #[test]
    fn a_pixel_with_no_reflection_moves_with_its_surface() {
        let eye = Vec3::new(0.0, 1.6, 0.0);
        let p = Vec3::new(1.0, 0.0, -4.0);
        assert!((reflected_point(eye, p, 0.0, 30.0) - p).length() < 1e-5);
        assert!((reflected_point(eye, p, 1.0, 0.0) - p).length() < 1e-5);
        let sky = reflected_point(eye, p, 1.0, 1e30);
        assert!(sky.is_finite() && ((sky - eye).length() - MAX_REFLECTED_DISTANCE).abs() < 1.0, "{sky}");
    }

    /// Half image, half surface: exactly halfway between the two layers'
    /// motions on screen, because a step of the eye moves a point by one over
    /// its distance and the blend is of inverse distances.
    #[test]
    fn a_half_reflected_pixel_moves_between_its_two_layers() {
        let eye = Vec3::new(0.0, 1.6, 0.0);
        let p = Vec3::new(0.3, 0.0, -3.0);
        let prev = view_proj(eye - Vec3::new(0.02, 0.0, 0.0), 0.0);
        let curr = view_proj(eye, 0.0);
        let surface = motion_vector(curr, prev, p).truncate();
        let image = motion_vector(curr, prev, reflected_point(eye, p, 1.0, 5.0)).truncate();
        let half = motion_vector(curr, prev, reflected_point(eye, p, 0.5, 5.0)).truncate();
        let mid = (surface + image) * 0.5;
        assert!((half - mid).length() < 1e-4 * (surface - image).length().max(1e-3), "{half} vs {mid}");
        assert!((surface - image).length() > 1e-3, "the two layers moved alike: {surface} {image}");
    }

    /// The shader moves a reflection as the CPU reference does: a wall 3 m
    /// ahead, every pixel half an image 8 m beyond it, the eye stepping right
    /// -- and with the switch off, as the wall.
    #[test]
    fn the_shader_moves_reflections_as_the_reference_does() {
        let Some((device, queue)) = crate::renderer::terrain_pipeline::tests::headless_gpu() else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        use wgpu::util::DeviceExt;
        const W: u32 = 64;
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let pipelines = MotionPipelines::new(&device, wgpu::TextureFormat::Depth32Float);
        let eye = Vec3::new(0.0, 1.6, 0.0);
        let prev = view_proj(eye - Vec3::new(0.05, 0.0, 0.0), 0.0);
        let curr = view_proj(eye, 0.0);

        // The eye image, four times the motion size, alpha 128: a share of
        // 127/255. The reach, half the eye image, 8 m everywhere.
        let texture = |label, size: u32, format, bytes: &[u8]| {
            device
                .create_texture_with_data(
                    &queue,
                    &wgpu::TextureDescriptor {
                        label: Some(label),
                        size: wgpu::Extent3d { width: size, height: size, depth_or_array_layers: 1 },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format,
                        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                        view_formats: &[],
                    },
                    wgpu::util::TextureDataOrder::LayerMajor,
                    bytes,
                )
                .create_view(&Default::default())
        };
        let eye_bytes: Vec<u8> = (0..(4 * W) * (4 * W)).flat_map(|_| [0u8, 0, 0, 128]).collect();
        let eye_view = texture("eye", 4 * W, wgpu::TextureFormat::Rgba8UnormSrgb, &eye_bytes);
        // The reach, 8 m, in R; no face's code in G.
        let reach_bytes: Vec<u8> = (0..(2 * W) * (2 * W))
            .flat_map(|_| [0x4800u16, 0].into_iter().flat_map(u16::to_le_bytes))
            .collect();
        let reach_view = texture("reach", 2 * W, crate::renderer::brush_pipeline::probe_pass::REACH_FORMAT, &reach_bytes);
        let reflect = pipelines.reflect_bind_group(&device, &eye_view, &reach_view);

        // Two slots: reflections on, then off.
        let mut ring = vec![0u8; 2 * SLOT_STRIDE as usize];
        for (slot, on) in [(0usize, 1.0f32), (1, 0.0)] {
            let cam = MotionCamera {
                curr: curr.to_cols_array_2d(),
                prev: prev.to_cols_array_2d(),
                params: [1.0, 1.0, 0.0, 0.0],
                eye: [eye.x, eye.y, eye.z, 1.0],
                reflect: [4.0, 4.0, on, 0.0],
            };
            let at = slot * SLOT_STRIDE as usize;
            ring[at..at + std::mem::size_of::<MotionCamera>()].copy_from_slice(bytemuck::bytes_of(&cam));
        }
        let ring = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ring"),
            contents: &ring,
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let cameras = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cameras"),
            layout: &pipelines.camera_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &ring,
                    offset: 0,
                    size: wgpu::BufferSize::new(std::mem::size_of::<MotionCamera>() as u64),
                }),
            }],
        });

        // The wall: a quad at z = -3 over the whole view, as brush vertices.
        let stride = std::mem::size_of::<crate::renderer::brush_pipeline::BrushVertex>();
        let mut vertices = vec![0u8; 4 * stride];
        for (i, (x, y)) in [(-10.0f32, -10.0f32), (10.0, -10.0), (10.0, 12.0), (-10.0, 12.0)].iter().enumerate() {
            vertices[i * stride..i * stride + 12].copy_from_slice(bytemuck::cast_slice(&[*x, *y, -3.0f32]));
        }
        let vertices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("wall"),
            contents: &vertices,
            usage: wgpu::BufferUsages::VERTEX,
        });
        let indices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("wall_indices"),
            contents: bytemuck::cast_slice(&[0u32, 1, 2, 0, 2, 3]),
            usage: wgpu::BufferUsages::INDEX,
        });

        let target = |label, format, usage| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d { width: W, height: W, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
        };
        let depth = target("depth", wgpu::TextureFormat::Depth32Float, wgpu::TextureUsages::RENDER_ATTACHMENT);
        let depth_view = depth.create_view(&Default::default());

        let run = |slot: u32| -> Vec3 {
            let motion = target(
                "motion",
                MOTION_FORMAT,
                wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            );
            let readback = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("readback"),
                size: (W * W * 8) as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut encoder = device.create_command_encoder(&Default::default());
            let draw = MotionDraw {
                kind: MotionKind::BrushReflect,
                vertices: &vertices,
                indices: &indices,
                first: 0,
                count: 6,
                slot,
                joints: Some(&reflect),
            };
            record(&mut encoder, &pipelines, &cameras, &motion.create_view(&Default::default()), &depth_view, &[draw], false, 1.0);
            encoder.copy_texture_to_buffer(
                motion.as_image_copy(),
                wgpu::TexelCopyBufferInfo {
                    buffer: &readback,
                    layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(W * 8), rows_per_image: None },
                },
                wgpu::Extent3d { width: W, height: W, depth_or_array_layers: 1 },
            );
            queue.submit(Some(encoder.finish()));
            readback.slice(..).map_async(wgpu::MapMode::Read, |_| {});
            let _ = device.poll(wgpu::PollType::wait_indefinitely());
            let bytes = readback.slice(..).get_mapped_range().unwrap();
            let at = ((W / 2) * W + W / 2) as usize * 8;
            let half = |i: usize| f16_to_f32(u16::from_le_bytes([bytes[at + 2 * i], bytes[at + 2 * i + 1]]));
            Vec3::new(half(0), half(1), half(2))
        };
        let reflected = run(0);
        let plain = run(1);
        let err = pollster::block_on(scope.pop());
        assert!(err.is_none(), "{err:?}");

        // The wall point at the centre of pixel (32, 32), and what it moves as.
        let ndc = glam::Vec2::new((W / 2) as f32 + 0.5, (W / 2) as f32 + 0.5) / W as f32 * 2.0 - glam::Vec2::ONE;
        let ndc = glam::Vec2::new(ndc.x, -ndc.y);
        let inv = curr.inverse();
        let near = inv.project_point3(ndc.extend(0.0));
        let far = inv.project_point3(ndc.extend(1.0));
        let p = near + (far - near) * ((-3.0 - near.z) / (far.z - near.z));
        let share = 1.0 - 128.0 / 255.0;
        let want = motion_vector(curr, prev, reflected_point(eye, p, share, 8.0));
        let want_plain = motion_vector(curr, prev, p);
        let close = |got: Vec3, want: Vec3| (got.truncate() - want.truncate()).abs().max_element() < 2e-4;
        assert!(close(reflected, want), "shader {reflected} vs reference {want}");
        assert!(close(plain, want_plain), "switched off: {plain} vs the wall's {want_plain}");
        assert!((want - want_plain).truncate().length() > 1e-3, "the test cannot tell them apart");
    }

    /// A half float's value: enough of IEEE 754 binary16 for the test above.
    fn f16_to_f32(h: u16) -> f32 {
        let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
        let exp = ((h >> 10) & 0x1f) as i32;
        let frac = (h & 0x3ff) as f32;
        match exp {
            0 => sign * frac * 2f32.powi(-24),
            31 => sign * f32::INFINITY,
            e => sign * (1.0 + frac / 1024.0) * 2f32.powi(e - 15),
        }
    }

    /// A point that was behind last frame's eye writes no motion, and no
    /// motion is ever larger than the whole screen: nothing the compositor
    /// samples can be infinite.
    #[test]
    fn motion_is_always_finite() {
        let vp = view_proj(Vec3::new(0.0, 1.6, 0.0), 0.0);
        let behind = view_proj(Vec3::new(0.0, 1.6, -6.0), 0.0);
        let mv = motion_vector(vp, behind, Vec3::new(0.0, 1.6, -4.0));
        assert_eq!(mv, Vec3::ZERO);
        let grazing = view_proj(Vec3::new(0.0, 1.6, 0.0), 1.5);
        let mv = motion_vector(vp, grazing, Vec3::new(0.3, 1.6, -4.0));
        assert!(mv.is_finite() && mv.abs().max_element() <= MAX_NDC_MOTION, "{mv}");
    }
}

/// DIAGNOSIS ONLY: what the motion and depth images really hold after the
/// motion pass, read straight back through Vulkan (wgpu will not copy a
/// D24S8 depth aspect). The compositor behaved as if every depth it read was
/// 0 whatever the pass cleared it to (desk test, 2026-09-29); this says
/// whether the pass's depth is in the image at all. `space_warp_debug` 8192.
#[cfg(target_os = "android")]
pub struct Readback {
    device: ash::Device,
    queue: ash::vk::Queue,
    pool: ash::vk::CommandPool,
    buffer: ash::vk::Buffer,
    memory: ash::vk::DeviceMemory,
    size: (u32, u32),
}

#[cfg(target_os = "android")]
impl Readback {
    pub fn new(
        instance: &ash::Instance,
        physical: ash::vk::PhysicalDevice,
        device: &ash::Device,
        queue: ash::vk::Queue,
        family: u32,
        size: (u32, u32),
    ) -> Result<Self, ash::vk::Result> {
        use ash::vk;
        let bytes = (size.0 * size.1) as u64 * (4 + 8);
        unsafe {
            let buffer = device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(bytes)
                    .usage(vk::BufferUsageFlags::TRANSFER_DST)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )?;
            let req = device.get_buffer_memory_requirements(buffer);
            let props = instance.get_physical_device_memory_properties(physical);
            let want = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
            let ty = (0..props.memory_type_count)
                .find(|&i| req.memory_type_bits & (1 << i) != 0 && props.memory_types[i as usize].property_flags.contains(want))
                .ok_or(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY)?;
            let memory = device.allocate_memory(&vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(ty), None)?;
            device.bind_buffer_memory(buffer, memory, 0)?;
            let pool = device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                    .queue_family_index(family),
                None,
            )?;
            Ok(Self { device: device.clone(), queue, pool, buffer, memory, size })
        }
    }

    /// Copy `layer` of both images back, wait, and describe them.
    /// `save`: also write both images, raw, to this file -- a 16-byte header
    /// (width, height, layer, whether depth is D24S8, each a little-endian
    /// u32), the depth words, then the motion texels as four half floats.
    pub fn read(
        &self,
        depth: ash::vk::Image,
        depth_has_stencil: bool,
        motion: ash::vk::Image,
        layer: u32,
        save: Option<&std::path::Path>,
    ) -> Result<String, ash::vk::Result> {
        use ash::vk;
        let (w, h) = self.size;
        let d = &self.device;
        unsafe {
            let cmd = d.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default().command_pool(self.pool).level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(1),
            )?[0];
            d.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))?;
            let depth_aspects = if depth_has_stencil { vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL } else { vk::ImageAspectFlags::DEPTH };
            let range = |aspect| vk::ImageSubresourceRange { aspect_mask: aspect, base_mip_level: 0, level_count: 1, base_array_layer: layer, layer_count: 1 };
            let barrier = |image, aspect, from, to, src, dst| {
                vk::ImageMemoryBarrier::default()
                    .src_access_mask(src)
                    .dst_access_mask(dst)
                    .old_layout(from)
                    .new_layout(to)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(image)
                    .subresource_range(range(aspect))
            };
            let to_copy = [
                barrier(depth, depth_aspects, vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL, vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE, vk::AccessFlags::TRANSFER_READ),
                barrier(motion, vk::ImageAspectFlags::COLOR, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    vk::AccessFlags::COLOR_ATTACHMENT_WRITE, vk::AccessFlags::TRANSFER_READ),
            ];
            d.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[], &to_copy);
            let layers = |aspect| vk::ImageSubresourceLayers { aspect_mask: aspect, mip_level: 0, base_array_layer: layer, layer_count: 1 };
            let extent = vk::Extent3D { width: w, height: h, depth: 1 };
            d.cmd_copy_image_to_buffer(cmd, depth, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, self.buffer,
                &[vk::BufferImageCopy::default().buffer_offset(0).image_subresource(layers(vk::ImageAspectFlags::DEPTH)).image_extent(extent)]);
            d.cmd_copy_image_to_buffer(cmd, motion, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, self.buffer,
                &[vk::BufferImageCopy::default().buffer_offset((w * h * 4) as u64).image_subresource(layers(vk::ImageAspectFlags::COLOR)).image_extent(extent)]);
            let back = [
                barrier(depth, depth_aspects, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                    vk::AccessFlags::TRANSFER_READ, vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE),
                barrier(motion, vk::ImageAspectFlags::COLOR, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                    vk::AccessFlags::TRANSFER_READ, vk::AccessFlags::COLOR_ATTACHMENT_WRITE),
            ];
            d.cmd_pipeline_barrier(cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::ALL_COMMANDS, vk::DependencyFlags::empty(), &[], &[], &back);
            d.end_command_buffer(cmd)?;
            let fence = d.create_fence(&vk::FenceCreateInfo::default(), None)?;
            let cmds = [cmd];
            let submitted = d.queue_submit(self.queue, &[vk::SubmitInfo::default().command_buffers(&cmds)], fence);
            let waited = submitted.and_then(|()| d.wait_for_fences(&[fence], true, u64::MAX));
            d.destroy_fence(fence, None);
            d.free_command_buffers(self.pool, &cmds);
            waited?;
            let n = (w * h) as usize;
            let ptr = d.map_memory(self.memory, 0, (n * 12) as u64, vk::MemoryMapFlags::empty())? as *const u8;
            let bytes = std::slice::from_raw_parts(ptr, n * 12);
            if let Some(path) = save {
                let mut file = Vec::with_capacity(16 + n * 12);
                for v in [w, h, layer, depth_has_stencil as u32] {
                    file.extend_from_slice(&v.to_le_bytes());
                }
                file.extend_from_slice(bytes);
                if let Err(e) = std::fs::write(path, &file) {
                    log::warn!("SWDUMP: could not write {}: {e}", path.display());
                }
            }
            let depth_word = |i: usize| u32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap());
            let depth_value = |i: usize| {
                if depth_has_stencil {
                    (depth_word(i) & 0x00FF_FFFF) as f32 / 16_777_215.0
                } else {
                    f32::from_bits(depth_word(i))
                }
            };
            let (mut dmin, mut dmax, mut dsum, mut dzero, mut done) = (f32::MAX, f32::MIN, 0.0f64, 0usize, 0usize);
            for i in 0..n {
                let v = depth_value(i);
                dmin = dmin.min(v);
                dmax = dmax.max(v);
                dsum += v as f64;
                dzero += (v == 0.0) as usize;
                done += (v >= 1.0) as usize;
            }
            let half = |i: usize, c: usize| {
                let at = n * 4 + i * 8 + c * 2;
                half_to_f32(u16::from_le_bytes([bytes[at], bytes[at + 1]]))
            };
            let (mut xmax, mut ymax, mut zmax, mut nan, mut moving) = (0.0f32, 0.0f32, 0.0f32, 0usize, 0usize);
            for i in 0..n {
                let (x, y, z) = (half(i, 0), half(i, 1), half(i, 2));
                if !(x.is_finite() && y.is_finite() && z.is_finite()) {
                    nan += 1;
                    continue;
                }
                xmax = xmax.max(x.abs());
                ymax = ymax.max(y.abs());
                zmax = zmax.max(z.abs());
                moving += (x.abs() + y.abs() > 1e-4) as usize;
            }
            let centre = depth_value((h / 2 * w + w / 2) as usize);
            d.unmap_memory(self.memory);
            Ok(format!(
                "layer {layer}: depth min {dmin:.6} max {dmax:.6} mean {:.6} centre {centre:.6}, {dzero} at 0, {done} at 1 of {n}; \
                 motion |x|<={xmax:.5} |y|<={ymax:.5} |z|<={zmax:.5}, {moving} moving, {nan} not finite",
                dsum / n as f64
            ))
        }
    }
}

#[cfg(target_os = "android")]
fn half_to_f32(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = ((h >> 10) & 0x1F) as i32;
    let frac = (h & 0x3FF) as f32;
    match exp {
        0 => sign * frac * 2f32.powi(-24),
        31 => if frac == 0.0 { sign * f32::INFINITY } else { f32::NAN },
        _ => sign * (1.0 + frac / 1024.0) * 2f32.powi(exp - 15),
    }
}
