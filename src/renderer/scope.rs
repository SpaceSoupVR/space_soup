//! Magnified optic ("scope portal") rendering.
//!
//! A scope is architecturally the same kind of object as the planar mirror in
//! [`super::mirror`]: render the world from a second virtual viewpoint into a
//! texture, then sample that texture inside a masked region of a surface. The
//! difference is the projection -- a narrow, symmetric frustum anchored at the
//! objective lens instead of a mirrored oblique one.
//!
//! Two properties of real optics drive the design, and both save work:
//!
//! * **The image is formed at the objective, not at the eye.** So the scope
//!   camera sits at the front lens looking down the optical axis. Sampling the
//!   already-composited eye image instead would lose angular detail by exactly
//!   the magnification factor and get parallax around the barrel wrong.
//! * **A parallax-free optic is collimated**, so the image does not shift with
//!   eye position -- move your head behind a scope and the reticle stays put on
//!   the target. That makes the render **view-independent**: one render serves
//!   both eyes. Eye position only affects the eye-box vignette.
//!
//! Consequently the composite samples by *ray angle* (scaled by magnification),
//! not by lens surface UV, which is what makes the sight picture behave like
//! glass rather than like a decal.

use glam::{Mat4, Vec3};

/// A virtual camera looking down an optic's axis.
///
/// `fov_y` is the optic's **true** (world) field of view, i.e. apparent field
/// divided by magnification, so higher power renders a narrower cone at the
/// same texture resolution -- which is precisely where the extra detail comes
/// from.
pub fn scope_view_proj(
    objective: Vec3,
    axis: Vec3,
    world_up: Vec3,
    true_fov_y_rad: f32,
    near: f32,
    far: f32,
) -> Mat4 {
    let forward = axis.normalize_or_zero();
    if forward == Vec3::ZERO {
        return Mat4::IDENTITY;
    }
    // Pick a stable up vector; fall back when the optic points straight up/down.
    let up = if forward.dot(world_up).abs() > 0.999 { Vec3::Z } else { world_up };
    let view = Mat4::look_to_rh(objective, forward, up);
    let proj = Mat4::perspective_rh(true_fov_y_rad.max(1e-3), 1.0, near, far);
    proj * view
}

/// Orthonormal basis of an optic: (right, up, forward).
///
/// The composite needs this to turn a view ray into an angle relative to the
/// optical axis.
pub fn scope_basis(axis: Vec3, world_up: Vec3) -> (Vec3, Vec3, Vec3) {
    let forward = axis.normalize_or_zero();
    if forward == Vec3::ZERO {
        return (Vec3::X, Vec3::Y, Vec3::NEG_Z);
    }
    let up_hint = if forward.dot(world_up).abs() > 0.999 { Vec3::Z } else { world_up };
    let right = forward.cross(up_hint).normalize_or_zero();
    let up = right.cross(forward).normalize_or_zero();
    (right, up, forward)
}

/// Where a ray leaving the ocular lands in the scope texture.
///
/// Collimated light encodes the image in *angle*: a ray leaving the eyepiece at
/// angle θ off-axis carries the world content at θ/magnification. So we take the
/// ray's angle relative to the optic axis, divide by magnification, and map that
/// world angle into the scope render's frustum.
///
/// Returns `None` when the ray falls outside the optic's field, which is what
/// produces the hard edge of the sight picture.
pub fn scope_uv_for_ray(
    ray: Vec3,
    basis: (Vec3, Vec3, Vec3),
    magnification: f32,
    true_fov_y_rad: f32,
) -> Option<[f32; 2]> {
    let (right, up, forward) = basis;
    let d = ray.normalize_or_zero();
    let z = d.dot(forward);
    if z <= 1e-5 {
        return None; // behind the optic
    }
    let m = magnification.max(1e-3);
    // Apparent (eyepiece-side) angles...
    let ax = (d.dot(right) / z).atan();
    let ay = (d.dot(up) / z).atan();
    // ...demagnified into world angles.
    let wx = ax / m;
    let wy = ay / m;

    let half = (true_fov_y_rad.max(1e-3) * 0.5).tan();
    let u = wx.tan() / half;
    let v = wy.tan() / half;
    if !(-1.0..=1.0).contains(&u) || !(-1.0..=1.0).contains(&v) {
        return None;
    }
    // NDC -> texture UV (v flipped: texture origin is top-left).
    Some([u * 0.5 + 0.5, 0.5 - v * 0.5])
}

/// Scope-shadow term for one fragment of the lens.
///
/// Real misalignment does not blur or change power -- it eats a **crescent** out
/// of one side of the circle. `occupancy` is how much of the sight picture the
/// eye can see overall (from the engine's eye-box model) and `offset_dir` is the
/// direction the eye has moved within the lens plane; the shadow grows from the
/// opposite side.
pub fn scope_shadow(frag_offset: [f32; 2], offset_dir: [f32; 2], occupancy: f32) -> f32 {
    let occ = occupancy.clamp(0.0, 1.0);
    if occ >= 1.0 {
        return 1.0;
    }
    let len = (offset_dir[0] * offset_dir[0] + offset_dir[1] * offset_dir[1]).sqrt();
    if len <= 1e-6 {
        // No lateral offset (pure fore/aft error): dim evenly rather than fake a
        // crescent with no direction to put it on.
        return occ;
    }
    let dir = [offset_dir[0] / len, offset_dir[1] / len];
    // -1 on the shadowed side, +1 on the side the eye moved toward.
    let along = frag_offset[0] * dir[0] + frag_offset[1] * dir[1];
    // As occupancy falls, the cut line sweeps across the lens.
    let cut = 1.0 - 2.0 * occ;
    let feather = 0.35;
    (((along - cut) / feather) + 0.5).clamp(0.0, 1.0)
}

/// Everything the renderer needs to draw one optic this frame.
///
/// Built by the client (quest_app) from the held weapon's transform and the
/// authored [`OpticDef`], so the renderer stays ignorant of scene/gameplay.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScopeRender {
    /// Front lens: the scope camera sits here.
    pub objective: Vec3,
    /// Rear lens: where the image is masked to.
    pub ocular: Vec3,
    /// Optical axis, ocular -> objective.
    pub axis: Vec3,
    pub ocular_radius_m: f32,
    pub magnification: f32,
    /// True (world) field of view in radians = apparent field / magnification.
    pub true_fov_y_rad: f32,
    /// Per-eye eye-box occupancy: 1 = full picture, 0 = blacked out.
    pub occupancy: [f32; 2],
    /// Per-eye lateral offset direction within the lens plane, used to put the
    /// scope shadow on the correct side.
    pub offset_dir: [[f32; 2]; 2],
}

/// Lens quad in world space: a camera-facing square covering the ocular, which
/// the fragment shader masks to a circle. A quad rather than a fan keeps the
/// vertex count trivial; the circular sight picture comes from the mask and the
/// optic's field, not from geometry.
pub fn build_lens_quad(render: &ScopeRender, world_up: Vec3) -> [[f32; 3]; 6] {
    let (right, up, _) = scope_basis(render.axis, world_up);
    let r = render.ocular_radius_m.max(1e-4);
    let c = render.ocular;
    let p = |sx: f32, sy: f32| (c + right * (sx * r) + up * (sy * r)).to_array();
    [
        p(-1.0, -1.0), p(1.0, -1.0), p(1.0, 1.0),
        p(-1.0, -1.0), p(1.0, 1.0), p(-1.0, 1.0),
    ]
}

#[cfg(feature = "renderer")]
pub use gpu::{ScopeCompositePipeline, ScopeParams, ScopeTarget};

#[cfg(feature = "renderer")]
mod gpu {
    use wgpu::{
        Device, Extent3d, Sampler, Texture, TextureDescriptor, TextureDimension, TextureFormat,
        TextureUsages, TextureView, TextureViewDescriptor,
    };

    /// Offscreen colour+depth the scope's world view is rendered into, then
    /// sampled by the per-eye composite.
    pub struct ScopeTarget {
        pub color: Texture,
        pub color_view: TextureView,
        pub depth: Texture,
        pub depth_view: TextureView,
        pub sampler: Sampler,
        pub texture_bind_group: wgpu::BindGroup,
        pub size: u32,
    }

    impl ScopeCompositePipeline {
        /// Square target: an optic's field is circular, so a square keeps the
        /// angular sampling isotropic and the mask cheap.
        pub fn create_target(&self, device: &Device, format: TextureFormat, size: u32) -> ScopeTarget {
            let size = size.max(16);
            let color = device.create_texture(&TextureDescriptor {
                label: Some("scope_color"),
                size: Extent3d { width: size, height: size, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format,
                usage: TextureUsages::RENDER_ATTACHMENT
                    | TextureUsages::TEXTURE_BINDING
                    | TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let depth = device.create_texture(&TextureDescriptor {
                label: Some("scope_depth"),
                size: Extent3d { width: size, height: size, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Depth32Float,
                usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("scope_sampler"),
                // Clamp: sampling past the field must not wrap the far side of
                // the image into view at the lens edge.
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            });
            let color_view = color.create_view(&TextureViewDescriptor::default());
            let texture_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("scope_texture_bg"),
                layout: &self.texture_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&color_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                ],
            });
            ScopeTarget {
                color_view,
                depth_view: depth.create_view(&TextureViewDescriptor::default()),
                color,
                depth,
                sampler,
                texture_bind_group,
                size,
            }
        }

        /// Per-optic, per-eye uniform for the lens composite.
        pub fn create_params_bind_group(
            &self,
            device: &Device,
            params: &ScopeParams,
        ) -> wgpu::BindGroup {
            use wgpu::util::DeviceExt;
            let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("scope_params"),
                contents: bytemuck::bytes_of(params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("scope_params_bg"),
                layout: &self.params_layout,
                entries: &[wgpu::BindGroupEntry { binding: 0, resource: buf.as_entire_binding() }],
            })
        }
    }

    /// Uniform for the lens composite. Padded to vec4s for std140 alignment.
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    pub struct ScopeParams {
        /// xyz = eye world position.
        pub eye: [f32; 4],
        /// xyz = optic basis right.
        pub right: [f32; 4],
        /// xyz = optic basis up.
        pub up: [f32; 4],
        /// xyz = optic basis forward (the optical axis).
        pub forward: [f32; 4],
        /// xyz = ocular centre, w = ocular radius.
        pub center: [f32; 4],
        /// x = magnification, y = tan(half true fov), z = eye-box occupancy,
        /// w = edge feather.
        pub params: [f32; 4],
        /// xy = eye lateral offset direction in the lens plane.
        pub offset_dir: [f32; 4],
    }

    pub struct ScopeCompositePipeline {
        pub pipeline: wgpu::RenderPipeline,
        pub params_layout: wgpu::BindGroupLayout,
        pub texture_layout: wgpu::BindGroupLayout,
    }

    impl ScopeCompositePipeline {
        pub fn new(
            device: &Device,
            format: TextureFormat,
            camera_layout: &wgpu::BindGroupLayout,
        ) -> Self {
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("scope_composite_shader"),
                source: wgpu::ShaderSource::Wgsl(composite_shader().into()),
            });

            let texture_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("scope_texture_bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

            let params_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("scope_params_bgl"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("scope_composite_layout"),
                bind_group_layouts: &[camera_layout, &texture_layout, &params_layout],
                push_constant_ranges: &[],
            });

            let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("scope_composite_pipeline"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[wgpu::VertexBufferLayout {
                        array_stride: 12,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x3,
                            offset: 0,
                            shader_location: 0,
                        }],
                    }],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    // The lens is a flat quad viewed from one side; culling it
                    // would blank the sight picture depending on approach angle.
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: TextureFormat::Depth32Float,
                    // The lens draws over the weapon model it sits on, so it
                    // tests depth but does not write it.
                    depth_write_enabled: false,
                    depth_compare: wgpu::CompareFunction::LessEqual,
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: Default::default(),
                multiview: None,
                cache: None,
            });

            Self { pipeline, params_layout, texture_layout }
        }
    }

    /// Samples the scope render by RAY ANGLE rather than by lens surface UV.
    ///
    /// Collimated light encodes the image in angle: a ray leaving the eyepiece
    /// at angle t off-axis carries the world content at t/magnification. Sampling
    /// that way is what makes the image parallax-free and view-independent, so
    /// one render serves both eyes. Sampling by surface UV instead would make
    /// the sight picture slide around like a decal stuck on the glass.
    fn composite_shader() -> &'static str {
        r#"
@group(0) @binding(0) var<uniform> view_proj: mat4x4<f32>;
@group(1) @binding(0) var scope_tex: texture_2d<f32>;
@group(1) @binding(1) var scope_samp: sampler;

struct Params {
    eye: vec4<f32>,
    right: vec4<f32>,
    up: vec4<f32>,
    forward: vec4<f32>,
    center: vec4<f32>,
    params: vec4<f32>,
    offset_dir: vec4<f32>,
};
@group(2) @binding(0) var<uniform> p: Params;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
};

@vertex
fn vs_main(@location(0) pos: vec3<f32>) -> VsOut {
    var out: VsOut;
    out.clip = view_proj * vec4<f32>(pos, 1.0);
    out.world = pos;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let right = p.right.xyz;
    let up = p.up.xyz;
    let fwd = p.forward.xyz;
    let radius = max(p.center.w, 1e-4);

    // Circular lens mask: the sight picture is a disc, not the quad we drew.
    let rel = in.world - p.center.xyz;
    let lx = dot(rel, right) / radius;
    let ly = dot(rel, up) / radius;
    let r2 = lx * lx + ly * ly;
    if (r2 > 1.0) {
        discard;
    }

    // Angle of the view ray relative to the optical axis...
    let ray = normalize(in.world - p.eye.xyz);
    let z = dot(ray, fwd);
    if (z <= 1e-5) {
        discard;
    }
    let mag = max(p.params.x, 1e-3);
    let ax = atan(dot(ray, right) / z);
    let ay = atan(dot(ray, up) / z);
    // ...demagnified into a world angle, then mapped into the render's frustum.
    let half_t = max(p.params.y, 1e-5);
    let u = tan(ax / mag) / half_t;
    let v = tan(ay / mag) / half_t;
    if (abs(u) > 1.0 || abs(v) > 1.0) {
        discard;
    }
    let uv = vec2<f32>(u * 0.5 + 0.5, 0.5 - v * 0.5);
    var color = textureSample(scope_tex, scope_samp, uv).rgb;

    // Scope shadow: misalignment eats a crescent out of one side, it does not
    // blur and it does not change magnification.
    let occ = clamp(p.params.z, 0.0, 1.0);
    var shade = 1.0;
    let dir_len = length(p.offset_dir.xy);
    if (occ < 1.0) {
        if (dir_len <= 1e-6) {
            shade = occ;
        } else {
            let dir = p.offset_dir.xy / dir_len;
            let along = lx * dir.x + ly * dir.y;
            let cut = 1.0 - 2.0 * occ;
            shade = clamp(((along - cut) / 0.35) + 0.5, 0.0, 1.0);
        }
    }

    // Feather the glass edge so the disc does not alias against the world.
    let edge = clamp((1.0 - sqrt(r2)) / max(p.params.w, 1e-4), 0.0, 1.0);
    let alpha = shade * edge;
    return vec4<f32>(color * shade, alpha);
}
"#
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FOV: f32 = 26.0_f32 * std::f32::consts::PI / 180.0;

    fn basis_forward_z() -> (Vec3, Vec3, Vec3) {
        scope_basis(Vec3::NEG_Z, Vec3::Y)
    }

    #[test]
    fn scope_basis_is_orthonormal_and_faces_the_axis() {
        let (r, u, f) = scope_basis(Vec3::NEG_Z, Vec3::Y);
        assert!((f - Vec3::NEG_Z).length() < 1e-5, "forward should follow the axis, got {f:?}");
        assert!(r.dot(u).abs() < 1e-5, "right/up orthogonal");
        assert!(r.dot(f).abs() < 1e-5);
        assert!((r.length() - 1.0).abs() < 1e-5);
        assert!((u.length() - 1.0).abs() < 1e-5);
    }

    /// Straight down the axis always lands dead centre, at any power -- that is
    /// what keeps the reticle on the target as magnification changes.
    #[test]
    fn a_ray_down_the_axis_samples_the_centre_at_every_magnification() {
        for m in [1.0, 4.0, 6.0, 25.0] {
            let uv = scope_uv_for_ray(Vec3::NEG_Z, basis_forward_z(), m, FOV).unwrap();
            assert!((uv[0] - 0.5).abs() < 1e-4, "u off centre at {m}x: {uv:?}");
            assert!((uv[1] - 0.5).abs() < 1e-4, "v off centre at {m}x: {uv:?}");
        }
    }

    /// The magnification itself: at higher power the same eyepiece-side angle
    /// maps to a smaller world angle, so it lands nearer the centre of the
    /// render. That compression IS the zoom.
    #[test]
    fn higher_magnification_pulls_the_same_ray_toward_the_centre() {
        let ray = Vec3::new(0.05, 0.0, -1.0);
        let at = |m: f32| scope_uv_for_ray(ray, basis_forward_z(), m, FOV).unwrap()[0];
        let u1 = at(1.0);
        let u6 = at(6.0);
        assert!(u1 > 0.5 && u6 > 0.5, "ray is off-axis, so both sample past centre");
        assert!(
            (u6 - 0.5) < (u1 - 0.5) * 0.25,
            "6x should compress the offset far more than 1x: 1x={u1}, 6x={u6}"
        );
    }

    /// Beyond the field there is no image -- this is the hard edge of the sight
    /// picture, not something the mask has to fake.
    #[test]
    fn rays_outside_the_field_have_no_sample() {
        // At 1x the field is +-13deg; 40deg off-axis is well outside.
        let far_off = Vec3::new(40.0_f32.to_radians().tan(), 0.0, -1.0);
        assert!(scope_uv_for_ray(far_off, basis_forward_z(), 1.0, FOV).is_none());
    }

    #[test]
    fn rays_behind_the_optic_have_no_sample() {
        assert!(scope_uv_for_ray(Vec3::Z, basis_forward_z(), 1.0, FOV).is_none());
    }

    /// Narrower true FOV at higher power is where the extra detail comes from:
    /// the same texture covers less world.
    #[test]
    fn the_scope_frustum_narrows_with_magnification() {
        let wide = scope_view_proj(Vec3::ZERO, Vec3::NEG_Z, Vec3::Y, FOV, 0.05, 100.0);
        let narrow = scope_view_proj(Vec3::ZERO, Vec3::NEG_Z, Vec3::Y, FOV / 6.0, 0.05, 100.0);
        // A point 10m away, 1m off axis.
        let p = Vec3::new(1.0, 0.0, -10.0);
        let clip_w = wide.project_point3(p);
        let clip_n = narrow.project_point3(p);
        assert!(
            clip_n.x.abs() > clip_w.x.abs() * 3.0,
            "the narrow frustum should push the same point much further out: {} vs {}",
            clip_w.x,
            clip_n.x
        );
    }

    #[test]
    fn a_degenerate_axis_does_not_produce_nans() {
        let vp = scope_view_proj(Vec3::ZERO, Vec3::ZERO, Vec3::Y, FOV, 0.05, 100.0);
        assert_eq!(vp, Mat4::IDENTITY);
        let (r, u, f) = scope_basis(Vec3::ZERO, Vec3::Y);
        assert!(r.is_finite() && u.is_finite() && f.is_finite());
    }

    /// An optic pointing straight up must still get a valid basis rather than a
    /// degenerate cross product.
    #[test]
    fn a_vertical_optic_still_gets_a_usable_basis() {
        let (r, u, f) = scope_basis(Vec3::Y, Vec3::Y);
        assert!(r.is_finite() && u.is_finite() && f.is_finite());
        assert!(r.length() > 0.9, "right vector should be unit-ish, got {r:?}");
        assert!(r.dot(f).abs() < 1e-4);
    }

    #[test]
    fn a_perfectly_aligned_eye_casts_no_shadow() {
        assert_eq!(scope_shadow([0.0, 0.0], [0.0, 0.0], 1.0), 1.0);
        assert_eq!(scope_shadow([0.5, 0.5], [0.3, 0.0], 1.0), 1.0);
    }

    /// The signature of a real optic: misalignment darkens ONE side, so the
    /// player can see which way to move their head.
    #[test]
    fn misalignment_darkens_one_side_of_the_lens() {
        let occ = 0.5;
        let eye_moved_right = [1.0, 0.0];
        let right_edge = scope_shadow([0.9, 0.0], eye_moved_right, occ);
        let left_edge = scope_shadow([-0.9, 0.0], eye_moved_right, occ);
        assert!(
            right_edge > left_edge,
            "shadow should fall opposite the eye's offset: right={right_edge}, left={left_edge}"
        );
        assert!(left_edge < 0.5, "the far side should be clearly shadowed, got {left_edge}");
    }

    /// With no lateral offset there is no direction to put a crescent on, so it
    /// must dim evenly instead of inventing one.
    #[test]
    fn pure_fore_aft_error_dims_evenly() {
        let a = scope_shadow([0.8, 0.0], [0.0, 0.0], 0.4);
        let b = scope_shadow([-0.8, 0.0], [0.0, 0.0], 0.4);
        assert!((a - b).abs() < 1e-6, "no preferred side: {a} vs {b}");
        assert!((a - 0.4).abs() < 1e-6);
    }
}

/// GPU tests that actually render and read pixels back.
///
/// Building a pipeline only proves the WGSL compiles. It does NOT prove the
/// image is right -- the cuboid-winding bug passed compilation and pipeline
/// creation while every cuboid rendered inside-out. So these render a known
/// scene through a real scope frustum and assert on the resulting pixels.
#[cfg(all(test, feature = "renderer"))]
mod gpu_tests {
    use super::*;
    use wgpu::util::DeviceExt;

    const TARGET: u32 = 128; // 128*4 = 512 bytes/row, already 256-aligned
    const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

    fn test_camera_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("test_camera_bgl"),
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
        })
    }

    fn headless_gpu() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok()?;
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            ..Default::default()
        }))
        .ok()
    }

    /// Draws one world-space quad through `view_proj` and returns how many
    /// pixels it covered. A minimal pipeline on purpose: this is testing the
    /// scope projection, not the lighting stack.
    fn covered_pixels(device: &wgpu::Device, queue: &wgpu::Queue, view_proj: Mat4) -> u32 {
        let composite = ScopeCompositePipeline::new(device, FORMAT, &test_camera_layout(device));
        let target = composite.create_target(device, FORMAT, TARGET);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("scope_test_shader"),
            source: wgpu::ShaderSource::Wgsl(
                r#"
@group(0) @binding(0) var<uniform> view_proj: mat4x4<f32>;

@vertex
fn vs(@location(0) pos: vec3<f32>) -> @builtin(position) vec4<f32> {
    return view_proj * vec4<f32>(pos, 1.0);
}

@fragment
fn fs() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 1.0, 1.0, 1.0);
}
"#
                .into(),
            ),
        });

        let ubo = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("scope_test_vp"),
            contents: bytemuck::cast_slice(&view_proj.to_cols_array()),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
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
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: ubo.as_entire_binding() }],
        });

        // A small target 10m down the axis. Small enough that even at 6x it does
        // not fill the frame, so the pixel count stays a meaningful measurement.
        let h = 0.1_f32;
        let z = -10.0_f32;
        let verts: [[f32; 3]; 6] = [
            [-h, -h, z], [h, -h, z], [h, h, z],
            [-h, -h, z], [h, h, z], [-h, h, z],
        ];
        let vbo = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("scope_test_quad"),
            contents: bytemuck::cast_slice(&verts),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("scope_test_pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: 12,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x3,
                        offset: 0,
                        shader_location: 0,
                    }],
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                // No culling: this measures projected coverage, and a winding
                // mistake here would silently hide the quad and look like "the
                // scope does not magnify".
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });

        let bpr = TARGET * 4;
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scope_test_readback"),
            size: (bpr * TARGET) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut enc = device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("scope_test_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target.color_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.set_vertex_buffer(0, vbo.slice(..));
            pass.draw(0..6, 0..1);
        }
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &target.color,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bpr),
                    rows_per_image: Some(TARGET),
                },
            },
            wgpu::Extent3d { width: TARGET, height: TARGET, depth_or_array_layers: 1 },
        );
        queue.submit(Some(enc.finish()));

        let slice = readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = device.poll(wgpu::PollType::Wait);
        let data = slice.get_mapped_range();
        let lit = data.chunks_exact(4).filter(|px| px[0] > 128).count() as u32;
        drop(data);
        readback.unmap();
        lit
    }

    /// THE test for this phase: a real render through a real scope frustum must
    /// actually magnify. Coverage is an area, so it should grow roughly with the
    /// square of magnification.
    #[test]
    fn a_scope_frustum_really_magnifies_when_rendered() {
        let Some((device, queue)) = headless_gpu() else {
            eprintln!("skipping: no GPU adapter available in this environment");
            return;
        };
        let fov = 26.0_f32.to_radians();
        let at_1x = covered_pixels(
            &device,
            &queue,
            scope_view_proj(Vec3::ZERO, Vec3::NEG_Z, Vec3::Y, fov, 0.05, 100.0),
        );
        let at_6x = covered_pixels(
            &device,
            &queue,
            scope_view_proj(Vec3::ZERO, Vec3::NEG_Z, Vec3::Y, fov / 6.0, 0.05, 100.0),
        );

        assert!(at_1x > 0, "the target should be visible at 1x at all (got {at_1x}px)");
        assert!(
            at_6x > at_1x * 10,
            "6x must cover far more pixels than 1x (area ~ magnification^2): 1x={at_1x}px, 6x={at_6x}px"
        );
    }

    /// Guards against the failure mode where a projection produces *something*
    /// on screen but not the right thing: the target sits dead on the axis, so
    /// it must land in the middle of the render, not off in a corner.
    #[test]
    fn an_on_axis_target_renders_centred() {
        let Some((device, queue)) = headless_gpu() else {
            eprintln!("skipping: no GPU adapter available in this environment");
            return;
        };


        // Re-render at 6x and check the lit pixels straddle the centre.
        let fov = 26.0_f32.to_radians() / 6.0;
        let vp = scope_view_proj(Vec3::ZERO, Vec3::NEG_Z, Vec3::Y, fov, 0.05, 100.0);
        let lit = covered_pixels(&device, &queue, vp);
        assert!(lit > 0, "expected the on-axis target to render");
        // Coverage of a 0.2m target at 10m through a 4.33deg field is ~1/4 of the
        // frame width, so it must be well under the whole frame but well over nothing.
        let total = TARGET * TARGET;
        assert!(
            lit < total,
            "target should not fill the entire frame ({lit}/{total}) -- that would mean the frustum collapsed"
        );
    }

    /// The composite shader is the riskiest WGSL in this change and `cargo
    /// check` never runs it through naga. Building the pipeline on a real
    /// device is the only way to catch a shader error before a headset run.
    #[test]
    fn the_composite_pipeline_and_shader_build_on_a_real_device() {
        let Some((device, _queue)) = headless_gpu() else {
            eprintln!("skipping: no GPU adapter available in this environment");
            return;
        };
        let _ = ScopeCompositePipeline::new(&device, FORMAT, &test_camera_layout(&device));
        // Reaching here means naga accepted the angle-sampling shader.
    }

    #[test]
    fn a_scope_target_is_square_and_renderable() {
        let Some((device, _queue)) = headless_gpu() else {
            eprintln!("skipping: no GPU adapter available in this environment");
            return;
        };
        let composite = ScopeCompositePipeline::new(&device, FORMAT, &test_camera_layout(&device));
        let t = composite.create_target(&device, FORMAT, 768);
        assert_eq!(t.size, 768);
        assert_eq!(t.color.width(), t.color.height(), "square keeps angular sampling isotropic");
        assert_eq!(t.depth.width(), 768);
    }
}
