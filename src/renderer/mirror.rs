//! Planar mirror reflection: a second full scene render from a reflected
//! camera angle into an offscreen texture per eye, then a quad in the main
//! pass that samples it using the fragment's own screen position. That
//! last part is an approximation (true per-pixel-correct reflection needs
//! the reflected view-projection baked into the quad's own vertices), valid
//! for a mirror viewed roughly head-on — the common case for "walk up and
//! look in a mirror."

use glam::{Mat3, Mat4, Quat, Vec3, Vec4};
use wgpu::*;

use super::mesh_pipeline::ModelUniform;

/// A flat mirror surface to render a reflection into, in the same
/// render-space coordinates as everything else passed to
/// `render_frame_with_meshes`. `rotation`'s local -Z axis is the mirror's
/// facing/normal direction; `half_size.x`/`.y` are the mirror's half-width
/// and half-height (`.z` is unused).
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

/// One eye's offscreen render target for the mirror's reflected view, plus
/// the bind group used to sample it when drawing the mirror quad itself.
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

        // Same shape as model_layout (one Mat4 uniform), but a distinct
        // layout/binding slot: the mirror's own reflected view-projection,
        // used so the quad can compute pixel-accurate reflection UVs from
        // its own vertices instead of approximating from the real camera's
        // screen position (which is only exactly right viewed head-on).
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
    // Not a builtin — this carries the *mirror's own reflected* camera's
    // clip-space position, perspective-interpolated per fragment, so each
    // pixel samples exactly the texel the reflected render actually put
    // there (pixel-accurate regardless of viewing angle), instead of
    // approximating from this quad's real-camera screen position (which is
    // only exactly right viewed dead-on).
    @location(0) reflected_clip: vec4<f32>,
}

@vertex
fn vs_main(v: VIn) -> VOut {
    var out: VOut;
    let world_pos = model_u.model * vec4<f32>(v.position, 1.0);
    out.clip = camera.view_proj * world_pos;
    // The quad is already nudged in front of the mirror frame's own surface
    // in world space (see quest_app's mirror_surface build), but a fixed
    // world-space offset loses to depth-buffer precision at any real
    // distance (non-linear depth + this projection's huge far:near ratio)
    // and the frame wins the depth test again, showing flat black instead
    // of the reflection. A clip-space bias scaled by w stays proportionally
    // correct regardless of distance — same technique as the wire pipeline.
    out.clip.z -= 0.0001 * out.clip.w;
    out.reflected_clip = reflected.view_proj * world_pos;
    return out;
}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4<f32> {
    let ndc = in.reflected_clip.xy / in.reflected_clip.w;
    // NDC Y points up; texture V points down (origin top-left) — flip it.
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    return textureSample(mirror_tex, mirror_samp, uv);
}
"#
    .to_string()
}

/// Local-space quad in the mirror's own XY plane (Z=0), sized from the
/// mirror's authored half-extents — transformed into place by the model
/// matrix built from the mirror object's own position/rotation.
pub fn build_mirror_quad(half_width: f32, half_height: f32) -> (Vec<MirrorVertex>, Vec<u32>) {
    let verts = vec![
        MirrorVertex {
            position: [-half_width, -half_height, 0.0],
        },
        MirrorVertex {
            position: [half_width, -half_height, 0.0],
        },
        MirrorVertex {
            position: [half_width, half_height, 0.0],
        },
        MirrorVertex {
            position: [-half_width, half_height, 0.0],
        },
    ];
    let idx = vec![0, 1, 2, 0, 2, 3];
    (verts, idx)
}

/// Affine reflection matrix across the plane through `plane_point` with
/// unit normal `plane_normal` — reflecting a camera's world transform by
/// this (`reflected_world = M * world`) gives the correct mirrored camera;
/// reflecting the camera's own *view* matrix by right-multiplying
/// (`reflected_view = view * M`) is equivalent and avoids decomposing/
/// recomposing the camera's rotation by hand, since a reflection is its
/// own inverse.
pub fn reflection_matrix(plane_point: Vec3, plane_normal: Vec3) -> Mat4 {
    let n = plane_normal.normalize();
    let linear = Mat3::from_cols(
        Vec3::X - 2.0 * n.x * n,
        Vec3::Y - 2.0 * n.y * n,
        Vec3::Z - 2.0 * n.z * n,
    );
    let translation = 2.0 * plane_point.dot(n) * n;
    Mat4::from_cols(
        linear.x_axis.extend(0.0),
        linear.y_axis.extend(0.0),
        linear.z_axis.extend(0.0),
        translation.extend(1.0),
    )
}

/// The mirror's world-space plane equation `(a,b,c,d)` such that a point
/// `X` satisfies `dot((a,b,c), X) + d == 0` when it's exactly on the
/// mirror's surface, and `> 0` on the side `plane_normal` points toward
/// (the side a viewer stands on to see their reflection).
pub fn world_plane_equation(plane_point: Vec3, plane_normal: Vec3) -> Vec4 {
    let n = plane_normal.normalize();
    n.extend(-plane_point.dot(n))
}

/// Transforms a plane equation from world space into a camera's own eye
/// space, given that camera's world (eye-to-world) matrix. Planes don't
/// transform the same way points do — they need the inverse-transpose of
/// the point transform, i.e. `transpose(world_to_eye)` here, since
/// `eye_to_world` is exactly `inverse(world_to_eye)`.
/// Takes the camera's own *world* matrix (eye-to-world — for a view matrix
/// `V` this is `V.inverse()`, **not** `V` itself). Planes transform by the
/// inverse-transpose of however points transform: points go
/// `p_eye = V * p_world`, so a plane goes
/// `Π_eye = (V^-1)^T * Π_world = eye_to_world^T * Π_world` — using `V^T`
/// directly (i.e. skipping the inverse) happens to get the plane's *normal*
/// right, since a rigid transform's rotational part is orthogonal
/// (`R^T == R^-1`), but gets its distance-from-origin term wrong, since
/// that isn't true of the translation component — silently clipping
/// geometry that should have been visible.
pub fn plane_to_eye_space(eye_to_world: Mat4, plane_world: Vec4) -> Vec4 {
    eye_to_world.transpose() * plane_world
}

/// Bends a perspective projection matrix's near clip plane to exactly
/// coincide with an arbitrary plane given in the *same eye space* the
/// projection matrix operates in (Eric Lengyel's oblique near-plane
/// clipping technique — see terathon.com/blog/oblique-clipping.html).
/// Used for the mirror's reflected render pass so nothing on the wrong
/// side of the mirror surface (between the reflected camera and the
/// mirror) can ever be rendered into it, regardless of what geometry
/// happens to be nearby — a structural fix rather than relying on the
/// mirror being placed somewhere with enough manual clearance.
///
/// `clip_plane` must satisfy the convention that the camera (at the eye
/// space origin) is strictly on its negative side, i.e. `clip_plane.w < 0`;
/// `plane_to_eye_space` applied to a plane built by `world_plane_equation`
/// with a normal pointing toward the real (non-reflected) camera already
/// satisfies this for the reflected camera.
pub fn oblique_near_clip(proj: Mat4, clip_plane: Vec4) -> Mat4 {
    let m = proj.to_cols_array();
    let sign = |v: f32| if v > 0.0 { 1.0 } else if v < 0.0 { -1.0 } else { 0.0 };

    let q = Vec4::new(
        (sign(clip_plane.x) + m[8]) / m[0],
        (sign(clip_plane.y) + m[9]) / m[5],
        -1.0,
        (1.0 + m[10]) / m[14],
    );

    let c = clip_plane * (2.0 / clip_plane.dot(q));

    let mut m2 = m;
    m2[2] = c.x;
    m2[6] = c.y;
    m2[10] = c.z + 1.0;
    m2[14] = c.w;
    Mat4::from_cols_array(&m2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reflection_matrix_reflects_a_point_across_the_plane() {
        // Mirror at x=2, facing +X: a point at x=0 should reflect to x=4.
        let m = reflection_matrix(Vec3::new(2.0, 0.0, 0.0), Vec3::X);
        let p = m.transform_point3(Vec3::new(0.0, 1.0, 3.0));
        assert!((p.x - 4.0).abs() < 1e-5, "expected x=4.0, got {p:?}");
        assert!((p.y - 1.0).abs() < 1e-5, "y/z should pass through unchanged, got {p:?}");
        assert!((p.z - 3.0).abs() < 1e-5, "y/z should pass through unchanged, got {p:?}");
    }

    #[test]
    fn reflection_matrix_is_its_own_inverse() {
        let m = reflection_matrix(Vec3::new(1.0, 2.0, 3.0), Vec3::new(0.3, 0.7, 0.1));
        let p = Vec3::new(5.0, -2.0, 0.5);
        let reflected_twice = m.transform_point3(m.transform_point3(p));
        assert!(
            p.distance(reflected_twice) < 1e-4,
            "reflecting twice should return the original point: {p:?} vs {reflected_twice:?}"
        );
    }

    #[test]
    fn reflection_matrix_leaves_plane_points_fixed() {
        let plane_point = Vec3::new(0.0, 1.5, -3.0);
        let m = reflection_matrix(plane_point, Vec3::Y);
        let reflected = m.transform_point3(plane_point);
        assert!(
            plane_point.distance(reflected) < 1e-5,
            "a point on the plane should map to itself, got {reflected:?}"
        );
    }

    /// Same perspective matrix shape `Camera::xr_projection` builds (this
    /// crate's actual VR projection), so the test exercises the real matrix
    /// form rather than glam's own (possibly differently-conventioned)
    /// perspective helper.
    fn test_projection(near: f32, far: f32) -> Mat4 {
        let fov_y = 60_f32.to_radians();
        let aspect = 1.0;
        let h_half = (fov_y * 0.5).tan();
        let w_half = h_half * aspect;
        let w = 2.0 * w_half;
        let h = 2.0 * h_half;
        Mat4::from_cols_array(&[
            2.0 / w,
            0.0,
            0.0,
            0.0,
            0.0,
            2.0 / h,
            0.0,
            0.0,
            0.0,
            0.0,
            -(far + near) / (far - near),
            -1.0,
            0.0,
            0.0,
            -(2.0 * far * near) / (far - near),
            0.0,
        ])
    }

    fn ndc_z(proj: Mat4, eye_space_point: Vec3) -> f32 {
        let clip = proj * eye_space_point.extend(1.0);
        clip.z / clip.w
    }

    #[test]
    fn oblique_near_clip_preserves_the_boundary_at_the_original_near_plane() {
        let near = 0.1;
        let proj = test_projection(near, 100.0);

        // A point straight ahead exactly at the existing near plane.
        let p_near = Vec3::new(0.0, 0.0, -near);
        let ndc_before = ndc_z(proj, p_near);

        // Clip plane coincides with that same near plane: normal points
        // away from the eye (down -Z, the view direction), eye (at the
        // origin) must land on the plane's negative side per the
        // algorithm's required convention.
        let clip_plane = Vec4::new(0.0, 0.0, -1.0, -near);
        assert!(clip_plane.w < 0.0, "camera must be on the negative side");

        let clipped = oblique_near_clip(proj, clip_plane);
        let ndc_after = ndc_z(clipped, p_near);

        assert!(
            (ndc_before - ndc_after).abs() < 1e-4,
            "a plane coinciding with the existing near plane shouldn't move \
             its boundary: before={ndc_before}, after={ndc_after}"
        );
    }

    #[test]
    fn oblique_near_clip_moves_the_boundary_to_an_arbitrary_plane() {
        let near = 0.1;
        let proj = test_projection(near, 100.0);
        let boundary_ndc_z = ndc_z(proj, Vec3::new(0.0, 0.0, -near));

        // A plane much farther out than the real near plane, still facing
        // the camera (eye at origin on its negative side).
        let plane_distance = 5.0;
        let clip_plane = Vec4::new(0.0, 0.0, -1.0, -plane_distance);
        assert!(clip_plane.w < 0.0);

        let clipped = oblique_near_clip(proj, clip_plane);

        // A point sitting exactly on that far-out plane should now land on
        // the same boundary the *original* near plane used to.
        let p_on_plane = Vec3::new(1.5, -0.7, -plane_distance);
        let ndc_on_plane = ndc_z(clipped, p_on_plane);
        assert!(
            (ndc_on_plane - boundary_ndc_z).abs() < 1e-3,
            "a point on the new clip plane should map to the near-plane \
             boundary ({boundary_ndc_z}), got {ndc_on_plane}"
        );

        // A point on the camera's side of that plane (closer than
        // plane_distance) should now fall outside the boundary — i.e. on
        // the "clipped away" side.
        let p_before_plane = Vec3::new(0.0, 0.0, -1.0);
        let ndc_before_plane = ndc_z(clipped, p_before_plane);
        assert!(
            ndc_before_plane < boundary_ndc_z,
            "a point nearer than the new clip plane should fall outside the \
             visible range (boundary={boundary_ndc_z}), got {ndc_before_plane}"
        );
    }

    #[test]
    fn world_plane_equation_matches_reflection_plane_convention() {
        let point = Vec3::new(2.0, 0.0, 0.0);
        let normal = Vec3::X;
        let plane = world_plane_equation(point, normal);
        // The plane's own point should satisfy the equation exactly.
        let value = plane.x * point.x + plane.y * point.y + plane.z * point.z + plane.w;
        assert!(value.abs() < 1e-5, "point on the plane should satisfy its own equation, got {value}");
        // A point on the normal's side should be positive.
        let ahead = point + normal * 3.0;
        let ahead_value = plane.x * ahead.x + plane.y * ahead.y + plane.z * ahead.z + plane.w;
        assert!(ahead_value > 0.0, "point on the normal side should be positive, got {ahead_value}");
    }

    /// Uses a view matrix with *both* rotation and translation — a
    /// rotation-only test wouldn't have caught the bug this guards against
    /// (using `view.transpose()` instead of `view.inverse().transpose()`
    /// happens to still work for the plane's normal, since a rigid
    /// transform's rotational part is orthogonal, but silently produces
    /// the wrong distance-from-origin term).
    #[test]
    fn plane_to_eye_space_matches_direct_point_transform() {
        let rot = Quat::from_rotation_y(35_f32.to_radians()) * Quat::from_rotation_x(10_f32.to_radians());
        let pos = Vec3::new(3.0, 1.5, -2.0);
        let camera_world = Mat4::from_rotation_translation(rot, pos);
        let view = camera_world.inverse();

        let plane_point = Vec3::new(-1.0, 2.0, 0.5);
        let plane_normal = Vec3::new(0.2, 0.9, 0.1);
        let plane_world = world_plane_equation(plane_point, plane_normal);

        let plane_eye = plane_to_eye_space(camera_world, plane_world);

        // A point on the world-space plane, transformed into eye space by
        // the *same* view matrix used to derive plane_eye, must satisfy
        // plane_eye's equation — this is the actual invariant that matters,
        // not just "the normal looks right."
        let p_world = plane_point + Vec3::new(4.0, -3.0, 7.0).reject_from(plane_normal.normalize());
        let p_eye = view.transform_point3(p_world);
        let value = plane_eye.x * p_eye.x + plane_eye.y * p_eye.y + plane_eye.z * p_eye.z + plane_eye.w;
        assert!(
            value.abs() < 1e-3,
            "a point on the world plane should satisfy the eye-space plane equation after \
             transforming by the same view matrix, got {value}"
        );
    }
}
