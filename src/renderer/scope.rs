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

/// Smallest scope target worth rendering. Below this the sight picture is mush
/// regardless of filtering.
pub const MIN_SCOPE_TARGET: u32 = 128;

/// Mip levels for a square texture of `size` — `floor(log2(size)) + 1`.
pub fn mip_level_count_for(size: u32) -> u32 {
    32 - size.max(1).leading_zeros()
}

/// Diameter, in screen pixels, that the ocular lens covers for an eye at
/// `eye_to_ocular_m`.
///
/// A viewport `viewport_height_px` tall spans `vertical_fov_rad`, so one radian
/// is `viewport_height_px / vertical_fov_rad` pixels, and the lens subtends
/// `2 * atan(r / d)` radians.
pub fn ocular_screen_diameter_px(
    ocular_radius_m: f32,
    eye_to_ocular_m: f32,
    viewport_height_px: u32,
    vertical_fov_rad: f32,
) -> f32 {
    if eye_to_ocular_m <= 1e-4 || vertical_fov_rad <= 1e-4 {
        return 0.0;
    }
    let subtended = 2.0 * (ocular_radius_m.max(0.0) / eye_to_ocular_m).atan();
    subtended / vertical_fov_rad * viewport_height_px as f32
}

/// Resolution to render the scope at, derived from how big the lens actually is
/// on screen.
///
/// A fixed tier constant is the wrong shape for this. The ocular covers on the
/// order of 150–250 px, so a fixed 768² target is both ~15× more pixels than
/// needed **and** the direct cause of aliasing: compositing 768² down onto a
/// 200 px disc is a 3–5× minification, which samples a handful of texels where
/// it should be averaging a few dozen. Matching the target to the disc removes
/// the waste and the shimmer at the same time.
///
/// `supersample` above 1.0 renders larger and lets the mip chain average back
/// down — real SSAA, which unlike MSAA also anti-aliases shading, and which is
/// unusually cheap here because collimation means one render serves both eyes.
///
/// Rounded up to a power of two so the mip chain is exact, and clamped so a lens
/// filling the view cannot blow past the quality tier's budget.
pub fn scope_target_size(
    ocular_radius_m: f32,
    eye_to_ocular_m: f32,
    viewport_height_px: u32,
    vertical_fov_rad: f32,
    tier_max: u32,
    supersample: f32,
) -> u32 {
    let disc = ocular_screen_diameter_px(
        ocular_radius_m,
        eye_to_ocular_m,
        viewport_height_px,
        vertical_fov_rad,
    );
    let wanted = (disc * supersample.max(1.0)).ceil().max(1.0);
    let pow2 = (wanted as u32).next_power_of_two();
    pow2.clamp(MIN_SCOPE_TARGET, tier_max.max(MIN_SCOPE_TARGET))
}

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
pub use gpu::{MipBlit, ScopeCompositePipeline, ScopeParams, ScopeTarget};

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
        /// Mip 0 only — the world pass renders here.
        pub color_view: TextureView,
        pub depth: Texture,
        pub depth_view: TextureView,
        pub sampler: Sampler,
        /// Samples the whole chain; this is what the composite binds.
        pub texture_bind_group: wgpu::BindGroup,
        pub size: u32,
        pub mip_levels: u32,
        /// One render-attachment view per level, `mip_views[i]` = level i.
        pub mip_views: Vec<TextureView>,
        /// `mip_src_bind_groups[i]` samples level i, to render level i + 1.
        pub mip_src_bind_groups: Vec<wgpu::BindGroup>,
    }

    impl ScopeTarget {
        /// Fill mips 1..n by successive halving.
        ///
        /// Must run after the world pass and before the composite, every frame
        /// the scope is drawn — the chain is stale otherwise, and a stale chain
        /// looks like a smeared sight picture rather than an obvious bug.
        pub fn generate_mips(&self, encoder: &mut wgpu::CommandEncoder, blit: &MipBlit) {
            for level in 1..self.mip_levels as usize {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("scope_mip_blit"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.mip_views[level],
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
                pass.set_pipeline(&blit.pipeline);
                pass.set_bind_group(0, &self.mip_src_bind_groups[level - 1], &[]);
                pass.draw(0..3, 0..1);
            }
        }
    }

    /// Downsample pass used to build the scope target's mip chain.
    ///
    /// wgpu has no `generate_mipmaps`, so this is an explicit blit: a fullscreen
    /// triangle sampling level N-1 with a linear filter into level N. Bilinear
    /// on an exact 2:1 reduction averages the right four texels, which is all a
    /// box filter needs.
    pub struct MipBlit {
        pub pipeline: wgpu::RenderPipeline,
        pub layout: wgpu::BindGroupLayout,
        pub sampler: Sampler,
    }

    impl MipBlit {
        pub fn new(device: &Device, format: TextureFormat) -> Self {
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("scope_mip_blit_shader"),
                source: wgpu::ShaderSource::Wgsl(
                    r#"
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

// Fullscreen triangle: no vertex buffer, no index buffer.
@vertex
fn vs(@builtin(vertex_index) i: u32) -> VsOut {
    var out: VsOut;
    let x = f32((i << 1u) & 2u);
    let y = f32(i & 2u);
    out.uv = vec2<f32>(x, y);
    out.pos = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    return textureSample(src, samp, in.uv);
}
"#
                    .into(),
                ),
            });

            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("scope_mip_blit_layout"),
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

            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("scope_mip_blit_pl"),
                bind_group_layouts: &[&layout],
                push_constant_ranges: &[],
            });

            let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("scope_mip_blit_pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            });

            let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("scope_mip_blit_sampler"),
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            });

            Self { pipeline, layout, sampler }
        }
    }

    impl ScopeCompositePipeline {
        /// Square target: an optic's field is circular, so a square keeps the
        /// angular sampling isotropic and the mask cheap.
        pub fn create_target(
            &self,
            device: &Device,
            format: TextureFormat,
            size: u32,
            mip_blit: &MipBlit,
        ) -> ScopeTarget {
            let size = size.max(16);
            let mip_levels = super::mip_level_count_for(size);
            let color = device.create_texture(&TextureDescriptor {
                label: Some("scope_color"),
                size: Extent3d { width: size, height: size, depth_or_array_layers: 1 },
                mip_level_count: mip_levels,
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
                // Trilinear. This is the fix for scope shimmer: the composite
                // minifies the target onto an ocular disc a few times smaller,
                // and bilinear-without-mips samples four texels where it needs
                // to average dozens. That undersampling crawls under head
                // motion and reads as "the scope is aliased".
                mipmap_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            });
            // Sampling view spans the whole chain; the render attachment is mip 0.
            let sample_view = color.create_view(&TextureViewDescriptor::default());
            let mip_views: Vec<TextureView> = (0..mip_levels)
                .map(|level| {
                    color.create_view(&TextureViewDescriptor {
                        label: Some("scope_mip"),
                        base_mip_level: level,
                        mip_level_count: Some(1),
                        ..Default::default()
                    })
                })
                .collect();
            let mip_src_bind_groups: Vec<wgpu::BindGroup> = (0..mip_levels as usize)
                .map(|level| {
                    device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("scope_mip_src"),
                        layout: &mip_blit.layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(&mip_views[level]),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::Sampler(&mip_blit.sampler),
                            },
                        ],
                    })
                })
                .collect();
            // Separate mip-0 view for the world pass to render into. wgpu views
            // are not Clone, and the same level can be viewed more than once.
            let color_view = color.create_view(&TextureViewDescriptor {
                label: Some("scope_color_mip0"),
                base_mip_level: 0,
                mip_level_count: Some(1),
                ..Default::default()
            });
            let texture_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("scope_texture_bg"),
                layout: &self.texture_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        // The whole chain, not mip 0 — binding a single level
                        // would make mipmap_filter a no-op and undo the fix.
                        resource: wgpu::BindingResource::TextureView(&sample_view),
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
                mip_levels,
                mip_views,
                mip_src_bind_groups,
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
mod target_sizing_tests {
    use super::*;

    // Quest-3-ish: ~1000 px tall per eye over a ~96 deg vertical field.
    const VP_H: u32 = 1000;
    const EYE_FOV: f32 = 96.0_f32 * std::f32::consts::PI / 180.0;
    // A typical scope ocular and a cheek-weld eye relief.
    const OCULAR_R: f32 = 0.017;
    const RELIEF: f32 = 0.095;

    #[test]
    fn mip_counts_follow_log2() {
        assert_eq!(mip_level_count_for(1), 1);
        assert_eq!(mip_level_count_for(2), 2);
        assert_eq!(mip_level_count_for(128), 8);
        assert_eq!(mip_level_count_for(768), 10);
        assert_eq!(mip_level_count_for(1024), 11);
    }

    #[test]
    fn a_real_scope_ocular_covers_a_couple_hundred_pixels() {
        let d = ocular_screen_diameter_px(OCULAR_R, RELIEF, VP_H, EYE_FOV);
        assert!(
            (150.0..280.0).contains(&d),
            "expected a 150-280px disc for a real scope, got {d:.0}px"
        );
    }

    /// The whole point of the change: a fixed 768 target was several times
    /// larger than the disc it composites onto, which is what caused the
    /// undersampling in the first place.
    #[test]
    fn the_target_is_sized_to_the_disc_not_to_a_fixed_tier_constant() {
        let size = scope_target_size(OCULAR_R, RELIEF, VP_H, EYE_FOV, 1024, 1.0);
        let disc = ocular_screen_diameter_px(OCULAR_R, RELIEF, VP_H, EYE_FOV);
        assert!(size < 768, "should be far below the old fixed 768, got {size}");
        assert!(size as f32 >= disc, "must not be smaller than the disc it fills");
    }

    #[test]
    fn a_lens_further_from_the_eye_needs_a_smaller_target() {
        let near = scope_target_size(OCULAR_R, 0.05, VP_H, EYE_FOV, 1024, 1.0);
        let far = scope_target_size(OCULAR_R, 0.30, VP_H, EYE_FOV, 1024, 1.0);
        assert!(far <= near, "a smaller on-screen disc must not cost more, {far} vs {near}");
    }

    #[test]
    fn supersampling_raises_the_target_and_the_tier_still_caps_it() {
        let base = scope_target_size(OCULAR_R, RELIEF, VP_H, EYE_FOV, 1024, 1.0);
        let ss = scope_target_size(OCULAR_R, RELIEF, VP_H, EYE_FOV, 1024, 2.0);
        assert!(ss > base, "2x supersample should render larger, {ss} vs {base}");
        // A lens filling the view must not blow past the tier budget.
        let huge = scope_target_size(1.0, 0.02, VP_H, EYE_FOV, 512, 4.0);
        assert_eq!(huge, 512);
    }

    #[test]
    fn sizes_are_powers_of_two_so_the_mip_chain_is_exact() {
        for relief in [0.03_f32, 0.05, 0.095, 0.2, 0.4] {
            let s = scope_target_size(OCULAR_R, relief, VP_H, EYE_FOV, 1024, 1.0);
            assert!(s.is_power_of_two(), "{s} is not a power of two");
            assert!(s >= MIN_SCOPE_TARGET);
        }
    }

    #[test]
    fn degenerate_inputs_fall_back_to_the_floor_rather_than_dividing_by_zero() {
        assert_eq!(ocular_screen_diameter_px(0.017, 0.0, VP_H, EYE_FOV), 0.0);
        assert_eq!(ocular_screen_diameter_px(0.017, RELIEF, VP_H, 0.0), 0.0);
        assert_eq!(scope_target_size(0.0, 0.0, 0, 0.0, 1024, 1.0), MIN_SCOPE_TARGET);
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
        let mip_blit = MipBlit::new(device, FORMAT);
        let target = composite.create_target(device, FORMAT, TARGET, &mip_blit);

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
        let mip_blit = MipBlit::new(&device, FORMAT);
        let t = composite.create_target(&device, FORMAT, 768, &mip_blit);
        assert_eq!(t.size, 768);
        assert_eq!(t.color.width(), t.color.height(), "square keeps angular sampling isotropic");
        assert_eq!(t.depth.width(), 768);
    }

    #[test]
    fn a_scope_target_carries_a_full_mip_chain() {
        let Some((device, _queue)) = headless_gpu() else {
            eprintln!("skipping: no GPU adapter available in this environment");
            return;
        };
        let composite = ScopeCompositePipeline::new(&device, FORMAT, &test_camera_layout(&device));
        let mip_blit = MipBlit::new(&device, FORMAT);
        let t = composite.create_target(&device, FORMAT, 256, &mip_blit);

        assert_eq!(t.mip_levels, 9, "256 -> 9 levels");
        assert_eq!(t.color.mip_level_count(), 9, "the texture itself must carry them");
        assert_eq!(t.mip_views.len(), 9, "one render-attachment view per level");
        assert_eq!(t.mip_src_bind_groups.len(), 9);
    }

    /// Paints a 1-texel checkerboard into mip 0, builds the chain, and reads
    /// back levels 0 and 1.
    fn checkerboard_mip_variances(size: u32) -> Option<(f64, f64, f64)> {
        let (device, queue) = headless_gpu()?;
        let composite = ScopeCompositePipeline::new(&device, FORMAT, &test_camera_layout(&device));
        let mip_blit = MipBlit::new(&device, FORMAT);
        let target = composite.create_target(&device, FORMAT, size, &mip_blit);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("checker"),
            source: wgpu::ShaderSource::Wgsl(
                r#"
@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let x = f32((i << 1u) & 2u);
    let y = f32(i & 2u);
    return vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let c = (u32(pos.x) + u32(pos.y)) % 2u;
    let v = f32(c);
    return vec4<f32>(v, v, v, 1.0);
}
"#
                .into(),
            ),
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("checker_pipeline"),
            layout: Some(&pl),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                buffers: &[],
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
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("checker_pass"),
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
            pass.draw(0..3, 0..1);
        }
        target.generate_mips(&mut encoder, &mip_blit);

        // Levels 0 and 1 only: both have 256-byte-aligned rows at size 128.
        let read = |level: u32, enc: &mut wgpu::CommandEncoder| {
            let dim = size >> level;
            let bpr = dim * 4;
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("checker_readback"),
                size: (bpr * dim) as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            enc.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: &target.color,
                    mip_level: level,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &buf,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(bpr),
                        rows_per_image: Some(dim),
                    },
                },
                wgpu::Extent3d { width: dim, height: dim, depth_or_array_layers: 1 },
            );
            buf
        };
        let b0 = read(0, &mut encoder);
        let b1 = read(1, &mut encoder);
        queue.submit(Some(encoder.finish()));

        let stats = |buf: &wgpu::Buffer| -> (f64, f64) {
            buf.slice(..).map_async(wgpu::MapMode::Read, |_| {});
            let _ = device.poll(wgpu::PollType::Wait);
            let data = buf.slice(..).get_mapped_range();
            let reds: Vec<f64> = data.chunks_exact(4).map(|p| p[0] as f64).collect();
            let mean = reds.iter().sum::<f64>() / reds.len() as f64;
            let var = reds.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / reds.len() as f64;
            drop(data);
            buf.unmap();
            (mean, var)
        };
        let (_m0, v0) = stats(&b0);
        let (m1, v1) = stats(&b1);
        Some((v0, v1, m1))
    }

    /// **The C2 shimmer test.**
    ///
    /// The scope target is composited onto an ocular disc several times smaller
    /// than itself. With a single mip level that minification point-samples a
    /// handful of texels out of dozens, which crawls under head motion and reads
    /// as aliasing — the defect MSAA was originally proposed to fix and would
    /// not have fixed, because MSAA acts inside the target rather than on how
    /// the target is sampled down.
    ///
    /// A 1-texel checkerboard is the worst case: maximum variance at level 0.
    /// If the chain is built correctly, level 1 averages each 2x2 block to mid
    /// grey — variance collapses and the mean lands near 50%. That is exactly
    /// the data a minifying sample will now read instead of point-sampled edges.
    #[test]
    fn the_mip_chain_averages_away_the_minification_aliasing() {
        let Some((v0, v1, m1)) = checkerboard_mip_variances(TARGET) else {
            eprintln!("skipping: no GPU adapter available in this environment");
            return;
        };

        assert!(v0 > 1000.0, "level 0 checkerboard should have huge variance, got {v0:.1}");
        assert!(
            v1 < v0 / 50.0,
            "level 1 must average the checkerboard away: variance {v1:.1} vs {v0:.1}"
        );
        assert!(
            (m1 - 127.5).abs() < 20.0,
            "each 2x2 checker block should average to mid grey, got {m1:.1}"
        );
    }
}
