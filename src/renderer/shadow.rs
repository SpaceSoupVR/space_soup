//! Real-time shadow mapping for the desktop render path.
//!
//! Two shadow slots share one depth-only pipeline set and one comparison
//! sampler:
//!   * **Sun** — an orthographic shadow for the directional light.
//!   * **Spot** — a perspective shadow for the flashlight / shadow-casting
//!     spot light (hand-attached on the Quest; placed in a scene in the editor).
//!
//! Each slot renders every caster from its light's point of view into its own
//! depth texture; the main pass samples both (bound into the shared camera/
//! lights group — see `uniforms::UniformBuffer`) and darkens fragments that
//! fail the depth comparison.
//!
//! Casters: world-space cuboids and terrain (SolidVertex), level brushes
//! (BrushVertex), and non-skinned meshes -- which includes baked caves, because
//! a layered mesh keeps its ordinary vertex buffer alongside the weighted one
//! and this pass only ever reads position. Skinned meshes do not cast yet: a
//! character-shaped hole in the shadows is far less noticeable than a wall that
//! does not cast one, and doing it properly means running the skinning in the
//! depth pass too.
//!
//! ONE PASS FOR BOTH EYES
//!
//! The sun's shadow map is built in LIGHT space and does not depend on where
//! the viewer is, so on the XR path it is rendered once per frame rather than
//! once per eye. That is not an optimisation to get to later -- doing it inside
//! the eye loop would double the cost of the most expensive thing here for an
//! identical result.

use glam::{Mat4, Vec3};
use wgpu::*;

use super::cuboid::SolidVertex;
use super::mesh::MeshVertex;

/// Side length of each (square) shadow depth texture on desktop.
///
/// A parameter rather than a constant because the headset cannot afford this
/// one: 2048 squared at Depth32Float is 16MB per slot, and the Quest is already
/// spending its bandwidth on two eyes at full resolution. See `QUEST_SHADOW_DIM`.
pub const SHADOW_DIM: u32 = 2048;

/// Side length used on the headset.
///
/// Half the desktop resolution in each axis, so a quarter of the memory and a
/// quarter of the fill. The honest cost is that a single map at this size,
/// stretched over an outdoor sightline, gives visibly chunky shadow edges far
/// from the viewer -- the real fix for that is cascades, which is a much larger
/// piece of work and is not this. Near shadows, which are the ones a player
/// actually reads cover from, hold up.
pub const QUEST_SHADOW_DIM: u32 = 1024;

/// How many spot lights can cast a real-time shadow at once.
///
/// A BUDGET, chosen rather than inherited. It was one, which is not a decision
/// anybody made -- the code took the first spot in the scene and the rest lit
/// without shadows, so a room with two matching lamps had one casting and one
/// not. Four covers a lit interior; each layer costs `dim * dim` of
/// Depth32Float, so at the Quest's 1024 that is 4MB apiece.
///
/// Lights beyond the budget still light the scene; they just do not occlude.
/// That is the honest failure -- a missing shadow rather than a missing light --
/// and the editor says which lights are affected.
pub const MAX_SPOT_SHADOWS: usize = 4;

/// Which shadow slot a pass/upload targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShadowKind {
    Sun,
    /// One of the spot layers, by index.
    Spot(usize),
}

/// Builds the sun's light-space view-projection: an orthographic box aimed
/// along `dir`, centered on `center`, sized to enclose a sphere of `radius`.
pub fn directional_light_matrix(dir: Vec3, center: Vec3, radius: f32) -> Mat4 {
    let radius = radius.max(0.001);
    let d = {
        let n = dir.normalize_or_zero();
        if n == Vec3::ZERO {
            Vec3::NEG_Y
        } else {
            n
        }
    };
    let dist = radius * 2.0;
    let eye = center - d * dist;
    let up = if d.dot(Vec3::Y).abs() > 0.99 {
        Vec3::Z
    } else {
        Vec3::Y
    };
    let view = Mat4::look_at_rh(eye, center, up);
    // glam `orthographic_rh` maps z into [0,1] to match wgpu clip space.
    let proj = Mat4::orthographic_rh(-radius, radius, -radius, radius, 0.0, dist + radius);
    proj * view
}

/// Builds a spot/flashlight light-space view-projection: a perspective frustum
/// from `pos` aimed along `dir`, with a vertical field of view covering the
/// spot's full cone angle (`cone_angle_deg`) and a far plane at `range`.
pub fn spot_light_matrix(pos: Vec3, dir: Vec3, cone_angle_deg: f32, range: f32) -> Mat4 {
    let d = {
        let n = dir.normalize_or_zero();
        if n == Vec3::ZERO {
            Vec3::NEG_Z
        } else {
            n
        }
    };
    let up = if d.dot(Vec3::Y).abs() > 0.99 {
        Vec3::Z
    } else {
        Vec3::Y
    };
    let view = Mat4::look_at_rh(pos, pos + d, up);
    // Pad the fov slightly beyond the full cone so the cone edge isn't clipped.
    let fov = (cone_angle_deg.to_radians() * 1.1).clamp(0.1, std::f32::consts::PI - 0.1);
    let far = range.max(0.2);
    let near = (far * 0.02).max(0.02);
    let proj = Mat4::perspective_rh(fov, 1.0, near, far);
    proj * view
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct LightMatrix {
    view_proj: [[f32; 4]; 4],
}

/// One mesh caster for a shadow pass: vertex buffer, index buffer, index count,
/// and the mesh's model-matrix bind group (reused from the main mesh pass).
pub type ShadowMeshDraw<'a> = (&'a Buffer, &'a Buffer, u32, &'a BindGroup);

/// One shadow map's own resources: depth texture and the per-frame light matrix.
struct ShadowSlot {
    _depth_texture: Texture,
    depth_view: TextureView,
    light_buffer: Buffer,
    light_bind_group: BindGroup,
}

impl ShadowSlot {
    fn new(device: &Device, light_layout: &BindGroupLayout, label: &str, dim: u32) -> Self {
        Self::new_layered(device, light_layout, label, dim, 1, 0)
    }

    /// A slot that renders into one layer of an existing array texture.
    fn from_texture(
        device: &Device,
        light_layout: &BindGroupLayout,
        texture: &Texture,
        layer: u32,
    ) -> Self {
        let depth_view = texture.create_view(&TextureViewDescriptor {
            dimension: Some(TextureViewDimension::D2),
            base_array_layer: layer,
            array_layer_count: Some(1),
            ..Default::default()
        });
        let (light_buffer, light_bind_group) = Self::light_uniform(device, light_layout);
        Self { _depth_texture: texture.clone(), depth_view, light_buffer, light_bind_group }
    }

    /// One layer of a shared array texture, or a standalone map when `layers`
    /// is 1.
    ///
    /// An ARRAY TEXTURE rather than an array of bindings. Binding several
    /// textures to one slot needs the TEXTURE_BINDING_ARRAY feature, which is
    /// not something to rely on for a headset; a depth_2d_array sampled by
    /// layer index is core WebGPU and works everywhere.
    fn new_layered(
        device: &Device,
        light_layout: &BindGroupLayout,
        label: &str,
        dim: u32,
        layers: u32,
        layer: u32,
    ) -> Self {
        let depth_texture = device.create_texture(&TextureDescriptor {
            label: Some(label),
            size: Extent3d { width: dim, height: dim, depth_or_array_layers: layers },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Depth32Float,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        // A single-layer view, which is what a render pass attaches to. The
        // array view used for sampling is built separately, from the same
        // texture.
        let depth_view = depth_texture.create_view(&TextureViewDescriptor {
            dimension: Some(TextureViewDimension::D2),
            base_array_layer: layer,
            array_layer_count: Some(1),
            ..Default::default()
        });
        let (light_buffer, light_bind_group) = Self::light_uniform(device, light_layout);
        Self {
            _depth_texture: depth_texture,
            depth_view,
            light_buffer,
            light_bind_group,
        }
    }

    /// The per-slot light matrix buffer and its bind group.
    fn light_uniform(device: &Device, light_layout: &BindGroupLayout) -> (Buffer, BindGroup) {
        let light_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("shadow_light_matrix"),
            size: std::mem::size_of::<LightMatrix>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let light_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("shadow_light_bg"),
            layout: light_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: light_buffer.as_entire_binding(),
            }],
        });
        (light_buffer, light_bind_group)
    }
}

pub struct ShadowMap {
    sun: ShadowSlot,
    /// One slot per shadow-casting spot. Separate textures rather than one
    /// atlas: an atlas needs every sample to offset and clamp into its own
    /// tile, and a single wrong clamp reads a neighbour's depth, which looks
    /// like a shadow from a light that is not there.
    spots: [ShadowSlot; MAX_SPOT_SHADOWS],
    _spot_texture: Texture,
    /// The whole spot array, as the shader samples it.
    spot_array_view: TextureView,
    sampler: Sampler,
    solid_pipeline: RenderPipeline,
    mesh_pipeline: RenderPipeline,
    /// Level geometry. A separate pipeline only because `BrushVertex` has a
    /// different stride -- the shader is `vs_solid`, unchanged, because a depth
    /// pass reads position and nothing else.
    brush_pipeline: RenderPipeline,
}

impl ShadowMap {
    /// Constructible from just the device — it owns a model bind-group layout
    /// with the same shape the mesh pipeline uses (wgpu treats identical
    /// descriptors as compatible), so the mesh pass's `ModelUniform` bind
    /// groups are reused here without rebinding, and there is no init-order
    /// cycle with `UniformBuffer`.
    pub fn new(device: &Device) -> Self {
        Self::with_dimension(device, SHADOW_DIM)
    }

    pub fn with_dimension(device: &Device, dim: u32) -> Self {
        let model_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("shadow_model_bgl"),
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
        let light_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("shadow_light_bgl"),
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

        let sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("shadow_sampler"),
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            compare: Some(CompareFunction::LessEqual),
            ..Default::default()
        });

        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("shadow_shader"),
            source: ShaderSource::Wgsl(SHADOW_SHADER.into()),
        });

        let bias = DepthBiasState {
            constant: 2,
            slope_scale: 2.0,
            clamp: 0.0,
        };

        let solid_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("shadow_solid_layout"),
            bind_group_layouts: &[&light_layout],
            push_constant_ranges: &[],
        });
        let solid_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("shadow_solid_pipeline"),
            layout: Some(&solid_layout),
            vertex: VertexState {
                module: &shader,
                entry_point: Some("vs_solid"),
                compilation_options: PipelineCompilationOptions::default(),
                buffers: &[SolidVertex::layout()],
            },
            fragment: None,
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                cull_mode: Some(Face::Back),
                front_face: FrontFace::Ccw,
                polygon_mode: PolygonMode::Fill,
                ..Default::default()
            },
            depth_stencil: Some(DepthStencilState {
                format: TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: CompareFunction::Less,
                stencil: StencilState::default(),
                bias,
            }),
            multisample: MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let mesh_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("shadow_mesh_layout"),
            bind_group_layouts: &[&light_layout, &model_layout],
            push_constant_ranges: &[],
        });
        let mesh_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("shadow_mesh_pipeline"),
            layout: Some(&mesh_layout),
            vertex: VertexState {
                module: &shader,
                entry_point: Some("vs_mesh"),
                compilation_options: PipelineCompilationOptions::default(),
                buffers: &[MeshVertex::layout()],
            },
            fragment: None,
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
                bias,
            }),
            multisample: MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let brush_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("shadow_brush_pipeline"),
            layout: Some(&solid_layout),
            vertex: VertexState {
                module: &shader,
                entry_point: Some("vs_solid"),
                compilation_options: PipelineCompilationOptions::default(),
                buffers: &[super::brush_pipeline::BrushVertex::layout()],
            },
            fragment: None,
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                cull_mode: Some(Face::Back),
                front_face: FrontFace::Ccw,
                polygon_mode: PolygonMode::Fill,
                ..Default::default()
            },
            depth_stencil: Some(DepthStencilState {
                format: TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: CompareFunction::Less,
                stencil: StencilState::default(),
                bias,
            }),
            multisample: MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let sun = ShadowSlot::new(device, &light_layout, "sun_shadow_depth", dim);
        // ONE texture with MAX_SPOT_SHADOWS layers, not N textures. The shader
        // samples it as a depth_2d_array by layer index, which is core WebGPU;
        // binding N separate textures to one slot would need an extension the
        // headset may not have.
        let spot_texture = device.create_texture(&TextureDescriptor {
            label: Some("spot_shadow_depth_array"),
            size: Extent3d {
                width: dim,
                height: dim,
                depth_or_array_layers: MAX_SPOT_SHADOWS as u32,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Depth32Float,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let spot_array_view = spot_texture.create_view(&TextureViewDescriptor {
            dimension: Some(TextureViewDimension::D2Array),
            ..Default::default()
        });
        let spots = std::array::from_fn(|i| {
            ShadowSlot::from_texture(device, &light_layout, &spot_texture, i as u32)
        });
        let spot = ShadowSlot::new(device, &light_layout, "spot_shadow_depth", dim);

        Self {
            sun,
            spots,
            _spot_texture: spot_texture,
            spot_array_view,
            sampler,
            solid_pipeline,
            mesh_pipeline,
            brush_pipeline,
        }
    }

    pub fn sun_depth_view(&self) -> &TextureView {
        &self.sun.depth_view
    }

    /// The spot shadow array, as the shading pass samples it.
    pub fn spot_depth_view(&self) -> &TextureView {
        &self.spot_array_view
    }

    pub fn sampler(&self) -> &Sampler {
        &self.sampler
    }

    fn slot(&self, kind: ShadowKind) -> &ShadowSlot {
        match kind {
            ShadowKind::Sun => &self.sun,
            ShadowKind::Spot(i) => &self.spots[i.min(MAX_SPOT_SHADOWS - 1)],
        }
    }

    /// Uploads a slot's light-space matrix for this frame.
    pub fn upload_light(&self, queue: &Queue, kind: ShadowKind, view_proj: Mat4) {
        let m = LightMatrix {
            view_proj: view_proj.to_cols_array_2d(),
        };
        queue.write_buffer(&self.slot(kind).light_buffer, 0, bytemuck::bytes_of(&m));
    }

    /// Records a depth-only shadow pass for `kind` into `encoder`: clears that
    /// slot's depth texture, then draws the solid geometry and mesh casters
    /// from the light's viewpoint.
    pub fn record(
        &self,
        encoder: &mut CommandEncoder,
        kind: ShadowKind,
        solid: Option<(&Buffer, &Buffer, u32)>,
        brushes: Option<(&Buffer, &Buffer, u32)>,
        mesh_draws: &[ShadowMeshDraw],
    ) {
        let slot = self.slot(kind);
        let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("shadow_pass"),
            color_attachments: &[],
            depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                view: &slot.depth_view,
                depth_ops: Some(Operations {
                    load: LoadOp::Clear(1.0),
                    store: StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            ..Default::default()
        });

        if let Some((vb, ib, count)) = solid {
            if count > 0 {
                pass.set_pipeline(&self.solid_pipeline);
                pass.set_bind_group(0, &slot.light_bind_group, &[]);
                pass.set_vertex_buffer(0, vb.slice(..));
                pass.set_index_buffer(ib.slice(..), IndexFormat::Uint32);
                pass.draw_indexed(0..count, 0, 0..1);
            }
        }

        if let Some((vb, ib, count)) = brushes {
            if count > 0 {
                pass.set_pipeline(&self.brush_pipeline);
                pass.set_bind_group(0, &slot.light_bind_group, &[]);
                pass.set_vertex_buffer(0, vb.slice(..));
                pass.set_index_buffer(ib.slice(..), IndexFormat::Uint32);
                pass.draw_indexed(0..count, 0, 0..1);
            }
        }

        if !mesh_draws.is_empty() {
            pass.set_pipeline(&self.mesh_pipeline);
            pass.set_bind_group(0, &slot.light_bind_group, &[]);
            for (vb, ib, count, model_bg) in mesh_draws {
                pass.set_bind_group(1, *model_bg, &[]);
                pass.set_vertex_buffer(0, vb.slice(..));
                pass.set_index_buffer(ib.slice(..), IndexFormat::Uint32);
                pass.draw_indexed(0..*count, 0, 0..1);
            }
        }
    }
}

/// Depth-only shadow-caster shaders shared by both slots. `vs_solid` draws
/// world-space cuboid geometry; `vs_mesh` applies the mesh's model matrix.
const SHADOW_SHADER: &str = r#"
struct LightMatrix { view_proj: mat4x4<f32> }
@group(0) @binding(0) var<uniform> light: LightMatrix;

struct ModelUniform { model: mat4x4<f32> }
@group(1) @binding(0) var<uniform> model_u: ModelUniform;

@vertex
fn vs_solid(@location(0) pos: vec3<f32>) -> @builtin(position) vec4<f32> {
    return light.view_proj * vec4<f32>(pos, 1.0);
}

@vertex
fn vs_mesh(@location(0) pos: vec3<f32>) -> @builtin(position) vec4<f32> {
    return light.view_proj * model_u.model * vec4<f32>(pos, 1.0);
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn light_matrix_places_center_inside_the_clip_box() {
        let center = Vec3::new(1.0, 2.0, -3.0);
        let m = directional_light_matrix(Vec3::new(0.0, -1.0, 0.0), center, 10.0);
        let clip = m * center.extend(1.0);
        let ndc = clip.truncate() / clip.w;
        assert!(ndc.x.abs() < 1e-4 && ndc.y.abs() < 1e-4, "center should project to the middle, got {ndc:?}");
        assert!(ndc.z > 0.0 && ndc.z < 1.0, "center depth should be inside [0,1], got {}", ndc.z);
    }

    #[test]
    fn points_farther_along_the_light_direction_have_greater_depth() {
        let center = Vec3::ZERO;
        let dir = Vec3::new(0.0, -1.0, 0.0);
        let m = directional_light_matrix(dir, center, 10.0);
        let high = Vec3::new(0.0, 4.0, 0.0);
        let low = Vec3::new(0.0, -4.0, 0.0);
        let z = |p: Vec3| {
            let c = m * p.extend(1.0);
            c.z / c.w
        };
        assert!(z(high) < z(low), "point nearer the sun should have smaller depth: {} vs {}", z(high), z(low));
    }

    #[test]
    fn degenerate_direction_does_not_panic() {
        let m = directional_light_matrix(Vec3::ZERO, Vec3::ZERO, 5.0);
        assert!(m.is_finite());
    }

    #[test]
    fn spot_matrix_projects_target_ahead_into_view() {
        // Flashlight at origin aimed down -Z; a point 3m ahead should land near
        // the center of the shadow map with depth inside [0,1].
        let m = spot_light_matrix(Vec3::ZERO, Vec3::NEG_Z, 45.0, 10.0);
        let ahead = Vec3::new(0.0, 0.0, -3.0);
        let clip = m * ahead.extend(1.0);
        let ndc = clip.truncate() / clip.w;
        assert!(ndc.x.abs() < 1e-3 && ndc.y.abs() < 1e-3, "aimed point should be centered, got {ndc:?}");
        assert!(ndc.z > 0.0 && ndc.z < 1.0, "aimed point depth should be inside [0,1], got {}", ndc.z);
    }

    #[test]
    fn spot_matrix_closer_point_has_smaller_depth() {
        let m = spot_light_matrix(Vec3::ZERO, Vec3::NEG_Z, 45.0, 10.0);
        let z = |p: Vec3| {
            let c = m * p.extend(1.0);
            c.z / c.w
        };
        assert!(z(Vec3::new(0.0, 0.0, -1.0)) < z(Vec3::new(0.0, 0.0, -5.0)), "closer point should have smaller depth");
    }
}

/// Render tests for the parts of shadowing that a matrix test cannot see.
///
/// The matrix tests above prove the projection is sane. They say nothing about
/// whether the depth pass runs, whether the main pass samples the map it wrote,
/// or whether a directional light is treated as directional -- all of which are
/// wiring, and all of which fail silently and look like a lighting bug.
#[cfg(test)]
mod render_tests {
    use super::*;
    use crate::renderer::cuboid::SolidVertex;
    use crate::renderer::lights::{Light, LightKind, LightsUniform};
    use crate::renderer::pipeline::{lightmap_bind_group_layout, SolidPipeline};
    use crate::renderer::uniforms::{ShadowUpload, SkyUpload, UniformBuffer};
    use crate::renderer::Color3;
    use wgpu::util::DeviceExt;

    /// ODD, so the centre texel sits exactly at NDC (0, 0).
    ///
    /// With an even width the middle pixel is half a texel off centre, and the
    /// quad's world position there is not the position the test asked for. That
    /// is invisible for a test comparing two renders -- both are off by the
    /// same amount -- and it is fatal for one comparing against an absolute
    /// number, which is how `the_falloff_the_editor_predicts...` first came out
    /// 2% low and looked like a real disagreement with the editor.
    const SIZE: u32 = 9;

    fn gpu() -> Option<(Device, Queue)> {
        crate::renderer::terrain_pipeline::tests::headless_gpu()
    }

    fn vertex(pos: [f32; 3], normal: [f32; 3]) -> SolidVertex {
        SolidVertex {
            position: pos,
            normal,
            color: [1.0, 1.0, 1.0, 1.0],
            uv2: [0.0, 0.0],
            reflectivity: 0.0,
        }
    }

    /// A horizontal quad of the given size at height `y`, wound so the sun
    /// overhead sees its front.
    fn ground(y: f32, half: f32) -> (Vec<SolidVertex>, Vec<u32>) {
        let n = [0.0, 1.0, 0.0];
        let v = vec![
            vertex([-half, y, -half], n),
            vertex([half, y, -half], n),
            vertex([half, y, half], n),
            vertex([-half, y, half], n),
        ];
        (v, vec![0, 2, 1, 0, 3, 2])
    }

    /// What one render asks for.
    struct Scene {
        lights: Vec<Light>,
        /// Extra geometry drawn ONLY into the shadow map: the thing casting.
        caster: Option<(Vec<SolidVertex>, Vec<u32>)>,
        /// Where the receiving quad sits in the world.
        ///
        /// A full position and not just a height: the shadow frustum can be
        /// escaped two ways -- past the far plane, or off the side of the map --
        /// and only the second isolates the uv guard from everything else.
        receiver_at: glam::Vec3,
        /// Where the eye is RELATIVE TO THE RECEIVER.
        ///
        /// Relative and not absolute, because specular depends on the angle
        /// between the eye and the surface: an absolute eye left behind while
        /// the receiver moved 500m turned a test of distance falloff into a
        /// test of the highlight's geometry, and it failed for the wrong
        /// reason with a very convincing-looking number.
        eye_offset: glam::Vec3,
        shadows_on: bool,
        /// The receiving quad's surface normal.
        ///
        /// Fixed pointing up until the sky arrived. The sky's ambient depends on
        /// which way a surface faces -- that being the whole difference between
        /// nine coefficients and one -- so it has to be a variable now.
        normal: [f32; 3],
        /// The sky's ambient. Defaults to the flat term the engine always had.
        sky: SkyUpload,
    }

    /// Renders a lit quad filling the view and returns its centre pixel.
    ///
    /// The receiver is drawn with an identity view_proj so it fills the target
    /// whatever its world position -- which is what lets the same quad be moved
    /// hundreds of metres to test that a sun does not attenuate.
    fn shade_receiver(scene: Scene) -> Option<[u8; 4]> {
        let (device, queue) = gpu()?;
        let format = TextureFormat::Rgba8Unorm;

        let shadow_map = ShadowMap::with_dimension(&device, 256);
        let lights_uniform = LightsUniform::new(&device);
        let uniforms = UniformBuffer::new(
            &device,
            &lights_uniform,
            shadow_map.sun_depth_view(),
            shadow_map.spot_depth_view(),
            shadow_map.sampler(),
        );
        // Spots claim shadow layers in scene order, exactly as both renderers
        // do -- so a test with two spots exercises two layers rather than one.
        let spot_indices: Vec<usize> = if scene.shadows_on {
            scene
                .lights
                .iter()
                .enumerate()
                .filter(|(_, l)| l.kind == LightKind::Spot)
                .map(|(i, _)| i)
                .take(MAX_SPOT_SHADOWS)
                .collect()
        } else {
            Vec::new()
        };
        lights_uniform.upload_with_shadow_layers(&queue, &scene.lights, &spot_indices);

        let sun = scene.lights.iter().find(|l| l.kind == LightKind::Directional);
        let sun_view_proj = sun
            .map(|l| directional_light_matrix(l.direction, Vec3::ZERO, 20.0))
            .unwrap_or(Mat4::IDENTITY);
        let mut spot_view_proj = [Mat4::IDENTITY; MAX_SPOT_SHADOWS];
        for (layer, &i) in spot_indices.iter().enumerate() {
            let l = &scene.lights[i];
            spot_view_proj[layer] =
                spot_light_matrix(l.position, l.direction, l.cone_angle_deg, l.range);
        }
        let upload = ShadowUpload {
            sun_view_proj,
            spot_view_proj,
            sun_enabled: scene.shadows_on && sun.is_some(),
            spot_count: spot_indices.len() as u32,
        };

        // The receiver, as clip-space coordinates that fill the target. Its
        // WORLD position -- which is what the shadow lookup and the lights use
        // -- rides on the model translation the view_proj cancels out.
        let world = scene.receiver_at;
        uniforms.upload_with_sky(
            &queue,
            glam::Mat4::from_translation(-world),
            world + scene.eye_offset,
            &upload,
            &scene.sky,
        );

        let pipeline = SolidPipeline::new(&device, format, &uniforms.layout);
        let white = crate::renderer::mesh::create_texture_from_rgba(
            &device,
            &queue,
            &lightmap_bind_group_layout(&device),
            &[255u8, 255, 255, 255],
            1,
            1,
        );

        let n = scene.normal;
        // z = 0 so the quad's world position is exactly `receiver_at`, for the
        // same reason SIZE is odd.
        let quad: Vec<SolidVertex> = [
            [-1.0, -1.0, 0.0],
            [3.0, -1.0, 0.0],
            [-1.0, 3.0, 0.0],
        ]
        .iter()
        .map(|p| {
            // Clip position, with the world position added back so world_pos in
            // the shader is the receiver's real place in the scene.
            vertex([p[0] + world.x, p[1] + world.y, p[2] + world.z], n)
        })
        .collect();

        let vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("recv_vb"),
            contents: bytemuck::cast_slice(&quad),
            usage: BufferUsages::VERTEX,
        });
        let ib = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("recv_ib"),
            contents: bytemuck::cast_slice(&[0u32, 1, 2]),
            usage: BufferUsages::INDEX,
        });

        let caster_bufs = scene.caster.as_ref().map(|(v, i)| {
            (
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("caster_vb"),
                    contents: bytemuck::cast_slice(v),
                    usage: BufferUsages::VERTEX,
                }),
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("caster_ib"),
                    contents: bytemuck::cast_slice(i),
                    usage: BufferUsages::INDEX,
                }),
                i.len() as u32,
            )
        });

        let desc = |fmt, usage| TextureDescriptor {
            label: None,
            size: Extent3d { width: SIZE, height: SIZE, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: fmt,
            usage,
            view_formats: &[],
        };
        let target = device.create_texture(&desc(
            format,
            TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
        ));
        let depth = device.create_texture(&desc(
            TextureFormat::Depth32Float,
            TextureUsages::RENDER_ATTACHMENT,
        ));
        let target_view = target.create_view(&Default::default());
        let depth_view = depth.create_view(&Default::default());
        let readback = device.create_buffer(&BufferDescriptor {
            label: Some("recv_readback"),
            size: (256 * SIZE) as u64,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&Default::default());
        if upload.sun_enabled {
            shadow_map.upload_light(&queue, ShadowKind::Sun, sun_view_proj);
            let solid = caster_bufs
                .as_ref()
                .map(|(vb, ib, count)| (vb, ib, *count));
            shadow_map.record(&mut encoder, ShadowKind::Sun, solid, None, &[]);
        }
        // One depth pass per spot layer, matching both renderers.
        for layer in 0..upload.spot_count as usize {
            shadow_map.upload_light(
                &queue,
                ShadowKind::Spot(layer),
                upload.spot_view_proj[layer],
            );
            let solid = caster_bufs
                .as_ref()
                .map(|(vb, ib, count)| (vb, ib, *count));
            shadow_map.record(&mut encoder, ShadowKind::Spot(layer), solid, None, &[]);
        }
        {
            let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("recv_pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &target_view,
                    resolve_target: None,
                    ops: Operations { load: LoadOp::Clear(Color::BLACK), store: StoreOp::Store },
                })],
                depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(Operations { load: LoadOp::Clear(1.0), store: StoreOp::Store }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&pipeline.pipeline);
            pass.set_bind_group(0, &uniforms.bind_group, &[]);
            pass.set_bind_group(1, &white.bind_group, &[]);
            pass.set_vertex_buffer(0, vb.slice(..));
            pass.set_index_buffer(ib.slice(..), IndexFormat::Uint32);
            pass.draw_indexed(0..3, 0, 0..1);
        }
        encoder.copy_texture_to_buffer(
            TexelCopyTextureInfo {
                texture: &target,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            TexelCopyBufferInfo {
                buffer: &readback,
                layout: TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(256),
                    rows_per_image: Some(SIZE),
                },
            },
            Extent3d { width: SIZE, height: SIZE, depth_or_array_layers: 1 },
        );
        queue.submit(Some(encoder.finish()));

        let slice = readback.slice(..);
        slice.map_async(MapMode::Read, |_| {});
        device.poll(PollType::Wait).ok();
        let data = slice.get_mapped_range();
        let at = (SIZE / 2) as usize * 256 + (SIZE / 2) as usize * 4;
        Some([data[at], data[at + 1], data[at + 2], data[at + 3]])
    }

    fn sun() -> Light {
        Light {
            position: Vec3::ZERO,
            // Straight down. `direction` is the way the light TRAVELS.
            direction: Vec3::NEG_Y,
            kind: LightKind::Directional,
            color: Color3(255, 255, 255, 255),
            // Dim on purpose. Ambient alone is already 0.6 of full white, so a
            // brighter sun saturates every channel and the specular difference
            // the eye test measures disappears into the clamp -- which it did,
            // at 0.5, with both eyes reading a flat 255.
            intensity: 0.2,
            // Ignored for a sun. Set small on purpose: if anything ever applied
            // range falloff to a directional light, these tests would go dark.
            range: 1.0,
            cone_angle_deg: 180.0,
        }
    }

    macro_rules! shot {
        ($s:expr) => {
            match shade_receiver($s) {
                Some(px) => px,
                None => {
                    eprintln!("skipping: no GPU adapter available");
                    return;
                }
            }
        };
    }

    fn base() -> Scene {
        Scene {
            lights: vec![sun()],
            caster: None,
            receiver_at: Vec3::ZERO,
            eye_offset: Vec3::new(0.0, 5.0, 0.0),
            shadows_on: true,
            normal: [0.0, 1.0, 0.0],
            sky: SkyUpload::none(),
        }
    }

    /// A spot above the receiver, aimed straight down.
    fn spot_above(x: f32, intensity: f32) -> Light {
        Light {
            position: Vec3::new(x, 6.0, 0.0),
            direction: Vec3::NEG_Y,
            kind: LightKind::Spot,
            color: Color3(255, 255, 255, 255),
            intensity,
            range: 30.0,
            cone_angle_deg: 120.0,
        }
    }

    #[test]
    fn a_spot_light_casts_a_shadow() {
        // The baseline the second-spot test is measured against. If this did
        // not hold, the two-spot result would prove nothing.
        let lit = shot!(Scene { lights: vec![spot_above(0.0, 4.0)], ..base() });
        let shaded = shot!(Scene {
            lights: vec![spot_above(0.0, 4.0)],
            caster: Some(ground(3.0, 8.0)),
            ..base()
        });
        assert!(
            (lit[0] as i32) - (shaded[0] as i32) > 8,
            "a spot cast no shadow: {lit:?} lit vs {shaded:?} with a caster between",
        );
    }

    #[test]
    fn a_second_spot_light_also_casts_a_shadow() {
        // THE POINT OF THE SHADOW ARRAY. There used to be one spot depth map,
        // claimed by the first spot in the scene, so a room with two matching
        // lamps had one casting a shadow and one not -- which reads as a broken
        // light rather than an exhausted budget.
        //
        // The first light here is deliberately dark, so anything measured comes
        // from the SECOND one. With a single shared map the second light gets no
        // shadow at all and the caster makes no difference.
        let dim = spot_above(-4.0, 0.0);
        let lit = shot!(Scene { lights: vec![dim.clone(), spot_above(0.0, 4.0)], ..base() });
        let shaded = shot!(Scene {
            lights: vec![dim, spot_above(0.0, 4.0)],
            caster: Some(ground(3.0, 8.0)),
            ..base()
        });
        assert!(
            (lit[0] as i32) - (shaded[0] as i32) > 8,
            "the SECOND spot cast no shadow: {lit:?} lit vs {shaded:?} with a caster between",
        );
    }

    #[test]
    fn a_spot_beyond_the_budget_still_lights_without_shadowing() {
        // The honest failure past MAX_SPOT_SHADOWS: a missing shadow, never a
        // missing light. A level that silently dropped its overflow lamps would
        // go dark in patches, which is far worse than an unshadowed one.
        let mut lights: Vec<Light> = (0..MAX_SPOT_SHADOWS + 1)
            .map(|i| spot_above(-8.0 + i as f32, 0.0))
            .collect();
        *lights.last_mut().unwrap() = spot_above(0.0, 4.0);

        let with_extra = shot!(Scene { lights: lights.clone(), ..base() });
        let alone = shot!(Scene { lights: vec![spot_above(0.0, 4.0)], ..base() });
        assert!(
            (with_extra[0] as i32 - alone[0] as i32).abs() < 6,
            "the light past the shadow budget stopped lighting: {with_extra:?} vs {alone:?}",
        );
    }

    #[test]
    fn a_sun_lights_a_surface_facing_it() {
        let lit = shot!(base());
        let unlit = shot!(Scene { lights: vec![], ..base() });
        assert!(
            lit[0] as i32 - unlit[0] as i32 > 8,
            "a directional light contributed nothing: {lit:?} vs ambient {unlit:?}",
        );
    }

    #[test]
    fn a_sun_does_not_dim_with_distance() {
        // THE DEFINING PROPERTY. A sun's rays are parallel and infinitely far
        // away, so a surface 500m from the origin must be lit exactly as one at
        // the origin. Treating it as a point light -- the easiest mistake, since
        // it shares the struct -- makes distant terrain fade to ambient, which
        // reads as fog nobody asked for.
        // At the origin and 500m up. Both unshadowed, so the only thing that
        // could separate them is falloff.
        //
        // An earlier version compared 100m with 500m, to keep the shadow
        // frustum out of it. That made the test USELESS: a point-light falloff
        // has already collapsed to nothing by 100m, so both readings came back
        // at ambient and matched. It only passed for the wrong reason, and a
        // deliberate break proved it -- the failure landed on a different test
        // entirely.
        let near = shot!(base());
        let far = shot!(Scene { receiver_at: Vec3::new(0.0, 500.0, 0.0), ..base() });
        for c in 0..3 {
            assert!(
                (near[c] as i32 - far[c] as i32).abs() <= 2,
                "the sun attenuated with distance: {near:?} at the origin vs {far:?} at y=500",
            );
        }
    }

    #[test]
    fn an_occluder_darkens_what_is_under_it() {
        // The whole point, end to end: a depth pass that actually runs, a map
        // that is actually sampled, and a comparison that comes out the right
        // way round. Each half is invisible on its own -- a depth pass that
        // never runs leaves a cleared map that shadows nothing, which looks
        // exactly like a scene with no caster in it.
        let clear = shot!(base());
        let shaded = shot!(Scene {
            caster: Some(ground(3.0, 8.0)),
            ..base()
        });
        assert!(
            clear[0] as i32 - shaded[0] as i32 > 8,
            "a slab three metres overhead cast no shadow: {clear:?} lit vs {shaded:?} shadowed",
        );
    }

    #[test]
    fn a_shadowed_surface_keeps_its_ambient_light() {
        // Shadow multiplies the LIGHT and not the surface, so a shadowed
        // fragment falls back to ambient rather than to black. A shadow that
        // reads as a hole in the world is the classic version of this bug.
        let shaded = shot!(Scene { caster: Some(ground(3.0, 8.0)), ..base() });
        let ambient_only = shot!(Scene { lights: vec![], ..base() });
        for c in 0..3 {
            assert!(
                (shaded[c] as i32 - ambient_only[c] as i32).abs() <= 3,
                "a shadowed fragment is not at ambient: {shaded:?} vs {ambient_only:?}",
            );
        }
    }

    #[test]
    fn geometry_outside_the_shadow_map_is_lit_rather_than_black() {
        // The `valid` guard in `shadow_coords`. Without it everything beyond the
        // map's reach reads as fully shadowed, and a level shows a hard line
        // across the ground at the edge of the shadow box -- far more
        // objectionable than a distant shadow simply going missing.
        // SIDEWAYS out of the map, not past its far plane: same height, same
        // distance from everything, 300m off the side of a box 20m across. That
        // is the `uv.x < 0 || uv.x > 1` branch specifically, which the distance
        // test above cannot reach.
        let inside = shot!(base());
        let beside = shot!(Scene { receiver_at: Vec3::new(300.0, 0.0, 0.0), ..base() });
        for c in 0..3 {
            assert!(
                (inside[c] as i32 - beside[c] as i32).abs() <= 2,
                "off the side of the shadow map went dark: {inside:?} vs {beside:?}",
            );
        }
    }

    #[test]
    fn the_falloff_the_editor_predicts_is_the_falloff_the_shader_applies() {
        // THE OTHER HALF OF A CROSS-LANGUAGE PIN. The editor warns about a
        // light's intensity using its own copy of this curve
        // (scene_editor_web/frontend/src/lib/lightFalloff.js), and a warning
        // computed from the wrong curve is worse than none -- it was authoring
        // against a mismatched preview that put a light of intensity 500 in the
        // lobby in the first place.
        //
        // Neither side transcribes the other's code. Both are asserted against
        // the same NUMBERS, so a change to the shader's attenuation fails here
        // and a change to the editor's fails there.
        //
        // The eye goes far off to the side so the specular lobe contributes
        // essentially nothing and this measures diffuse falloff alone, which is
        // what the editor models.
        let at = |intensity: f32, distance: f32| {
            shade_receiver(Scene {
                lights: vec![Light {
                    position: Vec3::new(0.0, distance, 0.0),
                    direction: Vec3::NEG_Y,
                    kind: LightKind::Point,
                    color: Color3(255, 255, 255, 255),
                    intensity,
                    range: 25.0,
                    cone_angle_deg: 180.0,
                }],
                eye_offset: Vec3::new(400.0, 0.5, 0.0),
                ..base()
            })
        };

        // ambient 0.6 + intensity * window^2 / (d^2 + 1), at d = 5, range = 25.
        for (intensity, expected) in [(1.0f32, 0.6383f32), (3.0, 0.7150), (10.0, 0.9834)] {
            let px = match at(intensity, 5.0) {
                Some(px) => px,
                None => {
                    eprintln!("skipping: no GPU adapter available");
                    return;
                }
            };
            let got = px[0] as f32 / 255.0;
            assert!(
                (got - expected).abs() < 0.01,
                "intensity {intensity} at 5m: shader gave {got:.4}, the editor \
                 predicts {expected:.4}",
            );
        }
    }

    #[test]
    fn the_skys_ambient_follows_the_surface_normal() {
        // WHAT NINE COEFFICIENTS BUY OVER ONE. The old ambient was a constant:
        // every surface received the same grey whichever way it faced, which is
        // why an unlit blockout reads as flat. A sky bright on one side must now
        // light the surfaces facing it more than those facing away, with no
        // light in the scene at all.
        let bright_half = {
            let mut pano = crate::renderer::sky::Panorama::solid([0.0, 0.0, 0.0], 64, 32);
            for y in 0..32 {
                for x in 0..32 {
                    let i = ((y * 64 + x) * 3) as usize;
                    pano.rgb[i] = 3.0;
                    pano.rgb[i + 1] = 3.0;
                    pano.rgb[i + 2] = 3.0;
                }
            }
            SkyUpload::from(&crate::renderer::sky::project_irradiance(&pano, 0.0, 1.0))
        };

        let toward = shot!(Scene {
            lights: vec![],
            normal: [-1.0, 0.0, 0.0],
            sky: bright_half.clone(),
            ..base()
        });
        let away = shot!(Scene {
            lights: vec![],
            normal: [1.0, 0.0, 0.0],
            sky: bright_half,
            ..base()
        });
        assert!(
            toward[0] as i32 - away[0] as i32 > 20,
            "the ambient did not follow the normal: facing the bright half gave \
             {toward:?}, facing away gave {away:?}",
        );
    }

    #[test]
    fn the_shader_evaluates_the_same_irradiance_the_cpu_does() {
        // THE CROSS-IMPLEMENTATION PIN, and the reason it exists: the
        // coefficients are projected in Rust and evaluated in WGSL, and those
        // are two separate transcriptions of the same nine constants. Every
        // other test here predicts the shader's output using the Rust side, so
        // if the two drift, they all keep passing and the headset renders
        // something else.
        //
        // Deliberately breaking the shader's l=1 band from 2/3 to 1.0 changed
        // nothing in this file until this test existed -- the directional test
        // below only asks which side is brighter, which a scaled band survives.
        let pano = {
            let mut p = crate::renderer::sky::Panorama::solid([0.05, 0.05, 0.05], 64, 32);
            // A bright quadrant, so bands 1 and 2 both carry real weight rather
            // than the constant term dominating and hiding a drift in them.
            for y in 4..20 {
                for x in 8..28 {
                    let i = ((y * 64 + x) * 3) as usize;
                    p.rgb[i] = 0.9;
                    p.rgb[i + 1] = 0.7;
                    p.rgb[i + 2] = 0.4;
                }
            }
            p
        };
        let irr = crate::renderer::sky::project_irradiance(&pano, 0.0, 1.0);
        let sky = SkyUpload::from(&irr);

        for normal in [
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, -0.7],
        ] {
            let px = shot!(Scene {
                lights: vec![],
                normal,
                sky: sky.clone(),
                ..base()
            });
            let want = irr.evaluate(normal);
            for c in 0..3 {
                // The target is linear Rgba8Unorm, so the byte IS the value.
                let got = px[c] as f32 / 255.0;
                assert!(
                    (got - want[c]).abs() < 0.012,
                    "normal {normal:?} channel {c}: the shader rendered {got:.4} \
                     and the CPU predicts {:.4}",
                    want[c],
                );
            }
        }
    }

    #[test]
    fn a_scene_with_no_sky_is_lit_exactly_as_before() {
        // The compatibility claim, at the pixel. `SkyUpload::none()` has to
        // evaluate to the old flat constant in every direction, or adding this
        // feature silently re-lights every level that does not use one.
        let up = shot!(Scene { lights: vec![], normal: [0.0, 1.0, 0.0], ..base() });
        let side = shot!(Scene { lights: vec![], normal: [1.0, 0.0, 0.0], ..base() });
        let down = shot!(Scene { lights: vec![], normal: [0.0, -1.0, 0.0], ..base() });
        for (a, b) in [(up, side), (up, down)] {
            for c in 0..3 {
                assert!(
                    (a[c] as i32 - b[c] as i32).abs() <= 1,
                    "flat ambient is no longer flat: {a:?} vs {b:?}",
                );
            }
        }
        let ambient = (crate::renderer::sky::AMBIENT * 255.0).round() as i32;
        assert!(
            (up[0] as i32 - ambient).abs() <= 2,
            "ambient came out {} rather than {ambient}",
            up[0],
        );
    }

    #[test]
    fn specular_depends_on_where_the_eye_is() {
        // Blinn-Phong's whole contribution. With a sun straight down on a
        // surface facing up, the highlight is directly overhead: an eye there
        // sees it and an eye far off to the side does not. If the camera
        // position never reaches the shader this is the test that notices.
        let overhead = shot!(Scene { eye_offset: Vec3::new(0.0, 5.0, 0.0), ..base() });
        let oblique = shot!(Scene { eye_offset: Vec3::new(50.0, 0.5, 0.0), ..base() });
        assert!(
            overhead[0] as i32 - oblique[0] as i32 > 4,
            "the specular highlight did not follow the eye: {overhead:?} vs {oblique:?}",
        );
    }
}

