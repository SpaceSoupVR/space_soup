pub mod camera;
pub mod cuboid;
pub mod icon;
pub mod lights;
pub mod mesh;
pub mod mesh_pipeline;
pub mod mirror;
pub mod panel;
pub mod scope;
pub mod particle;
pub mod pipeline;
pub mod ssr;
pub mod uniforms;

#[cfg(target_os = "android")]
pub mod xr_renderer;

pub use camera::Camera;
pub use cuboid::{Cuboid, CuboidShape, CuboidStyle};
pub use icon::{billboard_rotation, IconAssets, IconKind};
pub use lights::{Light, LightKind};
pub use mesh::GltfMesh;
pub use mirror::MirrorSurface;
pub use panel::WorldPanel;
pub use particle::{Beam, Particle, ParticlePipeline, ParticleVertex};
use std::collections::HashMap;
use wgpu::util::DeviceExt;
use wgpu::*;

use cuboid::{build_solid_mesh_one, build_wire_mesh_one, CuboidSnapshot, SolidVertex, WireVertex};
use lights::LightsUniform;
use mesh::{create_texture_from_rgba, LoadedTexture};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Color3(pub u8, pub u8, pub u8, pub u8);

impl Default for Color3 {
    fn default() -> Self {
        Color3(255, 255, 255, 255)
    }
}

impl Color3 {
    pub fn to_linear(&self) -> [f32; 4] {
        let c = |v: u8| {
            let f = v as f32 / 255.0;
            if f <= 0.04045 {
                f / 12.92
            } else {
                ((f + 0.055) / 1.055).powf(2.4)
            }
        };
        [c(self.0), c(self.1), c(self.2), self.3 as f32 / 255.0]
    }
}

pub struct MeshInstance<'a> {
    pub mesh: &'a GltfMesh,
    pub model: &'a mesh_pipeline::ModelUniform,
    pub lightmap_key: Option<&'a str>,
}

struct CuboidCacheEntry {
    snapshot: CuboidSnapshot,
    solid: Option<(Vec<SolidVertex>, Vec<u32>)>,
    wire: Option<(Vec<WireVertex>, Vec<u32>)>,
}

pub struct Renderer {
    pub device: Device,
    pub queue: Queue,
    solid_pipeline: pipeline::SolidPipeline,
    wire_pipeline: pipeline::WirePipeline,
    mesh_pipeline: mesh_pipeline::MeshPipeline,
    skinned_mesh_pipeline: mesh_pipeline::SkinnedMeshPipeline,
    uniform_buf: uniforms::UniformBuffer,
    lights_uniform: LightsUniform,
    depth_texture: Texture,
    depth_view: TextureView,
    pub width: u32,
    pub height: u32,
    cuboid_cache: HashMap<u64, CuboidCacheEntry>,
    cuboid_lightmaps: HashMap<String, LoadedTexture>,
    default_cuboid_lightmap: LoadedTexture,
    mesh_lightmaps: HashMap<String, LoadedTexture>,
    default_mesh_lightmap: LoadedTexture,
}

impl Renderer {
    pub fn from_device(
        device: Device,
        queue: Queue,
        format: TextureFormat,
        width: u32,
        height: u32,
    ) -> Self {
        let lights_uniform = LightsUniform::new(&device);
        let uniform_buf = uniforms::UniformBuffer::new(&device, &lights_uniform);
        let solid_pipeline = pipeline::SolidPipeline::new(&device, format, &uniform_buf.layout);
        let wire_pipeline = pipeline::WirePipeline::new(&device, format, &uniform_buf.layout);
        let mesh_pipeline = mesh_pipeline::MeshPipeline::new(&device, format, &uniform_buf.layout);
        let skinned_mesh_pipeline =
            mesh_pipeline::SkinnedMeshPipeline::new(&device, format, &uniform_buf.layout);
        let (depth_texture, depth_view) = Self::make_depth(&device, width, height);
        let white_pixel = [255u8, 255, 255, 255];
        let default_cuboid_lightmap =
            create_texture_from_rgba(&device, &queue, &solid_pipeline.lightmap_layout, &white_pixel, 1, 1);
        let default_mesh_lightmap =
            create_texture_from_rgba(&device, &queue, &mesh_pipeline.lightmap_layout, &white_pixel, 1, 1);

        Self {
            device,
            queue,
            solid_pipeline,
            wire_pipeline,
            mesh_pipeline,
            skinned_mesh_pipeline,
            uniform_buf,
            lights_uniform,
            depth_texture,
            depth_view,
            width,
            height,
            cuboid_cache: HashMap::new(),
            cuboid_lightmaps: HashMap::new(),
            default_cuboid_lightmap,
            mesh_lightmaps: HashMap::new(),
            default_mesh_lightmap,
        }
    }

    pub fn set_cuboid_lightmap(&mut self, key: &str, rgba: &[u8], width: u32, height: u32) {
        let tex = create_texture_from_rgba(&self.device, &self.queue, &self.solid_pipeline.lightmap_layout, rgba, width, height);
        self.cuboid_lightmaps.insert(key.to_string(), tex);
    }

    pub fn set_mesh_lightmap(&mut self, key: &str, rgba: &[u8], width: u32, height: u32) {
        let tex = create_texture_from_rgba(&self.device, &self.queue, &self.mesh_pipeline.lightmap_layout, rgba, width, height);
        self.mesh_lightmaps.insert(key.to_string(), tex);
    }

    pub fn mesh_texture_layout(&self) -> &BindGroupLayout {
        &self.mesh_pipeline.texture_layout
    }

    pub fn mesh_pipeline(&self) -> &mesh_pipeline::MeshPipeline {
        &self.mesh_pipeline
    }

    pub fn create_model_uniform(&self) -> mesh_pipeline::ModelUniform {
        self.mesh_pipeline.create_model_uniform(&self.device)
    }

    pub fn create_skinned_model_uniform(&self) -> mesh_pipeline::ModelUniform {
        self.skinned_mesh_pipeline.create_model_uniform(&self.device)
    }

    pub fn skin_joint_layout(&self) -> &BindGroupLayout {
        &self.skinned_mesh_pipeline.skin_joint_layout
    }

    pub fn create_icon_assets(&self) -> icon::IconAssets {
        icon::IconAssets::new(&self.device, &self.queue, &self.mesh_pipeline.texture_layout)
    }

    pub fn create_panel(
        &self,
        texture_format: TextureFormat,
        width_px: u32,
        height_px: u32,
        width_m: f32,
        height_m: f32,
    ) -> WorldPanel {
        WorldPanel::new(
            &self.device,
            &self.queue,
            texture_format,
            &self.mesh_pipeline,
            width_px,
            height_px,
            width_m,
            height_m,
        )
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
        let (t, v) = Self::make_depth(&self.device, width, height);
        self.depth_texture = t;
        self.depth_view = v;
    }

    pub fn render(&mut self, target_view: &TextureView, camera: &Camera, cuboids: &[Cuboid]) {
        self.render_with_meshes(target_view, camera, cuboids, &[]);
    }
    pub fn render_with_meshes(
        &mut self,
        target_view: &TextureView,
        camera: &Camera,
        cuboids: &[Cuboid],
        meshes: &[MeshInstance],
    ) {
        self.render_with_lights(target_view, camera, cuboids, meshes, &[]);
    }

    pub fn render_with_lights(
        &mut self,
        target_view: &TextureView,
        camera: &Camera,
        cuboids: &[Cuboid],
        meshes: &[MeshInstance],
        lights: &[lights::Light],
    ) {
        self.render_internal(target_view, camera, cuboids, meshes, &[], lights);
    }

    pub fn render_with_panels(
        &mut self,
        target_view: &TextureView,
        camera: &Camera,
        cuboids: &[Cuboid],
        meshes: &[MeshInstance],
        panels: &[&WorldPanel],
        lights: &[lights::Light],
    ) {
        self.render_internal(target_view, camera, cuboids, meshes, panels, lights);
    }

    #[allow(clippy::type_complexity)]
    fn bake_cuboids(
        &mut self,
        cuboids: &[Cuboid],
    ) -> (
        (Vec<SolidVertex>, Vec<u32>, Vec<(Option<String>, u32, u32)>),
        (Vec<WireVertex>, Vec<u32>),
    ) {
        let mut seen: std::collections::HashSet<u64> =
            std::collections::HashSet::with_capacity(cuboids.len());

        let mut solid_verts: Vec<SolidVertex> = Vec::new();
        let mut solid_indices: Vec<u32> = Vec::new();
        let mut solid_ranges: Vec<(Option<String>, u32, u32)> = Vec::new();
        let mut wire_verts: Vec<WireVertex> = Vec::new();
        let mut wire_indices: Vec<u32> = Vec::new();

        for c in cuboids {
            seen.insert(c.id);
            let snapshot = c.snapshot();

            let needs_rebuild = match self.cuboid_cache.get(&c.id) {
                Some(entry) => entry.snapshot != snapshot,
                None => true,
            };

            if needs_rebuild {
                let entry = CuboidCacheEntry {
                    snapshot,
                    solid: build_solid_mesh_one(c),
                    wire: build_wire_mesh_one(c),
                };
                self.cuboid_cache.insert(c.id, entry);
            }

            let entry = self
                .cuboid_cache
                .get(&c.id)
                .expect("just inserted or already present");

            if let Some((v, i)) = &entry.solid {
                let base = solid_verts.len() as u32;
                let index_start = solid_indices.len() as u32;
                solid_verts.extend_from_slice(v);
                solid_indices.extend(i.iter().map(|x| x + base));
                solid_ranges.push((c.lightmap_key.clone(), index_start, i.len() as u32));
            }
            if let Some((v, i)) = &entry.wire {
                let base = wire_verts.len() as u32;
                wire_verts.extend_from_slice(v);
                wire_indices.extend(i.iter().map(|x| x + base));
            }
        }

        self.cuboid_cache.retain(|id, _| seen.contains(id));

        (
            (solid_verts, solid_indices, solid_ranges),
            (wire_verts, wire_indices),
        )
    }

    fn render_internal(
        &mut self,
        target_view: &TextureView,
        camera: &Camera,
        cuboids: &[Cuboid],
        meshes: &[MeshInstance],
        panels: &[&WorldPanel],
        lights: &[lights::Light],
    ) {
        let vp = camera.projection() * camera.view();
        self.uniform_buf.upload(&self.queue, vp);
        self.lights_uniform.upload(&self.queue, lights);

        let ((solid_verts, solid_indices, solid_ranges), (wire_verts, wire_indices)) =
            self.bake_cuboids(cuboids);

        let solid_vb = self.device.create_buffer_init(&util::BufferInitDescriptor {
            label: Some("solid_vb"),
            contents: bytemuck::cast_slice(&solid_verts),
            usage: BufferUsages::VERTEX,
        });
        let solid_ib = self.device.create_buffer_init(&util::BufferInitDescriptor {
            label: Some("solid_ib"),
            contents: bytemuck::cast_slice(&solid_indices),
            usage: BufferUsages::INDEX,
        });
        let wire_vb = self.device.create_buffer_init(&util::BufferInitDescriptor {
            label: Some("wire_vb"),
            contents: bytemuck::cast_slice(&wire_verts),
            usage: BufferUsages::VERTEX,
        });
        let wire_ib = self.device.create_buffer_init(&util::BufferInitDescriptor {
            label: Some("wire_ib"),
            contents: bytemuck::cast_slice(&wire_indices),
            usage: BufferUsages::INDEX,
        });

        let mut panel_buffers: Vec<(Buffer, Buffer)> = Vec::with_capacity(panels.len());
        for panel in panels {
            panel.upload_model(&self.queue);
            let vb = self.device.create_buffer_init(&util::BufferInitDescriptor {
                label: Some("panel_vb"),
                contents: bytemuck::cast_slice(panel.vertices()),
                usage: BufferUsages::VERTEX,
            });
            let ib = self.device.create_buffer_init(&util::BufferInitDescriptor {
                label: Some("panel_ib"),
                contents: bytemuck::cast_slice(panel.indices()),
                usage: BufferUsages::INDEX,
            });
            panel_buffers.push((vb, ib));
        }

        let mut draws: Vec<(&Buffer, &Buffer, u32, &BindGroup, &BindGroup, &BindGroup)> = Vec::new();
        let mut skinned_draws: Vec<(&Buffer, &Buffer, u32, &BindGroup, &BindGroup, &BindGroup)> =
            Vec::new();

        for instance in meshes {
            instance
                .model
                .upload(&self.queue, instance.mesh.model_matrix());

            if let Some(skin) = &instance.mesh.skin {
                if let Some(joint_bg) = &skin.joint_bind_group {
                    for prim in &skin.primitives {
                        skinned_draws.push((
                            &prim.vertex_buffer,
                            &prim.index_buffer,
                            prim.indices.len() as u32,
                            &instance.model.bind_group,
                            &prim.texture.bind_group,
                            joint_bg,
                        ));
                    }
                }
            } else {
                let lightmap_bg = instance
                    .lightmap_key
                    .and_then(|k| self.mesh_lightmaps.get(k))
                    .map(|t| &t.bind_group)
                    .unwrap_or(&self.default_mesh_lightmap.bind_group);
                for prim in &instance.mesh.primitives {
                    draws.push((
                        &prim.vertex_buffer,
                        &prim.index_buffer,
                        prim.indices.len() as u32,
                        &instance.model.bind_group,
                        &prim.texture.bind_group,
                        lightmap_bg,
                    ));
                }
            }
        }

        for (panel, (vb, ib)) in panels.iter().zip(panel_buffers.iter()) {
            draws.push((
                vb,
                ib,
                panel.indices().len() as u32,
                &panel.model.bind_group,
                panel.bind_group(),
                &self.default_mesh_lightmap.bind_group,
            ));
        }

        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("frame"),
            });

        {
            let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("3d_pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: target_view,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(wgpu::Color {
                            r: 0.02,
                            g: 0.02,
                            b: 0.05,
                            a: 1.0,
                        }),
                        store: StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(Operations {
                        load: LoadOp::Clear(1.0),
                        store: StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });

            if !solid_verts.is_empty() {
                pass.set_pipeline(&self.solid_pipeline.pipeline);
                pass.set_bind_group(0, &self.uniform_buf.bind_group, &[]);
                pass.set_vertex_buffer(0, solid_vb.slice(..));
                pass.set_index_buffer(solid_ib.slice(..), IndexFormat::Uint32);
                for (lightmap_key, index_start, count) in &solid_ranges {
                    let lightmap_bg = lightmap_key
                        .as_deref()
                        .and_then(|k| self.cuboid_lightmaps.get(k))
                        .map(|t| &t.bind_group)
                        .unwrap_or(&self.default_cuboid_lightmap.bind_group);
                    pass.set_bind_group(1, lightmap_bg, &[]);
                    pass.draw_indexed(*index_start..*index_start + *count, 0, 0..1);
                }
            }

            if !wire_verts.is_empty() {
                pass.set_pipeline(&self.wire_pipeline.pipeline);
                pass.set_bind_group(0, &self.uniform_buf.bind_group, &[]);
                pass.set_vertex_buffer(0, wire_vb.slice(..));
                pass.set_index_buffer(wire_ib.slice(..), IndexFormat::Uint32);
                pass.draw_indexed(0..wire_indices.len() as u32, 0, 0..1);
            }

            if !draws.is_empty() {
                pass.set_pipeline(&self.mesh_pipeline.pipeline);
                pass.set_bind_group(0, &self.uniform_buf.bind_group, &[]);
                for (vb, ib, count, model_bg, tex_bg, lightmap_bg) in &draws {
                    pass.set_bind_group(1, *model_bg, &[]);
                    pass.set_bind_group(2, *tex_bg, &[]);
                    pass.set_bind_group(3, *lightmap_bg, &[]);
                    pass.set_vertex_buffer(0, vb.slice(..));
                    pass.set_index_buffer(ib.slice(..), IndexFormat::Uint32);
                    pass.draw_indexed(0..*count, 0, 0..1);
                }
            }

            if !skinned_draws.is_empty() {
                pass.set_pipeline(&self.skinned_mesh_pipeline.pipeline);
                pass.set_bind_group(0, &self.uniform_buf.bind_group, &[]);
                for (vb, ib, count, model_bg, tex_bg, joint_bg) in &skinned_draws {
                    pass.set_bind_group(1, *model_bg, &[]);
                    pass.set_bind_group(2, *tex_bg, &[]);
                    pass.set_bind_group(3, *joint_bg, &[]);
                    pass.set_vertex_buffer(0, vb.slice(..));
                    pass.set_index_buffer(ib.slice(..), IndexFormat::Uint32);
                    pass.draw_indexed(0..*count, 0, 0..1);
                }
            }
        }

        self.queue.submit(Some(encoder.finish()));
    }

    fn make_depth(device: &Device, width: u32, height: u32) -> (Texture, TextureView) {
        let tex = device.create_texture(&TextureDescriptor {
            label: Some("depth"),
            size: Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Depth32Float,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = tex.create_view(&TextureViewDescriptor::default());
        (tex, view)
    }
}


#[cfg(test)]
mod multiview_capability_tests {
    //! C3 (single-pass stereo) gate.
    //!
    //! Multiview is a *device* capability before it is a rendering technique:
    //! the VkDevice must enable `PhysicalDeviceMultiviewFeatures` (see
    //! `crate::xr::vulkan`), wgpu must be told about it on both halves of the
    //! hal adoption, and only then can a pipeline declare `multiview`.
    //!
    //! Like every WGSL/pipeline property in wgpu, that declaration is validated
    //! at **pipeline creation**, not at `cargo build` — so a clean compile
    //! proves nothing. These tests build a real multiview pipeline inside a
    //! validation error scope, which is the only way to find out before the
    //! headset does.

    fn multiview_device() -> Option<(wgpu::Device, wgpu::Queue, bool)> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok()?;
        let supported = adapter.features().contains(wgpu::Features::MULTIVIEW);
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_features: if supported {
                wgpu::Features::MULTIVIEW
            } else {
                wgpu::Features::empty()
            },
            required_limits: wgpu::Limits::default(),
            ..Default::default()
        }))
        .ok()?;
        Some((device, queue, supported))
    }

    /// Reports whether this machine can validate C3 locally at all. Never fails:
    /// a desktop adapter without multiview is a fact about the test machine, not
    /// a defect in the renderer. Quest's Adreno does support it.
    #[test]
    fn report_whether_this_adapter_can_validate_multiview() {
        let Some((_d, _q, supported)) = multiview_device() else {
            eprintln!("skipping: no GPU adapter available in this environment");
            return;
        };
        if supported {
            eprintln!("multiview: SUPPORTED on this adapter — C3 pipelines are locally verifiable");
        } else {
            eprintln!(
                "multiview: NOT supported on this adapter — C3 pipeline validation can only \
                 happen on device. Shader and uniform restructuring is still testable here."
            );
        }
    }

    /// A multiview pipeline plus a `@builtin(view_index)` shader must survive
    /// naga validation and pipeline creation. This is the shape every C3
    /// pipeline takes, so if it builds, the approach is sound.
    #[test]
    fn a_two_view_pipeline_with_view_index_builds_on_a_real_device() {
        let Some((device, _queue, supported)) = multiview_device() else {
            eprintln!("skipping: no GPU adapter available in this environment");
            return;
        };
        if !supported {
            eprintln!("skipping: adapter lacks MULTIVIEW (expected on some desktop GPUs)");
            return;
        }

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("multiview_probe"),
            source: wgpu::ShaderSource::Wgsl(
                r#"
// Per-eye matrices indexed by view. This is exactly how the real pipelines
// will pick their eye: one draw, two views, no CPU-side loop.
struct Eyes { view_proj: array<mat4x4<f32>, 2>, };
@group(0) @binding(0) var<uniform> eyes: Eyes;

@vertex
fn vs(
    @location(0) pos: vec3<f32>,
    @builtin(view_index) view: i32,
) -> @builtin(position) vec4<f32> {
    return eyes.view_proj[view] * vec4<f32>(pos, 1.0);
}

@fragment
fn fs() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 1.0, 1.0, 1.0);
}
"#
                .into(),
            ),
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
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let _pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("multiview_probe_pipeline"),
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
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: std::num::NonZeroU32::new(2),
            cache: None,
        });
        let err = pollster::block_on(device.pop_error_scope());
        assert!(err.is_none(), "multiview pipeline failed validation: {err:?}");
    }

    /// The real `SolidPipeline` must build in BOTH variants: single-view for the
    /// offscreen passes (scope world view, mirror reflection, which render into
    /// single-layer targets) and multiview for the eye pass. They share a bind
    /// group layout because the camera uniform is always `array<mat4x4, 2>` —
    /// single-view shaders read slot 0, multiview indexes by `view_index`.
    #[test]
    fn the_solid_pipeline_builds_single_view_and_multiview() {
        use crate::renderer::lights::LightsUniform;
        use crate::renderer::pipeline::SolidPipeline;
        use crate::renderer::uniforms::UniformBuffer;

        let Some((device, _queue, supported)) = multiview_device() else {
            eprintln!("skipping: no GPU adapter available in this environment");
            return;
        };
        let format = wgpu::TextureFormat::Rgba8Unorm;
        let lights = LightsUniform::new(&device);
        let uniforms = UniformBuffer::new(&device, &lights);

        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let _single = SolidPipeline::new(&device, format, &uniforms.layout);
        let err = pollster::block_on(device.pop_error_scope());
        assert!(err.is_none(), "single-view solid pipeline failed: {err:?}");

        if !supported {
            eprintln!("skipping multiview half: adapter lacks MULTIVIEW");
            return;
        }
        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let _stereo = SolidPipeline::new_multiview(&device, format, &uniforms.layout);
        let err = pollster::block_on(device.pop_error_scope());
        assert!(err.is_none(), "multiview solid pipeline failed: {err:?}");
    }

    /// Every pipeline the eye pass draws with must have a working multiview
    /// variant, or the pass cannot be collapsed to one draw per object.
    ///
    /// This builds them all under a validation scope, which is the only place
    /// multiview is actually checked — a clean `cargo build` says nothing about
    /// whether `@builtin(view_index)` survived naga or whether the pipeline is
    /// legal against a 2-layer attachment.
    #[test]
    fn every_eye_pass_pipeline_has_a_valid_multiview_variant() {
        use crate::renderer::lights::LightsUniform;
        use crate::renderer::mesh_pipeline::{MeshPipeline, SkinnedMeshPipeline};
        use crate::renderer::particle::ParticlePipeline;
        use crate::renderer::pipeline::{SolidPipeline, WirePipeline};
        use crate::renderer::uniforms::UniformBuffer;

        let Some((device, _queue, supported)) = multiview_device() else {
            eprintln!("skipping: no GPU adapter available in this environment");
            return;
        };
        if !supported {
            eprintln!("skipping: adapter lacks MULTIVIEW");
            return;
        }
        let format = wgpu::TextureFormat::Rgba8Unorm;
        let lights = LightsUniform::new(&device);
        let u = UniformBuffer::new(&device, &lights);

        // One scope per pipeline so a failure names the culprit.
        macro_rules! check {
            ($label:literal, $build:expr) => {{
                device.push_error_scope(wgpu::ErrorFilter::Validation);
                let _p = $build;
                let err = pollster::block_on(device.pop_error_scope());
                assert!(err.is_none(), concat!($label, " multiview variant failed: {:?}"), err);
            }};
        }

        check!("solid", SolidPipeline::new_multiview(&device, format, &u.layout));
        check!("wire", WirePipeline::new_multiview(&device, format, &u.layout));
        check!("mesh", MeshPipeline::new_multiview(&device, format, &u.layout));
        check!("skinned", SkinnedMeshPipeline::new_multiview(&device, format, &u.layout));
        check!("particle", ParticlePipeline::new_multiview(&device, format, &u.layout));
    }

    /// The offscreen passes must stay single-view: the scope world view and the
    /// mirror reflection render into single-layer targets, where a multiview
    /// pipeline is invalid. They share the camera uniform layout with the
    /// stereo variants because it is always `array<mat4x4, 2>`.
    #[test]
    fn offscreen_pipelines_remain_single_view() {
        use crate::renderer::lights::LightsUniform;
        use crate::renderer::mesh_pipeline::MeshPipeline;
        use crate::renderer::pipeline::SolidPipeline;
        use crate::renderer::uniforms::UniformBuffer;

        let Some((device, _queue, _supported)) = multiview_device() else {
            eprintln!("skipping: no GPU adapter available in this environment");
            return;
        };
        let format = wgpu::TextureFormat::Rgba8Unorm;
        let lights = LightsUniform::new(&device);
        let u = UniformBuffer::new(&device, &lights);

        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let _solid = SolidPipeline::new(&device, format, &u.layout);
        let _mirror_solid = SolidPipeline::new_mirror(&device, format, &u.layout);
        let _mesh = MeshPipeline::new(&device, format, &u.layout);
        let _mirror_mesh = MeshPipeline::new_mirror(&device, format, &u.layout);
        let err = pollster::block_on(device.pop_error_scope());
        assert!(err.is_none(), "single-view offscreen pipelines failed: {err:?}");
    }
}
