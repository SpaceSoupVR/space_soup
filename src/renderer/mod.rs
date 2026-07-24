pub mod camera;
pub mod cuboid;
pub mod icon;
pub mod lights;
pub mod mesh;
pub mod mesh_pipeline;
pub mod mirror;
pub mod panel;
pub mod particle;
pub mod pipeline;
pub mod uniforms;

#[cfg(target_os = "android")]
pub mod xr_renderer;

pub use camera::Camera;
pub use cuboid::{Cuboid, CuboidStyle};
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
        }
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

    fn bake_cuboids(
        &mut self,
        cuboids: &[Cuboid],
    ) -> ((Vec<SolidVertex>, Vec<u32>), (Vec<WireVertex>, Vec<u32>)) {
        let mut seen: std::collections::HashSet<u64> =
            std::collections::HashSet::with_capacity(cuboids.len());

        let mut solid_verts: Vec<SolidVertex> = Vec::new();
        let mut solid_indices: Vec<u32> = Vec::new();
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
                solid_verts.extend_from_slice(v);
                solid_indices.extend(i.iter().map(|x| x + base));
            }
            if let Some((v, i)) = &entry.wire {
                let base = wire_verts.len() as u32;
                wire_verts.extend_from_slice(v);
                wire_indices.extend(i.iter().map(|x| x + base));
            }
        }

        self.cuboid_cache.retain(|id, _| seen.contains(id));

        ((solid_verts, solid_indices), (wire_verts, wire_indices))
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

        let ((solid_verts, solid_indices), (wire_verts, wire_indices)) = self.bake_cuboids(cuboids);

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

        let mut draws: Vec<(&Buffer, &Buffer, u32, &BindGroup, &BindGroup)> = Vec::new();
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
                for prim in &instance.mesh.primitives {
                    draws.push((
                        &prim.vertex_buffer,
                        &prim.index_buffer,
                        prim.indices.len() as u32,
                        &instance.model.bind_group,
                        &prim.texture.bind_group,
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
                pass.draw_indexed(0..solid_indices.len() as u32, 0, 0..1);
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
                for (vb, ib, count, model_bg, tex_bg) in &draws {
                    pass.set_bind_group(1, *model_bg, &[]);
                    pass.set_bind_group(2, *tex_bg, &[]);
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

