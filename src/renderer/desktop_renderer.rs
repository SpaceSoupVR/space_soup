use std::collections::HashMap;
use wgpu::util::DeviceExt;
use wgpu::*;

use super::cuboid::{build_solid_mesh_one, build_solid_mesh_with_ranges, build_wire_mesh_one, CuboidSnapshot, SolidVertex, WireVertex};
use super::lights::LightsUniform;
use super::mesh::{create_texture_from_rgba, LoadedTexture};
use super::{icon, lights, mesh_pipeline, pipeline, uniforms};
use super::{Camera, Cuboid, MeshInstance, WorldPanel};

struct CuboidCacheEntry {
    snapshot: CuboidSnapshot,
    solid: Option<(Vec<SolidVertex>, Vec<u32>)>,
    wire: Option<(Vec<WireVertex>, Vec<u32>)>,
}

pub struct Renderer {
    pub device: Device,
    pub queue: Queue,
    format: TextureFormat,
    /// Color the 3D pass clears to. An alpha below 1.0 lets the OS composite
    /// whatever is behind the window through (used by the editor's
    /// transparent/vibrancy chrome); the default is the old opaque deep blue.
    pub clear_color: wgpu::Color,
    solid_pipeline: pipeline::SolidPipeline,
    wire_pipeline: pipeline::WirePipeline,
    overlay_pipeline: pipeline::SolidPipeline,
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
        let overlay_pipeline = pipeline::SolidPipeline::new_overlay(&device, format, &uniform_buf.layout);
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
            format,
            clear_color: wgpu::Color {
                r: 0.02,
                g: 0.02,
                b: 0.05,
                a: 1.0,
            },
            solid_pipeline,
            wire_pipeline,
            overlay_pipeline,
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

    /// Editor nicety: render solid cuboids double-sided so a camera flown
    /// inside one sees its interior instead of an x-ray hole.
    pub fn set_cuboids_double_sided(&mut self) {
        self.solid_pipeline =
            pipeline::SolidPipeline::new_double_sided(&self.device, self.format, &self.uniform_buf.layout);
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
        self.render_internal(target_view, camera, cuboids, meshes, &[], lights, &[]);
    }

    /// Like `render_with_lights`, but `overlay_cuboids` draw in an extra
    /// pass at the very end, through an "always passes the depth test"
    /// pipeline (see `pipeline::SolidPipeline::new_overlay`) -- so they
    /// read on top of everything else (meshes included) instead of being
    /// hidden inside whatever solid geometry already occupies that space.
    /// Meant for a highlight overlay (e.g. a skeleton), not real geometry.
    pub fn render_with_overlay(
        &mut self,
        target_view: &TextureView,
        camera: &Camera,
        cuboids: &[Cuboid],
        overlay_cuboids: &[Cuboid],
        meshes: &[MeshInstance],
        lights: &[lights::Light],
    ) {
        self.render_internal(target_view, camera, cuboids, meshes, &[], lights, overlay_cuboids);
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
        self.render_internal(target_view, camera, cuboids, meshes, panels, lights, &[]);
    }

    /// Renders just `meshes` (no cuboids, no overlay) to an off-screen
    /// `width`x`height` RGBA8 texture and reads it back to CPU memory --
    /// for a timeline filmstrip's thumbnails, which want a handful of real
    /// snapshots of a posed mesh, not one static icon repeated. Blocks
    /// (polls the device to completion) instead of returning a future:
    /// meant to run a few times whenever a clip's duration/keyframes
    /// change, not per frame, so a few milliseconds of stall here is an
    /// acceptable trade for not needing an async executor in the engine
    /// thread's otherwise-synchronous render loop.
    pub fn render_mesh_thumbnail(
        &mut self,
        camera: &Camera,
        meshes: &[MeshInstance],
        lights: &[lights::Light],
        width: u32,
        height: u32,
    ) -> Vec<u8> {
        let color_tex = self.device.create_texture(&TextureDescriptor {
            label: Some("thumb_color"),
            size: Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: self.format,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let color_view = color_tex.create_view(&TextureViewDescriptor::default());
        let (_depth_tex, depth_view) = Self::make_depth(&self.device, width, height);

        let vp = camera.projection() * camera.view();
        self.uniform_buf.upload(&self.queue, vp);
        self.lights_uniform.upload(&self.queue, lights);

        let mut draws: Vec<(&Buffer, &Buffer, u32, &BindGroup, &BindGroup, &BindGroup)> = Vec::new();
        let mut skinned_draws: Vec<(&Buffer, &Buffer, u32, &BindGroup, &BindGroup, &BindGroup)> =
            Vec::new();
        for instance in meshes {
            instance
                .model
                .upload(&self.queue, instance.mesh.model_matrix());
            instance.model.upload_eye(&self.queue, camera.position);
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

        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("thumb_frame"),
            });
        {
            let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("thumb_pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &color_view,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(self.clear_color),
                        store: StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(Operations {
                        load: LoadOp::Clear(1.0),
                        store: StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });

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

        // Copy to a mappable buffer -- rows must be padded to a 256-byte
        // stride (`COPY_BYTES_PER_ROW_ALIGNMENT`), unpadded again below.
        let unpadded_bytes_per_row = width * 4;
        let padded_bytes_per_row =
            unpadded_bytes_per_row.div_ceil(COPY_BYTES_PER_ROW_ALIGNMENT) * COPY_BYTES_PER_ROW_ALIGNMENT;
        let buffer_size = (padded_bytes_per_row * height) as BufferAddress;
        let out_buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some("thumb_readback"),
            size: buffer_size,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            TexelCopyTextureInfo {
                texture: &color_tex,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            TexelCopyBufferInfo {
                buffer: &out_buffer,
                layout: TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: Some(height),
                },
            },
            Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        self.queue.submit(Some(encoder.finish()));

        let slice = out_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        let _ = self.device.poll(PollType::Wait);
        let _ = rx.recv();

        let data = slice.get_mapped_range();
        let mut out = Vec::with_capacity((unpadded_bytes_per_row * height) as usize);
        for row in 0..height {
            let start = (row * padded_bytes_per_row) as usize;
            out.extend_from_slice(&data[start..start + unpadded_bytes_per_row as usize]);
        }
        drop(data);
        out_buffer.unmap();
        // The swapchain format on this platform is commonly BGRA, not
        // RGBA -- PNG/the `image` crate wants RGBA byte order, so swap R
        // and B per pixel when that's what we actually rendered into.
        if matches!(
            self.format,
            TextureFormat::Bgra8Unorm | TextureFormat::Bgra8UnormSrgb
        ) {
            for px in out.chunks_exact_mut(4) {
                px.swap(0, 2);
            }
        }
        out
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

    #[allow(clippy::too_many_arguments)]
    fn render_internal(
        &mut self,
        target_view: &TextureView,
        camera: &Camera,
        cuboids: &[Cuboid],
        meshes: &[MeshInstance],
        panels: &[&WorldPanel],
        lights: &[lights::Light],
        overlay_cuboids: &[Cuboid],
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

        // Overlay cuboids aren't cached (there are only ever a handful --
        // a skeleton's worth -- and they're rebuilt every frame anyway).
        let (overlay_verts, overlay_indices, _) = build_solid_mesh_with_ranges(overlay_cuboids);
        let overlay_vb = self.device.create_buffer_init(&util::BufferInitDescriptor {
            label: Some("overlay_vb"),
            contents: bytemuck::cast_slice(&overlay_verts),
            usage: BufferUsages::VERTEX,
        });
        let overlay_ib = self.device.create_buffer_init(&util::BufferInitDescriptor {
            label: Some("overlay_ib"),
            contents: bytemuck::cast_slice(&overlay_indices),
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
            instance.model.upload_eye(&self.queue, camera.position);

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
                        load: LoadOp::Clear(self.clear_color),
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

            // Last, so it reads on top of everything above (see
            // `SolidPipeline::new_overlay`'s own doc comment).
            if !overlay_verts.is_empty() {
                pass.set_pipeline(&self.overlay_pipeline.pipeline);
                pass.set_bind_group(0, &self.uniform_buf.bind_group, &[]);
                pass.set_bind_group(1, &self.default_cuboid_lightmap.bind_group, &[]);
                pass.set_vertex_buffer(0, overlay_vb.slice(..));
                pass.set_index_buffer(overlay_ib.slice(..), IndexFormat::Uint32);
                pass.draw_indexed(0..overlay_indices.len() as u32, 0, 0..1);
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
