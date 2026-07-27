pub mod camera;
pub mod cuboid;
pub mod icon;
pub mod lights;
pub mod mesh;
pub mod mesh_pipeline;
pub mod mirror;
pub mod offscreen;
pub mod panel;
pub mod pipeline;
pub mod shadow;
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
    shadow_map: shadow::ShadowMap,
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
        let shadow_map = shadow::ShadowMap::new(&device);
        let uniform_buf = uniforms::UniformBuffer::new(
            &device,
            &lights_uniform,
            shadow_map.sun_depth_view(),
            shadow_map.spot_depth_view(),
            shadow_map.sampler(),
        );
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
            shadow_map,
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

    /// Skinned meshes use their own pipeline/layouts (see `SkinnedMeshPipeline`), so a
    /// `ModelUniform` created here is **not** interchangeable with `create_model_uniform()`'s —
    /// use this one for any `GltfMesh` that has a `skin`.
    pub fn create_skinned_model_uniform(&self) -> mesh_pipeline::ModelUniform {
        self.skinned_mesh_pipeline.create_model_uniform(&self.device)
    }

    /// Bind group layout needed to call `GltfMesh::create_skin_bind_group` on a loaded skinned
    /// mesh before it can be drawn.
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
                            prim.index_count,
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

        // --- Shadows: the directional light gets an orthographic sun shadow
        // framed around the scene bounds; the first spot light is treated as
        // the shadow-casting flashlight and gets a perspective shadow. Both
        // fold their depth textures into the shared camera/lights group.
        let (center, radius) = scene_bounds(&solid_verts, meshes);
        let sun = lights.iter().find(|l| l.kind == LightKind::Directional);
        let sun_vp = sun
            .map(|l| shadow::directional_light_matrix(l.direction, center, radius))
            .unwrap_or(glam::Mat4::IDENTITY);

        let flashlight = lights
            .iter()
            .enumerate()
            .find(|(_, l)| l.kind == LightKind::Spot);
        let (spot_vp, flashlight_index) = match flashlight {
            Some((idx, l)) => (
                shadow::spot_light_matrix(l.position, l.direction, l.cone_angle_deg, l.range),
                idx as u32,
            ),
            None => (glam::Mat4::IDENTITY, 0),
        };

        let shadow_upload = uniforms::ShadowUpload {
            sun_view_proj: sun_vp,
            spot_view_proj: spot_vp,
            sun_enabled: sun.is_some(),
            spot_enabled: flashlight.is_some(),
            flashlight_index,
        };
        self.uniform_buf
            .upload(&self.queue, vp, camera.position, &shadow_upload);

        // Mesh casters for the shadow passes reuse the main pass's model bind
        // groups (skinned meshes are a follow-up); solid cuboids cast too.
        let shadow_mesh_draws: Vec<shadow::ShadowMeshDraw> = draws
            .iter()
            .map(|(vb, ib, count, model_bg, _tex_bg)| (*vb, *ib, *count, *model_bg))
            .collect();
        let solid_shadow = if !solid_verts.is_empty() {
            Some((&solid_vb, &solid_ib, solid_indices.len() as u32))
        } else {
            None
        };

        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("frame"),
            });

        // Depth-only shadow passes, recorded before the main pass samples them.
        if shadow_upload.sun_enabled {
            self.shadow_map
                .upload_light(&self.queue, shadow::ShadowKind::Sun, sun_vp);
            self.shadow_map.record(
                &mut encoder,
                shadow::ShadowKind::Sun,
                solid_shadow,
                &shadow_mesh_draws,
            );
        }
        if shadow_upload.spot_enabled {
            self.shadow_map
                .upload_light(&self.queue, shadow::ShadowKind::Spot, spot_vp);
            self.shadow_map.record(
                &mut encoder,
                shadow::ShadowKind::Spot,
                solid_shadow,
                &shadow_mesh_draws,
            );
        }

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

/// World-space bounding sphere (center, radius) of the shadow-casting content,
/// used to frame the sun's orthographic shadow box. Falls back to a sensible
/// default when there is nothing to bound.
fn scene_bounds(solid_verts: &[SolidVertex], meshes: &[MeshInstance]) -> (glam::Vec3, f32) {
    let mut min = glam::Vec3::splat(f32::INFINITY);
    let mut max = glam::Vec3::splat(f32::NEG_INFINITY);
    let mut any = false;
    for v in solid_verts {
        let p = glam::Vec3::from(v.position);
        min = min.min(p);
        max = max.max(p);
        any = true;
    }
    for m in meshes {
        let t = m.mesh.model_matrix().to_scale_rotation_translation().2;
        min = min.min(t);
        max = max.max(t);
        any = true;
    }
    if !any {
        return (glam::Vec3::ZERO, 20.0);
    }
    let center = (min + max) * 0.5;
    let radius = (max - center).length().max(5.0);
    (center, radius)
}

#[cfg(all(test, not(target_os = "android")))]
mod gpu_smoke {
    use super::*;
    use glam::{Quat, Vec3};

    const DIM: u32 = 256;

    /// Requests a headless device, or returns None (no adapter) so tests skip
    /// gracefully in an environment without a GPU.
    fn headless_device() -> Option<(Device, Queue)> {
        let instance = Instance::new(&InstanceDescriptor::default());
        let adapter = pollster::block_on(instance.request_adapter(&RequestAdapterOptions {
            power_preference: PowerPreference::default(),
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok()?;
        pollster::block_on(adapter.request_device(&DeviceDescriptor {
            required_features: Features::empty(),
            required_limits: Limits::default(),
            ..Default::default()
        }))
        .ok()
    }

    /// Builds every render pipeline on a real adapter, which forces naga to
    /// validate all WGSL shader strings — they are compiled at pipeline
    /// creation time, not by `cargo build`, so this is the only pure-Rust way
    /// to catch a shader error. A validation error scope makes any failure
    /// deterministic.
    #[test]
    fn shaders_and_pipelines_compile_on_gpu() {
        let Some((device, queue)) = headless_device() else {
            eprintln!("no GPU adapter; skipping shader validation");
            return;
        };
        let probe = device.clone();
        probe.push_error_scope(ErrorFilter::Validation);
        let _renderer = Renderer::from_device(device, queue, TextureFormat::Rgba8UnormSrgb, DIM, DIM);
        if let Some(err) = pollster::block_on(probe.pop_error_scope()) {
            panic!("shader/pipeline validation failed: {err:?}");
        }
    }

    fn grey_cuboid(id: u64, pos: Vec3, half: Vec3) -> Cuboid {
        Cuboid {
            position: pos,
            half_size: half,
            rotation: Quat::IDENTITY,
            color: Color3(200, 200, 200, 255),
            wire_color: Color3(0, 0, 0, 255),
            style: CuboidStyle::Solid,
            id,
        }
    }

    /// Renders `cuboids` under `lights` from a top-down camera and returns
    /// (total luminance, count of "shadowed floor" pixels). A shadowed-floor
    /// pixel is mid-brightness: darker than the fully-lit (clamped) floor but
    /// clearly brighter than the dark background — i.e. floor lit by ambient
    /// only. That count is a direct measure of shadow area, unconfounded by an
    /// occluder's own (bright) pixels.
    fn render_stats(renderer: &mut Renderer, cuboids: &[Cuboid], lights: &[Light]) -> (u64, u64) {
        let mut cam = Camera::new(1.0);
        cam.position = Vec3::new(8.0, 10.0, 8.0);
        cam.rotation = Quat::from_rotation_arc(Vec3::NEG_Z, (Vec3::ZERO - cam.position).normalize());
        cam.far = 100.0;

        let color = renderer.device.create_texture(&TextureDescriptor {
            label: Some("readback_color"),
            size: Extent3d { width: DIM, height: DIM, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8UnormSrgb,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = color.create_view(&TextureViewDescriptor::default());
        renderer.render_with_lights(&view, &cam, cuboids, &[], lights);

        // DIM*4 = 1024 bytes/row, already a multiple of 256 (no padding needed).
        let bpr = DIM * 4;
        let buffer = renderer.device.create_buffer(&BufferDescriptor {
            label: Some("readback_buf"),
            size: (bpr * DIM) as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = renderer
            .device
            .create_command_encoder(&CommandEncoderDescriptor { label: Some("readback") });
        enc.copy_texture_to_buffer(
            TexelCopyTextureInfo {
                texture: &color,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            TexelCopyBufferInfo {
                buffer: &buffer,
                layout: TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bpr),
                    rows_per_image: Some(DIM),
                },
            },
            Extent3d { width: DIM, height: DIM, depth_or_array_layers: 1 },
        );
        renderer.queue.submit(Some(enc.finish()));

        let slice = buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = renderer.device.poll(PollType::Wait);
        rx.recv().unwrap().expect("map readback buffer");

        let data = slice.get_mapped_range();
        let mut total: u64 = 0;
        let mut mid: u64 = 0;
        for px in data.chunks_exact(4) {
            // Rec. 601 luma; enough to compare "lit" vs "shadowed" brightness.
            let lum = (px[0] as u64 * 299 + px[1] as u64 * 587 + px[2] as u64 * 114) / 1000;
            total += lum;
            // Background clears to ~42, ambient-only floor ~160, fully-lit floor
            // clamps near 255. The [90, 215] band captures shadowed floor.
            if (90..=215).contains(&lum) {
                mid += 1;
            }
        }
        drop(data);
        buffer.unmap();
        (total, mid)
    }

    /// End-to-end proof that the spot (flashlight) shadow actually darkens the
    /// scene: an angled spot light over a floor, with vs without a wide bar
    /// occluder. The bar casts a shadow, so the frame must be measurably
    /// darker than the same scene with no occluder. Also exercises the real
    /// shadow passes + main pass (which `shaders_and_pipelines_compile_on_gpu`
    /// does not — that only builds pipelines).
    #[test]
    fn spot_shadow_darkens_the_scene() {
        let Some((device, queue)) = headless_device() else {
            eprintln!("no GPU adapter; skipping spot-shadow render test");
            return;
        };
        let mut renderer =
            Renderer::from_device(device, queue, TextureFormat::Rgba8UnormSrgb, DIM, DIM);

        // A strong spot at (5,5,0), 45deg above +X, aimed at the floor center.
        // It must dominate the floor's lighting (ambient is a flat 0.6), so
        // intensity is high to overcome the 1/dist^2 falloff. A post standing
        // at the origin casts its shadow toward -X (offset from the post, not
        // hidden under it), onto floor the top-down camera sees. The post's own
        // pixels are tiny; the offset shadow strip is large, so total frame
        // luminance drops clearly when the post is present.
        let light_pos = Vec3::new(5.0, 5.0, 0.0);
        let light = Light {
            position: light_pos,
            direction: (Vec3::ZERO - light_pos).normalize(),
            kind: LightKind::Spot,
            color: Color3(255, 255, 255, 255),
            intensity: 250.0,
            range: 50.0,
            cone_angle_deg: 120.0,
        };
        let floor = || grey_cuboid(1, Vec3::new(0.0, 0.0, 0.0), Vec3::new(5.0, 0.1, 5.0));
        let post = grey_cuboid(2, Vec3::new(0.0, 1.5, 0.0), Vec3::new(0.5, 1.5, 0.5));

        let (lit_total, lit_mid) = render_stats(&mut renderer, &[floor()], &[light]);
        let (shadow_total, shadow_mid) = render_stats(&mut renderer, &[floor(), post], &[light]);

        // The occluder's shadow both lowers total brightness and converts a
        // population of fully-lit floor pixels into ambient-only (mid) ones.
        assert!(
            shadow_total < lit_total,
            "spot shadow should darken the frame: with post={shadow_total} vs without={lit_total}"
        );
        assert!(
            shadow_mid > lit_mid + 300,
            "spot shadow should add shadowed-floor pixels: with post={shadow_mid} vs without={lit_mid}"
        );
    }
}
