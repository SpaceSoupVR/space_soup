use ash::vk::{self, Handle};
use log::info;
use openxr as xr;

use crate::renderer::{
    camera::Camera,
    cuboid::{build_solid_mesh_with_ranges, build_wire_mesh, Cuboid},
    lights::{Light, LightsUniform},
    mesh::{create_texture_from_rgba, LoadedTexture},
    mesh_pipeline::{MeshPipeline, ModelUniform, SkinnedMeshPipeline},
    mirror::{self, MirrorPipeline, MirrorSurface, MirrorTarget},
    panel::WorldPanel,
    particle::{self, Beam, Particle, ParticlePipeline},
    pipeline::{SolidPipeline, WirePipeline},
    ssr::{SceneTarget, SsrCameraUniform, SsrPipelines},
    uniforms::UniformBuffer,
    Color3, MeshInstance,
};
use crate::ui2d::{Area, Color as UiColor, Font as Ui2dFont, Item, Shape as ShapeItem, ShapeType, Span, Text};
use crate::xr::{VkContext, XrContext};
use glam::{Quat, Vec3};
use std::collections::HashMap;
use std::sync::Arc;
use wgpu::util::DeviceExt;

pub struct UiPanelRenderData {
    pub id: String,
    pub position: Vec3,
    pub rotation: Quat,
    pub width_m: f32,
    pub height_m: f32,
    pub background_color: Color3,
    pub text: String,
    pub text_color: Color3,
}

struct EyeTarget {
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
}

type MeshDraw<'a> = (
    &'a wgpu::BindGroup,
    &'a wgpu::BindGroup,
    &'a wgpu::BindGroup,
    &'a wgpu::Buffer,
    &'a wgpu::Buffer,
    u32,
);

type SkinnedDraw<'a> = (
    &'a wgpu::BindGroup,
    &'a wgpu::BindGroup,
    &'a wgpu::BindGroup,
    &'a wgpu::Buffer,
    &'a wgpu::Buffer,
    u32,
);

fn ui_color(c: Color3) -> UiColor {
    UiColor(c.0, c.1, c.2, c.3)
}

fn push_mesh_draws<'a>(
    instance: &'a MeshInstance,
    lightmap_bg: &'a wgpu::BindGroup,
    mesh_draws: &mut Vec<MeshDraw<'a>>,
    skinned_draws: &mut Vec<SkinnedDraw<'a>>,
) {
    if let Some(skin) = &instance.mesh.skin {
        if let Some(joint_bg) = &skin.joint_bind_group {
            for prim in &skin.primitives {
                skinned_draws.push((
                    &instance.model.bind_group,
                    &prim.texture.bind_group,
                    joint_bg,
                    &prim.vertex_buffer,
                    &prim.index_buffer,
                    prim.indices.len() as u32,
                ));
            }
        }
    } else {
        for prim in &instance.mesh.primitives {
            mesh_draws.push((
                &instance.model.bind_group,
                &prim.texture.bind_group,
                lightmap_bg,
                &prim.vertex_buffer,
                &prim.index_buffer,
                prim.indices.len() as u32,
            ));
        }
    }
}

pub struct XrRenderer {
    pub swapchain: xr::Swapchain<xr::Vulkan>,
    pub width: u32,
    pub height: u32,
    wgpu_device: wgpu::Device,
    wgpu_queue: wgpu::Queue,
    solid_pipeline: SolidPipeline,
    wire_pipeline: WirePipeline,
    mesh_pipeline: MeshPipeline,
    skinned_mesh_pipeline: SkinnedMeshPipeline,
    mirror_solid_pipeline: SolidPipeline,
    mirror_mesh_pipeline: MeshPipeline,
    mirror_pipeline: MirrorPipeline,
    mirror_targets: [MirrorTarget; 2],
    mirror_model_uniform: ModelUniform,
    mirror_reflected_vp_uniform: ModelUniform,
    ssr_pipelines: SsrPipelines,
    ssr_solid_pipeline: SolidPipeline,
    scene_targets: [SceneTarget; 2],
    ssr_camera_uniform: SsrCameraUniform,
    particle_pipeline: ParticlePipeline,
    uniform_buf: UniformBuffer,
    lights_uniform: LightsUniform,
    depth_view: wgpu::TextureView,
    eye_targets: Vec<[EyeTarget; 2]>,
    cuboid_lightmaps: HashMap<String, LoadedTexture>,
    default_cuboid_lightmap: LoadedTexture,
    mesh_lightmaps: HashMap<String, LoadedTexture>,
    default_mesh_lightmap: LoadedTexture,
    frame_stats: FrameStats,
}

struct FrameStats {
    window: u64,
    count: u64,
    last_frame: Option<std::time::Instant>,
    cpu_ms_sum: f64,
    gpu_ms_sum: f64,
    period_ms_sum: f64,
    period_samples: u64,
    cpu_ms_max: f64,
    gpu_ms_max: f64,
}

impl FrameStats {
    fn new(window: u64) -> Self {
        Self {
            window,
            count: 0,
            last_frame: None,
            cpu_ms_sum: 0.0,
            gpu_ms_sum: 0.0,
            period_ms_sum: 0.0,
            period_samples: 0,
            cpu_ms_max: 0.0,
            gpu_ms_max: 0.0,
        }
    }

    fn record(&mut self, cpu: std::time::Duration, gpu: std::time::Duration, now: std::time::Instant) {
        let cpu_ms = cpu.as_secs_f64() * 1000.0;
        let gpu_ms = gpu.as_secs_f64() * 1000.0;
        self.cpu_ms_sum += cpu_ms;
        self.gpu_ms_sum += gpu_ms;
        self.cpu_ms_max = self.cpu_ms_max.max(cpu_ms);
        self.gpu_ms_max = self.gpu_ms_max.max(gpu_ms);
        if let Some(prev) = self.last_frame {
            self.period_ms_sum += (now - prev).as_secs_f64() * 1000.0;
            self.period_samples += 1;
        }
        self.last_frame = Some(now);
        self.count += 1;

        if self.count % self.window == 0 {
            let n = self.window as f64;
            let avg_period = if self.period_samples > 0 {
                self.period_ms_sum / self.period_samples as f64
            } else {
                0.0
            };
            let fps = if avg_period > 0.0 { 1000.0 / avg_period } else { 0.0 };
            info!(
                "PERF: cpu_avg={:.2}ms cpu_max={:.2}ms | gpu_avg={:.2}ms gpu_max={:.2}ms | frame={:.2}ms (~{:.1}fps) over {} frames",
                self.cpu_ms_sum / n,
                self.cpu_ms_max,
                self.gpu_ms_sum / n,
                self.gpu_ms_max,
                avg_period,
                fps,
                self.window,
            );
            self.cpu_ms_sum = 0.0;
            self.gpu_ms_sum = 0.0;
            self.period_ms_sum = 0.0;
            self.period_samples = 0;
            self.cpu_ms_max = 0.0;
            self.gpu_ms_max = 0.0;
        }
    }
}

impl XrRenderer {
    pub fn new(
        vk: &VkContext,
        xr_ctx: &XrContext,
        session: &xr::Session<xr::Vulkan>,
        ui_font_bytes: &[u8],
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let view_configs = xr_ctx.instance.enumerate_view_configuration_views(
            xr_ctx.system,
            xr::ViewConfigurationType::PRIMARY_STEREO,
        )?;
        let width = view_configs[0].recommended_image_rect_width;
        let height = view_configs[0].recommended_image_rect_height;
        info!("XrRenderer: {width}x{height}");

        let vk_format = vk::Format::R8G8B8A8_SRGB;
        let wgpu_format = wgpu::TextureFormat::Rgba8UnormSrgb;

        let swapchain = session.create_swapchain(&xr::SwapchainCreateInfo {
            create_flags: xr::SwapchainCreateFlags::EMPTY,
            usage_flags: xr::SwapchainUsageFlags::COLOR_ATTACHMENT
                | xr::SwapchainUsageFlags::SAMPLED,
            format: vk_format.as_raw() as _,
            sample_count: 1,
            width,
            height,
            face_count: 1,
            array_size: 2,
            mip_count: 1,
        })?;

        let raw_images: Vec<vk::Image> = swapchain
            .enumerate_images()?
            .into_iter()
            .map(vk::Image::from_raw)
            .collect();
        info!("XrRenderer: {} swapchain images", raw_images.len());

        let (wgpu_device, wgpu_queue) = unsafe { build_wgpu_from_vulkan(vk)? };

        let lights_uniform = LightsUniform::new(&wgpu_device);
        let uniform_buf = UniformBuffer::new(&wgpu_device, &lights_uniform);
        let solid_pipeline = SolidPipeline::new(&wgpu_device, wgpu_format, &uniform_buf.layout);
        let wire_pipeline = WirePipeline::new(&wgpu_device, wgpu_format, &uniform_buf.layout);
        let mesh_pipeline = MeshPipeline::new(&wgpu_device, wgpu_format, &uniform_buf.layout);
        let skinned_mesh_pipeline =
            SkinnedMeshPipeline::new(&wgpu_device, wgpu_format, &uniform_buf.layout);
        let mirror_solid_pipeline =
            SolidPipeline::new_mirror(&wgpu_device, wgpu_format, &uniform_buf.layout);
        let mirror_mesh_pipeline =
            MeshPipeline::new_mirror(&wgpu_device, wgpu_format, &uniform_buf.layout);
        let mirror_pipeline = MirrorPipeline::new(&wgpu_device, wgpu_format, &uniform_buf.layout);
        let mirror_targets: [MirrorTarget; 2] =
            std::array::from_fn(|_| mirror_pipeline.create_target(&wgpu_device, wgpu_format, width, height));
        let mirror_model_uniform = mirror_pipeline.create_model_uniform(&wgpu_device);
        let mirror_reflected_vp_uniform = mirror_pipeline.create_reflected_vp_uniform(&wgpu_device);
        let ssr_pipelines = SsrPipelines::new(&wgpu_device, wgpu_format);
        let ssr_solid_pipeline = SolidPipeline::new_ssr(
            &wgpu_device,
            wgpu_format,
            &uniform_buf.layout,
            ssr_pipelines.camera_layout(),
            ssr_pipelines.scene_texture_layout(),
        );
        let scene_targets: [SceneTarget; 2] = std::array::from_fn(|_| {
            ssr_pipelines.create_scene_target(&wgpu_device, wgpu_format, width, height)
        });
        let ssr_camera_uniform = ssr_pipelines.create_camera_uniform(&wgpu_device);
        let particle_pipeline =
            ParticlePipeline::new(&wgpu_device, wgpu_format, &uniform_buf.layout);

        let depth_tex = wgpu_device.create_texture(&wgpu::TextureDescriptor {
            label: Some("xr_depth"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth_view = depth_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let mut eye_targets: Vec<[EyeTarget; 2]> = Vec::new();
        for &raw_image in &raw_images {
            let targets = std::array::from_fn(|eye| {
                let wgpu_tex = unsafe {
                    import_vk_image_as_wgpu(&wgpu_device, raw_image, wgpu_format, width, height, 2)
                };
                let view = wgpu_tex.create_view(&wgpu::TextureViewDescriptor {
                    format: Some(wgpu_format),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_array_layer: eye as u32,
                    array_layer_count: Some(1),
                    ..Default::default()
                });
                EyeTarget {
                    _texture: wgpu_tex,
                    view,
                }
            });
            eye_targets.push(targets);
        }

        let white_pixel = [255u8, 255, 255, 255];
        let default_cuboid_lightmap = create_texture_from_rgba(
            &wgpu_device,
            &wgpu_queue,
            &solid_pipeline.lightmap_layout,
            &white_pixel,
            1,
            1,
        );
        let default_mesh_lightmap = create_texture_from_rgba(
            &wgpu_device,
            &wgpu_queue,
            &mesh_pipeline.lightmap_layout,
            &white_pixel,
            1,
            1,
        );

        Ok(Self {
            swapchain,
            width,
            height,
            wgpu_device,
            wgpu_queue,
            solid_pipeline,
            wire_pipeline,
            mesh_pipeline,
            skinned_mesh_pipeline,
            mirror_solid_pipeline,
            mirror_mesh_pipeline,
            mirror_pipeline,
            mirror_targets,
            mirror_model_uniform,
            mirror_reflected_vp_uniform,
            ssr_pipelines,
            ssr_solid_pipeline,
            scene_targets,
            ssr_camera_uniform,
            particle_pipeline,
            uniform_buf,
            lights_uniform,
            depth_view,
            eye_targets,
            cuboid_lightmaps: HashMap::new(),
            default_cuboid_lightmap,
            mesh_lightmaps: HashMap::new(),
            default_mesh_lightmap,
            frame_stats: FrameStats::new(120),
        })
    }

    pub fn set_cuboid_lightmap(&mut self, key: &str, rgba: &[u8], width: u32, height: u32) {
        let tex = create_texture_from_rgba(
            &self.wgpu_device,
            &self.wgpu_queue,
            &self.solid_pipeline.lightmap_layout,
            rgba,
            width,
            height,
        );
        self.cuboid_lightmaps.insert(key.to_string(), tex);
    }

    pub fn set_mesh_lightmap(&mut self, key: &str, rgba: &[u8], width: u32, height: u32) {
        let tex = create_texture_from_rgba(
            &self.wgpu_device,
            &self.wgpu_queue,
            &self.mesh_pipeline.lightmap_layout,
            rgba,
            width,
            height,
        );
        self.mesh_lightmaps.insert(key.to_string(), tex);
    }

    fn cuboid_lightmap_bg(&self, key: Option<&str>) -> &wgpu::BindGroup {
        key.and_then(|k| self.cuboid_lightmaps.get(k))
            .map(|t| &t.bind_group)
            .unwrap_or(&self.default_cuboid_lightmap.bind_group)
    }

    fn mesh_lightmap_bg(&self, key: Option<&str>) -> &wgpu::BindGroup {
        key.and_then(|k| self.mesh_lightmaps.get(k))
            .map(|t| &t.bind_group)
            .unwrap_or(&self.default_mesh_lightmap.bind_group)
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.wgpu_device
    }
    pub fn queue(&self) -> &wgpu::Queue {
        &self.wgpu_queue
    }
    pub fn mesh_texture_layout(&self) -> &wgpu::BindGroupLayout {
        &self.mesh_pipeline.texture_layout
    }
    pub fn skinned_mesh_texture_layout(&self) -> &wgpu::BindGroupLayout {
        &self.skinned_mesh_pipeline.texture_layout
    }
    pub fn skin_joint_layout(&self) -> &wgpu::BindGroupLayout {
        &self.skinned_mesh_pipeline.skin_joint_layout
    }
    pub fn create_model_uniform(&self) -> crate::renderer::mesh_pipeline::ModelUniform {
        self.mesh_pipeline.create_model_uniform(&self.wgpu_device)
    }
    pub fn create_skinned_model_uniform(&self) -> crate::renderer::mesh_pipeline::ModelUniform {
        self.skinned_mesh_pipeline
            .create_model_uniform(&self.wgpu_device)
    }

    pub fn render_frame(
        &mut self,
        session: &xr::Session<xr::Vulkan>,
        stage: &xr::Space,
        time: xr::Time,
        cuboids: &[Cuboid],
    ) -> Result<Vec<xr::CompositionLayerProjectionView<xr::Vulkan>>, Box<dyn std::error::Error>>
    {
        self.render_frame_with_meshes(session, stage, time, cuboids, &[], &[], &[], &[], &[], None, &[])
    }

    #[allow(clippy::too_many_arguments)]
    pub fn render_frame_with_meshes(
        &mut self,
        session: &xr::Session<xr::Vulkan>,
        stage: &xr::Space,
        time: xr::Time,
        cuboids: &[Cuboid],
        meshes: &[MeshInstance],
        mirror_only_meshes: &[MeshInstance],
        lights: &[Light],
        particles: &[Particle],
        beams: &[Beam],
        mirror: Option<MirrorSurface>,
        ui_panels: &[UiPanelRenderData],
    ) -> Result<Vec<xr::CompositionLayerProjectionView<xr::Vulkan>>, Box<dyn std::error::Error>>
    {
        self.sync_ui_panels(ui_panels);

        let image_index = self.swapchain.acquire_image()? as usize;
        self.swapchain.wait_image(xr::Duration::INFINITE)?;
        let cpu_start = std::time::Instant::now();
        self.lights_uniform.upload(&self.wgpu_queue, lights);

        let (_, eye_views) =
            session.locate_views(xr::ViewConfigurationType::PRIMARY_STEREO, time, stage)?;

        let head_rot = {
            let o = eye_views[0].pose.orientation;
            glam::Quat::from_xyzw(o.x, o.y, o.z, o.w)
        };
        let cam_right = head_rot * glam::Vec3::X;
        let cam_up = head_rot * glam::Vec3::Y;
        let view_dir = head_rot * glam::Vec3::NEG_Z;

        let (solid_verts, solid_idx, solid_ranges) = build_solid_mesh_with_ranges(cuboids);
        let (wire_verts, wire_idx) = build_wire_mesh(cuboids);
        let (particle_verts, particle_idx) =
            particle::build_particle_mesh(particles, beams, cam_right, cam_up, view_dir);

        let solid_vb = self
            .wgpu_device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("solid_vb"),
                contents: bytemuck::cast_slice(&solid_verts),
                usage: wgpu::BufferUsages::VERTEX,
            });
        let solid_ib = self
            .wgpu_device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("solid_ib"),
                contents: bytemuck::cast_slice(&solid_idx),
                usage: wgpu::BufferUsages::INDEX,
            });
        let wire_vb = self
            .wgpu_device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("wire_vb"),
                contents: bytemuck::cast_slice(&wire_verts),
                usage: wgpu::BufferUsages::VERTEX,
            });
        let wire_ib = self
            .wgpu_device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("wire_ib"),
                contents: bytemuck::cast_slice(&wire_idx),
                usage: wgpu::BufferUsages::INDEX,
            });
        let particle_vb = self
            .wgpu_device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("particle_vb"),
                contents: bytemuck::cast_slice(&particle_verts),
                usage: wgpu::BufferUsages::VERTEX,
            });
        let particle_ib = self
            .wgpu_device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("particle_ib"),
                contents: bytemuck::cast_slice(&particle_idx),
                usage: wgpu::BufferUsages::INDEX,
            });

        let mut mesh_draws: Vec<MeshDraw> = Vec::new();
        let mut skinned_draws: Vec<SkinnedDraw> = Vec::new();
        for instance in meshes {
            instance
                .model
                .upload(&self.wgpu_queue, instance.mesh.model_matrix());
            let lightmap_bg = self.mesh_lightmap_bg(instance.lightmap_key);
            push_mesh_draws(instance, lightmap_bg, &mut mesh_draws, &mut skinned_draws);
        }

        let ui_panel_list: Vec<&WorldPanel> = self.ui_panel_instances.values().collect();
        let mut ui_panel_buffers: Vec<(wgpu::Buffer, wgpu::Buffer)> = Vec::with_capacity(ui_panel_list.len());
        for panel in &ui_panel_list {
            let vb = self
                .wgpu_device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("ui_panel_vb"),
                    contents: bytemuck::cast_slice(panel.vertices()),
                    usage: wgpu::BufferUsages::VERTEX,
                });
            let ib = self
                .wgpu_device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("ui_panel_ib"),
                    contents: bytemuck::cast_slice(panel.indices()),
                    usage: wgpu::BufferUsages::INDEX,
                });
            ui_panel_buffers.push((vb, ib));
        }
        for (panel, (vb, ib)) in ui_panel_list.iter().zip(ui_panel_buffers.iter()) {
            mesh_draws.push((
                &panel.model.bind_group,
                panel.bind_group(),
                &self.default_mesh_lightmap.bind_group,
                vb,
                ib,
                panel.indices().len() as u32,
            ));
        }

        let mut mirror_only_mesh_draws: Vec<MeshDraw> = Vec::new();
        let mut mirror_only_skinned_draws: Vec<SkinnedDraw> = Vec::new();
        for instance in mirror_only_meshes {
            instance
                .model
                .upload(&self.wgpu_queue, instance.mesh.model_matrix());
            let lightmap_bg = self.mesh_lightmap_bg(instance.lightmap_key);
            push_mesh_draws(instance, lightmap_bg, &mut mirror_only_mesh_draws, &mut mirror_only_skinned_draws);
        }

        let mirror_quad = mirror.map(|m| {
            let (verts, idx) = mirror::build_mirror_quad(m.half_size.x, m.half_size.y);
            let vb = self
                .wgpu_device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("mirror_quad_vb"),
                    contents: bytemuck::cast_slice(&verts),
                    usage: wgpu::BufferUsages::VERTEX,
                });
            let ib = self
                .wgpu_device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("mirror_quad_ib"),
                    contents: bytemuck::cast_slice(&idx),
                    usage: wgpu::BufferUsages::INDEX,
                });
            let model = glam::Mat4::from_rotation_translation(m.rotation, m.position);
            self.mirror_model_uniform.upload(&self.wgpu_queue, model);
            (vb, ib, idx.len() as u32)
        });

        for eye in 0..2usize {
            let ev = &eye_views[eye];
            let view = Camera::xr_view(ev.pose);
            let proj = Camera::xr_projection(ev.fov, 0.03, 1000.0);

            if let Some(m) = &mirror {
                let reflect = mirror::reflection_matrix(m.position, m.normal());
                let mirror_view = view * reflect;

                let world_plane = mirror::world_plane_equation(m.position, m.normal());
                let eye_plane = mirror::plane_to_eye_space(mirror_view.inverse(), world_plane);
                let mirror_proj = mirror::oblique_near_clip(proj, eye_plane);
                let mirror_view_proj = Camera::gl_to_wgpu_ndc(mirror_proj) * mirror_view;

                self.uniform_buf.upload(&self.wgpu_queue, mirror_view_proj);
                self.mirror_reflected_vp_uniform.upload(&self.wgpu_queue, mirror_view_proj);

                let mut encoder = self.wgpu_device.create_command_encoder(
                    &wgpu::CommandEncoderDescriptor { label: Some("mirror_eye") },
                );
                {
                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("mirror_pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &self.mirror_targets[eye].color_view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color {
                                    r: 0.02,
                                    g: 0.02,
                                    b: 0.05,
                                    a: 1.0,
                                }),
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: &self.mirror_targets[eye].depth_view,
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Clear(1.0),
                                store: wgpu::StoreOp::Store,
                            }),
                            stencil_ops: None,
                        }),
                        ..Default::default()
                    });

                    if !solid_verts.is_empty() {
                        pass.set_pipeline(&self.mirror_solid_pipeline.pipeline);
                        pass.set_bind_group(0, &self.uniform_buf.bind_group, &[]);
                        pass.set_vertex_buffer(0, solid_vb.slice(..));
                        pass.set_index_buffer(solid_ib.slice(..), wgpu::IndexFormat::Uint32);
                        for (lightmap_key, index_start, count, _reflectivity) in &solid_ranges {
                            pass.set_bind_group(1, self.cuboid_lightmap_bg(lightmap_key.as_deref()), &[]);
                            pass.draw_indexed(*index_start..*index_start + *count, 0, 0..1);
                        }
                    }
                    let all_mesh_draws = mesh_draws.iter().chain(mirror_only_mesh_draws.iter());
                    let all_skinned_draws =
                        skinned_draws.iter().chain(mirror_only_skinned_draws.iter());

                    if !mesh_draws.is_empty() || !mirror_only_mesh_draws.is_empty() {
                        pass.set_pipeline(&self.mirror_mesh_pipeline.pipeline);
                        pass.set_bind_group(0, &self.uniform_buf.bind_group, &[]);
                        for (model_bg, tex_bg, lightmap_bg, vb, ib, count) in all_mesh_draws {
                            pass.set_bind_group(1, *model_bg, &[]);
                            pass.set_bind_group(2, *tex_bg, &[]);
                            pass.set_bind_group(3, *lightmap_bg, &[]);
                            pass.set_vertex_buffer(0, vb.slice(..));
                            pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                            pass.draw_indexed(0..*count, 0, 0..1);
                        }
                    }
                    if !skinned_draws.is_empty() || !mirror_only_skinned_draws.is_empty() {
                        pass.set_pipeline(&self.skinned_mesh_pipeline.pipeline);
                        pass.set_bind_group(0, &self.uniform_buf.bind_group, &[]);
                        for (model_bg, tex_bg, joint_bg, vb, ib, count) in all_skinned_draws {
                            pass.set_bind_group(1, *model_bg, &[]);
                            pass.set_bind_group(2, *tex_bg, &[]);
                            pass.set_bind_group(3, *joint_bg, &[]);
                            pass.set_vertex_buffer(0, vb.slice(..));
                            pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                            pass.draw_indexed(0..*count, 0, 0..1);
                        }
                    }
                }
                self.wgpu_queue.submit(Some(encoder.finish()));
            }

            let eye_view_proj = Camera::gl_to_wgpu_ndc(proj) * view;
            self.uniform_buf.upload(&self.wgpu_queue, eye_view_proj);
            let cam_pos = glam::Vec3::new(ev.pose.position.x, ev.pose.position.y, ev.pose.position.z);
            self.ssr_camera_uniform.upload(&self.wgpu_queue, eye_view_proj, cam_pos);

            {
                let mut encoder = self.wgpu_device.create_command_encoder(
                    &wgpu::CommandEncoderDescriptor { label: Some("ssr_scene") },
                );
                {
                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("ssr_scene_pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &self.scene_targets[eye].color_view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color {
                                    r: 0.02,
                                    g: 0.02,
                                    b: 0.05,
                                    a: 1.0,
                                }),
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: &self.scene_targets[eye].depth_view,
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Clear(1.0),
                                store: wgpu::StoreOp::Store,
                            }),
                            stencil_ops: None,
                        }),
                        ..Default::default()
                    });

                    if !solid_verts.is_empty() {
                        pass.set_pipeline(&self.solid_pipeline.pipeline);
                        pass.set_bind_group(0, &self.uniform_buf.bind_group, &[]);
                        pass.set_vertex_buffer(0, solid_vb.slice(..));
                        pass.set_index_buffer(solid_ib.slice(..), wgpu::IndexFormat::Uint32);
                        for (lightmap_key, index_start, count, _reflectivity) in &solid_ranges {
                            pass.set_bind_group(1, self.cuboid_lightmap_bg(lightmap_key.as_deref()), &[]);
                            pass.draw_indexed(*index_start..*index_start + *count, 0, 0..1);
                        }
                    }
                    if !wire_verts.is_empty() {
                        pass.set_pipeline(&self.wire_pipeline.pipeline);
                        pass.set_bind_group(0, &self.uniform_buf.bind_group, &[]);
                        pass.set_vertex_buffer(0, wire_vb.slice(..));
                        pass.set_index_buffer(wire_ib.slice(..), wgpu::IndexFormat::Uint32);
                        pass.draw_indexed(0..wire_idx.len() as u32, 0, 0..1);
                    }
                    if !mesh_draws.is_empty() {
                        pass.set_pipeline(&self.mesh_pipeline.pipeline);
                        pass.set_bind_group(0, &self.uniform_buf.bind_group, &[]);
                        for (model_bg, tex_bg, lightmap_bg, vb, ib, count) in &mesh_draws {
                            pass.set_bind_group(1, *model_bg, &[]);
                            pass.set_bind_group(2, *tex_bg, &[]);
                            pass.set_bind_group(3, *lightmap_bg, &[]);
                            pass.set_vertex_buffer(0, vb.slice(..));
                            pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                            pass.draw_indexed(0..*count, 0, 0..1);
                        }
                    }
                    if !skinned_draws.is_empty() {
                        pass.set_pipeline(&self.skinned_mesh_pipeline.pipeline);
                        pass.set_bind_group(0, &self.uniform_buf.bind_group, &[]);
                        for (model_bg, tex_bg, joint_bg, vb, ib, count) in &skinned_draws {
                            pass.set_bind_group(1, *model_bg, &[]);
                            pass.set_bind_group(2, *tex_bg, &[]);
                            pass.set_bind_group(3, *joint_bg, &[]);
                            pass.set_vertex_buffer(0, vb.slice(..));
                            pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                            pass.draw_indexed(0..*count, 0, 0..1);
                        }
                    }
                    if !particle_verts.is_empty() {
                        pass.set_pipeline(&self.particle_pipeline.pipeline);
                        pass.set_bind_group(0, &self.uniform_buf.bind_group, &[]);
                        pass.set_vertex_buffer(0, particle_vb.slice(..));
                        pass.set_index_buffer(particle_ib.slice(..), wgpu::IndexFormat::Uint32);
                        pass.draw_indexed(0..particle_idx.len() as u32, 0, 0..1);
                    }
                }
                self.wgpu_queue.submit(Some(encoder.finish()));
            }

            let color_view = &self.eye_targets[image_index][eye].view;
            let mut encoder = self
                .wgpu_device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("eye") });

            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("eye_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: color_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.02,
                                g: 0.02,
                                b: 0.05,
                                a: 1.0,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &self.depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    ..Default::default()
                });

                pass.set_pipeline(&self.ssr_pipelines.blit_pipeline);
                pass.set_bind_group(0, &self.scene_targets[eye].bind_group, &[]);
                pass.draw(0..3, 0..1);

                if solid_ranges.iter().any(|(_, _, _, r)| *r > 0.0) {
                    pass.set_pipeline(&self.ssr_solid_pipeline.pipeline);
                    pass.set_bind_group(0, &self.uniform_buf.bind_group, &[]);
                    pass.set_bind_group(2, &self.ssr_camera_uniform.bind_group, &[]);
                    pass.set_bind_group(3, &self.scene_targets[eye].bind_group, &[]);
                    pass.set_vertex_buffer(0, solid_vb.slice(..));
                    pass.set_index_buffer(solid_ib.slice(..), wgpu::IndexFormat::Uint32);
                    for (lightmap_key, index_start, count, reflectivity) in &solid_ranges {
                        if *reflectivity <= 0.0 {
                            continue;
                        }
                        pass.set_bind_group(1, self.cuboid_lightmap_bg(lightmap_key.as_deref()), &[]);
                        pass.draw_indexed(*index_start..*index_start + *count, 0, 0..1);
                    }
                }

                if let Some((vb, ib, count)) = &mirror_quad {
                    pass.set_pipeline(&self.mirror_pipeline.pipeline);
                    pass.set_bind_group(0, &self.uniform_buf.bind_group, &[]);
                    pass.set_bind_group(1, &self.mirror_model_uniform.bind_group, &[]);
                    pass.set_bind_group(2, &self.mirror_targets[eye].texture_bind_group, &[]);
                    pass.set_bind_group(3, &self.mirror_reflected_vp_uniform.bind_group, &[]);
                    pass.set_vertex_buffer(0, vb.slice(..));
                    pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(0..*count, 0, 0..1);
                }
            }

            self.wgpu_queue.submit(Some(encoder.finish()));
        }

        let cpu_time = cpu_start.elapsed();
        let gpu_wait_start = std::time::Instant::now();
        self.wgpu_device.poll(wgpu::PollType::Wait);
        let gpu_time = gpu_wait_start.elapsed();
        self.frame_stats.record(cpu_time, gpu_time, std::time::Instant::now());
        self.swapchain.release_image()?;

        let proj_views = eye_views
            .iter()
            .enumerate()
            .map(|(i, ev)| {
                xr::CompositionLayerProjectionView::new()
                    .pose(ev.pose)
                    .fov(ev.fov)
                    .sub_image(
                        xr::SwapchainSubImage::new()
                            .swapchain(&self.swapchain)
                            .image_array_index(i as u32)
                            .image_rect(xr::Rect2Di {
                                offset: xr::Offset2Di { x: 0, y: 0 },
                                extent: xr::Extent2Di {
                                    width: self.width as i32,
                                    height: self.height as i32,
                                },
                            }),
                    )
            })
            .collect();

        Ok(proj_views)
    }

    pub fn cleanup(&self) {}
}

unsafe fn build_wgpu_from_vulkan(
    vk: &VkContext,
) -> Result<(wgpu::Device, wgpu::Queue), Box<dyn std::error::Error>> {
    use wgpu::hal::vulkan as hvk;

    let shared_instance = hvk::Instance::from_raw(
        ash::Entry::linked(),
        vk.instance.clone(),
        vk::make_api_version(0, 1, 1, 0),
        0,
        None,
        vec![],
        wgpu::InstanceFlags::empty(),
        false,
        None,
    )?;

    let exposed = shared_instance
        .expose_adapter(vk.physical_device)
        .ok_or("wgpu: failed to expose physical device")?;

    let open_device = exposed.adapter.device_from_raw(
        vk.device.clone(),
        None,
        &[],
        wgpu::Features::empty(),
        &wgpu::MemoryHints::default(),
        vk.queue_family_index,
        0,
    )?;

    let wgpu_instance = wgpu::Instance::from_hal::<hvk::Api>(shared_instance);
    let wgpu_adapter = wgpu_instance.create_adapter_from_hal(exposed);

    let adapter_limits = wgpu_adapter.limits();
    let (device, queue) = wgpu_adapter.create_device_from_hal(
        open_device,
        &wgpu::DeviceDescriptor {
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits {
                max_texture_dimension_2d: adapter_limits.max_texture_dimension_2d,
                ..wgpu::Limits::downlevel_defaults()
            },
            ..Default::default()
        },
    )?;

    Ok((device, queue))
}

unsafe fn import_vk_image_as_wgpu(
    device: &wgpu::Device,
    image: vk::Image,
    wgpu_format: wgpu::TextureFormat,
    width: u32,
    height: u32,
    array_layers: u32,
) -> wgpu::Texture {
    use wgpu::hal::vulkan as hvk;

    let hal_texture = hvk::Device::texture_from_raw(
        image,
        &wgpu::hal::TextureDescriptor {
            label: Some("xr_swapchain_image"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: array_layers,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu_format,
            usage: wgpu::TextureUses::COLOR_TARGET | wgpu::TextureUses::RESOURCE,
            memory_flags: wgpu::hal::MemoryFlags::empty(),
            view_formats: vec![],
        },
        None,
    );

    device.create_texture_from_hal::<hvk::Api>(
        hal_texture,
        &wgpu::TextureDescriptor {
            label: Some("xr_swapchain_tex"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: array_layers,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        },
    )
}

