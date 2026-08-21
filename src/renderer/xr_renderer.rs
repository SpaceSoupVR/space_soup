use ash::vk::{self, Handle};
use log::info;
use openxr as xr;

use crate::renderer::{
    lights::LightsUniform,
    mesh::{create_texture_from_rgba, LoadedTexture},
    mesh_pipeline::{MeshPipeline, ModelUniform, SkinnedMeshPipeline},
    mirror::{MirrorPipeline, MirrorTarget},
    particle::ParticlePipeline,
    pipeline::{SolidPipeline, WirePipeline},
    ssr::{SceneTarget, SsrCameraUniform, SsrPipelines},
    uniforms::UniformBuffer,
};
use crate::xr::{VkContext, XrContext};
use std::collections::HashMap;

mod render_frame;
mod vulkan_interop;

struct EyeTarget {
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
}

pub struct XrRenderer {
    pub swapchain: xr::Swapchain<xr::Vulkan>,
    pub width: u32,
    pub height: u32,
    wgpu_device: wgpu::Device,
    wgpu_queue: wgpu::Queue,
    solid_pipeline: SolidPipeline,
    terrain_pipeline: crate::renderer::terrain_pipeline::TerrainPipeline,
    // The material is per-scene, but it is built here with the fallback so
    // terrain renders through the real pipeline before any textures exist.
    // Replaced via `set_terrain_material` when a scene authors its own.
    terrain_material: crate::renderer::terrain_pipeline::TerrainMaterial,
    /// Layer textures and splat map are kept so either can be replaced alone.
    terrain_layers: Vec<Option<crate::renderer::terrain_pipeline::TerrainImage>>,
    terrain_splat: Option<crate::renderer::terrain_pipeline::TerrainImage>,
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

        let (wgpu_device, wgpu_queue) = unsafe { vulkan_interop::build_wgpu_from_vulkan(vk)? };

        let lights_uniform = LightsUniform::new(&wgpu_device);
        let uniform_buf = UniformBuffer::new(&wgpu_device, &lights_uniform);
        let solid_pipeline = SolidPipeline::new(&wgpu_device, wgpu_format, &uniform_buf.layout);
        let terrain_pipeline = crate::renderer::terrain_pipeline::TerrainPipeline::new(
            &wgpu_device, wgpu_format, &uniform_buf.layout,
        );
        let terrain_material = crate::renderer::terrain_pipeline::TerrainMaterial::fallback(
            &wgpu_device, &wgpu_queue, &terrain_pipeline.material_layout,
        );
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
                    vulkan_interop::import_vk_image_as_wgpu(&wgpu_device, raw_image, wgpu_format, width, height, 2)
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
            terrain_pipeline,
            terrain_material,
            terrain_layers: vec![None, None, None, None],
            terrain_splat: None,
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


    /// Apply a scene's authored splat map, or `None` to fall back to the
    /// slope- and height-driven blend.
    ///
    /// Rebuilds the material rather than writing into the existing texture:
    /// the map's resolution is per-scene, so a scene change can need a
    /// different texture entirely, and rebuilding once per scene load is not
    /// worth the branch to avoid.
    pub fn set_terrain_splat(
        &mut self,
        splat: Option<&crate::renderer::terrain_pipeline::TerrainImage>,
    ) {
        self.terrain_splat = splat.map(|s| crate::renderer::terrain_pipeline::TerrainImage {
            width: s.width,
            height: s.height,
            rgba: s.rgba.clone(),
        });
        self.rebuild_terrain_material();
    }

    /// Apply the scene's layer textures, filling any that did not load.
    pub fn set_terrain_layers(
        &mut self,
        layers: Vec<Option<crate::renderer::terrain_pipeline::TerrainImage>>,
    ) {
        self.terrain_layers = layers;
        self.rebuild_terrain_material();
    }

    /// Rebuild from whatever layer textures and splat map are currently held.
    ///
    /// Both arrive independently -- layers once per scene load, the splat map
    /// whenever the scene changes -- and the material needs both at once.
    /// Keeping each and rebuilding means whichever arrives second does not
    /// discard the first, which is exactly what a setter that took only its own
    /// half would do.
    fn rebuild_terrain_material(&mut self) {
        self.terrain_material = crate::renderer::terrain_pipeline::TerrainMaterial::from_layers(
            &self.wgpu_device,
            &self.wgpu_queue,
            &self.terrain_pipeline.material_layout,
            &self.terrain_layers,
            self.terrain_splat.as_ref(),
        );
    }

    pub fn cleanup(&self) {}
}
