use ash::vk::{self, Handle};
use std::sync::Arc;
use crate::renderer::mesh::create_lightmap_texture;
use log::{error, info};
use openxr as xr;

use crate::renderer::{
    lights::LightsUniform,
    mesh::{create_texture_from_rgba, LoadedTexture},
    mesh_pipeline::{MeshPipeline, ModelUniform, SkinnedMeshPipeline},
    mirror::{MirrorPipeline, MirrorTarget},
    particle::ParticlePipeline,
    pipeline::{SolidPipeline, WirePipeline},
    ssr::{SceneTarget, SsrCameraUniform, SsrPipelines, StereoSceneTextures},
    uniforms::UniformBuffer,
};
use crate::xr::{VkContext, XrContext};
use std::collections::HashMap;

mod render_frame;
mod time_of_day;
mod vulkan_interop;

struct FoveationState {
    eye_size: (u32, u32),

    probe_size: (u32, u32),

    maps: HashMap<crate::renderer::foveation::FoveationLevel, [usize; 4]>,

    applied: Option<crate::renderer::foveation::FoveationLevel>,
}

struct SpaceWarpState {
    motion: xr::Swapchain<xr::Vulkan>,
    depth: xr::Swapchain<xr::Vulkan>,

    motion_targets: Vec<[EyeTarget; 2]>,
    depth_targets: Vec<[EyeTarget; 2]>,
    size: (u32, u32),
    pipelines: crate::renderer::space_warp::MotionPipelines,

    cameras: wgpu::Buffer,
    camera_group: wgpu::BindGroup,

    prev: Option<([glam::Mat4; 2], glam::Mat4)>,

    prev_models: HashMap<wgpu::Buffer, glam::Mat4>,

    prev_clocks: Option<(f64, f64)>,

    acquired: Option<(usize, usize)>,

    info: [xr::sys::CompositionLayerSpaceWarpInfoFB; 2],

    motion_raw: Vec<vk::Image>,
    depth_raw: Vec<vk::Image>,
    depth_has_stencil: bool,
    readback: Option<crate::renderer::space_warp::Readback>,

    reflect_groups: Vec<[wgpu::BindGroup; 2]>,
}

impl SpaceWarpState {
    fn new(
        session: &xr::Session<xr::Vulkan>,
        device: &wgpu::Device,
        size: (u32, u32),
        vk_ctx: &VkContext,
    ) -> Result<Self, Box<dyn std::error::Error>> {

        let formats = session.enumerate_swapchain_formats()?;
        let d24s8 = vulkan_interop::supports_depth_attachment(device, vk::Format::D24_UNORM_S8_UINT);
        let (vk_depth, depth_format) = [
            (vk::Format::D24_UNORM_S8_UINT, wgpu::TextureFormat::Depth24PlusStencil8),
            (vk::Format::D32_SFLOAT, wgpu::TextureFormat::Depth32Float),
            (vk::Format::D16_UNORM, wgpu::TextureFormat::Depth16Unorm),
        ]
        .into_iter()
        .filter(|(f, _)| *f != vk::Format::D24_UNORM_S8_UINT || d24s8)
        .find(|(f, _)| formats.contains(&(f.as_raw() as u32)))
        .ok_or("no depth swapchain format the renderer can use")?;
        let make = |format: vk::Format, usage: xr::SwapchainUsageFlags| {
            session.create_swapchain(&xr::SwapchainCreateInfo {
                create_flags: xr::SwapchainCreateFlags::EMPTY,
                usage_flags: usage,
                format: format.as_raw() as _,
                sample_count: 1,
                width: size.0,
                height: size.1,
                face_count: 1,
                array_size: 2,
                mip_count: 1,
            })
        };

        let motion = make(
            vk::Format::R16G16B16A16_SFLOAT,
            xr::SwapchainUsageFlags::COLOR_ATTACHMENT | xr::SwapchainUsageFlags::SAMPLED | xr::SwapchainUsageFlags::TRANSFER_SRC,
        )?;
        let depth = make(
            vk_depth,
            xr::SwapchainUsageFlags::DEPTH_STENCIL_ATTACHMENT | xr::SwapchainUsageFlags::SAMPLED | xr::SwapchainUsageFlags::TRANSFER_SRC,
        )?;
        let motion_raw: Vec<vk::Image> = motion.enumerate_images()?.into_iter().map(vk::Image::from_raw).collect();
        let depth_raw: Vec<vk::Image> = depth.enumerate_images()?.into_iter().map(vk::Image::from_raw).collect();
        let readback = crate::renderer::space_warp::Readback::new(
            &vk_ctx.instance,
            vk_ctx.physical_device,
            &vk_ctx.device,
            vk_ctx.queue,
            vk_ctx.queue_family_index,
            size,
        )
        .map_err(|e| log::warn!("space warp: no readback ({e:?})"))
        .ok();
        let import = |images: Vec<u64>, format: wgpu::TextureFormat, hal: wgpu::TextureUses| -> Vec<[EyeTarget; 2]> {
            images
                .into_iter()
                .map(|raw| {
                    std::array::from_fn(|eye| {
                        let tex = unsafe {
                            vulkan_interop::import_vk_image_as_wgpu_with(
                                device,
                                vk::Image::from_raw(raw),
                                format,
                                (size.0, size.1, 2),
                                hal,
                                wgpu::TextureUsages::RENDER_ATTACHMENT,
                            )
                        };
                        let view = tex.create_view(&wgpu::TextureViewDescriptor {
                            dimension: Some(wgpu::TextureViewDimension::D2),
                            base_array_layer: eye as u32,
                            array_layer_count: Some(1),
                            ..Default::default()
                        });
                        EyeTarget { _texture: tex, view }
                    })
                })
                .collect()
        };
        let motion_targets = import(
            motion.enumerate_images()?,
            crate::renderer::space_warp::MOTION_FORMAT,
            wgpu::TextureUses::COLOR_TARGET,
        );
        let depth_targets = import(
            depth.enumerate_images()?,
            depth_format,
            wgpu::TextureUses::DEPTH_STENCIL_WRITE | wgpu::TextureUses::DEPTH_STENCIL_READ,
        );
        let pipelines = crate::renderer::space_warp::MotionPipelines::new(device, depth_format);
        let cameras = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("space_warp_cameras"),
            size: crate::renderer::space_warp::SLOT_STRIDE * crate::renderer::space_warp::MAX_SLOTS as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let camera_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("space_warp_cameras"),
            layout: &pipelines.camera_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &cameras,
                    offset: 0,
                    size: wgpu::BufferSize::new(std::mem::size_of::<crate::renderer::space_warp::MotionCamera>() as u64),
                }),
            }],
        });
        info!(
            "space warp: available, motion vectors {}x{}, depth {:?} ({:?}), {} + {} images",
            size.0,
            size.1,
            depth_format,
            vk_depth,
            motion_targets.len(),
            depth_targets.len(),
        );
        let info = std::array::from_fn(|eye| crate::renderer::space_warp::layer_info(&motion, &depth, eye as u32, size));
        Ok(Self {
            motion,
            depth,
            motion_targets,
            depth_targets,
            size,
            pipelines,
            cameras,
            camera_group,
            prev: None,
            prev_models: HashMap::new(),
            prev_clocks: None,
            acquired: None,
            info,
            motion_raw,
            depth_raw,
            depth_has_stencil: depth_format.has_stencil_aspect(),
            readback,

            reflect_groups: Vec::new(),
        })
    }
}

struct EyeTarget {
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShadowQuality {

    Off,

    SunOnly,

    SunAndSpot,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MsaaLevel {
    Off,
    X4,
}

pub const DEFAULT_MSAA: MsaaLevel = MsaaLevel::X4;

impl MsaaLevel {
    pub fn samples(self) -> u32 {
        match self {
            MsaaLevel::Off => 1,
            MsaaLevel::X4 => 4,
        }
    }
}

struct WaterBody {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    uniform_buffer: wgpu::Buffer,

    bind_groups: [wgpu::BindGroup; 2],
    index_count: u32,

    uniform: crate::renderer::water_pipeline::WaterUniform,

    waves: crate::renderer::water_waves::WaveField,

    motion_groups: Option<[wgpu::BindGroup; 2]>,

    bounds: (glam::Vec3, glam::Vec3),

    seen: std::cell::Cell<bool>,

    under_groups: [wgpu::BindGroup; 2],

    still: crate::renderer::underwater::StillDepth,
}

struct StereoProbePass {
    pass: crate::renderer::brush_pipeline::BrushPipeline,
    reader: crate::renderer::brush_pipeline::BrushPipeline,

    reader_sunless: crate::renderer::brush_pipeline::BrushPipeline,
    reader_baked: crate::renderer::brush_pipeline::BrushPipeline,
    target: crate::renderer::brush_pipeline::probe_pass::Target,
}

struct StereoScenePipelines {
    solid: SolidPipeline,
    wire: WirePipeline,
    mesh: MeshPipeline,
    skinned_mesh: SkinnedMeshPipeline,
    terrain: crate::renderer::terrain_pipeline::TerrainPipeline,
    layered_mesh: crate::renderer::layered_mesh_pipeline::LayeredMeshPipeline,
    water: crate::renderer::water_pipeline::WaterPipeline,
    sky: crate::renderer::sky::SkyPipeline,
    brush: crate::renderer::brush_pipeline::BrushPipeline,
    brush_opaque: crate::renderer::brush_pipeline::BrushPipeline,
    brush_sources: crate::renderer::brush_pipeline::BrushPipeline,
    brush_seal: crate::renderer::brush_pipeline::BrushSealPipeline,
    particle: ParticlePipeline,
    glare: crate::renderer::glare::GlarePipeline,
    effects: crate::renderer::effects::EffectsPipeline,
}

pub struct XrRenderer {
    pub swapchain: xr::Swapchain<xr::Vulkan>,
    pub width: u32,
    pub height: u32,
    wgpu_device: wgpu::Device,
    wgpu_queue: wgpu::Queue,
    solid_pipeline: SolidPipeline,
    water_pipeline: crate::renderer::water_pipeline::WaterPipeline,

    water_bodies: Vec<WaterBody>,

    underwater: crate::renderer::underwater::UnderwaterGpu,

    surfacing: crate::renderer::underwater::Surfacing,

    water_seconds: f64,

    water_step: f32,
    brush_pipeline: crate::renderer::brush_pipeline::BrushPipeline,

    brush_sources_pipeline: crate::renderer::brush_pipeline::BrushPipeline,

    brush_seal_pipeline: crate::renderer::brush_pipeline::BrushSealPipeline,

    brush_opaque_pipeline: crate::renderer::brush_pipeline::BrushPipeline,

    brush_probe_pass_pipeline: crate::renderer::brush_pipeline::BrushPipeline,
    brush_probe_reader_pipeline: crate::renderer::brush_pipeline::BrushPipeline,

    brush_probe_reader_sunless_pipeline: crate::renderer::brush_pipeline::BrushPipeline,
    brush_probe_reader_baked_pipeline: crate::renderer::brush_pipeline::BrushPipeline,

    spotless_readers: [crate::renderer::brush_pipeline::BrushPipeline; 3],
    spotless_mesh: MeshPipeline,

    spotless_frame: std::sync::atomic::AtomicBool,
    sun_faces: Option<crate::renderer::brush_pipeline::SunFaces>,
    probe_pass_targets: [crate::renderer::brush_pipeline::probe_pass::Target; 2],

    brush_probe_pass_deferred_pipeline: crate::renderer::brush_pipeline::BrushPipeline,

    brush_probe_pass_poolless_pipeline: crate::renderer::brush_pipeline::BrushPipeline,
    terrain_probe_pass_poolless_pipeline: crate::renderer::terrain_pipeline::TerrainPipeline,

    pass_cut_pipeline: Option<(String, crate::renderer::brush_pipeline::BrushPipeline)>,

    terrain_cut_pipeline: Option<(String, crate::renderer::terrain_pipeline::TerrainPipeline)>,

    thin_sampled_pipeline: Option<crate::renderer::mesh_pipeline::MeshPipeline>,
    scene_cut_pipeline: Option<(String, crate::renderer::brush_pipeline::BrushPipeline)>,

    water_cut_pipeline: Option<(String, wgpu::RenderPipeline)>,
    cut_inputs: (wgpu::TextureFormat, u32, wgpu::BindGroupLayout),
    probe_fixups: crate::renderer::probe_fixup::ProbeFixups,
    probe_fixup_targets: [wgpu::BindGroup; 2],

    probe_fixup_passes: [wgpu::BindGroup; 2],

    probe_blur: crate::renderer::probe_blur::ProbeBlur,
    probe_blur_groups: [Option<wgpu::BindGroup>; 2],

    character_cards: Option<crate::renderer::character_cards::CharacterCards>,

    skin_compute: crate::renderer::skin_compute::SkinCompute,

    posed_cache: std::sync::Mutex<crate::renderer::skin_compute::PosedCache>,

    card_atlas: Option<crate::renderer::proxy_cards::CardAtlas>,

    pool_cards: Option<crate::renderer::pool_cards::PoolCards>,

    terrain_probe_pass_pipeline: crate::renderer::terrain_pipeline::TerrainPipeline,
    terrain_probe_reader_pipeline: crate::renderer::terrain_pipeline::TerrainPipeline,

    terrain_reader_twins: [crate::renderer::terrain_pipeline::TerrainPipeline; 3],

    terrain_gentle: Option<[crate::renderer::terrain_pipeline::TerrainPipeline; 4]>,

    terrain_steep: Option<[crate::renderer::terrain_pipeline::TerrainPipeline; 4]>,

    slope_split: Option<crate::renderer::ground_twins::SlopeSplit>,

    terrain_dedup_passes: Option<[crate::renderer::terrain_pipeline::TerrainPipeline; 2]>,
    terrain_sun_baked: bool,

    terrain_inlined: Option<[crate::renderer::terrain_pipeline::TerrainPipeline; 3]>,

    reader_edits: Option<(
        String,
        [crate::renderer::brush_pipeline::BrushPipeline; 6],
        [crate::renderer::terrain_pipeline::TerrainPipeline; 4],
    )>,

    stereo_probe: Option<StereoProbePass>,

    brush_depth_prepass: crate::renderer::brush_pipeline::BrushPipeline,

    stereo_depth_prepass: Option<crate::renderer::brush_pipeline::BrushPipeline>,
    brush_mirror_pipeline: crate::renderer::brush_pipeline::BrushPipeline,
    brush_materials: crate::renderer::brush_pipeline::BrushMaterials,
    terrain_pipeline: crate::renderer::terrain_pipeline::TerrainPipeline,

    shadow_map: crate::renderer::shadow::ShadowMap,

    sky_pipeline: crate::renderer::sky::SkyPipeline,
    sky_mirror_pipeline: crate::renderer::sky::SkyPipeline,
    sky: crate::renderer::sky::Sky,
    post: crate::renderer::uniforms::PostUpload,
    player: crate::renderer::uniforms::PlayerUpload,

    pub shadow_quality: ShadowQuality,

    msaa: MsaaLevel,
    layered_mesh_pipeline: crate::renderer::layered_mesh_pipeline::LayeredMeshPipeline,
    layered_mesh_mirror_pipeline: crate::renderer::layered_mesh_pipeline::LayeredMeshPipeline,

    terrain_material: crate::renderer::terrain_pipeline::TerrainMaterial,

    terrain_layers: Vec<Option<crate::renderer::terrain_pipeline::TerrainImage>>,
    terrain_normals: Vec<Option<crate::renderer::terrain_pipeline::TerrainImage>>,
    terrain_rough: Vec<Option<crate::renderer::terrain_pipeline::TerrainImage>>,
    terrain_ao: Vec<Option<crate::renderer::terrain_pipeline::TerrainImage>>,
    terrain_splat: Option<crate::renderer::terrain_pipeline::TerrainImage>,

    terrain_sky_occlusion: Option<crate::renderer::terrain_pipeline::TerrainImage>,

    terrain_stationary: Vec<crate::renderer::terrain_pipeline::TerrainImage>,

    terrain_footprint: Option<(glam::Vec3, glam::Vec3)>,

    terrain_settings: crate::renderer::terrain_pipeline::TerrainMaterialUniform,
    wire_pipeline: WirePipeline,
    mesh_pipeline: MeshPipeline,
    skinned_mesh_mirror_pipeline: SkinnedMeshPipeline,
    skinned_mesh_pipeline: SkinnedMeshPipeline,
    mirror_solid_pipeline: SolidPipeline,
    mirror_mesh_pipeline: MeshPipeline,
    mirror_pipeline: MirrorPipeline,
    mirror_targets: [MirrorTarget; 2],
    mirror_model_uniform: ModelUniform,
    mirror_reflected_vp_uniform: ModelUniform,
    ssr_pipelines: SsrPipelines,
    ssr_solid_pipeline: SolidPipeline,

    brush_ssr_pipeline: crate::renderer::brush_pipeline::BrushPipeline,

    brush_ssr_trace_pipeline: crate::renderer::brush_pipeline::BrushPipeline,
    brush_ssr_composite_pipeline: crate::renderer::brush_pipeline::BrushPipeline,
    reflection_targets: [crate::renderer::ssr::ReflectionTarget; 2],

    reflection_composite_bg: [wgpu::BindGroup; 2],

    brush_ssr_sources_pipeline: crate::renderer::brush_pipeline::BrushPipeline,
    brush_ssr_debug_pipeline: crate::renderer::brush_pipeline::BrushPipeline,

    debug_view: crate::renderer::brush_pipeline::DebugView,
    scene_targets: [SceneTarget; 2],
    ssr_camera_uniform: SsrCameraUniform,
    particle_pipeline: ParticlePipeline,

    glare_pipeline: crate::renderer::glare::GlarePipeline,

    glare_sources: Vec<crate::renderer::glare::GlareSource>,

    glare_capsules: Vec<(glam::Vec3, glam::Vec3, f32)>,

    effect_emitters: Vec<crate::renderer::effects::EffectEmitter>,

    effect_seen_until: Vec<f64>,

    splashes: Vec<crate::renderer::effects::Splash>,
    effects_layout: wgpu::BindGroupLayout,
    effects_pipeline: crate::renderer::effects::EffectsPipeline,

    effects_gpu: Option<crate::renderer::effects::EffectsGpu>,

    weather: Option<crate::renderer::weather::WeatherScene>,

    floor_mirror_skinned: SkinnedMeshPipeline,
    floor_mirror_mips: crate::renderer::brush_pipeline::probe_pass::MirrorMips,

    uniform_buf: UniformBuffer,
    lights_uniform: LightsUniform,
    depth_view: wgpu::TextureView,
    eye_targets: Vec<[EyeTarget; 2]>,

    foveation: Option<FoveationState>,

    proxy_field_slots: Vec<crate::renderer::proxy_field::FieldSlot>,

    space_warp: Option<SpaceWarpState>,
    default_brush_lightmap: LoadedTexture,

    brush_lightmap: Option<LoadedTexture>,

    tod: time_of_day::TodState,
    cuboid_lightmaps: HashMap<String, LoadedTexture>,
    default_cuboid_lightmap: LoadedTexture,
    mesh_lightmaps: HashMap<String, LoadedTexture>,
    default_mesh_lightmap: LoadedTexture,

    probe_sampler: wgpu::Sampler,

    probe_view: Option<wgpu::TextureView>,

    probe_volumes: Vec<(u32, glam::Vec3, glam::Vec3, glam::Vec3)>,

    probe_brightness: Vec<f32>,

    room_descs: Vec<crate::renderer::probe_stream::ProbeDesc>,

    probe_rooms: Vec<u32>,

    probe_outdoor_volume: Option<u32>,

    probe_buildings: Vec<(glam::Vec3, glam::Vec3)>,

    pending_buildings: Vec<(glam::Vec3, glam::Vec3, Vec<u8>)>,

    ground_heights: Option<crate::renderer::ground_map::HeightGrid>,

    ground_dirty: bool,

    ground_placement: Option<([f32; 4], f32)>,

    probe_portals: Vec<crate::renderer::uniforms::ProbePortal>,

    cull_rooms: Vec<crate::renderer::portal_cull::CullRoom>,

    doors: Vec<crate::renderer::doors::DoorView>,

    shut_portals: Vec<bool>,

    probe_proxies: Vec<crate::renderer::uniforms::ProbeProxy>,

    levers: crate::renderer::levers::Levers,

    eye_capture_enabled: bool,
    eye_capture_served: u32,

    probe_stream: std::cell::RefCell<Option<crate::renderer::probe_stream::ProbeStream>>,

    eye: std::cell::RefCell<crate::renderer::exposure::EyeAdaptation>,

    auto_exposure: bool,

    last_frame_at: std::cell::Cell<Option<std::time::Instant>>,
    frame_stats: FrameStats,

    pass_timers: Option<crate::renderer::pass_timers::PassTimers>,

    perf_windows: u64,

    perf_window_index: u64,

    perf_warmup: bool,

    perf_metrics: Option<crate::xr::PerfMetrics>,

    perf_settings: Option<crate::xr::PerfSettings>,

    recommended_resolution: Option<crate::xr::RecommendedResolution>,
    recommendation_window: crate::renderer::dynamic_resolution::RecommendationWindow,

    perf_metric_window: crate::perf_metrics_log::WindowMeans,

    perf_log: Option<crate::perf_record::PerfLog>,

    started_at: std::time::Instant,

    pinned_head: Option<(glam::Vec3, glam::Quat)>,

    last_fov: Option<[xr::Fovf; 2]>,

    baked_lights: Vec<crate::renderer::lights::Light>,

    shadow_spot_incumbents: std::cell::RefCell<Vec<usize>>,

    shadow_slot_log: std::cell::RefCell<Vec<usize>>,

    character_shadow_held: std::cell::RefCell<Vec<glam::Vec3>>,

    static_sun_shadow: std::cell::RefCell<Option<crate::renderer::lights::StaticSunShadow>>,

    stereo_pipelines: Option<StereoScenePipelines>,

    stereo_scene: Option<StereoSceneTextures>,

    multiview_scene: bool,
    screen_space_reflections: bool,

    buffered_reflections: bool,

    shadow_diag_frames: std::cell::Cell<u64>,
}

struct WindowStats {
    frames: u64,
    cpu_avg: f64,
    cpu_max: f64,
    gpu_avg: f64,
    gpu_max: f64,
    frame_ms: f64,
    fps: f64,
}

struct FrameOutcome {

    measured: bool,

    closed: Option<WindowStats>,
}

struct FrameStats {
    window: u64,

    settle: u64,

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
    fn new(window: u64, settle: u64) -> Self {
        Self {
            window,
            settle: settle.min(window.saturating_sub(1)),
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

    fn restart(&mut self) {
        let (window, settle) = (self.window, self.settle);
        *self = Self::new(window, settle);
    }

    fn record(
        &mut self,
        cpu: std::time::Duration,
        gpu: std::time::Duration,
        now: std::time::Instant,
    ) -> FrameOutcome {
        self.count += 1;
        let measured = self.count > self.settle;
        if measured {
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
        }
        self.last_frame = Some(now);
        if self.count < self.window {
            return FrameOutcome { measured, closed: None };
        }

        let frames = self.window - self.settle;
        let n = frames as f64;
        let avg_period = if self.period_samples > 0 {
            self.period_ms_sum / self.period_samples as f64
        } else {
            0.0
        };
        let fps = if avg_period > 0.0 { 1000.0 / avg_period } else { 0.0 };
        let stats = WindowStats {
            frames,
            cpu_avg: self.cpu_ms_sum / n,
            cpu_max: self.cpu_ms_max,
            gpu_avg: self.gpu_ms_sum / n,
            gpu_max: self.gpu_ms_max,
            frame_ms: avg_period,
            fps,
        };
        info!(
            "PERF: cpu_avg={:.2}ms cpu_max={:.2}ms | gpu_avg={:.2}ms gpu_max={:.2}ms | frame={:.2}ms (~{:.1}fps) over {} frames",
            stats.cpu_avg, stats.cpu_max, stats.gpu_avg, stats.gpu_max, stats.frame_ms, stats.fps, stats.frames,
        );
        self.count = 0;
        self.cpu_ms_sum = 0.0;
        self.gpu_ms_sum = 0.0;
        self.period_ms_sum = 0.0;
        self.period_samples = 0;
        self.cpu_ms_max = 0.0;
        self.gpu_ms_max = 0.0;
        FrameOutcome { measured, closed: Some(stats) }
    }
}

impl XrRenderer {
    pub fn new(
        vk: &VkContext,
        xr_ctx: &XrContext,
        session: &xr::Session<xr::Vulkan>,
    ) -> Result<Self, Box<dyn std::error::Error>> {

        Self::new_with_msaa(vk, xr_ctx, session, DEFAULT_MSAA)
    }

    pub fn new_with_msaa(
        vk: &VkContext,
        xr_ctx: &XrContext,
        session: &xr::Session<xr::Vulkan>,
        msaa: MsaaLevel,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let view_configs = xr_ctx.instance.enumerate_view_configuration_views(
            xr_ctx.system,
            xr::ViewConfigurationType::PRIMARY_STEREO,
        )?;

        let recommended_w = view_configs[0].recommended_image_rect_width;
        let recommended_h = view_configs[0].recommended_image_rect_height;
        let scale = crate::renderer::RENDER_SCALE.clamp(0.3, 1.0);

        let round8 = |v: u32| (((v as f32 * scale) as u32).div_ceil(8) * 8).max(8);
        let width = round8(recommended_w);
        let height = round8(recommended_h);
        info!(
            "XrRenderer: {width}x{height} (render scale {scale:.2} of \
             {recommended_w}x{recommended_h} -- {:.0}% of the pixels)",
            100.0 * (f64::from(width) * f64::from(height))
                / (f64::from(recommended_w) * f64::from(recommended_h)),
        );

        let vk_format = vk::Format::R8G8B8A8_SRGB;
        let wgpu_format = wgpu::TextureFormat::Rgba8UnormSrgb;

        let eye_capture_enabled = crate::xr::vulkan::eye_capture_enabled();
        if eye_capture_enabled {
            info!("XrRenderer: eye capture enabled (debug.spacesoup.eyecapture)");
        }
        let swapchain = session.create_swapchain(&xr::SwapchainCreateInfo {
            create_flags: xr::SwapchainCreateFlags::EMPTY,
            usage_flags: xr::SwapchainUsageFlags::COLOR_ATTACHMENT
                | xr::SwapchainUsageFlags::SAMPLED
                | if eye_capture_enabled { xr::SwapchainUsageFlags::TRANSFER_SRC } else { xr::SwapchainUsageFlags::EMPTY },
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

        let foveation = (vk.fragment_density_map && unsafe { vulkan_interop::enable_foveation(&wgpu_device) }).then(|| {
            info!("foveation: available, {}x{} eye images", width, height);
            let probe = (width.div_ceil(2).max(1), height.div_ceil(2).max(1));
            FoveationState { eye_size: (width, height), probe_size: probe, maps: HashMap::new(), applied: None }
        });

        unsafe { vulkan_interop::start_pipeline_cache(vk, &wgpu_device) };

        wgpu_device.on_uncaptured_error(std::sync::Arc::new(|e| {
            error!("wgpu: {e}");
        }));

        let timestamp_period = vk.timestamp_period_ns;

        let pass_timers = timestamp_period.and_then(|period| {
            crate::renderer::pass_timers::PassTimers::new(
                &wgpu_device,

                &["scene_l", "eye_l", "scene_r", "eye_r", "prep_l", "prep_r", "refl_l", "refl_r", "probe_l", "probe_r", "fix_l", "fix_r", "mirror_l", "mirror_r", "mips_l", "mips_r", "blur_l", "blur_r", "cards", "card_mips", "pools", "pool_mips"],
                period,
            )
        });

        let lights_uniform = LightsUniform::new(&wgpu_device);

        let shadow_map = crate::renderer::shadow::ShadowMap::with_dimension(
            &wgpu_device,
            crate::renderer::shadow::QUEST_SHADOW_DIM,
        );
        log::info!(
            "shadow maps: Depth32Float filters linearly: {} (D16: {})",
            vulkan_interop::filters_linearly(&wgpu_device, vk::Format::D32_SFLOAT),
            vulkan_interop::filters_linearly(&wgpu_device, vk::Format::D16_UNORM),
        );
        let uniform_buf = UniformBuffer::new(
            &wgpu_device,
            &lights_uniform,
            shadow_map.sun_depth_view(),
            shadow_map.sun_dynamic_depth_view(),
            shadow_map.spot_depth_view(),
            shadow_map.sampler(),
        );
        let (_default_probe_view, probe_sampler) =
            crate::renderer::uniforms::default_probe_cube(&wgpu_device);

        let samples = msaa.samples();
        let solid_pipeline =
            SolidPipeline::new_multisampled(&wgpu_device, wgpu_format, &uniform_buf.layout, samples);
        let water_pipeline = crate::renderer::water_pipeline::WaterPipeline::new(
            &wgpu_device, wgpu_format, &uniform_buf.layout, samples,
        );
        let brush_pipeline = crate::renderer::brush_pipeline::BrushPipeline::new_multisampled(
            &wgpu_device, wgpu_format, &uniform_buf.layout, samples,
        );
        let brush_sources_pipeline = crate::renderer::brush_pipeline::BrushPipeline::new_multisampled_sources(
            &wgpu_device, wgpu_format, &uniform_buf.layout, samples,
        );
        let brush_seal_pipeline = crate::renderer::brush_pipeline::BrushSealPipeline::new(
            &wgpu_device, wgpu_format, &uniform_buf.layout, samples,
        );
        let brush_opaque_pipeline = crate::renderer::brush_pipeline::BrushPipeline::new_multisampled_opaque(
            &wgpu_device, wgpu_format, &uniform_buf.layout, samples,
        );
        let brush_mirror_pipeline = crate::renderer::brush_pipeline::BrushPipeline::new_mirror(
            &wgpu_device, wgpu_format, &uniform_buf.layout,
        );

        let probe_pass_layout = crate::renderer::brush_pipeline::probe_pass::bind_group_layout(&wgpu_device);
        let underwater = crate::renderer::underwater::UnderwaterGpu::new(&wgpu_device, wgpu_format, &uniform_buf.layout, &probe_pass_layout, samples);

        let effects_layout = crate::renderer::effects::bind_group_layout(&wgpu_device);
        let brush_probe_pass_pipeline = crate::renderer::brush_pipeline::BrushPipeline::new_probe_pass(
            &wgpu_device, &uniform_buf.layout, crate::renderer::multiview::ViewMode::Mono,
        );
        if crate::renderer::shader_checks::PIPELINE_STATISTICS.load(std::sync::atomic::Ordering::Relaxed) {
            crate::renderer::brush_pipeline::BrushPipeline::log_probe_pass_register_cuts(&wgpu_device, &uniform_buf.layout);
        }
        let brush_probe_reader_pipeline = crate::renderer::brush_pipeline::BrushPipeline::new_multisampled_probe_reader(
            &wgpu_device, wgpu_format, &uniform_buf.layout, samples, &probe_pass_layout,
            crate::renderer::multiview::ViewMode::Mono,
        );
        let spotless_readers = crate::renderer::brush_pipeline::BrushPipeline::new_spotless_probe_readers(
            &wgpu_device, wgpu_format, &uniform_buf.layout, samples, &probe_pass_layout,
        );
        let [brush_probe_reader_sunless_pipeline, brush_probe_reader_baked_pipeline] =
            [crate::renderer::brush_pipeline::FaceSun::Never, crate::renderer::brush_pipeline::FaceSun::Baked].map(|class| {
                crate::renderer::brush_pipeline::BrushPipeline::new_multisampled_probe_reader_for(
                    &wgpu_device, wgpu_format, &uniform_buf.layout, samples, &probe_pass_layout,
                    crate::renderer::multiview::ViewMode::Mono, class,
                )
            });
        if crate::renderer::shader_checks::PIPELINE_STATISTICS.load(std::sync::atomic::Ordering::Relaxed) {
            crate::renderer::brush_pipeline::BrushPipeline::log_scene_register_cuts(
                &wgpu_device, wgpu_format, &uniform_buf.layout, samples, &probe_pass_layout,
            );
            crate::renderer::mesh_pipeline::MeshPipeline::log_register_cuts(
                &wgpu_device, wgpu_format, &uniform_buf.layout, samples,
            );
        }
        let probe_pass_targets: [crate::renderer::brush_pipeline::probe_pass::Target; 2] = std::array::from_fn(|_| {
            crate::renderer::brush_pipeline::probe_pass::Target::new(&wgpu_device, &probe_pass_layout, width, height, 1)
        });
        let probe_fixups = crate::renderer::probe_fixup::ProbeFixups::new(
            &wgpu_device,
            &uniform_buf.layout,
            probe_pass_targets[0].width * probe_pass_targets[0].height,
        );
        let brush_probe_pass_deferred_pipeline =
            crate::renderer::brush_pipeline::BrushPipeline::new_probe_pass_deferred(&wgpu_device, &uniform_buf.layout, &probe_fixups);
        let probe_fixup_targets: [wgpu::BindGroup; 2] =
            std::array::from_fn(|eye| probe_fixups.target_bind_group(&wgpu_device, &probe_pass_targets[eye]));
        let probe_fixup_passes: [wgpu::BindGroup; 2] =
            std::array::from_fn(|eye| probe_fixups.pass_bind_group_for(&wgpu_device, &probe_pass_targets[eye]));
        let probe_blur = crate::renderer::probe_blur::ProbeBlur::new(&wgpu_device);
        let probe_blur_groups: [Option<wgpu::BindGroup>; 2] = std::array::from_fn(|eye| {
            probe_blur.bind_group(&wgpu_device, &probe_pass_targets[eye])
        });
        let terrain_probe_pass_pipeline =
            crate::renderer::terrain_pipeline::TerrainPipeline::new_probe_pass(&wgpu_device, &uniform_buf.layout, &probe_fixups);
        let brush_probe_pass_poolless_pipeline = crate::renderer::brush_pipeline::BrushPipeline::new_probe_pass_deferred_poolless(
            &wgpu_device, &uniform_buf.layout, &probe_fixups,
        );
        let terrain_probe_pass_poolless_pipeline = crate::renderer::terrain_pipeline::TerrainPipeline::new_probe_pass_poolless(
            &wgpu_device, &uniform_buf.layout, &probe_fixups,
        );
        let terrain_dedup_passes = crate::renderer::ground_twins::dedup_passes(&wgpu_device, &uniform_buf.layout, &probe_fixups, None);
        if terrain_dedup_passes.is_none() {
            log::warn!("terrain: the probe pass's dedup edits no longer match; it draws as it was");
        }
        let terrain_probe_reader_pipeline = crate::renderer::terrain_pipeline::TerrainPipeline::new_probe_reader(
            &wgpu_device, wgpu_format, &uniform_buf.layout, samples, &probe_pass_layout,
        );
        let terrain_reader_twins = crate::renderer::terrain_pipeline::TerrainPipeline::new_probe_reader_twins(
            &wgpu_device, wgpu_format, &uniform_buf.layout, samples, &probe_pass_layout,
        );
        let terrain_gentle = crate::renderer::ground_twins::gentle_readers(
            &wgpu_device, wgpu_format, &uniform_buf.layout, samples, &probe_pass_layout, None,
        );
        let terrain_steep = terrain_gentle.as_ref().and_then(|_| {
            crate::renderer::ground_twins::steep_readers(&wgpu_device, wgpu_format, &uniform_buf.layout, samples, &probe_pass_layout)
        });
        if terrain_gentle.is_none() {
            log::warn!("terrain: the ground's steep tests are not where its gentle twins take them out; the full readers draw everywhere");
        }
        if crate::renderer::shader_checks::PIPELINE_STATISTICS.load(std::sync::atomic::Ordering::Relaxed) {
            crate::renderer::brush_pipeline::BrushPipeline::log_deferred_register_cuts(
                &wgpu_device, &uniform_buf.layout, &probe_fixups,
            );
            crate::renderer::terrain_pipeline::TerrainPipeline::log_probe_pass_register_cuts(
                &wgpu_device, &uniform_buf.layout, &probe_fixups,
            );
            probe_fixups.log_register_cuts(&wgpu_device);
            crate::renderer::ground_cuts::log_ground_cuts(
                &wgpu_device, wgpu_format, &uniform_buf.layout, samples, &probe_pass_layout, &probe_fixups,
            );

            let _ = crate::renderer::terrain_pipeline::TerrainPipeline::new_inlined(
                &wgpu_device, wgpu_format, &uniform_buf.layout, samples, &probe_pass_layout, &probe_fixups,
            );
        }
        let brush_depth_prepass = crate::renderer::brush_pipeline::BrushPipeline::new_depth_prepass(
            &wgpu_device, wgpu_format, &uniform_buf.layout, samples, crate::renderer::multiview::ViewMode::Mono,
        );

        let brush_materials = crate::renderer::brush_pipeline::BrushMaterials::fallback(
            &wgpu_device, &wgpu_queue, &brush_pipeline.material_layout,
        );
        let terrain_pipeline = crate::renderer::terrain_pipeline::TerrainPipeline::new_multisampled(
            &wgpu_device, wgpu_format, &uniform_buf.layout, samples,
        );
        let terrain_material = crate::renderer::terrain_pipeline::TerrainMaterial::fallback(
            &wgpu_device, &wgpu_queue, &terrain_pipeline.material_layout,
        );
        let sky_pipeline = crate::renderer::sky::SkyPipeline::new(
            &wgpu_device, wgpu_format, &uniform_buf.layout, samples,
        );

        let sky_mirror_pipeline = crate::renderer::sky::SkyPipeline::new(
            &wgpu_device, wgpu_format, &uniform_buf.layout, 1,
        );

        let sky = crate::renderer::sky::Sky::none(
            &wgpu_device,
            &wgpu_queue,
            &sky_pipeline.layout,
            crate::renderer::sky::AMBIENT,
        );

        let layered_mesh_pipeline =
            crate::renderer::layered_mesh_pipeline::LayeredMeshPipeline::new_multisampled(
                &wgpu_device,
                wgpu_format,
                &uniform_buf.layout,
                &terrain_pipeline.material_layout,
                samples,
            );
        let layered_mesh_mirror_pipeline =
            crate::renderer::layered_mesh_pipeline::LayeredMeshPipeline::new_mirror(
                &wgpu_device, wgpu_format, &uniform_buf.layout, &terrain_pipeline.material_layout,
            );
        let wire_pipeline =
            WirePipeline::new_multisampled(&wgpu_device, wgpu_format, &uniform_buf.layout, samples);
        let mesh_pipeline =
            MeshPipeline::new_multisampled(&wgpu_device, wgpu_format, &uniform_buf.layout, samples);
        let spotless_mesh =
            MeshPipeline::new_multisampled_spotless(&wgpu_device, wgpu_format, &uniform_buf.layout, samples);
        let skinned_mesh_pipeline =
            SkinnedMeshPipeline::new_multisampled(&wgpu_device, wgpu_format, &uniform_buf.layout, samples);

        let skinned_mesh_mirror_pipeline =
            SkinnedMeshPipeline::new(&wgpu_device, wgpu_format, &uniform_buf.layout);

        let floor_mirror_skinned = SkinnedMeshPipeline::new(
            &wgpu_device,
            crate::renderer::brush_pipeline::probe_pass::MIRROR_FORMAT,
            &uniform_buf.layout,
        );
        let floor_mirror_mips = crate::renderer::brush_pipeline::probe_pass::MirrorMips::new(&wgpu_device);

        let mirror_solid_pipeline =
            SolidPipeline::new_mirror(&wgpu_device, wgpu_format, &uniform_buf.layout);
        let mirror_mesh_pipeline =
            MeshPipeline::new_mirror(&wgpu_device, wgpu_format, &uniform_buf.layout);
        let mirror_pipeline = MirrorPipeline::new(&wgpu_device, wgpu_format, &uniform_buf.layout);
        let mirror_targets: [MirrorTarget; 2] =
            std::array::from_fn(|_| mirror_pipeline.create_target(&wgpu_device, wgpu_format, width, height));
        let mirror_model_uniform = mirror_pipeline.create_model_uniform(&wgpu_device);
        let mirror_reflected_vp_uniform = mirror_pipeline.create_reflected_vp_uniform(&wgpu_device);

        let ssr_pipelines =
            SsrPipelines::new_with_depth_samples(&wgpu_device, wgpu_format, samples);
        let brush_ssr_pipeline = crate::renderer::brush_pipeline::BrushPipeline::new_ssr(
            &wgpu_device,
            wgpu_format,
            &uniform_buf.layout,
            ssr_pipelines.scene_texture_layout(),
        );
        let brush_ssr_sources_pipeline = crate::renderer::brush_pipeline::BrushPipeline::new_ssr_debug(
            &wgpu_device, wgpu_format, &uniform_buf.layout, ssr_pipelines.scene_texture_layout(),
            crate::renderer::brush_pipeline::DebugView::Sources,
        );
        let brush_ssr_debug_pipeline = crate::renderer::brush_pipeline::BrushPipeline::new_ssr_debug(
            &wgpu_device, wgpu_format, &uniform_buf.layout, ssr_pipelines.scene_texture_layout(),
            crate::renderer::brush_pipeline::DebugView::Ssr,
        );
        let brush_ssr_trace_pipeline = crate::renderer::brush_pipeline::BrushPipeline::new_ssr_trace(
            &wgpu_device,
            crate::renderer::ssr::REFLECTION_FORMAT,
            &uniform_buf.layout,
            ssr_pipelines.scene_texture_layout(),
        );
        let brush_ssr_composite_pipeline =
            crate::renderer::brush_pipeline::BrushPipeline::new_ssr_composite(
                &wgpu_device,
                wgpu_format,
                &uniform_buf.layout,
                ssr_pipelines.scene_texture_layout(),
            );
        let ssr_solid_pipeline = SolidPipeline::new_ssr(
            &wgpu_device,
            wgpu_format,
            &uniform_buf.layout,
            ssr_pipelines.camera_layout(),
            ssr_pipelines.scene_texture_layout(),
        );

        let can_multiview = wgpu_device
            .features()
            .contains(wgpu::Features::MULTIVIEW | wgpu::Features::MULTISAMPLE_ARRAY);
        info!(
            "renderer: stereo scene pass {}",
            if can_multiview { "available" } else { "UNAVAILABLE (missing MULTIVIEW/MULTISAMPLE_ARRAY)" },
        );

        let stereo_scope = wgpu_device.push_error_scope(wgpu::ErrorFilter::Validation);
        let stereo_pipelines = can_multiview.then(|| StereoScenePipelines {
            solid: SolidPipeline::new_multisampled_stereo(
                &wgpu_device, wgpu_format, &uniform_buf.layout, samples,
            ),
            wire: WirePipeline::new_multisampled_stereo(
                &wgpu_device, wgpu_format, &uniform_buf.layout, samples,
            ),
            mesh: MeshPipeline::new_multisampled_stereo(
                &wgpu_device, wgpu_format, &uniform_buf.layout, samples,
            ),
            skinned_mesh: SkinnedMeshPipeline::new_multisampled_stereo(
                &wgpu_device, wgpu_format, &uniform_buf.layout, samples,
            ),
            terrain: crate::renderer::terrain_pipeline::TerrainPipeline::new_multisampled_stereo(
                &wgpu_device, wgpu_format, &uniform_buf.layout, samples,
            ),
            layered_mesh:
                crate::renderer::layered_mesh_pipeline::LayeredMeshPipeline::new_multisampled_stereo(
                    &wgpu_device,
                    wgpu_format,
                    &uniform_buf.layout,

                    &terrain_pipeline.material_layout,
                    samples,
                ),
            water: crate::renderer::water_pipeline::WaterPipeline::new_stereo(
                &wgpu_device, wgpu_format, &uniform_buf.layout, samples,
            ),
            sky: crate::renderer::sky::SkyPipeline::new_stereo(
                &wgpu_device, wgpu_format, &uniform_buf.layout, samples,
            ),
            brush: crate::renderer::brush_pipeline::BrushPipeline::new_multisampled_stereo(
                &wgpu_device, wgpu_format, &uniform_buf.layout, samples,
            ),
            brush_opaque:
                crate::renderer::brush_pipeline::BrushPipeline::new_multisampled_opaque_stereo(
                    &wgpu_device, wgpu_format, &uniform_buf.layout, samples,
                ),
            brush_sources:
                crate::renderer::brush_pipeline::BrushPipeline::new_multisampled_sources_stereo(
                    &wgpu_device, wgpu_format, &uniform_buf.layout, samples,
                ),
            brush_seal: crate::renderer::brush_pipeline::BrushSealPipeline::new_stereo(
                &wgpu_device, wgpu_format, &uniform_buf.layout, samples,
            ),

            particle: ParticlePipeline::new_multisampled_stereo(
                &wgpu_device, wgpu_format, &uniform_buf.layout, samples,
            ),
            glare: crate::renderer::glare::GlarePipeline::new_multisampled_stereo(
                &wgpu_device, wgpu_format, &uniform_buf.layout, &probe_pass_layout, samples,
            ),
            effects: crate::renderer::effects::EffectsPipeline::new_multisampled_stereo(
                &wgpu_device, wgpu_format, &uniform_buf.layout, &probe_pass_layout, &effects_layout, samples,
            ),
        });

        let stereo_pipelines = match pollster::block_on(stereo_scope.pop()) {
            Some(e) => {
                error!("renderer: stereo pipelines REFUSED, multiview disabled -- {e}");
                None
            }
            None => stereo_pipelines,
        };
        let can_multiview = can_multiview && stereo_pipelines.is_some();

        let stereo_probe = if can_multiview {
            let scope = wgpu_device.push_error_scope(wgpu::ErrorFilter::Validation);
            let stereo = crate::renderer::multiview::ViewMode::Stereo;
            let built = StereoProbePass {
                pass: crate::renderer::brush_pipeline::BrushPipeline::new_probe_pass(&wgpu_device, &uniform_buf.layout, stereo),
                reader: crate::renderer::brush_pipeline::BrushPipeline::new_multisampled_probe_reader(
                    &wgpu_device, wgpu_format, &uniform_buf.layout, samples, &probe_pass_layout, stereo,
                ),
                reader_sunless: crate::renderer::brush_pipeline::BrushPipeline::new_multisampled_probe_reader_for(
                    &wgpu_device, wgpu_format, &uniform_buf.layout, samples, &probe_pass_layout, stereo,
                    crate::renderer::brush_pipeline::FaceSun::Never,
                ),
                reader_baked: crate::renderer::brush_pipeline::BrushPipeline::new_multisampled_probe_reader_for(
                    &wgpu_device, wgpu_format, &uniform_buf.layout, samples, &probe_pass_layout, stereo,
                    crate::renderer::brush_pipeline::FaceSun::Baked,
                ),
                target: crate::renderer::brush_pipeline::probe_pass::Target::new(
                    &wgpu_device, &probe_pass_layout, width, height, crate::renderer::multiview::STEREO_VIEWS,
                ),
            };
            match pollster::block_on(scope.pop()) {
                Some(e) => {
                    error!("renderer: stereo probe pass REFUSED, stereo traces reflections per pixel -- {e}");
                    None
                }
                None => Some(built),
            }
        } else {
            None
        };
        let stereo_depth_prepass = if can_multiview {
            let scope = wgpu_device.push_error_scope(wgpu::ErrorFilter::Validation);
            let built = crate::renderer::brush_pipeline::BrushPipeline::new_depth_prepass(
                &wgpu_device, wgpu_format, &uniform_buf.layout, samples, crate::renderer::multiview::ViewMode::Stereo,
            );
            match pollster::block_on(scope.pop()) {
                Some(e) => {
                    error!("renderer: stereo depth prepass REFUSED, stereo draws without one -- {e}");
                    None
                }
                None => Some(built),
            }
        } else {
            None
        };

        let (stereo_scene, scene_targets): (Option<StereoSceneTextures>, [SceneTarget; 2]) =
            if can_multiview {
                let (stereo, targets) = ssr_pipelines.create_scene_targets_stereo(
                    &wgpu_device, wgpu_format, width, height, samples,
                );
                (Some(stereo), targets)
            } else {
                (
                    None,
                    std::array::from_fn(|_| {
                        ssr_pipelines.create_scene_target_multisampled(
                            &wgpu_device, wgpu_format, width, height, samples,
                        )
                    }),
                )
            };

        let reflection_targets: [crate::renderer::ssr::ReflectionTarget; 2] =
            std::array::from_fn(|eye| {
                let _ = eye;
                ssr_pipelines.create_reflection_target(&wgpu_device, width, height)
            });
        let reflection_composite_bg: [wgpu::BindGroup; 2] = std::array::from_fn(|eye| {
            ssr_pipelines.create_composite_bind_group(
                &wgpu_device,
                &scene_targets[eye],
                &reflection_targets[eye],
            )
        });
        let ssr_camera_uniform = ssr_pipelines.create_camera_uniform(&wgpu_device);
        let particle_pipeline = ParticlePipeline::new_multisampled(
            &wgpu_device, wgpu_format, &uniform_buf.layout, samples,
        );
        let glare_pipeline = crate::renderer::glare::GlarePipeline::new_multisampled(
            &wgpu_device, wgpu_format, &uniform_buf.layout, &probe_pass_layout, samples,
        );
        let effects_pipeline = crate::renderer::effects::EffectsPipeline::new_multisampled(
            &wgpu_device, wgpu_format, &uniform_buf.layout, &probe_pass_layout, &effects_layout, samples,
        );

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

        let mut space_warp = xr_ctx.space_warp.and_then(|size| match SpaceWarpState::new(session, &wgpu_device, size, vk) {
            Ok(s) => Some(s),
            Err(e) => {
                log::warn!("space warp: unavailable ({e})");
                None
            }
        });

        let mut eye_targets: Vec<[EyeTarget; 2]> = Vec::new();
        for &raw_image in &raw_images {
            let targets = std::array::from_fn(|eye| {
                let wgpu_tex = unsafe {
                    if eye_capture_enabled {
                        vulkan_interop::import_vk_image_as_wgpu_with(
                            &wgpu_device,
                            raw_image,
                            wgpu_format,
                            (width, height, 2),
                            wgpu::TextureUses::COLOR_TARGET | wgpu::TextureUses::RESOURCE | wgpu::TextureUses::COPY_SRC,
                            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
                        )
                    } else {
                        vulkan_interop::import_vk_image_as_wgpu(&wgpu_device, raw_image, wgpu_format, width, height, 2)
                    }
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

        if let Some(sw) = space_warp.as_mut() {
            sw.reflect_groups = eye_targets
                .iter()
                .map(|eyes| {
                    std::array::from_fn(|eye| {
                        sw.pipelines.reflect_bind_group(&wgpu_device, &eyes[eye].view, &probe_pass_targets[eye].reach_view)
                    })
                })
                .collect();
        }

        let white_pixel = [255u8, 255, 255, 255];
        let default_cuboid_lightmap = create_lightmap_texture(
            &wgpu_device,
            &wgpu_queue,
            &solid_pipeline.lightmap_layout,
            &white_pixel,
            1,
            1,
            None,
        );

        let default_brush_lightmap = crate::renderer::brush_pipeline::default_brush_lightmap(
            &wgpu_device,
            &wgpu_queue,
            &brush_pipeline.lightmap_layout,
        );
        let default_mesh_lightmap = crate::renderer::brush_pipeline::default_brush_lightmap(
            &wgpu_device,
            &wgpu_queue,
            &mesh_pipeline.lightmap_layout,
        );

        let skin_compute = crate::renderer::skin_compute::SkinCompute::new(&wgpu_device);
        Ok(Self {
            swapchain,
            width,
            height,
            wgpu_device,
            wgpu_queue,
            solid_pipeline,
            brush_pipeline,
            brush_sources_pipeline,
            brush_seal_pipeline,
            brush_opaque_pipeline,
            brush_probe_pass_pipeline,
            brush_probe_reader_pipeline,
            brush_probe_reader_sunless_pipeline,
            brush_probe_reader_baked_pipeline,
            spotless_readers,
            spotless_mesh,
            spotless_frame: std::sync::atomic::AtomicBool::new(false),
            sun_faces: None,
            probe_pass_targets,
            brush_probe_pass_deferred_pipeline,
            brush_probe_pass_poolless_pipeline,
            terrain_probe_pass_poolless_pipeline,
            pass_cut_pipeline: None,
            terrain_cut_pipeline: None,
            thin_sampled_pipeline: None,
            scene_cut_pipeline: None,
            water_cut_pipeline: None,
            cut_inputs: (wgpu_format, samples, probe_pass_layout.clone()),
            probe_fixups,
            probe_fixup_targets,
            probe_fixup_passes,
            probe_blur,
            probe_blur_groups,
            character_cards: None,
            skin_compute,
            posed_cache: Default::default(),
            card_atlas: None,
            pool_cards: None,
            terrain_probe_pass_pipeline,
            terrain_probe_reader_pipeline,
            terrain_reader_twins,
            terrain_gentle,
            terrain_steep,
            slope_split: None,
            terrain_dedup_passes,
            terrain_sun_baked: false,
            terrain_inlined: None,
            weather: None,
            reader_edits: None,
            stereo_probe,
            brush_depth_prepass,
            stereo_depth_prepass,
            brush_mirror_pipeline,
            water_pipeline,
            water_bodies: Vec::new(),
            underwater,
            surfacing: Default::default(),
            water_seconds: 0.0,
            water_step: 0.0,
            brush_materials,
            terrain_pipeline,
            shadow_map,
            sky_pipeline,
            sky_mirror_pipeline,
            sky,

            post: crate::renderer::uniforms::PostUpload::default(),
            player: crate::renderer::uniforms::PlayerUpload::default(),

            shadow_quality: ShadowQuality::SunAndSpot,
            msaa,
            skinned_mesh_mirror_pipeline,
            layered_mesh_pipeline,
            layered_mesh_mirror_pipeline,
            terrain_material,
            terrain_layers: vec![None, None, None, None],
            terrain_normals: vec![None, None, None, None],
            terrain_rough: vec![None, None, None, None],
            terrain_ao: vec![None, None, None, None],
            terrain_splat: None,
            terrain_sky_occlusion: None,
            terrain_stationary: Vec::new(),
            terrain_footprint: None,
            terrain_settings: Default::default(),
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
            brush_ssr_pipeline,
            brush_ssr_trace_pipeline,
            brush_ssr_composite_pipeline,
            reflection_targets,
            reflection_composite_bg,
            brush_ssr_sources_pipeline,
            brush_ssr_debug_pipeline,
            debug_view: Default::default(),
            scene_targets,
            ssr_camera_uniform,
            particle_pipeline,
            glare_pipeline,
            glare_sources: Vec::new(),
            glare_capsules: Vec::new(),
            effect_emitters: Vec::new(),
            effect_seen_until: Vec::new(),
            splashes: Vec::new(),
            effects_layout,
            effects_pipeline,
            effects_gpu: None,
            floor_mirror_skinned,
            floor_mirror_mips,
            uniform_buf,
            lights_uniform,
            depth_view,
            eye_targets,
            foveation,
            proxy_field_slots: Vec::new(),
            space_warp,
            default_brush_lightmap,
            brush_lightmap: None,
            tod: Default::default(),
            cuboid_lightmaps: HashMap::new(),
            default_cuboid_lightmap,
            mesh_lightmaps: HashMap::new(),
            default_mesh_lightmap,
            probe_sampler,
            probe_view: None,
            probe_volumes: Vec::new(),
            probe_brightness: Vec::new(),
            room_descs: Vec::new(),
            probe_rooms: Vec::new(),
            probe_outdoor_volume: None,
            probe_buildings: Vec::new(),
            pending_buildings: Vec::new(),
            ground_heights: None,
            ground_dirty: false,
            ground_placement: None,
            probe_portals: Vec::new(),
            cull_rooms: Vec::new(),
            doors: Vec::new(),
            shut_portals: Vec::new(),
            probe_proxies: Vec::new(),
            levers: crate::renderer::levers::Levers::default(),
            eye_capture_enabled,
            eye_capture_served: 0,
            probe_stream: std::cell::RefCell::new(None),
            eye: std::cell::RefCell::new(crate::renderer::exposure::EyeAdaptation::sky_only(
                crate::renderer::sky::SkyIrradiance::flat(crate::renderer::sky::AMBIENT),
            )),
            auto_exposure: true,
            last_frame_at: std::cell::Cell::new(None),

            frame_stats: FrameStats::new(120, 8),
            pass_timers,
            perf_windows: 0,
            perf_window_index: 0,

            perf_warmup: true,
            perf_metrics: crate::xr::PerfMetrics::new(&xr_ctx.instance, session),
            perf_settings: crate::xr::PerfSettings::new(&xr_ctx.instance, session),
            recommended_resolution: crate::xr::RecommendedResolution::new(
                &xr_ctx.instance,
                session,
                xr_ctx.has_recommended_resolution,
            ),
            recommendation_window: Default::default(),
            perf_metric_window: Default::default(),
            perf_log: None,
            started_at: std::time::Instant::now(),
            pinned_head: None,
            last_fov: None,
            baked_lights: Vec::new(),
            shadow_spot_incumbents: std::cell::RefCell::new(Vec::new()),
            shadow_slot_log: std::cell::RefCell::new(Vec::new()),
            character_shadow_held: std::cell::RefCell::new(Vec::new()),
            static_sun_shadow: std::cell::RefCell::new(None),
            multiview_scene: false,
            stereo_pipelines,
            stereo_scene,
            screen_space_reflections: crate::renderer::scene_pass_plan::XR_SCREEN_SPACE_REFLECTIONS,
            buffered_reflections: crate::renderer::ssr::BUFFERED_REFLECTIONS,
            shadow_diag_frames: std::cell::Cell::new(0),
        })
    }

    pub fn set_cuboid_lightmap(&mut self, key: &str, light: crate::renderer::mesh::LightmapLight, width: u32, height: u32) {
        let tex = crate::renderer::mesh::create_lightmap_texture_with_sun(
            &self.wgpu_device,
            &self.wgpu_queue,
            &self.solid_pipeline.lightmap_layout,
            light,
            width,
            height,
            None,
            None,
        );
        self.cuboid_lightmaps.insert(key.to_string(), tex);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn set_brush_lightmap(
        &mut self,
        light: crate::renderer::mesh::LightmapLight,
        width: u32,
        height: u32,
        direction: Option<(&[u8], u32, u32)>,
        sun_mask: Option<(&[u8], u32, u32)>,
        stationary: &[&[u8]],
        stationary_size: (u32, u32),
    ) {

        let sun_mask = sun_mask.map(|(rgba, w, h)| (crate::renderer::brush_pipeline::dilate_sun_mask(rgba, w, h), w, h));
        let sun_mask = sun_mask.as_ref().map(|(rgba, w, h)| (rgba.as_slice(), *w, *h));
        self.sun_faces = sun_mask.and_then(|(rgba, w, h)| crate::renderer::brush_pipeline::SunFaces::from_mask(rgba, w, h));
        self.brush_lightmap = Some(crate::renderer::mesh::create_lightmap_texture_full(
            &self.wgpu_device,
            &self.wgpu_queue,
            &self.brush_pipeline.lightmap_layout,
            light,
            width,
            height,
            direction,
            sun_mask,
            (!stationary.is_empty()).then_some((stationary, stationary_size.0, stationary_size.1)),
        ));
        let shipped = match light {
            crate::renderer::mesh::LightmapLight::Linear(t) => Some((t.to_vec(), width, height)),
            crate::renderer::mesh::LightmapLight::Srgb8(_) => None,
        };
        self.tod_brush_atlas_changed(shipped);
    }

    fn brush_lightmap_bg(&self) -> &wgpu::BindGroup {

        if let Some(bg) = self.tod_brush_lightmap_bg() {
            return bg;
        }
        self.brush_lightmap
            .as_ref()
            .map(|t| &t.bind_group)
            .unwrap_or(&self.default_brush_lightmap.bind_group)
    }

    pub fn set_reflection_probes(
        &mut self,
        probes: &[(&[u8], u32, glam::Vec3, glam::Vec3, glam::Vec3)],
    ) {
        let resolution = probes.first().map(|p| p.1).unwrap_or(1);

        let usable: Vec<&(&[u8], u32, glam::Vec3, glam::Vec3, glam::Vec3)> =
            probes.iter().filter(|p| p.1 == resolution).collect();
        let mismatched = probes.len() - usable.len();
        if mismatched > 0 {
            log::warn!(
                "reflection probes: {mismatched} of {} dropped for a face size other than {resolution}px; re-bake the level's probes",
                probes.len(),
            );
        }

        let mut boxes: Vec<(glam::Vec3, glam::Vec3)> = Vec::new();
        let descs: Vec<crate::renderer::probe_stream::ProbeDesc> = usable
            .iter()
            .map(|p| {
                let volume = match boxes.iter().position(|b| *b == (p.3, p.4)) {
                    Some(v) => v,
                    None => {
                        boxes.push((p.3, p.4));
                        boxes.len() - 1
                    }
                };

                crate::renderer::probe_stream::ProbeDesc { centre: p.2, min: p.3, max: p.4, volume: volume as u32, has_depth: true, room_light: None }
            })
            .collect();
        let owned: Arc<Vec<Vec<u8>>> = Arc::new(usable.iter().map(|p| p.0.to_vec()).collect());
        let source: crate::renderer::probe_stream::ProbeSource =
            Arc::new(move |i| owned.get(i).cloned());
        self.set_reflection_probes_streamed(descs, resolution, Vec::new(), source);
    }

    pub fn set_reflection_probes_streamed(
        &mut self,
        descs: Vec<crate::renderer::probe_stream::ProbeDesc>,
        resolution: u32,
        portals: Vec<crate::renderer::uniforms::ProbePortal>,
        source: crate::renderer::probe_stream::ProbeSource,
    ) {
        self.set_reflection_probes_with_depth(descs, resolution, portals, source, None);
    }

    pub fn set_building_outsides(&mut self, buildings: Vec<(glam::Vec3, glam::Vec3, Vec<u8>)>) {
        self.pending_buildings = buildings;
    }

    pub fn set_reflection_probes_with_depth(
        &mut self,
        descs: Vec<crate::renderer::probe_stream::ProbeDesc>,
        resolution: u32,
        portals: Vec<crate::renderer::uniforms::ProbePortal>,
        source: crate::renderer::probe_stream::ProbeSource,
        depth: Option<crate::renderer::probe_stream::ProbeDepthSource>,
    ) {
        use crate::renderer::uniforms::{ProbeUpload, MAX_PROBES};

        self.room_descs = descs.clone();
        log::info!(
            "room light: {} of {} photograph(s) carry their room's light for the models",
            descs.iter().filter(|d| d.room_light.is_some()).count(),
            descs.len(),
        );

        let mut eye = self.tod_meter_probes(&descs, resolution, &source);
        let source = self.tod_probe_source(source);

        eye.set_portals(&portals);
        *self.eye.borrow_mut() = eye;
        self.tod_probes_loaded();
        log::info!("reflection probes: average radiance by probe {:?}", self.probe_brightness);

        let sky_faces = self.sky.reflection.as_ref().map(|r| r.cube_faces(resolution));

        let (buildings, building_faces): (Vec<(glam::Vec3, glam::Vec3)>, Vec<Vec<u8>>) =
            {
                let pending = std::mem::take(&mut self.pending_buildings);
                self.tod_take_buildings(pending).into_iter().map(|(lo, hi, f)| ((lo, hi), f)).unzip()
            };
        self.probe_buildings = buildings;
        let stream = crate::renderer::probe_stream::ProbeStream::new_with_extras(
            &self.wgpu_device,
            &self.wgpu_queue,
            resolution,
            descs.len(),
            source,
            depth,
            sky_faces,
            building_faces,
        );

        self.probe_outdoor_volume = crate::renderer::probe_stream::outdoor_volume(&descs);

        self.probe_volumes =
            descs.iter().enumerate().map(|(i, d)| (i as u32, d.centre, d.min, d.max)).collect();
        self.probe_rooms = descs.iter().map(|d| d.volume).collect();
        self.probe_portals = portals;
        *self.probe_stream.get_mut() = Some(stream);

        let mut upload = ProbeUpload { count: descs.len().min(MAX_PROBES) as u32, ..Default::default() };
        for (slot, (id, c, lo, hi)) in self.probe_volumes.iter().take(MAX_PROBES).enumerate() {
            upload.set(slot, *id, *c, *lo, *hi);
        }
        upload.fill_brightness(&self.probe_brightness);
        upload.fill_volumes(&self.probe_rooms);
        let sky_layer = self.probe_stream.get_mut().as_ref().and_then(|s| s.sky_layer());
        upload.set_outdoors(self.probe_outdoor_volume, sky_layer, self.ground_placement);
        let building_layer = self.probe_stream.get_mut().as_ref().and_then(|s| s.building_layer());
        upload.set_buildings(&self.probe_buildings, building_layer);
        let stream = self.probe_stream.get_mut().as_mut().expect("just set");
        stream.resolve(&self.wgpu_queue, &mut upload);
        let view = stream.view();
        self.uniform_buf.set_probe_depth(stream.depth_view());
        self.uniform_buf.rebind_probes(
            &self.wgpu_device,
            &self.lights_uniform,
            self.shadow_map.sun_depth_view(),
            self.shadow_map.sun_dynamic_depth_view(),
            self.shadow_map.spot_depth_view(),
            self.shadow_map.sampler(),
            &view,
            &self.probe_sampler,
            upload,
        );

        self.probe_view = Some(view);
        log::info!(
            "reflection probes: {} described, {} doorway(s), pool {} at {resolution}px",
            descs.len(),
            self.probe_portals.len(),
            self.probe_stream.get_mut().as_ref().map(|s| s.pool_layers()).unwrap_or(0),
        );
    }

    pub fn set_mesh_lightmap(
        &mut self,
        key: &str,
        light: crate::renderer::mesh::LightmapLight,
        width: u32,
        height: u32,
        stationary: &[&[u8]],
        stationary_size: (u32, u32),
    ) {
        let tex = crate::renderer::mesh::create_lightmap_texture_full(
            &self.wgpu_device,
            &self.wgpu_queue,
            &self.mesh_pipeline.lightmap_layout,
            light,
            width,
            height,
            None,
            None,
            (!stationary.is_empty()).then_some((stationary, stationary_size.0, stationary_size.1)),
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

    fn sp_solid(&self, stereo: bool) -> &wgpu::RenderPipeline {
        match (stereo, &self.stereo_pipelines) {
            (true, Some(p)) => &p.solid.pipeline,
            _ => &self.solid_pipeline.pipeline,
        }
    }
    fn sp_wire(&self, stereo: bool) -> &wgpu::RenderPipeline {
        match (stereo, &self.stereo_pipelines) {
            (true, Some(p)) => &p.wire.pipeline,
            _ => &self.wire_pipeline.pipeline,
        }
    }
    fn sp_mesh(&self, stereo: bool) -> &wgpu::RenderPipeline {
        match (stereo, &self.stereo_pipelines) {
            (true, Some(p)) => &p.mesh.pipeline,
            _ if self.spotless_frame.load(std::sync::atomic::Ordering::Relaxed) => &self.spotless_mesh.pipeline,
            _ => &self.mesh_pipeline.pipeline,
        }
    }
    fn sp_mesh_thin(&self, stereo: bool) -> &wgpu::RenderPipeline {
        match (stereo, &self.stereo_pipelines, &self.thin_sampled_pipeline) {
            (true, Some(p), _) => &p.mesh.thin_pipeline,
            (false, _, Some(p)) => &p.thin_pipeline,
            _ if self.spotless_frame.load(std::sync::atomic::Ordering::Relaxed) => &self.spotless_mesh.thin_pipeline,
            _ => &self.mesh_pipeline.thin_pipeline,
        }
    }
    fn sp_skinned(&self, stereo: bool) -> &wgpu::RenderPipeline {
        match (stereo, &self.stereo_pipelines) {
            (true, Some(p)) => &p.skinned_mesh.pipeline,
            _ => &self.skinned_mesh_pipeline.pipeline,
        }
    }
    fn sp_terrain(&self, stereo: bool) -> &wgpu::RenderPipeline {
        match (stereo, &self.stereo_pipelines) {
            (true, Some(p)) => &p.terrain.pipeline,
            _ => &self.terrain_pipeline.pipeline,
        }
    }

    fn terrain_weather_reader<'a>(
        &self,
        w: &'a crate::renderer::weather::WeatherScene,
    ) -> &'a crate::renderer::terrain_pipeline::TerrainPipeline {
        if self.levers.terrain_reader.as_deref() == Some("full") {
            return &w.twins.readers[0];
        }
        let spotless = self.spotless_frame.load(std::sync::atomic::Ordering::Relaxed);
        &w.twins.readers[match (spotless, self.terrain_sun_baked) {
            (false, false) => 0,
            (true, false) => 1,
            (false, true) => 2,
            (true, true) => 3,
        }]
    }

    fn terrain_twin(&self) -> usize {
        match (self.spotless_frame.load(std::sync::atomic::Ordering::Relaxed), self.terrain_sun_baked) {
            (false, false) => 0,
            (true, false) => 1,
            (false, true) => 2,
            (true, true) => 3,
        }
    }

    fn slope_twins_on(&self) -> bool {
        self.levers.slope_twins
            && self.terrain_inlined.is_none()
            && self.reader_edits.is_none()
            && self.scene_cut_pipeline.is_none()
            && self.levers.terrain_reader.as_deref() != Some("full")
    }

    fn terrain_reader(&self) -> &crate::renderer::terrain_pipeline::TerrainPipeline {
        if let Some([read, _, _]) = &self.terrain_inlined {
            return read;
        }

        let edited = self.reader_edits.as_ref().map(|(_, _, ground)| ground);
        if self.levers.terrain_reader.as_deref() == Some("full") || self.scene_cut_pipeline.is_some() {
            return edited.map_or(&self.terrain_probe_reader_pipeline, |e| &e[0]);
        }
        match (self.spotless_frame.load(std::sync::atomic::Ordering::Relaxed), self.terrain_sun_baked) {
            (false, false) => edited.map_or(&self.terrain_probe_reader_pipeline, |e| &e[0]),
            (true, false) => edited.map_or(&self.terrain_reader_twins[0], |e| &e[1]),
            (false, true) => edited.map_or(&self.terrain_reader_twins[1], |e| &e[2]),
            (true, true) => edited.map_or(&self.terrain_reader_twins[2], |e| &e[3]),
        }
    }
    fn sp_layered(&self, stereo: bool) -> &wgpu::RenderPipeline {
        match (stereo, &self.stereo_pipelines) {
            (true, Some(p)) => &p.layered_mesh.pipeline,
            _ => &self.layered_mesh_pipeline.pipeline,
        }
    }

    fn sp_water(&self, stereo: bool) -> &wgpu::RenderPipeline {
        let ringing = self.water_bodies.iter().any(|b| b.uniform.rings[0][3] > 0.0);
        let w = match (stereo, &self.stereo_pipelines) {
            (true, Some(p)) => &p.water,
            _ => &self.water_pipeline,
        };
        if let (false, Some((_, cut))) = (stereo, &self.water_cut_pipeline) {
            return cut;
        }
        if ringing { &w.pipeline } else { &w.ringless }
    }
    fn sp_sky(&self, stereo: bool) -> &wgpu::RenderPipeline {
        match (stereo, &self.stereo_pipelines) {
            (true, Some(p)) => &p.sky.pipeline,
            _ => &self.sky_pipeline.pipeline,
        }
    }
    fn sp_brush_seal(&self, stereo: bool) -> &wgpu::RenderPipeline {
        match (stereo, &self.stereo_pipelines) {
            (true, Some(p)) => &p.brush_seal.pipeline,
            _ => &self.brush_seal_pipeline.pipeline,
        }
    }
    fn sp_particle(&self, stereo: bool) -> &wgpu::RenderPipeline {
        match (stereo, &self.stereo_pipelines) {
            (true, Some(p)) => &p.particle.pipeline,
            _ => &self.particle_pipeline.pipeline,
        }
    }

    fn stereo_scene(&self) -> bool {
        self.multiview_scene && self.stereo_pipelines.is_some()
    }

    fn probe_pass_runs(&self, fx: &crate::renderer::levers::Levers, stereo: bool, has_brushes: bool) -> bool {
        fx.half_res_reflections
            && fx.probes
            && (!stereo || self.stereo_probe.is_some())
            && self.debug_view == crate::renderer::brush_pipeline::DebugView::Off
            && has_brushes
    }
    fn sp_glare(&self, stereo: bool) -> &crate::renderer::glare::GlarePipeline {
        match (stereo, &self.stereo_pipelines) {
            (true, Some(p)) => &p.glare,
            _ => &self.glare_pipeline,
        }
    }
    fn sp_effects(&self, stereo: bool) -> &crate::renderer::effects::EffectsPipeline {
        match (stereo, &self.stereo_pipelines) {
            (true, Some(p)) => &p.effects,
            _ => &self.effects_pipeline,
        }
    }

    fn sp_brush(&self, stereo: bool) -> &wgpu::RenderPipeline {
        use crate::renderer::brush_pipeline::DebugView;
        match (stereo, &self.stereo_pipelines) {
            (true, Some(p)) => match self.debug_view {
                DebugView::Sources => &p.brush_sources.pipeline,
                _ => &p.brush_opaque.pipeline,
            },
            _ => match self.debug_view {
                DebugView::Sources => &self.brush_sources_pipeline.pipeline,
                _ => &self.brush_opaque_pipeline.pipeline,
            },
        }
    }

    pub fn set_baked_lights(&mut self, lights: Vec<crate::renderer::lights::Light>) {
        self.baked_lights = lights;
    }

    pub fn set_closed_rooms(&mut self, closed: &[u32]) {
        let mut rooms: Vec<crate::renderer::portal_cull::CullRoom> = Vec::new();
        for (probe, (_, _, min, max)) in self.probe_volumes.iter().enumerate() {
            let Some(&id) = self.probe_rooms.get(probe) else { continue };
            if !rooms.iter().any(|r| r.id == id) {
                rooms.push(crate::renderer::portal_cull::CullRoom { id, min: *min, max: *max, closed: closed.contains(&id) });
            }
        }
        log::info!(
            "doorway culling: {} of {} rooms closed ({:?})",
            rooms.iter().filter(|r| r.closed).count(),
            rooms.len(),
            rooms.iter().filter(|r| r.closed).map(|r| r.id).collect::<Vec<_>>(),
        );
        self.cull_rooms = rooms;
    }

    pub fn set_doors(&mut self, doors: Vec<crate::renderer::doors::DoorView>) {
        self.shut_portals = crate::renderer::doors::shut_portals(&self.probe_portals, &doors);
        self.doors = doors;
    }

    pub fn set_reflection_proxies(
        &mut self,
        mut proxies: Vec<crate::renderer::uniforms::ProbeProxy>,
        fields: Vec<crate::renderer::proxy_field::ProxyField>,
        cards: Vec<crate::renderer::proxy_cards::ProxyCards>,
    ) {

        let atlas = crate::renderer::proxy_cards::atlas_with_characters(&self.wgpu_device, &self.wgpu_queue, &cards,
            crate::renderer::proxy_cards::CHARACTER_CARD_SETS,
        );
        self.uniform_buf.set_proxy_card_atlas(atlas.view.clone());
                for p in &mut proxies {
                    p.cards = p.cards.and_then(|i| atlas.rows.get(i as usize).copied().flatten());
                }
                log::info!(
            "reflection cards: {} model(s) on cards; {} character set(s), {} texels a card",
            atlas.rows.iter().flatten().count(),
            atlas.character_rows.len(),
            atlas.resolution,
        );
        if self
            .character_cards
            .as_ref()
            .is_none_or(|c| c.resolution() != atlas.resolution)
        {
                self.character_cards = Some(crate::renderer::character_cards::CharacterCards::new(&self.wgpu_device,
                atlas.resolution,
                &self.skinned_mesh_pipeline,
            ));
        }

        if self.pool_cards.as_ref().is_none_or(|p| p.resolution() != atlas.resolution) {
            self.pool_cards = Some(crate::renderer::pool_cards::PoolCards::new(
                &self.wgpu_device,
                &self.uniform_buf.layout,
                atlas.resolution,
            ));
        }
        self.lights_uniform.set_pool_row(
            self.pool_cards.as_ref().map(|p| p.first_row(atlas.pool_row)),
        );
        self.card_atlas = Some(atlas);
        self.probe_proxies = proxies;

        match crate::renderer::proxy_field::atlas(&self.wgpu_device, &self.wgpu_queue, &fields) {
            Some((view, slots)) => {
                self.uniform_buf.set_proxy_field_atlas(view);
                self.proxy_field_slots = slots;
            }
            None => {
                self.uniform_buf.set_proxy_field_atlas(crate::renderer::proxy_field::none(&self.wgpu_device));
                self.proxy_field_slots.clear();
            }
        }
        self.rebind_scene_group();
    }

    pub fn set_levers(&mut self, levers: crate::renderer::levers::Levers) {

        if levers != self.levers {
            self.frame_stats.restart();
            self.perf_metric_window.clear();
            self.perf_windows = 0;
            self.perf_warmup = true;
        }
        if let Some(on) = levers.ssr {
            self.set_screen_space_reflections(on);
        }
        if let Some(on) = levers.multiview {
            let got = self.set_multiview_scene(on);
            if got != on {
                log::warn!("LEVERS: multiview={on} asked, but this device cannot; it stays {got}");
            }
        }
        self.set_auto_exposure(levers.eye_adaptation);
        if levers.pass_cut != self.levers.pass_cut {

            let terrain = levers.pass_cut.as_ref().filter(|cut| cut.starts_with("terrain_cut_"));
            self.terrain_cut_pipeline = terrain.and_then(|cut| {
                let p = crate::renderer::terrain_pipeline::TerrainPipeline::new_probe_pass_with_cut(
                    &self.wgpu_device, &self.uniform_buf.layout, &self.probe_fixups, cut,
                );
                if p.is_none() {
                    log::warn!("LEVERS: pass_cut {cut}: no such cut, or it no longer matches the shader");
                }
                p.map(|p| (cut.clone(), p))
            });
            self.pass_cut_pipeline = levers.pass_cut.as_ref().filter(|_| terrain.is_none()).and_then(|cut| {
                let p = crate::renderer::brush_pipeline::BrushPipeline::new_probe_pass_deferred_with_cut(
                    &self.wgpu_device, &self.uniform_buf.layout, &self.probe_fixups, cut,
                );
                if p.is_none() {
                    log::warn!("LEVERS: pass_cut {cut}: no such cut, or it no longer matches the shader");
                }
                p.map(|p| (cut.clone(), p))
            });
        }
        if levers.thin_shading_sampled != self.levers.thin_shading_sampled {
            let (format, samples, _) = &self.cut_inputs;
            self.thin_sampled_pipeline = levers.thin_shading_sampled.then(|| {
                crate::renderer::mesh_pipeline::MeshPipeline::new_multisampled_thin_sampled(
                    &self.wgpu_device, *format, &self.uniform_buf.layout, *samples,
                )
            });
        }
        if levers.fixup_cut != self.levers.fixup_cut && !self.probe_fixups.set_cut(&self.wgpu_device, levers.fixup_cut.as_deref()) {
            log::warn!("LEVERS: fixup_cut {:?}: no such cut, or it no longer matches the shader", levers.fixup_cut);
        }
        if levers.terrain_reader != self.levers.terrain_reader {
            let (format, samples, layout) = &self.cut_inputs;
            self.terrain_inlined = match levers.terrain_reader.as_deref() {
                Some("inlined") => Some(crate::renderer::terrain_pipeline::TerrainPipeline::new_inlined(
                    &self.wgpu_device, *format, &self.uniform_buf.layout, *samples, layout, &self.probe_fixups,
                )),
                None | Some("full") => None,
                Some(other) => {
                    log::warn!("LEVERS: terrain_reader {other}: neither `full` nor `inlined`; the shipped readers draw");
                    None
                }
            };
        }
        if levers.reader_edit != self.levers.reader_edit {
            let (format, samples, layout) = &self.cut_inputs;
            self.reader_edits = levers.reader_edit.as_ref().and_then(|edit| {
                let brushes = crate::renderer::brush_pipeline::BrushPipeline::new_edited_probe_readers(
                    &self.wgpu_device, *format, &self.uniform_buf.layout, *samples, layout, edit,
                );
                let ground = crate::renderer::terrain_pipeline::TerrainPipeline::new_edited_probe_readers(
                    &self.wgpu_device, *format, &self.uniform_buf.layout, *samples, layout, edit,
                );
                match (brushes, ground) {
                    (Some(brushes), Some(ground)) => Some((edit.clone(), brushes, ground)),
                    _ => {
                        log::warn!("LEVERS: reader_edit {edit}: no such edit, or it no longer matches the readers; the shipped readers draw");
                        None
                    }
                }
            });
        }
        if levers.water_cut != self.levers.water_cut {
            self.water_cut_pipeline = levers.water_cut.as_ref().and_then(|cut| {
                let p = self.water_pipeline.with_cut(&self.wgpu_device, cut);
                if p.is_none() {
                    log::warn!("LEVERS: water_cut {cut}: no such cut, or it no longer matches the shader");
                }
                p.map(|p| (cut.clone(), p))
            });
        }
        if levers.scene_cut != self.levers.scene_cut {
            let (format, samples, layout) = &self.cut_inputs;
            self.scene_cut_pipeline = levers.scene_cut.as_ref().and_then(|cut| {
                let p = crate::renderer::brush_pipeline::BrushPipeline::new_multisampled_probe_reader_with_cut(
                    &self.wgpu_device, *format, &self.uniform_buf.layout, *samples, layout, cut,
                );
                if p.is_none() {
                    log::warn!("LEVERS: scene_cut {cut}: no such cut, or it no longer matches the shader");
                }
                p.map(|p| (cut.clone(), p))
            });
        }
        let requests = crate::renderer::performance_level::requests(
            self.levers.performance_levels(),
            levers.performance_levels(),
        );
        match &self.perf_settings {
            Some(settings) => requests.into_iter().for_each(|(domain, level)| settings.request(domain, level)),
            None if !requests.is_empty() => {
                log::warn!("LEVERS: a performance level asked, but the runtime has no XR_EXT_performance_settings")
            }
            None => {}
        }
        self.levers = levers;
    }

    pub fn levers(&self) -> crate::renderer::levers::Levers {
        self.levers.clone()
    }

    pub fn ab_phase(&self) -> crate::renderer::perf_ab::Phase {
        if crate::renderer::perf_ab::ENABLED || self.levers.ab_cycle {
            crate::renderer::perf_ab::Phase::cycle_phase(self.perf_windows)
        } else {
            crate::renderer::perf_ab::Phase::Baseline
        }
    }

    pub fn frame_levers(&self) -> crate::renderer::levers::Levers {
        self.levers.clone().with_phase(self.ab_phase())
    }

    pub fn space_warp_running(&self) -> bool {
        self.space_warp.is_some() && self.levers.space_warp
    }

    pub fn ask_recommended_resolution(&mut self, layer: &xr::sys::CompositionLayerProjection, display_time: xr::Time) {
        let Some(asker) = self.recommended_resolution.as_ref().filter(|_| self.levers.dynamic_resolution) else {
            return;
        };

        let answer = unsafe {
            asker.recommend(layer as *const xr::sys::CompositionLayerProjection as *const xr::sys::CompositionLayerBaseHeader, display_time)
        };
        self.recommendation_window.add(answer.map_err(|e| e.into_raw()));
    }

    pub fn set_pinned_head(&mut self, head: Option<(glam::Vec3, glam::Quat)>) {
        self.pinned_head = head;
    }

    pub fn set_perf_log(&mut self, path: std::path::PathBuf) {
        info!("perf log: every PERF window is also written to {}", path.display());
        self.perf_log = Some(crate::perf_record::PerfLog::new(path));
    }

    pub fn set_multiview_scene(&mut self, on: bool) -> bool {
        self.multiview_scene = on && self.stereo_pipelines.is_some();
        self.multiview_scene
    }

    pub fn multiview_scene(&self) -> bool {
        self.multiview_scene
    }

    pub fn multiview_available(&self) -> bool {
        self.stereo_pipelines.is_some()
    }

    pub fn set_buffered_reflections(&mut self, on: bool) {
        self.buffered_reflections = on;
    }

    pub fn buffered_reflections(&self) -> bool {
        self.buffered_reflections
    }

    pub fn set_screen_space_reflections(&mut self, on: bool) {
        self.screen_space_reflections = on;
    }

    pub fn screen_space_reflections(&self) -> bool {
        self.screen_space_reflections
    }

    pub fn set_debug_view(&mut self, view: crate::renderer::brush_pipeline::DebugView) {
        self.debug_view = view;
    }

    pub fn debug_view(&self) -> crate::renderer::brush_pipeline::DebugView {
        self.debug_view
    }

    pub fn set_post(&mut self, post: crate::renderer::uniforms::PostUpload) {
        self.post = post;
    }

    pub fn set_player_frame(&mut self, offset: glam::Vec3, yaw: f32) {

        self.player.offset = offset;
        self.player.yaw = yaw;
    }

    pub fn set_glare_sources(&mut self, sources: Vec<crate::renderer::glare::GlareSource>) {
        self.glare_sources = sources;
    }

    pub fn set_weather(&mut self, texels: &[(u32, u32)], mut chunks: Vec<(u32, u32)>) {
        use crate::renderer::weather;
        if texels.is_empty() {
            self.weather = None;
            return;
        }
        let device = &self.wgpu_device;
        let layout = weather::bind_group_layout(device);
        let probe_layout = crate::renderer::brush_pipeline::probe_pass::bind_group_layout(device);
        let twins = crate::renderer::terrain_pipeline::TerrainPipeline::new_weather_twins(
            device,
            wgpu::TextureFormat::Rgba8UnormSrgb,
            &self.uniform_buf.layout,
            self.msaa.samples(),
            &probe_layout,
            &self.probe_fixups,
            &layout,
        );
        let particles = weather::WeatherPipeline::new(
            device,
            wgpu::TextureFormat::Rgba8UnormSrgb,
            &self.uniform_buf.layout,
            &self.terrain_pipeline.material_layout,
            &layout,
            self.msaa.samples(),
            crate::renderer::multiview::ViewMode::Mono,
        );
        let maps = weather::WeatherMaps::new(device, &layout, texels);
        chunks.sort_unstable();

        let gentle: Vec<[crate::renderer::terrain_pipeline::TerrainPipeline; 4]> = weather::WeatherKinds::ALL
            .iter()
            .map_while(|kinds| {
                crate::renderer::ground_twins::gentle_readers(
                    device,
                    wgpu::TextureFormat::Rgba8UnormSrgb,
                    &self.uniform_buf.layout,
                    self.msaa.samples(),
                    &probe_layout,
                    Some((&layout, *kinds)),
                )
            })
            .collect();
        if gentle.len() != weather::WeatherKinds::ALL.len() {
            log::warn!("WEATHER: the ground's gentle weather twins did not build; its full ones draw everywhere");
        }
        let gentle = if gentle.len() == weather::WeatherKinds::ALL.len() { gentle } else { Vec::new() };
        let dedup_passes = crate::renderer::ground_twins::dedup_passes(device, &self.uniform_buf.layout, &self.probe_fixups, Some(&layout));
        log::info!("WEATHER: {} area(s), {} terrain chunk(s) take the weather twins", texels.len(), chunks.len());
        self.weather = Some(weather::WeatherScene {
            layout,
            maps,
            twins,
            particles,
            chunks,
            gentle,
            dedup_passes,
            holds: vec![(true, true); texels.len()],
            areas: Vec::new(),
            texels: texels.to_vec(),
            counts: (0, 0, 0),
            torch: None,
            seconds: 0.0,
        });
    }

    pub fn update_weather(&mut self, seconds: f64, areas: &[crate::renderer::weather::AreaParams], maps: &[Vec<[f32; 4]>]) {
        let Some(w) = self.weather.as_mut() else { return };
        w.seconds = seconds;
        w.areas = areas.to_vec();
        for (k, a) in areas.iter().enumerate() {
            let size = w.texels.get(k).copied().unwrap_or((1, 1));
            w.maps.set_area(k, a, size);
            if let Some(m) = maps.get(k) {
                w.maps.upload_map(&self.wgpu_queue, k as u32, size.0, size.1, m);
                if let Some(h) = w.holds.get_mut(k) {
                    *h = crate::renderer::weather::map_holds(m);
                }
            }
        }
    }

    pub fn set_weather_torch(&mut self, torch: Option<crate::renderer::Light>) {
        if let Some(w) = self.weather.as_mut() {
            w.torch = torch;
        }
    }

    pub fn set_effects(&mut self, emitters: Vec<crate::renderer::effects::EffectEmitter>) {
        if !emitters.is_empty() && self.effects_gpu.is_none() {
            let started = std::time::Instant::now();
            self.effects_gpu = Some(crate::renderer::effects::EffectsGpu::new(
                &self.wgpu_device,
                &self.wgpu_queue,
                &self.effects_layout,
            ));
            log::info!("EFFECTS textures made in {:.0} ms", started.elapsed().as_secs_f32() * 1000.0);
        }
        log::info!("EFFECTS {} emitters", emitters.len());
        self.effect_emitters = emitters;
        self.effect_seen_until.clear();
    }

    pub fn set_lit_surfaces(&self, surfaces: &[crate::renderer::lights::LitSurface]) {
        self.lights_uniform.set_lit_surfaces(surfaces);
    }

    pub fn set_capsules(&mut self, groups: &[crate::renderer::uniforms::CapsuleGroup]) {
        self.player.capsules = crate::renderer::uniforms::CapsuleUpload::from_groups(groups);

        self.glare_capsules.clear();
        self.glare_capsules.extend(groups.iter().flat_map(|g| {
            g.capsules.iter().enumerate().filter(|(k, _)| g.surfaces.get(*k).map_or(true, |s| *s == 0.0)).map(|(_, c)| *c)
        }));
    }

    pub fn set_sky(
        &mut self,
        pano: Option<&crate::renderer::sky::Panorama>,
        rotation_deg: f32,
        intensity: f32,
    ) {
        self.sky = match pano {
            Some(p) => crate::renderer::sky::Sky::new(
                &self.wgpu_device,
                &self.wgpu_queue,
                &self.sky_pipeline.layout,
                p,
                rotation_deg,
                intensity,
            ),
            None => crate::renderer::sky::Sky::none(
                &self.wgpu_device,
                &self.wgpu_queue,
                &self.sky_pipeline.layout,
                crate::renderer::sky::AMBIENT,
            ),
        };

        self.tod_sky_changed(pano.map(|p| (p.clone(), rotation_deg, intensity)));

        *self.eye.borrow_mut() = crate::renderer::exposure::EyeAdaptation::sky_only(self.sky.irradiance);

        self.ground_dirty = true;
    }

    pub fn set_terrain_heights(&mut self, heights: Option<crate::renderer::ground_map::HeightGrid>) {
        self.ground_heights = heights;
        self.ground_dirty = true;
    }

    pub fn ensure_ground_map(&mut self) {
        if !self.ground_dirty {
            return;
        }
        self.ground_dirty = false;
        let Some(heights) = self.ground_heights.as_ref() else {
            if self.ground_placement.take().is_some() {
                self.uniform_buf.set_ground_map(crate::renderer::uniforms::default_ground_map(&self.wgpu_device));
                self.rebind_scene_group();
            }
            return;
        };
        let started = std::time::Instant::now();
        let map = crate::renderer::ground_map::build(
            &crate::renderer::ground_map::GroundInputs {
                heights,
                sky: &self.sky.irradiance,
                sun: self.sky.sun.as_ref(),
                sky_occlusion: self.terrain_sky_occlusion.as_ref(),
                layers: &self.terrain_layers,
                splat: self.terrain_splat.as_ref(),
                settings: &self.terrain_settings,
            },
            crate::renderer::ground_map::GROUND_MAP_SIZE,
        );
        let view = crate::renderer::ground_map::upload(&self.wgpu_device, &self.wgpu_queue, &map);
        let extent = map.max - map.min;
        self.ground_placement = Some(([map.min.x, map.min.y, 1.0 / extent.x, 1.0 / extent.y], map.top));
        self.uniform_buf.set_ground_map(view);
        self.rebind_scene_group();
        log::info!(
            "ground map: {}x{} over {:.0} x {:.0} m, top {:.2} m, built in {} ms",
            map.width,
            map.height,
            extent.x,
            extent.y,
            map.top,
            started.elapsed().as_millis(),
        );
    }

    fn rebind_scene_group(&mut self) {
        let fallback;
        let probe_view = match self.probe_view.as_ref() {
            Some(v) => v,
            None => {
                fallback = crate::renderer::uniforms::default_probe_cube(&self.wgpu_device).0;
                &fallback
            }
        };
        let mut probes = self.uniform_buf.probes();
        let sky_layer = self.probe_stream.get_mut().as_ref().and_then(|s| s.sky_layer());
        probes.set_outdoors(self.probe_outdoor_volume, sky_layer, self.ground_placement);
        let building_layer = self.probe_stream.get_mut().as_ref().and_then(|s| s.building_layer());
        probes.set_buildings(&self.probe_buildings, building_layer);
        self.uniform_buf.rebind_probes(
            &self.wgpu_device,
            &self.lights_uniform,
            self.shadow_map.sun_depth_view(),
            self.shadow_map.sun_dynamic_depth_view(),
            self.shadow_map.spot_depth_view(),
            self.shadow_map.sampler(),
            probe_view,
            &self.probe_sampler,
            probes,
        );
    }

    pub fn set_auto_exposure(&mut self, on: bool) {
        self.auto_exposure = on;
    }

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

    pub fn sky_visibility_at(&self, world_x: f32, world_z: f32) -> f32 {
        let Some(img) = &self.terrain_sky_occlusion else {
            return 1.0;
        };
        let Some((min, max)) = self.terrain_footprint else {
            return 1.0;
        };
        let ex = max.x - min.x;
        let ez = max.z - min.z;
        if ex.abs() < 1e-6 || ez.abs() < 1e-6 || img.width == 0 || img.height == 0 {
            return 1.0;
        }

        let u = ((world_x - min.x) / ex).clamp(0.0, 1.0);
        let v = ((world_z - min.z) / ez).clamp(0.0, 1.0);
        let tx = ((u * img.width as f32) as u32).min(img.width - 1);
        let ty = ((v * img.height as f32) as u32).min(img.height - 1);
        let i = ((ty * img.width + tx) * 4) as usize;
        img.rgba.get(i).map(|r| *r as f32 / 255.0).unwrap_or(1.0)
    }

    pub fn set_terrain_stationary_masks(&mut self, masks: Vec<crate::renderer::terrain_pipeline::TerrainImage>) {
        self.terrain_stationary = masks;
        self.rebuild_terrain_material();
    }

    pub fn set_terrain_footprint(&mut self, min: glam::Vec3, max: glam::Vec3) {
        self.terrain_footprint = Some((min, max));
    }

    pub fn sun_toward(&self) -> Option<glam::Vec3> {
        self.sky.sun.as_ref().map(|s| glam::Vec3::from(s.direction).normalize_or_zero())
    }

    pub fn add_splash(&mut self, mut splash: crate::renderer::effects::Splash) {
        const KEPT: usize = 48;

        if let Some(b) = self.water_bodies.iter().find(|b| (b.uniform.extinction[3] - splash.position.y).abs() < 0.05) {
            let xz = glam::Vec2::new(splash.position.x, splash.position.z);
            splash.position.y += crate::renderer::water_pipeline::swash_lift(&b.uniform, splash.depth, xz);
        }
        if self.splashes.len() >= KEPT {
            self.splashes.remove(0);
        }
        self.splashes.push(splash);
    }

    pub fn set_water(
        &mut self,
        bodies: &[(
            Vec<crate::renderer::water_pipeline::WaterVertex>,
            Vec<u32>,
            crate::renderer::water_pipeline::WaterUniform,
            crate::renderer::water_waves::WaveParams,
        )],
    ) {
        use wgpu::util::DeviceExt;
        self.water_bodies = bodies
            .iter()
            .filter(|(v, i, _, _)| !v.is_empty() && !i.is_empty())
            .map(|(verts, indices, uniform, params)| {
                let vertex_buffer = self.wgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("water_vb"),
                    contents: bytemuck::cast_slice(verts),
                    usage: wgpu::BufferUsages::VERTEX,
                });
                let index_buffer = self.wgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("water_ib"),
                    contents: bytemuck::cast_slice(indices),
                    usage: wgpu::BufferUsages::INDEX,
                });
                let uniform_buffer = self.wgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("water_uniform"),
                    contents: bytemuck::bytes_of(uniform),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });
                let bounds = crate::renderer::water_pipeline::water_bounds(verts, params);
                let waves = crate::renderer::water_waves::WaveField::new(&self.wgpu_device, &self.wgpu_queue, *params);
                let bind_groups = self.water_pipeline.bind_groups(&self.wgpu_device, &uniform_buffer, &waves);
                let motion_groups = self.space_warp.as_ref().map(|sw| {
                    std::array::from_fn(|now| {
                        sw.pipelines.water_bind_group(
                            &self.wgpu_device,
                            &uniform_buffer,
                            &waves.displacement_views[now],
                            &waves.displacement_views[1 - now],
                            &self.water_pipeline.sampler,
                        )
                    })
                });
                let under_groups = self.underwater.pipes.water_groups(&self.wgpu_device, &uniform_buffer, &waves);
                WaterBody {
                    vertex_buffer,
                    index_buffer,
                    uniform_buffer,
                    bind_groups,
                    index_count: indices.len() as u32,
                    uniform: *uniform,
                    waves,
                    motion_groups,
                    bounds,
                    seen: std::cell::Cell::new(false),
                    under_groups,
                    still: crate::renderer::underwater::StillDepth::new(verts),
                }
            })
            .collect();
        log::info!(
            "water: {} body/bodies installed, {} blending",
            self.water_bodies.len(),
            if self.water_pipeline.dual_source { "dual-source" } else { "one-alpha" },
        );

        if !self.water_bodies.is_empty() && self.effects_gpu.is_none() {
            self.effects_gpu = Some(crate::renderer::effects::EffectsGpu::new(&self.wgpu_device, &self.wgpu_queue, &self.effects_layout));
        }
    }

    pub fn water_body_count(&self) -> usize {
        self.water_bodies.len()
    }

    pub fn set_water_time(&mut self, seconds: f32) {

        self.water_step = (seconds as f64 - self.water_seconds).clamp(0.0, 0.1) as f32;
        self.water_seconds = seconds as f64;
        let now = self.water_seconds;
        let lasts = crate::renderer::effects::SPLASH_SECONDS.max(crate::renderer::water_pipeline::RING_SECONDS) as f64;
        self.splashes.retain(|s| now - s.born < lasts && s.born <= now + 1.0);
        let rings = crate::renderer::effects::splash_rings(&self.splashes, now, self.player.offset);
        for body in &mut self.water_bodies {
            body.uniform.set_time(self.water_seconds, body.waves.params.loop_seconds);
            body.uniform.rings = rings;
            self.wgpu_queue.write_buffer(&body.uniform_buffer, 0, bytemuck::bytes_of(&body.uniform));
        }
    }

    pub fn set_terrain_sky_occlusion(
        &mut self,
        occ: Option<&crate::renderer::terrain_pipeline::TerrainImage>,
    ) {
        self.terrain_sky_occlusion = occ.map(|s| crate::renderer::terrain_pipeline::TerrainImage {
            width: s.width,
            height: s.height,
            rgba: s.rgba.clone(),
        });

        self.terrain_sun_baked = !self.tod.off_bake && occ.is_some_and(|s| s.sun_baked_everywhere());
        self.rebuild_terrain_material();
    }

    pub fn set_brush_materials(
        &mut self,
        colours: &[crate::renderer::terrain_pipeline::TerrainImage],
        normals: &[Option<crate::renderer::terrain_pipeline::TerrainImage>],
        roughs: &[Option<crate::renderer::terrain_pipeline::TerrainImage>],
        aos: &[Option<crate::renderer::terrain_pipeline::TerrainImage>],
    ) {
        let clone_opt = |v: &[Option<crate::renderer::terrain_pipeline::TerrainImage>]| {
            v.iter()
                .map(|o| {
                    o.as_ref().map(|i| crate::renderer::terrain_pipeline::TerrainImage {
                        width: i.width,
                        height: i.height,
                        rgba: i.rgba.clone(),
                    })
                })
                .collect::<Vec<_>>()
        };
        let roughs = clone_opt(roughs);
        let aos = clone_opt(aos);
        let colours: Vec<Option<crate::renderer::terrain_pipeline::TerrainImage>> = colours
            .iter()
            .map(|c| {
                Some(crate::renderer::terrain_pipeline::TerrainImage {
                    width: c.width,
                    height: c.height,
                    rgba: c.rgba.clone(),
                })
            })
            .collect();
        let normals: Vec<Option<crate::renderer::terrain_pipeline::TerrainImage>> = normals
            .iter()
            .map(|n| {
                n.as_ref().map(|n| crate::renderer::terrain_pipeline::TerrainImage {
                    width: n.width,
                    height: n.height,
                    rgba: n.rgba.clone(),
                })
            })
            .collect();
        self.brush_materials = crate::renderer::brush_pipeline::BrushMaterials::new(
            &self.wgpu_device,
            &self.wgpu_queue,
            &self.brush_pipeline.material_layout,
            &colours,
            &normals,
            &roughs,
            &aos,
        );
    }

    pub fn set_terrain_layers(
        &mut self,
        layers: Vec<Option<crate::renderer::terrain_pipeline::TerrainImage>>,
        normals: Vec<Option<crate::renderer::terrain_pipeline::TerrainImage>>,

        rough: Vec<Option<crate::renderer::terrain_pipeline::TerrainImage>>,
        ao: Vec<Option<crate::renderer::terrain_pipeline::TerrainImage>>,
    ) {
        self.terrain_layers = layers;
        self.terrain_normals = normals;
        self.terrain_rough = rough;
        self.terrain_ao = ao;
        self.rebuild_terrain_material();
    }

    pub fn set_terrain_settings(
        &mut self,
        settings: crate::renderer::terrain_pipeline::TerrainMaterialUniform,
    ) {
        self.terrain_settings = settings;
        self.rebuild_terrain_material();
    }

    fn rebuild_terrain_material(&mut self) {

        self.ground_dirty = true;

        let ground = match (&self.terrain_sky_occlusion, self.terrain_stationary.is_empty()) {
            (Some(map), false) => Some(map.with_stationary_masks(&self.terrain_stationary).unwrap_or_else(|| {
                log::warn!(
                    "terrain: {} stationary mask layer(s) do not match the {}x{} ground map; lamps light the ground unshadowed",
                    self.terrain_stationary.len(),
                    map.width,
                    map.height
                );
                crate::renderer::terrain_pipeline::TerrainImage { width: map.width, height: map.height, rgba: map.rgba.clone() }
            })),
            (map, _) => map.as_ref().map(|m| crate::renderer::terrain_pipeline::TerrainImage {
                width: m.width,
                height: m.height,
                rgba: m.rgba.clone(),
            }),
        };

        let ground = match (ground, self.tod.off_bake) {
            (Some(mut g), true) => {
                let first = self.terrain_sky_occlusion.as_ref().map_or(0, |m| m.rgba.len()).min(g.rgba.len());
                for texel in g.rgba[..first].chunks_exact_mut(4) {
                    texel[3] = 255;
                }
                Some(g)
            }
            (g, _) => g,
        };
        self.terrain_material =
            crate::renderer::terrain_pipeline::TerrainMaterial::from_layers_with(
                &self.wgpu_device,
                &self.wgpu_queue,
                &self.terrain_pipeline.material_layout,
                &self.terrain_layers,
                &self.terrain_normals,
                &self.terrain_rough,
                &self.terrain_ao,
                self.terrain_splat.as_ref(),
                ground.as_ref(),
                self.terrain_settings,
            );
    }

    pub fn cleanup(&self) {}
}

impl XrRenderer {

    fn apply_foveation(&mut self, level: crate::renderer::foveation::FoveationLevel) {
        use crate::renderer::foveation::{centre_of_view, density_pattern, FoveationLevel};
        let Some(state) = self.foveation.as_mut() else { return };
        if state.applied == Some(level) {
            return;
        }
        let Some(fov) = self.last_fov else { return };
        let maps = if level == FoveationLevel::Off {
            None
        } else {
            if !state.maps.contains_key(&level) {
                let centres: Vec<(f32, f32)> =
                    fov.iter().map(|f| centre_of_view(f.angle_left, f.angle_right, f.angle_up, f.angle_down)).collect();
                let patterns: Vec<_> = [state.eye_size, state.probe_size]
                    .iter()
                    .flat_map(|&(w, h)| centres.iter().map(move |&c| density_pattern(w, h, c, level)))
                    .collect();

                match unsafe { vulkan_interop::add_foveation_maps(&self.wgpu_device, &patterns) } {
                    Some(ids) if ids.len() == 4 => {
                        state.maps.insert(level, [ids[0], ids[1], ids[2], ids[3]]);
                    }
                    _ => {

                        state.applied = Some(level);
                        return;
                    }
                }
                let centres: Vec<String> = fov
                    .iter()
                    .map(|f| {
                        let (x, y) = centre_of_view(f.angle_left, f.angle_right, f.angle_up, f.angle_down);
                        format!("({x:.3}, {y:.3})")
                    })
                    .collect();
                info!("foveation: {} maps {}x{} texels, centres {}", level.label(), patterns[0].0, patterns[0].1, centres.join(" "));
            }
            state.maps.get(&level).copied()
        };
        for targets in &self.eye_targets {
            for (eye, target) in targets.iter().enumerate() {
                unsafe { vulkan_interop::set_foveation_target(&self.wgpu_device, &target.view, maps.map(|m| m[eye])) };
            }
        }
        for (eye, target) in self.probe_pass_targets.iter().enumerate() {
            unsafe { vulkan_interop::set_foveation_target(&self.wgpu_device, &target.color_view, maps.map(|m| m[2 + eye])) };
        }
        state.applied = Some(level);
        info!("foveation: {}", level.label());
    }
}
