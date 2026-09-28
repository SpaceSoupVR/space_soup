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
mod vulkan_interop;

struct EyeTarget {
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
}

/// How much of the frame budget goes on shadows.
///
/// An explicit knob rather than something inferred from the scene, because this
/// is the single biggest thing a level can spend on the headset and whoever is
/// tuning a map needs to be able to turn it down without editing their lights.
///
/// Each level costs a FULL EXTRA PASS over every caster in the scene. The sun's
/// map is built in light space and so is rendered once per frame for both eyes;
/// adding the flashlight doubles that, and it is a perspective map that has to
/// be rebuilt whenever the hand moves, which is every frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShadowQuality {
    /// No depth passes at all. Everything is lit unshadowed, as it was before
    /// shadows existed -- which is a real option for an indoor level lit
    /// entirely by point lights that never cast anyway.
    Off,
    /// The sun only. One pass, and it buys every outdoor shadow at once.
    ///
    /// NOT the default, and the reason is worth stating: a scene with no
    /// DIRECTIONAL light gets no shadows at all under this setting, silently.
    /// That is every indoor level -- a hall lit by three spots rendered with
    /// `sun_enabled == false` and `spot_count == 0`, so the shadow pass was
    /// skipped entirely and nothing in the level cast anything onto anything.
    /// It reads as "shadows are broken", not as "shadows are switched off".
    SunOnly,
    /// The sun and any shadow-casting spots, up to `MAX_SPOT_SHADOWS`.
    ///
    /// The default. Each source costs its own depth pass, but only for sources
    /// that actually exist: a scene with no sun pays nothing for the sun, and
    /// a scene with no spots pays nothing for spots. Defaulting here means an
    /// author gets shadows from whatever they lit the level with, rather than
    /// from whatever the engine guessed they would use.
    SunAndSpot,
}

/// How many samples the scene pass takes per pixel.
///
/// WHY THIS IS WORTH IT ON A HEADSET SPECIFICALLY
///
/// A tile GPU keeps the multisampled buffer in on-chip tile memory and resolves
/// it there, so the write out to main memory is the same size it always was.
/// The cost is tile memory and a little raster work, not bandwidth -- which is
/// the opposite of a desktop GPU, where 4x MSAA means a framebuffer four times
/// the size and four times the write traffic.
///
/// And aliasing is worth more to fix here than on a monitor: a shimmering edge
/// two metres from your face, moving with your head, is far more noticeable in
/// stereo than the same edge on a screen you are sitting still in front of.
///
/// THE ONE PLACE IT IS NOT CHEAP
///
/// Colour resolves and is discarded. DEPTH cannot resolve -- wgpu has no depth
/// resolve, and neither does the hardware in any portable way -- so if anything
/// downstream reads scene depth, the multisampled depth buffer has to be
/// STORED, at four times the size. `render_frame` therefore discards it unless
/// the frame actually has a reflective surface or a mirror in it, which is the
/// only thing that reads it. No shipped scene does.
/// NO 2x OPTION, deliberately.
///
/// WebGPU guarantees only 1 and 4 samples for a colour format; 2x needs the
/// optional `TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES`, and without it pipeline
/// creation fails outright. It was offered here until a headless test built
/// every pipeline at 2x and the driver said so in as many words. An option that
/// works on the machine you develop on and refuses to start on some headsets is
/// worse than not having it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MsaaLevel {
    Off,
    X4,
}

/// The MSAA level a headset session gets.
///
/// # This is a DIAGNOSTIC SWITCH as much as a quality setting
///
/// Nine separate fixes have been aimed at the thin light/dark lines along the
/// ceiling-wall junctions, and every one of them addressed a different VALUE
/// that MSAA edge shading is known to corrupt: the probe box test, the probe
/// parallax position, the lightmap `uv2`, the Fresnel normal. The mechanism is
/// real and documented on `BrushVertex::face_centre` -- with MSAA a pixel on a
/// polygon edge is shaded at its CENTRE, which can lie outside the polygon, so
/// every interpolated attribute arrives EXTRAPOLATED past the edge.
///
/// Patching the values one at a time cannot end: there is one such value per
/// thing the shader interpolates. Setting this to `Off` tests the whole family
/// in one build. If the seam vanishes, the cause is MSAA edge shading and the
/// answer is centroid interpolation (or per-face constants) rather than a tenth
/// clamp. If the seam SURVIVES, every extrapolation hypothesis is dead at once
/// and the cause is something MSAA never touched.
///
/// A build with this `Off` aliases everywhere and is not meant to be shipped --
/// it is meant to be looked at once, at a seam.
pub const DEFAULT_MSAA: MsaaLevel = MsaaLevel::X4;

impl MsaaLevel {
    pub fn samples(self) -> u32 {
        match self {
            MsaaLevel::Off => 1,
            MsaaLevel::X4 => 4,
        }
    }
}

/// One installed body of water: its buffers and its current optics.
struct WaterBody {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    index_count: u32,
    /// Kept so the animation time can be advanced without rebuilding the rest.
    uniform: crate::renderer::water_pipeline::WaterUniform,
}

/// Every scene pipeline again, built to draw BOTH EYES in one pass.
///
/// A separate set rather than a flag on each, because a pipeline's view mask is
/// fixed at creation: a pipeline cannot be told at draw time to cover two
/// layers. Held together so the whole stereo path is present or absent as one
/// thing -- a scene pass with nine stereo pipelines and one mono would draw
/// that one geometry type with the left eye's camera in both views, and nothing
/// would report it.
/// THE HALF-RESOLUTION PROBE PASS FOR BOTH EYES AT ONCE, for the multiview
/// scene pass: the pass, the brush that reads it, and the two-layer target.
///
/// Apart from `StereoScenePipelines` on purpose. If these fail to build, only
/// half-resolution reflections in stereo are lost -- the stereo scene pass
/// then traces its reflections per pixel, as it always has -- where inside
/// that set a failure would switch multiview off altogether.
struct StereoProbePass {
    pass: crate::renderer::brush_pipeline::BrushPipeline,
    reader: crate::renderer::brush_pipeline::BrushPipeline,
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
}


pub struct XrRenderer {
    pub swapchain: xr::Swapchain<xr::Vulkan>,
    pub width: u32,
    pub height: u32,
    wgpu_device: wgpu::Device,
    wgpu_queue: wgpu::Queue,
    solid_pipeline: SolidPipeline,
    water_pipeline: crate::renderer::water_pipeline::WaterPipeline,
    /// One body of water: its geometry in the player's frame, and its optics.
    ///
    /// Rebuilt on scene load rather than per frame -- the surface is static
    /// world geometry, so only the player-frame transform changes, and that is
    /// done by the caller exactly as it is for brushes and terrain.
    water_bodies: Vec<WaterBody>,
    brush_pipeline: crate::renderer::brush_pipeline::BrushPipeline,
    /// `brush_pipeline` drawing the lighting-sources diagnostic. See `DebugView`.
    brush_sources_pipeline: crate::renderer::brush_pipeline::BrushPipeline,
    /// The brushes' back faces, drawn flat to seal T-junction cracks. See
    /// `brush_pipeline::SEAL_BRUSH_CRACKS`.
    brush_seal_pipeline: crate::renderer::brush_pipeline::BrushSealPipeline,
    /// `brush_pipeline` without blending, for the `perf_ab::Phase::OpaqueBrushes` measurement.
    brush_opaque_pipeline: crate::renderer::brush_pipeline::BrushPipeline,
    /// THE HALF-RESOLUTION PROBE PASS and the brush that reads it, with each
    /// eye's target. See `brush_pipeline::probe_pass`.
    brush_probe_pass_pipeline: crate::renderer::brush_pipeline::BrushPipeline,
    brush_probe_reader_pipeline: crate::renderer::brush_pipeline::BrushPipeline,
    probe_pass_targets: [crate::renderer::brush_pipeline::probe_pass::Target; 2],
    /// The same for the multiview scene pass. `None` without multiview, or if
    /// the device refused these pipelines. See `StereoProbePass`.
    stereo_probe: Option<StereoProbePass>,
    brush_mirror_pipeline: crate::renderer::brush_pipeline::BrushPipeline,
    brush_materials: crate::renderer::brush_pipeline::BrushMaterials,
    terrain_pipeline: crate::renderer::terrain_pipeline::TerrainPipeline,
    // Caves and anything else baked from a voxel sculpt. Shares the terrain's
    // material bind group -- see layered_mesh_pipeline for why that is the
    // point rather than a shortcut.
    shadow_map: crate::renderer::shadow::ShadowMap,
    /// The panorama behind the level, and the ambient inside it.
    sky_pipeline: crate::renderer::sky::SkyPipeline,
    sky_mirror_pipeline: crate::renderer::sky::SkyPipeline,
    sky: crate::renderer::sky::Sky,
    post: crate::renderer::uniforms::PostUpload,
    player: crate::renderer::uniforms::PlayerUpload,
    /// How much shadow work this headset does per frame. See `ShadowQuality`.
    pub shadow_quality: ShadowQuality,
    /// Fixed at construction, because every pipeline is built against it.
    /// Changing it means rebuilding the renderer, which is why it is read-only.
    msaa: MsaaLevel,
    layered_mesh_pipeline: crate::renderer::layered_mesh_pipeline::LayeredMeshPipeline,
    layered_mesh_mirror_pipeline: crate::renderer::layered_mesh_pipeline::LayeredMeshPipeline,
    // The material is per-scene, but it is built here with the fallback so
    // terrain renders through the real pipeline before any textures exist.
    // Replaced via `set_terrain_material` when a scene authors its own.
    terrain_material: crate::renderer::terrain_pipeline::TerrainMaterial,
    /// Layer textures and splat map are kept so either can be replaced alone.
    terrain_layers: Vec<Option<crate::renderer::terrain_pipeline::TerrainImage>>,
    terrain_normals: Vec<Option<crate::renderer::terrain_pipeline::TerrainImage>>,
    terrain_rough: Vec<Option<crate::renderer::terrain_pipeline::TerrainImage>>,
    terrain_ao: Vec<Option<crate::renderer::terrain_pipeline::TerrainImage>>,
    terrain_splat: Option<crate::renderer::terrain_pipeline::TerrainImage>,
    /// Baked sky visibility over the terrain footprint, or `None` for a scene
    /// that has not been baked. `None` binds white -- full sky -- which is the
    /// shading terrain had before this existed.
    terrain_sky_occlusion: Option<crate::renderer::terrain_pipeline::TerrainImage>,
    /// World x/z bounds the occlusion map spans, needed to turn a world
    /// position back into a texel. Kept beside the image because one is
    /// meaningless without the other.
    terrain_footprint: Option<(glam::Vec3, glam::Vec3)>,
    /// How the layers tile, from the project's `settings.json`. Held alongside
    /// the textures for the same reason they are: it arrives independently.
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
    /// Brushes, redrawn over the blitted scene so a polished floor reflects it.
    brush_ssr_pipeline: crate::renderer::brush_pipeline::BrushPipeline,
    /// THE BUFFERED REFLECTION PATH. See `buffered_reflections`.
    ///
    /// `trace` marches into the reflection buffer, a resolve filters it, and
    /// `composite` puts the filtered result on the surface.
    brush_ssr_trace_pipeline: crate::renderer::brush_pipeline::BrushPipeline,
    brush_ssr_composite_pipeline: crate::renderer::brush_pipeline::BrushPipeline,
    reflection_targets: [crate::renderer::ssr::ReflectionTarget; 2],
    /// Binding 1 is the FILTERED REFLECTION rather than the depth pyramid --
    /// see `wgsl_ssr_composite_block`. Same layout, so the composite pipeline
    /// has the same shape as every other reflective one.
    reflection_composite_bg: [wgpu::BindGroup; 2],
    /// `brush_ssr_pipeline` drawing each diagnostic. See `DebugView`.
    brush_ssr_sources_pipeline: crate::renderer::brush_pipeline::BrushPipeline,
    brush_ssr_debug_pipeline: crate::renderer::brush_pipeline::BrushPipeline,
    /// Which diagnostic brushes are drawn with. `Off` in normal use.
    debug_view: crate::renderer::brush_pipeline::DebugView,
    scene_targets: [SceneTarget; 2],
    ssr_camera_uniform: SsrCameraUniform,
    particle_pipeline: ParticlePipeline,
    uniform_buf: UniformBuffer,
    lights_uniform: LightsUniform,
    depth_view: wgpu::TextureView,
    eye_targets: Vec<[EyeTarget; 2]>,
    default_brush_lightmap: LoadedTexture,
    /// The level's brushes share ONE atlas, because they share one draw call.
    brush_lightmap: Option<LoadedTexture>,
    cuboid_lightmaps: HashMap<String, LoadedTexture>,
    default_cuboid_lightmap: LoadedTexture,
    mesh_lightmaps: HashMap<String, LoadedTexture>,
    default_mesh_lightmap: LoadedTexture,
    /// The sampler the probe cube array is read through. Owned here so
    /// rebinding a new bake does not have to recreate it.
    probe_sampler: wgpu::Sampler,
    /// Kept alive because the scene bind group references it. Dropping it while
    /// bound is exactly the kind of use-after-free wgpu cannot warn about.
    probe_view: Option<wgpu::TextureView>,
    /// Every baked probe: `(probe index, centre, min, max)`. Residency picks
    /// slots from these by probe; `probe_stream` turns the probe into a layer.
    ///
    /// The whole level's worth, kept so residency can choose which of them
    /// occupy the shader's slots each frame without re-reading the level.
    probe_volumes: Vec<(u32, glam::Vec3, glam::Vec3, glam::Vec3)>,
    /// Each probe's average photographed radiance, indexed by PROBE, for the
    /// shader's normalisation. See `uniforms::probe_mean_radiance`.
    probe_brightness: Vec<f32>,
    /// Each probe's ROOM, by probe index. See `ProbeUpload::set_volume`.
    probe_rooms: Vec<u32>,
    /// The doorways between rooms, from the bake. See `ProbeUpload::set_portals`.
    probe_portals: Vec<crate::renderer::uniforms::ProbePortal>,
    /// What stands inside the rooms, for the reflection trace. See
    /// `ProbeUpload::set_proxies`.
    probe_proxies: Vec<crate::renderer::uniforms::ProbeProxy>,
    /// The runtime switches, from the headset's lever file. See `levers`.
    levers: crate::renderer::levers::Levers,
    /// The GPU pool the level's probes stream through. See `probe_stream`.
    ///
    /// A `RefCell` like `eye`: the frame is rendered through `&self`, and the
    /// stream advances once a frame.
    probe_stream: std::cell::RefCell<Option<crate::renderer::probe_stream::ProbeStream>>,
    /// Eye adaptation, metered from the probes. See `exposure::EyeAdaptation`.
    eye: std::cell::RefCell<crate::renderer::exposure::EyeAdaptation>,
    /// Whether exposure follows the eye; off pins it at `post.exposure`.
    auto_exposure: bool,
    /// When the last frame was rendered, for the adaptation's time step.
    last_frame_at: std::cell::Cell<Option<std::time::Instant>>,
    frame_stats: FrameStats,
    /// Per-pass GPU timing, or `None` when the queue cannot timestamp.
    ///
    /// `FrameStats` says the frame is GPU-bound; this says which pass. Four
    /// slots because the scene and eye passes each run PER EYE, and one shared
    /// slot per pass kind would silently report only whichever eye was
    /// submitted last.
    pass_timers: Option<crate::renderer::pass_timers::PassTimers>,
    /// How many `PERF` windows have been logged since the levers last
    /// changed, which selects the `perf_ab::Phase` for the frames of the next
    /// one. Restarted with the levers, so every viewpoint's schedule starts at
    /// its baseline.
    perf_windows: u64,
    /// Every window since start, warm-ups included: the results file's index.
    perf_window_index: u64,
    /// The next window straddles a change of levers, so it measures neither
    /// configuration: logged as `warmup` and left out of the schedule.
    perf_warmup: bool,
    /// The runtime's counters, logged beside `PERF`. `None` without the
    /// extension. See `xr::perf_metrics`.
    perf_metrics: Option<crate::xr::PerfMetrics>,
    /// Those counters summed over the window's measured frames.
    perf_metric_window: crate::perf_metrics_log::WindowMeans,
    /// Where each window is also written, one JSON line apiece: the host
    /// script's copy, which no ring buffer can drop. See `perf_record`.
    perf_log: Option<crate::perf_record::PerfLog>,
    /// When the renderer was made, for the results file's clock.
    started_at: std::time::Instant,
    /// The tracked head, pinned: position and orientation in stage space. The
    /// eyes are moved onto it after they are located. See `bench`.
    pinned_head: Option<(glam::Vec3, glam::Quat)>,
    /// Each eye's field of view the last time the runtime located the views,
    /// for a pinned frame the runtime could not locate.
    last_fov: Option<[xr::Fovf; 2]>,
    /// The level's Baked lamps, in the player's frame, for surfaces with no
    /// lightmap to carry them. See `lights::append_baked`.
    baked_lights: Vec<crate::renderer::lights::Light>,
    /// Screen-space reflections, switchable at runtime.
    ///
    /// Starts at `scene_pass_plan::XR_SCREEN_SPACE_REFLECTIONS`. A controller
    /// button flips it, so whether SSR looks better -- or causes or hides an
    /// artefact -- can be judged in one view instead of across two builds.
    /// Which spots held shadow slots last frame, named by their index in the
    /// light list the caller passes in. See `lights::spot_shadow_slots`.
    ///
    /// `RefCell` because the choice is made in the middle of a frame that is
    /// already holding an immutable borrow of `self` for the draw lists, and
    /// this is the one thing in there that has to remember something. The
    /// renderer runs on one thread, and the borrow lives for two statements.
    shadow_spot_incumbents: std::cell::RefCell<Vec<usize>>,
    /// The last spot-shadow slot assignment that was LOGGED.
    ///
    /// Separate from `shadow_spot_incumbents`, which is the hysteresis state
    /// the selector reads. This one exists only so the log can fire on change
    /// instead of on a timer -- see `SHADOWSLOTS` in `render_frame`.
    shadow_slot_log: std::cell::RefCell<Vec<usize>>,
    /// The sky sun's shadow map, drawn once over the level and redrawn only
    /// when the level or the sun changes. See `lights::StaticSunShadow`.
    static_sun_shadow: std::cell::RefCell<Option<crate::renderer::lights::StaticSunShadow>>,
    /// Every scene pipeline again, built to draw BOTH EYES in one pass.
    ///
    /// `None` when the device lacks MULTIVIEW or MULTISAMPLE_ARRAY, which is
    /// also every development machine -- wgpu has no multiview on Metal.
    stereo_pipelines: Option<StereoScenePipelines>,
    /// The layered colour and depth a stereo scene pass draws into. `None`
    /// exactly when `stereo_pipelines` is.
    stereo_scene: Option<StereoSceneTextures>,
    /// Whether the scene pass draws both eyes at once. Requires
    /// `stereo_pipelines`; see `set_multiview_scene`.
    multiview_scene: bool,
    screen_space_reflections: bool,
    /// Whether reflections go through the trace-resolve-composite path rather
    /// than being marched and blended inline.
    ///
    /// A RUNTIME SWITCH, because nothing on the development machine can tell
    /// whether the new path draws anything: a rejected pipeline draws nothing,
    /// logs nothing on the device and makes the frame faster. With a switch a
    /// bad build is one button from a working picture, and the two paths can be
    /// compared from the same viewpoint in one session -- which is the only
    /// kind of measurement that has ever held up here.
    buffered_reflections: bool,
    /// Frames counted purely to rate-limit the shadow-caster diagnostic.
    ///
    /// A `Cell` because the count is bumped inside the block that already holds
    /// borrows of the caster lists; a plain field would need the whole shadow
    /// pass restructured to satisfy the borrow checker for a log line.
    shadow_diag_frames: std::cell::Cell<u64>,
}

/// One `PERF` window's numbers, handed back on the frame that closes it.
struct WindowStats {
    frames: u64,
    cpu_avg: f64,
    cpu_max: f64,
    gpu_avg: f64,
    gpu_max: f64,
    frame_ms: f64,
    fps: f64,
}

/// What `FrameStats::record` made of a frame.
struct FrameOutcome {
    /// It counted toward the window's averages (it was not settling).
    measured: bool,
    /// It closed the window, which was logged.
    closed: Option<WindowStats>,
}

struct FrameStats {
    window: u64,
    /// The leading frames of every window left out of its averages. A phase
    /// or a lever switches on a window boundary, and the frames right after
    /// the switch pay for it -- a shadow map redrawn, probes uploaded -- which
    /// belongs to neither configuration.
    settle: u64,
    /// Frames into the current window.
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

    /// Starts the window again, dropping what the current one had: the frames
    /// so far ran under something that no longer holds.
    fn restart(&mut self) {
        let (window, settle) = (self.window, self.settle);
        *self = Self::new(window, settle);
    }

    /// The window closes on its `window`-th frame, which is when a caller
    /// hangs its own once-per-window reporting off the same beat instead of
    /// keeping a second counter that drifts from this one.
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
        // 4x by default. On a tile GPU this is the cheapest quality the engine
        // buys anywhere, and stereo is where aliasing hurts most.
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
        // RENDER SCALE. See `RENDER_SCALE`.
        let recommended_w = view_configs[0].recommended_image_rect_width;
        let recommended_h = view_configs[0].recommended_image_rect_height;
        let scale = crate::renderer::RENDER_SCALE.clamp(0.3, 1.0);
        // Rounded UP to a multiple of 8. A tile GPU bins in fixed-size tiles,
        // so an awkward width leaves a partial tile down the edge; it also
        // keeps the half-resolution reflection buffer an exact half.
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
        // SAY SOMETHING WHEN THE GPU REFUSES SOMETHING.
        //
        // Without this a validation error goes nowhere: `create_render_pipeline`
        // hands back an object either way, drawing with the broken one is a
        // no-op, and the picture simply loses whatever that pipeline drew. On
        // 2026-09-17 four reflective pipelines were built against a layout they
        // no longer matched; every one failed, reflections stopped appearing,
        // the false-colour view came back as the ordinary picture, and the
        // device log said NOTHING for two builds while it was chased from
        // screenshots.
        wgpu_device.on_uncaptured_error(std::sync::Arc::new(|e| {
            error!("wgpu: {e}");
        }));
        // Captured HERE rather than read back off `vk` later: the timers scale
        // raw ticks by it, and a period of 1.0 would report Adreno ticks as
        // though they were nanoseconds -- numbers that look like milliseconds
        // and are not.
        let timestamp_period = vk.timestamp_period_ns;
        // Built here, while `wgpu_device` is still a local: the struct literal
        // below moves it.
        let pass_timers = timestamp_period.and_then(|period| {
            crate::renderer::pass_timers::PassTimers::new(
                &wgpu_device,
                // APPENDED, not inserted: the existing slots are addressed by
                // index from the passes themselves, so a new label in the
                // middle would silently retime them.
                &["scene_l", "eye_l", "scene_r", "eye_r", "prep_l", "prep_r", "refl_l", "refl_r", "probe_l", "probe_r"],
                period,
            )
        });

        let lights_uniform = LightsUniform::new(&wgpu_device);
        // Before the uniform buffer, whose bind group points at these textures.
        // Quarter the desktop resolution -- see shadow::QUEST_SHADOW_DIM.
        let shadow_map = crate::renderer::shadow::ShadowMap::with_dimension(
            &wgpu_device,
            crate::renderer::shadow::QUEST_SHADOW_DIM,
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
        // The scene pass is multisampled; the mirror and eye passes are not. Each
        // pipeline is built for the pass it runs in -- wgpu will not let one
        // pipeline serve two different sample counts.
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
        // See `brush_pipeline::probe_pass`.
        let probe_pass_layout = crate::renderer::brush_pipeline::probe_pass::bind_group_layout(&wgpu_device);
        let brush_probe_pass_pipeline = crate::renderer::brush_pipeline::BrushPipeline::new_probe_pass(
            &wgpu_device, &uniform_buf.layout, crate::renderer::multiview::ViewMode::Mono,
        );
        let brush_probe_reader_pipeline = crate::renderer::brush_pipeline::BrushPipeline::new_multisampled_probe_reader(
            &wgpu_device, wgpu_format, &uniform_buf.layout, samples, &probe_pass_layout,
            crate::renderer::multiview::ViewMode::Mono,
        );
        let probe_pass_targets: [crate::renderer::brush_pipeline::probe_pass::Target; 2] = std::array::from_fn(|_| {
            crate::renderer::brush_pipeline::probe_pass::Target::new(&wgpu_device, &probe_pass_layout, width, height, 1)
        });
        // White until a scene loads its materials, so an untextured level draws
        // in its authored colours rather than in nothing.
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
        // The mirror pass renders into a single-sampled target.
        let sky_mirror_pipeline = crate::renderer::sky::SkyPipeline::new(
            &wgpu_device, wgpu_format, &uniform_buf.layout, 1,
        );
        // No sky until a scene names one. Always BOUND, like terrain's
        // placeholder layers -- an optional binding would mean two layouts and
        // therefore two pipelines, and this is one texel plus a constant-band
        // SH that evaluates to exactly the ambient the engine always had.
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
        let skinned_mesh_pipeline =
            SkinnedMeshPipeline::new_multisampled(&wgpu_device, wgpu_format, &uniform_buf.layout, samples);
        // The mirror pass renders into a single-sampled target, so the skinned
        // pipeline needs a 1x twin. Every other pipeline used there already had
        // a separate mirror variant for its winding; this one did not, because
        // it does not cull.
        let skinned_mesh_mirror_pipeline =
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
        // The blit and the reflective solids read the scene DEPTH, which is
        // multisampled whenever the scene pass is, so they have to be built
        // knowing that.
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
        // THE STEREO PIPELINES, and only when the device can actually run them.
        //
        // Both features are required: MULTIVIEW for the pass itself, and
        // MULTISAMPLE_ARRAY because a multiview attachment is layered and this
        // renderer runs the scene pass at 4x MSAA. Without either, these are
        // never built and `multiview_scene` can never turn on -- a claimed
        // feature the driver does not have is ten pipelines that fail to
        // create, draw nothing and say nothing.
        let can_multiview = wgpu_device
            .features()
            .contains(wgpu::Features::MULTIVIEW | wgpu::Features::MULTISAMPLE_ARRAY);
        info!(
            "renderer: stereo scene pass {}",
            if can_multiview { "available" } else { "UNAVAILABLE (missing MULTIVIEW/MULTISAMPLE_ARRAY)" },
        );
        // ANY ONE OF THEM FAILING MUST DISABLE THE WHOLE STEREO PATH.
        //
        // A pipeline that fails to create is still an object, and binding it
        // invalidates the command buffer -- so the frame is never submitted and
        // the headset shows the last one it got. That is not a missing model,
        // it is a FROZEN VIEW, and it is what one wrong bind group layout did
        // here on 2026-09-19.
        //
        // Catching it at construction turns the worst failure in this renderer
        // into the mildest: multiview simply reports itself unavailable and the
        // per-eye path carries on.
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
                    // THE TERRAIN'S material layout, which is what the mono
                    // pipeline beside it uses. Guessing the brush's here built
                    // a pipeline whose shader wanted a sampler where the layout
                    // had a texture: it failed to create, and binding a failed
                    // pipeline invalidates the whole command buffer, so the
                    // frame was never submitted and the view FROZE (2026-09-19).
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
            // DRAWN INSIDE THE SCENE PASS, so it needs a stereo twin like
            // everything else there. Missing it bound a mono pipeline in a
            // multiview pass, which wgpu refuses at submit -- the whole frame is
            // dropped and the headset shows the last one it got.
            particle: ParticlePipeline::new_multisampled_stereo(
                &wgpu_device, wgpu_format, &uniform_buf.layout, samples,
            ),
        });

        // ONE PAIR OF LAYERED TEXTURES when the device can draw stereo, two
        // separate ones when it cannot. Either way each eye gets the same
        // `SceneTarget` view bundle, so the depth resolve, the pyramid, the mip
        // chain and the eye pass carry on per eye without knowing which it is.
        let stereo_pipelines = match pollster::block_on(stereo_scope.pop()) {
            Some(e) => {
                error!("renderer: stereo pipelines REFUSED, multiview disabled -- {e}");
                None
            }
            None => stereo_pipelines,
        };
        let can_multiview = can_multiview && stereo_pipelines.is_some();

        // Both eyes' half-resolution probe pass, in a scope of its own. See
        // `StereoProbePass`.
        let stereo_probe = if can_multiview {
            let scope = wgpu_device.push_error_scope(wgpu::ErrorFilter::Validation);
            let stereo = crate::renderer::multiview::ViewMode::Stereo;
            let built = StereoProbePass {
                pass: crate::renderer::brush_pipeline::BrushPipeline::new_probe_pass(&wgpu_device, &uniform_buf.layout, stereo),
                reader: crate::renderer::brush_pipeline::BrushPipeline::new_multisampled_probe_reader(
                    &wgpu_device, wgpu_format, &uniform_buf.layout, samples, &probe_pass_layout, stereo,
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
        // One reflection buffer per eye, and the bind group the composite reads
        // it through. The scene colour comes from the eye's own scene target --
        // the composite still needs it for the environment fallback.
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
        let default_cuboid_lightmap = create_lightmap_texture(
            &wgpu_device,
            &wgpu_queue,
            &solid_pipeline.lightmap_layout,
            &white_pixel,
            1,
            1,
            None,
        );
        // Brushes and meshes both ADD their lightmap rather than multiplying
        // it, so their neutral value is BLACK -- with an opaque alpha, because
        // the alpha carries sky visibility and MULTIPLIES. The cuboid pipeline
        // is the odd one out and still wants white. Binding the wrong one is
        // not subtle: white here is a full stop of extra brightness on every
        // unbaked surface in the level.
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
            probe_pass_targets,
            stereo_probe,
            brush_mirror_pipeline,
            water_pipeline,
            water_bodies: Vec::new(),
            brush_materials,
            terrain_pipeline,
            shadow_map,
            sky_pipeline,
            sky_mirror_pipeline,
            sky,
            // ACES at exposure 1 until a scene says otherwise. Defaulting to
            // the hard clamp instead would ship every level with its highlights
            // blown out, which is the thing the curve exists to fix.
            post: crate::renderer::uniforms::PostUpload::default(),
            player: crate::renderer::uniforms::PlayerUpload::default(),
            // SunAndSpot, not SunOnly: see the enum. A spot-lit interior gets
            // no shadow pass at all under SunOnly, and the cost of this is
            // per-source, so a scene with only a sun still pays for only one.
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
            uniform_buf,
            lights_uniform,
            depth_view,
            eye_targets,
            default_brush_lightmap,
            brush_lightmap: None,
            cuboid_lightmaps: HashMap::new(),
            default_cuboid_lightmap,
            mesh_lightmaps: HashMap::new(),
            default_mesh_lightmap,
            probe_sampler,
            probe_view: None,
            probe_volumes: Vec::new(),
            probe_brightness: Vec::new(),
            probe_rooms: Vec::new(),
            probe_portals: Vec::new(),
            probe_proxies: Vec::new(),
            levers: crate::renderer::levers::Levers::default(),
            probe_stream: std::cell::RefCell::new(None),
            eye: std::cell::RefCell::new(crate::renderer::exposure::EyeAdaptation::sky_only(
                crate::renderer::sky::SkyIrradiance::flat(crate::renderer::sky::AMBIENT),
            )),
            auto_exposure: true,
            last_frame_at: std::cell::Cell::new(None),
            // 120 frames a window, the first 8 of each left to settle.
            frame_stats: FrameStats::new(120, 8),
            pass_timers,
            perf_windows: 0,
            perf_window_index: 0,
            // The first window pays for startup: streaming, pipeline caches.
            perf_warmup: true,
            perf_metrics: crate::xr::PerfMetrics::new(&xr_ctx.instance, session),
            perf_metric_window: Default::default(),
            perf_log: None,
            started_at: std::time::Instant::now(),
            pinned_head: None,
            last_fov: None,
            baked_lights: Vec::new(),
            shadow_spot_incumbents: std::cell::RefCell::new(Vec::new()),
            shadow_slot_log: std::cell::RefCell::new(Vec::new()),
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

    /// Install the level-wide baked lighting for brushes.
    ///
    /// RGB is baked direct and bounced light; ALPHA is baked sky visibility.
    /// `light` says how the bake stored them: half floats for any current bake
    /// (uploaded as `Rgba16Float`, linear, unclipped), 8-bit sRGB for an older
    /// one (`Rgba8UnormSrgb`, whose transfer covers colour only, so alpha still
    /// arrives as the fraction the baker wrote). See `LightmapLight`.
    ///
    /// `direction` is the companion bounce-direction atlas -- same charts, same
    /// texels, same uv2, but unit vectors rather than colour. `None` is a level
    /// baked before directional bounce existed, and it shades exactly as it did
    /// then rather than being a missing feature.
    /// `sun_mask` is the baked sky-sun visibility (RGBA, red = visible
    /// fraction, green = baked), at `SUN_MASK_SCALE` times this atlas's density.
    /// `None` shades the brushes' sun from the level's static shadow map.
    /// `stationary` is the stationary lamps' shadow masks, one RGBA image per
    /// two lamps, all `stationary_size`, at `STATIONARY_MASK_SCALE` times
    /// this atlas's density; empty binds the neutral mask, under which every
    /// stationary lamp is unshadowed.
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
    }

    fn brush_lightmap_bg(&self) -> &wgpu::BindGroup {
        self.brush_lightmap
            .as_ref()
            .map(|t| &t.bind_group)
            .unwrap_or(&self.default_brush_lightmap.bind_group)
    }

    /// Bind a scene's baked reflection probes.
    ///
    /// `probes` pairs each probe's six faces with the world-space box it is
    /// parallax-corrected against -- `(centre, min, max)`. Call once when a
    /// level's bake loads; passing an empty slice restores the neutral cube,
    /// which is what a level with no probes wants and is also how to turn them
    /// off.
    ///
    /// Every probe must share one resolution, because they occupy layers of a
    /// single cube array. The baker writes one resolution per scene for exactly
    /// this reason; a mismatched one is dropped rather than stretched, since a
    /// stretched cube face is a visibly wrong reflection and a missing probe is
    /// just the sky.
    pub fn set_reflection_probes(
        &mut self,
        probes: &[(&[u8], u32, glam::Vec3, glam::Vec3, glam::Vec3)],
    ) {
        let resolution = probes.first().map(|p| p.1).unwrap_or(1);
        // SAY SO when a probe is left out. The array has one face size, so a
        // probe baked at another cannot be uploaded -- and dropping it quietly
        // is how a level loses a room's reflection with nothing in the log.
        // The baker bakes every probe of a scene at one size
        // (`bake::probe::unify_resolution`); this catches a stale bake.
        let usable: Vec<&(&[u8], u32, glam::Vec3, glam::Vec3, glam::Vec3)> =
            probes.iter().filter(|p| p.1 == resolution).collect();
        let mismatched = probes.len() - usable.len();
        if mismatched > 0 {
            log::warn!(
                "reflection probes: {mismatched} of {} dropped for a face size other than {resolution}px; re-bake the level's probes",
                probes.len(),
            );
        }
        // No room names on this path: probes sharing a BOX are one room's
        // cells, which is what the bake writes for a subdivided volume.
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
                crate::renderer::probe_stream::ProbeDesc { centre: p.2, min: p.3, max: p.4, volume: volume as u32 }
            })
            .collect();
        let owned: Arc<Vec<Vec<u8>>> = Arc::new(usable.iter().map(|p| p.0.to_vec()).collect());
        let source: crate::renderer::probe_stream::ProbeSource =
            Arc::new(move |i| owned.get(i).cloned());
        self.set_reflection_probes_streamed(descs, resolution, Vec::new(), source);
    }

    /// Bind a level's reflection probes for STREAMING: every probe described
    /// up front, pixels fetched through `source` as they are needed, and the
    /// doorways between rooms as `portals`. See `probe_stream`.
    ///
    /// `source(i)` is called once here for each probe, to meter it for the
    /// eye and measure its brightness, and again whenever the pool needs it;
    /// nothing keeps the pixels between calls.
    pub fn set_reflection_probes_streamed(
        &mut self,
        descs: Vec<crate::renderer::probe_stream::ProbeDesc>,
        resolution: u32,
        portals: Vec<crate::renderer::uniforms::ProbePortal>,
        source: crate::renderer::probe_stream::ProbeSource,
    ) {
        self.set_reflection_probes_with_depth(descs, resolution, portals, source, None);
    }

    /// As [`Self::set_reflection_probes_streamed`], with each probe's
    /// per-texel distances, so reflections are traced against what the probe
    /// saw instead of projected onto its box. See `probe_trace` in the shader.
    pub fn set_reflection_probes_with_depth(
        &mut self,
        descs: Vec<crate::renderer::probe_stream::ProbeDesc>,
        resolution: u32,
        portals: Vec<crate::renderer::uniforms::ProbePortal>,
        source: crate::renderer::probe_stream::ProbeSource,
        depth: Option<crate::renderer::probe_stream::ProbeDepthSource>,
    ) {
        use crate::renderer::uniforms::{ProbeUpload, MAX_PROBES};
        // ONE PASS OVER THE PIXELS, now: the brightness the shader normalises
        // by and the eye's meter both need every probe, and neither needs the
        // pixels again.
        let mut eye = crate::renderer::exposure::EyeAdaptation::sky_only(self.sky.irradiance);
        self.probe_brightness = descs
            .iter()
            .enumerate()
            .map(|(i, d)| match source(i) {
                Some(faces) => {
                    eye.add_probe(&faces, resolution, d.centre, d.min, d.max);
                    crate::renderer::uniforms::probe_mean_radiance(&faces, resolution)
                }
                None => 0.0,
            })
            .collect();
        *self.eye.borrow_mut() = eye;
        log::info!("reflection probes: average radiance by probe {:?}", self.probe_brightness);

        let stream = crate::renderer::probe_stream::ProbeStream::new_with_depth(
            &self.wgpu_device,
            &self.wgpu_queue,
            resolution,
            descs.len(),
            source,
            depth,
        );
        // Every probe's box, by PROBE index, for residency. The layer is
        // decided by the stream, per frame.
        self.probe_volumes =
            descs.iter().enumerate().map(|(i, d)| (i as u32, d.centre, d.min, d.max)).collect();
        self.probe_rooms = descs.iter().map(|d| d.volume).collect();
        self.probe_portals = portals;
        *self.probe_stream.get_mut() = Some(stream);

        // The initial slots: the first `MAX_PROBES`, in bake order. Replaced on
        // the first frame that runs residency.
        let mut upload = ProbeUpload { count: descs.len().min(MAX_PROBES) as u32, ..Default::default() };
        for (slot, (id, c, lo, hi)) in self.probe_volumes.iter().take(MAX_PROBES).enumerate() {
            upload.set(slot, *id, *c, *lo, *hi);
        }
        upload.fill_brightness(&self.probe_brightness);
        upload.fill_volumes(&self.probe_rooms);
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
        // The view must outlive the bind group that references it.
        self.probe_view = Some(view);
        log::info!(
            "reflection probes: {} described, {} doorway(s), pool {} at {resolution}px",
            descs.len(),
            self.probe_portals.len(),
            self.probe_stream.get_mut().as_ref().map(|s| s.pool_layers()).unwrap_or(0),
        );
    }

    pub fn set_mesh_lightmap(&mut self, key: &str, light: crate::renderer::mesh::LightmapLight, width: u32, height: u32) {
        let tex = crate::renderer::mesh::create_lightmap_texture_with_sun(
            &self.wgpu_device,
            &self.wgpu_queue,
            &self.mesh_pipeline.lightmap_layout,
            light,
            width,
            height,
            None,
            None,
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
    /// Replace the scene's sky, or clear it.
    ///
    /// Projecting the irradiance walks every texel of the panorama, so this is
    /// a scene-change operation and emphatically not a per-frame one. At 1024 x
    /// 512 that is half a million texels times nine basis functions -- tens of
    /// milliseconds, which is fine once and impossible sixty times a second.
    /// How this scene's radiance is mapped to the display.
    ///
    /// Renderer state set by the engine when a scene loads, exactly like the
    /// sky: it is a property of the level being shown, and the renderer has no
    /// way to read a scene file itself.
    /// Turn screen-space reflections on or off from the next frame.
    /// THE SCENE PASS'S PIPELINES, mono or stereo, chosen once per frame.
    ///
    /// Accessors rather than two copies of the pass body: the scene pass is a
    /// couple of hundred lines of draw calls and duplicating it to change which
    /// pipeline each one binds is how the two copies drift. `stereo` is false
    /// unless `multiview_scene` is on, which requires the pipelines to exist.
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
            _ => &self.mesh_pipeline.pipeline,
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
    fn sp_layered(&self, stereo: bool) -> &wgpu::RenderPipeline {
        match (stereo, &self.stereo_pipelines) {
            (true, Some(p)) => &p.layered_mesh.pipeline,
            _ => &self.layered_mesh_pipeline.pipeline,
        }
    }
    fn sp_water(&self, stereo: bool) -> &wgpu::RenderPipeline {
        match (stereo, &self.stereo_pipelines) {
            (true, Some(p)) => &p.water.pipeline,
            _ => &self.water_pipeline.pipeline,
        }
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
    /// The scene-pass brush, whichever diagnostic is showing.
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

    /// Draw both eyes in ONE scene pass, when the device can.
    ///
    /// Returns whether it took: asking for stereo on a device without
    /// MULTIVIEW or MULTISAMPLE_ARRAY leaves it off rather than building a pass
    /// that cannot run. A caller that ignores the answer and reports "multiview
    /// on" would be describing a frame that is still drawing one eye at a time.
    /// The level's Baked lamps for this frame, in the player's frame like the
    /// live ones. Lightmapped surfaces ignore them; characters and the ground
    /// are lit by them. See `lights::append_baked`.
    pub fn set_baked_lights(&mut self, lights: Vec<crate::renderer::lights::Light>) {
        self.baked_lights = lights;
    }

    /// What stands inside the level's rooms -- a pillar, a lamp -- in WORLD
    /// space, named by the same volumes as the probes. The reflection trace
    /// stops at them; the rooms' own boxes are its walls. Set with the probes,
    /// and again whenever the scene changes.
    pub fn set_reflection_proxies(&mut self, proxies: Vec<crate::renderer::uniforms::ProbeProxy>) {
        self.probe_proxies = proxies;
    }

    /// The runtime switches. Per-frame features read them in `render_frame`;
    /// the three that are renderer STATE -- screen-space reflections, the
    /// multiview scene pass, eye adaptation -- are applied here, and only when
    /// the file names them, so a lever file that says nothing about SSR leaves
    /// the in-headset switch alone.
    pub fn set_levers(&mut self, levers: crate::renderer::levers::Levers) {
        // A window that straddles the change measures neither side of it:
        // start the window, and the schedule, again from the baseline.
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
        self.levers = levers;
    }

    pub fn levers(&self) -> crate::renderer::levers::Levers {
        self.levers.clone()
    }

    /// Pin the tracked head -- stage space, from `bench::BenchRig` -- or give
    /// it back to the headset with `None`. Per frame, beside
    /// `set_player_frame`, whose offset and yaw the pin is paired with.
    pub fn set_pinned_head(&mut self, head: Option<(glam::Vec3, glam::Quat)>) {
        self.pinned_head = head;
    }

    /// Also write every `PERF` window to `path` as a JSON line. See
    /// `perf_record`.
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

    /// Whether this device could draw a stereo scene pass at all.
    pub fn multiview_available(&self) -> bool {
        self.stereo_pipelines.is_some()
    }

    /// Switch the buffered reflection path on or off. See `buffered_reflections`.
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

    /// Choose the diagnostic brushes are drawn with. See `DebugView`.
    pub fn set_debug_view(&mut self, view: crate::renderer::brush_pipeline::DebugView) {
        self.debug_view = view;
    }

    pub fn debug_view(&self) -> crate::renderer::brush_pipeline::DebugView {
        self.debug_view
    }

    pub fn set_post(&mut self, post: crate::renderer::uniforms::PostUpload) {
        self.post = post;
    }

    /// Where the player is, so geometry pinned to the WORLD can undo the
    /// player-frame transform. Per frame, unlike the sky or the tone curve --
    /// it changes every time they take a step.
    pub fn set_player_frame(&mut self, offset: glam::Vec3, yaw: f32) {
        self.player = crate::renderer::uniforms::PlayerUpload { offset, yaw };
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
        // The eye meters the sky until probes arrive; `set_reflection_probes`
        // folds this sky into them, so it must come first -- as the client's
        // load order already has it.
        *self.eye.borrow_mut() = crate::renderer::exposure::EyeAdaptation::sky_only(self.sky.irradiance);
    }

    /// Turn eye adaptation on or off. Off, exposure is exactly `post.exposure`.
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

    /// Apply a scene's baked terrain sky occlusion.
    ///
    /// Separate from the splat map even though both are footprint textures:
    /// splat is AUTHORED and occlusion is BAKED, so they change at different
    /// times and a single setter would make a re-bake discard hand-painted
    /// weights.
    /// Sky visibility at a WORLD x/z, from the baked terrain map.
    ///
    /// For things that move and so cannot be baked against: characters, props,
    /// anything dynamic. Sampled at ground level rather than at the object's
    /// own height, which is the approximation being made and it is a mild one --
    /// the question it answers is "is this thing under a roof", and the answer
    /// does not change much over the height of a person.
    ///
    /// 1.0 when nothing is baked, so an unbaked level shades exactly as before.
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
        // Clamped rather than wrapped: a character standing off the edge of the
        // terrain takes the nearest edge value, which is the open sky it is
        // actually under. Wrapping would teleport the sample to the far side.
        let u = ((world_x - min.x) / ex).clamp(0.0, 1.0);
        let v = ((world_z - min.z) / ez).clamp(0.0, 1.0);
        let tx = ((u * img.width as f32) as u32).min(img.width - 1);
        let ty = ((v * img.height as f32) as u32).min(img.height - 1);
        let i = ((ty * img.width + tx) * 4) as usize;
        img.rgba.get(i).map(|r| *r as f32 / 255.0).unwrap_or(1.0)
    }

    /// The world-space x/z extent the terrain occlusion map covers.
    pub fn set_terrain_footprint(&mut self, min: glam::Vec3, max: glam::Vec3) {
        self.terrain_footprint = Some((min, max));
    }

    /// Install the scene's water. Replaces whatever was there.
    ///
    /// Takes the tessellated surface rather than the `WaterDef` because the
    /// depth at each vertex comes from the TERRAIN, and the renderer has no
    /// heightfield -- see `space_soup_engine::water::build_surface`.
    pub fn set_water(
        &mut self,
        bodies: &[(
            Vec<crate::renderer::water_pipeline::WaterVertex>,
            Vec<u32>,
            crate::renderer::water_pipeline::WaterUniform,
        )],
    ) {
        use wgpu::util::DeviceExt;
        self.water_bodies = bodies
            .iter()
            .filter(|(v, i, _)| !v.is_empty() && !i.is_empty())
            .map(|(verts, indices, uniform)| {
                let vb = self.wgpu_device.create_buffer_init(
                    &wgpu::util::BufferInitDescriptor {
                        label: Some("water_vb"),
                        contents: bytemuck::cast_slice(verts),
                        // COPY_DST as well as VERTEX: the surface is static in
                        // the WORLD and moves in the PLAYER's frame, so its
                        // positions are rewritten whenever the player walks --
                        // and only then. See `update_water_surface`.
                        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    },
                );
                let ib = self.wgpu_device.create_buffer_init(
                    &wgpu::util::BufferInitDescriptor {
                        label: Some("water_ib"),
                        contents: bytemuck::cast_slice(indices),
                        usage: wgpu::BufferUsages::INDEX,
                    },
                );
                let ub = self.wgpu_device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("water_uniform"),
                    size: std::mem::size_of::<
                        crate::renderer::water_pipeline::WaterUniform,
                    >() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                self.wgpu_queue.write_buffer(&ub, 0, bytemuck::bytes_of(uniform));
                let bind_group = self.wgpu_device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("water_material"),
                    layout: &self.water_pipeline.material_layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: ub.as_entire_binding(),
                    }],
                });
                WaterBody {
                    vertex_buffer: vb,
                    index_buffer: ib,
                    uniform_buffer: ub,
                    bind_group,
                    index_count: indices.len() as u32,
                    uniform: *uniform,
                }
            })
            .collect();
        log::info!("water: {} body/bodies installed", self.water_bodies.len());
    }

    /// Rewrite one body's vertices, after the player has moved.
    ///
    /// Only when they HAVE moved: a lake is thousands of vertices, and pushing
    /// them across the bus every frame for a player standing still would cost
    /// more than the water does to draw.
    pub fn update_water_surface(
        &self,
        index: usize,
        verts: &[crate::renderer::water_pipeline::WaterVertex],
    ) {
        let Some(body) = self.water_bodies.get(index) else {
            return;
        };
        self.wgpu_queue.write_buffer(&body.vertex_buffer, 0, bytemuck::cast_slice(verts));
    }

    /// How many bodies of water are installed.
    pub fn water_body_count(&self) -> usize {
        self.water_bodies.len()
    }

    /// Advance the wave animation.
    ///
    /// Separate from `set_water` because the geometry is static and the time is
    /// not: re-uploading vertices every frame to move a wave would be the most
    /// expensive possible way to add a sine.
    pub fn set_water_time(&mut self, seconds: f32) {
        for body in &mut self.water_bodies {
            body.uniform.anim[0] = seconds;
            self.wgpu_queue.write_buffer(
                &body.uniform_buffer,
                0,
                bytemuck::bytes_of(&body.uniform),
            );
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
        self.rebuild_terrain_material();
    }

    /// Apply the materials a scene's brushes use.
    ///
    /// Per SCENE, unlike the terrain layers, which are one art decision for the
    /// whole project. A level's walls are its own: a warehouse and a bunker
    /// share nothing, and the array holds only what the scene in front of you
    /// actually references.
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

    /// Apply the scene's layer textures, filling any that did not load.
    pub fn set_terrain_layers(
        &mut self,
        layers: Vec<Option<crate::renderer::terrain_pipeline::TerrainImage>>,
        normals: Vec<Option<crate::renderer::terrain_pipeline::TerrainImage>>,
        // Roughness and occlusion, same slot order. Empty or short is fine;
        // the missing slots take their neutral and terrain looks as it did
        // before these maps existed.
        rough: Vec<Option<crate::renderer::terrain_pipeline::TerrainImage>>,
        ao: Vec<Option<crate::renderer::terrain_pipeline::TerrainImage>>,
    ) {
        self.terrain_layers = layers;
        self.terrain_normals = normals;
        self.terrain_rough = rough;
        self.terrain_ao = ao;
        self.rebuild_terrain_material();
    }

    /// Apply the project's terrain material settings -- tile sizes and normal
    /// strength.
    ///
    /// Its own setter rather than an argument to `set_terrain_layers`, because
    /// the settings can change without the textures changing: retiling gravel
    /// from 4m to 2m is the commonest thing anyone does to terrain art, and it
    /// should not mean re-uploading four textures.
    pub fn set_terrain_settings(
        &mut self,
        settings: crate::renderer::terrain_pipeline::TerrainMaterialUniform,
    ) {
        self.terrain_settings = settings;
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
                self.terrain_sky_occlusion.as_ref(),
                self.terrain_settings,
            );
    }

    pub fn cleanup(&self) {}
}
