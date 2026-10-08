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

/// FIXED FOVEATED RENDERING's state: which density maps exist and which one
/// the eye images point at. See `foveation` and [`XrRenderer::apply_foveation`].
struct FoveationState {
    eye_size: (u32, u32),
    /// The half-resolution probe pass's targets, foveated the same way: its
    /// reflections at the edges are read by fragments that cover several
    /// pixels there anyway.
    probe_size: (u32, u32),
    /// Each level's maps, once made: both eyes' images, then both eyes' probe
    /// pass targets.
    maps: HashMap<crate::renderer::foveation::FoveationLevel, [usize; 4]>,
    /// The level the eye targets point at; `None` until a frame has located
    /// the views, since each eye's map is centred by its field of view.
    applied: Option<crate::renderer::foveation::FoveationLevel>,
}

/// APPLICATION SPACEWARP's swapchains, pipelines and history, where the
/// runtime has it. See `space_warp`.
struct SpaceWarpState {
    motion: xr::Swapchain<xr::Vulkan>,
    depth: xr::Swapchain<xr::Vulkan>,
    /// Per swapchain image, per eye (each swapchain has a layer an eye).
    motion_targets: Vec<[EyeTarget; 2]>,
    depth_targets: Vec<[EyeTarget; 2]>,
    size: (u32, u32),
    pipelines: crate::renderer::space_warp::MotionPipelines,
    /// Every draw's two clip transforms this frame, both eyes, one slot each.
    cameras: wgpu::Buffer,
    camera_group: wgpu::BindGroup,
    /// The previous frame's per-eye view-projections and world-to-player,
    /// kept every frame so that switching on starts with a true history.
    prev: Option<([glam::Mat4; 2], glam::Mat4)>,
    /// Each mesh's model matrix last frame, by its model buffer (which lives
    /// as long as the mesh does).
    prev_models: HashMap<wgpu::Buffer, glam::Mat4>,
    /// This frame's motion and depth images, while held.
    acquired: Option<(usize, usize)>,
    /// What each eye's projection view points at; alive until the frame is
    /// handed to the compositor.
    info: [xr::sys::CompositionLayerSpaceWarpInfoFB; 2],
    /// The swapchains' raw images, for the diagnostic readback.
    motion_raw: Vec<vk::Image>,
    depth_raw: Vec<vk::Image>,
    depth_has_stencil: bool,
    readback: Option<crate::renderer::space_warp::Readback>,
    /// Per eye swapchain image, per eye: what the brushes' motion reads to
    /// move reflections as what they show -- that eye's layer of the image,
    /// and its probe pass's reach. See `space_warp::MotionKind::BrushReflect`.
    reflect_groups: Vec<[wgpu::BindGroup; 2]>,
}

impl SpaceWarpState {
    fn new(
        session: &xr::Session<xr::Vulkan>,
        device: &wgpu::Device,
        size: (u32, u32),
        vk_ctx: &VkContext,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // A depth format the runtime takes that maps onto wgpu's exactly --
        // D24S8 first, the one Meta's native AppSW guide names for Vulkan.
        // The first build used D32_SFLOAT: the runtime accepted it, ran
        // `Type=App` on every frame, and the frames it made were BLACK
        // (headset, 2026-09-29).
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
        // SAMPLED on both: the compositor reads them. The spec says "should"
        // and Meta's sample and guide both set it; the first build did not.
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
            acquired: None,
            info,
            motion_raw,
            depth_raw,
            depth_has_stencil: depth_format.has_stencil_aspect(),
            readback,
            // Once the eye images exist; see `XrRenderer::new`.
            reflect_groups: Vec::new(),
        })
    }
}

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

/// One installed body of water: its buffers, its waves and its optics.
struct WaterBody {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    uniform_buffer: wgpu::Buffer,
    /// One per set of the wave field's textures: drawn with
    /// `[waves.current()]`.
    bind_groups: [wgpu::BindGroup; 2],
    index_count: u32,
    /// Kept so the shore's time can be advanced without rebuilding the rest.
    uniform: crate::renderer::water_pipeline::WaterUniform,
    /// Its own sea: a pond's wind and fetch raise different waves from a
    /// bay's. Advanced once a frame, before the scene pass reads it.
    waves: crate::renderer::water_waves::WaveField,
    /// SpaceWarp's group 1 for it, by which set holds this frame's surface:
    /// drawn with `[waves.current()]`. `None` without SpaceWarp.
    motion_groups: Option<[wgpu::BindGroup; 2]>,
    /// The world box its surface can reach, waves and all: what decides
    /// whether it is seen. See `water_pipeline::water_seen`.
    bounds: (glam::Vec3, glam::Vec3),
    /// Seen this frame (and, until it is decided, last frame). Out of sight
    /// its waves stand still, and the frame it comes back its other set holds
    /// a surface long gone. A cell, as the frame decides it while holding
    /// the renderer's other parts.
    seen: std::cell::Cell<bool>,
    /// Its groups for the view from under it, by wave set. See `underwater`.
    under_groups: [wgpu::BindGroup; 2],
    /// Its still depth over the ground, to ask at an eye.
    still: crate::renderer::underwater::StillDepth,
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
    /// `reader` for the faces the sun never reaches, and for those whose
    /// baked sun mask always answers. See `brush_pipeline::SunFaces`.
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
    /// The level's water: each body's surface in the WORLD -- the vertex
    /// shader poses it into the player's frame -- its optics and its waves.
    /// Built on scene load.
    water_bodies: Vec<WaterBody>,
    /// The view from under the water. See `underwater`.
    underwater: crate::renderer::underwater::UnderwaterGpu,
    /// When an eye last came out of the water: the wet film.
    surfacing: crate::renderer::underwater::Surfacing,
    /// The waves' clock, seconds. See `set_water_time`.
    water_seconds: f64,
    /// How far that clock moved since the last frame, seconds: the shore's
    /// swash's step, for SpaceWarp's motion.
    water_step: f32,
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
    /// The readers for the brushes the sky's sun never reaches and for those
    /// whose baked sun mask always answers, and which faces those are --
    /// `None` until a level brings a baked sun mask. See
    /// `brush_pipeline::SunFaces`.
    brush_probe_reader_sunless_pipeline: crate::renderer::brush_pipeline::BrushPipeline,
    brush_probe_reader_baked_pipeline: crate::renderer::brush_pipeline::BrushPipeline,
    /// THE SCENE'S SPOTLESS TWINS: the three readers above and the models'
    /// pipelines with the spots' shadow code left out, drawn in frames where
    /// no spot casts -- the registers the tent took given back. See
    /// `lights::without_spot_shadows`; the lever `spotless_shaders`.
    spotless_readers: [crate::renderer::brush_pipeline::BrushPipeline; 3],
    spotless_mesh: MeshPipeline,
    /// This frame draws with them: no spot casts, no lit surface's light
    /// (whose readers' loop they leave out), and the lever is on. Set
    /// once the frame knows its spots, while its draw lists borrow the
    /// renderer, hence atomic.
    spotless_frame: std::sync::atomic::AtomicBool,
    sun_faces: Option<crate::renderer::brush_pipeline::SunFaces>,
    probe_pass_targets: [crate::renderer::brush_pipeline::probe_pass::Target; 2],
    /// The single-eye probe pass that ships: its secondary lookups left to
    /// `probe_fixups`, which writes them into each eye's target through
    /// `probe_fixup_targets`. See `probe_fixup`; the lever
    /// `deferred_reflection_lookups`.
    brush_probe_pass_deferred_pipeline: crate::renderer::brush_pipeline::BrushPipeline,
    /// THE PROBE PASSES' POOLLESS TWINS, the brushes' and the ground's: the
    /// torch pool maps' lookup left out, drawn in frames where no surface is
    /// lit. See `lights::without_pool_maps`; the lever `poolless_shaders`.
    brush_probe_pass_poolless_pipeline: crate::renderer::brush_pipeline::BrushPipeline,
    terrain_probe_pass_poolless_pipeline: crate::renderer::terrain_pipeline::TerrainPipeline,
    /// MEASUREMENT: the `pass_cut` / `scene_cut` levers' pipelines, drawn in
    /// place of the shipped probe pass / scene reader while set, and what
    /// building them takes. See `Levers::pass_cut`.
    pass_cut_pipeline: Option<(String, crate::renderer::brush_pipeline::BrushPipeline)>,
    /// MEASUREMENT: the ground's probe pass with a `terrain_cut_` register cut,
    /// drawn in place of `terrain_probe_pass_pipeline` while `pass_cut` names
    /// one. See `TerrainPipeline::new_probe_pass_with_cut`.
    terrain_cut_pipeline: Option<(String, crate::renderer::terrain_pipeline::TerrainPipeline)>,
    /// MEASUREMENT: the mesh pipelines with the thin pass shaded per fragment,
    /// while `Levers::thin_shading_sampled` is set.
    thin_sampled_pipeline: Option<crate::renderer::mesh_pipeline::MeshPipeline>,
    scene_cut_pipeline: Option<(String, crate::renderer::brush_pipeline::BrushPipeline)>,
    /// `Levers::water_cut`'s water.
    water_cut_pipeline: Option<(String, wgpu::RenderPipeline)>,
    cut_inputs: (wgpu::TextureFormat, u32, wgpu::BindGroupLayout),
    probe_fixups: crate::renderer::probe_fixup::ProbeFixups,
    probe_fixup_targets: [wgpu::BindGroup; 2],
    /// Group 3 of each eye's deferring probe pass: the record list, and that
    /// eye's floor mirror. See `probe_fixup::ProbeFixups::pass_bind_group_for`.
    probe_fixup_passes: [wgpu::BindGroup; 2],
    /// A little blur on every reflection, after the fix-up: each eye's target
    /// into its blurred colour, which the scene pass then reads. See
    /// `probe_blur`; the lever `reflection_blur`.
    probe_blur: crate::renderer::probe_blur::ProbeBlur,
    probe_blur_groups: [Option<wgpu::BindGroup>; 2],
    /// THE PLAYER ON CARDS: six views of their body drawn every frame into
    /// the card atlas's rows kept for them, which a reflection reads where it
    /// meets them. Made with the atlas, at its cards' size. See
    /// `character_cards`; the lever `character_cards`.
    character_cards: Option<crate::renderer::character_cards::CharacterCards>,
    /// THE PLAYER POSED ONCE A FRAME for the shadow tiles and the cards, and
    /// each skinned primitive's posed copy. See `skin_compute`; the lever
    /// `skin_once`.
    skin_compute: crate::renderer::skin_compute::SkinCompute,
    /// Behind a lock: the frame fills it while its draw lists borrow the
    /// renderer.
    posed_cache: std::sync::Mutex<crate::renderer::skin_compute::PosedCache>,
    /// The card atlas the level's models and the characters share: the
    /// characters' rows are copied into it each frame. See
    /// `proxy_cards::atlas_with_characters`.
    card_atlas: Option<crate::renderer::proxy_cards::CardAtlas>,
    /// THE TORCH'S POOL IN REFLECTIONS: each lit surface's light, made once a
    /// frame into the rows the atlas keeps under the player's cards, which a
    /// reflection meeting the surface reads. Made with the atlas, at its
    /// cards' size. See `pool_cards`.
    pool_cards: Option<crate::renderer::pool_cards::PoolCards>,
    /// THE GROUND in the probe pass and reading it back. See
    /// `TerrainPipeline::new_probe_pass`; the lever `terrain_probe_pass`.
    terrain_probe_pass_pipeline: crate::renderer::terrain_pipeline::TerrainPipeline,
    terrain_probe_reader_pipeline: crate::renderer::terrain_pipeline::TerrainPipeline,
    /// The ground reader's twins, `[spotless, baked, baked_spotless]`, and
    /// whether the level's ground map lets it draw the baked ones. See
    /// `TerrainPipeline::new_probe_reader_twins`, `Self::terrain_reader`.
    terrain_reader_twins: [crate::renderer::terrain_pipeline::TerrainPipeline; 3],
    /// The ground's gentle readers, `[full, spotless, baked, baked_spotless]`,
    /// for the triangles no steep pixel comes from. See `ground_twins`.
    terrain_gentle: Option<[crate::renderer::terrain_pipeline::TerrainPipeline; 4]>,
    /// The same for the triangles no gentle pixel comes from: the steep
    /// ground's code alone. See `ground_twins::steep_readers`.
    terrain_steep: Option<[crate::renderer::terrain_pipeline::TerrainPipeline; 4]>,
    /// The terrain's indices with each chunk's gentle triangles first, for the
    /// terrain the frame was last handed. See `ground_twins::SlopeSplit`.
    slope_split: Option<crate::renderer::ground_twins::SlopeSplit>,
    /// The ground's probe pass and its poolless twin with the repeated code
    /// written once. See `ground_twins::dedup_passes`.
    terrain_dedup_passes: Option<[crate::renderer::terrain_pipeline::TerrainPipeline; 2]>,
    terrain_sun_baked: bool,
    /// MEASUREMENT: the ground's reader, probe pass and poolless pass with
    /// their layer reads inlined, drawn in the shipped ones' place while
    /// `Levers::terrain_reader` is `inlined`. See `TerrainPipeline::new_inlined`.
    terrain_inlined: Option<[crate::renderer::terrain_pipeline::TerrainPipeline; 3]>,
    /// MEASUREMENT: every scene reader rebuilt with one of
    /// `brush_pipeline::READER_EDITS` (`Levers::reader_edit`): the brushes'
    /// full, sunless and baked classes and their spotless twins, and the
    /// ground's full reader and its twins, drawn in place of the shipped ones.
    reader_edits: Option<(
        String,
        [crate::renderer::brush_pipeline::BrushPipeline; 6],
        [crate::renderer::terrain_pipeline::TerrainPipeline; 4],
    )>,
    /// The same for the multiview scene pass. `None` without multiview, or if
    /// the device refused these pipelines. See `StereoProbePass`.
    stereo_probe: Option<StereoProbePass>,
    /// The brushes' depth, drawn first in the scene pass. See
    /// `BrushPipeline::new_depth_prepass`; the lever `depth_prepass`.
    brush_depth_prepass: crate::renderer::brush_pipeline::BrushPipeline,
    /// The same for the multiview scene pass, in its own error scope like
    /// `stereo_probe`: refused, stereo simply runs without a prepass.
    stereo_depth_prepass: Option<crate::renderer::brush_pipeline::BrushPipeline>,
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
    /// The stationary lamps' shadows on the ground, two lamps a layer, on the
    /// ground map's grid. Bound after it as layers of one array; see
    /// `TerrainImage::with_stationary_masks`.
    terrain_stationary: Vec<crate::renderer::terrain_pipeline::TerrainImage>,
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
    /// The lamps' veils, drawn last in the scene pass. See `glare`.
    glare_pipeline: crate::renderer::glare::GlarePipeline,
    /// The light sources that glare this frame, in the player's frame. See
    /// `set_glare_sources`.
    glare_sources: Vec<crate::renderer::glare::GlareSource>,
    /// The characters' capsules, which can stand between an eye and a lamp.
    /// See `set_capsules`.
    glare_capsules: Vec<(glam::Vec3, glam::Vec3, f32)>,
    /// THE LEVEL'S EFFECTS -- fire, smoke, embers, dust -- in the world's
    /// frame, as `set_effects` hands them over; simulated, lit and drawn each
    /// frame after the glass. See `effects`.
    effect_emitters: Vec<crate::renderer::effects::EffectEmitter>,
    /// SPLASHES the client saw, born on the water's clock (`set_water_time`),
    /// kept while their drops fly or their rings spread. See `add_splash`.
    splashes: Vec<crate::renderer::effects::Splash>,
    effects_layout: wgpu::BindGroupLayout,
    effects_pipeline: crate::renderer::effects::EffectsPipeline,
    /// Their textures and buffers, made when a level first has an effect.
    effects_gpu: Option<crate::renderer::effects::EffectsGpu>,
    /// THE LEVEL'S WEATHER, when it has any: its maps, the ground's weather
    /// twins, the particles. See `weather` and [`Self::set_weather`].
    weather: Option<crate::renderer::weather::WeatherScene>,
    /// The characters mirrored in the floor, into the probe pass targets'
    /// floor mirror, and its blur. See `brush_pipeline::probe_pass::MIRROR_FORMAT`.
    floor_mirror_skinned: SkinnedMeshPipeline,
    floor_mirror_mips: crate::renderer::brush_pipeline::probe_pass::MirrorMips,

    uniform_buf: UniformBuffer,
    lights_uniform: LightsUniform,
    depth_view: wgpu::TextureView,
    eye_targets: Vec<[EyeTarget; 2]>,
    /// Fixed foveated rendering, where the device has it. See `foveation`.
    foveation: Option<FoveationState>,
    /// Where each model's distance field lies in the bound atlas. See
    /// `proxy_field` and [`Self::set_reflection_proxies`].
    proxy_field_slots: Vec<crate::renderer::proxy_field::FieldSlot>,
    /// Application SpaceWarp, where the runtime has it. See `space_warp`.
    space_warp: Option<SpaceWarpState>,
    default_brush_lightmap: LoadedTexture,
    /// The level's brushes share ONE atlas, because they share one draw call.
    brush_lightmap: Option<LoadedTexture>,
    /// THE TIME OF DAY: its clock, the photographed sky to go back to, the
    /// bake's daylight layers. Default: none, a photographed sky as before.
    /// See `xr_renderer::time_of_day`.
    tod: time_of_day::TodState,
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
    /// The level's photographs as the models take their room's light from
    /// them: see `room_light`. Empty without probes.
    room_descs: Vec<crate::renderer::probe_stream::ProbeDesc>,
    /// Each probe's ROOM, by probe index. See `ProbeUpload::set_volume`.
    probe_rooms: Vec<u32>,
    /// The outdoor volume, when the level's probes name one: the volume none of
    /// whose photographs has distances. See `ProbeDesc::has_depth`.
    probe_outdoor_volume: Option<u32>,
    /// The buildings' outsides reflections leaving a building can meet: world
    /// boxes, in the stream's building layers' order. See `outdoor_radiance`.
    probe_buildings: Vec<(glam::Vec3, glam::Vec3)>,
    /// Their cubes, held until the next probe set builds its stream.
    pending_buildings: Vec<(glam::Vec3, glam::Vec3, Vec<u8>)>,
    /// The terrain's heights, from which the ground map is built. See
    /// `ground_map` and [`Self::set_terrain_heights`].
    ground_heights: Option<crate::renderer::ground_map::HeightGrid>,
    /// Something the ground map is made from changed since it was built. It is
    /// rebuilt once, before the next frame -- not once per setter at load.
    ground_dirty: bool,
    /// Where the bound ground map lies and its top; `None` for no ground map.
    ground_placement: Option<([f32; 4], f32)>,
    /// The doorways between rooms, from the bake. See `ProbeUpload::set_portals`.
    probe_portals: Vec<crate::renderer::uniforms::ProbePortal>,
    /// The rooms, by the numbers the doorways use, with which of them are
    /// closed, for culling what lies outside the building. See `portal_cull`
    /// and `set_closed_rooms`.
    cull_rooms: Vec<crate::renderer::portal_cull::CullRoom>,
    /// The doors this frame, in the world. See `doors` and `set_doors`.
    doors: Vec<crate::renderer::doors::DoorView>,
    /// Which of `probe_portals` a shut door seals, from `doors`.
    shut_portals: Vec<bool>,
    /// What stands inside the rooms, for the reflection trace. See
    /// `ProbeUpload::set_proxies`.
    probe_proxies: Vec<crate::renderer::uniforms::ProbeProxy>,
    /// The runtime switches, from the headset's lever file. See `levers`.
    levers: crate::renderer::levers::Levers,
    /// Whether the eye images can be copied out, and the last
    /// `Levers::eye_capture` request served. See `capture_eyes`.
    eye_capture_enabled: bool,
    eye_capture_served: u32,
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
    /// The performance levels the levers ask for. `None` without the
    /// extension. See `performance_level`.
    perf_settings: Option<crate::xr::PerfSettings>,
    /// DYNAMIC RESOLUTION: the runtime's size for each frame's eye layer,
    /// asked while the lever is on, and the window's answers for the `DYNRES`
    /// line. `None` without the extension. See `dynamic_resolution`.
    recommended_resolution: Option<crate::xr::RecommendedResolution>,
    recommendation_window: crate::renderer::dynamic_resolution::RecommendationWindow,
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
    /// Where the lamps casting the player's crisp shadows stood last frame,
    /// for `shadow::character_shadow_lamps`' hysteresis.
    character_shadow_held: std::cell::RefCell<Vec<glam::Vec3>>,
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

        // Copied out only when asked to before the app started: see
        // `Levers::eye_capture`.
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
        // FIXED FOVEATED RENDERING, before any pipeline exists: once on, every
        // render pass carries a density map and every pipeline must be made
        // compatible with that. See `foveation`.
        let foveation = (vk.fragment_density_map && unsafe { vulkan_interop::enable_foveation(&wgpu_device) }).then(|| {
            info!("foveation: available, {}x{} eye images", width, height);
            let probe = (width.div_ceil(2).max(1), height.div_ceil(2).max(1));
            FoveationState { eye_size: (width, height), probe_size: probe, maps: HashMap::new(), applied: None }
        });
        // LAST LAUNCH'S COMPILED SHADERS, before any of the renderer's
        // pipelines exist. See `pipeline_cache_file`.
        unsafe { vulkan_interop::start_pipeline_cache(vk, &wgpu_device) };
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
                &["scene_l", "eye_l", "scene_r", "eye_r", "prep_l", "prep_r", "refl_l", "refl_r", "probe_l", "probe_r", "fix_l", "fix_r", "mirror_l", "mirror_r", "mips_l", "mips_r", "blur_l", "blur_r", "cards", "card_mips", "pools", "pool_mips"],
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
        let underwater = crate::renderer::underwater::UnderwaterGpu::new(&wgpu_device, wgpu_format, &uniform_buf.layout, &probe_pass_layout, samples);
        // The effects' own group, shared by their pipelines and by the
        // textures made when a level first has an effect. See `effects`.
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
            // Built only for the log: the `terrain_reader` lever's inlined set.
            let _ = crate::renderer::terrain_pipeline::TerrainPipeline::new_inlined(
                &wgpu_device, wgpu_format, &uniform_buf.layout, samples, &probe_pass_layout, &probe_fixups,
            );
        }
        let brush_depth_prepass = crate::renderer::brush_pipeline::BrushPipeline::new_depth_prepass(
            &wgpu_device, wgpu_format, &uniform_buf.layout, samples, crate::renderer::multiview::ViewMode::Mono,
        );

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
        let spotless_mesh =
            MeshPipeline::new_multisampled_spotless(&wgpu_device, wgpu_format, &uniform_buf.layout, samples);
        let skinned_mesh_pipeline =
            SkinnedMeshPipeline::new_multisampled(&wgpu_device, wgpu_format, &uniform_buf.layout, samples);
        // The mirror pass renders into a single-sampled target, so the skinned
        // pipeline needs a 1x twin. Every other pipeline used there already had
        // a separate mirror variant for its winding; this one did not, because
        // it does not cull.
        let skinned_mesh_mirror_pipeline =
            SkinnedMeshPipeline::new(&wgpu_device, wgpu_format, &uniform_buf.layout);
        // The floor mirror's: the same shader at the mirror's own format, its
        // light kept linear by the untoned curve. See `probe_pass::MIRROR_FORMAT`.
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
            glare: crate::renderer::glare::GlarePipeline::new_multisampled_stereo(
                &wgpu_device, wgpu_format, &uniform_buf.layout, &probe_pass_layout, samples,
            ),
            effects: crate::renderer::effects::EffectsPipeline::new_multisampled_stereo(
                &wgpu_device, wgpu_format, &uniform_buf.layout, &probe_pass_layout, &effects_layout, samples,
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

        // APPLICATION SPACEWARP's swapchains, where the runtime offers it.
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
        // What SpaceWarp's brush motion reads for reflections: each eye image's
        // layer and that eye's probe pass reach.
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
            // 120 frames a window, the first 8 of each left to settle.
            frame_stats: FrameStats::new(120, 8),
            pass_timers,
            perf_windows: 0,
            perf_window_index: 0,
            // The first window pays for startup: streaming, pipeline caches.
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
        // The atlas's empty texels near its charts filled first: what the GPU
        // samples, and so what each face's reader is chosen by. See
        // `brush_pipeline::dilate_sun_mask`.
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
        // A sun off its baked direction: the same atlas, the neutral mask.
        if let Some(bg) = self.tod_brush_lightmap_bg() {
            return bg;
        }
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
                // No distances come with this path: no room is told apart as
                // the outdoors, and nothing is traced as outdoors.
                crate::renderer::probe_stream::ProbeDesc { centre: p.2, min: p.3, max: p.4, volume: volume as u32, has_depth: true, room_light: None }
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
    /// THE BUILDINGS' OUTSIDES, for the next probe set
    /// ([`Self::set_reflection_probes_with_depth`]) to put in layers of their
    /// own: each building's world box and its six outside faces at the probes'
    /// size. See `outdoor_radiance` in the shader.
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
        // The models' room light, from the bake's harmonics: see `room_light`.
        self.room_descs = descs.clone();
        log::info!(
            "room light: {} of {} photograph(s) carry their room's light for the models",
            descs.iter().filter(|d| d.room_light.is_some()).count(),
            descs.len(),
        );
        // ONE PASS OVER THE PIXELS, now: the brightness the shader normalises
        // by and the eye's meter both need every probe, and neither needs the
        // pixels again.
        // Under a time of day, binned with the lamps-only photographs and
        // relit to the hour; the stream reads them relit too. See
        // `xr_renderer::time_of_day`.
        let mut eye = self.tod_meter_probes(&descs, resolution, &source);
        let source = self.tod_probe_source(source);
        // The doorways, which the meter hands over across as the
        // reflections do. See `exposure::EyeAdaptation::meter`.
        eye.set_portals(&portals);
        *self.eye.borrow_mut() = eye;
        self.tod_probes_loaded();
        log::info!("reflection probes: average radiance by probe {:?}", self.probe_brightness);

        // THE SKY REFLECTIONS SEE, at the probes' size, in a layer of its own.
        let sky_faces = self.sky.reflection.as_ref().map(|r| r.cube_faces(resolution));
        // THE BUILDINGS' OUTSIDES after the sky, when the level has them.
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
        // THE OUTDOORS: the one volume none of whose photographs has distances,
        // when others do. See `ProbeDesc::has_depth`.
        self.probe_outdoor_volume = crate::renderer::probe_stream::outdoor_volume(&descs);
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
        // The view must outlive the bind group that references it.
        self.probe_view = Some(view);
        log::info!(
            "reflection probes: {} described, {} doorway(s), pool {} at {resolution}px",
            descs.len(),
            self.probe_portals.len(),
            self.probe_stream.get_mut().as_ref().map(|s| s.pool_layers()).unwrap_or(0),
        );
    }

    /// A mesh's baked light, and its stationary lamps' shadow masks on the
    /// same atlas -- one RGBA8 image per two lamps, all `stationary_size`; empty
    /// binds the neutral mask, under which every stationary lamp is unshadowed.
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

    /// The ground's reader for this frame, chosen as the brushes' are: its
    /// spotless twin when no spot casts (`spotless_frame`), its baked one when
    /// the level's ground map is baked over every texel (`terrain_sun_baked`).
    /// The full reader under the `scene_cut` lever, as the brushes', and under
    /// `terrain_reader` `full`; its inlined reads under `inlined`.
    /// The ground reader's WEATHER TWIN for this frame, chosen as
    /// [`Self::terrain_reader`] chooses the dry one.
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

    /// Which of the ground's four readers this frame draws -- `[full,
    /// spotless, baked, baked_spotless]` -- chosen as [`Self::terrain_reader`]
    /// chooses.
    fn terrain_twin(&self) -> usize {
        match (self.spotless_frame.load(std::sync::atomic::Ordering::Relaxed), self.terrain_sun_baked) {
            (false, false) => 0,
            (true, false) => 1,
            (false, true) => 2,
            (true, true) => 3,
        }
    }

    /// Whether this frame draws the ground's gentle triangles with its slope
    /// twins: not while a measurement lever draws a ground reader of its own.
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
        // Under `reader_edit`, its edited set in the shipped ones' places.
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
    /// The water's pipeline: its ringless twin while no splash's rings are
    /// spreading. See `water_pipeline::RING_BLOCK`.
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
    /// Whether the scene pass draws both eyes at once this frame.
    fn stereo_scene(&self) -> bool {
        self.multiview_scene && self.stereo_pipelines.is_some()
    }
    /// Whether this frame's half-resolution probe pass runs: the brushes'
    /// reflections, and the depth the glare finds walls in front of its lamps
    /// by. The diagnostic views keep the per-pixel shader, which is the only
    /// one that paints them; a stereo scene pass needs the two-eye pass, which
    /// the device may have refused (`StereoProbePass`).
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
    /// Which rooms are CLOSED -- walled all round but their doorways -- by the
    /// numbers the probes and doorways use. After `set_reflection_probes*`,
    /// whose room boxes it takes. From inside one, the terrain is drawn only
    /// where a doorway shows it. See `portal_cull`.
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

    /// THE DOORS THIS FRAME, in the WORLD: each leaf where it stands and
    /// whether it is shut. Their meshes are drawn with the other models (and
    /// marked `MeshInstance::tile_caster`); this is what their shadow tiles
    /// are fitted to and which doorways portal culling may not see through.
    /// See `doors`.
    pub fn set_doors(&mut self, doors: Vec<crate::renderer::doors::DoorView>) {
        self.shut_portals = crate::renderer::doors::shut_portals(&self.probe_portals, &doors);
        self.doors = doors;
    }

    /// What stands in the rooms for the reflection trace: the proxies, the
    /// models' distance fields (`ProbeProxy::field` indexes `fields`) and the
    /// models' cards (`ProbeProxy::cards` indexes `cards`). See `proxy_field`
    /// and `proxy_cards`.
    pub fn set_reflection_proxies(
        &mut self,
        mut proxies: Vec<crate::renderer::uniforms::ProbeProxy>,
        fields: Vec<crate::renderer::proxy_field::ProxyField>,
        cards: Vec<crate::renderer::proxy_cards::ProxyCards>,
    ) {
        // The models' cards, two rows each (colours, then normals), and the
        // rows the characters' cards are drawn into every frame; a proxy then
        // names its colours' row, and a set the atlas could not take leaves
        // its proxies with none.
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
        // The torch's pool maps, in the rows under the player's cards. See
        // `pool_cards`.
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
        // The models' distance fields, packed and bound; where each lies goes
        // up with every frame's probes. See `proxy_field`.
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
        if levers.pass_cut != self.levers.pass_cut {
            // The ground's cuts are named for it; any other is the brushes'.
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

    /// The A/B schedule's phase for the frame about to be drawn: Baseline
    /// unless the schedule runs (`perf_ab::ENABLED`, `Levers::ab_cycle`).
    pub fn ab_phase(&self) -> crate::renderer::perf_ab::Phase {
        if crate::renderer::perf_ab::ENABLED || self.levers.ab_cycle {
            crate::renderer::perf_ab::Phase::cycle_phase(self.perf_windows)
        } else {
            crate::renderer::perf_ab::Phase::Baseline
        }
    }

    /// The levers the frame about to be drawn runs with -- the lever file and
    /// the schedule's phase together, as the frame's own `fx` has them -- for
    /// what the app decides before the frame: the flashlight's bounce.
    pub fn frame_levers(&self) -> crate::renderer::levers::Levers {
        self.levers.clone().with_phase(self.ab_phase())
    }

    /// Whether frames go out with SpaceWarp's motion and depth: the lever is
    /// on AND the runtime gave us its swapchains. What the compositor is asked
    /// for besides follows this, not the lever -- see
    /// `layer_settings::sharpening_for`.
    pub fn space_warp_running(&self) -> bool {
        self.space_warp.is_some() && self.levers.space_warp
    }

    /// DYNAMIC RESOLUTION, first step: ask the runtime its size for the eyes'
    /// layer about to be submitted for `display_time` -- while the lever
    /// `dynamic_resolution` is on and the runtime has the extension -- and keep
    /// the answer for the window's `DYNRES` line. The eyes are drawn at their
    /// fixed size whatever it says. See `dynamic_resolution`.
    pub fn ask_recommended_resolution(&mut self, layer: &xr::sys::CompositionLayerProjection, display_time: xr::Time) {
        let Some(asker) = self.recommended_resolution.as_ref().filter(|_| self.levers.dynamic_resolution) else {
            return;
        };
        // SAFETY: `layer` is borrowed for the call, and its views and their
        // swapchain are the ones the caller submits right after.
        let answer = unsafe {
            asker.recommend(layer as *const xr::sys::CompositionLayerProjection as *const xr::sys::CompositionLayerBaseHeader, display_time)
        };
        self.recommendation_window.add(answer.map_err(|e| e.into_raw()));
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
        // The frame only: the characters' capsules are set on their own.
        self.player.offset = offset;
        self.player.yaw = yaw;
    }

    /// The light sources that glare, in the player's frame: each lamp's bulb,
    /// what it gives off and which sides it shows from. Set every frame, as a
    /// lamp switches or dims. See `glare`.
    pub fn set_glare_sources(&mut self, sources: Vec<crate::renderer::glare::GlareSource>) {
        self.glare_sources = sources;
    }

    /// THE LEVEL'S EFFECTS, in the world's frame: set when a level loads.
    /// Their textures are made the first time there are any. See `effects`.
    /// THE LEVEL'S WEATHER: one map an area, `texels` across each, and the
    /// terrain chunks (by their first index into the terrain's own index
    /// buffer) an area touches, which are drawn with the ground's weather
    /// twins. Builds the twins and the particles' pipeline, so a level
    /// without weather -- `texels` empty -- builds and pays nothing. Then
    /// [`Self::update_weather`] each frame.
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
        // The gentle readers' twins, by what the ground holds. See `ground_twins`.
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

    /// This frame's weather: its clock, each area's numbers, and -- when the
    /// app worked them out again (a few times a second) -- each area's map,
    /// texels as `space_soup_engine::weather::surface` makes them.
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

    /// The player's torch this frame (its beam, player frame), which lights
    /// the rain and snow in its cone; `None` while it is off.
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
    }

    /// The characters as capsules for this frame, nearest first, in the
    /// player's frame. See `uniforms::CapsuleUpload`.
    /// The surfaces the live lamps' beams light this frame, for reflections to
    /// show their light. See `lights::LitSurface`.
    pub fn set_lit_surfaces(&self, surfaces: &[crate::renderer::lights::LitSurface]) {
        self.lights_uniform.set_lit_surfaces(surfaces);
    }

    pub fn set_capsules(&mut self, groups: &[crate::renderer::uniforms::CapsuleGroup]) {
        self.player.capsules = crate::renderer::uniforms::CapsuleUpload::from_groups(groups);
        // Every body's, for the hands that shield an eye from a lamp. See
        // `glare`. Not a carried thing's: a torch's glass is a source of glare
        // itself, and would hide its own.
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
        // What the time of day relights the bake against, and goes back to.
        self.tod_sky_changed(pano.map(|p| (p.clone(), rotation_deg, intensity)));
        // The eye meters the sky until probes arrive; `set_reflection_probes`
        // folds this sky into them, so it must come first -- as the client's
        // load order already has it.
        *self.eye.borrow_mut() = crate::renderer::exposure::EyeAdaptation::sky_only(self.sky.irradiance);
        // The ground's light is the sky's and the sun's.
        self.ground_dirty = true;
    }

    /// The terrain's heights over the footprint the terrain's own maps span
    /// (see [`Self::set_terrain_footprint`]), from which the ground map is
    /// built. `None` for a level without ground. See `ground_map`.
    pub fn set_terrain_heights(&mut self, heights: Option<crate::renderer::ground_map::HeightGrid>) {
        self.ground_heights = heights;
        self.ground_dirty = true;
    }

    /// Build the ground map if anything it is made from changed. Once, before
    /// a frame, rather than in every setter a level load passes through.
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

    /// Rebuild the scene bind group with what is bound now -- after the
    /// ground map changes -- keeping the probes and their upload as they are.
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

    /// The stationary lamps' shadow masks on the ground, in layer order, two
    /// lamps a layer as on the brushes. Empty: the lamps light the ground
    /// unshadowed.
    pub fn set_terrain_stationary_masks(&mut self, masks: Vec<crate::renderer::terrain_pipeline::TerrainImage>) {
        self.terrain_stationary = masks;
        self.rebuild_terrain_material();
    }

    /// The world-space x/z extent the terrain occlusion map covers.
    pub fn set_terrain_footprint(&mut self, min: glam::Vec3, max: glam::Vec3) {
        self.terrain_footprint = Some((min, max));
    }

    /// Install the scene's water. Replaces whatever was there.
    ///
    /// Takes the tessellated surface rather than the `WaterDef` because the
    /// depth at each vertex comes from the TERRAIN, and the renderer has no
    /// heightfield -- see `space_soup_engine::water::build_surface`. The
    /// surface is in the WORLD and stays there: the vertex shader poses it.
    /// Toward the sky's sun, in the world; `None` without one. The client
    /// looks for what stands between a splash and it.
    pub fn sun_toward(&self) -> Option<glam::Vec3> {
        self.sky.sun.as_ref().map(|s| glam::Vec3::from(s.direction).normalize_or_zero())
    }

    /// SOMETHING STRUCK THE WATER: its drops and spray fly with the effects,
    /// its rings spread on every body's surface. `born` is on the clock
    /// `set_water_time` is given. See `effects::Splash`.
    pub fn add_splash(&mut self, mut splash: crate::renderer::effects::Splash) {
        const KEPT: usize = 48;
        // Onto the surface as drawn: the body whose still surface it struck.
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
        // Splashes fly with the effects: their textures now, not at the
        // first splash, which would stall a frame making them.
        if !self.water_bodies.is_empty() && self.effects_gpu.is_none() {
            self.effects_gpu = Some(crate::renderer::effects::EffectsGpu::new(&self.wgpu_device, &self.wgpu_queue, &self.effects_layout));
        }
    }

    /// How many bodies of water are installed.
    pub fn water_body_count(&self) -> usize {
        self.water_bodies.len()
    }

    /// Advance the waves.
    ///
    /// Only a clock: the wave field itself is computed on the GPU once a
    /// frame, and the shore's breakers from this time in the shader.
    pub fn set_water_time(&mut self, seconds: f32) {
        // Clamped as the wave field clamps its own frame: a jump in time is a
        // restart or a scene change, not a long frame.
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
        // The ground's baked readers draw it only where the map is baked over
        // every texel. See `TerrainImage::sun_baked_everywhere`.
        self.terrain_sun_baked = !self.tod.off_bake && occ.is_some_and(|s| s.sun_baked_everywhere());
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
        // The ground map is built from the same maps and layers.
        self.ground_dirty = true;
        // The ground map with the lamps' masks after it, when they fit it.
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
        // A SUN OFF ITS BAKED DIRECTION (the time of day): the map's sun marked
        // unbaked -- alpha up -- in its own layer only, so the ground's reader
        // takes the sun's shadow from the static map. See `set_sun_off_bake`.
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
    /// Point the eye images at `level`'s density maps, making them the first
    /// time the level is asked for. Needs a frame that located the views:
    /// each eye's map is centred where that eye looks straight ahead, which
    /// its field of view says (`foveation::centre_of_view`).
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
                // Written and waited for here, on the render thread, so no
                // pass that reads them is recorded before they are ready.
                match unsafe { vulkan_interop::add_foveation_maps(&self.wgpu_device, &patterns) } {
                    Some(ids) if ids.len() == 4 => {
                        state.maps.insert(level, [ids[0], ids[1], ids[2], ids[3]]);
                    }
                    _ => {
                        // Not asked again every frame.
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
