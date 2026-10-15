//! Ground shading: splat materials, biplanar on slopes, macro variation.
//!
//! Terrain was appended to the cuboid solid pass, and the commit that did it
//! argued against giving it a pipeline of its own -- "a second place for
//! shading to drift". That was the right call while terrain was flat vertex
//! colour and is the wrong one now: terrain needs a material array and a
//! sampler that no cuboid will ever use, and forcing that into the solid
//! pipeline means every cuboid in the scene carries a binding it does not want.
//!
//! The drift concern is answered directly rather than ignored: this shares
//! `wgsl_lights_block` with the solid pipeline, so lighting, shadowing and
//! attenuation are literally the same code. What differs is only how albedo is
//! obtained, which is the thing that genuinely differs.
//!
//! Three techniques, chosen for a tile-based mobile GPU rendering stereo:
//!
//!  - SPLAT: up to four tiling materials in a texture array, blended by slope
//!    and height. This is what makes ground read as ground rather than as a
//!    coloured mesh.
//!
//!  - BIPLANAR rather than triplanar. A heightfield has no natural UVs, and
//!    planar UVs stretch badly on exactly the cliffs you sculpt for cover.
//!    Triplanar fixes that with three samples per material; biplanar drops the
//!    axis contributing least and uses two, for a difference nobody can see.
//!    Flat ground skips it entirely and takes a single planar sample, and most
//!    of a map is flat, so most pixels pay the cheap path.
//!
//!  - MACRO VARIATION: one low-frequency sample multiplied over the result.
//!    A single tiling texture over 500 metres repeats visibly from any ridge.
//!    Stochastic/hex tiling solves that properly at three samples plus a
//!    histogram transform, which is not affordable here -- and stochastic
//!    blending that is not temporally stable shimmers under head motion, which
//!    is far more noticeable in stereo than on a monitor. One extra sample
//!    removes the repetition you actually notice.

use wgpu::*;

use super::cuboid::SolidVertex;
use super::material_wgsl::{wgsl_biplanar_block, wgsl_whiteout_block};

/// Per-scene terrain material settings, matching the WGSL uniform.
///
/// `repeat` values are in metres per tile: a 4m stone tile and a 12m grass tile
/// read very differently, and having one global scale is what makes every
/// terrain in an engine look like the same terrain.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TerrainMaterialUniform {
    /// Metres per tile for each of the four splat layers.
    pub repeat: [f32; 4],
    /// Slope in degrees at which layer 1 (rock) fully replaces layer 0 (ground).
    pub slope_start_deg: f32,
    pub slope_end_deg: f32,
    /// World Y band over which layer 2 (high ground) fades in.
    pub height_start: f32,
    pub height_end: f32,
    /// Metres per tile of the macro variation texture, and how strongly it
    /// modulates. Zero strength disables it without a shader variant.
    pub macro_repeat: f32,
    pub macro_strength: f32,
    /// Degrees past which biplanar sampling kicks in. Below this the shader
    /// takes one planar sample.
    pub biplanar_start_deg: f32,
    /// Non-zero when the bound splat map carries authored weights.
    ///
    /// A flag rather than a sentinel resolution or an all-zero texel: the
    /// shader must know whether to trust the map BEFORE it reads it, and a
    /// "weights that happen to look unauthored" rule would make a legitimately
    /// black-painted texel indistinguishable from no map at all.
    pub use_splat: f32,

    /// How strongly the normal maps perturb the surface. 0 disables them.
    ///
    /// No companion flag, unlike `use_splat`, because a flat normal map is a
    /// genuine no-op: the fallback texel (128, 128, 255) unpacks to (0, 0, 1),
    /// which is "unchanged" in tangent space. An unauthored normal set costs
    /// its samples and changes nothing, so the shader never has to be told
    /// whether to trust it.
    pub normal_strength: f32,
    /// WET SAND: ground below this world y is darkened as the sea's wash
    /// leaves it wet, over `wet_band` metres above it. Off far below.
    pub wet_line: f32,
    pub wet_band: f32,
    /// UNDER STILL WATER: ground below this world y lies under a sea or a
    /// lake's still surface and is soaked ([`SEA_LINE_WGSL`]). Off far below.
    /// Above it a beach is wetted by the wash, which moves: that is the
    /// weather's (`weather::AreaParams::shore`).
    pub sea_line: f32,
    /// WET GROUND, per layer (`weather::wet_ground`): how far soaking darkens
    /// it, 0..1 -- how much of the wet-surface law it takes: open-grained sand
    /// and soil all of it, dense rock a third.
    pub porosity: [f32; 4],
    /// And how much of it a film of water covers once it is soaked, 0..1: its
    /// gloss. Packed sand and bare rock hold a film nearly all over; a lawn
    /// hardly any, its blades standing through it (at 0.3 roughness a wet lawn
    /// lost its sheen and drew the hill's horizon across itself,
    /// `docs/weather-2026-10-07.md`).
    pub film: [f32; 4],
    /// THE GROUND'S SHAPE (`ground_shape`): the cells its footprint uv runs
    /// over along x and z, then one over the metres of a cell along each. All 0 where
    /// the material has no shape bound: the ground then shades by its
    /// interpolated vertex normals and its drawn triangles, as it did.
    pub shape: [f32; 4],
}

impl Default for TerrainMaterialUniform {
    fn default() -> Self {
        Self {
            repeat: [8.0, 4.0, 10.0, 6.0],
            slope_start_deg: 22.0,
            slope_end_deg: 40.0,
            height_start: 1e9, // off by default: most scenes have no height band
            height_end: 1e9,
            macro_repeat: 140.0,
            macro_strength: 0.35,
            biplanar_start_deg: 18.0,
            use_splat: 0.0,
            normal_strength: 1.0,
            wet_line: -1e9,
            wet_band: 1.0,
            sea_line: -1e9,
            // By the layers' roles: ground (a lawn), rock, high ground (scree
            // and dry grass), sediment (sand).
            porosity: WET_POROSITY,
            film: WET_FILM,
            shape: [0.0; 4],
        }
    }
}

/// GROUND UNDER STILL WATER IS SOAKED: below `mat.sea_line`, darkened by the
/// wet-surface law as far as its layers take water in (`weather::wet_albedo`,
/// written as the weather's twin writes it, its numbers `weather::WET_BACK`
/// and `WET_OUT` -- a test holds them equal), whole from five centimetres
/// under the line. No film's gloss: there is no surface of air over it. The
/// one line of the ground's shader that is the sea's; its weather twin takes
/// it out, and reads the same from the shore's layer of the weather map
/// (`weather::with_weather`).
pub(crate) const SEA_LINE_WGSL: &str = "    // UNDER STILL WATER the ground is soaked: see `SEA_LINE_WGSL`.
    let sea_in = dot(w, mat.porosity) * clamp((mat.sea_line - in.tex_pos.y) * 20.0, 0.0, 1.0);
    let sea_back = 0.435 * albedo;
    albedo = albedo * ((1.0 - sea_in * 0.446) + (sea_in * 0.554) * (sea_back * sea_back + sea_back));
";

/// The layers' response to water where `settings.json` gives none: by their
/// roles -- ground (a lawn), rock, high ground (scree and dry grass), sediment
/// (sand). See [`TerrainMaterialUniform::porosity`].
pub const WET_POROSITY: [f32; 4] = [0.85, 0.35, 0.7, 1.0];
pub const WET_FILM: [f32; 4] = [0.12, 0.75, 0.3, 0.9];

pub struct TerrainPipeline {
    pub pipeline: RenderPipeline,
    pub material_layout: BindGroupLayout,
}

/// THE SCENE'S GROUND IS DRAWN WITHOUT BLENDING. Its shader writes alpha 1
/// (one `return`, a literal; `the_grounds_scene_shaders_write_alpha_one`), so
/// a blend kept nothing of the target -- but a blended draw does not write
/// the GPU's coarse depth, so nothing drawn after the ground was rejected
/// there against it. Quest 3, 2026-10-13, one build launched both ways
/// (`bench/shader_opt_1012/r13/opaque{0,1}.log`): cave_back_out 30.82 ->
/// 25.77 ms, cave_mouth 31.72 -> 29.71, snow_hills -0.30, the other five
/// views tried -0.0 to -0.12. `false` builds it blended as it always was:
/// `adb shell setprop debug.spacesoup.opaque 0` before a launch on the
/// headset, `OPAQUE_GROUND=0` in the offline harness. See `build_from_with`.
pub static OPAQUE_GROUND: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

/// THE GROUND'S VERTICES ARE THE WORLD'S (`true`), posed into the player's
/// frame by the vertex shader ([`world_vertices`]) -- where the app posed
/// every vertex on the CPU and sent the whole ground to the GPU again on
/// each frame the rig moved or turned (`docs/scale-audit-2026-10-13.md`, F1).
/// Read when a ground pipeline is BUILT, so it is set once, before the
/// renderer exists: the headset app sets it at launch (`xr::vulkan`;
/// `adb shell setprop debug.spacesoup.worldground 0` keeps the old way), the
/// offline harness by `WORLD_GROUND`. `false` here, so everything else that
/// builds a ground pipeline -- the tests -- hands it posed vertices as before.
pub static WORLD_GROUND: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// A solid vertex shader's opening, and the same posing its vertex first:
/// what [`world_vertices`] rewrites. The ground's shaders and the solid
/// pipeline's (`pipeline::solid_shader`) open alike.
const POSED_VERTEX: &str = "@vertex fn vs_main(v: VIn) -> VOut {\n    var out: VOut;\n";
const WORLD_VERTEX: &str = "@vertex fn vs_main(w: VIn) -> VOut {\n    var out: VOut;\n    var v = w;\n    v.pos = to_player_space(w.pos);\n    v.norm = to_player_direction(w.norm);\n";

/// `src` taking its vertices in the WORLD: position and normal are turned
/// into the player's frame (`lights`' `to_player_space`, the inverse of the
/// `to_world_space` every shader already has) before anything reads them, so
/// all that follows is as it was for a vertex the app had posed. `None` if
/// the shader does not open as the ground's does.
pub fn world_vertices(src: &str) -> Option<String> {
    (src.matches(POSED_VERTEX).count() == 1).then(|| src.replacen(POSED_VERTEX, WORLD_VERTEX, 1))
}

impl TerrainPipeline {
    pub fn new(device: &Device, format: TextureFormat, uniform_layout: &BindGroupLayout) -> Self {
        Self::new_multisampled(device, format, uniform_layout, 1)
    }

    /// See `pipeline::SolidPipeline::new_multisampled` -- a pipeline's sample
    /// count must match the pass it runs in, so a 4x eye pass needs its own.
    pub fn new_multisampled(

        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        samples: u32,
    ) -> Self {
        Self::new_with_view(device, format, uniform_layout, samples, crate::renderer::multiview::ViewMode::Mono)
    }

    /// The same, drawing BOTH EYES in one pass. See `multiview::ViewMode`.
    pub fn new_multisampled_stereo(

        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        samples: u32,
    ) -> Self {
        Self::new_with_view(device, format, uniform_layout, samples, crate::renderer::multiview::ViewMode::Stereo)
    }

    fn new_with_view(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        samples: u32,
        view: crate::renderer::multiview::ViewMode,
    ) -> Self {
        Self::build(device, format, uniform_layout, samples, view, TerrainRole::Scene, None)
    }

    /// THE GROUND IN THE SCENE PASS, reading its probe reflection from the
    /// half-resolution probe pass (group 3, `probe_pass::bind_group_layout`)
    /// instead of tracing it per pixel -- as the brushes do. Single eye.
    pub fn new_probe_reader(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        samples: u32,
        probe_layout: &BindGroupLayout,
    ) -> Self {
        Self::build(
            device,
            format,
            uniform_layout,
            samples,
            crate::renderer::multiview::ViewMode::Mono,
            TerrainRole::Read,
            Some(probe_layout),
        )
    }

    /// The ground reader's TWINS, as the brushes' readers have theirs, in the
    /// order `[spotless, baked, baked_spotless]`, for the frames and levels
    /// where each is the same picture:
    /// - SPOTLESS (`lights::without_spot_shadows`) for frames where no spot
    ///   casts -- the frames the brushes draw with theirs
    ///   (`XrRenderer::spotless_frame`);
    /// - BAKED (`SUN_MASK_EVERYWHERE`, as `brush_pipeline::FaceSun::Baked`)
    ///   for a level whose ground map carries a baked sun over every texel
    ///   ([`TerrainImage::sun_baked_everywhere`]): its `receiver_sun_mask` is
    ///   then never the -1 of "no bake", and the level's static sun map is
    ///   never read.
    /// 2026-10-06: the reader alone was 4,292 instructions, past the Quest's
    /// instruction cache in every outdoor view, and had no twin at all.
    pub fn new_probe_reader_twins(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        samples: u32,
        probe_layout: &BindGroupLayout,
    ) -> [Self; 3] {
        [(true, false), (false, true), (true, true)].map(|(spotless, sun_baked)| {
            Self::new_probe_reader_twin(device, format, uniform_layout, samples, probe_layout, spotless, sun_baked)
        })
    }

    /// MEASUREMENT: the ground's scene readers with one of
    /// `brush_pipeline::READER_EDITS`: the full reader, then
    /// [`Self::new_probe_reader_twins`]' order. `None` when `edit` names no
    /// entry or one of its edits no longer matches. `Levers::reader_edit`;
    /// `edit@class` as the brushes take it (the ground has no `sunless`).
    pub fn new_edited_probe_readers(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        samples: u32,
        probe_layout: &BindGroupLayout,
        edit: &str,
    ) -> Option<[Self; 4]> {
        let (edit, only) = edit.split_once('@').map_or((edit, None), |(e, c)| (e, Some(c)));
        let shipped = terrain_shader_for(TerrainRole::Read);
        let edited = crate::renderer::brush_pipeline::with_reader_edit(shipped.clone(), edit)?;
        let class = |name: &str| if only.map_or(true, |c| c == name) { edited.clone() } else { shipped.clone() };
        let baked = |s: String| crate::renderer::brush_pipeline::sun_reader_shader(s, crate::renderer::brush_pipeline::FaceSun::Baked);
        let spotless = crate::renderer::lights::without_spot_shadows;
        let build = |source: String, label: &str| {
            Self::build_from(
                device,
                format,
                uniform_layout,
                samples,
                crate::renderer::multiview::ViewMode::Mono,
                TerrainRole::Read,
                Some(probe_layout),
                source,
                label,
            )
        };
        Some([
            build(class("full"), "terrain_pipeline_read_edited"),
            build(spotless(class("full")), "terrain_pipeline_read_spotless_edited"),
            build(baked(class("baked")), "terrain_pipeline_read_baked_edited"),
            build(spotless(baked(class("baked"))), "terrain_pipeline_read_baked_spotless_edited"),
        ])
    }

    /// One of [`Self::new_probe_reader_twins`]: the ground's reader without
    /// the spots' shadows when `spotless`, without the static sun map's
    /// lookup when `sun_baked`; with neither, the full reader.
    pub fn new_probe_reader_twin(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        samples: u32,
        probe_layout: &BindGroupLayout,
        spotless: bool,
        sun_baked: bool,
    ) -> Self {
        let mut source = terrain_shader_for(TerrainRole::Read);
        if sun_baked {
            source = crate::renderer::brush_pipeline::sun_reader_shader(source, crate::renderer::brush_pipeline::FaceSun::Baked);
        }
        if spotless {
            source = crate::renderer::lights::without_spot_shadows(source);
        }
        let label = match (spotless, sun_baked) {
            (false, false) => "terrain_pipeline_read",
            (true, false) => "terrain_pipeline_read_spotless",
            (false, true) => "terrain_pipeline_read_baked",
            (true, true) => "terrain_pipeline_read_baked_spotless",
        };
        Self::build_from(
            device,
            format,
            uniform_layout,
            samples,
            crate::renderer::multiview::ViewMode::Mono,
            TerrainRole::Read,
            Some(probe_layout),
            source,
            label,
        )
    }

    /// DIAGNOSTIC: the ground's scene reader answering with one of
    /// [`GROUND_DEBUG_VIEWS`] (`GROUND_DEBUG` in the offline frame). `None`
    /// for a name that is not one.
    pub fn new_debug_reader(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        samples: u32,
        probe_layout: &BindGroupLayout,
        view: &str,
    ) -> Option<Self> {
        let source = ground_debug_shader(&terrain_shader_for(TerrainRole::Read), view)?;
        Some(Self::build_from(
            device,
            format,
            uniform_layout,
            samples,
            crate::renderer::multiview::ViewMode::Mono,
            TerrainRole::Read,
            Some(probe_layout),
            source,
            "terrain_pipeline_read_debug",
        ))
    }

    /// MEASUREMENT: the ground's reader, its probe pass and that pass's
    /// poolless twin, in that order, with the layer reads inlined per layer as
    /// they were before the loop ([`with_inlined_layer_reads`]) -- drawn in the
    /// shipped ones' place, and without the reader's twins, while
    /// `Levers::terrain_reader` is `inlined`.
    pub fn new_inlined(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        samples: u32,
        probe_layout: &BindGroupLayout,
        fixups: &crate::renderer::probe_fixup::ProbeFixups,
    ) -> [Self; 3] {
        let pass = with_inlined_layer_reads(terrain_shader_for(TerrainRole::ProbePass));
        let in_pass = |source: String, label: &str| {
            Self::build_from(
                device,
                crate::renderer::brush_pipeline::probe_pass::FORMAT,
                uniform_layout,
                1,
                crate::renderer::multiview::ViewMode::Mono,
                TerrainRole::ProbePass,
                Some(fixups.pass_layout()),
                source,
                label,
            )
        };
        [
            Self::build_from(
                device,
                format,
                uniform_layout,
                samples,
                crate::renderer::multiview::ViewMode::Mono,
                TerrainRole::Read,
                Some(probe_layout),
                with_inlined_layer_reads(terrain_shader_for(TerrainRole::Read)),
                "terrain_pipeline_read_inlined",
            ),
            in_pass(pass.clone(), "terrain_probe_pass_inlined"),
            in_pass(crate::renderer::lights::without_pool_maps(pass), "terrain_probe_pass_poolless_inlined"),
        ]
    }

    /// THE GROUND IN THE HALF-RESOLUTION PROBE PASS: its probe reflection and
    /// nothing else, with the secondary lookups left to `fixups`, exactly as the
    /// brushes' pass does. See `brush_pipeline::probe_pass` and `probe_fixup`.
    pub fn new_probe_pass(
        device: &Device,
        uniform_layout: &BindGroupLayout,
        fixups: &crate::renderer::probe_fixup::ProbeFixups,
    ) -> Self {
        Self::build(
            device,
            crate::renderer::brush_pipeline::probe_pass::FORMAT,
            uniform_layout,
            1,
            crate::renderer::multiview::ViewMode::Mono,
            TerrainRole::ProbePass,
            Some(fixups.pass_layout()),
        )
    }

    /// The ground's probe pass's POOLLESS TWIN: [`Self::new_probe_pass`]
    /// without the torch pool maps' lookup (`lights::without_pool_maps`), for
    /// frames where no surface is lit. `Levers::poolless_shaders`.
    pub fn new_probe_pass_poolless(
        device: &Device,
        uniform_layout: &BindGroupLayout,
        fixups: &crate::renderer::probe_fixup::ProbeFixups,
    ) -> Self {
        Self::build_from(
            device,
            crate::renderer::brush_pipeline::probe_pass::FORMAT,
            uniform_layout,
            1,
            crate::renderer::multiview::ViewMode::Mono,
            TerrainRole::ProbePass,
            Some(fixups.pass_layout()),
            crate::renderer::lights::without_pool_maps(terrain_shader_for(TerrainRole::ProbePass)),
            "terrain_probe_pass_poolless",
        )
    }

    /// THE GROUND'S PROBE PASS'S OPEN-GROUND TWIN ([`open_pass_shader`]), for
    /// the chunks the nowhere twin would draw, reflecting the outdoors; with
    /// `weather_layout`, its weather twin. `None` when the shader's switch or
    /// the weather's edits moved. `Levers::open_reflections`.
    pub fn new_probe_pass_open(
        device: &Device,
        uniform_layout: &BindGroupLayout,
        fixups: &crate::renderer::probe_fixup::ProbeFixups,
        weather_layout: Option<&BindGroupLayout>,
    ) -> Option<Self> {
        let src = open_pass_shader()?;
        let (src, label) = match weather_layout {
            Some(_) => (crate::renderer::weather::with_weather(&src)?, "terrain_weather_probe_pass_open"),
            None => (src, "terrain_probe_pass_open"),
        };
        Some(Self::build_from_with(
            device,
            crate::renderer::brush_pipeline::probe_pass::FORMAT,
            uniform_layout,
            1,
            crate::renderer::multiview::ViewMode::Mono,
            TerrainRole::ProbePass,
            weather_layout,
            Some(fixups.pass_layout()),
            src,
            label,
        ))
    }

    /// The weather's ground where water stands, mirroring the sky alone
    /// ([`puddle_pass_shader`]). `None` when the shader's line or the
    /// weather's edits no longer match.
    pub fn new_probe_pass_puddle(
        device: &Device,
        uniform_layout: &BindGroupLayout,
        fixups: &crate::renderer::probe_fixup::ProbeFixups,
        weather_layout: &BindGroupLayout,
    ) -> Option<Self> {
        let src = crate::renderer::weather::with_weather(&puddle_pass_shader()?)?;
        Some(Self::build_from_with(
            device,
            crate::renderer::brush_pipeline::probe_pass::FORMAT,
            uniform_layout,
            1,
            crate::renderer::multiview::ViewMode::Mono,
            TerrainRole::ProbePass,
            Some(weather_layout),
            Some(fixups.pass_layout()),
            src,
            "terrain_weather_probe_pass_puddle",
        ))
    }

    /// THE GROUND'S PROBE PASS'S NOWHERE TWIN ([`nowhere_pass_shader`]), for
    /// the chunks of ground that touch no photograph's box and no doorway's
    /// reach this frame (`outdoor_pass::NowhereFrame`); with `weather_layout`,
    /// its weather twin. `None` when the shader's switch or the weather's
    /// edits moved. `Levers::nowhere_pass`.
    pub fn new_probe_pass_nowhere(
        device: &Device,
        uniform_layout: &BindGroupLayout,
        fixups: &crate::renderer::probe_fixup::ProbeFixups,
        weather_layout: Option<&BindGroupLayout>,
    ) -> Option<Self> {
        let src = nowhere_pass_shader()?;
        let (src, label) = match weather_layout {
            Some(_) => (crate::renderer::weather::with_weather(&src)?, "terrain_weather_probe_pass_nowhere"),
            None => (src, "terrain_probe_pass_nowhere"),
        };
        Some(Self::build_from_with(
            device,
            crate::renderer::brush_pipeline::probe_pass::FORMAT,
            uniform_layout,
            1,
            crate::renderer::multiview::ViewMode::Mono,
            TerrainRole::ProbePass,
            weather_layout,
            Some(fixups.pass_layout()),
            src,
            label,
        ))
    }

    /// THE GROUND'S PROBE PASS'S OUTDOOR TWIN ([`outdoor_pass_shader`]), for
    /// the chunks of ground every texel of which starts out of doors this frame
    /// (`outdoor_pass::all_outdoors`); with `weather_layout`, its weather twin
    /// (`weather::with_weather`), for those of them a weather area touches.
    /// `None` when the shader's switch or the weather's edits moved.
    /// `Levers::outdoor_pass`.
    pub fn new_probe_pass_outdoors(
        device: &Device,
        uniform_layout: &BindGroupLayout,
        fixups: &crate::renderer::probe_fixup::ProbeFixups,
        weather_layout: Option<&BindGroupLayout>,
    ) -> Option<Self> {
        let src = outdoor_pass_shader()?;
        let (src, label) = match weather_layout {
            Some(_) => (crate::renderer::weather::with_weather(&src)?, "terrain_weather_probe_pass_outdoors"),
            None => (src, "terrain_probe_pass_outdoors"),
        };
        Some(Self::build_from_with(
            device,
            crate::renderer::brush_pipeline::probe_pass::FORMAT,
            uniform_layout,
            1,
            crate::renderer::multiview::ViewMode::Mono,
            TerrainRole::ProbePass,
            weather_layout,
            Some(fixups.pass_layout()),
            src,
            label,
        ))
    }

    /// MEASUREMENT: the ground's probe pass with the register cut `cut` (one of
    /// `PROBE_PASS_REGISTER_CUTS`) applied, to DRAW with in the shipped one's
    /// place (the `pass_cut` lever). `None` for a cut it does not have, or
    /// one whose line moved.
    pub fn new_probe_pass_with_cut(
        device: &Device,
        uniform_layout: &BindGroupLayout,
        fixups: &crate::renderer::probe_fixup::ProbeFixups,
        cut: &str,
    ) -> Option<Self> {
        let (label, edits) = PROBE_PASS_REGISTER_CUTS.iter().find(|(label, _)| *label == cut)?;
        let mut src = terrain_shader_for(TerrainRole::ProbePass);
        for (from, to) in edits.iter() {
            if !src.contains(from) {
                return None;
            }
            src = src.replacen(from, to, 1);
        }
        Some(Self::build_from(
            device,
            crate::renderer::brush_pipeline::probe_pass::FORMAT,
            uniform_layout,
            1,
            crate::renderer::multiview::ViewMode::Mono,
            TerrainRole::ProbePass,
            Some(fixups.pass_layout()),
            src,
            label,
        ))
    }

    /// MEASUREMENT ONLY: the ground's probe pass with each of
    /// `PROBE_PASS_REGISTER_CUTS` applied, one pipeline each, built so the
    /// PIPESTATS log lists its registers. Nothing draws with them.
    pub fn log_probe_pass_register_cuts(
        device: &Device,
        uniform_layout: &BindGroupLayout,
        fixups: &crate::renderer::probe_fixup::ProbeFixups,
    ) {
        let base = terrain_shader_for(TerrainRole::ProbePass);
        for (label, edits) in PROBE_PASS_REGISTER_CUTS {
            let mut src = base.clone();
            if let Some((from, _)) = edits.iter().find(|(from, _)| !src.contains(from)) {
                log::warn!("register cut {label}: `{from}` is not in the shader");
                continue;
            }
            for (from, to) in edits.iter() {
                src = src.replacen(from, to, 1);
            }
            let _ = Self::build_from(
                device,
                crate::renderer::brush_pipeline::probe_pass::FORMAT,
                uniform_layout,
                1,
                crate::renderer::multiview::ViewMode::Mono,
                TerrainRole::ProbePass,
                Some(fixups.pass_layout()),
                src,
                label,
            );
        }
    }

    /// THE GROUND'S WEATHER TWINS (`weather::with_weather`): the reader in
    /// [`TerrainWeatherTwins::readers`]' order and the probe pass with its
    /// poolless twin, each with the weather's group 2 (`weather_layout`).
    /// Drawn in place of the shipped ones for the terrain chunks a weather
    /// area touches, and built only for a level with weather.
    #[allow(clippy::too_many_arguments)]
    pub fn new_weather_twins(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        samples: u32,
        probe_layout: &BindGroupLayout,
        fixups: &crate::renderer::probe_fixup::ProbeFixups,
        weather_layout: &BindGroupLayout,
    ) -> TerrainWeatherTwins {
        let twin = |src: String| crate::renderer::weather::with_weather(&src).expect("the ground's weather twin no longer matches its shader");
        let read = terrain_shader_for(TerrainRole::Read);
        let baked = crate::renderer::brush_pipeline::sun_reader_shader(read.clone(), crate::renderer::brush_pipeline::FaceSun::Baked);
        let spotless = crate::renderer::lights::without_spot_shadows;
        let reader = |src: String, label: &str| {
            Self::build_from_with(
                device,
                format,
                uniform_layout,
                samples,
                crate::renderer::multiview::ViewMode::Mono,
                TerrainRole::Read,
                Some(weather_layout),
                Some(probe_layout),
                twin(src),
                label,
            )
        };
        let pass_src = terrain_shader_for(TerrainRole::ProbePass);
        let pass = |src: String, label: &str| {
            Self::build_from_with(
                device,
                crate::renderer::brush_pipeline::probe_pass::FORMAT,
                uniform_layout,
                1,
                crate::renderer::multiview::ViewMode::Mono,
                TerrainRole::ProbePass,
                Some(weather_layout),
                Some(fixups.pass_layout()),
                twin(src),
                label,
            )
        };
        TerrainWeatherTwins {
            readers: [
                reader(read.clone(), "terrain_weather_read"),
                reader(spotless(read), "terrain_weather_read_spotless"),
                reader(baked.clone(), "terrain_weather_read_baked"),
                reader(spotless(baked), "terrain_weather_read_baked_spotless"),
            ],
            pass: pass(pass_src.clone(), "terrain_weather_probe_pass"),
            pass_poolless: pass(crate::renderer::lights::without_pool_maps(pass_src), "terrain_weather_probe_pass_poolless"),
        }
    }

    fn build(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        samples: u32,
        view: crate::renderer::multiview::ViewMode,
        role: TerrainRole,
        group3: Option<&BindGroupLayout>,
    ) -> Self {
        let label = match role {
            TerrainRole::Scene => "terrain_pipeline",
            TerrainRole::Read => "terrain_pipeline_read",
            TerrainRole::ProbePass => "terrain_probe_pass",
        };
        Self::build_from(device, format, uniform_layout, samples, view, role, group3, terrain_shader_for(role), label)
    }

    #[allow(clippy::too_many_arguments)]
    fn build_from(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        samples: u32,
        view: crate::renderer::multiview::ViewMode,
        role: TerrainRole,
        group3: Option<&BindGroupLayout>,
        source: String,
        label: &str,
    ) -> Self {
        Self::build_from_with(device, format, uniform_layout, samples, view, role, None, group3, source, label)
    }

    /// [`Self::build_from`] with a group 2 -- the weather's, for the ground's
    /// weather twins (`weather::with_weather`).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn build_from_with(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        samples: u32,
        view: crate::renderer::multiview::ViewMode,
        role: TerrainRole,
        group2: Option<&BindGroupLayout>,
        group3: Option<&BindGroupLayout>,
        source: String,
        label: &str,
    ) -> Self {
        Self::build_faced(device, format, uniform_layout, samples, view, role, group2, group3, source, label, FrontFace::Ccw)
    }

    /// THE GROUND IN A PLANAR WATER MIRROR (`planar_water`): the scene's
    /// ground, single-sampled and single-eye, with the winding a mirrored
    /// view reverses taken as the front.
    pub fn new_mirror(device: &Device, format: TextureFormat, uniform_layout: &BindGroupLayout) -> Self {
        let source = terrain_shader_for(TerrainRole::Scene);
        Self::build_faced(
            device,
            format,
            uniform_layout,
            1,
            crate::renderer::multiview::ViewMode::Mono,
            TerrainRole::Scene,
            None,
            None,
            source,
            "terrain_pipeline_mirror",
            FrontFace::Cw,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn build_faced(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        samples: u32,
        view: crate::renderer::multiview::ViewMode,
        role: TerrainRole,
        group2: Option<&BindGroupLayout>,
        group3: Option<&BindGroupLayout>,
        mut source: String,
        label: &str,
        front_face: FrontFace,
    ) -> Self {
        if WORLD_GROUND.load(std::sync::atomic::Ordering::Relaxed) {
            match world_vertices(&source) {
                Some(world) => source = world,
                // Drawn from world vertices it would stand in the wrong place.
                None => log::error!("WORLD GROUND: the ground shader '{label}' does not open as expected; it reads its vertices as posed"),
            }
        }
        if role == TerrainRole::ProbePass && device.features().contains(wgpu::Features::SHADER_EARLY_DEPTH_TEST) {
            // It writes the fix-up list: see `BrushPipeline::new_probe_pass_deferred`.
            source = source.replacen("@fragment fn fs_main(", "@fragment @early_depth_test(force) fn fs_main(", 1);
        }
        // Audited: see `shader_checks`.
        let shader = crate::renderer::shader_checks::audited_shader_module(device, ShaderModuleDescriptor {
            label: Some("terrain_shader"),
            source: ShaderSource::Wgsl(crate::renderer::shader_precision::for_device(device, view.shader(source)).into()),
        });
        let material_layout = material_bind_group_layout(device);
        let mut layouts = vec![Some(uniform_layout), Some(&material_layout)];
        if group2.is_some() || group3.is_some() {
            layouts.push(group2);
        }
        if let Some(l) = group3 {
            layouts.push(Some(l));
        }
        let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("terrain_layout"),
            bind_group_layouts: &layouts,
            immediate_size: 0,
        });

        // In the probe pass, both of its targets; the reach is left as the
        // brushes wrote it. See `probe_pass::targets`.
        let probe_targets = crate::renderer::brush_pipeline::probe_pass::targets(false);
        // The scene's ground is opaque -- its shader writes alpha 1 -- and
        // was drawn with blending on until 2026-10-13. Without, it is the
        // same colour (alpha 1 keeps nothing of what lay under it) and the
        // ground writes the GPU's coarse depth. See `OPAQUE_GROUND`.
        let blend = if OPAQUE_GROUND.load(std::sync::atomic::Ordering::Relaxed) { None } else { Some(BlendState::ALPHA_BLENDING) };
        let scene_target = [Some(ColorTargetState { format, blend, write_mask: ColorWrites::ALL })];
        let targets: &[Option<ColorTargetState>] =
            if role == TerrainRole::ProbePass { &probe_targets } else { &scene_target };
        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(&layout),
            vertex: VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: PipelineCompilationOptions::default(),
                // Same vertex format as the solid pass, so the geometry path is
                // untouched: terrain still arrives as SolidVertex.
                buffers: &[Some(SolidVertex::layout())],
            },
            fragment: Some(FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: PipelineCompilationOptions::default(),
                // The probe pass stores a reflection, not a colour to blend.
                targets,
            }),
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                cull_mode: Some(Face::Back),
                front_face,
                polygon_mode: PolygonMode::Fill,
                ..Default::default()
            },
            depth_stencil: Some(DepthStencilState {
                format: TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(CompareFunction::Less),
                stencil: StencilState::default(),
                bias: DepthBiasState::default(),
            }),
            multisample: MultisampleState { count: samples, ..Default::default() },
            multiview_mask: view.mask(),
            cache: None,
        });

        Self { pipeline, material_layout }
    }
}

/// The ground's weather twins. See [`TerrainPipeline::new_weather_twins`].
pub struct TerrainWeatherTwins {
    /// `[full, spotless, baked, baked_spotless]`, as the reader's twins are
    /// chosen (`XrRenderer::terrain_reader`).
    pub readers: [TerrainPipeline; 4],
    pub pass: TerrainPipeline,
    pub pass_poolless: TerrainPipeline,
}

/// The switch [`outdoor_pass_shader`] turns: see `PROBE_ALL_OUTDOORS` in the
/// lights block.
const ALL_OUTDOORS_OFF: &str = "const PROBE_ALL_OUTDOORS: bool = false;";
const ALL_OUTDOORS_ON: &str = "const PROBE_ALL_OUTDOORS: bool = true;";

/// The switch [`nowhere_pass_shader`] turns: see `PROBE_NOWHERE` in the lights
/// block.
const NOWHERE_OFF: &str = "const PROBE_NOWHERE: bool = false;";
const NOWHERE_ON: &str = "const PROBE_NOWHERE: bool = true;";

/// The switch [`open_pass_shader`] turns: see `PROBE_OPEN_GROUND` in the
/// lights block.
const OPEN_GROUND_OFF: &str = "const PROBE_OPEN_GROUND: bool = false;";
const OPEN_GROUND_ON: &str = "const PROBE_OPEN_GROUND: bool = true;";

/// THE GROUND'S PROBE PASS FOR OPEN GROUND THAT REFLECTS THE OUTDOORS: the
/// pass with `PROBE_OPEN_GROUND` set, for the same chunks as
/// [`nowhere_pass_shader`] (`outdoor_pass::NowhereFrame::holds`). Ground
/// smooth enough to trace shows the ground map, the buildings' outsides and
/// the sky -- as ground inside the baked outdoor volume does -- where the
/// full pass and the nowhere twin show nothing. `Levers::open_reflections`.
/// `None` if the switch is not in the shader exactly once.
pub(crate) fn open_pass_shader() -> Option<String> {
    let src = terrain_shader_for(TerrainRole::ProbePass);
    (src.matches(OPEN_GROUND_OFF).count() == 1).then(|| src.replacen(OPEN_GROUND_OFF, OPEN_GROUND_ON, 1))
}

/// The open ground's answer, and the sky alone in its place: what
/// [`puddle_pass_shader`] rewrites.
const OPEN_COLOUR: &str = "        let open_col = probe_escape_colour(world_pos, open_dir, dir, roughness, open_lod);";
const OPEN_SKY_COLOUR: &str = "        let open_col = vec4<f32>(sky_reflection(open_dir, dir, open_lod), 1.0);";

/// THE GROUND'S PROBE PASS FOR STANDING WATER BEYOND THE OUTDOOR VOLUME: the
/// open-ground twin ([`open_pass_shader`]) answering with the SKY ALONE, at
/// the lobe's blur -- no walk over the ground map for a hill, no buildings'
/// outsides. A puddle then mirrors the sky it lies under and not the ridge
/// behind it; the open twin's whole answer put the weather's pass past the
/// Quest's instruction cache (3,755 instructions; rain_puddles_down +3.6 ms,
/// 2026-10-13). `Levers::puddle_sky`. `None` if the line is not in the shader
/// exactly once.
pub(crate) fn puddle_pass_shader() -> Option<String> {
    let src = open_pass_shader()?;
    (src.matches(OPEN_COLOUR).count() == 1).then(|| src.replacen(OPEN_COLOUR, OPEN_SKY_COLOUR, 1))
}

/// THE GROUND'S PROBE PASS FOR GROUND IN NO PHOTOGRAPH'S BOX: the pass with
/// `PROBE_NOWHERE` set, so the lookup answers "nothing" at once, as the full
/// pass does for such ground after choosing among every photograph, starting
/// a trace and walking the doorways -- and the pass is the ground's material
/// and the characters' reflections alone. The same picture wherever
/// `outdoor_pass::NowhereFrame::holds`, which is where it is drawn. `None` if
/// the switch is not in the shader exactly once.
pub(crate) fn nowhere_pass_shader() -> Option<String> {
    let src = terrain_shader_for(TerrainRole::ProbePass);
    (src.matches(NOWHERE_OFF).count() == 1).then(|| src.replacen(NOWHERE_OFF, NOWHERE_ON, 1))
}

/// THE GROUND'S PROBE PASS FOR GROUND THAT IS ALL OUT OF DOORS: the pass with
/// `PROBE_ALL_OUTDOORS` set, so its reflection is the outdoor one alone -- the
/// ground map's trace, the buildings' outsides, the sky -- and none of the
/// rooms' machinery is compiled in. The same picture wherever
/// `outdoor_pass::all_outdoors` holds, which is where it is drawn. `None` if
/// the switch is not in the shader exactly once.
///
/// A surface too rough to trace takes the photographs' path in the full pass;
/// here, the outdoor volume's photographs alone (`probe_outdoor_untraced`),
/// which is all that path finds further from a doorway than its fades.
pub(crate) fn outdoor_pass_shader() -> Option<String> {
    let src = terrain_shader_for(TerrainRole::ProbePass);
    if src.matches(ALL_OUTDOORS_OFF).count() != 1 {
        return None;
    }
    let src = src.replacen(ALL_OUTDOORS_OFF, ALL_OUTDOORS_ON, 1);
    // The sky read once where the outdoor answer has it twice -- behind a
    // building and with none -- as the pass's dedup twin has it
    // (`brush_pipeline::DEDUP_OUTDOOR_SKY`: the same picture to the byte).
    let (from, to) = crate::renderer::brush_pipeline::DEDUP_OUTDOOR_SKY;
    Some(if src.matches(from).count() == 1 { src.replacen(from, to, 1) } else { src })
}

/// MEASUREMENT ONLY (`terrain_cut_rough_skip`): nothing traced where the scene
/// gives the reflection no share of its own texel, as the brushes' pass has it
/// (`brush_pipeline::probe_pass::PASS_LIGHTING`) -- from this roughness up the
/// scene shader multiplies the reflection by `1 - lobe_is_hemispherical`, 0
/// (`shade_material_env_part`).
///
/// NOT THE SAME PICTURE, so not in the pass: the reflection is half the
/// resolution each way, and a scene pixel a little smoother than its texel
/// still reads the texel -- skipped, it reads the sky's and the lightmap's
/// light in the trace's place. Offline, 2026-10-12: 1.1% of the front lawn's
/// pixels and 1.9% of the field's differ, nearly all by one level of 255, the
/// worst by 11 (docs/shader-optimization-2026-10-12.md). The cut says what the
/// rough ground's reflection costs.
///
/// After `probe_fragment = in.clip`, which is where the weather twin puts its
/// own code (`weather::with_weather`): wet ground is made smooth there, before
/// this asks, so a puddle is still traced. And after every implicit-derivative
/// sample, at the entry point's top level.
pub(crate) const PASS_ROUGH_SKIP: &str =
    "    if (clamp(rough, 0.04, 1.0) >= PROBE_LOBE_HEMISPHERICAL) {\n        return vec4<f32>(0.0);\n    }\n";

/// Where [`PASS_ROUGH_SKIP`] goes, and the same with it in.
const PASS_ROUGH_SKIP_AT: &str = "    probe_fragment = in.clip;\n    return probe_env_for_pass(\n";
const PASS_ROUGH_SKIP_IN: &str = "    probe_fragment = in.clip;\n    if (clamp(rough, 0.04, 1.0) >= PROBE_LOBE_HEMISPHERICAL) {\n        return vec4<f32>(0.0);\n    }\n    return probe_env_for_pass(\n";

/// MEASUREMENT ONLY: what `TerrainPipeline::log_probe_pass_register_cuts` cuts
/// out of the ground's probe pass, one pipeline each -- text edits of the
/// generated WGSL; `pass_cut` draws with one. 2026-10-06: the glass's gate
/// took the pass from 24 registers to 25 until it was asked after the
/// capsules' answer was made (`lights::capsule_glass_beam`).
const PROBE_PASS_REGISTER_CUTS: &[(&str, &[(&str, &str)])] = &[
    ("terrain_cut_none", &[]),
    // The rough ground not traced: what its reflection costs. Not the same
    // picture -- see `PASS_ROUGH_SKIP`.
    ("terrain_cut_rough_skip", &[(PASS_ROUGH_SKIP_AT, PASS_ROUGH_SKIP_IN)]),
    // ALL OF THE GROUND BY ITS OUTDOOR TWIN (`outdoor_pass_shader`): for its
    // statistics, and to draw a view of open ground with it alone -- on the
    // headset by this lever, off it by `TERRAIN_PASS_CUT`. Not the same
    // picture on ground in or beside a building.
    ("terrain_cut_all_outdoors", &[(ALL_OUTDOORS_OFF, ALL_OUTDOORS_ON)]),
    // ALL OF THE GROUND BY ITS NOWHERE TWIN (`nowhere_pass_shader`): for its
    // statistics, and what the pass costs with no lookup at all. Not the same
    // picture on ground inside a photograph's box.
    ("terrain_cut_nowhere", &[(NOWHERE_OFF, NOWHERE_ON)]),
    // ALL OF THE GROUND BY ITS OPEN-GROUND TWIN (`open_pass_shader`): for its
    // statistics, and off the headset (`TERRAIN_PASS_CUT`) to see open ground's
    // reflections on a whole view. Not the full pass's picture anywhere.
    ("terrain_cut_open", &[(OPEN_GROUND_OFF, OPEN_GROUND_ON)]),
    // No gate: the glow as the capsules give it.
    (
        "terrain_cut_gate",
        &[(
            "rgb * seen / a\n            * capsule_glass_beam(p, camera.capsules[glow_i * 2 + 1].xyz);",
            "rgb * seen / a;",
        )],
    ),
    // No beam's light on a lit surface in the pass's colours. See
    // `lights::probe_surface_relit`. 2026-10-06: worked out at every hit, that
    // light took the pass from 24 registers to 25 in every form tried (seen
    // head on, summed at half precision, both, a one-tap shadow); read from
    // its pool map now.
    (
        "terrain_cut_relight",
        &[(crate::renderer::lights::SURFACE_RELIT_READ, ""), (crate::renderer::lights::SURFACE_RELIT_CALL, "")],
    ),
    // The repeated code written once, as for the brushes' pass: the same
    // picture. See `brush_pipeline::DEDUP_EDITS`.
    ("terrain_cut_dedup_all", crate::renderer::brush_pipeline::DEDUP_EDITS),
    // ... which took this pass 24 -> 26 registers (build 120): each of the
    // three alone, to find which.
    ("terrain_cut_dedup_sky", &[crate::renderer::brush_pipeline::DEDUP_OUTDOOR_SKY]),
    ("terrain_cut_dedup_portals", &[crate::renderer::brush_pipeline::DEDUP_PORTAL_SAMPLES]),
    ("terrain_cut_dedup_near_far", &[crate::renderer::brush_pipeline::DEDUP_NEAR_FAR]),
    // The two that left it at 24, together (build 122).
    (
        "terrain_cut_dedup_sky_near_far",
        &[crate::renderer::brush_pipeline::DEDUP_OUTDOOR_SKY, crate::renderer::brush_pipeline::DEDUP_NEAR_FAR],
    ),
    // No escape colour: what the dedup's sky rewrite reaches on the ground.
    // Not the same picture.
    ("terrain_cut_escape_colour", &[("    if (h.escaped) {\n        col = probe_escape_colour(", "    if (false) {\n        col = probe_escape_colour(")]),
];

/// The lights block as the ground takes it: with `options`, testing each
/// lamp's range before its baked mask -- out of range is what the building's
/// lamps are from nearly all the ground. See `CULL_RANGE_FIRST`. And a spot's
/// edge averaged over the ground's own slope, not its layers' bumps: see
/// `spot_long_step`.
fn terrain_lights_block(options: crate::renderer::lights::LightsBlockOptions) -> String {
    crate::renderer::lights::wgsl_lights_block_with(
        0,
        1,
        crate::renderer::lights::LightsBlockOptions { cull_range_first: true, long_step_from_plane: true, open_fallback: true, ..options },
    )
}

/// WHAT A TERRAIN SHADER IS FOR. See `TerrainPipeline::new_probe_reader` and
/// `new_probe_pass`.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum TerrainRole {
    /// Shades the ground, tracing its probe reflection per pixel. What
    /// shipped, and still what a stereo pass and full-resolution reflections
    /// draw with.
    Scene,
    /// Shades the ground, reading its probe reflection from the probe pass.
    Read,
    /// Writes the ground's probe reflection into the probe pass, deferring its
    /// secondary lookups to `probe_fixup`.
    ProbePass,
}

/// WHETHER THE GROUND'S SHADERS ARE BUILT FOR A PAINTED LEVEL: what
/// [`terrain_shader_for`] hands every ground pipeline built from now on.
///
/// The ground's layer weights are the splat map's where the level has one
/// (`mat.use_splat`, set by `TerrainMaterial::new`) and worked out from the
/// slope where it has not -- and the shader as written works out both for
/// every pixel and selects, because which it is arrives in a uniform. On a
/// painted level the slope's weights, an `acos` among them, were computed and
/// thrown away: 176 of the gentle reader's 2,719 instructions, 179 of the
/// steep one's 3,010, 80 of the mixed one's 3,141, 93 of the pass's 9,215
/// (PIPESTATS, 2026-10-12, `shader_opt`'s `splat_probe`).
///
/// So it is a fact of the BUILD, per level: the app says what it expects
/// before the renderer exists ([`expect_painted_ground`], from the entry
/// level's own file), and the renderer rebuilds the ground's pipelines when a
/// level's material turns out otherwise ([`ground_build_moves`],
/// `XrRenderer::set_terrain_splat`). With it set the shaders have the painted
/// answer alone -- the same picture on a painted level; with it clear, the
/// choice by the uniform, right for any level.
///
/// MEASUREMENT: `adb shell setprop debug.spacesoup.painted 0` (read once when
/// the device is created) pins the build with the choice in it, the control;
/// `1` pins the painted build whatever the level holds, which is the WRONG
/// picture on an unpainted one ([`pin_ground_build`]).
pub static GROUND_PAINTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Whether a measurement pinned [`GROUND_PAINTED`]: levels no longer move it.
static GROUND_BUILD_PINNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// What the app expects of the level it is about to load, said before the
/// renderer builds its pipelines so they are built once. A wrong guess costs
/// one rebuild of the ground's pipelines when the level arrives, nothing else.
pub fn expect_painted_ground(painted: bool) {
    if !GROUND_BUILD_PINNED.load(std::sync::atomic::Ordering::Relaxed) {
        GROUND_PAINTED.store(painted, std::sync::atomic::Ordering::Relaxed);
    }
}

/// MEASUREMENT: the ground's build held at `painted`, whatever levels load.
pub fn pin_ground_build(painted: bool) {
    GROUND_PAINTED.store(painted, std::sync::atomic::Ordering::Relaxed);
    GROUND_BUILD_PINNED.store(true, std::sync::atomic::Ordering::Relaxed);
}

/// What the ground's pipelines are to be built for, given what they are built
/// for now, whether the level in hand has a splat map, and a measurement's
/// pin: the level's fact unless pinned.
pub fn ground_build_wanted(built_painted: bool, level_painted: bool, pinned: bool) -> bool {
    if pinned { built_painted } else { level_painted }
}

/// A level with (or without) a splat map is in hand: sets [`GROUND_PAINTED`]
/// to what its ground wants and says whether that MOVED it -- the ground's
/// pipelines are then stale and the caller rebuilds them.
pub fn ground_build_moves(level_painted: bool) -> bool {
    use std::sync::atomic::Ordering::Relaxed;
    let built = GROUND_PAINTED.load(Relaxed);
    let want = ground_build_wanted(built, level_painted, GROUND_BUILD_PINNED.load(Relaxed));
    GROUND_PAINTED.store(want, Relaxed);
    want != built
}

/// Where the ground chooses between its two sets of layer weights, and the
/// painted choice alone.
pub(crate) const PAINTED_FROM: &str = "    return select(derived, authored, mat.use_splat > 0.5);\n";
pub(crate) const PAINTED_TO: &str = "    return authored;\n";

/// `src` built for a painted level: see [`GROUND_PAINTED`]. `None` if the
/// choice is not in the shader exactly once.
pub(crate) fn painted_shader(src: &str) -> Option<String> {
    (src.matches(PAINTED_FROM).count() == 1).then(|| src.replacen(PAINTED_FROM, PAINTED_TO, 1))
}

/// The ground reader's answer, which a diagnostic picture replaces.
const GROUND_ANSWER: &str = "    return vec4<f32>(tonemap(in.col.rgb * lit), 1.0);\n";

/// DIAGNOSTIC PICTURES of the ground's scene reader, for the offline frame
/// (`GROUND_DEBUG=<name>`): each replaces the reader's answer with ONE
/// quantity as a colour, so a fault seen on the ground can be traced to the
/// term that draws it. One picture, one question.
///
/// - `albedo`: the layers' blended colour, unlit -- is the fault in the
///   texturing or in the light?
/// - `plane`: the plane the pixel is textured on, red / green / blue for the
///   projection along x / y / z, dimmer as the second plane's share grows.
/// - `stretch`: how much that plane STRETCHES the texture on the triangle
///   actually drawn: white where a texel covers its own size, falling to
///   black as it is pulled out, RED past four times (a streak).
/// - `facet`: the drawn triangle's own slope against the slope the
///   interpolated normal claims, as |difference| / 45 degrees.
/// - `lod`: the colour read's mip level / 10 (white = the smallest mips).
/// - `rough`: the roughness handed to the shading.
/// - `weights`: the layer weights, ground / rock / high as red / green / blue.
/// - `sky`, `sun`: the ground map's sky visibility and its baked sun mask.
/// - `normal`, `shape`: the shaded (mapped) normal and the interpolated one,
///   in the world, as colours.
/// - `layer0` .. `layer2`: one layer's colour alone, as the frame reads it.
/// - `ao`: the occlusion handed to the shading.
///
/// And `cut:<name>`: the LIT picture with one of `ground_cuts::GROUND_CUTS`
/// taken out of the reader -- what that part draws. And
/// `edit:<from>=><to>[;;<from>=><to>...]`: the lit picture with those exact
/// text edits of the WGSL (each `from` once in the shader), to try a change
/// of the reader on a picture without a build.
pub const GROUND_DEBUG_VIEWS: &[(&str, &str)] = &[
    ("albedo", "albedo"),
    ("plane", "select(select(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(0.0, 1.0, 0.0), f.detail_axis == 1u), vec3<f32>(1.0, 0.0, 0.0), f.detail_axis == 0u) * (2.0 * f.blend - 1.0 + 0.25)"),
    (
        "stretch",
        "select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(dbg_face[f.detail_axis]), dbg_face[f.detail_axis] > 0.25)",
    ),
    ("facet", "vec3<f32>(abs(degrees(acos(clamp(dbg_face.y, 0.0, 1.0))) - slope_deg) / 45.0)"),
    (
        "lod",
        "vec3<f32>(0.5 * log2(max(max(dot(f.detail_ddx, f.detail_ddx), dot(f.detail_ddy, f.detail_ddy)), 1e-12) * 512.0 * 512.0) / 10.0)",
    ),
    ("rough", "vec3<f32>(rough)"),
    ("weights", "w.rgb"),
    ("sky", "vec3<f32>(sky_vis)"),
    ("sun", "vec3<f32>(receiver_sun_mask)"),
    ("normal", "to_world_direction(shaded_n) * 0.5 + 0.5"),
    ("shape", "to_world_direction(n) * 0.5 + 0.5"),
    ("layer0", "layer_colour_at(0, f)"),
    ("layer1", "layer_colour_at(1, f)"),
    ("layer2", "layer_colour_at(2, f)"),
    ("ao", "vec3<f32>(ao_map)"),
];

/// `src` (a ground scene shader) answering with one of
/// [`GROUND_DEBUG_VIEWS`]; `None` for a name that is not one, or a shader
/// whose answer is not where it is expected.
pub fn ground_debug_shader(src: &str, view: &str) -> Option<String> {
    if let Some(cut) = view.strip_prefix("cut:") {
        let (_, edits) = crate::renderer::ground_cuts::GROUND_CUTS.iter().find(|(name, _)| *name == cut)?;
        return crate::renderer::ground_cuts::with_cut(src, edits);
    }
    if let Some(edits) = view.strip_prefix("edit:") {
        let edits: Vec<(&str, &str)> = edits.split(";;").map(|e| e.split_once("=>")).collect::<Option<_>>()?;
        return crate::renderer::ground_cuts::with_cut(src, &edits);
    }
    let (_, colour) = GROUND_DEBUG_VIEWS.iter().find(|(name, _)| *name == view)?;
    (src.matches(GROUND_ANSWER).count() == 1).then(|| {
        src.replacen(
            GROUND_ANSWER,
            &format!(
                "    let dbg_face = abs(normalize(cross(dpdx(in.tex_pos), dpdy(in.tex_pos))));\n    return vec4<f32>(clamp({colour}, vec3<f32>(0.0), vec3<f32>(1.0)) + 0.0 * lit, 1.0);\n"
            ),
            1,
        )
    })
}

/// The terrain shader for `role`: `terrain_shader`, with its reflection read
/// from the probe pass or its fragment stage cut down to the reflection alone.
pub fn terrain_shader_for(role: TerrainRole) -> String {
    terrain_shader_built(role, GROUND_PAINTED.load(std::sync::atomic::Ordering::Relaxed))
}

/// [`terrain_shader_for`] for a level known to be `painted` or not: see
/// [`GROUND_PAINTED`].
pub fn terrain_shader_built(role: TerrainRole, painted: bool) -> String {
    let src = terrain_shader();
    let src = match painted {
        true => painted_shader(&src).unwrap_or(src),
        false => src,
    };
    if role == TerrainRole::Scene {
        return src;
    }
    let plain_lights = terrain_lights_block(crate::renderer::lights::LightsBlockOptions::default());
    assert!(src.contains(&plain_lights), "the terrain shader no longer embeds the lights block as generated");
    let shade = "    let lit = shade_material_env(\n";
    assert!(src.contains(shade), "the terrain shader no longer shades through `shade_material_env`");
    match role {
        TerrainRole::Scene => unreachable!(),
        TerrainRole::Read => {
            let read_lights = terrain_lights_block(crate::renderer::lights::LightsBlockOptions {
                probe_from_pass: true,
                ..Default::default()
            });
            let src = src.replacen(&plain_lights, &read_lights, 1).replacen(
                shade,
                &format!(
                    "    // From the half-resolution probe pass, as the brushes read it. See\n    // `brush_pipeline::probe_pass`.\n    let probe_pass_tolerance = max(4.0 * fwidth(in.clip.z), 1e-6);\n    probe_env_given = probe_pass_upsample(in.clip.xy, in.clip.z, probe_pass_tolerance, 0.0);\n{shade}"
                ),
                1,
            );
            // The ground writes no face code (`probe_pass::targets(false)`), so
            // it reads with none: 0, which every texel matches.
            format!(
                "{src}{}{}",
                crate::renderer::brush_pipeline::probe_pass::READER_WGSL,
                crate::renderer::brush_pipeline::probe_pass::FACE_CODE_WGSL,
            )
        }
        TerrainRole::ProbePass => {
            let pass_lights = terrain_lights_block(crate::renderer::lights::LightsBlockOptions {
                defer_secondary: true,
                ..Default::default()
            });
            let src = src.replacen(&plain_lights, &pass_lights, 1);
            let start = src.find(shade).expect("checked above");
            let end = start + src[start..].find("\n}\n").expect("the fragment stage ends");
            // The reflection for the pass to store, from the same inputs the
            // shading below hands `shade_material_env`: no baked bounce, the
            // ground's own position to choose from, its shape for Fresnel.
            format!(
                "{}    probe_fragment = in.clip;\n    return probe_env_for_pass(\n        in.world_pos, shaded_n, rough, clamp(ao_map, 0.0, 1.0), sky_vis, vec3<f32>(0.0), in.world_pos, n,\n    );{}",
                &src[..start],
                &src[end..],
            )
        }
    }
}

pub fn material_bind_group_layout(device: &Device) -> BindGroupLayout {
    device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("terrain_material_layout"),
        entries: &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::D2Array,
                    multisampled: false,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Sampler(SamplerBindingType::Filtering),
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 3,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Per-layer normal maps. Always bound, like the weights: an
            // optional binding would mean two layouts and therefore two
            // pipelines, and a flat placeholder costs one texel per layer and
            // changes nothing.
            BindGroupLayoutEntry {
                binding: 5,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::D2Array,
                    multisampled: false,
                },
                count: None,
            },
            // Baked sky visibility over the footprint. Always bound; an unbaked
            // terrain gets a 1x1 WHITE texel, which is "sees the whole sky" and
            // reproduces the old shading exactly. White is the neutral here
            // because this value is MULTIPLIED -- the opposite of the brush
            // lightmap's additive RGB, whose neutral is black.
            BindGroupLayoutEntry {
                binding: 6,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::D2Array,
                    multisampled: false,
                },
                count: None,
            },
            // Per-layer roughness and ambient occlusion, giving terrain the
            // same material set brush surfaces have. Always bound: a layer
            // with no map gets a solid neutral, so this is one pipeline rather
            // than two, for the same reason the normal array is.
            BindGroupLayoutEntry {
                binding: 7,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::D2Array,
                    multisampled: false,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 8,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::D2Array,
                    multisampled: false,
                },
                count: None,
            },
            // Authored blend weights over the terrain footprint. Always bound,
            // even when unauthored: an optional binding would mean two bind
            // group layouts and therefore two pipelines, and a 1x1 placeholder
            // costs one texel.
            BindGroupLayoutEntry {
                binding: 4,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            // THE GROUND'S STEP-1 SHAPE (`ground_shape`): its normals and
            // heights a texel a sample, loaded, never filtered. Always
            // bound: a 1x1 placeholder and `mat.shape` 0 where there is none.
            BindGroupLayoutEntry {
                binding: 9,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: false },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
        ],
    })
}

fn terrain_shader() -> String {
    format!(
        r#"
// Group 0 -- the camera, the lights and both shadow maps -- is declared by
// `wgsl_lights_block` below, so there is one description of that layout rather
// than one per shader.

@group(1) @binding(0) var layer_tex: texture_2d_array<f32>;
@group(1) @binding(1) var layer_samp: sampler;
@group(1) @binding(2) var macro_tex: texture_2d<f32>;

struct Material {{
    repeat: vec4<f32>,
    slope_start_deg: f32,
    slope_end_deg: f32,
    height_start: f32,
    height_end: f32,
    macro_repeat: f32,
    macro_strength: f32,
    biplanar_start_deg: f32,
    use_splat: f32,
    normal_strength: f32,
    wet_line: f32,
    wet_band: f32,
    sea_line: f32,
    porosity: vec4<f32>,
    film: vec4<f32>,
    shape: vec4<f32>,
}}
@group(1) @binding(3) var<uniform> mat: Material;
@group(1) @binding(4) var splat_tex: texture_2d<f32>;
@group(1) @binding(5) var normal_tex: texture_2d_array<f32>;
@group(1) @binding(7) var rough_tex: texture_2d_array<f32>;
@group(1) @binding(8) var ao_tex: texture_2d_array<f32>;
// Baked sky visibility over the terrain footprint, in R. 1 = open sky, 0 =
// sealed. Sampled by the same normalised footprint uv the splat map uses, so it
// needs no new vertex attribute.
//
// WHY TERRAIN NEEDS ITS OWN
//
// Brushes carry sky visibility in their lightmap's alpha, but terrain has no
// lightmap and no chart layout to put one in -- it is a heightfield sampled by
// world position. Without this the ground took the full open-sky term
// everywhere, so the floor INSIDE a sealed room lit exactly as brightly as the
// field outside it, and no amount of shadow-map work could change that: the
// term being wrong was the ambient, not the direct light.
// Layer 0 the ground's own map; after it the stationary lamps' masks, two
// lamps a layer. See `TerrainMaterial::new`.
@group(1) @binding(6) var sky_occ_tex: texture_2d_array<f32>;

{lights_block}
{biplanar_block}
{whiteout_block}
{shape_block}

struct VIn  {{ @location(0) pos: vec3<f32>, @location(1) norm: vec3<f32>, @location(2) col: vec4<f32>, @location(3) uv2: vec2<f32> }}
struct VOut {{
    @builtin(position) clip: vec4<f32>,
    @location(0) col: vec4<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) world_pos: vec3<f32>,
    // The same point before the player-frame transform. Texture projection and
    // the height blend are properties of the GROUND, not of where the player is
    // standing, so they read this and never `world_pos`.
    @location(4) tex_pos: vec3<f32>,
    // Normalised position over the terrain footprint, carried in the slot the
    // cuboid path uses for lightmap coordinates. Terrain is lit dynamically and
    // has no lightmap, so that slot was sitting at (0,0) doing nothing.
    @location(3) uv: vec2<f32>,
    // The same uv where the pixel's coverage of its triangle is (CENTROID):
    // where the ground's step-1 shape is read (`ground_shape`). At an MSAA
    // edge pixel whose centre lies across the triangle's edge, the centre's
    // uv reads the NEIGHBOUR's normal and plane -- the crease moved to the
    // pixel grid, as jagged as no MSAA (2026-10-14, offline: lines of 20 to
    // 94 levels along every crease of the cliffs).
    @location(5) @interpolate(perspective, centroid) shape_uv: vec2<f32>,
}}

@vertex fn vs_main(v: VIn) -> VOut {{
    var out: VOut;
    out.clip      = cam_view_proj() * vec4<f32>(v.pos, 1.0);
    out.col       = v.col;
    out.normal    = v.norm;
    out.world_pos = v.pos;
    out.tex_pos   = to_world_space(v.pos);
    out.uv        = v.uv2;
    out.shape_uv  = v.uv2;
    return out;
}}

// One splat layer, sampled planar from above. Ground is mostly flat, so this is
// the path most pixels take.
fn sample_planar(layer: i32, world: vec3<f32>, repeat: f32) -> vec3<f32> {{
    let uv = world.xz / max(repeat, 0.001);
    return textureSample(layer_tex, layer_samp, uv, layer).rgb;
}}

// Biplanar: the two strongest axes, weighted by the normal. Two samples rather
// than triplanar's three -- the third axis contributes least by construction,
// and dropping it is invisible while being a third cheaper.
//
// The axis choice itself lives in `material_wgsl`, because the cave shader needs
// exactly the same projection and a second copy of it would be a second thing to
// get wrong. What stays here is the sampling, which differs.
fn sample_biplanar(layer: i32, world: vec3<f32>, n: vec3<f32>, repeat: f32) -> vec3<f32> {{
    let b = biplanar_axes(world, n, repeat);
    let c_major = textureSample(layer_tex, layer_samp, b.uv_major, layer).rgb;
    let c_minor = textureSample(layer_tex, layer_samp, b.uv_minor, layer).rgb;
    return mix(c_minor, c_major, b.w);
}}

// Surface normal for one layer, sampled top-down and folded into the geometric
// normal by the "whiteout" blend.
//
// PLANAR ONLY, deliberately, while colour goes biplanar on steep ground. Each
// extra projection is another texture fetch per layer per pixel, and the four
// layers are already sampled unconditionally -- going biplanar here would take
// terrain from twelve fetches to sixteen on a machine that is not fast. The
// cost is that a near-vertical face gets a stretched normal; that is far less
// objectionable than stretched COLOUR, which is why colour keeps its second
// projection and this does not.
//
// Whiteout rather than simply replacing the normal: the map describes bumps
// relative to the surface, so it has to perturb the geometry rather than
// overwrite it. Replacing would make every slope light as though it were flat
// ground, which is exactly the artefact normal maps exist to avoid.
fn layer_normal(layer: i32, world: vec3<f32>, n: vec3<f32>, repeat: f32) -> vec3<f32> {{
    let uv = world.xz / max(repeat, 0.001);
    let packed = textureSample(normal_tex, layer_samp, uv, layer).rgb;

    // OpenGL convention: green is +Y in tangent space. ambientCG ships both
    // NormalGL and NormalDX; the installer takes GL, and a DX map loaded here
    // would light every bump from the opposite side.
    var tn = packed * 2.0 - 1.0;
    tn = vec3<f32>(tn.xy * mat.normal_strength, tn.z);

    // Axis 1 is the y projection, which is the only one this samples. For flat
    // ground (n = 0,1,0) and a flat texel (tn = 0,0,1) it returns the geometric
    // normal exactly, which is what makes an unauthored normal set a no-op.
    return normalize(whiteout(1u, world_planar_bend(tn), n));
}}

// A planar normal map's bend into the PLAYER's frame. The map is laid out
// along the world's x and z -- its uv is the world point's xz -- and the
// whiteout adds it along the x and z of `n`'s frame, which is the player's:
// turned by every snap and stick turn, so every blade of grass caught the
// light differently after one (headset `outdoors_front-y90`, 2026-10-01).
// The bend in the world is (tn.x, 0, -tn.y) (`material_wgsl`: red toward +u,
// green toward -v); turned into the player's frame and handed back in the
// whiteout's terms. At no turn it is `tn` exactly.
fn world_planar_bend(tn: vec3<f32>) -> vec3<f32> {{
    let p = to_player_direction(vec3<f32>(tn.x, 0.0, -tn.y));
    return vec3<f32>(p.x, -p.z, tn.z);
}}

fn layer_colour(layer: i32, world: vec3<f32>, n: vec3<f32>, slope_deg: f32) -> vec3<f32> {{
    let repeat = mat.repeat[layer];
    if (slope_deg < mat.biplanar_start_deg) {{
        return sample_planar(layer, world, repeat);
    }}
    return sample_biplanar(layer, world, n, repeat);
}}

// Everything a layer needs to sample itself, computed ONCE per fragment.
//
// The point is the derivatives. `textureSample` picks its own mip level from
// implicit derivatives, which is why it may only be called in uniform control
// flow -- and that requirement is what forced all four layers to be sampled
// whether or not they contributed. `textureSampleGrad` takes the gradients as
// arguments and carries no such restriction, so a layer with no weight can be
// skipped entirely.
//
// The gradients are taken here, before any branch, and every projection scales
// as 1/repeat -- so a per-layer repeat is a divide, never another derivative.
//
// ONE PLANE HELD, NOT THREE (2026-10-13). The frame carried the top-down
// plane, the cliff's major plane and the detail maps' -- which is one or the
// other of those two -- each with its two gradients: 18 floats across the
// layer loop where a pixel only ever reads one of them (the top-down plane
// when it is not steep, the major plane when it is). `detail_*` is that
// plane; the colour reads it too. The same values in every read, and the
// reader that draws triangles with both kinds of pixel holds what the steep
// twin holds (`ground_twins`; `shader_opt::frame_was` is the frame as it was).
//
// TWO PLANES, EVERY MAP ON BOTH, ONE SHARE (2026-10-13, the headset's line
// artifacts on steep ground). A steep pixel is read on the plane its surface
// faces most (`detail_*`) and, only where a second plane faces it nearly as
// much, on that one too (`minor_*`), the first by `blend` and the second by
// the rest -- colour, normal, roughness and occlusion alike. Until then the
// colour alone was blended and the other three changed plane along a line,
// which drew the polygon the two normals' tie traces across a face as a
// patch of another rock (the pale V on the scarp); and the share was the
// plain ratio of the normal's two largest parts, which leaves a plane that
// stretches the picture three times a third of the say.
struct SampleFrame {{
    minor_uv: vec2<f32>,
    minor_ddx: vec2<f32>,
    minor_ddy: vec2<f32>,
    // The first plane's share of every map: 1 where it is read alone (all
    // gentle ground, and steep ground away from a turn between two planes).
    blend: f32,
    // The share of the SECOND SCALE in the first plane's colour and normal:
    // a side plane's alone, and none at a tie, where the two planes must be
    // read alike for the picture to be the same from either side of it.
    second: f32,
    // THE FIRST PLANE: top-down on gentle ground, as it always was; on steep
    // ground the one the face looks along. Read top-down, a cliff took one
    // texel row of each map down its whole height -- the vertical streaks a
    // near-vertical face showed (2026-10-08, the headland).
    detail_uv: vec2<f32>,
    detail_ddx: vec2<f32>,
    detail_ddy: vec2<f32>,
    detail_axis: u32,
    // The macro variation, read once here (it steers the warp below).
    macro_value: f32,
    // How the top plane's bumps are folded into the normal (`fold_bumps`):
    // 1 where the triangle drawn is the surface the interpolated normal
    // describes, 0 where the planes were chosen by the triangle instead.
    fold: f32,
    minor_axis: u32,
}}

// How far, in metres, the macro variation shifts the side planes' uvs on a
// cliff: enough that one tile of rock never lines up with the next the same
// way, so a long face stops showing its normal map's repeat as rows of the
// same chevron (2026-10-08). Low frequency, so no texel is stretched.
const CLIFF_WARP_M: f32 = 2.5;
// The band warp: how far a face's stripes are bent, and over what distance
// the bend changes (the macro map at this repeat).
const CLIFF_BAND_WARP_M: f32 = 3.0;
const CLIFF_BAND_SCALE_M: f32 = 7.0;
// Whether the level's material carries a macro map (`sample_frame`).
const GROUND_MACRO_MAP: bool = false;

// A plane has a share of a steep pixel only where the surface faces it more
// than this: 1/sqrt(3), what all three face at once. So a plane is never read
// where it would stretch the picture more than 1.73 times, and the plane left
// out -- the third -- has no share to lose where it swaps with the second.
const PLANE_TIE: f32 = 0.57735;
// The first plane's share past which the second is not read at all.
const PLANE_ALONE: f32 = 0.999;
// How far the triangle drawn may turn from the interpolated normal before the
// planes are chosen by the triangle instead: on any smooth ground the two
// agree (their dot) within 6 degrees, and by 14 the triangle alone decides.
// `FACE_BAND` is one over the width between.
const FACE_DIFFERS: f32 = 0.9703;
const FACE_BAND: f32 = 41.3;
// What the macro variation reads where there is no macro map: the middle,
// which changes nothing.
const MACRO_NEUTRAL: f32 = 0.5019608;

// `n` is the interpolated normal, and `step_x` and `step_y` the position's
// two screen steps -- all three in the player's frame, where the steps stay
// small near the eye in a level of any size.
// `shape_face`: the step-1 triangle's plane under the pixel, which stands for
// the drawn one where `shaped` (`ground_shape`).
fn sample_frame(world: vec3<f32>, n: vec3<f32>, step_x: vec3<f32>, step_y: vec3<f32>, shape_face: vec3<f32>, shaped: bool) -> SampleFrame {{
    var f: SampleFrame;
    let planar_uv  = world.xz;
    // THE WORLD POSITION'S TWO SCREEN GRADIENTS, taken once: every plane's
    // are two of their components. Never the derivative of a uv a pixel
    // CHOSE: across the line where the choice changes that is the difference
    // of two unrelated coordinates, read as a gradient of metres a pixel, and
    // every map drops to its smallest mip along it -- the one-pixel pale
    // lines down a cliff (headset, 2026-10-13).
    let world_ddx = dpdx(world);
    let world_ddy = dpdy(world);
    let planar_ddx = world_ddx.xz;
    let planar_ddy = world_ddy.xz;

    // THE DRAWN TRIANGLE'S OWN NORMAL, on `n`'s side: both steps lie in the
    // triangle's plane, so their cross is its normal exactly, at a triangle's
    // edge too. (A triangle seen so edge-on that the steps have no area
    // between them has none, and the interpolated normal stands in below.)
    let across = cross(step_x, step_y);
    let along = dot(across, n);
    let across_len = inverseSqrt(max(dot(across, across), 1e-30));
    let face = select(across * (sign(along) * across_len), shape_face, shaped);
    // STEEP WHERE THE NORMAL OR THE TRIANGLE IS. The interpolated normal is
    // the light's; it is a blend of the faces round each vertex, and on a rise
    // one cell wide it reads twenty degrees gentler than the face it is drawn
    // on, and straight up along a knife ridge's crest, where the faces either
    // side cancel. Asked alone, a ridge's top was "gentle" and took the
    // top-down plane on seventy-degree faces: the picture pulled down each
    // side's fall line, the herringbone along the crest.
    let steep = min(n.y, face.y) <= cos(radians(mat.biplanar_start_deg));
    // THE PLANES ARE CHOSEN BY THE SURFACE THAT IS DRAWN: the interpolated
    // normal where the triangle agrees with it -- so on rounded ground the
    // shares turn smoothly, with no facet -- and the triangle's own where it
    // does not: at a cliff's foot and rim, on a one-cell step, along a ridge.
    // There the ground has a crease, and the picture changes plane on it.
    // In the world, as the planes are.
    let agree = clamp((abs(dot(face, n)) - FACE_DIFFERS) * FACE_BAND, 0.0001, 1.0);
    let pn = to_world_direction(normalize(mix(face, n, agree)));
    // At unit repeat, so each layer divides rather than recomputing.
    let b = biplanar_axes(world, pn, 1.0);
    let a = abs(pn);
    let ma = max(a.x, max(a.y, a.z));
    let me = a.x + a.y + a.z - ma - min(a.x, min(a.y, a.z));
    // Each plane's share is how far the surface faces it PAST the three-way
    // tie: the second plane's is nothing until then, and equal to the
    // first's where the two tie.
    let lead = ma - PLANE_TIE;
    let share = (lead + 0.0001) / (lead + max(me - PLANE_TIE, 0.0) + 0.0001);
    let major_ddx = select(select(world_ddx.xy, world_ddx.xz, b.axis_major == 1u), world_ddx.zy, b.axis_major == 0u);
    let major_ddy = select(select(world_ddy.xy, world_ddy.xz, b.axis_major == 1u), world_ddy.zy, b.axis_major == 0u);
    f.minor_uv  = b.uv_minor;
    f.minor_ddx = select(select(world_ddx.xy, world_ddx.xz, b.axis_minor == 1u), world_ddx.zy, b.axis_minor == 0u);
    f.minor_ddy = select(select(world_ddy.xy, world_ddy.xz, b.axis_minor == 1u), world_ddy.zy, b.axis_minor == 0u);
    f.blend     = select(1.0, share, steep);
    f.detail_uv   = select(planar_uv, b.uv_major, steep);
    f.detail_ddx  = select(planar_ddx, major_ddx, steep);
    f.detail_ddy  = select(planar_ddy, major_ddy, steep);
    f.detail_axis = select(1u, b.axis_major, steep);
    f.minor_axis  = b.axis_minor;
    f.fold        = select(1.0, agree, steep);
    // THE MACRO VARIATION, where a level has a macro map. None has: every
    // material is built with the one neutral texel (`TerrainMaterial::
    // from_layers_with`), and every ground pixel paid a fetch here -- a cliff
    // a second below -- to read the middle. `GROUND_MACRO_MAP` is the build
    // with the reads in it, for the day a level brings a map.
    f.macro_value = MACRO_NEUTRAL;
    if (GROUND_MACRO_MAP) {{
        let mr = max(mat.macro_repeat, 0.001);
        f.macro_value = textureSampleGrad(macro_tex, layer_samp, f.detail_uv / mr, f.detail_ddx / mr, f.detail_ddy / mr).r;
    }}
    // The second scale: on a side plane, by how far it leads (`SampleFrame`).
    f.second = select(0.0, (0.3 + 0.3 * f.macro_value) * clamp(4.0 * f.blend - 2.0, 0.0, 1.0), f.detail_axis != 1u);
    var warp = select(vec2<f32>(0.0), vec2<f32>(f.macro_value - 0.5, 0.4 - 0.8 * f.macro_value) * CLIFF_WARP_M, steep);
    // THE BANDS. The rock map is a scan of bedded rock, so it carries level
    // stripes, and tiled every 2 m up a 13 m face they read as strata laid
    // at an even pitch -- the "made" look from the beach (2026-10-08). The
    // macro map read again at a few metres' scale bends the face's v by up
    // to a metre and a half, so the stripes wander, thicken and thin along
    // the face instead of running level at one pitch. Steep faces only, so
    // its fetch is skipped on the ground.
    if (steep) {{
        let ms = 1.0 / CLIFF_BAND_SCALE_M;
        var m2 = MACRO_NEUTRAL;
        if (GROUND_MACRO_MAP) {{
            m2 = textureSampleGrad(macro_tex, layer_samp, f.detail_uv * ms + vec2<f32>(0.31, 0.57), f.detail_ddx * ms, f.detail_ddy * ms).r;
        }}
        warp = warp + vec2<f32>(0.35, 1.0) * (m2 - 0.5) * CLIFF_BAND_WARP_M;
    }}
    // The side planes' alone: the top plane is read where gentle ground
    // reads it, so nothing moves where steep ground begins.
    f.minor_uv  = f.minor_uv + select(warp, vec2<f32>(0.0), f.minor_axis == 1u);
    f.detail_uv = f.detail_uv + select(warp, vec2<f32>(0.0), f.detail_axis == 1u);
    return f;
}}

fn layer_colour_at(layer: i32, f: SampleFrame) -> vec3<f32> {{
    let r = max(mat.repeat[layer], 0.001);
    // The first plane (`SampleFrame`): top-down, and the whole colour, on
    // gentle ground; on steep ground the plane the face looks along.
    var c = textureSampleGrad(layer_tex, layer_samp, f.detail_uv / r, layer, f.detail_ddx / r, f.detail_ddy / r).rgb;
    // TWO SCALES ON THE FACE THE CLIFF LOOKS ALONG, as for the normal: the
    // rock scan's level stripes, tiled every 2 m, read as strata at one even
    // pitch from the beach. A second read 2.3x larger and tilted 37 degrees
    // puts a second, unrelated pitch over the first, so no period shows.
    if (f.second > 0.0) {{
        let rot = mat2x2<f32>(0.8, 0.6, -0.6, 0.8);
        let s2 = 1.0 / (r * 2.3);
        c = mix(
            c,
            textureSampleGrad(
                layer_tex, layer_samp, rot * f.detail_uv * s2 + vec2<f32>(0.37, 0.11), layer,
                rot * f.detail_ddx * s2, rot * f.detail_ddy * s2,
            ).rgb,
            f.second,
        );
    }}
    // The second plane, where it has a share.
    if (f.blend < PLANE_ALONE) {{
        c = mix(
            textureSampleGrad(layer_tex, layer_samp, f.minor_uv / r, layer, f.minor_ddx / r, f.minor_ddy / r).rgb,
            c,
            f.blend,
        );
    }}
    return c;
}}

// Roughness and occlusion for one layer, sampled with the same frame and
// gradients the normal uses so all four maps agree about which mip they are on.
//
// On the planes the colour is read on, by the same share: read on the first
// alone they changed plane along the line where two planes tie, and a patch
// of ground shone and shaded as another rock (`SampleFrame`). The second
// plane's fetch is taken only where it has a share.
fn layer_rough_at(layer: i32, f: SampleFrame) -> f32 {{
    let r = max(mat.repeat[layer], 0.001);
    var v = textureSampleGrad(
        rough_tex, layer_samp, f.detail_uv / r, layer, f.detail_ddx / r, f.detail_ddy / r,
    ).r;
    if (f.blend < PLANE_ALONE) {{
        v = mix(textureSampleGrad(rough_tex, layer_samp, f.minor_uv / r, layer, f.minor_ddx / r, f.minor_ddy / r).r, v, f.blend);
    }}
    return v;
}}

fn layer_ao_at(layer: i32, f: SampleFrame) -> f32 {{
    let r = max(mat.repeat[layer], 0.001);
    var v = textureSampleGrad(
        ao_tex, layer_samp, f.detail_uv / r, layer, f.detail_ddx / r, f.detail_ddy / r,
    ).r;
    if (f.blend < PLANE_ALONE) {{
        v = mix(textureSampleGrad(ao_tex, layer_samp, f.minor_uv / r, layer, f.minor_ddx / r, f.minor_ddy / r).r, v, f.blend);
    }}
    return v;
}}

// ONE PLANE'S BUMPS FOLDED INTO THE SURFACE NORMAL, without leaning it.
//
// `whiteout` shortens only the normal's part along the plane's axis, by the
// bump's z, and leaves the rest whole: the result leans away from the axis.
// The grass map's bumps average z 0.63 (rock's 0.97), so on the top plane a
// slope's mean normal was shaded half as steep again as it is, and on a side
// plane as much too upright. Where the two planes met on one normal -- a
// patch of the face chosen by the other plane, the level triangle at a bank's
// foot beside the bank's own -- one side shaded dark and the other pale: the
// pale V on the grass scarp (and, by their shape, the dark triangles in the
// beach hill's shots), a straight line along the foot of every rise
// (headset, 2026-10-13). It was never the
// highlight: with the specular and the reflections off, and at roughness 1,
// the V stayed.
//
// So a SIDE plane shortens the whole normal alike, and its bumps leave the
// direction where it was. The TOP plane keeps the fold every gentle pixel has
// always had -- level ground, where the two agree exactly, is most of every
// level and shades as before -- wherever the triangle drawn is the surface
// the interpolated normal describes (`keep` 1), and folds as a side plane
// does where the planes were chosen by the triangle (`keep` 0): at a bank's
// foot and rim, where the normal leans forty degrees over level ground.
fn fold_bumps(axis: u32, tn: vec3<f32>, n: vec3<f32>, keep: f32) -> vec3<f32> {{
    let z = abs(tn.z);
    if (axis == 1u) {{
        let g = mix(z, 1.0, keep);
        return vec3<f32>(tn.x + g * n.x, z * n.y, g * n.z - tn.y);
    }}
    return n * z + select(vec3<f32>(tn.x, -tn.y, 0.0), vec3<f32>(0.0, -tn.y, tn.x), axis == 0u);
}}

fn layer_normal_at(layer: i32, n: vec3<f32>, f: SampleFrame) -> vec3<f32> {{
    let r = max(mat.repeat[layer], 0.001);
    let packed = textureSampleGrad(
        normal_tex, layer_samp, f.detail_uv / r, layer, f.detail_ddx / r, f.detail_ddy / r,
    ).rgb;

    // Identical to `layer_normal` above, which stays for the planar path used
    // elsewhere: OpenGL green-up convention, strength applied to the tangent
    // axes only, folded into the geometry by the whiteout blend rather than
    // replacing it.
    //
    // RETURNED UNNORMALISED, AND THAT IS THE WHOLE CHANGE.
    //
    // Mipping a normal map averages its normals, and where they disagreed the
    // average is SHORT. That shortness is a measurement of how much detail the
    // mip threw away -- and `normalize()` discards exactly that measurement
    // while restoring full strength, so a distant bumpy surface keeps shading
    // as though every bump were still resolvable. In a headset, where head
    // tracking moves the sampling point every frame, that is shimmer.
    //
    // The caller renormalises for the shading normal and reads the length as
    // variance. Nothing is lost by handing both back.
    var tn = packed * 2.0 - 1.0;
    tn = vec3<f32>(tn.xy * mat.normal_strength, tn.z);
    // The top plane read alone -- all gentle ground, and steep ground whose
    // face still looks up -- keeps the bend it always had, so gentle ground
    // shades exactly as before.
    if (f.detail_axis == 1u && f.blend >= PLANE_ALONE) {{
        return fold_bumps(1u, world_planar_bend(tn), n, f.fold);
    }}
    // TWO SCALES ON A CLIFF. One normal map tiled down a long face showed
    // its repeat as rows of the same chevron. A second read, 2.3x larger
    // and turned 37 degrees, is mixed in: the two periods never line up,
    // so the sum does not repeat within any face we have. Its bumps are
    // turned back into the first read's axes before the mix, and the mix
    // (which averages two unrelated patterns, so comes out short) is
    // given back its strength. A side plane's alone, by `f.second`.
    if (f.second > 0.0) {{
        let rot = mat2x2<f32>(0.8, 0.6, -0.6, 0.8);
        let s2 = 1.0 / (r * 2.3);
        let p2 = textureSampleGrad(
            normal_tex, layer_samp, rot * f.detail_uv * s2 + vec2<f32>(0.37, 0.11), layer,
            rot * f.detail_ddx * s2, rot * f.detail_ddy * s2,
        ).rgb;
        var t2 = p2 * 2.0 - 1.0;
        t2 = vec3<f32>(transpose(rot) * (t2.xy * mat.normal_strength), t2.z);
        tn = vec3<f32>(mix(tn.xy, t2.xy, f.second) * (1.0 + 0.667 * f.second), mix(tn.z, t2.z, f.second));
    }}
    // Each plane's bumps are folded in where its axes are -- the world's --
    // and the sum turned back into the player's frame.
    let nw = to_world_direction(n);
    var bent = fold_bumps(f.detail_axis, tn, nw, f.fold);
    // THE SECOND PLANE'S, by its share, as the colour's: so the bumps do
    // not change direction along a line where a face turns from one plane
    // to the next. Its fetch is taken only where it has a share.
    if (f.blend < PLANE_ALONE) {{
        let pm = textureSampleGrad(
            normal_tex, layer_samp, f.minor_uv / r, layer, f.minor_ddx / r, f.minor_ddy / r,
        ).rgb;
        var tm = pm * 2.0 - 1.0;
        tm = vec3<f32>(tm.xy * mat.normal_strength, tm.z);
        bent = mix(fold_bumps(f.minor_axis, tm, nw, f.fold), bent, f.blend);
    }}
    return to_player_direction(bent);
}}


// Below this a layer changes the result by less than one 8-bit step, so
// sampling it buys nothing but bandwidth. Not renormalised afterwards:
// rescaling the surviving weights would move the shading further than the
// omission does.
const WEIGHT_EPS: f32 = 0.004;
// Metres over which the terrain's normal maps fade out past the detail
// distance, so no line marks where they stop.
const TERRAIN_DETAIL_FADE: f32 = 8.0;

// Blend weights for the four layers, authored or derived.
//
// One vec4 either way, so the fragment stage below has a single code path. The
// procedural branch reproduces the old mix chain exactly -- a weighted sum and
// a chain of mixes are the same arithmetic -- so turning authoring on and off
// is a change of WEIGHTS and never a change of shading model.
fn layer_weights(uv: vec2<f32>, world_y: f32, slope_deg: f32) -> vec4<f32> {{
    // Authored. Normalised by its own sum rather than trusted: the editor keeps
    // the four bytes summing to 255, but a hand-made or half-written file has
    // no such guarantee and unnormalised weights would blow out or black out
    // the ground rather than looking slightly wrong.
    let authored_raw = textureSample(splat_tex, layer_samp, uv);
    let total = authored_raw.r + authored_raw.g + authored_raw.b + authored_raw.a;
    let authored = authored_raw / max(total, 0.001);

    // Derived from slope, with the optional height band on top.
    let rock_w = smoothstep(mat.slope_start_deg, mat.slope_end_deg, slope_deg);
    var derived = vec4<f32>(1.0 - rock_w, rock_w, 0.0, 0.0);
    if (mat.height_end > mat.height_start) {{
        let high_w = smoothstep(mat.height_start, mat.height_end, world_y);
        derived = vec4<f32>(derived.x * (1.0 - high_w), derived.y * (1.0 - high_w), high_w, 0.0);
    }}

    // select, not an if: both sides are already computed and branching here
    // would put textureSample under non-uniform control flow.
    return select(derived, authored, mat.use_splat > 0.5);
}}

@fragment fn fs_main(in: VOut) -> @location(0) vec4<f32> {{
    // See `pixel_footprint` in the lights block: taken here, in uniform
    // control flow, so the light loop can keep a spot's edge a pixel wide.
    //
    // The GEOMETRIC MEAN of the two screen axes, not their sum: seen at a
    // grazing angle one axis stretches to metres while the other stays a
    // pixel, and the sum let that stretch widen a spot's pool across the floor.
    let step_x = dpdx(in.world_pos);
    let step_y = dpdy(in.world_pos);
    pixel_footprint = sqrt(length(step_x) * length(step_y));
    // THE STEP-1 GROUND UNDER THE PIXEL (`ground_shape`), at whatever step
    // its triangle is drawn: the normal the step-1 triangle there blends,
    // and that triangle's own plane -- so a coarser triangle of
    // `ground_lod` is lit and textured as step 1 is. The interpolated
    // normal and the drawn triangle where there is no shape, and on a
    // cave's ground patch (its vertices' alpha 0).
    let shape = ground_shape_at(in.shape_uv, in.col.a);
    let n = normalize(select(in.normal, to_player_direction(shape.n), shape.shaped));
    set_pixel_long_step(n, normalize(cam_pos() - in.world_pos));

    // Slope straight from the normal: no derivative, no extra sampling.
    let slope_deg = degrees(acos(clamp(n.y, -1.0, 1.0)));

    let w = layer_weights(in.uv, in.tex_pos.y, slope_deg);

    // Only the layers that contribute. Terrain is fill-bound, not geometry
    // bound -- quartering the triangle count moved the frame time by nothing,
    // while removing the ground entirely gave back 13 ms of a 13.9 ms budget --
    // and most fragments are one or two layers, not four.
    // The projections are chosen by the WORLD normal, as their uvs are world
    // axes: picked by the player-frame one, steep ground swapped its x and z
    // projections at every quarter turn (headset `outdoors_front-y90`,
    // 2026-10-01, the far bank).
    // And by the triangle that is drawn, where it is not the surface the
    // interpolated normal describes (`sample_frame`).
    let f = sample_frame(in.tex_pos, n, step_x, step_y, to_player_direction(shape.face), shape.shaped);

    // Blend the layers' normals by the same weights, then renormalise. Summing
    // unit vectors shortens the result wherever they disagree, and a shortened
    // normal darkens the surface -- so the renormalise is load-bearing, not
    // tidiness.
    //
    // FADED OUT WITH DISTANCE when `post_params.z` says where: past it a
    // layer's normal map only shakes a pixel's normal around an average the
    // vertex normal already is, and what it would have added -- a wider
    // highlight -- is baked into the roughness mips read below. So the far
    // field skips the samples and the blend, and loses nothing it could show.
    let detail_far = camera.post_params.z;
    let detail = select(1.0, 1.0 - smoothstep(detail_far, detail_far + TERRAIN_DETAIL_FADE, distance(in.world_pos, cam_pos())), detail_far > 0.0);

    // HOW MUCH DETAIL WAS AVERAGED AWAY, read before renormalising.
    //
    // Two things shortened this vector and both mean the same thing: the mip
    // chain averaging disagreeing normals inside one layer, and the weighted
    // blend of layers that disagree with each other. Either way a short result
    // says "the surface under this pixel is rougher than any single normal
    // here admits", and a wider highlight is the honest response.
    //
    // sigma2 = (1 - len)/len is the Toksvig variance; the factor scales the
    // specular exponent. Clamped at both ends: len can exceed 1 slightly where
    // normal_strength pushes past unit length, and a degenerate near-zero
    // blend must not divide by ~0.
    // ROUGHNESS AND OCCLUSION FROM THE MAPS, blended by the same weights the
    // colour and normal use so a splat boundary moves all four together.
    //
    // The normal's lost variation is ALREADY IN these mips -- baked in by
    // `roughness_chain_with_normal_variance` at load, per level, exactly as
    // brush materials get it. Adding a runtime term for it here as well would
    // count the same variance twice and over-roughen distant ground.
{layer_reads}
    shaded_n = normalize(select(n, shaded_n, length(shaded_n) > 0.0001));

    // What the MIPS could not see: the geometric normal's own variation across
    // this pixel, and the grazing-angle case where anisotropic filtering
    // fetches a sharper roughness mip than the isotropic level. Taken here, at
    // the top level of the entry point, because a derivative is only legal in
    // uniform control flow.
    let rough = specular_aa_roughness(clamp(rough_map, 0.0, 1.0), dpdx(shaded_n), dpdy(shaded_n));

    // Macro variation: one low-frequency sample, centred on 1 so it darkens and
    // lightens rather than only darkening.
    // Along the face on a cliff, as the detail maps are: read in `sample_frame`.
    let m = f.macro_value;
    albedo = albedo * (1.0 + (m - 0.5) * 2.0 * mat.macro_strength);
{sea_line}
    // Vertex colour survives as a tint, so the editor can still mark up ground
    // per-vertex without a second pipeline.
    // The ambient term is scaled by baked sky visibility BEFORE the lights are
    // added, which is what `shade_with_sky` does and `shade` cannot.
    let ground_map = textureSample(sky_occ_tex, layer_samp, in.uv, 0);
    let sky_vis = ground_map.r;
    // THE SKY SUN'S SHADOW, baked beside the sky visibility when alpha is 0:
    // a signed distance to the edge in texels (green) and the sun's penumbra
    // (blue), rebuilt exactly as the brushes rebuild theirs. The live static
    // map drew the walls' shadows on the grass with blocky edges (headset,
    // 2026-09-25). An older map is greyscale with alpha 1 and leaves the
    // ground on that live map, as before.
    let ground_sun_d = (ground_map.g - 0.5) * (2.0 * {sun_range:?});
    let ground_sun_w = max(max(ground_map.b * {sun_range:?}, 0.5 * fwidth(ground_sun_d)), 0.02);
    receiver_sun_mask = select(-1.0, smoothstep(-ground_sun_w, ground_sun_w, ground_sun_d), ground_map.a < 0.5);
    // THE STATIONARY LAMPS' SHADOWS ON THE GROUND, in the layers after the
    // ground map, in the brushes' format and channels. Without them every
    // lamp lit the grass straight through the walls -- the hallway's sconces
    // lit the ground outside the hallway (headset, 2026-09-24). A map with no
    // mask layers reads fully lit, as the ground always was.
    let st_layers = textureNumLayers(sky_occ_tex) - 1u;
    var st_0 = vec4<f32>(1.0);
    var st_1 = vec4<f32>(1.0);
    var st_2 = vec4<f32>(1.0);
    var st_3 = vec4<f32>(1.0);
    if (st_layers > 0u) {{
        st_0 = textureSample(sky_occ_tex, layer_samp, in.uv, 1);
    }}
    if (st_layers > 1u) {{
        st_1 = textureSample(sky_occ_tex, layer_samp, in.uv, 2);
    }}
    if (st_layers > 2u) {{
        st_2 = textureSample(sky_occ_tex, layer_samp, in.uv, 3);
    }}
    if (st_layers > 3u) {{
        st_3 = textureSample(sky_occ_tex, layer_samp, in.uv, 4);
    }}
    set_stationary_masks(st_0, st_1, st_2, st_3, {stationary_range:?});
    // THE SAME SHADING PATH THE BRUSHES USE, and the albedo goes IN rather
    // than being multiplied over the result.
    //
    // Terrain used to call `shade_with_sky` and multiply its albedo over the
    // answer. That is the exact bug this renderer already fixed for brushes:
    // a dielectric's highlight is not tinted by its diffuse colour, and
    // multiplying a 0.3-albedo ground over the lit result made every highlight
    // on it about three times too dim. Ground was a second-class material with
    // no Fresnel, no energy conservation, no probe reflection and no specular
    // occlusion, and it looked it next to marble.
    //
    // WHAT IS STILL MISSING, deliberately, until terrain ships the maps
    // brushes do: `ao` is 1.0 and the baked bounce is zero, because terrain has
    // neither an AO map nor a directional bounce bake -- it carries only a
    // scalar sky-occlusion, which is passed as `sky_vis` exactly as before.
    // Roughness is a constant widened by the normal's own variance rather than
    // a map. Each of those is a texture array away from parity.
    let lit = shade_material_env(
        in.world_pos,
        shaded_n,
        rough,
        clamp(ao_map, 0.0, 1.0),
        sky_vis,
        vec3<f32>(0.0),
        vec4<f32>(0.5, 0.5, 0.5, 0.0),
        albedo,
        // No face to stand on: ground is one continuous surface, so the
        // fragment's own position is the honest point to choose a probe from.
        in.world_pos,
        // The INTERPOLATED surface normal, before the layer normal maps
        // perturbed it -- the ground's own shape, which is what Fresnel wants.
        n,
    );
    return vec4<f32>(tonemap(in.col.rgb * lit), 1.0);
}}
"#,
        lights_block = terrain_lights_block(crate::renderer::lights::LightsBlockOptions::default()),
        sea_line = SEA_LINE_WGSL,
        sun_range = super::brush_pipeline::SUN_MASK_DISTANCE_TEXELS,
        stationary_range = super::brush_pipeline::STATIONARY_MASK_DISTANCE_TEXELS,
        biplanar_block = wgsl_biplanar_block(),
        whiteout_block = wgsl_whiteout_block(),
        shape_block = super::ground_shape::WGSL,
        layer_reads = LAYER_READS,
    )
}

/// THE GROUND'S LAYER READS -- each contributing layer's colour, normal,
/// roughness and occlusion, summed by its weight -- as `fs_main` takes them,
/// after the weights `w`, the sampling frame `f` and the normals' fade
/// `detail`.
///
/// ONE COPY OF EACH READ, walked over the layers that contribute. They were
/// twelve inlined copies -- colour, normal, and roughness with occlusion, for
/// each of the four layers ([`LAYER_READS_INLINED`]) -- and the scene's ground
/// reader came to 4,292 instructions, past what the Quest's instruction cache
/// holds (PIPESTATS, 2026-10-06; the brushes' readers draw a third slower past
/// about 3,390). The same reads in the same order: the mask is walked lowest
/// bit first, so every sum still adds layer 0 first and layer 3 last. Walked as
/// a mask rather than counted, as the lamps are (`surface_lights`), and each
/// weight picked by `select` rather than by indexing `w` at a runtime index,
/// which Adreno can keep in scratch memory.
const LAYER_READS: &str = "    var albedo = vec3<f32>(0.0);
    var shaded_n = vec3<f32>(0.0);
    var rough_map = 0.0;
    var ao_map = 0.0;
    var layers_left = select(0u, 1u, w.x > WEIGHT_EPS) | select(0u, 2u, w.y > WEIGHT_EPS)
        | select(0u, 4u, w.z > WEIGHT_EPS) | select(0u, 8u, w.w > WEIGHT_EPS);
    loop {
        if (layers_left == 0u) {
            break;
        }
        let layer = i32(countTrailingZeros(layers_left));
        layers_left = layers_left & (layers_left - 1u);
        let lw = select(select(select(w.x, w.y, layer == 1), w.z, layer == 2), w.w, layer == 3);
        albedo = albedo + layer_colour_at(layer, f) * lw;
        if (detail > 0.0) {
            shaded_n = shaded_n + layer_normal_at(layer, n, f) * lw;
        }
        rough_map = rough_map + layer_rough_at(layer, f) * lw;
        ao_map = ao_map + layer_ao_at(layer, f) * lw;
    }
    if (detail > 0.0) {
        shaded_n = mix(n, shaded_n, detail);
    }
";

/// MEASUREMENT: [`LAYER_READS`] as they were before the loop, each read
/// inlined for each layer, to draw against it in one build
/// (`Levers::terrain_reader`, `inlined`). The same picture.
const LAYER_READS_INLINED: &str = "    var albedo = vec3<f32>(0.0);
    if (w.x > WEIGHT_EPS) { albedo = albedo + layer_colour_at(0, f) * w.x; }
    if (w.y > WEIGHT_EPS) { albedo = albedo + layer_colour_at(1, f) * w.y; }
    if (w.z > WEIGHT_EPS) { albedo = albedo + layer_colour_at(2, f) * w.z; }
    if (w.w > WEIGHT_EPS) { albedo = albedo + layer_colour_at(3, f) * w.w; }
    var shaded_n = vec3<f32>(0.0);
    if (detail > 0.0) {
        if (w.x > WEIGHT_EPS) { shaded_n = shaded_n + layer_normal_at(0, n, f) * w.x; }
        if (w.y > WEIGHT_EPS) { shaded_n = shaded_n + layer_normal_at(1, n, f) * w.y; }
        if (w.z > WEIGHT_EPS) { shaded_n = shaded_n + layer_normal_at(2, n, f) * w.z; }
        if (w.w > WEIGHT_EPS) { shaded_n = shaded_n + layer_normal_at(3, n, f) * w.w; }
        shaded_n = mix(n, shaded_n, detail);
    }
    var rough_map = 0.0;
    var ao_map = 0.0;
    if (w.x > WEIGHT_EPS) { rough_map = rough_map + layer_rough_at(0, f) * w.x; ao_map = ao_map + layer_ao_at(0, f) * w.x; }
    if (w.y > WEIGHT_EPS) { rough_map = rough_map + layer_rough_at(1, f) * w.y; ao_map = ao_map + layer_ao_at(1, f) * w.y; }
    if (w.z > WEIGHT_EPS) { rough_map = rough_map + layer_rough_at(2, f) * w.z; ao_map = ao_map + layer_ao_at(2, f) * w.z; }
    if (w.w > WEIGHT_EPS) { rough_map = rough_map + layer_rough_at(3, f) * w.w; ao_map = ao_map + layer_ao_at(3, f) * w.w; }
";

/// MEASUREMENT: a ground shader's `src` with its layer reads inlined per
/// layer, as they were before the loop. See [`LAYER_READS_INLINED`].
pub fn with_inlined_layer_reads(src: String) -> String {
    assert_eq!(src.matches(LAYER_READS).count(), 1, "the ground shader's layer reads moved");
    src.replacen(LAYER_READS, LAYER_READS_INLINED, 1)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::renderer::cuboid::SolidVertex;
    use crate::renderer::lights::LightsUniform;
    use wgpu::util::DeviceExt;

    /// The pipeline is only validated when it is CREATED, and correctness only
    /// when something is drawn. A test that builds a pipeline proves the WGSL
    /// parses; it says nothing about whether the shading responds to slope.
    /// So this renders and reads the pixels back.
    pub fn headless_gpu() -> Option<(Device, Queue)> {
        let instance = Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&RequestAdapterOptions {
            apply_limit_buckets: false,
            power_preference: PowerPreference::default(),
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok()?;
        pollster::block_on(adapter.request_device(&DeviceDescriptor {
            required_features: Features::empty(),
            required_limits: crate::renderer::uniforms::scene_limits(Limits::default()),
            ..Default::default()
        }))
        .ok()
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

    /// Renders one full-screen quad with the given normal and returns its
    /// centre pixel.
    /// Which layer colours the material carries. The test palette is four
    /// primaries so a readback names the winning layer unambiguously; the
    /// fallback palette is the shipping one, whose colours are deliberately
    /// close together and cannot.
    #[derive(Copy, Clone)]
    pub enum Palette {
        Test,
        Fallback,
        /// `Test` with layer 0 a ramp across its width (red 4 per texel of
        /// 64), so WHERE it is sampled shows -- which projection, which world
        /// point -- and not only whether.
        Ramp,
    }

    pub fn render_quad_with_normal(normal: [f32; 3]) -> Option<[u8; 4]> {
        render_quad(normal, Palette::Test, None, None)
    }

    /// Same geometry, with authored weights bound.
    pub fn render_quad_with_splat(normal: [f32; 3], splat: &TerrainImage) -> Option<[u8; 4]> {
        render_quad(normal, Palette::Test, Some(splat), None)
    }

    /// Same geometry, but bound through the shipping `TerrainMaterial::fallback`
    /// rather than a bind group assembled here. Worth its own path because the
    /// hand-built test material proves the SHADER and proves nothing about the
    /// code the renderer will actually call -- layer padding, the D2Array view
    /// dimension and the non-sRGB macro format all live in `TerrainMaterial`
    /// and are exactly where a binding mistake would hide.
    pub fn render_quad_with_fallback_material(normal: [f32; 3]) -> Option<[u8; 4]> {
        render_quad(normal, Palette::Fallback, None, None)
    }

    fn render_quad(
        normal: [f32; 3],
        palette: Palette,
        splat: Option<&TerrainImage>,
        sky_occlusion: Option<&TerrainImage>,
    ) -> Option<[u8; 4]> {
        render_quad_full(normal, palette, splat, sky_occlusion, &[None, None, None, None], false)
    }

    /// Full harness: optional per-layer normal maps and an optional point light.
    ///
    /// The light matters. With an empty light list `shade` returns the ambient
    /// constant regardless of the surface normal, so a normal-map test on the
    /// default harness would pass whether the perturbation worked or not.
    pub fn render_quad_full(
        normal: [f32; 3],
        palette: Palette,
        splat: Option<&TerrainImage>,
        sky_occlusion: Option<&TerrainImage>,
        normals: &[Option<TerrainImage>],
        lit: bool,
    ) -> Option<[u8; 4]> {
        render_quad_lit_by(normal, palette, splat, sky_occlusion, normals, lit.then_some(None))
    }

    /// `render_quad_full`'s light, as a STATIONARY lamp reading mask channel
    /// `channel` when `Some(Some(channel))`.
    pub fn render_quad_lit_by(
        normal: [f32; 3],
        palette: Palette,
        splat: Option<&TerrainImage>,
        sky_occlusion: Option<&TerrainImage>,
        normals: &[Option<TerrainImage>],
        light: Option<Option<u8>>,
    ) -> Option<[u8; 4]> {
        render_quad_light_at(normal, palette, splat, sky_occlusion, normals, light, glam::Vec3::new(3.0, 2.0, 0.0))
    }

    /// `render_quad_lit_by` with the lamp at `light_pos`. The default sits
    /// level with the quad in z, so it cannot tell a tilt toward +z from one
    /// toward -z -- which is the axis green lives on in the top-down projection.
    pub fn render_quad_light_at(
        normal: [f32; 3],
        palette: Palette,
        splat: Option<&TerrainImage>,
        sky_occlusion: Option<&TerrainImage>,
        normals: &[Option<TerrainImage>],
        light: Option<Option<u8>>,
        light_pos: glam::Vec3,
    ) -> Option<[u8; 4]> {
        render_quad_turned(normal, palette, splat, sky_occlusion, normals, light, light_pos, 0.0)
    }

    /// `render_quad_light_at` seen by a player turned `yaw` radians: the lamp
    /// and the eye are handed over in the player's frame, as a frame uploads
    /// them, turned about the pixel read -- so its light comes from the same
    /// WORLD direction at any turn, and only what the shader gets wrong about
    /// the turn can change it. The quad's normal is straight up, which no
    /// turn moves.
    pub fn render_quad_turned(
        normal: [f32; 3],
        palette: Palette,
        splat: Option<&TerrainImage>,
        sky_occlusion: Option<&TerrainImage>,
        normals: &[Option<TerrainImage>],
        light: Option<Option<u8>>,
        light_pos: glam::Vec3,
        yaw: f32,
    ) -> Option<[u8; 4]> {
        // The middle of the pixel read (row and column 4 of 8), in clip space,
        // which is this harness's player frame.
        let read = glam::Vec3::new(0.125, -0.125, 0.5);
        let yaw_inv = glam::Quat::from_rotation_y(-yaw);
        let to_player = |p: glam::Vec3| read + yaw_inv * (p - read);
        let light_pos = to_player(light_pos);
        // The rig's offset holds the pixel read at one world point at every
        // turn -- away from the origin, so a ground texture is read there
        // well clear of its wrap -- and the quad's normal is the world's,
        // turned into the player's frame.
        const WORLD_SHIFT: glam::Vec3 = glam::Vec3::new(2.375, 0.625, 4.0);
        let offset = read + WORLD_SHIFT - glam::Quat::from_rotation_y(yaw) * read;
        let normal = (yaw_inv * glam::Vec3::from(normal)).to_array();
        // THE QUAD LIES IN THE PLANE ITS NORMAL SAYS, through the pixel read:
        // the ground's shader asks the triangle it is drawn on which way it
        // faces (`sample_frame`), and a quad square to the eye carrying
        // another surface's normal is a crease nothing on a level has. Its
        // corners are placed so that `quad_view` puts them where they always
        // were on screen, and the pixel read is the same point as before.
        let plane_n = glam::Vec3::from(normal).try_normalize().unwrap_or(glam::Vec3::Z);
        let plane_t = plane_n.any_orthonormal_vector();
        let plane_b = plane_n.cross(plane_t);
        let on_plane = |x: f32, y: f32| (read + plane_t * (x - read.x) + plane_b * (y - read.y)).to_array();
        let quad_view = glam::Mat4::from_cols(
            glam::Vec4::new(plane_t.x, plane_b.x, 0.0, 0.0),
            glam::Vec4::new(plane_t.y, plane_b.y, 0.0, 0.0),
            glam::Vec4::new(plane_t.z, plane_b.z, 0.0, 0.0),
            glam::Vec4::new(read.x - plane_t.dot(read), read.y - plane_b.dot(read), 0.5, 1.0),
        );
        let lit = light.is_some();
        let (device, queue) = headless_gpu()?;
        let format = TextureFormat::Rgba8Unorm;

        let lights = LightsUniform::new(&device);
        let (_shadows, uniforms) =
            crate::renderer::uniforms::test_support::scene_uniforms(&device, &lights);

        // Both uniform buffers are created UNINITIALISED. Without writing them
        // view_proj is garbage -- the triangle lands somewhere arbitrary and the
        // readback is just clear colour -- and the light count is undefined.
        // `quad_view` puts the quad's corners at the clip coordinates named
        // below, and an empty light list leaves the shader's AMBIENT term,
        // which is all this test wants: it is asserting the splat blend, not
        // the light rig.
        uniforms.upload_scene(
            &queue,
            quad_view,
            // The eye. Only specular reads it, and these tests assert the splat
            // blend -- so it goes straight out in front of the quad, where a
            // highlight lands symmetrically and cannot be mistaken for one of
            // the layers winning.
            to_player(glam::Vec3::new(0.0, 0.0, 5.0)),
            &crate::renderer::uniforms::ShadowUpload::disabled(),
            &crate::renderer::uniforms::SkyUpload::none(),
            &crate::renderer::uniforms::PostUpload::default(),
            &crate::renderer::uniforms::PlayerUpload { yaw, offset, ..Default::default() },
        );
        if lit {
            // Placed off to one side so a tilt in the surface normal changes
            // how much light the fragment receives. Directly overhead would
            // make an x-tilt symmetric and hide exactly what is being tested.
            lights.upload(&queue, &[crate::renderer::lights::Light {
                mask_channel: light.flatten(),
                shadow_near: None,
                source_radius: 0.0,
                in_level_bake: true,
                position: light_pos,
                direction: glam::Vec3::new(0.0, -1.0, 0.0),
                kind: crate::renderer::lights::LightKind::Point,
                color: crate::renderer::Color3(255, 255, 255, 255),
                // Deliberately dim. The test palette's layer 0 is pure red, so
                // ambient alone already puts that channel at 153 of 255; a
                // bright light clips it and the lighting difference this test
                // exists to measure vanishes into saturation. Both tilts
                // rendered [255, 0, 0] at intensity 40.
                intensity: 3.0,
                range: 50.0,
                cone_angle_deg: 180.0,
                inner_cone_angle_deg: 0.0,
            }]);
        } else {
            lights.upload(&queue, &[]);
        }

        let pipeline = TerrainPipeline::new(&device, format, &uniforms.layout);

        // Built through TerrainMaterial, not a hand-assembled bind group.
        // The hand-built version drifted the moment the layout gained a
        // binding, and worse, it meant every render test proved things about
        // test code rather than about the code the renderer calls.
        let solid = |rgb: [u8; 4]| TerrainImage { width: 1, height: 1, rgba: rgb.to_vec() };
        let test_layers = [
            solid([255, 0, 0, 255]),   // 0 ground -> red
            solid([0, 0, 255, 255]),   // 1 rock   -> blue
            solid([0, 255, 0, 255]),   // 2 high   -> green
            solid([255, 255, 255, 255]), // 3 spare -> white
        ];
        let neutral = solid([128, 128, 128, 255]);

        let material = match palette {
            Palette::Test => TerrainMaterial::new(
                &device, &queue, &pipeline.material_layout,
                &test_layers, &neutral, splat, sky_occlusion, normals,
                &[], &[], TerrainMaterialUniform::default(),
            ),
            Palette::Fallback => {
                TerrainMaterial::fallback(&device, &queue, &pipeline.material_layout)
            }
            Palette::Ramp => {
                let ramp = TerrainImage {
                    width: 64,
                    height: 64,
                    rgba: (0..64 * 64).flat_map(|i| [(i % 64) as u8 * 4, 0, 0, 255]).collect(),
                };
                // One size for all four, as a texture array needs.
                let layers = [ramp, solid_image([0, 0, 255], 64, 64), solid_image([0, 255, 0], 64, 64), solid_image([255, 255, 255], 64, 64)];
                TerrainMaterial::new(
                    &device, &queue, &pipeline.material_layout,
                    &layers, &neutral, splat, sky_occlusion, normals,
                    &[], &[], TerrainMaterialUniform::default(),
                )
            }
        };

        // A quad filling clip space, carrying the normal under test and lying
        // in its plane: the corners that `quad_view` sends to (-1, -1),
        // (3, -1) and (-1, 3).
        let verts = [
            vertex(on_plane(-1.0, -1.0), normal),
            vertex(on_plane(3.0, -1.0), normal),
            vertex(on_plane(-1.0, 3.0), normal),
        ];
        let vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("test_vb"),
            contents: bytemuck::cast_slice(&verts),
            usage: BufferUsages::VERTEX,
        });
        let ib = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("test_ib"),
            contents: bytemuck::cast_slice(&[0u32, 1, 2]),
            usage: BufferUsages::INDEX,
        });

        const SIZE: u32 = 8;
        let target = device.create_texture(&TextureDescriptor {
            label: Some("test_target"),
            size: Extent3d { width: SIZE, height: SIZE, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1,
            dimension: TextureDimension::D2,
            format,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let depth = device.create_texture(&TextureDescriptor {
            label: Some("test_depth"),
            size: Extent3d { width: SIZE, height: SIZE, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Depth32Float,
            usage: TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let target_view = target.create_view(&Default::default());
        let depth_view = depth.create_view(&Default::default());

        let readback = device.create_buffer(&BufferDescriptor {
            label: Some("test_readback"),
            size: (256 * SIZE) as u64,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("test_pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &target_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: Operations { load: LoadOp::Clear(Color::BLACK), store: StoreOp::Store },
                })],
                depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(Operations { load: LoadOp::Clear(1.0), store: StoreOp::Store }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                multiview_mask: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&pipeline.pipeline);
            pass.set_bind_group(0, &uniforms.bind_group, &[]);
            pass.set_bind_group(1, &material.bind_group, &[]);
            pass.set_vertex_buffer(0, vb.slice(..));
            pass.set_index_buffer(ib.slice(..), IndexFormat::Uint32);
            pass.draw_indexed(0..3, 0, 0..1);
        }
        encoder.copy_texture_to_buffer(
            TexelCopyTextureInfo {
                texture: &target, mip_level: 0, origin: Origin3d::ZERO, aspect: TextureAspect::All,
            },
            TexelCopyBufferInfo {
                buffer: &readback,
                layout: TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(256), rows_per_image: Some(SIZE) },
            },
            Extent3d { width: SIZE, height: SIZE, depth_or_array_layers: 1 },
        );
        queue.submit(Some(encoder.finish()));

        let slice = readback.slice(..);
        slice.map_async(MapMode::Read, |_| {});
        device.poll(PollType::Wait { submission_index: None, timeout: None }).ok();
        let data = slice.get_mapped_range().unwrap();
        let centre = (SIZE / 2) as usize * 256 + (SIZE / 2) as usize * 4;
        Some([data[centre], data[centre + 1], data[centre + 2], data[centre + 3]])
    }

    /// The outdoor twin is the pass with its one switch turned, valid, and
    /// much the smaller; the ground gives the lookup no face of its own (the
    /// twin's reasoning starts from that).
    #[test]
    fn the_outdoor_twin_is_the_pass_with_its_switch_turned() {
        use wgpu::naga;
        let pass = terrain_shader_for(TerrainRole::ProbePass);
        let twin = outdoor_pass_shader().expect("the outdoor switch moved");
        let (sky_twice, sky_once) = crate::renderer::brush_pipeline::DEDUP_OUTDOOR_SKY;
        assert_eq!(twin.replacen(ALL_OUTDOORS_ON, ALL_OUTDOORS_OFF, 1).replacen(sky_once, sky_twice, 1), pass);
        assert!(twin.contains(sky_once), "the outdoor sky's two reads are no longer where the dedup edit expects them");
        assert!(pass.contains("const PROBE_FACE_ALWAYS_GIVEN: bool = false;"));
        assert!(!pass.contains("probe_face_given = ") && !pass.contains("probe_volume_pos = "));
        let at = |s: &str, m: &str| s.find(m).unwrap_or_else(|| panic!("`{m}` is not in the shader"));
        // The lookup's outdoor answer is asked before anything of the rooms'.
        let env = &twin[at(&twin, "fn probe_environment(")..at(&twin, "fn probe_trace(")];
        assert!(at(env, "    if (PROBE_ALL_OUTDOORS) {") < at(env, "    let hit = probe_trace(world_pos, d, trace_room, roughness);"));
        assert!(at(env, "    if (PROBE_ALL_OUTDOORS) {") < at(env, "probe_choose(select_world, volume_world)"));
        assert!(at(env, "    if (PROBE_ALL_OUTDOORS) {") < at(env, "probe_portal_holding(volume_world)"));
        for (label, src) in [
            ("twin", twin.clone()),
            ("weather twin", crate::renderer::weather::with_weather(&twin).expect("the weather's edits moved")),
        ] {
            let module = naga::front::wgsl::parse_str(&src).unwrap_or_else(|e| panic!("{label}: {}", e.emit_to_string(&src)));
            naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
                .validate(&module)
                .unwrap_or_else(|e| panic!("{label}: {e:?}"));
        }
        let size = |s: &str| -> usize {
            crate::renderer::shader_inlining::inlined_sizes(s, "fs_main").iter().map(|f| f.own * f.copies).sum()
        };
        let (full, out) = (size(&pass), size(&twin));
        println!("the ground's pass: {full} proxy units; its outdoor twin: {out}");
        if std::env::var("TWIN_REPORT").is_ok() {
            println!("{}", crate::renderer::shader_inlining::report("outdoor twin", &twin, "fs_main", 30));
        }
        // 2,162 of 4,165 with the untraced answer's two photographs in it
        // (2026-10-12); the trace through the rooms alone is 1,500.
        assert!(out * 5 < full * 3, "the outdoor twin is {out} of {full}: the rooms' code is still in it");
        let Some((device, _queue)) = headless_gpu() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let lights = LightsUniform::new(&device);
        let (_shadows, uniforms) = crate::renderer::uniforms::test_support::scene_uniforms(&device, &lights);
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let fixups = crate::renderer::probe_fixup::ProbeFixups::new(&device, &uniforms.layout, 1024);
        let weather_layout = crate::renderer::weather::bind_group_layout(&device);
        assert!(TerrainPipeline::new_probe_pass_outdoors(&device, &uniforms.layout, &fixups, None).is_some());
        assert!(TerrainPipeline::new_probe_pass_outdoors(&device, &uniforms.layout, &fixups, Some(&weather_layout)).is_some());
        let err = pollster::block_on(scope.pop());
        assert!(err.is_none(), "the outdoor twins failed to build: {err:?}");
    }

    /// The painted build is each ground shader with its slope-driven weights
    /// gone and nothing else changed: valid, the choice in every role's
    /// shader exactly once, and what it keeps is what the choice returns on a
    /// painted level. (The switch itself is not set here: tests share it.)
    #[test]
    fn the_painted_build_keeps_the_splats_weights_alone() {
        use wgpu::naga;
        assert!(!GROUND_PAINTED.load(std::sync::atomic::Ordering::Relaxed));
        for role in [TerrainRole::Scene, TerrainRole::Read, TerrainRole::ProbePass] {
            let src = terrain_shader_for(role);
            let painted = painted_shader(&src).unwrap_or_else(|| panic!("{role:?}: the weights' choice moved"));
            assert_eq!(painted.replacen(PAINTED_TO, PAINTED_FROM, 1), src);
            let module = naga::front::wgsl::parse_str(&painted).unwrap_or_else(|e| panic!("{role:?}: {}", e.emit_to_string(&painted)));
            naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
                .validate(&module)
                .unwrap_or_else(|e| panic!("{role:?}: {e:?}"));
            let size = |s: &str| -> usize {
                crate::renderer::shader_inlining::inlined_sizes(s, "fs_main").iter().map(|f| f.own * f.copies).sum()
            };
            assert!(size(&painted) < size(&src), "{role:?}: nothing left the shader");
        }
        // `select(f, t, cond)` is `t` where `cond` holds, and the material sets
        // the flag to 1 exactly when it was given a splat map.
        assert!(PAINTED_FROM.contains("select(derived, authored, mat.use_splat > 0.5)"));
    }

    /// What the ground's pipelines are built for follows the level in hand,
    /// unless a measurement pinned it.
    #[test]
    fn the_ground_is_built_for_the_level_unless_pinned() {
        for built in [false, true] {
            for level in [false, true] {
                assert_eq!(ground_build_wanted(built, level, false), level, "built {built}, level {level}");
                assert_eq!(ground_build_wanted(built, level, true), built, "pinned at {built}, level {level}");
            }
        }
    }

    /// Every twin the renderer builds of the ground's shaders takes the
    /// painted build as it takes the one with the choice in it: each edit
    /// still finds its text once, and what comes out validates. (The switch
    /// itself is not set here: tests share it. `terrain_shader_built` is what
    /// it selects.)
    #[test]
    fn every_ground_twin_takes_the_painted_build() {
        use crate::renderer::brush_pipeline::{sun_reader_shader, FaceSun};
        use crate::renderer::ground_twins::{dedup_pass_shader, gentle_shader, steep_shader};
        use crate::renderer::lights::{without_pool_maps, without_spot_shadows};
        use crate::renderer::weather::{with_weather, with_weather_kinds, WeatherKinds};
        use wgpu::naga;
        let read = terrain_shader_built(TerrainRole::Read, true);
        let pass = terrain_shader_built(TerrainRole::ProbePass, true);
        for (role, src) in [(TerrainRole::Read, &read), (TerrainRole::ProbePass, &pass)] {
            assert_eq!(src.matches(PAINTED_TO).count(), 1, "{role:?}");
            assert!(!src.contains(PAINTED_FROM), "{role:?}: the choice is still in the painted build");
            assert_eq!(src.replacen(PAINTED_TO, PAINTED_FROM, 1), terrain_shader_built(role, false), "{role:?}");
        }
        let turned = |off: &str, on: &str| {
            assert_eq!(pass.matches(off).count(), 1, "`{off}` is not in the painted pass once");
            pass.replacen(off, on, 1)
        };
        let gentle = gentle_shader(&read).expect("gentle");
        let steep = steep_shader(&read).expect("steep");
        let dedup = dedup_pass_shader(&pass).expect("dedup");
        let nowhere = turned(NOWHERE_OFF, NOWHERE_ON);
        let outdoors = turned(ALL_OUTDOORS_OFF, ALL_OUTDOORS_ON);
        let open = turned(OPEN_GROUND_OFF, OPEN_GROUND_ON);
        let mut sources: Vec<(String, String)> = vec![
            ("read baked spotless".into(), without_spot_shadows(sun_reader_shader(read.clone(), FaceSun::Baked))),
            ("gentle baked spotless".into(), without_spot_shadows(sun_reader_shader(gentle.clone(), FaceSun::Baked))),
            ("steep baked spotless".into(), without_spot_shadows(sun_reader_shader(steep.clone(), FaceSun::Baked))),
            ("dedup poolless".into(), without_pool_maps(dedup.clone())),
            ("nowhere".into(), nowhere.clone()),
            ("outdoors".into(), outdoors.clone()),
            ("open".into(), open.clone()),
        ];
        for (label, src) in [("read", &read), ("pass", &pass), ("dedup", &dedup), ("nowhere", &nowhere), ("outdoors", &outdoors), ("open", &open)] {
            sources.push((format!("weather {label}"), with_weather(src).unwrap_or_else(|| panic!("{label}: the weather's edits moved"))));
        }
        for kinds in WeatherKinds::ALL {
            let twin = with_weather_kinds(&gentle, kinds).unwrap_or_else(|| panic!("gentle {kinds:?}: the weather's edits moved"));
            sources.push((format!("weather gentle {kinds:?}"), twin));
        }
        for (label, src) in &sources {
            let module = naga::front::wgsl::parse_str(src).unwrap_or_else(|e| panic!("{label}: {}", e.emit_to_string(src)));
            naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
                .validate(&module)
                .unwrap_or_else(|e| panic!("{label}: {e:?}"));
        }
    }

    /// The open-ground twin is the pass with its one switch turned: valid,
    /// much smaller, the outdoor answer and nothing of the photographs', the
    /// rooms' or the doorways' -- and it answers as the full pass does for a
    /// surface in no room and no doorway once the frame says open ground
    /// reflects (`ProbeUpload::open_ground`): escaped where it lies, coloured
    /// by `probe_escape_colour`; nothing where too rough to trace.
    /// The scene's ground is drawn without blending ([`OPAQUE_GROUND`]), which
    /// is the picture a blend drew only while every reader's one answer has
    /// alpha 1: the plain readers, the steep ones, and each with the weather.
    #[test]
    fn the_grounds_scene_shaders_write_alpha_one() {
        use crate::renderer::weather::{with_weather, with_weather_kinds, WeatherKinds};
        const ANSWER: &str = "return vec4<f32>(tonemap(in.col.rgb * lit), 1.0);";
        let mut seen = 0;
        for role in [TerrainRole::Scene, TerrainRole::Read] {
            let plain = terrain_shader_for(role);
            let mut sources = vec![plain.clone()];
            sources.extend(crate::renderer::ground_twins::steep_shader(&plain));
            for src in sources.clone() {
                sources.extend(with_weather(&src));
                sources.extend(WeatherKinds::ALL.iter().filter_map(|&k| with_weather_kinds(&src, k)));
            }
            for src in sources {
                let main = &src[src.find("fn fs_main(").expect("the ground's fragment shader")..];
                let main = &main[..main.find("\n}\n").expect("its end")];
                let returns: Vec<&str> = main.lines().map(str::trim).filter(|l| l.starts_with("return ")).collect();
                assert_eq!(returns, [ANSWER], "a ground reader answers with something a blend would not keep");
                seen += 1;
            }
        }
        assert!(seen >= 8, "{seen} readers looked at");
    }

    /// Every ground shader a frame may draw takes world vertices by the one
    /// rewrite ([`world_vertices`]), and validates with it; so does the solid
    /// pipeline's, which draws the ground into a mirror.
    #[test]
    fn every_ground_shader_takes_its_vertices_in_the_world() {
        use crate::renderer::weather::{with_weather, with_weather_kinds, WeatherKinds};
        let mut seen = 0;
        for role in [TerrainRole::Scene, TerrainRole::Read, TerrainRole::ProbePass] {
            let plain = terrain_shader_for(role);
            let mut sources = vec![plain.clone()];
            sources.extend(crate::renderer::ground_twins::steep_shader(&plain));
            if role == TerrainRole::ProbePass {
                sources.extend(nowhere_pass_shader());
                sources.extend(open_pass_shader());
                sources.extend(puddle_pass_shader());
            }
            for src in sources.clone() {
                sources.extend(with_weather(&src));
                sources.extend(WeatherKinds::ALL.iter().filter_map(|&k| with_weather_kinds(&src, k)));
            }
            for src in sources {
                let world = world_vertices(&src).expect("a ground shader that does not open as the ground's does");
                assert!(world.contains("v.pos = to_player_space(w.pos);") && world.contains("v.norm = to_player_direction(w.norm);"));
                assert_eq!(world.matches("fn vs_main(").count(), 1);
                // Nothing after the opening reads the world's vertex.
                let body = &world[world.find("v.norm = to_player_direction(w.norm);").unwrap() + 40..];
                let body = &body[..body.find("\n}\n").unwrap()];
                assert!(!body.contains("w."), "the vertex shader reads the world's vertex past its posing");
                seen += 1;
            }
        }
        assert!(seen >= 12, "{seen} shaders looked at");
        let validate = |label: &str, src: &str| {
            use wgpu::naga;
            let module = naga::front::wgsl::parse_str(src).unwrap_or_else(|e| panic!("{label}: {}", e.emit_to_string(src)));
            naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
                .validate(&module)
                .unwrap_or_else(|e| panic!("{label}: {e:?}"));
        };
        validate("the ground reader from world vertices", &world_vertices(&terrain_shader_for(TerrainRole::Read)).unwrap());
        validate("the ground's reflection pass from world vertices", &world_vertices(&terrain_shader_for(TerrainRole::ProbePass)).unwrap());
        let solid = world_vertices(&crate::renderer::pipeline::solid_shader_src()).expect("the solid pipeline's shader opens as the ground's");
        validate("the solid pipeline from world vertices", &solid);
    }

    /// The puddle twin is the open twin but for its answer, which is the sky
    /// the open twin's own answer ends in; and it validates.
    #[test]
    fn the_puddle_twin_is_the_open_twin_answering_with_the_sky() {
        use wgpu::naga;
        let open = open_pass_shader().expect("the open-ground switch moved");
        let puddle = puddle_pass_shader().expect("the open ground's answer moved");
        assert_eq!(puddle.replacen(OPEN_SKY_COLOUR, OPEN_COLOUR, 1), open);
        assert!(open.contains("    return sky_reflection(d, dir, lod);"), "the outdoors no longer ends in the sky");
        let wet = crate::renderer::weather::with_weather(&puddle).expect("the weather's edits");
        for src in [&puddle, &wet] {
            let module = naga::front::wgsl::parse_str(src).unwrap_or_else(|e| panic!("{}", e.emit_to_string(src)));
            naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
                .validate(&module)
                .unwrap_or_else(|e| panic!("{e:?}"));
        }
    }

    #[test]
    fn the_open_twin_is_the_pass_with_its_switch_turned() {
        use wgpu::naga;
        let pass = terrain_shader_for(TerrainRole::ProbePass);
        let twin = open_pass_shader().expect("the open-ground switch moved");
        assert_eq!(twin.replacen(OPEN_GROUND_ON, OPEN_GROUND_OFF, 1), pass);
        let at = |s: &str, m: &str| s.find(m).unwrap_or_else(|| panic!("`{m}` is not in the shader"));
        let env = &twin[at(&twin, "fn probe_environment(")..at(&twin, "fn probe_outdoor_untraced(")];
        let branch = at(env, "    if (PROBE_OPEN_GROUND) {");
        for later in [
            "    if (PROBE_NOWHERE) {",
            "    if (PROBE_ALL_OUTDOORS) {",
            "probe_portal_holding(volume_world)",
            "probe_choose(select_world, volume_world)",
            "    let hit = probe_trace(world_pos, d, trace_room, roughness);",
        ] {
            assert!(branch < at(env, later), "`{later}` comes before the open ground's answer");
        }
        let body = &env[branch..at(env, "    if (PROBE_NOWHERE) {")];
        // The trace's own roughness test, then the escape from the surface
        // itself in the full pass's direction and blur.
        assert!(body.contains("if (roughness > PROBE_TRACE_MAX_ROUGHNESS) {"));
        assert!(body.contains("let open_dir = normalize(world_dir);"));
        assert!(body.contains("let open_lod = clamp(roughness * PROBE_ROUGHNESS_MIPS, PROBE_MIN_LOD, PROBE_MAX_LOD);"));
        assert!(body.contains("probe_escape_colour(world_pos, open_dir, dir, roughness, open_lod)"));
        assert!(env.contains("    let d = normalize(world_dir);"));
        assert!(env.contains("    let probe_lod = clamp(roughness * PROBE_ROUGHNESS_MIPS, PROBE_MIN_LOD, PROBE_MAX_LOD);"));
        // THE FULL PASS'S HALF: in the trace, where no room and no doorway
        // holds the surface, under the table's word -- an escaped hit at the
        // surface, which `probe_traced_colour` colours the same way.
        let trace = &pass[at(&pass, "fn probe_trace_skipping(")..at(&pass, "fn ground_trace_until(")];
        let holding = at(trace, "        let p = probe_portal_holding(world_pos);\n        if (p < 0) {\n            if (camera.portal_params.y < -0.5) {");
        let fallback = &trace[holding..holding + at(&trace[holding..], "            return hit;\n        }\n")];
        for line in ["hit.pos = world_pos;", "hit.found = true;", "hit.escaped = true;"] {
            assert!(fallback.contains(line), "`{line}` is not in the open ground's hit");
        }
        assert!(at(trace, "if (roughness > PROBE_TRACE_MAX_ROUGHNESS || camera.portal_params.y > 0.5) {") < holding);
        assert!(pass.contains("        col = probe_escape_colour(h.pos, d, sky_dir, roughness, lod);"));
        // The ground's shaders alone: the brushes' pass keeps its text.
        let brush = crate::renderer::brush_pipeline::brush_shader_src();
        assert!(brush.contains("let p = probe_portal_holding(world_pos);"));
        assert!(!brush.contains("camera.portal_params.y < -0.5"));
        for (label, src) in [
            ("twin", twin.clone()),
            ("weather twin", crate::renderer::weather::with_weather(&twin).expect("the weather's edits moved")),
        ] {
            let module = naga::front::wgsl::parse_str(&src).unwrap_or_else(|e| panic!("{label}: {}", e.emit_to_string(&src)));
            naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
                .validate(&module)
                .unwrap_or_else(|e| panic!("{label}: {e:?}"));
        }
        let size = |s: &str| -> usize {
            crate::renderer::shader_inlining::inlined_sizes(s, "fs_main").iter().map(|f| f.own * f.copies).sum()
        };
        let (full, out) = (size(&pass), size(&twin));
        println!("the ground's pass: {full} proxy units; its open-ground twin: {out}");
        assert!(out * 5 < full * 3, "the open-ground twin is {out} of {full}: the rooms' machinery is still in it");
        let Some((device, _queue)) = headless_gpu() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let lights = LightsUniform::new(&device);
        let (_shadows, uniforms) = crate::renderer::uniforms::test_support::scene_uniforms(&device, &lights);
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let fixups = crate::renderer::probe_fixup::ProbeFixups::new(&device, &uniforms.layout, 1024);
        let weather_layout = crate::renderer::weather::bind_group_layout(&device);
        assert!(TerrainPipeline::new_probe_pass_open(&device, &uniforms.layout, &fixups, None).is_some());
        assert!(TerrainPipeline::new_probe_pass_open(&device, &uniforms.layout, &fixups, Some(&weather_layout)).is_some());
        let err = pollster::block_on(scope.pop());
        assert!(err.is_none(), "the open-ground twins failed to build: {err:?}");
    }

    /// The nowhere twin is the pass with its one switch turned: valid, far
    /// smaller, its lookup answering before anything of the photographs', the
    /// rooms' or the doorways' -- and what it answers is what the full lookup
    /// leaves when it finds nothing.
    #[test]
    fn the_nowhere_twin_is_the_pass_with_its_switch_turned() {
        use wgpu::naga;
        let pass = terrain_shader_for(TerrainRole::ProbePass);
        let twin = nowhere_pass_shader().expect("the nowhere switch moved");
        assert_eq!(twin.replacen(NOWHERE_ON, NOWHERE_OFF, 1), pass);
        let at = |s: &str, m: &str| s.find(m).unwrap_or_else(|| panic!("`{m}` is not in the shader"));
        let env = &twin[at(&twin, "fn probe_environment(")..at(&twin, "fn probe_outdoor_untraced(")];
        let branch = at(env, "    if (PROBE_NOWHERE) {");
        for later in [
            "    if (PROBE_ALL_OUTDOORS) {",
            "probe_portal_holding(volume_world)",
            "probe_choose(select_world, volume_world)",
            "    let hit = probe_trace(world_pos, d, trace_room, roughness);",
            "    return probe_through_portals(own, own_room, select_world, world_pos, d, probe_lod);",
        ] {
            assert!(branch < at(env, later), "`{later}` comes before the nowhere answer");
        }
        // What the full lookup leaves behind when it finds nothing: brightness
        // 0 before the branch, the eye's distance in it, the reach untouched.
        assert!(at(env, "    probe_brightness = 0.0;") < branch);
        let body = &env[branch..branch + at(&env[branch..], "    }\n")];
        assert!(body.contains("probe_eye_distance = distance(cam_pos(), frag_pos);"));
        assert!(body.contains("let nothing = vec4<f32>(0.0);") && body.contains("return nothing;"));
        assert!(!body.contains("probe_reach"));
        for (label, src) in [
            ("twin", twin.clone()),
            ("weather twin", crate::renderer::weather::with_weather(&twin).expect("the weather's edits moved")),
        ] {
            let module = naga::front::wgsl::parse_str(&src).unwrap_or_else(|e| panic!("{label}: {}", e.emit_to_string(&src)));
            naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
                .validate(&module)
                .unwrap_or_else(|e| panic!("{label}: {e:?}"));
        }
        let size = |s: &str| -> usize {
            crate::renderer::shader_inlining::inlined_sizes(s, "fs_main").iter().map(|f| f.own * f.copies).sum()
        };
        let (full, out) = (size(&pass), size(&twin));
        println!("the ground's pass: {full} proxy units; its nowhere twin: {out}");
        // 1,495 of 4,165 (2026-10-12): the ground's own layers, the
        // characters' capsules and the floor mirror.
        assert!(out * 5 < full * 2, "the nowhere twin is {out} of {full}: the lookup is still in it");
        if std::env::var("TWIN_REPORT").is_ok() {
            println!("{}", crate::renderer::shader_inlining::report("nowhere twin", &twin, "fs_main", 20));
        }
        let Some((device, _queue)) = headless_gpu() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let lights = LightsUniform::new(&device);
        let (_shadows, uniforms) = crate::renderer::uniforms::test_support::scene_uniforms(&device, &lights);
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let fixups = crate::renderer::probe_fixup::ProbeFixups::new(&device, &uniforms.layout, 1024);
        let weather_layout = crate::renderer::weather::bind_group_layout(&device);
        assert!(TerrainPipeline::new_probe_pass_nowhere(&device, &uniforms.layout, &fixups, None).is_some());
        assert!(TerrainPipeline::new_probe_pass_nowhere(&device, &uniforms.layout, &fixups, Some(&weather_layout)).is_some());
        let err = pollster::block_on(scope.pop());
        assert!(err.is_none(), "the nowhere twins failed to build: {err:?}");
    }

    /// The ground's pass traces every texel, rough ones too -- skipping them is
    /// not the same picture (`PASS_ROUGH_SKIP`) -- and the cut that prices the
    /// rough ones' reflection puts the skip where the scene shader's own
    /// weighting says it belongs: after the weather's code, which is what
    /// makes wet ground smooth, from the roughness where a texel's own pixel
    /// gives the reflection no share.
    #[test]
    fn the_rough_skip_is_a_cut_and_not_in_the_ground_pass() {
        let pass = terrain_shader_for(TerrainRole::ProbePass);
        assert!(!pass.contains(PASS_ROUGH_SKIP));
        assert_eq!(PASS_ROUGH_SKIP_IN, PASS_ROUGH_SKIP_AT.replacen("    return", &format!("{PASS_ROUGH_SKIP}    return"), 1));
        let (_, edits) = PROBE_PASS_REGISTER_CUTS.iter().find(|(l, _)| *l == "terrain_cut_rough_skip").unwrap();
        let wet = crate::renderer::weather::with_weather(&pass).expect("the ground's weather twin no longer matches its shader");
        assert!(wet.contains("    var rough = specular_aa_roughness("));
        let at = |s: &str, m: &str| s.find(m).unwrap_or_else(|| panic!("`{m}` is not in the shader"));
        for src in [&pass, &wet] {
            assert_eq!(src.matches(edits[0].0).count(), 1);
            let cut = src.replacen(edits[0].0, edits[0].1, 1);
            assert_eq!(cut.matches(PASS_ROUGH_SKIP).count(), 1);
            assert!(at(&cut, "    probe_fragment = in.clip;\n") < at(&cut, PASS_ROUGH_SKIP));
            assert!(at(&cut, PASS_ROUGH_SKIP) < at(&cut, "    return probe_env_for_pass(\n        in.world_pos"));
        }
        // The three places the scene shader weighs the probe's answer, each by
        // `1 - lobe_is_hemispherical`, which is 0 from the skip's roughness up.
        let read = terrain_shader_for(TerrainRole::Read);
        for line in [
            "    let lobe_is_hemispherical = smoothstep(0.25, PROBE_LOBE_HEMISPHERICAL, r);",
            "    let environment = mix(sharp, baseline, lobe_is_hemispherical);",
            "    let image_weight = (1.0 - lobe_is_hemispherical) * clamp(probe.a, 0.0, 1.0);",
            "    let r = clamp(roughness, 0.04, 1.0);",
        ] {
            assert!(read.contains(line), "the scene shader no longer has `{line}`");
        }
        let smoothstep = |lo: f32, hi: f32, x: f32| {
            let t = ((x - lo) / (hi - lo)).clamp(0.0, 1.0);
            t * t * (3.0 - 2.0 * t)
        };
        for r in [0.75f32, 0.7500001, 0.8, 0.93, 1.0] {
            assert_eq!(smoothstep(0.25, 0.75, r), 1.0, "{r}");
        }
        assert!(smoothstep(0.25, 0.75, 0.7499) < 1.0);
    }

    #[test]
    /// The optimisation itself, pinned in the source.
    ///
    /// Its effect is a frame time, which no unit test can see, and its absence
    /// is invisible: reverting to unconditional sampling renders exactly the
    /// same picture, only slower. So the guard is that the branches and the
    /// explicit-gradient sampling are still there.
    ///
    /// Measured cause: removing the ground took the frame from 24 ms to 11 ms
    /// of a 13.9 ms budget, while quartering the triangle count changed nothing
    /// -- terrain is fill-bound, and four layers sampled whether or not they
    /// contribute is where that fill goes.
    /// The ground's two probe-pass shaders -- the one reading the pass and the
    /// one writing it -- are valid WGSL and build on a real device with the
    /// layouts they are drawn with. See `TerrainRole`.
    #[test]
    fn the_probe_pass_terrain_shaders_validate_and_build() {
        use wgpu::naga;
        // The pass's poolless twin too, for frames no torch lights; the
        // reader's twins; and the measurement's inlined reads in both.
        let poolless = crate::renderer::lights::without_pool_maps(terrain_shader_for(TerrainRole::ProbePass));
        let read = terrain_shader_for(TerrainRole::Read);
        let baked = crate::renderer::brush_pipeline::sun_reader_shader(read.clone(), crate::renderer::brush_pipeline::FaceSun::Baked);
        let spotless = crate::renderer::lights::without_spot_shadows;
        for (role, src) in [
            ("read", read.clone()),
            ("pass", terrain_shader_for(TerrainRole::ProbePass)),
            ("poolless pass", poolless),
            ("spotless read", spotless(read.clone())),
            ("baked read", baked.clone()),
            ("baked spotless read", spotless(baked)),
            ("inlined read", with_inlined_layer_reads(read)),
            ("inlined pass", with_inlined_layer_reads(terrain_shader_for(TerrainRole::ProbePass))),
        ] {
            let module = naga::front::wgsl::parse_str(&src)
                .unwrap_or_else(|e| panic!("{role}: {}", e.emit_to_string(&src)));
            naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
                .validate(&module)
                .unwrap_or_else(|e| panic!("{role}: {e:?}"));
        }
        assert!(terrain_shader_for(TerrainRole::Read).contains("const PROBE_ENV_FROM_PASS: bool = true;"));
        let pass = terrain_shader_for(TerrainRole::ProbePass);
        assert!(pass.contains("const PROBE_SECONDARY_DEFERRED: bool = true;") && pass.contains("return probe_env_for_pass("));
        let Some((device, _queue)) = headless_gpu() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let lights = LightsUniform::new(&device);
        let (_shadows, uniforms) = crate::renderer::uniforms::test_support::scene_uniforms(&device, &lights);
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let probe_layout = crate::renderer::brush_pipeline::probe_pass::bind_group_layout(&device);
        let _read = TerrainPipeline::new_probe_reader(&device, TextureFormat::Rgba8UnormSrgb, &uniforms.layout, 4, &probe_layout);
        let fixups = crate::renderer::probe_fixup::ProbeFixups::new(&device, &uniforms.layout, 1024);
        let _pass = TerrainPipeline::new_probe_pass(&device, &uniforms.layout, &fixups);
        let _poolless = TerrainPipeline::new_probe_pass_poolless(&device, &uniforms.layout, &fixups);
        let _twins =
            TerrainPipeline::new_probe_reader_twins(&device, TextureFormat::Rgba8UnormSrgb, &uniforms.layout, 4, &probe_layout);
        let _inlined =
            TerrainPipeline::new_inlined(&device, TextureFormat::Rgba8UnormSrgb, &uniforms.layout, 4, &probe_layout, &fixups);
        let err = pollster::block_on(scope.pop());
        assert!(err.is_none(), "the terrain probe pipelines failed to build: {err:?}");
    }

    /// Each of the ground reader's twins is the reader with its own switches
    /// set and nothing else, and the measurement's inlined reads replace the
    /// loop and nothing else -- so any difference in their pictures is the
    /// switch's, and any difference in their cost is what they leave out.
    #[test]
    fn the_ground_readers_twins_and_inlined_reads_change_only_their_switches() {
        let read = terrain_shader_for(TerrainRole::Read);
        let baked = crate::renderer::brush_pipeline::sun_reader_shader(read.clone(), crate::renderer::brush_pipeline::FaceSun::Baked);
        assert_eq!(baked.replacen("const SUN_MASK_EVERYWHERE: bool = true;", "const SUN_MASK_EVERYWHERE: bool = false;", 1), read);
        let spotless = crate::renderer::lights::without_spot_shadows(read.clone());
        assert_ne!(spotless, read, "the spotless twin switched nothing off");
        let inlined = with_inlined_layer_reads(read.clone());
        assert_eq!(inlined.replacen(LAYER_READS_INLINED, LAYER_READS, 1), read);
        // One copy of each read in the loop; four, one a layer, inlined.
        for call in ["layer_colour_at(", "layer_normal_at(", "layer_rough_at(", "layer_ao_at("] {
            let body = |src: &str| src.split("@fragment").nth(1).map_or(0, |entry| entry.matches(call).count());
            assert_eq!((body(&read), body(&inlined)), (1, 4), "{call}");
        }
    }

    /// A ground map is baked everywhere only when every texel is: one texel
    /// from an older bake (alpha 255) or no map at all keeps the full reader.
    #[test]
    fn a_ground_map_is_baked_everywhere_only_when_every_texel_is() {
        let map = |alphas: &[u8]| TerrainImage {
            width: alphas.len() as u32,
            height: 1,
            rgba: alphas.iter().flat_map(|&a| [200, 128, 0, a]).collect(),
        };
        assert!(map(&[0, 0, 0, 0]).sun_baked_everywhere());
        assert!(map(&[0, 127]).sun_baked_everywhere(), "127 reads as 0.498, under the shader's half");
        assert!(!map(&[0, 128]).sun_baked_everywhere(), "128 reads as 0.502: not baked there");
        assert!(!map(&[255]).sun_baked_everywhere());
        assert!(!FULL_SKY.sun_baked_everywhere(), "no map is no bake");
        assert!(!TerrainImage { width: 0, height: 0, rgba: Vec::new() }.sun_baked_everywhere());
    }

    /// Every register cut of the ground's probe pass still finds its line, and
    /// what it leaves is valid WGSL -- a cut whose line moved logs a warning on
    /// the headset and measures nothing.
    #[test]
    fn every_terrain_register_cut_applies_and_validates() {
        use wgpu::naga;
        let base = terrain_shader_for(TerrainRole::ProbePass);
        for (label, edits) in PROBE_PASS_REGISTER_CUTS {
            let mut src = base.clone();
            for (from, to) in edits.iter() {
                assert!(src.contains(from), "{label}: `{from}` is not in the ground's probe pass");
                // Once: the edit is made at the first match.
                assert_eq!(src.matches(from).count(), 1, "{label}: `{from}` is in the ground's probe pass more than once");
                src = src.replacen(from, to, 1);
            }
            let module = naga::front::wgsl::parse_str(&src)
                .unwrap_or_else(|e| panic!("{label}: {}", e.emit_to_string(&src)));
            naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
                .validate(&module)
                .unwrap_or_else(|e| panic!("{label}: {e:?}"));
        }
    }

    #[test]
    fn weightless_layers_are_skipped_rather_than_sampled() {
        let src = super::terrain_shader();
        assert!(
            src.contains("textureSampleGrad"),
            "explicit gradients are what make a per-fragment branch legal at all",
        );
        for w in ["w.x > WEIGHT_EPS", "w.y > WEIGHT_EPS", "w.z > WEIGHT_EPS", "w.w > WEIGHT_EPS"] {
            assert!(src.contains(w), "missing the skip for {w}");
        }
    }

    #[test]
    fn the_gradients_are_taken_outside_every_branch() {
        // The reason this is legal. `textureSample` chooses its mip from
        // implicit derivatives and is only valid in uniform control flow --
        // which is exactly what forced all four layers to be sampled. Taking a
        // derivative INSIDE a branch reintroduces that problem while looking
        // like it had been solved, so every dpdx/dpdy must live in the one
        // function that runs before any branching.
        //
        // ONE DELIBERATE EXCEPTION, at the top level of `fs_main`: the
        // specular-antialiasing term takes `dpdx(shaded_n)`. That is not
        // inside a branch -- the layer blends have reconverged by then -- and
        // naga's own uniformity analysis accepts it, which
        // `every_scene_shader_survives_the_multiview_transform` exercises on
        // this very shader. The brush path does the same thing for the same
        // reason. The rule being enforced is "no derivative INSIDE a branch",
        // and counting them per function is how that is approximated cheaply.
        let src = super::terrain_shader();
        let frame = src
            .split("fn sample_frame")
            .nth(1)
            .and_then(|s| s.split("\nfn ").next())
            .expect("sample_frame present");
        let entry = src
            .split("@fragment")
            .nth(1)
            .expect("fragment entry present");
        let allowed = |tok: &str| frame.matches(tok).count() + entry.matches(tok).count();
        assert_eq!(
            src.matches("dpdx(").count(),
            allowed("dpdx("),
            "a derivative escaped both sample_frame and the fragment entry",
        );
        assert_eq!(src.matches("dpdy(").count(), allowed("dpdy("));
        // And the entry's own use must be exactly the specular-AA one, not a
        // second sampling gradient that has quietly moved out of the frame.
        // TWO named pairs are allowed at the top of the entry, both before any
        // branch: the specular-AA one above, and the pixel footprint the light
        // loop uses to keep a spot's edge a pixel wide (`pixel_footprint`).
        // Anything beyond those is a sampling gradient that escaped the frame.
        let footprint = "let step_x = dpdx(in.world_pos);\n    let step_y = dpdy(in.world_pos);";
        let extra = usize::from(entry.contains(footprint));
        assert!(
            entry.matches("dpdx(").count() <= 1 + extra && entry.matches("dpdy(").count() <= 1 + extra,
            "more than the two named derivative pairs in the fragment entry; \
             sampling gradients belong in sample_frame",
        );
    }

    #[test]
    fn baked_sky_occlusion_darkens_the_terrain() {
        // The ground inside a sealed room used to light exactly as brightly as
        // the field outside it, because `shade` had no occlusion term at all --
        // only the BRUSH shader did. No amount of shadow-map work could fix
        // that: the term that was wrong was the ambient, not the direct light.
        //
        // This test also compiles the terrain pipeline, which is the only thing
        // that validates its WGSL: a bad field accessor in that shader builds
        // clean and dies at pipeline creation on the device.
        let sealed = TerrainImage { width: 1, height: 1, rgba: vec![0, 0, 0, 255] };
        let Some(open) = render_quad([0.0, 1.0, 0.0], Palette::Test, None, None) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let dark = render_quad([0.0, 1.0, 0.0], Palette::Test, None, Some(&sealed)).unwrap();
        assert!(
            dark[0] < open[0],
            "zero sky visibility must darken the ground: {dark:?} vs open {open:?}",
        );
    }

    #[test]
    fn unbaked_terrain_shades_exactly_as_it_did_before() {
        // The neutral is WHITE here, not black: this value is MULTIPLIED into
        // the ambient term, the opposite of the brush lightmap's additive RGB.
        // Binding the wrong neutral would put every unbaked terrain in the game
        // into permanent night, which reads as a broken shader.
        let full = TerrainImage { width: 1, height: 1, rgba: vec![255, 255, 255, 255] };
        let Some(absent) = render_quad([0.0, 1.0, 0.0], Palette::Test, None, None) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let explicit = render_quad([0.0, 1.0, 0.0], Palette::Test, None, Some(&full)).unwrap();
        assert_eq!(
            absent, explicit,
            "no occlusion map must shade identically to a fully-open one: \
             {absent:?} vs {explicit:?}",
        );
    }

    #[test]
    fn a_mip_chain_runs_all_the_way_down_to_one_texel() {
        // Stopping early leaves the hardware clamped to the smallest level
        // present, which brings the shimmer back at exactly the distances mips
        // exist to fix.
        assert_eq!(mip_levels_for(1024, 1024), 11);
        assert_eq!(mip_levels_for(1, 1), 1);
        assert_eq!(mip_levels_for(256, 64), 9, "levels follow the LONGER side");

        let levels = mip_levels_for(8, 8);
        let chain = mip_chain(&[128u8; 8 * 8 * 4], 8, 8, levels);
        assert_eq!(chain.len() as u32, levels);
        assert_eq!((chain[0].1, chain[0].2), (8, 8));
        assert_eq!(chain.last().unwrap().1, 1, "chain must reach 1x1");
        for (data, w, h) in &chain {
            assert_eq!(data.len(), (w * h * 4) as usize, "level {w}x{h} is the wrong size");
        }
    }

    #[test]
    fn mips_are_averaged_in_linear_space_not_srgb() {
        // Averaging sRGB BYTES directly is the classic "distant ground goes
        // muddy" bug: the encoding is not linear in intensity, so the mean of
        // two encoded values is darker than the encoding of their mean. Black
        // and white must average to mid GREY -- sRGB 188, not sRGB 128.
        let mut px = vec![0u8; 2 * 2 * 4];
        for i in 0..4 {
            let v = if i % 2 == 0 { 0u8 } else { 255u8 };
            for c in 0..3 {
                px[i * 4 + c] = v;
            }
            px[i * 4 + 3] = 255;
        }
        let chain = mip_chain(&px, 2, 2, 2);
        let r = chain[1].0[0];
        assert!(
            (185..=191).contains(&r),
            "black+white must average to sRGB ~188 (linear 0.5), got {r} -- \
             {} suggests the average was taken on the encoded bytes",
            if r < 140 { "which" } else { "this" },
        );
        assert_eq!(chain[1].0[3], 255, "alpha is already linear and must not be transferred");
    }

    #[test]
    fn a_flat_colour_survives_every_mip_level() {
        // A round-trip check on the transfer functions: a uniform image cannot
        // change under a box filter, so any drift here is the sRGB conversion
        // being lossy in a direction that would tint every distant surface.
        let chain = mip_chain(&[200u8; 16 * 16 * 4], 16, 16, mip_levels_for(16, 16));
        for (data, w, h) in &chain {
            for (i, b) in data.iter().enumerate() {
                assert!(
                    b.abs_diff(200) <= 1,
                    "level {w}x{h} byte {i} drifted to {b} from 200",
                );
            }
        }
    }

    fn flat_ground_shades_from_the_ground_layer() {
        let Some(px) = render_quad_with_normal([0.0, 1.0, 0.0]) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        // Layer 0 is red. Lighting scales it, so assert the CHANNEL BALANCE
        // rather than an exact value -- an absolute assert would be a test of
        // the light rig rather than of the splat blend.
        assert!(px[0] > px[2], "flat ground should take the ground layer (red), got {px:?}");
    }

    #[test]
    fn a_cliff_shades_from_the_rock_layer() {
        // Normal pointing sideways: 90 degrees of slope, well past slope_end.
        let Some(px) = render_quad_with_normal([1.0, 0.0, 0.0]) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        assert!(px[2] > px[0], "a cliff should take the rock layer (blue), got {px:?}");
    }

    #[test]
    fn the_slope_blend_is_gradual_rather_than_a_hard_switch() {
        // Halfway through the 22..40 degree band, both layers should contribute
        // -- a hard switch reads as a visible seam right across a hillside.
        let a = 31.0_f32.to_radians();
        let Some(px) = render_quad_with_normal([a.sin(), a.cos(), 0.0]) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        assert!(px[0] > 8, "ground layer vanished mid-blend: {px:?}");
        assert!(px[2] > 8, "rock layer absent mid-blend: {px:?}");
    }
}

/// The textures and settings one terrain draws with.
///
/// Built separately from the pipeline because the pipeline is per-device and
/// this is per-scene: two levels want different ground, and rebuilding a
/// pipeline to change a texture would be absurd.
pub struct TerrainMaterial {
    pub bind_group: BindGroup,
    pub uniform: Buffer,
}

/// Mip levels for a texture of this size: down to 1x1, as the spec requires.
///
/// A chain that stops early leaves the smallest levels missing, and the
/// hardware clamps to the last one present -- which brings the aliasing back at
/// exactly the distances mips existed to fix.
pub fn mip_levels_for(w: u32, h: u32) -> u32 {
    32 - w.max(h).max(1).leading_zeros()
}

/// A box-filtered mip chain for one RGBA8 image, smallest-last.
///
/// Box filtering on the CPU rather than a blit pipeline: this runs once at
/// load, and a render pass per level per layer would need its own pipeline,
/// bind groups and a non-sRGB view of an sRGB texture. The quality difference
/// at these sizes is not visible on a headset; the difference in moving parts
/// is considerable.
///
/// Averaged in LINEAR space, not sRGB. Averaging sRGB bytes directly darkens
/// every mip -- the classic "distant ground goes muddy" artefact -- because the
/// encoding is not linear in intensity and the mean of two encoded values is
/// not the encoding of their mean.
/// The same box-filtered chain for data that is NOT colour.
///
/// Normal, roughness and occlusion maps are measurements, not colours. Putting
/// them through the sRGB transfer bends every value toward the dark end -- for
/// a normal map that tilts every bump, which this renderer has shipped once
/// already and does not intend to again.
/// Roughness for a layer that has no map, as an 8-bit value.
///
/// Matte, because dirt, gravel and grass are. A layer that disagrees ships a
/// roughness map and stops using this.
pub const DEFAULT_TERRAIN_ROUGHNESS: u8 = 200;

/// The most extra roughness-squared a normal's lost variation may add.
const TERRAIN_NORMAL_VARIANCE_CLAMP: f32 = 0.18;

/// A roughness mip chain that carries the variation the NORMAL map lost when it
/// was mipped -- the same treatment brush materials get.
///
/// Mipping a normal map averages its normals, and shading is not linear in the
/// normal: the lighting of an averaged normal is not the average of the
/// lighting. The LENGTH of the averaged (unnormalised) normal records how much
/// they disagreed, and that converts into variance, and variance is roughness.
///
/// `sigma2 = (1 - |Na|) / |Na|`, combined as `r' = sqrt(r^2 + min(2*sigma2, k))`.
/// ROUGHNESS VALUES DO NOT ADD -- they combine in variance space, which is why
/// this squares before summing and takes the root after.
///
/// Level 0 is untouched: nothing has been averaged there, so nothing was lost,
/// and ground seen underfoot keeps exactly the roughness its map specifies.
pub fn roughness_chain_with_normal_variance(
    rough: &[u8],
    normal: &[u8],
    w: u32,
    h: u32,
    levels: u32,
) -> Vec<(Vec<u8>, u32, u32)> {
    let mut chain = mip_chain_linear(rough, w, h, levels);

    // The normal halved repeatedly in FLOAT, kept UNNORMALISED -- the
    // shortening is the whole measurement, and doing it in f32 rather than
    // re-reading 8-bit mips avoids quantising twice.
    let mut vecs: Vec<[f32; 3]> = (0..(w * h) as usize)
        .map(|i| {
            [
                normal[i * 4] as f32 / 255.0 * 2.0 - 1.0,
                normal[i * 4 + 1] as f32 / 255.0 * 2.0 - 1.0,
                normal[i * 4 + 2] as f32 / 255.0 * 2.0 - 1.0,
            ]
        })
        .collect();
    let (mut cw, mut ch) = (w, h);

    for level in 1..chain.len() {
        let (nw, nh) = ((cw / 2).max(1), (ch / 2).max(1));
        let mut next = vec![[0.0f32; 3]; (nw * nh) as usize];
        for y in 0..nh {
            for x in 0..nw {
                let mut acc = [0.0f32; 3];
                for (dx, dy) in [(0u32, 0u32), (1, 0), (0, 1), (1, 1)] {
                    let sx = (x * 2 + dx).min(cw - 1);
                    let sy = (y * 2 + dy).min(ch - 1);
                    let v = vecs[(sy * cw + sx) as usize];
                    acc[0] += v[0];
                    acc[1] += v[1];
                    acc[2] += v[2];
                }
                next[(y * nw + x) as usize] = [acc[0] / 4.0, acc[1] / 4.0, acc[2] / 4.0];
            }
        }
        vecs = next;
        cw = nw;
        ch = nh;

        let (data, lw, lh) = &mut chain[level];
        if *lw != cw || *lh != ch {
            // The two chains fell out of step; stop rather than write variance
            // into the wrong texels.
            break;
        }
        for i in 0..(cw * ch) as usize {
            let v = vecs[i];
            let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            let sigma2 = if len > 1e-4 { (1.0 - len) / len } else { 1.0 };
            let kernel = (2.0 * sigma2).min(TERRAIN_NORMAL_VARIANCE_CLAMP);
            let base = data[i * 4] as f32 / 255.0;
            let filtered = (base * base + kernel).clamp(0.0, 1.0).sqrt();
            let byte = (filtered * 255.0).round().clamp(0.0, 255.0) as u8;
            data[i * 4] = byte;
            data[i * 4 + 1] = byte;
            data[i * 4 + 2] = byte;
        }
    }
    chain
}

pub fn mip_chain_linear(rgba: &[u8], w: u32, h: u32, levels: u32) -> Vec<(Vec<u8>, u32, u32)> {
    let mut out = vec![(rgba.to_vec(), w, h)];
    for _ in 1..levels {
        let (src, sw, sh) = out.last().unwrap();
        let (dw, dh) = ((sw / 2).max(1), (sh / 2).max(1));
        let mut dst = vec![0u8; (dw * dh * 4) as usize];
        for y in 0..dh {
            for x in 0..dw {
                for c in 0..4 {
                    let fetch = |sx: u32, sy: u32| -> f32 {
                        let i = ((sy.min(sh - 1) * sw + sx.min(sw - 1)) * 4 + c) as usize;
                        src[i] as f32 / 255.0
                    };
                    let (x0, y0) = (x * 2, y * 2);
                    let avg = (fetch(x0, y0) + fetch(x0 + 1, y0) + fetch(x0, y0 + 1)
                        + fetch(x0 + 1, y0 + 1))
                        * 0.25;
                    dst[((y * dw + x) * 4 + c) as usize] =
                        (avg.clamp(0.0, 1.0) * 255.0).round() as u8;
                }
            }
        }
        out.push((dst, dw, dh));
    }
    out
}

pub fn mip_chain(rgba: &[u8], w: u32, h: u32, levels: u32) -> Vec<(Vec<u8>, u32, u32)> {
    fn to_linear(c: u8) -> f32 {
        let s = c as f32 / 255.0;
        if s <= 0.040_45 { s / 12.92 } else { ((s + 0.055) / 1.055).powf(2.4) }
    }
    fn to_srgb(l: f32) -> u8 {
        let s = if l <= 0.003_130_8 { l * 12.92 } else { 1.055 * l.powf(1.0 / 2.4) - 0.055 };
        (s.clamp(0.0, 1.0) * 255.0).round() as u8
    }

    let mut out = vec![(rgba.to_vec(), w, h)];
    for _ in 1..levels {
        let (src, sw, sh) = out.last().unwrap();
        let (dw, dh) = ((sw / 2).max(1), (sh / 2).max(1));
        let mut dst = vec![0u8; (dw * dh * 4) as usize];
        for y in 0..dh {
            for x in 0..dw {
                for c in 0..4 {
                    // Alpha is linear already and must NOT go through the sRGB
                    // transfer; only the colour channels do.
                    let fetch = |sx: u32, sy: u32| -> f32 {
                        let i = ((sy.min(sh - 1) * sw + sx.min(sw - 1)) * 4 + c) as usize;
                        if c == 3 { src[i] as f32 / 255.0 } else { to_linear(src[i]) }
                    };
                    let (x0, y0) = (x * 2, y * 2);
                    let avg = (fetch(x0, y0) + fetch(x0 + 1, y0) + fetch(x0, y0 + 1)
                        + fetch(x0 + 1, y0 + 1))
                        * 0.25;
                    dst[((y * dw + x) * 4 + c) as usize] = if c == 3 {
                        (avg.clamp(0.0, 1.0) * 255.0).round() as u8
                    } else {
                        to_srgb(avg)
                    };
                }
            }
        }
        out.push((dst, dw, dh));
    }
    out
}

impl TerrainMaterial {
    /// Build from four RGBA8 layer images plus a macro image.
    ///
    /// Every layer must be the same size -- they go into one D2Array, which is
    /// what lets the shader index a layer by splat weight without a branch per
    /// layer. `layers` shorter than 4 is padded by repeating the last one, so a
    /// scene that only authors ground and rock still binds a complete array
    /// rather than failing validation.
    pub fn new(
        device: &Device,
        queue: &Queue,
        layout: &BindGroupLayout,
        layers: &[TerrainImage],
        macro_image: &TerrainImage,
        // Authored blend weights, or None to leave the shader on its slope- and
        // height-driven blend. None still binds a texture -- see the layout --
        // and clears `use_splat` so nothing reads it.
        splat: Option<&TerrainImage>,
        sky_occlusion: Option<&TerrainImage>,
        // Per-layer normal maps, in the same slot order as `layers`. Shorter or
        // sparser than four is fine; the gaps become flat, which the shader
        // treats as no perturbation at all.
        normals: &[Option<TerrainImage>],
        // Per-layer roughness and occlusion, same slot order. Gaps take the
        // neutral value for each -- `DEFAULT_TERRAIN_ROUGHNESS` and white --
        // so a project that has not authored them behaves as it always did.
        rough: &[Option<TerrainImage>],
        ao: &[Option<TerrainImage>],
        settings: TerrainMaterialUniform,
    ) -> Self {
        Self::new_with_shape(device, queue, layout, layers, macro_image, splat, sky_occlusion, normals, rough, ao, settings, None)
    }

    /// [`Self::new`], with the ground's step-1 shape (`ground_shape`) for its
    /// readers to take their normal and face from; `None` binds a
    /// placeholder and leaves them on the vertices.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_shape(
        device: &Device,
        queue: &Queue,
        layout: &BindGroupLayout,
        layers: &[TerrainImage],
        macro_image: &TerrainImage,
        splat: Option<&TerrainImage>,
        sky_occlusion: Option<&TerrainImage>,
        normals: &[Option<TerrainImage>],
        rough: &[Option<TerrainImage>],
        ao: &[Option<TerrainImage>],
        settings: TerrainMaterialUniform,
        shape: Option<&super::ground_shape::GroundShape>,
    ) -> Self {
        assert!(!layers.is_empty(), "terrain material needs at least one layer");
        let (w, h) = (layers[0].width, layers[0].height);
        assert!(
            layers.iter().all(|l| l.width == w && l.height == h),
            "all terrain layers must share one size to live in a D2Array",
        );

        // MIPMAPPED, and this is not a nicety.
        //
        // Ground is the one surface in the level that is always viewed at a
        // grazing angle running to the horizon, so a screen pixel a few metres
        // out covers many texels. With a single mip level the hardware picks
        // one of them, and which one changes as the head moves: the grass
        // crawls and sparkles. The sampler already asked for
        // `mipmap_filter: Linear` -- it just had nothing to filter between,
        // which is a silent no-op rather than an error.
        //
        // It is worth being precise about why this surfaced now: the layer
        // tiling used to be 8 m, and at that scale the ground was blurry enough
        // to hide the aliasing. Setting it to the material's authored 2 m
        // quadrupled the spatial frequency and made a pre-existing bug visible.
        // The tiling is right; the missing mips were always wrong.
        let mip_level_count = mip_levels_for(w, h);
        let array = device.create_texture(&TextureDescriptor {
            label: Some("terrain_layers"),
            size: Extent3d { width: w, height: h, depth_or_array_layers: 4 },
            mip_level_count,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8UnormSrgb,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });
        for slot in 0..4 {
            let src = layers.get(slot).unwrap_or_else(|| layers.last().unwrap());
            let chain = mip_chain(&src.rgba, w, h, mip_level_count);
            for (level, (data, lw, lh)) in chain.iter().enumerate() {
                queue.write_texture(
                    TexelCopyTextureInfo {
                        texture: &array,
                        mip_level: level as u32,
                        origin: Origin3d { x: 0, y: 0, z: slot as u32 },
                        aspect: TextureAspect::All,
                    },
                    data,
                    TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(4 * lw),
                        rows_per_image: Some(*lh),
                    },
                    Extent3d { width: *lw, height: *lh, depth_or_array_layers: 1 },
                );
            }
        }

        let macro_tex = device.create_texture(&TextureDescriptor {
            label: Some("terrain_macro"),
            size: Extent3d { width: macro_image.width, height: macro_image.height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            // NOT sRGB: this is a modulation factor, not a colour. Decoding it
            // through sRGB would bend the neutral point away from 0.5 and tint
            // the whole terrain.
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            TexelCopyTextureInfo {
                texture: &macro_tex,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            &macro_image.rgba,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * macro_image.width),
                rows_per_image: Some(macro_image.height),
            },
            Extent3d { width: macro_image.width, height: macro_image.height, depth_or_array_layers: 1 },
        );

        // Normal maps live in their own array, sized to the colour layers so
        // both index by the same slot. NOT sRGB: these encode a direction, and
        // decoding them through a colour curve bends every bump.
        // MIPPED, and that was the bug.
        //
        // The colour array above has had a mip chain all along; this one was
        // created with `mip_level_count: 1` while the shared sampler asks for
        // `mipmap_filter: Linear` and 8x anisotropy -- both silent no-ops
        // against a single level. So every distant pixel POINT-SAMPLED a
        // full-resolution normal map, the shading normal changed randomly from
        // pixel to pixel, and head tracking moved that around every frame.
        // That is the terrain shimmer, and it is why brush surfaces -- whose
        // normals are mipped -- never had it (headset, 2026-09-22).
        //
        // It also silently defeated the Toksvig term added just before this:
        // that measures how short the AVERAGED normal is, and with no mip
        // chain nothing is ever averaged, so the variance it read was always
        // zero. The method was right and had nothing to measure.
        let normal_array = device.create_texture(&TextureDescriptor {
            label: Some("terrain_normals"),
            size: Extent3d { width: w, height: h, depth_or_array_layers: 4 },
            mip_level_count,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });
        for slot in 0..4 {
            let flat = solid_image(FLAT_NORMAL, w, h);
            let src = match normals.get(slot).and_then(|n| n.as_ref()) {
                Some(img) if img.width == w && img.height == h => img.clone_image(),
                Some(img) => resample(img, w, h),
                None => flat,
            };
            // LINEAR, not sRGB: a normal is a direction.
            let chain = mip_chain_linear(&src.rgba, w, h, mip_level_count);
            for (level, (data, lw, lh)) in chain.iter().enumerate() {
                queue.write_texture(
                    TexelCopyTextureInfo {
                        texture: &normal_array,
                        mip_level: level as u32,
                        origin: Origin3d { x: 0, y: 0, z: slot as u32 },
                        aspect: TextureAspect::All,
                    },
                    data,
                    TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(4 * lw),
                        rows_per_image: Some(*lh),
                    },
                    Extent3d { width: *lw, height: *lh, depth_or_array_layers: 1 },
                );
            }
        }

        // ROUGHNESS AND OCCLUSION, mipped and linear like the normal array.
        //
        // Both are measurements rather than colours, so neither goes through
        // the sRGB transfer, and both need a mip chain for the same reason the
        // normals did: a distant pixel that point-samples a full-resolution
        // map gets a different answer every frame as the head moves.
        //
        // The roughness chain also folds in the NORMAL'S lost variation, which
        // is the same treatment brush materials get -- mipping a normal map
        // averages its normals, and the shortness of that average measures how
        // much detail the level threw away. Putting it back as roughness is
        // what keeps a highlight from popping in and out at distance.
        let mut rough_arr = Vec::with_capacity(4);
        let mut ao_arr = Vec::with_capacity(4);
        for slot in 0..4 {
            let pick = |src: &[Option<TerrainImage>], fallback: u8| -> TerrainImage {
                match src.get(slot).and_then(|x| x.as_ref()) {
                    Some(img) if img.width == w && img.height == h => img.clone_image(),
                    Some(img) => resample(img, w, h),
                    None => solid_image([fallback; 3], w, h),
                }
            };
            rough_arr.push(pick(rough, DEFAULT_TERRAIN_ROUGHNESS));
            ao_arr.push(pick(ao, 255));
        }

        let data_array = |label: &str| {
            device.create_texture(&TextureDescriptor {
                label: Some(label),
                size: Extent3d { width: w, height: h, depth_or_array_layers: 4 },
                mip_level_count,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba8Unorm,
                usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
                view_formats: &[],
            })
        };
        let rough_array = data_array("terrain_rough");
        let ao_array = data_array("terrain_ao");

        for slot in 0..4 {
            let normal_src = match normals.get(slot).and_then(|n| n.as_ref()) {
                Some(img) if img.width == w && img.height == h => img.clone_image(),
                Some(img) => resample(img, w, h),
                None => solid_image(FLAT_NORMAL, w, h),
            };
            let chains = [
                (
                    &rough_array,
                    roughness_chain_with_normal_variance(
                        &rough_arr[slot].rgba,
                        &normal_src.rgba,
                        w,
                        h,
                        mip_level_count,
                    ),
                ),
                (&ao_array, mip_chain_linear(&ao_arr[slot].rgba, w, h, mip_level_count)),
            ];
            for (tex, chain) in chains {
                for (level, (data, lw, lh)) in chain.iter().enumerate() {
                    queue.write_texture(
                        TexelCopyTextureInfo {
                            texture: tex,
                            mip_level: level as u32,
                            origin: Origin3d { x: 0, y: 0, z: slot as u32 },
                            aspect: TextureAspect::All,
                        },
                        data,
                        TexelCopyBufferLayout {
                            offset: 0,
                            bytes_per_row: Some(4 * lw),
                            rows_per_image: Some(*lh),
                        },
                        Extent3d { width: *lw, height: *lh, depth_or_array_layers: 1 },
                    );
                }
            }
        }

        let sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("terrain_sampler"),
            address_mode_u: AddressMode::Repeat,
            address_mode_v: AddressMode::Repeat,
            address_mode_w: AddressMode::Repeat,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: MipmapFilterMode::Linear,
            // Ground is seen almost entirely at grazing angles, which is the
            // exact case trilinear filtering handles worst: it picks a mip for
            // the SHORT axis of the footprint and so over-blurs along the long
            // one. 8x is the usual sweet spot and is cheap on a tile GPU
            // compared with the fill it saves by not needing a sharper mip.
            anisotropy_clamp: 8,
            ..Default::default()
        });

        // RGBA8 unorm, NOT sRGB: these are blend weights, and sRGB-decoding
        // them would bend a 50/50 blend away from the middle -- the same reason
        // the macro texture is linear.
        let splat_image = splat.unwrap_or(&NO_SPLAT);
        let splat_tex = device.create_texture(&TextureDescriptor {
            label: Some("terrain_splat"),
            size: Extent3d {
                width: splat_image.width,
                height: splat_image.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            TexelCopyTextureInfo {
                texture: &splat_tex,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            &splat_image.rgba,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * splat_image.width),
                rows_per_image: Some(splat_image.height),
            },
            Extent3d {
                width: splat_image.width,
                height: splat_image.height,
                depth_or_array_layers: 1,
            },
        );

        // Sky visibility over the footprint. WHITE when unbaked: this value is
        // multiplied into the ambient term, so 1.0 is the neutral that
        // reproduces the shading terrain had before the map existed.
        //
        // AN ARRAY: layer 0 is the ground's own map, and any further layers
        // the image's bytes carry -- whole images of the same size, back to
        // back -- are the stationary lamps' masks (`with_stationary_masks`).
        let occ_image = sky_occlusion.unwrap_or(&FULL_SKY);
        let occ_layers = (occ_image.rgba.len() / (4 * occ_image.width as usize * occ_image.height as usize).max(1)).max(1) as u32;
        let sky_occ_tex = device.create_texture(&TextureDescriptor {
            label: Some("terrain_sky_occlusion"),
            size: Extent3d {
                width: occ_image.width,
                height: occ_image.height,
                depth_or_array_layers: occ_layers,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            // NOT sRGB: a visibility fraction, not a colour. Decoding it would
            // bend every value and darken the ground non-linearly.
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            TexelCopyTextureInfo {
                texture: &sky_occ_tex,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            &occ_image.rgba[..(4 * occ_image.width * occ_image.height * occ_layers) as usize],
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * occ_image.width),
                rows_per_image: Some(occ_image.height),
            },
            Extent3d {
                width: occ_image.width,
                height: occ_image.height,
                depth_or_array_layers: occ_layers,
            },
        );

        // The flag is derived here rather than left to the caller: a material
        // built with a map that the shader then ignores, or without one that it
        // then reads, are both silent and both look like a shader bug.
        let mut settings = settings;
        settings.use_splat = if splat.is_some() { 1.0 } else { 0.0 };

        // THE GROUND'S SHAPE, or one texel and no shape.
        let placeholder = super::ground_shape::GroundShape { size: [1, 1], texels: vec![[0.0, 1.0, 0.0, 0.0]], ..Default::default() };
        let shape_image = shape.filter(|s| s.size[0] >= 2 && s.size[1] >= 2).unwrap_or(&placeholder);
        settings.shape = match shape {
            Some(s) if s.size[0] >= 2 && s.size[1] >= 2 => [s.uv_cells[0], s.uv_cells[1], 1.0 / s.cell[0], 1.0 / s.cell[1]],
            _ => [0.0; 4],
        };
        let shape_extent = Extent3d { width: shape_image.size[0], height: shape_image.size[1], depth_or_array_layers: 1 };
        let shape_tex = device.create_texture(&TextureDescriptor {
            label: Some("terrain_ground_shape"),
            size: shape_extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba32Float,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            TexelCopyTextureInfo { texture: &shape_tex, mip_level: 0, origin: Origin3d::ZERO, aspect: TextureAspect::All },
            shape_image.bytes(),
            TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(16 * shape_image.size[0]), rows_per_image: Some(shape_image.size[1]) },
            shape_extent,
        );

        let uniform = device.create_buffer(&BufferDescriptor {
            label: Some("terrain_material_uniform"),
            size: std::mem::size_of::<TerrainMaterialUniform>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform, 0, bytemuck::bytes_of(&settings));

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("terrain_material"),
            layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(
                        &array.create_view(&TextureViewDescriptor {
                            dimension: Some(TextureViewDimension::D2Array),
                            ..Default::default()
                        }),
                    ),
                },
                BindGroupEntry { binding: 1, resource: BindingResource::Sampler(&sampler) },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(
                        &macro_tex.create_view(&TextureViewDescriptor::default()),
                    ),
                },
                BindGroupEntry { binding: 3, resource: uniform.as_entire_binding() },
                BindGroupEntry {
                    binding: 4,
                    resource: BindingResource::TextureView(
                        &splat_tex.create_view(&TextureViewDescriptor::default()),
                    ),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: BindingResource::TextureView(
                        &normal_array.create_view(&TextureViewDescriptor {
                            dimension: Some(TextureViewDimension::D2Array),
                            ..Default::default()
                        }),
                    ),
                },
                BindGroupEntry {
                    binding: 6,
                    resource: BindingResource::TextureView(&sky_occ_tex.create_view(&TextureViewDescriptor {
                        dimension: Some(TextureViewDimension::D2Array),
                        ..Default::default()
                    })),
                },
                BindGroupEntry {
                    binding: 7,
                    resource: BindingResource::TextureView(
                        &rough_array.create_view(&TextureViewDescriptor {
                            dimension: Some(TextureViewDimension::D2Array),
                            ..Default::default()
                        }),
                    ),
                },
                BindGroupEntry {
                    binding: 8,
                    resource: BindingResource::TextureView(
                        &ao_array.create_view(&TextureViewDescriptor {
                            dimension: Some(TextureViewDimension::D2Array),
                            ..Default::default()
                        }),
                    ),
                },
                BindGroupEntry { binding: 9, resource: BindingResource::TextureView(&shape_tex.create_view(&TextureViewDescriptor::default())) },
            ],
        });

        Self { bind_group, uniform }
    }

    /// A material with no authored textures: flat colours per layer.
    ///
    /// This exists so terrain can go through the real pipeline from day one. A
    /// renderer path that only works once an artist has produced four tiling
    /// textures stays unwired and therefore unverified for as long as that
    /// takes, and the wiring bugs all surface later, at once, blamed on the art.
    /// The colours are the ones the previous flat-shaded terrain used, so
    /// turning this on changes shading but not palette.
    pub fn fallback(device: &Device, queue: &Queue, layout: &BindGroupLayout) -> Self {
        Self::fallback_with_splat(device, queue, layout, None, None)
    }

    /// The fallback palette, with authored blend weights applied to it.
    ///
    /// This is the state a level reaches first: someone has painted WHERE each
    /// material goes long before an artist has produced what each material
    /// looks like. Showing painted regions in flat colour is the honest render
    /// of that, and it matches what the editor previews.
    pub fn fallback_with_splat(
        device: &Device,
        queue: &Queue,
        layout: &BindGroupLayout,
        splat: Option<&TerrainImage>,
        sky_occlusion: Option<&TerrainImage>,
    ) -> Self {
        Self::from_layers(
            device, queue, layout, &[None, None, None, None], &[None, None, None, None], splat,
            sky_occlusion,
        )
    }

    /// Build from whatever layer textures loaded, filling the gaps.
    ///
    /// Per LAYER rather than all-or-nothing: a project part-way through
    /// authoring its materials should see the layers it has rather than lose
    /// all four because one file is missing.
    ///
    /// Gaps become solid images at the SAME SIZE as the loaded layers, not 1x1.
    /// The four share one D2Array, so a mismatched layer is a validation
    /// failure rather than a smaller texture -- and that failure would appear
    /// only on the project that happens to be missing one file.
    pub fn from_layers(
        device: &Device,
        queue: &Queue,
        layout: &BindGroupLayout,
        layers: &[Option<TerrainImage>],
        normals: &[Option<TerrainImage>],
        splat: Option<&TerrainImage>,
        sky_occlusion: Option<&TerrainImage>,
    ) -> Self {
        Self::from_layers_with(
            device,
            queue,
            layout,
            layers,
            normals,
            // This convenience form predates roughness and occlusion maps and
            // keeps its old shape: both fall back to their neutral, so callers
            // that have not been updated behave exactly as they did.
            &[],
            &[],
            splat,
            sky_occlusion,
            TerrainMaterialUniform::default(),
        )
    }

    /// The same, with the project's authored material settings.
    ///
    /// Split from `from_layers` rather than replacing it because the defaults
    /// are the right answer for every caller that has no project on disk to
    /// read from -- the fallback material and the pipeline's own tests.
    pub fn from_layers_with(
        device: &Device,
        queue: &Queue,
        layout: &BindGroupLayout,
        layers: &[Option<TerrainImage>],
        normals: &[Option<TerrainImage>],
        rough: &[Option<TerrainImage>],
        ao: &[Option<TerrainImage>],
        splat: Option<&TerrainImage>,
        sky_occlusion: Option<&TerrainImage>,
        settings: TerrainMaterialUniform,
    ) -> Self {
        Self::from_layers_with_shape(device, queue, layout, layers, normals, rough, ao, splat, sky_occlusion, settings, None)
    }

    /// [`Self::from_layers_with`], with the ground's step-1 shape: see
    /// [`Self::new_with_shape`].
    #[allow(clippy::too_many_arguments)]
    pub fn from_layers_with_shape(
        device: &Device,
        queue: &Queue,
        layout: &BindGroupLayout,
        layers: &[Option<TerrainImage>],
        normals: &[Option<TerrainImage>],
        rough: &[Option<TerrainImage>],
        ao: &[Option<TerrainImage>],
        splat: Option<&TerrainImage>,
        sky_occlusion: Option<&TerrainImage>,
        settings: TerrainMaterialUniform,
        shape: Option<&super::ground_shape::GroundShape>,
    ) -> Self {
        let (w, h) = layers
            .iter()
            .flatten()
            .map(|l| (l.width, l.height))
            .next()
            .unwrap_or((1, 1));

        let filled: Vec<TerrainImage> = FALLBACK_LAYER_COLOURS
            .iter()
            .enumerate()
            .map(|(i, rgb)| match layers.get(i).and_then(|l| l.as_ref()) {
                // Already the right size by construction when it is the one the
                // size came from; resampled otherwise so a set of mixed-size
                // files still binds.
                Some(img) if img.width == w && img.height == h => TerrainImage {
                    width: img.width,
                    height: img.height,
                    rgba: img.rgba.clone(),
                },
                Some(img) => resample(img, w, h),
                None => solid_image(*rgb, w, h),
            })
            .collect();

        Self::new_with_shape(
            device,
            queue,
            layout,
            &filled,
            &solid_image([128, 128, 128], 1, 1),
            splat,
            sky_occlusion,
            normals,
            rough,
            ao,
            settings,
            shape,
        )
    }

}

/// A decoded RGBA8 image destined for a texture array.
///
/// Named for terrain because that is what first needed it; brush materials load
/// through the same type rather than through a copy of it. A second decode-and-
/// resample path is exactly the shape of duplication this codebase has been
/// bitten by before -- one copy gets fixed and the other keeps the bug.
pub struct TerrainImage {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

impl TerrainImage {
    /// The ground's own map with the stationary lamps' masks after it, as the
    /// layers of one array -- see `TerrainMaterial::new`. `None` when a mask
    /// is not the map's size (a bake from before they shared its grid): the
    /// lamps then light the ground unshadowed, as they always had.
    pub fn with_stationary_masks(&self, masks: &[TerrainImage]) -> Option<TerrainImage> {
        if masks.iter().any(|m| m.width != self.width || m.height != self.height || m.rgba.len() != self.rgba.len()) {
            return None;
        }
        let mut rgba = self.rgba.clone();
        for m in masks {
            rgba.extend_from_slice(&m.rgba);
        }
        Some(TerrainImage { width: self.width, height: self.height, rgba })
    }

    /// Whether this GROUND MAP carries a baked sun over every texel: alpha
    /// under a half everywhere (the bake writes 0; a map from before the sun
    /// was baked beside the sky is 255). Then every read of it -- any blend of
    /// texels, at any mip -- is under a half too, `fs_main` never sets
    /// `receiver_sun_mask` to the -1 of "no bake", and the ground can be drawn
    /// by its baked reader ([`TerrainPipeline::new_probe_reader_twins`]).
    pub fn sun_baked_everywhere(&self) -> bool {
        !self.rgba.is_empty() && self.rgba.chunks_exact(4).all(|texel| texel[3] < 128)
    }
}

#[cfg(test)]
mod material_tests {
    use super::tests::*;

    /// A STATIONARY LAMP ON THE GROUND TAKES ITS SHADOW FROM THE MASK LAYER
    /// after the ground map: channel 0 fully shadowed leaves the ground as dark
    /// as with no lamp at all, fully lit as bright as an unmasked lamp. Before
    /// the masks, the ground took every lamp through the walls.
    #[test]
    fn a_stationary_lamp_on_the_ground_reads_its_mask() {
        let map = super::TerrainImage { width: 1, height: 1, rgba: vec![255, 128, 0, 255] };
        let mask = |d: u8| super::TerrainImage { width: 1, height: 1, rgba: vec![d, 0, 255, 255] };
        let layered = |d: u8| map.with_stationary_masks(&[mask(d)]).unwrap();
        // Flat ground, facing the harness's lamp up and to one side.
        let up = [0.0, 1.0, 0.0];
        let render = |sky: &super::TerrainImage, light: Option<Option<u8>>| {
            render_quad_lit_by(up, Palette::Test, None, Some(sky), &[None, None, None, None], light)
        };
        let Some(dark) = render(&map, None) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let lamp = render(&map, Some(None)).unwrap();
        let shadowed = render(&layered(0), Some(Some(0))).unwrap();
        let lit = render(&layered(255), Some(Some(0))).unwrap();
        assert!(lamp[0] > dark[0] + 10, "the test lamp does not light the ground: {lamp:?} vs {dark:?}");
        assert_eq!(shadowed, dark, "a lamp its mask hides still lit the ground");
        assert_eq!(lit, lamp, "a lamp its mask shows is not the unmasked lamp");
        // And a map with no mask layers is the ground as it always was.
        let other_channel = render(&layered(0), Some(Some(1))).unwrap();
        assert_eq!(other_channel, lamp, "a lamp on another channel read this one's shadow");
    }
    use super::*;

    /// The fallback material must produce the SAME layer selection as the
    /// hand-built test material: flat ground takes layer 0, a cliff takes layer
    /// 1. Colours differ (the fallback ships real terrain colours, the test
    /// material ships primaries), so this asserts on which layer won, via the
    /// channel that separates them, rather than on exact bytes.
    #[test]
    fn the_fallback_material_selects_layers_the_same_way() {
        let Some(flat) = render_quad_with_fallback_material([0.0, 1.0, 0.0]) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let cliff = render_quad_with_fallback_material([1.0, 0.05, 0.0]).unwrap();

        // Ground (86,112,62) is green-dominant; rock (104,100,94) is near-grey.
        //
        // The margin is smaller than it looks it should be because this quad
        // renders dark, and a tone curve compresses hardest at the bottom --
        // ACES puts 18% grey at 0.106. The ORDERING this test cares about is
        // untouched; only the absolute byte gap shrank, so the threshold is
        // stated against the range the pixels actually occupy rather than the
        // untonemapped one it was first calibrated in.
        let greenness = |p: [u8; 4]| p[1] as i32 - p[2] as i32;
        assert!(
            greenness(flat) > greenness(cliff) + 4 && greenness(flat) > 4,
            "flat ground should read greener than a cliff: flat {flat:?} cliff {cliff:?}",
        );
        assert!(flat[3] == 255 && cliff[3] == 255, "terrain must be opaque");
    }

    /// A material built with fewer than four layers still binds a complete
    /// D2Array. Without the padding this is a validation error at bind-group
    /// creation, which is the kind of thing that only shows up on the scene
    /// that happens to author two layers.
    #[test]
    fn a_short_layer_list_pads_to_a_full_array() {
        let Some((device, queue)) = headless_gpu() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let layout = material_bind_group_layout(&device);
        let one = TerrainImage { width: 1, height: 1, rgba: vec![10, 20, 30, 255] };
        let err_scope_1 = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let _m = TerrainMaterial::new(
            &device, &queue, &layout,
            std::slice::from_ref(&one),
            &one,
            None,
            None,
            &[None, None, None, None],
            &[],
            &[],
            TerrainMaterialUniform::default(),
        );
        assert!(
            pollster::block_on(err_scope_1.pop()).is_none(),
            "a single-layer terrain material must still bind validly",
        );
    }
}

/// The 1x1 stand-in bound when a scene authors no splat map.
///
/// Its contents are never read -- `use_splat` is 0 -- but something has to
/// satisfy the binding, and a shared constant is cheaper than each caller
/// inventing one.
static NO_SPLAT: std::sync::LazyLock<TerrainImage> = std::sync::LazyLock::new(|| TerrainImage {
    width: 1,
    height: 1,
    rgba: vec![255, 0, 0, 0],
});

/// The 1x1 stand-in bound when a scene has no baked terrain occlusion.
///
/// WHITE, and unlike `NO_SPLAT` its contents ARE read: sky visibility is
/// multiplied into the ambient term, so 1.0 means "sees the whole sky" and
/// reproduces exactly the shading terrain had before this map existed. Black
/// here would put every terrain in the game in permanent night.
static FULL_SKY: std::sync::LazyLock<TerrainImage> = std::sync::LazyLock::new(|| TerrainImage {
    width: 1,
    height: 1,
    rgba: vec![255, 255, 255, 255],
});

#[cfg(test)]
mod authored_splat_tests {
    use super::tests::*;
    use super::*;

    fn splat(rgba: [u8; 4]) -> TerrainImage {
        TerrainImage { width: 1, height: 1, rgba: rgba.to_vec() }
    }

    const FLAT: [f32; 3] = [0.0, 1.0, 0.0];
    const CLIFF: [f32; 3] = [1.0, 0.05, 0.0];

    /// The point of the whole feature: what an author painted wins over what
    /// the slope rule would have chosen. Flat ground takes layer 0 (red) by
    /// slope, so a map demanding layer 1 must come back blue.
    #[test]
    fn authored_weights_override_the_slope_blend() {
        let Some(unauthored) = render_quad_with_normal(FLAT) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        assert!(unauthored[0] > unauthored[2], "flat ground should be red without a map: {unauthored:?}");

        let painted = render_quad_with_splat(FLAT, &splat([0, 255, 0, 0])).unwrap();
        assert!(
            painted[2] > painted[0],
            "a map demanding layer 1 must beat the slope rule on flat ground: {painted:?}",
        );
    }

    /// And the other direction, so the test cannot pass by the map simply
    /// being ignored in one particular case.
    #[test]
    fn authored_weights_override_a_cliff_too() {
        let Some(unauthored) = render_quad_with_normal(CLIFF) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        assert!(unauthored[2] > unauthored[0], "a cliff should be blue without a map: {unauthored:?}");

        let painted = render_quad_with_splat(CLIFF, &splat([255, 0, 0, 0])).unwrap();
        assert!(
            painted[0] > painted[2],
            "a map demanding layer 0 must beat the slope rule on a cliff: {painted:?}",
        );
    }

    /// A blend, not a winner-takes-all. Half layer 0 (red) and half layer 2
    /// (green) must show both channels -- if the shader picked a dominant layer
    /// instead of summing weights, one of these would be zero.
    #[test]
    fn a_mixed_texel_blends_rather_than_picking_a_winner() {
        let Some(mixed) = render_quad_with_splat(FLAT, &splat([128, 0, 128, 0])) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        assert!(mixed[0] > 30, "layer 0 (red) should contribute: {mixed:?}");
        assert!(mixed[1] > 30, "layer 2 (green) should contribute: {mixed:?}");
        assert!(mixed[2] < mixed[0], "layer 1 (blue) was not painted: {mixed:?}");
    }

    /// Weights that do not sum to full are normalised rather than trusted. The
    /// editor keeps them summing to 255, but a hand-made or half-written file
    /// has no such guarantee, and unnormalised weights would blow out or black
    /// out the ground instead of merely looking slightly wrong.
    #[test]
    fn unnormalised_weights_are_normalised_not_amplified() {
        // Both of these mean "half layer 0, half layer 2" once normalised, but
        // one sums to 255 and the other to 510.
        let Some(normal_sum) = render_quad_with_splat(FLAT, &splat([128, 0, 128, 0])) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let double_sum = render_quad_with_splat(FLAT, &splat([255, 0, 255, 0])).unwrap();

        for channel in 0..3 {
            let delta = normal_sum[channel].abs_diff(double_sum[channel]);
            assert!(
                delta <= 4,
                "weights summing to 510 must shade like weights summing to 255 \
                 (channel {channel}: {normal_sum:?} vs {double_sum:?})",
            );
        }
    }

    /// An all-zero texel cannot divide by zero and black out the ground.
    #[test]
    fn an_empty_texel_does_not_produce_a_divide_by_zero() {
        let Some(px) = render_quad_with_splat(FLAT, &splat([0, 0, 0, 0])) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        assert_eq!(px[3], 255, "terrain must stay opaque: {px:?}");
        for channel in 0..3 {
            assert!(px[channel] < 250, "an empty texel must not blow out: {px:?}");
        }
    }
}

impl TerrainImage {
    /// Decode an image file into the RGBA8 a terrain layer needs.
    ///
    /// Returns `None` rather than failing the frame when the file is missing or
    /// unreadable: terrain that renders in flat fallback colours is diagnosable
    /// from the log, and a client that refuses to start because one texture is
    /// absent is not. That is the same call `terrain_render::load` makes about
    /// missing ground.
    pub fn load(path: &std::path::Path) -> Option<Self> {
        let decoded = match ::image::open(path) {
            Ok(d) => d.to_rgba8(),
            Err(e) => {
                log::warn!("terrain texture {}: {e}", path.display());
                return None;
            }
        };
        Some(Self {
            width: decoded.width(),
            height: decoded.height(),
            rgba: decoded.into_raw(),
        })
    }
}

/// The four layer textures a terrain material wants, by role.
///
/// Named by ROLE rather than by what they depict, matching the files on disk,
/// so replacing what "rock" looks like is a file swap and touches no code. The
/// order is the shader's fixed slot order and is not rearrangeable.
pub const TERRAIN_LAYER_FILES: [&str; 4] = ["ground.jpg", "rock.jpg", "high.jpg", "sediment.jpg"];

/// Normal maps for the same four layers, in the same order.
///
/// Suffixed rather than kept in a subdirectory so a layer's files sort together
/// and it is obvious at a glance which layers have normals and which do not.
pub const TERRAIN_NORMAL_FILES: [&str; 4] =
    ["ground_n.jpg", "rock_n.jpg", "high_n.jpg", "sediment_n.jpg"];

/// Roughness maps for the same four layers, in the same order.
///
/// Ground is not one gloss. Wet rock, dry gravel and grass differ, and a single
/// constant made every layer the same material under a different picture --
/// which is what a terrain looks like next to a brush surface that has a map.
pub const TERRAIN_ROUGH_FILES: [&str; 4] =
    ["ground_r.jpg", "rock_r.jpg", "high_r.jpg", "sediment_r.jpg"];

/// Ambient occlusion for the same four layers, in the same order.
pub const TERRAIN_AO_FILES: [&str; 4] =
    ["ground_ao.jpg", "rock_ao.jpg", "high_ao.jpg", "sediment_ao.jpg"];

/// Load the terrain layer set from a directory, falling back per layer.
///
/// Per LAYER, not all-or-nothing: a project part-way through authoring its
/// materials should see the layers it has, not lose all four because one is
/// missing. Each gap keeps that layer's flat colour, which is exactly what the
/// whole terrain looked like before any textures existed.
///
/// Sizes are normalised to the first successfully loaded layer, because they
/// share one D2Array and a mismatched layer would otherwise fail validation.
pub fn load_terrain_layers(dir: &std::path::Path) -> Vec<Option<TerrainImage>> {
    TERRAIN_LAYER_FILES
        .iter()
        .map(|name| TerrainImage::load(&dir.join(name)))
        .collect()
}

/// Load the four roughness maps, per layer, falling back to none.
///
/// A missing map is ordinary and not an error: that layer takes
/// `DEFAULT_TERRAIN_ROUGHNESS`, which is what every layer used before these
/// existed, so a project that has not authored them looks exactly as it did.
///
/// READ AS THE PROJECT READS THAT MATERIAL: a layer whose asset (`sources.json`
/// beside the maps) has a `"roughness": {"min", "max"}` in the game's
/// `material_overrides.json` takes it, as the brushes' copy of the same
/// material does (`quest_app::brush_render::roughness_range`). The rock layer
/// is Rock063, whose map ships at 0.33-0.59 -- wet, mirror-like stone, which is
/// why the hallway's walls were given 0.55-1.0 (2026-09-29). The ground's copy
/// never was: every cliff was the wet stone the walls had stopped being, bright
/// sheen down the steep faces (headset, 2026-10-13), and every rock pixel under
/// the probe pass's 0.75 took the reflection trace. Absent, a map is read as
/// it is.
pub fn load_terrain_rough(dir: &std::path::Path) -> Vec<Option<TerrainImage>> {
    TERRAIN_ROUGH_FILES
        .iter()
        .enumerate()
        .map(|(layer, name)| {
            let mut img = TerrainImage::load(&dir.join(name));
            if let (Some(img), Some((lo, hi))) = (img.as_mut(), terrain_roughness_range(dir, layer)) {
                for px in img.rgba.chunks_exact_mut(4) {
                    for c in &mut px[..3] {
                        *c = ((lo + (*c as f32 / 255.0) * (hi - lo)).clamp(0.0, 1.0) * 255.0).round() as u8;
                    }
                }
                log::info!("terrain: layer {layer}'s roughness read as {lo:.2}..{hi:.2}");
            }
            img
        })
        .collect()
}

/// What terrain layer `layer`'s roughness map's 0 and 1 stand for: its asset's
/// entry in the game's `material_overrides.json` (the game folder is two up
/// from the layer maps, `textures/terrain`), found through `sources.json`.
pub fn terrain_roughness_range(dir: &std::path::Path, layer: usize) -> Option<(f32, f32)> {
    let read = |path: std::path::PathBuf| -> Option<serde_json::Value> { serde_json::from_str(&std::fs::read_to_string(path).ok()?).ok() };
    let role = TERRAIN_LAYER_FILES.get(layer)?.strip_suffix(".jpg")?;
    let sources = read(dir.join("sources.json"))?;
    let id = sources.get(role)?.get("assetId")?.as_str()?;
    let overrides = read(dir.parent()?.parent()?.join("material_overrides.json"))?;
    let r = overrides.get(id)?.get("roughness")?;
    let lo = r.get("min")?.as_f64()? as f32;
    let hi = r.get("max")?.as_f64()? as f32;
    (lo.is_finite() && hi.is_finite()).then_some((lo.clamp(0.0, 1.0), hi.clamp(0.0, 1.0)))
}

/// Load the four occlusion maps, per layer, falling back to none.
///
/// A missing map means fully unoccluded -- white -- which is the neutral value
/// for something that MULTIPLIES, and is how terrain behaved before.
pub fn load_terrain_ao(dir: &std::path::Path) -> Vec<Option<TerrainImage>> {
    TERRAIN_AO_FILES
        .iter()
        .map(|name| TerrainImage::load(&dir.join(name)))
        .collect()
}

/// The project's terrain material settings, from `settings.json` beside the
/// layer textures.
///
/// WHY A FILE RATHER THAN A CONSTANT
///
/// `repeat` -- metres per tile, per layer -- is the control that decides whether
/// ground reads as gravel or as noise, and it is judged by eye against the art
/// that is actually installed. Baking it into the binary means the person who
/// can see the problem cannot fix it, and the person who can fix it has to
/// rebuild the engine to try a number.
///
/// Beside the textures, not in the scene: which four materials a project uses
/// and how big they tile is one art decision for the whole game, exactly like
/// the layer files themselves. Copying it into every scene would mean changing
/// "how big is our gravel" in twenty places.
///
/// ABSENT IS THE NORMAL CASE. Every project that predates this file has none,
/// and every key is independent, so a file naming only `repeat` leaves the rest
/// at their defaults rather than zeroing them. A malformed file logs and yields
/// the defaults: unreadable settings must not cost a project its terrain.
pub fn load_terrain_settings(dir: &std::path::Path) -> TerrainMaterialUniform {
    let mut out = TerrainMaterialUniform::default();
    let path = dir.join(TERRAIN_SETTINGS_FILE);
    let text = match std::fs::read_to_string(&path) {
        Ok(t) => t,
        Err(_) => return out,
    };
    let raw: serde_json::Value = match serde_json::from_str(&text) {
        Ok(v) => v,
        Err(e) => {
            log::warn!("{}: {e} -- using default terrain settings", path.display());
            return out;
        }
    };
    if let Some(values) = raw.get("repeat").and_then(|r| r.as_array()) {
        for (i, slot) in out.repeat.iter_mut().enumerate() {
            if let Some(v) = values.get(i).and_then(|v| v.as_f64()) {
                // Clamped, not rejected: a zero here divides by zero in the
                // shader, and one bad number should cost that layer its scale
                // rather than the project its ground.
                *slot = (v as f32).max(MIN_REPEAT);
            }
        }
    }
    if let Some(v) = raw.get("normal_strength").and_then(|v| v.as_f64()) {
        out.normal_strength = (v as f32).clamp(0.0, 4.0);
    }
    // How each layer takes water: see `TerrainMaterialUniform::porosity`.
    for (key, into) in [("porosity", &mut out.porosity), ("film", &mut out.film)] {
        if let Some(values) = raw.get(key).and_then(|r| r.as_array()) {
            for (i, slot) in into.iter_mut().enumerate() {
                if let Some(v) = values.get(i).and_then(|v| v.as_f64()) {
                    *slot = (v as f32).clamp(0.0, 1.0);
                }
            }
        }
    }
    out
}

/// Per-project terrain material settings, beside the layer textures.
pub const TERRAIN_SETTINGS_FILE: &str = "settings.json";

/// Smallest tile size in metres. Below this a tile is smaller than a texel of
/// anything and the ground reads as noise; at zero the shader divides by zero.
pub const MIN_REPEAT: f32 = 0.05;

/// Load the layer normal maps, per layer, from the same directory.
///
/// Missing is the NORMAL case rather than an error: a project with colour and
/// no normals is exactly what shipped before this existed, and each gap becomes
/// a flat map that perturbs nothing. `TerrainImage::load` already logs what it
/// could not read, so a typo is visible without failing the frame.
pub fn load_terrain_normals(dir: &std::path::Path) -> Vec<Option<TerrainImage>> {
    TERRAIN_NORMAL_FILES
        .iter()
        .map(|name| TerrainImage::load(&dir.join(name)))
        .collect()
}

/// Flat colours a layer falls back to when it has no texture.
///
/// The palette the terrain used before any textures existed, so a project with
/// no material art renders exactly as it always did rather than as black.
pub const FALLBACK_LAYER_COLOURS: [[u8; 3]; 4] = [
    [86, 112, 62],   // 0 ground
    [104, 100, 94],  // 1 rock
    [132, 128, 120], // 2 high ground
    [92, 84, 70],    // 3 sediment
];

pub(crate) fn solid_image(rgb: [u8; 3], width: u32, height: u32) -> TerrainImage {
    let mut rgba = Vec::with_capacity((width * height * 4) as usize);
    for _ in 0..(width * height) {
        rgba.extend_from_slice(&[rgb[0], rgb[1], rgb[2], 255]);
    }
    TerrainImage { width, height, rgba }
}

/// Nearest-neighbour resample onto a different size.
///
/// Only ever runs on a mismatched layer set, which is an authoring mistake
/// rather than a shipping configuration -- so this exists to keep such a
/// project RUNNING and legible, not to look good. Anything better would be
/// effort spent on a case the SOURCES.md tells authors to avoid.
pub(crate) fn resample(src: &TerrainImage, width: u32, height: u32) -> TerrainImage {
    let mut rgba = Vec::with_capacity((width * height * 4) as usize);
    for y in 0..height {
        let sy = (y as u64 * src.height as u64 / height.max(1) as u64) as u32;
        for x in 0..width {
            let sx = (x as u64 * src.width as u64 / width.max(1) as u64) as u32;
            let at = ((sy.min(src.height - 1) * src.width + sx.min(src.width - 1)) * 4) as usize;
            rgba.extend_from_slice(&src.rgba[at..at + 4]);
        }
    }
    TerrainImage { width, height, rgba }
}

#[cfg(test)]
mod layer_loading_tests {
    use super::tests::headless_gpu;
    use super::*;

    fn img(rgb: [u8; 3], w: u32, h: u32) -> TerrainImage {
        solid_image(rgb, w, h)
    }

    /// A project part-way through authoring its materials must see the layers
    /// it HAS, not lose all four because one file is missing.
    #[test]
    fn a_missing_layer_falls_back_without_taking_the_others_with_it() {
        let Some((device, queue)) = headless_gpu() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let layout = material_bind_group_layout(&device);

        let err_scope_2 = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let _m = TerrainMaterial::from_layers(
            &device,
            &queue,
            &layout,
            &[Some(img([10, 20, 30], 64, 64)), None, Some(img([1, 2, 3], 64, 64)), None],
            &[None, None, None, None],
            None,
            None,
        );
        assert!(
            pollster::block_on(err_scope_2.pop()).is_none(),
            "a partial layer set must still bind validly",
        );
    }

    /// The four share one D2Array, so a mismatched layer is a validation
    /// failure rather than a smaller texture -- and it would only appear on the
    /// project that happens to have mixed sizes.
    #[test]
    fn mixed_sizes_are_normalised_rather_than_failing_validation() {
        let Some((device, queue)) = headless_gpu() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let layout = material_bind_group_layout(&device);

        let err_scope_3 = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let _m = TerrainMaterial::from_layers(
            &device,
            &queue,
            &layout,
            &[
                Some(img([10, 20, 30], 64, 64)),
                Some(img([40, 50, 60], 16, 16)),
                None,
                Some(img([70, 80, 90], 128, 32)),
            ],
            &[None, None, None, None],
            None,
            None,
        );
        assert!(
            pollster::block_on(err_scope_3.pop()).is_none(),
            "mixed layer sizes must be normalised, not rejected",
        );
    }

    /// With nothing loaded the material must be exactly what it was before
    /// textures existed, so a project with no art is unchanged.
    #[test]
    fn no_layers_at_all_is_the_old_flat_palette() {
        let Some((device, queue)) = headless_gpu() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let layout = material_bind_group_layout(&device);
        let err_scope_4 = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let _m = TerrainMaterial::from_layers(
            &device, &queue, &layout, &[None, None, None, None], &[None, None, None, None], None, None,
        );
        assert!(pollster::block_on(err_scope_4.pop()).is_none());
    }

    #[test]
    fn a_missing_file_reports_none_rather_than_failing() {
        assert!(TerrainImage::load(std::path::Path::new("/nonexistent/ground.jpg")).is_none());
    }

    /// Role-named files, in the shader's fixed slot order.
    #[test]
    fn the_layer_file_names_match_the_shader_slots() {
        assert_eq!(TERRAIN_LAYER_FILES.len(), FALLBACK_LAYER_COLOURS.len());
        assert_eq!(TERRAIN_LAYER_FILES[0], "ground.jpg");
        assert_eq!(TERRAIN_LAYER_FILES[1], "rock.jpg");
    }

    /// A scratch directory holding exactly the `settings.json` given.
    fn settings_dir(label: &str, body: Option<&str>) -> std::path::PathBuf {
        let dir = std::env::temp_dir()
            .join(format!("terrain_settings_{label}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        if let Some(text) = body {
            std::fs::write(dir.join(TERRAIN_SETTINGS_FILE), text).unwrap();
        }
        dir
    }

    /// No file is the state every project was in before this existed, and it
    /// must render exactly as it did then.
    #[test]
    fn absent_settings_are_the_built_in_defaults() {
        let dir = settings_dir("absent", None);
        let loaded = load_terrain_settings(&dir);
        assert_eq!(loaded.repeat, TerrainMaterialUniform::default().repeat);
        assert_eq!(loaded.normal_strength, 1.0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The point of the file: the editor's tile sizes reach the headset.
    #[test]
    fn repeat_comes_from_the_file() {
        let dir = settings_dir("repeat", Some(r#"{"repeat": [2.0, 3.5, 12.0, 6.0]}"#));
        assert_eq!(load_terrain_settings(&dir).repeat, [2.0, 3.5, 12.0, 6.0]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Every key independent. A file that sets only `repeat` must not zero the
    /// normal maps -- which is what a `serde` struct with `Default::default()`
    /// per FIELD would do only if every field carried its own default, and what
    /// a plain deserialize would get wrong silently.
    #[test]
    fn an_unmentioned_key_keeps_its_default() {
        let dir = settings_dir("partial", Some(r#"{"repeat": [1.0, 1.0, 1.0, 1.0]}"#));
        let loaded = load_terrain_settings(&dir);
        assert_eq!(loaded.repeat, [1.0; 4]);
        assert_eq!(loaded.normal_strength, 1.0);
        assert_eq!(loaded.slope_start_deg, TerrainMaterialUniform::default().slope_start_deg);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A short array is a hand-edit, not a reason to lose the other layers.
    #[test]
    fn a_short_repeat_array_only_sets_what_it_names() {
        let dir = settings_dir("short", Some(r#"{"repeat": [2.0]}"#));
        let loaded = load_terrain_settings(&dir);
        let default = TerrainMaterialUniform::default().repeat;
        assert_eq!(loaded.repeat, [2.0, default[1], default[2], default[3]]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Zero would divide by zero in the shader. Clamped rather than rejected:
    /// one bad number costs that layer its scale, not the project its ground.
    #[test]
    fn a_zero_tile_size_is_clamped_not_taken() {
        let dir = settings_dir("zero", Some(r#"{"repeat": [0.0, -4.0, 10.0, 6.0]}"#));
        let loaded = load_terrain_settings(&dir);
        assert_eq!(loaded.repeat[0], MIN_REPEAT);
        assert_eq!(loaded.repeat[1], MIN_REPEAT);
        assert_eq!(loaded.repeat[2], 10.0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Unreadable settings must not cost a project its terrain.
    #[test]
    fn malformed_settings_fall_back_rather_than_failing() {
        let dir = settings_dir("broken", Some("{ this is not json"));
        assert_eq!(
            load_terrain_settings(&dir).repeat,
            TerrainMaterialUniform::default().repeat,
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn resampling_preserves_the_requested_size() {
        let out = resample(&img([9, 9, 9], 7, 3), 16, 16);
        assert_eq!((out.width, out.height), (16, 16));
        assert_eq!(out.rgba.len(), 16 * 16 * 4);
    }
}

/// The texel a normal map uses for "no perturbation".
///
/// (128, 128, 255) unpacks to (0, 0, 1) in tangent space -- straight out of the
/// surface. That is what lets an unauthored normal set be a true no-op and is
/// why the material needs no "has normals" flag, unlike the splat map, where
/// an all-zero texel is a legitimate authored value.
pub const FLAT_NORMAL: [u8; 3] = [128, 128, 255];

impl TerrainImage {
    fn clone_image(&self) -> TerrainImage {
        TerrainImage { width: self.width, height: self.height, rgba: self.rgba.clone() }
    }
}

#[cfg(test)]
mod normal_map_tests {
    use super::tests::{render_quad_full, Palette};
    use super::*;

    const FLAT: [f32; 3] = [0.0, 1.0, 0.0];

    fn normal_map(rgb: [u8; 3]) -> TerrainImage {
        solid_image(rgb, 1, 1)
    }

    /// Only layer 0 has weight, so only its normal map matters.
    fn only_layer0(map: TerrainImage) -> [Option<TerrainImage>; 4] {
        [Some(map), None, None, None]
    }

    fn brightness(px: [u8; 4]) -> u32 {
        px[0] as u32 + px[1] as u32 + px[2] as u32
    }

    /// The whole point: a normal map must change how the surface lights.
    ///
    /// Needs a real light. With the ambient-only harness the other tests use,
    /// `shade` returns a constant whatever the normal is, so this would pass
    /// with the perturbation entirely disconnected.
    #[test]
    fn a_tilted_normal_changes_the_lighting() {
        let Some(flat) = render_quad_full(FLAT, Palette::Test, None, None, &only_layer0(normal_map(FLAT_NORMAL)), true)
        else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        // Tangent normal tilted hard toward +x, where the test light sits.
        let toward = render_quad_full(FLAT, Palette::Test, None, None, &only_layer0(normal_map([230, 128, 160])), true).unwrap();
        // ...and hard away from it.
        let away = render_quad_full(FLAT, Palette::Test, None, None, &only_layer0(normal_map([25, 128, 160])), true).unwrap();

        assert_ne!(brightness(toward), brightness(flat), "a tilted normal must change shading: {toward:?} vs {flat:?}");
        assert!(
            brightness(toward) > brightness(away),
            "tilting toward the light must be brighter than tilting away: {toward:?} vs {away:?}",
        );
    }

    /// GREEN FACES THE TOP OF THE PICTURE, which on the ground is -z.
    ///
    /// Ground is textured top-down at uv = world.xz, and wgpu reads v = 0 at
    /// the picture's top row, so the top of the picture lies toward -z. An
    /// OpenGL map's green above 128 says "faces the top of the picture"; a lamp
    /// out toward -z must therefore light that texel more than one facing the
    /// bottom. The red-only test above cannot see this: red means the same in
    /// both conventions. Until 2026-09-28 the whiteout added green along +v and
    /// lit every bump upside down in z.
    #[test]
    fn green_tilts_the_ground_toward_the_top_of_its_picture() {
        let north = glam::Vec3::new(0.4, 2.0, -3.0);
        let at = |green: u8| {
            super::tests::render_quad_light_at(
                FLAT, Palette::Test, None, None, &only_layer0(normal_map([128, green, 180])), Some(None), north,
            )
        };
        let Some(faces_top) = at(230) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let faces_bottom = at(26).unwrap();
        assert!(
            brightness(faces_top) > brightness(faces_bottom) + 6,
            "a texel facing the top of the picture (-z) must be lit more by a lamp toward -z: \
             {faces_top:?} vs {faces_bottom:?}",
        );
    }

    /// THE BUMPS STAY WHERE THE WORLD PUT THEM AT ANY TURN. A snap or stick
    /// turn turns the frame the shaders work in, not the ground; a normal map
    /// laid out along the world's x and z must bend the ground the same way in
    /// the world whichever way the player faces. It was bent along the
    /// player's axes, so every blade of grass caught a lamp differently after a
    /// turn (headset `outdoors_front-y90`, 2026-10-01). Tilted on both axes, so
    /// a turn by a quarter or a half cannot map the tilt onto itself.
    #[test]
    fn a_normal_maps_bumps_do_not_turn_with_the_player() {
        let lamp = glam::Vec3::new(0.4, 2.0, -3.0);
        let at = |rgb: [u8; 3], yaw: f32| {
            super::tests::render_quad_turned(
                FLAT, Palette::Test, None, None, &only_layer0(normal_map(rgb)), Some(None), lamp, yaw,
            )
        };
        let tilted = [210, 60, 170];
        let Some(still) = at(tilted, 0.0) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        // That the lamp tells the tilts apart at all: tilted the other way on
        // both axes, it lights the pixel differently.
        let opposite = at([46, 196, 170], 0.0).unwrap();
        assert!(
            brightness(still).abs_diff(brightness(opposite)) > 6,
            "the test's lamp must tell the tilts apart: {still:?} vs {opposite:?}",
        );
        use std::f32::consts::{FRAC_PI_2, FRAC_PI_4, PI};
        for yaw in [FRAC_PI_2, PI, -FRAC_PI_4, 1.0] {
            let turned = at(tilted, yaw).unwrap();
            let off = still.iter().zip(&turned).map(|(a, b)| a.abs_diff(*b)).max().unwrap();
            assert!(off <= 1, "turned {yaw} radians: {turned:?}, unturned {still:?}");
        }
    }

    /// STEEP GROUND IS PROJECTED ALONG THE WORLD'S AXES AT ANY TURN. Past
    /// `biplanar_start_deg` a layer's colour comes from two projections chosen
    /// by the normal, their uvs world axes; chosen by the player-frame normal,
    /// a quarter turn swapped a bank's x and z projections and its texture
    /// changed under the player (headset `outdoors_front-y90`, 2026-10-01; a
    /// half turn hid it, the axes' sizes being the same). A 20-degree slope --
    /// biplanar, still one layer -- textured with a ramp, so the place read
    /// shows.
    #[test]
    fn steep_ground_is_projected_along_the_worlds_axes_at_any_turn() {
        let lamp = glam::Vec3::new(0.4, 2.0, -3.0);
        let tilt = 20f32.to_radians();
        let slope = [tilt.sin(), tilt.cos(), 0.0];
        let at = |yaw: f32| {
            super::tests::render_quad_turned(
                slope, Palette::Ramp, None, None, &[None, None, None, None], Some(None), lamp, yaw,
            )
        };
        let Some(still) = at(0.0) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        assert!(still[0] > 10, "the ramp's red shows: {still:?}");
        use std::f32::consts::{FRAC_PI_2, FRAC_PI_4, PI};
        for yaw in [FRAC_PI_2, PI, -FRAC_PI_4, 1.0] {
            let turned = at(yaw).unwrap();
            let off = still.iter().zip(&turned).map(|(a, b)| a.abs_diff(*b)).max().unwrap();
            assert!(off <= 1, "turned {yaw} radians: {turned:?}, unturned {still:?}");
        }
    }

    /// A flat normal map must be indistinguishable from none at all -- that is
    /// what lets an unauthored normal set cost nothing and need no flag.
    #[test]
    fn a_flat_normal_map_is_a_true_no_op() {
        let Some(without) = render_quad_full(FLAT, Palette::Test, None, None, &[None, None, None, None], true)
        else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let with_flat = render_quad_full(FLAT, Palette::Test, None, None, &only_layer0(normal_map(FLAT_NORMAL)), true).unwrap();
        assert_eq!(with_flat, without, "a flat normal map must shade identically to none");
    }

    /// Strength 0 disables perturbation without needing a second code path.
    #[test]
    fn zero_strength_disables_the_maps() {
        let Some((device, queue)) = super::tests::headless_gpu() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let layout = material_bind_group_layout(&device);
        let mut settings = TerrainMaterialUniform::default();
        settings.normal_strength = 0.0;

        let err_scope_5 = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let _m = TerrainMaterial::new(
            &device, &queue, &layout,
            &[solid_image([200, 200, 200], 4, 4)],
            &solid_image([128, 128, 128], 1, 1),
            None,
            None,
            &only_layer0(solid_image([230, 128, 160], 4, 4)),
            &[],
            &[],
            settings,
        );
        assert!(pollster::block_on(err_scope_5.pop()).is_none());
    }

    /// Sized to the colour layers, so a normal map at a different resolution
    /// still binds rather than failing validation on whichever project has one.
    #[test]
    fn a_normal_map_of_a_different_size_is_resampled() {
        let Some((device, queue)) = super::tests::headless_gpu() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let layout = material_bind_group_layout(&device);
        let err_scope_6 = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let _m = TerrainMaterial::from_layers(
            &device, &queue, &layout,
            &[Some(solid_image([10, 20, 30], 64, 64)), None, None, None],
            &[Some(solid_image(FLAT_NORMAL, 16, 16)), None, None, None],
            None,
            None,
        );
        assert!(
            pollster::block_on(err_scope_6.pop()).is_none(),
            "a mismatched normal map must be resampled, not rejected",
        );
    }

    #[test]
    fn the_flat_normal_texel_unpacks_to_straight_out() {
        // (128,128,255) / 255 * 2 - 1 ~= (0, 0, 1).
        let unpack = |v: u8| (v as f32 / 255.0) * 2.0 - 1.0;
        assert!(unpack(FLAT_NORMAL[0]).abs() < 0.01);
        assert!(unpack(FLAT_NORMAL[1]).abs() < 0.01);
        assert!((unpack(FLAT_NORMAL[2]) - 1.0).abs() < 0.01);
    }
}

// Scene shader sources, for `multiview::every_scene_shader_survives_the_multiview_transform`.
// Test-only: the gate has to see exactly the text each pipeline is built from,
// and nothing on a development machine can build a multiview pipeline to check.
#[cfg(test)]
pub fn terrain_shader_src() -> String {
    terrain_shader()
}

#[cfg(test)]
mod terrain_material_parity_tests {
    /// Terrain shades through the SAME path as brushes, with the albedo passed
    /// IN rather than multiplied over the answer.
    ///
    /// The old form -- `shade_with_sky(...)` then `albedo * col * lit` -- is the
    /// exact bug this renderer already fixed for brushes: a dielectric's
    /// highlight is not tinted by its diffuse colour, and multiplying a
    /// low-albedo ground over the lit result made every highlight on it far too
    /// dim. Ground had no Fresnel, no energy conservation, no probe reflection
    /// and no specular occlusion, and it read as a second-class material next
    /// to marble.
    #[test]
    fn terrain_uses_the_material_path_with_albedo_passed_in() {
        let src = super::terrain_shader_src();
        assert!(
            src.contains("let lit = shade_material_env("),
            "terrain no longer shades through the material path",
        );
        assert!(
            !src.contains("shade_with_sky(in.world_pos"),
            "terrain went back to the sky-only path",
        );
        assert!(
            src.contains("tonemap(in.col.rgb * lit)"),
            "the albedo is being multiplied over the lit result again; it must \
             go INTO the shading so highlights are not tinted by it",
        );
        assert!(
            !src.contains("tonemap(albedo * in.col.rgb * lit)"),
            "the old multiply-over form is back",
        );
    }

    /// The normal's lost variation goes into ROUGHNESS, the same place the
    /// brush bake puts it -- not into a bespoke specular-exponent tweak.
    #[test]
    fn the_normal_variance_lands_in_roughness() {
        let src = super::terrain_shader_src();
        // The normal's lost variation now lives in the BAKED roughness mips --
        // `roughness_chain_with_normal_variance` folds it in per level at load,
        // exactly as brush materials get it. It must NOT also be added at
        // runtime, or the same variance is counted twice and distant ground
        // goes flat.
        assert!(
            src.contains("layer_rough_at(layer, f)"),
            "terrain no longer samples a roughness map",
        );
        assert!(
            src.contains("layer_ao_at(layer, f)"),
            "terrain no longer samples an occlusion map",
        );
        assert!(
            !src.contains("min(2.0 * sigma2"),
            "the runtime variance term is back alongside the baked mips; that \
             counts the same measurement twice",
        );
        // What the bake cannot see -- geometric normal variation and the
        // grazing-angle anisotropy gap -- is still handled at runtime.
        assert!(
            src.contains("specular_aa_roughness(clamp(rough_map, 0.0, 1.0), dpdx(shaded_n), dpdy(shaded_n))"),
            "the geometric specular-AA term is gone; the bake alone cannot see \
             curvature or the anisotropy gap",
        );
    }
}

#[cfg(test)]
mod terrain_mip_tests {
    /// EVERY TERRAIN ARRAY THAT IS SAMPLED BY A SHRINKING FOOTPRINT NEEDS MIPS.
    ///
    /// The normal array shipped with `mip_level_count: 1` while the shared
    /// sampler asked for `mipmap_filter: Linear` and 8x anisotropy -- both
    /// silent no-ops against one level. Distant pixels point-sampled a
    /// full-resolution normal map, so the shading normal changed randomly
    /// pixel to pixel and head tracking moved it every frame. Brush normals
    /// were mipped all along, which is why only terrain shimmered.
    ///
    /// Asserted on the SOURCE because the texture is built on a device this
    /// test does not have. Crude, and it would have caught the bug.
    #[test]
    fn the_normal_array_is_created_with_a_mip_chain() {
        let src = include_str!("terrain_pipeline.rs");
        let i = src.find(r#"label: Some("terrain_normals")"#).expect("normal array is gone");
        let window = &src[i..i + 400];
        assert!(
            window.contains("mip_level_count,"),
            "terrain_normals is back to a single mip level; distant ground will \
             point-sample full-resolution normals and shimmer",
        );
        assert!(
            !window.contains("mip_level_count: 1"),
            "terrain_normals explicitly requests one mip level",
        );
    }

    /// Roughness and occlusion need mips for the same reason the normal does,
    /// and must be LINEAR for the same reason: they are measurements.
    #[test]
    fn the_data_arrays_are_mipped_and_linear() {
        let src = include_str!("terrain_pipeline.rs");
        let i = src.find("let data_array = |label: &str|").expect("data arrays are gone");
        let window = &src[i..i + 600];
        assert!(
            window.contains("mip_level_count,") && !window.contains("mip_level_count: 1"),
            "the roughness/occlusion arrays lost their mip chain",
        );
        assert!(
            window.contains("TextureFormat::Rgba8Unorm") && !window.contains("Rgba8UnormSrgb"),
            "a measurement array is being created as sRGB",
        );
        // The roughness chain must be the variance-baking one, not a plain
        // box filter -- that is what carries the normal's lost detail.
        assert!(
            src.contains("roughness_chain_with_normal_variance(")
                && src.contains("mip_chain_linear(&ao_arr[slot].rgba"),
            "roughness is no longer baked with the normal's variance, or AO is \
             no longer a plain linear chain",
        );
    }

    /// Every layer ships the full material set, so terrain is not a
    /// second-class surface next to a brush.
    /// A layer's roughness is read through its asset's override, as the
    /// brushes read the same material, and a layer with none as it is.
    #[test]
    fn a_layers_roughness_takes_its_materials_override() {
        let game = std::env::temp_dir().join(format!("terrain_rough_{}", std::process::id()));
        let dir = game.join("textures").join("terrain");
        let _ = std::fs::remove_dir_all(&game);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("sources.json"), r#"{"ground": {"assetId": "Grass005"}, "rock": {"assetId": "Rock063"}}"#).unwrap();
        std::fs::write(game.join("material_overrides.json"), r#"{"Rock063": {"roughness": {"min": 0.55, "max": 1.0}}}"#).unwrap();
        assert_eq!(super::terrain_roughness_range(&dir, 1), Some((0.55, 1.0)), "the rock layer is Rock063");
        assert_eq!(super::terrain_roughness_range(&dir, 0), None, "grass has no override");
        assert_eq!(super::terrain_roughness_range(&dir, 2), None, "a layer with no source");
        // The map itself, remapped: 0.4 reads as 0.55 + 0.4 * 0.45 = 0.73.
        let grey = image::RgbImage::from_pixel(4, 4, image::Rgb([102, 102, 102]));
        grey.save(dir.join("rock_r.jpg")).unwrap();
        grey.save(dir.join("ground_r.jpg")).unwrap();
        let maps = super::load_terrain_rough(&dir);
        let read = |m: &Option<super::TerrainImage>| m.as_ref().unwrap().rgba[0] as f32 / 255.0;
        assert!((read(&maps[1]) - 0.73).abs() < 0.02, "rock read as {}", read(&maps[1]));
        assert!((read(&maps[0]) - 0.4).abs() < 0.02, "grass read as {}", read(&maps[0]));
        let _ = std::fs::remove_dir_all(&game);
    }

    /// The shipped rock reads as the hallway's walls read the same stone:
    /// dry, its median 0.74 where the map's own is 0.43.
    #[test]
    fn the_shipped_cliffs_read_as_dry_rock() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../game/textures/terrain");
        let Some(Some(rock)) = super::load_terrain_rough(&dir).into_iter().nth(1) else {
            eprintln!("skipping: no terrain rock map");
            return;
        };
        let mut v: Vec<u8> = rock.rgba.chunks_exact(4).map(|p| p[0]).collect();
        v.sort_unstable();
        let median = v[v.len() / 2] as f32 / 255.0;
        assert!(median > 0.7, "the cliffs' rock reads at a median roughness of {median}: wet stone");
    }

    #[test]
    fn every_layer_has_a_roughness_and_occlusion_file_name() {
        assert_eq!(super::TERRAIN_ROUGH_FILES.len(), super::TERRAIN_LAYER_FILES.len());
        assert_eq!(super::TERRAIN_AO_FILES.len(), super::TERRAIN_LAYER_FILES.len());
        for (i, colour) in super::TERRAIN_LAYER_FILES.iter().enumerate() {
            let stem = colour.trim_end_matches(".jpg");
            assert_eq!(super::TERRAIN_ROUGH_FILES[i], format!("{stem}_r.jpg"));
            assert_eq!(super::TERRAIN_AO_FILES[i], format!("{stem}_ao.jpg"));
        }
    }

    /// A normal map mipped through the sRGB curve tilts every bump toward the
    /// surface. The chain for it must be the linear one.
    #[test]
    fn the_normal_chain_is_built_linear() {
        let src = include_str!("terrain_pipeline.rs");
        let i = src.find(r#"label: Some("terrain_normals")"#).expect("normal array is gone");
        let window = &src[i..i + 1600];
        assert!(
            window.contains("mip_chain_linear(&src.rgba, w, h, mip_level_count)"),
            "the normal mip chain is not the linear one",
        );
    }

    /// The linear chain must not apply any transfer curve: a mid-grey in is a
    /// mid-grey out at every level.
    #[test]
    fn the_linear_chain_preserves_a_flat_value() {
        let px = vec![128u8; 8 * 8 * 4];
        let chain = super::mip_chain_linear(&px, 8, 8, super::mip_levels_for(8, 8));
        assert!(chain.len() > 1, "no chain was built");
        for (data, w, h) in &chain {
            assert_eq!(data.len(), (w * h * 4) as usize);
            for b in data {
                assert!(
                    (*b as i32 - 128).abs() <= 1,
                    "a flat 128 became {b}; the linear chain is bending values",
                );
            }
        }
    }
}

/// STEEP SHAPES, as the ground's frame reads them (2026-10-13): a CPU
/// transcription of how `sample_frame` chooses its planes and shares and how
/// `layer_normal_at` folds a layer's bumps, run on the heightfield mesh of each
/// shape the headset showed faults on -- with the old reading beside it, which
/// each test must catch. Cells are test_room's (0.43 m), split on one diagonal,
/// normals area-weighted as `terrain_render::vertex_normals` makes them.
#[cfg(test)]
mod steep_shapes_frame {
    use glam::Vec3;

    const CELL: f32 = 0.43;
    const TIE: f32 = 0.57735;
    const STEEP_COS: f32 = 0.95106; // cos 18 degrees, `biplanar_start_deg`
    // The grass map's mean bump z (`ground_n.jpg`): what whiteout shortened.
    const BUMP_Z: f32 = 0.63;

    struct Mesh {
        p: Vec<Vec3>,
        n: Vec<Vec3>,
        tris: Vec<[usize; 3]>,
    }

    fn mesh(size: usize, h: impl Fn(usize, usize) -> f32) -> Mesh {
        let at = |i: usize, j: usize| i * size + j;
        let p: Vec<Vec3> = (0..size * size).map(|k| Vec3::new((k % size) as f32 * CELL, h(k / size, k % size), (k / size) as f32 * CELL)).collect();
        let mut tris = Vec::new();
        for i in 0..size - 1 {
            for j in 0..size - 1 {
                tris.push([at(i, j), at(i + 1, j), at(i, j + 1)]);
                tris.push([at(i, j + 1), at(i + 1, j), at(i + 1, j + 1)]);
            }
        }
        let mut n = vec![Vec3::ZERO; p.len()];
        for t in &tris {
            let c = (p[t[1]] - p[t[0]]).cross(p[t[2]] - p[t[0]]);
            let c = if c.y < 0.0 { -c } else { c };
            for &v in t {
                n[v] += c;
            }
        }
        Mesh { p, n: n.into_iter().map(|v| v.normalize()).collect(), tris }
    }

    /// The five shapes: a one-cell wall, an 85-degree face over four cells, a
    /// knife ridge, a cliff's foot onto level sand, and a stair of one-cell
    /// steps.
    fn shapes() -> Vec<(&'static str, Mesh)> {
        let s = 12;
        vec![
            ("wall", mesh(s, |_, j| if j >= 6 { 13.0 } else { 0.0 })),
            ("face85", mesh(s, |_, j| (j.clamp(4, 8) - 4) as f32 * CELL * 85f32.to_radians().tan())),
            ("ridge", mesh(s, |_, j| (6.0 - (j as f32 - 6.0).abs() * CELL * 70f32.to_radians().tan()).max(0.0))),
            ("foot", mesh(s, |_, j| (j.max(6) - 6) as f32 * CELL * 80f32.to_radians().tan())),
            ("stair", mesh(s, |_, j| (j / 2) as f32 * 0.8)),
        ]
    }

    /// The planes and their shares as the frame reads them: (axis, share)
    /// pairs, and the top plane's fold.
    fn planes_new(n: Vec3, face: Vec3) -> ([(usize, f32); 2], f32) {
        let steep = n.y.min(face.y) <= STEEP_COS;
        let agree = ((n.dot(face).abs() - 0.9703) * 41.3).clamp(0.0001, 1.0);
        let pn = face.lerp(n, agree).normalize();
        let (major, minor) = axes(pn);
        let a = pn.abs();
        let (ma, me) = (a[major], a[minor]);
        let lead = ma - TIE;
        let share = (lead + 0.0001) / (lead + (me - TIE).max(0.0) + 0.0001);
        if !steep {
            return ([(1, 1.0), (minor, 0.0)], 1.0);
        }
        ([(major, share), (minor, 1.0 - share)], agree)
    }

    /// As it was: the planes by the interpolated normal alone, steep by its
    /// slope, the second plane by `ma / (ma + me)` in the colour only, and the
    /// normal folded on the first plane alone.
    fn planes_old(n: Vec3) -> ([(usize, f32); 2], [(usize, f32); 2]) {
        let (major, minor) = axes(n);
        if n.y > STEEP_COS {
            return ([(1, 1.0), (minor, 0.0)], [(1, 1.0), (minor, 0.0)]);
        }
        let a = n.abs();
        let w = a[major] / (a[major] + a[minor]);
        ([(major, w), (minor, 1.0 - w)], [(major, 1.0), (minor, 0.0)])
    }

    fn axes(n: Vec3) -> (usize, usize) {
        let a = n.abs();
        let major = if a.y >= a.x && a.y >= a.z { 1 } else if a.x >= a.z { 0 } else { 2 };
        let minor = (0..3).filter(|&k| k != major).max_by(|&i, &j| a[i].total_cmp(&a[j])).unwrap();
        (major, minor)
    }

    /// A flat bump (its mean) folded on `axis`: `fold_bumps` and `whiteout`.
    fn fold(axis: usize, n: Vec3, keep: f32) -> Vec3 {
        if axis == 1 {
            let g = BUMP_Z + (1.0 - BUMP_Z) * keep;
            return Vec3::new(g * n.x, BUMP_Z * n.y, g * n.z);
        }
        n * BUMP_Z
    }
    fn whiteout(axis: usize, n: Vec3) -> Vec3 {
        let mut v = n;
        v[axis] *= BUMP_Z;
        v
    }

    /// Points across each triangle (barycentric), with the triangle's own
    /// normal and the interpolated one there.
    fn samples(m: &Mesh) -> Vec<(usize, Vec3, Vec3, Vec3)> {
        let mut out = Vec::new();
        for (t, tri) in m.tris.iter().enumerate() {
            let [a, b, c] = tri.map(|v| m.p[v]);
            let face = (b - a).cross(c - a).normalize();
            let face = if face.y < 0.0 { -face } else { face };
            for (u, v) in [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.5, 0.0), (0.0, 0.5), (0.5, 0.5), (0.33, 0.33), (0.1, 0.1), (0.8, 0.1), (0.1, 0.8)] {
                let w = 1.0 - u - v;
                let pos = a * w + b * u + c * v;
                let n = (m.n[tri[0]] * w + m.n[tri[1]] * u + m.n[tri[2]] * v).normalize();
                out.push((t, pos, n, face));
            }
        }
        out
    }

    /// The most a plane with any visible share (a tenth) stretches the picture
    /// on the triangle it is drawn on: 1 / |face . axis|.
    fn worst_stretch(planes: &[(usize, f32)], face: Vec3) -> f32 {
        planes.iter().filter(|(_, s)| *s >= 0.1).map(|&(k, _)| 1.0 / face.abs()[k].max(0.01)).fold(1.0, f32::max)
    }

    /// M1: NO PLANE IS READ STRETCHED. The old reading, taking its planes from
    /// the interpolated normal, pulled the picture out several times on every
    /// shape (the herringbone, the comb teeth); the frame keeps every plane
    /// with a share within 2.8 (the tie's 1.73, widened by the 14 degrees
    /// over which the triangle takes over from the normal).
    #[test]
    fn no_plane_is_read_stretched_on_any_steep_shape() {
        for (name, m) in shapes() {
            let (mut new, mut old) = (1.0f32, 1.0f32);
            for (_, _, n, face) in samples(&m) {
                new = new.max(worst_stretch(&planes_new(n, face).0, face));
                old = old.max(worst_stretch(&planes_old(n).0, face));
            }
            assert!(new <= 2.8, "{name}: a plane is read stretched {new:.1} times");
            assert!(old > 3.0, "{name}: the old reading stretched only {old:.1} times -- the shape no longer tests anything");
        }
    }

    /// M2: THE BUMPS SHADE ONE NORMAL ON EITHER SIDE OF ANY LINE. Where two
    /// triangles meet, and where a plane takes over inside one, the mean
    /// shading normal must not jump: the old reading turned it by up to
    /// fifteen degrees at once (the pale V, the dark facets, the line along a
    /// rise's foot).
    #[test]
    fn the_mean_shading_normal_never_jumps() {
        let shade_new = |n: Vec3, face: Vec3| {
            let (planes, keep) = planes_new(n, face);
            planes.iter().map(|&(k, s)| fold(k, n, keep) * s).sum::<Vec3>().normalize()
        };
        let shade_old = |n: Vec3| {
            let (_, normal) = planes_old(n);
            whiteout(normal[0].0, n).normalize()
        };
        for (name, m) in shapes() {
            let (mut new, mut old) = (0.0f32, 0.0f32);
            // The same point seen from every triangle it lies on: a vertex,
            // with that triangle's own normal and the shared interpolated one.
            let pts = samples(&m);
            for (i, &(ti, pi, ni, fi)) in pts.iter().enumerate() {
                for &(tj, pj, nj, fj) in &pts[i + 1..] {
                    if ti == tj || pi.distance(pj) > 1e-4 {
                        continue;
                    }
                    new = new.max(shade_new(ni, fi).angle_between(shade_new(nj, fj)).to_degrees());
                    old = old.max(shade_old(ni).angle_between(shade_old(nj)).to_degrees());
                }
            }
            // And along a fine line up each shape (a row of edges, across the
            // rise), where the old reading's plane flipped inside a triangle.
            let k = |r: usize, c: usize| r * 12 + c;
            let steps = 4000;
            let along = |t: f32| {
                let (j, f) = ((t as usize).min(10), t - (t as usize).min(10) as f32);
                m.n[k(5, j)].lerp(m.n[k(5, j + 1)], f).normalize()
            };
            for step in 0..steps {
                let (t0, t1) = (step as f32 * 11.0 / steps as f32, (step + 1) as f32 * 11.0 / steps as f32);
                old = old.max(shade_old(along(t0)).angle_between(shade_old(along(t1))).to_degrees());
            }
            assert!(new < 1.0, "{name}: the shading normal jumps {new:.1} degrees across a line");
            // (A smooth 85-degree face and a stair of 62-degree steps never
            // cross a tie: their old fault was the stretch, above.)
            if name != "face85" && name != "stair" {
                assert!(old > 4.0, "{name}: the old reading jumped only {old:.1} degrees -- the shape no longer tests anything");
            }
        }
    }

    /// M3: no derivative of a coordinate a pixel CHOSE: every gradient the
    /// frame hands the reads is two parts of the world position's own.
    #[test]
    fn the_frame_takes_no_derivative_of_a_chosen_coordinate() {
        let src = super::terrain_shader();
        let frame = src.split("fn sample_frame").nth(1).and_then(|s| s.split("\nfn ").next()).expect("sample_frame");
        for (i, _) in frame.match_indices("dpdx(").chain(frame.match_indices("dpdy(")) {
            assert!(frame[i + 5..].starts_with("world)"), "a derivative of something other than the world position: {}", &frame[i..i + 40]);
        }
    }

    /// The twins' classes agree with the frame: a triangle whose own face is
    /// steep is never drawn by the gentle twin, whatever its normals say --
    /// the ridge's crest and the stair's treads, whose normals read gentle.
    #[test]
    fn a_steep_faced_triangle_is_never_gentle() {
        for (name, m) in shapes() {
            for tri in &m.tris {
                let pos = tri.map(|v| m.p[v].to_array());
                let face_y = crate::renderer::ground_twins::face_height(pos);
                let normals = tri.map(|v| m.n[v].to_array());
                if face_y <= STEEP_COS {
                    assert!(!crate::renderer::ground_twins::triangle_is_gentle(normals, face_y, 18.0), "{name}: a steep face drawn gentle");
                }
            }
        }
    }
}
