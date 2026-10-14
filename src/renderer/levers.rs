//! RUNTIME LEVERS: switch a renderer feature off -- or force one on -- on the
//! headset, while it runs, to see what it costs and what it does.
//!
//! # Why a file and not a build
//!
//! Every question of the form "is it the trace or the portals that costs the
//! millisecond?" or "does the seam go away without the probe blend?" used to
//! cost a build and a deploy per answer. A lever answers it in the time it
//! takes to push a few bytes:
//!
//! ```text
//! adb -s 2G0YC5ZG7706YV shell "echo '{\"probe_trace\": false}' > \
//!     /sdcard/Android/data/com.example.questapp/files/levers.json"
//! ```
//!
//! The app polls the file about once a second and hands every change to the
//! renderer, and each `PERF` line names the levers it was measured under, so a
//! number is never read against the wrong configuration. Delete the file (or
//! write `{}`) to go back to the shipped state.
//!
//! # One set of switches for people and for the A/B schedule
//!
//! `perf_ab` cycles the same switches on its own: each phase is these levers
//! with exactly one more thing off. So a lever and a phase can never disagree
//! about what "no portals" means, and every feature added here is measurable by
//! the schedule for free.
//!
//! Unknown names are an ERROR, not ignored: a lever misspelt in the middle of a
//! headset session would otherwise measure nothing and look like a finding.
//!
//! # A viewpoint is a lever too
//!
//! `bench` pins the camera to a named place (see `bench`), so the same file
//! that switches a feature off also says where the frame is measured from. A
//! host script walks the views and the A/B schedule with the headset on a
//! desk: `quest_app/bench.py`.

use serde::Deserialize;

/// Every switchable feature. The default is the shipped renderer.
///
/// `Clone`, not `Copy`: the bench viewpoint carries its name.
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Levers {
    /// Reflection probes at all. Off: surfaces reflect the sky's harmonics.
    pub probes: bool,
    /// The exact reflection trace (rooms, doorways, proxies). Off: smooth
    /// surfaces fall back to the box projection.
    pub probe_trace: bool,
    /// What stands inside rooms -- the pillar, the lamps -- in the trace. Off:
    /// reflections pass through them to the walls.
    pub reflection_proxies: bool,
    /// The two-photograph blend across a room's cells. Off: every resident
    /// probe is its own room, so no fragment reads two.
    pub probe_blend: bool,
    /// Doorway portals: the blend across an opening, and the trace crossing
    /// into the next room.
    pub portals: bool,
    /// Every shadow pass and every shadow tap. Lights still shine, unshadowed.
    pub shadows: bool,
    /// The moving-objects sun map. The static map and the baked mask stay.
    pub sun_dynamic: bool,
    /// Each model left out of the shadow tiles its bounding sphere cannot
    /// reach -- the moving-objects map's two sun tiles and each spot's (see
    /// `shadow::sphere_in_frustum`). Lossless: the GPU clips those casters
    /// whole, so off changes no texel, only what the passes cost.
    pub shadow_mesh_cull: bool,
    /// The player's body posed ONCE a frame by a compute pass, and drawn into
    /// the moving-objects map, the spot atlas and the cards as a rigid mesh of
    /// those positions -- where each of those draws skinned every vertex again
    /// (`skin_compute`). The same arithmetic, so the same shadows and cards.
    pub skin_once: bool,
    /// In frames where no spot casts, the scene's brush readers and models
    /// drawn by their SPOTLESS TWINS: the spots' shadow code left out, which
    /// gives back the registers the tent took (`lights::without_spot_shadows`).
    /// The same pixels: with no spot casting, every spot's test is false.
    pub spotless_shaders: bool,
    /// In frames where no surface is lit by the torch, the probe pass and the
    /// ground's drawn by their POOLLESS TWINS: the pool maps' lookup left out
    /// (`lights::without_pool_maps`). The same pixels: with no lit surface,
    /// every lookup adds nothing.
    pub poolless_shaders: bool,
    /// DYNAMIC RESOLUTION: ask the runtime each frame for its size for the
    /// eyes' layer (`XR_META_recommended_layer_resolution`), which Quest 3
    /// makes the condition for GPU level 5. The eyes are still drawn at their
    /// fixed size. Off: never asked.
    pub dynamic_resolution: bool,
    /// The player's crisp shadows from the lamps lighting them most: a
    /// characters-only tile of the shadow atlas each. Off, those lamps shadow
    /// the player by the capsules alone. See `shadow::MAX_CHARACTER_SHADOWS`.
    pub character_shadows: bool,
    /// The doors' shadows from the lamps: the lamps reaching a door near the
    /// eye take moving casters' tiles that hold the door, so a closed leaf
    /// stops the light its doorway was baked letting through. Off, a door
    /// casts only in the sun's and the torch's maps. See `doors`.
    pub door_shadows: bool,
    /// A shut door seals its doorway for portal culling: what lies beyond is
    /// culled as if the doorway were wall. Off, doors cull nothing. See
    /// `doors::shut_portals`.
    pub door_culling: bool,
    /// The characters as capsules: their soft shadows, contact darkening and
    /// reflections. Off, the shaders see no characters (and the characters'
    /// shadow tiles go with them). See `uniforms::CapsuleUpload`.
    pub capsules: bool,
    /// The live light loop, the sky's sun included. Baked light and probes stay.
    pub direct_lights: bool,
    /// The STATIONARY lamps -- shaded live, shadowed from their baked masks.
    /// MEASUREMENT: off, their light is in no lightmap, so it is simply gone.
    pub stationary_lights: bool,
    /// Eye adaptation. Off: exposure is pinned at the scene's post setting.
    pub eye_adaptation: bool,
    /// The light loop skips a lamp that cannot reach the pixel -- past its
    /// range, or behind a wall by its baked mask -- before any of its maths.
    /// Lossless: off changes no pixel, only what the frame costs.
    pub light_culling: bool,
    /// Brush reflections from the half-resolution probe pass rather than
    /// traced per pixel. See `brush_pipeline::probe_pass`. Mono scene passes.
    pub half_res_reflections: bool,
    /// The brushes' depth drawn first in the scene pass, so nothing hidden
    /// behind a wall or under a floor is shaded. See
    /// `BrushPipeline::new_depth_prepass`. Lossless: off changes no pixel.
    pub depth_prepass: bool,
    /// From inside a closed room, whatever lies outside the building -- the
    /// terrain -- is drawn only where a doorway shows it. See `portal_cull`.
    /// Lossless for a closed room: off changes no pixel.
    pub portal_culling: bool,
    /// The single-eye probe pass leaves a traced hit's secondary lookups to
    /// a compute pass over just the texels that need them. See
    /// `probe_fixup`. Lossless: off changes no pixel.
    pub deferred_reflection_lookups: bool,
    /// The ground's probe reflection made in the half-resolution probe pass
    /// and read back, as the brushes' is, instead of traced per pixel in the
    /// scene pass. See `TerrainPipeline::new_probe_pass`. Single eye, with
    /// `deferred_reflection_lookups`.
    pub terrain_probe_pass: bool,
    /// Reflections that leave the building meet the ground: the trace over
    /// the ground map's heights. Off: they see only the sky. See
    /// `ground_map::trace`.
    pub ground_trace: bool,
    /// Fixed foveated rendering: how coarsely the edges of each eye's image
    /// are shaded, where the headset has it. See `foveation`.
    pub foveation: crate::renderer::foveation::FoveationLevel,
    /// APPLICATION SPACEWARP: motion vectors and depth with every frame, so
    /// the compositor can make every other one. ON by default since
    /// 2026-09-29, once its black frames were fixed: 36 frames a second are
    /// rendered, a 27.8 ms budget. Meshes carry their own motion (previous
    /// model, previous joints); cuboids, particles, water waves, reflections
    /// and highlights move only with the camera. It also turns sharpening off
    /// -- see `layer_settings::sharpening_for`. See `space_warp`.
    pub space_warp: bool,
    /// DIAGNOSIS ONLY: SpaceWarp variants, as bits, to find what makes the
    /// compositor's frames black (headset, 2026-09-29) without a rebuild per
    /// guess. 1 = the motion pass draws nothing (cleared vectors and depth);
    /// 2 = vectors written y DOWN (Vulkan's NDC; shipped is y up); 4 =
    /// vectors zeroed (depth still drawn);
    /// 8 = `appSpaceDeltaPose` the identity; 16 = `farZ` infinite; 64 = no
    /// layer settings (sharpening) on the projection layer; 128 = views drawn
    /// and submitted 5 cm right of the head, 256 = swaying +-5 cm -- head
    /// motion for the compositor to correct, on a headset lying on a desk;
    /// 512 = depth declared reversed; 1024 = depth declared 500 m .. 1 km;
    /// 2048 = the depth image's stencil stored rather than discarded;
    /// 4096 = the depth image cleared to 0 rather than 1;
    /// 16384 = reflections move with their surface, as before 2026-09-29
    /// (`space_warp`'s reflections section), to compare;
    /// 32768 = the brushes write what the reflection motion READ instead of
    /// motion -- reflected share, reach in metres, distance ratio;
    /// 65536 = with 8192, the images also saved raw to the app's files
    /// (`swdump.bin`, see `space_warp::Readback::read`);
    /// 131072 = the motion pass unculled -- every brush and mesh in both
    /// eyes, as before 2026-10-08 round 4 -- to price the cull;
    /// 262144 = no particle motion (rain, snow, effects), to compare.
    pub space_warp_debug: u32,
    /// DIAGNOSIS: both eyes' finished images of the next frame, raw, to the
    /// app's files as `eyecapture_<n>.bin`, each time this changes to a new
    /// nonzero `n` -- to see whether something differs between the eyes,
    /// which no screenshot shows (the system's is one view). Needs the
    /// `debug.spacesoup.eyecapture` property set to 1 before the app starts;
    /// see `xr::vulkan::eye_capture_enabled`. `quest_app/bench.py
    /// --eye-capture` drives it. Captures only: it changes nothing drawn, so
    /// it is not in `summary`.
    pub eye_capture: u32,
    /// Each lamp's terminator shaded over the pixel's footprint rather than at
    /// its centre, so normal-mapped stone lit at a grazing angle does not
    /// flip whole pixels between lit and black. See `lights::terminator_aa`.
    pub terminator_aa: bool,
    /// The lamps' veils: each light source's glare, drawn where an HDR
    /// framebuffer would have bloomed it. See `glare`.
    pub glare: bool,
    /// How strong the veils are against `glare::VEIL_SHARE` of the CIE young
    /// eye's: 1 as shipped.
    pub glare_strength: f32,
    /// The level's effects -- fire, smoke, embers, dust -- simulated, lit and
    /// drawn. See `effects`.
    pub effects: bool,
    /// THE MUZZLE TEST: the right hand's A fires the selected firearm effect
    /// from its muzzle point (a tap one shot, held automatic fire) and B
    /// steps to the next effect (`quest_app::muzzle_test`). A way to look at
    /// `game/effects` in the headset before a weapon's script raises them.
    /// On for now, for that look; a shipped build turns it off.
    pub muzzle_test: bool,
    /// Each effect particle drawn by the shader VARIANT that holds only its
    /// own shading (`effects::variants`): smoke without the flame's code, an
    /// ember without a flash's. Off, the union shader draws them all, as
    /// before 2026-10-13 -- the same pixels; the A/B that prices the variants
    /// (`fire_close`, `fire_and_smoke`, `brick_hall`).
    pub effect_variants: bool,
    /// DEBUG VIEW: every effect particle drawn as a flat eighth-step of
    /// white over black, added, so the count of layers a pixel pays for
    /// reads as its brightness (8 or more is white). The same instances and
    /// cutouts as the real draw. See `effects::overdraw_shader`.
    pub effects_overdraw: bool,
    /// The level's water -- its surfaces, their waves and, under SpaceWarp,
    /// their motion -- drawn. See `water_pipeline`.
    pub water: bool,
    /// The view from under the water -- the veil, the surface's underside,
    /// the waterline and the wet film -- when an eye is in it. Off, an eye
    /// under the water sees the surface's top and an unfogged bed. See
    /// `underwater`.
    pub underwater: bool,
    /// The level's weather -- the ground's weather twins over its areas, and
    /// the falling rain, snow and splashes -- drawn. Off draws the dry
    /// ground's shaders everywhere and no particles. See `weather`.
    pub weather: bool,
    /// Where a fixture's bulb meets the tone curve, exposed, when it would be
    /// brighter: its own light -- the bulb's glow and its lamp on the inside
    /// of its shade -- scaled as one, as an eye adapts to a lamp it looks into
    /// (`tonemap::own_light_scale`). 16 as shipped, the bulb white and its
    /// reflector graded down to the rim; 0 leaves it unscaled, the mouth one
    /// white with the bulb lost in it, as until 2026-10-02.
    pub fixture_bulb_level: f32,
    /// The player's flashlight throwing its light back off what its beam
    /// lands on, as one more light standing at the lit patch. The renderer
    /// draws it as any light; the app places it (`quest_app::flashlight_bounce`).
    pub flashlight_bounce: bool,
    /// The lit surfaces' lights -- that bounce -- shaded by the scene readers
    /// apart from the lamps, with none of the lamp loop's work for a shadow
    /// or a highlight they never have (`lights::Light::is_surface_light`).
    /// Off, they go through the lamp loop as before, the shaders unchanged.
    pub surface_light_loop: bool,
    /// A carried torch in reflections, as its own capsules: its body, and its
    /// glass glowing. The app hands them over (`quest_app::flashlight`); off,
    /// it hands none, and a torch shows in no reflection, as until 2026-10-02.
    pub torch_reflection: bool,
    /// The eye's meter shown the pools the live lamps throw -- a torch close
    /// to a wall -- beside the probes' photographs
    /// (`lights::LightsUniform::set_pool_light`). Off, it meters the level's
    /// own light alone, and a torch a hand's width from pale stone draws its
    /// pool flat white at the dark room's exposure, as until 2026-10-13.
    pub pool_metering: bool,
    /// The characters mirrored in the floor the player stands on, in place of
    /// their capsules there. See `brush_pipeline::probe_pass::MIRROR_FORMAT`.
    /// OFF as shipped: its pass and blur levels cost about 1 ms at GPU level 5
    /// wherever a mirrored body could be in view -- which, with the Quest's
    /// tall field of view, is most views (2026-09-30). On once they cost less.
    pub floor_mirror: bool,
    /// A little blur on every reflection: each half-resolution reflection
    /// texel averaged with its neighbours on the same surface, so reflected
    /// edges and highlights stop crawling with the texel grid. See
    /// `probe_blur`.
    pub reflection_blur: bool,
    /// The player on cards: six views of their body drawn every frame, which
    /// a reflection meeting them reads for their real outline and colours in
    /// place of their capsules' one colour. See `character_cards`.
    pub character_cards: bool,
    /// The thin pass's kernel unit, in eye pixels: a model's wires, chain
    /// links and rims are drawn wider by the kernel's reach and each fragment
    /// takes its share of their true light (see `mesh::thin_parts` and the
    /// mesh shader's `thin_share`), so a wire a third of a pixel thick holds
    /// its light as it moves, where 4x MSAA alone let it swing by a third. 0
    /// draws them as they are, among the opaque parts. 1 as shipped: the
    /// kernel then adds up to the same light wherever a wire falls, upright
    /// or diagonal; more draws a wider band than the kernel needs (where
    /// foveation shades a block of pixels as one fragment, the kernel is that
    /// much wider); less cuts its tails. Until 2026-10-01 this was the drawn
    /// width in pixels, 3 as shipped.
    pub thin_parts: f32,
    /// MEASUREMENT: the thin pass shaded as it was before 2026-10-01, from
    /// the normal interpolated at each fragment of the widened band, instead
    /// of the mean over the stretch of the part each fragment's kernel
    /// reaches (`mesh_pipeline::THIN_SHADE`). Single-eye passes.
    pub thin_shading_sampled: bool,
    /// The CPU performance level asked of the runtime. `None` asks nothing,
    /// as shipped: the app keeps the level it started at. See
    /// `performance_level`.
    pub cpu_level: Option<crate::renderer::performance_level::PerformanceLevel>,
    /// The GPU's, likewise. Play ran at GPU level 2 with nothing asked
    /// (2026-09-29); this is the lever for when the game needs more.
    pub gpu_level: Option<crate::renderer::performance_level::PerformanceLevel>,
    /// Metres past which the terrain's layer normal maps fade out; 0 keeps
    /// them everywhere. See `terrain_pipeline` (`post_params.z`).
    pub terrain_detail_distance: f32,
    /// MEASUREMENT: the right eye drawn from the LEFT eye's view -- its pose
    /// and field of view -- so the two finished images must match texel for
    /// texel; any difference is state one eye's passes keep and the other's
    /// do not (user, 2026-09-30: "one eye pass that is not doing something the
    /// same as the other"). Capture both with `bench.py --eye-capture`.
    pub same_eyes: bool,
    /// MEASUREMENT: the probe pass drawn with one of its register cuts
    /// (`brush_pipeline::DEFERRED_REGISTER_CUTS`, e.g. `def_cut_characters`)
    /// -- what a part costs by its PRESENCE in the shader, which a runtime
    /// switch cannot measure (the code stays compiled in). Mono passes. A cut
    /// named `terrain_cut_...` is the ground's instead
    /// (`terrain_pipeline::PROBE_PASS_REGISTER_CUTS`), the brushes' as shipped.
    pub pass_cut: Option<String>,
    /// MEASUREMENT: the scene pass's brush likewise, with one of
    /// `SCENE_REGISTER_CUTS` (e.g. `scene_cut_contact`). Mono passes.
    pub scene_cut: Option<String>,
    /// MEASUREMENT: the reflection fix-up run with one of
    /// `probe_fixup::FIXUP_CUTS` (e.g. `fixup_cut_rims`) -- what each kind of
    /// record costs. Lossy: the cut lookups are not made.
    pub fixup_cut: Option<String>,
    /// MEASUREMENT: the water drawn with one of `water_pipeline::WATER_CUTS`
    /// (e.g. `breaker_face`) in place of its ringless shader -- what a part
    /// costs by its presence and its work. Not the same picture.
    pub water_cut: Option<String>,
    /// The ground's slope twins (`ground_twins`): its gentle triangles drawn
    /// by the readers without the steep ground's code. Off draws every
    /// triangle with the full readers. The same picture.
    pub slope_twins: bool,
    /// The ground's probe pass with its repeated code written once
    /// (`ground_twins::dedup_passes`). Off draws the pass as it was. The same
    /// picture.
    pub pass_dedup: bool,
    /// MEASUREMENT: each time this changes to a new non-zero value, the next
    /// frame's command buffers, draws and dispatches are logged with their
    /// pipelines' labels (`DRAWLOG` lines), in the order a per-draw GPU trace
    /// numbers them. See `wgpu_hal::vulkan::DRAW_LOG`.
    pub draw_log: u32,
    /// The cave drawn by its shader's twins, each triangle class by the twin
    /// without the path its pixels never take (`cave_twins`). Off draws the
    /// whole cave with the full shader. The same picture.
    pub cave_twins: bool,
    /// The cave's depth drawn first in the reflection (probe) pass, so the
    /// ground behind its rock is not traced there (`cave_twins`). Off leaves
    /// the cave out of that pass, as before. The same picture.
    pub cave_occluder: bool,
    /// The scene pass draws a model, or a cave, only for an eye whose view
    /// its bound reaches: the motion pass's cull (`space_warp`), for the pass
    /// that shades. Off draws every model for both eyes wherever it stands --
    /// its vertices shaded twice a frame for nothing
    /// (`docs/shader-optimization-2026-10-12.md`). The same picture: the GPU
    /// clipped them.
    ///
    /// ON since 2026-10-13 (rounds 6 and 7, `docs/bench/shader_opt_1012`):
    /// -0.02 to -1.01 ms in 18 of 20 views, 0 wgpu errors. outdoors_field
    /// read +0.28 twice with every model culled and nothing else changed:
    /// see docs/shader-optimization-2026-10-12.md for where that goes.
    pub scene_cull: bool,
    /// The reflection pass draws the ground that is all out of doors -- in the
    /// outdoor volume, clear of every room and of every doorway's fades --
    /// with its outdoor twin, which has the outdoor reflection and none of
    /// the rooms' machinery (`outdoor_pass`). Off draws all of the ground
    /// with the full pass. The same picture.
    ///
    /// ON since 2026-10-13 (round 8) with `ground_classes`, which is what
    /// lets it reach this level's ground: by whole chunks it drew 1 of 64.
    pub outdoor_pass: bool,
    /// The reflection pass draws the ground that lies in no photograph's box
    /// and beyond every doorway's reach -- most of a level whose outdoor
    /// volume is the box round its buildings -- with its nowhere twin, whose
    /// lookup answers "nothing" at once, as the full pass's does after all
    /// its searching (`outdoor_pass::NowhereFrame`). Off draws that ground
    /// with the full pass. The same picture.
    ///
    /// ON since 2026-10-13 (round 7): -0.2 to -5.0 ms in the 11 views that
    /// look at such ground (cave_back_out 35.5 -> 30.5, cave_mouth 25.2 ->
    /// 21.7, under_cove 16.4 -> 14.2, beach_south 19.7 -> 17.8), nothing over
    /// the noise elsewhere, 0 wgpu errors.
    pub nowhere_pass: bool,
    /// OPEN GROUND REFLECTS THE OUTDOORS: the ground the nowhere twin would
    /// draw -- beyond the baked outdoor volume, where the reflection lookup
    /// finds nothing -- is drawn by the pass's open-ground twin instead, and
    /// where it is smooth enough to trace (a puddle, wet sand) mirrors the
    /// ground, the buildings' outsides and the sky, as ground inside the
    /// volume does (`PROBE_OPEN_GROUND`, `terrain_pipeline::open_pass_shader`).
    /// Off, that ground reflects nothing, as it did until 2026-10-13: by the
    /// nowhere twin under `nowhere_pass`, else by the full pass. NOT the same
    /// picture: reflections where there were none.
    ///
    /// OFF, JUDGED 2026-10-13 (round 8; docs/shader-optimization-2026-10-12.md,
    /// "Open ground"): it costs 0.3 to 3.2 ms over the nowhere twin where the
    /// view is such ground (rain_puddles_down +3.2, cave_back_out +2.6,
    /// snow_hills +1.8, beach_south +1.6, cave_mouth +1.5), and beside far
    /// puddles that now mirror the hills it puts a pale sky sheen on every
    /// cliff -- a change of look for the user to choose, not a correction.
    /// One line turns it on: this default, or `"open_reflections": true` in
    /// the lever file.
    pub open_reflections: bool,
    /// WET GROUND BEYOND THE BAKED OUTDOOR VOLUME REFLECTS THE OUTDOORS: the
    /// weather's ground there -- rained-on ground, its puddles, a shore's
    /// washed sand -- is drawn by the reflection pass's OPEN-GROUND twin (the
    /// ground map's trace, the buildings' outsides, the sky) where the dry
    /// ground beside it keeps the nowhere twin. A film of water and a puddle
    /// are mirrors, and a mirror with nothing in it is a dark patch: without
    /// this the washed sand at the water's edge is only darker, and far
    /// puddles lie flat and black (`docs/ground-wetness-2026-10-12.md`).
    /// `open_reflections` for the weather's chunks alone: no dry cliff takes
    /// a sheen from it. ON, NOT YET MEASURED ON THE HEADSET (2026-10-13): the
    /// weather's open twin is 3,755 instructions at 16 registers against its
    /// nowhere twin's 2,907, at half resolution, on the weather's chunks
    /// only; `open_reflections` over every such chunk cost rain_puddles_down
    /// 3.2 ms. One line turns it off.
    pub wet_reflections: bool,
    /// THE WET-ONLY TWINS OF THE MIXED AND STEEP READERS
    /// (`ground_twins::wet_readers`): a weather chunk that holds no standing
    /// water and no snow -- a beach's washed chunks with the cliff over them
    /// -- draws its triangles that are not gentle by them, not by the
    /// weather's full reader (3,820 instructions, past the instruction cache;
    /// these about 3,270 by the host proxy). The same picture. Off: the full
    /// reader, as on 2026-10-13 09:05, which measured cave_mouth +3.8 ms.
    pub wet_steep_twins: bool,
    /// ... AND ONLY WHERE WATER STANDS: with `wet_reflections`, the
    /// open-ground twin draws only the weather chunks that hold standing
    /// water (a puddle, a flooded hollow); wet sand, rained-on slopes and
    /// snow keep the nowhere twin, as before `wet_reflections`. A film of
    /// water on sand then lies as dark as it did; a puddle mirrors the sky.
    pub puddle_reflections: bool,
    /// THE SKY IN WET GROUND, by a twin that fits: the weather's chunks that
    /// hold a film of water or a puddle THIS FRAME (`WeatherScene::films_of`:
    /// a washed shore, an area whose map is wet) are drawn in the reflection
    /// pass by the puddle twin (`terrain_pipeline::puddle_pass_shader`) --
    /// the sky at the lobe's blur, with no walk over the ground map and no
    /// buildings' outsides -- so washed sand, wet rock and puddles mirror the
    /// sky they lie under; a dry area's chunks keep the nowhere twin. In
    /// place of `wet_reflections`' open twin, whose whole answer puts the
    /// weather's pass past the Quest's instruction cache. With
    /// `puddle_reflections`, only where water stands.
    pub wet_sky: bool,
    /// The reflection pass's ground twins reach TRIANGLES, not only whole
    /// chunks: a chunk that is neither all out of doors nor all nowhere by
    /// its bounds -- one across the baked outdoor volume's edge, one with a
    /// building in it -- is drawn in its triangles' classes, sorted once a
    /// level from the level's whole probe table (`outdoor_pass::LevelGround`,
    /// `ground_twins::SlopeSplit::classed`), each class by the twin
    /// `outdoor_pass`, `nowhere_pass` and `open_reflections` give it. Off
    /// draws such a chunk whole by the full pass. The same picture (offline,
    /// against the same triangles in the same order all by the full pass:
    /// beach_south to the byte, rain_puddles_down 3 pixels of 2.2 M by one
    /// level, outdoors_field 521 by one to four -- rounding).
    ///
    /// ON since 2026-10-13 (round 8, `docs/bench/shader_opt_1012/r8`), with
    /// `outdoor_pass`: rain_puddles_down -4.9 ms, pond -4.1, snow_hills -3.1,
    /// outdoors_field -2.3, beach_south -1.9, torch_doorway_out -1.7,
    /// outdoors_front -0.8, cave_back_out -0.8, shore -0.7; hall_back +0.16
    /// (two more pipelines in its reflection pass for the ground through its
    /// door); nothing over the 0.15 ms floor in the other ten. 0 wgpu errors.
    pub ground_classes: bool,
    /// The scene pass and the reflection pass draw the ground's chunks
    /// nearest the eye first -- in the scene pass each chunk's gentle, mixed
    /// and steep triangles together, by their own readers -- so ground hidden
    /// behind nearer ground fails its depth test before it is shaded. Off
    /// draws them in the order they lie in the buffer, slope class by slope
    /// class. The same picture.
    ///
    /// OFF, MEASURED 2026-10-13 (round 8): -0.2 to -0.6 ms where near ground
    /// hides far (outdoors_front, cave_mouth, rain_field, outdoors_field,
    /// cave_back_out) and +0.2 to +0.4 where it does not (beach_south, pond,
    /// rain_puddles_down, shore) or where no ground shows at all
    /// (cave_chamber, cave_back_exit): the reader changes up to three times a
    /// chunk. See `ground_near_chunks`.
    pub ground_near_first: bool,
    /// THE NEAREST FEW CHUNKS OF THE GROUND FIRST, the rest reader by reader:
    /// this many of the chunks nearest the eye are drawn as
    /// `ground_near_first` draws every chunk -- one walk, each chunk's slope
    /// ranges by their own readers -- and the chunks beyond them reader by
    /// reader as before, nearest first within each. The near ground is what
    /// hides the far, and it is few chunks; walking every chunk that way
    /// changes the reader up to three times a chunk, which the far ground's
    /// many small chunks pay for (round 8: beach_south +0.42 ms under
    /// `ground_near_first`, outdoors_front -0.64). 0 leaves the order alone.
    /// Ignored under `ground_near_first`. The same picture.
    pub ground_near_chunks: u32,
    /// THE SCENE PASS DRAWS THE GROUND'S STEEP TRIANGLES FIRST, then those with
    /// both kinds of pixel, then the gentle ones, where it drew mixed, gentle,
    /// steep -- the chunks in the order they lie in the buffer either way
    /// (nearest first only under `ground_near_chunks`). What stands up is what hides: a cliff covers the slopes behind
    /// it, level ground covers next to nothing, so the cliff's depth goes down
    /// before the ground behind it is shaded. Three readers bound once each,
    /// as before -- none of `ground_near_first`'s changes of reader. The same
    /// picture.
    ///
    /// ON since 2026-10-13 (rounds 9b and 9c, all 20 views): cave_back_out
    /// -0.65 to -0.90 ms (its scene pass -0.4 an eye), cave_mouth +0.21 (there
    /// the sand in front hides the cliff's foot, and now goes down after it),
    /// the other 18 within 0.1. 0 wgpu errors.
    pub ground_steep_first: bool,
    /// THE GROUND AND THE WATER THE GROUND HIDES ARE NOT DRAWN: a chunk of the
    /// ground, or a body of water, wholly behind a hill for both eyes is left
    /// out of the reflection pass, the scene pass and the motion pass (and
    /// the water's waves are not moved), by a horizon taken over the drawn
    /// ground itself, with a cave's mouth a hole in it. The same picture:
    /// only what lies behind a drawn, depth-writing pixel is left out. See
    /// `ground_horizon`.
    ///
    /// ON since 2026-10-13 (rounds 11 and 13, with `ground_horizon_caves`):
    /// cave_back_out -1.28 ms, cave_mouth -0.83, outdoors_field and pond
    /// -0.30, rain_puddles_down -0.28, nothing over 0.15 the other way.
    pub ground_horizon: bool,
    /// ... AND THE GROUND HIDDEN BY A BAKED CAVE'S ROCK: where every cave
    /// drawn is a baked one (a closed sheet joined to the ground along its
    /// mouths), a sight line that is under the ground reaches no ground
    /// beyond until a mouth lets it out -- so from inside a cave only the
    /// ground its mouths show is drawn, and from outside none behind a mouth
    /// that the cave does not show. With `ground_horizon`. The same picture
    /// for an eye in the air; an eye pushed into rock inside a cave's box no
    /// longer sees ground through the rock's back.
    pub ground_horizon_caves: bool,
    /// THE SCENE PASS DRAWS A BAKED CAVE THE EYE IS INSIDE BEFORE THE GROUND,
    /// where it drew it after the ground and the brushes: from inside, the
    /// rock covers most of the picture and the ground lies behind it, so the
    /// rock's depth goes down before the ground's readers shade what it
    /// hides. A cave the eye is outside of is drawn where it was (the hill
    /// round it hides it, not it the hill). The same picture: both are
    /// opaque and depth-tested.
    ///
    /// ON since 2026-10-13 (round 13): cave_back_exit -3.2 ms, cave_chamber
    /// -2.8, cave_back_out -0.2; the cave's mouth from outside unchanged.
    pub cave_first: bool,
    /// A BODY OF WATER IS DRAWN ONLY WHERE IT CAN BE SEEN: its index buffer,
    /// in the order it lies, in runs of consecutive triangles; the scene pass
    /// and the motion pass draw the runs some eye's view reaches (through a
    /// doorway, from inside a closed room) and the ground does not hide
    /// (with `ground_horizon`), where they drew the whole body if any of its
    /// box was in view. The same picture, the same order. See `water_runs`.
    ///
    /// ON since 2026-10-13 (round 13): snow_hills -0.87 ms, cave_back_out
    /// -0.36, rain_puddles_down, beach_south, pond, torch_doorway_out and
    /// hall_back -0.2, nothing the other way.
    pub water_runs: bool,
    /// THE GROUND'S BUFFERS KEPT ON THE GPU: the frame's solid vertex and
    /// index buffers live across frames, the ground first in each, and the
    /// ground's part is written only when it is not what was last written
    /// (the app re-posed it for a rig that moved; the slope split redone;
    /// another level) -- where every frame copied the whole ground, rebased
    /// its indices and made two new GPU buffers of them. The same bytes at
    /// other offsets; the same picture. Off: as it was. See `static_solid`.
    pub ground_buffers: bool,
    /// THE BRUSHES' BUFFERS KEPT ON THE GPU: the rooms' vertex and index
    /// buffers, their split by sun class and the motion pass's runs live
    /// across frames and are made again only when the brushes handed over
    /// are not the ones they were made from -- where every frame split all
    /// of them and made two new GPU buffers. With the brushes handed over in
    /// the world (`brush_pipeline::WORLD_BRUSHES`, a launch switch) that is
    /// once a level; with the app posing them, every frame the rig moves, as
    /// before. The same bytes; the same picture. Off: as it was. See
    /// `static_brush`. ON since 2026-10-13 on the offline byte comparison and
    /// tests alone: NOT YET MEASURED ON THE HEADSET.
    pub brush_buffers: bool,
    /// A BAKED CAVE'S DEPTH AND MOTION IN THE CHUNKS IN VIEW: the scene pass
    /// draws a baked cave by the chunks an eye can see into
    /// (`baked_cave::visible_ranges`), but the SpaceWarp motion pass and the
    /// reflection pass's occluder drew every cave whole -- their cost
    /// followed the cave's size, not the view. On: both draw the chunks in
    /// view. Lossless (what is left out the GPU clipped). OFF: written
    /// 2026-10-13 with no headset, and neither pass is in the offline frame,
    /// so nothing but the range test (`baked_cave::occluder_ranges`) has
    /// checked it. Turn on after a headset look at cave_mouth / cave_floor.
    pub cave_chunk_cull: bool,
    /// THE GROUND COARSER WITH DISTANCE: each node of the chunks' quadtree
    /// drawn at the step its distance from the head gives it (1, 2, 4, 8 or
    /// 16 samples over the same vertices), borders stitched to the coarser
    /// neighbour (`ground_lod`). The reflection, scene, motion and spot
    /// shadow passes draw the one choice. Does nothing unless the level's
    /// load built the lists (`adb shell setprop debug.spacesoup.groundlod 1`
    /// before the launch).
    ///
    /// OFF: built 2026-10-13 with NO HEADSET. Offline it is watertight and
    /// near the step-1 picture (`docs/ground-lod-status-2026-10-13.md`);
    /// ridge shimmer under MSAA, SpaceWarp on a chunk as its step changes,
    /// shadows, the cave mouth and wet cliffs have to be looked at first.
    pub ground_lod: bool,
    /// WITH `ground_lod`: no node of the ground drawn coarser than its own
    /// error allows at ANY distance (a pixel, `ground_lod::PIXEL_RAD`). Off,
    /// the design's distance caps apply too: step 1 ends at 120 m and step 2
    /// at 240 m whatever a coarser step moves -- on the shipped ground, with
    /// its cliffs, that is where most of the triangles saved come from, and
    /// where a cliff moves by up to 10 mrad (8 pixels) as the cap is crossed.
    /// ON, so that `ground_lod` alone is the rule that keeps the surface
    /// within a pixel: offline at stress_625's hilltop the caps change 8.7%
    /// of the picture's pixels (7,098 by more than 20 levels, far cliffs
    /// visibly another shape), this rule 2.9% (766 by more than 20: shading
    /// along rims, the normals being the vertices').
    /// `docs/ground-lod-status-2026-10-13.md`.
    pub ground_lod_strict: bool,
    /// HOW FAR A BIG THING IS STILL DRAWN, in its own bounding radii: the
    /// app hands the renderer a model, a fixture, a baked cave or a cuboid
    /// while its nearest point is within 40 m OR it is nearer than this many
    /// radii (quest_app's `draw_range::drawn_at`). 229 is where its sphere is
    /// half a degree across; 76, a degree and a half; 0 leaves the 40 m rule
    /// alone, by bounds. Measured in `docs/shader-optimization-2026-10-12.md`.
    pub size_reach: u32,
    /// THE REFLECTION PASS LAYS THE BRUSHES' DEPTH FIRST, as the scene pass
    /// does (`BrushPipeline::new_probe_depth_prepass`): its shader, the most
    /// expensive in the frame, then runs once a pixel, on the nearest wall,
    /// where one draw of the whole level shaded whatever lay behind it too
    /// if it came first in the buffer. The same picture: the pass's shader
    /// passes on equal depth, as the scene's does.
    pub probe_prepass: bool,
    /// THE MODELS A CLOSED ROOM'S WALLS HIDE ARE NOT DRAWN: from inside a
    /// closed room, a model (not a character) is drawn in the scene pass and
    /// the motion pass only where its bound reaches into the room or into
    /// what one of the room's open doorways shows. The shadow passes and the
    /// reflections keep them all. The same picture: `portal_cull::room_view`,
    /// by the rule the ground outside has been drawn by since 2026-09-28.
    pub model_rooms: bool,
    /// MEASUREMENT: the ground drawn by its full reader everywhere, with no
    /// twin (`full`), or as it was before its layer reads became one loop
    /// (`inlined`: reader and probe pass, no twin). The same picture as the
    /// shipped readers. See `TerrainPipeline::new_probe_reader_twins`.
    pub terrain_reader: Option<String>,
    /// MEASUREMENT: every scene reader -- the brushes' sun classes, their
    /// spotless twins, the ground's four -- rebuilt with one of
    /// `brush_pipeline::READER_EDITS` (e.g. `sky_walked`) and drawn in
    /// place of the shipped ones, chosen per frame and per face as they are.
    /// Two forms of one piece of reader code, priced against each other in
    /// one session in the configuration that ships; `reader_edit_none` is
    /// the control, rebuilt unedited. The same picture as the shipped readers.
    pub reader_edit: Option<String>,
    /// MEASUREMENT: block on the GPU at the end of every frame, as the
    /// renderer used to. Its wait is then exactly the GPU's time, which is
    /// what the A/B schedule attributes costs with; shipped, the CPU prepares
    /// the next frame while the GPU draws this one.
    pub gpu_sync: bool,
    /// Screen-space reflections. `None` leaves the in-headset switch alone.
    pub ssr: Option<bool>,
    /// Both eyes in one multiview scene pass. `None` leaves it as set.
    pub multiview: Option<bool>,
    /// MEASUREMENT ONLY -- the scene pass shades a quarter of the pixels. If
    /// the scene cost drops toward a quarter, the frame is fill-bound.
    pub half_viewport: bool,
    /// MEASUREMENT ONLY -- straight to the swapchain: no offscreen copy, no eye
    /// pass. Reflections that need the copy break.
    pub direct_path: bool,
    /// Cycle `perf_ab`'s phases, one per `PERF` window, on top of these levers.
    pub ab_cycle: bool,
    /// MEASUREMENT: the camera pinned to a named viewpoint instead of the
    /// head. The app moves the rig and the renderer pins the tracked head;
    /// see `bench`.
    pub bench: Option<crate::renderer::bench::BenchPose>,
    /// THE TIME OF DAY'S HOUR, local solar time: setting it turns a level
    /// with a photographed sky over to the time-of-day sky (its defaults,
    /// north turned so the sun keeps the photograph's side) and pins the
    /// clock there; a level that authors a time of day jumps to it. `None`
    /// leaves the level's own clock. See `renderer::time_of_day`.
    pub time_of_day_hour: Option<f32>,
    /// Real minutes a whole day takes, over the level's own: 0 freezes the
    /// clock, 24 runs an hour a minute. With `time_of_day_hour`, the clock
    /// runs on from that hour.
    pub time_of_day_minutes: Option<f32>,
    /// The eye's night vision -- the Purkinje shift toward rod vision's
    /// blue-grey under moonlight (`tonemap::night_vision`). Off for an A/B:
    /// the shader's code stays; its amount is 0.
    pub night_vision: bool,
}

/// Set by a change of [`Levers::draw_log`]; taken by the next frame.
pub static DRAW_LOG_PENDING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

impl Default for Levers {
    fn default() -> Self {
        Self {
            probes: true,
            probe_trace: true,
            reflection_proxies: true,
            probe_blend: true,
            portals: true,
            shadows: true,
            sun_dynamic: true,
            shadow_mesh_cull: true,
            skin_once: true,
            spotless_shaders: true,
            poolless_shaders: true,
            dynamic_resolution: true,
            character_shadows: true,
            door_shadows: true,
            door_culling: true,
            capsules: true,
            direct_lights: true,
            stationary_lights: true,
            eye_adaptation: true,
            light_culling: true,
            half_res_reflections: true,
            depth_prepass: true,
            portal_culling: true,
            deferred_reflection_lookups: true,
            terrain_probe_pass: true,
            ground_trace: true,
            foveation: crate::renderer::foveation::SHIPPED,
            space_warp: true,
            space_warp_debug: 0,
            eye_capture: 0,
            terminator_aa: true,
            glare: true,
            glare_strength: 1.0,
            effects: true,
            muzzle_test: true,
            effect_variants: true,
            effects_overdraw: false,
            water: true,
            underwater: true,
            weather: true,
            fixture_bulb_level: 16.0,
            flashlight_bounce: true,
            surface_light_loop: true,
            torch_reflection: true,
            pool_metering: true,
            floor_mirror: false,
            reflection_blur: true,
            character_cards: true,
            thin_parts: 1.0,
            thin_shading_sampled: false,
            cpu_level: None,
            gpu_level: None,
            terrain_detail_distance: 0.0,
            same_eyes: false,
            pass_cut: None,
            scene_cut: None,
            fixup_cut: None,
            water_cut: None,
            slope_twins: true,
            pass_dedup: true,
            draw_log: 0,
            cave_twins: true,
            cave_occluder: true,
            scene_cull: true,
            outdoor_pass: true,
            nowhere_pass: true,
            open_reflections: false,
            wet_reflections: false,
            wet_steep_twins: true,
            puddle_reflections: false,
            wet_sky: false,
            ground_classes: true,
            ground_near_first: false,
            ground_near_chunks: 0,
            ground_steep_first: true,
            ground_horizon: true,
            ground_horizon_caves: true,
            cave_first: true,
            water_runs: true,
            ground_buffers: true,
            brush_buffers: true,
            cave_chunk_cull: false,
            ground_lod: false,
            ground_lod_strict: true,
            size_reach: 229,
            probe_prepass: false,
            model_rooms: false,
            terrain_reader: None,
            reader_edit: None,
            gpu_sync: false,
            ssr: None,
            multiview: None,
            half_viewport: false,
            direct_path: false,
            ab_cycle: false,
            bench: None,
            time_of_day_hour: None,
            time_of_day_minutes: None,
            night_vision: true,
        }
    }
}

impl Levers {
    /// Parse a lever file. `{}` -- or an empty file -- is the shipped state.
    pub fn parse(text: &str) -> Result<Self, String> {
        if text.trim().is_empty() {
            return Ok(Self::default());
        }
        let levers: Self = serde_json::from_str(text).map_err(|e| e.to_string())?;
        if let Some(problem) = levers.bench.as_ref().and_then(|b| b.problem()) {
            return Err(problem);
        }
        Ok(levers)
    }

    /// These levers with the `perf_ab` phase's one extra switch applied.
    pub fn with_phase(self, phase: crate::renderer::perf_ab::Phase) -> Self {
        use crate::renderer::perf_ab::Phase;
        let mut l = self;
        match phase {
            Phase::Baseline => {}
            Phase::HalfViewport => l.half_viewport = true,
            Phase::DirectPath => l.direct_path = true,
            Phase::NoProbes => l.probes = false,
            Phase::NoShadows => l.shadows = false,
            Phase::NoSunDynamic => l.sun_dynamic = false,
            Phase::NoProbeBlend => l.probe_blend = false,
            Phase::NoPortals => l.portals = false,
            Phase::NoDirectLights => l.direct_lights = false,
            Phase::NoProbeTrace => l.probe_trace = false,
            Phase::NoProxies => l.reflection_proxies = false,
            Phase::NoStationary => l.stationary_lights = false,
            Phase::NoLightCulling => l.light_culling = false,
            Phase::FullResReflections => l.half_res_reflections = false,
            Phase::NoDepthPrepass => l.depth_prepass = false,
            Phase::NoPortalCulling => l.portal_culling = false,
            Phase::InlineReflectionLookups => l.deferred_reflection_lookups = false,
            Phase::TerrainPerPixelReflections => l.terrain_probe_pass = false,
            Phase::NoGroundTrace => l.ground_trace = false,
            Phase::NoFoveation => l.foveation = crate::renderer::foveation::FoveationLevel::Off,
            Phase::NoGlare => l.glare = false,
            Phase::NoTerminatorAa => l.terminator_aa = false,
            Phase::FloorMirror => l.floor_mirror = true,
            Phase::NoReflectionBlur => l.reflection_blur = false,
            Phase::NoCharacterCards => l.character_cards = false,
            Phase::NoFlashlightBounce => l.flashlight_bounce = false,
            Phase::NoTorchReflection => l.torch_reflection = false,
            Phase::NoEffects => l.effects = false,
            Phase::NoWater => l.water = false,
            Phase::NoWeather => l.weather = false,
            Phase::NoDoorShadows => l.door_shadows = false,
            Phase::TimeFrozen => l.time_of_day_minutes = Some(0.0),
        }
        l
    }

    /// The performance levels asked for, CPU then GPU, as
    /// `performance_level::requests` takes them.
    pub fn performance_levels(&self) -> [Option<crate::renderer::performance_level::PerformanceLevel>; 2] {
        [self.cpu_level, self.gpu_level]
    }

    /// What differs from the shipped state, for the `PERF` line: `-` when
    /// nothing does.
    pub fn summary(&self) -> String {
        let d = Self::default();
        let mut out: Vec<String> = Vec::new();
        let mut flag = |name: &str, on: bool, default: bool| {
            if on != default {
                out.push(if on { name.to_string() } else { format!("no_{name}") });
            }
        };
        flag("probes", self.probes, d.probes);
        flag("probe_trace", self.probe_trace, d.probe_trace);
        flag("proxies", self.reflection_proxies, d.reflection_proxies);
        flag("probe_blend", self.probe_blend, d.probe_blend);
        flag("portals", self.portals, d.portals);
        flag("shadows", self.shadows, d.shadows);
        flag("sun_dynamic", self.sun_dynamic, d.sun_dynamic);
        flag("shadow_mesh_cull", self.shadow_mesh_cull, d.shadow_mesh_cull);
        flag("skin_once", self.skin_once, d.skin_once);
        flag("spotless_shaders", self.spotless_shaders, d.spotless_shaders);
        flag("poolless_shaders", self.poolless_shaders, d.poolless_shaders);
        flag("dynamic_resolution", self.dynamic_resolution, d.dynamic_resolution);
        flag("character_shadows", self.character_shadows, d.character_shadows);
        flag("door_shadows", self.door_shadows, d.door_shadows);
        flag("door_culling", self.door_culling, d.door_culling);
        flag("capsules", self.capsules, d.capsules);
        flag("direct_lights", self.direct_lights, d.direct_lights);
        flag("stationary", self.stationary_lights, d.stationary_lights);
        flag("eye_adaptation", self.eye_adaptation, d.eye_adaptation);
        flag("light_culling", self.light_culling, d.light_culling);
        flag("half_res_reflections", self.half_res_reflections, d.half_res_reflections);
        flag("depth_prepass", self.depth_prepass, d.depth_prepass);
        flag("portal_culling", self.portal_culling, d.portal_culling);
        flag("deferred_lookups", self.deferred_reflection_lookups, d.deferred_reflection_lookups);
        flag("terrain_probe_pass", self.terrain_probe_pass, d.terrain_probe_pass);
        flag("ground_trace", self.ground_trace, d.ground_trace);
        flag("space_warp", self.space_warp, d.space_warp);
        flag("glare", self.glare, d.glare);
        flag("effects", self.effects, d.effects);
        flag("muzzle_test", self.muzzle_test, d.muzzle_test);
        flag("effect_variants", self.effect_variants, d.effect_variants);
        flag("effects_overdraw", self.effects_overdraw, d.effects_overdraw);
        flag("water", self.water, d.water);
        flag("underwater", self.underwater, d.underwater);
        flag("weather", self.weather, d.weather);
        flag("terminator_aa", self.terminator_aa, d.terminator_aa);
        flag("floor_mirror", self.floor_mirror, d.floor_mirror);
        flag("reflection_blur", self.reflection_blur, d.reflection_blur);
        flag("character_cards", self.character_cards, d.character_cards);
        flag("flashlight_bounce", self.flashlight_bounce, d.flashlight_bounce);
        flag("surface_light_loop", self.surface_light_loop, d.surface_light_loop);
        flag("torch_reflection", self.torch_reflection, d.torch_reflection);
        flag("pool_metering", self.pool_metering, d.pool_metering);
        flag("gpu_sync", self.gpu_sync, d.gpu_sync);
        flag("same_eyes", self.same_eyes, d.same_eyes);
        flag("thin_shading_sampled", self.thin_shading_sampled, d.thin_shading_sampled);
        flag("half_viewport", self.half_viewport, d.half_viewport);
        flag("direct_path", self.direct_path, d.direct_path);
        flag("ab_cycle", self.ab_cycle, d.ab_cycle);
        if self.foveation != d.foveation {
            out.push(format!("foveation={}", self.foveation.label()));
        }
        if self.space_warp_debug != 0 {
            out.push(format!("swdbg={}", self.space_warp_debug));
        }
        if self.glare && self.glare_strength != d.glare_strength {
            out.push(format!("glare_strength={}", self.glare_strength));
        }
        if self.fixture_bulb_level != d.fixture_bulb_level {
            out.push(format!("fixture_bulb_level={}", self.fixture_bulb_level));
        }
        if let Some(level) = self.cpu_level {
            out.push(format!("cpu_level={}", level.label()));
        }
        if let Some(level) = self.gpu_level {
            out.push(format!("gpu_level={}", level.label()));
        }
        if self.thin_parts != d.thin_parts {
            out.push(format!("thin_parts={}", self.thin_parts));
        }
        if self.terrain_detail_distance != d.terrain_detail_distance {
            out.push(format!("terrain_detail={}", self.terrain_detail_distance));
        }
        if let Some(on) = self.ssr {
            out.push(format!("ssr={}", if on { "on" } else { "off" }));
        }
        if let Some(on) = self.multiview {
            out.push(format!("multiview={}", if on { "on" } else { "off" }));
        }
        if let Some(cut) = &self.pass_cut {
            out.push(format!("pass_cut={cut}"));
        }
        if let Some(cut) = &self.scene_cut {
            out.push(format!("scene_cut={cut}"));
        }
        if let Some(cut) = &self.fixup_cut {
            out.push(format!("fixup_cut={cut}"));
        }
        if let Some(cut) = &self.water_cut {
            out.push(format!("water_cut={cut}"));
        }
        if !self.slope_twins {
            out.push("no_slope_twins".to_string());
        }
        if !self.pass_dedup {
            out.push("no_pass_dedup".to_string());
        }
        if self.draw_log != 0 {
            out.push(format!("draw_log={}", self.draw_log));
        }
        if !self.cave_twins {
            out.push("no_cave_twins".to_string());
        }
        if !self.cave_occluder {
            out.push("no_cave_occluder".to_string());
        }
        // Against the default, as the flags above: `x` where it is on and the
        // default is off, `no_x` the other way, nothing where they agree -- so
        // a default accepted later changes the token and an arm that names it
        // (`armbench.py`) cannot take the other arm's windows for its own.
        for (name, on, default) in [
            ("scene_cull", self.scene_cull, d.scene_cull),
            ("outdoor_pass", self.outdoor_pass, d.outdoor_pass),
            ("nowhere_pass", self.nowhere_pass, d.nowhere_pass),
            ("open_reflections", self.open_reflections, d.open_reflections),
            ("wet_reflections", self.wet_reflections, d.wet_reflections),
            ("wet_steep_twins", self.wet_steep_twins, d.wet_steep_twins),
            ("puddle_reflections", self.puddle_reflections, d.puddle_reflections),
            ("wet_sky", self.wet_sky, d.wet_sky),
            ("ground_classes", self.ground_classes, d.ground_classes),
            ("ground_near_first", self.ground_near_first, d.ground_near_first),
            ("ground_steep_first", self.ground_steep_first, d.ground_steep_first),
            ("ground_horizon", self.ground_horizon, d.ground_horizon),
            ("ground_horizon_caves", self.ground_horizon_caves, d.ground_horizon_caves),
            ("cave_first", self.cave_first, d.cave_first),
            ("water_runs", self.water_runs, d.water_runs),
            ("ground_buffers", self.ground_buffers, d.ground_buffers),
            ("brush_buffers", self.brush_buffers, d.brush_buffers),
            ("cave_chunk_cull", self.cave_chunk_cull, d.cave_chunk_cull),
            ("ground_lod", self.ground_lod, d.ground_lod),
            ("ground_lod_strict", self.ground_lod_strict, d.ground_lod_strict),
            ("probe_prepass", self.probe_prepass, d.probe_prepass),
            ("model_rooms", self.model_rooms, d.model_rooms),
        ] {
            if on != default {
                out.push(if on { name.to_string() } else { format!("no_{name}") });
            }
        }
        if self.size_reach != d.size_reach {
            out.push(format!("size_reach={}", self.size_reach));
        }
        if self.ground_near_chunks != d.ground_near_chunks {
            out.push(format!("ground_near_chunks={}", self.ground_near_chunks));
        }
        if let Some(reader) = &self.terrain_reader {
            out.push(format!("terrain_reader={reader}"));
        }
        if let Some(edit) = &self.reader_edit {
            out.push(format!("reader_edit={edit}"));
        }
        if let Some(b) = &self.bench {
            out.push(format!("bench={}", b.name));
        }
        if let Some(h) = self.time_of_day_hour {
            out.push(format!("time_of_day_hour={h}"));
        }
        if let Some(m) = self.time_of_day_minutes {
            out.push(format!("time_of_day_minutes={m}"));
        }
        if !self.night_vision {
            out.push("no_night_vision".to_string());
        }
        if out.is_empty() {
            "-".to_string()
        } else {
            out.join(",")
        }
    }
}

/// A lever file, re-read when it changes.
pub struct LeverFile {
    path: std::path::PathBuf,
    /// The modification time and length last read; `None` before the first
    /// read and while the file does not exist.
    seen: Option<(std::time::SystemTime, u64)>,
}

impl LeverFile {
    pub fn new(path: impl Into<std::path::PathBuf>) -> Self {
        Self { path: path.into(), seen: None }
    }

    pub fn path(&self) -> &std::path::Path {
        &self.path
    }

    /// `Some` when the file changed since the last call: the levers it now
    /// holds, or why it could not be read. A file that disappears reads as the
    /// shipped state, so deleting it undoes every lever.
    pub fn poll(&mut self) -> Option<Result<Levers, String>> {
        match std::fs::metadata(&self.path) {
            Ok(m) => {
                let stamp = (m.modified().unwrap_or(std::time::UNIX_EPOCH), m.len());
                if self.seen == Some(stamp) {
                    return None;
                }
                self.seen = Some(stamp);
                Some(std::fs::read_to_string(&self.path).map_err(|e| e.to_string()).and_then(|t| Levers::parse(&t)))
            }
            Err(_) => {
                if self.seen.take().is_some() {
                    Some(Ok(Levers::default()))
                } else {
                    None
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::renderer::perf_ab::Phase;

    #[test]
    fn an_empty_file_is_the_shipped_renderer() {
        assert_eq!(Levers::parse("").unwrap(), Levers::default());
        assert_eq!(Levers::parse("{}").unwrap(), Levers::default());
        assert_eq!(Levers::default().summary(), "-");
    }

    #[test]
    fn a_lever_changes_only_itself() {
        let l = Levers::parse(r#"{"probe_trace": false, "ssr": true}"#).unwrap();
        assert!(!l.probe_trace && l.ssr == Some(true));
        assert_eq!(Levers { probe_trace: true, ssr: None, ..l.clone() }, Levers::default());
        assert_eq!(l.summary(), "no_probe_trace,ssr=on");
    }

    /// The ground's levers are named against their defaults, whichever way
    /// those stand: an arm that names one (`armbench.py`) keeps its meaning
    /// when a default is accepted later.
    #[test]
    fn the_grounds_levers_are_named_against_their_defaults() {
        let d = Levers::default();
        let l = Levers { scene_cull: !d.scene_cull, ground_near_first: !d.ground_near_first, ground_near_chunks: d.ground_near_chunks + 3, ..d.clone() };
        let name = |name: &str, default: bool| if default { format!("no_{name}") } else { name.to_string() };
        assert_eq!(
            l.summary(),
            format!(
                "{},{},ground_near_chunks={}",
                name("scene_cull", d.scene_cull),
                name("ground_near_first", d.ground_near_first),
                d.ground_near_chunks + 3,
            ),
        );
        assert_eq!(Levers::parse(r#"{"ground_near_chunks": 4}"#).unwrap().ground_near_chunks, 4);
    }

    /// A clock request is named on every line measured under it, since it
    /// moves every timing; a level the extension does not have is refused.
    #[test]
    fn a_performance_level_is_read_and_named() {
        use crate::renderer::performance_level::PerformanceLevel;
        let l = Levers::parse(r#"{"gpu_level": "sustained_high", "cpu_level": "boost"}"#).unwrap();
        assert_eq!(l.performance_levels(), [Some(PerformanceLevel::Boost), Some(PerformanceLevel::SustainedHigh)]);
        assert_eq!(l.summary(), "cpu_level=boost,gpu_level=sustained_high");
        assert_eq!(Levers::default().performance_levels(), [None, None]);
        assert!(Levers::parse(r#"{"gpu_level": "max"}"#).is_err());
    }

    /// A misspelt lever measures nothing and would look like a finding.
    #[test]
    fn an_unknown_lever_is_an_error() {
        assert!(Levers::parse(r#"{"probe_tarce": false}"#).is_err());
    }

    /// The viewpoint rides in the same file and is named on every line.
    #[test]
    fn a_bench_view_is_read_and_named() {
        let l = Levers::parse(r#"{"bench": {"name": "hall_back", "eye": [-1.3, 1.6, -14.5], "at": [0.4, 2.4, 3.7]}, "ab_cycle": true}"#)
            .unwrap();
        let b = l.bench.as_ref().unwrap();
        assert_eq!((b.name.as_str(), b.eye, b.at), ("hall_back", [-1.3, 1.6, -14.5], [0.4, 2.4, 3.7]));
        assert_eq!(l.summary(), "ab_cycle,bench=hall_back");
    }

    /// A view that cannot be pinned -- a misspelt field, a name that would
    /// split the summary, no direction -- is refused like a misspelt lever.
    #[test]
    fn a_bench_view_that_cannot_be_used_is_an_error() {
        for bad in [
            r#"{"bench": {"name": "a", "eye": [0, 1, 0], "look_at": [0, 1, -1]}}"#,
            r#"{"bench": {"name": "a,b", "eye": [0, 1, 0], "at": [0, 1, -1]}}"#,
            r#"{"bench": {"name": "a", "eye": [0, 1, 0], "at": [0, 1, 0]}}"#,
            r#"{"bench": {"name": "a", "eye": [0, 1], "at": [0, 1, -1]}}"#,
        ] {
            assert!(Levers::parse(bad).is_err(), "accepted {bad}");
        }
    }

    /// Each phase is the levers with exactly one more thing switched, so the
    /// schedule and a person flipping the same lever measure the same thing.
    #[test]
    fn each_phase_is_one_lever() {
        for p in Phase::ALL {
            let l = Levers::default().with_phase(p);
            let changed = l.summary();
            if p == Phase::Baseline {
                assert_eq!(changed, "-");
            } else {
                assert!(!changed.contains(','), "{p:?} switched more than one lever: {changed}");
                assert_ne!(changed, "-", "{p:?} switched nothing");
            }
        }
    }

    #[test]
    fn the_file_is_reread_when_it_changes_and_forgotten_when_deleted() {
        let dir = std::env::temp_dir().join(format!("levers_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("levers.json");
        let _ = std::fs::remove_file(&path);
        let mut f = LeverFile::new(&path);
        assert!(f.poll().is_none(), "no file, nothing to report");
        std::fs::write(&path, r#"{"portals": false}"#).unwrap();
        assert_eq!(f.poll().unwrap().unwrap().portals, false);
        assert!(f.poll().is_none(), "unchanged, nothing to report");
        std::fs::write(&path, r#"{"portals": false, "shadows": false}"#).unwrap();
        let l = f.poll().unwrap().unwrap();
        assert!(!l.portals && !l.shadows);
        std::fs::remove_file(&path).unwrap();
        assert_eq!(f.poll().unwrap().unwrap(), Levers::default(), "deleting the file undoes every lever");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
