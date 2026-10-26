use serde::Deserialize;

/// `Phase::LightCutoff`'s cutoff when the lever has none: displayed
/// brightness 0.02, under a tenth of the darkest tone the eye reads.
pub const TRIAL_LIGHT_CUTOFF: f32 = 0.02;

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Levers {

    pub probes: bool,

    pub probe_trace: bool,

    pub reflection_proxies: bool,

    pub probe_blend: bool,

    pub portals: bool,

    pub shadows: bool,

    pub sun_dynamic: bool,

    pub shadow_mesh_cull: bool,

    pub skin_once: bool,

    pub spotless_shaders: bool,

    pub poolless_shaders: bool,

    pub dynamic_resolution: bool,

    pub character_shadows: bool,

    pub door_shadows: bool,

    pub door_culling: bool,

    pub capsules: bool,

    pub direct_lights: bool,

    pub stationary_lights: bool,

    pub eye_adaptation: bool,

    pub light_culling: bool,

    pub half_res_reflections: bool,

    pub depth_prepass: bool,

    pub portal_culling: bool,

    pub deferred_reflection_lookups: bool,

    pub terrain_probe_pass: bool,

    pub ground_trace: bool,

    pub foveation: crate::renderer::foveation::FoveationLevel,

    pub space_warp: bool,

    pub space_warp_debug: u32,

    pub spot_caster_drop: u32,

    pub eye_capture: u32,

    pub terminator_aa: bool,

    pub glare: bool,

    pub glare_strength: f32,

    /// Closed rooms seen from outside drawn at their own adaptation
    /// (`EyeAdaptation::room_exposure`).
    pub room_exposure: bool,

    /// The clouds' shadows drawn per pixel, sweeping over the ground with
    /// the wind (`Sky::step_cloud_shadow`); off dims the whole level's sun
    /// at once, by the clouds over its middle.
    pub cloud_shadows: bool,

    /// The volumetric clouds' sky map built a slice a frame
    /// (`Sky::step_clouds`); off keeps the last map and its shadow as they
    /// were, to price the build on the headset (2026-10-22).
    pub clouds: bool,
    /// The sky map's build slowed (up to a slice every fourth frame) while
    /// the headset's GPU frame time is over budget (`cloud_pace`); off
    /// builds a slice every frame outside closed rooms, to price it.
    pub cloud_pace: bool,
    /// A cloud map's history let go where it and the new march disagree by
    /// 0.03 to 0.1 of the sky showing (`cloud_sky::HISTORY_REJECT`: no ghost
    /// fringes at changing edges), and where it differs by 15-35% of the
    /// cloud there (`HISTORY_REJECT_SHARE`: no ghost rims in clear sky); off
    /// lets it go only at 0.2 to 0.5, as before 2026-10-22.
    pub cloud_history_tight: bool,
    /// The cirrus, drawn per pixel by the sky shader (`sky_cirrus`, 2026-10-23:
    /// in the maps it shared the lower clouds' carry wind and warped where
    /// they crossed). Off: no cirrus, to time it.
    pub sky_cirrus: bool,
    /// From a lit closed room the eye adapts part of the way toward the night
    /// through a doorway it looks at, the room held at its own adaptation
    /// (`exposure::DOOR_LEAN`, 2026-10-23). Off: metered on the room alone.
    pub door_lean: bool,
    /// The reflection probes relit on the GPU at every frame's daylight
    /// (`ProbeStream::new_smooth`), not rebuilt on the CPU at 25% steps; off
    /// is the stepped relight. Read when the level's probes load. Costs two
    /// more cube arrays (2 x 51 MB at 256 px) and the probe array a storage
    /// usage (2026-10-22).
    pub smooth_probes: bool,
    /// The static sun map drawn along the gliding sun, again each 0.1
    /// degrees it moves (`STATIC_SUN_REDRAW_COS`), as two maps blended by
    /// the sun's way between them (`lights::SunPair`, 2026-10-23); off draws
    /// one once a snapshot along the snapshot's sun, as before 2026-10-22.
    pub smooth_sun_shadow: bool,
    /// The ground map's light (what reflections of the ground and the
    /// water's bed show) read at the frame's daylight between its rebuilds,
    /// which come at x1.5 or 3 degrees (`PostUpload::ground_stops`); off
    /// reads it as built, stepping at each.
    pub smooth_ground: bool,
    /// The displayed brightness (radiance times the eye's exposure) a
    /// lamp's reach ends at (`lights::effective_range`): a dim lamp with a
    /// long range stops costing the pixels its light barely reaches. 0 is
    /// off (colleague's plan part 2, item 5; A/B before turning on).
    pub light_cutoff: f32,
    /// Native frames (no SpaceWarp synthesis) whenever the GPU time holds
    /// under 75% of the display period for 5 s, back to SpaceWarp once over
    /// 90% for a quarter second (`frame_governor`; the AAA doc's rank 11).
    /// Off by default: A/B before turning on.
    pub frame_governor: bool,
    /// Models drawn from their coarsest LOD whose error (x
    /// `mesh_lod::ERROR_MARGIN`) projects under `model_lod_pixels` eye
    /// pixels: by construction no switch is visible (part 1, model LODs).
    pub model_lods: bool,
    pub model_lod_pixels: f32,
    /// Dynamic resolution: the eye drawn at a share of its targets, down a
    /// step after three frames over 90% of the frame, up after 60 under 75%,
    /// never under `dynres_min` (`dynamic_resolution`; part 2, item 4). Off
    /// by default: A/B before turning on.
    pub dynres: bool,
    pub dynres_min: f32,

    pub effects: bool,

    pub muzzle_test: bool,

    /// THE MUD TEST, dev only: the right hand's A raises a mud splash where
    /// its aim meets mud ground, B steps through a dropped crate, a bullet and
    /// a blast (`space_soup_app::mud_test`). Takes A and B from the muzzle
    /// test while on. Off.
    pub mud_test: bool,

    pub effect_variants: bool,

    pub effects_overdraw: bool,

    pub cave_dust: bool,

    pub water: bool,

    pub underwater: bool,

    pub weather: bool,

    pub fixture_bulb_level: f32,

    pub flashlight_bounce: bool,

    pub surface_light_loop: bool,

    pub torch_reflection: bool,

    pub pool_bounds: bool,

    pub fps_hud: bool,

    pub pool_metering: bool,

    pub floor_mirror: bool,

    pub reflection_blur: bool,
    /// The reflections carried from frame to frame and averaged
    /// (`probe_blur::History`): from mid distance a lamp cage's wires and
    /// other sub-pixel bright parts of a reflection came out whole or not at
    /// all in each texel and danced as the head moved. Off: each frame's
    /// reflections alone, as before 2026-10-23.
    pub reflection_history: bool,

    pub character_cards: bool,

    pub thin_parts: f32,

    pub thin_shading_sampled: bool,

    pub cpu_level: Option<crate::renderer::performance_level::PerformanceLevel>,

    pub gpu_level: Option<crate::renderer::performance_level::PerformanceLevel>,

    pub terrain_detail_distance: f32,

    pub same_eyes: bool,

    pub pass_cut: Option<String>,

    pub scene_cut: Option<String>,

    pub fixup_cut: Option<String>,

    pub water_cut: Option<String>,

    pub slope_twins: bool,

    pub pass_dedup: bool,

    pub draw_log: u32,

    pub cave_twins: bool,

    pub cave_occluder: bool,

    pub scene_cull: bool,

    pub outdoor_pass: bool,

    pub nowhere_pass: bool,

    pub open_reflections: bool,

    pub wet_reflections: bool,

    pub wet_steep_twins: bool,

    pub puddle_reflections: bool,

    pub wet_sky: bool,

    pub ground_classes: bool,

    pub ground_near_first: bool,

    pub ground_near_chunks: u32,

    pub ground_steep_first: bool,

    pub ground_horizon: bool,

    pub ground_horizon_caves: bool,

    pub cave_first: bool,

    pub opaque_models: bool,

    pub body_prepass: bool,

    pub sky_before_water: bool,

    pub planar_water: bool,

    pub water_runs: bool,

    pub ground_buffers: bool,

    pub brush_buffers: bool,
    /// The brushes' material arrays uploaded as ASTC encoded ahead
    /// (`bin/brush_astc`, `BrushMaterials::from_astc`) where every material
    /// in the level has an encoding of the images it loaded: a quarter of
    /// the memory and no mips built at load. Without them, or on a GPU with
    /// no ASTC, the images are uploaded as before; off always uploads them.
    /// Read when a level's materials load.
    pub brush_astc: bool,

    pub cave_chunk_cull: bool,

    pub ground_lod: bool,

    pub ground_lod_strict: bool,

    pub size_reach: u32,

    pub probe_prepass: bool,

    pub model_rooms: bool,
    /// The lamps a closed room's walls hide from both eyes left out of the
    /// frame's lights, as `model_rooms` leaves out models: a lamp counts if
    /// its reach touches the eye's room or shows through one of its open
    /// doorways (`portal_cull::RoomView::shows`), so nothing it lights is in
    /// view. It frees the budget (`MAX_LIGHTS`) and the meshes' and ground's
    /// lamp loops in levels of many rooms (2026-10-23).
    pub light_rooms: bool,

    pub terrain_reader: Option<String>,

    pub reader_edit: Option<String>,

    pub gpu_sync: bool,

    pub ssr: Option<bool>,

    pub debug_view: Option<String>,

    pub multiview: Option<bool>,

    pub half_viewport: bool,

    pub direct_path: bool,

    pub ab_cycle: bool,
    /// With `ab_cycle`, only these phases (labels, comma separated, e.g.
    /// "baseline,no_smooth_shadow,light_cutoff,fovea30,fovea45").
    pub ab_only: Option<String>,

    pub bench: Option<crate::renderer::bench::BenchPose>,

    pub time_of_day_hour: Option<f32>,

    pub time_of_day_minutes: Option<f32>,

    pub night_vision: bool,
}

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
            spot_caster_drop: 0,
            eye_capture: 0,
            terminator_aa: true,
            glare: true,
            glare_strength: 1.0,
            room_exposure: true,
            cloud_shadows: true,
            clouds: true,
            cloud_pace: true,
            cloud_history_tight: true,
            sky_cirrus: true,
            door_lean: true,
            smooth_probes: true,
            smooth_sun_shadow: true,
            smooth_ground: true,
            light_cutoff: 0.0,
            frame_governor: false,
            model_lods: true,
            model_lod_pixels: 0.5,
            dynres: false,
            dynres_min: 0.75,
            effects: true,
            muzzle_test: true,
            mud_test: false,
            effect_variants: true,
            effects_overdraw: false,
            cave_dust: false,
            water: true,
            underwater: true,
            weather: true,
            fixture_bulb_level: 16.0,
            flashlight_bounce: true,
            surface_light_loop: true,
            torch_reflection: true,
            pool_bounds: true,
            fps_hud: false,
            pool_metering: true,
            floor_mirror: false,
            reflection_blur: true,
            reflection_history: true,
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
            opaque_models: true,
            body_prepass: true,
            sky_before_water: true,
            planar_water: true,
            water_runs: true,
            ground_buffers: true,
            brush_buffers: true,
            brush_astc: true,
            cave_chunk_cull: false,
            ground_lod: true,
            ground_lod_strict: true,
            size_reach: 229,
            probe_prepass: false,
            model_rooms: false,
            light_rooms: true,
            terrain_reader: None,
            reader_edit: None,
            gpu_sync: false,
            ssr: None,
            debug_view: None,
            multiview: None,
            half_viewport: false,
            direct_path: false,
            ab_cycle: false,
            ab_only: None,
            bench: None,
            time_of_day_hour: None,
            time_of_day_minutes: None,
            night_vision: true,
        }
    }
}

impl Levers {

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
            Phase::NoSmoothShadow => l.smooth_sun_shadow = false,
            Phase::NoSmoothGround => l.smooth_ground = false,
            Phase::LightCutoff => {
                if l.light_cutoff <= 0.0 {
                    l.light_cutoff = TRIAL_LIGHT_CUTOFF;
                }
            }
            Phase::Fovea30 => l.pass_cut = Some("def_cut_fovea30".into()),
            Phase::Fovea45 => l.pass_cut = Some("def_cut_fovea45".into()),
            Phase::NoReflectionHistory => l.reflection_history = false,
        }
        l
    }

    pub fn performance_levels(&self) -> [Option<crate::renderer::performance_level::PerformanceLevel>; 2] {
        [self.cpu_level, self.gpu_level]
    }

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
        flag("room_exposure", self.room_exposure, d.room_exposure);
        flag("cloud_shadows", self.cloud_shadows, d.cloud_shadows);
        flag("clouds", self.clouds, d.clouds);
        flag("cloud_pace", self.cloud_pace, d.cloud_pace);
        flag("cloud_history_tight", self.cloud_history_tight, d.cloud_history_tight);
        flag("sky_cirrus", self.sky_cirrus, d.sky_cirrus);
        flag("door_lean", self.door_lean, d.door_lean);
        flag("smooth_probes", self.smooth_probes, d.smooth_probes);
        flag("smooth_sun_shadow", self.smooth_sun_shadow, d.smooth_sun_shadow);
        flag("smooth_ground", self.smooth_ground, d.smooth_ground);
        flag("frame_governor", self.frame_governor, d.frame_governor);
        flag("model_lods", self.model_lods, d.model_lods);
        flag("dynres", self.dynres, d.dynres);
        flag("effects", self.effects, d.effects);
        flag("muzzle_test", self.muzzle_test, d.muzzle_test);
        flag("mud_test", self.mud_test, d.mud_test);
        flag("effect_variants", self.effect_variants, d.effect_variants);
        flag("effects_overdraw", self.effects_overdraw, d.effects_overdraw);
        flag("cave_dust", self.cave_dust, d.cave_dust);
        flag("water", self.water, d.water);
        flag("underwater", self.underwater, d.underwater);
        flag("weather", self.weather, d.weather);
        flag("terminator_aa", self.terminator_aa, d.terminator_aa);
        flag("floor_mirror", self.floor_mirror, d.floor_mirror);
        flag("reflection_blur", self.reflection_blur, d.reflection_blur);
        flag("reflection_history", self.reflection_history, d.reflection_history);
        flag("character_cards", self.character_cards, d.character_cards);
        flag("flashlight_bounce", self.flashlight_bounce, d.flashlight_bounce);
        flag("surface_light_loop", self.surface_light_loop, d.surface_light_loop);
        flag("torch_reflection", self.torch_reflection, d.torch_reflection);
        flag("pool_bounds", self.pool_bounds, d.pool_bounds);
        flag("fps_hud", self.fps_hud, d.fps_hud);
        flag("pool_metering", self.pool_metering, d.pool_metering);
        flag("gpu_sync", self.gpu_sync, d.gpu_sync);
        flag("same_eyes", self.same_eyes, d.same_eyes);
        flag("thin_shading_sampled", self.thin_shading_sampled, d.thin_shading_sampled);
        flag("half_viewport", self.half_viewport, d.half_viewport);
        flag("direct_path", self.direct_path, d.direct_path);
        flag("ab_cycle", self.ab_cycle, d.ab_cycle);
        if let Some(only) = &self.ab_only {
            out.push(format!("ab_only={only}"));
        }
        if self.foveation != d.foveation {
            out.push(format!("foveation={}", self.foveation.label()));
        }
        if self.space_warp_debug != 0 {
            out.push(format!("swdbg={}", self.space_warp_debug));
        }
        if self.spot_caster_drop != 0 {
            out.push(format!("spot_drop={}", self.spot_caster_drop));
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
        if self.dynres_min != d.dynres_min {
            out.push(format!("dynres_min={}", self.dynres_min));
        }
        if self.model_lod_pixels != d.model_lod_pixels {
            out.push(format!("model_lod_pixels={}", self.model_lod_pixels));
        }
        if self.light_cutoff != d.light_cutoff {
            out.push(format!("light_cutoff={}", self.light_cutoff));
        }
        if self.terrain_detail_distance != d.terrain_detail_distance {
            out.push(format!("terrain_detail={}", self.terrain_detail_distance));
        }
        if let Some(on) = self.ssr {
            out.push(format!("ssr={}", if on { "on" } else { "off" }));
        }
        if let Some(view) = &self.debug_view {
            out.push(format!("debug_view={view}"));
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
            ("opaque_models", self.opaque_models, d.opaque_models),
            ("body_prepass", self.body_prepass, d.body_prepass),
            ("sky_before_water", self.sky_before_water, d.sky_before_water),
            ("planar_water", self.planar_water, d.planar_water),
            ("water_runs", self.water_runs, d.water_runs),
            ("ground_buffers", self.ground_buffers, d.ground_buffers),
            ("brush_buffers", self.brush_buffers, d.brush_buffers),
            ("brush_astc", self.brush_astc, d.brush_astc),
            ("cave_chunk_cull", self.cave_chunk_cull, d.cave_chunk_cull),
            ("ground_lod", self.ground_lod, d.ground_lod),
            ("ground_lod_strict", self.ground_lod_strict, d.ground_lod_strict),
            ("probe_prepass", self.probe_prepass, d.probe_prepass),
            ("model_rooms", self.model_rooms, d.model_rooms),
            ("light_rooms", self.light_rooms, d.light_rooms),
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

pub struct LeverFile {
    path: std::path::PathBuf,

    seen: Option<(std::time::SystemTime, u64)>,
}

impl LeverFile {
    pub fn new(path: impl Into<std::path::PathBuf>) -> Self {
        Self { path: path.into(), seen: None }
    }

    pub fn path(&self) -> &std::path::Path {
        &self.path
    }

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

    #[test]
    fn the_debug_view_and_opaque_models_are_levers() {
        let l = Levers::parse(r#"{"debug_view": "sources", "opaque_models": false}"#).unwrap();
        assert_eq!(l.debug_view.as_deref(), Some("sources"));
        assert!(!l.opaque_models && Levers::default().opaque_models && Levers::default().debug_view.is_none());
        assert_eq!(l.summary(), "debug_view=sources,no_opaque_models");
        let s = Levers::parse(r#"{"sky_before_water": false}"#).unwrap();
        assert!(!s.sky_before_water && Levers::default().sky_before_water);
        assert_eq!(s.summary(), "no_sky_before_water");
        let m = Levers::parse(r#"{"planar_water": false}"#).unwrap();
        assert!(!m.planar_water && Levers::default().planar_water);
        assert_eq!(m.summary(), "no_planar_water");
    }

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

    #[test]
    fn a_performance_level_is_read_and_named() {
        use crate::renderer::performance_level::PerformanceLevel;
        let l = Levers::parse(r#"{"gpu_level": "sustained_high", "cpu_level": "boost"}"#).unwrap();
        assert_eq!(l.performance_levels(), [Some(PerformanceLevel::Boost), Some(PerformanceLevel::SustainedHigh)]);
        assert_eq!(l.summary(), "cpu_level=boost,gpu_level=sustained_high");
        assert_eq!(Levers::default().performance_levels(), [None, None]);
        assert!(Levers::parse(r#"{"gpu_level": "max"}"#).is_err());
    }

    #[test]
    fn an_unknown_lever_is_an_error() {
        assert!(Levers::parse(r#"{"probe_tarce": false}"#).is_err());
    }

    #[test]
    fn a_bench_view_is_read_and_named() {
        let l = Levers::parse(r#"{"bench": {"name": "hall_back", "eye": [-1.3, 1.6, -14.5], "at": [0.4, 2.4, 3.7]}, "ab_cycle": true}"#)
            .unwrap();
        let b = l.bench.as_ref().unwrap();
        assert_eq!((b.name.as_str(), b.eye, b.at), ("hall_back", [-1.3, 1.6, -14.5], [0.4, 2.4, 3.7]));
        assert_eq!(l.summary(), "ab_cycle,bench=hall_back");
    }

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
