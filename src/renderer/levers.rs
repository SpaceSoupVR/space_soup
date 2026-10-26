use serde::Deserialize;

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

    pub eye_capture: u32,

    pub terminator_aa: bool,

    pub glare: bool,

    pub glare_strength: f32,

    pub effects: bool,

    pub water: bool,

    pub underwater: bool,

    pub weather: bool,

    pub fixture_bulb_level: f32,

    pub flashlight_bounce: bool,

    pub surface_light_loop: bool,

    pub torch_reflection: bool,

    pub floor_mirror: bool,

    pub reflection_blur: bool,

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

    pub terrain_reader: Option<String>,

    pub reader_edit: Option<String>,

    pub gpu_sync: bool,

    pub ssr: Option<bool>,

    pub multiview: Option<bool>,

    pub half_viewport: bool,

    pub direct_path: bool,

    pub ab_cycle: bool,

    pub bench: Option<crate::renderer::bench::BenchPose>,

    pub time_of_day_hour: Option<f32>,

    pub time_of_day_minutes: Option<f32>,

    pub night_vision: bool,
}

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
            water: true,
            underwater: true,
            weather: true,
            fixture_bulb_level: 16.0,
            flashlight_bounce: true,
            surface_light_loop: true,
            torch_reflection: true,
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

    pub fn parse(text: &str) -> Result<Self, String> {
        if text.trim().is_empty() {
            return Ok(Self::default());
        }
        let levers: Self = serde_json::from_str(text).map_err(|e| e.to_string())?;
        if let Some(problem) = levers.bench.as_ref().and_then(super::bench::BenchPose::problem) {
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
        flag("effects", self.effects, d.effects);
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
        assert!(!f.poll().unwrap().unwrap().portals);
        assert!(f.poll().is_none(), "unchanged, nothing to report");
        std::fs::write(&path, r#"{"portals": false, "shadows": false}"#).unwrap();
        let l = f.poll().unwrap().unwrap();
        assert!(!l.portals && !l.shadows);
        std::fs::remove_file(&path).unwrap();
        assert_eq!(f.poll().unwrap().unwrap(), Levers::default(), "deleting the file undoes every lever");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
