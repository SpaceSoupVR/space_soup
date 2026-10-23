pub const ENABLED: bool = false;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {

    Baseline,

    HalfViewport,

    DirectPath,

    NoProbes,

    NoShadows,

    NoSunDynamic,

    NoProbeBlend,

    NoPortals,

    NoDirectLights,

    NoProbeTrace,

    NoProxies,

    NoStationary,

    NoLightCulling,

    FullResReflections,

    NoDepthPrepass,

    NoPortalCulling,

    InlineReflectionLookups,

    TerrainPerPixelReflections,

    NoGroundTrace,

    NoFoveation,

    NoGlare,

    NoTerminatorAa,

    FloorMirror,

    NoReflectionBlur,

    NoCharacterCards,

    NoFlashlightBounce,

    NoTorchReflection,

    NoEffects,

    NoWater,

    NoWeather,

    NoDoorShadows,

    TimeFrozen,
    /// The trial levers of 2026-10-22, each against the baseline: smooth sun
    /// shadow off, smooth ground light off (`smooth_probes` is read at load,
    /// so it stays), the brightness cutoff on, the foveated trace at 30 and
    /// 45 deg.
    NoSmoothShadow,
    NoSmoothGround,
    LightCutoff,
    Fovea30,
    Fovea45,
}

impl Phase {
    pub const ALL: [Phase; 37] = [
        Phase::Baseline,
        Phase::HalfViewport,
        Phase::DirectPath,
        Phase::NoProbes,
        Phase::NoShadows,
        Phase::NoSunDynamic,
        Phase::NoProbeBlend,
        Phase::NoPortals,
        Phase::NoDirectLights,
        Phase::NoProbeTrace,
        Phase::NoProxies,
        Phase::NoStationary,
        Phase::NoLightCulling,
        Phase::FullResReflections,
        Phase::NoDepthPrepass,
        Phase::NoPortalCulling,
        Phase::InlineReflectionLookups,
        Phase::TerrainPerPixelReflections,
        Phase::NoGroundTrace,
        Phase::NoFoveation,
        Phase::NoGlare,
        Phase::NoTerminatorAa,
        Phase::FloorMirror,
        Phase::NoReflectionBlur,
        Phase::NoCharacterCards,
        Phase::NoFlashlightBounce,
        Phase::NoTorchReflection,
        Phase::NoEffects,
        Phase::NoWater,
        Phase::NoWeather,
        Phase::NoDoorShadows,
        Phase::TimeFrozen,
        Phase::NoSmoothShadow,
        Phase::NoSmoothGround,
        Phase::LightCutoff,
        Phase::Fovea30,
        Phase::Fovea45,
    ];

    pub fn for_window(window: u64) -> Phase {
        if !ENABLED {
            return Phase::Baseline;
        }
        Self::cycle_phase(window)
    }

    pub fn cycle_phase(window: u64) -> Phase {
        Self::ALL[(window % Self::ALL.len() as u64) as usize]
    }

    /// The cycle through only the phases `only` names (labels, comma
    /// separated: `Levers::ab_only`), in `ALL`'s order; every phase when it
    /// names none.
    pub fn cycle_phase_among(window: u64, only: Option<&str>) -> Phase {
        let named: Vec<Phase> = match only {
            Some(list) => Self::ALL.iter().copied().filter(|p| list.split(',').any(|l| l.trim() == p.label())).collect(),
            None => Vec::new(),
        };
        if named.is_empty() {
            return Self::cycle_phase(window);
        }
        named[(window % named.len() as u64) as usize]
    }

    pub fn label(self) -> &'static str {
        match self {
            Phase::Baseline => "baseline",
            Phase::HalfViewport => "half_viewport",
            Phase::DirectPath => "direct_path",
            Phase::NoProbes => "no_probes",
            Phase::NoShadows => "no_shadows",
            Phase::NoSunDynamic => "no_sun_dynamic",
            Phase::NoProbeBlend => "no_probe_blend",
            Phase::NoPortals => "no_portals",
            Phase::NoDirectLights => "no_direct_lights",
            Phase::NoProbeTrace => "no_probe_trace",
            Phase::NoProxies => "no_proxies",
            Phase::NoStationary => "no_stationary",
            Phase::NoLightCulling => "no_light_culling",
            Phase::FullResReflections => "full_res_reflections",
            Phase::NoDepthPrepass => "no_depth_prepass",
            Phase::NoPortalCulling => "no_portal_culling",
            Phase::InlineReflectionLookups => "inline_reflection_lookups",
            Phase::TerrainPerPixelReflections => "terrain_per_pixel_reflections",
            Phase::NoGroundTrace => "no_ground_trace",
            Phase::NoFoveation => "no_foveation",
            Phase::NoGlare => "no_glare",
            Phase::NoTerminatorAa => "no_terminator_aa",
            Phase::FloorMirror => "floor_mirror",
            Phase::NoReflectionBlur => "no_reflection_blur",
            Phase::NoCharacterCards => "no_character_cards",
            Phase::NoFlashlightBounce => "no_flashlight_bounce",
            Phase::NoTorchReflection => "no_torch_reflection",
            Phase::NoEffects => "no_effects",
            Phase::NoWater => "no_water",
            Phase::NoWeather => "no_weather",
            Phase::NoDoorShadows => "no_door_shadows",
            Phase::TimeFrozen => "time_frozen",
            Phase::NoSmoothShadow => "no_smooth_shadow",
            Phase::NoSmoothGround => "no_smooth_ground",
            Phase::LightCutoff => "light_cutoff",
            Phase::Fovea30 => "fovea30",
            Phase::Fovea45 => "fovea45",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_ab_schedule_is_off() {
        assert!(
            !ENABLED,
            "perf_ab::ENABLED is on: the renderer blanks three quarters of the \
             image, drops reflections and drops probes in turn every PERF window. \
             Set it back to false.",
        );
    }

    #[test]
    fn every_phase_is_visited_and_the_cycle_repeats() {
        if !ENABLED {

            assert!((0..8).all(|w| Phase::for_window(w) == Phase::Baseline));
        }
        let n = Phase::ALL.len() as u64;
        let first: Vec<Phase> = (0..n).map(Phase::cycle_phase).collect();
        let second: Vec<Phase> = (n..2 * n).map(Phase::cycle_phase).collect();
        assert_eq!(first, Phase::ALL.to_vec(), "a phase was skipped or repeated within one cycle");
        assert_eq!(first, second, "the cycle did not repeat, so later lines cannot be matched to earlier ones");
    }

    #[test]
    fn a_named_cycle_visits_only_its_phases_in_order() {
        let only = Some("baseline, fovea30,light_cutoff,nonsense");
        let got: Vec<Phase> = (0..6).map(|w| Phase::cycle_phase_among(w, only)).collect();
        assert_eq!(got, vec![Phase::Baseline, Phase::LightCutoff, Phase::Fovea30, Phase::Baseline, Phase::LightCutoff, Phase::Fovea30]);
        assert_eq!(Phase::cycle_phase_among(5, None), Phase::cycle_phase(5));
        assert_eq!(Phase::cycle_phase_among(5, Some("nonsense")), Phase::cycle_phase(5));
    }

    #[test]
    fn labels_are_distinct_so_log_lines_cannot_be_confused() {
        let mut labels: Vec<&str> = Phase::ALL.iter().map(|p| p.label()).collect();
        labels.sort();
        labels.dedup();
        assert_eq!(labels.len(), Phase::ALL.len());
    }
}
