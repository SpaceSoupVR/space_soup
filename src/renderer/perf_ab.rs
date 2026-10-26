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
}

impl Phase {
    pub const ALL: [Phase; 32] = [
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
    fn labels_are_distinct_so_log_lines_cannot_be_confused() {
        let mut labels: Vec<&str> = Phase::ALL.iter().map(|p| p.label()).collect();
        labels.sort_unstable();
        labels.dedup();
        assert_eq!(labels.len(), Phase::ALL.len());
    }
}
