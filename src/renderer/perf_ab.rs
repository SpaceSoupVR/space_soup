//! A timed A/B schedule for finding where a frame's GPU time goes.
//!
//! Its own module, off the android-only renderer, for the same reason as
//! `scene_pass_plan`: a decision compiled only for the headset is one no test on
//! a development machine ever runs.
//!
//! # Why a schedule rather than a build per experiment
//!
//! Every experiment used to cost a build, a deploy and a headset session, so
//! each hypothesis got tested one at a time -- and three in a row were wrong.
//! This cycles the configurations on the `PERF` window boundary instead, so one
//! session produces every comparison, each line labelled with what was switched
//! off, all under the same thermal state and the same view.
//!
//! # What a phase switches
//!
//! A phase is the running `levers::Levers` with exactly one more thing off --
//! see `Levers::with_phase` -- so the schedule and a person flipping a lever
//! measure the same thing. `Levers::ab_cycle` runs the schedule without a
//! build; `ENABLED` is the compile-time way.
//!
//! # Every phase is DELIBERATELY WRONG to look at
//!
//! These switches break rendering to measure it: half the image goes black, or
//! reflections vanish. That is the price of measuring in place. `ENABLED` must
//! be false in anything that is meant to be looked at.

/// Whether the renderer cycles through `Phase`s at all.
///
/// Off except in a measurement build. See the module note: every phase but
/// `Baseline` renders the scene incorrectly on purpose. Guarded by
/// `the_ab_schedule_is_off`, because a build with this on looks broken every
/// few seconds on a headset and looks like nothing at all in a diff.
pub const ENABLED: bool = false;

/// One configuration held for a whole `PERF` window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    /// As shipped. The reference every other phase is read against.
    Baseline,
    /// The scene pass shades a quarter of the pixels. If the scene cost drops
    /// toward a quarter, the frame is fill-bound and resolution or per-pixel
    /// shading is the lever; if it barely moves, the cost is elsewhere --
    /// vertex work, pass setup, or bandwidth.
    HalfViewport,
    /// Forces the straight-to-swapchain path: no offscreen copy, no eye pass,
    /// no stored multisampled depth, no per-frame mip chain. Reflections break.
    /// Separates what the REFLECTIVE path costs from what the scene itself costs.
    DirectPath,
    /// No reflection probes: the per-pixel probe search and cube samples skip.
    NoProbes,
    /// No shadows: neither the shadow-map passes nor the per-pixel shadow taps.
    /// The lights still shine, unshadowed, so what drops is shadowing alone.
    NoShadows,
    /// No moving-objects sun map: its per-frame shadow pass and its per-pixel
    /// taps. The static sun map and the baked mask stay.
    NoSunDynamic,
    /// Every resident probe treated as its own room, so no fragment blends two
    /// photographs: the cost of the second cube sample and its parallax.
    NoProbeBlend,
    /// No doorway portals: the per-fragment portal loop finds nothing.
    NoPortals,
    /// No live lights at all, the sky's sun included: the per-pixel light loop,
    /// its shadow taps and the shadow passes. Baked light and probes remain.
    NoDirectLights,
    /// No reflection trace: smooth surfaces fall back to the box projection.
    /// The cost of tracing reflections through the rooms and doorways.
    NoProbeTrace,
    /// No proxies in the trace: reflections pass through the pillar and the
    /// lamps. The cost of what stands inside rooms.
    NoProxies,
    /// No stationary lamps: the cost of shading them live with their baked
    /// shadow masks. Their light is in no lightmap, so the room goes dark.
    NoStationary,
    /// No light culling: every lamp's maths on every pixel, reachable or not.
    /// What skipping lamps that cannot reach a pixel saves; the picture is
    /// identical.
    NoLightCulling,
    /// Reflections traced per pixel in the scene pass again, instead of by the
    /// half-resolution probe pass. What that pass saves.
    FullResReflections,
    /// No brush depth prepass: hidden surfaces are shaded again. What the
    /// prepass saves.
    NoDepthPrepass,
}

impl Phase {
    pub const ALL: [Phase; 15] = [
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
    ];

    /// The phase for the frames of the `window`-th `PERF` window.
    ///
    /// Keyed on the window index rather than a frame count or a clock, so that
    /// a phase covers EXACTLY the frames one `PERF` line averages. A phase that
    /// changed mid-window would average two configurations into one number and
    /// label it as one of them.
    pub fn for_window(window: u64) -> Phase {
        if !ENABLED {
            return Phase::Baseline;
        }
        Self::cycle_phase(window)
    }

    /// The cycle, independent of `ENABLED`, so it stays tested while the
    /// schedule is switched off.
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
            // Off: every window is the baseline. The cycle itself is checked
            // through `cycle_phase`, which does not depend on the switch.
            assert!((0..8).all(|w| Phase::for_window(w) == Phase::Baseline));
        }
        let n = Phase::ALL.len() as u64;
        let first: Vec<Phase> = (0..n).map(Phase::cycle_phase).collect();
        let second: Vec<Phase> = (n..2 * n).map(Phase::cycle_phase).collect();
        assert_eq!(first, Phase::ALL.to_vec(), "a phase was skipped or repeated within one cycle");
        assert_eq!(first, second, "the cycle did not repeat, so later lines cannot be matched to earlier ones");
    }

    // That each phase switches exactly one thing is checked where the switches
    // are: `levers::tests::each_phase_is_one_lever`.

    #[test]
    fn labels_are_distinct_so_log_lines_cannot_be_confused() {
        let mut labels: Vec<&str> = Phase::ALL.iter().map(|p| p.label()).collect();
        labels.sort();
        labels.dedup();
        assert_eq!(labels.len(), Phase::ALL.len());
    }
}
