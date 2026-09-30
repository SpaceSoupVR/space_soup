//! The CPU and GPU PERFORMANCE LEVELS the app asks the runtime for, via
//! `XR_EXT_performance_settings`: a hint the headset's clock governor takes as
//! its baseline, not a clock the app sets.
//!
//! Its own module for the reason `layer_settings` gives: `openxr` is an
//! Android-only dependency of this crate, so what to ask for, and when, is
//! decided here where a development machine's tests run it, and
//! `xr::perf_settings` only maps it onto the raw call.
//!
//! # Why it exists while nothing asks
//!
//! In play the headset ran this app at GPU level 2 (456 MHz), while `bench.py`
//! measures at level 5 (599 MHz) by a developer-only property (2026-09-29).
//! Under SpaceWarp the governor never sees a late frame, so it never raises
//! the level itself. A polished game will want more GPU than it has today, and
//! the public way to ask a stock headset for it is this extension -- so the
//! plumbing ships dormant, driven by the `cpu_level` / `gpu_level` levers, and
//! the default asks for nothing.
//!
//! # What the Quest does with a request (Meta's documentation, read 2026-09-30)
//!
//! - The levels are suggestions: Dynamic Clock Throttling treats them as a
//!   baseline and moves the clocks either way for power and heat.
//! - An app starts at `sustained_low` for the CPU and `sustained_high` for the
//!   GPU -- so asking the GPU for `sustained_high` asks for what it has.
//! - `boost` is currently the same as `sustained_high` on Quest.
//! - GPU level 5 is granted only to apps with dynamic resolution, and only
//!   with thermal headroom: none of the four levels reaches it.
//! - Quest 3 can trade a CPU level for a GPU level with the manifest entry
//!   `com.oculus.trade_cpu_for_gpu_amount` (1: GPU +1, CPU -1).
//!
//! What each request did to the levels on this headset is measured in the
//! engine plan (section 9) -- the documentation above is a claim until then.

use serde::Deserialize;

/// `XrPerfSettingsLevelEXT`, by the names the lever file uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PerformanceLevel {
    /// A non-XR section: power first, late frames accepted.
    PowerSavings,
    /// Low, steady complexity: less power over the odd late frame.
    SustainedLow,
    /// High or changing complexity, within what the device can sustain.
    SustainedHigh,
    /// Past what the device can sustain, for short sections (under 30 s).
    Boost,
}

impl PerformanceLevel {
    pub fn label(self) -> &'static str {
        match self {
            PerformanceLevel::PowerSavings => "power_savings",
            PerformanceLevel::SustainedLow => "sustained_low",
            PerformanceLevel::SustainedHigh => "sustained_high",
            PerformanceLevel::Boost => "boost",
        }
    }

    /// The extension's value for this level.
    pub fn raw(self) -> i32 {
        match self {
            PerformanceLevel::PowerSavings => 0,
            PerformanceLevel::SustainedLow => 25,
            PerformanceLevel::SustainedHigh => 50,
            PerformanceLevel::Boost => 75,
        }
    }
}

/// `XrPerfSettingsDomainEXT`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Domain {
    Cpu,
    Gpu,
}

impl Domain {
    pub const ALL: [Domain; 2] = [Domain::Cpu, Domain::Gpu];

    pub fn label(self) -> &'static str {
        match self {
            Domain::Cpu => "cpu",
            Domain::Gpu => "gpu",
        }
    }

    /// The extension's value for this domain.
    pub fn raw(self) -> i32 {
        match self {
            Domain::Cpu => 1,
            Domain::Gpu => 2,
        }
    }

    /// The level an app starts at, by Meta's documentation: what a domain is
    /// put back to when its lever is removed, since the extension has no call
    /// that withdraws a request.
    pub fn startup_level(self) -> PerformanceLevel {
        match self {
            Domain::Cpu => PerformanceLevel::SustainedLow,
            Domain::Gpu => PerformanceLevel::SustainedHigh,
        }
    }
}

/// The requests to make when the levels asked for go from `was` to `now`
/// (CPU, GPU): a level newly asked, or changed, is asked; a level no longer
/// asked puts its domain back to where the app started. Nothing asked before
/// or after, nothing is called -- the shipped state never calls at all.
pub fn requests(
    was: [Option<PerformanceLevel>; 2],
    now: [Option<PerformanceLevel>; 2],
) -> Vec<(Domain, PerformanceLevel)> {
    Domain::ALL
        .iter()
        .zip(was.iter().zip(now.iter()))
        .filter_map(|(&domain, (&was, &now))| match (was, now) {
            (a, b) if a == b => None,
            (_, Some(level)) => Some((domain, level)),
            (Some(_), None) => Some((domain, domain.startup_level())),
            (None, None) => None,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use PerformanceLevel::*;

    #[test]
    fn nothing_asked_calls_nothing() {
        assert!(requests([None, None], [None, None]).is_empty());
    }

    #[test]
    fn a_level_is_asked_when_it_changes_and_only_then() {
        assert_eq!(requests([None, None], [None, Some(Boost)]), vec![(Domain::Gpu, Boost)]);
        assert!(requests([None, Some(Boost)], [None, Some(Boost)]).is_empty());
        assert_eq!(
            requests([Some(SustainedLow), Some(Boost)], [Some(SustainedHigh), Some(Boost)]),
            vec![(Domain::Cpu, SustainedHigh)]
        );
    }

    /// Deleting the lever file undoes every lever; for a request the runtime
    /// has already taken, that means asking for the startup level again.
    #[test]
    fn a_level_no_longer_asked_goes_back_to_where_the_app_started() {
        assert_eq!(
            requests([Some(Boost), Some(PowerSavings)], [None, None]),
            vec![(Domain::Cpu, SustainedLow), (Domain::Gpu, SustainedHigh)]
        );
    }

    /// The values `XR_EXT_performance_settings` defines (openxr is not built
    /// on this machine, so they are pinned here against the specification).
    #[test]
    fn the_levels_and_domains_are_the_extensions_values() {
        let levels: Vec<i32> = [PowerSavings, SustainedLow, SustainedHigh, Boost].iter().map(|l| l.raw()).collect();
        assert_eq!(levels, vec![0, 25, 50, 75]);
        assert_eq!((Domain::Cpu.raw(), Domain::Gpu.raw()), (1, 2));
    }

    #[test]
    fn a_lever_names_a_level_as_its_label_does() {
        for level in [PowerSavings, SustainedLow, SustainedHigh, Boost] {
            let parsed: PerformanceLevel = serde_json::from_str(&format!("\"{}\"", level.label())).unwrap();
            assert_eq!(parsed, level);
        }
    }
}
