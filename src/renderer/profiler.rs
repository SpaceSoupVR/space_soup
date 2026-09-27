//! Where the frame actually goes, measured on the device rather than guessed.
//!
//! Every performance number in this project's roadmap is an estimate derived
//! from how the code is structured -- "the scene is submitted twice, so
//! multiview should save about 40% of CPU". Estimates from structure are a
//! reasonable way to decide what to try and a terrible way to decide whether it
//! worked, and this project's own history is the argument: the light-culling
//! fix earlier this year was *exactly inverted* and looked plausible until
//! somebody measured it.
//!
//! WHY GPU TIMESTAMPS AND NOT A CPU CLOCK
//!
//! Wrapping a pass in `Instant::now()` measures how long it took to *record*
//! commands, not how long the GPU spent running them. On a tile GPU those
//! numbers are close to unrelated: recording is cheap and the tiler may not
//! start work until the pass ends. A timestamp query is written into the
//! command stream and resolved by the GPU itself, so it reports GPU time.
//!
//! WHY THE READBACK IS A FRAME BEHIND
//!
//! Resolved timestamps live in a buffer that has to be mapped, and mapping
//! blocks until the GPU is done with it. Blocking on the frame you just
//! submitted would stall the pipeline and change the very number being
//! measured. So a frame's timings are read on a later frame, which is fine for
//! a profiler and would not be fine for anything that fed back into rendering.

use std::collections::BTreeMap;

/// One pass's GPU time, in milliseconds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PassTiming {
    pub start_ms: f64,
    pub end_ms: f64,
}

impl PassTiming {
    pub fn duration_ms(&self) -> f64 {
        (self.end_ms - self.start_ms).max(0.0)
    }
}

/// Convert a raw timestamp delta to milliseconds.
///
/// `period_ns` is the queue's `get_timestamp_period()` -- nanoseconds per tick,
/// which is hardware-specific and is emphatically not 1. Treating ticks as
/// nanoseconds happens to be almost right on some desktop GPUs and is wrong by
/// an order of magnitude on others, which produces a profile that looks
/// plausible and ranks the passes correctly while being numerically nonsense.
pub fn ticks_to_ms(ticks: u64, period_ns: f32) -> f64 {
    (ticks as f64) * (period_ns as f64) / 1_000_000.0
}

/// A rolling average per pass.
///
/// Averaged because a single frame is noise: the first frame after a scene
/// change compiles pipelines, any frame can be interrupted by the compositor,
/// and a headset throttles when it warms up. A number that jumps by 3 ms
/// between frames cannot be compared against a number from last week, which is
/// the entire use for it.
#[derive(Debug, Default)]
pub struct TimingAggregator {
    window: usize,
    samples: BTreeMap<String, Vec<f64>>,
}

impl TimingAggregator {
    pub fn new(window: usize) -> Self {
        Self { window: window.max(1), samples: BTreeMap::new() }
    }

    pub fn record(&mut self, pass: &str, ms: f64) {
        // A negative or absurd sample means the query was never written -- a
        // pass that did not run this frame, or a driver that returned zero.
        // Recording it would drag the average toward a frame that never
        // happened.
        if !ms.is_finite() || ms < 0.0 {
            return;
        }
        let v = self.samples.entry(pass.to_string()).or_default();
        v.push(ms);
        if v.len() > self.window {
            let excess = v.len() - self.window;
            v.drain(0..excess);
        }
    }

    pub fn average(&self, pass: &str) -> Option<f64> {
        let v = self.samples.get(pass)?;
        if v.is_empty() {
            return None;
        }
        Some(v.iter().sum::<f64>() / v.len() as f64)
    }

    /// Every pass, slowest first -- the ordering a profile is read in.
    pub fn ranked(&self) -> Vec<(String, f64)> {
        let mut out: Vec<(String, f64)> = self
            .samples
            .keys()
            .filter_map(|k| self.average(k).map(|ms| (k.clone(), ms)))
            .collect();
        out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        out
    }

    pub fn total_ms(&self) -> f64 {
        self.ranked().iter().map(|(_, ms)| ms).sum()
    }

    /// One line per pass, plus the share of a frame budget each one took.
    ///
    /// Against the budget rather than as a bare number, because "the shadow
    /// pass is 2.1 ms" only means something next to how long a frame is allowed
    /// to be.
    pub fn report(&self, frame_budget_ms: f64) -> Vec<String> {
        let ranked = self.ranked();
        let total = self.total_ms();
        let mut lines = Vec::with_capacity(ranked.len() + 1);
        for (name, ms) in &ranked {
            let pct = if frame_budget_ms > 0.0 { ms / frame_budget_ms * 100.0 } else { 0.0 };
            lines.push(format!("{name:<24} {ms:7.3} ms  {pct:5.1}% of budget"));
        }
        lines.push(format!(
            "{:<24} {:7.3} ms  {:5.1}% of budget",
            "TOTAL (GPU)",
            total,
            if frame_budget_ms > 0.0 { total / frame_budget_ms * 100.0 } else { 0.0 },
        ));
        lines
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ticks_are_not_nanoseconds() {
        // The trap this exists to avoid. A period of 1.0 and a period of 38.4
        // (a real Adreno value) differ by 38x, and both produce a profile that
        // ranks the passes identically -- so the mistake survives every sanity
        // check that only looks at which pass is slowest.
        assert!((ticks_to_ms(1_000_000, 1.0) - 1.0).abs() < 1e-9);
        // Tolerance sized for f32: the period arrives as f32 from wgpu, and
        // 38.4f32 widens to 38.400001525878906, so a tighter bound would fail
        // on correct code.
        assert!((ticks_to_ms(1_000_000, 38.4) - 38.4).abs() < 1e-4);
    }

    #[test]
    fn a_pass_that_starts_after_it_ends_reports_zero_not_a_negative() {
        let t = PassTiming { start_ms: 5.0, end_ms: 4.0 };
        assert_eq!(t.duration_ms(), 0.0);
    }

    #[test]
    fn averages_over_the_window_and_forgets_older_frames() {
        // The window is what makes two runs comparable. Without it the number
        // reported depends on which frame you happened to look at.
        let mut a = TimingAggregator::new(3);
        for ms in [10.0, 10.0, 10.0, 1.0, 1.0, 1.0] {
            a.record("shadow", ms);
        }
        assert!((a.average("shadow").unwrap() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn ignores_a_sample_from_a_pass_that_did_not_run() {
        // A query never written reads back as garbage. Averaging it in would
        // drag the number toward a frame that never happened.
        let mut a = TimingAggregator::new(8);
        a.record("sky", 2.0);
        a.record("sky", f64::NAN);
        a.record("sky", -5.0);
        assert!((a.average("sky").unwrap() - 2.0).abs() < 1e-9);
    }

    #[test]
    fn ranks_the_expensive_pass_first() {
        // A profile is read from the top. Alphabetical order would bury the
        // pass you opened it to find.
        let mut a = TimingAggregator::new(4);
        a.record("aaa_cheap", 0.2);
        a.record("zzz_expensive", 9.0);
        a.record("mmm_middling", 3.0);
        let ranked = a.ranked();
        assert_eq!(ranked[0].0, "zzz_expensive");
        assert_eq!(ranked[2].0, "aaa_cheap");
    }

    #[test]
    fn reports_a_share_of_the_frame_budget_not_just_a_duration() {
        // "2.1 ms" means nothing on its own; against 13.9 ms it means 15%.
        let mut a = TimingAggregator::new(4);
        a.record("shadow", 1.39);
        let report = a.report(13.9);
        assert!(report[0].contains("10.0%"), "{report:?}");
        assert!(report.last().unwrap().contains("TOTAL"));
    }

    #[test]
    fn an_unknown_pass_has_no_average_rather_than_zero() {
        // Zero would read as "free", which is the opposite of "not measured".
        let a = TimingAggregator::new(4);
        assert_eq!(a.average("never_ran"), None);
    }

    #[test]
    fn a_zero_budget_does_not_divide_by_zero() {
        let mut a = TimingAggregator::new(2);
        a.record("x", 1.0);
        assert!(a.report(0.0).iter().all(|l| !l.contains("NaN") && !l.contains("inf")));
    }
}
