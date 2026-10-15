use std::collections::BTreeMap;

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

pub fn ticks_to_ms(ticks: u64, period_ns: f32) -> f64 {
    (ticks as f64) * (period_ns as f64) / 1_000_000.0
}

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

        assert!((ticks_to_ms(1_000_000, 1.0) - 1.0).abs() < 1e-9);

        assert!((ticks_to_ms(1_000_000, 38.4) - 38.4).abs() < 1e-4);
    }

    #[test]
    fn a_pass_that_starts_after_it_ends_reports_zero_not_a_negative() {
        let t = PassTiming { start_ms: 5.0, end_ms: 4.0 };
        assert_eq!(t.duration_ms(), 0.0);
    }

    #[test]
    fn averages_over_the_window_and_forgets_older_frames() {

        let mut a = TimingAggregator::new(3);
        for ms in [10.0, 10.0, 10.0, 1.0, 1.0, 1.0] {
            a.record("shadow", ms);
        }
        assert!((a.average("shadow").unwrap() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn ignores_a_sample_from_a_pass_that_did_not_run() {

        let mut a = TimingAggregator::new(8);
        a.record("sky", 2.0);
        a.record("sky", f64::NAN);
        a.record("sky", -5.0);
        assert!((a.average("sky").unwrap() - 2.0).abs() < 1e-9);
    }

    #[test]
    fn ranks_the_expensive_pass_first() {

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

        let mut a = TimingAggregator::new(4);
        a.record("shadow", 1.39);
        let report = a.report(13.9);
        assert!(report[0].contains("10.0%"), "{report:?}");
        assert!(report.last().unwrap().contains("TOTAL"));
    }

    #[test]
    fn an_unknown_pass_has_no_average_rather_than_zero() {

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
