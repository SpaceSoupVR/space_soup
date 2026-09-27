//! The `XRPERF` log line: what the runtime's frame counters look like once
//! written down. Pure, so it is tested on a development machine; the
//! Android-only `xr::perf_metrics` fills it from `XR_META_performance_metrics`.

/// How a counter is measured, as the runtime typed it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unit {
    Generic,
    Percentage,
    Milliseconds,
    Bytes,
    Hertz,
}

/// One sampled counter.
#[derive(Debug, Clone, PartialEq)]
pub struct Sample {
    /// The counter's path without its `/perfmetrics_meta/` prefix.
    pub short: String,
    pub value: f64,
    pub unit: Unit,
}

/// `/perfmetrics_meta/app/gpu_frametime` -> `app/gpu_frametime`.
pub fn short_name(path: &str) -> String {
    let trimmed = path.trim_start_matches('/');
    trimmed.strip_prefix("perfmetrics_meta/").unwrap_or(trimmed).to_string()
}

/// The per-core CPU utilisations are left out: a Quest 3 has eight, the
/// average is logged, and eight more numbers a line would bury the ones
/// that decide anything.
pub fn worth_logging(short: &str) -> bool {
    let per_core =
        short.starts_with("device/cpu") && short.as_bytes().get(10).is_some_and(|b| b.is_ascii_digit());
    !per_core
}

/// `XRPERF: app/gpu_frametime=12.34ms compositor/gpu_frametime=1.20ms ...`
pub fn format_line(samples: &[Sample]) -> String {
    let mut out = String::from("XRPERF:");
    for s in samples {
        out.push(' ');
        out.push_str(&s.short);
        out.push('=');
        let text = match s.unit {
            Unit::Milliseconds => format!("{:.2}ms", s.value),
            Unit::Percentage => format!("{:.0}%", s.value),
            Unit::Hertz => format!("{:.0}Hz", s.value),
            Unit::Bytes => format!("{:.0}B", s.value),
            Unit::Generic => format!("{}", s.value),
        };
        out.push_str(&text);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counter_paths_lose_their_prefix() {
        assert_eq!(short_name("/perfmetrics_meta/app/gpu_frametime"), "app/gpu_frametime");
        assert_eq!(short_name("/other/thing"), "other/thing");
    }

    #[test]
    fn per_core_utilisations_are_left_out_but_the_average_stays() {
        assert!(!worth_logging("device/cpu0_util_percentage"));
        assert!(!worth_logging("device/cpu7_util_percentage"));
        assert!(worth_logging("device/cpu_util_average_percentage"));
        assert!(worth_logging("device/gpu_util_percentage"));
        assert!(worth_logging("app/gpu_frametime"));
    }

    #[test]
    fn the_line_carries_each_value_in_its_unit() {
        let line = format_line(&[
            Sample { short: "app/gpu_frametime".into(), value: 12.345, unit: Unit::Milliseconds },
            Sample { short: "device/gpu_util_percentage".into(), value: 87.0, unit: Unit::Percentage },
            Sample { short: "compositor/dropped_frame_count".into(), value: 3.0, unit: Unit::Generic },
        ]);
        assert_eq!(
            line,
            "XRPERF: app/gpu_frametime=12.35ms device/gpu_util_percentage=87% compositor/dropped_frame_count=3"
        );
    }
}
