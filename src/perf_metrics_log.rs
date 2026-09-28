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

/// Every counter over the frames of one `PERF` window.
///
/// A counter read once, on the frame a window closes, is ONE frame, and one
/// frame of a GPU-bound scene sits easily a millisecond either side of the
/// window it stands for -- the size of most of the effects being measured.
/// Times, rates and percentages are summed every measured frame and averaged
/// at the close; counts and sizes are left at their last value, which is what
/// a running total means.
#[derive(Default)]
pub struct WindowMeans {
    counters: Vec<(String, Unit, f64, u32)>,
}

impl WindowMeans {
    pub fn add(&mut self, samples: &[Sample]) {
        for s in samples {
            let averaged = matches!(s.unit, Unit::Milliseconds | Unit::Percentage | Unit::Hertz);
            match self.counters.iter_mut().find(|c| c.0 == s.short) {
                Some(c) if averaged => {
                    c.2 += s.value;
                    c.3 += 1;
                }
                Some(c) => {
                    c.2 = s.value;
                    c.3 = 1;
                }
                None => self.counters.push((s.short.clone(), s.unit, s.value, 1)),
            }
        }
    }

    /// The window's values, and a fresh start for the next one.
    pub fn take(&mut self) -> Vec<Sample> {
        let out = self
            .counters
            .iter()
            .map(|(short, unit, sum, n)| Sample { short: short.clone(), value: sum / f64::from(*n), unit: *unit })
            .collect();
        self.counters.clear();
        out
    }

    pub fn clear(&mut self) {
        self.counters.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_window_averages_times_and_keeps_the_last_count() {
        let mut w = WindowMeans::default();
        let frame = |gpu: f64, dropped: f64| {
            vec![
                Sample { short: "app/gpu_frametime".into(), value: gpu, unit: Unit::Milliseconds },
                Sample { short: "compositor/dropped_frame_count".into(), value: dropped, unit: Unit::Generic },
            ]
        };
        w.add(&frame(10.0, 1.0));
        w.add(&frame(14.0, 3.0));
        // A counter the runtime declined one frame is averaged over the frames
        // it answered, not diluted by a zero.
        w.add(&[Sample { short: "app/gpu_frametime".into(), value: 12.0, unit: Unit::Milliseconds }]);
        let out = w.take();
        assert_eq!(out[0].value, 12.0, "{out:?}");
        assert_eq!(out[1].value, 3.0, "{out:?}");
        assert!(w.take().is_empty(), "the next window starts empty");
    }

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
