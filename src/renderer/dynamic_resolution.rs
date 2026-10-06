//! DYNAMIC RESOLUTION: what the runtime recommends for the eyes' layer each
//! frame (`xr::recommended_resolution`, Android only), kept here where a
//! development machine's tests run it.
//!
//! First step (cortex wall #16): ask every frame and draw the eyes at their
//! fixed size whatever the answer -- asking is what Quest 3 grants GPU level 5
//! for -- and log each window's answers beside the clock the runtime chose, so
//! one bench says both whether the level comes and what size the runtime would
//! have had us draw.

/// One window's answers, for the `DYNRES` line.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RecommendationWindow {
    /// Frames asked.
    pub asked: u32,
    /// Answers with a recommendation (`isValid`).
    pub valid: u32,
    /// Calls the runtime refused, and the last refusal's code.
    pub errors: u32,
    pub last_error: i32,
    /// The latest recommendation, and the smallest and largest of the window,
    /// each as (width, height) of one view.
    pub last: Option<(u32, u32)>,
    pub smallest: Option<(u32, u32)>,
    pub largest: Option<(u32, u32)>,
}

impl RecommendationWindow {
    /// One frame's answer: `Ok(None)` when the runtime had no recommendation,
    /// `Err(code)` when it refused the call.
    pub fn add(&mut self, answer: Result<Option<(u32, u32)>, i32>) {
        self.asked += 1;
        match answer {
            Ok(Some(size)) => {
                self.valid += 1;
                self.last = Some(size);
                let area = |s: (u32, u32)| u64::from(s.0) * u64::from(s.1);
                if self.smallest.is_none_or(|s| area(size) < area(s)) {
                    self.smallest = Some(size);
                }
                if self.largest.is_none_or(|s| area(size) > area(s)) {
                    self.largest = Some(size);
                }
            }
            Ok(None) => {}
            Err(code) => {
                self.errors += 1;
                self.last_error = code;
            }
        }
    }

    /// The window so far, starting a new one.
    pub fn take(&mut self) -> Self {
        std::mem::take(self)
    }

    /// The `DYNRES` line: what was asked and answered, against `drawn`, the
    /// size each eye was drawn at; the recommendation as a share of it too,
    /// since that is the render scale the runtime wanted.
    pub fn format_line(&self, drawn: (u32, u32)) -> String {
        let size = |s: Option<(u32, u32)>| s.map_or("-".to_string(), |(w, h)| format!("{w}x{h}"));
        let scale = |s: Option<(u32, u32)>| {
            s.map_or("-".to_string(), |(w, h)| {
                let share = ((f64::from(w) * f64::from(h)) / (f64::from(drawn.0.max(1)) * f64::from(drawn.1.max(1)))).sqrt();
                format!("{share:.3}")
            })
        };
        let mut line = format!(
            "DYNRES: asked={} valid={} recommended={} scale={} range={}..{} drawn={}x{}",
            self.asked,
            self.valid,
            size(self.last),
            scale(self.last),
            size(self.smallest),
            size(self.largest),
            drawn.0,
            drawn.1,
        );
        if self.errors > 0 {
            line.push_str(&format!(" errors={} last_error={}", self.errors, self.last_error));
        }
        line
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_window_keeps_the_latest_smallest_and_largest_answers() {
        let mut w = RecommendationWindow::default();
        w.add(Ok(Some((1680, 1760))));
        w.add(Ok(None));
        w.add(Ok(Some((1440, 1504))));
        w.add(Ok(Some((1600, 1680))));
        w.add(Err(-1));
        assert_eq!((w.asked, w.valid, w.errors, w.last_error), (5, 3, 1, -1));
        assert_eq!(w.last, Some((1600, 1680)));
        assert_eq!(w.smallest, Some((1440, 1504)));
        assert_eq!(w.largest, Some((1680, 1760)));
        let line = w.format_line((1680, 1760));
        assert_eq!(
            line,
            "DYNRES: asked=5 valid=3 recommended=1600x1680 scale=0.953 range=1440x1504..1680x1760 drawn=1680x1760 errors=1 last_error=-1"
        );
        let taken = w.take();
        assert_eq!(taken.asked, 5);
        assert_eq!(w, RecommendationWindow::default(), "taking starts a new window");
    }

    #[test]
    fn a_window_with_no_recommendation_says_so() {
        let mut w = RecommendationWindow::default();
        w.add(Ok(None));
        assert_eq!(w.format_line((1440, 1584)), "DYNRES: asked=1 valid=0 recommended=- scale=- range=-..- drawn=1440x1584");
    }
}
