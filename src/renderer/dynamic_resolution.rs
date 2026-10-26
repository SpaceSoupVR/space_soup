#[derive(Clone, Debug, Default, PartialEq)]
pub struct RecommendationWindow {

    pub asked: u32,

    pub valid: u32,

    pub errors: u32,
    pub last_error: i32,

    pub last: Option<(u32, u32)>,
    pub smallest: Option<(u32, u32)>,
    pub largest: Option<(u32, u32)>,
}

impl RecommendationWindow {

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

    pub fn take(&mut self) -> Self {
        std::mem::take(self)
    }

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

/// DYNAMIC RESOLUTION (the colleague's plan part 2, item 4): the eye drawn
/// into the top-left `scale` of its targets, allocated once at full size,
/// and handed to the compositor as that rectangle. The scale is folded into
/// the eye's projection (`fold`): the whole field of view lands in the
/// rectangle's NDC corner, so every shader that maps NDC to texels over its
/// whole texture stays right, and a scissor keeps the rest of the target
/// unshaded. The frame is fill-bound (shading scales with pixels), so this
/// buys GPU time back where the frame is over budget and gives it back
/// where it is not.
///
/// Down fast (a step after `DOWN_FRAMES` over `DOWN_OVER` of the frame),
/// up slowly (a step after `UP_FRAMES` under `UP_UNDER`), a step at most
/// every `HOLD_S`, so the sharpness never pumps.
pub const STEP: f32 = 0.05;
pub const DOWN_OVER: f32 = 0.9;
pub const DOWN_FRAMES: u32 = 3;
pub const UP_UNDER: f32 = 0.75;
pub const UP_FRAMES: u32 = 60;
pub const HOLD_S: f32 = 0.5;

#[derive(Clone, Debug)]
pub struct ScaleController {
    scale: f32,
    over: u32,
    under: u32,
    since: f32,
}

impl Default for ScaleController {
    fn default() -> Self {
        Self { scale: 1.0, over: 0, under: 0, since: HOLD_S }
    }
}

impl ScaleController {
    /// One frame: its GPU time (ms; None keeps the scale), the app's frame
    /// period (ms) and length (s), and the least scale allowed. Returns the
    /// scale to draw at, a whole number of `STEP`s.
    pub fn step(&mut self, gpu_ms: Option<f32>, period_ms: f32, dt: f32, min_scale: f32) -> f32 {
        self.since += dt;
        let min_scale = min_scale.clamp(STEP, 1.0);
        if let Some(gpu) = gpu_ms.filter(|g| g.is_finite() && *g > 0.0 && period_ms > 0.0) {
            if gpu > DOWN_OVER * period_ms {
                self.over += 1;
                self.under = 0;
            } else if gpu < UP_UNDER * period_ms {
                self.under += 1;
                self.over = 0;
            } else {
                self.over = 0;
                self.under = 0;
            }
            if self.since >= HOLD_S {
                if self.over >= DOWN_FRAMES && self.scale > min_scale {
                    self.scale = (self.scale - STEP).max(min_scale);
                    self.over = 0;
                    self.since = 0.0;
                } else if self.under >= UP_FRAMES && self.scale < 1.0 {
                    self.scale = (self.scale + STEP).min(1.0);
                    self.under = 0;
                    self.since = 0.0;
                }
            }
        }
        self.scale = (self.scale / STEP).round() * STEP;
        self.scale.clamp(min_scale, 1.0)
    }

    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

/// The step a scale is (for caching per-scale data: foveation maps).
pub fn step_of(scale: f32) -> u32 {
    (scale / STEP).round() as u32
}

/// The rectangle drawn at `scale` of a `w` x `h` target: its top-left part,
/// at least one texel.
pub fn rect(w: u32, h: u32, scale: f32) -> (u32, u32) {
    if scale >= 1.0 {
        return (w, h);
    }
    (((w as f32 * scale).ceil() as u32).clamp(1, w), ((h as f32 * scale).ceil() as u32).clamp(1, h))
}

/// The clip-space map that draws a whole view into the top-left `scale` of
/// the target (wgpu NDC, y up): x' = s x + (s - 1) w, y' = s y + (1 - s) w.
pub fn fold(scale: f32) -> glam::Mat4 {
    if scale >= 1.0 {
        return glam::Mat4::IDENTITY;
    }
    let s = scale;
    glam::Mat4::from_cols(
        glam::Vec4::new(s, 0.0, 0.0, 0.0),
        glam::Vec4::new(0.0, s, 0.0, 0.0),
        glam::Vec4::new(0.0, 0.0, 1.0, 0.0),
        glam::Vec4::new(s - 1.0, 1.0 - s, 0.0, 1.0),
    )
}

#[cfg(test)]
mod scale_tests {
    use super::*;

    const PERIOD: f32 = 1000.0 / 36.0;

    fn run(c: &mut ScaleController, gpu: f32, frames: u32) -> f32 {
        (0..frames).map(|_| c.step(Some(gpu), PERIOD, 1.0 / 36.0, 0.7)).last().unwrap_or(1.0)
    }

    #[test]
    fn a_heavy_load_drops_within_three_frames_and_recovers_only_slowly() {
        let mut c = ScaleController::default();
        assert_eq!(run(&mut c, 20.0, 100), 1.0, "a frame in budget stays at full size");
        assert!((run(&mut c, 26.0, 3) - 0.95).abs() < 1e-4, "three frames over: one step down");
        assert!((run(&mut c, 26.0, 3) - 0.95).abs() < 1e-4, "not again inside the hold");
        assert!((run(&mut c, 26.0, 200) - 0.7).abs() < 1e-4, "down to the floor and no further");
        assert!((run(&mut c, 18.0, 59) - 0.7).abs() < 1e-4, "not up before 60 frames");
        assert!((run(&mut c, 18.0, 1) - 0.75).abs() < 1e-4, "then one step up");
        assert!((run(&mut c, 18.0, 2000) - 1.0).abs() < 1e-4, "and back to full, never past it");
    }

    #[test]
    fn between_the_lines_the_scale_holds() {
        let mut c = ScaleController::default();
        run(&mut c, 26.0, 40);
        let held = run(&mut c, 0.8 * PERIOD, 1000);
        assert!((held - run(&mut c, 0.8 * PERIOD, 10)).abs() < 1e-6);
        assert!(held < 1.0);
    }

    #[test]
    fn the_fold_puts_the_whole_view_in_the_top_left_of_the_target() {
        let f = fold(0.5);
        let at = |x: f32, y: f32| {
            let c = f * glam::Vec4::new(x * 2.0, y * 2.0, 0.5, 2.0);
            (c.x / c.w, c.y / c.w)
        };
        // NDC -1..1 lands in x -1..0 (left half) and y 0..1 (top half).
        assert_eq!(at(-1.0, 1.0), (-1.0, 1.0));
        assert_eq!(at(1.0, -1.0), (0.0, 0.0));
        assert_eq!(fold(1.0), glam::Mat4::IDENTITY);
        assert_eq!(rect(1448, 1552, 0.5), (724, 776));
        assert_eq!(rect(1448, 1552, 1.0), (1448, 1552));
        assert_eq!(step_of(0.75), 15);
    }
}
