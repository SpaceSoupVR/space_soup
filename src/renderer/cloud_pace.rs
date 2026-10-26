//! How often the sky map is built, from the headset's own GPU frame time.
//!
//! A slice of the sky map costs about 1.5 ms of GPU a frame on the Quest 3
//! (`time_the_sky_map`, 2026-10-22). The map is carried on its winds between
//! builds, so building a slice every second or fourth frame holds the clouds
//! where they are and only renews their shape more slowly. When the frame is
//! over budget that is the cheapest time to buy back: in the user's live
//! session (2026-10-22, 239 windows of `XRPERF`) frames dropped from a mean
//! `app/gpu_frametime` of about 22 ms in a 27.7 ms `SpaceWarp` period (11 drops
//! a 3.5 s window at 20-22 ms, 16 at 22-24, 26 at 24-26, 36 at 26-28), and
//! half the windows were past it.

/// Past this share of the frame period (smoothed GPU time) the build slows.
pub const OVER: f32 = 0.8;
/// Under this share it speeds up again; the gap (3.3 ms at 36 fps) is wider
/// than what a halving buys (0.4-0.75 ms), so the pace does not swing.
pub const UNDER: f32 = 0.68;
/// The slowest pace: a slice every this many frames.
pub const MOST: u32 = 4;
/// Frames held at a pace before it may change again (2 s at 36 fps): long
/// enough for the GPU counter, which lags, to show what the change did.
pub const HOLD: u32 = 72;
/// The smoothing of the GPU time, per frame (about 20 frames).
const SMOOTH: f32 = 0.05;

#[derive(Clone, Debug)]
pub struct CloudPace {
    smoothed: Option<f32>,
    every: u32,
    held: u32,
}

impl Default for CloudPace {
    fn default() -> Self {
        Self { smoothed: None, every: 1, held: 0 }
    }
}

impl CloudPace {
    /// This frame's pace from the app's GPU frame time (ms, `None` when the
    /// runtime gives none: the pace stays) and the frame period it must fit.
    pub fn step(&mut self, gpu_ms: Option<f32>, period_ms: f32) -> u32 {
        let Some(gpu) = gpu_ms.filter(|g| g.is_finite() && *g > 0.0) else {
            return self.every;
        };
        let s = match self.smoothed {
            Some(s) => s + (gpu - s) * SMOOTH,
            None => gpu,
        };
        self.smoothed = Some(s);
        self.held = self.held.saturating_add(1);
        if self.held >= HOLD && period_ms > 0.0 {
            if s > OVER * period_ms && self.every < MOST {
                self.every *= 2;
                self.held = 0;
            } else if s < UNDER * period_ms && self.every > 1 {
                self.every /= 2;
                self.held = 0;
            }
        }
        self.every
    }

    /// The pace now: a slice every this many frames.
    pub fn every(&self) -> u32 {
        self.every
    }

    /// The smoothed GPU time, ms.
    pub fn smoothed(&self) -> Option<f32> {
        self.smoothed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PERIOD: f32 = 1000.0 / 36.0;

    fn run(pace: &mut CloudPace, gpu: f32, frames: u32) -> u32 {
        (0..frames).map(|_| pace.step(Some(gpu), PERIOD)).last().unwrap_or(1)
    }

    #[test]
    fn an_over_budget_frame_slows_the_sky_map_and_a_light_one_restores_it() {
        let mut pace = CloudPace::default();
        assert_eq!(run(&mut pace, 18.0, 500), 1, "inside the budget every frame builds");
        assert_eq!(run(&mut pace, 24.0, 40), 2, "over it, half the rate");
        assert_eq!(run(&mut pace, 24.0, HOLD), 4, "still over, a quarter");
        assert_eq!(run(&mut pace, 24.0, 10 * HOLD), MOST, "and no slower");
        assert_eq!(run(&mut pace, 20.0, 10 * HOLD), MOST, "between the lines it holds");
        assert_eq!(run(&mut pace, 17.0, 40), 2);
        assert_eq!(run(&mut pace, 17.0, HOLD), 1);
    }

    #[test]
    fn one_slow_frame_does_not_move_the_pace() {
        let mut pace = CloudPace::default();
        run(&mut pace, 18.0, 500);
        pace.step(Some(40.0), PERIOD);
        assert_eq!(run(&mut pace, 18.0, 500), 1);
    }

    #[test]
    fn no_reading_keeps_the_pace() {
        let mut pace = CloudPace::default();
        run(&mut pace, 25.0, HOLD + 40);
        assert_eq!(pace.step(None, PERIOD), 2);
        assert_eq!(pace.step(Some(f32::NAN), PERIOD), 2);
    }
}
