//! Native frame rate when the GPU has room for it, `SpaceWarp` when it has not
//! (the AAA doc's rank 11: "a 72 <-> 36+ASW governor driven by GPU time").
//!
//! Under `SpaceWarp` the app renders every other display frame and the runtime
//! synthesises the rest from motion vectors; a frame rendered natively shows
//! no synthesis error at all (no smear on thin or transparent things). So the
//! frame is rendered natively whenever its GPU time fits the display's
//! period with a margin, and handed back to `SpaceWarp` the moment it does not.
//! Going native is slow (held for seconds: a view must stay light), coming
//! back is fast (a quarter of a second: a dropped native frame judders), and
//! after a return the governor waits before trying again, so it never flaps.

/// Go native once the smoothed GPU time is under this share of the display
/// period (10.4 ms at 72 Hz), held for `NATIVE_HOLD_S`.
pub const NATIVE_UNDER: f32 = 0.75;
/// Back to `SpaceWarp` once it is over this share (12.5 ms at 72 Hz), held for
/// `WARP_HOLD_S`.
pub const WARP_OVER: f32 = 0.9;
pub const NATIVE_HOLD_S: f32 = 5.0;
pub const WARP_HOLD_S: f32 = 0.25;
/// After a return to `SpaceWarp`, no native try for this long.
pub const RETRY_AFTER_S: f32 = 10.0;
/// The smoothing of the GPU time, per frame.
const SMOOTH: f32 = 0.1;

#[derive(Clone, Debug, Default)]
pub struct FrameGovernor {
    native: bool,
    smoothed: Option<f32>,
    /// Seconds the current condition has held.
    held: f32,
    /// Seconds since the last return to `SpaceWarp`.
    since_return: f32,
}

impl FrameGovernor {
    /// One rendered frame: its GPU time (ms; `None` keeps the state), the
    /// display's own period (ms: 13.9 at 72 Hz, whatever the app renders at)
    /// and the frame's length in seconds. Returns whether to render natively.
    pub fn step(&mut self, gpu_ms: Option<f32>, display_period_ms: f32, dt: f32) -> bool {
        self.since_return += dt;
        let Some(gpu) = gpu_ms.filter(|g| g.is_finite() && *g > 0.0) else {
            return self.native;
        };
        let s = match self.smoothed {
            Some(s) => s + (gpu - s) * SMOOTH,
            None => gpu,
        };
        self.smoothed = Some(s);
        let leave = if self.native {
            s > WARP_OVER * display_period_ms
        } else {
            s < NATIVE_UNDER * display_period_ms && self.since_return >= RETRY_AFTER_S
        };
        self.held = if leave { self.held + dt } else { 0.0 };
        let hold = if self.native { WARP_HOLD_S } else { NATIVE_HOLD_S };
        if leave && self.held >= hold {
            self.native = !self.native;
            self.held = 0.0;
            if !self.native {
                self.since_return = 0.0;
            }
        }
        self.native
    }

    pub fn native(&self) -> bool {
        self.native
    }

    pub fn smoothed(&self) -> Option<f32> {
        self.smoothed
    }

    /// Back to `SpaceWarp` now, as when the governor is switched off.
    pub fn reset(&mut self) {
        *self = Self { since_return: 0.0, ..Self::default() };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PERIOD: f32 = 1000.0 / 72.0;

    fn run(g: &mut FrameGovernor, gpu: f32, seconds: f32, fps: f32) -> bool {
        let n = (seconds * fps) as usize;
        (0..n).map(|_| g.step(Some(gpu), PERIOD, 1.0 / fps)).last().unwrap_or(g.native())
    }

    #[test]
    fn a_light_frame_goes_native_only_after_holding_and_a_heavy_one_returns_fast() {
        let mut g = FrameGovernor { since_return: RETRY_AFTER_S, ..FrameGovernor::default() };
        assert!(!run(&mut g, 9.0, 4.0, 36.0), "not before the hold");
        assert!(run(&mut g, 9.0, 1.5, 36.0), "native once held");
        assert!(run(&mut g, 13.5, 0.1, 72.0), "a moment over does not return");
        assert!(!run(&mut g, 13.5, 0.6, 72.0), "held over, back to SpaceWarp");
        assert!(!run(&mut g, 9.0, 8.0, 36.0), "no retry inside the wait");
        assert!(run(&mut g, 9.0, 8.0, 36.0), "then native again");
    }

    #[test]
    fn a_frame_between_the_lines_stays_where_it_is() {
        let mut g = FrameGovernor { since_return: RETRY_AFTER_S, ..FrameGovernor::default() };
        assert!(!run(&mut g, 11.5, 30.0, 36.0), "11.5 ms is not under 10.4: stays on SpaceWarp");
        run(&mut g, 9.0, 6.0, 36.0);
        assert!(run(&mut g, 11.5, 30.0, 72.0), "nor over 12.5: stays native");
    }

    #[test]
    fn no_reading_keeps_the_state() {
        let mut g = FrameGovernor::default();
        assert!(!g.step(None, PERIOD, 1.0));
        assert!(!g.step(Some(f32::NAN), PERIOD, 1.0));
    }
}
