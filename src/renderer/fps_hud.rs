//! THE FRAME-RATE READOUT: a small panel in front of the player's view with
//! the frames a second the app is making, and what the runtime says the app's
//! GPU and CPU spent on each. Toggled in play (the Y button).
//!
//! Drawn by the COMPOSITOR as its own quad layer (`xr_renderer::fps_layer`),
//! not into the eye images: it is sharp whatever the eyes' resolution, SpaceWarp
//! never extrapolates it (so it writes no motion vectors), and it costs the
//! frame nothing but a small upload twice a second.
//!
//! This half is what can be tested on the host: the meter and the picture.

use std::time::{Duration, Instant};

/// How often the numbers change: often enough to follow, slow enough to read.
pub const UPDATE_EVERY: Duration = Duration::from_millis(500);

/// The panel's pixels: three lines of the 5x7 font at `SCALE`.
pub const WIDTH_PX: u32 = 320;
pub const HEIGHT_PX: u32 = 144;
/// Its size and place in VIEW space (metres; x right, y up, -z ahead): low and
/// to the left, where it can be read with a glance and stays out of the way.
pub const WIDTH_M: f32 = 0.14;
pub const HEIGHT_M: f32 = WIDTH_M * HEIGHT_PX as f32 / WIDTH_PX as f32;
pub const POSITION: [f32; 3] = [-0.11, -0.11, -0.75];

const SCALE: usize = 4;
const MARGIN: usize = 16;
const LINE_HEIGHT: usize = 7 * SCALE + 12;
const ADVANCE: usize = 6 * SCALE;

/// One window's numbers.
#[derive(Clone, Debug, PartialEq)]
pub struct Reading {
    /// Frames the app made a second.
    pub fps: f64,
    /// The rate it is aiming at: the display's, or half of it under SpaceWarp.
    pub target_fps: f64,
    /// The runtime's `app/gpu_frametime` and `app/cpu_frametime`, averaged
    /// (`None` without `XR_META_performance_metrics`).
    pub gpu_ms: Option<f64>,
    pub cpu_ms: Option<f64>,
}

/// Counts frames and averages the runtime's counters between updates.
#[derive(Default)]
pub struct FpsMeter {
    start: Option<Instant>,
    frames: u32,
    gpu: (f64, u32),
    cpu: (f64, u32),
}

impl FpsMeter {
    /// One frame made at `now`, with the runtime's counters for it if it has
    /// them (`app/gpu_frametime`, `app/cpu_frametime`, in milliseconds). On
    /// the frame that closes a window, that window's reading.
    pub fn frame(&mut self, now: Instant, gpu_ms: Option<f64>, cpu_ms: Option<f64>, target_fps: f64) -> Option<Reading> {
        let Some(start) = self.start else {
            // The first frame only starts the clock: a window counts the
            // frames that END in it.
            self.start = Some(now);
            return None;
        };
        self.frames += 1;
        if let Some(g) = gpu_ms {
            self.gpu = (self.gpu.0 + g, self.gpu.1 + 1);
        }
        if let Some(c) = cpu_ms {
            self.cpu = (self.cpu.0 + c, self.cpu.1 + 1);
        }
        let elapsed = now.duration_since(start);
        if elapsed < UPDATE_EVERY {
            return None;
        }
        let mean = |(sum, n): (f64, u32)| (n > 0).then(|| sum / n as f64);
        let reading = Reading {
            fps: self.frames as f64 / elapsed.as_secs_f64(),
            target_fps,
            gpu_ms: mean(self.gpu),
            cpu_ms: mean(self.cpu),
        };
        *self = Self { start: Some(now), ..Default::default() };
        Some(reading)
    }
}

/// The readout's lines and the colour of each (sRGB), the frame rate's by how
/// close it is to the target: green at it, amber a little under, red well under.
pub fn lines(r: &Reading) -> [(String, [u8; 3]); 3] {
    let ms = |v: Option<f64>| v.map_or("--".to_string(), |v| format!("{v:.1}"));
    let share = if r.target_fps > 0.0 { r.fps / r.target_fps } else { 1.0 };
    let colour = if share >= 0.97 {
        [110, 235, 120]
    } else if share >= 0.85 {
        [255, 196, 70]
    } else {
        [255, 92, 80]
    };
    [
        (format!("{:.0}/{:.0} FPS", r.fps, r.target_fps), colour),
        (format!("GPU {} MS", ms(r.gpu_ms)), [235, 235, 235]),
        (format!("CPU {} MS", ms(r.cpu_ms)), [235, 235, 235]),
    ]
}

/// The panel as RGBA8 (sRGB, premultiplied alpha, rows top first): a dark
/// translucent card with the lines on it.
pub fn picture(lines: &[(String, [u8; 3])]) -> Vec<u8> {
    let (w, h) = (WIDTH_PX as usize, HEIGHT_PX as usize);
    let mut px = vec![0u8; w * h * 4];
    // The card: black at 60%, its corners rounded. Premultiplied, so black
    // is zeros whatever its alpha.
    let radius = 14.0f32;
    for y in 0..h {
        for x in 0..w {
            let dx = (radius - (x as f32 + 0.5)).max((x as f32 + 0.5) - (w as f32 - radius)).max(0.0);
            let dy = (radius - (y as f32 + 0.5)).max((y as f32 + 0.5) - (h as f32 - radius)).max(0.0);
            let cover = (radius + 0.5 - (dx * dx + dy * dy).sqrt()).clamp(0.0, 1.0);
            px[(y * w + x) * 4 + 3] = (153.0 * cover).round() as u8;
        }
    }
    for (row, (text, colour)) in lines.iter().enumerate() {
        let top = MARGIN + row * LINE_HEIGHT;
        for (i, ch) in text.chars().enumerate() {
            let left = MARGIN + i * ADVANCE;
            let Some(rows) = glyph(ch) else { continue };
            for (gy, bits) in rows.iter().enumerate() {
                for gx in 0..5 {
                    if bits & (0b10000 >> gx) == 0 {
                        continue;
                    }
                    for sy in 0..SCALE {
                        for sx in 0..SCALE {
                            let (x, y) = (left + gx * SCALE + sx, top + gy * SCALE + sy);
                            if x < w && y < h {
                                let p = (y * w + x) * 4;
                                px[p..p + 4].copy_from_slice(&[colour[0], colour[1], colour[2], 255]);
                            }
                        }
                    }
                }
            }
        }
    }
    px
}

/// A 5x7 glyph, one row a byte (bit 4 the left column); `None` for a space or
/// anything the readout never writes.
fn glyph(ch: char) -> Option<[u8; 7]> {
    Some(match ch {
        '0' => [0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E],
        '1' => [0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E],
        '2' => [0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F],
        '3' => [0x1F, 0x02, 0x04, 0x02, 0x01, 0x11, 0x0E],
        '4' => [0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02],
        '5' => [0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E],
        '6' => [0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E],
        '7' => [0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08],
        '8' => [0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E],
        '9' => [0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C],
        '.' => [0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C],
        '-' => [0x00, 0x00, 0x00, 0x1F, 0x00, 0x00, 0x00],
        '/' => [0x01, 0x01, 0x02, 0x04, 0x08, 0x10, 0x10],
        'C' => [0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E],
        'F' => [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10],
        'G' => [0x0E, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0F],
        'M' => [0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11],
        'P' => [0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10],
        'S' => [0x0F, 0x10, 0x10, 0x0E, 0x01, 0x01, 0x1E],
        'U' => [0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E],
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_window_counts_the_frames_that_end_in_it_and_averages_the_runtimes_times() {
        let mut m = FpsMeter::default();
        let t0 = Instant::now();
        assert_eq!(m.frame(t0, Some(99.0), None, 36.0), None, "the first frame only starts the clock");
        let mut reading = None;
        // 36 a second: a frame every 1/36 s until half a second has passed.
        for i in 1..=18 {
            let at = t0 + Duration::from_secs_f64(i as f64 / 36.0);
            if let Some(r) = m.frame(at, Some(20.0 + i as f64 % 2.0), Some(5.0), 36.0) {
                reading = Some(r);
            }
        }
        let r = reading.expect("half a second of frames closes a window");
        assert!((r.fps - 36.0).abs() < 0.01, "{}", r.fps);
        assert!((r.gpu_ms.unwrap() - 20.5).abs() < 0.01, "{:?}", r.gpu_ms);
        assert_eq!(r.cpu_ms, Some(5.0));
    }

    #[test]
    fn without_the_runtimes_counters_the_times_read_as_dashes() {
        let r = Reading { fps: 35.6, target_fps: 36.0, gpu_ms: None, cpu_ms: None };
        let l = lines(&r);
        assert_eq!(l[0].0, "36/36 FPS");
        assert_eq!(l[1].0, "GPU -- MS");
    }

    #[test]
    fn the_frame_rate_turns_amber_then_red_as_it_falls_below_the_target() {
        let at = |fps| lines(&Reading { fps, target_fps: 36.0, gpu_ms: Some(1.0), cpu_ms: Some(1.0) })[0].1;
        assert_eq!(at(36.0)[1], 235);
        assert_eq!(at(32.0), [255, 196, 70]);
        assert_eq!(at(22.0), [255, 92, 80]);
    }

    #[test]
    fn every_character_the_readout_writes_has_a_glyph_and_fits_the_panel() {
        let r = Reading { fps: 120.0, target_fps: 120.0, gpu_ms: Some(123.4), cpu_ms: Some(56.7) };
        for (text, _) in lines(&r) {
            for ch in text.chars().filter(|c| *c != ' ') {
                assert!(glyph(ch).is_some(), "no glyph for {ch:?}");
            }
            let right = MARGIN + text.chars().count() * ADVANCE;
            assert!(right <= WIDTH_PX as usize, "{text} runs off the panel ({right} px)");
        }
        assert!(MARGIN + 3 * LINE_HEIGHT <= HEIGHT_PX as usize);
        let px = picture(&lines(&r));
        assert_eq!(px.len(), (WIDTH_PX * HEIGHT_PX * 4) as usize);
        // Premultiplied: no channel above its alpha anywhere.
        assert!(px.chunks(4).all(|p| p[0] <= p[3] && p[1] <= p[3] && p[2] <= p[3]));
        // The corner is clear, the card's middle dark, and the text drawn.
        assert_eq!(px[3], 0);
        assert!(px.chunks(4).any(|p| p == [235, 235, 235, 255]));
    }
}
