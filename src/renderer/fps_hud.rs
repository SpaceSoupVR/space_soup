use std::time::{Duration, Instant};

pub const UPDATE_EVERY: Duration = Duration::from_millis(500);

pub const WIDTH_PX: u32 = 320;
pub const HEIGHT_PX: u32 = 144;

pub const WIDTH_M: f32 = 0.14;
pub const HEIGHT_M: f32 = WIDTH_M * HEIGHT_PX as f32 / WIDTH_PX as f32;
pub const POSITION: [f32; 3] = [-0.17, -0.16, -0.75];

const SCALE: usize = 4;
const MARGIN: usize = 16;
const LINE_HEIGHT: usize = 7 * SCALE + 12;
const ADVANCE: usize = 6 * SCALE;

#[derive(Clone, Debug, PartialEq)]
pub struct Reading {

    pub fps: f64,

    pub target_fps: f64,

    pub gpu_ms: Option<f64>,
    pub cpu_ms: Option<f64>,
}

#[derive(Default)]
pub struct FpsMeter {
    start: Option<Instant>,
    frames: u32,
    gpu: (f64, u32),
    cpu: (f64, u32),
}

impl FpsMeter {

    pub fn frame(&mut self, now: Instant, gpu_ms: Option<f64>, cpu_ms: Option<f64>, target_fps: f64) -> Option<Reading> {
        let Some(start) = self.start else {

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
        let mean = |(sum, n): (f64, u32)| (n > 0).then(|| sum / f64::from(n));
        let reading = Reading {
            fps: f64::from(self.frames) / elapsed.as_secs_f64(),
            target_fps,
            gpu_ms: mean(self.gpu),
            cpu_ms: mean(self.cpu),
        };
        *self = Self { start: Some(now), ..Default::default() };
        Some(reading)
    }
}

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

pub fn picture(lines: &[(String, [u8; 3])]) -> Vec<u8> {
    let (w, h) = (WIDTH_PX as usize, HEIGHT_PX as usize);
    let mut px = vec![0u8; w * h * 4];

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

        for i in 1..=18 {
            let at = t0 + Duration::from_secs_f64(f64::from(i) / 36.0);
            if let Some(r) = m.frame(at, Some(20.0 + f64::from(i) % 2.0), Some(5.0), 36.0) {
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

        assert!(px.chunks(4).all(|p| p[0] <= p[3] && p[1] <= p[3] && p[2] <= p[3]));

        assert_eq!(px[3], 0);
        assert!(px.chunks(4).any(|p| p == [235, 235, 235, 255]));
    }
}
