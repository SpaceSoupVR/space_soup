//! TIME OF DAY in the renderer: the clock, and the thread that turns an hour
//! into the sky, the lights and the relit brush atlas.
//!
//! # What changes with the hour, and what it costs
//!
//! Nothing here adds a per-pixel cost to the level's shaders. The sky's
//! harmonics are nine uniforms whatever they hold; the sun or the moon is the
//! scene's one directional light, which every shader already shades; the brush
//! atlas is rewritten in place. What the hour moves:
//!
//! - THE SKY: [`space_soup_sky::time_of_day`] solves the atmosphere and the
//!   night sky into a 256 x 128 panorama on a worker thread (~12 ms on a
//!   desktop core, ~50 ms on a Quest 3 core), projects the harmonics, and
//!   places the sun and the moon. The sky shader draws the discs and the stars
//!   over the panorama.
//! - THE SUN'S SHADOW: a sun off its baked direction takes the level's shadow
//!   from the static sun map instead of the baked mask (`XrRenderer` binds the
//!   brushes' sunless lightmap group and the ground's unbaked map), and that
//!   map is redrawn when the direction moves -- [`REFRESH_HOURS`] apart, so a
//!   shadow edge steps less than a texel between redraws.
//! - THE BAKED BOUNCE: the brush atlas is rebuilt from its daylight layers
//!   (`space_soup_engine::daylight`, `AtlasLayers`) as `lamps + sky x k_sky + sun x k_sun`,
//!   on the worker, and uploaded when the weights move by [`RELIGHT_STEP`].
//! - THE PROBES: each photograph is relit as `lamps + (full - lamps) x
//!   daylight` from the lamps-only bake ([`relit_probe_source`]) and
//!   prefiltered again on the probe stream's worker, with the sky reflections
//!   see and the buildings' outsides, when the daylight moves by
//!   [`PROBE_RELIGHT_RATIO`] or the light by [`PROBE_RELIGHT_ANGLE_COS`].
//! - THE EYE: the meter is relit (`EyeAdaptation::relight`), its ceiling is
//!   raised for the night, and the Purkinje shift follows the adapted light.
//!
//! Frozen time (`day_minutes` 0, no lever) refreshes nothing after the first
//! snapshot: the cost of a frozen time of day is the cost of a photographed
//! sky.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{mpsc, Arc};

use space_soup_sky::time_of_day::{AtlasLayers, LayerWeights, SkySnapshot, TimeOfDayParams, TimeOfDaySky};
use space_soup_sky::{SkyIrradiance, SkySun};

/// Hours of game time between sky refreshes while time runs: 0.02 h, the sun
/// ~0.3 degrees -- a shadow from a 4 m wall moves ~2 cm, under a texel of the
/// static map, so the redraws read as continuous motion.
pub const REFRESH_HOURS: f32 = 0.02;

/// How far a layer weight must move (as a ratio) before the brush atlas is
/// relit: 3%, under what the eye resolves between two moments a second apart.
pub const RELIGHT_STEP: f32 = 0.03;

/// The exposure ceiling while the time of day lets the night in. Moonlight is
/// ~2e-6 of sunlight; at the meter's partial adaptation (0.8) a moonlit
/// meter of ~1e-6 asks for ~4e4.
pub const NIGHT_MAX_EXPOSURE: f32 = 2.0e5;

/// How far the daylight must move (as a ratio) before the probes are relit:
/// 25%, a third of a stop. Through dusk, with a 24-minute day, that is about
/// one relight every two seconds; through the day and the night, none.
pub const PROBE_RELIGHT_RATIO: f32 = 1.25;

/// How far the light must turn before the probes (their sky above all) are
/// relit: 3 degrees.
pub const PROBE_RELIGHT_ANGLE_COS: f32 = 0.9986;

/// A probe photograph relit: `lamps + (full - lamps) x daylight` in RGB, the
/// coverage (alpha) as photographed. Both as the stream holds them: six faces
/// of RGBA half floats. `full` back unchanged if the two differ in size.
///
/// The daylight part of a moonlit photograph (~1e-7) is below half-float's
/// normal range and keeps only a few bits: the reflections of a moonlit room
/// are nearly black, which is what they are beside its lamps. The meter does
/// not read these (`EyeAdaptation::relight` relights in f32).
pub fn relight_probe_faces(full: &[u8], lamps: &[u8], daylight: f32) -> Vec<u8> {
    use super::uniforms::{f16_to_f32, f32_to_f16};
    if full.len() != lamps.len() {
        return full.to_vec();
    }
    let mut out = Vec::with_capacity(full.len());
    for (f, l) in full.chunks_exact(8).zip(lamps.chunks_exact(8)) {
        for c in 0..4 {
            let a = f16_to_f32(u16::from_le_bytes([f[2 * c], f[2 * c + 1]]));
            let b = f16_to_f32(u16::from_le_bytes([l[2 * c], l[2 * c + 1]]));
            let v = if c == 3 { a } else { b + (a - b).max(0.0) * daylight };
            out.extend_from_slice(&f32_to_f16(v).to_le_bytes());
        }
    }
    out
}

/// The daylight the probes are lit at, shared between the renderer, which
/// sets it, and the probe source, which reads it whenever a photograph is
/// streamed or relit.
#[derive(Clone)]
pub struct ProbeDaylight(Arc<AtomicU32>);

impl Default for ProbeDaylight {
    fn default() -> Self {
        Self(Arc::new(AtomicU32::new(1.0f32.to_bits())))
    }
}

impl ProbeDaylight {
    pub fn get(&self) -> f32 {
        f32::from_bits(self.0.load(Ordering::Relaxed))
    }
    pub fn set(&self, daylight: f32) {
        self.0.store(daylight.to_bits(), Ordering::Relaxed);
    }
}

/// The level's probe photographs relit at whatever `daylight` holds when each
/// is read. A probe the lamps-only bake has no photograph of is scaled whole.
/// At a daylight of exactly 1 the photograph comes back untouched.
pub fn relit_probe_source(
    full: super::probe_stream::ProbeSource,
    lamps: super::probe_stream::ProbeSource,
    daylight: ProbeDaylight,
) -> super::probe_stream::ProbeSource {
    Arc::new(move |i| {
        let faces = full(i)?;
        let k = daylight.get();
        if k == 1.0 {
            return Some(faces);
        }
        let unlit;
        let lamps = match lamps(i) {
            Some(l) => l,
            None => {
                unlit = vec![0u8; faces.len()];
                unlit
            }
        };
        Some(relight_probe_faces(&faces, &lamps, k))
    })
}

/// What the worker hands back for one moment.
pub struct Prepared {
    pub snapshot: SkySnapshot,
    pub weights: LayerWeights,
    /// The daylight's share as one number (the ground's light now over the
    /// bake's), for what has no layers of its own: the meter's photographs.
    pub daylight: f32,
    /// The brush atlas relit, as its half-float mip chain, when the weights
    /// moved enough since the last one sent.
    pub atlas: Option<Vec<(Vec<u8>, u32, u32)>>,
}

struct Request {
    hour: f32,
    days: f64,
}

/// The clock and its worker.
pub struct TimeOfDayRuntime {
    pub params: TimeOfDayParams,
    /// Real seconds since the level started its day.
    seconds: f64,
    requests: mpsc::Sender<Request>,
    results: mpsc::Receiver<Prepared>,
    in_flight: bool,
    requested_hour: Option<f32>,
    /// The `(hour, minutes)` the lever forced last, so a change is seen.
    forced: Option<(Option<f32>, Option<f32>)>,
    /// The optical depth straight up, which dims the stars.
    pub zenith_depth: [f32; 3],
    pub has_layers: bool,
}

/// The hour a lever or a level asks for: `hour` pins it (the clock restarts
/// from there), `minutes` the day's length.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ClockOverride {
    pub hour: Option<f32>,
    pub day_minutes: Option<f32>,
}

impl TimeOfDayRuntime {
    /// Start the clock. `baked_sky` and `baked_sun` are what the level's bake
    /// was lit with (the photographed sky the scene names) -- the weights are
    /// relative to them. `layers` are the brush atlas's daylight layers, when
    /// the level has them.
    pub fn new(params: TimeOfDayParams, baked_sky: SkyIrradiance, baked_sun: Option<SkySun>, layers: Option<Arc<AtlasLayers>>) -> (Self, Prepared) {
        let sky = Arc::new(TimeOfDaySky::new(params));
        let zenith_depth = {
            let t = sky.atmosphere.eye_transmittance([0.0, 1.0, 0.0]);
            t.map(|v| -v.max(1e-6).ln())
        };
        let has_layers = layers.is_some();
        let mut worker = Worker { sky: sky.clone(), baked_sky, baked_sun, layers, sent: None };
        let first = worker.prepare(params.start_hour, 0.0, true);
        let (req_tx, req_rx) = mpsc::channel::<Request>();
        let (res_tx, res_rx) = mpsc::channel::<Prepared>();
        std::thread::Builder::new()
            .name("time_of_day".into())
            .spawn(move || {
                while let Ok(mut r) = req_rx.recv() {
                    // Only the newest: a backlog is moments already past.
                    while let Ok(next) = req_rx.try_recv() {
                        r = next;
                    }
                    if res_tx.send(worker.prepare(r.hour, r.days, false)).is_err() {
                        break;
                    }
                }
            })
            .ok();
        let rt = Self {
            params,
            seconds: 0.0,
            requests: req_tx,
            results: res_rx,
            in_flight: false,
            requested_hour: Some(params.start_hour),
            forced: None,
            zenith_depth,
            has_layers,
        };
        (rt, first)
    }

    /// The hour now.
    pub fn hour(&self) -> f32 {
        self.params.hour_at(self.seconds)
    }

    /// The level's own clock seconds, for the stars' twinkle.
    pub fn seconds(&self) -> f64 {
        self.seconds
    }

    /// Advance `dt` real seconds, with the lever's override, and ask the
    /// worker for a new sky when the hour has moved [`REFRESH_HOURS`]. Returns
    /// a finished one when there is one.
    pub fn advance(&mut self, dt: f32, clock: ClockOverride) -> Option<Prepared> {
        let forced = (clock.hour, clock.day_minutes);
        if self.forced != Some(forced) {
            if self.forced.is_some() || clock.hour.is_some() || clock.day_minutes.is_some() {
                if let Some(h) = clock.hour {
                    self.params.start_hour = h;
                    self.seconds = 0.0;
                }
                if let Some(m) = clock.day_minutes {
                    // Keep the hour where it is while the speed changes.
                    let hour = self.hour();
                    self.params.day_minutes = m.max(0.0);
                    self.params.start_hour = hour;
                    self.seconds = 0.0;
                }
            }
            self.forced = Some(forced);
        }
        self.seconds += dt.max(0.0) as f64;
        let hour = self.hour();
        let moved = self.requested_hour.is_none_or(|h| {
            let d = (hour - h).rem_euclid(24.0);
            d.min(24.0 - d) >= REFRESH_HOURS
        });
        if moved && !self.in_flight {
            self.in_flight = true;
            self.requested_hour = Some(hour);
            let _ = self.requests.send(Request { hour, days: self.params.days_at(self.seconds) });
        }
        match self.results.try_recv() {
            Ok(p) => {
                self.in_flight = false;
                Some(p)
            }
            Err(_) => None,
        }
    }
}

struct Worker {
    sky: Arc<TimeOfDaySky>,
    baked_sky: SkyIrradiance,
    baked_sun: Option<SkySun>,
    layers: Option<Arc<AtlasLayers>>,
    /// The weights the atlas was last relit with.
    sent: Option<LayerWeights>,
}

impl Worker {
    fn prepare(&mut self, hour: f32, days: f64, force: bool) -> Prepared {
        let snapshot = self.sky.snapshot(hour, days);
        let weights = LayerWeights::between(&self.baked_sky, self.baked_sun.as_ref(), &snapshot.irradiance, snapshot.light.as_ref());
        let daylight = daylight_share(&self.baked_sky, self.baked_sun.as_ref(), &snapshot);
        let moved = |a: &LayerWeights, b: &LayerWeights| {
            let far = |x: f32, y: f32| {
                let (x, y) = (x.max(1e-9), y.max(1e-9));
                (x / y).max(y / x) > 1.0 + RELIGHT_STEP && (x - y).abs() > 1e-7
            };
            (0..3).any(|c| far(a.sky[c], b.sky[c]) || far(a.sun[c], b.sun[c]))
        };
        let atlas = match &self.layers {
            Some(layers) if force || self.sent.as_ref().is_none_or(|s| moved(s, &weights)) => {
                self.sent = Some(weights);
                Some(crate::renderer::mesh::lightmap_mips_f16(&layers.combine(weights.sky, weights.sun), layers.width, layers.height))
            }
            _ => None,
        };
        Prepared { snapshot, weights, daylight, atlas }
    }
}

/// The daylight now over the bake's, as one number: the ground's light (sky
/// on a horizontal plane plus the sun or moon at its elevation), luminance.
pub fn daylight_share(baked_sky: &SkyIrradiance, baked_sun: Option<&SkySun>, now: &SkySnapshot) -> f32 {
    let lum = |c: [f32; 3]| 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
    let ground = |sky: &SkyIrradiance, sun: Option<&SkySun>| {
        lum(sky.evaluate([0.0, 1.0, 0.0])) + sun.map_or(0.0, |s| lum(s.light_rgb) * s.direction[1].max(0.0))
    };
    let then = ground(baked_sky, baked_sun);
    if then > 0.0 { ground(&now.irradiance, now.light.as_ref()) / then } else { 0.0 }
}

/// The night-vision amount for an eye adapted to `adapted` engine units of
/// luminance under a time-of-day sky whose calibration is
/// `candelas_per_engine`. See `tonemap::night_vision_for`.
pub fn night_vision(adapted: f32, candelas_per_engine: f32) -> f32 {
    crate::renderer::tonemap::night_vision_for(adapted * candelas_per_engine)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relit_photographs_keep_the_lamps_and_scale_the_daylight() {
        use crate::renderer::uniforms::{f16_to_f32, f32_to_f16};
        let texel = |v: [f32; 4]| v.iter().flat_map(|x| f32_to_f16(*x).to_le_bytes()).collect::<Vec<u8>>();
        let read = |b: &[u8]| -> Vec<f32> { b.chunks_exact(2).map(|p| f16_to_f32(u16::from_le_bytes([p[0], p[1]]))).collect() };
        let full = texel([1.0, 0.5, 0.25, 0.75]);
        let lamps = texel([0.25, 0.25, 0.25, 0.75]);
        assert_eq!(read(&relight_probe_faces(&full, &lamps, 1.0)), read(&full));
        assert_eq!(read(&relight_probe_faces(&full, &lamps, 0.0)), read(&lamps));
        assert_eq!(read(&relight_probe_faces(&full, &lamps, 0.5))[0], 0.625);
        // Through the source: the daylight it reads when asked.
        let (f, l) = (full.clone(), lamps.clone());
        let daylight = ProbeDaylight::default();
        let source = relit_probe_source(Arc::new(move |_| Some(f.clone())), Arc::new(move |_| Some(l.clone())), daylight.clone());
        assert_eq!(source(0).unwrap(), full);
        daylight.set(0.0);
        assert_eq!(read(&source(0).unwrap()), read(&lamps));
    }

    fn hdri_like() -> (SkyIrradiance, SkySun) {
        let sky = SkyIrradiance::flat(0.53);
        let sun = SkySun { direction: [0.3775, 0.7417, 0.5544], light_rgb: [1.34, 1.35, 1.23], texels: 9, energy_fraction: 0.48 };
        (sky, sun)
    }

    #[test]
    fn a_frozen_clock_asks_once_and_a_running_one_every_refresh() {
        let (sky, sun) = hdri_like();
        let (mut rt, first) = TimeOfDayRuntime::new(TimeOfDayParams::default(), sky, Some(sun), None);
        assert!(first.snapshot.light.is_some());
        // Frozen: the hour never moves, nothing is asked for.
        for _ in 0..200 {
            assert!(rt.advance(0.1, ClockOverride::default()).is_none());
        }
        assert!(!rt.in_flight);
        // A one-minute day: 0.02 h is 0.05 s of real time.
        let mut got = 0;
        for _ in 0..400 {
            if rt.advance(0.01, ClockOverride { hour: None, day_minutes: Some(1.0) }).is_some() {
                got += 1;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        assert!(got >= 2, "{got}");
    }

    #[test]
    fn the_lever_pins_the_hour() {
        let (sky, sun) = hdri_like();
        let (mut rt, _) = TimeOfDayRuntime::new(TimeOfDayParams::default(), sky, Some(sun), None);
        rt.advance(0.0, ClockOverride { hour: Some(22.5), day_minutes: None });
        assert!((rt.hour() - 22.5).abs() < 1e-4);
        // Waits for the worker's answer for 22.5 h.
        let mut answer = None;
        for _ in 0..2000 {
            if let Some(p) = rt.advance(0.0, ClockOverride { hour: Some(22.5), day_minutes: None }) {
                answer = Some(p);
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        let p = answer.expect("the worker answers");
        assert!((p.snapshot.hour - 22.5).abs() < 1e-4);
        assert!(p.daylight < 1e-4, "night: {}", p.daylight);
    }

    #[test]
    fn the_atlas_is_relit_at_first_and_then_only_when_the_light_moves() {
        let (sky, sun) = hdri_like();
        let layers = Arc::new(AtlasLayers {
            width: 2,
            height: 2,
            lamps: vec![0.01; 16],
            sky: vec![0.2; 16],
            sun: vec![0.5; 16],
        });
        let params = TimeOfDayParams::default();
        let ts = Arc::new(TimeOfDaySky::new(params));
        let mut w = Worker { sky: ts, baked_sky: sky, baked_sun: Some(sun), layers: Some(layers), sent: None };
        assert!(w.prepare(15.0, 0.0, true).atlas.is_some());
        assert!(w.prepare(15.0, 0.0, false).atlas.is_none(), "same light, no upload");
        assert!(w.prepare(3.0, 0.0, false).atlas.is_some(), "night relights");
    }
}
