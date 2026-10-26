use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{mpsc, Arc};

use space_soup_sky::time_of_day::{AtlasLayers, LayerWeights, SkySnapshot, TimeOfDayParams, TimeOfDaySky};
use space_soup_sky::{SkyIrradiance, SkySun};

pub const REFRESH_HOURS: f32 = 0.02;

pub const RELIGHT_STEP: f32 = 0.03;

/// The night eye's widest opening: past what starlight alone asks
/// (`exposure::DEEP_NIGHT_LUMINANCE`, about x1.4e7), so the adaptation curve
/// decides it, not the cap. At 1e6 a moonless night drew black (2026-10-22).
pub const NIGHT_MAX_EXPOSURE: f32 = 1.0e8;

pub const PROBE_RELIGHT_RATIO: f32 = 1.25;

/// The brightest texel a relit atlas is stored at, in half floats: room
/// below f16's 65504 for the mips' averages to stay finite.
pub const ATLAS_PEAK: f32 = 16384.0;

/// HOW MANY STOPS BRIGHTER A RELIT ATLAS IS STORED, so its brightest texel
/// sits near `ATLAS_PEAK`: whole stops, so the scale moves no mantissa bit.
/// Under a new moon the daylight layers are x4e-9 of the bake's, below f16's
/// smallest (6e-8), and the atlas stored exactly zero however far the eye
/// opened (2026-10-22); stored brighter, starlight keeps its texels. Never
/// dimmer than as baked. Alpha (sky visibility) is not light, and is left.
pub fn atlas_stops(rgba: &[f32]) -> f32 {
    let peak = rgba.chunks_exact(4).map(|p| p[0].max(p[1]).max(p[2])).fold(0.0f32, f32::max);
    if !(peak > 0.0) {
        return 0.0;
    }
    (ATLAS_PEAK / peak).log2().floor().clamp(0.0, 100.0)
}

/// `rgba`'s light (not its alpha) `stops` brighter.
pub fn scale_atlas(rgba: &mut [f32], stops: f32) {
    if stops == 0.0 {
        return;
    }
    let k = stops.exp2();
    for p in rgba.chunks_exact_mut(4) {
        for v in &mut p[..3] {
            *v *= k;
        }
    }
}

pub const PROBE_RELIGHT_ANGLE_COS: f32 = 0.9986;

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

/// A BUILDING'S OUTSIDE under the hour, from its daylight alone: `(full -
/// lamps) * daylight`, alpha kept. Its lamps-only photograph is the inside
/// (lit through doorways, and the hallway's through its walls), which every
/// wall reflecting a neighbour showed at night as a lit stone face
/// (2026-10-20). A building's outside takes almost none of its lamps' light.
pub fn building_daylight_faces(full: &[u8], lamps: &[u8], daylight: f32) -> Vec<u8> {
    use super::uniforms::{f16_to_f32, f32_to_f16};
    let mut out = Vec::with_capacity(full.len());
    for (k, f) in full.chunks_exact(8).enumerate() {
        let l = lamps.get(k * 8..k * 8 + 8);
        for c in 0..4 {
            let a = f16_to_f32(u16::from_le_bytes([f[2 * c], f[2 * c + 1]]));
            let b = l.map_or(0.0, |l| f16_to_f32(u16::from_le_bytes([l[2 * c], l[2 * c + 1]])));
            let v = if c == 3 { a } else { (a - b).max(0.0) * daylight };
            out.extend_from_slice(&f32_to_f16(v).to_le_bytes());
        }
    }
    out
}

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
        let lamps = if let Some(l) = lamps(i) { l } else {
            unlit = vec![0u8; faces.len()];
            unlit
        };
        Some(relight_probe_faces(&faces, &lamps, k))
    })
}

pub struct Prepared {
    pub snapshot: SkySnapshot,
    pub weights: LayerWeights,

    pub daylight: f32,

    pub atlas: Option<Vec<(Vec<u8>, u32, u32)>>,
    /// The stops `atlas` was stored brighter by (`atlas_stops`); the brushes
    /// read it back down by as many (`PostUpload::atlas_stops`).
    pub atlas_stops: f32,
}

struct Request {
    hour: f32,
    days: f64,
}

pub struct TimeOfDayRuntime {
    pub params: TimeOfDayParams,

    seconds: f64,
    requests: mpsc::Sender<Request>,
    results: mpsc::Receiver<Prepared>,
    in_flight: bool,
    requested_hour: Option<f32>,

    forced: Option<(Option<f32>, Option<f32>)>,

    pub zenith_depth: [f32; 3],
    pub has_layers: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ClockOverride {
    pub hour: Option<f32>,
    pub day_minutes: Option<f32>,
}

impl TimeOfDayRuntime {

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

    pub fn hour(&self) -> f32 {
        self.params.hour_at(self.seconds)
    }

    pub fn seconds(&self) -> f64 {
        self.seconds
    }

    pub fn advance(&mut self, dt: f32, clock: ClockOverride) -> Option<Prepared> {
        let forced = (clock.hour, clock.day_minutes);
        if self.forced != Some(forced) {
            if self.forced.is_some() || clock.hour.is_some() || clock.day_minutes.is_some() {
                if let Some(h) = clock.hour {
                    self.params.start_hour = h;
                    self.seconds = 0.0;
                }
                if let Some(m) = clock.day_minutes {

                    let hour = self.hour();
                    self.params.day_minutes = m.max(0.0);
                    self.params.start_hour = hour;
                    self.seconds = 0.0;
                }
            }
            self.forced = Some(forced);
        }
        self.seconds += f64::from(dt.max(0.0));
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

    sent: Option<LayerWeights>,
}

impl Worker {
    fn prepare(&mut self, hour: f32, days: f64, force: bool) -> Prepared {
        let snapshot = self.sky.snapshot(hour, days);
        let weights = LayerWeights::between(&self.baked_sky, self.baked_sun.as_ref(), &snapshot.irradiance, snapshot.light.as_ref());
        let weights = match &self.layers {
            Some(layers) => weights.with_keys(layers, snapshot.light.as_ref()),
            None => weights,
        };
        let daylight = daylight_share(&self.baked_sky, self.baked_sun.as_ref(), &snapshot);
        let moved = |a: &LayerWeights, b: &LayerWeights| {
            let far = |x: f32, y: f32| {
                let (x, y) = (x.max(1e-9), y.max(1e-9));
                (x / y).max(y / x) > 1.0 + RELIGHT_STEP && (x - y).abs() > 1e-7
            };
            // Between keyframes, a step along the arc moves the bounce too.
            let keyed = match (a.key, b.key) {
                (Some(x), Some(y)) => {
                    (x.a, x.b) != (y.a, y.b) && (x.a, x.b) != (y.b, y.a)
                        || (x.t - y.t).abs() > RELIGHT_STEP
                        || (0..3).any(|c| far(x.light[c], y.light[c]))
                }
                (x, y) => x.is_some() != y.is_some(),
            };
            keyed || (0..3).any(|c| far(a.sky[c], b.sky[c]) || far(a.sun[c], b.sun[c]))
        };
        let (atlas, atlas_stops) = match &self.layers {
            Some(layers) if force || self.sent.as_ref().is_none_or(|s| moved(s, &weights)) => {
                self.sent = Some(weights);
                let mut relit = layers.relit(&weights);
                let stops = atlas_stops(&relit);
                scale_atlas(&mut relit, stops);
                (Some(crate::renderer::mesh::lightmap_mips_f16(&relit, layers.width, layers.height)), stops)
            }
            _ => (None, 0.0),
        };
        Prepared { snapshot, weights, daylight, atlas, atlas_stops }
    }
}

pub fn daylight_share(baked_sky: &SkyIrradiance, baked_sun: Option<&SkySun>, now: &SkySnapshot) -> f32 {
    let lum = |c: [f32; 3]| 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
    let ground = |sky: &SkyIrradiance, sun: Option<&SkySun>| {
        lum(sky.evaluate([0.0, 1.0, 0.0])) + sun.map_or(0.0, |s| lum(s.light_rgb) * s.direction[1].max(0.0))
    };
    let then = ground(baked_sky, baked_sun);
    if then > 0.0 { ground(&now.irradiance, now.light.as_ref()) / then } else { 0.0 }
}

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

        let (f, l) = (full.clone(), lamps.clone());
        let daylight = ProbeDaylight::default();
        let source = relit_probe_source(Arc::new(move |_| Some(f.clone())), Arc::new(move |_| Some(l.clone())), daylight.clone());
        assert_eq!(source(0).unwrap(), full);
        daylight.set(0.0);
        assert_eq!(read(&source(0).unwrap()), read(&lamps));
    }

    /// A building's outside keeps only its daylight: dark at night however
    /// its lamps light the inside, and with no lamps photograph the whole.
    #[test]
    fn a_buildings_outside_is_dark_at_night_whatever_its_lamps() {
        use crate::renderer::uniforms::{f16_to_f32, f32_to_f16};
        let texel = |v: [f32; 4]| v.iter().flat_map(|x| f32_to_f16(*x).to_le_bytes()).collect::<Vec<u8>>();
        let read = |b: &[u8]| -> Vec<f32> { b.chunks_exact(2).map(|p| f16_to_f32(u16::from_le_bytes([p[0], p[1]]))).collect() };
        let full = texel([1.0, 0.5, 0.25, 0.75]);
        let lamps = texel([0.25, 0.25, 0.25, 0.75]);
        assert_eq!(read(&building_daylight_faces(&full, &lamps, 0.0)), vec![0.0, 0.0, 0.0, 0.75]);
        assert_eq!(read(&building_daylight_faces(&full, &lamps, 0.5)), vec![0.375, 0.125, 0.0, 0.75]);
        assert_eq!(read(&building_daylight_faces(&full, &[], 0.5)), vec![0.5, 0.25, 0.125, 0.75]);
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

        for _ in 0..200 {
            assert!(rt.advance(0.1, ClockOverride::default()).is_none());
        }
        assert!(!rt.in_flight);

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
            sun_keys: Vec::new(),
        });
        let params = TimeOfDayParams::default();
        let ts = Arc::new(TimeOfDaySky::new(params));
        let mut w = Worker { sky: ts, baked_sky: sky, baked_sun: Some(sun), layers: Some(layers), sent: None };
        assert!(w.prepare(15.0, 0.0, true).atlas.is_some());
        assert!(w.prepare(15.0, 0.0, false).atlas.is_none(), "same light, no upload");
        assert!(w.prepare(3.0, 0.0, false).atlas.is_some(), "night relights");
    }

    /// With sun keyframes the bounce follows the sun along the day: at a key
    /// the atlas holds that key's bounce, and walking on to the next relights
    /// toward it.
    #[test]
    fn the_sun_bounce_walks_its_keyframes() {
        use space_soup_sky::time_of_day::{sun_key_hours, SunKey};
        let (sky, sun) = hdri_like();
        let params = TimeOfDayParams::default();
        let hours = sun_key_hours(&params, 4);
        let flat = |v: f32| vec![v, v, v, 1.0, v, v, v, 1.0, v, v, v, 1.0, v, v, v, 1.0];
        let layers = Arc::new(AtlasLayers {
            width: 2,
            height: 2,
            lamps: vec![0.0; 16],
            sky: vec![0.0; 16],
            sun: flat(0.5),
            sun_keys: hours.iter().enumerate().map(|(i, (h, d))| SunKey::from_rgba(*h, *d, &flat(1.0 + i as f32))).collect(),
        });
        let ts = Arc::new(TimeOfDaySky::new(params));
        let mut w = Worker { sky: ts.clone(), baked_sky: sky, baked_sun: Some(sun), layers: Some(layers), sent: None };
        let at = |w: &mut Worker, h: f32| {
            let p = w.prepare(h, 0.0, true);
            let light = p.snapshot.light.unwrap().light_rgb;
            (p.weights.key.unwrap(), light)
        };
        let (k1, _) = at(&mut w, hours[1].0);
        let on = if k1.t < 0.5 { k1.a } else { k1.b };
        assert!(on == 1 && (k1.t < 0.02 || k1.t > 0.98), "{k1:?}");
        let (k, _) = at(&mut w, f32::midpoint(hours[1].0, hours[2].0));
        assert!([k.a, k.b].contains(&1) && [k.a, k.b].contains(&2) && (k.t - 0.5).abs() < 0.15, "{k:?}");
        let prepared = w.prepare(f32::midpoint(hours[1].0, hours[2].0), 0.0, true);
        let lit = crate::renderer::mesh::lightmap_mips_f16(
            &w.layers.as_ref().unwrap().relit(&prepared.weights),
            2,
            2,
        );
        assert!(prepared.atlas.is_some() && !lit.is_empty());
        assert!(w.prepare(f32::midpoint(hours[1].0, hours[2].0) + 0.5, 0.0, false).atlas.is_some(), "half an hour on relights");
    }
}
