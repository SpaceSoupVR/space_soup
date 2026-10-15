use glam::Vec3;

use super::probe_stream::ProbeDesc;
use super::sky::SkyIrradiance;
use super::uniforms::ProbePortal;

pub const METER_BINS: u32 = 4;

pub const REFERENCE_LUMINANCE: f32 = 0.5;

pub const ADAPTATION_STRENGTH: f32 = 0.8;

pub const MIN_EXPOSURE: f32 = 0.25;

pub const MAX_EXPOSURE: f32 = 12.0;

pub const ADAPT_BRIGHTER_STOPS_PER_S: f32 = 3.0;
pub const ADAPT_DARKER_STOPS_PER_S: f32 = 1.5;

const CENTRE_WEIGHT_FLOOR: f32 = 0.1;
const CENTRE_WEIGHT_POWER: f32 = 4.0;

const LOG_FLOOR: f32 = 1e-4;

const DOORWAY_HANDOVER: f32 = 0.5;

const WINDOW_NEAREST: f32 = 0.05;

const NEAREST_PROBE_FLOOR: f32 = 0.25;

pub const METER_LOW_PERCENTILE: f32 = 0.7;
pub const METER_HIGH_PERCENTILE: f32 = 0.98;

struct ProbeMeter {
    centre: Vec3,
    min: Vec3,
    max: Vec3,

    room: u32,

    bins: Vec<(Vec3, f32, f32)>,

    parts: Vec<BinParts>,
}

#[derive(Clone, Copy, Debug)]
struct BinParts {
    surface: f32,
    lamps: Option<f32>,
    sky: f32,
    sky_ref: f32,
}

pub struct EyeAdaptation {
    meters: Vec<ProbeMeter>,

    portals: Vec<ProbePortal>,

    sky: SkyIrradiance,

    adapted_log2: Option<f32>,

    max_exposure: f32,
}

impl EyeAdaptation {

    pub fn sky_only(sky: SkyIrradiance) -> Self {
        Self { meters: Vec::new(), portals: Vec::new(), sky, adapted_log2: None, max_exposure: MAX_EXPOSURE }
    }

    pub fn from_probes(probes: &[(&[u8], u32, ProbeDesc)], sky: SkyIrradiance) -> Self {
        let mut eye = Self::sky_only(sky);
        for (faces, res, desc) in probes {
            eye.add_probe(faces, *res, desc);
        }
        eye
    }

    pub fn add_probe(&mut self, faces: &[u8], res: u32, desc: &ProbeDesc) {
        self.add_probe_layered(faces, None, res, desc);
    }

    pub fn add_probe_layered(&mut self, faces: &[u8], lamps: Option<&[u8]>, res: u32, desc: &ProbeDesc) {
        if let Some(texels) = super::uniforms::decode_probe_texels(faces, res) {
            let lamps = lamps.and_then(|l| super::uniforms::decode_probe_texels(l, res));
            let (bins, parts) = bin_probe_parts(&texels, lamps.as_deref(), res, &self.sky);
            self.meters.push(ProbeMeter { centre: desc.centre, min: desc.min, max: desc.max, room: desc.volume, bins, parts });
        }
    }

    pub fn relight(&mut self, sky: SkyIrradiance, daylight: f32) {
        for m in &mut self.meters {
            for (bin, part) in m.bins.iter_mut().zip(&m.parts) {
                let sky_now = luminance(sky.radiance(bin.0.to_array()));
                let ratio = if part.sky_ref > 0.0 { sky_now / part.sky_ref } else { 0.0 };
                let surface = match part.lamps {
                    Some(l) => l + (part.surface - l).max(0.0) * daylight,
                    None => part.surface,
                };
                bin.1 = surface + part.sky * ratio;
            }
        }
        self.sky = sky;
    }

    pub fn set_max_exposure(&mut self, max: f32) {
        self.max_exposure = max.max(MIN_EXPOSURE);
    }

    fn log_floor(&self) -> f32 {
        if self.max_exposure > MAX_EXPOSURE {
            NIGHT_LOG_FLOOR
        } else {
            LOG_FLOOR
        }
    }

    pub fn adapted_luminance(&self) -> Option<f32> {
        self.adapted_log2.map(f32::exp2)
    }

    pub fn set_portals(&mut self, portals: &[ProbePortal]) {
        self.portals = portals.to_vec();
    }

    pub fn meter(&self, head: Vec3, gaze: Vec3) -> f32 {
        self.meter_with(head, gaze, &[])
    }

    pub fn meter_with(&self, head: Vec3, gaze: Vec3, extra: &[(Vec3, f32, f32)]) -> f32 {
        let gaze = gaze.normalize_or_zero();

        for p in &self.portals {
            let a = p.axis.min(2) as usize;
            let (mut lo, mut hi) = (p.min, p.max);
            lo[a] -= DOORWAY_HANDOVER;
            hi[a] += DOORWAY_HANDOVER;
            if head.cmpge(lo).all() && head.cmple(hi).all() {
                let f = smoothstep(lo[a], hi[a], head[a]);
                match (self.room_log(p.low, head, gaze, extra), self.room_log(p.high, head, gaze, extra)) {
                    (Some(lo), Some(hi)) => return (lo + (hi - lo) * f).exp(),
                    (Some(one), None) | (None, Some(one)) => return one.exp(),
                    (None, None) => {}
                }
            }
        }
        let log = match self.room_at(head) {
            Some(room) => self.room_log(room, head, gaze, extra),
            None => {
                let samples: Vec<(f32, f32)> =
                    sky_bins(&self.sky).into_iter().chain(extra.iter().copied()).map(|b| weighted(b, gaze, 1.0, self.log_floor())).collect();
                band_log(samples)
            }
        };
        log.map_or(REFERENCE_LUMINANCE, f32::exp)
    }

    fn room_at(&self, p: Vec3) -> Option<u32> {
        let mut best: Option<(u32, f32)> = None;
        for m in &self.meters {
            if p.cmplt(m.min).any() || p.cmpgt(m.max).any() {
                continue;
            }
            let d = m.max - m.min;
            let volume = d.x * d.y * d.z;
            if best.is_none_or(|(_, v)| volume < v * 0.999) {
                best = Some((m.room, volume));
            }
        }
        best.map(|b| b.0)
    }

    fn room_log(&self, room: u32, head: Vec3, gaze: Vec3, extra: &[(Vec3, f32, f32)]) -> Option<f32> {
        let mut samples = Vec::new();
        if !self.photographs(room, head, gaze, 1.0, &mut samples) {
            return None;
        }
        for (beyond, share) in self.windows(room, head, gaze) {
            let mut through = Vec::new();
            if self.photographs(beyond, head, gaze, 1.0, &mut through) {

                let total: f32 = through.iter().map(|s| s.1).sum();
                if total > 0.0 {
                    samples.extend(through.into_iter().map(|(l, w)| (l, w * share / total)));
                }
            }
        }
        samples.extend(extra.iter().map(|&b| weighted(b, gaze, 1.0, self.log_floor())));
        band_log(samples)
    }

    fn photographs(&self, room: u32, head: Vec3, gaze: Vec3, scale: f32, out: &mut Vec<(f32, f32)>) -> bool {
        let near = |m: &ProbeMeter| 1.0 / m.centre.distance_squared(head).max(NEAREST_PROBE_FLOOR);
        let total: f32 = self.meters.iter().filter(|m| m.room == room).map(near).sum();
        if total <= 0.0 {
            return false;
        }
        for m in self.meters.iter().filter(|m| m.room == room) {
            let k = scale * near(m) / total;
            out.extend(m.bins.iter().map(|&b| weighted(b, gaze, k, self.log_floor())));
        }
        true
    }

    fn windows(&self, room: u32, head: Vec3, gaze: Vec3) -> Vec<(u32, f32)> {
        let mut out = Vec::new();
        for p in &self.portals {
            let a = p.axis.min(2) as usize;

            let (plane, beyond, ahead) = if p.low == room {
                (p.max[a], p.high, p.max[a] - head[a])
            } else if p.high == room {
                (p.min[a], p.low, head[a] - p.min[a])
            } else {
                continue;
            };

            let ahead = ahead.max(WINDOW_NEAREST);
            let mut at = head;
            at[a] = if p.low == room { plane - ahead } else { plane + ahead };
            let (u, v) = ((a + 1) % 3, (a + 2) % 3);
            let omega = rectangle_solid_angle(
                ahead,
                (p.min[u] - at[u], p.max[u] - at[u]),
                (p.min[v] - at[v], p.max[v] - at[v]),
            );
            let mut middle = (p.min + p.max) * 0.5;
            middle[a] = plane;
            let toward = (middle - at).normalize_or_zero();
            let share = omega * (CENTRE_WEIGHT_FLOOR + toward.dot(gaze).max(0.0).powf(CENTRE_WEIGHT_POWER));
            if share > 0.0 {
                out.push((beyond, share));
            }
        }
        out
    }

    pub fn update(&mut self, metered: f32, dt: f32) -> f32 {
        let target = metered.max(self.log_floor()).log2();
        let current = match self.adapted_log2 {
            None => target,
            Some(c) => {
                let (rate, delta) = if target > c {
                    (ADAPT_BRIGHTER_STOPS_PER_S, target - c)
                } else {
                    (ADAPT_DARKER_STOPS_PER_S, c - target)
                };
                let step = (rate * dt.max(0.0)).min(delta);
                if target > c { c + step } else { c - step }
            }
        };
        self.adapted_log2 = Some(current);
        exposure_for_up_to(current.exp2(), self.max_exposure)
    }
}

pub fn exposure_for(luminance: f32) -> f32 {
    exposure_for_up_to(luminance, MAX_EXPOSURE)
}

pub fn exposure_for_up_to(luminance: f32, max: f32) -> f32 {
    let floor = if max > MAX_EXPOSURE { NIGHT_LOG_FLOOR } else { LOG_FLOOR };
    let ratio = REFERENCE_LUMINANCE / luminance.max(floor);
    ratio.powf(ADAPTATION_STRENGTH).clamp(MIN_EXPOSURE, max)
}

const NIGHT_LOG_FLOOR: f32 = 1e-10;

fn weighted((d, lum, omega): (Vec3, f32, f32), gaze: Vec3, k: f32, floor: f32) -> (f32, f32) {
    let w = k * omega * (CENTRE_WEIGHT_FLOOR + d.dot(gaze).max(0.0).powf(CENTRE_WEIGHT_POWER));
    (lum.max(floor).ln(), w)
}

fn rectangle_solid_angle(ahead: f32, x: (f32, f32), y: (f32, f32)) -> f32 {
    let f = |x: f32, y: f32| (x * y / (ahead * (ahead * ahead + x * x + y * y).sqrt())).atan();
    (f(x.1, y.1) - f(x.0, y.1) - f(x.1, y.0) + f(x.0, y.0)).max(0.0)
}

fn band_log(mut samples: Vec<(f32, f32)>) -> Option<f32> {

    samples.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let total: f32 = samples.iter().map(|s| s.1).sum();
    if total <= 0.0 {
        return None;
    }

    let (lo, hi) = (METER_LOW_PERCENTILE * total, METER_HIGH_PERCENTILE * total);
    let (mut acc, mut sum, mut weight) = (0.0f32, 0.0f32, 0.0f32);
    for (l, w) in samples {
        let inside = (acc + w).min(hi) - acc.max(lo);
        if inside > 0.0 {
            sum += l * inside;
            weight += inside;
        }
        acc += w;
    }
    (weight > 0.0).then(|| sum / weight)
}

fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0).max(1e-6)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn luminance(c: [f32; 3]) -> f32 {
    0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
}

fn bin_probe_parts(texels: &[[f32; 4]], lamps: Option<&[[f32; 4]]>, res: u32, sky: &SkyIrradiance) -> (Vec<(Vec3, f32, f32)>, Vec<BinParts>) {
    let bins = bin_probe(texels, res, sky);
    let per = (res / METER_BINS).max(1);
    let mut parts = Vec::with_capacity(bins.len());
    for face in 0..6usize {
        for by in 0..METER_BINS {
            for bx in 0..METER_BINS {
                let (mut surface, mut lamp, mut sky_part, mut omega_sum) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
                for ty in by * per..((by + 1) * per).min(res) {
                    for tx in bx * per..((bx + 1) * per).min(res) {
                        let u = (tx as f32 + 0.5) / res as f32;
                        let v = (ty as f32 + 0.5) / res as f32;
                        let d = super::probe_prefilter::texel_direction(face, u, v);
                        let (a, b) = (2.0 * u - 1.0, 2.0 * v - 1.0);
                        let omega = (1.0 + a * a + b * b).powf(-1.5);
                        let i = (face as u32 * res * res + ty * res + tx) as usize;
                        let t = texels[i];
                        let cover = t[3].clamp(0.0, 1.0);
                        surface += cover * luminance([t[0], t[1], t[2]]) * omega;
                        if let Some(l) = lamps {
                            let l = l[i];
                            lamp += cover * luminance([l[0], l[1], l[2]]) * omega;
                        }
                        sky_part += (1.0 - cover) * luminance(sky.radiance(d.to_array())) * omega;
                        omega_sum += omega;
                    }
                }
                if omega_sum > 0.0 {
                    let u = (bx as f32 + 0.5) / METER_BINS as f32;
                    let v = (by as f32 + 0.5) / METER_BINS as f32;
                    let d = super::probe_prefilter::texel_direction(face, u, v);
                    parts.push(BinParts {
                        surface: surface / omega_sum,
                        lamps: lamps.map(|_| lamp / omega_sum),
                        sky: sky_part / omega_sum,
                        sky_ref: luminance(sky.radiance(d.to_array())),
                    });
                }
            }
        }
    }
    (bins, parts)
}

fn bin_probe(texels: &[[f32; 4]], res: u32, sky: &SkyIrradiance) -> Vec<(Vec3, f32, f32)> {
    let per = (res / METER_BINS).max(1);
    let mut out = Vec::with_capacity((6 * METER_BINS * METER_BINS) as usize);
    for face in 0..6usize {
        for by in 0..METER_BINS {
            for bx in 0..METER_BINS {
                let (mut sum, mut omega_sum) = (0.0f32, 0.0f32);
                for ty in by * per..((by + 1) * per).min(res) {
                    for tx in bx * per..((bx + 1) * per).min(res) {
                        let u = (tx as f32 + 0.5) / res as f32;
                        let v = (ty as f32 + 0.5) / res as f32;
                        let d = super::probe_prefilter::texel_direction(face, u, v);

                        let (a, b) = (2.0 * u - 1.0, 2.0 * v - 1.0);
                        let omega = (1.0 + a * a + b * b).powf(-1.5);
                        let t = texels[(face as u32 * res * res + ty * res + tx) as usize];
                        let cover = t[3].clamp(0.0, 1.0);
                        let l = cover * luminance([t[0], t[1], t[2]])
                            + (1.0 - cover) * luminance(sky.radiance(d.to_array()));
                        sum += l * omega;
                        omega_sum += omega;
                    }
                }
                let u = (bx as f32 + 0.5) / METER_BINS as f32;
                let v = (by as f32 + 0.5) / METER_BINS as f32;
                let d = super::probe_prefilter::texel_direction(face, u, v);
                if omega_sum > 0.0 {

                    let texel = 2.0 / res as f32;
                    out.push((d, sum / omega_sum, omega_sum * texel * texel));
                }
            }
        }
    }
    out
}

fn sky_bins(sky: &SkyIrradiance) -> Vec<(Vec3, f32, f32)> {
    let mut out = Vec::new();
    for face in 0..6usize {
        for by in 0..METER_BINS {
            for bx in 0..METER_BINS {
                let u = (bx as f32 + 0.5) / METER_BINS as f32;
                let v = (by as f32 + 0.5) / METER_BINS as f32;
                let d = super::probe_prefilter::texel_direction(face, u, v);
                let (a, b) = (2.0 * u - 1.0, 2.0 * v - 1.0);
                let bin = 2.0 / METER_BINS as f32;
                out.push((d, luminance(sky.radiance(d.to_array())), bin * bin * (1.0 + a * a + b * b).powf(-1.5)));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_darker_room_is_exposed_up_and_a_brighter_one_down_within_the_limits() {
        let bright = exposure_for(REFERENCE_LUMINANCE * 14.0);
        let reference = exposure_for(REFERENCE_LUMINANCE);
        let hall = exposure_for(REFERENCE_LUMINANCE / 14.0);
        assert!((reference - 1.0).abs() < 1e-5);
        assert!(hall > 4.0 && hall <= MAX_EXPOSURE, "{hall}");
        assert!(bright < 0.25 + 1e-5 || bright < 1.0, "{bright}");

        assert!(hall < 14.0, "full adaptation would be 14x; got {hall}");
        assert_eq!(exposure_for(1e-9), MAX_EXPOSURE);
    }

    #[test]
    fn the_eye_adapts_at_its_rates_and_settles() {
        let mut eye = EyeAdaptation::sky_only(SkyIrradiance::flat(0.3));
        let start = eye.update(REFERENCE_LUMINANCE, 0.016);
        assert!((start - 1.0).abs() < 1e-5, "first frame adapts instantly");

        let dark = REFERENCE_LUMINANCE / 8.0;
        let after_1s = eye.update(dark, 1.0);
        let expect = 2f32.powf(1.5 * ADAPTATION_STRENGTH);
        assert!((after_1s / expect - 1.0).abs() < 1e-3, "{after_1s} vs {expect}");
        for _ in 0..20 {
            eye.update(dark, 1.0);
        }
        assert!((eye.update(dark, 0.0) - exposure_for(dark)).abs() < 1e-4);

        let back = eye.update(REFERENCE_LUMINANCE, 1.0);
        assert!((back - 1.0).abs() < 1e-4, "{back}");
    }

    fn faces(res: u32, lum: impl Fn(Vec3) -> f32) -> Vec<u8> {
        let mut out = Vec::new();
        for face in 0..6usize {
            for y in 0..res {
                for x in 0..res {
                    let d = crate::renderer::probe_prefilter::texel_direction(
                        face,
                        (x as f32 + 0.5) / res as f32,
                        (y as f32 + 0.5) / res as f32,
                    );
                    let l = lum(d);
                    for v in [l, l, l, 1.0] {
                        out.extend_from_slice(&crate::renderer::sky::f32_to_f16(v).to_le_bytes());
                    }
                }
            }
        }
        out
    }

    fn desc(centre: Vec3, min: Vec3, max: Vec3, room: u32, has_depth: bool) -> ProbeDesc {
        ProbeDesc { centre, min, max, volume: room, has_depth, room_light: None }
    }

    fn two_rooms() -> EyeAdaptation {
        let (dark, bright, outdoors) = (faces(16, |_| 0.01), faces(16, |_| 0.3), faces(16, |_| 1.0));
        let a = desc(Vec3::new(-2.5, 1.5, 0.0), Vec3::new(-5.0, 0.0, -2.0), Vec3::new(-0.1, 3.0, 2.0), 0, true);
        let b = desc(Vec3::new(2.5, 1.5, 0.0), Vec3::new(0.1, 0.0, -2.0), Vec3::new(5.0, 3.0, 2.0), 1, true);
        let out = desc(Vec3::new(0.0, 5.0, 0.0), Vec3::splat(-50.0), Vec3::splat(50.0), 2, false);
        let mut eye = EyeAdaptation::from_probes(
            &[(&dark, 16, a), (&bright, 16, b), (&outdoors, 16, out)],
            SkyIrradiance::flat(1.0),
        );
        eye.set_portals(&[ProbePortal {
            min: Vec3::new(-0.3, 0.0, -0.6),
            max: Vec3::new(0.3, 2.2, 0.6),
            axis: 0,
            low: 0,
            high: 1,
            wall: None,
        }]);
        eye
    }

    #[test]
    fn a_doorway_hands_the_meter_over_without_a_step_or_the_outdoors() {
        let eye = two_rooms();
        let walk: Vec<f32> = (0..=400).map(|i| eye.meter(Vec3::new(-2.0 + i as f32 * 0.01, 1.6, 0.0), Vec3::X)).collect();

        assert!((eye.meter(Vec3::new(-2.0, 1.6, 0.0), Vec3::NEG_X) / 0.01 - 1.0).abs() < 0.05);
        assert!((walk[400] / 0.3 - 1.0).abs() < 0.01, "in the bright room: {}", walk[400]);
        for (i, w) in walk.windows(2).enumerate() {
            assert!(w[1] <= 0.3 * 1.001, "{} cm along: read {} -- the outdoors", i + 1, w[1]);
            assert!((w[1] / w[0]).ln().abs() < 0.15, "{} cm along: {} then {}", i + 1, w[0], w[1]);
        }
    }

    #[test]
    fn a_bright_doorway_meters_the_more_the_nearer_it_is() {
        let eye = two_rooms();
        let far = eye.meter(Vec3::new(-4.5, 1.6, 0.0), Vec3::X);
        let near = eye.meter(Vec3::new(-1.0, 1.6, 0.0), Vec3::X);
        let away = eye.meter(Vec3::new(-1.0, 1.6, 0.0), Vec3::NEG_X);

        assert!(near > far * 1.5 && far > away, "near {near}, far {far}, looking away {away}");
        assert!(near > 0.01 * 4.0 && near < 0.3, "{near}");

        assert!(away < 0.01 * 1.5, "{away}");
    }

    #[test]
    fn a_rectangle_fills_the_solid_angle_it_should() {
        let face = rectangle_solid_angle(1.0, (-1.0, 1.0), (-1.0, 1.0));
        assert!((face - 4.0 * std::f32::consts::PI / 6.0).abs() < 1e-4, "{face}");
        let tiny = rectangle_solid_angle(10.0, (-0.05, 0.05), (-0.05, 0.05));
        assert!((tiny / (0.01 / 100.0) - 1.0).abs() < 1e-3, "{tiny}");
        assert_eq!(rectangle_solid_angle(1.0, (2.0, 3.0), (2.0, 3.0)) > 0.0, true);
    }

    #[test]
    fn light_the_photographs_never_saw_is_metered_too() {

        let eye = two_rooms();
        let (head, gaze) = (Vec3::new(-2.5, 1.6, 0.0), Vec3::new(0.0, -0.6, -0.8));
        let alone = eye.meter(head, gaze);
        assert_eq!(eye.meter_with(head, gaze, &[]), alone);
        let lit = eye.meter_with(head, gaze, &[(gaze, 0.5, 1.5)]);
        assert!(lit > 5.0 * alone, "{lit} vs {alone}");
    }

    #[test]
    fn a_night_sky_takes_the_daylight_out_of_the_meter_and_keeps_the_lamps() {
        let res = 8;
        let day = faces(res, |_| 0.4);
        let lamps = faces(res, |_| 0.004);
        let mut eye = EyeAdaptation::sky_only(SkyIrradiance::flat(0.5));
        eye.add_probe_layered(&day, Some(&lamps), res, &desc(Vec3::ZERO, Vec3::splat(-5.0), Vec3::splat(5.0), 0, true));
        let noon = eye.meter(Vec3::ZERO, Vec3::NEG_Z);
        eye.relight(SkyIrradiance::flat(0.5), 1.0);
        assert!((eye.meter(Vec3::ZERO, Vec3::NEG_Z) - noon).abs() < 1e-3 * noon, "as baked");
        eye.relight(SkyIrradiance::flat(1e-7), 0.0);
        let night = eye.meter(Vec3::ZERO, Vec3::NEG_Z);
        assert!((night - 0.004).abs() < 1e-4, "only the lamps: {night}");

        assert_eq!(exposure_for(1e-6), MAX_EXPOSURE);
        assert!(exposure_for_up_to(1e-6, 1e5) > 1e4);
    }

    #[test]
    fn a_moonlit_meter_is_not_read_as_the_day_floor() {
        let res = 8;
        let field = faces(res, |_| 0.2);
        let unlit = faces(res, |_| 0.0);
        let mut eye = EyeAdaptation::sky_only(SkyIrradiance::flat(0.1));
        eye.add_probe_layered(&field, Some(&unlit), res, &desc(Vec3::ZERO, Vec3::splat(-5.0), Vec3::splat(5.0), 0, true));
        eye.relight(SkyIrradiance::flat(1e-7), 1e-6);
        assert!((eye.meter(Vec3::ZERO, Vec3::NEG_Z) - 1e-4).abs() < 1e-6, "the day reads its floor");
        eye.set_max_exposure(2e5);
        let night = eye.meter(Vec3::ZERO, Vec3::NEG_Z);
        assert!((night / 2e-7 - 1.0).abs() < 0.05, "the night reads the field: {night}");
        let e = eye.update(night, 0.0);
        assert!(e > 1e4, "and opens the eye: x{e}");
        assert!((eye.adapted_luminance().unwrap() / night - 1.0).abs() < 0.01);
    }

    #[test]
    fn looking_at_the_bright_side_meters_brighter() {

        let mut sky = SkyIrradiance::flat(0.2);
        sky.sh[3] = [0.3, 0.3, 0.3];
        let eye = EyeAdaptation::sky_only(sky);
        let toward = eye.meter(Vec3::ZERO, Vec3::X);
        let away = eye.meter(Vec3::ZERO, Vec3::NEG_X);
        assert!(toward > away * 1.2, "centre-weighting ignored the gaze: {toward} vs {away}");
    }
}
