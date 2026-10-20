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
/// How fast the eye keeps adapting to the dark once past the indoor range
/// (`MAX_EXPOSURE`): rods take minutes where cones take seconds, so stepping
/// out of a lit hall into the night leaves the eye night-blind for a while.
pub const ADAPT_NIGHT_STOPS_PER_S: f32 = 0.3;

/// LOCAL ADAPTATION to a lit room seen through a doorway: the eye adapts in
/// part to what it looks at, so it never sits more than this many times darker
/// than a lit room filling enough of the view. Without it a night-adapted eye
/// (exposure ~1e4) drew the lit hall's doorway as one sheet of white.
pub const BRIGHT_VIEW_RATIO: f32 = 12.0;
/// The doorway's share of the view (steradians, by the meter's lobe) from
/// which the cap starts: a 1.6 x 2.2 m door at ~34 m...
/// The same with closed rooms drawn at their own adaptation (`local_rooms`):
/// the room reads by itself, so the pull is only the eye's simultaneous range
/// (about three decades). At 12 it blacked the moonlit ground out round the
/// door from 12 m (exposure x70,000 -> x520).
pub const BRIGHT_VIEW_RATIO_LOCAL: f32 = 1000.0;
pub const BRIGHT_VIEW_FROM_SR: f32 = 0.003;
/// ...to where it holds in full (the same door at ~6.6 m).
pub const BRIGHT_VIEW_TO_SR: f32 = 0.08;

/// LOCAL ADAPTATION BY ROOM: a closed room seen from outside it is drawn at
/// up to this many times the exposure an eye inside it has -- a little
/// brighter, as a dark-adapted eye sees a lit room -- and never brighter than
/// the eye's own. With one exposure for the whole view a lit hall seen from
/// the night was either a white sheet or blacked the night out (headset,
/// 2026-10-20).
pub const ROOM_SEEN_BRIGHTER: f32 = 2.5;
/// How far outside a room's or doorway's box its adaptation fades out,
/// metres: per pixel, so the edge is the opening's outline.
pub const ROOM_EXPOSURE_BAND: f32 = 0.05;

const CENTRE_WEIGHT_FLOOR: f32 = 0.1;
const CENTRE_WEIGHT_POWER: f32 = 4.0;

const OPEN_SKY_FROM: f32 = 0.08;
const OPEN_SKY_TO: f32 = 0.3;

const DAY_SKY_FROM: f32 = REFERENCE_LUMINANCE / 2000.0;
const DAY_SKY_TO: f32 = REFERENCE_LUMINANCE / 200.0;

const FREE_LOBE_SCALE: f32 = 3.0 / 16.0;

#[derive(Clone, Copy, Debug)]
struct Lobe {
    gaze: Vec3,
    free: f32,
}

impl Lobe {
    fn weight(&self, d: Vec3) -> f32 {
        let centre = d.dot(self.gaze).max(0.0).powf(CENTRE_WEIGHT_POWER);
        let level = 1.0 - d.y * d.y;
        let free = FREE_LOBE_SCALE * level * level;
        CENTRE_WEIGHT_FLOOR + centre + (free - centre) * self.free
    }
}

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

    open: f32,
}

#[derive(Clone, Copy, Debug)]
struct BinParts {
    surface: f32,
    lamps: Option<f32>,
    sky: f32,
    sky_ref: f32,
}

/// A closed room's box and the luminance its own eye adapts to (`band_log`
/// over its photographs, every direction), for drawing it at that adaptation
/// when seen from outside (`EyeAdaptation::room_exposure`).
#[derive(Clone, Copy, Debug)]
struct ClosedRoom {
    room: u32,
    min: Vec3,
    max: Vec3,
    luminance: f32,
}

pub struct EyeAdaptation {
    meters: Vec<ProbeMeter>,

    closed_rooms: Vec<ClosedRoom>,

    /// Closed rooms are drawn at their own adaptation (`room_exposure`), so
    /// the light seen into one through a doorway is not metered.
    local_rooms: bool,

    portals: Vec<ProbePortal>,

    sky: SkyIrradiance,

    adapted_log2: Option<f32>,

    max_exposure: f32,

    day: f32,
}

impl EyeAdaptation {

    pub fn sky_only(sky: SkyIrradiance) -> Self {
        let day = sky_daylight(&sky);
        Self { meters: Vec::new(), closed_rooms: Vec::new(), local_rooms: false, portals: Vec::new(), sky, adapted_log2: None, max_exposure: MAX_EXPOSURE, day }
    }

    pub fn gaze_free_share(&self, head: Vec3) -> f32 {
        match self.room_at(head) {
            Some(room) => self.free_in(room, head),
            None => self.day,
        }
    }

    fn free_in(&self, room: u32, head: Vec3) -> f32 {
        let near = |m: &ProbeMeter| 1.0 / m.centre.distance_squared(head).max(NEAREST_PROBE_FLOOR);
        let (mut open, mut total) = (0.0f32, 0.0f32);
        for m in self.meters.iter().filter(|m| m.room == room) {
            open += near(m) * m.open;
            total += near(m);
        }
        let open = if total > 0.0 { open / total } else { 1.0 };
        smoothstep(OPEN_SKY_FROM, OPEN_SKY_TO, open) * self.day
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
            let open = open_sky(&texels, res);
            self.meters.push(ProbeMeter { centre: desc.centre, min: desc.min, max: desc.max, room: desc.volume, bins, parts, open });
            self.find_closed_rooms();
        }
    }

    /// The rooms no sky shows into, each with the luminance an eye inside it
    /// adapts to. Rebuilt when the photographs or their light change.
    fn find_closed_rooms(&mut self) {
        let mut rooms: Vec<u32> = self.meters.iter().map(|m| m.room).collect();
        rooms.sort_unstable();
        rooms.dedup();
        self.closed_rooms = rooms
            .into_iter()
            .filter_map(|room| {
                let own: Vec<&ProbeMeter> = self.meters.iter().filter(|m| m.room == room).collect();
                if own.iter().any(|m| m.open >= OPEN_SKY_FROM) {
                    return None;
                }
                let (min, max) = own.iter().fold((Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)), |(lo, hi), m| (lo.min(m.min), hi.max(m.max)));
                let centre = own.iter().map(|m| m.centre).sum::<Vec3>() / own.len() as f32;
                let lobe = Lobe { gaze: Vec3::Z, free: 1.0 };
                let mut samples = Vec::new();
                self.photographs(room, centre, &lobe, 1.0, &mut samples);
                Some(ClosedRoom { room, min, max, luminance: band_log(samples)?.exp() })
            })
            .collect();
    }

    /// The rows of `uniforms::RoomExposure` for an eye at `head` exposed at
    /// `exposure`: each closed room near it drawn at no more than
    /// `ROOM_SEEN_BRIGHTER` times the exposure an eye inside it would have,
    /// and never brighter than the eye's own. Continuous: walking in, the
    /// eye's exposure meets the room's and the factor reaches 1 by itself.
    pub fn room_exposure(&self, head: Vec3, exposure: f32) -> super::uniforms::RoomExposure {
        use super::uniforms::ROOM_EXPOSURE_ROOMS;
        let mut rows = [[0.0f32; 4]; 2 * ROOM_EXPOSURE_ROOMS];
        if !(exposure > 0.0) {
            return rows;
        }
        let mut rooms = self.closed_rooms.clone();
        let gap = |lo: Vec3, hi: Vec3| (head.clamp(lo, hi) - head).length_squared();
        rooms.sort_by(|a, b| gap(a.min, a.max).total_cmp(&gap(b.min, b.max)));
        rooms.truncate(ROOM_EXPOSURE_ROOMS / 2);
        let factor = |r: &ClosedRoom| (ROOM_SEEN_BRIGHTER * exposure_for_up_to(r.luminance, self.max_exposure) / exposure).clamp(1e-6, 1.0);
        // Each room, then its doorways: a doorway's jamb is lit by the room
        // and belongs to its adaptation, or it drew white round the opening.
        let mut boxes: Vec<(Vec3, Vec3, f32)> = rooms.iter().map(|r| (r.min, r.max, factor(r))).collect();
        for p in &self.portals {
            let inside = rooms.iter().find(|r| r.room == p.low || r.room == p.high);
            let both = rooms.iter().filter(|r| r.room == p.low || r.room == p.high).count() == 2;
            if let (Some(r), false) = (inside, both) {
                boxes.push((p.min, p.max, factor(r)));
            }
        }
        boxes.sort_by(|a, b| gap(a.0, a.1).total_cmp(&gap(b.0, b.1)));
        for (i, (lo, hi, f)) in boxes.into_iter().take(ROOM_EXPOSURE_ROOMS).enumerate() {
            rows[2 * i] = [lo.x, lo.y, lo.z, f];
            rows[2 * i + 1] = [hi.x, hi.y, hi.z, ROOM_EXPOSURE_BAND];
        }
        rows
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
        self.day = sky_daylight(&sky);
        self.sky = sky;
        self.find_closed_rooms();
    }

    /// Whether closed rooms are drawn at their own adaptation; see `local_rooms`.
    pub fn set_local_rooms(&mut self, on: bool) {
        self.local_rooms = on;
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

                let lobe = Lobe { gaze, free: self.day };
                let samples: Vec<(f32, f32)> =
                    sky_bins(&self.sky).into_iter().chain(extra.iter().copied()).map(|b| weighted(b, &lobe, 1.0, self.log_floor())).collect();
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
        let lobe = Lobe { gaze, free: self.free_in(room, head) };
        if !self.photographs(room, head, &lobe, 1.0, &mut samples) {
            return None;
        }
        let mut bright = Vec::new();
        for (beyond, share, seen_sr) in self.windows(room, head, &lobe) {
            let mut through = Vec::new();
            if self.photographs(beyond, head, &lobe, 1.0, &mut through) {

                let total: f32 = through.iter().map(|s| s.1).sum();
                if total > 0.0 {
                    bright.push((through.iter().map(|(l, w)| l * w).sum::<f32>() / total, seen_sr));
                    // A room drawn at its own adaptation lights only its own
                    // pixels: metered, its doorway's light blacked out the
                    // moonlit ground round the door from 12 m (exposure
                    // x70,000 -> x520).
                    let local = self.local_rooms && self.closed_rooms.iter().any(|c| c.room == beyond);
                    if !local {
                        samples.extend(through.into_iter().map(|(l, w)| (l, w * share / total)));
                    }
                }
            }
        }
        samples.extend(extra.iter().map(|&b| weighted(b, &lobe, 1.0, self.log_floor())));
        let log = band_log(samples)?;
        let ratio = if self.local_rooms { BRIGHT_VIEW_RATIO_LOCAL } else { BRIGHT_VIEW_RATIO };
        Some(bright.into_iter().fold(log, |log, (seen, share)| log + bright_view_pull(seen, share, log, ratio)))
    }

    fn photographs(&self, room: u32, head: Vec3, lobe: &Lobe, scale: f32, out: &mut Vec<(f32, f32)>) -> bool {
        let near = |m: &ProbeMeter| 1.0 / m.centre.distance_squared(head).max(NEAREST_PROBE_FLOOR);
        let total: f32 = self.meters.iter().filter(|m| m.room == room).map(near).sum();
        if total <= 0.0 {
            return false;
        }
        for m in self.meters.iter().filter(|m| m.room == room) {
            let k = scale * near(m) / total;
            out.extend(m.bins.iter().map(|&b| weighted(b, lobe, k, self.log_floor())));
        }
        true
    }

    /// Each doorway out of `room` seen from `head`: the room beyond, its share
    /// of the meter's lobe, and the solid angle of it inside the view.
    fn windows(&self, room: u32, head: Vec3, lobe: &Lobe) -> Vec<(u32, f32, f32)> {
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
            let share = omega * lobe.weight(toward);
            if share > 0.0 {
                out.push((beyond, share, omega * in_view(toward, lobe.gaze)));
            }
        }
        out
    }

    pub fn update(&mut self, metered: f32, dt: f32) -> f32 {
        let target = metered.max(self.log_floor()).log2();
        let dt = dt.max(0.0);
        let current = match self.adapted_log2 {
            None => target,
            Some(c) if target > c => (c + ADAPT_BRIGHTER_STOPS_PER_S * dt).min(target),
            Some(c) => {
                // Cones down to the night threshold, rods past it; a step that
                // crosses it spends the rest of its time at the rods' rate.
                let night = night_adapted_log2();
                let fast_time = ((c - night).max(0.0) / ADAPT_DARKER_STOPS_PER_S).min(dt);
                let fast = c - ADAPT_DARKER_STOPS_PER_S * fast_time;
                (fast - ADAPT_NIGHT_STOPS_PER_S * (dt - fast_time)).max(target).min(c)
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
    let luminance = luminance.max(floor);
    let indoor = (REFERENCE_LUMINANCE / luminance).powf(ADAPTATION_STRENGTH);
    if max > MAX_EXPOSURE && indoor > MAX_EXPOSURE {
        // Past the indoor range, on from where it ends (`NIGHT_ADAPTATION_STRENGTH`).
        let knee = REFERENCE_LUMINANCE / MAX_EXPOSURE.powf(1.0 / ADAPTATION_STRENGTH);
        return (MAX_EXPOSURE * (knee / luminance).powf(NIGHT_ADAPTATION_STRENGTH)).clamp(MIN_EXPOSURE, max);
    }
    indoor.clamp(MIN_EXPOSURE, max)
}

/// THE DARK-ADAPTED EYE takes in more of the night than the day's partial
/// adaptation: past the indoor range (`MAX_EXPOSURE`) exposure follows the
/// luminance at this power. At 0.8 throughout, moonlit ground (about 0.17
/// lux) drew four stops under the day and the user found the moon's and
/// stars' light too low (2026-10-20); this lifts it about 1.6 stops.
pub const NIGHT_ADAPTATION_STRENGTH: f32 = 0.9;

const NIGHT_LOG_FLOOR: f32 = 1e-10;

/// The adapted luminance (log2) below which the eye is past the indoor
/// range: where `exposure_for` reaches `MAX_EXPOSURE`.
/// How much of a direction is in the headset's view of `gaze`: whole within
/// ~25 deg of it, none past ~60 deg.
fn in_view(toward: Vec3, gaze: Vec3) -> f32 {
    smoothstep(0.5, 0.9, toward.dot(gaze))
}

fn night_adapted_log2() -> f32 {
    (REFERENCE_LUMINANCE / MAX_EXPOSURE.powf(1.0 / ADAPTATION_STRENGTH)).log2()
}

/// How far (natural log) a lit room seen at log luminance `seen`, through a
/// doorway filling `seen_sr` of the view, raises a meter reading `log`: up to
/// within `BRIGHT_VIEW_RATIO` of it, by how much of the view it fills.
/// The factor `room_local_exposure` gives a point, on the CPU: for what is
/// drawn with its exposure set there (the lamps' glare).
pub fn room_factor_at(rows: &super::uniforms::RoomExposure, p: Vec3) -> f32 {
    let mut e = 1.0f32;
    for pair in rows.chunks_exact(2) {
        let (lo, hi) = (pair[0], pair[1]);
        if lo[3] <= 0.0 {
            break;
        }
        let inside = (p.x - lo[0]).min(hi[0] - p.x).min(p.y - lo[1]).min(hi[1] - p.y).min(p.z - lo[2]).min(hi[2] - p.z);
        e = e.min(lo[3].powf(((inside + hi[3]) / hi[3].max(1e-3)).clamp(0.0, 1.0)));
    }
    e
}

fn bright_view_pull(seen: f32, seen_sr: f32, log: f32, ratio: f32) -> f32 {
    let held = smoothstep(BRIGHT_VIEW_FROM_SR.ln(), BRIGHT_VIEW_TO_SR.ln(), seen_sr.max(1e-9).ln());
    held * (seen - ratio.ln() - log).max(0.0)
}

fn weighted((d, lum, omega): (Vec3, f32, f32), lobe: &Lobe, k: f32, floor: f32) -> (f32, f32) {
    let w = k * omega * lobe.weight(d);
    (lum.max(floor).ln(), w)
}

fn open_sky(texels: &[[f32; 4]], res: u32) -> f32 {
    let (mut open, mut total) = (0.0f32, 0.0f32);
    for (i, t) in texels.iter().enumerate() {
        let r = i as u32 % (res * res);
        let (a, b) = (2.0 * ((r % res) as f32 + 0.5) / res as f32 - 1.0, 2.0 * ((r / res) as f32 + 0.5) / res as f32 - 1.0);
        let omega = (1.0 + a * a + b * b).powf(-1.5);
        open += (1.0 - t[3].clamp(0.0, 1.0)) * omega;
        total += omega;
    }
    if total > 0.0 { open / total } else { 0.0 }
}

fn sky_daylight(sky: &SkyIrradiance) -> f32 {
    let (mut sum, mut omega) = (0.0f32, 0.0f32);
    for (d, l, w) in sky_bins(sky) {
        if d.y > 0.0 {
            sum += l * w;
            omega += w;
        }
    }
    let mean = if omega > 0.0 { sum / omega } else { 0.0 };
    smoothstep(DAY_SKY_FROM.ln(), DAY_SKY_TO.ln(), mean.max(1e-12).ln())
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

    /// A night-dark outdoors with a lamp-lit room behind a doorway: looked
    /// at from near, the eye stays within `BRIGHT_VIEW_RATIO` of the room;
    /// from far off, or looking away, it keeps the night.
    #[test]
    fn a_lit_doorway_seen_from_the_night_holds_the_eye_within_reach_of_it() {
        let (hall, night) = (faces(16, |_| 0.2), faces(16, |_| 1e-5));
        let a = desc(Vec3::new(0.0, 1.5, -5.0), Vec3::new(-3.0, 0.0, -10.0), Vec3::new(3.0, 3.0, 0.0), 0, true);
        let out = desc(Vec3::new(0.0, 5.0, 20.0), Vec3::new(-50.0, -5.0, 0.0), Vec3::new(50.0, 50.0, 60.0), 1, false);
        let mut eye = EyeAdaptation::from_probes(&[(&hall, 16, a), (&night, 16, out)], SkyIrradiance::flat(1e-5));
        eye.set_max_exposure(2.0e5);
        eye.set_portals(&[ProbePortal { min: Vec3::new(-0.8, 0.0, -0.4), max: Vec3::new(0.8, 2.2, 0.0), axis: 2, low: 0, high: 1, wall: None }]);
        let near = eye.meter(Vec3::new(0.0, 1.6, 3.0), Vec3::NEG_Z);
        let far = eye.meter(Vec3::new(0.0, 1.6, 45.0), Vec3::NEG_Z);
        let away = eye.meter(Vec3::new(0.0, 1.6, 3.0), Vec3::Z);
        assert!(near >= 0.2 / BRIGHT_VIEW_RATIO * 0.99, "near the door the eye sits within reach of the hall: {near}");
        assert!(far < 0.2 / BRIGHT_VIEW_RATIO / 10.0, "a far door hardly moves the night eye: {far}");
        assert!(away < near / 10.0, "looking away keeps the night: {away} vs {near}");
    }

    #[test]
    fn a_lit_room_seen_from_the_night_is_drawn_at_its_own_adaptation_and_its_doorway_with_it() {
        let (hall, night) = (faces(16, |_| 0.2), faces_open(16, |_| 1e-5, |d| d.y > 0.0));
        let a = desc(Vec3::new(0.0, 1.5, -5.0), Vec3::new(-3.0, 0.0, -10.0), Vec3::new(3.0, 3.0, 0.0), 0, true);
        let out = desc(Vec3::new(0.0, 5.0, 20.0), Vec3::new(-50.0, -5.0, 0.0), Vec3::new(50.0, 50.0, 60.0), 1, false);
        let mut eye = EyeAdaptation::from_probes(&[(&hall, 16, a), (&night, 16, out)], SkyIrradiance::flat(1e-5));
        eye.set_max_exposure(2.0e5);
        let door = ProbePortal { min: Vec3::new(-0.8, 0.0, 0.0), max: Vec3::new(0.8, 2.2, 0.4), axis: 2, low: 0, high: 1, wall: None };
        eye.set_portals(&[door]);
        eye.set_local_rooms(true);
        let night_eye = exposure_for_up_to(eye.meter(Vec3::new(0.0, 1.6, 20.0), Vec3::NEG_Z), 2.0e5);
        let rows = eye.room_exposure(Vec3::new(0.0, 1.6, 20.0), night_eye);
        let inside_hall = room_factor_at(&rows, Vec3::new(0.0, 1.0, -5.0));
        let hall_eye = exposure_for_up_to(0.2, 2.0e5);
        assert!((night_eye * inside_hall / hall_eye - ROOM_SEEN_BRIGHTER).abs() < 0.3, "the hall at {} of its own eye", night_eye * inside_hall / hall_eye);
        assert!((room_factor_at(&rows, Vec3::new(0.0, 1.0, 0.2)) - inside_hall).abs() < 1e-6, "the doorway's jamb with the hall");
        assert_eq!(room_factor_at(&rows, Vec3::new(2.0, 1.0, 0.3)), 1.0, "the wall's outside face beside the door keeps the night eye");
        assert_eq!(room_factor_at(&rows, Vec3::new(0.0, 0.0, 10.0)), 1.0, "the yard keeps the night eye");
        // Inside, the eye is the hall's: nothing changes.
        let inner = exposure_for_up_to(eye.meter(Vec3::new(0.0, 1.6, -5.0), Vec3::NEG_Z), 2.0e5);
        assert_eq!(room_factor_at(&eye.room_exposure(Vec3::new(0.0, 1.6, -5.0), inner), Vec3::new(0.0, 1.0, -5.0)), 1.0);
        // Its light through the doorway no longer meters the night eye.
        let near = eye.meter(Vec3::new(0.0, 1.6, 6.0), Vec3::NEG_Z);
        assert!(near < 0.2 / BRIGHT_VIEW_RATIO / 10.0, "the yard by the door keeps the night: {near}");
    }

    #[test]
    fn the_eye_takes_its_time_to_adapt_to_the_night() {
        let mut eye = EyeAdaptation::sky_only(SkyIrradiance::flat(0.3));
        eye.set_max_exposure(2.0e5);
        eye.update(REFERENCE_LUMINANCE / 14.0, 0.016);
        let start = eye.adapted_luminance().unwrap().log2();
        for _ in 0..10 {
            eye.update(1e-6, 1.0);
        }
        let after = eye.adapted_luminance().unwrap().log2();
        let indoor_part = start - night_adapted_log2();
        let night_part = 10.0 - indoor_part / ADAPT_DARKER_STOPS_PER_S;
        let want = start - indoor_part - night_part * ADAPT_NIGHT_STOPS_PER_S;
        assert!((after - want).abs() < 0.2, "10 s from a lit room into the night: {after} vs {want}");
        assert!(after > 1e-6f32.log2() + 5.0, "still far from night-adapted after 10 s");
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

    /// Past the indoor range the dark-adapted eye takes in more: the curve
    /// runs on unbroken from `MAX_EXPOSURE`, lifts moonlit ground by more
    /// than a stop and a half, and leaves the indoor range as it was.
    #[test]
    fn the_night_eye_adapts_further_than_the_day_eye() {
        let knee = REFERENCE_LUMINANCE / MAX_EXPOSURE.powf(1.0 / ADAPTATION_STRENGTH);
        assert!((exposure_for_up_to(knee * 0.999, 1e6) / exposure_for_up_to(knee * 1.001, 1e6) - 1.0).abs() < 0.01, "a step at the knee");
        let moonlit = 3.5e-7;
        let day_curve = (REFERENCE_LUMINANCE / moonlit).powf(ADAPTATION_STRENGTH);
        assert!(exposure_for_up_to(moonlit, 1e6) > 3.0 * day_curve, "{} vs {day_curve}", exposure_for_up_to(moonlit, 1e6));
        assert_eq!(exposure_for_up_to(0.05, 1e6), exposure_for(0.05), "the indoor range is untouched");
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

        let room = faces(16, |d| 0.05 * (1.0 + 0.5 * d.x));
        let mut eye = EyeAdaptation::sky_only(SkyIrradiance::flat(0.5));
        eye.add_probe(&room, 16, &desc(Vec3::ZERO, Vec3::splat(-5.0), Vec3::splat(5.0), 0, true));
        let toward = eye.meter(Vec3::ZERO, Vec3::X);
        let away = eye.meter(Vec3::ZERO, Vec3::NEG_X);
        assert!(toward > away * 1.2, "centre-weighting ignored the gaze in a room: {toward} vs {away}");

        let mut night = SkyIrradiance::flat(2e-5);
        night.sh[3] = [3e-5, 3e-5, 3e-5];
        let mut eye = EyeAdaptation::sky_only(night);
        eye.set_max_exposure(2e5);
        let toward = eye.meter(Vec3::ZERO, Vec3::X);
        let away = eye.meter(Vec3::ZERO, Vec3::NEG_X);
        assert!(toward > away * 1.2, "centre-weighting ignored the gaze at night: {toward} vs {away}");
    }

    fn faces_open(res: u32, lum: impl Fn(Vec3) -> f32, open: impl Fn(Vec3) -> bool) -> Vec<u8> {
        let mut out = Vec::new();
        for face in 0..6usize {
            for y in 0..res {
                for x in 0..res {
                    let d = crate::renderer::probe_prefilter::texel_direction(
                        face,
                        (x as f32 + 0.5) / res as f32,
                        (y as f32 + 0.5) / res as f32,
                    );
                    let (l, a) = if open(d) { (0.0, 0.0) } else { (lum(d), 1.0) };
                    for v in [l, l, l, a] {
                        out.extend_from_slice(&crate::renderer::sky::f32_to_f16(v).to_le_bytes());
                    }
                }
            }
        }
        out
    }

    #[test]
    fn in_daylight_under_open_sky_turning_does_not_re_expose() {

        let mut sky = SkyIrradiance::flat(0.4);
        sky.sh[3] = [0.6, 0.6, 0.6];
        let outside = EyeAdaptation::sky_only(sky);

        let ground = faces_open(16, |d| if d.x < -0.5 { 0.03 } else { 0.3 }, |d| d.y > 0.0 && d.x > -0.5);
        let mut open = EyeAdaptation::sky_only(sky);
        open.add_probe(&ground, 16, &desc(Vec3::new(0.0, 2.0, 0.0), Vec3::splat(-20.0), Vec3::splat(20.0), 0, true));
        assert!(open.gaze_free_share(Vec3::new(0.0, 1.6, 0.0)) > 0.99);
        for (name, eye) in [("outside every probe", &outside), ("on open ground", &open)] {
            let head = Vec3::new(0.0, 1.6, 0.0);
            let looks: Vec<f32> = (0..16)
                .map(|i| {
                    let a = i as f32 * std::f32::consts::TAU / 16.0;
                    eye.meter(head, Vec3::new(a.cos(), -0.1, a.sin()))
                })
                .collect();
            let (lo, hi) = looks.iter().fold((f32::MAX, 0.0f32), |(lo, hi), &m| (lo.min(m), hi.max(m)));
            assert!((hi / lo).log2() < 0.02, "{name}: {:.3} stops as the head turns: {looks:?}", (hi / lo).log2());

            let down = eye.meter(head, Vec3::new(1.0, -0.6, 0.0));
            assert!((down / hi).log2().abs() < 0.3, "{name}: looking down {down} vs {hi}");
        }
    }

    #[test]
    fn a_room_with_a_door_to_the_sky_still_follows_the_gaze() {
        let mut sky = SkyIrradiance::flat(0.4);
        sky.sh[3] = [0.6, 0.6, 0.6];

        let room = faces_open(16, |_| 0.01, |d| d.x > 0.9 && d.y.abs() < 0.12 && d.z.abs() < 0.12);
        let mut eye = EyeAdaptation::sky_only(sky);
        eye.add_probe(&room, 16, &desc(Vec3::ZERO, Vec3::splat(-5.0), Vec3::splat(5.0), 0, true));
        assert!(eye.gaze_free_share(Vec3::ZERO) < 1e-3);
        assert!(eye.meter(Vec3::ZERO, Vec3::X) > 2.0 * eye.meter(Vec3::ZERO, Vec3::NEG_X));
    }
}
