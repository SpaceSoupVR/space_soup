//! EYE ADAPTATION, metered from the baked reflection probes.
//!
//! # Why a level needs it
//!
//! The hall in `test_room` averages 0.033 radiance; the open ground outside
//! averages 0.47 -- fourteen times brighter -- and both were shown at one fixed
//! exposure. Outdoors looked right and the hall read as murky, and the back of
//! the hall, which a path tracer confirms really is about ten times darker
//! than its front, read as black. A real eye walking in from daylight adapts
//! within a second or two; so does every contemporary engine's camera (Unity
//! HDRP and Unreal both meter the frame and ease the exposure toward it).
//!
//! # Why from the probes and not from the frame
//!
//! Those engines meter with a GPU pass that reduces the frame to its average
//! brightness. On the Quest the frame is fill-bound, and a full-screen pass is
//! the one thing it cannot afford. But the answer is already baked: every
//! reflection probe is a photograph of the room around it, taken in every
//! direction. Reduced once at load to 96 directions, it meters the view the
//! player is looking at for a few hundred multiply-adds a frame on the CPU,
//! and nothing on the GPU at all.
//!
//! What it cannot see is what the probe did not: a lamp switched on after the
//! bake, or the player's own torch. Those are the cases a frame meter handles
//! and this does not -- worth knowing, and not this level.
//!
//! # From where the player stands, not where the photograph was taken
//!
//! A photograph is taken from one point, and a room's light reaches the eye
//! from where the eye is: walk up to a doorway and the brighter room beyond
//! it fills half your view, though from the capture point it is a slot in a
//! far wall. Read from the capture point, the meter barely counted the next
//! room until the head crossed into it, then switched rooms at once -- and in
//! the thickness of the wall, inside no room at all, it read the OUTDOORS.
//! From the brick hall the hallway drew six times brighter than it looks once
//! you are in it, then dimmed as you stepped through (headset, 2026-10-01).
//! So a room's photographs are blended by how near they are, every doorway
//! of the room adds the light of the room beyond it in proportion to how much
//! of the view its opening fills FROM THE HEAD (`windows`) -- the opening is
//! known geometry, which the photographs' coarse bins are not -- and across a
//! doorway the two rooms it joins are handed over along its depth, as the
//! reflections are.

use glam::Vec3;

use super::probe_stream::ProbeDesc;
use super::sky::SkyIrradiance;
use super::uniforms::ProbePortal;

/// Faces are reduced to this many bins a side for metering: 6 x 4 x 4 = 96
/// directions per probe. A meter wants the broad distribution of light, not
/// detail, and a camera's own meter has fewer zones than this.
pub const METER_BINS: u32 = 4;

/// The metered luminance at which exposure is 1 -- the brightness the level
/// was already being shown at correctly.
///
/// Calibrated to `test_room`'s outdoor probe, metered with the sky filled in,
/// looking at the horizon (0.61): outdoors was the one place the fixed
/// exposure looked right, so it keeps its look, and everywhere else is exposed
/// relative to it. The hall, metered the same way, reads 0.002-0.03 -- the
/// back of it genuinely five to eight stops darker than daylight, which the
/// path-traced reference agrees with. See
/// `quest_app::offline_frame::exposure_calibration`.
///
/// Then set a little under that, at 0.5: on the headset (2026-09-23) outdoors
/// read slightly bright at exactly the old look, so it is exposed about 15%
/// down -- a camera's exposure compensation, not physics.
pub const REFERENCE_LUMINANCE: f32 = 0.5;

/// How completely the eye compensates, 0..1.
///
/// 1 would make every room exactly as bright as daylight, which is what a
/// camera on automatic does and not what a person sees: a dim room still
/// LOOKS dim after your eyes adjust, just not black. 0.8 leaves a factor of
/// ten in scene brightness looking about 1.6 times dimmer -- the hall darker
/// than outdoors, its back darker than its front, all of it readable.
pub const ADAPTATION_STRENGTH: f32 = 0.8;

/// The range the adaptation may move exposure through, in multiples.
///
/// A floor and a ceiling on how far the eye goes: a sealed, unlit room must
/// still read as dark rather than be pulled up to grey noise, and staring into
/// the sun must not black out the scene.
pub const MIN_EXPOSURE: f32 = 0.25;
///
/// 12 rather than 16: at 16 the dimmest views of the hall read a little too
/// bright on the headset (2026-09-23). A dim room should still look dim.
pub const MAX_EXPOSURE: f32 = 12.0;

/// How fast the eye adapts, in stops per second.
///
/// Faster going INTO brightness than out of it, as eyes are: stepping out
/// into daylight dazzles for a moment, stepping into a room takes longer to
/// see in. Unreal's defaults are 3 and 1; these are a little quicker on the
/// dark side because in VR a slow fade reads as the game stalling.
pub const ADAPT_BRIGHTER_STOPS_PER_S: f32 = 3.0;
pub const ADAPT_DARKER_STOPS_PER_S: f32 = 1.5;

/// Metering weight: `floor + max(0, cos)^power` for a bin at angle to the
/// gaze. Centre-weighted, like a camera's default: what you look at counts
/// most, but the rest of the room still counts for something.
const CENTRE_WEIGHT_FLOOR: f32 = 0.1;
const CENTRE_WEIGHT_POWER: f32 = 4.0;

/// Luminance floor for the log average, so a black texel cannot drag the
/// geometric mean to zero.
const LOG_FLOOR: f32 = 1e-4;

/// How far past each side of a doorway's box the meter hands over from one
/// room to the other, in metres. The box is only the wall's thickness and a
/// little: handed over across that alone, the brick hall's meter fell 0.8
/// stops in 28 cm.
const DOORWAY_HANDOVER: f32 = 0.5;

/// The nearest a doorway's opening is taken to be, in metres: see `windows`.
const WINDOW_NEAREST: f32 = 0.05;

/// A room's photographs are blended by `1 / max(d^2, this)`, `d` the distance
/// to each capture point: the nearest counts most, and the handover from one
/// to the next is gradual rather than a switch at the midpoint.
const NEAREST_PROBE_FLOOR: f32 = 0.25;

/// The band of the metered view that sets exposure, as weighted percentiles
/// of brightness: everything darker than the low one and brighter than the
/// high one is ignored.
///
/// A plain log-average of a room is ruled by its dark corners -- measured on
/// the hall, it read 0.0003 against an arithmetic mean of 0.033, and pinned
/// the exposure at its ceiling. Dark areas should not set exposure; the lit
/// part of what you are looking at should. Unreal's histogram metering works
/// the same way (its classic defaults average the 80th-98th percentile); the
/// top 2% is dropped so a lamp's own glare cannot black out the room.
pub const METER_LOW_PERCENTILE: f32 = 0.7;
pub const METER_HIGH_PERCENTILE: f32 = 0.98;

/// One probe, reduced for metering.
struct ProbeMeter {
    centre: Vec3,
    min: Vec3,
    max: Vec3,
    /// The room it photographs (`ProbeDesc::volume`), which the doorways name.
    room: u32,
    /// (direction, luminance, solid angle in steradians) per bin, from
    /// `centre`.
    bins: Vec<(Vec3, f32, f32)>,
}

/// The adapted eye: what it meters from, and where it has got to.
pub struct EyeAdaptation {
    meters: Vec<ProbeMeter>,
    /// The doorways between rooms, across which the meter hands over.
    portals: Vec<ProbePortal>,
    /// The sky, for metering where no probe covers the player.
    sky: SkyIrradiance,
    /// The luminance the eye is currently adapted to, as log2. `None` until
    /// the first frame, which adapts instantly rather than fading in from an
    /// arbitrary start.
    adapted_log2: Option<f32>,
}

impl EyeAdaptation {
    /// No probes: meters the sky alone.
    pub fn sky_only(sky: SkyIrradiance) -> Self {
        Self { meters: Vec::new(), portals: Vec::new(), sky, adapted_log2: None }
    }

    /// Reduce each probe -- `(faces, resolution, where and what it is)`, as
    /// `XrRenderer::set_reflection_probes` receives them -- to its metering
    /// bins. Where a probe texel saw sky (alpha < 1) the sky's own radiance
    /// fills in, since the probe stores coverage rather than sky.
    pub fn from_probes(probes: &[(&[u8], u32, ProbeDesc)], sky: SkyIrradiance) -> Self {
        let mut eye = Self::sky_only(sky);
        for (faces, res, desc) in probes {
            eye.add_probe(faces, *res, desc);
        }
        eye
    }

    /// Add one probe's meter -- for a level whose probes are read one at a
    /// time rather than held together. See `probe_stream`.
    pub fn add_probe(&mut self, faces: &[u8], res: u32, desc: &ProbeDesc) {
        if let Some(texels) = super::uniforms::decode_probe_texels(faces, res) {
            self.meters.push(ProbeMeter {
                centre: desc.centre,
                min: desc.min,
                max: desc.max,
                room: desc.volume,
                bins: bin_probe(&texels, res, &self.sky),
            });
        }
    }

    /// The level's doorways, which the meter hands over across.
    pub fn set_portals(&mut self, portals: &[ProbePortal]) {
        self.portals = portals.to_vec();
    }

    /// The luminance a centre-weighted meter reads at `head` looking along
    /// `gaze`, both in WORLD space.
    pub fn meter(&self, head: Vec3, gaze: Vec3) -> f32 {
        let gaze = gaze.normalize_or_zero();
        // IN A DOORWAY: both rooms it joins, handed over along its depth. In
        // the wall's thickness the head is in neither room's box, and the
        // smallest box round it was the outdoors'.
        for p in &self.portals {
            let a = p.axis.min(2) as usize;
            let (mut lo, mut hi) = (p.min, p.max);
            lo[a] -= DOORWAY_HANDOVER;
            hi[a] += DOORWAY_HANDOVER;
            if head.cmpge(lo).all() && head.cmple(hi).all() {
                let f = smoothstep(lo[a], hi[a], head[a]);
                match (self.room_log(p.low, head, gaze), self.room_log(p.high, head, gaze)) {
                    (Some(lo), Some(hi)) => return (lo + (hi - lo) * f).exp(),
                    (Some(one), None) | (None, Some(one)) => return one.exp(),
                    (None, None) => {}
                }
            }
        }
        let log = match self.room_at(head) {
            Some(room) => self.room_log(room, head, gaze),
            None => {
                let samples: Vec<(f32, f32)> =
                    sky_bins(&self.sky).into_iter().map(|b| weighted(b, gaze, 1.0)).collect();
                band_log(samples)
            }
        };
        log.map_or(REFERENCE_LUMINANCE, f32::exp)
    }

    /// The room the player is in: the same rule the shader uses to pick a
    /// probe for a surface -- the smallest box containing the point.
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

    /// What the meter reads in `room` from `head`, as a log luminance: the
    /// room's photographs, and through each of its doorways the room beyond,
    /// as much as the opening fills of the view. `None` for a room with no
    /// photographs.
    fn room_log(&self, room: u32, head: Vec3, gaze: Vec3) -> Option<f32> {
        let mut samples = Vec::new();
        if !self.photographs(room, head, gaze, 1.0, &mut samples) {
            return None;
        }
        for (beyond, share) in self.windows(room, head, gaze) {
            let mut through = Vec::new();
            if self.photographs(beyond, head, gaze, 1.0, &mut through) {
                // The room beyond as the opening frames it: its light, in the
                // weight the opening has here.
                let total: f32 = through.iter().map(|s| s.1).sum();
                if total > 0.0 {
                    samples.extend(through.into_iter().map(|(l, w)| (l, w * share / total)));
                }
            }
        }
        band_log(samples)
    }

    /// `room`'s photographs as weighted log-luminance samples, each probe's
    /// weight `scale` times its share of `1 / max(d^2, floor)`: the nearest
    /// counts most. False for a room with none.
    fn photographs(&self, room: u32, head: Vec3, gaze: Vec3, scale: f32, out: &mut Vec<(f32, f32)>) -> bool {
        let near = |m: &ProbeMeter| 1.0 / m.centre.distance_squared(head).max(NEAREST_PROBE_FLOOR);
        let total: f32 = self.meters.iter().filter(|m| m.room == room).map(near).sum();
        if total <= 0.0 {
            return false;
        }
        for m in self.meters.iter().filter(|m| m.room == room) {
            let k = scale * near(m) / total;
            out.extend(m.bins.iter().map(|&b| weighted(b, gaze, k)));
        }
        true
    }

    /// `room`'s doorways seen from `head`: the room beyond each, and the
    /// centre-weighted solid angle its opening fills -- in the units of a
    /// photograph's bins, whose weights sum to about 2.5 for a whole view.
    /// The opening is taken at the far face of the doorway's box, so it
    /// fills the view as the head reaches it.
    fn windows(&self, room: u32, head: Vec3, gaze: Vec3) -> Vec<(u32, f32)> {
        let mut out = Vec::new();
        for p in &self.portals {
            let a = p.axis.min(2) as usize;
            // The plane of the opening, how far ahead of the head, and the room
            // beyond it.
            let (plane, beyond, ahead) = if p.low == room {
                (p.max[a], p.high, p.max[a] - head[a])
            } else if p.high == room {
                (p.min[a], p.low, head[a] - p.min[a])
            } else {
                continue;
            };
            // A head past the opening -- which the handover reaches -- sees
            // it as from just short of it: the view through it stays at its
            // fullest rather than vanishing as the plane is crossed.
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

    /// Advance the eye by `dt` seconds toward `metered` luminance and return
    /// the exposure multiplier to render with.
    pub fn update(&mut self, metered: f32, dt: f32) -> f32 {
        let target = metered.max(LOG_FLOOR).log2();
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
        exposure_for(current.exp2())
    }
}

/// The steady-state exposure for an eye fully settled at `luminance`.
pub fn exposure_for(luminance: f32) -> f32 {
    let ratio = REFERENCE_LUMINANCE / luminance.max(LOG_FLOOR);
    ratio.powf(ADAPTATION_STRENGTH).clamp(MIN_EXPOSURE, MAX_EXPOSURE)
}

/// A bin as a weighted log-luminance sample: its solid angle, centre-weighted
/// toward `gaze`, times `k`.
fn weighted((d, lum, omega): (Vec3, f32, f32), gaze: Vec3, k: f32) -> (f32, f32) {
    let w = k * omega * (CENTRE_WEIGHT_FLOOR + d.dot(gaze).max(0.0).powf(CENTRE_WEIGHT_POWER));
    (lum.max(LOG_FLOOR).ln(), w)
}

/// The solid angle of a rectangle `ahead` metres in front of a point, its
/// sides spanning `x` and `y` across the plane relative to the point's foot.
fn rectangle_solid_angle(ahead: f32, x: (f32, f32), y: (f32, f32)) -> f32 {
    let f = |x: f32, y: f32| (x * y / (ahead * (ahead * ahead + x * x + y * y).sqrt())).atan();
    (f(x.1, y.1) - f(x.0, y.1) - f(x.1, y.0) + f(x.0, y.0)).max(0.0)
}

/// The meter's reading of weighted log-luminance samples: the weighted
/// average over the percentile band. `None` for no weight.
fn band_log(mut samples: Vec<(f32, f32)>) -> Option<f32> {
    // Darkest first.
    samples.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let total: f32 = samples.iter().map(|s| s.1).sum();
    if total <= 0.0 {
        return None;
    }
    // Average over the percentile band, taking the part of each sample's
    // weight that falls inside it.
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

/// A probe's texels reduced to `METER_BINS` x `METER_BINS` per face.
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
                        // A cube texel's solid angle shrinks toward the face
                        // corners as 1 / (1 + a^2 + b^2)^(3/2).
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
                    // In steradians: a texel is 2 / res across a face of side 2.
                    let texel = 2.0 / res as f32;
                    out.push((d, sum / omega_sum, omega_sum * texel * texel));
                }
            }
        }
    }
    out
}

/// The sky alone, binned the same way, for a player outside every probe.
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
        // Partial: the dim room is exposed up LESS than its darkness, so it
        // still reads darker once adapted.
        assert!(hall < 14.0, "full adaptation would be 14x; got {hall}");
        assert_eq!(exposure_for(1e-9), MAX_EXPOSURE);
    }

    #[test]
    fn the_eye_adapts_at_its_rates_and_settles() {
        let mut eye = EyeAdaptation::sky_only(SkyIrradiance::flat(0.3));
        let start = eye.update(REFERENCE_LUMINANCE, 0.016);
        assert!((start - 1.0).abs() < 1e-5, "first frame adapts instantly");
        // Walk into a room 3 stops darker -- inside the clamp, so the rate is
        // what is measured: after 1 s the eye has moved 1.5 stops, not all 3.
        let dark = REFERENCE_LUMINANCE / 8.0;
        let after_1s = eye.update(dark, 1.0);
        let expect = 2f32.powf(1.5 * ADAPTATION_STRENGTH);
        assert!((after_1s / expect - 1.0).abs() < 1e-3, "{after_1s} vs {expect}");
        for _ in 0..20 {
            eye.update(dark, 1.0);
        }
        assert!((eye.update(dark, 0.0) - exposure_for(dark)).abs() < 1e-4);
        // And back out into the light, twice as fast: all 3 stops in 1 s.
        let back = eye.update(REFERENCE_LUMINANCE, 1.0);
        assert!((back - 1.0).abs() < 1e-4, "{back}");
    }

    /// A probe's faces, RGBA half floats as baked, every texel seen (alpha 1)
    /// at the grey `lum(direction)` gives.
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

    /// Two rooms side by side along x, a dark one and a bright one, a wall
    /// between them 0.2 m thick with a doorway through it -- and the outdoors
    /// round both, brighter than either.
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

    /// THROUGH A DOORWAY THE METER HANDS OVER: from a dark room into a bright
    /// one, a centimetre at a time, it moves from the one's reading to the
    /// other's without a step -- and never reads the outdoors. In the wall's
    /// thickness the head is in neither room, and the smallest box round it
    /// was the outdoors': walking through any door flashed the exposure to
    /// daylight's (headset, 2026-10-01).
    #[test]
    fn a_doorway_hands_the_meter_over_without_a_step_or_the_outdoors() {
        let eye = two_rooms();
        let walk: Vec<f32> = (0..=400).map(|i| eye.meter(Vec3::new(-2.0 + i as f32 * 0.01, 1.6, 0.0), Vec3::X)).collect();
        // Facing away from the doorway, each room reads as itself.
        assert!((eye.meter(Vec3::new(-2.0, 1.6, 0.0), Vec3::NEG_X) / 0.01 - 1.0).abs() < 0.05);
        assert!((walk[400] / 0.3 - 1.0).abs() < 0.01, "in the bright room: {}", walk[400]);
        for (i, w) in walk.windows(2).enumerate() {
            assert!(w[1] <= 0.3 * 1.001, "{} cm along: read {} -- the outdoors", i + 1, w[1]);
            assert!((w[1] / w[0]).ln().abs() < 0.15, "{} cm along: {} then {}", i + 1, w[0], w[1]);
        }
    }

    /// LOOKING INTO A BRIGHT ROOM THROUGH ITS DOORWAY meters it, the more the
    /// nearer: the opening fills more of the view. From the brick hall the
    /// hallway barely counted, standing at its door and looking through,
    /// until the head crossed into it -- then the exposure fell 2.6 stops
    /// (headset, 2026-10-01). Looking away, the room reads as itself.
    #[test]
    fn a_bright_doorway_meters_the_more_the_nearer_it_is() {
        let eye = two_rooms();
        let far = eye.meter(Vec3::new(-4.5, 1.6, 0.0), Vec3::X);
        let near = eye.meter(Vec3::new(-1.0, 1.6, 0.0), Vec3::X);
        let away = eye.meter(Vec3::new(-1.0, 1.6, 0.0), Vec3::NEG_X);
        // A doorway 4.8 m off fills under a twentieth of the view, and moves
        // the meter a little; at 1.3 m it fills a third.
        assert!(near > far * 1.5 && far > away, "near {near}, far {far}, looking away {away}");
        assert!(near > 0.01 * 4.0 && near < 0.3, "{near}");
        // Behind the head it still counts a little -- the meter's floor
        // weight, as for the photographs.
        assert!(away < 0.01 * 1.5, "{away}");
    }

    /// The solid angle of a rectangle: a quarter of the view of a square
    /// seen from its corner's normal at no distance, and a sphere's whole
    /// sixth for a cube face seen from the cube's middle.
    #[test]
    fn a_rectangle_fills_the_solid_angle_it_should() {
        let face = rectangle_solid_angle(1.0, (-1.0, 1.0), (-1.0, 1.0));
        assert!((face - 4.0 * std::f32::consts::PI / 6.0).abs() < 1e-4, "{face}");
        let tiny = rectangle_solid_angle(10.0, (-0.05, 0.05), (-0.05, 0.05));
        assert!((tiny / (0.01 / 100.0) - 1.0).abs() < 1e-3, "{tiny}");
        assert_eq!(rectangle_solid_angle(1.0, (2.0, 3.0), (2.0, 3.0)) > 0.0, true);
    }

    #[test]
    fn looking_at_the_bright_side_meters_brighter() {
        // A sky twice as bright on +x as on -x.
        let mut sky = SkyIrradiance::flat(0.2);
        sky.sh[3] = [0.3, 0.3, 0.3]; // L11: the x band
        let eye = EyeAdaptation::sky_only(sky);
        let toward = eye.meter(Vec3::ZERO, Vec3::X);
        let away = eye.meter(Vec3::ZERO, Vec3::NEG_X);
        assert!(toward > away * 1.2, "centre-weighting ignored the gaze: {toward} vs {away}");
    }
}
