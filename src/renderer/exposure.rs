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

use glam::Vec3;

use super::sky::SkyIrradiance;

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
    /// (direction, luminance, solid-angle weight) per bin.
    bins: Vec<(Vec3, f32, f32)>,
}

/// The adapted eye: what it meters from, and where it has got to.
pub struct EyeAdaptation {
    meters: Vec<ProbeMeter>,
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
        Self { meters: Vec::new(), sky, adapted_log2: None }
    }

    /// Reduce each probe -- `(faces, resolution, capture point, box min, box
    /// max)`, as `XrRenderer::set_reflection_probes` receives them -- to its
    /// metering bins. Where a probe texel saw sky (alpha < 1) the sky's own
    /// radiance fills in, since the probe stores coverage rather than sky.
    pub fn from_probes(
        probes: &[(&[u8], u32, Vec3, Vec3, Vec3)],
        sky: SkyIrradiance,
    ) -> Self {
        let meters = probes
            .iter()
            .filter_map(|&(faces, res, centre, min, max)| {
                let texels = super::uniforms::decode_probe_texels(faces, res)?;
                Some(ProbeMeter { centre, min, max, bins: bin_probe(&texels, res, &sky) })
            })
            .collect();
        Self { meters, sky, adapted_log2: None }
    }

    /// Add one probe's meter -- for a level whose probes are read one at a
    /// time rather than held together. See `probe_stream`.
    pub fn add_probe(&mut self, faces: &[u8], res: u32, centre: Vec3, min: Vec3, max: Vec3) {
        if let Some(texels) = super::uniforms::decode_probe_texels(faces, res) {
            self.meters.push(ProbeMeter { centre, min, max, bins: bin_probe(&texels, res, &self.sky) });
        }
    }

    /// The luminance a centre-weighted meter reads at `head` looking along
    /// `gaze`, both in WORLD space.
    pub fn meter(&self, head: Vec3, gaze: Vec3) -> f32 {
        let gaze = gaze.normalize_or_zero();
        let bins: Vec<(Vec3, f32, f32)> = match self.probe_at(head) {
            Some(m) => m.bins.clone(),
            None => sky_bins(&self.sky),
        };
        // Weighted log luminances, darkest first.
        let mut samples: Vec<(f32, f32)> = bins
            .iter()
            .map(|&(d, lum, omega)| {
                let w = omega * (CENTRE_WEIGHT_FLOOR + d.dot(gaze).max(0.0).powf(CENTRE_WEIGHT_POWER));
                (lum.max(LOG_FLOOR).ln(), w)
            })
            .collect();
        samples.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let total: f32 = samples.iter().map(|s| s.1).sum();
        if total <= 0.0 {
            return REFERENCE_LUMINANCE;
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
        if weight > 0.0 { (sum / weight).exp() } else { REFERENCE_LUMINANCE }
    }

    /// The probe whose room the player is in: the same rule the shader uses to
    /// pick a probe for a surface -- smallest box containing the point, then
    /// the nearest capture point.
    fn probe_at(&self, p: Vec3) -> Option<&ProbeMeter> {
        let mut best: Option<(&ProbeMeter, f32, f32)> = None;
        for m in &self.meters {
            if p.cmplt(m.min).any() || p.cmpgt(m.max).any() {
                continue;
            }
            let d = m.max - m.min;
            let volume = d.x * d.y * d.z;
            let dist = m.centre.distance_squared(p);
            let better = match best {
                None => true,
                Some((_, v, dd)) => volume < v * 0.999 || (volume < v * 1.001 && dist < dd),
            };
            if better {
                best = Some((m, volume, dist));
            }
        }
        best.map(|b| b.0)
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
                    out.push((d, sum / omega_sum, omega_sum));
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
                out.push((d, luminance(sky.radiance(d.to_array())), (1.0 + a * a + b * b).powf(-1.5)));
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
