use bytemuck::{Pod, Zeroable};
use glam::{Quat, Vec3};
use wgpu::*;

use super::lights::{Light, LightKind};
use super::probe_stream::ProbeDesc;

mod flash;
pub mod mud_splash;
pub mod authoring;
pub mod catalog;
pub mod seen;
pub mod stack;
pub mod templates;
pub mod variants;
pub use variants::FxVariant;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EffectKind {

    Fire,

    Smoke,

    Embers,

    Dust,

    Coals,
}

impl EffectKind {

    pub fn from_name(name: &str) -> Option<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "fire" | "flame" | "flames" => Some(Self::Fire),
            "smoke" => Some(Self::Smoke),
            "embers" | "sparks" => Some(Self::Embers),
            "dust" | "motes" | "dust_motes" => Some(Self::Dust),
            "coals" | "coal" | "fire_bed" | "ember_bed" => Some(Self::Coals),
            _ => None,
        }
    }

    pub fn screened(self) -> bool {
        matches!(self, Self::Embers | Self::Dust)
    }
}

#[derive(Clone, Debug)]
pub struct EffectEmitter {

    pub id: String,
    pub kind: EffectKind,
    pub position: Vec3,

    pub direction: Vec3,

    pub extent: [Vec3; 3],

    pub scale: f32,

    pub rate: f32,

    pub tint: [f32; 3],

    pub ceiling: Option<f32>,

    pub variation: Variation,
}

impl EffectEmitter {

    pub fn volume(&self) -> f32 {
        8.0 * self.extent[0].length() * self.extent[1].length() * self.extent[2].length()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Variation {

    pub intensity: f32,

    pub size_variation: f32,

    pub life_variation: f32,

    pub speed_variation: f32,

    pub temperature_variation: f32,

    pub flicker: f32,

    pub flicker_rate: f32,

    pub turbulence: f32,

    pub seed: u32,
}

impl Default for Variation {
    fn default() -> Self {
        Self {
            intensity: 1.0,
            size_variation: 1.0,
            life_variation: 1.0,
            speed_variation: 1.0,
            temperature_variation: 0.0,
            flicker: 1.0,
            flicker_rate: 1.0,
            turbulence: 1.0,
            seed: 0,
        }
    }
}

impl Variation {

    pub fn clamped(self) -> Self {
        let c = |v: f32, lo: f32, hi: f32, or: f32| if v.is_nan() { or } else { v.clamp(lo, hi) };
        Self {
            intensity: c(self.intensity, 0.0, 4.0, 1.0),
            size_variation: c(self.size_variation, 0.0, 3.0, 1.0),
            life_variation: c(self.life_variation, 0.0, 2.0, 1.0),
            speed_variation: c(self.speed_variation, 0.0, 3.0, 1.0),
            temperature_variation: c(self.temperature_variation, 0.0, 1.0, 0.0),
            flicker: c(self.flicker, 0.0, 3.0, 1.0),
            flicker_rate: c(self.flicker_rate, 0.25, 4.0, 1.0),
            turbulence: c(self.turbulence, 0.0, 4.0, 1.0),
            seed: self.seed.min(65535),
        }
    }
}

fn emitter_seed(e: &EffectEmitter) -> u64 {
    let base = id_hash(&e.id);
    match e.variation.seed.min(65535) {
        0 => base,
        s => base ^ hash64(u64::from(s)),
    }
}

fn life_range(p: &Preset, v: &Variation) -> (f32, f32) {
    if v.life_variation == 1.0 {
        return p.life;
    }
    let m = f32::midpoint(p.life.0, p.life.1);
    let h = 0.5 * (p.life.1 - p.life.0) * v.life_variation;
    ((m - h).max(0.05), m + h)
}

fn speed_range(p: &Preset, v: &Variation) -> (f32, f32) {
    if v.speed_variation == 1.0 {
        return p.speed;
    }
    let m = f32::midpoint(p.speed.0, p.speed.1);
    let h = 0.5 * (p.speed.1 - p.speed.0) * v.speed_variation;
    ((m - h).max(0.0), m + h)
}

fn size_spread(u: f32, v: &Variation) -> f32 {
    if v.size_variation == 1.0 {
        0.8 + 0.4 * u
    } else {
        (1.0 + (u - 0.5) * 0.4 * v.size_variation).max(0.05)
    }
}

fn temperature(u: f32, v: &Variation) -> Vec3 {
    if v.temperature_variation == 0.0 {
        return Vec3::ONE;
    }
    let k = (2.0 * u - 1.0) * v.temperature_variation;
    Vec3::new(1.0, 2f32.powf(0.8 * k), 2f32.powf(1.6 * k))
}

#[derive(Clone, Debug, PartialEq)]
struct Preset {

    rate: f32,
    life: (f32, f32),
    speed: (f32, f32),
    spread_deg: f32,

    bed: f32,
    size: (f32, f32),
    drag: f32,
    accel: Vec3,

    swirl: (f32, f32),

    spin: (f32, f32),

    streak: f32,

    soft: f32,

    frames: (u32, u32),

    phase_g: f32,
    albedo: f32,
}

pub const SMOKE_FRAMES: u32 = 16;
pub const SMOKE_BACK: u32 = SMOKE_FRAMES;
pub const FIRE_FIRST: u32 = 2 * SMOKE_FRAMES;

pub const FIRE_FRAMES: u32 = 48;
pub const FLAME_SECONDS: f32 = 2.4;
pub const EMBER_LAYER: u32 = FIRE_FIRST + FIRE_FRAMES;
pub const MOTE_LAYER: u32 = EMBER_LAYER + 1;
pub const COALS_LAYER: u32 = MOTE_LAYER + 1;

pub const CROWN_LAYER: u32 = COALS_LAYER + 1;
pub const CROWN_VARIANTS: u32 = 4;
pub const CROWN_FRAMES: u32 = 3;

pub const JET_LAYER: u32 = CROWN_LAYER + CROWN_VARIANTS * CROWN_FRAMES;
pub const JET_VARIANTS: u32 = 2;
pub const JET_FRAMES: u32 = 3;

pub const SPRAY_LAYER: u32 = JET_LAYER + JET_VARIANTS * JET_FRAMES;
pub const SPRAY_VARIANTS: u32 = 2;
pub const SPRAY_FRAMES: u32 = 3;
pub const FROTH_LAYER: u32 = SPRAY_LAYER + SPRAY_VARIANTS * SPRAY_FRAMES;
pub const FROTH_VARIANTS: u32 = 2;
pub const FROTH_FRAMES: u32 = 3;

pub const FLASH_LAYER: u32 = FROTH_LAYER + FROTH_VARIANTS * FROTH_FRAMES;
pub const FLASH_VARIANTS: u32 = 4;
pub const FLASH_FRAMES: u32 = 3;
pub const CORE_LAYER: u32 = FLASH_LAYER + FLASH_VARIANTS * FLASH_FRAMES;
pub const CORE_VARIANTS: u32 = 2;
/// Mud clumps: each a clod's normal and cover (`mud_splash::lump_frame`).
pub const MUD_LAYER: u32 = CORE_LAYER + CORE_VARIANTS;
pub const MUD_VARIANTS: u32 = 4;
pub const ATLAS_LAYERS: u32 = MUD_LAYER + MUD_VARIANTS;
pub const ATLAS_SIZE: u32 = 128;
/// A mud lump's wet sheen: its peak (over the mud's albedo, so a thin water
/// film's 3% reflectance against an albedo of about 0.1) and its sharpness.
const LUMP_SHEEN: f32 = 1.0;
const LUMP_GLOSS: f32 = 40.0;
const ATLAS_MIPS: u32 = 8;

const MAX_SLOTS: usize = 2048;

const DUST_SEEN_WITHIN: f32 = 5.0;

const DUST_NEAREST: (f32, f32) = (0.12, 0.35);

const DUST_AMBIENT_SEEN: f32 = 0.15;

const DUST_LIGHT_SIZE: (f32, f32) = (0.00058, 0.0035);

const GLINT_FLOOR: f32 = 0.08;
const GLINT_MEAN: f32 = 0.357;

pub const VERTICES_PER_PARTICLE: u32 = 24;

const CONVEX: f32 = 0.35;

fn preset(kind: EffectKind) -> Preset {
    templates::template(kind).preset.clone()
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct EffectInstance {

    pub centre: [f32; 4],

    pub colour: [f32; 4],

    pub light: [f32; 4],

    pub light_dir: [f32; 4],

    pub axis: [f32; 4],

    pub params: [f32; 4],

    pub motion: [f32; 4],
}

impl EffectInstance {
    pub const ATTRIBS: [VertexAttribute; 6] = vertex_attr_array![
        0 => Float32x4, 1 => Float32x4, 2 => Float32x4, 3 => Float32x4, 4 => Float32x4, 5 => Float32x4
    ];

    pub fn layout() -> VertexBufferLayout<'static> {
        VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as BufferAddress,
            step_mode: VertexStepMode::Instance,
            attributes: &Self::ATTRIBS,
        }
    }

    pub const MOTION_ATTRIBS: [VertexAttribute; 7] = vertex_attr_array![
        0 => Float32x4, 1 => Float32x4, 2 => Float32x4, 3 => Float32x4, 4 => Float32x4, 5 => Float32x4, 6 => Float32x4
    ];

    pub fn motion_layout() -> VertexBufferLayout<'static> {
        VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as BufferAddress,
            step_mode: VertexStepMode::Instance,
            attributes: &Self::MOTION_ATTRIBS,
        }
    }
}

#[derive(Default)]
pub struct EffectFrame {
    pub instances: Vec<EffectInstance>,
    pub over: u32,
    pub screened: u32,

    pub screened_heat: u32,

    pub bursts: stack::BurstCount,

    pub splash_costs: Vec<(u64, f32, f32)>,
}

#[derive(Clone, Copy)]
pub struct Surroundings<'a> {

    pub head: Vec3,

    pub offset: Vec3,
    pub yaw_inv: Quat,

    pub lights: &'a [Light],

    pub reaches: &'a dyn Fn(&EffectEmitter, usize) -> bool,

    pub ambient: &'a dyn Fn(Vec3) -> Vec3,
    pub exposure: f32,
    /// The room-local adaptation at a world point (`exposure::room_factor_at`):
    /// an effect in a closed room is drawn at the room's adaptation, as its
    /// walls are. None: every effect at `exposure`.
    pub room: Option<&'a dyn Fn(Vec3) -> f32>,
}

impl Surroundings<'_> {
    /// These surroundings for an effect at `world`: the exposure of the room
    /// it is in.
    pub fn at_point(&self, world: Vec3) -> Self {
        Self { exposure: self.exposure * self.room.map_or(1.0, |r| r(world)), ..*self }
    }
}

fn hash64(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58476d1ce4e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}

fn id_hash(id: &str) -> u64 {
    id.bytes().fold(0xcbf29ce484222325u64, |h, b| (h ^ u64::from(b)).wrapping_mul(0x100000001b3))
}

fn uniforms(seed: u64) -> [f32; 12] {
    let mut out = [0.0; 12];
    let mut h = seed;
    for v in &mut out {
        h = hash64(h.wrapping_add(0x9e3779b97f4a7c15));
        *v = (h >> 40) as f32 / (1u64 << 24) as f32;
    }
    out
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn cone_direction(forward: Vec3, spread_deg: f32, u1: f32, u2: f32) -> Vec3 {
    let forward = forward.try_normalize().unwrap_or(Vec3::Y);
    let right = forward.any_orthonormal_vector();
    let up = forward.cross(right);
    let cos_max = spread_deg.min(180.0).to_radians().cos();
    let cos_t = lerp(1.0, cos_max, u1);
    let sin_t = (1.0 - cos_t * cos_t).max(0.0).sqrt();
    let phi = u2 * std::f32::consts::TAU;
    right * (sin_t * phi.cos()) + up * (sin_t * phi.sin()) + forward * cos_t
}

fn ballistic(p0: Vec3, v0: Vec3, accel: Vec3, k: f32, age: f32) -> (Vec3, Vec3) {
    if k < 1e-4 {
        return (p0 + v0 * age + accel * (0.5 * age * age), v0 + accel * age);
    }
    let decay = (-k * age).exp();
    let a1 = (1.0 - decay) / k;
    let pos = p0 + v0 * a1 + accel * ((age - a1) / k);
    let vel = accel / k + (v0 - accel / k) * decay;
    (pos, vel)
}

const MOTION_H: f32 = 1.0 / 64.0;

fn swirl_motion(growth: f32, radius: f32, turn: f32, angle: f32) -> Vec3 {
    Vec3::new(angle.cos(), 0.0, angle.sin()) * growth + Vec3::new(-angle.sin(), 0.0, angle.cos()) * (radius * turn)
}

fn lamp_at(l: &Light, p: Vec3) -> Option<(Vec3, Vec3)> {
    lamp_seen_at(&lamp_seen(l, 1.0)?, p)
}

/// A lamp as the motes read it, worked out once a frame instead of once a
/// mote: its linear colour (three `powf`), its cone's cosines and the fire
/// adaptation share `own` (`simulate_governed`). Dust spent 0.94 ms of the
/// Quest's CPU a frame on 893 motes, ~1 us a mote, most of it here and in
/// `air_drift`'s clocks (`time_a_frames_effects`).
struct LampSeen {
    position: Vec3,
    range: f32,
    colour: Vec3,
    intensity: f32,
    source_radius: f32,
    /// A spot's (cos outer, cos inner, unit direction).
    spot: Option<(f32, f32, Vec3)>,
    own: f32,
}

fn lamp_seen(l: &Light, own: f32) -> Option<LampSeen> {
    if matches!(l.kind, LightKind::Directional) || l.intensity <= 0.0 {
        return None;
    }
    let c = l.color.to_linear();
    let spot = matches!(l.kind, LightKind::Spot).then(|| {
        let (cos_outer, cos_inner) = l.cone_cosines();
        (cos_outer, cos_inner, l.direction.normalize_or_zero())
    });
    Some(LampSeen { position: l.position, range: l.range, colour: Vec3::new(c[0], c[1], c[2]), intensity: l.intensity, source_radius: l.source_radius, spot, own })
}

fn lamp_seen_at(l: &LampSeen, p: Vec3) -> Option<(Vec3, Vec3)> {
    let to = l.position - p;
    let d = to.length();
    if d >= l.range || d < 1e-5 {
        return None;
    }
    let dir = to / d;
    let x = d / l.range.max(1e-4);
    let window = (1.0 - x * x * x * x).clamp(0.0, 1.0);
    let near = super::glare::LAMP_RADIUS;
    let atten = window * window / (d * d + l.source_radius * l.source_radius).max(near * near);
    let cone = match l.spot {
        Some((cos_outer, cos_inner, axis)) => smoothstep(cos_outer, cos_inner, (-dir).dot(axis)),
        None => 1.0,
    };
    if cone <= 0.0 {
        return None;
    }
    Some((l.colour * (l.intensity * atten * cone), dir))
}

fn luminance(c: Vec3) -> f32 {
    c.dot(Vec3::new(0.2126, 0.7152, 0.0722))
}

fn henyey_greenstein(g: f32, travel: Vec3, to_eye: Vec3) -> f32 {
    let cos_t = travel.dot(to_eye);
    let g2 = g * g;
    let d = (1.0 + g2 - 2.0 * g * cos_t).max(1e-4);
    ((1.0 - g2) / (d * d.sqrt())).min(12.0)
}

pub fn mist_conditions(hour: f32) -> f32 {
    const AT: [(f32, f32); 7] = [(0.0, 0.85), (5.0, 1.0), (7.5, 1.0), (10.5, 0.05), (17.0, 0.05), (21.0, 0.8), (24.0, 0.85)];
    let h = hour.rem_euclid(24.0);
    let i = AT.iter().rposition(|&(t, _)| t <= h).unwrap_or(0).min(AT.len() - 2);
    let ((t0, a), (t1, b)) = (AT[i], AT[i + 1]);
    let x = ((h - t0) / (t1 - t0)).clamp(0.0, 1.0);
    a + (b - a) * x * x * (3.0 - 2.0 * x)
}

fn noise1(x: f32, seed: u64) -> f32 {
    let i = x.floor();
    let f = x - i;
    let slope = |k: f32| (hash64(seed ^ (k as i64 as u64).wrapping_mul(0x9e3779b97f4a7c15)) >> 40) as f32 / (1u64 << 23) as f32 - 1.0;
    let a = slope(i) * f;
    let b = slope(i + 1.0) * (f - 1.0);
    let s = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    2.0 * lerp(a, b, s)
}

fn noise1d(x: f64, seed: u64) -> f32 {
    let i = x.floor();
    let f = (x - i) as f32;
    let slope = |k: f64| (hash64(seed ^ (k as i64 as u64).wrapping_mul(0x9e3779b97f4a7c15)) >> 40) as f32 / (1u64 << 23) as f32 - 1.0;
    let a = slope(i) * f;
    let b = slope(i + 1.0) * (f - 1.0);
    let s = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    2.0 * lerp(a, b, s)
}

#[derive(Clone, Copy)]
struct AirWave {
    k: Vec3,
    across: Vec3,
    reach: f32,
    w: f32,
    phase: f32,
}

fn air_waves(seed: u64) -> [AirWave; 6] {
    std::array::from_fn(|i| {
        let u = uniforms(hash64(seed ^ (i as u64 + 1).wrapping_mul(0xd6e8feb86659fd93)));
        let broad = i < 3;
        let wavelength = if broad { lerp(1.5, 4.0, u[0]) } else { lerp(0.4, 0.8, u[0]) };

        let dir = cone_direction(Vec3::Y, 180.0, u[1], u[2]);
        let dir = (dir * Vec3::new(1.0, 0.5, 1.0)).normalize_or(Vec3::X);
        let k = dir * (std::f32::consts::TAU / wavelength);
        let across = dir.cross(cone_direction(Vec3::Y, 180.0, u[3], u[4])).normalize_or(dir.any_orthonormal_vector());
        let (reach, w) = if broad { (lerp(0.05, 0.09, u[5]), lerp(0.12, 0.25, u[6])) } else { (lerp(0.008, 0.015, u[5]), lerp(0.5, 1.0, u[6])) };
        AirWave { k, across, reach, w, phase: u[7] * std::f32::consts::TAU }
    })
}

#[cfg(test)]
fn air_drift(waves: &[AirWave; 6], p: Vec3, time: f64) -> Vec3 {
    air_drift_back(waves, &air_clocks(waves, time), p, 0.0)
}

/// Each wave's phase at `time`, wrapped in f64 once a frame: `air_drift`
/// took the f64 remainder six times for each of four reads a mote.
fn air_clocks(waves: &[AirWave; 6], time: f64) -> [f32; 6] {
    std::array::from_fn(|i| (f64::from(waves[i].w) * time).rem_euclid(std::f64::consts::TAU) as f32)
}

/// `air_drift` at the clocks' time less `back` seconds (a mote's age, a
/// motion step: seconds, small enough for f32).
fn air_drift_back(waves: &[AirWave; 6], clocks: &[f32; 6], p: Vec3, back: f32) -> Vec3 {
    waves.iter().zip(clocks).fold(Vec3::ZERO, |sum, (a, clock)| sum + a.across * (a.reach * (a.k.dot(p) + (clock - a.w * back) + a.phase).sin()))
}

fn air_reach(waves: &[AirWave; 6]) -> f32 {
    2.0 * waves.iter().map(|a| a.reach).sum::<f32>()
}

fn mote_glint(age: f32, rate: f32, seed: u64) -> f32 {

    let s = smoothstep(-0.35, 0.35, noise1(age * rate, seed));
    (GLINT_FLOOR + (1.0 - GLINT_FLOOR) * s * s * s) / GLINT_MEAN
}

pub const FIRE_RADIANCE: f32 = 40.0;
pub const EMBER_RADIANCE: f32 = 60.0;

pub const FLAME_COVER: f32 = 0.8;

pub const FLAME_EROSION: f32 = 0.15;

const LIFT_FROM: f32 = 0.02;
const LIFT_TO: f32 = 0.3;

const BURN_OUT: f32 = 0.45;

pub const BLUE_ROOT: f32 = 0.0;

pub const FIRE_LEVEL: f32 = 4.0;

const FLAME_STRETCH: f32 = 1.9;

const FLAME_BASE: f32 = 0.04;
const FLAME_W0: f32 = 0.35;
const FLAME_W1: f32 = 2.4;

pub fn flame_colour(heat: f32) -> Vec3 {
    let t = heat.clamp(0.0, 1.0);
    let bright = 0.03 + 0.97 * t.powf(1.4);
    Vec3::new(1.0, 0.02 + 0.5 * t * t, 0.02 * t + 0.08 * t * t * t * t) * bright
}

fn flame_v(v: f32, dt: f32) -> f32 {
    let h = (v - FLAME_BASE) / (1.0 - FLAME_BASE);
    if h <= 0.0 {
        return v;
    }
    let c = FLAME_W0 / FLAME_W1;
    let moved = ((h + c) * (-FLAME_W1 * dt).exp() - c).max(0.0);
    FLAME_BASE + moved * (1.0 - FLAME_BASE)
}

const FLAME_PACE: f32 = 1.3;

const FIRE_BED: f32 = 0.15;

const PUFF_K: f32 = 1.5;

const PUFF_SWELL: f32 = 0.25;
const PUFF_PINCH: f32 = 0.55;

const PUFF_WANDER: f32 = 0.3;

fn puff_book_hz() -> f32 {
    PUFF_K / (2.0 * FIRE_BED).sqrt() / FLAME_PACE
}

fn fire_tempo(e: &EffectEmitter) -> f32 {
    e.scale.sqrt() / e.variation.clamped().flicker_rate
}

fn fire_puff_hz(e: &EffectEmitter) -> f32 {
    puff_book_hz() * FLAME_PACE / fire_tempo(e)
}

fn fire_phase(e: &EffectEmitter, time: f64) -> f32 {
    let f = f64::from(fire_puff_hz(e));
    let wander = PUFF_WANDER * noise1d(time * f * 0.25, emitter_seed(e) ^ 0x7f4a_7c15_9e37_79b9);
    ((time * f).rem_euclid(1.0) as f32 + wander).rem_euclid(1.0)
}

fn fire_spot(birth: f64, ju: f32, jv: f32) -> (f32, f32) {
    let x = (0.5 + birth * 0.754_877_666_246_692_8).rem_euclid(1.0) as f32;
    let y = (0.5 + birth * 0.569_840_290_998_053_3).rem_euclid(1.0) as f32;
    let x = (x + 0.1 * (ju - 0.5)).rem_euclid(1.0);
    let y = (y + 0.1 * (jv - 0.5)).rem_euclid(1.0);
    (x.sqrt(), y)
}

struct FlameSheet {

    base: Vec3,

    size: f32,
    stretch: f32,

    frame: f32,

    risen: f32,
    stands: f32,
    rotation: f32,
    mirrored: bool,

    heat: f32,

    book: f32,
}

fn fire_sheets(e: &EffectEmitter, time: f64) -> Vec<FlameSheet> {
    let p = preset(EffectKind::Fire);
    let v = e.variation.clamped();
    let rate = e.rate.max(0.0) * p.rate;
    if e.kind != EffectKind::Fire || !(rate > 0.0) || !(e.scale > 0.0) {
        return Vec::new();
    }
    let tempo = fire_tempo(e);
    let (l0, l1) = life_range(&p, &v);
    let (l0, l1) = (l0 * tempo, l1 * tempo);
    let slots = ((rate * l1).ceil() as usize).clamp(1, MAX_SLOTS);
    let period = (slots as f64 / f64::from(rate)).max(f64::from(l1));
    let base = emitter_seed(e);
    let up = e.direction.try_normalize().unwrap_or(Vec3::Y);
    let side = up.any_orthonormal_vector();
    let wide = e.scale.powf(-0.25);

    let per = FIRE_FRAMES as f32 / FLAME_SECONDS / puff_book_hz();
    let last = (FIRE_FRAMES - 1) as f32;
    let mut out = Vec::with_capacity(slots);
    for i in 0..slots {
        let since = time - i as f64 / f64::from(rate);
        let n = (since / period).floor();
        let age = (since - n * period) as f32;
        let u = uniforms(hash64(base ^ (i as u64).wrapping_mul(0x632be59bd9b4e019) ^ (n as i64 as u64).wrapping_mul(0x8cb92ba72f3d8dd7)));
        let life = lerp(l0, l1, u[0]);
        if age >= life {
            continue;
        }
        let t = age / life;

        let (r2, a2) = fire_spot(n * slots as f64 + i as f64, u[4], u[5]);
        let r = p.bed * e.scale * r2;
        let phi = a2 * std::f32::consts::TAU;
        let spot = e.position + (side * phi.cos() + up.cross(side) * phi.sin()) * r;
        let size = lerp(p.size.0, p.size.1, smoothstep(0.0, 1.0, t)) * e.scale * size_spread(u[8], &v) * wide * (1.15 - 0.45 * r2);
        let tall = if v.size_variation == 1.0 { 0.85 + 0.35 * u[7] } else { (1.025 + (u[7] - 0.5) * 0.35 * v.size_variation).max(0.2) };
        let spin_dir = if u[10] < 0.5 { -1.0 } else { 1.0 };
        let pace_of = if v.speed_variation == 1.0 { 0.85 + 0.3 * u[1] } else { (1.0 + (u[1] - 0.5) * 0.3 * v.speed_variation).max(0.1) };
        let pace = FIRE_FRAMES as f32 / FLAME_SECONDS * FLAME_PACE * pace_of / tempo;
        let range = (last - life * pace).max(0.0);
        let phase = fire_phase(e, time - f64::from(age) + 0.5 * f64::from(life));
        let want = (phase * per - 0.5 * life * pace).rem_euclid(per);
        let mut start = want + ((u[6] * range - want) / per).round() * per;
        if start > range {
            start -= per;
        }
        if start < 0.0 {
            start += per;
        }
        let frame = start.clamp(0.0, range) + age * pace;
        out.push(FlameSheet {
            base: spot,
            size,
            stretch: FLAME_STRETCH * tall / wide,
            frame: frame.min(last),
            book: if frame < last { pace * FLAME_SECONDS / FIRE_FRAMES as f32 } else { 0.0 },
            risen: smoothstep(0.0, 0.25, t),
            stands: 1.0 - smoothstep(0.6, 1.0, t),
            rotation: (2.0 * u[9] - 1.0) * (p.spin.1 * v.turbulence) + p.spin.0 * age * spin_dir,
            mirrored: u[3] < 0.5,
            heat: u[11],
        });
    }
    out
}

const FLAME_AREA: [f32; FIRE_FRAMES as usize] = [
    0.9271, 0.8415, 0.8425, 0.8990, 0.9877, 1.0703, 1.1098, 1.0776, 1.0018, 0.9146, 0.8234, 0.8473, 0.9395, 1.0476, 1.1368, 1.1838,
    1.1488, 1.1276, 1.0433, 0.9555, 0.8993, 0.8743, 0.8739, 0.9395, 1.0028, 1.0094, 1.0443, 1.1089, 1.0977, 1.0181, 0.9195, 0.9046,
    0.9783, 1.0576, 1.1141, 1.1200, 1.0657, 1.0794, 1.0408, 0.9906, 0.9489, 0.9382, 1.0067, 1.0420, 1.0302, 1.0040, 0.9796, 0.9859,
];

const EROSION_COVER: [f32; 9] = [0.0014, 0.0114, 0.0565, 0.1624, 0.3356, 0.5441, 0.7620, 0.9568, 1.0];

const FLAME_AREA_MEAN: f32 = 0.182531;

fn flame_area(frame: f32) -> f32 {
    let f = frame.clamp(0.0, (FIRE_FRAMES - 1) as f32);
    let i = (f.floor() as usize).min(FIRE_FRAMES as usize - 2);
    lerp(FLAME_AREA[i], FLAME_AREA[i + 1], f - i as f32)
}

fn erosion_cover(stands: f32) -> f32 {
    let f = stands.clamp(0.0, 1.0) * 8.0;
    let i = (f.floor() as usize).min(7);
    lerp(EROSION_COVER[i], EROSION_COVER[i + 1], f - i as f32)
}

const RISE_COVER: [f32; 9] = [0.0, 0.0033, 0.1533, 0.4406, 0.6905, 0.8668, 0.9673, 0.9997, 1.0];

fn rise_cover(risen: f32) -> f32 {
    let f = risen.clamp(0.0, 1.0) * 8.0;
    let i = (f.floor() as usize).min(7);
    lerp(RISE_COVER[i], RISE_COVER[i + 1], f - i as f32)
}

const RISE_FROM: f32 = -0.3;
const RISE_TO: f32 = 1.15;
const RISE_SOFT: f32 = 0.12;
const RISE_LEAD: f32 = 0.25;

#[cfg(test)]
fn flame_tables() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let frames: Vec<Vec<u8>> = (0..FIRE_FRAMES).map(fire_frame).collect();
    let cover: Vec<f32> = frames.iter().map(|f| f.chunks(4).map(|t| f32::from(t[3]) / 255.0).sum::<f32>()).collect();
    let mean = cover.iter().sum::<f32>() / cover.len() as f32;
    let area = cover.iter().map(|c| c / mean).collect();
    let n = ATLAS_SIZE as usize;
    let erosion = (0..9)
        .map(|k| {
            let gone = 1.0 - k as f32 / 8.0;
            let (mut stands, mut all) = (0.0f32, 0.0f32);
            for f in &frames {
                for (i, t) in f.chunks(4).enumerate() {
                    let a = f32::from(t[3]) / 255.0;
                    let v = 1.0 - ((i / n) as f32 + 0.5) / n as f32;
                    stands += a * smoothstep(gone - FLAME_EROSION, gone, f32::from(t[1]) / 255.0 * smoothstep(LIFT_FROM, LIFT_TO, v));
                    all += a;
                }
            }
            stands / all
        })
        .collect();
    let rise = (0..9)
        .map(|k| {
            let front = lerp(RISE_FROM, RISE_TO, k as f32 / 8.0);
            let (mut stands, mut all) = (0.0f32, 0.0f32);
            for f in &frames {
                for (i, t) in f.chunks(4).enumerate() {
                    let a = f32::from(t[3]) / 255.0;
                    let v = 1.0 - ((i / n) as f32 + 0.5) / n as f32;
                    stands += a * (1.0 - smoothstep(front - RISE_SOFT, front, v - RISE_LEAD * f32::from(t[1]) / 255.0));
                    all += a;
                }
            }
            stands / all
        })
        .collect();
    (area, erosion, rise)
}

const HANDOVER_SHARE: f32 = 0.35;

pub fn fire_glow(e: &EffectEmitter, time: f64) -> f32 {
    let sheets = fire_sheets(e, time);
    let wide = e.scale.powf(-0.25);
    let unit = e.scale * wide;
    let (mut held, mut puffed) = (0.0f32, 0.0f32);
    for s in &sheets {
        let w = s.size / unit;
        let weight = w * w * (s.stretch * wide) * erosion_cover(s.stands) * rise_cover(s.risen);
        held += weight;
        puffed += weight * flame_area(s.frame);
    }
    if !(held > 0.0) {
        return 0.0;
    }
    (puffed / held) * (held / (FLAME_AREA_MEAN * e.rate * fire_tempo(e))).powf(HANDOVER_SHARE)
}

const COALS_HALF: f32 = 0.24;

const COALS_GLOW: f32 = 0.6;

fn coals_instance(e: &EffectEmitter, time: f64, at: &Surroundings, own: &dyn Fn(&Light) -> f32, breath: f32) -> Option<EffectInstance> {
    if !(e.rate > 0.0) || !(e.scale > 0.0) {
        return None;
    }
    let v = e.variation.clamped();
    let up = e.direction.try_normalize().unwrap_or(Vec3::Y);
    let centre = at.yaw_inv * (e.position + up * 0.005 - at.offset);
    let normal = at.yaw_inv * up;
    let mut light = (at.ambient)(e.position);
    for (i, l) in at.lights.iter().enumerate() {
        if (at.reaches)(e, i) {
            if let Some((c, toward)) = lamp_at(l, centre) {
                light += c * (own(l) * toward.dot(normal).max(0.0));
            }
        }
    }
    let tint = Vec3::from(e.tint);
    let lit = light * tint * at.exposure;
    let hot = tint * (FIRE_RADIANCE * COALS_GLOW * at.exposure * fire_adaptation(e, at) * v.intensity * breath);
    let spin = uniforms(emitter_seed(e))[0] * std::f32::consts::TAU;
    Some(EffectInstance {
        centre: [centre.x, centre.y, centre.z, COALS_HALF * e.scale],
        colour: [hot.x, hot.y, hot.z, e.rate.min(1.0)],
        light: [lit.x, lit.y, lit.z, 2.0],

        light_dir: [0.0, 1.0, 0.0, v.flicker - 1.0],
        axis: [normal.x, normal.y, normal.z, 0.0],
        params: [spin, COALS_LAYER as f32, ((time * f64::from(v.flicker_rate)) % 3600.0) as f32, 3.0],

        motion: [0.0; 4],
    })
}

fn coals_breath(bed: &EffectEmitter, emitters: &[EffectEmitter], time: f64) -> f32 {
    let Some(fire) = emitters.iter().find(|f| {
        f.kind == EffectKind::Fire && f.position.distance(bed.position) < 0.5 * COALS_HALF * bed.scale.max(f.scale) + 1e-3
    }) else {
        return 1.0;
    };
    let glow = fire_glow(fire, time);
    if glow <= 0.0 {
        return 1.0;
    }
    1.0 + 0.25 * bed.variation.clamped().flicker * (glow.min(2.5) - 1.0)
}

fn fire_adaptation(e: &EffectEmitter, at: &Surroundings) -> f32 {
    let distance = (at.yaw_inv * (e.position - at.offset) - at.head).length();
    let adapted = super::tonemap::bulb_adaptation(0.15 * e.scale, distance, 1.0);
    super::tonemap::own_light_scale(at.exposure, FIRE_RADIANCE, FIRE_LEVEL, adapted)
}

pub fn simulate(emitters: &[EffectEmitter], time: f64, at: &Surroundings) -> EffectFrame {
    simulate_with(emitters, &[], time, at)
}

pub fn simulate_with(emitters: &[EffectEmitter], splashes: &[Splash], time: f64, at: &Surroundings) -> EffectFrame {
    simulate_seen(emitters, &|_| true, splashes, time, at)
}

pub fn simulate_seen(emitters: &[EffectEmitter], seen: &dyn Fn(usize) -> bool, splashes: &[Splash], time: f64, at: &Surroundings) -> EffectFrame {
    simulate_all(emitters, seen, splashes, None, time, at)
}

pub fn simulate_all(
    emitters: &[EffectEmitter],
    seen: &dyn Fn(usize) -> bool,
    splashes: &[Splash],
    bursts: Option<&stack::BurstFrame>,
    time: f64,
    at: &Surroundings,
) -> EffectFrame {
    simulate_governed(emitters, seen, splashes, bursts, None, time, at)
}

pub struct Shares<'a> {
    pub emitter: &'a dyn Fn(&EffectEmitter) -> f32,
    pub splash: &'a dyn Fn(&Splash) -> f32,

    pub sight: &'a [Vec<seen::Eye>],
}

pub fn emitter_key(e: &EffectEmitter) -> u64 {
    hash64(id_hash(&e.id) ^ 0x51ac_ed00_e1u64)
}

pub fn splash_key(s: &Splash) -> u64 {
    hash64(s.seed ^ 0x5b1a_54u64)
}

const CUTOUT_MEAN: f32 = 0.6;

pub fn instance_fill(i: &EffectInstance, head: Vec3) -> f32 {
    use super::effect_asset::{REFERENCE_EYES, REFERENCE_FOCAL_PIXELS};
    let d = (Vec3::from_slice(&i.centre[..3]) - head).length().max(0.05);
    let across = (2.0 * i.centre[3] / d * REFERENCE_FOCAL_PIXELS).max(1.5);
    let along = (i.light_dir[3] / d * REFERENCE_FOCAL_PIXELS).max(across);
    across * along * CUTOUT_MEAN * REFERENCE_EYES / 1e6
}

struct KindCost {

    area_each: f32,
    flat: bool,
    importance: f32,
    points: bool,
    instances: f32,
}

fn kind_cost(kind: EffectKind) -> &'static KindCost {
    static COSTS: std::sync::OnceLock<Vec<KindCost>> = std::sync::OnceLock::new();
    let all = COSTS.get_or_init(|| {
        EffectKind::ALL
            .iter()
            .map(|k| {
                let a = templates::asset(*k);
                let r = a.check(books()).unwrap_or_else(|e| panic!("the built-in {} template: {}", k.template_name(), e.iter().map(std::string::ToString::to_string).collect::<Vec<_>>().join("; ")));
                let points = *k == EffectKind::Dust;
                let area = r.area_facing + r.area_flat;
                let area_each = if points || r.instances_steady == 0 { 0.0 } else { area / r.instances_steady as f32 };
                KindCost { area_each, flat: r.area_flat > r.area_facing, importance: a.scalability.importance, points, instances: r.instances_steady as f32 }
            })
            .collect()
    });
    &all[EffectKind::ALL.iter().position(|k| *k == kind).expect("every kind is listed")]
}

pub fn governor_wants(
    emitters: &[EffectEmitter],
    seen: &dyn Fn(usize) -> bool,
    splashes: &[Splash],
    measured: &std::collections::HashMap<u64, (f32, f32)>,
    head_world: Vec3,
    sight: &[Vec<seen::Eye>],
) -> Vec<stack::Want> {
    use super::effect_asset::REFERENCE_FOCAL_PIXELS;
    let mut out = Vec::new();
    for (index, e) in emitters.iter().enumerate() {
        if !seen(index) || !(e.scale > 0.0) || !(e.rate > 0.0) {
            continue;
        }
        let cost = kind_cost(e.kind);
        let tpl = templates::template(e.kind);
        let (count, sheddable) = match tpl.form {
            templates::Form::Slots(s) => {
                let p = &tpl.preset;
                let v = e.variation.clamped();
                let rate = e.rate * if s.density { p.rate * e.volume() / f32::midpoint(p.life.0, p.life.1) } else { p.rate };
                let slots = ((rate * life_range(p, &v).1).ceil()).clamp(1.0, MAX_SLOTS as f32);

                let eyes = sight.get(index);
                let margin = mote_margin(e);
                let within = if s.boxed && s.held {

                    let ext = e.extent.map(|v| v.length().max(1e-4));
                    let rel = head_world - e.position;
                    let span: [(f32, f32); 3] = std::array::from_fn(|k| {
                        let u = rel.dot(e.extent[k]) / (ext[k] * ext[k]);
                        let r = DUST_SEEN_WITHIN / ext[k];
                        ((u - r).max(-1.0), (u + r).min(1.0))
                    });
                    if span.iter().any(|(a, b)| a >= b) {
                        0.0
                    } else {
                        let at = |k: usize, i: usize| span[k].0 + (span[k].1 - span[k].0) * (i as f32 + 0.5) / 5.0;
                        let near = (0..125)
                            .filter(|n| {
                                let p0 = e.position + e.extent[0] * at(0, n % 5) + e.extent[1] * at(1, n / 5 % 5) + e.extent[2] * at(2, n / 25);
                                p0.distance_squared(head_world) <= DUST_SEEN_WITHIN * DUST_SEEN_WITHIN
                                    && eyes.is_none_or(|eyes| eyes.iter().any(|v| v.holds(p0, margin)))
                            })
                            .count();
                        let part: f32 = span.iter().map(|(a, b)| (b - a) / 2.0).product();
                        near as f32 / 125.0 * part
                    }
                } else {
                    1.0
                };
                ((slots * within).ceil(), true)
            }
            _ => (cost.instances, false),
        };
        let (lo, hi) = emitter_bounds(e);
        let d = (0.5 * (lo + hi)).distance(head_world).max(0.3);
        let pixels = (hi - lo).length() / d * REFERENCE_FOCAL_PIXELS;

        let looking = [seen::Eye::looking(head_world, 0.5 * (lo + hi)); 2];
        let eyes: &[seen::Eye] = match sight.get(index) {
            Some(v) => v,
            None => &looking,
        };
        let area = if cost.points { 0.0 } else { count * cost.area_each * e.scale * e.scale };
        let (facing, flat) = if cost.flat { (0.0, area) } else { (area, 0.0) };
        let fill = seen::seen_fill(eyes, lo, hi, facing, flat, if cost.points { count } else { 0.0 }, 0.0);
        out.push(stack::Want {
            key: emitter_key(e),
            score: cost.importance * 100.0 + pixels.max(1.0).log2(),
            fill,
            count,
            sheddable,
            kind: stack::WantKind::Emitter,
        });
    }
    for s in splashes {
        let (lo, hi) = splash_bounds(s);
        let d = (0.5 * (lo + hi)).distance(head_world).max(0.3);
        let pixels = (hi - lo).length() / d * REFERENCE_FOCAL_PIXELS;
        let (fill, count) = measured.get(&splash_key(s)).copied().unwrap_or((0.0, 0.0));
        out.push(stack::Want {
            key: splash_key(s),
            score: templates::water_entry_asset().scalability.importance * 100.0 + pixels.max(1.0).log2(),
            fill,
            count,
            sheddable: true,
            kind: stack::WantKind::Splash,
        });
    }
    out
}

pub fn simulate_governed(
    emitters: &[EffectEmitter],
    seen: &dyn Fn(usize) -> bool,
    splashes: &[Splash],
    bursts: Option<&stack::BurstFrame>,
    shares: Option<&Shares>,
    time: f64,
    at: &Surroundings,
) -> EffectFrame {
    let mut emitter_shed = 0u32;
    let mut splash_shed = 0u32;
    let mut splash_costs: Vec<(u64, f32, f32)> = Vec::new();
    let mut over: Vec<(f32, EffectInstance)> = Vec::new();
    let mut screened: Vec<EffectInstance> = Vec::new();
    let player = |p: Vec3| at.yaw_inv * (p - at.offset);
    let head_world = at.yaw_inv.inverse() * at.head + at.offset;

    let fires: Vec<(Vec3, f32, f32)> = emitters
        .iter()
        .filter(|e| fire_burns(e))
        .map(|e| (player(fire_light_rest(e)), FIRE_LIGHT_REACH * e.scale, fire_adaptation(e, at)))
        .collect();
    let own = |l: &Light| {
        fires
            .iter()
            .find(|f| !l.in_level_bake && f.0.distance_squared(l.position) < f.1 * f.1)
            .map_or(1.0, |f| f.2)
    };
    for (index, e) in emitters.iter().enumerate() {
        if !seen(index) {
            continue;
        }
        let at = &at.at_point(e.position);

        let tpl = templates::template(e.kind);
        let s = match tpl.form {
            templates::Form::Bed => {

                let breath = coals_breath(e, emitters, time);
                over.extend(coals_instance(e, time, at, &own, breath).map(|i| (f32::INFINITY, i)));
                continue;
            }
            templates::Form::Sheets => {
                fire_instances(e, time, at, &mut over);
                continue;
            }
            templates::Form::WaterEntry => continue,
            templates::Form::Slots(s) => s,
        };
        if s.pops {
            ember_pops(e, time, at, &mut screened);
        }
        let p = &tpl.preset;
        let v = e.variation.clamped();
        let rate = e.rate.max(0.0) * if s.density { p.rate * e.volume() / f32::midpoint(p.life.0, p.life.1) } else { p.rate };
        if !(rate > 0.0) || !(e.scale > 0.0) {
            continue;
        }
        let lives = life_range(p, &v);
        let speeds = speed_range(p, &v);
        let slots = ((rate * lives.1).ceil() as usize).clamp(1, MAX_SLOTS);

        let period = (slots as f64 / f64::from(rate)).max(f64::from(lives.1));
        let base = emitter_seed(e);

        let share = shares.map_or(1.0, |sh| (sh.emitter)(e));
        let sight = shares.and_then(|sh| sh.sight.get(index));
        let margin = mote_margin(e);
        let shed_phase = (hash64(base ^ 0x0de5_ead) >> 40) as f32 / (1u64 << 24) as f32;
        let tint = Vec3::from(e.tint);
        let ambient = (at.ambient)(e.position);
        let lamps: Vec<LampSeen> = (0..at.lights.len())
            .filter(|&i| (at.reaches)(e, i))
            .filter_map(|i| lamp_seen(&at.lights[i], own(&at.lights[i])))
            .collect();
        let glint_seed = id_hash("glint");

        let glow = if s.own_light { at.exposure * fire_adaptation(e, at) } else { at.exposure };
        let up = e.direction.try_normalize().unwrap_or(Vec3::Y);
        let air = s.air.then(|| air_waves(base));
        let clocks = air.as_ref().map(|w| air_clocks(w, time));
        for i in 0..slots {
            let since = time - i as f64 / f64::from(rate);
            let n = (since / period).floor();
            let age = (since - n * period) as f32;
            let u = uniforms(hash64(
                base ^ (i as u64).wrapping_mul(0x632be59bd9b4e019) ^ (n as i64 as u64).wrapping_mul(0x8cb92ba72f3d8dd7),
            ));
            let life = lerp(lives.0, lives.1, u[0]);
            if age >= life {
                continue;
            }
            let t = age / life;
            let speed = lerp(speeds.0, speeds.1, u[1]) * e.scale.sqrt();
            let p0 = if s.boxed {
                let p0 = e.position + e.extent[0] * (2.0 * u[4] - 1.0) + e.extent[1] * (2.0 * u[5] - 1.0) + e.extent[2] * (2.0 * u[6] - 1.0);

                if s.held && p0.distance_squared(head_world) > DUST_SEEN_WITHIN * DUST_SEEN_WITHIN {
                    continue;
                }

                if s.held && !sight.is_none_or(|eyes| eyes.iter().any(|v| v.holds(p0, margin))) {
                    continue;
                }
                p0
            } else {

                let r = p.bed * e.scale * u[4].sqrt();
                let side = up.any_orthonormal_vector();
                let phi = u[5] * std::f32::consts::TAU;
                e.position + (side * phi.cos() + up.cross(side) * phi.sin()) * r
            };

            if share < 1.0 && !stack::kept(i as u32, share, shed_phase) {
                emitter_shed += 1;
                continue;
            }
            let dir = cone_direction(up, p.spread_deg, u[2], u[3]);
            let (mut pos, vel) = ballistic(p0, dir * speed, p.accel, p.drag, age);

            let mut moving = vel;
            if let (Some(air), Some(clocks)) = (&air, &clocks) {
                let now = air_drift_back(air, clocks, p0, 0.0);
                pos += (now - air_drift_back(air, clocks, p0, age)) * v.turbulence;
                let seed = base ^ (i as u64).wrapping_mul(0x632be59bd9b4e019) ^ (n as i64 as u64);
                let wander = |jitter: f32| Vec3::new(noise1(jitter, seed), 0.6 * noise1(jitter, seed ^ 1), noise1(jitter, seed ^ 2)) * (p.swirl.0 * v.turbulence);
                let jitter = age * p.swirl.1;
                pos += wander(jitter);
                moving += (now - air_drift_back(air, clocks, p0, MOTION_H)) * (v.turbulence / MOTION_H)
                    + (wander(jitter) - wander((age - MOTION_H) * p.swirl.1)) / MOTION_H;
            } else {
                let swirl_r = p.swirl.0 * v.turbulence * t * e.scale;
                let swirl_a = p.swirl.1 * age + u[7] * std::f32::consts::TAU;
                pos += Vec3::new(swirl_a.cos(), 0.0, swirl_a.sin()) * swirl_r;
                moving += swirl_motion(p.swirl.0 * v.turbulence * e.scale / life, swirl_r, p.swirl.1, swirl_a);
            }
            let mut size = if s.log_size {

                let k = if v.size_variation == 1.0 { u[11] * u[11] } else { 1.0 / 3.0 + (u[11] * u[11] - 1.0 / 3.0) * v.size_variation };
                p.size.0 * (p.size.1 / p.size.0).powf(k) * e.scale
            } else {
                lerp(p.size.0, p.size.1, smoothstep(0.0, 1.0, t)) * e.scale * size_spread(u[8], &v)
            };

            if let Some(top) = e.ceiling {
                match s.ceiling {
                    templates::Ceiling::Spread => {
                        let under = top - 0.6 * size;
                        let excess = pos.y - under;
                        if excess > 0.0 {
                            let a = u[7] * std::f32::consts::TAU + 0.3 * age;
                            pos.y = under - 0.15 * size * (1.0 - (-excess).exp());
                            pos += Vec3::new(a.cos(), 0.0, a.sin()) * (0.9 * excess);
                            size *= 1.0 + 0.5 * excess.min(2.0);

                            let rise = moving.y;
                            moving.y = 0.15 * size * (-excess).exp() * -rise;
                            moving += Vec3::new(a.cos(), 0.0, a.sin()) * (0.9 * rise);
                        }
                    }
                    templates::Ceiling::Stop => {
                        if pos.y >= top - 0.02 {
                            pos.y = top - 0.02;
                            moving.y = moving.y.min(0.0);
                        }
                    }
                    templates::Ceiling::Open => {}
                }
            }
            let spin_dir = if u[10] < 0.5 { -1.0 } else { 1.0 };
            let rotation = (2.0 * u[9] - 1.0) * p.spin.1 + p.spin.0 * age * spin_dir;
            let frame = p.frames.0 as f32 + t * (p.frames.1.saturating_sub(1)) as f32;
            let streak = if p.streak > 0.0 { vel.length() * p.streak } else { 0.0 };
            let centre = player(pos);
            let axis = at.yaw_inv * vel.try_normalize().unwrap_or(Vec3::Y);
            let to_eye = (at.head - centre).normalize_or_zero();
            let moving = at.yaw_inv * moving;
            let mut inst = EffectInstance {
                centre: [centre.x, centre.y, centre.z, size],
                colour: [0.0; 4],
                light: [0.0; 4],
                light_dir: [0.0, 1.0, 0.0, streak],
                axis: [axis.x, axis.y, axis.z, p.soft * e.scale.sqrt()],
                params: [rotation, frame, 0.0, 0.0],
                motion: [moving.x, moving.y, moving.z, 0.0],
            };
            match s.shade {
                templates::SlotShade::Ember => {
                    inst.colour = ember_colour(&v, tint, glow, age, t, u[11]);
                }
                templates::SlotShade::Scatter { ambient: seen } => {

                    let mut scattered = ambient * seen;
                    for l in &lamps {
                        if let Some((c, d)) = lamp_seen_at(l, centre) {
                            scattered += c * (l.own * henyey_greenstein(p.phase_g, -d, to_eye));
                        }
                    }

                    let glint = if s.held { mote_glint(age, lerp(0.35, 1.3, u[7]), glint_seed ^ u64::from(u[8].to_bits())) } else { 1.0 };
                    let c = scattered * p.albedo * tint * (at.exposure * glint * v.intensity);
                    let held = smoothstep(0.0, 0.2, t) * smoothstep(1.0, 0.7, t);
                    let d = (centre - at.head).length();
                    let focus = if s.held { smoothstep(DUST_NEAREST.0, DUST_NEAREST.1, d) * smoothstep(DUST_SEEN_WITHIN, 0.75 * DUST_SEEN_WITHIN, d) } else { 1.0 };
                    inst.colour = [c.x, c.y, c.z, held * focus];
                    inst.params[3] = -1.0;
                }
                templates::SlotShade::Sixway { peak, rise, power } => {
                    let mut strongest: Option<(f32, Vec3, Vec3)> = None;
                    let mut rest = ambient;
                    for l in &lamps {
                        let Some((c, d)) = lamp_seen_at(l, centre) else { continue };
                        let c = c * l.own;
                        let y = luminance(c);
                        match strongest {
                            Some((sy, _, _)) if sy >= y => rest += c,
                            Some((_, sc, _)) => {
                                rest += sc;
                                strongest = Some((y, c, d));
                            }
                            None => strongest = Some((y, c, d)),
                        }
                    }
                    let a = Vec3::splat(p.albedo) * tint * at.exposure;
                    let base = rest * a;
                    let opacity = peak * smoothstep(0.0, rise, t) * (1.0 - t).powf(power) * v.intensity;
                    inst.colour = [base.x, base.y, base.z, opacity];
                    if let Some((_, c, d)) = strongest {
                        let lit = c * a * henyey_greenstein(p.phase_g, -d, to_eye);
                        inst.light = [lit.x, lit.y, lit.z, 0.0];
                        inst.light_dir = [d.x, d.y, d.z, streak];
                    }
                    inst.params[3] = 1.0;
                }
            }
            if inst.colour[3] <= 0.0 {
                continue;
            }
            if s.screened {
                screened.push(inst);
            } else {
                over.push(((centre - at.head).length_squared(), inst));
            }
        }
    }
    for s in splashes {
        let (o0, s0) = (over.len(), screened.len());
        splash(s, time, &at.at_point(s.position), &mut over, &mut screened);

        let key = splash_key(s);
        let fill: f32 = over[o0..].iter().map(|(_, i)| i).chain(screened[s0..].iter()).map(|i| instance_fill(i, at.head)).sum();
        splash_costs.push((key, fill, (over.len() - o0 + screened.len() - s0) as f32));
        let share = shares.map_or(1.0, |sh| (sh.splash)(s));
        if share < 1.0 {
            let phase = uniforms(key)[0];
            let mut j = 0u32;
            let mut keep = |_: &EffectInstance| {
                j += 1;
                let kept = stack::kept(j - 1, share, phase);
                splash_shed += u32::from(!kept);
                kept
            };
            let tail: Vec<(f32, EffectInstance)> = over.drain(o0..).collect();
            over.extend(tail.into_iter().filter(|(_, i)| keep(i)));
            let tail: Vec<EffectInstance> = screened.drain(s0..).collect();
            screened.extend(tail.into_iter().filter(|i| keep(i)));
        }
    }
    let mut burst_count = bursts.map_or_else(Default::default, |b| stack::simulate_bursts(b, time, at, &own, &mut over, &mut screened));
    burst_count.emitter_shed = emitter_shed;
    burst_count.splash_shed = splash_shed;

    let hot = |i: &EffectInstance| i.params[3] == stack::SHADE_SCREEN_HEAT;
    let screened_heat = screened.iter().filter(|i| hot(i)).count() as u32;
    if screened_heat > 0 {
        screened.sort_by_key(|i| hot(i));
    }

    over.sort_by(|a, b| b.0.total_cmp(&a.0));
    let mut frame = EffectFrame {
        instances: Vec::with_capacity(over.len() + screened.len()),
        over: over.len() as u32,
        screened: screened.len() as u32,
        screened_heat,
        bursts: burst_count,
        splash_costs,
    };
    frame.instances.extend(over.into_iter().map(|(_, i)| i));
    frame.instances.extend(screened);
    frame
}

pub fn emitter_bounds(e: &EffectEmitter) -> (Vec3, Vec3) {
    let s = e.scale.max(0.0);
    let root = s.sqrt();
    let p = e.position;
    let (lo, hi) = match e.kind {
        EffectKind::Fire => (p - Vec3::new(0.45 * s, 0.15 * s, 0.45 * s), p + Vec3::new(0.45 * s, 1.4 * s, 0.45 * s)),
        EffectKind::Coals => (p - Vec3::splat(0.36 * s), p + Vec3::splat(0.36 * s)),
        EffectKind::Smoke => {
            let wide = 0.8 * s + 0.6 + if e.ceiling.is_some() { 2.5 } else { 0.0 };
            (p - Vec3::new(wide, 0.6 * s, wide), p + Vec3::new(wide, 4.5 * s.max(root) + 0.6 * s, wide))
        }
        EffectKind::Embers => (p - Vec3::new(0.6 * s + 0.4, 0.3, 0.6 * s + 0.4), p + Vec3::new(0.6 * s + 0.4, 3.5 * root, 0.6 * s + 0.4)),
        EffectKind::Dust => {
            let half = e.extent[0].abs() + e.extent[1].abs() + e.extent[2].abs() + Vec3::splat(0.25 * e.variation.clamped().turbulence + 0.05);
            (p - half, p + half)
        }
    };
    let hi = match (e.ceiling, e.kind) {
        (Some(top), EffectKind::Smoke | EffectKind::Embers) => hi.with_y(hi.y.min(top + 0.05).max(lo.y)),
        _ => hi,
    };
    (lo - Vec3::splat(EMITTER_MARGIN), hi + Vec3::splat(EMITTER_MARGIN))
}

pub const EMITTER_MARGIN: f32 = 0.3;

pub const EMITTER_HOLD: f64 = 0.5;

pub fn emitter_seen(
    e: &EffectEmitter,
    views: &[[glam::Vec4; 6]],
    head: Vec3,
    rooms: &[super::portal_cull::CullRoom],
    portals: &[super::uniforms::ProbePortal],
) -> bool {
    !emitter_sight(e, views, head, rooms, portals).is_empty()
}

pub fn emitter_sight(
    e: &EffectEmitter,
    views: &[[glam::Vec4; 6]],
    head: Vec3,
    rooms: &[super::portal_cull::CullRoom],
    portals: &[super::uniforms::ProbePortal],
) -> Vec<seen::Eye> {
    let (lo, hi) = emitter_bounds(e);
    seen::sight(views, head, lo, hi, e.position, rooms, portals)
}

fn mote_margin(e: &EffectEmitter) -> f32 {
    0.25 * e.variation.clamped().turbulence + 0.05
}

fn fire_instances(e: &EffectEmitter, time: f64, at: &Surroundings, over: &mut Vec<(f32, EffectInstance)>) {
    let sheets = fire_sheets(e, time);
    if sheets.is_empty() {
        return;
    }
    let p = preset(EffectKind::Fire);
    let v = e.variation.clamped();
    let glow = at.exposure * fire_adaptation(e, at);
    let tint = Vec3::from(e.tint);
    let up = e.direction.try_normalize().unwrap_or(Vec3::Y);
    let axis = at.yaw_inv * Vec3::Y;
    for s in sheets {
        if s.stands <= 0.0 || s.risen <= 0.0 {
            continue;
        }
        let centre = at.yaw_inv * (s.base + up * (s.size * s.stretch * 0.92) - at.offset);
        let c = tint * (FIRE_RADIANCE * glow * (0.88 + 0.12 * s.heat) * v.intensity) * temperature(s.heat, &v);
        over.push((
            (centre - at.head).length_squared(),
            EffectInstance {
                centre: [centre.x, centre.y, centre.z, s.size],
                colour: [c.x, c.y, c.z, s.stands],
                light: [s.risen, 0.0, 0.0, 1.0],
                light_dir: [0.0, 1.0, 0.0, 0.0],
                axis: [axis.x, axis.y, axis.z, p.soft * e.scale.sqrt()],
                params: [s.rotation, FIRE_FIRST as f32 + s.frame, if s.mirrored { -s.stretch } else { s.stretch }, 2.0],

                motion: [0.0, 0.0, 0.0, s.book],
            },
        ));
    }
}

fn ember_colour(v: &Variation, tint: Vec3, glow: f32, age: f32, t: f32, heat: f32) -> [f32; 4] {
    let flicker = 0.7 + 0.3 * (v.flicker * (age * 23.0 * v.flicker_rate + heat * 40.0).sin() + (1.0 - v.flicker));
    let c = Vec3::new(1.0, 0.45, 0.12) * tint * (EMBER_RADIANCE * glow * flicker * v.intensity) * temperature(heat, v);
    [c.x, c.y, c.z, (1.0 - t * t).max(0.0)]
}

const EMBER_POPS: f32 = 0.3;

const EMBER_POP_BIN: f64 = 0.5;

const EMBER_PLUME: f32 = 0.9;

fn ember_pops(e: &EffectEmitter, time: f64, at: &Surroundings, screened: &mut Vec<EffectInstance>) {
    let p = preset(EffectKind::Embers);
    let v = e.variation.clamped();
    let chance = (EMBER_POPS * EMBER_POP_BIN as f32 * e.rate.max(0.0)).min(0.9);
    if !(chance > 0.0) || !(e.scale > 0.0) {
        return;
    }
    let lives = life_range(&p, &v);
    let base = emitter_seed(e) ^ 0xb5ad_4ece_da1c_e2a9;
    let up = e.direction.try_normalize().unwrap_or(Vec3::Y);
    let side = up.any_orthonormal_vector();
    let glow = at.exposure * fire_adaptation(e, at);
    let tint = Vec3::from(e.tint);
    let root = e.scale.sqrt();
    let accel = up * (p.drag * EMBER_PLUME * root) + p.accel;
    let first = ((time - f64::from(lives.1)) / EMBER_POP_BIN).floor() as i64;
    let last = (time / EMBER_POP_BIN).floor() as i64;
    for bin in first..=last {
        let pop = hash64(base ^ (bin as u64).wrapping_mul(0x8cb92ba72f3d8dd7));
        let b = uniforms(pop);
        if b[0] >= chance {
            continue;
        }
        let born = (bin as f64 + f64::from(b[1])) * EMBER_POP_BIN;
        let since = (time - born) as f32;
        if since < 0.0 {
            continue;
        }
        let r = p.bed * e.scale * b[2].sqrt();
        let phi = b[3] * std::f32::consts::TAU;
        let from = e.position + (side * phi.cos() + up.cross(side) * phi.sin()) * r;
        let count = 6 + (b[4] * 11.0) as usize;
        for j in 0..count {
            let u = uniforms(hash64(pop ^ (j as u64 + 1).wrapping_mul(0x632be59bd9b4e019)));
            let life = lerp(lives.0, lives.1, u[0]);

            let age = since - 0.04 * u[5];
            if age < 0.0 || age >= life {
                continue;
            }
            let t = age / life;
            let speed = lerp(1.8, 3.0, u[1]) * root;
            let dir = cone_direction(up, 15.0, u[2], u[3]);
            let (mut pos, vel) = ballistic(from, dir * speed, accel, p.drag, age);
            let swirl_r = p.swirl.0 * v.turbulence * t * e.scale;
            let swirl_a = p.swirl.1 * age + u[7] * std::f32::consts::TAU;
            pos += Vec3::new(swirl_a.cos(), 0.0, swirl_a.sin()) * swirl_r;
            let mut moving = vel + swirl_motion(p.swirl.0 * v.turbulence * e.scale / life, swirl_r, p.swirl.1, swirl_a);
            if let Some(top) = e.ceiling {
                if pos.y >= top - 0.02 {
                    pos.y = top - 0.02;
                    moving.y = moving.y.min(0.0);
                }
            }
            let moving = at.yaw_inv * moving;
            let size = lerp(p.size.0, p.size.1, smoothstep(0.0, 1.0, t)) * e.scale * size_spread(u[8], &v);
            let centre = at.yaw_inv * (pos - at.offset);
            let axis = at.yaw_inv * vel.try_normalize().unwrap_or(Vec3::Y);
            let colour = ember_colour(&v, tint, glow, age, t, u[11]);
            if colour[3] <= 0.0 {
                continue;
            }
            screened.push(EffectInstance {
                centre: [centre.x, centre.y, centre.z, size],
                colour,
                light: [0.0; 4],
                light_dir: [0.0, 1.0, 0.0, vel.length() * p.streak],
                axis: [axis.x, axis.y, axis.z, p.soft * root],
                params: [0.0, p.frames.0 as f32, 0.0, 0.0],
                motion: [moving.x, moving.y, moving.z, 0.0],
            });
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Splash {

    pub position: Vec3,

    pub born: f64,

    pub velocity: Vec3,

    pub size: f32,

    pub mass: f32,

    pub sunlit: f32,

    pub depth: f32,

    pub calm: f32,

    pub held: f32,

    pub flat: f32,
    pub seed: u64,

    /// MUD, not water: how wet the mud struck is, 0 (water, every splash
    /// before mud had them) to 1 (slurry). A mud splash throws clumps and
    /// spray and leaves a crater and spatter in the mud map
    /// (`mud_splash`, docs/engine/mud-and-deform.md).
    pub mud: f32,
    /// A BLAST in the mud, not a strike: the energy that went into the
    /// ground, joules (`Splash::mud_blast`). 0 for anything thrown or shot.
    pub blast: f32,
}

impl Splash {

    pub fn new(position: Vec3, born: f64, velocity: Vec3, size: f32, mass: f32) -> Self {
        let seed = hash64(born.to_bits() ^ hash64(u64::from(position.x.to_bits()) << 32 | u64::from(position.z.to_bits())));
        Self { position, born, velocity, size, mass, sunlit: 1.0, depth: 30.0, calm: 1.0, held: 0.0, flat: 0.0, seed, mud: 0.0, blast: 0.0 }
    }

    pub fn projectile(position: Vec3, born: f64, velocity: Vec3, diameter: f32, mass: f32) -> Self {
        Self::new(position, born, velocity, 0.5 * diameter, mass)
    }
}

pub const SPLASH_SECONDS: f32 = 3.0;

const DROP_ALBEDO: f32 = 1.0;
const DROP_PHASE_G: f32 = 0.55;

const DROP_STREAK: f32 = 0.012;

const WATER_DENSITY: f32 = 1000.0;
const GRAVITY: f32 = 9.81;
const SURFACE_TENSION: f32 = 0.072;
const ATMOSPHERE: f32 = 101_325.0;

const BODY_DRAG: f32 = 0.8;

const DROP_DRAG: f32 = 4.5e-4;

const SEAL_SPEED: (f32, f32) = (8.0, 40.0);

const SPLASH_DROPS: usize = 110;

const SPLASH_PETALS: usize = 14;

const SHEET_THICKNESS: f32 = 0.01;

const SPLASH_H: f32 = 1.0 / 256.0;

const RING_ENERGY_SHARE: f32 = 0.10;

const RING_STEEPEST: f32 = 0.6;

const RING_EDGE: f32 = 0.5;

const RING_PEAK: f32 = 0.4;

pub const RING_FILM_VISCOSITY: f32 = 1.0e-5;

pub const DROP_CELLS: (f32, f32) = (4.0, 24.0);

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SplashShape {

    pub cavity: f32,

    pub depth: f32,

    pub collapse: f32,

    pub sealed: f32,

    pub fine: f32,

    pub sheet_speed: f32,
    pub sheet_angle: f32,
    pub crown_height: f32,
    pub crown_seconds: f32,
    pub petals: usize,

    pub drop: f32,
    pub drops: usize,
    pub spray_speed: f32,

    pub jet_height: f32,
    pub jet_width: f32,

    pub broken: f32,

    pub white: f32,

    pub spray: usize,
    pub froth: usize,

    pub plow: f32,

    pub struck: f32,

    pub mist: usize,

    pub ring_wavelength: f32,
    pub ring_speed: f32,
    pub ring_height: f32,

    pub wave_source: f32,

    pub ring_gravity: f32,

    pub ring_release: f32,

    pub drops_land: f32,
    pub drops_band: f32,
    pub drops_from: f32,
    pub drops_over: f32,
    pub drops_density: f32,

    pub foam: f32,

    pub foam_seconds: f32,
    pub settle: f32,

    pub forward: Vec3,

    pub carry: Vec3,

    pub seconds: f32,
}

fn thrown_height(up: f32, k: f32, t: f32) -> f32 {
    let x = k * t;

    let (a1, phi) = if x < 0.05 {
        (t * (1.0 - x * (0.5 - x * (1.0 / 6.0 - x / 24.0))), 0.5 - x * (1.0 / 6.0 - x * (1.0 / 24.0 - x / 120.0)))
    } else {
        let e = (-x).exp_m1();
        (-e / k, (x + e) / (x * x))
    };
    up * a1 - GRAVITY * t * t * phi
}

fn aloft(up: f32, k: f32) -> f32 {
    if up <= 0.0 {
        return 0.0;
    }
    if k < 1e-3 {
        return 2.0 * up / GRAVITY;
    }
    let apex = (up * k / GRAVITY).ln_1p() / k;
    let (mut lo, mut hi) = (apex, apex + 2.0 * up / GRAVITY + 1.0 / k);
    for _ in 0..24 {
        let mid = f32::midpoint(lo, hi);
        if thrown_height(up, k, mid) > 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    f32::midpoint(lo, hi)
}

fn apex(up: f32, k: f32) -> f32 {
    if up <= 0.0 {
        return 0.0;
    }

    let y = up * k / GRAVITY;
    if y < 0.1 {
        return up * up / GRAVITY * (0.5 - y * (1.0 / 3.0 - y * (0.25 - y * (0.2 - y / 6.0))));
    }
    up / k - GRAVITY / (k * k) * y.ln_1p()
}

pub fn splash_shape(s: &Splash) -> SplashShape {
    use std::f32::consts::{PI, TAU};
    let speed = s.velocity.length();
    let r = s.size.max(1e-3);
    let calm = s.calm.clamp(0.0, 1.0);
    let normal = s.velocity.y.abs();
    let sin_entry = if speed > 1e-4 { (normal / speed).clamp(0.0, 1.0) } else { 1.0 };
    let flat = Vec3::new(s.velocity.x, 0.0, s.velocity.z);
    let forward = if speed > 1e-4 { flat / speed } else { Vec3::ZERO };
    let sealed = smoothstep(SEAL_SPEED.0, SEAL_SPEED.1, speed);

    let pressure = |a: f32| WATER_DENSITY * GRAVITY * a + ATMOSPHERE * sealed;
    let drive = 0.5 * WATER_DENSITY * BODY_DRAG * r * r * speed * speed + pressure(r) * r * r;
    let mut cavity = r;
    for _ in 0..24 {
        cavity = f32::midpoint(cavity, (drive / pressure(cavity)).sqrt());
    }
    let closes = |a: f32| 1.83 * a * (WATER_DENSITY / pressure(a)).sqrt();

    let stops = 2.0 * s.mass.max(1e-5) / (WATER_DENSITY * BODY_DRAG * PI * r * r);
    let travel = stops * (1.0 + speed * closes(cavity) / stops).ln();

    cavity = r + (cavity - r).min(travel);

    let flat_hit = s.flat.clamp(0.0, 1.0);
    let mut depth = (travel * sin_entry).min(s.depth.max(0.0)).max(0.0) * (1.0 - 0.6 * flat_hit);

    let energy = 0.5 * s.mass.max(0.0) * speed * speed;
    let dug = PI * cavity * cavity * depth * (WATER_DENSITY * GRAVITY * depth / 12.0 + ATMOSPHERE * sealed / 3.0);
    if dug > energy && dug > 0.0 {
        let k = (energy / dug).powf(0.25);
        cavity = (cavity * k).max(r);
        depth *= k;
    }
    let collapse = closes(cavity).max(1e-3);

    let deep = smoothstep(0.5, 2.0, s.depth / cavity);
    let passes = smoothstep(1.0, 2.5, depth / cavity);

    let shoulder = (WATER_DENSITY * normal * normal * r / SURFACE_TENSION).sqrt();
    let thrown = 0.8 * shoulder / (shoulder + 150.0) * normal * (1.0 - sealed) * (travel / r).min(1.0);

    let sheet_speed = (3.5 * (cavity - r) / collapse).max(thrown).max(1.3 * normal * flat_hit);

    let steep = smoothstep(0.3, 1.5, s.depth / r);
    let sheet_angle = lerp(lerp(28.0f32, 68.0, steep), 14.0, flat_hit).to_radians();
    let plow = 0.9 * flat.length() * (1.0 - sealed);
    let weber = WATER_DENSITY * sheet_speed * sheet_speed * cavity / SURFACE_TENSION;
    let drop = (cavity / weber.max(1.0).sqrt()).clamp(1.5e-4, 4e-3);
    let fine = 1.0 - smoothstep(3e-4, 1e-3, drop);
    let up = sheet_speed * sheet_angle.sin();
    let air = DROP_DRAG * up / drop;
    let crown_height = apex(up, air);
    let crown_seconds = aloft(up, air);

    let rim = 1.2 * cavity;
    let petals = if crown_height > 0.004 && fine < 0.75 && flat_hit < 0.5 { ((TAU * rim / (1.6 * crown_height)).round() as usize).clamp(10, SPLASH_PETALS) } else { 0 };

    let fingers = TAU * cavity / (9.0 * drop.max(1.5e-3));

    let drops = if sheet_speed > 0.15 { ((1.5 * fingers * (0.15 + 0.85 * (1.0 - fine)) * (0.4 + 0.6 * calm) * (1.0 + flat_hit)).round() as usize).clamp(4, SPLASH_DROPS) } else { 0 };

    let spray_speed = (0.5 + 0.75 * shoulder / (shoulder + 150.0)) * normal;

    let jet_height = 1.4 * depth * passes * deep * calm * calm * (1.0 - s.held.clamp(0.0, 1.0));
    let jet_width = (0.3 * cavity * (cavity / depth.max(1e-3)).sqrt()).clamp(0.08 * cavity, 0.45 * cavity);
    let jet_seconds = 2.0 * (2.0 * jet_height / GRAVITY).sqrt();

    let retracts = (2.0 * SURFACE_TENSION / (WATER_DENSITY * SHEET_THICKNESS * cavity)).sqrt();
    let broken = smoothstep(0.6, 2.5, crown_height / (retracts * crown_seconds.max(1e-3))) * (1.0 - fine);
    let white = broken.max(0.85 * (1.0 - calm)) * (1.0 - fine);
    let throws = sheet_speed > 0.15;
    let spray = if throws { ((1.5 + 9.0 * cavity.sqrt() * (1.0 - sealed) + 2.5 * sealed).round() as usize).clamp(3, 6) } else { 0 };

    let froth = if throws && white > 0.4 && (cavity > 0.1 || broken > 0.5) { ((white * (6.0 + 4.0 * cavity.sqrt())).round() as usize).clamp(6, 9) } else { 0 };
    let mist = if sealed > 0.05 { (2.0 + 6.0 * sealed).round() as usize } else { 0 };

    let into = 0.5 * s.mass.max(0.0) * (normal * normal + 0.1 * flat.length_squared());
    let a = cavity.max(0.01);
    let wave_source = ((8.0 * RING_ENERGY_SHARE * into / (WATER_DENSITY * GRAVITY * PI * a * a)).sqrt()).min(RING_STEEPEST * a)
        * (0.3 + 0.7 * calm)
        * (1.0 - 0.5 * sealed);
    let k = 3.0 / (RING_EDGE * a);
    let ring_wavelength = TAU / k;
    let bond = SURFACE_TENSION * k * k / (WATER_DENSITY * GRAVITY);
    let ring_gravity = GRAVITY * (1.0 + bond);
    let ring_speed = 0.5 * (ring_gravity / k).sqrt();

    let ring_height = 0.5 * RING_PEAK * wave_source;
    let ring_release = 0.5 * collapse;

    let foam = ((0.35 + 1.2 * smoothstep(0.03, 0.5, cavity) + 0.4 * sealed + 0.4 * (1.0 - calm)) * smoothstep(0.1, 0.6, sheet_speed) * (1.0 - 0.5 * flat_hit)).min(0.65 + 0.3 * (1.0 - calm));
    let foam_seconds = 1.2 + 3.0 * cavity.sqrt();
    let settle = (1.5 + 4.0 * cavity.sqrt()).min(6.0);

    let carry = (flat * 0.4).clamp_length_max(sheet_speed) * (1.0 - 0.8 * sealed);

    let fastest = sheet_speed + spray_speed + plow;
    let finest = (drop * (sheet_speed / fastest.max(1e-3)).clamp(0.08, 1.0)).max(1.5e-4);

    let fan = 1.25 * sheet_speed + 0.45 * spray_speed + plow;
    let fan_seconds = if spray > 0 { aloft(fan * sheet_angle.sin(), DROP_DRAG * fan / (0.6 * drop)).min(2.5) + 0.25 * collapse } else { 0.0 };
    let seconds = (0.5 * collapse + aloft(fastest * sheet_angle.sin(), DROP_DRAG * fastest / finest).min(2.5) * (1.0 - fine))
        .max(collapse + jet_seconds + 0.1)
        .max(if petals > 0 { 1.3 * crown_seconds + 0.2 * collapse } else { 0.0 })
        .max(fan_seconds * (1.0 - sealed))
        .max(0.6 * collapse + 0.5 + 1.2 * cavity.sqrt() + 0.4 * sealed)
        .min(SPLASH_SECONDS);

    let fly_k = DROP_DRAG * sheet_speed / drop;
    let flight = aloft(up, fly_k);
    let reach = sheet_speed * sheet_angle.cos() * if fly_k < 1e-4 { flight } else { (1.0 - (-fly_k * flight).exp()) / fly_k };
    let drops_land = 1.1 * cavity + 0.7 * reach;
    let drops_band = 0.3 * cavity + 0.5 * reach;
    let drops_from = 0.25 * collapse + 0.6 * flight;
    let drops_over = (0.8 * flight).max(0.15);
    let falling = if throws { (drops as f32 * 3.0 + spray as f32 * 25.0) * (1.0 - fine) } else { 0.0 };
    let drops_density = falling / (TAU * drops_land * 2.0 * drops_band).max(1e-3);
    SplashShape {
        cavity,
        depth,
        collapse,
        sealed,
        fine,
        sheet_speed,
        sheet_angle,
        crown_height,
        crown_seconds,
        petals,
        drop,
        drops,
        spray_speed,
        jet_height,
        jet_width,
        broken,
        white,
        spray,
        froth,
        plow,
        struck: speed,
        mist,
        ring_wavelength,
        ring_speed,
        ring_height,
        wave_source,
        ring_gravity,
        ring_release,
        drops_land,
        drops_band,
        drops_from,
        drops_over,
        drops_density,
        foam,
        foam_seconds,
        settle,
        forward,
        carry,
        seconds,
    }
}

pub fn splash_range(shape: &SplashShape) -> f32 {
    (250.0 * (shape.cavity + shape.crown_height + shape.jet_height)).max(15.0)
}

pub fn splash_bounds(s: &Splash) -> (Vec3, Vec3) {
    if s.mud > 0.0 {
        return mud_splash::bounds(s);
    }
    let shape = splash_shape(s);
    let fastest = shape.sheet_speed + shape.spray_speed + shape.plow;

    let (up, out) = (fastest * shape.sheet_angle.sin(), fastest * shape.sheet_angle.cos() + shape.carry.length());
    let slowed = DROP_DRAG / (1.6 * shape.drop);
    let high = 1.3 * shape.crown_height.max(shape.jet_height).max(apex(up, slowed * fastest)).max(shape.sheet_speed / 3.0) + 2.0 * shape.cavity + 0.1;
    let wide = 2.5 * shape.cavity + (out * shape.seconds).min(if slowed * out > 1e-3 { 1.0 / slowed } else { f32::MAX }).max(shape.sheet_speed / 3.0) + 0.1;
    (s.position - Vec3::new(wide, 0.05, wide), s.position + Vec3::new(wide, high, wide))
}

pub fn splash_seen(s: &Splash, time: f64, views: &[[glam::Vec4; 6]]) -> bool {
    let age = (time - s.born) as f32;
    if !(0.0..SPLASH_SECONDS).contains(&age) {
        return false;
    }
    let (lo, hi) = splash_bounds(s);
    views.iter().any(|planes| {
        let mut unbounded = *planes;
        unbounded[5] = glam::Vec4::new(0.0, 0.0, 0.0, 1.0);
        super::shadow::aabb_in_frustum(&unbounded, lo, hi)
    })
}

pub fn strike(live: &mut Vec<Splash>, mut new: Splash, kept: usize) {
    new.calm = 1.0;
    if new.mud > 0.0 {
        // Mud keeps no churned surface for the next strike to land in.
        if live.len() >= kept.max(1) {
            live.remove(0);
        }
        live.push(new);
        return;
    }
    let own = splash_shape(&new);
    let mut calm = 1.0f32;
    for old in live.iter() {
        let age = (new.born - old.born) as f32;
        if age < 0.0 || old.mud > 0.0 {
            continue;
        }
        let was = splash_shape(old);
        if age > was.settle {
            continue;
        }
        let apart = Vec3::new(new.position.x - old.position.x, 0.0, new.position.z - old.position.z).length();
        let reach = 1.2 * was.cavity + own.cavity;
        let within = 1.0 - smoothstep(0.5 * reach, reach, apart);

        let churned = if age < was.collapse { 1.0 } else { 0.7 * (1.0 - smoothstep(0.0, was.settle - was.collapse, age - was.collapse)) };
        let outweighs = (was.cavity / own.cavity).powi(2).min(1.0);
        calm *= 1.0 - within * churned * outweighs;
    }
    new.calm = calm.clamp(0.0, 1.0);
    if live.len() >= kept.max(1) {
        live.remove(0);
    }
    live.push(new);
}

fn splash(s: &Splash, time: f64, at: &Surroundings, over: &mut Vec<(f32, EffectInstance)>, screened: &mut Vec<EffectInstance>) {
    use std::f32::consts::{FRAC_PI_2, PI, TAU};
    let age = (time - s.born) as f32;
    if !(0.0..SPLASH_SECONDS).contains(&age) {
        return;
    }
    if s.mud > 0.0 {
        mud_splash::draw(s, time, at, over);
        return;
    }
    let shape = splash_shape(s);
    if age >= shape.seconds || shape.sheet_speed <= 0.02 {
        return;
    }
    let player = |p: Vec3| at.yaw_inv * (p - at.offset);
    let head_world = at.yaw_inv.inverse() * at.head + at.offset;
    let away = head_world.distance(s.position);
    if away > splash_range(&shape) {
        return;
    }
    let ambient = (at.ambient)(s.position);
    let sun = at.lights.iter().find(|l| matches!(l.kind, LightKind::Directional) && l.intensity > 0.0).map(|l| {
        let c = l.color.to_linear();
        (-l.direction.normalize_or_zero(), Vec3::new(c[0], c[1], c[2]) * (l.intensity * s.sunlit.clamp(0.0, 1.0)))
    });
    let lit = |centre: Vec3, albedo: f32, g: f32| {
        let to_eye = (at.head - centre).normalize_or_zero();
        let mut c = ambient * albedo;
        if let Some((to_sun, rgb)) = sun {
            c += rgb * (albedo * henyey_greenstein(g, -to_sun, to_eye));
        }
        c * at.exposure
    };

    let glassy = |centre: Vec3, face: Vec3, opacity: f32| -> Vec3 {
        let to_eye = (at.head - centre).normalize_or_zero();
        let f = at.yaw_inv * face.normalize_or(Vec3::Y);
        let n = if f.dot(to_eye) < 0.0 { -f } else { f };
        let cos_v = n.dot(to_eye).clamp(0.05, 1.0);
        let fresnel = 0.02 + 0.98 * (1.0 - cos_v).powi(5);
        let mut c = ambient * (0.15 + fresnel);
        if let Some((to_sun, rgb)) = sun {

            let h = (to_sun + to_eye).normalize_or_zero();
            let nh = n.dot(h).abs();
            let a2 = 0.09f32;
            let d = a2 / (PI * (nh * nh * (a2 - 1.0) + 1.0).powi(2));
            let glint = d * (0.02 + 0.98 * (1.0 - h.dot(to_eye).clamp(0.0, 1.0)).powi(5)) / (4.0 * cos_v);
            c += rgb * (glint * n.dot(to_sun).abs() + 0.08 * henyey_greenstein(0.8, -to_sun, to_eye));
        }
        c * (at.exposure / opacity.max(0.05))
    };
    let gravity = Vec3::new(0.0, -GRAVITY, 0.0);
    let calm = s.calm.clamp(0.0, 1.0);
    let lean = shape.forward.length();
    let toward = shape.forward.normalize_or_zero();

    let middle = s.position + shape.forward * (0.4 * shape.cavity);

    let seen_along = |p: Vec3, axis: Vec3| -> (f32, f32) {
        let to_eye = (head_world - p).normalize_or(Vec3::Z);
        let up_seen = Vec3::Y - to_eye * to_eye.y;
        if up_seen.length_squared() < 1e-4 {
            return (0.0, 1.0);
        }
        let u = up_seen.normalize();
        let r = u.cross(to_eye);
        let (ar, au) = (axis.dot(r), axis.dot(u));
        (-ar.atan2(au), (ar * ar + au * au).sqrt())
    };

    let shown = ((shape.drops as f32 * (8.0 / away.max(0.1)).min(1.0)).ceil() as usize).min(shape.drops);
    for i in 0..shown {
        let u = uniforms(hash64(s.seed ^ (i as u64 + 1).wrapping_mul(0x632be59bd9b4e019)));
        let azimuth = u[0] * TAU;
        let out = Vec3::new(azimuth.cos(), 0.0, azimuth.sin());
        let down_range = out.dot(toward) * lean;

        let ahead = down_range.max(0.0);
        let launch = shape.sheet_speed * lerp(0.6, 1.0, u[6]) + shape.spray_speed * u[2] * u[2] * (1.0 + 0.8 * down_range).max(0.2) + shape.plow * ahead * ahead * u[6];
        let width = (shape.drop * lerp(0.6, 1.6, u[5]) * (shape.sheet_speed / launch.max(1e-3)).clamp(0.08, 1.0)).max(1.5e-4);
        let elevation = shape.sheet_angle * (1.0 - 0.45 * ahead) + (u[1] - 0.5) * 0.42;
        let v0 = out * (elevation.cos() * launch) + Vec3::Y * (elevation.sin() * launch) + shape.carry * (u[7] * (0.5 + 0.5 * down_range));
        let p0 = middle + out * (shape.cavity * lerp(0.8, 1.1, u[3]));
        let a = age - 0.5 * shape.collapse * u[4];
        if a <= 0.0 {
            continue;
        }
        let (pos, vel) = ballistic(p0, v0, gravity, DROP_DRAG * launch / width, a);
        if pos.y < s.position.y && vel.y < 0.0 {
            continue;
        }
        let centre = player(pos);
        let axis = at.yaw_inv * vel.normalize_or(Vec3::Y);
        let moving = at.yaw_inv * vel;
        let c = lit(centre, DROP_ALBEDO, DROP_PHASE_G);
        screened.push(EffectInstance {
            centre: [centre.x, centre.y, centre.z, 0.5 * width],
            colour: [c.x, c.y, c.z, 1.0 - smoothstep(0.85, 1.0, age / shape.seconds)],
            light: [0.0; 4],
            light_dir: [0.0, 1.0, 0.0, vel.length() * DROP_STREAK],
            axis: [axis.x, axis.y, axis.z, 0.01],
            params: [0.0, EMBER_LAYER as f32, 0.0, 0.0],
            motion: [moving.x, moving.y, moving.z, 0.0],
        });
    }

    let whole = (1.0 - shape.fine) * calm * (1.0 - 0.85 * shape.broken);
    let crown_opacity = lerp(0.55, 0.92, shape.broken) * whole;

    if shape.petals > 0 && crown_opacity > 0.2 {
        let turn = uniforms(hash64(s.seed ^ 0xc80))[0] * TAU;
        for k in 0..shape.petals {
            let u = uniforms(hash64(s.seed ^ 0xc80 ^ (k as u64 + 1).wrapping_mul(0x8cb92ba72f3d8dd7)));
            let azimuth = turn + TAU * (k as f32 + 0.5 * (u[0] - 0.5)) / shape.petals as f32;
            let out = Vec3::new(azimuth.cos(), 0.0, azimuth.sin());
            let down_range = out.dot(toward) * lean;
            let ahead = down_range.max(0.0);
            let angle = shape.sheet_angle * (1.0 - 0.45 * ahead);
            let up = (shape.sheet_speed + shape.plow * ahead * ahead) * angle.sin() * lerp(0.8, 1.25, u[1]) * (1.0 + 0.6 * down_range.min(0.0)).max(0.3);
            let air = DROP_DRAG * up / shape.drop;
            let life = aloft(up, air).max(0.04);
            let start = 0.2 * shape.collapse * u[2];
            let t = (age - start) / life;
            if !(t > 0.0 && t < 1.0) {
                continue;
            }

            let axis = (Vec3::Y * angle.sin() + out * angle.cos()).normalize();
            let tangent = Vec3::new(-out.z, 0.0, out.x);
            let place = |since: f32| -> (Vec3, f32, f32, f32) {
                let since = since.clamp(0.0, life);
                let high = thrown_height(up, air, since).max(0.0);
                let long = high / angle.sin();
                let rim = shape.cavity * (0.8 + 0.45 * smoothstep(0.0, 0.5 * shape.collapse, since)) + (0.3 * shape.carry.length() + 0.5 * shape.plow * angle.cos()) * since * ahead;
                let centre = middle + out * rim + axis * (0.5 * long);
                let (spin, shows) = seen_along(centre, axis);
                let (_, across) = seen_along(centre, tangent);
                let half = (0.5 * long * shows.max(0.2)).max(1e-4);

                let wide = (2.0 * PI * rim / shape.petals as f32 * across.max(0.45)).min(6.0 * half).max(half);
                (centre, half, wide / half, spin)
            };
            let since = age - start;
            let (centre_w, half, aspect, spin) = place(since);
            let moving = (place(since + 0.5 * SPLASH_H).0 - place(since - 0.5 * SPLASH_H).0) / SPLASH_H;
            let centre = player(centre_w);
            let moving = at.yaw_inv * moving;

            let face = out * angle.sin() - Vec3::Y * angle.cos();
            let c = glassy(centre, face, crown_opacity).lerp(lit(centre, 0.95, 0.15), shape.broken);

            let stage = 1.999 * smoothstep(0.3 - 0.3 * shape.broken, 0.95 - 0.35 * shape.broken, t);
            let variant = (u[3] * CROWN_VARIANTS as f32) as u32 % CROWN_VARIANTS;
            let opacity = crown_opacity * smoothstep(0.0, 0.06, t) * (1.0 - smoothstep(0.8, 1.0, t));
            over.push((
                (centre - at.head).length_squared(),
                EffectInstance {
                    centre: [centre.x, centre.y, centre.z, half],
                    colour: [c.x, c.y, c.z, opacity],
                    light: [0.0, 0.0, 0.0, 1.0],
                    light_dir: [0.0, 1.0, 0.0, 0.0],
                    axis: [0.0, 1.0, 0.0, (0.25 * half).clamp(0.005, 0.03)],

                    params: [FRAC_PI_2 + spin, (CROWN_LAYER + variant * CROWN_FRAMES) as f32 + stage, aspect, 0.0],
                    motion: [moving.x, moving.y, moving.z, 0.0],
                },
            ));
        }
    }

    struct Fan {
        count: usize,
        layer: u32,
        variants: u32,
        salt: u64,

        fast: (f32, f32),
        coarse: f32,

        tall: (f32, f32),

        stand: f32,
        opacity: f32,
        phase_g: f32,
    }
    let plume = shape.sealed;
    let fans = [
        Fan { count: shape.froth, layer: FROTH_LAYER, variants: FROTH_VARIANTS, salt: 0xf207, fast: (0.0, 0.05), coarse: 1.5, tall: (1.0, 1.15), stand: 1.0, opacity: 0.95 * shape.white, phase_g: 0.0 },

        Fan { count: shape.spray, layer: SPRAY_LAYER, variants: SPRAY_VARIANTS, salt: 0x5b7a, fast: (0.15, 0.45), coarse: 0.6, tall: (1.0, 1.5), stand: 0.0, opacity: 0.8 * (shape.drops as f32 / 60.0).clamp(0.45, 1.0).max(plume), phase_g: 0.55 },
    ];
    for fan in &fans {
        if fan.opacity < 0.02 {
            continue;
        }
        let turn = uniforms(hash64(s.seed ^ fan.salt))[0] * TAU;
        for j in 0..fan.count {
            let u = uniforms(hash64(s.seed ^ fan.salt ^ (j as u64 + 1).wrapping_mul(0x8cb92ba72f3d8dd7)));
            let azimuth = turn + TAU * (j as f32 + 0.7 * (u[0] - 0.5)) / fan.count as f32;
            let out = Vec3::new(azimuth.cos(), 0.0, azimuth.sin());
            let down_range = out.dot(toward) * lean;
            let ahead = down_range.max(0.0);
            let open_launch = ((shape.sheet_speed * lerp(0.9, 1.25, u[1]) + shape.spray_speed * lerp(fan.fast.0, fan.fast.1, u[2]) + shape.plow * ahead * ahead) * (1.0 + 0.5 * down_range.min(0.0)).max(0.4)).min(0.9 * shape.struck);

            let launch = lerp(open_launch, (2.0 * GRAVITY * shape.jet_height.max(0.3 * shape.depth)).sqrt() * lerp(0.6, 1.1, u[1]), plume);
            let angle = lerp(shape.sheet_angle * (1.0 - 0.45 * ahead), 1.5, plume);
            let up = launch * angle.sin();

            let air = DROP_DRAG * launch / (fan.coarse * shape.drop) * (1.0 - plume) / 2.2;
            let life = aloft(up, air).max(0.1);
            let rises = if air < 1e-3 { up / GRAVITY } else { (up * air / GRAVITY).ln_1p() / air };
            let start = 0.25 * shape.collapse * u[3] * (1.0 - plume) + 0.3 * shape.collapse * plume * u[3];
            let since = age - start;
            let t = since / life;
            if !(t > 0.0 && t < 1.0) {
                continue;
            }
            let leans = lerp(angle, FRAC_PI_2, fan.stand);
            let axis = (Vec3::Y * leans.sin() + out * leans.cos()).normalize();

            let tall_by = lerp(fan.tall.0, fan.tall.1, u[4]) * (1.0 + 1.2 * plume) * (1.0 + 0.35 * (1.0 - fan.stand) * smoothstep(0.2, 0.7, shape.cavity));
            let place = |since: f32| -> (Vec3, f32, f32) {
                let since = since.clamp(0.0, life);

                let top = thrown_height(up, air, since.min(rises)).max(0.0);
                let high = top * (1.0 - 0.4 * smoothstep(0.55, 1.0, since / life));
                let long = high / leans.sin();
                let rim = shape.cavity * (0.85 + 0.35 * smoothstep(0.0, 0.5 * shape.collapse, since)) * (1.0 - plume) * lerp(0.6 + 0.4 * u[5], 1.0, fan.stand);
                let foot = middle + out * rim + (shape.carry * 0.3 + toward * (0.5 * shape.plow * angle.cos())) * (since * ahead * (1.0 - plume));
                let centre = foot + axis * (0.5 * long);
                let (spin, shows) = seen_along(centre, axis);
                ((centre), (0.5 * long * shows.max(0.25)).max(1e-4), spin)
            };
            let (centre_w, tall, spin) = place(since);
            let moving = at.yaw_inv * ((place(since + 0.5 * SPLASH_H).0 - place(since - 0.5 * SPLASH_H).0) / SPLASH_H);
            let centre = player(centre_w);
            let half = tall / tall_by;
            let c = lit(centre, 0.97, fan.phase_g);

            let stage = 1.999 * smoothstep(lerp(0.12, 0.4, fan.stand), lerp(0.8, 0.97, fan.stand), t);
            let variant = (u[6] * fan.variants as f32) as u32 % fan.variants;
            let opacity = fan.opacity * lerp(0.8, 1.0, u[7]) * smoothstep(0.0, 0.05, t) * (1.0 - smoothstep(0.7, 1.0, t));
            over.push((
                (centre - at.head).length_squared(),
                EffectInstance {
                    centre: [centre.x, centre.y, centre.z, half],
                    colour: [c.x, c.y, c.z, opacity],
                    light: [0.0, 0.0, 0.0, 1.0],
                    light_dir: [0.0, 1.0, 0.0, 0.0],
                    axis: [0.0, 1.0, 0.0, (0.25 * half).clamp(0.005, 0.05)],
                    params: [spin, (fan.layer + variant * 3) as f32 + stage, if u[8] < 0.5 { -tall_by } else { tall_by }, 0.0],
                    motion: [moving.x, moving.y, moving.z, 0.0],
                },
            ));
        }
    }

    let since_close = age - shape.collapse;
    if shape.jet_height > 0.01 && since_close > 0.0 {
        let throw = (2.0 * GRAVITY * shape.jet_height).sqrt();
        let life = 2.0 * throw / GRAVITY;
        let u = uniforms(hash64(s.seed ^ 0x6e7));
        let axis = (Vec3::Y + shape.forward * 0.35 + Vec3::new(u[0] - 0.5, 0.0, u[1] - 0.5) * 0.12).normalize();
        if since_close < life {
            let place = |since: f32| -> (Vec3, f32, f32, f32) {
                let since = since.clamp(0.0, life);
                let high = (throw * since - 0.5 * GRAVITY * since * since).max(0.0);
                let centre = middle + axis * (0.5 * high);
                let (spin, shows) = seen_along(centre, axis);
                let tall = (0.5 * high * shows.max(0.2)).max(1e-4);

                let half = (2.0 * shape.jet_width * (1.0 - 0.4 * since / life)).min(tall).max(1e-4);
                (centre, half, tall / half, spin)
            };
            let t = since_close / life;
            let (centre_w, half, aspect, spin) = place(since_close);
            let moving = at.yaw_inv * ((place(since_close + 0.5 * SPLASH_H).0 - place(since_close - 0.5 * SPLASH_H).0) / SPLASH_H);
            let centre = player(centre_w);
            let stage = 1.999 * smoothstep(0.3, 0.85, t);
            let variant = (u[2] * JET_VARIANTS as f32) as u32 % JET_VARIANTS;

            let opacity = lerp(0.6, 0.9, shape.fine.max(0.5 * t)) * smoothstep(0.0, 0.05, t) * (1.0 - smoothstep(0.85, 1.0, t));

            let face = (head_world - centre_w) * Vec3::new(1.0, 0.0, 1.0);
            let c = glassy(centre, face, opacity).lerp(lit(centre, 0.95, 0.0), shape.fine.max(0.5 * t));
            over.push((
                (centre - at.head).length_squared(),
                EffectInstance {
                    centre: [centre.x, centre.y, centre.z, half],
                    colour: [c.x, c.y, c.z, opacity],
                    light: [0.0, 0.0, 0.0, 1.0],
                    light_dir: [0.0, 1.0, 0.0, 0.0],
                    axis: [0.0, 1.0, 0.0, (0.25 * half).clamp(0.005, 0.03)],
                    params: [spin, (JET_LAYER + variant * JET_FRAMES) as f32 + stage, if u[3] < 0.5 { -aspect } else { aspect }, 0.0],
                    motion: [moving.x, moving.y, moving.z, 0.0],
                },
            ));
        }

        let release = 0.35 * throw / GRAVITY;
        for i in 0..3u64 {
            let u = uniforms(hash64(s.seed ^ 0x6e7 ^ (i + 1).wrapping_mul(0x9e3779b97f4a7c15)));
            let width = (shape.jet_width * lerp(0.5, 1.0, u[1])).clamp(0.002, 0.012) * (1.0 - 0.7 * shape.fine);
            let v0 = axis * (throw * lerp(0.8, 1.12, u[0])) + Vec3::new(u[2] - 0.5, 0.0, u[3] - 0.5) * (0.12 * throw) + shape.carry * 0.2;
            let (pos, vel) = ballistic(middle, v0, gravity, DROP_DRAG * throw / width, since_close);
            if since_close < release || (pos.y < s.position.y && vel.y < 0.0) {
                continue;
            }
            let centre = player(pos);
            let dir = at.yaw_inv * vel.normalize_or(Vec3::Y);
            let moving = at.yaw_inv * vel;
            let c = lit(centre, DROP_ALBEDO, DROP_PHASE_G);
            screened.push(EffectInstance {
                centre: [centre.x, centre.y, centre.z, 0.5 * width],
                colour: [c.x, c.y, c.z, 1.0],
                light: [0.0; 4],
                light_dir: [0.0, 1.0, 0.0, vel.length() * DROP_STREAK],
                axis: [dir.x, dir.y, dir.z, 0.01],
                params: [0.0, EMBER_LAYER as f32, 0.0, 0.0],
                motion: [moving.x, moving.y, moving.z, 0.0],
            });
        }
    }

    let mist_turn = uniforms(hash64(s.seed ^ 0x5bd1e995))[0] * TAU;
    for i in 0..shape.mist {
        let u = uniforms(hash64(s.seed ^ 0x5bd1e995 ^ (i as u64 + 1).wrapping_mul(0x8cb92ba72f3d8dd7)));
        let begins = 0.6 * shape.collapse * u[6];
        let life = (lerp(0.5, 1.0, u[0]) * (0.5 + 1.2 * shape.cavity.sqrt() + 0.4 * shape.sealed)).min(shape.seconds - begins);
        let since = age - begins;
        if !(since > 0.0 && since < life) {
            continue;
        }
        let t = since / life;

        let azimuth = mist_turn + TAU * (i as f32 + 0.8 * (u[1] - 0.5)) / shape.mist as f32;
        let out = Vec3::new(azimuth.cos(), 0.0, azimuth.sin());

        let share = ((i as f32 + 1.0) * 0.618_034 + u[2] / shape.mist as f32).fract();
        let launch = shape.sheet_speed * lerp(0.35, 1.0, share);
        let width = shape.drop * lerp(0.35, 1.2, share);
        let v0 = out * (launch * shape.sheet_angle.cos() * (1.0 - 0.9 * plume)) + Vec3::Y * (launch * lerp(shape.sheet_angle.sin() * 0.6, 1.0, plume)) + shape.carry * 0.3;

        let drag = (DROP_DRAG * launch / (2.2 * width)).max(3.0);
        let (pos, vel) = ballistic(middle + out * (shape.cavity * 0.6 * (1.0 - plume)), v0, gravity, drag, since);

        let size = lerp(0.3 * shape.cavity + shape.crown_height, 0.45 * shape.cavity, plume) * lerp(0.7, 1.3, u[4]) * (0.5 + 1.7 * t);
        let floor = s.position.y + 0.02;
        let centre = player(pos.max(Vec3::new(f32::MIN, floor, f32::MIN)));
        let moving = at.yaw_inv * if pos.y < floor { vel * Vec3::new(1.0, 0.0, 1.0) } else { vel };

        let c = lit(centre, 0.95, 0.0);
        let opacity = (0.05 + 0.7 * shape.fine) * plume * smoothstep(0.0, 0.08, t) * (1.0 - t).powf(1.4);
        let frame = SMOKE_FRAMES as f32 * (0.35 + 0.55 * t);
        over.push((
            (centre - at.head).length_squared(),
            EffectInstance {
                centre: [centre.x, centre.y, centre.z, size],
                colour: [c.x, c.y, c.z, opacity],
                light: [0.0, 0.0, 0.0, if plume > 0.5 { 1.0 } else { 0.0 }],
                light_dir: [0.0, 1.0, 0.0, 0.0],
                axis: [0.0, 1.0, 0.0, 0.05],
                params: [if plume > 0.5 { 0.0 } else { (2.0 * u[5] - 1.0) * PI }, frame.min((SMOKE_FRAMES - 1) as f32), 1.0 + 1.3 * plume, 0.0],
                motion: [moving.x, moving.y, moving.z, 0.0],
            },
        ));
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SplashRings<const N: usize> {

    pub rings: [[f32; 4]; N],

    pub waves: [[f32; 4]; N],

    pub pools: [[f32; 4]; N],

    pub drops: [[f32; 4]; N],
}

pub fn splash_rings<const N: usize>(splashes: &[Splash], time: f64, head_world: Vec3) -> SplashRings<N> {
    use std::f32::consts::PI;
    let mut live: Vec<(f32, [[f32; 4]; 4])> = splashes
        .iter()
        .filter_map(|s| {
            let age = (time - s.born) as f32;
            // Mud makes no rings.
            if age < 0.0 || s.mud > 0.0 {
                return None;
            }
            let shape = splash_shape(s);
            let lasts = ring_seconds(&shape);
            if age >= lasts.max(shape.foam_seconds + shape.collapse) || shape.wave_source <= 0.0 && shape.foam <= 0.0 {
                return None;
            }
            let a = shape.cavity.max(0.01);
            let run = (age - shape.ring_release).max(0.0);
            let fade = 1.0 - smoothstep(0.6 * lasts, lasts, age);

            let c = RING_EDGE * a;
            let size = RING_PEAK * shape.wave_source * a * c.powi(3) * (3.0f32).exp() / 27.0 * fade;

            let g4 = 0.25 * shape.ring_gravity;
            let steep = [1.5f32, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0]
                .iter()
                .map(|m| {
                    let k = m / c;
                    let from_rim = run * (g4 / k).sqrt();
                    size * k.powi(4) * (-k * (c + 2.0 * RING_FILM_VISCOSITY * k * run)).exp() * smoothstep(0.0, a, from_rim) / (a + from_rim)
                })
                .fold(0.0, f32::max);

            let bowl = if age < shape.collapse {
                0.5 * shape.depth.min(shape.cavity) * (PI * age / shape.collapse).sin().max(0.0)
            } else {
                let x = (age - shape.collapse) / (0.3 * shape.collapse + 0.15);
                -0.25 * shape.depth.min(shape.cavity) * x * (1.0 - x).exp()
            };

            let foamed = (age - 0.3 * shape.collapse) / shape.foam_seconds;
            let reach = shape.cavity * (1.0 + 0.9 * smoothstep(0.0, 1.0, foamed));
            let foam = shape.foam * smoothstep(0.3 * shape.collapse, 0.9 * shape.collapse, age) * (1.0 - smoothstep(0.0, 1.0, foamed)).powf(1.5);
            let streaked = 0.2 * smoothstep(0.1, 0.6, foamed);

            let landed = age - shape.drops_from;
            let raining = landed < shape.drops_over + DROP_RING_SECONDS;
            let cells = (3.0 / shape.drops_band.max(1e-3)).clamp(DROP_CELLS.0, DROP_CELLS.1).round();
            let share = if raining { (shape.drops_density / (cells * cells)).min(1.0) } else { 0.0 };
            let apart = (s.position - head_world).length();

            let rain = share * smoothstep(0.0, 0.05, landed) * (1.0 - smoothstep(shape.drops_over, shape.drops_over + DROP_RING_SECONDS, landed));

            let weight = ring_weight(steep + 2.0 * bowl.abs() / reach + 0.05 * rain, foam, apart) * (age / 0.5).min(1.0).powi(3);
            (weight > 0.0).then_some((
                weight,
                [
                    [s.position.x, s.position.z, run, size.max(1e-12)],
                    [0.25 * shape.ring_gravity, c, a, 2.0 * bowl / (reach * reach)],
                    [foam, 1.0 / (reach * reach), streaked, shape.drops_over],
                    [shape.drops_land, 1.0 / (shape.drops_band * shape.drops_band).max(1e-6), landed, if share > 0.0 { cells + 0.999 * share } else { 0.0 }],
                ],
            ))
        })
        .collect();
    live.sort_by(|a, b| b.0.total_cmp(&a.0));

    let cut = live.get(N).map_or(0.0, |l| l.0);
    let mut out = SplashRings { rings: [[0.0; 4]; N], waves: [[0.0; 4]; N], pools: [[0.0; 4]; N], drops: [[0.0; 4]; N] };
    for (i, (weight, [ring, wave, pool, drop])) in live.into_iter().take(N).enumerate() {

        let keep = if cut > 0.0 { (0.5 * (weight / cut - 1.0)).clamp(0.0, 1.0) } else { 1.0 };
        out.rings[i] = [ring[0], ring[1], ring[2], (ring[3] * keep).max(1e-12)];
        out.waves[i] = [wave[0], wave[1], wave[2], wave[3] * keep];
        out.pools[i] = [pool[0] * keep, pool[1], pool[2], pool[3]];

        out.drops[i] = [drop[0], drop[1], drop[2], if drop[3] > 0.0 { drop[3].floor() + drop[3].fract() * keep } else { 0.0 }];
    }
    out
}

fn ring_weight(slope: f32, foam: f32, apart: f32) -> f32 {
    (slope + 0.3 * foam) / (1.0 + apart / 4.0).powi(2)
}

pub const DROP_RING_SECONDS: f32 = 0.45;

pub fn ring_seconds(shape: &SplashShape) -> f32 {
    let a = shape.cavity.max(0.01);
    let k = 3.0 / (RING_EDGE * a);

    let reach = 300.0 * RING_PEAK * shape.wave_source * a * k;
    (shape.ring_release + reach / shape.ring_speed.max(1e-3)).clamp(1.0, super::water_pipeline::RING_SECONDS)
}

pub const FIRE_LIGHT_INTENSITY: f32 = 0.6;
pub const FIRE_LIGHT_RANGE: f32 = 4.5;

fn fire_burns(e: &EffectEmitter) -> bool {
    e.kind == EffectKind::Fire && e.rate > 0.0 && e.scale > 0.0
}

fn fire_light_rest(e: &EffectEmitter) -> Vec3 {
    e.position + e.direction.try_normalize().unwrap_or(Vec3::Y) * (0.35 * e.scale)
}

const FIRE_LIGHT_REACH: f32 = 0.25;

pub fn fire_light(e: &EffectEmitter, time: f64, offset: Vec3, yaw_inv: Quat) -> Option<Light> {
    if !fire_burns(e) {
        return None;
    }
    let v = e.variation.clamped();
    let glow = fire_glow(e, time);
    let f = f64::from(fire_puff_hz(e));
    let seed = emitter_seed(e);
    let flutter = 0.12 * noise1d(time * f * 2.0, seed ^ 0x51) + 0.06 * noise1d(time * f * 4.0, seed ^ 0x52);
    let flicker = 0.8 * (1.0 + v.flicker * (glow - 1.0 + flutter)).clamp(0.25, 2.0);
    let up = e.direction.try_normalize().unwrap_or(Vec3::Y);
    let side = up.any_orthonormal_vector();
    let sway = (side * noise1d(time * f * 0.5, seed ^ 0x53) + up.cross(side) * noise1d(time * f * 0.5, seed ^ 0x54)) * (0.12 * e.scale);
    let rise = up * (0.35 * e.scale * 0.15 * (glow.min(2.5) - 1.0));
    Some(Light {
        position: yaw_inv * (fire_light_rest(e) + sway + rise - offset),
        direction: yaw_inv * -up,
        kind: LightKind::Point,
        color: super::Color3(255, 150, 72, 255),
        intensity: FIRE_LIGHT_INTENSITY * e.scale * e.scale * e.rate.min(1.0) * flicker * v.intensity,
        range: FIRE_LIGHT_RANGE * e.scale.sqrt(),
        cone_angle_deg: 0.0,
        inner_cone_angle_deg: 0.0,
        mask_channel: None,
        shadow_near: None,
        source_radius: 0.22 * e.scale,
        in_level_bake: false,
    })
}

pub fn fire_glare(e: &EffectEmitter, time: f64, offset: Vec3, yaw_inv: Quat) -> Option<super::glare::GlareSource> {
    let light = fire_light(e, time, offset, yaw_inv)?;
    let up = e.direction.try_normalize().unwrap_or(Vec3::Y);
    let middle = yaw_inv * (e.position + up * (0.25 * e.scale) - offset);
    let c = light.color.to_linear();

    let table = super::glare::GlareTable {
        rows: 1,
        cols: 1,
        share: vec![1.0],
        centre: vec![Vec3::ZERO],
        split: Some(super::glare::GlareTableSplit {
            bulb: vec![0.0],
            bulb_centre: vec![Vec3::ZERO],
            lit_centre: vec![Vec3::ZERO],
            lit_spread: vec![0.13 * e.scale],
            bulb_fine: None,
        }),
        air: None,
    };
    Some(super::glare::GlareSource {
        position: middle,
        radiance: Vec3::new(c[0], c[1], c[2]) * light.intensity,
        sides: [1.0; 6],
        rotation: Quat::IDENTITY,
        cone: None,
        centres: None,
        table: Some((std::sync::Arc::new(table), middle)),
        halo_only: false,
        mirror: None,
    })
}

pub fn room_ambient(descs: &[ProbeDesc], world: Vec3) -> Vec3 {
    let l = super::room_light::room_light_at_or_nearest(descs, world);
    Vec3::new(l[0][0], l[0][1], l[0][2]) * 0.282_095
}

pub const UNDER_GROUND: (f32, f32) = (0.5, 2.0);

pub fn room_ambient_now(descs: &[ProbeDesc], world: Vec3, relit: &[f32], ground: Option<&super::ground_map::HeightGrid>) -> Vec3 {
    let ratio = |i: usize| if relit.is_empty() { 1.0 } else { relit.get(i).copied().filter(|k| k.is_finite()).unwrap_or(1.0).max(0.0) };
    let baked = room_ambient(descs, world);
    let lit = || descs.iter().enumerate().filter(|(_, d)| d.room_light.is_some());
    let size = |d: &ProbeDesc| (d.max - d.min).max(Vec3::ZERO).element_product();
    let near = |a: &(usize, &ProbeDesc), b: &(usize, &ProbeDesc)| (a.1.centre - world).length_squared().total_cmp(&(b.1.centre - world).length_squared());
    let inside = lit().filter(|(_, d)| (world - d.min).min(d.max - world).min_element() >= 0.0).min_by(|a, b| size(a.1).total_cmp(&size(b.1)));
    let now = match inside.or_else(|| lit().min_by(near)) {
        Some((i, _)) if !relit.is_empty() => baked * ratio(i),
        _ => baked,
    };

    let held = lit().map(|(_, d)| smoothstep(-super::room_light::ROOM_EDGE, 0.0, (world - d.min).min(d.max - world).min_element())).fold(0.0, f32::max);
    let buried = ground.map_or(0.0, |g| smoothstep(UNDER_GROUND.0, UNDER_GROUND.1, g.at(world.x, world.z) - world.y));
    let open_share = (1.0 - held) * (1.0 - buried);
    let outdoor = |v: u32| descs.iter().filter(|d| d.volume == v).all(|d| !d.has_depth);
    match lit().filter(|(_, d)| outdoor(d.volume)).min_by(near) {
        Some((i, d)) if open_share > 0.0 => {
            let l = d.room_light.unwrap_or_default();
            let open = Vec3::new(l[0][0], l[0][1], l[0][2]) * (0.282_095 * ratio(i));
            now.lerp(open, open_share)
        }
        _ => now,
    }
}

pub fn lamp_in_room(descs: &[ProbeDesc], emitter: Vec3, lamp: Vec3) -> bool {
    let size = |d: &ProbeDesc| (d.max - d.min).max(Vec3::ZERO).element_product();
    let room = descs
        .iter()
        .filter(|d| emitter.cmpge(d.min).all() && emitter.cmple(d.max).all())
        .min_by(|a, b| size(a).total_cmp(&size(b)));
    match room {
        Some(d) => lamp.cmpge(d.min - Vec3::splat(0.25)).all() && lamp.cmple(d.max + Vec3::splat(0.25)).all(),
        None => true,
    }
}

const POOL_ALBEDO: f32 = 0.35;

const FIRE_METER_WEIGHT: f32 = 4.0;

pub fn meter_samples(emitters: &[EffectEmitter], head: Vec3, descs: &[ProbeDesc]) -> Vec<(Vec3, f32, f32)> {
    let size = |d: &ProbeDesc| (d.max - d.min).max(Vec3::ZERO).element_product();
    let mut out = Vec::new();
    for e in emitters {
        let Some(light) = fire_light(e, 0.0, Vec3::ZERO, Quat::IDENTITY) else { continue };

        let intensity = e.variation.clamped().intensity;
        let light = Light {
            intensity: FIRE_LIGHT_INTENSITY * e.scale * e.scale * e.rate.min(1.0) * 0.8 * intensity,
            position: fire_light_rest(e),
            ..light
        };
        let up = e.direction.try_normalize().unwrap_or(Vec3::Y);
        let room = descs
            .iter()
            .filter(|d| e.position.cmpge(d.min).all() && e.position.cmple(d.max).all())
            .min_by(|a, b| size(a).total_cmp(&size(b)));

        if room.is_some_and(|d| head.cmplt(d.min - Vec3::splat(0.5)).any() || head.cmpgt(d.max + Vec3::splat(0.5)).any()) {
            continue;
        }

        let to = e.position + up * (0.3 * e.scale) - head;
        let d2 = to.length_squared().max(1e-4);
        out.push((to / d2.sqrt(), FIRE_RADIANCE * intensity * luminance(flame_colour(0.6)), FIRE_METER_WEIGHT * 0.24 * e.scale * e.scale / d2));
        if (head - e.position).dot(up) <= 0.0 {
            continue;
        }

        let side = up.any_orthonormal_vector();
        let across = up.cross(side);
        let height = (light.position - e.position).length();
        const EDGES: [f32; 7] = [0.0, 0.5, 1.0, 1.6, 2.4, 3.6, 5.2];
        for k in 0..EDGES.len() - 1 {
            let (r0, r1) = (EDGES[k] * height, EDGES[k + 1] * height);
            let area = std::f32::consts::PI * (r1 * r1 - r0 * r0) / 8.0;
            for s in 0..8 {
                let phi = (s as f32 + 0.5) * std::f32::consts::TAU / 8.0;
                let p = e.position + (side * phi.cos() + across * phi.sin()) * f32::midpoint(r0, r1);
                if room.is_some_and(|d| p.cmplt(d.min).any() || p.cmpgt(d.max).any()) {
                    continue;
                }
                let Some((c, toward)) = lamp_at(&light, p) else { continue };
                let v = head - p;
                let dist2 = v.length_squared().max(1e-4);
                let omega = FIRE_METER_WEIGHT * area * v.dot(up).max(0.0) / (dist2 * dist2.sqrt());
                let lum = POOL_ALBEDO * luminance(c) * toward.dot(up).max(0.0);
                if lum > 0.0 && omega > 0.0 {
                    out.push((-v / dist2.sqrt(), lum, omega));
                }
            }
        }
    }
    out
}

fn grad(h: u64) -> Vec3 {

    const G: [[f32; 3]; 12] = [
        [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0], [1.0, -1.0, 0.0], [-1.0, -1.0, 0.0],
        [1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 0.0, -1.0],
        [0.0, 1.0, 1.0], [0.0, -1.0, 1.0], [0.0, 1.0, -1.0], [0.0, -1.0, -1.0],
    ];
    Vec3::from(G[(h % 12) as usize])
}

fn noise3(p: Vec3, seed: u64) -> f32 {
    let i = p.floor();
    let f = p - i;
    let fade = |t: f32| t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
    let (u, v, w) = (fade(f.x), fade(f.y), fade(f.z));
    let (ix, iy, iz) = (i.x as i64, i.y as i64, i.z as i64);
    let corner = |dx: i64, dy: i64, dz: i64| {
        let h = hash64(
            seed ^ ((ix + dx) as u64).wrapping_mul(0x9e3779b97f4a7c15)
                ^ ((iy + dy) as u64).wrapping_mul(0xc2b2ae3d27d4eb4f)
                ^ ((iz + dz) as u64).wrapping_mul(0x165667b19e3779f9),
        );
        grad(h).dot(f - Vec3::new(dx as f32, dy as f32, dz as f32))
    };
    let x00 = lerp(corner(0, 0, 0), corner(1, 0, 0), u);
    let x10 = lerp(corner(0, 1, 0), corner(1, 1, 0), u);
    let x01 = lerp(corner(0, 0, 1), corner(1, 0, 1), u);
    let x11 = lerp(corner(0, 1, 1), corner(1, 1, 1), u);
    lerp(lerp(x00, x10, v), lerp(x01, x11, v), w) * 1.4
}

fn fbm(p: Vec3, octaves: u32, seed: u64) -> f32 {
    let mut sum = 0.0;
    let mut amp = 0.5;
    let mut q = p;
    for o in 0..octaves {
        sum += amp * noise3(q, seed.wrapping_add(u64::from(o) * 7919));
        q = q * 2.03 + Vec3::new(17.1, 9.3, 3.7);
        amp *= 0.5;
    }
    sum
}

fn billows(p: Vec3, octaves: u32, seed: u64) -> f32 {
    let mut sum = 0.0;
    let mut amp = 0.5;
    let mut q = p;
    for o in 0..octaves {
        sum += amp * noise3(q, seed.wrapping_add(u64::from(o) * 7919)).abs();
        q = q * 2.03 + Vec3::new(17.1, 9.3, 3.7);
        amp *= 0.5;
    }
    sum
}

const VOL: usize = 64;
const VOL_DEPTH: usize = 48;

const SMOKE_SIGMA: f32 = 6.0;

const LUMPS: [([f32; 3], f32); 8] = [
    ([0.0, -0.06, 0.0], 0.55),
    ([0.36, 0.04, 0.1], 0.38),
    ([-0.28, 0.2, -0.1], 0.44),
    ([0.12, 0.38, 0.0], 0.34),
    ([-0.22, -0.32, 0.16], 0.3),
    ([0.3, -0.3, -0.18], 0.26),
    ([-0.42, -0.08, 0.2], 0.24),
    ([0.02, 0.1, 0.38], 0.3),
];

fn smoke_density(p: Vec3, age: f32) -> f32 {
    let grow = 1.0 + 0.1 * age;
    let w = p * 1.4 + Vec3::splat(age * 0.5);
    let bent = p + Vec3::new(fbm(w, 2, 41), fbm(w + Vec3::splat(7.3), 2, 43), fbm(w + Vec3::splat(13.1), 2, 47)) * 0.22;
    let mut body = 0.0f32;
    for (c, r) in LUMPS {
        let q = (bent - Vec3::from(c) * grow).length() / (r * grow);
        let s = (1.0 - q * q).max(0.0);
        body += s * s;
    }
    let body = body.min(1.0);
    let roll = billows(p * 3.0 + Vec3::new(0.0, -1.1 * age, 0.8 * age), 4, 11);
    let d = body - (0.16 + 0.32 * age) + (roll - 0.25) * (1.1 + 0.7 * age) * smoothstep(0.0, 0.3, body);
    let edge = 1.0 - smoothstep(0.86, 0.98, p.abs().max_element());
    smoothstep(0.0, 0.22, d) * (1.0 - 0.75 * age) * edge
}

fn smoke_frame(k: u32) -> [Vec<u8>; 2] {
    let age = k as f32 / (SMOKE_FRAMES - 1) as f32;
    let (n, nz) = (VOL, VOL_DEPTH);
    let at = |x: usize, y: usize, z: usize| (z * n + y) * n + x;
    let mut density = vec![0.0f32; n * n * nz];
    for z in 0..nz {
        for y in 0..n {
            for x in 0..n {
                let p = Vec3::new(
                    (x as f32 + 0.5) / n as f32 * 2.0 - 1.0,
                    1.0 - (y as f32 + 0.5) / n as f32 * 2.0,
                    (z as f32 + 0.5) / nz as f32 * 2.0 - 1.0,
                );
                density[at(x, y, z)] = smoke_density(p, age);
            }
        }
    }

    let (dx, dz) = (2.0 / n as f32, 2.0 / nz as f32);
    let mut depth = vec![[0.0f32; 6]; n * n * nz];
    for z in 0..nz {
        for y in 0..n {
            let mut run = 0.0;
            for x in (0..n).rev() {
                let d = density[at(x, y, z)];
                depth[at(x, y, z)][0] = (run + 0.5 * d) * dx;
                run += d;
            }
            run = 0.0;
            for x in 0..n {
                let d = density[at(x, y, z)];
                depth[at(x, y, z)][3] = (run + 0.5 * d) * dx;
                run += d;
            }
        }
        for x in 0..n {
            let mut run = 0.0;
            for y in 0..n {
                let d = density[at(x, y, z)];
                depth[at(x, y, z)][1] = (run + 0.5 * d) * dx;
                run += d;
            }
            run = 0.0;
            for y in (0..n).rev() {
                let d = density[at(x, y, z)];
                depth[at(x, y, z)][4] = (run + 0.5 * d) * dx;
                run += d;
            }
        }
    }
    for y in 0..n {
        for x in 0..n {
            let mut run = 0.0;
            for z in (0..nz).rev() {
                let d = density[at(x, y, z)];
                depth[at(x, y, z)][2] = (run + 0.5 * d) * dz;
                run += d;
            }
            run = 0.0;
            for z in 0..nz {
                let d = density[at(x, y, z)];
                depth[at(x, y, z)][5] = (run + 0.5 * d) * dz;
                run += d;
            }
        }
    }

    let mut light = vec![[0.0f32; 6]; n * n];
    let mut alpha = vec![0.0f32; n * n];
    for y in 0..n {
        for x in 0..n {
            let mut seen = 1.0f32;
            let mut sum = [0.0f32; 6];
            let mut through = [0.0f32; 6];
            for z in (0..nz).rev() {
                let v = at(x, y, z);
                let a = 1.0 - (-SMOKE_SIGMA * density[v] * dz).exp();
                for s in 0..6 {
                    let t = (-SMOKE_SIGMA * depth[v][s]).exp();
                    sum[s] += seen * a * t;
                    through[s] += t;
                }
                seen *= 1.0 - a;
            }
            let o = y * n + x;
            alpha[o] = 1.0 - seen;
            for s in 0..6 {

                light[o][s] = if alpha[o] > 1.0 / 512.0 { sum[s] / alpha[o] } else { through[s] / nz as f32 };
            }
        }
    }

    let m = ATLAS_SIZE as usize;
    let mut front = vec![0u8; m * m * 4];
    let mut back = vec![0u8; m * m * 4];
    let sample = |fx: f32, fy: f32, get: &dyn Fn(usize) -> f32| -> f32 {
        let x = (fx * n as f32 - 0.5).clamp(0.0, (n - 1) as f32);
        let y = (fy * n as f32 - 0.5).clamp(0.0, (n - 1) as f32);
        let (x0, y0) = (x.floor() as usize, y.floor() as usize);
        let (x1, y1) = ((x0 + 1).min(n - 1), (y0 + 1).min(n - 1));
        let (tx, ty) = (x - x0 as f32, y - y0 as f32);
        lerp(lerp(get(y0 * n + x0), get(y0 * n + x1), tx), lerp(get(y1 * n + x0), get(y1 * n + x1), tx), ty)
    };
    let byte = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
    for y in 0..m {
        for x in 0..m {
            let (fx, fy) = ((x as f32 + 0.5) / m as f32, (y as f32 + 0.5) / m as f32);
            let a = byte(sample(fx, fy, &|i| alpha[i]));
            let o = (y * m + x) * 4;
            for c in 0..3 {
                front[o + c] = byte(sample(fx, fy, &|i| light[i][c]));
                back[o + c] = byte(sample(fx, fy, &|i| light[i][c + 3]));
            }
            front[o + 3] = a;
            back[o + 3] = a;
        }
    }
    [front, back]
}

fn fire_frame(k: u32) -> Vec<u8> {
    let n = ATLAS_SIZE as usize;
    let time = k as f32 * FLAME_SECONDS / FIRE_FRAMES as f32;
    let beat_hz = puff_book_hz();
    let mut out = vec![0u8; n * n * 4];
    for y in 0..n {
        for x in 0..n {
            let px = (x as f32 + 0.5) / n as f32 * 2.0 - 1.0;
            let py = 1.0 - (y as f32 + 0.5) / n as f32 * 2.0;
            let v = f32::midpoint(py, 1.0);
            let h = ((v - FLAME_BASE) / (1.0 - FLAME_BASE)).max(0.0);

            let flight = (1.0 + FLAME_W1 * h / FLAME_W0).ln() / FLAME_W1;
            let born = flight - time;

            let beat = (std::f32::consts::TAU * beat_hz * (time - flight)).sin();
            let sway = fbm(Vec3::new(px * 0.9 + 3.0, born * 1.6, time * 0.5), 3, 71) * 0.35 * h.powf(1.2);
            let sx = px + sway;
            let neck = 1.0 - 0.25 * (1.0 - smoothstep(0.0, 0.18, h));
            let swell = 1.0 + PUFF_SWELL * beat * smoothstep(0.1, 0.4, h);
            let half = (0.58 * (1.0 - h).max(0.0).sqrt() + 0.02) * neck * swell;
            let across = (1.0 - (sx / half) * (sx / half)).max(0.0);
            let lick = fbm(Vec3::new(sx * 2.6, born * 4.5, time * 0.8 + 20.0), 3, 83);
            let pinch = PUFF_PINCH * (-beat).max(0.0).powi(2) * smoothstep(0.35, 0.65, h);
            let body = across.sqrt() * (1.0 - 0.6 * h) + lick * (0.5 + 0.5 * h) - 0.18 - 0.35 * h - pinch;
            let lift = v - 0.06 * (0.5 + fbm(Vec3::new(px * 4.0, time * 1.5, 7.0), 2, 91));

            let inside = (1.0 - smoothstep(0.75, 0.95, px.abs())) * (1.0 - smoothstep(0.8, 0.97, v));
            let a = smoothstep(0.0, 0.1, body) * smoothstep(0.0, 0.08, lift) * inside;
            let root = 1.0 - smoothstep(0.0, 0.1, h);
            let heat = smoothstep(0.0, 1.0, body) * (0.6 + 0.4 * smoothstep(0.0, 0.15, h)) * (1.0 - 0.12 * root);
            let o = (y * n + x) * 4;
            out[o] = (heat * 255.0).round() as u8;
            out[o + 1] = ((body / FLAME_DEPTH).clamp(0.0, 1.0) * 255.0).round() as u8;
            out[o + 2] = (root * a * 255.0).round() as u8;
            out[o + 3] = (a * 255.0).round() as u8;
        }
    }
    out
}

const FLAME_DEPTH: f32 = 0.9;

fn cells(x: f32, y: f32, seed: u64) -> (f32, f32, u64) {
    let (ix, iy) = (x.floor() as i64, y.floor() as i64);
    let (mut f1, mut f2, mut id) = (f32::MAX, f32::MAX, 0u64);
    for dy in -1..=1 {
        for dx in -1..=1 {
            let (cx, cy) = (ix + dx, iy + dy);
            let h = hash64(seed ^ (cx as u64).wrapping_mul(0x9e3779b97f4a7c15) ^ (cy as u64).wrapping_mul(0xc2b2ae3d27d4eb4f));
            let jx = (h >> 40) as f32 / (1u64 << 24) as f32;
            let jy = ((h >> 16) & 0xff_ffff) as f32 / (1u64 << 24) as f32;
            let d = (x - (cx as f32 + 0.15 + 0.7 * jx)).hypot(y - (cy as f32 + 0.15 + 0.7 * jy));
            if d < f1 {
                (f2, f1, id) = (f1, d, h);
            } else if d < f2 {
                f2 = d;
            }
        }
    }
    (f1, f2, id)
}

fn coals_frame() -> Vec<u8> {
    let n = ATLAS_SIZE as usize;
    let mut out = vec![0u8; n * n * 4];
    for y in 0..n {
        for x in 0..n {
            let px = (x as f32 + 0.5) / n as f32 * 2.0 - 1.0;
            let py = 1.0 - (y as f32 + 0.5) / n as f32 * 2.0;
            let wobble = fbm(Vec3::new(px * 2.5, py * 2.5, 3.0), 3, 101);
            let rim = px.hypot(py) + 0.12 * wobble;
            let a = 1.0 - smoothstep(0.7, 0.88, rim);
            let (f1, f2, id) = cells(px * 4.5 + 0.37 * wobble, py * 4.5, 103);
            let dome = (1.0 - f1 / 0.75).clamp(0.0, 1.0);
            let crack = 1.0 - smoothstep(0.0, 0.3, f2 - f1);
            let core = 1.0 - smoothstep(0.1, 0.72, rim);
            let lump = (id >> 8) as f32 / (1u64 << 56) as f32;
            let grain = 0.5 + fbm(Vec3::new(px * 14.0, py * 14.0, 9.0), 2, 107);

            let skin = smoothstep(0.35, 0.65, 0.5 + fbm(Vec3::new(px * 9.0, py * 9.0, 4.0), 3, 109));
            let glowing = smoothstep(0.15, 0.6, lump) * (0.35 + 0.65 * dome) * skin;
            let heat = (core * (0.35 * crack + 0.75 * glowing) * (0.8 + 0.4 * grain)).clamp(0.0, 0.85);
            let ash = smoothstep(0.4, 0.8, rim + 0.25 * (grain - 0.5));
            let albedo = lerp(0.04 + 0.12 * dome, 0.3 + 0.15 * grain, ash);
            let o = (y * n + x) * 4;
            out[o] = (heat * 255.0).round() as u8;
            out[o + 1] = (albedo.clamp(0.0, 1.0) * 255.0).round() as u8;
            out[o + 2] = (id >> 32) as u8;
            out[o + 3] = (a * 255.0).round() as u8;
        }
    }
    out
}

fn dot_frame(sharpness: f32) -> Vec<u8> {
    let n = ATLAS_SIZE as usize;
    let mut out = vec![0u8; n * n * 4];
    for y in 0..n {
        for x in 0..n {
            let px = (x as f32 + 0.5) / n as f32 * 2.0 - 1.0;
            let py = 1.0 - (y as f32 + 0.5) / n as f32 * 2.0;
            let r2 = px * px + py * py;
            let a = (-r2 * sharpness).exp() * smoothstep(1.0, 0.9, r2.sqrt());
            let o = (y * n + x) * 4;
            out[o] = 255;
            out[o + 1] = 255;
            out[o + 2] = 255;
            out[o + 3] = (a * 255.0).round() as u8;
        }
    }
    out
}

pub fn atlas_layers() -> Vec<Vec<u8>> {
    fn on_every_core<T: Send>(count: u32, make: impl Fn(u32) -> T + Sync) -> Vec<T> {
        let threads = std::thread::available_parallelism().map_or(4, std::num::NonZero::get).clamp(1, count.max(1) as usize);
        let mut made: Vec<Option<T>> = (0..count).map(|_| None).collect();
        let make = &make;
        std::thread::scope(|s| {
            let chunk = (count as usize).div_ceil(threads).max(1);
            for (c, part) in made.chunks_mut(chunk).enumerate() {
                s.spawn(move || {
                    for (i, slot) in part.iter_mut().enumerate() {
                        *slot = Some(make((c * chunk + i) as u32));
                    }
                });
            }
        });
        made.into_iter().map(|f| f.expect("every frame made")).collect()
    }
    let smoke = on_every_core(SMOKE_FRAMES, smoke_frame);
    let fire = on_every_core(FIRE_FRAMES, fire_frame);
    let mut layers = Vec::with_capacity(ATLAS_LAYERS as usize);
    layers.extend(smoke.iter().map(|[f, _]| f.clone()));
    layers.extend(smoke.into_iter().map(|[_, b]| b));
    layers.extend(fire);
    layers.push(dot_frame(6.0));
    layers.push(dot_frame(3.0));
    layers.push(coals_frame());
    for variant in 0..CROWN_VARIANTS {
        layers.extend((0..CROWN_FRAMES).map(|stage| crown_frame(variant, stage)));
    }
    for variant in 0..JET_VARIANTS {
        layers.extend((0..JET_FRAMES).map(|stage| jet_frame(variant, stage)));
    }
    for variant in 0..SPRAY_VARIANTS {
        layers.extend((0..SPRAY_FRAMES).map(|stage| spray_frame(variant, stage)));
    }
    for variant in 0..FROTH_VARIANTS {
        layers.extend((0..FROTH_FRAMES).map(|stage| froth_frame(variant, stage)));
    }
    for variant in 0..FLASH_VARIANTS {
        layers.extend((0..FLASH_FRAMES).map(|form| flash::lobe_frame(variant, form)));
    }
    layers.extend((0..CORE_VARIANTS).map(flash::core_frame));
    layers.extend((0..MUD_VARIANTS).map(mud_splash::lump_frame));
    layers
}

static CUTOUT_SHARES: std::sync::OnceLock<Vec<f32>> = std::sync::OnceLock::new();

fn cutout_shares_of(cut: &[[f32; 16]]) -> Vec<f32> {
    cut.iter()
        .map(|c| {

            let area: f32 = (0..8).map(|k| (c[2 * k] * c[2 * ((k + 1) % 8) + 1] - c[2 * ((k + 1) % 8)] * c[2 * k + 1]) * 0.5).sum();
            (area.abs() / 4.0).clamp(0.0, 1.0)
        })
        .collect()
}

pub fn books() -> &'static [super::effect_asset::BookInfo] {
    use super::effect_asset::{BookInfo, Channels};
    static BOOKS: std::sync::OnceLock<Vec<BookInfo>> = std::sync::OnceLock::new();
    BOOKS.get_or_init(|| {
        let shares = CUTOUT_SHARES.get_or_init(|| cutout_shares_of(&cutouts(&atlas_layers())));
        let layer_bytes = (0..ATLAS_MIPS).map(|m| u64::from((ATLAS_SIZE >> m).max(1)).pow(2) * 4).sum::<u64>();
        let book = |name, first: u32, frames: u32, variants: u32, channels, layers: u32| BookInfo {
            name,
            first_layer: first,
            frames,
            variants,
            channels,
            cutout_share: shares[first as usize..(first + frames * variants) as usize].iter().sum::<f32>() / (frames * variants) as f32,
            bytes: u64::from(layers) * layer_bytes,
        };
        vec![
            book("smoke", 0, SMOKE_FRAMES, 1, Channels::Sixway, 2 * SMOKE_FRAMES),
            book("flame", FIRE_FIRST, FIRE_FRAMES, 1, Channels::HeatCover, FIRE_FRAMES),
            book("ember", EMBER_LAYER, 1, 1, Channels::Alpha, 1),
            book("mote", MOTE_LAYER, 1, 1, Channels::Alpha, 1),
            book("coals", COALS_LAYER, 1, 1, Channels::Bed, 1),
            book("crown", CROWN_LAYER, CROWN_FRAMES, CROWN_VARIANTS, Channels::Alpha, CROWN_FRAMES * CROWN_VARIANTS),
            book("jet", JET_LAYER, JET_FRAMES, JET_VARIANTS, Channels::Alpha, JET_FRAMES * JET_VARIANTS),
            book("spray", SPRAY_LAYER, SPRAY_FRAMES, SPRAY_VARIANTS, Channels::Alpha, SPRAY_FRAMES * SPRAY_VARIANTS),
            book("froth", FROTH_LAYER, FROTH_FRAMES, FROTH_VARIANTS, Channels::Alpha, FROTH_FRAMES * FROTH_VARIANTS),
            book("muzzle_flash", FLASH_LAYER, FLASH_FRAMES, FLASH_VARIANTS, Channels::HeatCover, FLASH_FRAMES * FLASH_VARIANTS),
            book("muzzle_core", CORE_LAYER, 1, CORE_VARIANTS, Channels::HeatCover, CORE_VARIANTS),
        ]
    })
}

struct Dabs {
    clear: Vec<f32>,
}

impl Dabs {
    fn new() -> Self {
        Self { clear: vec![1.0; (ATLAS_SIZE * ATLAS_SIZE) as usize] }
    }

    fn dab(&mut self, cx: f32, cy: f32, r: f32, a: f32) {
        let n = ATLAS_SIZE as usize;
        let soft = 1.0 / n as f32;
        let r = r.max(0.5 * soft);
        let reach = r + soft;
        let (x0, x1) = (((cx - reach) * n as f32).floor().max(0.0) as usize, ((cx + reach) * n as f32).ceil().clamp(0.0, n as f32) as usize);
        let (y0, y1) = (((cy - reach) * n as f32).floor().max(0.0) as usize, ((cy + reach) * n as f32).ceil().clamp(0.0, n as f32) as usize);
        for y in y0..y1 {
            for x in x0..x1 {
                let (px, py) = ((x as f32 + 0.5) / n as f32, (y as f32 + 0.5) / n as f32);
                let d = ((px - cx) * (px - cx) + (py - cy) * (py - cy)).sqrt();

                let cover = a * (1.0 - smoothstep(r - soft, r + soft, d));
                let c = &mut self.clear[y * n + x];
                *c = c.min(1.0 - cover);
            }
        }
    }

    fn finish(self, film: impl Fn(f32, f32) -> f32, edge: impl Fn(f32, f32) -> f32, on_its_side: bool) -> Vec<u8> {
        let n = ATLAS_SIZE as usize;
        let mut out = vec![0u8; n * n * 4];
        for y in 0..n {
            for x in 0..n {

                let (ux, uy) = if on_its_side { (y, n - 1 - x) } else { (x, y) };
                let (px, py) = ((ux as f32 + 0.5) / n as f32, (uy as f32 + 0.5) / n as f32);
                let a = (1.0 - self.clear[uy * n + ux] * (1.0 - film(px, py).clamp(0.0, 1.0))) * edge(px, py);
                out[(y * n + x) * 4..(y * n + x) * 4 + 4].copy_from_slice(&[255, 255, 255, (a.clamp(0.0, 1.0) * 255.0).round() as u8]);
            }
        }
        out
    }
}

fn crown_frame(variant: u32, stage: u32) -> Vec<u8> {
    let seed = 0xc0ffee ^ ((u64::from(variant) + 1) << 20);
    let stage = stage.min(2);
    let mut d = Dabs::new();
    let base = 0.98f32;

    let rim = move |x: f32| {
        let side = (2.0 * x - 1.0) * (2.0 * x - 1.0);
        0.38 + 0.1 * noise1(x * 3.3, seed) + 0.045 * noise1(x * 9.0, seed ^ 7) + 0.07 * side
    };

    for c in 0..110 {
        let x = 0.01 + 0.98 * (c as f32 + 0.5) / 110.0;
        let u = uniforms(hash64(seed ^ 0x71 ^ (c as u64).wrapping_mul(0x9e3779b97f4a7c15)));

        let thick = 0.024 + 0.011 * noise1(x * 9.0, seed ^ 0x33);
        match stage {
            0 => d.dab(x, rim(x), thick, 0.95),

            1 => {
                if noise1(x * 9.0, seed ^ 0x33) > -0.4 {
                    d.dab(x, rim(x) + 0.012, 0.9 * thick, 0.95);
                }
            }

            _ => {
                if c % 4 == 0 {
                    d.dab(x + 0.012 * (u[0] - 0.5), rim(x) + 0.03 + 0.05 * u[1], lerp(0.011, 0.02, u[2]), 0.95);
                }
            }
        }
    }

    for f in 0..7u64 {
        let u = uniforms(hash64(seed ^ 0xf1 ^ f.wrapping_mul(0x632be59bd9b4e019)));
        let x0 = 0.06 + 0.88 * (f as f32 + 0.15 + 0.7 * u[0]) / 7.0;
        let lean = (u[1] - 0.5) * 0.7;
        let long = lerp(0.07, 0.2, u[2] * u[2]) * [1.0f32, 1.35, 1.6][stage as usize];
        let wide = lerp(0.012, 0.02, u[3]);
        let y0 = rim(x0);
        let steps = 28;
        for k in 0..=steps {
            let t = k as f32 / steps as f32;
            let (x, y) = (x0 + lean * long * t, y0 - long * t * (1.0 - 0.25 * t));
            let neck = match stage {
                0 => 1.0,

                1 => 0.35 + 0.65 * (t * std::f32::consts::TAU + 6.0 * u[4]).cos().abs(),

                _ => {
                    if (t * 2.0 + u[4]).fract() < 0.22 {
                        1.4
                    } else {
                        0.0
                    }
                }
            };
            if neck > 0.0 {
                d.dab(x, y, wide * (1.0 - 0.45 * t) * neck, 0.95);
            }
        }
        let tip = (x0 + lean * long, y0 - long * 0.75);
        d.dab(tip.0, tip.1 - 0.008, wide * 1.4, 0.97);

        for k in 0..2 * stage + 2 {
            let v = uniforms(hash64(seed ^ 0xd0 ^ (f * 8 + u64::from(k)).wrapping_mul(0x8cb92ba72f3d8dd7)));
            d.dab(tip.0 + (v[0] - 0.5) * 0.16, (tip.1 - 0.03 - 0.22 * v[1] * v[1]).max(0.03), lerp(0.006, 0.012, v[2]), 0.95);
        }
    }

    let film = move |x: f32, y: f32| -> f32 {
        if y < rim(x) || y > base {
            return 0.0;
        }
        let up = 1.0 - (y - rim(x)) / (base - rim(x));
        let warp = 0.5 * noise1(y * 4.0, seed ^ 0x52);
        let streak = 0.5 + 0.5 * noise1(x * 21.0 + warp, seed ^ 0x51);
        let fine = 0.5 + 0.5 * noise1(x * 55.0 + 2.0 * warp, seed ^ 0x53);
        let under_rim = smoothstep(0.75, 1.0, up);

        let body = (0.13 + 0.42 * streak * streak * streak + 0.16 * fine * fine + 0.22 * under_rim) * (0.12 + 0.88 * smoothstep(0.0, 0.65, up));
        match stage {
            0 => body,

            1 => 0.8 * body * smoothstep(-0.25, 0.2, noise3(Vec3::new(x * 8.0, y * 6.0, 3.7), seed ^ 0x40) + 0.6 * (1.0 - up) - 0.25),

            _ => 0.55 * body * smoothstep(0.05, 0.4, noise3(Vec3::new(x * 12.0, y * 5.0, 9.1), seed ^ 0x41) + 0.35 * (1.0 - up) - 0.2),
        }
    };

    d.finish(film, |x, y| smoothstep(0.0, 0.24, x) * smoothstep(1.0, 0.76, x) * smoothstep(0.0, 0.03, y) * smoothstep(1.0, 0.96, y), true)
}

fn spray_frame(variant: u32, stage: u32) -> Vec<u8> {
    let seed = 0x5b7a1 ^ ((u64::from(variant) + 1) << 20);
    let stage = stage.min(2) as usize;
    let mut d = Dabs::new();
    let foot = (0.5f32, 0.975f32);
    let count = [340u64, 340, 170][stage];
    for i in 0..count {
        let u = uniforms(hash64(seed ^ (i + 1).wrapping_mul(0x8cb92ba72f3d8dd7)));

        let angle = (u[0] + u[1] - 1.0) * 0.8;
        let far = [0.8f32, 0.93, 0.95][stage] * u[2].powf(0.55) * (1.0 - 0.3 * angle * angle);
        let (sx, sy) = (angle.sin(), -angle.cos());
        let (x, y) = (foot.0 + sx * far, foot.1 + sy * far);

        let r = lerp(0.0035, 0.012, u[3] * u[3] * u[3]) * (1.15 - 0.4 * far) * [1.0f32, 1.0, 0.8][stage];
        if stage == 0 {
            let long = lerp(0.03, 0.09, u[4]) * (0.4 + 0.6 * far);
            for k in 0..6 {
                let t = k as f32 / 5.0;
                d.dab(x - sx * long * t, y - sy * long * t, r * (1.0 - 0.55 * t), 0.92);
            }
        } else {
            d.dab(x, y + 0.03 * u[5] * (stage as f32 - 1.0), r, 0.92);
        }
    }
    let haze = move |x: f32, y: f32| -> f32 {
        let (dx, dy) = (x - foot.0, foot.1 - y);
        let far = (dx * dx + dy * dy).sqrt();
        let angle = dx.atan2(dy.max(1e-4));
        let cone = (1.0 - (angle / 0.75) * (angle / 0.75)).max(0.0) * smoothstep(0.0, 0.08, far) * (1.0 - smoothstep(0.25, 0.8, far));
        [0.2f32, 0.1, 0.0][stage] * cone * (0.55 + 0.45 * noise3(Vec3::new(x * 9.0, y * 9.0, 0.7), seed ^ 0x11))
    };
    d.finish(haze, |x, y| smoothstep(0.0, 0.04, x) * smoothstep(1.0, 0.96, x) * smoothstep(0.0, 0.03, y) * smoothstep(1.0, 0.985, y), false)
}

fn froth_frame(variant: u32, stage: u32) -> Vec<u8> {
    let seed = 0xf207 ^ ((u64::from(variant) + 1) << 20);
    let stage = stage.min(2) as usize;
    let mut d = Dabs::new();
    let base = 0.98f32;

    let top = move |x: f32| {
        let side = (2.0 * x - 1.0) * (2.0 * x - 1.0);
        (0.34 + 0.26 * noise1(x * 3.7, seed) + 0.12 * noise1(x * 10.0, seed ^ 3) + 0.16 * side).clamp(0.06, 0.85)
    };

    for i in 0..[60u64, 120, 150][stage] {
        let u = uniforms(hash64(seed ^ 0xbead ^ (i + 1).wrapping_mul(0x632be59bd9b4e019)));
        let x = 0.08 + 0.84 * u[0];
        let reach = base - top(x);
        let h = reach * lerp([0.85f32, 0.5, 0.15][stage], 1.25, u[2]);
        d.dab(x, (base - h).max(0.02), lerp(0.005, 0.014, u[3] * u[3]), 0.92);
    }
    let body = move |x: f32, y: f32| -> f32 {
        if y > base {
            return 0.0;
        }
        let up = (base - y) / (base - top(x));
        let lives = 1.0 - smoothstep(0.65, 1.0, up);
        let warp = 0.6 * noise1(y * 3.0, seed ^ 0x21);
        let streak = 0.5 + 0.5 * noise3(Vec3::new(x * 22.0 + warp, y * 4.0, 1.9), seed ^ 0x22);
        let lumps = 0.5 + 0.5 * noise3(Vec3::new(x * 6.0, y * 6.0, 4.3), seed ^ 0x23);
        let dense = [0.85f32 - 0.3 * up, 0.72 - 0.3 * up, 0.5 - 0.2 * up][stage];

        let a = lives * dense * (0.25 + 0.4 * streak + 0.5 * lumps) * smoothstep(0.0, 0.3, up);
        match stage {
            0 => a,
            1 => a * smoothstep(-0.3, 0.2, noise3(Vec3::new(x * 7.0, y * 6.0, 7.7), seed ^ 0x24) + 0.5 - 0.8 * up),
            _ => a * smoothstep(0.0, 0.35, noise3(Vec3::new(x * 9.0 + warp, y * 4.0, 2.2), seed ^ 0x25) + 0.25 - 0.35 * up),
        }
    };
    d.finish(body, |x, y| smoothstep(0.0, 0.4, x) * smoothstep(1.0, 0.6, x) * smoothstep(0.0, 0.03, y) * smoothstep(1.0, 0.97, y), false)
}

fn jet_frame(variant: u32, stage: u32) -> Vec<u8> {
    let seed = 0x1e7 ^ ((u64::from(variant) + 1) << 20);
    let stage = stage.min(2);
    let u = uniforms(hash64(seed));
    let mut d = Dabs::new();
    let swell = [0.12f32, 0.5, 1.0][stage as usize];
    let waves = lerp(3.0, 4.5, u[0]);
    let bend = |t: f32| 0.5 + 0.04 * noise1(t * 3.0, seed ^ 3) * t;
    let steps = 180;
    for k in 0..=steps {
        let t = k as f32 / steps as f32;
        let y = 0.97 - 0.84 * t;
        let wave = (t * waves * std::f32::consts::TAU + 6.0 * u[1]).cos();

        let body = 0.085 * (1.0 - 0.6 * t) * (1.0 + 0.7 * (1.0 - smoothstep(0.0, 0.1, t)));
        let wide = match stage {
            0 | 1 => body * (1.0 + swell * 0.8 * wave).max(0.15),

            _ => {
                if t > 0.1 {
                    body * (0.2 + 1.3 * wave).max(0.0)
                } else {
                    body * (1.0 - 7.0 * t).max(0.0)
                }
            }
        };
        if wide > 0.004 {
            d.dab(bend(t), y, wide, 0.93);
        }
    }

    d.dab(bend(1.0), 0.11, lerp(0.04, 0.055, u[2]), 0.95);
    for k in 0..(3 * u64::from(stage)) {
        let v = uniforms(hash64(seed ^ 0x5a7 ^ k.wrapping_mul(0x9e3779b97f4a7c15)));
        d.dab(0.5 + (v[0] - 0.5) * 0.16, lerp(0.15, 0.85, v[1]), lerp(0.008, 0.018, v[2]), 0.9);
    }
    d.finish(|_, _| 0.0, |x, y| smoothstep(0.0, 0.05, x) * smoothstep(1.0, 0.95, x) * smoothstep(0.0, 0.03, y) * smoothstep(1.0, 0.94, y), false)
}

fn outline(alpha: &[&[u8]]) -> [[f32; 2]; 8] {
    let n = ATLAS_SIZE as usize;
    let margin = 2.0 * 2.0 / n as f32;
    let (mut lo, mut hi) = ([f32::MAX; 4], [f32::MIN; 4]);
    for y in 0..n {
        for x in 0..n {
            if !alpha.iter().any(|a| a[(y * n + x) * 4 + 3] > 0) {
                continue;
            }
            for (cx, cy) in [(x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1)] {
                let (qx, qy) = (cx as f32 / n as f32 * 2.0 - 1.0, 1.0 - cy as f32 / n as f32 * 2.0);
                for (k, v) in [qx, qy, qx + qy, qx - qy].into_iter().enumerate() {
                    lo[k] = lo[k].min(v);
                    hi[k] = hi[k].max(v);
                }
            }
        }
    }
    if lo[0] > hi[0] {

        return [[0.0; 2]; 8];
    }
    let diag = std::f32::consts::SQRT_2;
    let (xmin, ymin, smin, dmin) = ((lo[0] - margin).max(-1.0), (lo[1] - margin).max(-1.0), lo[2] - margin * diag, lo[3] - margin * diag);
    let (xmax, ymax, smax, dmax) = ((hi[0] + margin).min(1.0), (hi[1] + margin).min(1.0), hi[2] + margin * diag, hi[3] + margin * diag);
    let cx = |x: f32| x.clamp(xmin, xmax);
    let cy = |y: f32| y.clamp(ymin, ymax);
    [
        [cx(smin - ymin), ymin],
        [cx(dmax + ymin), ymin],
        [xmax, cy(xmax - dmax)],
        [xmax, cy(smax - xmax)],
        [cx(smax - ymax), ymax],
        [cx(dmin + ymax), ymax],
        [xmin, cy(xmin - dmin)],
        [xmin, cy(smin - xmin)],
    ]
}

fn flame_smear(frame: &[u8], dt: f32) -> Vec<u8> {
    let n = ATLAS_SIZE as usize;
    let mut out = vec![0u8; n * n * 4];
    let row_of = |v: f32| ((1.0 - v) * n as f32 - 0.5).round().clamp(0.0, (n - 1) as f32) as usize;
    for y in 0..n {
        let v = 1.0 - (y as f32 + 0.5) / n as f32;
        let w = flame_v(v, -dt);
        let (top, bottom) = (row_of(v.max(w)), row_of(v.min(w)));
        for x in 0..n {
            if frame[(y * n + x) * 4 + 3] > 0 {
                for r in top..=bottom {
                    out[(r * n + x) * 4 + 3] = 255;
                }
            }
        }
    }
    out
}

pub fn cutouts(layers: &[Vec<u8>]) -> Vec<[f32; 16]> {
    let dt = FLAME_SECONDS / FIRE_FRAMES as f32;
    (0..layers.len() as u32)
        .map(|l| {
            let next = match l {
                l if l < SMOKE_FRAMES - 1 => Some(l + 1),
                l if (FIRE_FIRST..FIRE_FIRST + FIRE_FRAMES - 1).contains(&l) => Some(l + 1),

                l if (CROWN_LAYER..FLASH_LAYER).contains(&l) && (l - CROWN_LAYER) % CROWN_FRAMES < CROWN_FRAMES - 1 => Some(l + 1),
                _ => None,
            };
            let smeared: Vec<Vec<u8>> = match next {
                Some(m) if (FIRE_FIRST..FIRE_FIRST + FIRE_FRAMES).contains(&l) => vec![flame_smear(&layers[l as usize], dt), flame_smear(&layers[m as usize], -dt)],
                _ => Vec::new(),
            };
            let mut with: Vec<&[u8]> = vec![&layers[l as usize]];
            if let Some(m) = next {
                with.push(&layers[m as usize]);
            }
            with.extend(smeared.iter().map(std::vec::Vec::as_slice));
            let o = outline(&with);
            let mut out = [0.0f32; 16];
            for (k, c) in o.iter().enumerate() {
                out[2 * k] = c[0];
                out[2 * k + 1] = c[1];
            }
            out
        })
        .collect()
}

fn half_mip(src: &[u8], size: usize) -> Vec<u8> {
    let h = (size / 2).max(1);
    let mut out = vec![0u8; h * h * 4];
    for y in 0..h {
        for x in 0..h {
            let mut sum = [0.0f32; 4];
            let mut weight = 0.0;
            for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                let o = ((2 * y + dy).min(size - 1) * size + (2 * x + dx).min(size - 1)) * 4;
                let a = f32::from(src[o + 3]);
                for c in 0..3 {
                    sum[c] += f32::from(src[o + c]) * (a + 1.0);
                }
                sum[3] += a;
                weight += a + 1.0;
            }
            let o = (y * h + x) * 4;
            for c in 0..3 {
                out[o + c] = (sum[c] / weight).round() as u8;
            }
            out[o + 3] = (sum[3] / 4.0).round() as u8;
        }
    }
    out
}

fn create_atlas(device: &Device, queue: &Queue, layers: &[Vec<u8>]) -> Texture {
    let texture = device.create_texture(&TextureDescriptor {
        label: Some("effects_atlas"),
        size: Extent3d { width: ATLAS_SIZE, height: ATLAS_SIZE, depth_or_array_layers: ATLAS_LAYERS },
        mip_level_count: ATLAS_MIPS,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
        view_formats: &[],
    });
    for (layer, image) in layers.iter().enumerate() {
        let mut level = image.clone();
        let mut size = ATLAS_SIZE as usize;
        for mip in 0..ATLAS_MIPS {
            queue.write_texture(
                TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: mip,
                    origin: Origin3d { x: 0, y: 0, z: layer as u32 },
                    aspect: TextureAspect::All,
                },
                &level,
                TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(size as u32 * 4), rows_per_image: Some(size as u32) },
                Extent3d { width: size as u32, height: size as u32, depth_or_array_layers: 1 },
            );
            if size > 1 {
                level = half_mip(&level, size);
                size /= 2;
            }
        }
    }
    texture
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct EffectsUniform {
    pub right: [f32; 4],
    pub up: [f32; 4],
    pub head: [f32; 4],

    pub depth: [f32; 4],
}

pub fn bind_group_layout(device: &Device) -> BindGroupLayout {
    let uniform = |binding, visibility| BindGroupLayoutEntry {
        binding,
        visibility,
        ty: BindingType::Buffer { ty: BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
        count: None,
    };
    device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("effects_layout"),
        entries: &[
            uniform(0, ShaderStages::VERTEX | ShaderStages::FRAGMENT),
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::D2Array,
                    multisampled: false,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Sampler(SamplerBindingType::Filtering),
                count: None,
            },
            uniform(3, ShaderStages::VERTEX),
        ],
    })
}

pub struct EffectsGpu {
    _atlas: Texture,
    uniform: Buffer,
    _cutouts: Buffer,
    bind_group: BindGroup,
    instances: Buffer,
    capacity: u64,
    over: u32,
    screened: u32,

    runs: Vec<variants::Run>,

    pub run_stats: variants::RunStats,

    pub use_variants: bool,

    pub overdraw: bool,
}

impl EffectsGpu {
    pub fn new(device: &Device, queue: &Queue, layout: &BindGroupLayout) -> Self {
        use wgpu::util::DeviceExt;
        let layers = atlas_layers();
        let atlas = create_atlas(device, queue, &layers);
        let view = atlas.create_view(&TextureViewDescriptor {
            dimension: Some(TextureViewDimension::D2Array),
            ..Default::default()
        });
        let sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("effects_sampler"),
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: MipmapFilterMode::Linear,
            ..Default::default()
        });
        let uniform = device.create_buffer(&BufferDescriptor {
            label: Some("effects_uniform"),
            size: std::mem::size_of::<EffectsUniform>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cut: Vec<[f32; 16]> = cutouts(&layers);
        let _ = CUTOUT_SHARES.set(cutout_shares_of(&cut));
        let cutouts = device.create_buffer_init(&util::BufferInitDescriptor {
            label: Some("effects_cutouts"),
            contents: bytemuck::cast_slice(&cut),
            usage: BufferUsages::UNIFORM,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("effects_bind_group"),
            layout,
            entries: &[
                BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: BindingResource::TextureView(&view) },
                BindGroupEntry { binding: 2, resource: BindingResource::Sampler(&sampler) },
                BindGroupEntry { binding: 3, resource: cutouts.as_entire_binding() },
            ],
        });
        let capacity = 256;
        let instances = Self::instance_buffer(device, capacity);
        Self {
            _atlas: atlas,
            uniform,
            _cutouts: cutouts,
            bind_group,
            instances,
            capacity,
            over: 0,
            screened: 0,
            runs: Vec::new(),
            run_stats: Default::default(),
            use_variants: true,
            overdraw: false,
        }
    }

    fn instance_buffer(device: &Device, capacity: u64) -> Buffer {
        device.create_buffer(&BufferDescriptor {
            label: Some("effects_instances"),
            size: capacity * std::mem::size_of::<EffectInstance>() as u64,
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    pub fn upload(&mut self, device: &Device, queue: &Queue, frame: &EffectFrame) {
        let n = frame.instances.len() as u64;
        if n > self.capacity {
            self.capacity = n.next_power_of_two();
            self.instances = Self::instance_buffer(device, self.capacity);
        }
        if n > 0 {
            queue.write_buffer(&self.instances, 0, bytemuck::cast_slice(&frame.instances));
        }
        self.over = frame.over;
        self.screened = frame.screened;
        (self.runs, self.run_stats) = variants::runs(frame, self.use_variants);
    }

    pub fn set_view(&self, queue: &Queue, uniform: &EffectsUniform) {
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(uniform));
    }

    pub fn motion_parts(&self) -> (&Buffer, &BindGroup, &[variants::Run]) {
        (&self.instances, &self.bind_group, &self.runs)
    }

    pub fn any(&self) -> bool {
        self.over + self.screened > 0
    }

    /// The particles uploaded last: drawn over, and screened.
    pub fn counts(&self) -> (u32, u32) {
        (self.over, self.screened)
    }

    pub fn draw(&self, pass: &mut RenderPass<'_>, pipelines: &EffectsPipeline, camera: &BindGroup, probe: &BindGroup) {
        if !self.any() {
            return;
        }
        pass.set_bind_group(0, camera, &[]);
        pass.set_bind_group(1, probe, &[]);
        pass.set_bind_group(2, &self.bind_group, &[]);
        pass.set_vertex_buffer(0, self.instances.slice(..));
        if self.overdraw {
            pass.set_pipeline(&pipelines.overdraw);
            pass.draw(0..VERTICES_PER_PARTICLE, 0..self.over + self.screened);
            return;
        }
        for run in &self.runs {
            pass.set_pipeline(&pipelines.variants[run.variant as usize]);
            pass.draw(0..VERTICES_PER_PARTICLE, run.first..run.first + run.count);
        }
    }
}

pub struct EffectsPipeline {

    pub variants: Vec<RenderPipeline>,

    pub overdraw: RenderPipeline,
}

impl EffectsPipeline {
    pub fn new_multisampled(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        probe_layout: &BindGroupLayout,
        effects_layout: &BindGroupLayout,
        samples: u32,
    ) -> Self {
        Self::new_with_view(device, format, uniform_layout, probe_layout, effects_layout, samples, crate::renderer::multiview::ViewMode::Mono)
    }

    pub fn new_multisampled_stereo(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        probe_layout: &BindGroupLayout,
        effects_layout: &BindGroupLayout,
        samples: u32,
    ) -> Self {
        Self::new_with_view(device, format, uniform_layout, probe_layout, effects_layout, samples, crate::renderer::multiview::ViewMode::Stereo)
    }

    fn new_with_view(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        probe_layout: &BindGroupLayout,
        effects_layout: &BindGroupLayout,
        samples: u32,
        view: crate::renderer::multiview::ViewMode,
    ) -> Self {
        let module = |label: &str, source: String| {
            device.create_shader_module(ShaderModuleDescriptor { label: Some(label), source: ShaderSource::Wgsl(view.shader(source).into()) })
        };
        let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("effects_layout"),
            bind_group_layouts: &[Some(uniform_layout), Some(probe_layout), Some(effects_layout)],
            immediate_size: 0,
        });

        let over = BlendComponent { src_factor: BlendFactor::One, dst_factor: BlendFactor::OneMinusSrcAlpha, operation: BlendOperation::Add };
        let screen = BlendComponent { src_factor: BlendFactor::OneMinusDst, dst_factor: BlendFactor::One, operation: BlendOperation::Add };
        let build = |label: &str, shader: &ShaderModule, fragment: &str, color: BlendComponent| {
            device.create_render_pipeline(&RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                vertex: VertexState {
                    module: shader,
                    entry_point: Some("vs_main"),
                    compilation_options: PipelineCompilationOptions::default(),
                    buffers: &[Some(EffectInstance::layout())],
                },
                fragment: Some(FragmentState {
                    module: shader,
                    entry_point: Some(fragment),
                    compilation_options: PipelineCompilationOptions::default(),
                    targets: &[Some(ColorTargetState {
                        format,
                        blend: Some(BlendState { color, alpha: BlendComponent::OVER }),

                        write_mask: ColorWrites::COLOR,
                    })],
                }),
                primitive: PrimitiveState {
                    topology: PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    front_face: FrontFace::Ccw,
                    polygon_mode: PolygonMode::Fill,
                    ..Default::default()
                },
                depth_stencil: Some(DepthStencilState {
                    format: TextureFormat::Depth32Float,
                    depth_write_enabled: Some(false),
                    depth_compare: Some(CompareFunction::LessEqual),
                    stencil: StencilState::default(),
                    bias: DepthBiasState::default(),
                }),
                multisample: MultisampleState { count: samples, ..Default::default() },
                multiview_mask: view.mask(),
                cache: None,
            })
        };
        let added = BlendComponent { src_factor: BlendFactor::One, dst_factor: BlendFactor::One, operation: BlendOperation::Add };
        let variants = FxVariant::ALL
            .iter()
            .map(|v| {
                let shader = module(v.label(), v.shader());
                if v.screened() {
                    build(v.label(), &shader, "fs_screen", screen)
                } else {
                    build(v.label(), &shader, "fs_over", over)
                }
            })
            .collect();
        if crate::renderer::shader_checks::PIPELINE_STATISTICS.load(std::sync::atomic::Ordering::Relaxed) {

            for (name, entry, _) in variants::EFFECT_MODULE_CUTS {
                let Some(source) = variants::module_cut(name) else { continue };
                let label = format!("effects_cut_{name}");
                let _ = build(&label, &module(&label, source), entry, if *entry == "fs_screen" { screen } else { over });
            }
        }
        let overdraw = build("effects_overdraw", &module("effects_overdraw", variants::overdraw_shader()), "fs_overdraw", added);
        Self { variants, overdraw }
    }
}

pub fn effects_shader() -> String {
    format!(
        "{}{}",
        crate::renderer::tonemap::wgsl_aces_block(),
        r"
var<private> view_slot: i32 = 0;
struct Camera { view_proj: array<mat4x4<f32>, 2> }
@group(0) @binding(0) var<uniform> camera: Camera;
@group(1) @binding(1) var probe_pass_depth: texture_depth_2d_array;
struct Fx {
    right: vec4<f32>,
    up: vec4<f32>,
    head: vec4<f32>,
    depth: vec4<f32>,
}
@group(2) @binding(0) var<uniform> fx: Fx;
@group(2) @binding(1) var fx_atlas: texture_2d_array<f32>;
@group(2) @binding(2) var fx_samp: sampler;
struct Cutouts { corners: array<vec4<f32>, CUT_VEC4S> }
@group(2) @binding(3) var<uniform> fx_cut: Cutouts;

fn fx_eye() -> vec3<f32> {
    let m = camera.view_proj[view_slot];
    let r0 = vec4<f32>(m[0].x, m[1].x, m[2].x, m[3].x);
    let r1 = vec4<f32>(m[0].y, m[1].y, m[2].y, m[3].y);
    let r3 = vec4<f32>(m[0].w, m[1].w, m[2].w, m[3].w);
    let c13 = cross(r1.xyz, r3.xyz);
    let c30 = cross(r3.xyz, r0.xyz);
    let c01 = cross(r0.xyz, r1.xyz);
    return -(r0.w * c13 + r1.w * c30 + r3.w * c01) / dot(r0.xyz, c13);
}

var<private> fx_fog_point: vec3<f32> = vec3<f32>(0.0);

struct IIn {
    @location(0) centre: vec4<f32>,
    @location(1) colour: vec4<f32>,
    @location(2) light: vec4<f32>,
    @location(3) light_dir: vec4<f32>,
    @location(4) axis: vec4<f32>,
    @location(5) params: vec4<f32>,
}
struct VOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) colour: vec4<f32>,
    @location(2) @interpolate(flat) light: vec3<f32>,

    @location(3) @interpolate(flat) light_q: vec3<f32>,

    @location(4) @interpolate(flat) layer: vec3<f32>,

    @location(5) at: vec4<f32>,
    @location(6) @interpolate(flat) soft: f32,

    @location(7) @interpolate(flat) time: f32,
    @location(8) @interpolate(flat) flicker: f32,

    @location(9) world: vec3<f32>,
    @location(10) @interpolate(flat) fog_a: vec4<f32>,
    @location(11) @interpolate(flat) fog_b: vec4<f32>,
}

fn fx_fog_vertex(vi: u32, inst: IIn) -> VOut {
    var out: VOut;
    let tri = vi / 3u;
    let k = vi % 3u;
    var corner = vec2<f32>(0.0);
    if (k != 0u) {
        let a = f32((tri + k - 1u) % 8u) * 0.78539816;
        corner = vec2<f32>(cos(a), sin(a)) * 1.0824;
    }
    let c = inst.centre.xyz;
    let r = length(inst.axis.xyz);
    let eye = fx_eye();
    let d = length(c - eye);
    var mid = eye;
    var across = fx.right.xyz;
    var along = fx.up.xyz;
    var reach = 1.0;
    if (d > r * 1.02) {
        let f = (c - eye) / d;
        let s = d - r;
        mid = eye + f * s;
        reach = s * r / sqrt(d * d - r * r);
        across = normalize(select(cross(f, vec3<f32>(0.0, 1.0, 0.0)), cross(f, vec3<f32>(1.0, 0.0, 0.0)), abs(f.y) > 0.99));
        along = cross(across, f);
    } else {
        mid = eye + normalize(cross(fx.up.xyz, fx.right.xyz)) * 0.25;
    }
    let world = mid + (across * corner.x + along * corner.y) * reach;
    out.clip = camera.view_proj[view_slot] * vec4<f32>(world, 1.0);
    out.at = out.clip;
    out.world = world;
    out.uv = corner;
    out.colour = inst.colour;
    out.light = inst.light.xyz;
    out.light_q = inst.light_dir.xyz;
    out.layer = vec3<f32>(0.0, inst.params.y, inst.params.w);
    out.soft = inst.light.w;
    out.time = inst.params.x;
    out.flicker = inst.params.z;
    out.fog_a = inst.centre;
    out.fog_b = inst.axis;
    return out;
}

@vertex fn vs_main(@builtin(vertex_index) vi: u32, inst: IIn) -> VOut {
    if (inst.params.w < -1.5) {
        return fx_fog_vertex(vi, inst);
    }
    var out: VOut;

    let tri = vi / 3u;
    let k = vi % 3u;
    let layer = min(u32(max(inst.params.y, 0.0)), LAST_LAYERu);
    var corner: vec2<f32>;
    if (k == 0u) {

        var sum = vec2<f32>(0.0);
        for (var q = 0u; q < 4u; q++) {
            let c = fx_cut.corners[layer * 4u + q];
            sum += c.xy + c.zw;
        }
        corner = sum * 0.125;
    } else {
        let j = (tri + k - 1u) % 8u;
        let pair = fx_cut.corners[layer * 4u + j / 2u];
        corner = select(pair.zw, pair.xy, (j & 1u) == 0u);
    }
    let centre = inst.centre.xyz;
    let half = inst.centre.w;
    let to_eye = normalize(fx.head.xyz - centre);
    var across = fx.right.xyz;
    var along = fx.up.xyz;
    var reach = vec2<f32>(half, half);
    var middle = centre;
    let streak = inst.light_dir.w;
    if (inst.light.w > 1.5) {

        let n = inst.axis.xyz;
        let a0 = normalize(select(cross(n, vec3<f32>(0.0, 0.0, 1.0)), cross(n, vec3<f32>(1.0, 0.0, 0.0)), abs(n.z) > 0.9));
        let b0 = cross(n, a0);
        let s = sin(inst.params.x);
        let c = cos(inst.params.x);
        across = a0 * c + b0 * s;
        along = b0 * c - a0 * s;
    } else if (streak > 0.0) {

        along = inst.axis.xyz;
        let side = cross(along, to_eye);
        across = select(fx.right.xyz, normalize(side), dot(side, side) > 1e-8);
        reach = vec2<f32>(half, half + 0.5 * streak);
        middle = centre - along * (0.5 * streak);
    } else {
        var r = fx.right.xyz;
        var u = fx.up.xyz;
        if (inst.light.w > 0.5) {

            let up_seen = vec3<f32>(0.0, 1.0, 0.0) - to_eye * to_eye.y;
            if (dot(up_seen, up_seen) > 1e-4) {
                u = normalize(up_seen);
                r = cross(u, to_eye);
            }
        }

        let s = sin(inst.params.x);
        let c = cos(inst.params.x);
        across = (r * c + u * s) * select(1.0, -1.0, inst.params.z < 0.0);
        along = u * c - r * s;
        reach = vec2<f32>(half, half * max(abs(inst.params.z), 1.0));
    }

    let least = 0.75 * fx.depth.w * (camera.view_proj[view_slot] * vec4<f32>(centre, 1.0)).w;
    var drawn = max(reach, vec2<f32>(least));
    if (inst.params.w < -0.5) {

        drawn = vec2<f32>(least);
    }
    out.colour = inst.colour;
    out.colour.a = inst.colour.a * (reach.x * reach.y) / max(drawn.x * drawn.y, 1e-12);
    reach = drawn;

    let dome = select(0.0, CONVEX * min(reach.x, reach.y), inst.params.w > 0.5 && inst.params.w < 2.5 && inst.light.w < 1.5 && streak <= 0.0);
    let world = middle + across * (corner.x * reach.x) + along * (corner.y * reach.y) + to_eye * (select(-0.5, 0.5, k == 0u) * dome);
    out.clip = camera.view_proj[view_slot] * vec4<f32>(world, 1.0);
    out.at = out.clip;

    out.uv = vec2<f32>(corner.x * 0.5 + 0.5, 0.5 - corner.y * 0.5);
    out.light = inst.light.xyz;
    let l = inst.light_dir.xyz;
    out.light_q = vec3<f32>(dot(l, across), dot(l, along), dot(l, cross(across, along)));
    out.layer = vec3<f32>(floor(inst.params.y), fract(inst.params.y), inst.params.w);
    out.soft = inst.axis.w;
    out.time = inst.params.z;
    out.flicker = inst.light_dir.w;
    return out;
}

fn fx_depth_fade(in: VOut) -> f32 {
    if (fx.depth.z < 0.5 || in.soft <= 0.0 || in.at.w <= 0.0) {
        return 1.0;
    }
    let ndc = in.at.xyz / in.at.w;
    let size = vec2<f32>(textureDimensions(probe_pass_depth));
    let texel = clamp(vec2<i32>(vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5) * size), vec2<i32>(0), vec2<i32>(size) - vec2<i32>(1));
    let z = textureLoad(probe_pass_depth, texel, view_slot, 0);
    if (z >= 1.0) {
        return 1.0;
    }
    let n = fx.depth.x;
    let f = fx.depth.y;
    let scene = n * f / (f - z * (f - n));
    return clamp((scene - in.at.w) / in.soft, 0.0, 1.0);
}

const FOG_TILE: f32 = 0.42;
const FOG_STEPS: i32 = 10;

fn fx_fog_gyroid(p: vec3<f32>) -> f32 {
    return dot(sin(p), cos(p.yzx));
}

const FOG_TURN: mat3x3<f32> = mat3x3<f32>(vec3<f32>(0.8, -0.6, 0.0), vec3<f32>(0.36, 0.48, 0.8), vec3<f32>(-0.48, -0.64, 0.6));
fn fx_fog_tuft(m: vec3<f32>, tall: f32, half: f32, wind: vec2<f32>, drift: f32, blur: f32) -> f32 {
    let t = 1.2 * drift;
    let at = vec3<f32>(m.x - wind.x * drift, m.y, m.z - wind.y * drift);
    let c = fx_fog_gyroid(FOG_TURN * vec3<f32>(at.x / (2.6 * FOG_TILE) + 1.0, at.y / (5.0 * FOG_TILE) + 0.2 * t, at.z / (2.6 * FOG_TILE) - 2.0));
    var p = FOG_TURN * vec3<f32>(at.x / FOG_TILE, at.y / FOG_TILE + 0.3 * t, at.z / FOG_TILE);
    p += 0.45 * sin(vec3<f32>(0.9 * p.y + 0.5 * t, 0.8 * p.z + 1.0, 0.7 * p.x - 0.4 * t));
    let n1 = fx_fog_gyroid(p);
    let n2 = fx_fog_gyroid(FOG_TURN * (2.3 * p.zxy + vec3<f32>(5.0, 0.6 * t, 3.0)));
    let band = (m.y - 0.375 * tall) / max(half, 1e-3);
    let n = 0.5 * n1 + 0.3 * n2 * (1.0 - smoothstep(0.3, 0.8, blur)) + 0.45 * c - band * band;
    let guard = smoothstep(0.025 * tall, 0.125 * tall, m.y) * (1.0 - smoothstep(0.8 * tall, tall, m.y));
    return guard * smoothstep(0.1, 0.95, n);
}

fn fx_fog(in: VOut) -> vec4<f32> {
    let eye = fx_eye();
    let ray = in.world - eye;
    let t_quad = max(length(ray), 1e-6);
    let d = ray / t_quad;
    var t_far = 1e6;
    if (fx.depth.z > 0.5 && in.at.w > 0.0) {
        let ndc = in.at.xyz / in.at.w;
        let size = vec2<f32>(textureDimensions(probe_pass_depth));
        let texel = clamp(vec2<i32>(vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5) * size), vec2<i32>(0), vec2<i32>(size) - vec2<i32>(1));
        let z = textureLoad(probe_pass_depth, texel, view_slot, 0);
        if (z < 1.0) {
            let n = fx.depth.x;
            let f = fx.depth.y;
            t_far = t_quad * (n * f / (f - z * (f - n))) / in.at.w;
        }
    }

    let bx = vec3<f32>(cos(in.time), 0.0, sin(in.time));
    let bz = vec3<f32>(-bx.z, 0.0, bx.x);
    let o = eye - in.fog_a.xyz;
    let lo = vec3<f32>(dot(o, bx), o.y, dot(o, bz));
    let ld = vec3<f32>(dot(d, bx), d.y, dot(d, bz));
    let h = in.fog_b.xyz;
    let inv = 1.0 / select(ld, vec3<f32>(1e-6), abs(ld) < vec3<f32>(1e-6));
    let ta = (-h - lo) * inv;
    let tb = (h - lo) * inv;
    let t0 = max(max(max(min(ta.x, tb.x), min(ta.y, tb.y)), min(ta.z, tb.z)), 0.0);
    let t1 = min(min(min(max(ta.x, tb.x), max(ta.y, tb.y)), max(ta.z, tb.z)), t_far);
    if (t1 <= t0) {
        return vec4<f32>(0.0);
    }
    let half = max(in.fog_b.w, 0.01);
    let wind = vec2<f32>(cos(in.soft), sin(in.soft));
    let rim = in.fog_a.w / max(min(h.x, h.z), 1e-3);
    let span = (t1 - t0) / f32(FOG_STEPS);
    let blur = span / FOG_TILE;
    var depth = 0.0;
    var top = 0.0;
    var at_t = 0.0;
    for (var i = 0; i < FOG_STEPS; i++) {
        let t = t0 + (f32(i) + 0.5) * span;
        let m = lo + ld * t;
        let lies = 1.0 - smoothstep(1.0 - rim, 1.0, length(m.xz / max(h.xz, vec2<f32>(1e-3))));
        let up = m.y + h.y;
        let k = span * lies * fx_fog_tuft(vec3<f32>(m.x, up, m.z), 2.0 * h.y, half, wind, in.flicker, blur);
        depth += k;
        top += k * clamp(up / (1.5 * h.y), 0.0, 1.0);
        at_t += k * t;
    }
    fx_fog_point = eye + d * select(0.5 * (t0 + t1), at_t / depth, depth > 1e-6);

    let lit = 0.7 + 0.45 * top / max(depth, 1e-6);
    depth *= in.colour.a;
    let alpha = 1.0 - exp(-depth);
    let g = in.layer.y;
    let cosine = dot(d, in.light_q);

    let hg = min((1.0 - g * g) / pow(max(1.0 + g * g - 2.0 * g * cosine, 1e-4), 1.5), 12.0);
    let g4 = 0.25 * g;
    let hg4 = (1.0 - g4 * g4) / pow(max(1.0 + g4 * g4 - 2.0 * g4 * cosine, 1e-4), 1.5);
    let shown = aces_fitted(lit * (in.colour.rgb + in.light * mix(hg, hg4, alpha)));
    return vec4<f32>(shown * alpha, alpha);
}

fn fx_flame(hottest: vec3<f32>, heat: f32) -> vec3<f32> {
    let t = clamp(heat, 0.0, 1.0);
    let t2 = t * t;
    let bright = 0.03 + 0.97 * pow(max(t, 1e-6), 1.4);
    return vec3<f32>(1.0, 0.02 + 0.5 * t2, 0.02 * t + 0.08 * t2 * t2) * (hottest * bright);
}

fn fx_flame_v(v: f32, dt: f32) -> f32 {
    let h = (v - FLAME_BASE) / (1.0 - FLAME_BASE);
    let c = FLAME_W0 / FLAME_W1;
    let moved = max((h + c) * exp(-FLAME_W1 * dt) - c, 0.0);
    return select(v, FLAME_BASE + moved * (1.0 - FLAME_BASE), h > 0.0);
}

@fragment fn fs_over(in: VOut) -> @location(0) vec4<f32> {
    let gx = dpdx(in.uv);
    let gy = dpdy(in.uv);
    let fade = fx_depth_fade(in);
    let layer = i32(in.layer.x);
    let next = select(layer, layer + 1, in.layer.y > 0.0);
    // <fx:lump>
    if (in.layer.z > 4.5) {
        let m = textureSampleGrad(fx_atlas, fx_samp, in.uv, layer, gx, gy);
        let n = normalize(m.rgb * 2.0 - vec3<f32>(1.0));
        let ndl = max(dot(n, in.light_q), 0.0);
        let h = normalize(in.light_q + vec3<f32>(0.0, 0.0, 1.0));
        let sheen = LUMP_SHEEN * pow(max(dot(n, h), 0.0), LUMP_GLOSS) * ndl;
        let cover = clamp(in.colour.a * m.a * fade, 0.0, 1.0);
        let mud = aces_fitted(in.colour.rgb * (0.6 + 0.4 * n.z) + in.light * (ndl + sheen));
        return vec4<f32>(mud * cover, cover);
    }
    // </fx:lump>
    // <fx:coals>
    if (in.layer.z > 2.5) {

        let b = textureSampleGrad(fx_atlas, fx_samp, in.uv, layer, gx, gy);
        let phase = b.b * 6.2831853;
        let flicker = 0.8 + 0.2 * (1.0 + in.flicker) * sin(in.time * 2.3 + phase) * sin(in.time * 0.9 + 3.0 * phase);
        let alpha = clamp(in.colour.a * b.a, 0.0, 1.0);
        let shown = aces_fitted(fx_flame(in.colour.rgb, b.r * flicker) + in.light * b.g);
        return vec4<f32>(shown * alpha, alpha);
    }
    // </fx:coals>
    // <fx:flame>
    if (in.layer.z > 1.5) {
        let v = 1.0 - in.uv.y;
        let was = vec2<f32>(in.uv.x, 1.0 - fx_flame_v(v, in.layer.y * FLAME_DT));
        let will = vec2<f32>(in.uv.x, 1.0 - fx_flame_v(v, (in.layer.y - 1.0) * FLAME_DT));
        let f = mix(
            textureSampleGrad(fx_atlas, fx_samp, was, layer, gx, gy),
            textureSampleGrad(fx_atlas, fx_samp, will, next, gx, gy),
            in.layer.y,
        );

        let top = 1.0 - smoothstep(0.88, 1.0, v);

        let gone = 1.0 - in.colour.a;
        let front = mix(RISE_FROM, RISE_TO, in.light.x);
        let lasts = f.g * smoothstep(LIFT_FROM, LIFT_TO, v);
        let stands = smoothstep(gone - FLAME_EROSION, gone, lasts) * (1.0 - smoothstep(front - RISE_SOFT, front, v - RISE_LEAD * f.g));
        let alpha = clamp(f.a * stands * fade * top, 0.0, 1.0);

        let soot = aces_fitted(fx_flame(in.colour.rgb, f.r * (0.45 + 0.55 * alpha) * (1.0 - BURN_OUT * gone)));

        let root = vec3<f32>(0.05, 0.12, 0.6) * (BLUE_ROOT * in.colour.r * f.b * alpha);
        let shown = soot;

        let covers = alpha * FLAME_COVER * clamp(max(shown.r, max(shown.g, shown.b)), 0.0, 1.0);
        return vec4<f32>(shown * alpha + root, covers);
    }
    // </fx:flame>
    // <fx:fog>
    if (in.layer.z < -1.5) {
        return fx_fog(in);
    }
    // </fx:fog>
    let six = in.layer.z > 0.5;
    let back = select(layer, layer + SMOKE_BACK, six);
    let back_next = select(next, next + SMOKE_BACK, six);
    let a = mix(textureSampleGrad(fx_atlas, fx_samp, in.uv, layer, gx, gy), textureSampleGrad(fx_atlas, fx_samp, in.uv, next, gx, gy), in.layer.y);
    let b = mix(textureSampleGrad(fx_atlas, fx_samp, in.uv, back, gx, gy), textureSampleGrad(fx_atlas, fx_samp, in.uv, back_next, gx, gy), in.layer.y);
    let lp = max(in.light_q, vec3<f32>(0.0));
    let ln = max(-in.light_q, vec3<f32>(0.0));
    let toward = dot(lp * lp, a.rgb) + dot(ln * ln, b.rgb);
    let around = (a.r + a.g + a.b + b.r + b.g + b.b) / 6.0;
    let lit = select(in.colour.rgb + in.light, in.colour.rgb * around + in.light * toward, six);
    let alpha = clamp(in.colour.a * a.a * fade, 0.0, 1.0);
    return vec4<f32>(aces_fitted(lit) * alpha, alpha);
}

@fragment fn fs_screen(in: VOut) -> @location(0) vec4<f32> {
    let layer = i32(in.layer.x);
    let next = select(layer, layer + 1, in.layer.y > 0.0);
    let t = mix(textureSample(fx_atlas, fx_samp, in.uv, layer), textureSample(fx_atlas, fx_samp, in.uv, next), in.layer.y);
    let strength = in.colour.a * t.a * fx_depth_fade(in);

    let light = select(in.colour.rgb, fx_flame(in.colour.rgb, t.r), in.layer.z > 3.5);
    return vec4<f32>(aces_fitted(light * strength), 0.0);
}
"
        .replace("CUT_VEC4S", &(ATLAS_LAYERS * 4).to_string())
        .replace("LAST_LAYER", &(ATLAS_LAYERS - 1).to_string())
        .replace("SMOKE_BACK", &SMOKE_BACK.to_string())
        .replace("FLAME_COVER", &format!("{FLAME_COVER:?}"))
        .replace("FLAME_EROSION", &format!("{FLAME_EROSION:?}"))
        .replace("BLUE_ROOT", &format!("{BLUE_ROOT:?}"))
        .replace("CONVEX", &format!("{CONVEX:?}"))
        .replace("LUMP_SHEEN", &format!("{LUMP_SHEEN:?}"))
        .replace("LUMP_GLOSS", &format!("{LUMP_GLOSS:?}"))
        .replace("LIFT_FROM", &format!("{LIFT_FROM:?}"))
        .replace("LIFT_TO", &format!("{LIFT_TO:?}"))
        .replace("BURN_OUT", &format!("{BURN_OUT:?}"))
        .replace("RISE_FROM", &format!("{RISE_FROM:?}"))
        .replace("RISE_TO", &format!("{RISE_TO:?}"))
        .replace("RISE_SOFT", &format!("{RISE_SOFT:?}"))
        .replace("RISE_LEAD", &format!("{RISE_LEAD:?}"))
        .replace("FLAME_DT", &format!("{:?}", FLAME_SECONDS / FIRE_FRAMES as f32))
        .replace("FLAME_BASE", &format!("{FLAME_BASE:?}"))
        .replace("FLAME_W0", &format!("{FLAME_W0:?}"))
        .replace("FLAME_W1", &format!("{FLAME_W1:?}"))
    )
}

pub fn motion_shader() -> String {
    motion_shader_of(effects_shader())
}

pub fn motion_shader_of(source: String) -> String {
    let mut s = source;
    let swap = |s: &mut String, from: &str, to: &str, n: usize| {
        assert_eq!(s.matches(from).count(), n, "effects motion twin: `{from}` moved in the effects' shader");
        *s = s.replace(from, to);
    };

    let from = s.find("        var r = fx.right.xyz;\n").expect("effects motion twin: the upright axes moved");
    let end = "        along = u * c - r * s;\n";
    let to = from + s[from..].find(end).expect("effects motion twin: the upright axes moved") + end.len();
    let upright = s[from..to].to_string();
    swap(&mut s, "@group(0) @binding(0) var<uniform> camera: Camera;", "", 1);
    swap(&mut s, "camera.view_proj[view_slot]", "fx_motion_view()", 4);
    swap(&mut s, "0.75 * fx.depth.w * ", "0.75 * fx.depth.w * fx_pixel_scale * ", 1);
    swap(&mut s, "@vertex fn vs_main(@builtin(vertex_index) vi: u32, inst: IIn) -> VOut", "fn fx_vertex(vi: u32, inst: IIn) -> VOut", 1);
    swap(&mut s, "@fragment fn fs_over(in: VOut) -> @location(0) vec4<f32>", "fn fx_over(in: VOut) -> vec4<f32>", 1);
    swap(&mut s, "@fragment fn fs_screen(in: VOut) -> @location(0) vec4<f32>", "fn fx_screen(in: VOut) -> vec4<f32>", 1);
    format!(
        r"{motion}
{s}
var<private> fx_before: bool = false;
var<private> fx_pixel_scale: f32 = 1.0;

fn fx_motion_view() -> mat4x4<f32> {{
    if (fx_before) {{
        return cam.prev;
    }}
    return cam.curr;
}}

fn fx_flame_along(inst: IIn) -> vec3<f32> {{
    let to_eye = normalize(fx.head.xyz - inst.centre.xyz);
    var across = fx.right.xyz;
    var along = fx.up.xyz;
{upright}    return along;
}}

struct FxMotionOut {{
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) colour: vec4<f32>,
    @location(2) @interpolate(flat) light: vec3<f32>,
    @location(3) @interpolate(flat) light_q: vec3<f32>,
    @location(4) @interpolate(flat) layer: vec3<f32>,
    @location(5) at: vec4<f32>,

    @location(6) @interpolate(flat) stfb: vec4<f32>,
    @location(7) curr: vec4<f32>,
    @location(8) prev: vec4<f32>,

    @location(9) @interpolate(flat) gas: vec4<f32>,
    @location(10) world: vec3<f32>,
    @location(11) @interpolate(flat) fog_a: vec4<f32>,
    @location(12) @interpolate(flat) fog_b: vec4<f32>,
}}

@vertex fn vs_fx_motion(@builtin(vertex_index) vi: u32, inst: IIn, @location(6) motion: vec4<f32>) -> FxMotionOut {{
    fx_pixel_scale = cam.reflect.x;
    let step = cam.params.w;
    var was = inst;
    was.centre = vec4<f32>(inst.centre.xyz - motion.xyz * step, inst.centre.w);
    fx_before = true;
    let b = fx_vertex(vi, was);
    fx_before = false;
    let n = fx_vertex(vi, inst);
    var out: FxMotionOut;
    out.clip = n.clip;
    out.uv = n.uv;
    out.colour = n.colour;
    out.light = n.light;
    out.light_q = n.light_q;
    out.layer = n.layer;
    out.at = n.at;
    out.stfb = vec4<f32>(n.soft, n.time, n.flicker, 0.0);
    out.curr = n.clip;
    out.prev = b.clip;
    out.world = n.world;
    out.fog_a = n.fog_a;
    out.fog_b = n.fog_b;
    out.gas = vec4<f32>(0.0);

    if (inst.params.w < -1.5) {{
        out.gas = vec4<f32>(motion.xyz * step, 0.0);
    }}
    if (inst.params.w > 1.5 && inst.params.w < 2.5 && motion.w > 0.0) {{
        let height = 2.0 * inst.centre.w * max(abs(inst.params.z), 1.0);
        out.gas = cam.prev * vec4<f32>(fx_flame_along(inst) * height, 0.0);
        out.stfb.w = motion.w * step;
    }}
    return out;
}}

fn fx_vout(in: FxMotionOut) -> VOut {{
    var v: VOut;
    v.clip = in.clip;
    v.uv = in.uv;
    v.colour = in.colour;
    v.light = in.light;
    v.light_q = in.light_q;
    v.layer = in.layer;
    v.at = in.at;
    v.soft = in.stfb.x;
    v.time = in.stfb.y;
    v.flicker = in.stfb.z;
    v.world = in.world;
    v.fog_a = in.fog_a;
    v.fog_b = in.fog_b;
    return v;
}}

@fragment fn fs_fx_motion_over(in: FxMotionOut) -> @location(0) vec4<f32> {{
    let shown = fx_over(fx_vout(in));
    var prev = in.prev;
    if (in.stfb.w > 0.0) {{

        let v = 1.0 - in.uv.y;
        prev = prev - in.gas * (v - fx_flame_v(v, in.stfb.w));
    }}
    var curr = in.curr;
    if (in.layer.z < -1.5) {{

        curr = cam.curr * vec4<f32>(fx_fog_point, 1.0);
        prev = cam.prev * vec4<f32>(fx_fog_point - in.gas.xyz, 1.0);
    }}
    let flame = in.layer.z > 1.5 && in.layer.z < 2.5;
    let w = layer_weight(shown.a, select({soft:?}, {bright:?}, flame));
    return vec4<f32>(motion_of(curr, prev).xyz * w, w);
}}

@fragment fn fs_fx_motion_screened(in: FxMotionOut) -> @location(0) vec4<f32> {{
    let shown = fx_screen(fx_vout(in));
    let w = layer_weight(max(shown.r, max(shown.g, shown.b)), {bright:?});
    return vec4<f32>(motion_of(in.curr, in.prev).xyz * w, w);
}}
",
        motion = crate::renderer::space_warp::motion_block(),
        soft = crate::renderer::space_warp::SOFT_LAYER_CONTRAST,
        bright = crate::renderer::space_warp::BRIGHT_LAYER_CONTRAST,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_fog_slab_integrates_from_each_eye_not_their_middle() {
        let eye_of = |m: glam::Mat4| {
            let r = |i: usize| m.row(i);
            let (r0, r1, r3) = (r(0), r(1), r(3));
            let c13 = r1.truncate().cross(r3.truncate());
            let c30 = r3.truncate().cross(r0.truncate());
            let c01 = r0.truncate().cross(r1.truncate());
            -(r0.w * c13 + r1.w * c30 + r3.w * c01) / r0.truncate().dot(c13)
        };
        let proj = glam::Mat4::from_cols_array(&[0.9, 0.0, 0.0, 0.0, 0.0, 1.1, 0.0, 0.0, 0.08, -0.05, 0.0, -1.0, 0.0, 0.0, 0.03, 0.0]);
        for (eye, at) in [(Vec3::new(-14.032, 1.6, -24.0), Vec3::new(-26.0, -0.4, -27.0)), (Vec3::new(-13.968, 1.6, -24.0), Vec3::new(-26.0, -0.4, -27.0)), (Vec3::new(3.0, -2.0, 7.5), Vec3::new(3.0, 5.0, 7.6))] {
            let vp = proj * glam::Mat4::look_at_rh(eye, at, Vec3::Y);
            assert!((eye_of(vp) - eye).length() < 1e-3, "{eye} came back as {}", eye_of(vp));
        }
        let src = effects_shader();
        let body = |name: &str| {
            let from = src.find(&format!("fn {name}(")).unwrap();
            let to = from + src[from..].find("\n}\n").unwrap();
            src[from..to].to_string()
        };
        for f in ["fx_fog_vertex", "fx_fog"] {
            let b = body(f);
            assert!(b.contains("fx_eye()") && !b.contains("fx.head"), "{f} must start its rays at this eye");
        }
        assert!(motion_shader().contains("curr = cam.curr * vec4<f32>(fx_fog_point, 1.0);"), "a fog slab's motion is its mist's");
    }

    #[test]
    fn mist_follows_the_hour_thick_at_dawn_gone_by_midday() {
        assert_eq!(mist_conditions(6.0), 1.0);
        assert!(mist_conditions(10.5) <= 0.05 + 1e-6);
        assert!(mist_conditions(14.0) < 0.06);
        assert!((mist_conditions(21.0) - 0.8).abs() < 1e-6);
        assert!(mist_conditions(2.0) > 0.85 && mist_conditions(23.0) > 0.8);
        assert!(mist_conditions(9.0) < 0.6 && mist_conditions(9.0) > 0.3, "burning off: {}", mist_conditions(9.0));
        assert!((mist_conditions(24.0) - mist_conditions(0.0)).abs() < 1e-6 && (mist_conditions(-1.0) - mist_conditions(23.0)).abs() < 1e-6);
        let mut last = mist_conditions(0.0);
        for k in 1..=2400 {
            let m = mist_conditions(k as f32 * 0.01);
            assert!((m - last).abs() < 0.01, "a step at {}: {last} -> {m}", k as f32 * 0.01);
            last = m;
        }
    }

    fn emitter(kind: EffectKind) -> EffectEmitter {
        EffectEmitter {
            id: "test".into(),
            kind,
            position: Vec3::ZERO,
            direction: Vec3::Y,
            extent: [Vec3::X, Vec3::Y, Vec3::Z],
            scale: 1.0,
            rate: 1.0,
            tint: [1.0; 3],
            ceiling: None,
            variation: Variation::default(),
        }
    }

    fn every(_: &EffectEmitter, _: usize) -> bool {
        true
    }

    fn dim(_: Vec3) -> Vec3 {
        Vec3::splat(0.1)
    }

    fn seen_from(head: Vec3) -> Surroundings<'static> {
        Surroundings { head, offset: Vec3::ZERO, yaw_inv: Quat::IDENTITY, lights: &[], reaches: &every, ambient: &dim, exposure: 1.0, room: None }
    }

    fn lamp() -> Light {
        Light {
            position: Vec3::new(0.0, 2.0, 0.0),
            direction: Vec3::NEG_Y,
            kind: LightKind::Point,
            color: super::super::Color3(255, 255, 255, 255),
            intensity: 3.0,
            range: 4.0,
            cone_angle_deg: 0.0,
            inner_cone_angle_deg: 0.0,
            mask_channel: None,
            shadow_near: None,
            source_radius: 0.0,
            in_level_bake: true,
        }
    }

    fn centre(i: &EffectInstance) -> Vec3 {
        Vec3::new(i.centre[0], i.centre[1], i.centre[2])
    }

    #[test]
    fn the_same_time_gives_the_same_particles() {
        let e = [emitter(EffectKind::Fire), emitter(EffectKind::Smoke)];
        let at = seen_from(Vec3::new(0.0, 1.6, 3.0));
        let a = simulate(&e, 12.345, &at);
        let b = simulate(&e, 12.345, &at);
        assert!(a.instances.len() > 10);
        assert_eq!(bytemuck::cast_slice::<_, u8>(&a.instances), bytemuck::cast_slice::<_, u8>(&b.instances));
    }

    #[test]
    fn a_turn_moves_nothing_in_the_world() {

        let e = [emitter(EffectKind::Smoke), EffectEmitter { id: "f".into(), ..emitter(EffectKind::Fire) }];
        let straight = simulate(&e, 31.0, &seen_from(Vec3::new(0.0, 1.6, 3.0)));
        let (yaw, offset) = (1.1f32, Vec3::new(2.0, 0.0, -1.0));
        let turned_at = Surroundings { offset, yaw_inv: Quat::from_rotation_y(-yaw), ..seen_from(Vec3::new(0.0, 1.6, 3.0)) };
        let turned = simulate(&e, 31.0, &turned_at);
        let world = |i: &EffectInstance| Quat::from_rotation_y(yaw) * centre(i) + offset;
        let mut a: Vec<[i32; 3]> = straight.instances.iter().map(|i| (centre(i) * 1e4).round().as_ivec3().to_array()).collect();
        let mut b: Vec<[i32; 3]> = turned.instances.iter().map(|i| (world(i) * 1e4).round().as_ivec3().to_array()).collect();
        a.sort_unstable();
        b.sort_unstable();
        assert_eq!(a.len(), b.len());
        for (p, q) in a.iter().zip(&b) {
            assert!(p.iter().zip(q).all(|(x, y)| (x - y).abs() <= 2), "{p:?} vs {q:?}");
        }
    }

    #[test]
    fn an_emitter_keeps_about_rate_times_life_alive() {

        let e = [emitter(EffectKind::Smoke)];
        let at = seen_from(Vec3::new(0.0, 1.6, 3.0));
        let counts: Vec<f32> = (0..20).map(|k| simulate(&e, 100.0 + f64::from(k) * 0.37, &at).over as f32).collect();
        let mean = counts.iter().sum::<f32>() / counts.len() as f32;
        assert!((22.0..38.0).contains(&mean), "about 6 x 5 = 30 alive, found {mean}");
    }

    #[test]
    fn dust_fills_its_box_by_volume() {

        let small = EffectEmitter { extent: [Vec3::X * 0.5, Vec3::Y * 0.5, Vec3::Z * 0.5], ..emitter(EffectKind::Dust) };
        let large = emitter(EffectKind::Dust);
        let at = seen_from(Vec3::new(0.0, 1.6, 3.0));
        let count = |e: &EffectEmitter| (0..10).map(|k| simulate(std::slice::from_ref(e), 50.0 + f64::from(k), &at).screened).sum::<u32>() as f32 / 10.0;
        let (s, l) = (count(&small), count(&large));
        assert!((60.0..120.0).contains(&s), "90 a cubic metre, found {s}");
        assert!((l / s - 8.0).abs() < 1.5, "{l} vs {s}");
        let wander = air_reach(&air_waves(id_hash(&small.id))) + 0.05;
        for i in simulate(std::slice::from_ref(&small), 50.0, &at).instances {
            assert!(centre(&i).abs().max_element() < 0.5 + wander, "inside its box, as far as the air carries it: {}", centre(&i));
        }
    }

    fn step_splash(born: f64) -> Splash {
        Splash { depth: 0.3, seed: 11, ..Splash::new(Vec3::ZERO, born, Vec3::new(0.0, -2.2, 0.0), 0.05, 3.0) }
    }

    fn pebble() -> Splash {
        Splash { depth: 2.0, seed: 3, ..Splash::new(Vec3::ZERO, 0.0, Vec3::new(0.0, -4.4, 0.0), 0.015, 0.035) }
    }
    fn pistol() -> Splash {
        Splash { size: 0.07, mass: 0.9, ..pebble() }
    }
    fn crate_box() -> Splash {
        Splash { velocity: Vec3::new(0.0, -6.3, 0.0), size: 0.4, mass: 30.0, ..pebble() }
    }
    fn bullet() -> Splash {
        Splash { depth: 2.0, seed: 5, ..Splash::projectile(Vec3::ZERO, 0.0, Vec3::new(247.0, -247.0, 0.0), 0.009, 0.008) }
    }
    fn foot() -> Splash {
        Splash { depth: 0.2, seed: 9, ..Splash::new(Vec3::ZERO, 0.0, Vec3::new(-2.3, -1.1, 0.0), 0.05, 3.0) }
    }

    #[test]
    fn a_splash_throws_drops_that_rise_fall_back_and_are_gone() {
        let at = seen_from(Vec3::new(0.0, 1.6, 2.0));
        let s = [pistol()];
        let s = [Splash { born: 10.0, ..s[0] }];
        let shape = splash_shape(&s[0]);
        let count = |t: f64| simulate_with(&[], &s, t, &at).screened;
        assert_eq!(count(9.9), 0, "nothing before it struck");
        assert!(count(10.15) >= 12, "drops in the air: {}", count(10.15));
        assert_eq!(count(10.0 + f64::from(shape.seconds) + 0.01), 0, "all fallen back");
        assert!(shape.seconds <= SPLASH_SECONDS);

        let f = simulate_with(&[], &s, 10.2, &at);
        let ys: Vec<f32> = f.instances[f.over as usize..].iter().map(|i| i.centre[1]).collect();
        assert!(ys.iter().all(|&y| y >= -1e-3), "{ys:?}");
        assert!(ys.iter().copied().fold(0.0, f32::max) > 0.05, "{ys:?}");
        assert!(f.over >= 5, "its crown, in pieces: {}", f.over);
    }

    #[test]
    fn a_cavity_grows_as_its_froude_number_and_a_bigger_splash_is_slower() {

        let heavy = |speed: f32, size: f32| Splash { velocity: Vec3::new(0.0, -speed, 0.0), size, mass: 1e4 * size.powi(3), ..pebble() };
        let (a, b) = (splash_shape(&heavy(3.0, 0.05)), splash_shape(&heavy(6.0, 0.05)));

        let grows = (b.cavity - 0.02) / (a.cavity - 0.02);
        assert!((1.45..1.75).contains(&grows), "cavity {} -> {} ({grows})", a.cavity, b.cavity);

        let (small, big) = (splash_shape(&heavy(3.0, 0.03)), splash_shape(&heavy(6.0, 0.12)));
        assert!((big.cavity / small.cavity - 4.0).abs() < 0.2, "{} vs {}", big.cavity, small.cavity);
        assert!((big.collapse / small.collapse - 2.0).abs() < 0.1, "{} vs {}", big.collapse, small.collapse);
        assert!((1.9..5.0).contains(&(big.sheet_speed / small.sheet_speed)), "{} vs {}", big.sheet_speed, small.sheet_speed);
        assert!((3.4..25.0).contains(&(big.crown_height / small.crown_height)), "{} vs {}", big.crown_height, small.crown_height);

        let (p, c) = (splash_shape(&pebble()), splash_shape(&crate_box()));
        assert!((0.18..0.32).contains(&(p.sheet_speed / 4.4)) && (0.5..0.8).contains(&(c.sheet_speed / 6.3)), "{} {}", p.sheet_speed, c.sheet_speed);
        assert!(big.ring_wavelength > 3.5 * small.ring_wavelength && big.ring_speed > 1.5 * small.ring_speed);

        let lowered = splash_shape(&heavy(0.05, 0.05));

        assert!(lowered.sheet_speed < 0.02 && lowered.cavity < 0.0501 && lowered.ring_height < 1e-3, "{lowered:?}");
        let at = seen_from(Vec3::new(0.0, 1.6, 2.0));
        assert_eq!(simulate_with(&[], &[heavy(0.05, 0.05)], 0.1, &at).instances.len(), 0);
    }

    #[test]
    fn a_pebble_a_crate_a_bullet_and_a_foot_each_splash_as_what_they_are() {
        let (p, g, c, b, f) = (splash_shape(&pebble()), splash_shape(&pistol()), splash_shape(&crate_box()), splash_shape(&bullet()), splash_shape(&foot()));

        assert!(p.cavity < g.cavity && g.cavity < c.cavity, "{} {} {}", p.cavity, g.cavity, c.cavity);
        assert!(p.collapse < g.collapse && g.collapse < c.collapse);
        assert!(p.crown_height < g.crown_height && g.crown_height < c.crown_height);
        assert!(p.ring_wavelength < g.ring_wavelength && g.ring_wavelength < c.ring_wavelength);
        assert!((0.02..0.08).contains(&p.cavity) && (0.1..0.25).contains(&g.cavity) && (0.5..1.0).contains(&c.cavity));

        assert!(p.jet_height > 0.08 && p.jet_width < 0.012, "{p:?}");
        assert!(g.jet_height > 0.1 && g.jet_width > p.jet_width);
        assert_eq!(c.jet_height, 0.0, "{c:?}");

        assert!(b.sealed > 0.99 && b.cavity < 0.1 && b.collapse < 0.02, "{b:?}");
        assert!(b.fine > 0.95 && b.drop < 5e-4 && b.petals == 0);
        assert!(b.jet_height > 0.5 && b.jet_width < 0.015 && b.mist >= 4, "{b:?}");
        assert!(p.sealed == 0.0 && p.fine == 0.0 && p.drop > 1.5e-3);

        assert!(f.crown_height < 0.1 && f.jet_height < 0.15, "{f:?}");
        assert!(f.forward.x < -0.8 && f.carry.x < -0.5);

        let ankle = splash_shape(&Splash { depth: 0.03, ..pistol() });
        assert!(ankle.jet_height < 0.01 && ankle.sheet_angle < g.sheet_angle - 0.4, "{ankle:?}");

        let cork = splash_shape(&Splash { mass: 0.02, ..pistol() });
        assert!(cork.cavity < 0.8 * g.cavity && cork.crown_height < 0.5 * g.crown_height, "{} vs {}", cork.cavity, g.cavity);
    }

    #[test]
    fn a_harder_strike_throws_more_and_higher() {
        let at = seen_from(Vec3::new(0.0, 1.6, 2.0));
        let soft = [pebble()];
        let hard = [pistol()];
        let (a, b) = (simulate_with(&[], &soft, 0.12, &at), simulate_with(&[], &hard, 0.12, &at));
        let top = |f: &EffectFrame| f.instances[..f.over as usize].iter().map(|i| i.centre[1] + i.centre[3]).fold(0.0, f32::max);
        assert!(b.screened > 2 * a.screened, "{} vs {}", b.screened, a.screened);
        assert!(top(&b) > 1.3 * top(&a), "{} vs {}", top(&b), top(&a));
        let (p, g) = (splash_shape(&pebble()), splash_shape(&pistol()));
        assert!(g.crown_height > 1.8 * p.crown_height && g.spray >= p.spray, "{} vs {}", g.crown_height, p.crown_height);
    }

    #[test]
    fn a_glancing_strike_throws_its_water_ahead() {
        let at = seen_from(Vec3::new(0.0, 1.6, 2.0));
        let f = simulate_with(&[], &[foot()], 0.12, &at);
        let drops = &f.instances[f.over as usize..];
        assert!(drops.len() >= 8, "{}", drops.len());
        let ahead = drops.iter().map(|d| d.motion[0]).sum::<f32>() / drops.len() as f32;
        assert!(ahead < -0.15, "the drops' mean speed along its way: {ahead}");
    }

    #[test]
    fn a_splash_is_the_same_every_time_and_no_two_are_alike() {
        let at = seen_from(Vec3::new(0.0, 1.6, 2.0));
        let a = simulate_with(&[], &[pistol()], 0.1, &at);
        let b = simulate_with(&[], &[pistol()], 0.1, &at);
        assert_eq!(bytemuck::cast_slice::<_, u8>(&a.instances), bytemuck::cast_slice::<_, u8>(&b.instances));
        let other = simulate_with(&[], &[Splash { seed: 4, ..pistol() }], 0.1, &at);
        assert_eq!(a.over, other.over);
        let crown = |f: &EffectFrame| -> Vec<[f32; 4]> { f.instances[..f.over as usize].iter().filter(|i| i.params[1] >= CROWN_LAYER as f32).map(|i| i.params).collect() };
        let (ca, cb) = (crown(&a), crown(&other));
        assert!(ca.len() >= 5 && ca != cb, "another seed's crown is other pieces, turned otherwise");
        let moved = a.instances.iter().zip(&other.instances).filter(|(x, y)| (Vec3::from_slice(&x.centre[..3]) - Vec3::from_slice(&y.centre[..3])).length() > 0.01).count();
        assert!(moved * 2 > a.instances.len(), "{moved} of {} moved", a.instances.len());
    }

    #[test]
    fn a_strike_into_disturbed_water_raises_no_second_crown() {
        let at = seen_from(Vec3::new(0.0, 1.6, 2.5));
        let first = pistol();
        let second = |after: f64, apart: f32| {
            let mut live = vec![first];
            strike(&mut live, Splash { born: after, position: Vec3::new(apart, 0.0, 0.0), seed: 21, ..pistol() }, 48);
            live[1]
        };
        let shape = splash_shape(&first);
        let into_the_hole = second(0.5 * f64::from(shape.collapse), 0.03);
        assert!(into_the_hole.calm < 0.05, "{}", into_the_hole.calm);
        let churned = second(f64::from(shape.collapse) + 0.3, 0.03);
        assert!((0.25..0.7).contains(&churned.calm), "{}", churned.calm);
        assert!(second(0.1, 2.0).calm == 1.0 && second(f64::from(shape.settle) + 0.1, 0.0).calm == 1.0);

        let (calm, rough) = (splash_shape(&first), splash_shape(&into_the_hole));
        assert_eq!(rough.jet_height, 0.0);
        assert!(rough.drops < calm.drops && rough.ring_height < 0.5 * calm.ring_height);
        assert!(calm.froth == 0 && rough.froth >= 6 && rough.white > 0.8 && rough.spray >= 3 && rough.foam > calm.foam, "{rough:?}");
        let pieces = |s: Splash, layers: std::ops::Range<u32>| {
            let f = simulate_with(&[], &[Splash { born: 0.0, ..s }], 0.08, &at);
            f.instances[..f.over as usize].iter().filter(|i| layers.contains(&(i.params[1] as u32)) && i.colour[3] > 0.02).count()
        };
        assert!(pieces(first, CROWN_LAYER..JET_LAYER) >= 5 && pieces(into_the_hole, CROWN_LAYER..JET_LAYER) == 0);
        assert!(pieces(first, FROTH_LAYER..ATLAS_LAYERS) == 0 && pieces(into_the_hole, FROTH_LAYER..ATLAS_LAYERS) >= 4);
        assert!(pieces(into_the_hole, SPRAY_LAYER..FROTH_LAYER) >= 2);

        let mut live = vec![crate_box()];
        strike(&mut live, Splash { born: 0.2, ..pebble() }, 48);
        assert!(live[1].calm < 0.05);
        let mut live = vec![pebble()];
        strike(&mut live, Splash { born: 0.05, ..crate_box() }, 48);
        assert!(live[1].calm > 0.95, "{}", live[1].calm);

        let mut live = Vec::new();
        for k in 0..60 {
            strike(&mut live, Splash { born: f64::from(k), ..pebble() }, 48);
        }
        assert!(live.len() == 48 && live[0].born == 12.0);
    }

    #[test]
    fn drops_sparkle_against_the_sun_and_not_away_from_it() {

        let sun = [Light { kind: LightKind::Directional, direction: Vec3::new(0.0, -0.3, 1.0).normalize(), intensity: 5.0, ..lamp() }];
        let dark = |_: Vec3| Vec3::ZERO;
        let s = [step_splash(0.0)];
        let mean = |head: Vec3| {
            let f = simulate_with(&[], &s, 0.1, &Surroundings { lights: &sun, ambient: &dark, ..seen_from(head) });
            let drops = &f.instances[f.over as usize..];
            assert!(drops.len() >= 6);
            drops.iter().map(|i| i.colour[0]).sum::<f32>() / drops.len().max(1) as f32
        };
        let (into, away) = (mean(Vec3::new(0.0, 0.6, 3.0)), mean(Vec3::new(0.0, 0.6, -3.0)));
        assert!(into > 3.0 * away, "into the sun {into}, with it behind {away}");
        let shaded = [Splash { sunlit: 0.0, ..step_splash(0.0) }];
        let f = simulate_with(&[], &shaded, 0.1, &Surroundings { lights: &sun, ambient: &dark, ..seen_from(Vec3::new(0.0, 0.6, 3.0)) });
        assert!(f.instances.iter().all(|i| i.colour[0] < 1e-6), "a shaded splash has no sun in it");
    }

    fn ring_drawn(r: [f32; 4], w: [f32; 4], dist: f32) -> f32 {
        let from_rim = (dist - w[2]).max(1e-4);
        let ph = w[0] * r[2] * r[2] / from_rim;
        let k = ph / from_rim;
        if r[3] <= 0.0 || k <= 0.0 {
            return 0.0;
        }
        let t = (from_rim / w[2].max(1e-6)).min(1.0);
        -r[3] * k * k * k * (-k * (w[1] + k * r[2] * 2.0 * RING_FILM_VISCOSITY)).exp() * t / dist * ph.cos()
    }

    #[test]
    fn the_rings_that_matter_most_go_to_the_water_and_none_pops() {
        let head = Vec3::new(0.0, 1.6, 0.0);
        let shape = splash_shape(&pistol());

        let early = splash_rings::<4>(&[pistol()], 0.9 * f64::from(shape.ring_release), head);
        assert_eq!(early.rings[0][2], 0.0);
        assert!((0..400).all(|i| ring_drawn(early.rings[0], early.waves[0], 0.01 * i as f32) == 0.0));
        let one = splash_rings::<4>(&[pistol()], 1.0, head);
        assert!(one.rings[0][3] > 0.0 && one.rings[1][3] == 0.0, "{one:?}");
        assert!((one.rings[0][2] - (1.0 - shape.ring_release)).abs() < 1e-5);
        assert!((one.waves[0][0] - 0.25 * shape.ring_gravity).abs() < 1e-5 && one.waves[0][2] == shape.cavity);
        assert!(one.pools[0][0] > 0.05, "foam where it struck: {one:?}");

        let three = splash_rings::<4>(&[pistol()], 3.0, head);
        let profile: Vec<f32> = (0..3000).map(|i| ring_drawn(three.rings[0], three.waves[0], 0.001 * i as f32)).collect();
        let most = profile.iter().fold(0.0f32, |m, h| m.max(h.abs()));
        let crests: Vec<usize> = (1..2999).filter(|&i| profile[i] > profile[i - 1] && profile[i] >= profile[i + 1] && profile[i] > 0.01 * most).collect();
        assert!(crests.len() >= 4, "a train of crests, not one: {crests:?}");
        let gaps: Vec<usize> = crests.windows(2).map(|w| w[1] - w[0]).collect();
        assert!(gaps.windows(2).all(|g| g[1] >= g[0]), "crests further apart outward: {gaps:?}");
        let highest = crests.iter().copied().max_by(|a, b| profile[*a].total_cmp(&profile[*b])).unwrap();
        assert!(profile[*crests.last().unwrap()] < profile[highest] && highest > crests[0], "{crests:?}");

        let later = splash_rings::<4>(&[pistol()], 3.02, head);
        let next = (crests[1]..3000).find(|&i| ring_drawn(later.rings[0], later.waves[0], 0.001 * i as f32) > ring_drawn(later.rings[0], later.waves[0], 0.001 * (i - 1) as f32) && ring_drawn(later.rings[0], later.waves[0], 0.001 * i as f32) >= ring_drawn(later.rings[0], later.waves[0], 0.001 * (i + 1) as f32));
        assert!(next.is_some_and(|n| n > crests[1]));

        let open = splash_rings::<4>(&[pistol()], 0.5 * f64::from(shape.collapse), head);
        let boil = splash_rings::<4>(&[pistol()], f64::from(shape.collapse + 0.1), head);
        assert!(open.waves[0][3] > 1.0 && boil.waves[0][3] < 0.0, "{open:?} {boil:?}");

        let big = splash_rings::<4>(&[crate_box()], 1.0, head);
        let (crate_shape, small) = (splash_shape(&crate_box()), splash_shape(&pebble()));
        assert!(big.waves[0][2] > one.waves[0][2] && crate_shape.ring_height > shape.ring_height && shape.ring_height > small.ring_height, "{} {} {}", crate_shape.ring_height, shape.ring_height, small.ring_height);
        assert!(crate_shape.ring_wavelength > shape.ring_wavelength && crate_shape.ring_speed > shape.ring_speed);

        let gone = splash_rings::<4>(&[pistol(), crate_box()], f64::from(super::super::water_pipeline::RING_SECONDS) + 0.1, head);
        assert!(gone.rings.iter().all(|r| r[3] == 0.0));

        const N: usize = super::super::water_pipeline::RINGS;
        let steps: Vec<Splash> =
            (0..10).map(|k| Splash { born: 0.5 * f64::from(k), position: Vec3::new(0.3 * (k % 2) as f32, 0.0, 0.65 * k as f32), seed: k as u64, ..foot() }).collect();
        let drawn = |t: f64| {
            let r = splash_rings::<N>(&steps, t, Vec3::new(0.0, 1.6, 3.0));
            let line: Vec<f32> = (0..160)
                .map(|j| {
                    let p = glam::Vec2::new(0.6, -0.5 + 0.045 * j as f32);
                    (0..N).map(|i| ring_drawn(r.rings[i], r.waves[i], (p - glam::Vec2::new(r.rings[i][0], r.rings[i][1])).length())).sum::<f32>()
                })
                .collect();
            (line, r.pools.iter().map(|x| x[0]).sum::<f32>(), r.rings.iter().filter(|x| x[3] > 1e-9).count())
        };
        let peak = (0..1100).map(|k| drawn(0.005 * f64::from(k)).0.iter().fold(0.0f32, |m, h| m.max(h.abs()))).fold(0.0, f32::max);
        assert!(peak > 1e-4, "{peak}");
        assert!((0..1100).any(|k| drawn(0.005 * f64::from(k)).2 == N), "the wader fills the water's rings");
        for k in 0..1100 {
            let (a, b) = (drawn(0.005 * f64::from(k)), drawn(0.005 * f64::from(k + 1)));
            let jump = a.0.iter().zip(&b.0).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max);
            assert!(jump < 0.2 * peak, "heights jump at {} s: {jump} of {peak}", 0.005 * f64::from(k));

            assert!((a.1 - b.1).abs() < 0.1, "foam jumps at {} s: {} -> {}", 0.005 * f64::from(k), a.1, b.1);
        }
    }

    #[test]
    fn a_splash_out_of_range_or_out_of_sight_is_not_drawn() {
        let near = seen_from(Vec3::new(0.0, 1.6, 3.0));
        assert!(simulate_with(&[], &[pebble()], 0.1, &near).instances.len() > 5);
        let far = seen_from(Vec3::new(0.0, 1.6, 200.0));
        assert_eq!(simulate_with(&[], &[pebble()], 0.1, &far).instances.len(), 0);

        assert!(simulate_with(&[], &[crate_box()], 0.2, &far).instances.len() > 5);

        let mid = seen_from(Vec3::new(0.0, 1.6, 30.0));
        assert!(simulate_with(&[], &[pistol()], 0.1, &mid).screened * 2 < simulate_with(&[], &[pistol()], 0.1, &near).screened);

        let planes = |at: Vec3| {
            let vp = glam::Mat4::perspective_rh(1.5, 1.0, 0.03, 1000.0) * glam::Mat4::look_at_rh(Vec3::new(0.0, 1.6, 3.0), at, Vec3::Y);
            super::super::shadow::frustum_planes(vp)
        };
        assert!(splash_seen(&pistol(), 0.1, &[planes(Vec3::ZERO)]));
        assert!(!splash_seen(&pistol(), 0.1, &[planes(Vec3::new(0.0, 1.6, 9.0))]));
        assert!(!splash_seen(&pistol(), f64::from(SPLASH_SECONDS) + 0.1, &[planes(Vec3::ZERO)]));

        for s in [pebble(), pistol(), crate_box(), bullet(), foot()] {
            let (lo, hi) = splash_bounds(&s);
            for k in 1..40 {
                let f = simulate_with(&[], &[s], 0.05 * f64::from(k), &near);
                for i in &f.instances {
                    let p = Vec3::from_slice(&i.centre[..3]);
                    assert!(p.cmpge(lo - Vec3::splat(0.05)).all() && p.cmple(hi + Vec3::splat(0.05)).all(), "{p:?} outside {lo:?}..{hi:?} at {} s", 0.05 * k as f32);
                }
            }
        }
    }

    #[test]
    #[ignore]
    fn save_the_splash_textures() {
        let out = std::path::PathBuf::from(std::env::var("OUT").unwrap_or_else(|_| "/tmp".into()));
        let n = ATLAS_SIZE as usize;
        let cols = (CROWN_VARIANTS + JET_VARIANTS + SPRAY_VARIANTS + FROTH_VARIANTS) as usize;
        let (w, h) = (cols * n, 3 * n);
        let mut sheet = vec![0u8; w * h * 4];
        for stage in 0..3u32 {
            for c in 0..cols as u32 {
                let (layer, on_its_side) = if c < CROWN_VARIANTS {
                    (crown_frame(c, stage), true)
                } else if c < CROWN_VARIANTS + JET_VARIANTS {
                    (jet_frame(c - CROWN_VARIANTS, stage), false)
                } else if c < CROWN_VARIANTS + JET_VARIANTS + SPRAY_VARIANTS {
                    (spray_frame(c - CROWN_VARIANTS - JET_VARIANTS, stage), false)
                } else {
                    (froth_frame(c - CROWN_VARIANTS - JET_VARIANTS - SPRAY_VARIANTS, stage), false)
                };
                for y in 0..n {
                    for x in 0..n {

                        let (sx, sy) = if on_its_side { (n - 1 - y, x) } else { (x, y) };
                        let a = f32::from(layer[(sy * n + sx) * 4 + 3]) / 255.0;
                        let o = ((stage as usize * n + y) * w + c as usize * n + x) * 4;
                        let over = |under: f32| ((under * (1.0 - a) + a).powf(1.0 / 2.2) * 255.0) as u8;
                        sheet[o..o + 4].copy_from_slice(&[over(0.01), over(0.03), over(0.05), 255]);
                    }
                }
            }
        }
        image::save_buffer(out.join("splash_textures.png"), &sheet, w as u32, h as u32, image::ExtendedColorType::Rgba8).expect("textures");
    }

    #[test]
    fn a_crate_throws_white_water_and_foam_dissolves() {
        let (p, g, c) = (splash_shape(&pebble()), splash_shape(&pistol()), splash_shape(&crate_box()));
        assert!(p.broken == 0.0 && p.froth == 0 && g.froth == 0, "{} {}", p.froth, g.froth);
        assert!(c.broken > 0.6 && c.froth >= 6 && c.white > 0.6, "{c:?}");
        assert!(p.foam_seconds < g.foam_seconds && g.foam_seconds < c.foam_seconds);
        assert!((1.2..2.2).contains(&p.foam_seconds) && (3.0..4.5).contains(&c.foam_seconds));
        let head = Vec3::new(0.0, 1.6, 3.0);
        let foam = |s: Splash, t: f64| splash_rings::<4>(&[s], t, head).pools[0];

        let shut = f64::from(c.collapse) + 0.1;
        assert!(foam(crate_box(), shut)[0] > 0.5 && foam(crate_box(), 2.0)[0] < 0.4 && foam(crate_box(), 4.3)[0] == 0.0, "{:?}", foam(crate_box(), 2.0));
        assert!(foam(crate_box(), 2.0)[2] > foam(crate_box(), shut)[2]);
        assert!(foam(pebble(), 0.25)[0] > 0.2 && foam(pebble(), 1.8)[0] < 0.02, "{:?}", foam(pebble(), 0.25));

        assert!(foam(crate_box(), 2.0)[1] < foam(crate_box(), shut)[1]);
    }

    #[test]
    fn what_is_held_in_its_cavity_throws_no_jet() {
        let free = Splash { depth: 0.6, ..foot() };
        let held = Splash { held: 1.0, ..free };
        let (a, b) = (splash_shape(&free), splash_shape(&held));
        assert!(a.jet_height > 0.05 && b.jet_height == 0.0, "{} {}", a.jet_height, b.jet_height);
        assert!(a.cavity == b.cavity && a.sheet_speed == b.sheet_speed && a.spray == b.spray);
    }

    pub(crate) fn user_strikes() -> [(&'static str, Splash); 6] {
        let at = |v: Vec3, r: f32, m: f32, depth: f32| Splash { depth, seed: 11, ..Splash::new(Vec3::ZERO, 0.0, v, r, m) };
        [
            ("pebble 0.1 kg dropped 1 m", at(Vec3::new(0.0, -4.43, 0.0), 0.021, 0.1, 1.5)),
            ("fist rock 1 kg thrown", at(Vec3::new(6.0, -6.0, 0.0), 0.045, 1.0, 1.5)),
            ("boulder 8 kg dropped 1.5 m", at(Vec3::new(0.0, -5.4, 0.0), 0.09, 8.0, 1.5)),
            ("boulder 8 kg thrown", at(Vec3::new(4.0, -6.0, 0.0), 0.09, 8.0, 1.5)),
            ("crate 5 kg thrown", at(Vec3::new(3.0, -4.0, 0.0), 0.22, 5.0, 1.5)),
            ("crate 5 kg bobbing", Splash { held: 1.0, ..at(Vec3::new(0.0, -0.25, 0.0), 0.22, 5.0, 1.5) }),
        ]
    }

    #[test]
    #[ignore]
    fn print_the_causal_chain() {
        for (name, s) in user_strikes() {
            let c = splash_shape(&s);
            eprintln!(
                "{name}: cavity {:.3} m wide x {:.3} deep, open {:.0} ms; crown {:.2} m/s at {:.0} deg, {:.3} m high for {:.2} s, broken {:.2}; jet {:.3} m at {:.0} ms; \
                 RINGS leave at {:.0} ms from a {:.1} mm deep source: middle {:.3} m long at {:.2} m/s, first crest {:.1} mm, seen {:.1} s; \
                 drops land {:.2} m out (band {:.2}) from {:.0} ms over {:.0} ms, {:.0} a m2; foam {:.2} for {:.1} s",
                c.cavity, c.depth, c.collapse * 1e3, c.sheet_speed, c.sheet_angle.to_degrees(), c.crown_height, c.crown_seconds, c.broken,
                c.jet_height, c.collapse * 1e3, c.ring_release * 1e3, c.wave_source * 1e3, c.ring_wavelength, c.ring_speed, c.ring_height * 1e3,
                ring_seconds(&c), c.drops_land, c.drops_band, c.drops_from * 1e3, c.drops_over * 1e3, c.drops_density, c.foam, c.foam_seconds,
            );
        }
    }

    #[test]
    #[ignore]
    fn print_the_splashes() {
        let at = seen_from(Vec3::new(0.0, 1.6, 3.0));

        let share: Vec<f32> = cutouts(&atlas_layers())
            .iter()
            .map(|c| {
                let p: Vec<[f32; 2]> = (0..8).map(|k| [c[2 * k], c[2 * k + 1]]).collect();
                (0..8).map(|k| p[k][0] * p[(k + 1) % 8][1] - p[(k + 1) % 8][0] * p[k][1]).sum::<f32>().abs() / 8.0
            })
            .collect();
        for (name, s) in [("pebble", pebble()), ("pistol", pistol()), ("bullet", bullet()), ("foot", foot()), ("crate", crate_box())] {
            let shape = splash_shape(&s);
            let (mut most, mut most_over, mut area, mut drawn) = (0u32, 0u32, 0.0f32, 0.0f32);
            for k in 0..600 {
                let f = simulate_with(&[], &[s], 0.005 * f64::from(k), &at);
                most = most.max(f.over + f.screened);
                most_over = most_over.max(f.over);
                let a: f32 = f.instances[..f.over as usize].iter().map(|i| 4.0 * i.centre[3] * i.centre[3] * i.params[2].abs().max(1.0)).sum();
                area = area.max(a);

                let d: f32 = f.instances[..f.over as usize].iter().map(|i| 4.0 * i.centre[3] * i.centre[3] * i.params[2].abs().max(1.0) * share[(i.params[1] as usize).min(share.len() - 1)]).sum();
                drawn = drawn.max(d);
            }
            eprintln!(
                "{name:7} cavity {:.3} m x {:.3} deep, closes in {:.3} s, sealed {:.2}; sheet {:.2} m/s at {:.0} deg, crown {:.3} m for {:.2} s in {} pieces, broken {:.2}; drops {} of {:.2} mm (fine {:.2}), spray +{:.1} m/s; jet {:.3} m high, {:.3} wide; mist {}; rings {:.2} m long at {:.2} m/s, {:.1} mm high; foam {:.2}, settles {:.1} s; lasts {:.2} s | most instances {most} ({most_over} over), quads {area:.3} m2, drawn within their cutouts {drawn:.3} m2",
                shape.cavity, shape.depth, shape.collapse, shape.sealed, shape.sheet_speed, shape.sheet_angle.to_degrees(), shape.crown_height,
                shape.crown_seconds, shape.petals, shape.broken, shape.drops, shape.drop * 1e3, shape.fine, shape.spray_speed, shape.jet_height, shape.jet_width,
                shape.mist, shape.ring_wavelength, shape.ring_speed, shape.ring_height * 1e3, shape.foam, shape.settle, shape.seconds,
            );
        }
    }

    #[test]
    fn smoke_spreads_out_under_a_ceiling_instead_of_rising_through_it() {
        let at = seen_from(Vec3::new(0.0, 1.6, 3.0));
        let free = [emitter(EffectKind::Smoke)];
        let roofed = [EffectEmitter { ceiling: Some(1.2), ..emitter(EffectKind::Smoke) }];
        let tops = |e: &[EffectEmitter]| {
            let f = simulate(e, 40.0, &at);
            let ys: Vec<f32> = f.instances.iter().map(|i| i.centre[1]).collect();
            let wide = f.instances.iter().map(|i| (i.centre[0].powi(2) + (i.centre[2] - 0.0).powi(2)).sqrt()).fold(0.0, f32::max);
            (ys.iter().copied().fold(f32::MIN, f32::max), wide)
        };
        let (free_top, free_wide) = tops(&free);
        let (roof_top, roof_wide) = tops(&roofed);
        assert!(free_top > 1.5, "free smoke rises past 1.5 m: {free_top}");
        assert!(roof_top < 1.2, "roofed smoke stays under it: {roof_top}");
        assert!(roof_wide > free_wide, "and spreads wider: {roof_wide} vs {free_wide}");
    }

    #[test]
    fn a_tumbling_mote_flashes_and_is_lost_and_averages_its_plain_light() {

        let samples: Vec<f32> = (0..200_000).map(|k| mote_glint(k as f32 * 0.013, 1.0, 7 + (k / 5000) as u64)).collect();
        let mean = samples.iter().sum::<f32>() / samples.len() as f32;
        let dim = samples.iter().filter(|&&g| g < 0.6).count() as f32 / samples.len() as f32;
        let peak = samples.iter().copied().fold(0.0, f32::max);
        assert!((mean - 1.0).abs() < 0.06, "mean {mean}");
        assert!(dim > 0.3, "lost {dim} of the time");
        assert!(peak > 2.5, "brightest flash {peak}");
    }

    #[test]
    fn the_air_carries_neighbours_together_at_a_few_centimetres_a_second() {
        let air = air_waves(id_hash("hall_dust"));
        let moved = |p: Vec3, t: f64| air_drift(&air, p, t + 1.0) - air_drift(&air, p, t);
        let (mut speed, mut apart_near, mut apart_far) = (0.0f32, 0.0f32, 0.0f32);
        let n = 400;
        for k in 0..n {
            let u = uniforms(k as u64);
            let p = Vec3::new(u[0], u[1], u[2]) * 4.0;
            let t = 10.0 + 37.0 * f64::from(u[3]);
            let a = moved(p, t);
            speed += a.length();
            apart_near += (a - moved(p + Vec3::new(0.03, 0.0, 0.02), t)).length();
            apart_far += (a - moved(p + Vec3::new(2.5, 0.4, -1.7), t)).length();
        }
        let (speed, near, far) = (speed / n as f32, apart_near / n as f32, apart_far / n as f32);
        assert!((0.008..0.05).contains(&speed), "mean drift {speed} m/s");
        assert!(near < 0.35 * speed, "4 cm apart they mostly move together: {near} vs {speed}");
        assert!(far > 0.6 * speed, "metres apart they do not: {far} vs {speed}");
    }

    #[test]
    fn a_mote_is_lost_out_of_focus_and_past_its_distance() {

        let e = [EffectEmitter { extent: [Vec3::X * 0.01, Vec3::Y * 0.01, Vec3::Z * 0.01], rate: 3000.0, ..emitter(EffectKind::Dust) }];
        let strength = |head: Vec3| {
            let f = simulate(&e, 40.0, &seen_from(head));
            f.instances.iter().map(|i| i.colour[3]).sum::<f32>() / f.instances.len().max(1) as f32
        };
        let (held, close, far) = (strength(Vec3::new(0.0, 0.0, 2.0)), strength(Vec3::new(0.0, 0.0, 0.08)), strength(Vec3::new(0.0, 0.0, 4.9)));
        assert!(held > 0.3, "{held}");
        assert!(close < 0.05 * held, "out of focus {close} vs {held}");
        assert!(far < 0.15 * held, "at the edge of sight {far} vs {held}");
    }

    #[test]
    fn a_cycle_does_not_repeat_the_one_before() {

        let e = [EffectEmitter { rate: 0.2, ..emitter(EffectKind::Smoke) }];
        let p = preset(EffectKind::Smoke);
        let slots = ((p.rate * 0.2 * p.life.1).ceil() as usize).max(1);
        let period = (slots as f64 / (f64::from(p.rate) * 0.2)).max(f64::from(p.life.1));
        let at = seen_from(Vec3::new(0.0, 1.6, 3.0));
        let a = simulate(&e, 3.0, &at);
        let b = simulate(&e, 3.0 + period, &at);
        assert!(!a.instances.is_empty() && !b.instances.is_empty());
        assert_ne!(a.instances[0].centre, b.instances[0].centre);
    }

    #[test]
    fn smoke_rises_and_spreads_and_flames_stand_on_their_bed() {
        let at = seen_from(Vec3::new(0.0, 1.6, 3.0));
        let smoke = simulate(&[emitter(EffectKind::Smoke)], 50.0, &at);
        let high = smoke.instances.iter().map(|i| i.centre[1]).fold(f32::MIN, f32::max);
        let biggest = smoke.instances.iter().map(|i| i.centre[3]).fold(0.0f32, f32::max);
        assert!(high > 1.0, "smoke reaches above a metre, highest {high}");
        assert!(biggest > 0.4, "old puffs are big, biggest {biggest}");
        for time in [50.0, 51.3, 52.7] {
            let fire = simulate(&[emitter(EffectKind::Fire)], time, &at);
            for i in &fire.instances {

                let bottom = i.centre[1] - i.centre[3] * i.params[2].abs();
                assert!(bottom < 0.0 && bottom > -0.04, "a flame stands on its bed: bottom {bottom}");
                assert!(i.params[0].abs() < 0.1 && i.light[3] == 1.0, "upright: {}", i.params[0]);
                assert!((0.8 * FLAME_STRETCH..1.25 * FLAME_STRETCH).contains(&i.params[2].abs()), "{}", i.params[2]);
                let frame = i.params[1] - FIRE_FIRST as f32;
                assert!((0.0..=(FIRE_FRAMES - 1) as f32).contains(&frame), "inside the book: {frame}");
            }
            assert!(fire.over >= 3 && fire.screened == 0, "sheets drawn over, sorted with their smoke: {}", fire.over);
            assert!(fire.instances.iter().any(|i| i.params[2] < 0.0), "some mirrored");
        }
    }

    #[test]
    fn a_flame_is_deep_red_at_its_edges_and_pale_yellow_at_its_core() {
        let (edge, body, core) = (flame_colour(0.25), flame_colour(0.6), flame_colour(1.0));
        assert!(edge.y < 0.1 * edge.x && edge.z < 0.02 * edge.x, "{edge}");
        assert!(body.y > 0.15 * body.x && body.y < 0.4 * body.x, "orange: {body}");
        assert!(core.y > 0.5 * core.x && core.z > 0.05 * core.x, "{core}");
        let lum: Vec<f32> = (0..=20).map(|k| luminance(flame_colour(k as f32 / 20.0))).collect();
        assert!(lum.windows(2).all(|w| w[1] > w[0]), "brighter as it is hotter");
        assert!(lum[20] > 10.0 * lum[5], "steeply: {} vs {}", lum[20], lum[5]);
    }

    #[test]
    fn a_flame_frame_is_read_where_its_gas_was() {

        let dt = FLAME_SECONDS / FIRE_FRAMES as f32;
        for v in [0.1f32, 0.4, 0.8] {
            assert!(flame_v(v, dt) < v && flame_v(v, -dt) > v);
            assert!((flame_v(flame_v(v, dt), -dt) - v).abs() < 1e-5);
        }
        assert!(v_rise(0.8, dt) > 2.0 * v_rise(0.1, dt), "tongues stretch as they rise");
        assert_eq!(flame_v(0.02, dt), 0.02);
        fn v_rise(v: f32, dt: f32) -> f32 {
            flame_v(v, -dt) - v
        }
    }

    #[test]
    fn smoke_is_drawn_back_to_front() {
        let head = Vec3::new(0.0, 1.0, 2.0);
        let f = simulate(&[emitter(EffectKind::Smoke)], 77.0, &seen_from(head));
        let d: Vec<f32> = f.instances[..f.over as usize].iter().map(|i| (centre(i) - head).length()).collect();
        assert!(d.windows(2).all(|w| w[0] >= w[1]), "far to near");
    }

    #[test]
    fn a_lamp_lights_the_smoke_it_reaches_and_not_what_it_does_not() {
        let lamps = [lamp()];
        let none = |_: &EffectEmitter, _: usize| false;
        let dark = |_: Vec3| Vec3::ZERO;
        let head = Vec3::new(0.0, 1.6, 3.0);
        let lit_at = Surroundings { lights: &lamps, ambient: &dark, ..seen_from(head) };
        let lit = simulate(&[emitter(EffectKind::Smoke)], 40.0, &lit_at);
        let dark_at = Surroundings { lights: &lamps, reaches: &none, ambient: &dark, ..seen_from(head) };
        let unlit = simulate(&[emitter(EffectKind::Smoke)], 40.0, &dark_at);
        let sum = |f: &EffectFrame| f.instances.iter().map(|i| i.light[0] + i.colour[0]).sum::<f32>();
        assert!(sum(&lit) > 0.1, "the lamp lights the smoke");
        assert_eq!(sum(&unlit), 0.0, "a lamp an emitter may not take lights nothing");

        let i = lit.instances.iter().find(|i| i.light[0] > 0.0).unwrap();
        assert!(i.light_dir[1] > 0.5);
    }

    #[test]
    fn a_mote_sparkles_looking_toward_its_lamp() {

        let lamps = [Light { position: Vec3::new(0.0, 0.0, -2.0), ..lamp() }];
        let dark = |_: Vec3| Vec3::ZERO;
        let e = [EffectEmitter { extent: [Vec3::X * 0.1, Vec3::Y * 0.1, Vec3::Z * 0.1], rate: 4.0, ..emitter(EffectKind::Dust) }];
        let mean = |head: Vec3| {
            let f = simulate(&e, 60.0, &Surroundings { lights: &lamps, ambient: &dark, ..seen_from(head) });
            f.instances.iter().map(|i| i.colour[0]).sum::<f32>() / f.instances.len().max(1) as f32
        };
        let (into, from) = (mean(Vec3::new(0.0, 0.0, 2.0)), mean(Vec3::new(0.0, 0.0, -1.5)));
        assert!(into > 10.0 * from, "into the light {into}, from its side {from}");
    }

    #[test]
    fn a_fire_in_a_dim_room_is_shown_as_the_adapted_eye_sees_it() {

        let bright = Surroundings { exposure: 30.0, ..seen_from(Vec3::new(0.0, 0.5, 1.5)) };
        let f = simulate(&[emitter(EffectKind::Fire)], 20.0, &bright);
        let peak = f.instances.iter().map(|i| i.colour[0]).fold(0.0f32, f32::max);
        assert!(peak <= FIRE_LEVEL * 1.01 && peak > 0.5 * FIRE_LEVEL, "{peak}");
    }

    #[test]
    fn a_fires_smoke_is_lit_by_it_as_its_flames_are_seen() {

        let fire = EffectEmitter { id: "f".into(), scale: 1.5, ..emitter(EffectKind::Fire) };
        let smoke = EffectEmitter { id: "s".into(), position: Vec3::new(0.0, 0.5, 0.0), ..emitter(EffectKind::Smoke) };
        let lights = [fire_light(&fire, 10.0, Vec3::ZERO, Quat::IDENTITY).unwrap()];
        let dark = |_: Vec3| Vec3::ZERO;
        let at = Surroundings { lights: &lights, ambient: &dark, exposure: 9.0, ..seen_from(Vec3::new(0.0, 1.2, 1.8)) };
        let f = simulate(&[fire, smoke], 10.0, &at);
        let brightest = |mode: f32, of: &dyn Fn(&EffectInstance) -> f32| {
            f.instances.iter().filter(|i| i.params[3] == mode).map(of).fold(0.0f32, f32::max)
        };
        let flames = brightest(2.0, &|i| i.colour[0]);
        let smoke = brightest(1.0, &|i| i.light[0] + i.colour[0]);
        assert!(flames > 0.8 * FIRE_LEVEL, "{flames}");
        assert!(smoke > 0.0 && smoke < 0.5 * flames, "smoke {smoke}, flames {flames}");
    }

    #[test]
    fn a_bed_of_coals_lies_flat_under_its_fire_and_is_drawn_first() {
        let fire = EffectEmitter { id: "f".into(), scale: 1.5, ..emitter(EffectKind::Fire) };
        let smoke = EffectEmitter { id: "s".into(), position: Vec3::new(0.0, 0.5, 0.0), ..emitter(EffectKind::Smoke) };
        let coals = EffectEmitter { id: "c".into(), scale: 1.5, ..emitter(EffectKind::Coals) };
        let at = seen_from(Vec3::new(0.0, 1.6, 2.0));
        let f = simulate(&[fire, smoke, coals], 20.0, &at);
        let bed = &f.instances[0];
        assert_eq!(bed.params[3], 3.0, "the bed first, under everything");
        assert_eq!(bed.light[3], 2.0, "lying flat");
        assert_eq!(bed.axis[3], 0.0, "not faded into the floor it lies on");
        assert!(bed.centre[1].abs() < 0.01 && (bed.centre[3] - COALS_HALF * 1.5).abs() < 1e-6, "{:?}", bed.centre);
        assert_eq!(f.instances.iter().filter(|i| i.params[3] == 3.0).count(), 1, "one bed");
        let off = simulate(&[EffectEmitter { rate: 0.0, ..emitter(EffectKind::Coals) }], 20.0, &at);
        assert!(off.instances.is_empty(), "stopped with its rate");
    }

    #[test]
    fn coals_glow_in_the_middle_and_go_to_ash_at_the_rim() {
        let c = coals_frame();
        let n = ATLAS_SIZE as usize;
        let mean = |r0: f32, r1: f32, channel: usize| {
            let (mut sum, mut count) = (0.0f32, 0.0f32);
            for y in 0..n {
                for x in 0..n {
                    let r = ((x as f32 + 0.5) / n as f32 * 2.0 - 1.0).hypot((y as f32 + 0.5) / n as f32 * 2.0 - 1.0);
                    let o = (y * n + x) * 4;
                    if (r0..r1).contains(&r) && c[o + 3] > 128 {
                        sum += f32::from(c[o + channel]);
                        count += 1.0;
                    }
                }
            }
            sum / count.max(1.0)
        };
        assert!(mean(0.0, 0.3, 0) > 2.0 * mean(0.55, 0.75, 0), "hotter in the middle");
        assert!(mean(0.55, 0.75, 1) > 1.5 * mean(0.0, 0.3, 1), "paler ash at the rim");
        assert_eq!(c[3], 0, "nothing in the corner");
    }

    #[test]
    fn a_fire_is_metered_from_its_own_room_and_not_through_a_wall() {
        let room = |lo: Vec3, hi: Vec3, volume: u32| ProbeDesc {
            centre: 0.5 * (lo + hi),
            min: lo,
            max: hi,
            volume,
            has_depth: true,
            room_light: None,
        };
        let descs = [room(Vec3::new(-3.0, 0.0, -3.0), Vec3::new(3.0, 3.0, 3.0), 1), room(Vec3::new(3.0, 0.0, -3.0), Vec3::new(9.0, 3.0, 3.0), 2)];
        let fire = [EffectEmitter { scale: 1.5, ..emitter(EffectKind::Fire) }];
        let near = meter_samples(&fire, Vec3::new(0.3, 1.6, 2.0), &descs);
        assert!(near.len() > 20, "its flames and its pool: {}", near.len());
        assert!(near.iter().all(|s| s.1 > 0.0 && s.2 > 0.0 && (s.0.length() - 1.0).abs() < 1e-4));

        let pool = &near[1..];
        let at_foot = pool.iter().map(|s| s.1).fold(0.0f32, f32::max);
        let far = pool.iter().map(|s| s.1).fold(f32::MAX, f32::min);
        assert!(at_foot > 10.0 * far, "{at_foot} vs {far}");
        assert!(near[0].1 > 10.0 * at_foot, "the flames outshine their pool");
        assert!(meter_samples(&fire, Vec3::new(6.0, 1.6, 0.0), &descs).is_empty(), "the next room sees a wall");
        assert!(!meter_samples(&fire, Vec3::new(6.0, 1.6, 0.0), &[]).is_empty(), "without rooms, from anywhere");
        assert!(meter_samples(&[emitter(EffectKind::Smoke)], Vec3::new(0.3, 1.6, 2.0), &descs).is_empty());
    }

    #[test]
    fn a_fire_lights_the_room_with_a_flicker_and_nothing_else_does() {
        let fire = EffectEmitter { position: Vec3::new(14.0, 0.0, -4.0), scale: 1.5, ..emitter(EffectKind::Fire) };
        let (offset, yaw_inv) = (Vec3::new(14.0, 0.0, 0.0), Quat::from_rotation_y(-0.7));
        let at = |t: f64| fire_light(&fire, t, offset, yaw_inv).unwrap();

        let world = |l: &Light| Quat::from_rotation_y(0.7) * l.position + offset;
        let samples: Vec<Light> = (0..2000).map(|k| at(1000.0 + f64::from(k) * 0.013)).collect();
        let rest = Vec3::new(14.0, 0.525, -4.0);
        let far = samples.iter().map(|l| (world(l) - rest).length()).fold(0.0f32, f32::max);
        let moved = samples.windows(2).map(|w| (world(&w[0]) - world(&w[1])).length()).sum::<f32>();
        assert!(far < FIRE_LIGHT_REACH * 1.5 && far > 0.01, "sways within reach: {far}");
        assert!(moved > 0.05, "and does move: {moved}");

        let nominal = FIRE_LIGHT_INTENSITY * 1.5 * 1.5;
        let k: Vec<f32> = samples.iter().map(|l| l.intensity / nominal).collect();
        let mean = k.iter().sum::<f32>() / k.len() as f32;
        let (lo, hi) = k.iter().fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
        assert!((mean - 0.8).abs() < 0.06 && lo > 0.3 && hi < 1.4 && hi - lo > 0.2, "mean {mean}, {lo} to {hi}");

        let l = &samples[0];
        assert!(l.source_radius > 0.1 && !l.in_level_bake && l.mask_channel.is_none());
        assert!(fire_light(&emitter(EffectKind::Smoke), 1.0, offset, yaw_inv).is_none());
        assert!(fire_light(&EffectEmitter { rate: 0.0, ..fire.clone() }, 1.0, offset, yaw_inv).is_none());

        let steady = EffectEmitter { variation: Variation { flicker: 0.0, ..Variation::default() }, ..fire };
        let s = fire_light(&steady, 1000.3, offset, yaw_inv).unwrap();
        assert!((s.intensity / nominal - 0.8).abs() < 1e-5, "{}", s.intensity / nominal);
    }

    fn spectrum(x: &[f32], rate: f32, freqs: &[f32]) -> Vec<f32> {
        let n = x.len();
        let mean = x.iter().sum::<f32>() / n as f32;
        let w = |i: usize| 0.5 - 0.5 * (std::f32::consts::TAU * i as f32 / n as f32).cos();
        let at = |f: f32| {
            let (mut re, mut im) = (0.0f64, 0.0f64);
            for (i, v) in x.iter().enumerate() {
                let a = std::f64::consts::TAU * f64::from(f) * i as f64 / f64::from(rate);
                re += f64::from((v - mean) * w(i)) * a.cos();
                im += f64::from((v - mean) * w(i)) * a.sin();
            }
            (re * re + im * im) as f32
        };
        freqs.iter().map(|&f| (at(f - 0.1) + at(f) + at(f + 0.1)) / 3.0).collect()
    }

    #[test]
    fn a_fires_light_pulses_at_the_rate_its_flames_puff() {

        for scale in [1.0f32, 4.0] {
            let fire = EffectEmitter { scale, ..emitter(EffectKind::Fire) };
            let puff = 1.5 / (0.3 * scale).sqrt();
            assert!((fire_puff_hz(&fire) - puff).abs() < 0.01 * puff, "{} vs {puff}", fire_puff_hz(&fire));
            let rate = 40.0;
            let light: Vec<f32> = (0..(rate as usize * 60)).map(|k| fire_light(&fire, 500.0 + k as f64 / f64::from(rate), Vec3::ZERO, Quat::IDENTITY).unwrap().intensity).collect();
            let freqs: Vec<f32> = (2..=60).map(|k| k as f32 * 0.2).collect();
            let power = spectrum(&light, rate, &freqs);
            let peak = freqs[power.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0];
            let band = |lo: f32, hi: f32| {
                let v: Vec<f32> = freqs.iter().zip(&power).filter(|(f, _)| (lo..hi).contains(*f)).map(|(_, p)| *p).collect();
                v.iter().sum::<f32>() / v.len() as f32
            };
            let (near, high) = (band(0.7 * puff, 1.3 * puff), band(3.0 * puff, 4.0 * puff));
            eprintln!("scale {scale}: puff {puff:.2} Hz, peak {peak:.2} Hz, near {near:.3e}, three to four times above {high:.3e}");
            assert!((0.6 * puff..1.5 * puff).contains(&peak), "scale {scale}: peak {peak} Hz, puffing {puff} Hz");
            assert!(near > 8.0 * high, "scale {scale}: {near} vs {high}");
        }
    }

    #[test]
    fn a_fires_light_follows_how_much_flame_is_in_view() {
        let fire = EffectEmitter { scale: 1.5, ..emitter(EffectKind::Fire) };
        let times: Vec<f64> = (0..3000).map(|k| 200.0 + f64::from(k) * 0.021).collect();
        let glow: Vec<f32> = times.iter().map(|&t| fire_glow(&fire, t)).collect();
        let light: Vec<f32> = times.iter().map(|&t| fire_light(&fire, t, Vec3::ZERO, Quat::IDENTITY).unwrap().intensity).collect();
        let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len() as f32;
        let (mg, ml) = (mean(&glow), mean(&light));
        let cov: f32 = glow.iter().zip(&light).map(|(g, l)| (g - mg) * (l - ml)).sum();
        let sd = |v: &[f32], m: f32| v.iter().map(|x| (x - m) * (x - m)).sum::<f32>().sqrt();
        let r = cov / (sd(&glow, mg) * sd(&light, ml));
        let spread = sd(&glow, mg) / (glow.len() as f32).sqrt();
        eprintln!("glow mean {mg:.3}, sd {spread:.3}; light against it r = {r:.3}");
        assert!((mg - 1.0).abs() < 0.08, "the glow is 1 on average: {mg}");
        assert!((0.06..0.4).contains(&spread), "and swells and dims: {spread}");
        assert!(r > 0.85, "the light follows the flames: r = {r}");
    }

    #[test]
    fn a_bigger_fire_takes_time_as_the_root_of_its_size_and_is_more_tongues() {

        let small = emitter(EffectKind::Fire);
        let big = EffectEmitter { scale: 4.0, ..emitter(EffectKind::Fire) };
        assert!((fire_tempo(&big) / fire_tempo(&small) - 2.0).abs() < 1e-5);
        assert!((fire_puff_hz(&small) / fire_puff_hz(&big) - 2.0).abs() < 1e-4);
        let stats = |e: &EffectEmitter| {
            let (mut count, mut tall, mut wide, mut pace) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
            let n = 400;
            for k in 0..n {
                let t = 300.0 + f64::from(k) * 0.37;
                let (a, b) = (fire_sheets(e, t), fire_sheets(e, t + 0.01));
                count += a.len() as f32;
                tall += a.iter().map(|s| s.size * s.stretch).sum::<f32>() / a.len() as f32;
                wide += a.iter().map(|s| s.size).sum::<f32>() / a.len() as f32;

                let moved: Vec<f32> = a.iter().zip(&b).filter(|(x, y)| x.base == y.base && y.frame > x.frame && y.frame < (FIRE_FRAMES - 1) as f32).map(|(x, y)| (y.frame - x.frame) / 0.01).collect();
                pace += moved.iter().sum::<f32>() / moved.len().max(1) as f32;
            }
            let n = n as f32;
            (count / n, tall / n, wide / n, pace / n)
        };
        let (c1, t1, w1, p1) = stats(&small);
        let (c4, t4, w4, p4) = stats(&big);
        eprintln!("scale 1: {c1:.1} sheets {t1:.3} m tall {w1:.3} wide {p1:.1} frames/s; scale 4: {c4:.1} {t4:.3} {w4:.3} {p4:.1}");
        assert!((c4 / c1 - 2.0).abs() < 0.3, "sheets alive as sqrt(scale): {c1} vs {c4}");
        assert!((t4 / t1 - 4.0).abs() < 0.4, "as tall as the fire: {t1} vs {t4}");
        assert!((w4 / w1 - 4f32.powf(0.75)).abs() < 0.3, "narrower for their height: {w1} vs {w4}");
        assert!((p1 / p4 - 2.0).abs() < 0.2, "the book played at 1/sqrt(scale): {p1} vs {p4}");

        assert!(((t4 * p4) / (t1 * p1) - 2.0).abs() < 0.3);
    }

    #[test]
    fn a_fires_sheets_puff_together() {

        let fire = EffectEmitter { scale: 1.5, ..emitter(EffectKind::Fire) };
        let rate = 40.0;

        let area: Vec<f32> = (0..(rate as usize * 60))
            .map(|k| {
                let sheets = fire_sheets(&fire, 700.0 + k as f64 / f64::from(rate));
                let w = |s: &FlameSheet| s.size * s.size * s.stretch * erosion_cover(s.stands) * rise_cover(s.risen);
                sheets.iter().map(|s| w(s) * flame_area(s.frame)).sum::<f32>() / sheets.iter().map(w).sum::<f32>()
            })
            .collect();
        let puff = fire_puff_hz(&fire);
        let freqs: Vec<f32> = (2..=40).map(|k| k as f32 * 0.2).collect();
        let power = spectrum(&area, rate, &freqs);
        let peak = freqs[power.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0];
        assert!((0.75 * puff..1.3 * puff).contains(&peak), "the book's puffs add up at the beat: peak {peak} Hz, beat {puff} Hz");
    }

    #[test]
    fn the_book_puffs_and_its_sheets_erode_from_their_tips() {

        let (area, cover, _) = flame_tables();
        let cycles = FLAME_SECONDS * puff_book_hz();
        let power = |c: f32| {
            let (mut re, mut im) = (0.0f32, 0.0f32);
            for (k, a) in area.iter().enumerate() {
                let x = std::f32::consts::TAU * c * k as f32 / FIRE_FRAMES as f32;
                re += (a - 1.0) * x.cos();
                im += (a - 1.0) * x.sin();
            }
            re * re + im * im
        };
        let at_beat = power(cycles.round());
        assert!((1..12).filter(|&c| c as f32 != cycles.round()).all(|c| power(c as f32) < at_beat), "{cycles} cycles");

        assert!(cover.windows(2).all(|w| w[1] > w[0]) && cover[0] < 0.05 && cover[8] == 1.0, "{cover:?}");

        let f = fire_frame(20);
        let n = ATLAS_SIZE as usize;
        let depth = |rows: std::ops::Range<usize>| {
            let (mut sum, mut count) = (0.0f32, 0.0f32);
            for y in rows {
                for x in 0..n {
                    let t = &f[(y * n + x) * 4..(y * n + x) * 4 + 4];
                    if t[3] > 128 {
                        sum += f32::from(t[1]);
                        count += 1.0;
                    }
                }
            }
            sum / count.max(1.0)
        };
        assert!(depth(n / 4..n / 2) < 0.6 * depth(3 * n / 4..n), "tips shallow, root deep");

        let blue = |rows: std::ops::Range<usize>| rows.flat_map(|y| (0..n).map(move |x| (y, x))).map(|(y, x)| f32::from(f[(y * n + x) * 4 + 2])).sum::<f32>();
        assert!(blue(n - n / 12..n) > 0.0 && blue(0..n / 2) == 0.0);
    }

    #[test]
    fn the_flame_tables_are_the_books() {
        let (area, cover, rise) = flame_tables();
        for (k, (a, b)) in rise.iter().zip(RISE_COVER).enumerate() {
            assert!((a - b).abs() < 2e-3, "RISE_COVER[{k}]: book {a}, table {b}: print_the_flame_tables");
        }
        for (k, (a, b)) in area.iter().zip(FLAME_AREA).enumerate() {
            assert!((a - b).abs() < 2e-3, "FLAME_AREA[{k}]: book {a}, table {b}: print_the_flame_tables");
        }
        for (k, (a, b)) in cover.iter().zip(EROSION_COVER).enumerate() {
            assert!((a - b).abs() < 2e-3, "EROSION_COVER[{k}]: book {a}, table {b}: print_the_flame_tables");
        }
    }

    #[test]
    fn variation_at_its_defaults_changes_nothing_and_a_seed_changes_the_pattern() {
        let at = Surroundings { lights: &[], ..seen_from(Vec3::new(0.3, 1.6, 3.0)) };
        let kinds = [EffectKind::Fire, EffectKind::Smoke, EffectKind::Embers, EffectKind::Dust, EffectKind::Coals];
        let plain: Vec<EffectEmitter> = kinds.iter().enumerate().map(|(k, &kind)| EffectEmitter { id: format!("e{k}"), ..emitter(kind) }).collect();

        let named: Vec<EffectEmitter> = plain
            .iter()
            .map(|e| EffectEmitter {
                variation: Variation {
                    intensity: 1.0,
                    size_variation: 1.0,
                    life_variation: 1.0,
                    speed_variation: 1.0,
                    temperature_variation: 0.0,
                    flicker: 1.0,
                    flicker_rate: 1.0,
                    turbulence: 1.0,
                    seed: 0,
                }
                .clamped(),
                ..e.clone()
            })
            .collect();
        for time in [10.0, 47.31] {
            let (a, b) = (simulate(&plain, time, &at), simulate(&named, time, &at));
            assert_eq!(bytemuck::cast_slice::<_, u8>(&a.instances), bytemuck::cast_slice::<_, u8>(&b.instances));
        }

        let seeded: Vec<EffectEmitter> = plain.iter().map(|e| EffectEmitter { variation: Variation { seed: 7, ..Variation::default() }, ..e.clone() }).collect();
        for (p, s) in plain.iter().zip(&seeded) {
            let (a, b) = (simulate(std::slice::from_ref(p), 33.3, &at), simulate(std::slice::from_ref(s), 33.3, &at));
            assert!(!a.instances.is_empty(), "{:?}", p.kind);
            assert_ne!(bytemuck::cast_slice::<_, u8>(&a.instances), bytemuck::cast_slice::<_, u8>(&b.instances), "{:?}", p.kind);
            if p.kind != EffectKind::Embers {

                let slots = |e: &EffectEmitter| (0..20).map(|k| simulate(std::slice::from_ref(e), 40.0 + f64::from(k) * 0.31, &at).instances.len()).sum::<usize>() as f32;
                assert!((slots(p) / slots(s) - 1.0).abs() < 0.25, "{:?}", p.kind);
            }
        }

        let wild = Variation { turbulence: 99.0, flicker_rate: 0.0, seed: 70000, intensity: f32::NAN, ..Variation::default() }.clamped();
        assert_eq!((wild.turbulence, wild.flicker_rate, wild.seed, wild.intensity), (4.0, 0.25, 65535, 1.0));
    }

    #[test]
    fn variation_does_what_it_says() {
        let at = seen_from(Vec3::new(0.3, 1.6, 3.0));
        let with = |kind: EffectKind, v: Variation| EffectEmitter { variation: v, ..emitter(kind) };
        let d = Variation::default();

        let f1 = simulate(&[with(EffectKind::Fire, d)], 20.0, &at);
        let f2 = simulate(&[with(EffectKind::Fire, Variation { intensity: 2.0, ..d })], 20.0, &at);
        assert!((f2.instances[0].colour[0] / f1.instances[0].colour[0] - 2.0).abs() < 1e-4);
        let l = |v: Variation| fire_light(&with(EffectKind::Fire, v), 20.0, Vec3::ZERO, Quat::IDENTITY).unwrap().intensity;
        assert!((l(Variation { intensity: 2.0, ..d }) / l(d) - 2.0).abs() < 1e-4);

        let hot = simulate(&[with(EffectKind::Fire, Variation { temperature_variation: 1.0, ..d })], 20.0, &at);
        let ratios: Vec<f32> = hot.instances.iter().map(|i| i.colour[2] / i.colour[0]).collect();
        assert!(ratios.iter().copied().fold(0.0, f32::max) > 1.5 * ratios.iter().copied().fold(f32::MAX, f32::min));

        assert!((fire_puff_hz(&with(EffectKind::Fire, Variation { flicker_rate: 2.0, ..d })) / fire_puff_hz(&with(EffectKind::Fire, d)) - 2.0).abs() < 1e-4);

        let s1 = simulate(&[with(EffectKind::Smoke, d)], 40.0, &at);
        let s2 = simulate(&[with(EffectKind::Smoke, Variation { intensity: 0.5, ..d })], 40.0, &at);
        assert!((s2.instances[3].colour[3] / s1.instances[3].colour[3] - 0.5).abs() < 1e-4);

        let top = |v: Variation| (0..20).flat_map(|k| simulate(&[with(EffectKind::Smoke, v)], 60.0 + f64::from(k), &at).instances).map(|i| i.centre[1]).fold(0.0f32, f32::max);
        assert!(top(Variation { life_variation: 2.0, ..d }) > top(d) + 0.1);

        let c = simulate(&[with(EffectKind::Coals, Variation { flicker: 0.0, flicker_rate: 2.0, ..d })], 20.0, &at);
        assert_eq!(c.instances[0].light_dir[3], -1.0);
        assert!((c.instances[0].params[2] - 40.0).abs() < 1e-4);
    }

    #[test]
    fn embers_pop_in_bursts_carried_up_by_the_plume() {

        let at = seen_from(Vec3::new(0.3, 1.6, 3.0));
        let embers = emitter(EffectKind::Embers);
        let (mut most, mut least, mut high) = (0u32, u32::MAX, 0.0f32);
        for k in 0..600 {
            let mut pops = Vec::new();
            ember_pops(&embers, 100.0 + f64::from(k) * 0.1, &at, &mut pops);
            let n = pops.len() as u32;
            most = most.max(n);
            least = least.min(n);
            high = pops.iter().map(|i| i.centre[1]).fold(high, f32::max);
        }

        let stream_high = (0..200)
            .flat_map(|k| {
                let time = 100.0 + f64::from(k) * 0.3;
                let mut pops = Vec::new();
                ember_pops(&embers, time, &at, &mut pops);
                let popped: Vec<[f32; 4]> = pops.iter().map(|i| i.centre).collect();
                simulate(std::slice::from_ref(&embers), time, &at).instances.into_iter().filter(move |i| !popped.contains(&i.centre))
            })
            .map(|i| i.centre[1])
            .fold(0.0f32, f32::max);
        eprintln!("pops: most {most} alight at once, least {least}, highest {high:.2} m; stream's highest {stream_high:.2} m");
        assert_eq!(least, 0, "between pops there are none");
        assert!(most >= 6, "a pop is a handful of sparks: {most}");
        assert!(high > 1.3 * stream_high, "carried up by the plume: {high} vs {stream_high}");
        let none = EffectEmitter { rate: 0.0, ..embers };
        let mut pops = Vec::new();
        ember_pops(&none, 100.0, &at, &mut pops);
        assert!(pops.is_empty());
    }

    #[test]
    fn a_fire_glares_as_wide_as_its_flames() {
        let fire = EffectEmitter { scale: 1.5, ..emitter(EffectKind::Fire) };
        let s = fire_glare(&fire, 30.0, Vec3::ZERO, Quat::IDENTITY).unwrap();
        let eye = Vec3::new(0.0, 1.6, 2.5);
        let lobes = super::super::glare::glare_lobes(&s, eye);
        assert_eq!(lobes.len(), 1, "no bulb: one lobe, its body");
        assert!((lobes[0].radius - std::f32::consts::SQRT_2 * 0.13 * 1.5).abs() < 1e-4, "{}", lobes[0].radius);
        assert!((lobes[0].centre - Vec3::new(0.0, 0.375, 0.0)).length() < 1e-4);

        let near = Vec3::new(0.0, 1.0, 1.2);
        let lobe = super::super::glare::glare_lobes(&s, near)[0];
        let q = super::super::glare::glare_quad(&s, &lobe, near, 20.0, 1.0).expect("a veil");
        assert!(q.peak > 0.02 && q.peak < 0.5, "{}", q.peak);
        assert!(super::super::glare::glare_quad(&s, &lobes[0], eye, 4.0, 1.0).is_none());
        assert!(fire_glare(&emitter(EffectKind::Smoke), 30.0, Vec3::ZERO, Quat::IDENTITY).is_none());
    }

    #[test]
    fn a_door_opening_onto_dust_works_out_only_what_its_doorway_shows_ramped_in() {
        use super::super::portal_cull::CullRoom;
        use super::super::uniforms::ProbePortal;
        let head = Vec3::new(0.0, 1.6, 2.5);
        let view = glam::Mat4::perspective_rh(1.6, 1.0, 0.05, 100.0) * glam::Mat4::look_at_rh(head, Vec3::new(0.0, 1.4, -3.0), Vec3::Y);
        let views = [super::super::shadow::frustum_planes(view)];
        let rooms = [
            CullRoom { id: 1, min: Vec3::new(-1.5, -0.5, -0.5), max: Vec3::new(1.5, 3.5, 5.0), closed: true },
            CullRoom { id: 2, min: Vec3::new(-8.0, -0.5, -14.0), max: Vec3::new(8.0, 6.0, -0.5), closed: true },
        ];
        let door = ProbePortal { min: Vec3::new(-0.6, 0.0, -0.7), max: Vec3::new(0.6, 2.1, -0.3), axis: 2, low: 2, high: 1, wall: None };
        let dust = [EffectEmitter {
            id: "hall_dust".into(),
            position: Vec3::new(0.0, 1.5, -5.0),
            extent: [Vec3::X * 7.5, Vec3::Y * 1.5, Vec3::Z * 4.4],
            ..emitter(EffectKind::Dust)
        }];
        assert!(emitter_sight(&dust[0], &views, head, &rooms, &[]).is_empty(), "the door shut: out of sight");
        let sight = vec![emitter_sight(&dust[0], &views, head, &rooms, &[door])];
        assert!(!sight[0].is_empty(), "the door open: seen through it");
        let at = seen_from(head);
        let t = 50.0;
        let run = |share: &dyn Fn(&EffectEmitter) -> f32, sight: &[Vec<seen::Eye>]| {
            let shares = Shares { emitter: share, splash: &|_| 1.0, sight };
            simulate_governed(&dust, &|_| true, &[], None, Some(&shares), t, &at)
        };
        let whole = run(&|_| 1.0, &[]);
        let through = run(&|_| 1.0, &sight);
        assert!(through.screened > 0 && (through.screened as f32) < 0.4 * whole.screened as f32, "{} of {} motes", through.screened, whole.screened);
        let same = |a: &EffectInstance, b: &EffectInstance| bytemuck::bytes_of(a) == bytemuck::bytes_of(b);
        for i in &through.instances {
            assert!(whole.instances.iter().any(|w| same(w, i)), "a mote through the doorway is the mote drawn whole");
        }

        let shown = whole.instances.iter().filter(|w| sight[0].iter().any(|e| e.holds(centre(w), 0.0))).count();
        let kept = through.instances.iter().filter(|w| sight[0].iter().any(|e| e.holds(centre(w), 0.0))).count();
        assert_eq!(kept, shown, "every mote in the doorway's view is drawn");

        let lib = stack::EffectLibrary::default();
        let mut set = stack::BurstSet::default();
        let narrowed = governor_wants(&dust, &|_| true, &[], &set.splash_measured, head, &sight);
        let open_room = governor_wants(&dust, &|_| true, &[], &set.splash_measured, head, &[vec![]]);
        assert!(narrowed[0].count < 0.4 * governor_wants(&dust, &|_| true, &[], &set.splash_measured, head, &[])[0].count, "weighed through the doorway");
        assert_eq!(open_room[0].fill, 0.0, "nothing in sight weighs nothing");
        let mut shares = Vec::new();
        for frame in 0..14 {
            let now = t + f64::from(frame) / 72.0;
            set.frame_with(&lib, now, &views, head, &narrowed);
            shares.push(set.share_of(emitter_key(&dust[0])));
        }
        assert!((shares[0] - 0.1).abs() < 1e-6, "the opening frame works out a tenth: {shares:?}");
        assert!(shares.windows(2).all(|w| w[1] >= w[0]) && shares[13] == 1.0, "{shares:?}");
        let first = run(&|_| shares[0], &sight);
        assert!(first.screened as f32 <= 0.2 * through.screened as f32 + 1.0, "{} of {}", first.screened, through.screened);
        for i in &first.instances {
            assert!(through.instances.iter().any(|w| same(w, i)), "the motes ramping in are the same motes");
        }
    }

    #[test]
    fn an_emitter_out_of_view_or_behind_a_shut_door_is_not_simulated() {
        use super::super::portal_cull::CullRoom;
        use super::super::uniforms::ProbePortal;

        let view = glam::Mat4::perspective_rh(1.6, 1.0, 0.05, 100.0) * glam::Mat4::look_at_rh(Vec3::new(0.0, 1.6, 2.0), Vec3::new(0.0, 1.2, 0.0), Vec3::Y);
        let views = [super::super::shadow::frustum_planes(view)];
        let head = Vec3::new(0.0, 1.6, 2.0);
        let fire = EffectEmitter { scale: 1.5, ..emitter(EffectKind::Fire) };
        assert!(emitter_seen(&fire, &views, head, &[], &[]), "in front");
        let behind = EffectEmitter { position: Vec3::new(0.0, 0.0, 6.0), ..fire.clone() };
        assert!(!emitter_seen(&behind, &views, head, &[], &[]), "behind the head");

        let rooms = [
            CullRoom { id: 1, min: Vec3::new(-3.0, -1.0, -0.5), max: Vec3::new(3.0, 4.0, 4.0), closed: true },
            CullRoom { id: 2, min: Vec3::new(-3.0, -1.0, -4.0), max: Vec3::new(3.0, 4.0, -0.5), closed: true },
        ];
        let far = EffectEmitter { position: Vec3::new(0.0, 0.0, -2.0), ..fire.clone() };
        let door = ProbePortal { min: Vec3::new(-0.5, 0.0, -0.7), max: Vec3::new(0.5, 2.0, -0.3), axis: 2, low: 2, high: 1, wall: None };
        assert!(emitter_seen(&far, &views, head, &rooms, &[door]), "through the open doorway");
        assert!(!emitter_seen(&far, &views, head, &rooms, &[]), "the doorway shut");
        assert!(emitter_seen(&fire, &views, head, &rooms, &[]), "in the head's own room");

        let smoke = EffectEmitter { id: "s".into(), position: Vec3::new(0.0, 0.5, 0.0), ..emitter(EffectKind::Smoke) };
        let lights = [fire_light(&fire, 10.0, Vec3::ZERO, Quat::IDENTITY).unwrap()];
        let dark = |_: Vec3| Vec3::ZERO;
        let at = Surroundings { lights: &lights, ambient: &dark, exposure: 9.0, ..seen_from(Vec3::new(0.0, 1.2, 1.8)) };
        let both = [fire.clone(), smoke];
        let all = simulate(&both, 10.0, &at);
        let smoke_only = simulate_seen(&both, &|i| i == 1, &[], 10.0, &at);
        let smoke_of = |f: &EffectFrame| f.instances.iter().filter(|i| i.params[3] == 1.0).map(|i| i.light[0]).collect::<Vec<_>>();
        assert!(smoke_only.instances.iter().all(|i| i.params[3] == 1.0) && !smoke_only.instances.is_empty());
        assert_eq!(smoke_of(&all), smoke_of(&smoke_only));
    }

    #[test]
    fn a_bed_breathes_with_the_fire_on_it() {
        let at = seen_from(Vec3::new(0.3, 1.6, 3.0));
        let fire = EffectEmitter { id: "f".into(), scale: 1.5, ..emitter(EffectKind::Fire) };
        let bed = EffectEmitter { id: "c".into(), scale: 1.5, ..emitter(EffectKind::Coals) };
        let glow = |e: &[EffectEmitter], t: f64| simulate(e, t, &at).instances.iter().find(|i| i.params[3] == 3.0).unwrap().colour[0];
        let with: Vec<f32> = (0..200).map(|k| glow(&[fire.clone(), bed.clone()], 50.0 + f64::from(k) * 0.05)).collect();
        let alone = glow(std::slice::from_ref(&bed), 50.0);
        let (lo, hi) = with.iter().fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
        assert!(hi > 1.03 * lo && hi < 1.25 * alone && lo > 0.8 * alone, "{lo} to {hi}, alone {alone}");
        assert!((0..50).map(|k| glow(std::slice::from_ref(&bed), 50.0 + f64::from(k) * 0.05)).all(|g| g == alone), "alone it does not");
    }

    #[test]
    fn the_rooms_ambient_follows_the_time_of_day() {
        let level = [[1.0f32; 3]; 9];
        let room = |volume: u32, lo: Vec3, hi: Vec3| ProbeDesc { centre: 0.5 * (lo + hi), min: lo, max: hi, volume, has_depth: true, room_light: Some(level) };
        let descs = [room(0, Vec3::splat(-50.0), Vec3::splat(50.0)), room(1, Vec3::ZERO, Vec3::splat(4.0))];
        let (outdoors, indoors) = (Vec3::new(20.0, 1.0, 20.0), Vec3::splat(2.0));
        let baked = room_ambient(&descs, outdoors);
        assert!(baked.x > 0.0);
        assert_eq!(room_ambient_now(&descs, outdoors, &[], None), baked);

        let relit = [0.01, 0.8];
        assert!((room_ambient_now(&descs, outdoors, &relit, None).x / baked.x - 0.01).abs() < 1e-4);
        let hall = room_ambient(&descs, indoors);
        assert!((room_ambient_now(&descs, indoors, &relit, None).x / hall.x - 0.8).abs() < 1e-4, "the smallest room that holds the point is its room");

        let far = Vec3::new(-200.0, 0.0, -200.0);
        assert!((room_ambient_now(&descs, far, &relit, None).x / room_ambient(&descs, far).x - 0.01).abs() < 1e-4);
        assert_eq!(room_ambient_now(&descs, outdoors, &[f32::NAN, 1.0], None), baked);
    }

    #[test]
    fn out_in_the_open_past_every_box_the_light_is_the_outdoors() {
        let photo = |volume: u32, lo: Vec3, hi: Vec3, depth: bool, v: f32| ProbeDesc { centre: 0.5 * (lo + hi), min: lo, max: hi, volume, has_depth: depth, room_light: Some([[v; 3]; 9]) };

        let descs = [photo(0, Vec3::new(0.0, -5.0, 0.0), Vec3::new(40.0, 20.0, 40.0), false, 2.0), photo(1, Vec3::new(-12.0, 0.0, -12.0), Vec3::new(-4.0, 4.0, -4.0), true, 0.05)];
        let pond = Vec3::new(-20.0, 0.0, -20.0);
        let open = 2.0 * 0.282_095;
        assert!((room_ambient_now(&descs, pond, &[], None).x - open).abs() < 1e-4, "{:?}", room_ambient_now(&descs, pond, &[], None));
        assert!((room_ambient_now(&descs, pond, &[0.5, 1.0], None).x - 0.5 * open).abs() < 1e-4);

        assert!((room_ambient_now(&descs, Vec3::new(-8.0, 2.0, -8.0), &[], None).x - 0.05 * 0.282_095).abs() < 1e-4);
        assert!((room_ambient_now(&descs, Vec3::new(20.0, 1.0, 20.0), &[], None).x - open).abs() < 1e-4);

        let hill = super::super::ground_map::HeightGrid::sample(glam::Vec2::splat(-60.0), glam::Vec2::splat(60.0), 121, 121, |x, _| if x < -25.0 { 14.0 } else { 0.0 });
        let in_cave = Vec3::new(-30.0, 0.5, -20.0);
        let dim = room_ambient(&descs, in_cave).x;
        assert!((room_ambient_now(&descs, in_cave, &[], Some(&hill)).x - dim).abs() < 1e-4, "the cave keeps its nearest room's light");
        assert!(dim < 0.2 * open);

        assert!((room_ambient_now(&descs, pond, &[], Some(&hill)).x - open).abs() < 1e-4);
        assert!((room_ambient_now(&descs, Vec3::new(-30.0, 14.3, -20.0), &[], Some(&hill)).x - open).abs() < 1e-4);
    }

    #[test]
    fn only_a_lamp_in_the_emitters_room_lights_it() {
        let room = |volume: u32, lo: Vec3, hi: Vec3| ProbeDesc {
            centre: 0.5 * (lo + hi),
            min: lo,
            max: hi,
            volume,
            has_depth: true,
            room_light: None,
        };
        let descs = [
            room(0, Vec3::splat(-50.0), Vec3::splat(50.0)),
            room(1, Vec3::ZERO, Vec3::splat(4.0)),
            room(2, Vec3::new(4.0, 0.0, 0.0), Vec3::new(8.0, 4.0, 4.0)),
        ];
        let fire = Vec3::new(2.0, 0.2, 2.0);
        assert!(lamp_in_room(&descs, fire, Vec3::new(1.0, 3.0, 1.0)));
        assert!(lamp_in_room(&descs, fire, Vec3::new(4.1, 3.0, 1.0)), "in the doorway");
        assert!(!lamp_in_room(&descs, fire, Vec3::new(6.0, 3.0, 1.0)), "the next room's");
        assert!(lamp_in_room(&[], fire, Vec3::new(60.0, 3.0, 1.0)), "no rooms: every lamp");
    }

    #[test]
    fn the_flame_stands_on_its_base_in_the_texture() {

        let f = &fire_frame(4);
        let n = ATLAS_SIZE as usize;
        let sum_in = |rows: std::ops::Range<usize>, channel: usize| -> f32 {
            rows.flat_map(|y| (0..n).map(move |x| f32::from(f[(y * n + x) * 4 + channel]))).sum()
        };
        assert!(sum_in(n / 2..n, 3) > 1.5 * sum_in(0..n / 2, 3), "more flame below the middle");
        assert!(sum_in(n / 2..n, 0) > 2.0 * sum_in(0..n / 2, 0), "hotter below the middle");
    }

    #[test]
    fn a_puff_lit_from_one_side_is_bright_on_that_side_and_glows_against_the_light() {
        let [front, back] = smoke_frame(3);
        let n = ATLAS_SIZE as usize;
        let texel = |img: &[u8], x: usize, y: usize, c: usize| f32::from(img[(y * n + x) * 4 + c]);
        assert!(texel(&front, n / 2, n / 2, 3) > 200.0, "dense in the middle: {}", texel(&front, n / 2, n / 2, 3));
        assert_eq!(texel(&front, 1, 1, 3), 0.0, "nothing in the corner");

        let row = n / 2;
        let right: f32 = (n * 5 / 8..n * 3 / 4).map(|x| texel(&front, x, row, 0)).sum();
        let left: f32 = (n / 4..n * 3 / 8).map(|x| texel(&front, x, row, 0)).sum();
        assert!(right > 1.5 * left, "lit side {right}, far side {left}");

        let edge = (0..n).map(|x| (texel(&back, x, row, 3), texel(&back, x, row, 2))).find(|&(a, _)| a > 40.0).unwrap();
        assert!(edge.1 > texel(&back, n / 2, row, 2) + 40.0, "edge {} middle {}", edge.1, texel(&back, n / 2, row, 2));

        assert!(texel(&front, n / 2, row, 2) > texel(&back, n / 2, row, 2) + 60.0);
    }

    #[test]
    fn a_cutout_holds_everything_visible_and_cuts_the_corners() {
        let layers = atlas_layers();
        let cuts = cutouts(&layers);
        let n = ATLAS_SIZE as usize;
        for (l, cut) in cuts.iter().enumerate() {
            let corners: Vec<Vec3> = (0..8).map(|k| Vec3::new(cut[2 * k], cut[2 * k + 1], 0.0)).collect();

            let inside = |p: Vec3| {
                (0..8).all(|k| {
                    let (a, b) = (corners[k], corners[(k + 1) % 8]);
                    let e = b - a;
                    e.length_squared() < 1e-12 || e.x * (p.y - a.y) - e.y * (p.x - a.x) >= -1e-5
                })
            };
            for y in 0..n {
                for x in 0..n {
                    if layers[l][(y * n + x) * 4 + 3] == 0 {
                        continue;
                    }
                    let p = Vec3::new((x as f32 + 0.5) / n as f32 * 2.0 - 1.0, 1.0 - (y as f32 + 0.5) / n as f32 * 2.0, 0.0);
                    assert!(inside(p), "layer {l} texel {x},{y} outside its cutout {cut:?}");
                }
            }

            let area: f32 = (0..8).map(|k| corners[k].x * corners[(k + 1) % 8].y - corners[(k + 1) % 8].x * corners[k].y).sum::<f32>() * 0.5;
            assert!(area <= 4.0 + 1e-4, "layer {l}: {area}");
            if l == EMBER_LAYER as usize || l == (FIRE_FIRST + 8) as usize {
                assert!(area < 3.5, "a dot's or a flame's corners are cut: layer {l} keeps {area} of 4");
            }
        }
    }

    #[test]
    fn the_atlas_is_the_same_bytes_every_time() {
        let a = atlas_layers();
        let b = atlas_layers();
        assert_eq!(a.len(), ATLAS_LAYERS as usize);
        assert!(a.iter().all(|l| l.len() == (ATLAS_SIZE * ATLAS_SIZE * 4) as usize));
        assert_eq!(a, b);
    }

    #[test]
    fn mips_keep_a_transparent_edge_from_darkening() {

        let src = vec![255, 255, 255, 255, 0, 0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 0];
        let m = half_mip(&src, 2);
        assert!(m[0] > 240 && m[3] > 120 && m[3] < 135, "{m:?}");
    }

    #[test]
    fn kinds_are_named_as_scenes_name_them() {
        assert_eq!(EffectKind::from_name("Fire"), Some(EffectKind::Fire));
        assert_eq!(EffectKind::from_name("sparks"), Some(EffectKind::Embers));
        assert_eq!(EffectKind::from_name("dust_motes"), Some(EffectKind::Dust));
        assert_eq!(EffectKind::from_name("coals"), Some(EffectKind::Coals));
        assert_eq!(EffectKind::from_name("rain"), None);
    }

    #[test]
    #[ignore]
    fn export_effects_for_the_editor() {
        let out = std::path::PathBuf::from(std::env::var("OUT").unwrap_or_else(|_| "/tmp".into()));
        let layers = atlas_layers();
        let n = ATLAS_SIZE as usize;
        let cols = 16usize;
        let rows = (layers.len()).div_ceil(cols);
        let (w, h) = (cols * n, rows * n);
        let mut sheet = vec![0u8; w * h * 4];
        for (k, l) in layers.iter().enumerate() {
            let (cx, cy) = (k % cols, k / cols);
            for y in 0..n {
                let o = ((cy * n + y) * w + cx * n) * 4;
                sheet[o..o + n * 4].copy_from_slice(&l[y * n * 4..(y + 1) * n * 4]);
            }
        }
        image::save_buffer(out.join("effects_atlas_layers.png"), &sheet, w as u32, h as u32, image::ExtendedColorType::Rgba8).expect("atlas");
        let cut: Vec<Vec<f32>> = cutouts(&layers).iter().map(|c| c.to_vec()).collect();
        std::fs::write(out.join("effects_cutouts.json"), serde_json::to_string(&cut).unwrap()).expect("cutouts");

        let point = Light { position: Vec3::new(0.5, 2.5, -0.5), ..lamp() };
        let sun = Light { kind: LightKind::Directional, direction: Vec3::new(0.3, -0.6, 0.7).normalize(), intensity: 2.0, ..lamp() };
        let lights = [point, sun];
        let ambient = |_: Vec3| Vec3::new(0.05, 0.06, 0.08);
        let emitters = vec![
            EffectEmitter { id: "fire".into(), scale: 1.5, ..emitter(EffectKind::Fire) },
            EffectEmitter { id: "coals".into(), scale: 1.5, ..emitter(EffectKind::Coals) },
            EffectEmitter { id: "smoke".into(), position: Vec3::new(0.0, 0.5, 0.0), ..emitter(EffectKind::Smoke) },
            EffectEmitter { id: "smoke_roofed".into(), position: Vec3::new(2.0, 0.5, 0.0), ceiling: Some(1.6), ..emitter(EffectKind::Smoke) },
            EffectEmitter { id: "embers".into(), scale: 1.5, ..emitter(EffectKind::Embers) },
            EffectEmitter { id: "dust".into(), position: Vec3::new(0.0, 1.5, 1.0), extent: [Vec3::X * 0.6, Vec3::Y * 0.5, Vec3::Z * 0.6], rate: 0.6, ..emitter(EffectKind::Dust) },

            EffectEmitter {
                id: "fire_varied".into(),
                position: Vec3::new(3.0, 0.0, 0.0),
                scale: 0.8,
                variation: Variation {
                    intensity: 1.3,
                    size_variation: 1.6,
                    life_variation: 0.6,
                    speed_variation: 2.0,
                    temperature_variation: 0.5,
                    flicker: 1.5,
                    flicker_rate: 1.3,
                    turbulence: 2.0,
                    seed: 4242,
                },
                ..emitter(EffectKind::Fire)
            },
            EffectEmitter {
                id: "coals_varied".into(),
                position: Vec3::new(3.0, 0.0, 0.0),
                scale: 0.8,
                variation: Variation { intensity: 0.7, flicker: 0.4, flicker_rate: 2.0, seed: 9, ..Variation::default() },
                ..emitter(EffectKind::Coals)
            },
            EffectEmitter {
                id: "smoke_varied".into(),
                position: Vec3::new(-2.0, 0.5, 0.0),
                variation: Variation { intensity: 0.6, size_variation: 2.0, life_variation: 1.5, speed_variation: 0.5, turbulence: 3.0, seed: 77, ..Variation::default() },
                ..emitter(EffectKind::Smoke)
            },
            EffectEmitter {
                id: "embers_varied".into(),
                position: Vec3::new(3.0, 0.0, 0.0),
                scale: 0.8,
                rate: 2.0,
                variation: Variation {
                    intensity: 1.2,
                    size_variation: 0.5,
                    life_variation: 0.5,
                    speed_variation: 2.0,
                    temperature_variation: 1.0,
                    flicker: 0.3,
                    flicker_rate: 2.5,
                    turbulence: 0.0,
                    seed: 5,
                },
                ..emitter(EffectKind::Embers)
            },
            EffectEmitter {
                id: "dust_varied".into(),
                position: Vec3::new(0.0, 1.5, -1.0),
                extent: [Vec3::X * 0.4, Vec3::Y * 0.4, Vec3::Z * 0.4],
                rate: 0.8,
                variation: Variation { intensity: 2.0, size_variation: 0.5, life_variation: 1.4, turbulence: 2.0, seed: 3, ..Variation::default() },
                ..emitter(EffectKind::Dust)
            },
        ];

        let splashes = [
            Splash { depth: 0.3, seed: 7, ..Splash::new(Vec3::new(-1.0, 0.0, 1.0), 9.8, Vec3::new(0.0, -2.5, 0.0), 0.08, 3.0) },
            Splash { depth: 2.0, seed: 3, ..Splash::new(Vec3::new(1.5, 0.0, 1.5), 9.85, Vec3::new(0.0, -4.4, 0.0), 0.015, 0.035) },
            Splash { depth: 2.0, seed: 5, ..Splash::projectile(Vec3::new(-2.5, 0.0, 0.5), 9.9, Vec3::new(247.0, -247.0, 0.0), 0.009, 0.008) },
            Splash { depth: 0.2, held: 1.0, seed: 9, ..Splash::new(Vec3::new(0.5, 0.0, 2.0), 9.85, Vec3::new(-2.3, -1.1, 0.4), 0.07, 3.0) },
            Splash { depth: 0.5, seed: 13, ..Splash::new(Vec3::new(-3.0, 0.0, -1.0), 9.7, Vec3::new(0.0, -6.3, 0.0), 0.4, 30.0) },
            Splash { depth: 1.0, calm: 0.15, sunlit: 0.4, seed: 21, ..Splash::new(Vec3::new(2.5, 0.0, -0.5), 9.9, Vec3::new(0.0, -4.4, 0.0), 0.07, 0.9) },
        ];
        let mut frames = Vec::new();
        for (time, yaw, offset) in
            [(10.0f64, 0.0f32, Vec3::ZERO), (10.35, 0.0, Vec3::ZERO), (10.2, 0.9, Vec3::new(1.0, 0.0, -2.0)), (23.7, 0.9, Vec3::new(1.0, 0.0, -2.0))]
        {
            let head = Vec3::new(0.3, 1.6, 3.0);
            let at = Surroundings { head, offset, yaw_inv: Quat::from_rotation_y(-yaw), lights: &lights, ambient: &ambient, exposure: 1.3, ..seen_from(head) };
            let f = simulate_with(&emitters, &splashes, time, &at);
            let fire_lights: Vec<serde_json::Value> = emitters
                .iter()
                .filter_map(|e| fire_light(e, time, offset, Quat::from_rotation_y(-yaw)))
                .map(|l| serde_json::json!({ "position": l.position.to_array(), "intensity": l.intensity }))
                .collect();
            frames.push(serde_json::json!({
                "time": time, "yaw": yaw, "offset": offset.to_array(), "head": head.to_array(),
                "over": f.over, "screened": f.screened, "fire_lights": fire_lights,
                "instances": f.instances.iter().map(|i| [i.centre, i.colour, i.light, i.light_dir, i.axis, i.params]).collect::<Vec<_>>(),
            }));
        }
        let golden = serde_json::json!({
            "lights": lights.iter().map(|l| serde_json::json!({
                "position": l.position.to_array(), "direction": l.direction.to_array(), "kind": format!("{:?}", l.kind),
                "color": [l.color.0, l.color.1, l.color.2, l.color.3], "intensity": l.intensity, "range": l.range,
                "cone_angle_deg": l.cone_angle_deg, "inner_cone_angle_deg": l.inner_cone_angle_deg, "source_radius": l.source_radius,
            })).collect::<Vec<_>>(),
            "ambient": [0.05, 0.06, 0.08], "exposure": 1.3,
            "emitters": emitters.iter().map(|e| serde_json::json!({
                "id": e.id, "kind": format!("{:?}", e.kind), "position": e.position.to_array(), "direction": e.direction.to_array(),
                "extent": e.extent.map(|v| v.to_array()), "scale": e.scale, "rate": e.rate, "tint": e.tint, "ceiling": e.ceiling,
                "variation": {
                    "intensity": e.variation.intensity, "size_variation": e.variation.size_variation,
                    "life_variation": e.variation.life_variation, "speed_variation": e.variation.speed_variation,
                    "temperature_variation": e.variation.temperature_variation, "flicker": e.variation.flicker,
                    "flicker_rate": e.variation.flicker_rate, "turbulence": e.variation.turbulence, "seed": e.variation.seed,
                },
            })).collect::<Vec<_>>(),
            "splashes": splashes.iter().map(|s| serde_json::json!({
                "position": s.position.to_array(), "born": s.born, "velocity": s.velocity.to_array(), "size": s.size, "mass": s.mass,
                "sunlit": s.sunlit, "depth": s.depth, "calm": s.calm, "held": s.held, "seed": s.seed.to_string(),
            })).collect::<Vec<_>>(),
            "frames": frames,
        });
        std::fs::write(out.join("effects_golden.json"), serde_json::to_string(&golden).unwrap()).expect("golden");
        eprintln!("wrote the atlas ({} layers), cutouts and {} golden frames to {}", layers.len(), golden["frames"].as_array().map_or(0, std::vec::Vec::len), out.display());
    }

    #[test]
    #[ignore]
    fn save_the_effects_atlas() {
        let out = std::path::PathBuf::from(std::env::var("OUT").unwrap_or_else(|_| "/tmp".into()));
        let layers = atlas_layers();
        let n = ATLAS_SIZE as usize;
        let fire_rows = (FIRE_FRAMES as usize).div_ceil(16);
        let (w, h) = (16 * n, (3 + fire_rows) * n);
        let mut sheet = vec![0u8; w * h * 4];
        let encode = |v: f32| (v.clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0).round() as u8;
        let mut put = |cell: (usize, usize), x: usize, y: usize, c: [f32; 3]| {
            let o = ((cell.1 * n + y) * w + cell.0 * n + x) * 4;
            sheet[o..o + 4].copy_from_slice(&[encode(c[0]), encode(c[1]), encode(c[2]), 255]);
        };
        for (row, l) in [(0usize, Vec3::new(0.6, 0.45, 0.65)), (1, Vec3::new(0.1, 0.25, -0.96))] {
            let l = l.normalize();
            let (lp, ln) = (l.max(Vec3::ZERO), (-l).max(Vec3::ZERO));
            for k in 0..SMOKE_FRAMES as usize {
                let (a, b) = (&layers[k], &layers[k + SMOKE_BACK as usize]);
                for y in 0..n {
                    for x in 0..n {
                        let o = (y * n + x) * 4;
                        let f = |img: &[u8], c: usize| f32::from(img[o + c]) / 255.0;
                        let toward = lp.x * lp.x * f(a, 0) + lp.y * lp.y * f(a, 1) + lp.z * lp.z * f(a, 2)
                            + ln.x * ln.x * f(b, 0) + ln.y * ln.y * f(b, 1) + ln.z * ln.z * f(b, 2);
                        let around = (f(a, 0) + f(a, 1) + f(a, 2) + f(b, 0) + f(b, 1) + f(b, 2)) / 6.0;
                        let lit = 0.08 * around + 0.9 * toward;
                        let alpha = f(a, 3);
                        let v = lit * alpha + 0.18 * (1.0 - alpha);
                        put((k, row), x, y, [v, v, v]);
                    }
                }
            }
        }
        for k in 0..FIRE_FRAMES as usize {
            let img = &layers[(FIRE_FIRST as usize) + k];
            for y in 0..n {
                for x in 0..n {
                    let o = (y * n + x) * 4;
                    let (heat, a) = (f32::from(img[o]) / 255.0, f32::from(img[o + 3]) / 255.0);
                    let c = super::super::tonemap::aces_fitted(flame_colour(heat) * FIRE_LEVEL) * a;

                    put((k % 16, 2 + k / 16), x, y, c.to_array());
                }
            }
        }
        for (cell, layer) in [(0usize, EMBER_LAYER), (1, MOTE_LAYER)] {
            let img = &layers[layer as usize];
            for y in 0..n {
                for x in 0..n {
                    let a = f32::from(img[(y * n + x) * 4 + 3]) / 255.0;
                    put((cell, 2 + fire_rows), x, y, [a, a, a]);
                }
            }
        }
        let path = out.join("effects_atlas.png");
        image::save_buffer(&path, &sheet, w as u32, h as u32, image::ExtendedColorType::Rgba8).expect("write the sheet");
        eprintln!("wrote {}", path.display());
    }

    #[test]
    #[ignore]
    fn render_effects_alone() {
        let out = std::path::PathBuf::from(std::env::var("OUT").unwrap_or_else(|_| "/tmp".into()));
        let instance = wgpu::Instance::default();
        let Ok(adapter) = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())) else {
            eprintln!("skipping: no GPU");
            return;
        };
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).unwrap();
        let (w, h) = (640u32, 640u32);
        let format = TextureFormat::Rgba8UnormSrgb;
        let camera_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: None,
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
                ty: BindingType::Buffer { ty: BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            }],
        });
        let probe_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: None,
            entries: &[BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Depth,
                    view_dimension: TextureViewDimension::D2Array,
                    multisampled: false,
                },
                count: None,
            }],
        });
        let effects_layout = bind_group_layout(&device);
        let pipeline = EffectsPipeline::new_multisampled(&device, format, &camera_layout, &probe_layout, &effects_layout, 1);
        let mut gpu = EffectsGpu::new(&device, &queue, &effects_layout);

        let v3 = |key: &str, or: Vec3| {
            std::env::var(key).ok().map_or(or, |s| {
                let v: Vec<f32> = s.split(',').map(|x| x.trim().parse().unwrap()).collect();
                Vec3::new(v[0], v[1], v[2])
            })
        };
        let (eye, at_point) = (v3("EYE", Vec3::new(0.0, 0.5, 1.6)), v3("AT", Vec3::new(0.0, 0.3, 0.0)));
        let view_proj = glam::Mat4::perspective_rh(50f32.to_radians(), w as f32 / h as f32, 0.03, 1000.0)
            * glam::Mat4::look_at_rh(eye, at_point, Vec3::Y);
        let camera = {
            use wgpu::util::DeviceExt;
            device.create_buffer_init(&util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&[view_proj.to_cols_array(), view_proj.to_cols_array()]),
                usage: BufferUsages::UNIFORM,
            })
        };
        let camera_bg = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &camera_layout,
            entries: &[BindGroupEntry { binding: 0, resource: camera.as_entire_binding() }],
        });
        let probe_depth = device.create_texture(&TextureDescriptor {
            label: None,
            size: Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Depth32Float,
            usage: TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let probe_bg = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &probe_layout,
            entries: &[BindGroupEntry {
                binding: 1,
                resource: BindingResource::TextureView(&probe_depth.create_view(&TextureViewDescriptor {
                    dimension: Some(TextureViewDimension::D2Array),
                    ..Default::default()
                })),
            }],
        });

        let time = std::env::var("EFFECTS_TIME").ok().and_then(|t| t.parse().ok()).unwrap_or(30.0);
        let emitters = [
            EffectEmitter { id: "fire".into(), scale: 1.5, ..emitter(EffectKind::Fire) },
            EffectEmitter { id: "smoke".into(), position: Vec3::new(0.0, 0.5, 0.0), ..emitter(EffectKind::Smoke) },
            EffectEmitter { id: "embers".into(), scale: 1.5, ..emitter(EffectKind::Embers) },
            EffectEmitter { id: "coals".into(), scale: 1.5, ..emitter(EffectKind::Coals) },
        ];
        let fire_lights: Vec<Light> = emitters.iter().filter_map(|e| fire_light(e, time, Vec3::ZERO, Quat::IDENTITY)).collect();
        let faint = |_: Vec3| Vec3::splat(0.02);
        // MUD=crate|rock|bullet|blast draws that mud splash alone, in
        // daylight, EFFECTS_TIME seconds after the strike.
        let mud = std::env::var("MUD").ok().map(|kind| match kind.as_str() {
            "rock" => Splash { seed: 5, ..Splash::mud(Vec3::ZERO, 0.0, Vec3::new(0.0, -6.3, 0.0), 0.05, 1.0, 1.0) },
            "bullet" => Splash { seed: 7, ..Splash::mud_projectile(Vec3::ZERO, 0.0, Vec3::new(280.0, -280.0, 0.0), 0.009, 0.008, 1.0) },
            "blast" => Splash { seed: 9, ..Splash::mud_blast(Vec3::ZERO, 0.0, 1.0, 1.0) },
            _ => Splash { seed: 3, ..Splash::mud(Vec3::ZERO, 0.0, Vec3::new(0.0, -6.3, 0.0), 0.25, 30.0, 1.0) },
        });
        let sun = [Light { kind: LightKind::Directional, direction: Vec3::new(0.4, -0.7, -0.6).normalize(), intensity: 3.0, ..lamp() }];
        let sky = |_: Vec3| Vec3::new(0.35, 0.4, 0.5);
        let at = match mud {
            Some(_) => Surroundings { head: eye, lights: &sun, ambient: &sky, exposure: 1.0, ..seen_from(eye) },
            None => Surroundings { head: eye, lights: &fire_lights, ambient: &faint, exposure: 4.0, ..seen_from(eye) },
        };
        let frame = match &mud {
            Some(s) => simulate_with(&[], std::slice::from_ref(s), time, &at),
            None => simulate(&emitters, time, &at),
        };
        eprintln!("{} over, {} screened", frame.over, frame.screened);
        gpu.upload(&device, &queue, &frame);
        let forward = (at_point - eye).normalize();
        let right = forward.cross(Vec3::Y).normalize();
        gpu.set_view(
            &queue,
            &EffectsUniform {
                right: right.extend(0.0).to_array(),
                up: right.cross(forward).extend(0.0).to_array(),
                head: eye.extend(1.0).to_array(),
                depth: [0.03, 1000.0, 0.0, 0.0],
            },
        );

        let target = |format, usage| {
            device.create_texture(&TextureDescriptor {
                label: None,
                size: Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
        };
        let colour = target(format, TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC);
        let depth = target(TextureFormat::Depth32Float, TextureUsages::RENDER_ATTACHMENT);
        let (colour_view, depth_view) = (colour.create_view(&Default::default()), depth.create_view(&Default::default()));
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &colour_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: Operations { load: LoadOp::Clear(if mud.is_some() { Color { r: 0.25, g: 0.3, b: 0.36, a: 1.0 } } else { Color { r: 0.015, g: 0.015, b: 0.018, a: 1.0 } }), store: StoreOp::Store },
                })],
                depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(Operations { load: LoadOp::Clear(1.0), store: StoreOp::Discard }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            gpu.draw(&mut pass, &pipeline, &camera_bg, &probe_bg);
        }
        let row = (w * 4).div_ceil(256) * 256;
        let readback = device.create_buffer(&BufferDescriptor {
            label: None,
            size: u64::from(row * h),
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            colour.as_image_copy(),
            TexelCopyBufferInfo {
                buffer: &readback,
                layout: TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(h) },
            },
            Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        queue.submit(Some(encoder.finish()));
        let slice = readback.slice(..);
        slice.map_async(MapMode::Read, |_| {});
        let _ = device.poll(PollType::Wait { submission_index: None, timeout: None });
        let data = slice.get_mapped_range().expect("map the frame");
        let mut rgba = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h {
            let start = (y * row) as usize;
            rgba.extend_from_slice(&data[start..start + (w * 4) as usize]);
        }
        let path = out.join("effects_alone.png");
        image::save_buffer(&path, &rgba, w, h, image::ExtendedColorType::Rgba8).expect("write the frame");
        eprintln!("wrote {}", path.display());
    }

    #[test]
    #[ignore]
    fn print_where_the_flames_stand() {
        let fire = EffectEmitter { id: "brick_fire".into(), scale: 1.5, ..emitter(EffectKind::Fire) };
        let (mut sum, mut w, mut worst) = (Vec3::ZERO, 0.0f32, 0.0f32);
        for k in 0..2000 {
            let (mut s1, mut w1) = (Vec3::ZERO, 0.0f32);
            for s in fire_sheets(&fire, 100.0 + f64::from(k) * 0.05) {
                let a = s.size * s.size * s.stretch * erosion_cover(s.stands) * rise_cover(s.risen);
                s1 += s.base * a;
                w1 += a;
            }
            if w1 > 0.0 {
                worst = worst.max((s1 / w1).length());
                sum += s1;
                w += w1;
            }
        }
        eprintln!("mean flame centre {:?}, worst moment {worst:.3} m off", sum / w);
    }

    #[test]
    #[ignore]
    fn print_the_flame_tables() {
        let (area, cover, rise) = flame_tables();
        let fmt = |v: &[f32]| v.iter().map(|x| format!("{x:.4}")).collect::<Vec<_>>().join(", ");
        eprintln!("const FLAME_AREA: [f32; FIRE_FRAMES as usize] = [{}];", fmt(&area));
        eprintln!("const EROSION_COVER: [f32; 9] = [{}];", fmt(&cover));
        eprintln!("const RISE_COVER: [f32; 9] = [{}];", fmt(&rise));
        let fire = emitter(EffectKind::Fire);
        let lerp_at = |t: &[f32], x: f32| {
            let i = (x.floor() as usize).min(t.len() - 2);
            lerp(t[i], t[i + 1], x - i as f32)
        };
        let mut sum = 0.0f64;
        let samples = 20000;
        for k in 0..samples {
            let time = 100.0 + f64::from(k) * 0.0173;
            sum += fire_sheets(&fire, time)
                .iter()
                .map(|s| f64::from(s.size * s.size * s.stretch * lerp_at(&area, s.frame) * lerp_at(&cover, s.stands * 8.0) * lerp_at(&rise, s.risen * 8.0)))
                .sum::<f64>();
        }
        eprintln!("const FLAME_AREA_MEAN: f32 = {:.6};", sum / f64::from(samples));
        let g: Vec<u8> = (0..FIRE_FRAMES).flat_map(|k| fire_frame(k).chunks(4).filter(|t| t[3] > 128).map(|t| t[1]).collect::<Vec<_>>()).collect();
        let q = |f: f32| g.iter().filter(|&&x| f32::from(x) >= f * 255.0).count() as f32 / g.len() as f32;
        eprintln!("depth over covered texels: >=0.25 {:.3}, >=0.5 {:.3}, >=0.75 {:.3}, >=0.9 {:.3}", q(0.25), q(0.5), q(0.75), q(0.9));
    }

    #[test]
    #[ignore]
    fn print_a_fires_particles() {
        let id = std::env::var("FIRE_ID").unwrap_or_else(|_| "test".into());
        let time = std::env::var("EFFECTS_TIME").ok().and_then(|t| t.parse().ok()).unwrap_or(30.0);
        let fire = EffectEmitter { id, scale: 1.5, ..emitter(EffectKind::Fire) };
        let at = Surroundings { exposure: 12.0, ..seen_from(Vec3::new(0.0, 1.2, 2.2)) };
        let f = simulate(&[fire], time, &at);
        for i in &f.instances {
            eprintln!(
                "centre {:6.3} {:6.3} {:6.3} half {:5.3} colour {:5.2} {:5.2} {:5.2} a {:4.2} rot {:5.2} frame {:5.2} mode {}",
                i.centre[0], i.centre[1], i.centre[2], i.centre[3], i.colour[0], i.colour[1], i.colour[2], i.colour[3], i.params[0], i.params[1], i.params[3]
            );
        }
        let cuts = cutouts(&atlas_layers());
        eprintln!("fire layer 40 cutout {:?}", cuts[40]);
    }

    #[test]
    #[ignore]
    fn time_a_frames_effects() {
        let lamps: Vec<Light> = (0..6).map(|k| Light { position: Vec3::new(k as f32 - 2.5, 2.8, -1.0), ..lamp() }).collect();
        let emitters = [
            EffectEmitter { id: "fire".into(), scale: 1.5, ..emitter(EffectKind::Fire) },
            EffectEmitter { id: "coals".into(), scale: 1.5, ..emitter(EffectKind::Coals) },
            EffectEmitter { id: "smoke".into(), position: Vec3::new(0.0, 0.5, 0.0), ..emitter(EffectKind::Smoke) },
            EffectEmitter { id: "embers".into(), scale: 1.5, ..emitter(EffectKind::Embers) },
            EffectEmitter { id: "dust".into(), extent: [Vec3::X * 1.2, Vec3::Y * 1.45, Vec3::Z * 1.2], rate: 0.6, ..emitter(EffectKind::Dust) },
        ];
        let at = Surroundings { lights: &lamps, ..seen_from(Vec3::new(0.3, 1.6, 2.0)) };
        let time = |emitters: &[EffectEmitter]| {
            let start = std::time::Instant::now();
            let mut count = 0;
            for k in 0..200 {
                count += simulate(emitters, 30.0 + f64::from(k) / 72.0, &at).instances.len();
            }
            (count / 200, start.elapsed().as_secs_f64() / 200.0 * 1e3)
        };
        let (count, each) = time(&emitters);
        eprintln!("{count} particles a frame, {each:.3} ms a frame");
        for e in &emitters {
            let (count, each) = time(std::slice::from_ref(e));
            eprintln!("  {:>6}: {count} particles, {each:.3} ms", e.id);
        }

        let start = std::time::Instant::now();
        for k in 0..2000 {
            let t = 30.0 + f64::from(k) / 72.0;
            std::hint::black_box(fire_light(&emitters[0], t, Vec3::ZERO, Quat::IDENTITY));
            std::hint::black_box(fire_glare(&emitters[0], t, Vec3::ZERO, Quat::IDENTITY));
        }
        eprintln!("  fire_light + fire_glare: {:.4} ms", start.elapsed().as_secs_f64() / 2000.0 * 1e3);
    }

    #[test]
    fn the_effects_motion_twin_validates_and_is_the_effects_shader() {
        let twin = motion_shader();
        let error = crate::renderer::multiview::multiview_validation_error_of(&twin, false);
        assert!(error.is_none(), "{error:?}");
        for part in ["fn fx_vertex(vi: u32, inst: IIn) -> VOut", "fn fx_over(in: VOut) -> vec4<f32>", "fn fx_flame_along", "along = u * c - r * s;"] {
            assert!(twin.contains(part), "{part}");
        }
        assert!(!effects_shader().contains("fx_motion_view"), "the draw shader carries none of the twin");
    }

    #[test]
    fn every_particle_carries_its_velocity() {

        let check = |label: &str, at: f64, h: f32, frame: &dyn Fn(f64) -> EffectFrame, zeroed: bool| {
            let (a, b) = (frame(at), frame(at + f64::from(h)));
            let mut moving = 0;
            let mut found = 0;
            for i in &a.instances {
                let v = Vec3::from_slice(&i.motion[..3]);
                if v.length() < 0.005 {
                    continue;
                }
                moving += 1;
                let v = if zeroed { Vec3::ZERO } else { v };
                let next = Vec3::from_slice(&i.centre[..3]) + v * h;
                let nearest = b.instances.iter().map(|j| Vec3::from_slice(&j.centre[..3]).distance(next)).fold(f32::INFINITY, f32::min);
                let speed = Vec3::from_slice(&i.motion[..3]).length();
                if nearest < 0.1 * speed * h + 2e-5 {
                    found += 1;
                }
            }
            assert!(moving >= 5, "{label}: {moving} moving");
            (found, moving)
        };
        let head = Vec3::new(0.4, 1.2, 1.5);
        let at = seen_from(head);
        let mut dust = emitter(EffectKind::Dust);
        dust.position = Vec3::new(0.0, 1.2, 0.0);
        dust.extent = [Vec3::X * 1.5, Vec3::Y * 0.8, Vec3::Z * 1.5];
        dust.rate = 3.0;
        let mut embers = emitter(EffectKind::Embers);
        embers.rate = 4.0;
        let smoke = emitter(EffectKind::Smoke);
        let cases: Vec<(&str, f64, f32, Box<dyn Fn(f64) -> EffectFrame>)> = vec![
            ("embers", 7.3, 1e-3, Box::new(|t| simulate(std::slice::from_ref(&embers), t, &at))),
            ("dust", 20.0, 0.01, Box::new(|t| simulate(std::slice::from_ref(&dust), t, &at))),
            ("smoke", 9.0, 1e-3, Box::new(|t| simulate(std::slice::from_ref(&smoke), t, &at))),
            ("splash", 0.1, 1e-3, Box::new(|t| simulate_with(&[], &[pistol()], t, &at))),
            ("splash's jet", 0.33, 1e-3, Box::new(|t| simulate_with(&[], &[pistol()], t, &at))),
            ("bullet's splash", 0.2, 1e-3, Box::new(|t| simulate_with(&[], &[bullet()], t, &at))),
        ];
        for (label, t, h, frame) in &cases {
            let (found, moving) = check(label, *t, *h, frame.as_ref(), false);
            assert!(found as f32 >= 0.95 * moving as f32, "{label}: {found} of {moving} found where their velocity took them");
            let (found, moving) = check(label, *t, *h, frame.as_ref(), true);
            assert!((found as f32) < 0.5 * moving as f32, "{label}: still found with no velocity ({found} of {moving}): the test cannot fail");
        }
    }

    #[test]
    fn a_flame_sheet_stands_still_and_its_gas_rises_at_its_book_pace() {
        let fire = emitter(EffectKind::Fire);
        let at = seen_from(Vec3::new(0.0, 0.6, 1.5));
        let h = 1e-3;
        let (a, b) = (simulate(std::slice::from_ref(&fire), 12.0, &at), simulate(std::slice::from_ref(&fire), 12.0 + h, &at));
        let flames: Vec<&EffectInstance> = a.instances.iter().filter(|i| i.params[3] == 2.0).collect();
        assert!(flames.len() >= 3, "{} flames", flames.len());
        let mut checked = 0;
        for f in flames {
            assert_eq!(&f.motion[..3], &[0.0; 3], "a sheet stands on its spot");
            let Some(g) = b.instances.iter().find(|g| g.params[3] == 2.0 && g.centre[0] == f.centre[0] && g.centre[2] == f.centre[2]) else { continue };
            if f.motion[3] == 0.0 {
                continue;
            }
            let frames_a_second = (g.params[1] - f.params[1]) / h as f32;
            let book = frames_a_second * FLAME_SECONDS / FIRE_FRAMES as f32;
            assert!((book - f.motion[3]).abs() < 0.02 * f.motion[3] + 1e-3, "book pace {book} against {}", f.motion[3]);
            checked += 1;
        }
        assert!(checked >= 3, "{checked} sheets checked");
    }

    #[test]
    fn the_shader_validates() {
        let src = effects_shader();
        let error = crate::renderer::multiview::multiview_validation_error_of(&src, false);
        assert!(error.is_none(), "{error:?}");
    }
}
