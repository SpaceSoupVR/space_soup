use bytemuck::{Pod, Zeroable};
use glam::{Quat, Vec3};
use wgpu::*;

use super::lights::{Light, LightKind};
use super::probe_stream::ProbeDesc;

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
pub const ATLAS_LAYERS: u32 = CROWN_LAYER + 1;
pub const ATLAS_SIZE: u32 = 128;
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
    match kind {
        EffectKind::Fire => Preset {

            rate: 9.0,
            life: (0.9, 1.3),
            speed: (0.0, 0.0),
            spread_deg: 0.0,
            bed: 0.15,
            size: (0.145, 0.13),
            drag: 0.0,
            accel: Vec3::ZERO,
            swirl: (0.0, 0.0),
            spin: (0.0, 0.07),
            streak: 0.0,
            soft: 0.05,
            frames: (FIRE_FIRST, FIRE_FRAMES),
            phase_g: 0.0,
            albedo: 0.0,
        },
        EffectKind::Smoke => Preset {
            rate: 6.0,
            life: (4.0, 6.0),
            speed: (0.15, 0.28),
            spread_deg: 10.0,
            bed: 0.06,
            size: (0.1, 0.55),
            drag: 0.6,
            accel: Vec3::new(0.0, 0.3, 0.0),
            swirl: (0.14, 0.7),
            spin: (0.2, std::f32::consts::PI),
            streak: 0.0,
            soft: 0.3,
            frames: (0, SMOKE_FRAMES),
            phase_g: 0.35,
            albedo: 0.3,
        },
        EffectKind::Embers => Preset {
            rate: 5.0,
            life: (1.2, 2.4),
            speed: (0.9, 1.8),
            spread_deg: 28.0,
            bed: 0.06,
            size: (0.005, 0.003),
            drag: 1.3,
            accel: Vec3::new(0.0, -1.1, 0.0),
            swirl: (0.06, 3.0),
            spin: (0.0, 0.0),
            streak: 0.035,
            soft: 0.02,
            frames: (EMBER_LAYER, 1),
            phase_g: 0.0,
            albedo: 0.0,
        },

        EffectKind::Coals => Preset {
            rate: 0.0,
            life: (1.0, 1.0),
            speed: (0.0, 0.0),
            spread_deg: 0.0,
            bed: 0.0,
            size: (COALS_HALF, COALS_HALF),
            drag: 0.0,
            accel: Vec3::ZERO,
            swirl: (0.0, 0.0),
            spin: (0.0, 0.0),
            streak: 0.0,
            soft: 0.0,
            frames: (COALS_LAYER, 1),
            phase_g: 0.0,
            albedo: 0.0,
        },

        EffectKind::Dust => Preset {
            rate: 90.0,
            life: (3.5, 6.5),
            speed: (0.0, 0.004),
            spread_deg: 180.0,
            bed: 0.0,
            size: DUST_LIGHT_SIZE,
            drag: 0.4,
            accel: Vec3::new(0.0, -0.0012, 0.0),
            swirl: (0.006, 0.8),
            spin: (0.0, 0.0),
            streak: 0.0,
            soft: 0.02,
            frames: (MOTE_LAYER, 1),
            phase_g: 0.7,
            albedo: 0.8,
        },
    }
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
}

pub struct Surroundings<'a> {

    pub head: Vec3,

    pub offset: Vec3,
    pub yaw_inv: Quat,

    pub lights: &'a [Light],

    pub reaches: &'a dyn Fn(&EffectEmitter, usize) -> bool,

    pub ambient: &'a dyn Fn(Vec3) -> Vec3,
    pub exposure: f32,
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
    if matches!(l.kind, LightKind::Directional) || l.intensity <= 0.0 {
        return None;
    }
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
    let cone = if matches!(l.kind, LightKind::Spot) {
        let (cos_outer, cos_inner) = l.cone_cosines();
        smoothstep(cos_outer, cos_inner, (-dir).dot(l.direction.normalize_or_zero()))
    } else {
        1.0
    };
    if cone <= 0.0 {
        return None;
    }
    let c = l.color.to_linear();
    Some((Vec3::new(c[0], c[1], c[2]) * (l.intensity * atten * cone), dir))
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

fn air_drift(waves: &[AirWave; 6], p: Vec3, time: f64) -> Vec3 {
    waves.iter().fold(Vec3::ZERO, |sum, a| {
        let clock = (f64::from(a.w) * time).rem_euclid(std::f64::consts::TAU) as f32;
        sum + a.across * (a.reach * (a.k.dot(p) + clock + a.phase).sin())
    })
}

#[cfg(test)]
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
        match e.kind {
            EffectKind::Coals => {

                let breath = coals_breath(e, emitters, time);
                over.extend(coals_instance(e, time, at, &own, breath).map(|i| (f32::INFINITY, i)));
                continue;
            }
            EffectKind::Fire => {
                fire_instances(e, time, at, &mut over);
                continue;
            }
            EffectKind::Embers => ember_pops(e, time, at, &mut screened),
            _ => {}
        }
        let p = preset(e.kind);
        let v = e.variation.clamped();
        let rate = e.rate.max(0.0)
            * match e.kind {
                EffectKind::Dust => p.rate * e.volume() / f32::midpoint(p.life.0, p.life.1),
                _ => p.rate,
            };
        if !(rate > 0.0) || !(e.scale > 0.0) {
            continue;
        }
        let lives = life_range(&p, &v);
        let speeds = speed_range(&p, &v);
        let slots = ((rate * lives.1).ceil() as usize).clamp(1, MAX_SLOTS);

        let period = (slots as f64 / f64::from(rate)).max(f64::from(lives.1));
        let base = emitter_seed(e);
        let tint = Vec3::from(e.tint);
        let ambient = (at.ambient)(e.position);
        let lamps: Vec<usize> = (0..at.lights.len()).filter(|&i| (at.reaches)(e, i)).collect();

        let glow = if e.kind == EffectKind::Embers { at.exposure * fire_adaptation(e, at) } else { at.exposure };
        let up = e.direction.try_normalize().unwrap_or(Vec3::Y);
        let air = (e.kind == EffectKind::Dust).then(|| air_waves(base));
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
            let p0 = if e.kind == EffectKind::Dust {
                let p0 = e.position + e.extent[0] * (2.0 * u[4] - 1.0) + e.extent[1] * (2.0 * u[5] - 1.0) + e.extent[2] * (2.0 * u[6] - 1.0);
                if p0.distance_squared(head_world) > DUST_SEEN_WITHIN * DUST_SEEN_WITHIN {
                    continue;
                }
                p0
            } else {

                let r = p.bed * e.scale * u[4].sqrt();
                let side = up.any_orthonormal_vector();
                let phi = u[5] * std::f32::consts::TAU;
                e.position + (side * phi.cos() + up.cross(side) * phi.sin()) * r
            };
            let dir = cone_direction(up, p.spread_deg, u[2], u[3]);
            let (mut pos, vel) = ballistic(p0, dir * speed, p.accel, p.drag, age);

            let mut moving = vel;
            if let Some(air) = &air {

                pos += (air_drift(air, p0, time) - air_drift(air, p0, time - f64::from(age))) * v.turbulence;
                let seed = base ^ (i as u64).wrapping_mul(0x632be59bd9b4e019) ^ (n as i64 as u64);
                let wander = |jitter: f32| Vec3::new(noise1(jitter, seed), 0.6 * noise1(jitter, seed ^ 1), noise1(jitter, seed ^ 2)) * (p.swirl.0 * v.turbulence);
                let jitter = age * p.swirl.1;
                pos += wander(jitter);
                moving += (air_drift(air, p0, time) - air_drift(air, p0, time - f64::from(MOTION_H))) * (v.turbulence / MOTION_H)
                    + (wander(jitter) - wander((age - MOTION_H) * p.swirl.1)) / MOTION_H;
            } else {
                let swirl_r = p.swirl.0 * v.turbulence * t * e.scale;
                let swirl_a = p.swirl.1 * age + u[7] * std::f32::consts::TAU;
                pos += Vec3::new(swirl_a.cos(), 0.0, swirl_a.sin()) * swirl_r;
                moving += swirl_motion(p.swirl.0 * v.turbulence * e.scale / life, swirl_r, p.swirl.1, swirl_a);
            }
            let mut size = if e.kind == EffectKind::Dust {

                let k = if v.size_variation == 1.0 { u[11] * u[11] } else { 1.0 / 3.0 + (u[11] * u[11] - 1.0 / 3.0) * v.size_variation };
                p.size.0 * (p.size.1 / p.size.0).powf(k) * e.scale
            } else {
                lerp(p.size.0, p.size.1, smoothstep(0.0, 1.0, t)) * e.scale * size_spread(u[8], &v)
            };

            if let Some(top) = e.ceiling {
                match e.kind {
                    EffectKind::Smoke => {
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
                    EffectKind::Embers
                        if pos.y >= top - 0.02 => {
                            pos.y = top - 0.02;
                            moving.y = moving.y.min(0.0);
                        }
                    _ => {}
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
            match e.kind {
                EffectKind::Embers => {
                    inst.colour = ember_colour(&v, tint, glow, age, t, u[11]);
                }
                EffectKind::Dust => {

                    let mut scattered = ambient * DUST_AMBIENT_SEEN;
                    for &li in &lamps {
                        if let Some((c, d)) = lamp_at(&at.lights[li], centre) {
                            scattered += c * (own(&at.lights[li]) * henyey_greenstein(p.phase_g, -d, to_eye));
                        }
                    }

                    let glint = mote_glint(age, lerp(0.35, 1.3, u[7]), id_hash("glint") ^ u64::from(u[8].to_bits()));
                    let c = scattered * p.albedo * tint * (at.exposure * glint * v.intensity);
                    let held = smoothstep(0.0, 0.2, t) * smoothstep(1.0, 0.7, t);
                    let d = (centre - at.head).length();
                    let focus = smoothstep(DUST_NEAREST.0, DUST_NEAREST.1, d) * smoothstep(DUST_SEEN_WITHIN, 0.75 * DUST_SEEN_WITHIN, d);
                    inst.colour = [c.x, c.y, c.z, held * focus];
                    inst.params[3] = -1.0;
                }
                EffectKind::Smoke => {
                    let mut strongest: Option<(f32, Vec3, Vec3)> = None;
                    let mut rest = ambient;
                    for &li in &lamps {
                        let Some((c, d)) = lamp_at(&at.lights[li], centre) else { continue };
                        let c = c * own(&at.lights[li]);
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
                    let opacity = 0.55 * smoothstep(0.0, 0.1, t) * (1.0 - t).powf(1.4) * v.intensity;
                    inst.colour = [base.x, base.y, base.z, opacity];
                    if let Some((_, c, d)) = strongest {
                        let lit = c * a * henyey_greenstein(p.phase_g, -d, to_eye);
                        inst.light = [lit.x, lit.y, lit.z, 0.0];
                        inst.light_dir = [d.x, d.y, d.z, streak];
                    }
                    inst.params[3] = 1.0;
                }

                EffectKind::Coals | EffectKind::Fire => unreachable!("coals and flames are drawn apart"),
            }
            if inst.colour[3] <= 0.0 {
                continue;
            }
            if e.kind.screened() {
                screened.push(inst);
            } else {
                over.push(((centre - at.head).length_squared(), inst));
            }
        }
    }
    for s in splashes {
        splash(s, time, at, &mut over, &mut screened);
    }

    over.sort_by(|a, b| b.0.total_cmp(&a.0));
    let mut frame = EffectFrame {
        instances: Vec::with_capacity(over.len() + screened.len()),
        over: over.len() as u32,
        screened: screened.len() as u32,
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
    let (lo, hi) = emitter_bounds(e);
    let in_view = views.iter().any(|planes| {
        let mut unbounded = *planes;
        unbounded[5] = glam::Vec4::new(0.0, 0.0, 0.0, 1.0);
        super::shadow::aabb_in_frustum(&unbounded, lo, hi)
    });
    if !in_view {
        return false;
    }
    let room_of = |p: Vec3| {
        rooms
            .iter()
            .filter(|r| (r.min.cmplt(p) & p.cmplt(r.max)).all())
            .min_by(|a, b| (a.max - a.min).element_product().total_cmp(&(b.max - b.min).element_product()))
    };
    let Some(start) = room_of(head).filter(|r| r.closed) else { return true };
    let goal = room_of(e.position);
    if goal.is_some_and(|g| g.id == start.id) {
        return true;
    }

    let mut reached = vec![start.id];
    let mut frontier = vec![start.id];
    for _ in 0..super::portal_cull::MAX_DOORWAYS {
        let mut next = Vec::new();
        for p in portals {
            for (from, to) in [(p.low, p.high), (p.high, p.low)] {
                if frontier.contains(&from) && !reached.contains(&to) {
                    reached.push(to);
                    next.push(to);
                }
            }
        }
        match goal {
            Some(g) if reached.contains(&g.id) => return true,

            None if reached.iter().any(|id| rooms.iter().any(|r| r.id == *id && !r.closed)) => return true,
            _ => {}
        }
        if next.is_empty() {
            return false;
        }
        frontier = next;
    }
    false
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

    pub speed: f32,

    pub size: f32,

    pub sunlit: f32,

    pub depth: f32,
    pub seed: u64,
}

pub const SPLASH_SECONDS: f32 = 1.6;

const DROP_ALBEDO: f32 = 1.0;
const DROP_PHASE_G: f32 = 0.55;

const DROP_STREAK: f32 = 0.012;

pub fn splash_strength(speed: f32, size: f32) -> f32 {
    let v = (speed / 1.3).max(0.0);
    (v * v * (size / 0.08).max(0.0)).min(12.0)
}

fn splash(s: &Splash, time: f64, at: &Surroundings, over: &mut Vec<(f32, EffectInstance)>, screened: &mut Vec<EffectInstance>) {
    let age = (time - s.born) as f32;
    let strength = splash_strength(s.speed, s.size);
    if !(0.0..SPLASH_SECONDS).contains(&age) || strength <= 0.01 {
        return;
    }
    let player = |p: Vec3| at.yaw_inv * (p - at.offset);
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
    let gravity = Vec3::new(0.0, -9.81, 0.0);
    let drops = ((40.0 * strength.sqrt()) as usize).clamp(6, 200);
    for i in 0..drops {
        let u = uniforms(hash64(s.seed ^ (i as u64).wrapping_mul(0x632be59bd9b4e019)));
        let azimuth = u[0] * std::f32::consts::TAU;
        let out = Vec3::new(azimuth.cos(), 0.0, azimuth.sin());
        let elevation = lerp(40.0f32, 85.0, u[1]).to_radians();

        let launch = (s.speed * lerp(0.4, 1.6, u[2] * u[2])).clamp(0.3, 8.0);
        let v0 = (out * elevation.cos() + Vec3::Y * elevation.sin()) * launch;
        let p0 = s.position + out * (s.size * lerp(0.5, 1.1, u[3]));
        let born_late = 0.06 * u[4];
        let a = age - born_late;
        if a <= 0.0 {
            continue;
        }
        let (pos, vel) = ballistic(p0, v0, gravity, 0.4, a);
        if pos.y < s.position.y && vel.y < 0.0 {
            continue;
        }
        let centre = player(pos);
        let axis = at.yaw_inv * vel.normalize_or(Vec3::Y);
        let moving = at.yaw_inv * vel;
        let c = lit(centre, DROP_ALBEDO, DROP_PHASE_G);
        let radius = lerp(0.0015, 0.004, u[5]);
        screened.push(EffectInstance {
            centre: [centre.x, centre.y, centre.z, radius],
            colour: [c.x, c.y, c.z, 1.0 - smoothstep(0.75, 1.0, a / SPLASH_SECONDS)],
            light: [0.0; 4],
            light_dir: [0.0, 1.0, 0.0, vel.length() * DROP_STREAK],
            axis: [axis.x, axis.y, axis.z, 0.01],
            params: [0.0, EMBER_LAYER as f32, 0.0, 0.0],
            motion: [moving.x, moving.y, moving.z, 0.0],
        });
    }

    let rise = ((1.2 * s.speed).powi(2) / (2.0 * 9.81)).clamp(0.04, 2.0);
    let crown_life = (2.0 * (2.0 * rise / 9.81).sqrt()).clamp(0.3, 1.4);
    let t = age / crown_life;
    if t < 1.0 {
        let u = uniforms(hash64(s.seed ^ 0xc80));
        let half_at = |t: f32| 0.5 * (rise * smoothstep(0.0, 0.3, t) * (1.0 - 0.45 * smoothstep(0.45, 1.0, t))).max(s.size);
        let half = half_at(t);
        let centre = player(s.position + Vec3::Y * (half * 0.92));

        let lift = 0.92 * (half - half_at(t - MOTION_H / crown_life)) / MOTION_H;
        let c = lit(centre, 0.95, 0.0);
        let opacity = 0.85 * smoothstep(0.0, 0.04, t) * (1.0 - smoothstep(0.5, 1.0, t)) * strength.sqrt().clamp(0.35, 1.0);
        over.push((
            (centre - at.head).length_squared(),
            EffectInstance {
                centre: [centre.x, centre.y, centre.z, half],
                colour: [c.x, c.y, c.z, opacity],
                light: [0.0, 0.0, 0.0, 1.0],
                light_dir: [0.0, 1.0, 0.0, 0.0],
                axis: [0.0, 1.0, 0.0, 0.03],
                params: [(2.0 * u[0] - 1.0) * 0.15, CROWN_LAYER as f32, if u[1] < 0.5 { -1.0 } else { 1.0 }, 0.0],
                motion: [0.0, lift, 0.0, 0.0],
            },
        ));
    }

    let puffs = ((2.0 + 2.0 * strength.sqrt()) as usize).min(10);
    for i in 0..puffs {
        let u = uniforms(hash64(s.seed ^ 0x5bd1e995 ^ (i as u64).wrapping_mul(0x8cb92ba72f3d8dd7)));
        let life = lerp(0.45, 0.9, u[0]) * strength.sqrt().clamp(0.7, 1.6);
        if age >= life {
            continue;
        }
        let t = age / life;
        let azimuth = u[1] * std::f32::consts::TAU;
        let out = Vec3::new(azimuth.cos(), 0.0, azimuth.sin());
        let v0 = out * lerp(0.2, 0.7, u[2]) * s.speed.sqrt() + Vec3::Y * lerp(0.3, 1.0, u[3]) * s.speed.sqrt();
        let (pos, vel) = ballistic(s.position + out * s.size * 0.6, v0, Vec3::new(0.0, -2.0, 0.0), 3.0, age);
        let floor = s.position.y + 0.02;
        let centre = player(pos.max(Vec3::new(f32::MIN, floor, f32::MIN)));
        let moving = at.yaw_inv * if pos.y < floor { vel * Vec3::new(1.0, 0.0, 1.0) } else { vel };
        let size = s.size * lerp(0.7, 1.2, u[4]) * (0.6 + 1.6 * t) * strength.sqrt().clamp(0.6, 1.8);

        let c = lit(centre, 0.95, 0.0);
        let opacity = 0.3 * smoothstep(0.0, 0.06, t) * (1.0 - t).powf(1.4) * strength.sqrt().min(1.3).max(0.4);
        let frame = SMOKE_FRAMES as f32 * (0.2 + 0.6 * t);
        over.push((
            (centre - at.head).length_squared(),
            EffectInstance {
                centre: [centre.x, centre.y, centre.z, size],
                colour: [c.x, c.y, c.z, opacity],
                light: [0.0; 4],
                light_dir: [0.0, 1.0, 0.0, 0.0],
                axis: [0.0, 1.0, 0.0, 0.05],
                params: [(2.0 * u[5] - 1.0) * std::f32::consts::PI, frame.min((SMOKE_FRAMES - 1) as f32), 0.0, 0.0],
                motion: [moving.x, moving.y, moving.z, 0.0],
            },
        ));
    }
}

pub fn splash_rings<const N: usize>(splashes: &[Splash], time: f64, head_world: Vec3) -> [[f32; 4]; N] {
    let mut live: Vec<(f32, [f32; 4])> = splashes
        .iter()
        .filter_map(|s| {
            let age = (time - s.born) as f32;
            (0.0..super::water_pipeline::RING_SECONDS).contains(&age).then(|| {
                ((s.position - head_world).length_squared(), [s.position.x, s.position.z, age, splash_strength(s.speed, s.size)])
            })
        })
        .collect();
    live.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut out = [[0.0; 4]; N];
    for (o, (_, r)) in out.iter_mut().zip(live) {
        *o = r;
    }
    out
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
    let l = super::room_light::room_light_at(descs, world);
    Vec3::new(l[0][0], l[0][1], l[0][2]) * 0.282_095
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
    layers.push(crown_frame());
    layers
}

fn crown_frame() -> Vec<u8> {
    let n = ATLAS_SIZE as usize;
    let mut clear = vec![1.0f32; n * n];
    let mut splat = |cx: f32, cy: f32, r: f32, a: f32| {
        let (x0, x1) = (((cx - 3.0 * r) * n as f32).floor().max(0.0) as usize, ((cx + 3.0 * r) * n as f32).ceil().min(n as f32) as usize);
        let (y0, y1) = (((cy - 3.0 * r) * n as f32).floor().max(0.0) as usize, ((cy + 3.0 * r) * n as f32).ceil().min(n as f32) as usize);
        for y in y0..y1 {
            for x in x0..x1 {
                let (px, py) = ((x as f32 + 0.5) / n as f32, (y as f32 + 0.5) / n as f32);
                let d2 = ((px - cx) * (px - cx) + (py - cy) * (py - cy)) / (r * r);
                clear[y * n + x] *= 1.0 - a * (-d2 * 2.0).exp();
            }
        }
    };
    let base = (0.5f32, 0.95f32);
    for j in 0..26u64 {
        let u = uniforms(hash64(0xc0ffee ^ j.wrapping_mul(0x9e3779b97f4a7c15)));

        let lean = (2.0 * u[0] - 1.0) * (0.25 + 0.75 * u[1] * u[1]) * 1.1;
        let len = lerp(0.4, 0.88, u[2]) * (1.0 - 0.3 * lean.abs());
        let w0 = lerp(0.016, 0.03, u[3]);
        let droop = 0.12 * u[4];
        let mut tip = base;
        for k in 0..48 {
            let t = k as f32 / 47.0;
            let x = base.0 + 0.8 * (lean.sin() * len * t + lean.signum() * droop * t * t);
            let y = base.1 - lean.cos() * len * t * (1.0 - 0.25 * t * t);
            splat(x, y, w0 * (1.0 - 0.65 * t), 0.55 * (1.0 - 0.35 * t));
            tip = (x, y);
        }
        splat(tip.0, tip.1 - 0.02, w0 * 1.1, 0.8);
    }
    for k in 0..40u64 {
        let u = uniforms(hash64(0xd06e ^ k.wrapping_mul(0x632be59bd9b4e019)));
        splat(base.0 + (2.0 * u[0] - 1.0) * 0.3, base.1 - 0.06 * u[1], lerp(0.03, 0.06, u[2]), 0.25);
    }
    let mut out = vec![0u8; n * n * 4];
    for (i, c) in clear.iter().enumerate() {
        let (x, y) = ((i % n) as f32 / n as f32, (i / n) as f32 / n as f32);

        let edge = smoothstep(0.0, 0.04, x) * smoothstep(1.0, 0.96, x) * smoothstep(0.0, 0.04, y);
        out[i * 4..i * 4 + 4].copy_from_slice(&[255, 255, 255, ((1.0 - c) * edge * 255.0).round() as u8]);
    }
    out
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
                _ => None,
            };
            let smeared: Vec<Vec<u8>> = match next {
                Some(m) if l >= FIRE_FIRST => vec![flame_smear(&layers[l as usize], dt), flame_smear(&layers[m as usize], -dt)],
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
        Self { _atlas: atlas, uniform, _cutouts: cutouts, bind_group, instances, capacity, over: 0, screened: 0 }
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
    }

    pub fn set_view(&self, queue: &Queue, uniform: &EffectsUniform) {
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(uniform));
    }

    pub fn motion_parts(&self) -> (&Buffer, &BindGroup, u32, u32) {
        (&self.instances, &self.bind_group, self.over, self.screened)
    }

    pub fn any(&self) -> bool {
        self.over + self.screened > 0
    }

    pub fn draw(&self, pass: &mut RenderPass<'_>, pipelines: &EffectsPipeline, camera: &BindGroup, probe: &BindGroup) {
        if !self.any() {
            return;
        }
        pass.set_bind_group(0, camera, &[]);
        pass.set_bind_group(1, probe, &[]);
        pass.set_bind_group(2, &self.bind_group, &[]);
        pass.set_vertex_buffer(0, self.instances.slice(..));
        if self.over > 0 {
            pass.set_pipeline(&pipelines.over);
            pass.draw(0..VERTICES_PER_PARTICLE, 0..self.over);
        }
        if self.screened > 0 {
            pass.set_pipeline(&pipelines.screened);
            pass.draw(0..VERTICES_PER_PARTICLE, self.over..self.over + self.screened);
        }
    }
}

pub struct EffectsPipeline {
    pub over: RenderPipeline,
    pub screened: RenderPipeline,
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
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("effects_shader"),
            source: ShaderSource::Wgsl(view.shader(effects_shader()).into()),
        });
        let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("effects_layout"),
            bind_group_layouts: &[Some(uniform_layout), Some(probe_layout), Some(effects_layout)],
            immediate_size: 0,
        });

        let over = BlendComponent { src_factor: BlendFactor::One, dst_factor: BlendFactor::OneMinusSrcAlpha, operation: BlendOperation::Add };
        let screen = BlendComponent { src_factor: BlendFactor::OneMinusDst, dst_factor: BlendFactor::One, operation: BlendOperation::Add };
        let build = |label: &str, fragment: &str, color: BlendComponent| {
            device.create_render_pipeline(&RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                vertex: VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    compilation_options: PipelineCompilationOptions::default(),
                    buffers: &[Some(EffectInstance::layout())],
                },
                fragment: Some(FragmentState {
                    module: &shader,
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
        Self { over: build("effects_over", "fs_over", over), screened: build("effects_screened", "fs_screen", screen) }
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
}

@vertex fn vs_main(@builtin(vertex_index) vi: u32, inst: IIn) -> VOut {
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
    if (in.layer.z > 2.5) {

        let b = textureSampleGrad(fx_atlas, fx_samp, in.uv, layer, gx, gy);
        let phase = b.b * 6.2831853;
        let flicker = 0.8 + 0.2 * (1.0 + in.flicker) * sin(in.time * 2.3 + phase) * sin(in.time * 0.9 + 3.0 * phase);
        let alpha = clamp(in.colour.a * b.a, 0.0, 1.0);
        let shown = aces_fitted(fx_flame(in.colour.rgb, b.r * flicker) + in.light * b.g);
        return vec4<f32>(shown * alpha, alpha);
    }
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
    return vec4<f32>(aces_fitted(in.colour.rgb * strength), 0.0);
}
"
        .replace("CUT_VEC4S", &(ATLAS_LAYERS * 4).to_string())
        .replace("LAST_LAYER", &(ATLAS_LAYERS - 1).to_string())
        .replace("SMOKE_BACK", &SMOKE_BACK.to_string())
        .replace("FLAME_COVER", &format!("{FLAME_COVER:?}"))
        .replace("FLAME_EROSION", &format!("{FLAME_EROSION:?}"))
        .replace("BLUE_ROOT", &format!("{BLUE_ROOT:?}"))
        .replace("CONVEX", &format!("{CONVEX:?}"))
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
    let mut s = effects_shader();
    let swap = |s: &mut String, from: &str, to: &str, n: usize| {
        assert_eq!(s.matches(from).count(), n, "effects motion twin: `{from}` moved in the effects' shader");
        *s = s.replace(from, to);
    };

    let from = s.find("        var r = fx.right.xyz;\n").expect("effects motion twin: the upright axes moved");
    let end = "        along = u * c - r * s;\n";
    let to = from + s[from..].find(end).expect("effects motion twin: the upright axes moved") + end.len();
    let upright = s[from..to].to_string();
    swap(&mut s, "@group(0) @binding(0) var<uniform> camera: Camera;", "", 1);
    swap(&mut s, "camera.view_proj[view_slot]", "fx_motion_view()", 2);
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
    @location(6) @interpolate(flat) soft: f32,
    @location(7) @interpolate(flat) time: f32,
    @location(8) @interpolate(flat) flicker: f32,
    @location(9) curr: vec4<f32>,
    @location(10) prev: vec4<f32>,

    @location(11) @interpolate(flat) gas: vec4<f32>,
    @location(12) @interpolate(flat) book: f32,
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
    out.soft = n.soft;
    out.time = n.time;
    out.flicker = n.flicker;
    out.curr = n.clip;
    out.prev = b.clip;
    out.gas = vec4<f32>(0.0);
    out.book = 0.0;
    if (inst.params.w > 1.5 && inst.params.w < 2.5 && motion.w > 0.0) {{
        let height = 2.0 * inst.centre.w * max(abs(inst.params.z), 1.0);
        out.gas = cam.prev * vec4<f32>(fx_flame_along(inst) * height, 0.0);
        out.book = motion.w * step;
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
    v.soft = in.soft;
    v.time = in.time;
    v.flicker = in.flicker;
    return v;
}}

@fragment fn fs_fx_motion_over(in: FxMotionOut) -> @location(0) vec4<f32> {{
    let shown = fx_over(fx_vout(in));
    var prev = in.prev;
    if (in.book > 0.0) {{

        let v = 1.0 - in.uv.y;
        prev = prev - in.gas * (v - fx_flame_v(v, in.book));
    }}
    let flame = in.layer.z > 1.5 && in.layer.z < 2.5;
    let w = layer_weight(shown.a, select({soft:?}, {bright:?}, flame));
    return vec4<f32>(motion_of(in.curr, prev).xyz * w, w);
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
        Surroundings { head, offset: Vec3::ZERO, yaw_inv: Quat::IDENTITY, lights: &[], reaches: &every, ambient: &dim, exposure: 1.0 }
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
        Splash { position: Vec3::new(0.0, 0.0, 0.0), born, speed: 1.3, size: 0.08, sunlit: 1.0, depth: 0.3, seed: 11 }
    }

    #[test]
    fn a_splash_throws_drops_that_rise_fall_back_and_are_gone() {
        let at = seen_from(Vec3::new(0.0, 1.6, 2.0));
        let s = [step_splash(10.0)];
        let count = |t: f64| simulate_with(&[], &s, t, &at).screened;
        assert_eq!(count(9.9), 0, "nothing before it struck");
        assert!(count(10.15) >= 12, "drops in the air: {}", count(10.15));
        assert_eq!(count(10.0 + f64::from(SPLASH_SECONDS) + 0.01), 0, "all fallen back");

        let f = simulate_with(&[], &s, 10.2, &at);
        let ys: Vec<f32> = f.instances[f.over as usize..].iter().map(|i| i.centre[1]).collect();
        assert!(ys.iter().all(|&y| y >= -1e-3), "{ys:?}");
        assert!(ys.iter().copied().fold(0.0, f32::max) > 0.05, "{ys:?}");
        assert!(f.over >= 2, "spray where it struck");
    }

    #[test]
    fn a_harder_strike_throws_more_and_higher() {
        let at = seen_from(Vec3::new(0.0, 1.6, 2.0));
        let soft = [step_splash(0.0)];
        let hard = [Splash { speed: 4.0, size: 0.15, ..step_splash(0.0) }];
        let (a, b) = (simulate_with(&[], &soft, 0.25, &at), simulate_with(&[], &hard, 0.25, &at));
        let top = |f: &EffectFrame| f.instances[f.over as usize..].iter().map(|i| i.centre[1]).fold(0.0, f32::max);
        assert!(b.screened > 2 * a.screened, "{} vs {}", b.screened, a.screened);
        assert!(top(&b) > 2.0 * top(&a), "{} vs {}", top(&b), top(&a));
    }

    #[test]
    fn drops_sparkle_against_the_sun_and_not_away_from_it() {

        let sun = [Light { kind: LightKind::Directional, direction: Vec3::new(0.0, -0.3, 1.0).normalize(), intensity: 5.0, ..lamp() }];
        let dark = |_: Vec3| Vec3::ZERO;
        let s = [step_splash(0.0)];
        let mean = |head: Vec3| {
            let f = simulate_with(&[], &s, 0.2, &Surroundings { lights: &sun, ambient: &dark, ..seen_from(head) });
            let drops = &f.instances[f.over as usize..];
            drops.iter().map(|i| i.colour[0]).sum::<f32>() / drops.len().max(1) as f32
        };
        let (into, away) = (mean(Vec3::new(0.0, 0.6, 3.0)), mean(Vec3::new(0.0, 0.6, -3.0)));
        assert!(into > 3.0 * away, "into the sun {into}, with it behind {away}");
        let shaded = [Splash { sunlit: 0.0, ..step_splash(0.0) }];
        let f = simulate_with(&[], &shaded, 0.2, &Surroundings { lights: &sun, ambient: &dark, ..seen_from(Vec3::new(0.0, 0.6, 3.0)) });
        assert!(f.instances.iter().all(|i| i.colour[0] < 1e-6), "a shaded splash has no sun in it");
    }

    #[test]
    fn the_nearest_live_rings_go_to_the_water() {
        let s: Vec<Splash> = (0..6).map(|k| Splash { position: Vec3::new(k as f32, 0.0, 0.0), ..step_splash(1.0) }).collect();
        let r = splash_rings::<4>(&s, 2.0, Vec3::new(5.0, 1.6, 0.0));
        assert_eq!(r.map(|r| r[0]), [5.0, 4.0, 3.0, 2.0]);
        assert!((r[0][2] - 1.0).abs() < 1e-6 && r[0][3] > 0.5);
        let gone = splash_rings::<4>(&s, 1.0 + f64::from(super::super::water_pipeline::RING_SECONDS) + 0.1, Vec3::ZERO);
        assert!(gone.iter().all(|r| r[3] == 0.0));
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
    #[ignore = "diagnostic that prints sizes or timings or writes files; run by hand with --ignored"]
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
        let splashes = [Splash { position: Vec3::new(-1.0, 0.0, 1.0), born: 9.8, speed: 2.5, size: 0.08, sunlit: 1.0, depth: 0.3, seed: 7 }];
        let mut frames = Vec::new();
        for (time, yaw, offset) in [(10.0f64, 0.0f32, Vec3::ZERO), (10.35, 0.0, Vec3::ZERO), (23.7, 0.9, Vec3::new(1.0, 0.0, -2.0))] {
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
                "position": s.position.to_array(), "born": s.born, "speed": s.speed, "size": s.size, "sunlit": s.sunlit, "depth": s.depth, "seed": s.seed.to_string(),
            })).collect::<Vec<_>>(),
            "frames": frames,
        });
        std::fs::write(out.join("effects_golden.json"), serde_json::to_string(&golden).unwrap()).expect("golden");
        eprintln!("wrote the atlas ({} layers), cutouts and {} golden frames to {}", layers.len(), 3, out.display());
    }

    #[test]
    #[ignore = "diagnostic that prints sizes or timings or writes files; run by hand with --ignored"]
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
    #[ignore = "diagnostic that prints sizes or timings or writes files; run by hand with --ignored"]
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
        let at = Surroundings { head: eye, lights: &fire_lights, ambient: &faint, exposure: 4.0, ..seen_from(eye) };
        let frame = simulate(&emitters, time, &at);
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
                    ops: Operations { load: LoadOp::Clear(Color { r: 0.015, g: 0.015, b: 0.018, a: 1.0 }), store: StoreOp::Store },
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
    #[ignore = "diagnostic that prints sizes or timings or writes files; run by hand with --ignored"]
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
    #[ignore = "diagnostic that prints sizes or timings or writes files; run by hand with --ignored"]
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
    #[ignore = "diagnostic that prints sizes or timings or writes files; run by hand with --ignored"]
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
    #[ignore = "diagnostic that prints sizes or timings or writes files; run by hand with --ignored"]
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
            ("splash", 0.25, 1e-3, Box::new(|t| simulate_with(&[], &[step_splash(0.0)], t, &at))),
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
