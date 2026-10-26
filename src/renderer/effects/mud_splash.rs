//! MUD SPLASHES: what wet mud does when something heavy lands in it, a blast
//! goes off in it or a bullet strikes it.
//!
//! A mud splash is a [`Splash`] with `mud` above 0, so it rides everything a
//! water splash does (the governor, the view cull, the server's entries, the
//! renderer's list), and draws here instead of as water. What it is comes
//! from the strike, as a water splash's does, by the scaling laws for
//! cratering:
//!
//! - THE CRATER. The energy the ground takes digs a bowl as wide as gravity
//!   allows, `K (E / rho g)^(1/4)`, or as the mud's strength allows,
//!   `K (E / 3 tau)^(1/3)`, whichever is smaller (Housen and Holsapple's two
//!   regimes, blended). Wet mud is weak, so most strikes dig in the gravity
//!   regime; stiff mud holds a pebble to a dent.
//!   - A STRIKE gives the ground only the drag work of its first few body
//!     radii, and only part of that: a bullet keeps going into the mud with
//!     most of its energy, a dropped crate gives up all of its own.
//!   - A BLAST is told by its crater (`Splash::mud_blast`), and the energy
//!     is found from it, so a blast and a strike dig by the same law.
//! - THE EJECTA. A third of the bowl is thrown, the rest pushed into the
//!   rim. Each piece leaves from where it lay, faster the nearer the middle:
//!   `v = C sqrt(g R + 3 tau / rho) (x / R)^(-1/mu)`, capped at half the
//!   striker's speed, and earlier the nearer the middle.
//! - CLUMPS, not drops. Mud breaks where the stretching beats its yield
//!   stress, so a piece thrown at `v` is about `We tau / (rho v^2)` across:
//!   wet mud flung fast goes to spray, stiff mud and slow throws come off in
//!   clods (up to a third of the crater). Wet mud strings out into ligaments
//!   as it flies.
//! - LANDINGS. Every clump comes down where its flight says, and is stamped
//!   into the mud map there as a pat of mud ([`landings`],
//!   `MudBody::land`); the crater is stamped when the splash is raised.
//!
//! Clumps are drawn as LUMPS (`SHADE_LUMP`): one atlas read of a lumpy
//! blob's normal, lit by the sun with a wet sheen and by the room's light,
//! so a clump is a solid thing and not a flat dab. Every clump carries its
//! velocity for the `SpaceWarp` motion pass.
//!
//! See docs/engine/mud-and-deform.md.

use glam::Vec3;

use super::{
    aloft, apex, hash64, lerp, noise3, smoothstep, uniforms, EffectInstance, LightKind, Splash, Surroundings, ATLAS_SIZE, BODY_DRAG, DROP_DRAG, GRAVITY,
    MUD_LAYER, MUD_VARIANTS, SPLASH_SECONDS, WATER_DENSITY,
};

/// The shading kind of a lump (`EffectInstance::params[3]`, `FxVariant::Lump`).
pub const SHADE_LUMP: f32 = 5.0;

/// Wet mud, kg/m3.
pub const MUD_DENSITY: f32 = 1700.0;
/// The yield stress of mud a boot barely marks, Pa, and of the wettest mud
/// that still holds a print. `Splash::mud` runs between them, by ratio.
pub const YIELD_STIFF: f32 = 2000.0;
pub const YIELD_SLURRY: f32 = 60.0;
/// Wet mud's colour, linear: a dark brown, darker than dry soil because the
/// water fills its pores.
pub const MUD_ALBEDO: Vec3 = Vec3::new(0.07, 0.052, 0.036);

/// The most clumps, and spray specks, one splash draws.
pub const MUD_CLUMPS: usize = 90;
pub const MUD_SPRAY: usize = 90;

/// The crater laws' constants: gravity regime, strength regime, and the
/// bowl's depth over its radius.
const GRAVITY_CRATER: f32 = 0.9;
const STRENGTH_CRATER: f32 = 0.8;
const BOWL: f32 = 0.42;
/// A strike's drag work over this many body radii goes into the ground, and
/// this share of that digs (the rest is heat and the body's own way down).
const COUPLED_RADII: f32 = 3.0;
const CRATER_SHARE: f32 = 0.15;
/// Mud flows under a strike once the push into it passes its bearing
/// capacity, about this many times its yield stress (Skempton's `N_c`).
const BEARING: f32 = 9.0;
/// The share of the bowl that is thrown, not pushed into the rim.
const EJECTA_SHARE: f32 = 0.35;
/// The ejecta speed law: its constant, its exponent `mu`, and the inmost
/// launch point (of the crater's radius) it is followed to.
const THROW: f32 = 0.7;
const MU: f32 = 0.55;
const INNER: f32 = 0.3;
/// The Weber number a thrown clump breaks at, against the yield stress.
const CLUMP_WEBER: f32 = 4.0;
/// The finest spray speck, metres across.
const FINEST: f32 = 2.5e-3;
/// MUD IS STRONGER DEEPER: a soft clay's shear strength rises about this
/// much per metre down, Pa/m, as the weight above consolidates it. What a
/// crater throws comes from down to half its depth, so a blast's deep bowl
/// tears out clods where a strike's shallow one throws spray.
const STRENGTH_PER_METRE: f32 = 1500.0;
/// Below the top this deep, metres: the wet top is soft right through.
const SOFT_TOP: f32 = 0.1;
/// The share of the throw that goes to spray, by mass, and how big a cloud
/// of it one speck may stand for (of the crater's radius, over a floor).
const SPRAY_SHARE: f32 = 0.25;
const SPECK_CLOUD: (f32, f32) = (0.012, 0.06);
/// A speck's streak: how far it moves in this long (about a frame and a
/// half), as a water drop's (`effects::DROP_STREAK`).
const SPECK_STREAK: f32 = 0.012;
/// How long wet mud strings out behind a flying clump, seconds of its speed.
const LIGAMENT: f32 = 0.012;
/// What a landed clump makes: a pat this many times its width across, and
/// this share of its width high.
const PAT_WIDTH: f32 = 1.6;
const PAT_HEIGHT: f32 = 0.25;

/// The yield stress of mud this wet (`Splash::mud`), Pa.
pub fn yield_stress(mud: f32) -> f32 {
    YIELD_STIFF * (YIELD_SLURRY / YIELD_STIFF).powf(mud.clamp(0.0, 1.0))
}

impl Splash {
    /// Something striking mud: as `Splash::new`, into mud this wet (0..1).
    pub fn mud(position: Vec3, born: f64, velocity: Vec3, size: f32, mass: f32, mud: f32) -> Self {
        Self { mud: mud.clamp(0.01, 1.0), ..Self::new(position, born, velocity, size, mass) }
    }

    /// A bullet into mud: as `Splash::projectile`.
    pub fn mud_projectile(position: Vec3, born: f64, velocity: Vec3, diameter: f32, mass: f32, mud: f32) -> Self {
        Self::mud(position, born, velocity, 0.5 * diameter, mass, mud)
    }

    /// A blast in mud that digs a crater `radius` metres across the middle
    /// (as `prop_system::BurstSpec::crater_radius_m`). The energy is the one
    /// the crater law needs for it.
    pub fn mud_blast(position: Vec3, born: f64, radius: f32, mud: f32) -> Self {
        let mud = mud.clamp(0.01, 1.0);
        Self { blast: blast_energy(radius, mud), size: radius.max(0.01), mud, ..Self::new(position, born, Vec3::ZERO, radius.max(0.01), 0.0) }
    }
}

/// The crater, metres, that `energy` joules dig in mud this wet.
pub fn crater_radius(energy: f32, mud: f32) -> f32 {
    if !(energy > 0.0) {
        return 0.0;
    }
    let gravity = GRAVITY_CRATER * (energy / (MUD_DENSITY * GRAVITY)).powf(0.25);
    let strength = STRENGTH_CRATER * (energy / (3.0 * yield_stress(mud))).cbrt();
    (gravity.powi(-4) + strength.powi(-4)).powf(-0.25)
}

/// The energy, joules, that digs a crater `radius` metres across in mud this
/// wet: [`crater_radius`] turned round.
pub fn blast_energy(radius: f32, mud: f32) -> f32 {
    if !(radius > 0.0) {
        return 0.0;
    }
    let (mut lo, mut hi) = (1e-6f32.ln(), 1e10f32.ln());
    for _ in 0..60 {
        let mid = f32::midpoint(lo, hi);
        if crater_radius(mid.exp(), mud) < radius {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    f32::midpoint(lo, hi).exp()
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MudShape {
    /// The crater's radius and depth, metres.
    pub crater: f32,
    pub depth: f32,
    /// How much it throws at all, 0..1: a strike too gentle to make the mud
    /// flow only dents it.
    pub throws: f32,
    /// The ejecta's speed at the rim, and its fastest (from the inmost
    /// launch point, or half the striker's speed), m/s.
    pub rim_speed: f32,
    pub top_speed: f32,
    /// The biggest clump, metres across.
    pub largest: f32,
    /// Clumps and spray specks drawn, and how many clumps each clump stands
    /// for, by its width (a blast throws far more than it draws).
    pub clumps: usize,
    pub spray: usize,
    pub cluster: f32,
    /// The mass thrown, kg.
    pub thrown: f32,
    /// How wide a speck sprite draws: the cloud of specks it stands for.
    pub speck: f32,
    /// How long the crater takes to open, seconds: the rim's throws leave last.
    pub forms: f32,
    /// Downrange, as long as the strike was glancing (0..1).
    pub forward: Vec3,
    pub yield_stress: f32,
    /// Seconds until the last clump lands.
    pub seconds: f32,
}

/// A thrown piece: where and when it leaves, how fast, how big, and when it
/// lands, all from the splash's own seed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Clump {
    pub from: Vec3,
    pub launch: Vec3,
    /// Linear drag rate, 1/s (as a drop's: `effects::ballistic`).
    pub drag: f32,
    pub size: f32,
    /// Seconds after the strike it leaves, and how long it flies.
    pub leaves: f32,
    pub flies: f32,
    pub turn: f32,
    pub spin: f32,
    pub variant: u32,
    pub fine: bool,
}

/// A throw under gravity and linear drag `k`, `age` seconds on: where it is
/// and how fast it goes. As `effects::ballistic`, but kept exact at the small
/// drag of a big clump, where `(1 - e^-kt) / k` cancels in f32 and the fall
/// came out a fifth short (a clump hit the ground 12 ms before its pat).
fn flight(p0: Vec3, v0: Vec3, k: f32, age: f32) -> (Vec3, Vec3) {
    let x = k * age;
    let (first, second) = if x < 0.05 {
        (1.0 - x * (0.5 - x * (1.0 / 6.0 - x / 24.0)), 0.5 - x * (1.0 / 6.0 - x * (1.0 / 24.0 - x / 120.0)))
    } else {
        let e = (-x).exp_m1();
        (-e / x, (x + e) / (x * x))
    };
    let gravity = Vec3::new(0.0, -GRAVITY, 0.0);
    let pos = p0 + v0 * (age * first) + gravity * (age * age * second);
    let vel = v0 * (1.0 - x * first) + gravity * (age * first);
    (pos, vel)
}

fn drag_of(speed: f32, size: f32) -> f32 {
    DROP_DRAG * (WATER_DENSITY / MUD_DENSITY) * speed / size.max(1e-4)
}

fn clump_size(speed: f32, tau: f32, largest: f32) -> f32 {
    (CLUMP_WEBER * tau / (MUD_DENSITY * speed * speed).max(1e-6)).clamp(FINEST, largest.max(FINEST))
}

pub fn mud_shape(s: &Splash) -> MudShape {
    use std::f32::consts::PI;
    let mud = s.mud.clamp(0.0, 1.0);
    let tau = yield_stress(mud);
    let speed = s.velocity.length();
    let r = s.size.max(1e-3);
    let flat = Vec3::new(s.velocity.x, 0.0, s.velocity.z);
    let (energy, throws, cap, smallest) = if s.blast > 0.0 {
        (s.blast, 1.0, f32::INFINITY, 0.0)
    } else {
        let normal = (-s.velocity.y).max(0.0);
        let ke = 0.5 * s.mass.max(0.0) * speed * speed;
        let stops = 2.0 * s.mass.max(1e-5) / (MUD_DENSITY * BODY_DRAG * PI * r * r);
        let coupled = 1.0 - (-COUPLED_RADII * r / stops).exp();
        // A glancing hit still digs, by the share of its push into the mud.
        let pressure = MUD_DENSITY * (normal * normal + 0.15 * flat.length_squared());
        (ke * coupled * CRATER_SHARE, smoothstep(1.0, 6.0, pressure / (BEARING * tau)), 0.5 * speed, 1.1 * r)
    };
    let crater = crater_radius(energy, mud).max(smallest);
    let depth = BOWL * crater;
    let rim_speed = THROW * (GRAVITY * crater + 3.0 * tau / MUD_DENSITY).sqrt();
    let top_speed = (rim_speed * INNER.powf(-1.0 / MU)).min(cap).max(rim_speed.min(cap));
    let largest = if s.blast > 0.0 { 0.3 * crater } else { (0.3 * crater).min(1.2 * r.max(0.03)) };
    let tau = tau + STRENGTH_PER_METRE * (0.5 * depth - SOFT_TOP).max(0.0);
    let thrown = EJECTA_SHARE * MUD_DENSITY * 0.5 * PI * crater * crater * depth * throws;
    let middle = (rim_speed * top_speed).sqrt();
    let typical = clump_size(middle, tau, largest);
    let pieces = thrown / (MUD_DENSITY * PI / 6.0 * typical.powi(3));
    let clumps = if throws > 0.05 { pieces.min(MUD_CLUMPS as f32).round() as usize } else { 0 };
    let cluster = if clumps > 0 { (pieces / clumps as f32).cbrt().clamp(1.0, 2.0) } else { 1.0 };
    let spray = if clumps > 0 { (MUD_SPRAY as f32 * throws * smoothstep(2.0, 8.0, top_speed) * (0.4 + 0.6 * mud)).round() as usize } else { 0 };
    let specks = SPRAY_SHARE * thrown / (MUD_DENSITY * PI / 6.0 * FINEST.powi(3));
    let speck = if spray > 0 { ((specks / spray as f32).cbrt() * FINEST).clamp(FINEST, SPECK_CLOUD.0 + SPECK_CLOUD.1 * crater) } else { FINEST };
    let forms = (0.5 * crater / rim_speed.max(0.05)).clamp(0.02, 0.4);
    let forward = if speed > 1e-3 && s.blast <= 0.0 { flat / speed } else { Vec3::ZERO };
    let flight = |v: f32, size: f32| aloft(v * 62f32.to_radians().sin(), drag_of(v, size));
    let seconds = if clumps > 0 {
        (forms + flight(top_speed, FINEST.max(clump_size(top_speed, tau, largest))).max(flight(rim_speed, largest)) * 1.15 + 0.1).min(SPLASH_SECONDS)
    } else {
        0.0
    };
    MudShape { crater, depth, throws, rim_speed, top_speed, largest, clumps, spray, cluster, thrown, speck, forms, forward, yield_stress: tau, seconds }
}

/// Every piece the splash throws, clumps first and then spray.
pub fn clumps(s: &Splash, shape: &MudShape) -> Vec<Clump> {
    use std::f32::consts::TAU;
    let lean = shape.forward.length();
    let toward = shape.forward.normalize_or_zero();
    let mut out = Vec::with_capacity(shape.clumps + shape.spray);
    for i in 0..shape.clumps + shape.spray {
        let fine = i >= shape.clumps;
        let (k, n) = if fine { (i - shape.clumps, shape.spray) } else { (i, shape.clumps) };
        let u = uniforms(hash64(s.seed ^ 0x6d75_64 ^ (i as u64 + 1).wrapping_mul(0x632be59bd9b4e019)));
        // Stratified round the rim, so any seed surrounds it.
        let azimuth = TAU * (k as f32 + u[0]) / n.max(1) as f32 + if fine { 0.37 } else { 0.0 };
        let out_dir = Vec3::new(azimuth.cos(), 0.0, azimuth.sin());
        let down_range = out_dir.dot(toward) * lean;
        // Spray is the fast fine fraction from near the middle.
        let x = if fine { lerp(INNER, 0.55, u[1]) } else { lerp(INNER, 1.0, u[1].sqrt()) };
        let speed = ((shape.rim_speed * x.powf(-1.0 / MU)).min(shape.top_speed) * lerp(0.85, 1.15, u[2]) * (1.0 + 0.5 * down_range).max(0.25)).max(0.05);
        let along = (x - INNER) / (1.0 - INNER);
        let elevation = (lerp(62.0f32, 44.0, along).to_radians() * (1.0 - 0.35 * down_range.max(0.0)) + (u[3] - 0.5) * 0.25).clamp(0.12, 1.45);
        let launch = out_dir * (elevation.cos() * speed) + Vec3::Y * (elevation.sin() * speed);
        let size = if fine { lerp(FINEST, 2.0 * FINEST, u[4]) } else { (clump_size(speed, shape.yield_stress, shape.largest) * lerp(0.6, 1.5, u[4])).clamp(FINEST, shape.largest) };
        let drag = drag_of(speed, size);
        let leaves = shape.forms * x.powf(1.0 + 1.0 / MU) * lerp(0.8, 1.2, u[5]);
        let flies = aloft(launch.y, drag).max(0.02);
        out.push(Clump {
            from: s.position + out_dir * (x * shape.crater),
            launch,
            drag,
            size,
            leaves,
            flies,
            turn: u[6] * TAU,
            spin: (u[7] - 0.5) * 2.0 * (0.3 * speed / size).min(25.0),
            variant: ((u[8] * MUD_VARIANTS as f32) as u32).min(MUD_VARIANTS - 1),
            fine,
        });
    }
    out
}

/// Where a piece is `age` seconds after the strike, and how fast it goes;
/// none before it leaves or after it lands.
pub fn clump_at(c: &Clump, age: f32) -> Option<(Vec3, Vec3)> {
    let a = age - c.leaves;
    if a <= 0.0 || a >= c.flies {
        return None;
    }
    Some(flight(c.from, c.launch, c.drag, a))
}

/// The space every piece of the splash stays within.
pub fn bounds(s: &Splash) -> (Vec3, Vec3) {
    let shape = mud_shape(s);
    let fastest = shape.top_speed * 1.15 * 1.8;
    let slowest_drag = drag_of(fastest, shape.largest.max(FINEST));
    let high = apex(fastest, slowest_drag) + 2.0 * shape.largest + 0.1;
    let reach = if slowest_drag * fastest > 1e-3 { (fastest * shape.seconds).min(fastest / slowest_drag) } else { fastest * shape.seconds };
    let wide = shape.crater + reach + 2.0 * shape.largest + 0.1;
    (s.position - Vec3::new(wide, shape.depth + 0.1, wide), s.position + Vec3::new(wide, high, wide))
}

/// The splash's own pieces, drawn as lumps into `over`.
pub fn draw(s: &Splash, time: f64, at: &Surroundings, over: &mut Vec<(f32, EffectInstance)>) {
    let age = (time - s.born) as f32;
    let shape = mud_shape(s);
    if !(age >= 0.0 && age < shape.seconds) {
        return;
    }
    let player = |p: Vec3| at.yaw_inv * (p - at.offset);
    let head_world = at.yaw_inv.inverse() * at.head + at.offset;
    let away = head_world.distance(s.position);
    if away > (250.0 * (shape.crater + shape.largest)).max(20.0) {
        return;
    }
    let ambient = (at.ambient)(s.position) * MUD_ALBEDO * at.exposure;
    let sun = at.lights.iter().find(|l| matches!(l.kind, LightKind::Directional) && l.intensity > 0.0).map(|l| {
        let c = l.color.to_linear();
        (-l.direction.normalize_or_zero(), Vec3::new(c[0], c[1], c[2]) * (l.intensity * s.sunlit.clamp(0.0, 1.0)) * MUD_ALBEDO * at.exposure)
    });
    let (to_sun, lit) = sun.unwrap_or((Vec3::Y, Vec3::ZERO));
    let to_sun = at.yaw_inv * to_sun;
    // Far off, the specks are below a pixel: draw fewer, as the water's drops.
    let specks = ((shape.spray as f32 * (8.0 / away.max(0.1)).min(1.0)).ceil() as usize).min(shape.spray);
    let stretch = LIGAMENT * s.mud.clamp(0.0, 1.0);
    for (i, c) in clumps(s, &shape).iter().enumerate() {
        if c.fine && i - shape.clumps >= specks {
            continue;
        }
        let Some((pos, vel)) = clump_at(c, age) else { continue };
        if pos.y < s.position.y - shape.depth && vel.y < 0.0 {
            continue;
        }
        let centre = player(pos);
        let moving = at.yaw_inv * vel;
        let axis = moving.normalize_or(Vec3::Y);
        let half = 0.5 * if c.fine { shape.speck } else { c.size * shape.cluster };
        let streak = if c.fine { vel.length() * SPECK_STREAK } else { (vel.length() * stretch).min(2.5 * c.size) };
        over.push((
            (centre - at.head).length_squared(),
            EffectInstance {
                centre: [centre.x, centre.y, centre.z, half],
                colour: [ambient.x, ambient.y, ambient.z, 1.0],
                light: [lit.x, lit.y, lit.z, 0.0],
                light_dir: [to_sun.x, to_sun.y, to_sun.z, streak],
                axis: [axis.x, axis.y, axis.z, 0.02],
                params: [c.turn + c.spin * (age - c.leaves), (MUD_LAYER + c.variant) as f32, 0.0, SHADE_LUMP],
                motion: [moving.x, moving.y, moving.z, 0.0],
            },
        ));
    }
}

/// THE CRATER a mud splash digs, stamped when it is raised: its middle (x, z)
/// and its radius and depth, metres.
pub fn crater(s: &Splash) -> ([f32; 2], f32, f32) {
    let shape = mud_shape(s);
    ([s.position.x, s.position.z], shape.crater, shape.depth)
}

/// A clump come down: stamped into the mud map at `at` (the splash's clock)
/// as a pat `radius` metres across the narrow way and `stretch` times as long
/// along `heading` (a glancing landing smears), `height` metres high. Spray
/// only wets (`height` 0).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Landing {
    pub at: f64,
    pub centre: [f32; 2],
    pub radius: f32,
    pub heading: [f32; 2],
    pub stretch: f32,
    pub height: f32,
}

/// Where every piece of a mud splash comes down, by the same flights it is
/// drawn with, so each pat lies where its clump vanished.
pub fn landings(s: &Splash) -> Vec<Landing> {
    let shape = mud_shape(s);
    clumps(s, &shape)
        .iter()
        .map(|c| {
            let (pos, vel) = flight(c.from, c.launch, c.drag, c.flies);
            let flat = Vec3::new(vel.x, 0.0, vel.z);
            let heading = flat.normalize_or(Vec3::X);
            let size = if c.fine { shape.speck } else { c.size * shape.cluster };
            Landing {
                at: s.born + f64::from(c.leaves + c.flies),
                centre: [pos.x, pos.z],
                radius: 0.5 * PAT_WIDTH * size,
                heading: [heading.x, heading.z],
                stretch: 1.0 + (0.15 * flat.length() / vel.y.abs().max(0.3)).min(1.5),
                height: if c.fine { 0.0 } else { PAT_HEIGHT * size },
            }
        })
        .collect()
}

/// The landings still to come, of every mud splash raised: handed out as
/// their time comes ([`MudLandings::take`]), so a pat appears as its clump
/// comes down and not when the splash is raised.
#[derive(Default)]
pub struct MudLandings {
    due: Vec<Landing>,
}

impl MudLandings {
    /// The most kept waiting; the soonest are kept.
    pub const KEPT: usize = 4096;

    pub fn add(&mut self, s: &Splash) {
        self.due.extend(landings(s));
        if self.due.len() > Self::KEPT {
            self.due.sort_by(|a, b| a.at.total_cmp(&b.at));
            self.due.truncate(Self::KEPT);
        }
    }

    /// Every landing due by `now`, taken out.
    pub fn take(&mut self, now: f64) -> Vec<Landing> {
        let (due, later): (Vec<Landing>, Vec<Landing>) = self.due.drain(..).partition(|l| l.at <= now);
        self.due = later;
        due
    }

    pub fn waiting(&self) -> usize {
        self.due.len()
    }
}

/// A LUMP'S SPRITE: a clod's outline and surface, stored as its normal in
/// RGB (x right, y up, z toward the eye; 0.5 is 0) and its cover in alpha.
/// Each variant is a different clod: an uneven outline of a few harmonics,
/// a domed top, and small bumps over it.
pub fn lump_frame(variant: u32) -> Vec<u8> {
    let n = ATLAS_SIZE as usize;
    let seed = 0x6c75_6d70u64 ^ ((u64::from(variant) + 1) << 24);
    let u = uniforms(hash64(seed));
    let harmonics: [(f32, f32); 5] = std::array::from_fn(|k| (0.13 * lerp(0.4, 1.2, u[k]) / (k as f32 + 1.0).powf(0.8), u[k + 5] * std::f32::consts::TAU));
    let outline = move |a: f32| 0.72 * (1.0 + harmonics.iter().enumerate().map(|(k, (amp, ph))| amp * ((k as f32 + 2.0) * a + ph).cos()).sum::<f32>());
    let height = |px: f32, py: f32| -> f32 {
        let rho = (px * px + py * py).sqrt();
        let edge = outline(py.atan2(px));
        let q = rho / edge;
        if q >= 1.0 {
            return 0.0;
        }
        let dome = (1.0 - q * q).sqrt();
        let bumps = noise3(Vec3::new(px * 3.0, py * 3.0, variant as f32 * 3.1), seed) + 0.3 * noise3(Vec3::new(px * 7.0, py * 7.0, 0.5), seed ^ 1);
        0.72 * dome * (0.85 + 0.05 * bumps)
    };
    let step = 2.0 / n as f32;
    let mut out = vec![0u8; n * n * 4];
    for y in 0..n {
        for x in 0..n {
            let px = (x as f32 + 0.5) * step - 1.0;
            let py = 1.0 - (y as f32 + 0.5) * step;
            let rho = (px * px + py * py).sqrt();
            let inside = (outline(py.atan2(px)) - rho) / step;
            let cover = smoothstep(-0.75, 0.75, inside);
            let o = (y * n + x) * 4;
            if cover <= 0.0 {
                out[o..o + 4].copy_from_slice(&[128, 128, 255, 0]);
                continue;
            }
            let dx = (height(px + step, py) - height(px - step, py)) / (2.0 * step);
            let dy = (height(px, py + step) - height(px, py - step)) / (2.0 * step);
            let normal = Vec3::new(-dx, -dy, 1.0).normalize();
            let byte = |v: f32| ((v * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0).round() as u8;
            out[o..o + 4].copy_from_slice(&[byte(normal.x), byte(normal.y), byte(normal.z), (cover * 255.0).round() as u8]);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn crate_in_mud(mud: f32) -> Splash {
        Splash { seed: 3, ..Splash::mud(Vec3::ZERO, 0.0, Vec3::new(0.0, -6.3, 0.0), 0.25, 30.0, mud) }
    }
    fn rock(mud: f32) -> Splash {
        Splash { seed: 5, ..Splash::mud(Vec3::ZERO, 0.0, Vec3::new(0.0, -6.3, 0.0), 0.05, 1.0, mud) }
    }
    fn pistol_round(mud: f32) -> Splash {
        Splash { seed: 7, ..Splash::mud_projectile(Vec3::ZERO, 0.0, Vec3::new(280.0, -280.0, 0.0), 0.009, 0.008, mud) }
    }
    fn grenade(mud: f32) -> Splash {
        Splash { seed: 9, ..Splash::mud_blast(Vec3::ZERO, 0.0, 1.0, mud) }
    }

    /// Each strike digs and throws about what it does in life: the ranges are
    /// read off films of drops into mud, bullets into wet ground and blasts in
    /// fields, generously.
    #[test]
    fn a_crate_a_rock_a_bullet_and_a_blast_each_splash_mud_as_what_they_are() {
        for (name, s, crater, top, apex_m) in [
            ("crate", crate_in_mud(1.0), (0.25, 0.5), (2.0, 4.0), (0.15, 0.8)),
            ("rock", rock(1.0), (0.06, 0.2), (1.5, 4.0), (0.08, 0.8)),
            ("bullet", pistol_round(1.0), (0.05, 0.2), (3.0, 12.0), (0.3, 3.0)),
            ("blast", grenade(1.0), (0.95, 1.05), (12.0, 25.0), (4.0, 20.0)),
        ] {
            let shape = mud_shape(&s);
            let highest = clumps(&s, &shape).iter().map(|c| apex(c.launch.y, c.drag)).fold(0.0, f32::max);
            eprintln!(
                "{name:<7} crater {:.3} m deep {:.3}, throws {:.2}, rim {:.2} top {:.2} m/s, largest {:.3} m, {} clumps x{:.2} + {} spray ({:.3} m clouds), {:.2} kg, apex {:.2} m, {:.2} s",
                shape.crater, shape.depth, shape.throws, shape.rim_speed, shape.top_speed, shape.largest, shape.clumps, shape.cluster, shape.spray, shape.speck, shape.thrown, highest, shape.seconds
            );
            assert!(shape.crater >= crater.0 && shape.crater <= crater.1, "{name}: crater {}", shape.crater);
            assert!(shape.top_speed >= top.0 && shape.top_speed <= top.1, "{name}: top speed {}", shape.top_speed);
            assert!(highest >= apex_m.0 && highest <= apex_m.1, "{name}: highest clump {highest}");
            assert!(shape.clumps > 4, "{name}");
            assert!(shape.seconds > 0.2 && shape.seconds <= SPLASH_SECONDS, "{name}: {}", shape.seconds);
        }
    }

    #[test]
    fn stiff_mud_holds_a_rock_to_a_dent_and_breaks_into_clods() {
        let (wet, stiff) = (mud_shape(&rock(1.0)), mud_shape(&rock(0.0)));
        assert!(stiff.crater < wet.crater, "{} {}", stiff.crater, wet.crater);
        assert!(stiff.throws < wet.throws);
        let (wet, stiff) = (mud_shape(&grenade(1.0)), mud_shape(&grenade(0.0)));
        let mean = |s: &Splash, sh: &MudShape| {
            let c = clumps(s, sh);
            c.iter().filter(|c| !c.fine).map(|c| c.size).sum::<f32>() / c.iter().filter(|c| !c.fine).count().max(1) as f32
        };
        assert!(mean(&grenade(0.0), &stiff) > 2.0 * mean(&grenade(1.0), &wet), "stiff mud comes off in clods");
        let lowered = mud_shape(&Splash { seed: 1, ..Splash::mud(Vec3::ZERO, 0.0, Vec3::new(0.0, -0.1, 0.0), 0.25, 30.0, 0.5) });
        assert_eq!((lowered.clumps, lowered.spray), (0, 0), "a crate set down gently throws nothing");
    }

    #[test]
    fn a_blast_digs_the_crater_it_was_told() {
        for mud in [0.1, 0.5, 1.0] {
            for radius in [0.2, 1.0, 3.0] {
                let s = Splash::mud_blast(Vec3::ZERO, 0.0, radius, mud);
                assert!((mud_shape(&s).crater - radius).abs() < 0.01 * radius, "{mud} {radius}: {}", mud_shape(&s).crater);
            }
        }
    }

    #[test]
    fn a_glancing_bullet_throws_mud_downrange() {
        let s = pistol_round(1.0);
        let shape = mud_shape(&s);
        let ahead: f32 = clumps(&s, &shape).iter().map(|c| c.launch.x).sum();
        assert!(ahead > 0.0, "the spray leans the way the bullet went: {ahead}");
    }

    #[test]
    fn every_piece_stays_within_its_splashs_bounds_and_lands_where_it_vanishes() {
        for s in [crate_in_mud(1.0), rock(0.5), pistol_round(1.0), grenade(1.0), grenade(0.0)] {
            let shape = mud_shape(&s);
            let (lo, hi) = bounds(&s);
            let pieces = clumps(&s, &shape);
            let mut t = 0.0;
            while t < shape.seconds {
                for c in &pieces {
                    if let Some((p, _)) = clump_at(c, t) {
                        assert!(p.cmpge(lo).all() && p.cmple(hi).all(), "{p} outside {lo}..{hi}");
                    }
                }
                t += 0.01;
            }
            for (c, l) in pieces.iter().zip(landings(&s)) {
                let (last, vel) = clump_at(c, c.leaves + c.flies - 1e-3).expect("drawn until it lands");
                assert!((last.x - l.centre[0]).abs() < 0.05 && (last.z - l.centre[1]).abs() < 0.05, "{last} vs {:?}", l.centre);
                assert!(last.y.abs() < 0.01 + 1.1e-3 * vel.y.abs(), "it lands on the mud: {} at {} m/s", last.y, vel.y);
                assert!(l.at <= s.born + f64::from(shape.seconds) + 1e-3, "{} after {}", l.at, shape.seconds);
            }
        }
    }

    #[test]
    fn landings_are_handed_out_as_their_time_comes() {
        let mut due = MudLandings::default();
        due.add(&crate_in_mud(1.0));
        let all = due.waiting();
        assert!(all > 0);
        let first = due.take(0.0).len();
        let mid = due.take(0.4).len();
        let rest = due.take(10.0).len();
        assert_eq!(first + mid + rest, all);
        assert!(first < all && rest < all, "{first} {mid} {rest}");
        assert_eq!(due.waiting(), 0);
    }

    #[test]
    fn a_lump_sprite_is_a_lit_solid_facing_the_eye() {
        for v in 0..MUD_VARIANTS {
            let f = lump_frame(v);
            let n = ATLAS_SIZE as usize;
            let at = |x: usize, y: usize| &f[(y * n + x) * 4..(y * n + x) * 4 + 4];
            // Bumpy, but as a whole it faces the eye.
            let mut mean = Vec3::ZERO;
            for y in n / 4..3 * n / 4 {
                for x in n / 4..3 * n / 4 {
                    let p = at(x, y);
                    if p[3] == 255 {
                        mean += Vec3::new(f32::from(p[0]), f32::from(p[1]), f32::from(p[2])) / 127.5 - Vec3::ONE;
                    }
                }
            }
            let mean = mean.normalize();
            assert!(mean.z > 0.9, "the middle faces the eye: {mean}");
            assert_eq!(at(n / 2, n / 2)[3], 255, "the middle is solid");
            assert_eq!(at(0, 0)[3], 0, "the corner is clear");
            // Toward the right edge the surface turns right, toward the top up.
            let right = (n / 2..n).rev().map(|x| at(x, n / 2)).find(|p| p[3] > 200).unwrap();
            let top = (0..n / 2).map(|y| at(n / 2, y)).find(|p| p[3] > 200).unwrap();
            assert!(right[0] > 200 && top[1] > 200, "{right:?} {top:?}");
        }
    }
}
