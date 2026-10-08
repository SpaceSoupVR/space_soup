//! THE EFFECTS: fire, the coals it burns on, smoke, embers and dust motes,
//! drawn in the scene pass after everything opaque and the glass.
//!
//! EVERY PARTICLE IS A CLOSED-FORM FUNCTION of its emitter, its slot and the
//! time, as the old particles were (`quest_app::particles`): nothing is held
//! from frame to frame, so the scene's emitters and the time are the whole
//! state, both eyes and every client agree, and a frame SpaceWarp makes up
//! needs nothing either. Each slot respawns every `period` seconds with a
//! seed of its own incarnation, so a cycle never repeats the last one.
//!
//! IN THE WORLD'S FRAME, then turned into the player's: a particle's spread
//! and swirl are directions, and worked out in the player's frame they would
//! turn with every snap turn. The motion is ballistic with linear drag
//! (`x(t)` exact), a constant acceleration (gravity, or buoyancy for flame
//! and smoke) and a sideways swirl or wander. Size, colour and opacity follow
//! the age.
//!
//! LIT ON THE CPU, once a particle a frame, by the lamps of the room its
//! emitter stands in (`lamp_in_room`) with the lights block's falloff and cone,
//! unshadowed, and by that room's baked light (`room_ambient`). Smoke takes
//! its strongest lamp through SIX-WAY LIGHTING: each frame of a puff holds how
//! much light it passes to the eye from each of six sides, worked out through
//! the puff's own volume when the textures are made (`smoke_frame`), so it
//! shadows itself and glows round its edges against the light. Dust scatters
//! forward (Henyey-Greenstein): motes sparkle looking toward a lamp. Fire,
//! its coals and embers emit, seen as an eye adapted to the fire sees them
//! (`tonemap::own_light_scale`) -- and so is the fire's own light on its
//! smoke, which at the room's exposure burned white over its flames.
//!
//! A FIRE is sheets of flame standing on its bed, each playing a stretch of a
//! flipbook of rising gas (`fire_frame`) and handing over to the next as it
//! fades; its coals are one bed lying under them (`coals_instance`). It
//! lights the room round it, and its smoke from below, with a flickering lamp
//! of its own (`fire_light`), and the eye adapts to it as to what it fixes on
//! (`meter_samples`).
//!
//! TWO PIPELINES: smoke, flames and coals premultiplied OVER what is behind
//! them, drawn back to front together, a flame covering only as much of what
//! is behind it as it outshines it (`FLAME_COVER`); embers and dust SCREENED
//! like the lamps' glare (`glare`), which needs no order -- dust only adds the
//! light it scatters, so a mote in shadow is not a dark speck. Both
//! pipelines write colour only: the eye image's alpha holds
//! each pixel's reflected share for SpaceWarp, whose motion pass they stay out
//! of. Both fade into what they meet from the probe pass's depth (half
//! resolution, brushes only; off when that pass did not run).
//!
//! CUT OUT, NOT SQUARE: each particle is drawn as the eight-sided outline
//! round its texture's visible part (`cutouts`), not its whole quad, so the
//! transparent corners of a puff or a flame cost no fill.
//!
//! THE TEXTURES are made here at load, from noise (`atlas_layers`): sixteen
//! frames of a smoke puff in two layers each (its light from the six sides
//! and its opacity), forty-eight of a sheet of flame (its heat in red, its
//! cover in alpha), an ember's dot and a mote's, and a bed of coals. No
//! download, the same bytes on every run.

use bytemuck::{Pod, Zeroable};
use glam::{Quat, Vec3};
use wgpu::*;

use super::lights::{Light, LightKind};
use super::probe_stream::ProbeDesc;

/// What an emitter makes: one of a few presets, each a whole look.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EffectKind {
    /// Flames: sheets of emitted light standing on the bed, over what is
    /// behind.
    Fire,
    /// Smoke: grey puffs that grow and thin as they rise, lit, over.
    Smoke,
    /// Embers: small hot sparks streaked along their motion, screened.
    Embers,
    /// Dust motes drifting in a box, seen where a lamp lights them, screened.
    Dust,
    /// A bed of glowing coals lying on the floor under a fire, over.
    Coals,
}

impl EffectKind {
    /// The kind a scene names, `"fire"`, `"smoke"`, `"embers"` or `"dust"`.
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

    /// The small light-like kinds, drawn screened; smoke and flames go over
    /// what is behind them, sorted together.
    pub fn screened(self) -> bool {
        matches!(self, Self::Embers | Self::Dust)
    }
}

/// One emitter, in the WORLD's frame, as the scene places it.
#[derive(Clone, Debug)]
pub struct EffectEmitter {
    /// Seeds its particles: two emitters of one kind look different.
    pub id: String,
    pub kind: EffectKind,
    pub position: Vec3,
    /// Where it emits toward; up for fire and smoke. Dust ignores it.
    pub direction: Vec3,
    /// Dust: half its box along each of the box's three axes, which the motes
    /// fill. Ignored by the other kinds.
    pub extent: [Vec3; 3],
    /// Size of everything it makes, and of the bed it rises from; 1 is the
    /// preset's.
    pub scale: f32,
    /// Particles a second, as a multiple of the preset's; 0 stops it.
    pub rate: f32,
    /// Linear multiplier on the preset's colour.
    pub tint: [f32; 3],
    /// The rock or ceiling over it, world y, if any: smoke rising into it
    /// spreads out beneath it and embers stop at it. Found by the client
    /// (`quest_app::scene_effects::find_ceilings`), not authored.
    pub ceiling: Option<f32>,
    /// How its particles differ from one another, and how strong and lively
    /// it is: the author's fine tuning over the preset. The default is the
    /// preset exactly. See [`Variation`].
    pub variation: Variation,
}

impl EffectEmitter {
    /// Its box's volume, cubic metres: what a dust emitter fills.
    pub fn volume(&self) -> f32 {
        8.0 * self.extent[0].length() * self.extent[1].length() * self.extent[2].length()
    }
}

/// AN EMITTER'S VARIATION, as the scene's `effect` names it (all optional
/// there): the editor's preview reads the same fields with the same formulas
/// (`scene_editor_web` `effectsSim.js`, held to this file by its golden
/// frames). Each field is applied only where it differs from its default,
/// through an expression that is the preset's own at the default, so an
/// emitter that names none is simulated bit for bit as one without them.
/// Out of range values are clamped ([`Variation::clamped`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Variation {
    /// x [0, 4]: fire -- its flames' light and its lamp's; coals -- their
    /// glow; embers -- their light; smoke -- its opacity; dust -- the light it
    /// scatters.
    pub intensity: f32,
    /// x the preset's spread of sizes [0, 3]: 1 is +-20%, 0 all alike; a
    /// flame's height too. Dust: the spread of its motes' light, in octaves.
    pub size_variation: f32,
    /// x the preset's spread of lifetimes about their mean [0, 2].
    pub life_variation: f32,
    /// x the preset's spread of launch speeds about their mean [0, 3]; for
    /// fire, of how fast each sheet plays its book.
    pub speed_variation: f32,
    /// [0, 1]: each flame sheet or spark hotter (toward yellow-white) or
    /// cooler (deep red) than the next; 0 is one colour.
    pub temperature_variation: f32,
    /// x the preset's flicker [0, 3]: a fire's light pulsing with its
    /// flames, a bed's coals, the sparks. 0 is steady.
    pub flicker: f32,
    /// x the preset's pace of flicker [0.25, 4]. For a fire it is the fire's
    /// whole tempo -- its flames' puffing and its light's together, as wind
    /// quickens a real one -- since a light pulsing at a rate its flames do
    /// not show is what gave the old firelight away.
    pub flicker_rate: f32,
    /// x the preset's turbulence [0, 4]: smoke's and sparks' swirl, how far
    /// the air carries and jostles dust, how far a fire's sheets lean.
    pub turbulence: f32,
    /// [0, 65535]: 0 keeps the id's own pattern; anything else mixes into it
    /// (`emitter_seed`), so two emitters of one id move differently.
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
    /// Every field inside its range, as the editor clamps it; not a number is
    /// the default.
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

/// The seed an emitter's particles, air, light and bed hang off: its id's, or
/// that mixed with its variation's `seed`.
fn emitter_seed(e: &EffectEmitter) -> u64 {
    let base = id_hash(&e.id);
    match e.variation.seed.min(65535) {
        0 => base,
        s => base ^ hash64(s as u64),
    }
}

/// The lifetimes a kind's particles are drawn between: its preset's, or
/// spread about their mean by `life_variation`.
fn life_range(p: &Preset, v: &Variation) -> (f32, f32) {
    if v.life_variation == 1.0 {
        return p.life;
    }
    let m = 0.5 * (p.life.0 + p.life.1);
    let h = 0.5 * (p.life.1 - p.life.0) * v.life_variation;
    ((m - h).max(0.05), m + h)
}

/// The launch speeds, likewise by `speed_variation`.
fn speed_range(p: &Preset, v: &Variation) -> (f32, f32) {
    if v.speed_variation == 1.0 {
        return p.speed;
    }
    let m = 0.5 * (p.speed.0 + p.speed.1);
    let h = 0.5 * (p.speed.1 - p.speed.0) * v.speed_variation;
    ((m - h).max(0.0), m + h)
}

/// A particle's own size factor from its uniform: the preset's 0.8-1.2, or
/// spread by `size_variation`.
fn size_spread(u: f32, v: &Variation) -> f32 {
    if v.size_variation == 1.0 {
        0.8 + 0.4 * u
    } else {
        (1.0 + (u - 0.5) * 0.4 * v.size_variation).max(0.05)
    }
}

/// A sheet's or spark's colour shift for `temperature_variation`: hotter ones
/// (by the uniform that also makes them brighter) toward yellow-white, cooler
/// toward deep red. White at the default 0.
fn temperature(u: f32, v: &Variation) -> Vec3 {
    if v.temperature_variation == 0.0 {
        return Vec3::ONE;
    }
    let k = (2.0 * u - 1.0) * v.temperature_variation;
    Vec3::new(1.0, 2f32.powf(0.8 * k), 2f32.powf(1.6 * k))
}

/// A kind's look and motion. Sizes are the quad's half size in metres at the
/// emitter's scale 1.
struct Preset {
    /// Particles a second; for dust, alive per cubic metre.
    rate: f32,
    life: (f32, f32),
    speed: (f32, f32),
    spread_deg: f32,
    /// Radius of the bed particles are born on.
    bed: f32,
    size: (f32, f32),
    drag: f32,
    accel: Vec3,
    /// Swirl radius reached at the end of a life, and its angular speed; for
    /// dust, how far a mote wanders and how fast.
    swirl: (f32, f32),
    /// Spin a second, and how far from upright it starts (radians either way).
    spin: (f32, f32),
    /// Seconds of motion an ember's streak covers; 0 faces the eye.
    streak: f32,
    /// Metres over which it fades into a wall or floor it meets.
    soft: f32,
    /// First texture layer and how many frames its flipbook has.
    frames: (u32, u32),
    /// Henyey-Greenstein asymmetry of its scattering; 0 for what emits.
    phase_g: f32,
    albedo: f32,
}

/// The texture array's layout: smoke frames' light from right, top and front
/// with their opacity; the same frames' light from left, bottom and back;
/// flame frames; an ember; a mote; a bed of coals.
pub const SMOKE_FRAMES: u32 = 16;
pub const SMOKE_BACK: u32 = SMOKE_FRAMES;
pub const FIRE_FIRST: u32 = 2 * SMOKE_FRAMES;
/// The flame's flipbook: `FIRE_FRAMES` frames over `FLAME_SECONDS` of one
/// sheet of flame, of which each of a fire's sheets plays a stretch.
pub const FIRE_FRAMES: u32 = 48;
pub const FLAME_SECONDS: f32 = 2.4;
pub const EMBER_LAYER: u32 = FIRE_FIRST + FIRE_FRAMES;
pub const MOTE_LAYER: u32 = EMBER_LAYER + 1;
pub const COALS_LAYER: u32 = MOTE_LAYER + 1;
pub const CROWN_LAYER: u32 = COALS_LAYER + 1;
pub const ATLAS_LAYERS: u32 = CROWN_LAYER + 1;
pub const ATLAS_SIZE: u32 = 128;
const ATLAS_MIPS: u32 = 8;

/// The most particles one emitter keeps alive.
const MAX_SLOTS: usize = 2048;

/// How far off a dust mote is drawn, metres: past it a mote is under a pixel
/// -- a sparkle of aliasing, not a mote -- and a hall of them cost the CPU
/// for nothing from the next room. Motes fade out over its last quarter, so
/// walking does not switch them on and off at a line.
const DUST_SEEN_WITHIN: f32 = 5.0;

/// How close a mote may come before it fades: nearer than this the eye is not
/// focused on it, and a 4 mm mote a hand's width away is a soft blot.
const DUST_NEAREST: (f32, f32) = (0.12, 0.35);

/// How much of the room's even light a mote is seen by. A mote lit as evenly
/// as the wall behind it is about as bright as that wall, and lost against
/// it: dust shows in a beam, by the light it scatters forward. Drawn
/// screened, it can only ADD to what is behind it, so the light it adds over
/// that wall is what it takes this share of.
const DUST_AMBIENT_SEEN: f32 = 0.15;

/// WHAT A MOTE IS SEEN AS. A sunbeam's motes are lint and skin flakes 10-50
/// microns across: at arm's length a twentieth of a headset pixel, so each is
/// a POINT whose brightness alone the eye sees, never a disc (drawn as 4 mm
/// discs, near ones were blots several pixels wide -- headset, 2026-10-07).
/// A mote's size here is only the light it sends: the radius of a disc as
/// bright that sends as much. Real dust is many small grains and few large,
/// and the light goes as the area, so from the faintest to 6x the radius,
/// most near the faint end; drawn always at the pixel floor
/// (`vs_main`'s point). The mean light is 0.4 of the old 4 mm disc's.
const DUST_LIGHT_SIZE: (f32, f32) = (0.00058, 0.0035);

/// A tumbling flake turning toward the mirror angle flashes, and turning away
/// all but vanishes (`mote_glint`): its brightness over its tumble, shifted
/// and scaled so its mean is its plain scattering.
const GLINT_FLOOR: f32 = 0.08;
const GLINT_MEAN: f32 = 0.357;

/// Vertices a particle is drawn with: its cutout's eight corners fanned from
/// its middle into eight triangles -- the middle pushed toward the eye for a
/// flame or a puff, so each is a low dome in stereo and not a flat card
/// (`CONVEX`).
pub const VERTICES_PER_PARTICLE: u32 = 24;

/// HOW FAR A FLAME'S OR A PUFF'S MIDDLE STANDS TOWARD THE EYE, as a share of
/// its half width: the card is a shallow pyramid, its outline where it was.
/// Seen with two eyes a flat sheet is a picture of a flame, every sheet of a
/// fire at its own depth like cards in a rack; domed, each has a body
/// (fire research R6). Vertex work only: 24 vertices a particle where there
/// were 18.
const CONVEX: f32 = 0.35;

fn preset(kind: EffectKind) -> Preset {
    match kind {
        EffectKind::Fire => Preset {
            // Sheets of flame standing on the bed, each playing a stretch of
            // the flipbook and fading in and out over it: the flames' motion
            // is the book's, the base stays on the fuel.
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
        // One bed, drawn whole (`coals_instance`); only its look is here.
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
        // A life is how long the eye keeps hold of one mote, not how long
        // it is in the air: it is noticed as it tumbles into the light, held
        // a few seconds and lost, while others are found elsewhere. Its
        // motion is the room's air (`air_drift`) plus its own jitter and a
        // slow settling; `swirl` is that jitter's reach and pace.
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

/// One particle as the GPU draws it: its cutout, fanned from
/// `vertex_index`. 96 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct EffectInstance {
    /// Centre (player frame) and the quad's half size.
    pub centre: [f32; 4],
    /// Smoke: the room's light and its weaker lamps' on it, times its colour,
    /// exposed; the screened kinds: the light it gives or scatters, exposed.
    /// Alpha: its opacity (over) or its strength (screened).
    pub colour: [f32; 4],
    /// Smoke: its strongest lamp's light on it, times its colour, exposed;
    /// a bed of coals: the light on its char, exposed. w: 1 to stand upright
    /// (a flame), 2 to lie flat across `axis` (coals), 0 to face the eye.
    pub light: [f32; 4],
    /// Unit direction toward that lamp (player frame); w the streak's length.
    pub light_dir: [f32; 4],
    /// Unit direction of its motion, for a streak, or the normal it lies
    /// across; w its soft fade (metres).
    pub axis: [f32; 4],
    /// Its spin (radians), its texture layer with the fraction toward the
    /// next frame, how much taller than wide an upright one is drawn (a
    /// flame, negative mirrored; 0 is square) or a bed's clock, and how it
    /// is shaded: 1 six-way smoke, 2 a flame, 3 a bed of coals, 0 plain.
    pub params: [f32; 4],
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
}

/// A frame's particles: those drawn over what is behind them, back to front,
/// then the screened ones. One buffer, `over` first.
#[derive(Default)]
pub struct EffectFrame {
    pub instances: Vec<EffectInstance>,
    pub over: u32,
    pub screened: u32,
}

/// What lights the particles and where they are seen from.
pub struct Surroundings<'a> {
    /// The head, in the player's frame: what the particles face and are
    /// sorted from.
    pub head: Vec3,
    /// The player's frame: a world point `p` is `yaw_inv * (p - offset)` in it.
    pub offset: Vec3,
    pub yaw_inv: Quat,
    /// The frame's lights, in the player's frame.
    pub lights: &'a [Light],
    /// Whether light `i` of `lights` may light an emitter's particles.
    pub reaches: &'a dyn Fn(&EffectEmitter, usize) -> bool,
    /// The light round a WORLD point from every direction, its mean radiance,
    /// linear: the room's baked light.
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
    id.bytes().fold(0xcbf29ce484222325u64, |h, b| (h ^ b as u64).wrapping_mul(0x100000001b3))
}

/// Twelve uniform numbers in [0, 1) for one incarnation of one slot.
fn uniforms(seed: u64) -> [f32; 12] {
    let mut out = [0.0; 12];
    let mut h = seed;
    for v in out.iter_mut() {
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

/// A direction within `spread_deg` of `forward`, from two uniforms, even over
/// the cap's area.
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

/// Where and how fast, `age` seconds after leaving `p0` at `v0`, under
/// `accel` with linear drag `k`: exact.
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

/// The light a lamp gives a point, as the lights block's falloff and cone
/// have it at a pixel with no averaging, unshadowed: (rgb, direction toward
/// the lamp). Suns are left out: indoors their light would come through the
/// walls, and a particle has no mask to say so.
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

/// How much more than an even scatterer a particle sends toward the eye of
/// light arriving along `travel` (from the lamp), for asymmetry `g`; capped,
/// since the six-way frames already hold the puff's own forward glow.
fn henyey_greenstein(g: f32, travel: Vec3, to_eye: Vec3) -> f32 {
    let cos_t = travel.dot(to_eye);
    let g2 = g * g;
    let d = (1.0 + g2 - 2.0 * g * cos_t).max(1e-4);
    ((1.0 - g2) / (d * d.sqrt())).min(12.0)
}

/// Smooth noise along one axis, in [-1, 1]: a random slope at each whole
/// number, eased between them. Never repeats, unlike a sum of sines.
fn noise1(x: f32, seed: u64) -> f32 {
    let i = x.floor();
    let f = x - i;
    let slope = |k: f32| (hash64(seed ^ (k as i64 as u64).wrapping_mul(0x9e3779b97f4a7c15)) >> 40) as f32 / (1u64 << 23) as f32 - 1.0;
    let a = slope(i) * f;
    let b = slope(i + 1.0) * (f - 1.0);
    let s = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    2.0 * lerp(a, b, s)
}

/// [`noise1`] along a long clock: `x` in f64, so hours of display time do not
/// step it (in f32 a clock past a day moves in hundredths of a second).
fn noise1d(x: f64, seed: u64) -> f32 {
    let i = x.floor();
    let f = (x - i) as f32;
    let slope = |k: f64| (hash64(seed ^ (k as i64 as u64).wrapping_mul(0x9e3779b97f4a7c15)) >> 40) as f32 / (1u64 << 23) as f32 - 1.0;
    let a = slope(i) * f;
    let b = slope(i + 1.0) * (f - 1.0);
    let s = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    2.0 * lerp(a, b, s)
}

/// One slow wave of the room's air: it carries the air ACROSS its wave
/// vector, so the flow it makes neither gathers nor thins the dust
/// (divergence-free), and neighbouring motes ride it together.
#[derive(Clone, Copy)]
struct AirWave {
    k: Vec3,
    across: Vec3,
    reach: f32,
    w: f32,
    phase: f32,
}

/// THE ROOM'S AIR round a dust emitter: three broad drifts (1.5-4 m waves,
/// up to about 2 cm/s) and three smaller eddies (0.4-0.8 m), each a shear
/// wave of its own direction and pace, so the flow wanders and never
/// repeats in any time anyone watches. Still indoor air moves a few
/// centimetres a second; a beam of sunlight shows it.
fn air_waves(seed: u64) -> [AirWave; 6] {
    std::array::from_fn(|i| {
        let u = uniforms(hash64(seed ^ (i as u64 + 1).wrapping_mul(0xd6e8feb86659fd93)));
        let broad = i < 3;
        let wavelength = if broad { lerp(1.5, 4.0, u[0]) } else { lerp(0.4, 0.8, u[0]) };
        // Mostly sideways: a room's air turns over slowly in height.
        let dir = cone_direction(Vec3::Y, 180.0, u[1], u[2]);
        let dir = (dir * Vec3::new(1.0, 0.5, 1.0)).normalize_or(Vec3::X);
        let k = dir * (std::f32::consts::TAU / wavelength);
        let across = dir.cross(cone_direction(Vec3::Y, 180.0, u[3], u[4])).normalize_or(dir.any_orthonormal_vector());
        let (reach, w) = if broad { (lerp(0.05, 0.09, u[5]), lerp(0.12, 0.25, u[6])) } else { (lerp(0.008, 0.015, u[5]), lerp(0.5, 1.0, u[6])) };
        AirWave { k, across, reach, w, phase: u[7] * std::f32::consts::TAU }
    })
}

/// How far the air at `p` has carried what was there at time 0, by `time`
/// (seconds); a mote noticed at `since` has moved by this at `time` minus it
/// at `since`. Read at where it was noticed: its whole trip is a small part
/// of the shortest wave.
fn air_drift(waves: &[AirWave; 6], p: Vec3, time: f64) -> Vec3 {
    waves.iter().fold(Vec3::ZERO, |sum, a| {
        let clock = (a.w as f64 * time).rem_euclid(std::f64::consts::TAU) as f32;
        sum + a.across * (a.reach * (a.k.dot(p) + clock + a.phase).sin())
    })
}

/// The farthest `air_drift` can carry a mote between two times.
fn air_reach(waves: &[AirWave; 6]) -> f32 {
    2.0 * waves.iter().map(|a| a.reach).sum::<f32>()
}

/// A mote's brightness as it tumbles, mean 1: a flat flake sends its light
/// toward the eye in a flash as it turns through the mirror angle and little
/// otherwise, so a mote is lost while it shows its edge and found again as
/// it turns -- or, as here, another is found first. `rate` turns a second.
fn mote_glint(age: f32, rate: f32, seed: u64) -> f32 {
    // The noise rarely leaves +-0.4; spread over the whole turn.
    let s = smoothstep(-0.35, 0.35, noise1(age * rate, seed));
    (GLINT_FLOOR + (1.0 - GLINT_FLOOR) * s * s * s) / GLINT_MEAN
}

/// A flame's radiance at its hottest, in the lights block's units: a lamp's
/// bulb is its intensity over `LAMP_RADIUS` squared, 1,200 for the hallway
/// sconce; a wood fire's flames are some thirty times dimmer per area than a
/// frosted bulb.
pub const FIRE_RADIANCE: f32 = 40.0;
pub const EMBER_RADIANCE: f32 = 60.0;

/// How much of what is behind it a flame covers, drawn over it: see
/// `fs_over`.
pub const FLAME_COVER: f32 = 0.8;

/// How soft an eroding flame's edge is, in its depth (`fire_frame`'s green):
/// sharp enough to tear, soft enough not to alias (fire research R2).
pub const FLAME_EROSION: f32 = 0.15;
/// A dying sheet's root goes with its tips: its depth is taken as nothing at
/// its root, all of it from `LIFT_TO` of its height up, so what is left last
/// is off the fuel (`fs_over`).
const LIFT_FROM: f32 = 0.02;
const LIFT_TO: f32 = 0.3;
/// How much of its heat a sheet has lost by the time it is eroded away: a
/// flamelet burning out cools to deep red.
const BURN_OUT: f32 = 0.45;

/// A flame's ROOT glows blue as well -- CH* and C2* radicals where no soot has
/// formed yet -- this share of its hottest light, in blue (0.05, 0.12, 0.6),
/// added after the cover (fire research R9). OFF: over a floor lit orange by
/// the fire, 0.04 drew a purple seam where the flames meet it and 0.02 still
/// tinted the rim purple (hall renders, 2026-10-08). The root mask (blue in
/// the book) and the shader path stay, for a look on the headset over a dark
/// bed of logs.
pub const BLUE_ROOT: f32 = 0.0;

/// Where a fire's hottest part meets the tone curve once the eye has adapted
/// to it: exposed past this, its light is scaled down as one (see
/// `tonemap::own_light_scale`). Its hottest core then shows pale yellow, its
/// body orange, its edges deep red (`flame_colour`) -- as photographs of a
/// fire show it, and not one flat white. Every hue past about 1 exposed
/// washes toward cream through the curve, so only the core is put there.
pub const FIRE_LEVEL: f32 = 4.0;

/// A flame sheet's height over its width.
const FLAME_STRETCH: f32 = 1.9;
/// Where a flame frame's base sits up its height (0 the bottom row), and how
/// fast its gas rises from there: `FLAME_W0` heights a second at the base and
/// `FLAME_W1` more for each height climbed -- tongues stretch as they rise.
/// The shader follows the same rise between two frames (`fx_flame_v`).
const FLAME_BASE: f32 = 0.04;
const FLAME_W0: f32 = 0.35;
const FLAME_W1: f32 = 2.4;

/// A flame's light by its heat (0-1), relative to its hottest: black through
/// deep red, orange and amber to pale yellow, brightening steeply -- the
/// shader's `fx_flame`, which this is the twin of.
pub fn flame_colour(heat: f32) -> Vec3 {
    let t = heat.clamp(0.0, 1.0);
    let bright = 0.03 + 0.97 * t.powf(1.4);
    Vec3::new(1.0, 0.02 + 0.5 * t * t, 0.02 * t + 0.08 * t * t * t * t) * bright
}

/// Where the gas at height `v` of a flame frame (0 its bottom row, 1 its top)
/// was `dt` seconds before (after, for a negative `dt`), as it rises at
/// `FLAME_W0 + FLAME_W1 h` heights a second: `h` above the base goes to
/// `(h + c) e^(-W1 dt) - c`, `c = W0 / W1`. At or below the base, it stays.
fn flame_v(v: f32, dt: f32) -> f32 {
    let h = (v - FLAME_BASE) / (1.0 - FLAME_BASE);
    if h <= 0.0 {
        return v;
    }
    let c = FLAME_W0 / FLAME_W1;
    let moved = ((h + c) * (-FLAME_W1 * dt).exp() - c).max(0.0);
    FLAME_BASE + moved * (1.0 - FLAME_BASE)
}

/// THE FIRE'S PACE: book seconds a sheet plays a second, at scale 1. Its gas
/// rises through the book at `FLAME_W0 + FLAME_W1 h` heights a book second;
/// played at this pace a tongue crosses a campfire's half-metre flame in
/// about 0.65 s. McCaffrey's centreline gas moves 2-3 m/s 10-25 cm up, and
/// what the eye follows rides it at half that or less: 0.4-0.7 s a crossing
/// (fire research G12, R12). At 1 a crossing took 0.86 s, a fire seen
/// slowed.
const FLAME_PACE: f32 = 1.3;

/// The bed a fire's sheets stand on, its radius at scale 1: the preset's.
const FIRE_BED: f32 = 0.15;

/// PUFFING. A buoyant flame sheds a ring vortex at its base about `f =
/// PUFF_K / sqrt(D)` times a second, for a fire `D` metres across: k is
/// 1.68 across many fires (Malalasekera et al.), 1.33 (USTC), 1.5 (Cetegen and
/// Ahmed) -- 2.7 Hz for this 0.3 m bed at scale 1, 1.2 Hz for a 1.5 m
/// bonfire. Each bulge swells as it rises, a neck pinches in under it, and
/// above the flame's continuous zone the neck cuts through and the bulge tears
/// off as a flamelet: the rhythm the eye reads as fire, and the beat its light
/// pulses at (fire research 1.2, R1, R3).
const PUFF_K: f32 = 1.5;
/// How much a bulge widens a sheet, as a share of its width, and how deep its
/// neck cuts into the flame's body, as `fire_frame` bakes them.
const PUFF_SWELL: f32 = 0.25;
const PUFF_PINCH: f32 = 0.55;
/// How far a fire's beat wanders, in cycles: puffing is periodic, not a
/// clock -- a spectrum peaked at `f`, not a line.
const PUFF_WANDER: f32 = 0.3;

/// The book's puffs a book second: the fire's at scale 1, played at
/// `FLAME_PACE`.
fn puff_book_hz() -> f32 {
    PUFF_K / (2.0 * FIRE_BED).sqrt() / FLAME_PACE
}

/// SECONDS AT A FIRE'S SIZE: how much longer everything about it takes than
/// at scale 1. Froude scaling -- buoyant flows of one shape take time as the
/// square root of their size -- so a bonfire's tongues live longer and play
/// their book slower, its puffs come slower, and its gas still rises faster in
/// metres a second (as `sqrt(size)`). Scaled linearly, as before, a bonfire
/// was a campfire on fast-forward. Its `flicker_rate` is its tempo besides.
fn fire_tempo(e: &EffectEmitter) -> f32 {
    e.scale.sqrt() / e.variation.clamped().flicker_rate
}

/// A fire's puffs a second: `PUFF_K / sqrt(D)`, times its `flicker_rate`.
fn fire_puff_hz(e: &EffectEmitter) -> f32 {
    puff_book_hz() * FLAME_PACE / fire_tempo(e)
}

/// WHERE A FIRE IS IN ITS PUFFING at `time`, in cycles 0-1: its beat, wandering
/// slowly off it. Its sheets play their books in step with it, and its
/// light pulses with them.
fn fire_phase(e: &EffectEmitter, time: f64) -> f32 {
    let f = fire_puff_hz(e) as f64;
    let wander = PUFF_WANDER * noise1d(time * f * 0.25, emitter_seed(e) ^ 0x7f4a_7c15_9e37_79b9);
    ((time * f).rem_euclid(1.0) as f32 + wander).rem_euclid(1.0)
}

/// One sheet of a fire at one moment: what `simulate_with` draws and
/// `fire_glow` adds up. In the WORLD's frame.
struct FlameSheet {
    /// Its spot on the bed.
    base: Vec3,
    /// Its quad's half width, and its height over that (`params.z`).
    size: f32,
    stretch: f32,
    /// Its place in the book, 0 to `FIRE_FRAMES - 1`.
    frame: f32,
    /// How far it has stood up off the fuel since it was born, 0 to 1, and
    /// how much of it still stands as it dies, 1 to 0 (see `fs_over`).
    risen: f32,
    stands: f32,
    rotation: f32,
    mirrored: bool,
    /// Its uniform for brightness and temperature.
    heat: f32,
}

/// A FIRE'S SHEETS at `time`: standing on their spots of the bed, their base a
/// little into the fuel -- larger toward the middle, where a fire burns
/// tallest, and each of its own height -- each playing its stretch of the book
/// (`fire_frame`) at about the book's own pace, in step with the fire's beat
/// (`fire_phase`): a stretch begun so the book's puffing is at the fire's
/// halfway through its life. (Out of step, every sheet puffed on its own and
/// the fire as a whole only boiled.) Froude-scaled (`fire_tempo`); a bigger
/// fire's sheets are narrower for their height, `scale^0.75` wide and `scale`
/// tall, and live longer, so a bonfire is many tongues where a campfire is a
/// few: sheets alive go as `sqrt(scale)`, layers over a pixel as
/// `scale^0.25`.
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
    let period = (slots as f64 / rate as f64).max(l1 as f64);
    let base = emitter_seed(e);
    let up = e.direction.try_normalize().unwrap_or(Vec3::Y);
    let side = up.any_orthonormal_vector();
    let wide = e.scale.powf(-0.25);
    // Book frames a puff.
    let per = FIRE_FRAMES as f32 / FLAME_SECONDS / puff_book_hz();
    let last = (FIRE_FRAMES - 1) as f32;
    let mut out = Vec::with_capacity(slots);
    for i in 0..slots {
        let since = time - i as f64 / rate as f64;
        let n = (since / period).floor();
        let age = (since - n * period) as f32;
        let u = uniforms(hash64(base ^ (i as u64).wrapping_mul(0x632be59bd9b4e019) ^ (n as i64 as u64).wrapping_mul(0x8cb92ba72f3d8dd7)));
        let life = lerp(l0, l1, u[0]);
        if age >= life {
            continue;
        }
        let t = age / life;
        let r = p.bed * e.scale * u[4].sqrt();
        let phi = u[5] * std::f32::consts::TAU;
        let spot = e.position + (side * phi.cos() + up.cross(side) * phi.sin()) * r;
        let size = lerp(p.size.0, p.size.1, smoothstep(0.0, 1.0, t)) * e.scale * size_spread(u[8], &v) * wide * (1.15 - 0.45 * u[4].sqrt());
        let tall = if v.size_variation == 1.0 { 0.85 + 0.35 * u[7] } else { (1.025 + (u[7] - 0.5) * 0.35 * v.size_variation).max(0.2) };
        let spin_dir = if u[10] < 0.5 { -1.0 } else { 1.0 };
        let pace_of = if v.speed_variation == 1.0 { 0.85 + 0.3 * u[1] } else { (1.0 + (u[1] - 0.5) * 0.3 * v.speed_variation).max(0.1) };
        let pace = FIRE_FRAMES as f32 / FLAME_SECONDS * FLAME_PACE * pace_of / tempo;
        let range = (last - life * pace).max(0.0);
        let phase = fire_phase(e, time - age as f64 + 0.5 * life as f64);
        let want = (phase * per - 0.5 * life * pace).rem_euclid(per);
        let mut start = want + ((u[6] * range - want) / per).round() * per;
        if start > range {
            start -= per;
        }
        if start < 0.0 {
            start += per;
        }
        out.push(FlameSheet {
            base: spot,
            size,
            stretch: FLAME_STRETCH * tall / wide,
            frame: (start.clamp(0.0, range) + age * pace).min(last),
            risen: smoothstep(0.0, 0.25, t),
            stands: 1.0 - smoothstep(0.6, 1.0, t),
            rotation: (2.0 * u[9] - 1.0) * (p.spin.1 * v.turbulence) + p.spin.0 * age * spin_dir,
            mirrored: u[3] < 0.5,
            heat: u[11],
        });
    }
    out
}

/// THE FLAME A BOOK FRAME HOLDS: each frame's cover (alpha) summed, over the
/// book's mean -- puffing and tearing off, a frame holds more or less of it.
/// Measured from `fire_frame` (`print_the_flame_tables`) and held to it by
/// `the_flame_tables_are_the_books`; the editor's port has the same numbers.
const FLAME_AREA: [f32; FIRE_FRAMES as usize] = [
    0.9271, 0.8415, 0.8425, 0.8990, 0.9877, 1.0703, 1.1098, 1.0776, 1.0018, 0.9146, 0.8234, 0.8473, 0.9395, 1.0476, 1.1368, 1.1838,
    1.1488, 1.1276, 1.0433, 0.9555, 0.8993, 0.8743, 0.8739, 0.9395, 1.0028, 1.0094, 1.0443, 1.1089, 1.0977, 1.0181, 0.9195, 0.9046,
    0.9783, 1.0576, 1.1141, 1.1200, 1.0657, 1.0794, 1.0408, 0.9906, 0.9489, 0.9382, 1.0067, 1.0420, 1.0302, 1.0040, 0.9796, 0.9859,
];
/// How much of a sheet's cover stands as it is eroded (`fs_over`), by how much
/// of it stands (`FlameSheet::stands`) at 0, 1/8 .. 1: the book's mean.
const EROSION_COVER: [f32; 9] = [0.0014, 0.0114, 0.0565, 0.1624, 0.3356, 0.5441, 0.7620, 0.9568, 1.0];
/// The mean of `fire_glow`'s sum at scale 1, rate 1: what makes it 1.
const FLAME_AREA_MEAN: f32 = 0.183015;

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

/// How much of a sheet's cover has stood up as it is born, by how far
/// (`FlameSheet::risen`) at 0, 1/8 .. 1: the book's mean.
const RISE_COVER: [f32; 9] = [0.0, 0.0033, 0.1533, 0.4406, 0.6905, 0.8668, 0.9673, 0.9997, 1.0];

fn rise_cover(risen: f32) -> f32 {
    let f = risen.clamp(0.0, 1.0) * 8.0;
    let i = (f.floor() as usize).min(7);
    lerp(RISE_COVER[i], RISE_COVER[i + 1], f - i as f32)
}

/// How far up its frame a sheet `risen` (0-1) has stood, and how soft that
/// front is, in the frame's height: from under its root to over its tip, the
/// deepest of its body leading (`fs_over`).
const RISE_FROM: f32 = -0.3;
const RISE_TO: f32 = 1.15;
const RISE_SOFT: f32 = 0.12;
const RISE_LEAD: f32 = 0.25;

/// `FLAME_AREA` and `EROSION_COVER` as the book has them, worked out from its
/// frames: each frame's cover over the book's mean, and the share of the
/// book's cover that stands as a sheet erodes, as `fs_over` erodes it.
#[cfg(test)]
fn flame_tables() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let frames: Vec<Vec<u8>> = (0..FIRE_FRAMES).map(fire_frame).collect();
    let cover: Vec<f32> = frames.iter().map(|f| f.chunks(4).map(|t| t[3] as f32 / 255.0).sum::<f32>()).collect();
    let mean = cover.iter().sum::<f32>() / cover.len() as f32;
    let area = cover.iter().map(|c| c / mean).collect();
    let n = ATLAS_SIZE as usize;
    let erosion = (0..9)
        .map(|k| {
            let gone = 1.0 - k as f32 / 8.0;
            let (mut stands, mut all) = (0.0f32, 0.0f32);
            for f in &frames {
                for (i, t) in f.chunks(4).enumerate() {
                    let a = t[3] as f32 / 255.0;
                    let v = 1.0 - ((i / n) as f32 + 0.5) / n as f32;
                    stands += a * smoothstep(gone - FLAME_EROSION, gone, t[1] as f32 / 255.0 * smoothstep(LIFT_FROM, LIFT_TO, v));
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
                    let a = t[3] as f32 / 255.0;
                    let v = 1.0 - ((i / n) as f32 + 0.5) / n as f32;
                    stands += a * (1.0 - smoothstep(front - RISE_SOFT, front, v - RISE_LEAD * t[1] as f32 / 255.0));
                    all += a;
                }
            }
            stands / all
        })
        .collect();
    (area, erosion, rise)
}

/// How much of the slow swell and ebb of a fire's sheets as they hand over
/// reaches its light, as a power of it. The sheets are how a fire is drawn,
/// not how one burns: whole, their hand-overs swung the light at about 1 Hz,
/// over the puffing it should peak at.
const HANDOVER_SHARE: f32 = 0.35;

/// HOW MUCH FLAME A FIRE SHOWS at `time`, 1 on average: its sheets' cover as
/// they stand -- each its book frame's flame (`FLAME_AREA`), eroded as far as
/// it is (`EROSION_COVER`), times its size. Its light follows this
/// (`fire_light`), so the room brightens as the flames swell and dims as
/// flamelets tear off, at the beat the eye sees: what makes light and flames
/// read as one fire (fire research R1; Niagara's lights take the particles'
/// values for the same reason). The puffing whole -- the sheets' frames'
/// flame over what their frames would hold on average -- and their total over
/// its long-run mean to `HANDOVER_SHARE`.
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

/// A bed of coals' quad half size at scale 1, metres: its coals some 40 cm
/// across.
const COALS_HALF: f32 = 0.24;
/// A bed's hottest coal, as a share of its fire's hottest flame: coals glow
/// orange, well under the flames that rise from them.
const COALS_GLOW: f32 = 0.6;

/// A BED OF COALS: one quad lying on the floor across the emitter's up, spun
/// by its seed, drawn before every flame and puff -- they stand on it or rise
/// from it. Its coals glow as the eye adapted to its fire sees the fire, each
/// flickering on its own (`fs_over`), and all together breathing a little with
/// the fire burning on them (`coals_breath`, its `fire_glow`: the flames' light on
/// the bed and the draught they pull through it; fire research R13); its char
/// and ash are lit like a floor, by the room and its lamps -- its fire's own
/// light among them, seen as the flames are (`own`). Not faded into the floor
/// it lies on, or it would fade away whole. Under the flames it hides the
/// floor their light burns brightest on, as a fire's fuel does.
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
        // w: its coals' flicker over the preset's, less one (`fs_over`).
        light_dir: [0.0, 1.0, 0.0, v.flicker - 1.0],
        axis: [normal.x, normal.y, normal.z, 0.0],
        params: [spin, COALS_LAYER as f32, ((time * v.flicker_rate as f64) % 3600.0) as f32, 3.0],
    })
}

/// How much a bed of coals breathes with the fire on it: a quarter of that
/// fire's swell and ebb (`fire_glow`), times its coals' flicker amount; 1 for
/// a bed with no fire on it (one whose middle is within half its size of the
/// bed's).
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

/// How far the eye adapted to a fire scales its light down: the flames' and
/// embers' radiance, and the fire's own light on its smoke. 1 for a fire too
/// small or too far off to adapt to.
fn fire_adaptation(e: &EffectEmitter, at: &Surroundings) -> f32 {
    let distance = (at.yaw_inv * (e.position - at.offset) - at.head).length();
    let adapted = super::tonemap::bulb_adaptation(0.15 * e.scale, distance, 1.0);
    super::tonemap::own_light_scale(at.exposure, FIRE_RADIANCE, FIRE_LEVEL, adapted)
}

/// THE FRAME'S PARTICLES at `time` seconds. See the module notes.
pub fn simulate(emitters: &[EffectEmitter], time: f64, at: &Surroundings) -> EffectFrame {
    simulate_with(emitters, &[], time, at)
}

/// [`simulate`], with the splashes still in the air at `time` (`splash`).
pub fn simulate_with(emitters: &[EffectEmitter], splashes: &[Splash], time: f64, at: &Surroundings) -> EffectFrame {
    let mut over: Vec<(f32, EffectInstance)> = Vec::new();
    let mut screened: Vec<EffectInstance> = Vec::new();
    let player = |p: Vec3| at.yaw_inv * (p - at.offset);
    let head_world = at.yaw_inv.inverse() * at.head + at.offset;
    // A FIRE'S OWN LIGHT on smoke and motes is seen as its flames are, by the
    // eye adapted to the fire: lit at the room's exposure, the smoke over a
    // fire burned white above flames a hundred times dimmer than it is. Found
    // among the frame's lights near where `fire_light` hangs it -- it sways
    // with the flames, and the client may hang it on another clock's moment.
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
    for e in emitters {
        match e.kind {
            EffectKind::Coals => {
                // Drawn before all it lies under: as the farthest of all.
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
                EffectKind::Dust => p.rate * e.volume() / (0.5 * (p.life.0 + p.life.1)),
                _ => p.rate,
            };
        if !(rate > 0.0) || !(e.scale > 0.0) {
            continue;
        }
        let lives = life_range(&p, &v);
        let speeds = speed_range(&p, &v);
        let slots = ((rate * lives.1).ceil() as usize).clamp(1, MAX_SLOTS);
        // Each slot respawns every `period`: never sooner than its longest
        // life, and the emitter as a whole at `rate`.
        let period = (slots as f64 / rate as f64).max(lives.1 as f64);
        let base = emitter_seed(e);
        let tint = Vec3::from(e.tint);
        let ambient = (at.ambient)(e.position);
        let lamps: Vec<usize> = (0..at.lights.len()).filter(|&i| (at.reaches)(e, i)).collect();
        // Embers as the eye adapted to the fire sees them.
        let glow = if e.kind == EffectKind::Embers { at.exposure * fire_adaptation(e, at) } else { at.exposure };
        let up = e.direction.try_normalize().unwrap_or(Vec3::Y);
        let air = (e.kind == EffectKind::Dust).then(|| air_waves(base));
        for i in 0..slots {
            let since = time - i as f64 / rate as f64;
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
                // A disc across the emitter's direction.
                let r = p.bed * e.scale * u[4].sqrt();
                let side = up.any_orthonormal_vector();
                let phi = u[5] * std::f32::consts::TAU;
                e.position + (side * phi.cos() + up.cross(side) * phi.sin()) * r
            };
            let dir = cone_direction(up, p.spread_deg, u[2], u[3]);
            let (mut pos, vel) = ballistic(p0, dir * speed, p.accel, p.drag, age);
            if let Some(air) = &air {
                // Carried by the room's air with its neighbours, and jostled
                // on its own by the smallest eddies -- a wander that never
                // retraces itself, where a sum of sines drew loops.
                pos += (air_drift(air, p0, time) - air_drift(air, p0, time - age as f64)) * v.turbulence;
                let seed = base ^ (i as u64).wrapping_mul(0x632be59bd9b4e019) ^ (n as i64 as u64);
                let jitter = age * p.swirl.1;
                pos += Vec3::new(noise1(jitter, seed), 0.6 * noise1(jitter, seed ^ 1), noise1(jitter, seed ^ 2)) * (p.swirl.0 * v.turbulence);
            } else {
                let swirl_r = p.swirl.0 * v.turbulence * t * e.scale;
                let swirl_a = p.swirl.1 * age + u[7] * std::f32::consts::TAU;
                pos += Vec3::new(swirl_a.cos(), 0.0, swirl_a.sin()) * swirl_r;
            }
            let mut size = if e.kind == EffectKind::Dust {
                // Many faint, few bright (`DUST_LIGHT_SIZE`); spread over
                // fewer or more octaves about its middle by `size_variation`.
                let k = if v.size_variation == 1.0 { u[11] * u[11] } else { 1.0 / 3.0 + (u[11] * u[11] - 1.0 / 3.0) * v.size_variation };
                p.size.0 * (p.size.1 / p.size.0).powf(k) * e.scale
            } else {
                lerp(p.size.0, p.size.1, smoothstep(0.0, 1.0, t)) * e.scale * size_spread(u[8], &v)
            };
            // UNDER A CEILING smoke cannot keep rising: what would have
            // risen past it is turned sideways and spreads out beneath it,
            // flattening and widening -- the layer that gathers under the
            // roof of a cave or a hall. Embers stop against it.
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
                        }
                    }
                    EffectKind::Embers => pos.y = pos.y.min(top - 0.02),
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
            let mut inst = EffectInstance {
                centre: [centre.x, centre.y, centre.z, size],
                colour: [0.0; 4],
                light: [0.0; 4],
                light_dir: [0.0, 1.0, 0.0, streak],
                axis: [axis.x, axis.y, axis.z, p.soft * e.scale.sqrt()],
                params: [rotation, frame, 0.0, 0.0],
            };
            match e.kind {
                EffectKind::Embers => {
                    inst.colour = ember_colour(&v, tint, glow, age, t, u[11]);
                }
                EffectKind::Dust => {
                    // Only the light it scatters, each lamp's by its own
                    // angle to the eye; the room's even light hardly shows
                    // it (`DUST_AMBIENT_SEEN`).
                    let mut scattered = ambient * DUST_AMBIENT_SEEN;
                    for &li in &lamps {
                        if let Some((c, d)) = lamp_at(&at.lights[li], centre) {
                            scattered += c * (own(&at.lights[li]) * henyey_greenstein(p.phase_g, -d, to_eye));
                        }
                    }
                    // Seen as the eye holds it: found as it turns into the
                    // light, flashing and dimming as it tumbles, then lost;
                    // and only where the eye could focus on it.
                    let glint = mote_glint(age, lerp(0.35, 1.3, u[7]), id_hash("glint") ^ u[8].to_bits() as u64);
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
                // Drawn above, each its own way.
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
    // Back to front, so each puff lands over what is behind it.
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

/// A FIRE'S SHEETS as the GPU draws them (`fire_sheets`): upright, taller than
/// wide, half of them mirrored; their light at its hottest, exposed as the eye
/// adapted to the fire sees it, and the texture's heat colours it
/// (`flame_colour`). As it is born it STANDS UP off the fuel, a front rising
/// through its frame with its body's core leading (`light.x`, how far); as it
/// dies the shader ERODES it from its thin tips and edges in toward its root
/// (alpha, how much still stands). A fade dimmed the whole sheet at once, see-
/// through as no flame is (fire research R2); grown out of its root by depth,
/// a newborn sheet was a pale ball on the fuel.
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
            },
        ));
    }
}

/// A spark's light and strength: hot orange, flickering as it tumbles,
/// fading as it cools.
fn ember_colour(v: &Variation, tint: Vec3, glow: f32, age: f32, t: f32, heat: f32) -> [f32; 4] {
    let flicker = 0.7 + 0.3 * (v.flicker * (age * 23.0 * v.flicker_rate + heat * 40.0).sin() + (1.0 - v.flicker));
    let c = Vec3::new(1.0, 0.45, 0.12) * tint * (EMBER_RADIANCE * glow * flicker * v.intensity) * temperature(heat, v);
    [c.x, c.y, c.z, (1.0 - t * t).max(0.0)]
}

/// EMBER POPS, a second, on average, at rate 1: a pocket of sap or gas
/// bursting, a log settling. A fire's sparks come in bursts as well as a
/// thin stream, and VR players notice them (fire research R10).
const EMBER_POPS: f32 = 0.3;
/// The clock is cut into bins this long, each holding a pop or not.
const EMBER_POP_BIN: f64 = 0.5;
/// How fast the plume over a fire rises where it carries a pop's sparks,
/// m/s at scale 1 (as `sqrt(scale)`): they are thrown up into it and drag
/// toward its rise, rather than flying free and falling back.
const EMBER_PLUME: f32 = 0.9;

/// AN EMBERS EMITTER'S POPS still alight at `time`: in each `EMBER_POP_BIN`
/// of the clock a pop with the chance its rate gives, at a moment within it,
/// of 6-16 sparks thrown up at 1.8-3 m/s within 15 degrees of its up, each a
/// function of the pop and the time like every particle here.
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
    let first = ((time - lives.1 as f64) / EMBER_POP_BIN).floor() as i64;
    let last = (time / EMBER_POP_BIN).floor() as i64;
    for bin in first..=last {
        let pop = hash64(base ^ (bin as u64).wrapping_mul(0x8cb92ba72f3d8dd7));
        let b = uniforms(pop);
        if b[0] >= chance {
            continue;
        }
        let born = (bin as f64 + b[1] as f64) * EMBER_POP_BIN;
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
            // A few hundredths of a second apart, so a pop is a spray, not one point.
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
            if let Some(top) = e.ceiling {
                pos.y = pos.y.min(top - 0.02);
            }
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
            });
        }
    }
}

/// SOMETHING STRIKING WATER: a foot, a hand, a thrown thing. An event, not an
/// emitter -- the client sees it happen (`quest_app::splashes`) and every
/// particle it throws is a function of it and the time since, as an
/// emitter's are of the clock, so it needs no state either. In the WORLD's
/// frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Splash {
    /// Where it struck the still surface.
    pub position: Vec3,
    /// When, on the clock it is simulated with.
    pub born: f64,
    /// How fast what struck it was moving, m/s.
    pub speed: f32,
    /// How big what struck it was: its radius at the surface, metres.
    pub size: f32,
    /// How much of the sun reaches it, 0..1: the client's ray toward the sun.
    pub sunlit: f32,
    /// The still water's depth there, metres: how far the shore's swash
    /// lifts it (`XrRenderer::add_splash`).
    pub depth: f32,
    pub seed: u64,
}

/// How long a splash lasts, seconds: its highest drop falls back by then.
pub const SPLASH_SECONDS: f32 = 1.6;

/// Water drops' scattering: clear spheres absorb nothing and send most of the
/// light that reaches them on forward, bent through them, mirroring a little
/// back -- so a splash against the sun sparkles and with the sun behind the
/// eye is fainter.
const DROP_ALBEDO: f32 = 1.0;
const DROP_PHASE_G: f32 = 0.55;
/// One frame's travel at 72-90 Hz, seconds: a falling drop is a streak.
const DROP_STREAK: f32 = 0.012;

/// How strong a splash is, about 1 for a walking step into shallow water: the
/// energy it throws, from its speed and the size of what struck.
pub fn splash_strength(speed: f32, size: f32) -> f32 {
    let v = (speed / 1.3).max(0.0);
    (v * v * (size / 0.08).max(0.0)).min(12.0)
}

/// A splash's drops and spray at `time`.
///
/// DROPS: thrown from a ring round where it struck at 40-85 degrees, each as
/// fast as a share of the strike, falling under gravity with a little drag,
/// gone when they reach the surface again. Streaked along their motion, lit
/// by the sun (when the client found it in sight) and the light round them,
/// screened like the embers. SPRAY: a few puffs of mist where it struck,
/// thinning as they spread -- drawn over, as the smoke is.
fn splash(s: &Splash, time: f64, at: &Surroundings, over: &mut Vec<(f32, EffectInstance)>, screened: &mut Vec<EffectInstance>) {
    let age = (time - s.born) as f32;
    let strength = splash_strength(s.speed, s.size);
    if !(age >= 0.0 && age < SPLASH_SECONDS) || strength <= 0.01 {
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
        // The sheet a strike throws up leaves faster than what struck: a few
        // drops up to half as fast again, most slower.
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
        let c = lit(centre, DROP_ALBEDO, DROP_PHASE_G);
        let radius = lerp(0.0015, 0.004, u[5]);
        screened.push(EffectInstance {
            centre: [centre.x, centre.y, centre.z, radius],
            colour: [c.x, c.y, c.z, 1.0 - smoothstep(0.75, 1.0, a / SPLASH_SECONDS)],
            light: [0.0; 4],
            light_dir: [0.0, 1.0, 0.0, vel.length() * DROP_STREAK],
            axis: [axis.x, axis.y, axis.z, 0.01],
            params: [0.0, EMBER_LAYER as f32, 0.0, 0.0],
        });
    }
    // THE CROWN: the body of the splash, the jets it throws up -- thousands
    // of drops too small to draw one by one, close enough together to be
    // thick: white as a cloud is, where each drop alone is clear. It rises as
    // fast as the strike throws it, sags as it falls back, and thins.
    let rise = ((1.2 * s.speed).powi(2) / (2.0 * 9.81)).clamp(0.04, 2.0);
    let crown_life = (2.0 * (2.0 * rise / 9.81).sqrt()).clamp(0.3, 1.4);
    let t = age / crown_life;
    if t < 1.0 {
        let u = uniforms(hash64(s.seed ^ 0xc80));
        let height = rise * smoothstep(0.0, 0.3, t) * (1.0 - 0.45 * smoothstep(0.45, 1.0, t));
        let half = 0.5 * height.max(s.size);
        let centre = player(s.position + Vec3::Y * (half * 0.92));
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
            },
        ));
    }
    // SPRAY: mist where it struck, rising a little and spreading.
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
        let (pos, _) = ballistic(s.position + out * s.size * 0.6, v0, Vec3::new(0.0, -2.0, 0.0), 3.0, age);
        let centre = player(pos.max(Vec3::new(f32::MIN, s.position.y + 0.02, f32::MIN)));
        let size = s.size * lerp(0.7, 1.2, u[4]) * (0.6 + 1.6 * t) * strength.sqrt().clamp(0.6, 1.8);
        // Thick, its light is scattered many times over: as bright from any
        // side, so even (g = 0), not a single drop's forward lobe.
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
            },
        ));
    }
}

/// The newest splashes' rings on the water, as [`WaterUniform::rings`]
/// takes them: world x, z, seconds since, and strength; strength 0 for none.
/// Nearest the head first when there are more than fit.
///
/// [`WaterUniform::rings`]: super::water_pipeline::WaterUniform
pub fn splash_rings<const N: usize>(splashes: &[Splash], time: f64, head_world: Vec3) -> [[f32; 4]; N] {
    let mut live: Vec<(f32, [f32; 4])> = splashes
        .iter()
        .filter_map(|s| {
            let age = (time - s.born) as f32;
            (age >= 0.0 && age < super::water_pipeline::RING_SECONDS).then(|| {
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

/// How strong a fire's light is at scale 1, in the lights block's units (the
/// brick hall's ceiling lamp is 7), and how far it reaches: the floor a metre
/// off takes about what that lamp gives it from the ceiling, the walls a dim
/// glow. Three times this burned the floor round the fire white at the brick
/// hall's exposure, which the baked probes meter without it, and with it
/// every flame drawn over that floor.
pub const FIRE_LIGHT_INTENSITY: f32 = 0.6;
pub const FIRE_LIGHT_RANGE: f32 = 4.5;

/// Whether `e` is a fire that burns: one that has a light and sheets.
fn fire_burns(e: &EffectEmitter) -> bool {
    e.kind == EffectKind::Fire && e.rate > 0.0 && e.scale > 0.0
}

/// Where a fire's light hangs when its flames are at their mean, in the WORLD's
/// frame: 0.35 of its scale up its axis. `fire_light` sways it from here.
fn fire_light_rest(e: &EffectEmitter) -> Vec3 {
    e.position + e.direction.try_normalize().unwrap_or(Vec3::Y) * (0.35 * e.scale)
}

/// How far from its rest a fire's light may be found, at scale 1 (as scale):
/// past its sway and rise, short of anything else.
const FIRE_LIGHT_REACH: f32 = 0.25;

/// A FIRE'S LIGHT on the room round it, in the player's frame: a point light
/// in its flames, as wide as they are (so the floor under them has no hot
/// spot), unshadowed -- its reach is kept inside a room rather than masked.
/// Its smoke takes it from below like any lamp. `None` for any other kind, or
/// a fire stopped. `time` in seconds, worked in f64 so a long-running clock
/// does not step it -- and the clock the effects are simulated on, so the
/// light pulses with the flames the eye sees.
///
/// IT FOLLOWS THE FLAMES: its strength goes with how much flame is in view
/// (`fire_glow`) -- swelling with each puff, dimming as a flamelet tears off
/// and as sheets hand over -- with a faint 1/f flutter over that, an octave and
/// two above the beat; its centre sways a few centimetres with the same
/// flames and rises as they swell, so the shading on walls and hands moves
/// (fire research R1, R11). It flickered with three fixed sines of its own
/// (1.1, 2.2 and 3.7 Hz), unrelated to the flames drawn and the same for a
/// bonfire as a candle.
pub fn fire_light(e: &EffectEmitter, time: f64, offset: Vec3, yaw_inv: Quat) -> Option<Light> {
    if !fire_burns(e) {
        return None;
    }
    let v = e.variation.clamped();
    let glow = fire_glow(e, time);
    let f = fire_puff_hz(e) as f64;
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

/// A FIRE'S GLARE, in the player's frame: its flames as one source of the
/// lamps' veil (`glare`), glowing from the middle of its flame body as wide as
/// that is, from every side, as bright as its light. A night fire with no veil
/// round it looked painted on (fire research R5). Its veil is the CIE young
/// eye's, as a lamp's, so it is faint: a fire is some thirty times dimmer per
/// area than a frosted bulb.
pub fn fire_glare(e: &EffectEmitter, time: f64, offset: Vec3, yaw_inv: Quat) -> Option<super::glare::GlareSource> {
    let light = fire_light(e, time, offset, yaw_inv)?;
    let up = e.direction.try_normalize().unwrap_or(Vec3::Y);
    let middle = yaw_inv * (e.position + up * (0.25 * e.scale) - offset);
    let c = light.color.to_linear();
    // One entry, from every side: none of it a bulb, all of it a lit body
    // whose spread is the flames'.
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

/// The mean radiance of the room's baked light round a world point: band 0
/// of `room_light`, which holds every lit surface the room's photographs saw.
pub fn room_ambient(descs: &[ProbeDesc], world: Vec3) -> Vec3 {
    let l = super::room_light::room_light_at(descs, world);
    Vec3::new(l[0][0], l[0][1], l[0][2]) * 0.282_095
}

/// Whether a lamp lights an emitter's particles: it hangs in the smallest
/// room box holding the emitter, or a little outside it (a lamp in a
/// doorway). Both points in the world's frame. Without rooms, every lamp
/// does. A lamp in the next room shines into this one through its doorway
/// only, and particles have no shadow mask to say where.
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

/// How much of a fire's light the floor round it sends back, for the meter:
/// stone, as the brick hall's.
const POOL_ALBEDO: f32 = 0.35;

/// How many times a photograph's bin of the same size a fire's light weighs
/// in the meter: a fire draws the eye, and the eye adapts to what it fixes
/// on. Weighed as one bin among the room's, the pool was a sliver of the
/// meter's band and the eye stayed adapted to the dim room round it.
const FIRE_METER_WEIGHT: f32 = 4.0;

/// WHAT A FIRE ADDS TO THE VIEW THE EYE ADAPTS TO, which the baked
/// photographs the meter reads never saw (`exposure::EyeAdaptation::meter_with`):
/// its flames, and the pool its light throws on the floor round its foot as
/// far as its room's box holds that floor, for a head in that room -- as
/// (direction from `head`, luminance, solid angle), in the WORLD's frame, its
/// light at its mean flicker. Metered without it, the eye stayed adapted to
/// the dim room, and the pool burned white round flames shown as the eye
/// adapted to them sees them.
pub fn meter_samples(emitters: &[EffectEmitter], head: Vec3, descs: &[ProbeDesc]) -> Vec<(Vec3, f32, f32)> {
    let size = |d: &ProbeDesc| (d.max - d.min).max(Vec3::ZERO).element_product();
    let mut out = Vec::new();
    for e in emitters {
        let Some(light) = fire_light(e, 0.0, Vec3::ZERO, Quat::IDENTITY) else { continue };
        // At its mean, hung at its rest.
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
        // Seen only from its own room, or its doorway: through a wall it is
        // nothing to adapt to.
        if room.is_some_and(|d| head.cmplt(d.min - Vec3::splat(0.5)).any() || head.cmpgt(d.max + Vec3::splat(0.5)).any()) {
            continue;
        }
        // The flames: a sheet facing the eye at the radiance of their body.
        let to = e.position + up * (0.3 * e.scale) - head;
        let d2 = to.length_squared().max(1e-4);
        out.push((to / d2.sqrt(), FIRE_RADIANCE * intensity * luminance(flame_colour(0.6)), FIRE_METER_WEIGHT * 0.24 * e.scale * e.scale / d2));
        if (head - e.position).dot(up) <= 0.0 {
            continue;
        }
        // Rings out to fifteen times the light's height, eight patches each.
        let side = up.any_orthonormal_vector();
        let across = up.cross(side);
        let height = (light.position - e.position).length();
        const EDGES: [f32; 7] = [0.0, 0.5, 1.0, 1.6, 2.4, 3.6, 5.2];
        for k in 0..EDGES.len() - 1 {
            let (r0, r1) = (EDGES[k] * height, EDGES[k + 1] * height);
            let area = std::f32::consts::PI * (r1 * r1 - r0 * r0) / 8.0;
            for s in 0..8 {
                let phi = (s as f32 + 0.5) * std::f32::consts::TAU / 8.0;
                let p = e.position + (side * phi.cos() + across * phi.sin()) * (0.5 * (r0 + r1));
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

// ---------------------------------------------------------------------------
// The textures.

fn grad(h: u64) -> Vec3 {
    // Twelve cube-edge directions, Perlin's.
    const G: [[f32; 3]; 12] = [
        [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0], [1.0, -1.0, 0.0], [-1.0, -1.0, 0.0],
        [1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 0.0, -1.0],
        [0.0, 1.0, 1.0], [0.0, -1.0, 1.0], [0.0, 1.0, -1.0], [0.0, -1.0, -1.0],
    ];
    Vec3::from(G[(h % 12) as usize])
}

/// Gradient noise in about [-1, 1], from integer lattice hashes: the same
/// value for the same point on any machine.
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
        sum += amp * noise3(q, seed.wrapping_add(o as u64 * 7919));
        q = q * 2.03 + Vec3::new(17.1, 9.3, 3.7);
        amp *= 0.5;
    }
    sum
}

/// Turbulence: the sum of each octave's noise folded to its size, which
/// rounds into the bulging cells of a cloud's surface. About 0 to 0.7, its
/// mean near 0.25.
fn billows(p: Vec3, octaves: u32, seed: u64) -> f32 {
    let mut sum = 0.0;
    let mut amp = 0.5;
    let mut q = p;
    for o in 0..octaves {
        sum += amp * noise3(q, seed.wrapping_add(o as u64 * 7919)).abs();
        q = q * 2.03 + Vec3::new(17.1, 9.3, 3.7);
        amp *= 0.5;
    }
    sum
}

/// The smoke's volume: `VOL` across and up, `VOL_DEPTH` through.
const VOL: usize = 64;
const VOL_DEPTH: usize = 48;
/// Extinction at density 1, per unit of the puff's half width: a young
/// puff's middle lets a thousandth of the light through, so it shadows
/// itself.
const SMOKE_SIGMA: f32 = 6.0;

/// The puff's lumps, centres and radii in its own space (-1 to 1 each way):
/// the same in every frame, so it grows and thins as one. Lopsided on
/// purpose -- set evenly round the middle they drew a flower -- and out to
/// about 0.86 grown, inside the cube's soft walls.
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

/// How dense the puff is at `p`, `age` (0 to 1) into its life: lumps,
/// bent out of round by a slow warp of the space they sit in, whose surface
/// bulges into billows that roll as it rises; eaten into and thinned as it
/// ages; nothing at the cube's walls.
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

/// A smoke puff `k` frames of `SMOKE_FRAMES` into its life, as two layers:
/// the light it passes to the eye from a lamp on its right, above it and in
/// front of it (red, green, blue), and from its left, below and behind; alpha
/// its opacity in both. Each is single scattering through the puff's own
/// volume, the light's way in and the eye's way out both attenuated -- a lamp
/// behind a thick puff lights its thin edges and leaves its middle dark.
/// Stored as a share of the opacity, so 1 is a sliver lit through. Row 0 is
/// the TOP.
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
    // Optical depth from each voxel out to each side, its own half included:
    // right (+x), top (+y, the rows above), front (+z, toward the eye), left,
    // bottom, back.
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
    // The eye's way in, front to back.
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
                // Where there is next to nothing, the light along the ray, so
                // filtering toward an empty texel meets a sensible value.
                light[o][s] = if alpha[o] > 1.0 / 512.0 { sum[s] / alpha[o] } else { through[s] / nz as f32 };
            }
        }
    }
    // Up to the atlas's size, bilinearly.
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

/// Frame `k` of the flame's flipbook, `FLAME_SECONDS / FIRE_FRAMES` seconds
/// apart: one sheet of flame on its fuel. Gas rises through it, faster as it
/// climbs (`FLAME_W0`, `FLAME_W1`), carrying noise keyed to the moment it left
/// the base -- so its lumps rise and stretch into tongues -- and swaying more
/// the higher it is; the sheet's body, wide at the base and tapering to a
/// point, is eaten by that noise, more toward the top, where the tongues break
/// off and die. Its bottom edge is ragged, lifting off the fuel.
///
/// IT PUFFS (`PUFF_K`): the gas that left the base on the beat swells the
/// sheet as it rises, the gas between pinches it into a neck, and above the
/// continuous zone -- the steady lower third of a flame, McCaffrey's -- the
/// neck cuts through and the bulge above tears off as a flamelet and dies.
/// Keyed, like the noise, to when the gas left the base, so a bulge rises with
/// the gas and stretches as it does. (Advected noise alone boiled but never
/// pulsed.)
///
/// Red its heat; green how deep inside the flame each texel is, which the
/// shader erodes from the outside in as a sheet dies (`fs_over`) -- its thin
/// tips and edges go first; blue its ROOT, the first centimetres off the fuel,
/// where the gas has made no soot yet: there it glows less and blue
/// (CH*, C2*) shows; alpha its cover. Row 0 is the TOP. (Tried before: one
/// smooth tongue a particle, which stacked into cones; many small rising
/// blobs, which summed to a smooth block.)
fn fire_frame(k: u32) -> Vec<u8> {
    let n = ATLAS_SIZE as usize;
    let time = k as f32 * FLAME_SECONDS / FIRE_FRAMES as f32;
    let beat_hz = puff_book_hz();
    let mut out = vec![0u8; n * n * 4];
    for y in 0..n {
        for x in 0..n {
            let px = (x as f32 + 0.5) / n as f32 * 2.0 - 1.0;
            let py = 1.0 - (y as f32 + 0.5) / n as f32 * 2.0;
            let v = (py + 1.0) * 0.5;
            let h = ((v - FLAME_BASE) / (1.0 - FLAME_BASE)).max(0.0);
            // The seconds the gas here took to rise from the base, and so the
            // moment it left it.
            let flight = (1.0 + FLAME_W1 * h / FLAME_W0).ln() / FLAME_W1;
            let born = flight - time;
            // Its puff: +1 the bulge, -1 the neck.
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
            // Nothing at the frame's sides or top: the noise reaches past the
            // body, and a lick there was cut straight by the quad's edge.
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

/// How deep in a flame's body is its deepest (green 1 in `fire_frame`): the
/// body's value at the middle of its root, about.
const FLAME_DEPTH: f32 = 0.9;

/// The cell a point falls in on a jittered grid, Worley's: the distance to the
/// nearest cell's point, to the next nearest, and the nearest's hash.
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

/// A BED OF COALS seen from above: lumps of char packed in a ragged disc,
/// glowing in the cracks between them and over many of their faces --
/// hottest in the middle, under the flames -- going to grey ash at the rim.
/// Red its heat, green the char's albedo, blue each lump's own phase of
/// flicker, alpha its cover. Row 0 is the far edge; it lies flat.
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
            // A lump glows in patches over its face, under grey skins of ash;
            // the gaps between lumps glow softly, deeper in. (Bright thin
            // gaps alone drew cracked lava.)
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

/// A round soft dot: an ember's (tight) or a mote's (soft), lit evenly.
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

/// Every layer of the effects' texture array, mip 0, RGBA8, `ATLAS_SIZE`
/// square, in `ATLAS_LAYERS` order. The smoke's frames, worked through their
/// volume, and the flame's are made on every core.
pub fn atlas_layers() -> Vec<Vec<u8>> {
    fn on_every_core<T: Send>(count: u32, make: impl Fn(u32) -> T + Sync) -> Vec<T> {
        let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).clamp(1, count.max(1) as usize);
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

/// A SPLASH'S CROWN, seen from the side: jets of water thrown up and out from
/// where it struck, tapering as they thin, arcing over as they fall, each
/// ending in the drop it breaks into, over a low dome of mist at the base.
/// White in alpha (aerated water), its base on the bottom row's middle.
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
        // Most jets near upright, a few flung wide.
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
        // Nothing at the quad's sides or top: the cutout and the fade end it.
        let edge = smoothstep(0.0, 0.04, x) * smoothstep(1.0, 0.96, x) * smoothstep(0.0, 0.04, y);
        out[i * 4..i * 4 + 4].copy_from_slice(&[255, 255, 255, ((1.0 - c) * edge * 255.0).round() as u8]);
    }
    out
}

/// The eight-sided outline round every texel of `alpha` (RGBA8, `ATLAS_SIZE`
/// square, row 0 the top) above nothing, two texels' margin besides for its
/// coarser mips: in the quad's own coordinates, -1 to 1 with y up, counter-
/// clockwise from the bottom edge's left end. Where an axis's diagonal cuts
/// nothing off, two corners meet at the box's.
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
        // Nothing visible: a point, drawn as nothing.
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

/// A flame frame's cover carried along its gas's rise for `dt` seconds (back
/// down for a negative `dt`), every height between kept: everywhere the
/// shader may read it between two frames (`fx_flame_v`). Alpha only.
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

/// Every layer's cutout, as the shader's `fx_cut` holds it: four vec4s, two
/// corners each. A flipbook frame's takes in the next frame too, which it is
/// blended toward -- a flame frame's both carried along the rise between
/// them, as the shader reads them.
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
            with.extend(smeared.iter().map(|s| s.as_slice()));
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

/// The next mip of a square RGBA8 image: a 2x2 box, alpha-weighted for the
/// colour so a transparent texel does not pull its neighbours' toward its own.
fn half_mip(src: &[u8], size: usize) -> Vec<u8> {
    let h = (size / 2).max(1);
    let mut out = vec![0u8; h * h * 4];
    for y in 0..h {
        for x in 0..h {
            let mut sum = [0.0f32; 4];
            let mut weight = 0.0;
            for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                let o = ((2 * y + dy).min(size - 1) * size + (2 * x + dx).min(size - 1)) * 4;
                let a = src[o + 3] as f32;
                for c in 0..3 {
                    sum[c] += src[o + c] as f32 * (a + 1.0);
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

/// The texture array with its mips, uploaded.
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

// ---------------------------------------------------------------------------
// The pipelines.

/// What the effects' shader needs besides the scene's camera: the head's
/// right and up and its position (every quad faces the head, so both eyes see
/// one quad), and the depth range for the soft fade. 64 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct EffectsUniform {
    pub right: [f32; 4],
    pub up: [f32; 4],
    pub head: [f32; 4],
    /// near, far, 1 when the probe pass's depth is this frame's, and an eye
    /// pixel's size at unit depth (0: no least size).
    pub depth: [f32; 4],
}

/// Group 2: the uniform, the texture array, its sampler and the cutouts.
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

/// The effects' GPU side: textures made once, a uniform and an instance
/// buffer written each frame, grown when a frame needs more and never
/// remade otherwise. Made when a level first has an effect, so a level
/// without one pays nothing; its group's layout is the renderer's
/// ([`bind_group_layout`]), which the pipelines were built with.
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

    /// This frame's particles, for [`Self::draw`].
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

    /// This frame's view of them, for [`Self::draw`]: written apart from the
    /// particles, since whether the probe pass's depth is this frame's is
    /// known only later in a frame.
    pub fn set_view(&self, queue: &Queue, uniform: &EffectsUniform) {
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(uniform));
    }

    /// Whether there is anything to draw this frame.
    pub fn any(&self) -> bool {
        self.over + self.screened > 0
    }

    /// The smoke and flames over what is behind them, then embers and dust
    /// screened.
    /// `camera` is the scene's group 0, `probe` the probe pass's read group.
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

/// The effects' pipelines: `over` for smoke and flames, `screened` for embers
/// and dust. Group 0 is the scene's camera, group 1 the probe pass's
/// (`brush_pipeline::probe_pass::bind_group_layout`) for its depth, group 2
/// [`bind_group_layout`]. Mono and stereo twins, like every scene-pass
/// pipeline. See `multiview`.
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
        // Premultiplied over; screened as the glare is.
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
                        // COLOUR ONLY: the eye image's alpha is SpaceWarp's.
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

/// The effects' shader. Each instance is its cutout facing the head (or
/// upright, for a flame; or lying along its motion and turned across the
/// eye, for a streak), its corners from `vertex_index`. This eye's camera and
/// depth layer by `view_slot`, which a stereo pass sets per view.
pub fn effects_shader() -> String {
    format!(
        "{}{}",
        crate::renderer::tonemap::wgsl_aces_block(),
        r#"
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
    // The strongest lamp's direction in the quad's own axes: x across it, y
    // up it (as its texture's rows run upward), z toward the eye.
    @location(3) @interpolate(flat) light_q: vec3<f32>,
    // Its texture layer, the fraction toward the next frame, and how it is
    // shaded: 1 six-way smoke, 2 a flame, 3 a bed of coals, 0 plain.
    @location(4) @interpolate(flat) layer: vec3<f32>,
    // Its clip position again, for the depth it is faded against.
    @location(5) at: vec4<f32>,
    @location(6) @interpolate(flat) soft: f32,
    // A bed of coals' clock, seconds, for its flicker, and its flicker over
    // the preset's, less one.
    @location(7) @interpolate(flat) time: f32,
    @location(8) @interpolate(flat) flicker: f32,
}

@vertex fn vs_main(@builtin(vertex_index) vi: u32, inst: IIn) -> VOut {
    var out: VOut;
    // Eight triangles fanned from its middle to its cutout's eight corners.
    let tri = vi / 3u;
    let k = vi % 3u;
    let layer = min(u32(max(inst.params.y, 0.0)), LAST_LAYERu);
    var corner: vec2<f32>;
    if (k == 0u) {
        // Its middle: the mean of its corners, inside the convex outline.
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
        // Lying flat across its normal -- a bed of coals on the floor --
        // spun about it.
        let n = inst.axis.xyz;
        let a0 = normalize(select(cross(n, vec3<f32>(0.0, 0.0, 1.0)), cross(n, vec3<f32>(1.0, 0.0, 0.0)), abs(n.z) > 0.9));
        let b0 = cross(n, a0);
        let s = sin(inst.params.x);
        let c = cos(inst.params.x);
        across = a0 * c + b0 * s;
        along = b0 * c - a0 * s;
    } else if (streak > 0.0) {
        // A streak lies along its motion, its head at the particle, its
        // tail behind it.
        along = inst.axis.xyz;
        let side = cross(along, to_eye);
        across = select(fx.right.xyz, normalize(side), dot(side, side) > 1e-8);
        reach = vec2<f32>(half, half + 0.5 * streak);
        middle = centre - along * (0.5 * streak);
    } else {
        var r = fx.right.xyz;
        var u = fx.up.xyz;
        if (inst.light.w > 0.5) {
            // Upright: its up the world's, as far as the eye's view of it
            // allows, so a flame stands however the head tilts.
            let up_seen = vec3<f32>(0.0, 1.0, 0.0) - to_eye * to_eye.y;
            if (dot(up_seen, up_seen) > 1e-4) {
                u = normalize(up_seen);
                r = cross(u, to_eye);
            }
        }
        // Spun about the view: the quad's axes, and with them its texture.
        // A flame drawn mirrored has a negative stretch: its texture's left
        // on the quad's right.
        let s = sin(inst.params.x);
        let c = cos(inst.params.x);
        across = (r * c + u * s) * select(1.0, -1.0, inst.params.z < 0.0);
        along = u * c - r * s;
        reach = vec2<f32>(half, half * max(abs(inst.params.z), 1.0));
    }
    // NO SMALLER THAN A PIXEL AND A HALF: a mote or a spark under a pixel
    // falls between the samples and twinkles as it drifts, in each eye
    // differently. Drawn at least that wide with its light spread over it,
    // the same light reaches the eye, and a far mote dims as it shrinks
    // instead of flickering.
    let least = 0.75 * fx.depth.w * (camera.view_proj[view_slot] * vec4<f32>(centre, 1.0)).w;
    var drawn = max(reach, vec2<f32>(least));
    if (inst.params.w < -0.5) {
        // A POINT (a dust mote): a twentieth of a pixel across however near,
        // its light always gathered into the pixel and a half.
        drawn = vec2<f32>(least);
    }
    out.colour = inst.colour;
    out.colour.a = inst.colour.a * (reach.x * reach.y) / max(drawn.x * drawn.y, 1e-12);
    reach = drawn;
    // A FLAME OR A PUFF IS A LOW DOME, its middle toward the eye by CONVEX of
    // its half width, its outline where it was: a body in stereo, not a card.
    let domed = k == 0u && inst.params.w > 0.5 && inst.params.w < 2.5 && inst.light.w < 1.5 && streak <= 0.0;
    let dome = select(0.0, CONVEX * min(reach.x, reach.y), domed);
    let world = middle + across * (corner.x * reach.x) + along * (corner.y * reach.y) + to_eye * dome;
    out.clip = camera.view_proj[view_slot] * vec4<f32>(world, 1.0);
    out.at = out.clip;
    // Row 0 of a frame is its TOP, so the quad's top takes v = 0.
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

// How far in front of what the probe pass drew this point is, over its soft
// distance: 0 touching a wall, 1 clear of it; 1 where that depth is not
// this frame's.
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

// A flame's light by its heat, `hottest` at heat 1: black through deep red,
// orange and amber to pale yellow, brightening steeply (`flame_colour`).
fn fx_flame(hottest: vec3<f32>, heat: f32) -> vec3<f32> {
    let t = clamp(heat, 0.0, 1.0);
    let t2 = t * t;
    let bright = 0.03 + 0.97 * pow(max(t, 1e-6), 1.4);
    return vec3<f32>(1.0, 0.02 + 0.5 * t2, 0.02 * t + 0.08 * t2 * t2) * (hottest * bright);
}

// Where the gas at height `v` of a flame frame (0 its bottom row) was `dt`
// seconds before, rising as the frames were made (`flame_v`).
fn fx_flame_v(v: f32, dt: f32) -> f32 {
    let h = (v - FLAME_BASE) / (1.0 - FLAME_BASE);
    let c = FLAME_W0 / FLAME_W1;
    let moved = max((h + c) * exp(-FLAME_W1 * dt) - c, 0.0);
    return select(v, FLAME_BASE + moved * (1.0 - FLAME_BASE), h > 0.0);
}

// Smoke and flames, over what is behind them, sorted together.
// SIX-WAY SMOKE: its strongest lamp's light by how the puff passes light from
// each side, weighted by the lamp's direction squared along each axis (they
// sum to one); the rest of its light from everywhere, by the six sides' mean.
// A FLAME adds its light and covers FLAME_COVER of what is behind it: a flame
// lets light through, but screened over a floor its own light had lit, in a
// picture already through the tone curve, it washed out to white. Between two
// frames of its book, each is read where this pixel's gas was in it or will
// be, so a tongue rises between them instead of fading from one place into
// the next. Sampled with the quad's own gradients, taken before the kinds
// part ways.
@fragment fn fs_over(in: VOut) -> @location(0) vec4<f32> {
    let gx = dpdx(in.uv);
    let gy = dpdy(in.uv);
    let fade = fx_depth_fade(in);
    let layer = i32(in.layer.x);
    let next = select(layer, layer + 1, in.layer.y > 0.0);
    if (in.layer.z > 2.5) {
        // A BED OF COALS: red how hot each coal is, green its char's albedo,
        // blue its own phase of flicker. Its glow as its fire's flames are
        // shown, its char lit as a floor.
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
        // Gas read from below rises past the frame's top between frames: gone
        // before the quad's edge, which would cut it straight.
        let top = 1.0 - smoothstep(0.88, 1.0, v);
        // ERODED, NOT FADED: born, it stands up off the fuel to `light.x` of
        // its height, the core of its body (green) leading; dying, its
        // colour's alpha is how much still stands, and what goes first is what
        // lies shallowest in its body -- its tips and edges, tearing -- and its
        // root, so what is left lifts off the fuel as a flamelet and burns out
        // as the gas carries it up, cooling as it goes. (Eroded toward its
        // root, the last of a sheet was a pale ball on the fuel.) A faded
        // sheet was all there and see-through, which no flame is.
        let gone = 1.0 - in.colour.a;
        let front = mix(RISE_FROM, RISE_TO, in.light.x);
        let lasts = f.g * smoothstep(LIFT_FROM, LIFT_TO, v);
        let stands = smoothstep(gone - FLAME_EROSION, gone, lasts) * (1.0 - smoothstep(front - RISE_SOFT, front, v - RISE_LEAD * f.g));
        let alpha = clamp(f.a * stands * fade * top, 0.0, 1.0);
        // Thin flame is cooler: where it barely covers -- its ragged base,
        // its edges, an eroding edge -- it burns red, not a faint yellow,
        // which reads as olive. Its root, short of soot, glows blue as well.
        let soot = aces_fitted(fx_flame(in.colour.rgb, f.r * (0.45 + 0.55 * alpha) * (1.0 - BURN_OUT * gone)));
        // The root's blue is a thin gas's light: it adds, and covers nothing
        // (as cover it drew a purple seam where the flames meet the floor).
        let root = vec3<f32>(0.05, 0.12, 0.6) * (BLUE_ROOT * in.colour.r * f.b * alpha);
        let shown = soot;
        // It covers what is behind only as far as it outshines it: a dim red
        // edge adds its light and hides nothing (covering, one sheet's edge
        // drew a dark seam across the bright sheet behind it).
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

// Embers and dust, screened over the scene as the glare is: their light
// through the scene's tone curve.
@fragment fn fs_screen(in: VOut) -> @location(0) vec4<f32> {
    let layer = i32(in.layer.x);
    let next = select(layer, layer + 1, in.layer.y > 0.0);
    let t = mix(textureSample(fx_atlas, fx_samp, in.uv, layer), textureSample(fx_atlas, fx_samp, in.uv, next), in.layer.y);
    let strength = in.colour.a * t.a * fx_depth_fade(in);
    return vec4<f32>(aces_fitted(in.colour.rgb * strength), 0.0);
}
"#
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
        // The particles are worked out in the world's frame: seen from a
        // player turned and moved, each is where it was, only expressed in
        // the player's frame. A spread or swirl worked out in that frame
        // would turn with every snap turn.
        let e = [emitter(EffectKind::Smoke), EffectEmitter { id: "f".into(), ..emitter(EffectKind::Fire) }];
        let straight = simulate(&e, 31.0, &seen_from(Vec3::new(0.0, 1.6, 3.0)));
        let (yaw, offset) = (1.1f32, Vec3::new(2.0, 0.0, -1.0));
        let turned_at = Surroundings { offset, yaw_inv: Quat::from_rotation_y(-yaw), ..seen_from(Vec3::new(0.0, 1.6, 3.0)) };
        let turned = simulate(&e, 31.0, &turned_at);
        let world = |i: &EffectInstance| Quat::from_rotation_y(yaw) * centre(i) + offset;
        let mut a: Vec<[i32; 3]> = straight.instances.iter().map(|i| (centre(i) * 1e4).round().as_ivec3().to_array()).collect();
        let mut b: Vec<[i32; 3]> = turned.instances.iter().map(|i| (world(i) * 1e4).round().as_ivec3().to_array()).collect();
        a.sort();
        b.sort();
        assert_eq!(a.len(), b.len());
        for (p, q) in a.iter().zip(&b) {
            assert!(p.iter().zip(q).all(|(x, y)| (x - y).abs() <= 2), "{p:?} vs {q:?}");
        }
    }

    #[test]
    fn an_emitter_keeps_about_rate_times_life_alive() {
        // Smoke: 6 a second for about 5 s on average.
        let e = [emitter(EffectKind::Smoke)];
        let at = seen_from(Vec3::new(0.0, 1.6, 3.0));
        let counts: Vec<f32> = (0..20).map(|k| simulate(&e, 100.0 + k as f64 * 0.37, &at).over as f32).collect();
        let mean = counts.iter().sum::<f32>() / counts.len() as f32;
        assert!((22.0..38.0).contains(&mean), "about 6 x 5 = 30 alive, found {mean}");
    }

    #[test]
    fn dust_fills_its_box_by_volume() {
        // A 2 m box holds eight times the motes of a 1 m one.
        let small = EffectEmitter { extent: [Vec3::X * 0.5, Vec3::Y * 0.5, Vec3::Z * 0.5], ..emitter(EffectKind::Dust) };
        let large = emitter(EffectKind::Dust);
        let at = seen_from(Vec3::new(0.0, 1.6, 3.0));
        let count = |e: &EffectEmitter| (0..10).map(|k| simulate(std::slice::from_ref(e), 50.0 + k as f64, &at).screened).sum::<u32>() as f32 / 10.0;
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
        assert_eq!(count(10.0 + SPLASH_SECONDS as f64 + 0.01), 0, "all fallen back");
        // Every drop is above the surface, and some rise a hand high.
        let f = simulate_with(&[], &s, 10.2, &at);
        let ys: Vec<f32> = f.instances[f.over as usize..].iter().map(|i| i.centre[1]).collect();
        assert!(ys.iter().all(|&y| y >= -1e-3), "{ys:?}");
        assert!(ys.iter().cloned().fold(0.0, f32::max) > 0.05, "{ys:?}");
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
        // The sun low in the -z sky, shining toward +z.
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
        let gone = splash_rings::<4>(&s, 1.0 + super::super::water_pipeline::RING_SECONDS as f64 + 0.1, Vec3::ZERO);
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
            (ys.iter().cloned().fold(f32::MIN, f32::max), wide)
        };
        let (free_top, free_wide) = tops(&free);
        let (roof_top, roof_wide) = tops(&roofed);
        assert!(free_top > 1.5, "free smoke rises past 1.5 m: {free_top}");
        assert!(roof_top < 1.2, "roofed smoke stays under it: {roof_top}");
        assert!(roof_wide > free_wide, "and spreads wider: {roof_wide} vs {free_wide}");
    }

    #[test]
    fn a_tumbling_mote_flashes_and_is_lost_and_averages_its_plain_light() {
        // Over many turns its mean is 1 (no light made or lost), yet it
        // spends a good share of the time too dim to hold and flashes well
        // above its mean now and then.
        let samples: Vec<f32> = (0..200_000).map(|k| mote_glint(k as f32 * 0.013, 1.0, 7 + (k / 5000) as u64)).collect();
        let mean = samples.iter().sum::<f32>() / samples.len() as f32;
        let dim = samples.iter().filter(|&&g| g < 0.6).count() as f32 / samples.len() as f32;
        let peak = samples.iter().cloned().fold(0.0, f32::max);
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
            let t = 10.0 + 37.0 * u[3] as f64;
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
        // One mote seen from 2 m, a hand's width, and 4.9 m.
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
        // A slot's next life must not retrace its last: the old particles
        // did, every `lifetime` seconds.
        let e = [EffectEmitter { rate: 0.2, ..emitter(EffectKind::Smoke) }];
        let p = preset(EffectKind::Smoke);
        let slots = ((p.rate * 0.2 * p.life.1).ceil() as usize).max(1);
        let period = (slots as f64 / (p.rate as f64 * 0.2)).max(p.life.1 as f64);
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
                // The sheet's bottom edge a little under the bed, which is
                // its base in the texture; drawn upright, its own height.
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
        // Gas rises, faster as it climbs, and following it back and forth
        // returns where it started; at the base and below it does not move.
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
        // The light comes from above: the lamp hangs over the fire.
        let i = lit.instances.iter().find(|i| i.light[0] > 0.0).unwrap();
        assert!(i.light_dir[1] > 0.5);
    }

    #[test]
    fn a_mote_sparkles_looking_toward_its_lamp() {
        // Forward scattering: seen with the lamp behind it, a mote sends far
        // more of its light to the eye than seen from the lamp's side.
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
        // Exposed for a dim room, a fire's hottest part would be far past
        // white; adapted to, it meets the curve at FIRE_LEVEL.
        let bright = Surroundings { exposure: 30.0, ..seen_from(Vec3::new(0.0, 0.5, 1.5)) };
        let f = simulate(&[emitter(EffectKind::Fire)], 20.0, &bright);
        let peak = f.instances.iter().map(|i| i.colour[0]).fold(0.0f32, f32::max);
        assert!(peak <= FIRE_LEVEL * 1.01 && peak > 0.5 * FIRE_LEVEL, "{peak}");
    }

    #[test]
    fn a_fires_smoke_is_lit_by_it_as_its_flames_are_seen() {
        // At the room's exposure the smoke over a fire burned white above
        // flames the adapted eye holds at FIRE_LEVEL; seen as the flames are,
        // it glows a good deal dimmer than they do.
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
                        sum += c[o + channel] as f32;
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
        // The pool brightest at the fire's foot, inside the room's walls.
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
        // In its flames, in the player's frame: swaying a few centimetres
        // about 0.35 of its scale up.
        let world = |l: &Light| Quat::from_rotation_y(0.7) * l.position + offset;
        let samples: Vec<Light> = (0..2000).map(|k| at(1000.0 + k as f64 * 0.013)).collect();
        let rest = Vec3::new(14.0, 0.525, -4.0);
        let far = samples.iter().map(|l| (world(l) - rest).length()).fold(0.0f32, f32::max);
        let moved = samples.windows(2).map(|w| (world(&w[0]) - world(&w[1])).length()).sum::<f32>();
        assert!(far < FIRE_LIGHT_REACH * 1.5 && far > 0.01, "sways within reach: {far}");
        assert!(moved > 0.05, "and does move: {moved}");
        // A flicker about 0.8 of its strength, never out.
        let nominal = FIRE_LIGHT_INTENSITY * 1.5 * 1.5;
        let k: Vec<f32> = samples.iter().map(|l| l.intensity / nominal).collect();
        let mean = k.iter().sum::<f32>() / k.len() as f32;
        let (lo, hi) = k.iter().fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
        assert!((mean - 0.8).abs() < 0.06 && lo > 0.3 && hi < 1.4 && hi - lo > 0.2, "mean {mean}, {lo} to {hi}");
        // Unshadowed, so it must say how wide it is.
        let l = &samples[0];
        assert!(l.source_radius > 0.1 && !l.in_level_bake && l.mask_channel.is_none());
        assert!(fire_light(&emitter(EffectKind::Smoke), 1.0, offset, yaw_inv).is_none());
        assert!(fire_light(&EffectEmitter { rate: 0.0, ..fire.clone() }, 1.0, offset, yaw_inv).is_none());
        // Steady with no flicker, at its mean.
        let steady = EffectEmitter { variation: Variation { flicker: 0.0, ..Variation::default() }, ..fire };
        let s = fire_light(&steady, 1000.3, offset, yaw_inv).unwrap();
        assert!((s.intensity / nominal - 0.8).abs() < 1e-5, "{}", s.intensity / nominal);
    }

    /// The power of `x` (sampled `rate` times a second, mean removed) about
    /// each of `freqs`, Hz: a Hann-windowed DFT, each bin averaged with its
    /// neighbours a tenth of a hertz either side.
    fn spectrum(x: &[f32], rate: f32, freqs: &[f32]) -> Vec<f32> {
        let n = x.len();
        let mean = x.iter().sum::<f32>() / n as f32;
        let w = |i: usize| 0.5 - 0.5 * (std::f32::consts::TAU * i as f32 / n as f32).cos();
        let at = |f: f32| {
            let (mut re, mut im) = (0.0f64, 0.0f64);
            for (i, v) in x.iter().enumerate() {
                let a = std::f64::consts::TAU * f as f64 * i as f64 / rate as f64;
                re += ((v - mean) * w(i)) as f64 * a.cos();
                im += ((v - mean) * w(i)) as f64 * a.sin();
            }
            (re * re + im * im) as f32
        };
        freqs.iter().map(|&f| (at(f - 0.1) + at(f) + at(f + 0.1)) / 3.0).collect()
    }

    #[test]
    fn a_fires_light_pulses_at_the_rate_its_flames_puff() {
        // A fire D across puffs about 1.5/sqrt(D) times a second: 2.7 Hz for
        // the 0.3 m bed at scale 1, half that four times as big. Its light's
        // spectrum peaks there, with only a weak tail above -- not the
        // 1-4 Hz sines of before, the same for every fire.
        for scale in [1.0f32, 4.0] {
            let fire = EffectEmitter { scale, ..emitter(EffectKind::Fire) };
            let puff = 1.5 / (0.3 * scale).sqrt();
            assert!((fire_puff_hz(&fire) - puff).abs() < 0.01 * puff, "{} vs {puff}", fire_puff_hz(&fire));
            let rate = 40.0;
            let light: Vec<f32> = (0..(rate as usize * 60)).map(|k| fire_light(&fire, 500.0 + k as f64 / rate as f64, Vec3::ZERO, Quat::IDENTITY).unwrap().intensity).collect();
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
        let times: Vec<f64> = (0..3000).map(|k| 200.0 + k as f64 * 0.021).collect();
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
        // Froude: a fire four times the size puffs half as fast, its sheets
        // live twice as long and play their book half as fast, so each covers
        // as much of it. Twice as many stand at once, each as tall for the
        // fire's height but narrower: many tongues, not a campfire's few made
        // huge.
        let small = emitter(EffectKind::Fire);
        let big = EffectEmitter { scale: 4.0, ..emitter(EffectKind::Fire) };
        assert!((fire_tempo(&big) / fire_tempo(&small) - 2.0).abs() < 1e-5);
        assert!((fire_puff_hz(&small) / fire_puff_hz(&big) - 2.0).abs() < 1e-4);
        let stats = |e: &EffectEmitter| {
            let (mut count, mut tall, mut wide, mut pace) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
            let n = 400;
            for k in 0..n {
                let t = 300.0 + k as f64 * 0.37;
                let (a, b) = (fire_sheets(e, t), fire_sheets(e, t + 0.01));
                count += a.len() as f32;
                tall += a.iter().map(|s| s.size * s.stretch).sum::<f32>() / a.len() as f32;
                wide += a.iter().map(|s| s.size).sum::<f32>() / a.len() as f32;
                // The same sheets a hundredth of a second on, where they are.
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
        // And the gas in it rises faster in metres a second, as sqrt(scale).
        assert!(((t4 * p4) / (t1 * p1) - 2.0).abs() < 0.3);
    }

    #[test]
    fn a_fires_sheets_puff_together() {
        // Each sheet plays its stretch of the book in step with the fire's
        // beat, so the flame as a whole swells and necks -- the book's own
        // puffing, summed over the sheets, shows in how much flame there is.
        let fire = EffectEmitter { scale: 1.5, ..emitter(EffectKind::Fire) };
        let rate = 40.0;
        // Each sheet's book frame's flame, weighed by how much of the sheet
        // stands: the puffing alone, the hand-overs left out.
        let area: Vec<f32> = (0..(rate as usize * 60))
            .map(|k| {
                let sheets = fire_sheets(&fire, 700.0 + k as f64 / rate as f64);
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
        // The book's flame swells and shrinks at its puffing rate: about
        // `FLAME_SECONDS x puff_book_hz` cycles over its frames.
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
        // Erosion takes the shallowest first, all of it at the end and none
        // of it at the start, steadily between.
        assert!(cover.windows(2).all(|w| w[1] > w[0]) && cover[0] < 0.05 && cover[8] == 1.0, "{cover:?}");
        // The shallowest is high up: a dying sheet's tips go before its root.
        let f = fire_frame(20);
        let n = ATLAS_SIZE as usize;
        let depth = |rows: std::ops::Range<usize>| {
            let (mut sum, mut count) = (0.0f32, 0.0f32);
            for y in rows {
                for x in 0..n {
                    let t = &f[(y * n + x) * 4..(y * n + x) * 4 + 4];
                    if t[3] > 128 {
                        sum += t[1] as f32;
                        count += 1.0;
                    }
                }
            }
            sum / count.max(1.0)
        };
        assert!(depth(n / 4..n / 2) < 0.6 * depth(3 * n / 4..n), "tips shallow, root deep");
        // And the root is marked blue at the bottom, nowhere above.
        let blue = |rows: std::ops::Range<usize>| rows.flat_map(|y| (0..n).map(move |x| (y, x))).map(|(y, x)| f[(y * n + x) * 4 + 2] as f32).sum::<f32>();
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
        // Every field named at its default, as a scene may write them.
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
        // A seed moves every kind's particles, and keeps how many there are.
        let seeded: Vec<EffectEmitter> = plain.iter().map(|e| EffectEmitter { variation: Variation { seed: 7, ..Variation::default() }, ..e.clone() }).collect();
        for (p, s) in plain.iter().zip(&seeded) {
            let (a, b) = (simulate(std::slice::from_ref(p), 33.3, &at), simulate(std::slice::from_ref(s), 33.3, &at));
            assert!(!a.instances.is_empty(), "{:?}", p.kind);
            assert_ne!(bytemuck::cast_slice::<_, u8>(&a.instances), bytemuck::cast_slice::<_, u8>(&b.instances), "{:?}", p.kind);
            if p.kind != EffectKind::Embers {
                // (A pop is in one pattern's moment and not the other's.)
                let slots = |e: &EffectEmitter| (0..20).map(|k| simulate(std::slice::from_ref(e), 40.0 + k as f64 * 0.31, &at).instances.len()).sum::<usize>() as f32;
                assert!((slots(p) / slots(s) - 1.0).abs() < 0.25, "{:?}", p.kind);
            }
        }
        // Clamped to the editor's ranges.
        let wild = Variation { turbulence: 99.0, flicker_rate: 0.0, seed: 70000, intensity: f32::NAN, ..Variation::default() }.clamped();
        assert_eq!((wild.turbulence, wild.flicker_rate, wild.seed, wild.intensity), (4.0, 0.25, 65535, 1.0));
    }

    #[test]
    fn variation_does_what_it_says() {
        let at = seen_from(Vec3::new(0.3, 1.6, 3.0));
        let with = |kind: EffectKind, v: Variation| EffectEmitter { variation: v, ..emitter(kind) };
        let d = Variation::default();
        // Intensity scales a fire's flames and its light.
        let f1 = simulate(&[with(EffectKind::Fire, d)], 20.0, &at);
        let f2 = simulate(&[with(EffectKind::Fire, Variation { intensity: 2.0, ..d })], 20.0, &at);
        assert!((f2.instances[0].colour[0] / f1.instances[0].colour[0] - 2.0).abs() < 1e-4);
        let l = |v: Variation| fire_light(&with(EffectKind::Fire, v), 20.0, Vec3::ZERO, Quat::IDENTITY).unwrap().intensity;
        assert!((l(Variation { intensity: 2.0, ..d }) / l(d) - 2.0).abs() < 1e-4);
        // A cooler-or-hotter spread tints sheets apart.
        let hot = simulate(&[with(EffectKind::Fire, Variation { temperature_variation: 1.0, ..d })], 20.0, &at);
        let ratios: Vec<f32> = hot.instances.iter().map(|i| i.colour[2] / i.colour[0]).collect();
        assert!(ratios.iter().cloned().fold(0.0, f32::max) > 1.5 * ratios.iter().cloned().fold(f32::MAX, f32::min));
        // A faster tempo puffs faster.
        assert!((fire_puff_hz(&with(EffectKind::Fire, Variation { flicker_rate: 2.0, ..d })) / fire_puff_hz(&with(EffectKind::Fire, d)) - 2.0).abs() < 1e-4);
        // Smoke thickens with intensity; no turbulence, no swirl.
        let s1 = simulate(&[with(EffectKind::Smoke, d)], 40.0, &at);
        let s2 = simulate(&[with(EffectKind::Smoke, Variation { intensity: 0.5, ..d })], 40.0, &at);
        assert!((s2.instances[3].colour[3] / s1.instances[3].colour[3] - 0.5).abs() < 1e-4);
        // A wider spread of lives: as many alive on average, the longest
        // lived rising higher.
        let top = |v: Variation| (0..20).flat_map(|k| simulate(&[with(EffectKind::Smoke, v)], 60.0 + k as f64, &at).instances).map(|i| i.centre[1]).fold(0.0f32, f32::max);
        assert!(top(Variation { life_variation: 2.0, ..d }) > top(d) + 0.1);
        // Coals: steady with no flicker; their clock by the flicker rate.
        let c = simulate(&[with(EffectKind::Coals, Variation { flicker: 0.0, flicker_rate: 2.0, ..d })], 20.0, &at);
        assert_eq!(c.instances[0].light_dir[3], -1.0);
        assert!((c.instances[0].params[2] - 40.0).abs() < 1e-4);
    }

    #[test]
    fn embers_pop_in_bursts_carried_up_by_the_plume() {
        // Besides the steady stream, now and then a burst of sparks thrown up
        // into the plume and carried higher than the stream's.
        let at = seen_from(Vec3::new(0.3, 1.6, 3.0));
        let embers = emitter(EffectKind::Embers);
        let (mut most, mut least, mut high) = (0u32, u32::MAX, 0.0f32);
        for k in 0..600 {
            let mut pops = Vec::new();
            ember_pops(&embers, 100.0 + k as f64 * 0.1, &at, &mut pops);
            let n = pops.len() as u32;
            most = most.max(n);
            least = least.min(n);
            high = pops.iter().map(|i| i.centre[1]).fold(high, f32::max);
        }
        // The stream alone: what `simulate` draws that is not a pop.
        let stream_high = (0..200)
            .flat_map(|k| {
                let time = 100.0 + k as f64 * 0.3;
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
        // A veil round it as the CIE eye's for a lamp of its light: faint,
        // since flames are some thirty times dimmer per area than a frosted
        // bulb -- seen close in a dark night, and none from across a room.
        let near = Vec3::new(0.0, 1.0, 1.2);
        let lobe = super::super::glare::glare_lobes(&s, near)[0];
        let q = super::super::glare::glare_quad(&s, &lobe, near, 20.0, 1.0).expect("a veil");
        assert!(q.peak > 0.02 && q.peak < 0.5, "{}", q.peak);
        assert!(super::super::glare::glare_quad(&s, &lobes[0], eye, 4.0, 1.0).is_none());
        assert!(fire_glare(&emitter(EffectKind::Smoke), 30.0, Vec3::ZERO, Quat::IDENTITY).is_none());
    }

    #[test]
    fn a_bed_breathes_with_the_fire_on_it() {
        let at = seen_from(Vec3::new(0.3, 1.6, 3.0));
        let fire = EffectEmitter { id: "f".into(), scale: 1.5, ..emitter(EffectKind::Fire) };
        let bed = EffectEmitter { id: "c".into(), scale: 1.5, ..emitter(EffectKind::Coals) };
        let glow = |e: &[EffectEmitter], t: f64| simulate(e, t, &at).instances.iter().find(|i| i.params[3] == 3.0).unwrap().colour[0];
        let with: Vec<f32> = (0..200).map(|k| glow(&[fire.clone(), bed.clone()], 50.0 + k as f64 * 0.05)).collect();
        let alone = glow(&[bed.clone()], 50.0);
        let (lo, hi) = with.iter().fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
        assert!(hi > 1.03 * lo && hi < 1.25 * alone && lo > 0.8 * alone, "{lo} to {hi}, alone {alone}");
        assert!((0..50).map(|k| glow(&[bed.clone()], 50.0 + k as f64 * 0.05)).all(|g| g == alone), "alone it does not");
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
        // Row 0 is the top: a flame's base (wide, hot) is in the bottom
        // half, its tip in the top. Effect textures have been drawn upside
        // down before, and a symmetric test sprite could never show it.
        let f = &fire_frame(4);
        let n = ATLAS_SIZE as usize;
        let sum_in = |rows: std::ops::Range<usize>, channel: usize| -> f32 {
            rows.flat_map(|y| (0..n).map(move |x| f[(y * n + x) * 4 + channel] as f32)).sum()
        };
        assert!(sum_in(n / 2..n, 3) > 1.5 * sum_in(0..n / 2, 3), "more flame below the middle");
        assert!(sum_in(n / 2..n, 0) > 2.0 * sum_in(0..n / 2, 0), "hotter below the middle");
    }

    #[test]
    fn a_puff_lit_from_one_side_is_bright_on_that_side_and_glows_against_the_light() {
        let [front, back] = smoke_frame(3);
        let n = ATLAS_SIZE as usize;
        let texel = |img: &[u8], x: usize, y: usize, c: usize| img[(y * n + x) * 4 + c] as f32;
        assert!(texel(&front, n / 2, n / 2, 3) > 200.0, "dense in the middle: {}", texel(&front, n / 2, n / 2, 3));
        assert_eq!(texel(&front, 1, 1, 3), 0.0, "nothing in the corner");
        // Lit from the right (+x, front red): its right side brighter than
        // its left, where the light has gone through it first.
        let row = n / 2;
        let right: f32 = (n * 5 / 8..n * 3 / 4).map(|x| texel(&front, x, row, 0)).sum();
        let left: f32 = (n / 4..n * 3 / 8).map(|x| texel(&front, x, row, 0)).sum();
        assert!(right > 1.5 * left, "lit side {right}, far side {left}");
        // From behind (back blue): its thin edge passes more than its thick
        // middle.
        let edge = (0..n).map(|x| (texel(&back, x, row, 3), texel(&back, x, row, 2))).find(|&(a, _)| a > 40.0).unwrap();
        assert!(edge.1 > texel(&back, n / 2, row, 2) + 40.0, "edge {} middle {}", edge.1, texel(&back, n / 2, row, 2));
        // And from the front (front blue) more than from behind, at its
        // middle.
        assert!(texel(&front, n / 2, row, 2) > texel(&back, n / 2, row, 2) + 60.0);
    }

    #[test]
    fn a_cutout_holds_everything_visible_and_cuts_the_corners() {
        let layers = atlas_layers();
        let cuts = cutouts(&layers);
        let n = ATLAS_SIZE as usize;
        for (l, cut) in cuts.iter().enumerate() {
            let corners: Vec<Vec3> = (0..8).map(|k| Vec3::new(cut[2 * k], cut[2 * k + 1], 0.0)).collect();
            // Inside a convex counter-clockwise polygon: left of every edge.
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
            // The polygon's area, by the shoelace.
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
        assert!(a == b);
    }

    #[test]
    fn mips_keep_a_transparent_edge_from_darkening() {
        // A white texel beside a transparent black one: the mip is white,
        // half covered.
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

    /// FOR THE EDITOR'S PREVIEW (`scene_editor_web`, `effectsSim.js`):
    /// `OUT=dir cargo test --release --lib export_effects_for_the_editor --
    /// --ignored` writes the atlas's layers as they are uploaded
    /// (`effects_atlas_layers.png`: `ATLAS_LAYERS` squares of `ATLAS_SIZE`,
    /// 16 to a row, RGBA as stored), their cutouts (`effects_cutouts.json`),
    /// and GOLDEN FRAMES (`effects_golden.json`): what `simulate_with` makes
    /// for fixed emitters, splashes, lights and times. The editor's port of
    /// the simulation is tested against those, so its preview is this one.
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
            // Every variation field away from its default, so the port's
            // formulas are held to these too.
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

    /// A LOOK AT THE TEXTURES: `OUT=dir cargo test --release --lib
    /// save_the_effects_atlas -- --ignored` writes `effects_atlas.png` -- the
    /// smoke's frames lit from the front right, the same lit from behind, both
    /// over grey, shaded as `fs_over` shades them; the flame's book over black,
    /// coloured and toned as `fs_over` shows it adapted to; the ember and the
    /// mote.
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
                        let f = |img: &[u8], c: usize| img[o + c] as f32 / 255.0;
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
                    let (heat, a) = (img[o] as f32 / 255.0, img[o + 3] as f32 / 255.0);
                    let c = super::super::tonemap::aces_fitted(flame_colour(heat) * FIRE_LEVEL) * a;
                    // `encode` takes linear light and the curve's output is
                    // already the display's linear light.
                    put((k % 16, 2 + k / 16), x, y, c.to_array());
                }
            }
        }
        for (cell, layer) in [(0usize, EMBER_LAYER), (1, MOTE_LAYER)] {
            let img = &layers[layer as usize];
            for y in 0..n {
                for x in 0..n {
                    let a = img[(y * n + x) * 4 + 3] as f32 / 255.0;
                    put((cell, 2 + fire_rows), x, y, [a, a, a]);
                }
            }
        }
        let path = out.join("effects_atlas.png");
        image::save_buffer(&path, &sheet, w as u32, h as u32, image::ExtendedColorType::Rgba8).expect("write the sheet");
        eprintln!("wrote {}", path.display());
    }

    /// THE EFFECTS ALONE, drawn through their own pipelines over a dark
    /// background with no scene round them: `OUT=dir cargo test --release
    /// --lib render_effects_alone -- --ignored` writes `effects_alone.png`, a
    /// fire on its coals with its smoke and embers 1.6 m off, at
    /// `EFFECTS_TIME` (30 s); `EYE=x,y,z` and `AT=x,y,z` move the eye.
    /// The scene's camera and probe groups are stood in for by the bindings
    /// the shader reads from them.
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
            size: (row * h) as u64,
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

    /// THE FLAME TABLES, measured from the book: `cargo test --release --lib
    /// print_the_flame_tables -- --ignored --nocapture` prints `FLAME_AREA`,
    /// `EROSION_COVER` and `FLAME_AREA_MEAN` to paste in (here and in the
    /// editor's `effectsSim.js`) after `fire_frame` changes.
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
            let time = 100.0 + k as f64 * 0.0173;
            sum += fire_sheets(&fire, time)
                .iter()
                .map(|s| (s.size * s.size * s.stretch * lerp_at(&area, s.frame) * lerp_at(&cover, s.stands * 8.0) * lerp_at(&rise, s.risen * 8.0)) as f64)
                .sum::<f64>();
        }
        eprintln!("const FLAME_AREA_MEAN: f32 = {:.6};", sum / samples as f64);
        let g: Vec<u8> = (0..FIRE_FRAMES).flat_map(|k| fire_frame(k).chunks(4).filter(|t| t[3] > 128).map(|t| t[1]).collect::<Vec<_>>()).collect();
        let q = |f: f32| g.iter().filter(|&&x| x as f32 >= f * 255.0).count() as f32 / g.len() as f32;
        eprintln!("depth over covered texels: >=0.25 {:.3}, >=0.5 {:.3}, >=0.75 {:.3}, >=0.9 {:.3}", q(0.25), q(0.5), q(0.75), q(0.9));
    }

    /// What a fire hands the GPU: `cargo test --release --lib
    /// print_a_fires_particles -- --ignored --nocapture`.
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

    /// What a frame's simulation costs the CPU for the test room's effects --
    /// a fire on its coals with its smoke and embers, and a hall of dust,
    /// six lamps about: `cargo test --release --lib time_a_frames_effects --
    /// --ignored --nocapture`.
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
                count += simulate(emitters, 30.0 + k as f64 / 72.0, &at).instances.len();
            }
            (count / 200, start.elapsed().as_secs_f64() / 200.0 * 1e3)
        };
        let (count, each) = time(&emitters);
        eprintln!("{count} particles a frame, {each:.3} ms a frame");
        for e in &emitters {
            let (count, each) = time(std::slice::from_ref(e));
            eprintln!("  {:>6}: {count} particles, {each:.3} ms", e.id);
        }
        // The fire's light and glare, which the client asks for each frame.
        let start = std::time::Instant::now();
        for k in 0..2000 {
            let t = 30.0 + k as f64 / 72.0;
            std::hint::black_box(fire_light(&emitters[0], t, Vec3::ZERO, Quat::IDENTITY));
            std::hint::black_box(fire_glare(&emitters[0], t, Vec3::ZERO, Quat::IDENTITY));
        }
        eprintln!("  fire_light + fire_glare: {:.4} ms", start.elapsed().as_secs_f64() / 2000.0 * 1e3);
    }

    #[test]
    fn the_shader_validates() {
        let src = effects_shader();
        let error = crate::renderer::multiview::multiview_validation_error_of(&src, false);
        assert!(error.is_none(), "{error:?}");
    }
}
