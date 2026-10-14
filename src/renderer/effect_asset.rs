//! EFFECT ASSETS: a particle effect as DATA -- emitters, each a STACK OF
//! MODULES in five stages (spawn, initialize, update, render, events) -- read
//! from `game/effects/<name>.json`, checked, and handed to the compiler
//! (`effects::stack`), which turns each emitter into a flat CPU program and
//! a render key. See `docs/particle-authoring-plan-2026-10-12.md`.
//!
//! This file is the model, the loader and the validator. It holds no wgpu and
//! nothing of the renderer but [`BookInfo`], which the renderer fills in
//! (`effects::books`), so the editor's preview can be built from it too.
//!
//! WHAT IT REFUSES. An unknown module, an unknown parameter, a missing
//! lifetime, a second placement: load errors, with the emitter and module
//! named -- a typo must not ship as a default. An effect over a budget
//! (instances, fill, lights, texture) is an error unless it carries
//! `over_budget` with its reason, and then it is listed
//! ([`Report::over_budget`]).
//!
//! STATELESS BY CONSTRUCTION. Every module is a function of a particle's
//! seed (initialize), of its age, seed and the effect's inputs (update), or
//! of the frame's surroundings where it is (lighting). None reads the
//! particle's own previous frame: the set is closed and has no such module.

use serde::{Deserialize, Serialize};

/// The version this code writes; older files are upgraded in memory
/// ([`EffectAsset::parse`]).
pub const ASSET_VERSION: u32 = 1;
/// The most keys one curve or gradient holds.
pub const MAX_CURVE_KEYS: usize = 8;
/// A curve is baked into this many samples; evaluating one is a lookup.
pub const CURVE_SAMPLES: usize = 32;
/// The most named inputs an effect takes (`EffectBurst::inputs`).
pub const MAX_INPUTS: usize = 4;

// PROVISIONAL LIMITS (plan section 5.3 and its decision 8): confirmed only by
// a headset run. Each is checked when an asset is validated.
/// Live instances one effect may keep, worst case.
pub const LIMIT_INSTANCES: u32 = 400;
/// Fill of one burst alone at its reference view, megapixel-layers.
pub const LIMIT_FILL_BURST: f32 = 0.9;
/// Fill of a placed effect, or of one source's bursts at their `max_live`
/// cap, megapixel-layers.
pub const LIMIT_FILL_STEADY: f32 = 1.5;
/// Live lights one effect may raise.
pub const LIMIT_LIGHTS: u32 = 1;
/// Texture bytes of every effect book in a level together.
pub const LIMIT_TEXTURE_BYTES: u64 = 16 * 1024 * 1024;
/// How near the eye a burst may spawn before it must name a `near_fade`.
pub const NEAR_EYE: f32 = 0.35;

/// The reference view fill is worked out at: the offline harness's eye view,
/// 1546 pixels over 104 degrees, both eyes.
pub const REFERENCE_FOCAL_PIXELS: f32 = 604.0;
pub const REFERENCE_EYES: f32 = 2.0;
/// How far from the eye a held effect (a burst) and a placed one are taken
/// to be, metres, where the asset does not say.
pub const REFERENCE_HELD: f32 = 1.0;
pub const REFERENCE_PLACED: f32 = 2.5;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EffectAsset {
    pub name: String,
    pub version: u32,
    /// The effect's named inputs, in the order `EffectBurst::inputs` holds
    /// them: `energy`, `heat`, `suppressed`... At most [`MAX_INPUTS`].
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub inputs: Vec<String>,
    /// A burst's ENERGY, shot to shot (1 at its median): a muzzle flash's
    /// size and light vary -- some shots twice the median, some a third. An
    /// input named `energy` multiplies it. Its lobes' length, its emissive
    /// sprites' size and light, and its light pulse follow it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub energy: Option<Value>,
    pub emitters: Vec<EmitterDef>,
    pub scalability: Scalability,
    #[serde(default)]
    pub budget: Budget,
    /// Why this effect may exceed a budget. It then loads, and is listed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub over_budget: Option<String>,
    #[serde(default, skip_serializing_if = "SeedPolicy::is_default")]
    pub seed_policy: SeedPolicy,
}

fn is_false(b: &bool) -> bool {
    !*b
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SeedPolicy {
    /// Each burst (or each respawn of a slot) draws anew.
    #[default]
    PerSpawn,
    /// Every burst is the same burst: for tuning, and for tests.
    Locked,
}

impl SeedPolicy {
    fn is_default(&self) -> bool {
        *self == Self::PerSpawn
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EmitterDef {
    pub name: String,
    /// Where it is born: at the event's point, or at its second (a weapon's
    /// ejection port).
    #[serde(default, skip_serializing_if = "At::is_origin")]
    pub at: At,
    #[serde(default)]
    pub spawn: Vec<SpawnModule>,
    #[serde(default)]
    pub initialize: Vec<InitModule>,
    #[serde(default)]
    pub update: Vec<UpdateModule>,
    #[serde(default)]
    pub render: Vec<RenderModule>,
    #[serde(default)]
    pub events: Vec<EventModule>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum At {
    #[default]
    Origin,
    Port,
}

impl At {
    fn is_origin(&self) -> bool {
        *self == Self::Origin
    }
}

/// THE SCALABILITY BLOCK, required: what the frame may do to this effect
/// when it is far, small, near the eye or one of too many.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Scalability {
    /// Past this, metres from the eye, nothing of it is simulated or drawn.
    pub cull_distance: f32,
    /// The whole effect is skipped when its bounds are under this many
    /// pixels across.
    pub min_pixels: f32,
    /// 0-1. The governor sheds the lowest first, and among equals the
    /// farthest and smallest.
    pub importance: f32,
    /// Bursts of this asset from one source alive at once. A later one
    /// replaces the oldest, and tells the rest how many were shed
    /// (`heat`), so a string of shots is denser smoke and not more puffs.
    pub max_live: u32,
    /// Particles fade and shrink from the second distance in to the first,
    /// metres from the eye: near-eye fill is the largest cost a held
    /// effect has.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub near_fade: Option<[f32; 2]>,
    /// How far from the eye its budgets are worked out, metres
    /// ([`REFERENCE_HELD`], [`REFERENCE_PLACED`] when absent).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference_distance: Option<f32>,
}

/// What the effect says it needs. Each is checked against what its stacks
/// add up to, and against the hard limits.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Budget {
    pub fill_megapixel_layers: f32,
    pub instances: u32,
    #[serde(default)]
    pub lights: u32,
}

impl Default for Budget {
    fn default() -> Self {
        Self { fill_megapixel_layers: LIMIT_FILL_BURST, instances: LIMIT_INSTANCES, lights: 0 }
    }
}

// ---------------------------------------------------------------------------
// Values.

/// A number, or how one is drawn for each particle. Every draw is from one
/// of the particle's own uniforms, so an effect is reproducible.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Value {
    Number(f32),
    Range(RangeValue),
    Uniform(UniformValue),
    Lognormal(LognormalValue),
    Squared(SquaredValue),
    LogSquared(LogSquaredValue),
    Curve(CurveValue),
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RangeValue {
    pub range: [f32; 2],
}
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UniformValue {
    pub uniform: [f32; 2],
}
/// `[median, sigma]`: most near the median, a few several times it.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LognormalValue {
    pub lognormal: [f32; 2],
}
/// `[lo, hi]` by the uniform squared: many near `lo`, few near `hi`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SquaredValue {
    pub squared: [f32; 2],
}
/// `[lo, hi]` by octaves of the uniform squared, `lo * (hi / lo)^(u^2)`:
/// most near `lo`, few large -- how real dust's grains are sized.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LogSquaredValue {
    pub log_squared: [f32; 2],
}
/// Keys `[t, v]`, `t` rising in 0-1, joined by monotone cubics (no
/// overshoot: an opacity curve never dips below its keys).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CurveValue {
    pub curve: Vec<[f32; 2]>,
    #[serde(default)]
    pub over: Over,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Over {
    /// The particle's age over its life.
    #[default]
    Life,
    /// The particle's own uniform.
    Seed,
}

/// How far out a log-normal draw may land, in sigmas either way.
pub const LOGNORMAL_SIGMAS: f32 = 2.5;

impl Value {
    /// The value for a particle whose uniform is `u` (0-1) at `t` of its
    /// life.
    pub fn sample(&self, u: f32, t: f32) -> f32 {
        match self {
            Self::Number(v) => *v,
            Self::Range(RangeValue { range: [lo, hi] }) | Self::Uniform(UniformValue { uniform: [lo, hi] }) => lo + (hi - lo) * u,
            Self::Squared(SquaredValue { squared: [lo, hi] }) => lo + (hi - lo) * u * u,
            Self::LogSquared(LogSquaredValue { log_squared: [lo, hi] }) => lo * (hi / lo).powf(u * u),
            Self::Lognormal(LognormalValue { lognormal: [median, sigma] }) => median * (sigma * normal_of(u)).exp(),
            Self::Curve(c) => curve_at(&c.curve, if c.over == Over::Seed { u } else { t }),
        }
    }

    /// The least and the most it can be.
    pub fn bounds(&self) -> (f32, f32) {
        match self {
            Self::Number(v) => (*v, *v),
            Self::Range(RangeValue { range: [a, b] })
            | Self::Uniform(UniformValue { uniform: [a, b] })
            | Self::Squared(SquaredValue { squared: [a, b] })
            | Self::LogSquared(LogSquaredValue { log_squared: [a, b] }) => (a.min(*b), a.max(*b)),
            Self::Lognormal(LognormalValue { lognormal: [median, sigma] }) => {
                let k = (sigma.abs() * LOGNORMAL_SIGMAS).exp();
                (median / k, median * k)
            }
            Self::Curve(c) => c.curve.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), k| (lo.min(k[1]), hi.max(k[1]))),
        }
    }

    /// Its mean over particles (and, for a curve over life, over the life).
    pub fn mean(&self) -> f32 {
        match self {
            Self::Number(v) => *v,
            Self::Range(RangeValue { range: [a, b] }) | Self::Uniform(UniformValue { uniform: [a, b] }) => 0.5 * (a + b),
            Self::Squared(SquaredValue { squared: [a, b] }) => a + (b - a) / 3.0,
            Self::LogSquared(LogSquaredValue { log_squared: [a, b] }) => (0..16).map(|i| a * (b / a).powf(((i as f32 + 0.5) / 16.0).powi(2))).sum::<f32>() / 16.0,
            Self::Lognormal(LognormalValue { lognormal: [median, sigma] }) => median * (0.5 * sigma * sigma).exp(),
            Self::Curve(c) => {
                let table = bake_curve(&c.curve);
                table.iter().sum::<f32>() / table.len() as f32
            }
        }
    }

    fn check(&self, what: &str, least: f32, issues: &mut Vec<String>) {
        if let Self::Curve(c) = self {
            check_keys(c.curve.iter().map(|k| k[0]), what, issues);
        }
        let (lo, hi) = self.bounds();
        if !(lo.is_finite() && hi.is_finite()) {
            issues.push(format!("{what} is not a finite number"));
        } else if lo < least {
            issues.push(format!("{what} can be {lo}, under its least {least}"));
        }
    }
}

fn check_keys(times: impl Iterator<Item = f32>, what: &str, issues: &mut Vec<String>) {
    let times: Vec<f32> = times.collect();
    if times.is_empty() || times.len() > MAX_CURVE_KEYS {
        issues.push(format!("{what}: a curve holds 1 to {MAX_CURVE_KEYS} keys, this one {}", times.len()));
    }
    if times.iter().any(|t| !(0.0..=1.0).contains(t)) || times.windows(2).any(|w| w[1] <= w[0]) {
        issues.push(format!("{what}: a curve's keys run from 0 to 1, each after the last"));
    }
}

/// A standard normal's value at the uniform `u`, clamped to
/// [`LOGNORMAL_SIGMAS`] either way (Acklam's rational approximation of the
/// inverse normal, central region and tails).
pub fn normal_of(u: f32) -> f32 {
    let p = (u as f64).clamp(1e-6, 1.0 - 1e-6);
    const A: [f64; 6] = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.3577518672690, -30.66479806614716, 2.506628277459239];
    const B: [f64; 5] = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572];
    const C: [f64; 6] = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
    const D: [f64; 4] = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416];
    let tail = |q: f64| (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5]) / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0);
    let z = if p < 0.02425 {
        tail((-2.0 * p.ln()).sqrt())
    } else if p > 1.0 - 0.02425 {
        -tail((-2.0 * (1.0 - p).ln()).sqrt())
    } else {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    };
    (z as f32).clamp(-LOGNORMAL_SIGMAS, LOGNORMAL_SIGMAS)
}

/// A curve's value at `x` (0-1): a monotone cubic through its keys (Fritsch-
/// Carlson), flat before the first and after the last.
pub fn curve_at(keys: &[[f32; 2]], x: f32) -> f32 {
    match keys {
        [] => 0.0,
        [only] => only[1],
        _ => {
            if x <= keys[0][0] {
                return keys[0][1];
            }
            let last = keys.len() - 1;
            if x >= keys[last][0] {
                return keys[last][1];
            }
            let i = keys.windows(2).position(|w| x < w[1][0]).unwrap_or(last - 1);
            let slope = |j: usize| (keys[j + 1][1] - keys[j][1]) / (keys[j + 1][0] - keys[j][0]).max(1e-6);
            // A key's tangent: the harmonic mean of the slopes either side
            // where they agree in sign, flat where they do not.
            let tangent = |j: usize| {
                if j == 0 {
                    slope(0)
                } else if j == last {
                    slope(last - 1)
                } else {
                    let (a, b) = (slope(j - 1), slope(j));
                    if a * b <= 0.0 {
                        0.0
                    } else {
                        2.0 * a * b / (a + b)
                    }
                }
            };
            let h = keys[i + 1][0] - keys[i][0];
            let s = (x - keys[i][0]) / h.max(1e-6);
            let (m0, m1) = (tangent(i) * h, tangent(i + 1) * h);
            let (s2, s3) = (s * s, s * s * s);
            (2.0 * s3 - 3.0 * s2 + 1.0) * keys[i][1] + (s3 - 2.0 * s2 + s) * m0 + (-2.0 * s3 + 3.0 * s2) * keys[i + 1][1] + (s3 - s2) * m1
        }
    }
}

/// A curve as the CPU program holds it: [`CURVE_SAMPLES`] values across 0-1.
pub fn bake_curve(keys: &[[f32; 2]]) -> [f32; CURVE_SAMPLES] {
    std::array::from_fn(|i| curve_at(keys, i as f32 / (CURVE_SAMPLES - 1) as f32))
}

/// A baked curve's value at `x`: a lookup and a lerp.
pub fn table_at(table: &[f32; CURVE_SAMPLES], x: f32) -> f32 {
    let f = x.clamp(0.0, 1.0) * (CURVE_SAMPLES - 1) as f32;
    let i = (f as usize).min(CURVE_SAMPLES - 2);
    table[i] + (table[i + 1] - table[i]) * (f - i as f32)
}

/// A particle's lifetime: seconds, or the one app frame of its event.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum LifeValue {
    Named(LifeName),
    Seconds(Value),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LifeName {
    /// Drawn on one app frame and no other, whatever the frame time: a
    /// muzzle flash lasts 1-3 ms.
    OneFrame,
}

// ---------------------------------------------------------------------------
// Modules.

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "module", rename_all = "snake_case", deny_unknown_fields)]
pub enum SpawnModule {
    /// Particles a second. In a burst's effect: for `for_seconds` after the
    /// event (a smoking barrel).
    Rate {
        per_second: f32,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        for_seconds: Option<f32>,
    },
    /// So many at the event, each burst with this chance of any at all.
    Burst {
        count: Value,
        #[serde(default = "one", skip_serializing_if = "is_one")]
        chance: f32,
    },
    /// Alive per cubic metre of the placed object's box (dust).
    Density { per_cubic_metre: f32 },
    /// Sheets handing over on a shared beat (a fire's puffing).
    Beat { per_second: f32 },
    /// One of the effect's inputs scales the count, the rate or the chance,
    /// by a curve of the input (0-1).
    ByInput { input: String, target: SpawnTarget, curve: Vec<[f32; 2]> },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpawnTarget {
    Count,
    Rate,
    Chance,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "module", rename_all = "snake_case", deny_unknown_fields)]
pub enum InitModule {
    Life { value: LifeValue },
    /// The sprite's half size, metres.
    Size { value: Value },
    /// Launch speed, m/s.
    Speed { value: Value },
    /// Radians a second (either way by the seed) and radians it may start
    /// turned (either way).
    Spin {
        #[serde(default)]
        rate: f32,
        #[serde(default)]
        start: f32,
    },
    ShapePoint {},
    /// Born on a disc of this radius across the emitter's direction. With
    /// `fills_object`, a PLACED effect's disc spans its object's footprint
    /// instead -- stretched to the box's width and depth, its rate grown
    /// with the area so it stays as dense (`stack::Placed::footprint`): one
    /// pond mist for every pond, sized by the pond's box. The editor's
    /// preview, with no object, draws the disc as authored.
    ShapeDisc {
        radius: f32,
        #[serde(default, skip_serializing_if = "is_false")]
        fills_object: bool,
    },
    /// Born anywhere in the placed object's box.
    ShapeBox {},
    /// Launched within this angle of the emitter's direction, or of the
    /// world's up (`about: "up"`: what rises whichever way a barrel points).
    ShapeCone {
        angle_deg: f32,
        #[serde(default, skip_serializing_if = "About::is_direction")]
        about: About,
    },
    /// A MUZZLE DEVICE'S PORTS: `lobes` jets leaving round the bore, opened
    /// `open_deg` from it and raked `rake_deg` forward of square, none
    /// pointing down with `skip_down` (a birdcage's closed bottom), plus
    /// `side_pairs` pairs square to the bore (a brake's). Each particle is
    /// one lobe, `length` metres long (its sprite stretched along it); the
    /// particle past the last lobe is the core at the muzzle. 0 lobes is a
    /// bare muzzle: every particle a forward tongue.
    ShapeLobes {
        lobes: u32,
        open_deg: f32,
        #[serde(default)]
        rake_deg: f32,
        #[serde(default)]
        skip_down: bool,
        #[serde(default)]
        side_pairs: u32,
        length: Value,
    },
    /// Spots spread evenly over a bed of this radius by a low-discrepancy
    /// sequence (a fire's sheets).
    SpotsLowDiscrepancy { radius: f32 },
    /// Its particles appear this many app frames after the event (an
    /// afterglow on the frame after a flash).
    Delay { frames: u32 },
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum About {
    #[default]
    Direction,
    Up,
}

impl About {
    fn is_direction(&self) -> bool {
        *self == Self::Direction
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "module", rename_all = "snake_case", deny_unknown_fields)]
pub enum UpdateModule {
    /// Constant acceleration and linear drag (1/s), exact at any age.
    Ballistic {
        #[serde(default)]
        accel: [f32; 3],
        #[serde(default)]
        drag: f32,
    },
    /// Carried round a circle growing to `radius` metres over the life, at
    /// `turn` radians a second.
    Swirl { radius: f32, turn: f32 },
    /// Carried by the room's shared air: neighbours move together.
    AirDrift {
        #[serde(default = "one", skip_serializing_if = "is_one")]
        strength: f32,
    },
    /// A wander of its own, `reach` metres at `pace` a second.
    Jitter { reach: f32, pace: f32 },
    /// Rising stops under the ceiling found over the emitter, and spreads.
    CeilingSpread {},
    /// Stops against that ceiling.
    CeilingStop {},
    /// The size over life: `to` grows the initial size to that many metres
    /// (eased), or `value` multiplies it (a curve).
    SizeOverLife {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        to: Option<f32>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        value: Option<Value>,
    },
    /// Opacity (over) or strength (screened), 0-1.
    OpacityOverLife { value: Value },
    /// A puff's opacity in closed form: up to `peak` over the first `rise`
    /// of its life (eased), thinning as `(1 - t)^power`.
    PuffFade { peak: f32, rise: f32, power: f32 },
    /// Multiplies its colour: keys `[t, r, g, b, a]`.
    ColourOverLife { gradient: Vec<[f32; 5]> },
    /// A tumbling flake's brightness (dust).
    Glint {},
    /// Lit once a particle a frame by the room's baked light and the lamps
    /// that reach it: `sixway` through a puff's own volume, `scatter` by a
    /// Henyey-Greenstein phase function. A flash in the same effect lights
    /// it on its frame.
    LightCpu {
        mode: LightMode,
        #[serde(default)]
        phase_g: f32,
        albedo: [f32; 3],
        /// The share of the room's even light it is seen by (dust: little).
        #[serde(default = "one", skip_serializing_if = "is_one")]
        ambient: f32,
    },
    /// An emitter's light seen as the eye adapted to it sees it: `radius`
    /// metres of source at scale 1, held at `level` on the tone curve.
    AdaptOwnLight { radius: f32, level: f32 },
    /// One of the effect's inputs scales opacity, size or brightness.
    ByInput { input: String, target: UpdateTarget, curve: Vec<[f32; 2]> },
    /// A spark's glow: hot orange, flickering as it tumbles, fading as it
    /// cools.
    EmberGlow { radiance: f32 },
    /// A fire's sheets of flame standing on its bed (the fused fire module:
    /// with `beat`, `spots_low_discrepancy` and `sprite_flame`).
    FlameSheets { radiance: f32 },
    /// A bed of coals under a fire, breathing with it (with `sprite_bed`).
    CoalBed { glow: f32 },
    /// Pops: bursts of sparks carried up by the plume, so many a second.
    Pops { per_second: f32 },
    /// One part of what something striking water throws (the fused splash
    /// module): its count, size, speed and timing are worked out from what
    /// struck -- its width, mass and velocity (`effects::splash_shape`) --
    /// and the emitter's burst count is the most of it there can be.
    WaterEntry { part: WaterPart },
}

/// What a strike into water throws: the drops its rim breaks into (and its
/// jet's beads), the crown's petals, fans of spray, pieces of white water,
/// the jet up its middle, and a sealed strike's mist.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WaterPart {
    Drops,
    Crown,
    Spray,
    Froth,
    Jet,
    Mist,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UpdateTarget {
    Opacity,
    Size,
    Brightness,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LightMode {
    Sixway,
    Scatter,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "module", rename_all = "snake_case", deny_unknown_fields)]
pub enum RenderModule {
    /// Alpha from a layer, one lit colour.
    SpritePlain { book: String },
    /// Two layers, the light from six sides (smoke).
    SpriteSixway { book: String },
    /// Heat ramp, frames blended along the flow, eroded (fire).
    SpriteFlame { book: String },
    /// A bed of coals.
    SpriteBed { book: String },
    /// The light it scatters (`light_cpu`, `scatter`), screened: it only
    /// adds, so one in shadow is not a dark speck (dust, a splash's drops).
    SpriteScatter { book: String },
    /// Light, screened through the tone curve: `radiance` at its hottest,
    /// coloured by `colour`, or through the flame's heat ramp from the
    /// book's own heat (`heat_ramp`: white-yellow core, orange body, red
    /// thin edges).
    SpriteEmissive {
        book: String,
        radiance: f32,
        #[serde(default = "white", skip_serializing_if = "is_white")]
        colour: [f32; 3],
        #[serde(default)]
        heat_ramp: bool,
        #[serde(default)]
        adapt: Adapt,
    },
    Placement {
        mode: PlacementMode,
        /// A streak covers this many seconds of its motion.
        #[serde(default)]
        streak_seconds: f32,
    },
    /// Which frame of its book: by age (blended toward the next), or one
    /// per particle by its seed, with the book's variants by seed too.
    Flipbook {
        #[serde(default)]
        play: Play,
    },
    /// A MIST LAYER HOVERING IN TUFTS, not particles: one layer over a box
    /// that fills its placed object's footprint (`shape_disc` --
    /// `fills_object` for a level's pond, its radius in the editor), standing
    /// on its origin (the water's surface) and `height` metres up. Soft
    /// cloud-like tufts -- a 3D billow noise some 0.5-2 m across, gathered in
    /// clusters with clear water between -- in a band 3/8 of the height up,
    /// `falloff` metres either side of its middle: clear just over the water,
    /// soft bottom and top. `density` is per metre of sight inside a tuft. It
    /// lies in the ellipse the box's footprint holds (size the object to the
    /// water), thinning to nothing over `edge` metres in from its rim, and
    /// ends where the scene's depth does (a bank, the shore). Summed along
    /// each pixel's line of sight in ten stretches. Drifts with `wind` (m/s,
    /// world), the tufts slowly changing shape as it goes; that drift is its
    /// SpaceWarp vectors. Its density is also its object's `strength` and,
    /// while a time of day runs, the hour's `effects::mist_conditions`. Lit
    /// as `light_cpu` (scatter) works out -- the room's or sky's even light,
    /// its lamps, and the sun by `sunlit` -- a tuft's top brighter than its
    /// underside.
    SpriteFog {
        density: f32,
        height: f32,
        falloff: f32,
        #[serde(default)]
        edge: f32,
        #[serde(default)]
        wind: [f32; 3],
    },
    /// Fades into what it meets over this many metres.
    SoftDepth { metres: f32 },
    /// How SpaceWarp moves its pixels: `bright` for sharp bright layers,
    /// `soft` for those that move themselves only where they hide what is
    /// behind, `none` with the reason.
    Motion {
        layer: MotionLayer,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        reason: Option<String>,
    },
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Adapt {
    /// At the room's exposure: what a source too brief to adapt to is seen
    /// at -- blinding in the dark, faint in the sun.
    #[default]
    Room,
    /// As the eye adapted to this emitter (`adapt_own_light` in its update).
    OwnLight,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlacementMode {
    Facing,
    Upright,
    Streak,
    Flat,
    Point,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Play {
    #[default]
    Age,
    Seed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MotionLayer {
    Bright,
    Soft,
    None,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "module", rename_all = "snake_case", deny_unknown_fields)]
pub enum EventModule {
    /// A LIVE LIGHT for `frames` app frames at the emitter's event: a point
    /// light of `range` metres, unshadowed, in the frame's surface-light
    /// list. `visible_only` leaves it out where it could raise no pixel by
    /// two levels at the room's exposure (every daylight shot).
    LightPulse {
        intensity: LightIntensity,
        range: f32,
        #[serde(default = "one_frame", skip_serializing_if = "is_one_frame")]
        frames: u32,
        #[serde(default)]
        gate: Gate,
        #[serde(default = "white", skip_serializing_if = "is_white")]
        colour: [f32; 3],
    },
}

/// A light's intensity: a number, or a share of an emitter's radiance
/// (`"flash.radiance"`), so the light follows the flash it is the light of.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum LightIntensity {
    Number(f32),
    From(IntensityFrom),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IntensityFrom {
    pub from: String,
    pub scale: f32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Gate {
    #[default]
    VisibleOnly,
    Always,
}

fn one() -> f32 {
    1.0
}
fn is_one(v: &f32) -> bool {
    *v == 1.0
}
fn one_frame() -> u32 {
    1
}
fn is_one_frame(v: &u32) -> bool {
    *v == 1
}
fn white() -> [f32; 3] {
    [1.0; 3]
}
fn is_white(v: &[f32; 3]) -> bool {
    *v == [1.0; 3]
}

// ---------------------------------------------------------------------------
// Books.

/// A baked book as an asset names it: a set of atlas layers. Filled in by
/// the renderer (`effects::books`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BookInfo {
    pub name: &'static str,
    pub first_layer: u32,
    /// Frames one variant plays through.
    pub frames: u32,
    pub variants: u32,
    pub channels: Channels,
    /// The share of its quad a frame's cutout covers, mean over the book:
    /// what a particle pays fill for.
    pub cutout_share: f32,
    /// Every layer of it with its mips, bytes.
    pub bytes: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Channels {
    /// Two layers a frame: the light passed from six sides, and opacity.
    Sixway,
    /// Heat in red, cover in alpha.
    HeatCover,
    Alpha,
    Bed,
}

// ---------------------------------------------------------------------------
// Checking.

#[derive(Clone, Debug, PartialEq)]
pub struct Issue {
    /// An error refuses the asset; a budget error does not when the asset
    /// carries `over_budget`.
    pub budget: bool,
    pub emitter: Option<String>,
    pub message: String,
}

impl std::fmt::Display for Issue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.emitter {
            Some(e) => write!(f, "emitter `{e}`: {}", self.message),
            None => write!(f, "{}", self.message),
        }
    }
}

/// What one emitter adds up to, worst case, at the effect's reference view.
#[derive(Clone, Debug, PartialEq)]
pub struct EmitterCost {
    pub name: String,
    /// Live particles from one burst (or the placed emitter), worst case.
    pub instances: u32,
    /// Seconds its longest-lived particle lasts (0 for one frame).
    pub longest_life: f32,
    /// Fill of those particles, megapixel-layers, at the reference view: a
    /// burst's every particle at its reference distance (held in the hand);
    /// a placed effect's seen from its stated view (`effects::seen`).
    pub fill: f32,
    /// Spawned at a rate: one stream per source however many bursts are
    /// alive (a newer burst takes the stream over), so it is not multiplied
    /// by `max_live`.
    pub stream: bool,
    /// What its particles draw, worst case, square metres within their
    /// cutouts, whatever the distance: turned to the eye, or lying level
    /// (`placement: flat`). With `points`, what a frame's fill is worked out
    /// from where the eyes are (`effects::seen`).
    pub area: f32,
    pub flat: bool,
    /// Particles drawn at the 1.5 pixel floor (`placement: point`).
    pub points: u32,
    /// Where its particles get to from its origin (its up the emitter's).
    pub extent: Extent,
    /// The radius of a disc that fills its placed object (`shape_disc`
    /// `fills_object`), 0 for none.
    pub fills_disc: f32,
    /// A fog slab (`sprite_fog`): ONE layer over its box, whatever the view.
    pub volume: bool,
}

/// WHERE AN EFFECT'S PARTICLES GET TO from its origin, metres, its up the
/// emitter's direction: out to the side, down, up (`stack::emitter_extent`).
/// The box its fill is seen through (`effects::seen`).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Extent {
    pub lateral: f32,
    pub below: f32,
    pub above: f32,
}

impl Extent {
    pub fn union(self, o: Self) -> Self {
        Self { lateral: self.lateral.max(o.lateral), below: self.below.max(o.below), above: self.above.max(o.above) }
    }
}

/// What an effect costs, from its stacks alone, and what it was held to.
#[derive(Clone, Debug, PartialEq)]
pub struct Report {
    pub emitters: Vec<EmitterCost>,
    /// One burst alone (or the placed effect).
    pub instances_per_burst: u32,
    pub fill_per_burst: f32,
    /// One source's bursts at the `max_live` cap: the most this effect
    /// keeps alive and fills from one weapon, whatever the rate of fire.
    pub instances_steady: u32,
    pub fill_steady: f32,
    pub lights: u32,
    pub texture_bytes: u64,
    pub reference_distance: f32,
    /// Whether any of its emitters is spawned by an event.
    pub burst: bool,
    /// HOW ITS FILL WAS STATED. A burst: every particle at
    /// `reference_distance` (held in the hand). A placed effect: seen
    /// `reference_distance` off the side of its box, standing (`seen`); the
    /// frame's governor weighs it from the eyes of each frame.
    pub seen_from_side: bool,
    /// Every emitter's box together, and their particles' area at the cap
    /// (turned to the eye; lying level) and points: what the frame weighs.
    pub extent: Extent,
    pub area_facing: f32,
    pub area_flat: f32,
    pub points: u32,
    /// Fog slabs: each one layer over the box.
    pub volumes: u32,
    /// The reason it may exceed a budget, and the budget errors it then
    /// carries.
    pub over_budget: Option<(String, Vec<String>)>,
}

impl EffectAsset {
    /// An asset from its JSON text. Unknown modules and parameters are
    /// errors. A file of an older `version` is upgraded here.
    pub fn parse(text: &str) -> Result<Self, String> {
        let asset: Self = serde_json::from_str(text).map_err(|e| e.to_string())?;
        match asset.version {
            ASSET_VERSION => Ok(asset),
            v => Err(format!("effect `{}`: version {v} is not one this build reads (it writes {ASSET_VERSION})", asset.name)),
        }
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("an effect asset is plain data")
    }

    /// Every effect in a folder (`game/effects`): `<name>.json`, the name
    /// inside matching the file's. Files that will not parse are returned
    /// with why; `module_costs.json` and names beginning `_` are not
    /// effects. A missing folder is no effects.
    pub fn load_dir(dir: &std::path::Path) -> (Vec<Self>, Vec<(std::path::PathBuf, String)>) {
        let (mut assets, mut failed) = (Vec::new(), Vec::new());
        let Ok(entries) = std::fs::read_dir(dir) else { return (assets, failed) };
        let mut paths: Vec<_> = entries.filter_map(|e| e.ok().map(|e| e.path())).collect();
        paths.sort();
        for path in paths {
            let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("").to_string();
            if path.extension().and_then(|e| e.to_str()) != Some("json") || stem.starts_with('_') || stem == "module_costs" {
                continue;
            }
            match std::fs::read_to_string(&path).map_err(|e| e.to_string()).and_then(|t| Self::parse(&t)) {
                Ok(a) if a.name == stem => assets.push(a),
                Ok(a) => failed.push((path, format!("the effect inside is named `{}`, not `{stem}`", a.name))),
                Err(e) => failed.push((path, e)),
            }
        }
        (assets, failed)
    }

    /// A burst's energy as it is drawn: 1 where the asset names none.
    pub fn energy(&self) -> Value {
        self.energy.clone().unwrap_or(Value::Number(1.0))
    }

    /// The index of a named input.
    pub fn input(&self, name: &str) -> Option<usize> {
        self.inputs.iter().position(|i| i == name)
    }

    /// CHECK IT, and say what it costs: its structure (each emitter exactly
    /// one spawn, one life, one placement, one sprite, one motion; modules
    /// in an order they can run in), its values, and its budgets against
    /// what it declares and against the hard limits. `books` are the books
    /// that exist.
    pub fn check(&self, books: &[BookInfo]) -> Result<Report, Vec<Issue>> {
        let mut issues: Vec<Issue> = Vec::new();
        let mut note = |budget: bool, emitter: Option<&str>, message: String| issues.push(Issue { budget, emitter: emitter.map(str::to_string), message });
        if self.name.is_empty() || !self.name.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_') {
            note(false, None, format!("an effect's name is lower case letters, digits and `_`; `{}` is not", self.name));
        }
        if self.inputs.len() > MAX_INPUTS {
            note(false, None, format!("{} inputs; an effect takes at most {MAX_INPUTS}", self.inputs.len()));
        }
        if self.emitters.is_empty() {
            note(false, None, "an effect needs an emitter".into());
        }
        if let Some(e) = &self.energy {
            let (lo, hi) = e.bounds();
            if !(lo > 0.0 && hi <= 5.0) || matches!(e, Value::Curve(_)) {
                note(false, None, format!("energy runs from {lo} to {hi}; it is a draw over 0 and at most 5"));
            }
        }
        let s = &self.scalability;
        if !(s.cull_distance > 0.0) || !(0.0..=1.0).contains(&s.importance) || s.max_live == 0 || !(s.min_pixels >= 0.0) {
            note(false, None, "scalability: cull_distance over 0, importance 0-1, max_live at least 1, min_pixels 0 or more".into());
        }
        if let Some([a, b]) = s.near_fade {
            if !(a >= 0.0 && b > a) {
                note(false, None, format!("scalability: near_fade [{a}, {b}] fades from the second distance in to the first"));
            }
        }
        let burst = self.emitters.iter().any(|e| e.spawn.iter().any(|m| matches!(m, SpawnModule::Burst { .. })));
        let reference = s.reference_distance.unwrap_or(if burst { REFERENCE_HELD } else { REFERENCE_PLACED }).max(0.05);
        let mut costs = Vec::new();
        let mut lights = 0u32;
        let mut texture: Vec<&'static str> = Vec::new();
        for e in &self.emitters {
            let mut local: Vec<(bool, String)> = Vec::new();
            let cost = self.check_emitter(e, books, reference, &mut local, &mut texture);
            lights += e.events.iter().filter(|m| matches!(m, EventModule::LightPulse { .. })).count() as u32;
            for (budget, message) in local {
                note(budget, Some(&e.name), message);
            }
            costs.push(cost);
        }
        let mut names: Vec<&str> = self.emitters.iter().map(|e| e.name.as_str()).collect();
        names.sort();
        if names.windows(2).any(|w| w[0] == w[1]) {
            note(false, None, "two emitters share a name".into());
        }
        // A PLACED EFFECT'S FILL, seen from beside it: its box from its
        // stated view, times each emitter's layers (`effects::seen`) -- not
        // every particle at one distance, which priced a wide mist's far
        // sheets as near ones and past the edges of the view.
        let extent = costs.iter().fold(Extent::default(), |a, c| a.union(c.extent));
        if !burst {
            let (lo, hi, eyes) = crate::renderer::effects::seen::stated_view(extent.lateral, extent.below, extent.above, reference);
            for c in costs.iter_mut() {
                let (facing, flat) = if c.flat { (0.0, c.area) } else { (c.area, 0.0) };
                c.fill = crate::renderer::effects::seen::seen_fill(&eyes, lo, hi, facing, flat, c.points as f32, if c.volume { 1.0 } else { 0.0 });
            }
        }
        let instances_per_burst: u32 = costs.iter().map(|c| c.instances).sum();
        let fill_per_burst: f32 = costs.iter().map(|c| c.fill).sum();
        // AT THE CAP: an emitter whose particles outlive the event keeps up
        // to `max_live` bursts' worth alive; one-frame particles only ever
        // their own burst's.
        let held = |c: &EmitterCost| if burst && c.longest_life > 0.0 && !c.stream { s.max_live.max(1) } else { 1 };
        let instances_steady: u32 = costs.iter().map(|c| c.instances * held(c)).sum();
        let fill_steady: f32 = costs.iter().map(|c| c.fill * held(c) as f32).sum();
        let area_facing: f32 = costs.iter().filter(|c| !c.flat).map(|c| c.area * held(c) as f32).sum();
        let area_flat: f32 = costs.iter().filter(|c| c.flat).map(|c| c.area * held(c) as f32).sum();
        let points: u32 = costs.iter().map(|c| c.points * held(c)).sum();
        let volumes = costs.iter().filter(|c| c.volume).count() as u32;
        let texture_bytes: u64 = texture.iter().filter_map(|n| books.iter().find(|b| b.name == *n)).map(|b| b.bytes).sum();
        let mut over = |what: String| note(true, None, what);
        if instances_steady > self.budget.instances {
            over(format!("{instances_steady} live instances at its cap; it budgets {}", self.budget.instances));
        }
        if self.budget.instances > LIMIT_INSTANCES {
            over(format!("it budgets {} instances; the limit is {LIMIT_INSTANCES}", self.budget.instances));
        }
        if fill_steady > self.budget.fill_megapixel_layers {
            let seen = if burst { format!("at {reference} m") } else { format!("seen from {reference} m off its side") };
            over(format!("fill {fill_steady:.2} megapixel-layers {seen} at its cap; it budgets {}", self.budget.fill_megapixel_layers));
        }
        if self.budget.fill_megapixel_layers > LIMIT_FILL_STEADY {
            over(format!("it budgets {} megapixel-layers; the limit is {LIMIT_FILL_STEADY}", self.budget.fill_megapixel_layers));
        }
        if burst && fill_per_burst > LIMIT_FILL_BURST {
            over(format!("one burst fills {fill_per_burst:.2} megapixel-layers at {reference} m; the limit is {LIMIT_FILL_BURST}"));
        }
        if lights > self.budget.lights {
            over(format!("{lights} live lights; it budgets {}", self.budget.lights));
        }
        if self.budget.lights > LIMIT_LIGHTS {
            over(format!("it budgets {} lights; the limit is {LIMIT_LIGHTS}", self.budget.lights));
        }
        if texture_bytes > LIMIT_TEXTURE_BYTES {
            over(format!("its books are {texture_bytes} bytes; a level's limit is {LIMIT_TEXTURE_BYTES}"));
        }
        if burst && s.near_fade.is_none() && reference < 4.0 * NEAR_EYE {
            note(false, None, format!("a burst that can spawn within {NEAR_EYE} m of the eye needs scalability.near_fade"));
        }
        if matches!(&self.over_budget, Some(r) if r.trim().is_empty()) {
            note(false, None, "over_budget needs its reason".into());
        }
        let over_budget: Vec<String> = issues.iter().filter(|i| i.budget).map(|i| i.to_string()).collect();
        let fatal: Vec<Issue> = issues.iter().filter(|i| !i.budget || self.over_budget.is_none()).cloned().collect();
        if !fatal.is_empty() {
            return Err(fatal);
        }
        Ok(Report {
            emitters: costs,
            instances_per_burst,
            fill_per_burst,
            instances_steady,
            fill_steady,
            lights,
            texture_bytes,
            reference_distance: reference,
            burst,
            seen_from_side: !burst,
            extent,
            area_facing,
            area_flat,
            points,
            volumes,
            over_budget: self.over_budget.clone().map(|r| (r, over_budget)),
        })
    }

    fn check_emitter(&self, e: &EmitterDef, books: &[BookInfo], reference: f32, issues: &mut Vec<(bool, String)>, texture: &mut Vec<&'static str>) -> EmitterCost {
        let mut bad = |m: String| issues.push((false, m));
        let mut values: Vec<String> = Vec::new();
        let known_input = |name: &str| self.input(name).is_some();
        // SPAWN: exactly one source.
        let sources = e.spawn.iter().filter(|m| !matches!(m, SpawnModule::ByInput { .. })).count();
        if sources != 1 {
            bad(format!("needs exactly one of rate, burst, density or beat in spawn; it has {sources}"));
        }
        let mut count = 0.0f32;
        let mut rate: Option<(f32, Option<f32>)> = None;
        for m in &e.spawn {
            match m {
                SpawnModule::Burst { count: c, chance } => {
                    c.check("burst count", 0.0, &mut values);
                    count = c.bounds().1.ceil();
                    if !(0.0..=1.0).contains(chance) {
                        bad(format!("burst chance {chance} is not 0-1"));
                    }
                }
                SpawnModule::Rate { per_second, for_seconds } => {
                    if !(*per_second > 0.0) || for_seconds.is_some_and(|s| !(s > 0.0)) {
                        bad("rate: per_second and for_seconds over 0".into());
                    }
                    rate = Some((*per_second, *for_seconds));
                }
                SpawnModule::Density { per_cubic_metre } if !(*per_cubic_metre > 0.0) => bad("density over 0".into()),
                SpawnModule::Beat { per_second } => {
                    if !(*per_second > 0.0) {
                        bad("beat over 0".into());
                    }
                    rate = Some((*per_second, None));
                }
                SpawnModule::ByInput { input, curve, target } => {
                    if !known_input(input) {
                        bad(format!("by_input names `{input}`, which is not one of the effect's inputs"));
                    }
                    check_keys(curve.iter().map(|k| k[0]), "by_input", &mut values);
                    let most = curve.iter().map(|k| k[1]).fold(0.0f32, f32::max);
                    match target {
                        SpawnTarget::Count => count *= most.max(1.0),
                        SpawnTarget::Rate => rate = rate.map(|(r, d)| (r * most.max(1.0), d)),
                        SpawnTarget::Chance => {}
                    }
                }
                _ => {}
            }
        }
        // INITIALIZE.
        let lives: Vec<&LifeValue> = e.initialize.iter().filter_map(|m| if let InitModule::Life { value } = m { Some(value) } else { None }).collect();
        if lives.len() != 1 {
            bad(format!("needs exactly one life; it has {}", lives.len()));
        }
        let (mut longest, mut one_frame) = (0.0f32, false);
        match lives.first() {
            Some(LifeValue::Named(LifeName::OneFrame)) => one_frame = true,
            Some(LifeValue::Seconds(v)) => {
                v.check("life", 0.01, &mut values);
                if matches!(v, Value::Curve(_)) {
                    bad("a life is a number or a range, not a curve".into());
                }
                longest = v.bounds().1;
            }
            None => {}
        }
        let mut size: Option<&Value> = None;
        let mut lobe_length: Option<&Value> = None;
        let mut shapes = 0;
        for m in &e.initialize {
            match m {
                InitModule::Size { value } => {
                    value.check("size", 1e-5, &mut values);
                    size = Some(value);
                }
                InitModule::Speed { value } => value.check("speed", 0.0, &mut values),
                InitModule::ShapeLobes { lobes, open_deg, side_pairs, length, .. } => {
                    shapes += 1;
                    length.check("lobe length", 1e-4, &mut values);
                    lobe_length = Some(length);
                    if *lobes > 12 || *side_pairs > 4 || !(0.0..=120.0).contains(open_deg) {
                        bad("shape_lobes: at most 12 lobes and 4 side pairs, opened 0-120 degrees".into());
                    }
                }
                InitModule::ShapeCone { angle_deg, .. } if !(0.0..=180.0).contains(angle_deg) => bad("shape_cone: 0-180 degrees".into()),
                InitModule::ShapeDisc { radius, .. } if !(*radius >= 0.0) => bad("shape_disc: radius 0 or more".into()),
                InitModule::ShapeDisc { radius, fills_object: true } if !(*radius > 0.0) => bad("shape_disc: a disc that fills its object needs a radius over 0".into()),
                InitModule::ShapePoint {} | InitModule::ShapeDisc { .. } | InitModule::ShapeBox {} | InitModule::SpotsLowDiscrepancy { .. } => shapes += 1,
                InitModule::Delay { frames } if *frames > 8 => bad("delay: at most 8 frames".into()),
                _ => {}
            }
        }
        if shapes > 1 {
            bad(format!("{shapes} shapes of birth; an emitter has one (a cone is a direction, and may join one)"));
        }
        // UPDATE: values, inputs, and an order the steps can run in -- what
        // reads a particle's place or size comes after all that move or
        // size it, and what scales its light after what lights it.
        let mut last_mover = None;
        let mut first_reader = None;
        let mut grow: Option<f32> = None;
        let mut fused = (false, false);
        for (k, m) in e.update.iter().enumerate() {
            match m {
                UpdateModule::Ballistic { .. } | UpdateModule::Swirl { .. } | UpdateModule::AirDrift { .. } | UpdateModule::Jitter { .. } | UpdateModule::CeilingSpread {} | UpdateModule::CeilingStop {} => {
                    last_mover = Some(k)
                }
                UpdateModule::SizeOverLife { to, value } => {
                    last_mover = Some(k);
                    match (to, value) {
                        (Some(to), None) => grow = Some(grow.unwrap_or(0.0).max(*to)),
                        (None, Some(v)) => {
                            v.check("size_over_life", 0.0, &mut values);
                            grow = Some(-v.bounds().1);
                        }
                        _ => bad("size_over_life takes `to` (metres) or `value` (a multiplier), one of them".into()),
                    }
                }
                UpdateModule::LightCpu { phase_g, .. } => {
                    first_reader.get_or_insert(k);
                    if !(-0.95..=0.95).contains(phase_g) {
                        bad("light_cpu: phase_g within 0.95 of 0".into());
                    }
                }
                UpdateModule::OpacityOverLife { value } => value.check("opacity_over_life", 0.0, &mut values),
                UpdateModule::PuffFade { peak, rise, power } if !(*peak > 0.0 && *peak <= 1.0 && *rise > 0.0 && *rise < 1.0 && *power > 0.0) => {
                    bad("puff_fade: peak 0-1, rise inside the life, power over 0".into())
                }
                UpdateModule::ColourOverLife { gradient } => check_keys(gradient.iter().map(|k| k[0]), "colour_over_life", &mut values),
                UpdateModule::ByInput { input, curve, .. } => {
                    if !known_input(input) {
                        bad(format!("by_input names `{input}`, which is not one of the effect's inputs"));
                    }
                    check_keys(curve.iter().map(|k| k[0]), "by_input", &mut values);
                }
                UpdateModule::FlameSheets { .. } => fused.0 = true,
                UpdateModule::CoalBed { .. } => fused.1 = true,
                _ => {}
            }
        }
        if let (Some(mover), Some(reader)) = (last_mover, first_reader) {
            if reader < mover {
                bad("light_cpu lights a particle where it is: it comes after every module that moves or sizes it".into());
            }
        }
        // RENDER: one sprite, one placement, one motion.
        let sprites: Vec<&RenderModule> = e
            .render
            .iter()
            .filter(|m| {
                matches!(
                    m,
                    RenderModule::SpritePlain { .. }
                        | RenderModule::SpriteSixway { .. }
                        | RenderModule::SpriteFlame { .. }
                        | RenderModule::SpriteBed { .. }
                        | RenderModule::SpriteScatter { .. }
                        | RenderModule::SpriteEmissive { .. }
                        | RenderModule::SpriteFog { .. }
                )
            })
            .collect();
        let fog = sprites.first().and_then(|m| if let RenderModule::SpriteFog { density, height, falloff, edge, .. } = m { Some((*density, *height, *falloff, *edge)) } else { None });
        let placements: Vec<PlacementMode> = e.render.iter().filter_map(|m| if let RenderModule::Placement { mode, .. } = m { Some(*mode) } else { None }).collect();
        let motions: Vec<(&MotionLayer, &Option<String>)> = e.render.iter().filter_map(|m| if let RenderModule::Motion { layer, reason } = m { Some((layer, reason)) } else { None }).collect();
        let soft = e.render.iter().any(|m| matches!(m, RenderModule::SoftDepth { metres } if *metres > 0.0));
        for (what, n) in [("sprite", sprites.len()), ("placement", placements.len()), ("motion", motions.len())] {
            if what == "placement" && fog.is_some() {
                if n != 0 {
                    bad("a fog slab (sprite_fog) is its box: it takes no placement".into());
                }
                continue;
            }
            if n != 1 {
                bad(format!("needs exactly one {what} module in render; it has {n}"));
            }
        }
        if let Some((density, height, falloff, edge)) = fog {
            if !(density > 0.0 && density <= 20.0 && height > 0.0 && falloff > 0.0 && edge >= 0.0) {
                bad("sprite_fog: density over 0 (at most 20 a metre), height and falloff over 0, edge 0 or more".into());
            }
            if !e.update.iter().any(|m| matches!(m, UpdateModule::LightCpu { mode: LightMode::Scatter, .. })) {
                bad("sprite_fog is lit as light_cpu (scatter) works out; the stack has none".into());
            }
            if !e.initialize.iter().any(|m| matches!(m, InitModule::ShapeDisc { radius, .. } if *radius > 0.0)) {
                bad("sprite_fog spans its shape_disc (radius over 0; fills_object for a level's box)".into());
            }
            if !e.spawn.iter().any(|m| matches!(m, SpawnModule::Rate { .. })) {
                bad("sprite_fog stands while its effect is placed: its spawn is a rate (one slab, whatever the rate)".into());
            }
        }
        if let Some((MotionLayer::None, reason)) = motions.first() {
            if reason.as_deref().map_or(true, |r| r.trim().is_empty()) {
                bad("motion `none` needs its reason: every moving draw writes SpaceWarp motion vectors".into());
            }
        }
        let mut share = 1.0;
        let mut screened = false;
        if let Some(sprite) = sprites.first().filter(|_| fog.is_none()) {
            let (book, wants) = match sprite {
                RenderModule::SpritePlain { book } => (book, None),
                RenderModule::SpriteSixway { book } => (book, Some(Channels::Sixway)),
                RenderModule::SpriteFlame { book } => (book, Some(Channels::HeatCover)),
                RenderModule::SpriteBed { book } => (book, Some(Channels::Bed)),
                RenderModule::SpriteScatter { book } => {
                    if !e.update.iter().any(|m| matches!(m, UpdateModule::LightCpu { mode: LightMode::Scatter, .. })) {
                        bad("sprite_scatter draws the light light_cpu (scatter) works out; the stack has none".into());
                    }
                    (book, None)
                }
                RenderModule::SpriteEmissive { book, radiance, heat_ramp, adapt, .. } => {
                    screened = true;
                    if !(*radiance > 0.0) {
                        bad("sprite_emissive: radiance over 0".into());
                    }
                    if *adapt == Adapt::OwnLight && !e.update.iter().any(|m| matches!(m, UpdateModule::AdaptOwnLight { .. })) {
                        bad("sprite_emissive adapts to its own light, but the stack has no adapt_own_light".into());
                    }
                    (book, heat_ramp.then_some(Channels::HeatCover))
                }
                _ => unreachable!(),
            };
            match books.iter().find(|b| b.name == book.as_str()) {
                None => bad(format!("names the book `{book}`, which does not exist (books: {})", books.iter().map(|b| b.name).collect::<Vec<_>>().join(", "))),
                Some(b) => {
                    if wants.is_some_and(|w| w != b.channels) {
                        bad(format!("its sprite module reads {:?} channels; the book `{book}` holds {:?}", wants.unwrap(), b.channels));
                    }
                    if b.cutout_share > 0.98 {
                        bad(format!("the book `{book}`'s cutout is its whole quad: it pays fill for transparent corners"));
                    }
                    share = b.cutout_share;
                    if !texture.contains(&b.name) {
                        texture.push(b.name);
                    }
                }
            }
            if matches!(sprite, RenderModule::SpriteSixway { .. }) && !e.update.iter().any(|m| matches!(m, UpdateModule::LightCpu { mode: LightMode::Sixway, .. })) {
                bad("sprite_sixway draws the light light_cpu (sixway) works out; the stack has none".into());
            }
            if fused.0 != matches!(sprite, RenderModule::SpriteFlame { .. }) || fused.1 != matches!(sprite, RenderModule::SpriteBed { .. }) {
                bad("flame_sheets goes with sprite_flame, and coal_bed with sprite_bed".into());
            }
        }
        if !soft && fog.is_none() && !matches!(placements.first(), Some(PlacementMode::Point | PlacementMode::Flat)) {
            bad("a sprite that can touch geometry needs soft_depth, or it ends in a hard line where it meets a wall".into());
        }
        if screened && e.events.iter().any(|m| matches!(m, EventModule::LightPulse { .. })) && one_frame {
            // A one-frame source is seen at the room's exposure and its light
            // with it: nothing to adapt.
        } else if screened
            && e.events.iter().any(|m| matches!(m, EventModule::LightPulse { .. }))
            && !e.update.iter().any(|m| matches!(m, UpdateModule::AdaptOwnLight { .. }))
        {
            bad("an emissive emitter that also raises a lasting light needs adapt_own_light, or what it lights burns white".into());
        }
        for m in &e.events {
            let EventModule::LightPulse { intensity, range, frames, .. } = m;
            if !(*range > 0.0 && *range <= 8.0) || *frames == 0 || *frames > 4 {
                bad("light_pulse: range up to 8 m, 1-4 frames".into());
            }
            if let LightIntensity::From(f) = intensity {
                let known = self.emitters.iter().any(|o| {
                    f.from == format!("{}.radiance", o.name) && o.render.iter().any(|r| matches!(r, RenderModule::SpriteEmissive { .. }))
                });
                if !known {
                    bad(format!("light_pulse takes its intensity from `{}`: no emitter of that name has a sprite_emissive", f.from));
                }
            }
        }
        for v in values {
            bad(v);
        }
        // WHAT IT ADDS UP TO. Instances: a burst's count, or a rate's
        // particles alive at once. Fill: each particle's mean drawn area at
        // the reference view, both eyes.
        let instances = match rate {
            _ if fog.is_some() => 1.0,
            Some((per_second, for_seconds)) => (per_second * for_seconds.map_or(longest, |d| d.min(longest))).ceil().max(1.0),
            None => count,
        }
        .max(0.0) as u32;
        let half = size.map_or(0.0, |s| s.mean());
        let mean_half = match grow {
            // Eased from its first size to `to`: the mean of the squares.
            Some(to) if to > 0.0 => ((half * half + half * to + to * to) / 3.0).sqrt(),
            Some(scale) if scale < 0.0 => half * (-scale).max(1.0) * 0.75,
            _ => half,
        };
        let across = 2.0 * mean_half / reference * REFERENCE_FOCAL_PIXELS;
        // A streak is as long as its speed for `streak_seconds`: its launch,
        // and what its acceleration has brought it to against its drag by
        // the middle of its life.
        let streak_metres = e.render.iter().find_map(|m| match m {
            RenderModule::Placement { mode: PlacementMode::Streak, streak_seconds } => {
                let launch = e.initialize.iter().find_map(|m| if let InitModule::Speed { value } = m { Some(value.mean()) } else { None }).unwrap_or(0.0);
                let gained = e.update.iter().find_map(|m| match m {
                    UpdateModule::Ballistic { accel, drag } => {
                        let a = (accel[0] * accel[0] + accel[1] * accel[1] + accel[2] * accel[2]).sqrt();
                        Some(if *drag > 1e-4 { a / drag * (1.0 - (-drag * 0.5 * longest).exp()) } else { a * 0.5 * longest })
                    }
                    _ => None,
                });
                Some(*streak_seconds * (launch + gained.unwrap_or(0.0)))
            }
            _ => None,
        });
        let along = match (lobe_length, placements.first(), streak_metres) {
            (Some(l), _, _) => (l.mean() / reference * REFERENCE_FOCAL_PIXELS).max(across),
            (None, Some(PlacementMode::Point), _) => 1.5,
            (None, _, Some(m)) => (m / reference * REFERENCE_FOCAL_PIXELS).max(across),
            _ => across,
        };
        let across = if matches!(placements.first(), Some(PlacementMode::Point)) { 1.5 } else { across };
        let fill = instances as f32 * across * along * share * REFERENCE_EYES / 1e6;
        // The same area, whatever the distance (`across` and `along` in
        // metres): what the frame's governor sees through.
        let point = matches!(placements.first(), Some(PlacementMode::Point));
        let metres = |px: f32| px * reference / REFERENCE_FOCAL_PIXELS;
        let area = if point || fog.is_some() { 0.0 } else { instances as f32 * metres(across) * metres(along) * share };
        let fills_disc = e.initialize.iter().find_map(|m| if let InitModule::ShapeDisc { radius, fills_object: true } = m { Some(*radius) } else { None }).unwrap_or(0.0);
        EmitterCost {
            name: e.name.clone(),
            instances,
            longest_life: if one_frame { 0.0 } else { longest },
            fill,
            stream: rate.is_some(),
            area,
            flat: matches!(placements.first(), Some(PlacementMode::Flat)),
            points: if point { instances } else { 0 },
            extent: crate::renderer::effects::stack::emitter_extent(e),
            fills_disc,
            volume: fog.is_some(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(crate) const BOOKS: &[BookInfo] = &[
        BookInfo { name: "smoke", first_layer: 0, frames: 16, variants: 1, channels: Channels::Sixway, cutout_share: 0.7, bytes: 2_800_000 },
        BookInfo { name: "muzzle_flash", first_layer: 113, frames: 3, variants: 4, channels: Channels::HeatCover, cutout_share: 0.5, bytes: 1_200_000 },
        BookInfo { name: "ember", first_layer: 80, frames: 1, variants: 1, channels: Channels::Alpha, cutout_share: 0.6, bytes: 87_000 },
    ];

    fn flash() -> String {
        r#"{
          "name": "t", "version": 1, "inputs": ["heat"],
          "emitters": [{
            "name": "flash",
            "spawn": [{ "module": "burst", "count": 6, "chance": 0.8 }],
            "initialize": [
              { "module": "life", "value": "one_frame" },
              { "module": "shape_lobes", "lobes": 5, "open_deg": 42, "skip_down": true, "length": { "lognormal": [0.08, 0.45] } },
              { "module": "size", "value": 0.012 }],
            "update": [],
            "render": [
              { "module": "sprite_emissive", "book": "muzzle_flash", "radiance": 900, "heat_ramp": true },
              { "module": "placement", "mode": "streak" },
              { "module": "soft_depth", "metres": 0.03 },
              { "module": "motion", "layer": "bright" }],
            "events": [{ "module": "light_pulse", "intensity": { "from": "flash.radiance", "scale": 0.02 }, "range": 4.0 }]
          }],
          "scalability": { "cull_distance": 60, "min_pixels": 2, "importance": 0.9, "max_live": 4, "near_fade": [0.15, 0.35] },
          "budget": { "fill_megapixel_layers": 0.9, "instances": 13, "lights": 1 }
        }"#
        .to_string()
    }

    /// An asset reads, checks, and writes back what it read.
    #[test]
    fn an_asset_reads_checks_and_round_trips() {
        let a = EffectAsset::parse(&flash()).unwrap();
        let report = a.check(BOOKS).unwrap_or_else(|i| panic!("{}", i.iter().map(|i| i.to_string()).collect::<Vec<_>>().join("; ")));
        assert_eq!((report.instances_per_burst, report.lights, report.burst), (6, 1, true));
        assert_eq!(report.instances_steady, 6, "one-frame particles are only ever their own burst's");
        assert!(report.fill_per_burst > 0.0 && report.fill_per_burst < 0.2, "{}", report.fill_per_burst);
        assert_eq!(EffectAsset::parse(&a.to_json()).unwrap(), a);
        assert_eq!(a.input("heat"), Some(0));
    }

    /// A typo is an error that names itself: an unknown module, an unknown
    /// parameter of a module, of a value, of the scalability block.
    #[test]
    fn a_typo_is_a_load_error_not_a_default() {
        for (from, to, says) in [
            ("\"module\": \"burst\"", "\"module\": \"brust\"", "brust"),
            ("\"chance\": 0.8", "\"chanse\": 0.8", "chanse"),
            ("\"skip_down\": true", "\"skip_dwon\": true", "skip_dwon"),
            ("\"min_pixels\": 2", "\"min_pixel\": 2", "min_pixel"),
            ("\"module\": \"placement\", \"mode\": \"streak\"", "\"module\": \"placement\", \"mode\": \"streak\", \"strek\": 1", "strek"),
            ("\"version\": 1", "\"version\": 7", "version 7"),
        ] {
            let text = flash().replace(from, to);
            assert_ne!(text, flash(), "{from}");
            let e = EffectAsset::parse(&text).expect_err(to);
            assert!(e.contains(says), "{to}: {e}");
        }
        // A value with a key too many is no value at all.
        let text = flash().replace("{ \"lognormal\": [0.08, 0.45] }", "{ \"lognormal\": [0.08, 0.45], \"sigma\": 2 }");
        assert!(EffectAsset::parse(&text).is_err());
    }

    /// What must be there is checked, and each failure names its emitter.
    #[test]
    fn a_stack_missing_a_required_module_is_refused() {
        let cases = [
            ("{ \"module\": \"life\", \"value\": \"one_frame\" },", "", "exactly one life"),
            ("{ \"module\": \"motion\", \"layer\": \"bright\" }", "{ \"module\": \"motion\", \"layer\": \"none\" }", "reason"),
            ("{ \"module\": \"placement\", \"mode\": \"streak\" },", "", "exactly one placement"),
            ("{ \"module\": \"soft_depth\", \"metres\": 0.03 },", "", "soft_depth"),
            ("\"book\": \"muzzle_flash\"", "\"book\": \"muzle_flash\"", "does not exist"),
            ("\"book\": \"muzzle_flash\"", "\"book\": \"ember\"", "HeatCover"),
            ("\"from\": \"flash.radiance\"", "\"from\": \"flsh.radiance\"", "flsh.radiance"),
            ("\"near_fade\": [0.15, 0.35]", "\"reference_distance\": 0.7", "near_fade"),
            ("\"lights\": 1", "\"lights\": 0", "live lights"),
            ("\"count\": 6", "\"count\": 60", "live instances"),
        ];
        for (from, to, says) in cases {
            let text = flash().replace(from, to);
            assert_ne!(text, flash(), "{from}");
            let a = EffectAsset::parse(&text).unwrap_or_else(|e| panic!("{to}: {e}"));
            let issues = a.check(BOOKS).expect_err(to);
            let all = issues.iter().map(|i| i.to_string()).collect::<Vec<_>>().join("; ");
            assert!(all.contains(says), "{to}: {all}");
        }
    }

    /// An effect over a budget is refused, unless it says why -- and then it
    /// is listed with what it exceeds. A structural error is never excused.
    #[test]
    fn a_budget_is_a_refusal_unless_overridden_with_a_reason() {
        let over = flash().replace("\"count\": 6", "\"count\": 60");
        assert!(EffectAsset::parse(&over).unwrap().check(BOOKS).is_err());
        let excused = over.replace("\"version\": 1,", "\"version\": 1, \"over_budget\": \"a hero shot for the trailer\",");
        let report = EffectAsset::parse(&excused).unwrap().check(BOOKS).unwrap();
        let (reason, what) = report.over_budget.unwrap();
        assert_eq!(reason, "a hero shot for the trailer");
        assert!(what.iter().any(|w| w.contains("live instances")), "{what:?}");
        let broken = excused.replace("{ \"module\": \"life\", \"value\": \"one_frame\" },", "");
        assert!(EffectAsset::parse(&broken).unwrap().check(BOOKS).is_err());
        let blank = over.replace("\"version\": 1,", "\"version\": 1, \"over_budget\": \" \",");
        assert!(EffectAsset::parse(&blank).unwrap().check(BOOKS).is_err());
    }

    /// Particles that outlive their event are held to `max_live` bursts.
    #[test]
    fn a_lasting_emitter_is_costed_at_its_cap() {
        let smoke = flash()
            .replace("\"value\": \"one_frame\"", "\"value\": { \"range\": [0.9, 1.8] }")
            .replace("\"events\": [{ \"module\": \"light_pulse\", \"intensity\": { \"from\": \"flash.radiance\", \"scale\": 0.02 }, \"range\": 4.0 }]", "\"events\": []")
            .replace("\"instances\": 13, \"lights\": 1", "\"instances\": 24, \"lights\": 0");
        let report = EffectAsset::parse(&smoke).unwrap().check(BOOKS).unwrap();
        assert_eq!((report.instances_per_burst, report.instances_steady), (6, 24));
        assert!((report.fill_steady - 4.0 * report.fill_per_burst).abs() < 1e-6);
    }

    /// A curve passes through its keys, never overshoots them, and its
    /// table is what the program reads.
    #[test]
    fn a_curve_holds_its_keys_and_does_not_overshoot() {
        let keys = [[0.0, 0.0], [0.08, 0.2], [0.3, 0.12], [1.0, 0.0]];
        for k in keys {
            assert!((curve_at(&keys, k[0]) - k[1]).abs() < 1e-5);
        }
        let table = bake_curve(&keys);
        for i in 0..=200 {
            let x = i as f32 / 200.0;
            let v = curve_at(&keys, x);
            assert!((-1e-6..=0.2 + 1e-6).contains(&v), "{x}: {v}");
            assert!((table_at(&table, x) - v).abs() < 0.02, "{x}");
        }
        assert_eq!(curve_at(&[[0.5, 3.0]], 0.1), 3.0);
    }

    /// A log-normal draw has its median at the middle uniform, is as often
    /// over as under, and is bounded.
    #[test]
    fn a_lognormal_has_its_median_and_its_bounds() {
        let v = Value::Lognormal(LognormalValue { lognormal: [0.055, 0.45] });
        assert!((v.sample(0.5, 0.0) - 0.055).abs() < 1e-4);
        let (lo, hi) = v.bounds();
        let draws: Vec<f32> = (0..1000).map(|i| v.sample((i as f32 + 0.5) / 1000.0, 0.0)).collect();
        assert!(draws.iter().all(|d| *d >= lo * 0.999 && *d <= hi * 1.001));
        assert!(draws.iter().filter(|d| **d > 2.0 * 0.055).count() > 30, "some shots are twice the median");
        assert!(draws.iter().filter(|d| **d < 0.055 / 2.0).count() > 30);
        assert!((normal_of(0.8413) - 1.0).abs() < 0.01 && (normal_of(0.0228) + 2.0).abs() < 0.02);
    }
}
