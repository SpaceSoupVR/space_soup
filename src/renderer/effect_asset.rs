use serde::{Deserialize, Serialize};

pub const ASSET_VERSION: u32 = 1;

pub const MAX_CURVE_KEYS: usize = 8;

pub const CURVE_SAMPLES: usize = 32;

pub const MAX_INPUTS: usize = 4;

pub const LIMIT_INSTANCES: u32 = 400;

pub const LIMIT_FILL_BURST: f32 = 0.9;

pub const LIMIT_FILL_STEADY: f32 = 1.5;

pub const LIMIT_LIGHTS: u32 = 1;

pub const LIMIT_TEXTURE_BYTES: u64 = 16 * 1024 * 1024;

pub const NEAR_EYE: f32 = 0.35;

pub const REFERENCE_FOCAL_PIXELS: f32 = 604.0;
pub const REFERENCE_EYES: f32 = 2.0;

pub const REFERENCE_HELD: f32 = 1.0;
pub const REFERENCE_PLACED: f32 = 2.5;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EffectAsset {
    pub name: String,
    pub version: u32,

    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub inputs: Vec<String>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub energy: Option<Value>,
    pub emitters: Vec<EmitterDef>,
    pub scalability: Scalability,
    #[serde(default)]
    pub budget: Budget,

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

    #[default]
    PerSpawn,

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

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Scalability {

    pub cull_distance: f32,

    pub min_pixels: f32,

    pub importance: f32,

    pub max_live: u32,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub near_fade: Option<[f32; 2]>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference_distance: Option<f32>,
}

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

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LognormalValue {
    pub lognormal: [f32; 2],
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SquaredValue {
    pub squared: [f32; 2],
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LogSquaredValue {
    pub log_squared: [f32; 2],
}

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

    #[default]
    Life,

    Seed,
}

pub const LOGNORMAL_SIGMAS: f32 = 2.5;

impl Value {

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

pub fn bake_curve(keys: &[[f32; 2]]) -> [f32; CURVE_SAMPLES] {
    std::array::from_fn(|i| curve_at(keys, i as f32 / (CURVE_SAMPLES - 1) as f32))
}

pub fn table_at(table: &[f32; CURVE_SAMPLES], x: f32) -> f32 {
    let f = x.clamp(0.0, 1.0) * (CURVE_SAMPLES - 1) as f32;
    let i = (f as usize).min(CURVE_SAMPLES - 2);
    table[i] + (table[i + 1] - table[i]) * (f - i as f32)
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum LifeValue {
    Named(LifeName),
    Seconds(Value),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LifeName {

    OneFrame,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "module", rename_all = "snake_case", deny_unknown_fields)]
pub enum SpawnModule {

    Rate {
        per_second: f32,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        for_seconds: Option<f32>,
    },

    Burst {
        count: Value,
        #[serde(default = "one", skip_serializing_if = "is_one")]
        chance: f32,
    },

    Density { per_cubic_metre: f32 },

    Beat { per_second: f32 },

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

    Size { value: Value },

    Speed { value: Value },

    Spin {
        #[serde(default)]
        rate: f32,
        #[serde(default)]
        start: f32,
    },
    ShapePoint {},

    ShapeDisc {
        radius: f32,
        #[serde(default, skip_serializing_if = "is_false")]
        fills_object: bool,
    },

    ShapeBox {},

    ShapeCone {
        angle_deg: f32,
        #[serde(default, skip_serializing_if = "About::is_direction")]
        about: About,
    },

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

    SpotsLowDiscrepancy { radius: f32 },

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

    Ballistic {
        #[serde(default)]
        accel: [f32; 3],
        #[serde(default)]
        drag: f32,
    },

    Swirl { radius: f32, turn: f32 },

    AirDrift {
        #[serde(default = "one", skip_serializing_if = "is_one")]
        strength: f32,
    },

    Jitter { reach: f32, pace: f32 },

    CeilingSpread {},

    CeilingStop {},

    SizeOverLife {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        to: Option<f32>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        value: Option<Value>,
    },

    OpacityOverLife { value: Value },

    PuffFade { peak: f32, rise: f32, power: f32 },

    ColourOverLife { gradient: Vec<[f32; 5]> },

    Glint {},

    LightCpu {
        mode: LightMode,
        #[serde(default)]
        phase_g: f32,
        albedo: [f32; 3],

        #[serde(default = "one", skip_serializing_if = "is_one")]
        ambient: f32,
    },

    AdaptOwnLight { radius: f32, level: f32 },

    ByInput { input: String, target: UpdateTarget, curve: Vec<[f32; 2]> },

    EmberGlow { radiance: f32 },

    FlameSheets { radiance: f32 },

    CoalBed { glow: f32 },

    Pops { per_second: f32 },

    WaterEntry { part: WaterPart },
}

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

    SpritePlain { book: String },

    SpriteSixway { book: String },

    SpriteFlame { book: String },

    SpriteBed { book: String },

    SpriteScatter { book: String },

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

        #[serde(default)]
        streak_seconds: f32,
    },

    Flipbook {
        #[serde(default)]
        play: Play,
    },

    SpriteFog {
        density: f32,
        height: f32,
        falloff: f32,
        #[serde(default)]
        edge: f32,
        #[serde(default)]
        wind: [f32; 3],
    },

    SoftDepth { metres: f32 },

    Motion {
        layer: MotionLayer,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        reason: Option<String>,
    },
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Adapt {

    #[default]
    Room,

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

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BookInfo {
    pub name: &'static str,
    pub first_layer: u32,

    pub frames: u32,
    pub variants: u32,
    pub channels: Channels,

    pub cutout_share: f32,

    pub bytes: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Channels {

    Sixway,

    HeatCover,
    Alpha,
    Bed,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Issue {

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

#[derive(Clone, Debug, PartialEq)]
pub struct EmitterCost {
    pub name: String,

    pub instances: u32,

    pub longest_life: f32,

    pub fill: f32,

    pub stream: bool,

    pub area: f32,
    pub flat: bool,

    pub points: u32,

    pub extent: Extent,

    pub fills_disc: f32,

    pub volume: bool,
}

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

#[derive(Clone, Debug, PartialEq)]
pub struct Report {
    pub emitters: Vec<EmitterCost>,

    pub instances_per_burst: u32,
    pub fill_per_burst: f32,

    pub instances_steady: u32,
    pub fill_steady: f32,
    pub lights: u32,
    pub texture_bytes: u64,
    pub reference_distance: f32,

    pub burst: bool,

    pub seen_from_side: bool,

    pub extent: Extent,
    pub area_facing: f32,
    pub area_flat: f32,
    pub points: u32,

    pub volumes: u32,

    pub over_budget: Option<(String, Vec<String>)>,
}

impl EffectAsset {

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

    pub fn energy(&self) -> Value {
        self.energy.clone().unwrap_or(Value::Number(1.0))
    }

    pub fn input(&self, name: &str) -> Option<usize> {
        self.inputs.iter().position(|i| i == name)
    }

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

        let instances = match rate {
            _ if fog.is_some() => 1.0,
            Some((per_second, for_seconds)) => (per_second * for_seconds.map_or(longest, |d| d.min(longest))).ceil().max(1.0),
            None => count,
        }
        .max(0.0) as u32;
        let half = size.map_or(0.0, |s| s.mean());
        let mean_half = match grow {

            Some(to) if to > 0.0 => ((half * half + half * to + to * to) / 3.0).sqrt(),
            Some(scale) if scale < 0.0 => half * (-scale).max(1.0) * 0.75,
            _ => half,
        };
        let across = 2.0 * mean_half / reference * REFERENCE_FOCAL_PIXELS;

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

        let text = flash().replace("{ \"lognormal\": [0.08, 0.45] }", "{ \"lognormal\": [0.08, 0.45], \"sigma\": 2 }");
        assert!(EffectAsset::parse(&text).is_err());
    }

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
