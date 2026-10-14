//! STACKS, COMPILED: an effect asset's emitters (`effect_asset`) turned into
//! flat CPU programs, and the EVENT BURSTS that run them.
//!
//! WHAT COMPILING DOES. Each emitter's spawn, initialize and update modules
//! become a list of [`Step`]s over one small particle record, in the stack's
//! order, with every curve baked to a table and every parameter resolved; its
//! render modules become a shade mode, a placement, a book and a blend -- the
//! render key its shader variant is chosen by (`variants`). Nothing is
//! looked up by name in a frame.
//!
//! A BURST ([`EffectBurst`]) is an event: a shot, an impact. Like a splash it
//! holds no particles: every particle is a function of the burst and the time
//! since. What IS kept, per burst and never per particle: how many app
//! frames it has been drawn (a muzzle flash lives ONE frame, whatever the
//! frame time), and whether it was admitted whole or as its flash only
//! ([`BurstSet::add`]).
//!
//! WHAT EACH THING COSTS, and what it scales with:
//! - a live burst out of view: one bounds test a frame (its travel bound,
//!   [`CompiledEffect::reach`]); out of range or under `min_pixels`: the same;
//! - a live burst in view: its particles, once each (the steps of its
//!   emitters); a burst is retired when its longest life has passed;
//! - one SOURCE (a weapon) firing at any rate: at most `max_live` bursts'
//!   lasting particles and one stream, by admission ([`BurstSet::add`]);
//! - the frame: the governor ([`BurstSet::govern`]) holds the bursts'
//!   estimated fill and instances under a budget by thinning the lasting
//!   particles of the least important, farthest, smallest bursts first --
//!   sparser, never smaller or dimmer -- and counts what it shed.

use super::*;
use crate::renderer::effect_asset::{
    self as asset, bake_curve, table_at, Adapt, BookInfo, EffectAsset, EmitterDef, EventModule, Gate, InitModule, Issue, LifeName, LifeValue, LightIntensity, LightMode,
    MotionLayer, Over, PlacementMode, Play, RenderModule, Report, Scalability, SeedPolicy, SpawnModule, SpawnTarget, UpdateModule, UpdateTarget, Value, CURVE_SAMPLES,
};

/// An effect in a library, resolved once at load: never a string a frame.
pub type EffectId = u32;

/// HOW A SCREENED PARTICLE IS SHADED, `EffectInstance::params.w`: its light
/// through the flame's heat ramp from its book's own heat (a muzzle flash).
/// 0 is plain, as embers and drops are.
pub const SHADE_SCREEN_HEAT: f32 = 4.0;

/// The estimated fill of every burst in view the governor holds the frame
/// to, megapixel-layers, and the instances. PROVISIONAL: to be set from a
/// headset run of the stress scene.
pub const FRAME_FILL_BUDGET: f32 = 6.0;
pub const FRAME_INSTANCE_BUDGET: u32 = 1500;
/// How fast a burst's share of its lasting particles moves toward what the
/// governor wants, a frame: ten frames from whole to none, so particles
/// leave one at a time and none pops.
const SHARE_STEP: f32 = 0.1;
/// The share of the budget the governor aims at, so that the frame stays
/// under the budget itself while its shares move.
const GOVERN_AIM: f32 = 0.9;

/// A light pulse is left out where, as the eye in that room takes it
/// ([`brief_light_scale`]), it would raise a surface this far off by under
/// two levels of 255 -- 0.016 of full scale on a sunlit surface.
const PULSE_GATE_DISTANCE: f32 = 1.0;
const PULSE_GATE_ALBEDO: f32 = 0.5;
const PULSE_GATE_LEVELS: f32 = 0.016;

/// The exposure the eye holds in full sun in this engine (the offline
/// harness meters 0.33 outdoors at 13:00 and exposes x1.4).
pub const SUN_EXPOSURE: f32 = 1.4;
/// How much more a brief source shows to an eye in a dim room than in the
/// sun, beyond what the frame's exposure already says.
const BRIEF_GAIN: f32 = 4.0;

/// HOW A SOURCE TOO BRIEF TO ADAPT TO IS SEEN -- a muzzle flash, its sparks:
/// at the room's exposure, times this.
///
/// A flash lasts 1-3 ms, so the eye that sees it is adapted to the room, and
/// that alone should make it blinding in the dark and faint in the sun. But
/// the rooms here are not as dark against the sun as real ones: a lamp-lit
/// hall is exposed x3 to x9 where full sun is x1.4 (offline harness,
/// 2026-10-13) -- a few times dimmer, where a real room is hundreds of times
/// dimmer. (Only the night outdoors has its real darkness: x80,000.) A
/// flash bright enough for the hall would be as white in the sun. So part
/// of the gap the exposure does not span is put back for what is too brief
/// to be adapted to: 1 from four times the sun's exposure on, falling in
/// step with the exposure to 1/4 at the sun's. With the exposure itself a
/// flash in the brick hall (x9.2) is 26 times what it is in the sun.
/// PROVISIONAL: set from the offline strips; to be looked at in a headset.
pub fn brief_source_scale(exposure: f32) -> f32 {
    (exposure / SUN_EXPOSURE).clamp(1.0, BRIEF_GAIN) / BRIEF_GAIN
}

/// The same for a flash's LIGHT ON THE ROOM: [`brief_source_scale`] twice
/// over. Its sprite is screened, so what is behind it takes its share away
/// (over a sunlit wall most of it); its light on a wall is added, and
/// nothing takes any. In the sun this leaves it under the gate
/// (`pulse_visible`), which is what keeps it out of every daylight shot.
pub fn brief_light_scale(exposure: f32) -> f32 {
    brief_source_scale(exposure).powi(2)
}

/// A value as a program holds it: resolved, a curve as its table.
#[derive(Clone, Debug)]
enum Val {
    Plain(Value),
    Table([f32; CURVE_SAMPLES], Over),
}

impl Val {
    fn new(v: &Value) -> Self {
        match v {
            Value::Curve(c) => Self::Table(bake_curve(&c.curve), c.over),
            other => Self::Plain(other.clone()),
        }
    }

    fn at(&self, u: f32, t: f32) -> f32 {
        match self {
            Self::Plain(v) => v.sample(u, t),
            Self::Table(table, over) => table_at(table, if *over == Over::Seed { u } else { t }),
        }
    }
}

/// One step of an emitter's CPU program. Each reads and writes the particle
/// record ([`P`]); the list runs in the stack's order.
#[derive(Clone, Debug)]
enum Step {
    /// Its radius, and whether a placed effect's disc fills its object.
    Disc(f32, bool),
    Lobes { lobes: u32, open: f32, rake: f32, skip_down: bool, side_pairs: u32, length: Val },
    Cone(f32, bool),
    Speed(Val),
    Size(Val),
    Spin { rate: f32, start: f32 },
    Ballistic { accel: Vec3, drag: f32 },
    Swirl { radius: f32, turn: f32 },
    AirDrift(f32),
    Jitter { reach: f32, pace: f32 },
    CeilingSpread,
    CeilingStop,
    SizeTo(f32),
    SizeBy(Val),
    Opacity(Val),
    PuffFade { peak: f32, rise: f32, power: f32 },
    Colour(Box<[[f32; 4]; CURVE_SAMPLES]>),
    ByInput { input: usize, target: UpdateTarget, table: [f32; CURVE_SAMPLES] },
    Light { mode: LightMode, g: f32, albedo: Vec3, ambient: f32 },
    AdaptOwnLight { radius: f32, level: f32 },
}

/// How an emitter's particles come to be.
#[derive(Clone, Debug)]
enum Spawn {
    Burst { count: Val, chance: f32 },
    Rate { per_second: f32, for_seconds: f32 },
}

/// How a program's particles are shaded and blended: with its placement and
/// book, its render key.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Shade {
    /// Over: one lit colour, alpha from the book.
    Plain,
    /// Over: six-way smoke.
    Sixway,
    /// Screened: the light it scatters.
    Scatter,
    /// Screened: light of one colour.
    Emissive { radiance: f32, colour: Vec3, own: bool },
    /// Screened: light through the heat ramp.
    EmissiveHeat { radiance: f32, colour: Vec3, own: bool },
}

impl Shade {
    pub fn screened(&self) -> bool {
        matches!(self, Self::Scatter | Self::Emissive { .. } | Self::EmissiveHeat { .. })
    }
}

/// One emitter, compiled.
#[derive(Clone, Debug)]
pub struct Program {
    pub name: String,
    spawn: Spawn,
    spawn_by: Vec<(usize, SpawnTarget, [f32; CURVE_SAMPLES])>,
    /// `None`: one app frame.
    life: Option<Val>,
    delay_frames: u32,
    steps: Vec<Step>,
    pub shade: Shade,
    pub placement: PlacementMode,
    streak_seconds: f32,
    pub book: BookInfo,
    play: Play,
    soft: f32,
    pub motion: MotionLayer,
    /// Born at the burst's second point (a weapon's ejection port).
    at_port: bool,
    /// The radius of its disc where it fills a placed object's footprint
    /// (`shape_disc` `fills_object`), 0 for none.
    fills_disc: f32,
    /// How far any of its particles gets from where the burst was, metres.
    reach: f32,
    /// The longest one of its particles can still be drawn after the burst,
    /// seconds (0 for one frame).
    seconds: f32,
}

/// A light a burst raises for a few frames.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LightPulse {
    /// Its emitter: the pulse is raised when that emitter's burst is.
    emitter: usize,
    pub intensity: f32,
    pub range: f32,
    pub frames: u32,
    pub gate: Gate,
    pub colour: [f32; 3],
}

/// An effect asset, checked and compiled.
#[derive(Clone, Debug)]
pub struct CompiledEffect {
    pub name: String,
    pub programs: Vec<Program>,
    pub report: Report,
    pub scalability: Scalability,
    /// A burst's energy, shot to shot: 1 at its median.
    energy: Val,
    energy_input: Option<usize>,
    pub light: Option<LightPulse>,
    /// How long after its event any of it can still be drawn, seconds.
    pub seconds: f32,
    /// A CONSERVATIVE BOUND OF PARTICLE TRAVEL: every particle of a burst
    /// stays within this of where the burst was, metres (its launch against
    /// its drag over its life, its acceleration, its swirl, the air's
    /// reach, its lobes and its own size). What a burst is culled by.
    pub reach: f32,
    locked: bool,
}

/// How far the room's air can carry a particle of a burst, metres: the air
/// every burst rides (`air_waves`), the same for all, so a weapon's smoke
/// drifts as one body.
fn burst_air() -> &'static [AirWave; 6] {
    static AIR: std::sync::OnceLock<[AirWave; 6]> = std::sync::OnceLock::new();
    AIR.get_or_init(|| air_waves(id_hash("the room's air")))
}

fn curve_table(keys: &[[f32; 2]]) -> [f32; CURVE_SAMPLES] {
    bake_curve(keys)
}

impl CompiledEffect {
    /// Check an asset against the books that exist and compile it.
    pub fn compile(a: &EffectAsset, books: &[BookInfo]) -> Result<Self, Vec<Issue>> {
        let report = a.check(books)?;
        let mut programs = Vec::new();
        let mut light = None;
        for (index, e) in a.emitters.iter().enumerate() {
            let program = compile_emitter(a, e, books).map_err(|m| vec![Issue { budget: false, emitter: Some(e.name.clone()), message: m }])?;
            for m in &e.events {
                let EventModule::LightPulse { intensity, range, frames, gate, colour } = m;
                let intensity = match intensity {
                    LightIntensity::Number(n) => *n,
                    LightIntensity::From(f) => {
                        let radiance = a
                            .emitters
                            .iter()
                            .filter(|o| f.from == format!("{}.radiance", o.name))
                            .find_map(|o| o.render.iter().find_map(|r| if let RenderModule::SpriteEmissive { radiance, .. } = r { Some(*radiance) } else { None }))
                            .unwrap_or(0.0);
                        radiance * f.scale
                    }
                };
                light = Some(LightPulse { emitter: index, intensity, range: *range, frames: *frames, gate: *gate, colour: *colour });
            }
            programs.push(program);
        }
        let seconds = programs.iter().map(|p| p.seconds).fold(0.0, f32::max);
        let reach = programs.iter().map(|p| p.reach).fold(0.05, f32::max);
        Ok(Self {
            name: a.name.clone(),
            programs,
            report,
            scalability: a.scalability,
            energy: Val::new(&a.energy()),
            energy_input: a.input("energy"),
            light,
            seconds,
            reach,
            locked: a.seed_policy == SeedPolicy::Locked,
        })
    }

    /// The most instances one burst of it draws.
    pub fn instances(&self) -> u32 {
        self.report.instances_per_burst
    }

    /// The radius of its widest disc that fills a placed object, 0 for none.
    pub fn fills_disc(&self) -> f32 {
        self.programs.iter().map(|p| p.fills_disc).fold(0.0, f32::max)
    }
}

fn compile_emitter(a: &EffectAsset, e: &EmitterDef, books: &[BookInfo]) -> Result<Program, String> {
    let input = |name: &str| a.input(name).ok_or_else(|| format!("no input `{name}`"));
    let mut spawn = None;
    let mut spawn_by = Vec::new();
    for m in &e.spawn {
        match m {
            SpawnModule::Burst { count, chance } => spawn = Some(Spawn::Burst { count: Val::new(count), chance: *chance }),
            SpawnModule::Rate { per_second, for_seconds } => spawn = Some(Spawn::Rate { per_second: *per_second, for_seconds: for_seconds.unwrap_or(f32::INFINITY) }),
            SpawnModule::ByInput { input: name, target, curve } => spawn_by.push((input(name)?, *target, curve_table(curve))),
            SpawnModule::Density { .. } | SpawnModule::Beat { .. } => return Err("density and beat spawn only the built-in templates' emitters".into()),
        }
    }
    let spawn = spawn.ok_or("no spawn module")?;
    let mut life = None;
    let mut delay_frames = 0;
    let mut steps = Vec::new();
    // How far a particle can get: launch, drag, acceleration, and the rest.
    let (mut speed_most, mut size_most, mut extra) = (0.0f32, 0.0f32, 0.0f32);
    let mut life_most = 0.0f32;
    for m in &e.initialize {
        match m {
            InitModule::Life { value: LifeValue::Named(LifeName::OneFrame) } => life = None,
            InitModule::Life { value: LifeValue::Seconds(v) } => {
                life_most = v.bounds().1;
                life = Some(Val::new(v));
            }
            InitModule::Size { value } => {
                size_most = value.bounds().1;
                steps.push(Step::Size(Val::new(value)));
            }
            InitModule::Speed { value } => {
                speed_most = value.bounds().1;
                steps.push(Step::Speed(Val::new(value)));
            }
            InitModule::Spin { rate, start } => steps.push(Step::Spin { rate: *rate, start: *start }),
            InitModule::ShapePoint {} => {}
            InitModule::ShapeDisc { radius, fills_object } => {
                extra += radius;
                steps.push(Step::Disc(*radius, *fills_object));
            }
            InitModule::ShapeCone { angle_deg, about } => steps.push(Step::Cone(*angle_deg, *about == asset::About::Up)),
            InitModule::ShapeLobes { lobes, open_deg, rake_deg, skip_down, side_pairs, length } => {
                extra += 1.2 * length.bounds().1 * LOBE_ENERGY_MOST + 0.02 * *side_pairs as f32;
                steps.push(Step::Lobes { lobes: *lobes, open: *open_deg, rake: *rake_deg, skip_down: *skip_down, side_pairs: *side_pairs, length: Val::new(length) });
            }
            InitModule::Delay { frames } => delay_frames = *frames,
            InitModule::ShapeBox {} | InitModule::SpotsLowDiscrepancy { .. } => return Err("shape_box and spots_low_discrepancy place only the built-in templates' emitters".into()),
        }
    }
    let (mut accel_most, mut drag) = (0.0f32, 0.0f32);
    let mut size_scale = 1.0f32;
    for m in &e.update {
        match m {
            UpdateModule::Ballistic { accel, drag: k } => {
                accel_most = Vec3::from(*accel).length();
                drag = *k;
                steps.push(Step::Ballistic { accel: Vec3::from(*accel), drag: *k });
            }
            UpdateModule::Swirl { radius, turn } => {
                extra += radius;
                steps.push(Step::Swirl { radius: *radius, turn: *turn });
            }
            UpdateModule::AirDrift { strength } => {
                extra += air_reach(burst_air()) * strength;
                steps.push(Step::AirDrift(*strength));
            }
            UpdateModule::Jitter { reach, pace } => {
                extra += 1.8 * reach;
                steps.push(Step::Jitter { reach: *reach, pace: *pace });
            }
            UpdateModule::CeilingSpread {} => steps.push(Step::CeilingSpread),
            UpdateModule::CeilingStop {} => steps.push(Step::CeilingStop),
            UpdateModule::SizeOverLife { to: Some(to), .. } => {
                size_most = size_most.max(*to);
                steps.push(Step::SizeTo(*to));
            }
            UpdateModule::SizeOverLife { value: Some(v), .. } => {
                size_scale *= v.bounds().1.max(1.0);
                steps.push(Step::SizeBy(Val::new(v)));
            }
            UpdateModule::SizeOverLife { .. } => return Err("size_over_life takes `to` or `value`".into()),
            UpdateModule::OpacityOverLife { value } => steps.push(Step::Opacity(Val::new(value))),
            UpdateModule::PuffFade { peak, rise, power } => steps.push(Step::PuffFade { peak: *peak, rise: *rise, power: *power }),
            UpdateModule::ColourOverLife { gradient } => {
                let channel = |c: usize| bake_curve(&gradient.iter().map(|k| [k[0], k[1 + c]]).collect::<Vec<_>>());
                let (r, g, b, al) = (channel(0), channel(1), channel(2), channel(3));
                steps.push(Step::Colour(Box::new(std::array::from_fn(|i| [r[i], g[i], b[i], al[i]]))));
            }
            UpdateModule::ByInput { input: name, target, curve } => {
                let table = curve_table(curve);
                if *target == UpdateTarget::Size {
                    size_scale *= table.iter().fold(1.0f32, |a, b| a.max(*b));
                }
                steps.push(Step::ByInput { input: input(name)?, target: *target, table });
            }
            UpdateModule::LightCpu { mode, phase_g, albedo, ambient } => steps.push(Step::Light { mode: *mode, g: *phase_g, albedo: Vec3::from(*albedo), ambient: *ambient }),
            UpdateModule::AdaptOwnLight { radius, level } => steps.push(Step::AdaptOwnLight { radius: *radius, level: *level }),
            UpdateModule::Glint {}
            | UpdateModule::EmberGlow { .. }
            | UpdateModule::FlameSheets { .. }
            | UpdateModule::CoalBed { .. }
            | UpdateModule::Pops { .. }
            | UpdateModule::WaterEntry { .. } => {
                return Err("glint, ember_glow, flame_sheets, coal_bed, pops and water_entry run only in the built-in templates (`effects::templates`)".into())
            }
        }
    }
    let mut shade = None;
    let mut book = None;
    let mut placement = PlacementMode::Facing;
    let (mut streak_seconds, mut soft, mut play, mut motion) = (0.0, 0.0, Play::Age, MotionLayer::Soft);
    let find = |name: &str| books.iter().find(|b| b.name == name).copied().ok_or_else(|| format!("no book `{name}`"));
    for m in &e.render {
        match m {
            RenderModule::SpritePlain { book: b } => (shade, book) = (Some(Shade::Plain), Some(find(b)?)),
            RenderModule::SpriteSixway { book: b } => (shade, book) = (Some(Shade::Sixway), Some(find(b)?)),
            RenderModule::SpriteScatter { book: b } => (shade, book) = (Some(Shade::Scatter), Some(find(b)?)),
            RenderModule::SpriteEmissive { book: b, radiance, colour, heat_ramp, adapt } => {
                let (radiance, colour, own) = (*radiance, Vec3::from(*colour), *adapt == Adapt::OwnLight);
                shade = Some(if *heat_ramp { Shade::EmissiveHeat { radiance, colour, own } } else { Shade::Emissive { radiance, colour, own } });
                book = Some(find(b)?);
            }
            RenderModule::SpriteFlame { .. } | RenderModule::SpriteBed { .. } => return Err("sprite_flame and sprite_bed draw only the built-in templates' emitters".into()),
            RenderModule::Placement { mode, streak_seconds: s } => (placement, streak_seconds) = (*mode, *s),
            RenderModule::Flipbook { play: p } => play = *p,
            RenderModule::SoftDepth { metres } => soft = *metres,
            RenderModule::Motion { layer, .. } => motion = *layer,
        }
    }
    let life_for_travel = if life.is_some() { life_most } else { 0.0 };
    // Launched at `v` against drag `k` it gets no further than `v / k`;
    // under `a` it drifts at no more than `a / k`, or falls `a t^2 / 2`.
    let coast = if drag > 1e-4 { speed_most * (1.0 / drag).min(life_for_travel) } else { speed_most * life_for_travel };
    let fall = if drag > 1e-4 { accel_most / drag * life_for_travel } else { 0.5 * accel_most * life_for_travel * life_for_travel };
    let reach = coast + fall + extra + 1.5 * size_most * size_scale + streak_seconds * speed_most;
    let seconds = match (&spawn, life.is_some()) {
        (Spawn::Rate { for_seconds, .. }, _) => for_seconds.min(600.0) + life_most,
        (_, true) => life_most,
        (_, false) => 0.0,
    };
    Ok(Program {
        name: e.name.clone(),
        spawn,
        spawn_by,
        life,
        delay_frames,
        steps,
        shade: shade.ok_or("no sprite module")?,
        placement,
        streak_seconds,
        book: book.ok_or("no book")?,
        play,
        soft,
        motion,
        at_port: e.at == asset::At::Port,
        fills_disc: e.initialize.iter().find_map(|m| if let InitModule::ShapeDisc { radius, fills_object: true } = m { Some(*radius) } else { None }).unwrap_or(0.0),
        reach,
        seconds,
    })
}

/// How much longer than its drawn length the most energetic shot's lobe is.
const LOBE_ENERGY_MOST: f32 = 2.2;

/// WHERE AN EMITTER'S PARTICLES GET TO from its origin, out to the side, down
/// and up, its up the emitter's direction: [`compile_emitter`]'s travel bound
/// taken apart by way -- a disc and a swirl go sideways, a launch along its
/// cone, a fall along its acceleration, a level sheet's own size only
/// sideways. The box an effect's fill is seen through (`seen`), and
/// conservative as the reach is: drag, air and jitter at their most.
pub(crate) fn emitter_extent(e: &EmitterDef) -> asset::Extent {
    let (mut size_most, mut size_scale, mut speed_most, mut life_most) = (0.0f32, 1.0f32, 0.0f32, 0.0f32);
    let (mut lateral, mut vertical) = (0.0f32, 0.0f32);
    let mut cone: Option<f32> = None;
    for m in &e.initialize {
        match m {
            InitModule::Life { value: LifeValue::Seconds(v) } => life_most = v.bounds().1,
            InitModule::Size { value } => size_most = value.bounds().1,
            InitModule::Speed { value } => speed_most = value.bounds().1,
            InitModule::ShapeDisc { radius, .. } => lateral += radius,
            InitModule::ShapeBox {} => {
                lateral += 0.5;
                vertical += 0.5;
            }
            InitModule::ShapeCone { angle_deg, .. } => cone = Some(*angle_deg),
            InitModule::ShapeLobes { side_pairs, length, .. } => {
                let l = 1.2 * length.bounds().1 * LOBE_ENERGY_MOST + 0.02 * *side_pairs as f32;
                lateral += l;
                vertical += l;
            }
            _ => {}
        }
    }
    let (mut accel, mut drag) = (Vec3::ZERO, 0.0f32);
    for m in &e.update {
        match m {
            UpdateModule::Ballistic { accel: a, drag: k } => (accel, drag) = (Vec3::from(*a), *k),
            UpdateModule::Swirl { radius, .. } => lateral += radius,
            UpdateModule::AirDrift { strength } => {
                let air = air_reach(burst_air()) * strength;
                lateral += air;
                vertical += air;
            }
            UpdateModule::Jitter { reach, .. } => {
                lateral += 1.8 * reach;
                vertical += 0.6 * 1.8 * reach;
            }
            UpdateModule::SizeOverLife { to: Some(to), .. } => size_most = size_most.max(*to),
            UpdateModule::SizeOverLife { value: Some(v), .. } => size_scale *= v.bounds().1.max(1.0),
            UpdateModule::ByInput { target: UpdateTarget::Size, curve, .. } => size_scale *= curve.iter().map(|k| k[1]).fold(1.0f32, f32::max),
            _ => {}
        }
    }
    let (mut flat, mut streak_seconds) = (false, 0.0f32);
    for m in &e.render {
        if let RenderModule::Placement { mode, streak_seconds: s } = m {
            flat = *mode == PlacementMode::Flat;
            streak_seconds = if *mode == PlacementMode::Streak { *s } else { 0.0 };
        }
    }
    let t = life_most;
    let coast = if drag > 1e-4 { speed_most * (1.0 / drag).min(t) } else { speed_most * t };
    let fall = |a: f32| if drag > 1e-4 { a / drag * t } else { 0.5 * a * t * t };
    // A streak lies along its velocity: its launch, and the most its
    // acceleration brings it to against its drag, sideways and up apart.
    let terminal = |a: f32| if drag > 1e-4 { a / drag } else { a * t };
    // A launch along its cone (along its direction without one); `about:
    // up` is the world's up, which a placed effect's direction is.
    let open = cone.unwrap_or(0.0).clamp(0.0, 180.0).to_radians();
    let (up, down) = (coast, coast * (-open.cos()).max(0.0));
    let side = coast * open.min(std::f32::consts::FRAC_PI_2).sin();
    let size = 1.5 * size_most * size_scale;
    let sheet_up = if flat { size * open.min(std::f32::consts::FRAC_PI_2).sin() } else { size };
    let fall_side = fall(Vec3::new(accel.x, 0.0, accel.z).length());
    let streak_side = streak_seconds * (speed_most * open.min(std::f32::consts::FRAC_PI_2).sin() + terminal(Vec3::new(accel.x, 0.0, accel.z).length()));
    let streak_up = streak_seconds * (speed_most + terminal(accel.y.abs()));
    asset::Extent {
        lateral: lateral + side + fall_side + size + streak_side,
        below: vertical + down + fall((-accel.y).max(0.0)) + sheet_up + streak_up,
        above: vertical + up + fall(accel.y.max(0.0)) + sheet_up + streak_up,
    }
}

/// THE EFFECTS A LEVEL CAN RAISE, compiled at load, by name.
#[derive(Clone, Debug, Default)]
pub struct EffectLibrary {
    effects: Vec<CompiledEffect>,
}

impl EffectLibrary {
    pub fn new() -> Self {
        Self::default()
    }

    /// Every effect in `game/effects`, checked and compiled, and why each
    /// that was refused was. The books are the renderer's ([`books`]).
    pub fn from_dir(dir: &std::path::Path) -> (Self, Vec<String>) {
        let (assets, failed) = EffectAsset::load_dir(dir);
        let mut errors: Vec<String> = failed.into_iter().map(|(p, e)| format!("{}: {e}", p.display())).collect();
        let mut lib = Self::new();
        for a in &assets {
            if let Err(issues) = lib.add(a) {
                errors.push(format!("effect `{}`: {}", a.name, issues.iter().map(|i| i.to_string()).collect::<Vec<_>>().join("; ")));
            }
        }
        (lib, errors)
    }

    /// Compile one asset into the library (replacing one of its name).
    pub fn add(&mut self, a: &EffectAsset) -> Result<EffectId, Vec<Issue>> {
        let compiled = CompiledEffect::compile(a, books())?;
        Ok(match self.effects.iter().position(|e| e.name == a.name) {
            Some(i) => {
                self.effects[i] = compiled;
                i as EffectId
            }
            None => {
                self.effects.push(compiled);
                self.effects.len() as EffectId - 1
            }
        })
    }

    pub fn id(&self, name: &str) -> Option<EffectId> {
        self.effects.iter().position(|e| e.name == name).map(|i| i as EffectId)
    }

    pub fn get(&self, id: EffectId) -> Option<&CompiledEffect> {
        self.effects.get(id as usize)
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.effects.iter().map(|e| e.name.as_str())
    }

    pub fn len(&self) -> usize {
        self.effects.len()
    }

    pub fn is_empty(&self) -> bool {
        self.effects.is_empty()
    }
}

/// AN EVENT THAT RAISES AN EFFECT: a shot, an impact. In the WORLD's frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EffectBurst {
    pub asset: EffectId,
    /// Seconds since it happened, as it is handed over: 0 for the local
    /// player's own shot, the message's age for one the server told of.
    pub age: f32,
    /// Where: a muzzle. And which way it points, and its up (a flash
    /// hider's closed bottom is opposite it).
    pub position: Vec3,
    pub direction: Vec3,
    pub up: Vec3,
    /// How what it is attached to was moving, m/s: its flash is moved with
    /// it by SpaceWarp's made-up frame, so it stays on the barrel in recoil.
    pub velocity: Vec3,
    /// A second point and direction: a weapon's ejection port. Emitters
    /// placed `at: "port"` are skipped without one.
    pub port: Option<(Vec3, Vec3)>,
    /// The asset's named inputs, in its order (`EffectAsset::inputs`).
    pub inputs: [f32; 4],
    /// What raised it: one weapon. `max_live` and the stream are per source.
    pub source: u64,
    pub seed: u64,
}

impl EffectBurst {
    /// A burst of `asset` at `position` along `direction`, now, at rest,
    /// its inputs zero.
    pub fn new(asset: EffectId, position: Vec3, direction: Vec3, source: u64, seed: u64) -> Self {
        let direction = direction.try_normalize().unwrap_or(Vec3::NEG_Z);
        Self { asset, age: 0.0, position, direction, up: Vec3::Y, velocity: Vec3::ZERO, port: None, inputs: [0.0; 4], source, seed }
    }
}

/// A burst in play.
#[derive(Clone, Copy, Debug)]
pub struct LiveBurst {
    pub burst: EffectBurst,
    /// When it happened, on the effects' clock; `None` until the first frame
    /// after it was added gives it one ([`BurstSet::frame`]).
    pub born: Option<f64>,
    /// App frames it has been drawn (or would have been, out of view).
    pub frames: u32,
    /// Admitted whole; false when its source was at its cap: its one-frame
    /// emitters and its stream only.
    pub lasting: bool,
    /// When a later burst of its source took its stream over.
    superseded: Option<f64>,
    /// The share of its lasting particles drawn (the governor's).
    pub share: f32,
    /// A PLACED EFFECT (a scene object's `effect.asset`): never retired,
    /// never taken over, its streams running for as long as the level is
    /// loaded; with the object's own scale, rate and tint ([`BurstSet::place`]).
    pub standing: Option<Placed>,
}

/// What a scene object's effect says beside its asset's name, as the
/// templates' `scale`, `rate` and `tint` do: its sprites' size, its streams'
/// rate and its colour, each a multiple of the asset's.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Placed {
    pub scale: f32,
    pub rate: f32,
    /// Linear (the scene's sRGB tint, decoded).
    pub tint: [f32; 3],
    /// How much of the sky's sun reaches it, 0-1, found at load by a ray
    /// toward the sun (`quest_app::scene_effects::find_sunlit`): 0 in a cave
    /// or a room, 1 out in the open. Its particles are lit by the frame's
    /// directional light times this, as a lamp would light them.
    pub sunlit: f32,
    /// Its object's footprint, half its width along the world's x and z,
    /// metres: what a disc that fills its object (`shape_disc`
    /// `fills_object`) spans, its rate grown with the area. 0 for none.
    pub footprint: [f32; 2],
}

impl Placed {
    /// The footprint, where it has one.
    pub fn footprint(&self) -> Option<[f32; 2]> {
        (self.footprint[0] > 0.0 && self.footprint[1] > 0.0).then_some(self.footprint)
    }

    /// How many times as many particles a disc of `radius` that fills its
    /// object makes (its footprint's area over the disc's); 1 for one that
    /// does not (`radius` 0) or an object with no footprint.
    pub fn fills(&self, radius: f32) -> f32 {
        match self.footprint() {
            Some([x, z]) if radius > 0.0 => x * z / (radius * radius),
            _ => 1.0,
        }
    }
}

impl Default for Placed {
    fn default() -> Self {
        Self { scale: 1.0, rate: 1.0, tint: [1.0; 3], sunlit: 0.0, footprint: [0.0; 2] }
    }
}

/// A scene object's effect that names an asset, as the client loads it
/// (`quest_app::scene_effects::placed`), in the WORLD's frame.
#[derive(Clone, Debug, PartialEq)]
pub struct PlacedEffect {
    /// The object's id: what a log names, and (hashed) its source.
    pub object: String,
    pub asset: String,
    pub position: Vec3,
    /// Its object's up.
    pub direction: Vec3,
    pub source: u64,
    pub placed: Placed,
}

/// SOMETHING ELSE THE GOVERNOR WEIGHS beside the bursts
/// ([`BurstSet::frame_with`]): a placed template emitter or a splash in view,
/// by a key of its own (`effects::governor_wants`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Want {
    pub key: u64,
    /// Lowest sheds first: importance x 100 plus log2 of its pixels across,
    /// so among equals the farthest and smallest go first.
    pub score: f32,
    /// Its fill this frame drawn whole, megapixel-layers (estimated), and
    /// its instances.
    pub fill: f32,
    pub count: f32,
    /// False for what fewer of would be a smaller thing, not a sparser one
    /// (a fire's sheets, a bed of coals): counted in the frame, never shed.
    pub sheddable: bool,
    pub kind: WantKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WantKind {
    Emitter,
    Splash,
}

/// What a frame's bursts came to.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BurstStats {
    pub live: u32,
    /// In view, in range and big enough to draw.
    pub seen: u32,
    /// Out of every eye's view, out of range or under `min_pixels`.
    pub culled: u32,
    /// Estimated fill of those in view before the governor, megapixel-
    /// layers, and after it.
    pub fill_wanted: f32,
    pub fill: f32,
    /// Bursts the governor is thinning.
    pub thinned: u32,
    /// Light pulses raised, and those left out as not visible (since the
    /// set was made).
    pub lights: u32,
    pub lights_gated: u32,
    /// Bursts admitted as their flash only, their source at its cap (since
    /// the set was made).
    pub merged: u32,
    /// Bursts dropped before their time because [`BURSTS_KEPT`] were alive
    /// (since the set was made): what a level firing more than it can keep
    /// shows up as.
    pub dropped: u32,
    /// Placed emitters and splashes in view the governor is thinning this
    /// frame ([`Want`]), and the frame's estimated instances before and
    /// after it (bursts, emitters and splashes together; `fill_wanted` and
    /// `fill` are of all three too).
    pub emitters_thinned: u32,
    pub splashes_thinned: u32,
    pub instances_wanted: f32,
    pub instances: f32,
    /// Frames the governor shed anything in, since the set was made.
    pub frames_governed: u32,
}

/// THE BURSTS IN PLAY, and what admits, culls, thins and retires them.
#[derive(Clone, Debug, Default)]
pub struct BurstSet {
    pub live: Vec<LiveBurst>,
    /// The effects' clock at the last frame.
    now: f64,
    pub stats: BurstStats,
    /// What the governor holds the frame's effects to.
    pub budget: FrameBudget,
    /// The governor's share of each placed emitter and splash it weighed
    /// last frame, by [`Want::key`]; what it did not weigh is whole.
    shares: std::collections::HashMap<u64, f32>,
    /// The placed emitters it weighed last frame, by key. One it did not --
    /// newly in sight, its door just opened -- RAMPS IN from none of its
    /// particles, [`SHARE_STEP`] a frame, in its fixed order (`kept`): the
    /// same particles it would draw whole, arriving over a few frames, not
    /// its whole population worked out on the frame it was revealed.
    weighed: std::collections::HashSet<u64>,
    /// Each splash's fill and instances as last drawn whole, by its key
    /// (`effects::simulate_governed` measures them): what it is weighed by
    /// on the next frame.
    pub splash_measured: std::collections::HashMap<u64, (f32, f32)>,
}

/// The estimated fill of the bursts in view, megapixel-layers, and their
/// lasting instances, that the governor holds a frame to.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrameBudget {
    pub fill: f32,
    pub instances: u32,
}

impl Default for FrameBudget {
    fn default() -> Self {
        Self { fill: FRAME_FILL_BUDGET, instances: FRAME_INSTANCE_BUDGET }
    }
}

/// The most bursts kept at all: past it one is dropped, and counted
/// (`BurstStats::dropped`) -- the oldest that is only a flash and a stream
/// (its smoke already merged into its source's), else the oldest. A guard
/// on memory and on the work of a frame's cull (one bound test a burst);
/// about 170 weapons' worth at five whole bursts and a stream each.
pub const BURSTS_KEPT: usize = 1024;

impl BurstSet {
    fn age(&self, l: &LiveBurst) -> f32 {
        match l.born {
            Some(b) => (self.now - b) as f32,
            None => l.burst.age,
        }
    }

    /// A NEW BURST. Its source keeps at most `max_live` bursts' lasting
    /// particles: one that arrives sooner after the last whole one than
    /// `longest life / max_live` is admitted as its flash and its stream
    /// alone (its inputs -- `heat` -- still reach the puffs already there),
    /// so ten shots a second cost what three do and read as denser smoke.
    /// Spaced evenly: a plain cap let five through and then none.
    pub fn add(&mut self, lib: &EffectLibrary, burst: EffectBurst) {
        let Some(c) = lib.get(burst.asset) else { return };
        let most = c.scalability.max_live.max(1);
        let same = |l: &LiveBurst| l.burst.asset == burst.asset && l.burst.source == burst.source && l.lasting && l.standing.is_none();
        let ages: Vec<f32> = self.live.iter().filter(|l| same(l)).map(|l| self.age(l) - burst.age).collect();
        let whole_alive = ages.iter().filter(|a| **a < c.seconds).count() as u32;
        let since_last = ages.iter().copied().fold(f32::INFINITY, f32::min);
        let lasting = whole_alive < most && !(since_last < c.seconds / most as f32);
        if !lasting {
            self.stats.merged += 1;
        }
        if self.live.len() >= BURSTS_KEPT {
            let oldest_flash = self.live.iter().position(|l| !l.lasting && l.standing.is_none());
            if let Some(k) = oldest_flash.or_else(|| self.live.iter().position(|l| l.standing.is_none())) {
                self.live.remove(k);
                self.stats.dropped += 1;
            }
        }
        self.live.push(LiveBurst { burst, born: None, frames: 0, lasting, superseded: None, share: 1.0, standing: None });
    }

    /// A PLACED EFFECT: a scene object's `effect.asset`, at the object, along
    /// its up, for as long as the level is loaded. Its streams run from the
    /// first frame on; a burst emitter in it fires once, then. `source` is
    /// the object's (one key a placed effect, for the governor and the seed).
    pub fn place(&mut self, lib: &EffectLibrary, asset: EffectId, position: Vec3, direction: Vec3, source: u64, placed: Placed) {
        if lib.get(asset).is_none() {
            return;
        }
        let burst = EffectBurst::new(asset, position, direction, source, hash64(source ^ 0x5eed_0f_91ace));
        self.live.push(LiveBurst { burst, born: None, frames: 0, lasting: true, superseded: None, share: 1.0, standing: Some(placed) });
    }

    /// Every placed effect taken away (a scene change), the bursts left.
    pub fn clear_placed(&mut self) {
        self.live.retain(|l| l.standing.is_none());
    }

    /// The governor's share of a placed emitter or splash ([`Want::key`]):
    /// 1 for one it has not thinned.
    pub fn share_of(&self, key: u64) -> f32 {
        self.shares.get(&key).copied().unwrap_or(1.0)
    }

    /// What each splash drawn this frame filled whole (`EffectFrame::splash_costs`),
    /// kept for the next frame's governor.
    pub fn measured(&mut self, costs: &[(u64, f32, f32)]) {
        self.splash_measured.clear();
        self.splash_measured.extend(costs.iter().map(|&(k, f, n)| (k, (f, n))));
    }

    /// THE FRAME'S BURSTS at `now` on the effects' clock: new ones are given
    /// their moment, finished ones retired, and each of the rest is either
    /// drawn this frame or not -- in one of `views` (the eyes' frusta, in
    /// the WORLD's frame; far plane ignored) by its travel bound, within its
    /// `cull_distance` of `head` and at least `min_pixels` across. Then the
    /// governor: if those drawn would fill more than [`FRAME_FILL_BUDGET`]
    /// or number more than [`FRAME_INSTANCE_BUDGET`], the lasting particles
    /// of the least important, then the farthest and smallest, are thinned
    /// a tenth a frame until they do not. Returns, per live burst, whether
    /// it is drawn.
    pub fn frame(&mut self, lib: &EffectLibrary, now: f64, views: &[[glam::Vec4; 6]], head: Vec3) -> Vec<bool> {
        self.frame_with(lib, now, views, head, &[])
    }

    /// [`Self::frame`], with THE PLACED EMITTERS AND SPLASHES in view weighed
    /// by the same governor (`others`, from `effects::governor_wants`): the
    /// whole frame's effects -- bursts, emitters, splashes -- are held to one
    /// budget, the lowest scored shed first whatever kind it is. Their shares
    /// are read back with [`Self::share_of`].
    pub fn frame_with(&mut self, lib: &EffectLibrary, now: f64, views: &[[glam::Vec4; 6]], head: Vec3, others: &[Want]) -> Vec<bool> {
        self.now = now;
        // New ones: their moment, and their source's stream.
        for i in 0..self.live.len() {
            if self.live[i].born.is_some() {
                continue;
            }
            let born = now - self.live[i].burst.age.max(0.0) as f64;
            self.live[i].born = Some(born);
            if self.live[i].standing.is_some() {
                continue;
            }
            let (asset, source) = (self.live[i].burst.asset, self.live[i].burst.source);
            for (k, l) in self.live.iter_mut().enumerate() {
                if k != i && l.standing.is_none() && l.burst.asset == asset && l.burst.source == source && l.superseded.is_none() && l.born.is_some_and(|b| b <= born) {
                    l.superseded = Some(born);
                }
            }
        }
        self.live.retain(|l| lib.get(l.burst.asset).is_some_and(|c| l.standing.is_some() || now < l.until(c)));
        let mut drawn = vec![false; self.live.len()];
        // (score, fill, instances, index): what the governor weighs. An
        // index past the bursts is `others[index - live]`.
        let mut wants: Vec<(f32, f32, f32, usize)> = Vec::new();
        let (mut fixed_fill, mut fixed_count) = (0.0f32, 0.0f32);
        for (i, l) in self.live.iter().enumerate() {
            let c = lib.get(l.burst.asset).expect("retained with its effect");
            let placed = l.standing.unwrap_or_default();
            let d = l.burst.position.distance(head);
            let pixels = 2.0 * c.reach * placed.scale.max(0.0) / d.max(0.05) * asset::REFERENCE_FOCAL_PIXELS;
            if d > c.scalability.cull_distance || pixels < c.scalability.min_pixels {
                continue;
            }
            let mut lo = l.burst.position - Vec3::splat(c.reach);
            let mut hi = l.burst.position + Vec3::splat(c.reach);
            if let Some((port, _)) = l.burst.port {
                lo = lo.min(port - Vec3::splat(c.reach));
                hi = hi.max(port + Vec3::splat(c.reach));
            }
            let in_view = views.is_empty()
                || views.iter().any(|planes| {
                    let mut unbounded = *planes;
                    unbounded[5] = glam::Vec4::new(0.0, 0.0, 0.0, 1.0);
                    super::super::shadow::aabb_in_frustum(&unbounded, lo, hi)
                });
            if !in_view {
                continue;
            }
            drawn[i] = true;
            if l.lasting {
                // WHAT IT FILLS IN THESE EYES (`seen`): its box clipped to
                // each view, times its particles' layers. A placed one's
                // sprites are `scale` times the size, its streams `rate`
                // times as dense, and a disc that fills its object as many
                // more as the footprint is wider.
                let (fill, count) = l.seen_cost(c, views, head);
                wants.push((c.scalability.importance * 100.0 + pixels.max(1.0).log2(), fill, count, i));
            }
        }
        let bursts = self.live.len();
        for (k, w) in others.iter().enumerate() {
            if w.sheddable {
                wants.push((w.score, w.fill, w.count, bursts + k));
            } else {
                fixed_fill += w.fill;
                fixed_count += w.count;
            }
        }
        // THE GOVERNOR. Lowest score first loses its particles.
        wants.sort_by(|a, b| a.0.total_cmp(&b.0));
        let fill_wanted: f32 = wants.iter().map(|w| w.1).sum::<f32>() + fixed_fill;
        let count_wanted: f32 = wants.iter().map(|w| w.2).sum::<f32>() + fixed_count;
        // Aimed a little under the budget: what it sheds goes a tenth a
        // frame, and new bursts and splashes arrive whole meanwhile.
        let mut over_fill = fill_wanted - GOVERN_AIM * self.budget.fill;
        let mut over_count = count_wanted - GOVERN_AIM * self.budget.instances as f32;
        let mut target = vec![1.0f32; bursts + others.len()];
        for &(_, fill, count, i) in &wants {
            if over_fill <= 0.0 && over_count <= 0.0 {
                break;
            }
            // Wholly, or as far as brings the frame back under.
            let cut = (over_fill / fill.max(1e-6)).max(over_count / count.max(1.0)).clamp(0.0, 1.0);
            target[i] = 1.0 - cut;
            over_fill -= fill * cut;
            over_count -= count * cut;
        }
        let (mut fill, mut count) = (fixed_fill, fixed_count);
        let (mut thinned, mut emitters_thinned, mut splashes_thinned) = (0, 0, 0);
        let mut shares = std::collections::HashMap::with_capacity(others.len());
        for &(_, f, n, i) in &wants {
            let share = if i < bursts {
                let l = &mut self.live[i];
                // One never drawn yet starts where the governor wants it:
                // nothing of it was on screen to go.
                l.share = if l.frames == 0 { target[i] } else { l.share + (target[i] - l.share).clamp(-SHARE_STEP, SHARE_STEP) };
                thinned += (l.share < 1.0) as u32;
                l.share
            } else {
                let w = &others[i - bursts];
                let ramps = w.kind == WantKind::Emitter && !self.weighed.contains(&w.key);
                let was = if ramps { 0.0 } else { self.share_of(w.key) };
                let share = was + (target[i] - was).clamp(-SHARE_STEP, SHARE_STEP);
                if share < 1.0 {
                    shares.insert(w.key, share);
                    match w.kind {
                        WantKind::Emitter => emitters_thinned += 1,
                        WantKind::Splash => splashes_thinned += 1,
                    }
                }
                share
            };
            fill += f * share;
            count += n * share;
        }
        // What was not weighed this frame (out of view) comes back whole --
        // a placed emitter ramping in again when it is next in sight.
        self.shares = shares;
        self.weighed = others.iter().filter(|w| w.kind == WantKind::Emitter && w.sheddable).map(|w| w.key).collect();
        // One out of view comes back whole when the frame has room for it.
        for (i, l) in self.live.iter_mut().enumerate() {
            if !drawn[i] {
                l.share = 1.0;
            }
        }
        let seen = drawn.iter().filter(|d| **d).count() as u32;
        let shed = thinned + emitters_thinned + splashes_thinned > 0;
        self.stats = BurstStats {
            live: self.live.len() as u32,
            seen,
            culled: self.live.len() as u32 - seen,
            fill_wanted,
            fill,
            thinned,
            emitters_thinned,
            splashes_thinned,
            instances_wanted: count_wanted,
            instances: count,
            frames_governed: self.stats.frames_governed + shed as u32,
            ..self.stats
        };
        drawn
    }

    /// The frame is drawn: each burst has been shown one frame more.
    pub fn end_frame(&mut self) {
        for l in self.live.iter_mut().filter(|l| l.born.is_some()) {
            l.frames += 1;
        }
    }

    /// THE LIGHTS THE BURSTS RAISE THIS FRAME, in the player's frame: each
    /// flash's pulse on the frames it lasts, a point light with no shadow --
    /// so the scene readers shade it in their surface-light loop, apart from
    /// the lamps (`Light::is_surface_light`). GATED: a pulse that could not
    /// raise a surface a metre off by two levels of 255 at `exposure` -- any
    /// shot in daylight -- is not raised at all. Asked for before the frame
    /// that draws the flash, as the fires' lights are.
    pub fn lights(&mut self, lib: &EffectLibrary, exposure: f32, offset: Vec3, yaw_inv: Quat) -> Vec<Light> {
        let mut out = Vec::new();
        for l in &self.live {
            let Some(c) = lib.get(l.burst.asset) else { continue };
            let Some((pulse, energy)) = c.pulse_of(l) else { continue };
            let intensity = pulse.intensity * energy * brief_light_scale(exposure);
            if pulse.gate == Gate::VisibleOnly && !pulse_visible(intensity, exposure) {
                self.stats.lights_gated += (l.frames == 0) as u32;
                continue;
            }
            self.stats.lights += (l.frames == 0) as u32;
            out.push(pulse_light(&pulse, intensity, yaw_inv * (l.burst.position + l.burst.direction * (0.4 * c.reach.min(0.2)) - offset)));
        }
        out
    }
}

/// Whether a pulse of `intensity` (as the eye in the room takes it,
/// [`brief_light_scale`]) shows at `exposure`.
pub fn pulse_visible(intensity: f32, exposure: f32) -> bool {
    exposure * PULSE_GATE_ALBEDO * intensity / (PULSE_GATE_DISTANCE * PULSE_GATE_DISTANCE) > PULSE_GATE_LEVELS
}

fn pulse_light(pulse: &LightPulse, intensity: f32, position: Vec3) -> Light {
    let byte = |c: f32| (c.clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0).round() as u8;
    Light {
        position,
        direction: Vec3::NEG_Y,
        kind: LightKind::Point,
        color: super::super::Color3(byte(pulse.colour[0]), byte(pulse.colour[1]), byte(pulse.colour[2]), 255),
        intensity,
        range: pulse.range,
        cone_angle_deg: 0.0,
        inner_cone_angle_deg: 0.0,
        mask_channel: None,
        // No shadow: a surface light (`Light::casts_shadow`). Nobody sees a
        // shadow's edge in 28 ms, and a lamp's shadow slot costs every frame.
        shadow_near: Some(pulse.range + 1.0),
        source_radius: 0.06,
        in_level_bake: false,
    }
}

impl LiveBurst {
    /// WHAT ITS LASTING PARTICLES COST IN THESE EYES, as the governor weighs
    /// it: the fill they make seen through `views` from `head` (`seen`: its
    /// box clipped to each view, times their layers; both reference eyes
    /// looking at it where no views are given) and how many they are. A
    /// placed one's sprites are `scale` times the size, its streams `rate`
    /// times as dense, and a disc that fills its object as many more as the
    /// footprint is wider.
    pub fn seen_cost(&self, c: &CompiledEffect, views: &[[glam::Vec4; 6]], head: Vec3) -> (f32, f32) {
        let placed = self.standing.unwrap_or_default();
        let (lo, hi) = self.seen_box(c);
        let eyes: Vec<super::seen::Eye> = if views.is_empty() {
            vec![super::seen::Eye::looking(head, 0.5 * (lo + hi)); asset::REFERENCE_EYES as usize]
        } else {
            views.iter().filter_map(|p| super::seen::Eye::from_planes(head, p)).collect()
        };
        let fills = placed.fills(c.fills_disc());
        let grown = placed.scale * placed.scale * placed.rate * fills;
        let lasting = || c.report.emitters.iter().filter(|e| e.longest_life > 0.0);
        let facing: f32 = lasting().filter(|e| !e.flat).map(|e| e.area).sum();
        let flat: f32 = lasting().filter(|e| e.flat).map(|e| e.area).sum();
        let points: u32 = lasting().map(|e| e.points).sum();
        let fill = super::seen::seen_fill(&eyes, lo, hi, facing * grown, flat * grown, points as f32 * placed.rate * fills);
        let count: u32 = lasting().map(|e| e.instances).sum();
        (fill, count as f32 * placed.rate * fills)
    }

    /// THE BOX ITS PARTICLES ARE SEEN THROUGH (`seen`), world frame: its
    /// emitters' extent about where it is -- level with the world's up when
    /// it stands upright (a placed effect, smoke rising), a cube of its
    /// widest way otherwise -- a disc that fills its object widened to the
    /// footprint.
    fn seen_box(&self, c: &CompiledEffect) -> (Vec3, Vec3) {
        let e = c.report.extent;
        let placed = self.standing.unwrap_or_default();
        let grow = placed.scale.max(1.0);
        let (mut x, mut z) = (e.lateral, e.lateral);
        if let (Some([fx, fz]), r) = (placed.footprint(), c.fills_disc()) {
            if r > 0.0 {
                x += fx - r;
                z += fz - r;
            }
        }
        let p = self.burst.position;
        let (lo, hi) = if self.burst.direction.normalize_or_zero().y > 0.99 {
            (p - Vec3::new(x, e.below, z) * grow, p + Vec3::new(x, e.above, z) * grow)
        } else {
            let m = x.max(z).max(e.below).max(e.above) * grow;
            (p - Vec3::splat(m), p + Vec3::splat(m))
        };
        match self.burst.port {
            Some((port, _)) => (lo.min(port - Vec3::splat(c.reach)), hi.max(port + Vec3::splat(c.reach))),
            None => (lo, hi),
        }
    }

    /// When nothing of it is left, on the effects' clock.
    fn until(&self, c: &CompiledEffect) -> f64 {
        let Some(born) = self.born else { return f64::INFINITY };
        let mut seconds = 0.0f32;
        for (k, p) in c.programs.iter().enumerate() {
            let s = match (&p.spawn, &p.life) {
                (Spawn::Rate { per_second, for_seconds }, _) => {
                    let window = self.superseded.map_or(*for_seconds, |s| ((s - born) as f32).min(*for_seconds));
                    // A stream taken over before its first particle left
                    // (`simulate_bursts`: particle 0 leaves `phase / rate`
                    // after the event) holds nothing: at 12 shots a second
                    // two in three of a string's bursts, gone the frame
                    // after their flash rather than kept for a thread's life.
                    let rate = per_second * p.by(SpawnTarget::Rate, &self.burst.inputs);
                    let phase = uniforms(hash64(self.seed(c) ^ (k as u64 + 1).wrapping_mul(0x632b_e59b_d9b4_e019)))[1];
                    if self.superseded.is_some() && !(window * rate > phase) {
                        0.0
                    } else {
                        window.min(600.0) + p.seconds - for_seconds.min(600.0)
                    }
                }
                (Spawn::Burst { .. }, Some(_)) if self.lasting => p.seconds,
                _ => 0.0,
            };
            seconds = seconds.max(s);
        }
        // Its one-frame emitters and its light need their frames.
        let frames = c.programs.iter().filter(|p| p.life.is_none()).map(|p| p.delay_frames + 1).max().unwrap_or(0).max(c.light.map_or(0, |l| l.frames));
        if self.frames < frames {
            return f64::INFINITY;
        }
        born + seconds as f64
    }

    fn seed(&self, c: &CompiledEffect) -> u64 {
        if c.locked {
            id_hash(&c.name)
        } else {
            self.burst.seed
        }
    }
}

impl CompiledEffect {
    /// A burst's own draws: whether its emitters fire at all, and its
    /// energy -- one draw each for the whole burst, so its flash, its core
    /// and its light agree.
    fn draws(&self, l: &LiveBurst) -> (f32, f32) {
        let b = uniforms(hash64(l.seed(self) ^ 0xb025_7c0d_e5e7_11a5));
        let by_input = self.energy_input.map_or(1.0, |i| if l.burst.inputs[i] > 0.0 { l.burst.inputs[i] } else { 1.0 });
        (b[0], (self.energy.at(b[1], 0.0) * by_input).clamp(0.05, LOBE_ENERGY_MOST * LOBE_ENERGY_MOST))
    }

    /// Its light pulse and energy, if it raises one on the frame coming.
    fn pulse_of(&self, l: &LiveBurst) -> Option<(LightPulse, f32)> {
        let pulse = self.light?;
        let (chance, energy) = self.draws(l);
        let p = &self.programs[pulse.emitter];
        (l.frames < pulse.frames && p.fires(chance, &l.burst.inputs)).then_some((pulse, energy))
    }
}

impl Program {
    fn by(&self, target: SpawnTarget, inputs: &[f32; 4]) -> f32 {
        self.spawn_by.iter().filter(|(_, t, _)| *t == target).map(|(i, _, table)| table_at(table, inputs[*i].clamp(0.0, 1.0))).product()
    }

    /// Whether a burst whose chance draw is `draw` spawns this emitter.
    fn fires(&self, draw: f32, inputs: &[f32; 4]) -> bool {
        match &self.spawn {
            Spawn::Burst { chance, .. } => draw < chance * self.by(SpawnTarget::Chance, inputs),
            Spawn::Rate { .. } => true,
        }
    }
}

/// Whether particle `j` of a burst is among the `share` of them kept: a
/// fixed order, so a falling share drops them one at a time and a rising
/// one brings the same ones back.
pub(super) fn kept(j: u32, share: f32, phase: f32) -> bool {
    share >= 1.0 || ((j as f32 + 1.0) * 0.618_034 + phase).fract() < share
}

/// What a particle's steps run in: its burst, and the frame.
struct Ctx<'a> {
    at: &'a Surroundings<'a>,
    own: &'a dyn Fn(&Light) -> f32,
    time: f64,
    origin: Vec3,
    dir: Vec3,
    up: Vec3,
    carrier: Vec3,
    inputs: [f32; 4],
    energy: f32,
    lamps: &'a [usize],
    ambient: Vec3,
    /// The burst's own flash this frame: where (player frame) and its light.
    flash: Option<Light>,
    near_fade: Option<[f32; 2]>,
    ceiling: Option<f32>,
    /// A placed effect's own scale and tint ([`Placed`]); 1 for a burst.
    scale: f32,
    tint: Vec3,
    /// The sun as it reaches a placed effect ([`Placed::sunlit`]): its light
    /// and the way toward it, player frame; none for a burst.
    sun: Option<(Vec3, Vec3)>,
    /// A placed effect's object footprint ([`Placed::footprint`]).
    footprint: Option<[f32; 2]>,
}

/// The particle record a program's steps read and write.
struct P {
    u: [f32; 12],
    index: u32,
    age: f32,
    life: f32,
    t: f32,
    p0: Vec3,
    dir: Vec3,
    speed: f32,
    pos: Vec3,
    vel: Vec3,
    moving: Vec3,
    size: f32,
    size0: f32,
    length: f32,
    rotation: f32,
    opacity: f32,
    bright: f32,
    tint: Vec3,
    /// From `light_cpu`: its even light, its strongest lamp's, and the way
    /// to that lamp (player frame).
    lit: Option<(Vec3, Vec3, Vec3)>,
    adapt: Option<(f32, f32)>,
}

impl Program {
    /// One particle at `age` of its life, as the GPU draws it.
    fn particle(&self, cx: &Ctx, index: u32, age: f32, u: [f32; 12]) -> Option<EffectInstance> {
        use std::f32::consts::TAU;
        let at = cx.at;
        let (life, age) = match &self.life {
            Some(v) => (v.at(u[0], 0.0), age),
            None => (1.0, 0.0),
        };
        if !(age >= 0.0 && age < life) {
            return None;
        }
        let emissive = matches!(self.shade, Shade::Emissive { .. } | Shade::EmissiveHeat { .. });
        let mut p = P {
            u,
            index,
            age,
            life,
            t: age / life,
            p0: cx.origin,
            dir: cx.dir,
            speed: 0.0,
            pos: cx.origin,
            vel: Vec3::ZERO,
            moving: Vec3::ZERO,
            size: 0.01,
            size0: 0.01,
            length: 0.0,
            rotation: 0.0,
            opacity: 1.0,
            bright: 1.0,
            tint: cx.tint,
            lit: None,
            adapt: None,
        };
        let player = |w: Vec3| at.yaw_inv * (w - at.offset);
        for step in &self.steps {
            match step {
                Step::Disc(radius, fills) => {
                    let r = radius * p.u[4].sqrt();
                    let side = cx.dir.any_orthonormal_vector();
                    let phi = p.u[5] * TAU;
                    let mut off = (side * phi.cos() + cx.dir.cross(side) * phi.sin()) * r;
                    // Stretched to the object's footprint (world x and z).
                    if let (true, Some([fx, fz])) = (*fills && *radius > 0.0, cx.footprint) {
                        off = Vec3::new(off.x * fx / radius, off.y, off.z * fz / radius);
                    }
                    p.p0 += off;
                    p.pos = p.p0;
                }
                Step::Lobes { lobes, open, rake, skip_down, side_pairs, length } => {
                    let f = cx.dir;
                    let up = (cx.up - f * cx.up.dot(f)).try_normalize().unwrap_or_else(|| f.any_orthonormal_vector());
                    let right = f.cross(up);
                    let j = p.index;
                    let (aim, from, wobble) = if j < *lobes {
                        // Round the bore from the device's up; a closed
                        // bottom leaves the lower third without a jet.
                        let azimuth = match (*skip_down, *lobes) {
                            (true, 1) => 0.0,
                            (true, n) => (-120.0 + 240.0 * j as f32 / (n - 1) as f32).to_radians(),
                            (false, n) => TAU * j as f32 / n as f32,
                        };
                        let radial = up * azimuth.cos() + right * azimuth.sin();
                        let a = open.to_radians();
                        (f * a.cos() + radial * a.sin(), Vec3::ZERO, 3.5)
                    } else if j < lobes + 2 * side_pairs {
                        let k = j - lobes;
                        let side = if k % 2 == 0 { 1.0 } else { -1.0 };
                        let a = (90.0 - rake).to_radians();
                        (f * a.cos() + right * (side * a.sin()), f * (-0.018 * (k / 2) as f32), 5.0)
                    } else {
                        // Past the device's ports: a tongue straight out.
                        (f, Vec3::ZERO, 7.0)
                    };
                    p.dir = cone_direction(aim, wobble, p.u[2], p.u[3]);
                    p.length = length.at(p.u[6], 0.0) * cx.energy.sqrt().min(LOBE_ENERGY_MOST);
                    // A streak's head is at the particle: the lobe's tip.
                    p.p0 = cx.origin + from + p.dir * p.length;
                    p.pos = p.p0;
                }
                Step::Cone(angle, up) => p.dir = cone_direction(if *up { Vec3::Y } else { p.dir }, *angle, p.u[2], p.u[3]),
                Step::Speed(v) => p.speed = v.at(p.u[1], 0.0),
                Step::Size(v) => {
                    p.size = v.at(p.u[8], 0.0) * if emissive { cx.energy.sqrt() } else { 1.0 };
                    p.size0 = p.size;
                }
                Step::Spin { rate, start } => {
                    let way = if p.u[10] < 0.5 { -1.0 } else { 1.0 };
                    p.rotation = (2.0 * p.u[9] - 1.0) * start + rate * p.age * way;
                }
                Step::Ballistic { accel, drag } => {
                    let (pos, vel) = ballistic(p.p0, p.dir * p.speed, *accel, *drag, p.age);
                    p.pos = pos;
                    p.vel = vel;
                    p.moving = vel;
                }
                Step::Swirl { radius, turn } => {
                    let r = radius * p.t;
                    let a = turn * p.age + p.u[7] * TAU;
                    p.pos += Vec3::new(a.cos(), 0.0, a.sin()) * r;
                    p.moving += swirl_motion(radius / p.life, r, *turn, a);
                }
                Step::AirDrift(strength) => {
                    let air = burst_air();
                    let here = air_drift(air, p.p0, cx.time);
                    p.pos += (here - air_drift(air, p.p0, cx.time - p.age as f64)) * *strength;
                    p.moving += (here - air_drift(air, p.p0, cx.time - MOTION_H as f64)) * (*strength / MOTION_H);
                }
                Step::Jitter { reach, pace } => {
                    let seed = hash64(p.u[7].to_bits() as u64 ^ (p.u[9].to_bits() as u64) << 32);
                    let wander = |x: f32| Vec3::new(noise1(x, seed), 0.6 * noise1(x, seed ^ 1), noise1(x, seed ^ 2)) * *reach;
                    let x = p.age * pace;
                    p.pos += wander(x);
                    p.moving += (wander(x) - wander((p.age - MOTION_H) * pace)) / MOTION_H;
                }
                Step::CeilingSpread => {
                    if let Some(top) = cx.ceiling {
                        let under = top - 0.6 * p.size;
                        let excess = p.pos.y - under;
                        if excess > 0.0 {
                            let a = p.u[7] * TAU + 0.3 * p.age;
                            p.pos.y = under - 0.15 * p.size * (1.0 - (-excess).exp());
                            p.pos += Vec3::new(a.cos(), 0.0, a.sin()) * (0.9 * excess);
                            p.size *= 1.0 + 0.5 * excess.min(2.0);
                            let rise = p.moving.y;
                            p.moving.y = 0.15 * p.size * (-excess).exp() * -rise;
                            p.moving += Vec3::new(a.cos(), 0.0, a.sin()) * (0.9 * rise);
                        }
                    }
                }
                Step::CeilingStop => {
                    if let Some(top) = cx.ceiling {
                        if p.pos.y >= top - 0.02 {
                            p.pos.y = top - 0.02;
                            p.moving.y = p.moving.y.min(0.0);
                        }
                    }
                }
                Step::SizeTo(to) => p.size = lerp(p.size0, *to, smoothstep(0.0, 1.0, p.t)),
                Step::SizeBy(v) => p.size *= v.at(p.u[8], p.t),
                Step::Opacity(v) => p.opacity *= v.at(p.u[11], p.t),
                Step::PuffFade { peak, rise, power } => p.opacity *= peak * smoothstep(0.0, *rise, p.t) * (1.0 - p.t).powf(*power),
                Step::Colour(table) => {
                    let f = p.t.clamp(0.0, 1.0) * (CURVE_SAMPLES - 1) as f32;
                    let i = (f as usize).min(CURVE_SAMPLES - 2);
                    let c = glam::Vec4::from(table[i]).lerp(glam::Vec4::from(table[i + 1]), f - i as f32);
                    p.tint *= c.truncate();
                    p.opacity *= c.w;
                }
                Step::ByInput { input, target, table } => {
                    let f = table_at(table, cx.inputs[*input].clamp(0.0, 1.0));
                    match target {
                        UpdateTarget::Opacity => p.opacity *= f,
                        UpdateTarget::Size => p.size *= f,
                        UpdateTarget::Brightness => p.bright *= f,
                    }
                }
                Step::Light { mode, g, albedo, ambient } => {
                    let centre = player(p.pos);
                    let to_eye = (at.head - centre).normalize_or_zero();
                    let mut strongest: Option<(f32, Vec3, Vec3)> = None;
                    let mut rest = cx.ambient * *ambient;
                    let mut take = |c: Vec3, d: Vec3| {
                        // Scattered by its angle to the eye; six-way keeps
                        // its strongest lamp apart, for the puff's own sides.
                        let c = if *mode == LightMode::Scatter { c * henyey_greenstein(*g, -d, to_eye) } else { c };
                        let y = luminance(c);
                        match strongest {
                            _ if *mode == LightMode::Scatter => rest += c,
                            Some((sy, _, _)) if sy >= y => rest += c,
                            Some((_, sc, _)) => {
                                rest += sc;
                                strongest = Some((y, c, d));
                            }
                            None => strongest = Some((y, c, d)),
                        }
                    };
                    for &li in cx.lamps {
                        if let Some((c, d)) = lamp_at(&at.lights[li], centre) {
                            take(c * (cx.own)(&at.lights[li]), d);
                        }
                    }
                    if let Some((c, d)) = cx.flash.as_ref().and_then(|f| lamp_at(f, centre)) {
                        take(c, d);
                    }
                    if let Some((c, d)) = cx.sun {
                        take(c, d);
                    }
                    let a = *albedo * at.exposure;
                    p.lit = Some(match strongest {
                        Some((_, c, d)) => (rest * a, c * a * henyey_greenstein(*g, -d, to_eye), d),
                        None => (rest * a, Vec3::ZERO, Vec3::Y),
                    });
                }
                Step::AdaptOwnLight { radius, level } => p.adapt = Some((*radius, *level)),
            }
        }
        p.size *= cx.scale;
        let centre = player(p.pos);
        let distance = (centre - at.head).length();
        if let Some([near, far]) = cx.near_fade {
            // Near the eye it thins and shrinks: a puff a hand's width off
            // would cover a third of the view.
            let f = smoothstep(near, far, distance);
            p.opacity *= f;
            p.size *= 0.5 + 0.5 * f;
        }
        if !(p.opacity > 0.0) || !(p.size > 0.0) {
            return None;
        }
        let (streak, along) = match self.placement {
            PlacementMode::Streak if p.length > 0.0 => (p.length, p.dir),
            PlacementMode::Streak => (p.vel.length() * self.streak_seconds, p.vel.try_normalize().unwrap_or(p.dir)),
            PlacementMode::Flat => (0.0, p.dir),
            _ => (0.0, p.vel.try_normalize().unwrap_or(Vec3::Y)),
        };
        let axis = at.yaw_inv * along;
        let moving = match (self.motion, &self.life) {
            (MotionLayer::None, _) => Vec3::ZERO,
            // What lives one frame is where its source is: it rides it.
            (_, None) => at.yaw_inv * cx.carrier,
            _ => at.yaw_inv * p.moving,
        };
        let frames = self.book.frames.max(1);
        let variant = ((p.u[10] * self.book.variants as f32) as u32).min(self.book.variants.saturating_sub(1));
        let frame = match self.play {
            Play::Age => p.t * (frames - 1) as f32,
            Play::Seed => ((p.u[11] * frames as f32) as u32).min(frames - 1) as f32,
        };
        let layer = (self.book.first_layer + variant * frames) as f32 + frame;
        let stands = match self.placement {
            PlacementMode::Upright => 1.0,
            PlacementMode::Flat => 2.0,
            _ => 0.0,
        };
        let mut inst = EffectInstance {
            centre: [centre.x, centre.y, centre.z, p.size],
            colour: [0.0; 4],
            light: [0.0, 0.0, 0.0, stands],
            light_dir: [0.0, 1.0, 0.0, streak],
            axis: [axis.x, axis.y, axis.z, self.soft],
            params: [p.rotation, layer, 0.0, 0.0],
            motion: [moving.x, moving.y, moving.z, 0.0],
        };
        match self.shade {
            Shade::Sixway | Shade::Plain | Shade::Scatter => {
                let (base, lit, d) = p.lit.unwrap_or((cx.ambient * at.exposure, Vec3::ZERO, Vec3::Y));
                let (base, lit) = (base * p.tint * p.bright, lit * p.tint * p.bright);
                if self.shade == Shade::Sixway {
                    inst.colour = [base.x, base.y, base.z, p.opacity];
                    inst.light = [lit.x, lit.y, lit.z, stands];
                    inst.light_dir = [d.x, d.y, d.z, streak];
                    inst.params[3] = 1.0;
                } else {
                    let c = base + lit;
                    inst.colour = [c.x, c.y, c.z, p.opacity.min(1.0)];
                    if self.placement == PlacementMode::Point {
                        inst.params[3] = -1.0;
                    }
                }
            }
            Shade::Emissive { radiance, colour, own } | Shade::EmissiveHeat { radiance, colour, own } => {
                // As the eye adapted to it sees it, or -- too brief for that --
                // as the eye in this room does.
                let adapted = match (own, p.adapt) {
                    (true, Some((radius, level))) => {
                        crate::renderer::tonemap::own_light_scale(at.exposure, radiance, level, crate::renderer::tonemap::bulb_adaptation(radius, distance, 1.0))
                    }
                    _ => brief_source_scale(at.exposure),
                };
                let c = colour * p.tint * (radiance * at.exposure * adapted * p.bright * cx.energy.sqrt());
                inst.colour = [c.x, c.y, c.z, p.opacity.min(1.0)];
                inst.params[3] = match (self.shade, self.placement) {
                    (Shade::EmissiveHeat { .. }, _) => SHADE_SCREEN_HEAT,
                    (_, PlacementMode::Point) => -1.0,
                    _ => 0.0,
                };
            }
        }
        Some(inst)
    }
}

/// What simulating the bursts counted.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BurstCount {
    pub over: u32,
    pub screened: u32,
    /// Particles the governor's share left out.
    pub shed: u32,
    /// The same for the placed emitters and the splashes
    /// (`effects::simulate_governed`).
    pub emitter_shed: u32,
    pub splash_shed: u32,
}

/// The bursts' part of a frame: the library, the bursts in play and which
/// of them are drawn ([`BurstSet::frame`]).
#[derive(Clone, Copy)]
pub struct BurstFrame<'a> {
    pub lib: &'a EffectLibrary,
    pub set: &'a BurstSet,
    pub drawn: &'a [bool],
}

/// THE BURSTS' PARTICLES at `time`, the clock `BurstSet::frame` was given.
pub(super) fn simulate_bursts(
    bursts: &BurstFrame,
    time: f64,
    at: &Surroundings,
    own: &dyn Fn(&Light) -> f32,
    over: &mut Vec<(f32, EffectInstance)>,
    screened: &mut Vec<EffectInstance>,
) -> BurstCount {
    let mut count = BurstCount::default();
    let set = bursts.set;
    // What each source says NOW: its newest burst's inputs, found once (a
    // search a burst made the frame's work go as the square of the bursts).
    let mut newest: std::collections::HashMap<(EffectId, u64), [f32; 4]> = std::collections::HashMap::with_capacity(set.live.len());
    for l in &set.live {
        newest.insert((l.burst.asset, l.burst.source), l.burst.inputs);
    }
    for (i, l) in set.live.iter().enumerate() {
        if !bursts.drawn.get(i).copied().unwrap_or(false) {
            continue;
        }
        let (Some(c), Some(born)) = (bursts.lib.get(l.burst.asset), l.born) else { continue };
        let placed = l.standing.unwrap_or_default();
        let age = (time - born) as f32;
        let seed = l.seed(c);
        let (chance, energy) = c.draws(l);
        // The newest burst's inputs, so the puffs already in the air
        // thicken as the string goes on.
        let inputs = newest.get(&(l.burst.asset, l.burst.source)).copied().unwrap_or(l.burst.inputs);
        // The lamps of the room it is in, as an emitter there would have.
        let stand_in = EffectEmitter {
            id: String::new(),
            kind: EffectKind::Smoke,
            position: l.burst.position,
            direction: l.burst.direction,
            extent: [Vec3::ZERO; 3],
            scale: 1.0,
            rate: 1.0,
            tint: [1.0; 3],
            ceiling: None,
            variation: Variation::default(),
        };
        let lamps: Vec<usize> = (0..at.lights.len()).filter(|&k| (at.reaches)(&stand_in, k)).collect();
        // The sky's sun, where it reaches a placed effect (`lamp_at` leaves
        // suns out: nothing else knows whether one reaches the particles).
        let sun = (placed.sunlit > 0.0)
            .then(|| at.lights.iter().find(|l| matches!(l.kind, LightKind::Directional) && l.intensity > 0.0))
            .flatten()
            .map(|l| {
                let c = l.color.to_linear();
                (Vec3::new(c[0], c[1], c[2]) * (l.intensity * placed.sunlit.min(1.0)), -l.direction.normalize_or_zero())
            });
        let flash = c.pulse_of(l).map(|(pulse, energy)| {
            let intensity = pulse.intensity * energy * brief_light_scale(at.exposure);
            pulse_light(&pulse, intensity, at.yaw_inv * (l.burst.position + l.burst.direction * (0.4 * c.reach.min(0.2)) - at.offset))
        });
        for (k, p) in c.programs.iter().enumerate() {
            let (origin, dir) = match (p.at_port, l.burst.port) {
                (false, _) => (l.burst.position, l.burst.direction),
                (true, Some(port)) => port,
                (true, None) => continue,
            };
            if !p.fires(chance, &l.burst.inputs) {
                continue;
            }
            let cx = Ctx {
                at,
                own,
                time,
                origin,
                dir: dir.try_normalize().unwrap_or(Vec3::NEG_Z),
                up: l.burst.up,
                carrier: l.burst.velocity,
                inputs,
                energy,
                lamps: &lamps,
                ambient: (at.ambient)(origin),
                flash,
                near_fade: c.scalability.near_fade,
                ceiling: None,
                scale: placed.scale,
                tint: Vec3::from(placed.tint),
                sun,
                footprint: placed.footprint(),
            };
            let own_seed = hash64(seed ^ (k as u64 + 1).wrapping_mul(0x632b_e59b_d9b4_e019));
            // Each burst sheds in an order of its own: a row of them thinned
            // together does not lose the same puff of each on one frame.
            let shed_phase = uniforms(own_seed)[2];
            let mut push = |inst: EffectInstance, count: &mut BurstCount| {
                if p.shade.screened() {
                    count.screened += 1;
                    screened.push(inst);
                } else {
                    count.over += 1;
                    over.push(((Vec3::from_slice(&inst.centre[..3]) - at.head).length_squared(), inst));
                }
            };
            match &p.spawn {
                Spawn::Burst { count: n, .. } => {
                    let one_frame = p.life.is_none();
                    if one_frame && l.frames != p.delay_frames || !one_frame && !l.lasting {
                        continue;
                    }
                    let n = (n.at(uniforms(own_seed)[0], 0.0) * p.by(SpawnTarget::Count, &l.burst.inputs)).round().max(0.0) as u32;
                    for j in 0..n {
                        if !one_frame && !kept(j, l.share, shed_phase) {
                            count.shed += 1;
                            continue;
                        }
                        let u = uniforms(hash64(own_seed ^ (j as u64 + 1).wrapping_mul(0x8cb9_2ba7_2f3d_8dd7)));
                        if let Some(inst) = p.particle(&cx, j, age, u) {
                            push(inst, &mut count);
                        }
                    }
                }
                Spawn::Rate { per_second, for_seconds } => {
                    // A STREAM: particle `j` leaves `(j + phase) / rate`
                    // after the event, until the window closes or a later
                    // burst of the source takes the stream over.
                    let rate = per_second * p.by(SpawnTarget::Rate, &l.burst.inputs) * placed.rate * placed.fills(p.fills_disc);
                    if !(rate > 0.0) {
                        continue;
                    }
                    let window = l.superseded.map_or(*for_seconds, |s| ((s - born) as f32).min(*for_seconds));
                    let phase = uniforms(own_seed)[1];
                    let longest = p.seconds - for_seconds.min(600.0);
                    let first = (((age - longest) * rate - phase).ceil().max(0.0)) as u32;
                    let last = (age.min(window) * rate - phase).floor();
                    if last < first as f32 {
                        continue;
                    }
                    for j in first..=last as u32 {
                        let left = (j as f32 + phase) / rate;
                        if left >= window {
                            break;
                        }
                        if !kept(j, l.share, shed_phase) {
                            count.shed += 1;
                            continue;
                        }
                        let u = uniforms(hash64(own_seed ^ (j as u64 + 1).wrapping_mul(0x8cb9_2ba7_2f3d_8dd7)));
                        if let Some(inst) = p.particle(&cx, j, age - left, u) {
                            push(inst, &mut count);
                        }
                    }
                }
            }
        }
    }
    count
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    fn every(_: &EffectEmitter, _: usize) -> bool {
        true
    }
    fn dim(_: Vec3) -> Vec3 {
        Vec3::splat(0.1)
    }
    fn seen_from(head: Vec3, exposure: f32) -> Surroundings<'static> {
        Surroundings { head, offset: Vec3::ZERO, yaw_inv: Quat::IDENTITY, lights: &[], reaches: &every, ambient: &dim, exposure }
    }

    /// The shipped effects (`game/effects`), compiled; `None` where the game
    /// folder is not beside this crate.
    pub(crate) fn shipped() -> Option<EffectLibrary> {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../game/effects");
        if !dir.is_dir() {
            eprintln!("skipping: no {}", dir.display());
            return None;
        }
        let (lib, errors) = EffectLibrary::from_dir(&dir);
        assert!(errors.is_empty(), "{}", errors.join("\n"));
        Some(lib)
    }

    fn muzzle() -> (Vec3, Vec3) {
        (Vec3::new(0.0, 1.4, 0.0), Vec3::NEG_Z)
    }

    fn shot(lib: &EffectLibrary, name: &str, seed: u64) -> EffectBurst {
        let (at, dir) = muzzle();
        EffectBurst { port: Some((at + Vec3::new(0.03, 0.0, 0.25), Vec3::new(1.0, 0.3, 0.2).normalize())), ..EffectBurst::new(lib.id(name).unwrap(), at, dir, 1, seed) }
    }

    /// One frame of a set at `time`, every burst drawn.
    fn frame(lib: &EffectLibrary, set: &mut BurstSet, time: f64, at: &Surroundings) -> EffectFrame {
        let drawn = set.frame(lib, time, &[], at.head);
        let f = simulate_all(&[], &|_| true, &[], Some(&BurstFrame { lib, set, drawn: &drawn }), time, at);
        set.end_frame();
        f
    }

    /// EVERY SHIPPED EFFECT LOADS, checks and compiles inside its budgets,
    /// and says what it costs: `cargo test --release --lib
    /// every_shipped_effect -- --nocapture` prints the table.
    #[test]
    fn every_shipped_effect_compiles_inside_its_budgets() {
        let Some(lib) = shipped() else { return };
        assert!(lib.len() >= 3, "{:?}", lib.names().collect::<Vec<_>>());
        for name in ["muzzle_556_a2", "muzzle_9mm_bare", "muzzle_suppressed"] {
            let c = lib.get(lib.id(name).expect(name)).unwrap();
            let r = &c.report;
            eprintln!(
                "{name}: one burst {} instances, {:.3} megapixel-layers at {} m; at its cap {} instances, {:.3}; {} light; reach {:.2} m; lasts {:.1} s",
                r.instances_per_burst, r.fill_per_burst, r.reference_distance, r.instances_steady, r.fill_steady, r.lights, c.reach, c.seconds
            );
            for e in &r.emitters {
                eprintln!("    {:>16}: {:>3} instances, {:.4} fill, longest {:.2} s{}", e.name, e.instances, e.fill, e.longest_life, if e.stream { ", a stream" } else { "" });
            }
            assert!(r.over_budget.is_none(), "{name} ships over a budget");
            assert!(r.instances_steady <= asset::LIMIT_INSTANCES && r.fill_steady <= asset::LIMIT_FILL_STEADY && r.fill_per_burst <= asset::LIMIT_FILL_BURST);
        }
    }

    /// EVERY EFFECT ASSET IN `game/effects` -- what the editor saves -- reads,
    /// passes the validator against the books that exist and compiles to a
    /// program the engine runs; and every one that exceeds a budget says why.
    /// The built-in templates are checked the same way. `cargo test --release
    /// --lib every_effect_asset -- --nocapture` LISTS THE OVERRIDES: each
    /// asset over a budget, its reason, and what it exceeds.
    #[test]
    fn every_effect_asset_validates_and_its_overrides_are_listed() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../game/effects");
        let (assets, failed) = EffectAsset::load_dir(&dir);
        assert!(failed.is_empty(), "{}", failed.iter().map(|(p, e)| format!("{}: {e}", p.display())).collect::<Vec<_>>().join("\n"));
        let mut overrides = Vec::new();
        let shipped = assets.len();
        for (a, origin) in assets.into_iter().map(|a| (a, "game/effects")).chain(templates::assets().into_iter().map(|a| (a, "built-in template"))) {
            let report = a.check(books()).unwrap_or_else(|issues| panic!("{origin} `{}` is refused:\n  {}", a.name, issues.iter().map(|i| i.to_string()).collect::<Vec<_>>().join("\n  ")));
            if origin == "game/effects" {
                CompiledEffect::compile(&a, books()).unwrap_or_else(|issues| panic!("`{}` checks but does not compile: {}", a.name, issues.iter().map(|i| i.to_string()).collect::<Vec<_>>().join("; ")));
            }
            eprintln!(
                "{origin:>17} {:<20} {:>4} instances, {:.3} megapixel-layers at {} m (one burst {} / {:.3}){}",
                a.name,
                report.instances_steady,
                report.fill_steady,
                report.reference_distance,
                report.instances_per_burst,
                report.fill_per_burst,
                if report.over_budget.is_some() { "  OVER BUDGET" } else { "" }
            );
            if let Some((reason, what)) = report.over_budget {
                assert!(!reason.trim().is_empty());
                overrides.push(format!("{origin} `{}`: \"{reason}\"\n      exceeds: {}", a.name, if what.is_empty() { "nothing now".to_string() } else { what.join("; ") }));
            }
        }
        eprintln!("\nOVERRIDES ({} of {} assets and {} templates):", overrides.len(), shipped, templates::assets().len());
        for o in &overrides {
            eprintln!("  {o}");
        }
        assert!(shipped >= 3, "game/effects holds {shipped} assets");
    }

    /// A FLASH IS ONE APP FRAME, whatever the frame time; its smoke goes on.
    /// And the same shot is the same particles, and no two shots are.
    #[test]
    fn a_flash_lives_one_frame_and_its_smoke_goes_on() {
        let Some(lib) = shipped() else { return };
        let at = seen_from(Vec3::new(0.0, 1.5, 0.7), 1.0);
        let hot = |f: &EffectFrame| f.instances.iter().filter(|i| i.params[3] == SHADE_SCREEN_HEAT).count();
        let mut flashes = 0;
        for seed in 0..40u64 {
            for dt in [1.0 / 36.0, 1.0 / 72.0, 0.2] {
                let mut set = BurstSet::default();
                set.add(&lib, shot(&lib, "muzzle_556_a2", seed));
                let first = frame(&lib, &mut set, 100.0, &at);
                let second = frame(&lib, &mut set, 100.0 + dt, &at);
                assert_eq!(hot(&second), 0, "a flash on its second frame (dt {dt})");
                assert_eq!(first.screened_heat as usize, hot(&first));
                assert!(second.over >= 3, "its smoke goes on: {} puffs", second.over);
                if dt == 0.2 {
                    flashes += (hot(&first) > 0) as u32;
                    assert!(hot(&first) == 0 || hot(&first) == 6, "five jets and a core, or nothing: {}", hot(&first));
                }
            }
        }
        // A birdcage hides its flash on some shots: the asset says 0.55.
        assert!((12..=32).contains(&flashes), "{flashes} of 40 shots flashed");
        let one = |seed| {
            let mut set = BurstSet::default();
            set.add(&lib, shot(&lib, "muzzle_9mm_bare", seed));
            frame(&lib, &mut set, 5.0, &at).instances
        };
        assert_eq!(bytemuck::cast_slice::<_, u8>(&one(3)), bytemuck::cast_slice::<_, u8>(&one(3)));
        assert_ne!(bytemuck::cast_slice::<_, u8>(&one(3)), bytemuck::cast_slice::<_, u8>(&one(4)));
    }

    /// A birdcage's jets leave upward and sideways and none downward; a bare
    /// muzzle's tongue leaves forward.
    #[test]
    fn a_flash_hiders_jets_leave_no_lobe_downward() {
        let Some(lib) = shipped() else { return };
        let at = seen_from(Vec3::new(0.0, 1.5, 0.7), 1.0);
        let (from, bore) = muzzle();
        let (mut jets, mut tongues) = (0, 0);
        for seed in 0..60u64 {
            for (name, hider) in [("muzzle_556_a2", true), ("muzzle_9mm_bare", false)] {
                let mut set = BurstSet::default();
                set.add(&lib, shot(&lib, name, seed));
                for i in frame(&lib, &mut set, 9.0, &at).instances.iter().filter(|i| i.params[3] == SHADE_SCREEN_HEAT && i.light_dir[3] > 0.0) {
                    let (tip, along) = (Vec3::from_slice(&i.centre[..3]), Vec3::from_slice(&i.axis[..3]));
                    assert!((tip - along * i.light_dir[3] - from).length() < 0.03, "{name}: a jet's root is at the muzzle");
                    if hider {
                        jets += 1;
                        assert!(along.y > -0.45, "{name}: a jet leaves downward ({along})");
                        assert!(along.dot(bore) > 0.5 && along.dot(bore) < 0.9, "{name}: opened from the bore");
                    } else {
                        tongues += 1;
                        assert!(along.dot(bore) > 0.98, "{name}: a bare muzzle's tongue is forward");
                    }
                }
            }
        }
        assert!(jets >= 100 && tongues >= 60, "{jets} jets, {tongues} tongues");
    }

    /// A FLASH IN THE DARK IS BLINDING AND IN THE SUN IS FAINT, by one rule:
    /// it is seen by the eye the room has left (`brief_source_scale`). And
    /// its light on the room is raised only where it could be seen.
    #[test]
    fn a_flash_is_seen_at_the_rooms_exposure_and_its_light_only_where_it_shows() {
        let Some(lib) = shipped() else { return };
        let mut seed = 0;
        let mut flash_at = |exposure: f32| loop {
            seed += 1;
            let mut set = BurstSet::default();
            set.add(&lib, shot(&lib, "muzzle_556_a2", seed));
            let lights = set.lights(&lib, exposure, Vec3::ZERO, Quat::IDENTITY);
            let f = frame(&lib, &mut set, 1.0, &seen_from(Vec3::new(0.0, 1.5, 0.7), exposure));
            if f.screened_heat > 0 {
                let brightest = f.instances.iter().filter(|i| i.params[3] == SHADE_SCREEN_HEAT).map(|i| i.colour[0]).fold(0.0, f32::max);
                break (brightest, lights, set.stats);
            }
        };
        let (dark, lit, stats) = flash_at(9.0);
        assert!(dark > 8.0, "far past white in a dim room: {dark}");
        assert_eq!((lit.len(), stats.lights, stats.lights_gated), (1, 1, 0));
        assert!(lit[0].is_surface_light() && !lit[0].casts_shadow() && !lit[0].in_level_bake);
        assert!((lit[0].position - muzzle().0).length() < 0.2);
        let (sun, lit, stats) = flash_at(SUN_EXPOSURE);
        assert!(sun < 1.5 && sun > 0.1, "faint in daylight, and there: {sun}");
        assert!(dark > 20.0 * sun, "by more than the exposures' ratio: {dark} against {sun}");
        // No daylight shot raises its light, however much energy it drew.
        let c = lib.get(lib.id("muzzle_556_a2").unwrap()).unwrap();
        let Val::Plain(energy) = &c.energy else { panic!("an energy is a draw") };
        let most = c.light.unwrap().intensity * energy.bounds().1 * brief_light_scale(SUN_EXPOSURE);
        assert!(!pulse_visible(most, SUN_EXPOSURE) && pulse_visible(c.light.unwrap().intensity * brief_light_scale(3.2), 3.2));
        assert_eq!((lit.len(), stats.lights, stats.lights_gated), (0, 0, 1), "its light is not raised where it could not be seen");
        // The suppressed weapon raises none at all, and glows only on the
        // first shot of a string.
        let mut set = BurstSet::default();
        let cold = shot(&lib, "muzzle_suppressed", 5);
        set.add(&lib, cold);
        assert!(set.lights(&lib, 4.0, Vec3::ZERO, Quat::IDENTITY).is_empty());
        assert_eq!(frame(&lib, &mut set, 1.0, &seen_from(Vec3::new(0.0, 1.5, 0.7), 4.0)).screened_heat, 1, "the first round's pop");
        let mut set = BurstSet::default();
        set.add(&lib, EffectBurst { inputs: [0.5, 0.0, 0.0, 0.0], ..cold });
        assert_eq!(frame(&lib, &mut set, 1.0, &seen_from(Vec3::new(0.0, 1.5, 0.7), 4.0)).screened_heat, 0, "none in a string");
    }

    /// AUTOMATIC FIRE COSTS WHAT A FEW SHOTS DO: one source's lasting
    /// particles are held to `max_live` bursts, evenly spaced, each shot
    /// still flashing; the puffs in the air thicken with the source's heat;
    /// and all of it is gone when its longest life has passed.
    #[test]
    fn automatic_fire_is_held_to_its_cap_and_thickens_instead() {
        let Some(lib) = shipped() else { return };
        let at = seen_from(Vec3::new(0.0, 1.5, 0.7), 1.0);
        let c = lib.get(lib.id("muzzle_556_a2").unwrap()).unwrap();
        let mut set = BurstSet::default();
        let (mut most, mut flashes, mut opacity_cold, mut opacity_hot) = (0u32, 0, 0.0f32, 0.0f32);
        // Three seconds at ten shots a second, drawn at 36 frames a second.
        for k in 0..108 {
            let t = 50.0 + k as f64 / 36.0;
            if k % 4 == 0 && k < 108 {
                let heat = (k as f32 / 36.0 / 2.0).min(1.0);
                set.add(&lib, EffectBurst { inputs: [heat, 0.0, 0.0, 0.0], ..shot(&lib, "muzzle_556_a2", 1000 + k as u64) });
            }
            let f = frame(&lib, &mut set, t, &at);
            most = most.max(f.over);
            flashes += (f.screened_heat > 0) as u32;
            let mean = f.instances[..f.over as usize].iter().map(|i| i.colour[3]).sum::<f32>() / f.over.max(1) as f32;
            if k == 12 {
                opacity_cold = mean;
            }
            if k == 100 {
                opacity_hot = mean;
            }
        }
        assert!(most <= c.report.instances_steady, "{most} puffs alive; the cap is {}", c.report.instances_steady);
        assert!(most >= 12, "a haze, not a puff: {most}");
        assert!(set.stats.merged >= 12, "most shots of a string add no puffs: {} merged", set.stats.merged);
        assert!(flashes >= 8, "each shot may still flash: {flashes} frames with one");
        assert!(opacity_hot > 1.2 * opacity_cold, "the haze thickens with heat: {opacity_cold} -> {opacity_hot}");
        // The string over: the barrel's thread for a few seconds, then nothing.
        let later = frame(&lib, &mut set, 53.0 + 2.5, &at);
        assert!(later.over >= 2 && later.over <= 14, "a thread from the hot barrel: {}", later.over);
        let gone = frame(&lib, &mut set, 53.0 + c.seconds as f64 + 0.1, &at);
        assert_eq!((gone.instances.len(), set.live.len()), (0, 0), "retired when its longest life has passed");
    }

    /// EVERY PARTICLE STAYS INSIDE ITS BURST'S TRAVEL BOUND, which is what a
    /// burst is culled by: none can be drawn from a burst judged out of view.
    #[test]
    fn every_particle_stays_within_its_bursts_reach() {
        let Some(lib) = shipped() else { return };
        let at = seen_from(Vec3::new(0.0, 1.5, 3.0), 1.0);
        for name in lib.names().map(str::to_string).collect::<Vec<_>>() {
            let c = lib.get(lib.id(&name).unwrap()).unwrap();
            let mut farthest = 0.0f32;
            for seed in 0..30u64 {
                let mut set = BurstSet::default();
                let b = EffectBurst { inputs: [1.0, 0.0, 0.0, 0.0], ..shot(&lib, &name, seed) };
                set.add(&lib, b);
                for k in 0..80 {
                    for i in &frame(&lib, &mut set, 10.0 + k as f64 * 0.07, &at).instances {
                        let p = Vec3::from_slice(&i.centre[..3]);
                        let from = b.position.distance(p).min(b.port.unwrap().0.distance(p));
                        farthest = farthest.max(from + i.centre[3]);
                    }
                }
            }
            assert!(farthest <= c.reach, "{name}: a particle {farthest} m out; its bound is {}", c.reach);
            assert!(farthest > 0.05 * c.reach, "{name}: the bound {} is far looser than it need be ({farthest})", c.reach);
        }
    }

    /// EVERY PARTICLE CARRIES ITS VELOCITY for SpaceWarp
    /// (`EffectInstance::motion`): a moment later it is where that took it.
    /// A flash, alive one frame, carries its muzzle's.
    #[test]
    fn every_burst_particle_carries_its_velocity() {
        let Some(lib) = shipped() else { return };
        let at = seen_from(Vec3::new(0.3, 1.5, 1.2), 1.0);
        for name in lib.names().map(str::to_string).collect::<Vec<_>>() {
            let (mut moving, mut found, mut seen) = (0, 0, 0);
            for seed in 0..12u64 {
                let recoil = Vec3::new(0.1, 0.6, 1.5);
                let b = EffectBurst { velocity: recoil, inputs: [0.6, 0.0, 0.0, 0.0], ..shot(&lib, &name, seed) };
                let h = 1e-3;
                for t in [0.0f64, 0.02, 0.2, 0.7] {
                    let at_time = |dt: f64| {
                        let mut set = BurstSet::default();
                        set.add(&lib, b);
                        let first = frame(&lib, &mut set, 30.0, &at);
                        if t + dt == 0.0 {
                            first
                        } else {
                            frame(&lib, &mut set, 30.0 + t + dt, &at)
                        }
                    };
                    let (a, later) = (at_time(0.0), at_time(h));
                    for i in &a.instances {
                        let v = Vec3::from_slice(&i.motion[..3]);
                        if i.params[3] == SHADE_SCREEN_HEAT {
                            assert_eq!(v, recoil, "{name}: a flash rides its muzzle");
                            continue;
                        }
                        seen += 1;
                        // (A millimetre a second: a pond's mist drifts at a few.)
                        if v.length() < 0.001 {
                            continue;
                        }
                        moving += 1;
                        let next = Vec3::from_slice(&i.centre[..3]) + v * h as f32;
                        let nearest = later.instances.iter().map(|j| Vec3::from_slice(&j.centre[..3]).distance(next)).fold(f32::INFINITY, f32::min);
                        found += (nearest < 0.1 * v.length() * h as f32 + 2e-5) as u32;
                    }
                }
            }
            // Ten at least, or all of them: a slow stream (pond_mist, one a
            // second) has only a handful in the first 0.7 s after it starts.
            assert!(moving >= 10.min(seen) && seen > 0, "{name}: {moving} of {seen} moving");
            assert!(found as f32 >= 0.95 * moving as f32, "{name}: {found} of {moving} found where their velocity took them");
        }
    }

    /// A BURST OUT OF SIGHT, OUT OF RANGE OR TOO SMALL TO SEE costs a bounds
    /// test: it is not simulated. And it is still retired on time.
    #[test]
    fn a_burst_out_of_view_or_range_is_not_simulated() {
        let Some(lib) = shipped() else { return };
        let head = Vec3::new(0.0, 1.5, 0.7);
        let at = seen_from(head, 1.0);
        // An eye at `head` looking down -z.
        let vp = glam::Mat4::perspective_rh(1.6, 1.0, 0.03, 1000.0) * glam::Mat4::look_at_rh(head, head + Vec3::NEG_Z, Vec3::Y);
        let views = [crate::renderer::shadow::frustum_planes(vp)];
        let mut set = BurstSet::default();
        let ahead = shot(&lib, "muzzle_556_a2", 1);
        let behind = EffectBurst { position: head + Vec3::Z * 30.0, port: None, source: 2, ..ahead };
        let far = EffectBurst { position: head - Vec3::Z * 200.0, port: None, source: 3, ..ahead };
        for b in [ahead, behind, far] {
            set.add(&lib, b);
        }
        let drawn = set.frame(&lib, 1.0, &views, head);
        assert_eq!(drawn, [true, false, false]);
        assert_eq!((set.stats.live, set.stats.seen, set.stats.culled), (3, 1, 2));
        let all = simulate_all(&[], &|_| true, &[], Some(&BurstFrame { lib: &lib, set: &set, drawn: &drawn }), 1.0, &at);
        let one = simulate_all(&[], &|_| true, &[], Some(&BurstFrame { lib: &lib, set: &set, drawn: &[true, false, false] }), 1.0, &at);
        assert_eq!(all.instances.len(), one.instances.len());
        assert!(all.instances.iter().all(|i| i.centre[2] < head.z));
        set.end_frame();
        assert!(set.frame(&lib, 1.0 + 20.0, &views, head).is_empty() && set.live.is_empty());
    }

    /// A stream as a placed asset would be: puffs rising from a vent.
    pub(crate) const VENT: &str = r#"{
      "name": "test_vent", "version": 1,
      "emitters": [{
        "name": "steam",
        "spawn": [{ "module": "rate", "per_second": 10 }],
        "initialize": [
          { "module": "life", "value": { "range": [1.0, 2.0] } },
          { "module": "shape_cone", "angle_deg": 12 },
          { "module": "speed", "value": { "range": [0.6, 1.0] } },
          { "module": "size", "value": 0.05 }],
        "update": [
          { "module": "ballistic", "accel": [0, 0.4, 0], "drag": 1.2 },
          { "module": "size_over_life", "to": 0.3 },
          { "module": "puff_fade", "peak": 0.3, "rise": 0.15, "power": 1.5 },
          { "module": "light_cpu", "mode": "sixway", "phase_g": 0.4, "albedo": [0.8, 0.8, 0.8] }],
        "render": [
          { "module": "sprite_sixway", "book": "smoke" },
          { "module": "placement", "mode": "facing" },
          { "module": "flipbook" },
          { "module": "soft_depth", "metres": 0.2 },
          { "module": "motion", "layer": "soft" }]
      }],
      "scalability": { "cull_distance": 60, "min_pixels": 2, "importance": 0.5, "max_live": 1 },
      "budget": { "fill_megapixel_layers": 1.5, "instances": 40 }
    }"#;

    /// Every generic module once, for the editor's port to be held to.
    const SINK: &str = r#"{
      "name": "fixture_sink", "version": 1, "inputs": ["heat"],
      "energy": { "lognormal": [1.0, 0.3] },
      "emitters": [
        { "name": "glow",
          "spawn": [{ "module": "burst", "count": { "range": [3, 6] }, "chance": 0.9 }],
          "initialize": [
            { "module": "life", "value": { "squared": [0.3, 0.8] } },
            { "module": "shape_disc", "radius": 0.05 },
            { "module": "shape_cone", "angle_deg": 25 },
            { "module": "speed", "value": { "lognormal": [2.0, 0.3] } },
            { "module": "size", "value": { "log_squared": [0.01, 0.03] } },
            { "module": "spin", "rate": 2.0, "start": 0.5 }],
          "update": [
            { "module": "ballistic", "accel": [0, -2, 0], "drag": 3.0 },
            { "module": "jitter", "reach": 0.02, "pace": 1.5 },
            { "module": "size_over_life", "value": { "curve": [[0, 1], [0.5, 1.6], [1, 0.4]] } },
            { "module": "colour_over_life", "gradient": [[0, 1, 0.8, 0.5, 1], [0.4, 1, 0.5, 0.2, 0.9], [1, 0.6, 0.2, 0.1, 0.3]] },
            { "module": "by_input", "input": "heat", "target": "brightness", "curve": [[0, 0.5], [1, 1.5]] },
            { "module": "adapt_own_light", "radius": 0.05, "level": 3.0 }],
          "render": [
            { "module": "sprite_emissive", "book": "ember", "radiance": 30, "colour": [1, 0.5, 0.2], "adapt": "own_light" },
            { "module": "placement", "mode": "streak", "streak_seconds": 0.03 },
            { "module": "soft_depth", "metres": 0.02 },
            { "module": "motion", "layer": "bright" }] },
        { "name": "haze",
          "spawn": [{ "module": "rate", "per_second": 8, "for_seconds": 1.5 }, { "module": "by_input", "input": "heat", "target": "rate", "curve": [[0, 0.5], [1, 1]] }],
          "initialize": [
            { "module": "life", "value": { "range": [0.8, 1.6] } },
            { "module": "shape_cone", "angle_deg": 15, "about": "up" },
            { "module": "speed", "value": { "uniform": [0.3, 0.6] } },
            { "module": "size", "value": 0.04 }],
          "update": [
            { "module": "ballistic", "accel": [0, 0.3, 0], "drag": 1.5 },
            { "module": "swirl", "radius": 0.05, "turn": 1.2 },
            { "module": "air_drift", "strength": 0.7 },
            { "module": "size_over_life", "to": 0.2 },
            { "module": "opacity_over_life", "value": { "curve": [[0, 0], [0.2, 0.3], [1, 0]] } },
            { "module": "puff_fade", "peak": 0.9, "rise": 0.1, "power": 0.5 },
            { "module": "by_input", "input": "heat", "target": "opacity", "curve": [[0, 0.6], [1, 1]] },
            { "module": "light_cpu", "mode": "sixway", "phase_g": 0.3, "albedo": [0.7, 0.7, 0.75] }],
          "render": [
            { "module": "sprite_sixway", "book": "smoke" },
            { "module": "placement", "mode": "facing" },
            { "module": "flipbook" },
            { "module": "soft_depth", "metres": 0.2 },
            { "module": "motion", "layer": "soft" }] },
        { "name": "motes",
          "spawn": [{ "module": "burst", "count": 12 }],
          "initialize": [
            { "module": "life", "value": { "range": [1, 2] } },
            { "module": "shape_point" },
            { "module": "shape_cone", "angle_deg": 180 },
            { "module": "speed", "value": { "range": [0, 0.2] } },
            { "module": "size", "value": 0.002 }],
          "update": [
            { "module": "ballistic", "drag": 0.5 },
            { "module": "light_cpu", "mode": "scatter", "phase_g": 0.6, "albedo": [0.9, 0.9, 0.9], "ambient": 0.3 }],
          "render": [
            { "module": "sprite_scatter", "book": "mote" },
            { "module": "placement", "mode": "point" },
            { "module": "flipbook", "play": "seed" },
            { "module": "soft_depth", "metres": 0.02 },
            { "module": "motion", "layer": "bright" }] },
        { "name": "after", "at": "port",
          "spawn": [{ "module": "burst", "count": 2 }],
          "initialize": [{ "module": "life", "value": "one_frame" }, { "module": "delay", "frames": 1 }, { "module": "size", "value": 0.05 }],
          "update": [],
          "render": [
            { "module": "sprite_plain", "book": "crown" },
            { "module": "placement", "mode": "upright" },
            { "module": "flipbook", "play": "seed" },
            { "module": "soft_depth", "metres": 0.05 },
            { "module": "motion", "layer": "soft" }] }
      ],
      "scalability": { "cull_distance": 60, "min_pixels": 1, "importance": 0.5, "max_live": 3, "near_fade": [0.1, 0.3] },
      "budget": { "fill_megapixel_layers": 1.5, "instances": 200, "lights": 0 }
    }"#;

    /// THE EDITOR'S PREVIEW IS HELD TO THESE: every generic module, the
    /// three muzzle assets and a placed stream, run through `BurstSet` and
    /// `simulate_all` in a turned, moved player frame with a lamp, and every
    /// instance's 28 floats written to the editor
    /// (`scene_editor_web/frontend/src/lib/__effect_stack_fixtures.json`),
    /// which `effectStack.test.js` replays through its port. Writes them with
    /// `WRITE_FIXTURES=1`; otherwise fails if what is there is not what the
    /// engine makes now (re-export after any change to the stacks).
    #[test]
    fn the_editors_stack_fixtures_are_current() {
        use serde_json::json;
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../scene_editor_web/frontend/src/lib/__effect_stack_fixtures.json");
        let game = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../game/effects");
        let read = |name: &str| std::fs::read_to_string(game.join(format!("{name}.json"))).ok();
        let (Some(rifle), Some(pistol), Some(quiet)) = (read("muzzle_556_a2"), read("muzzle_9mm_bare"), read("muzzle_suppressed")) else { return };
        // The effect authored in the editor (2026-10-14), placed as a scene's object places it.
        let steam = read("steam_vent").unwrap_or_default();
        let mist = read("pond_mist").unwrap_or_default();
        let yaw = 0.4f32;
        let offset = Vec3::new(0.5, 0.0, -0.2);
        let lamp = Light {
            position: Vec3::new(-0.6, 2.2, -0.4),
            direction: Vec3::NEG_Y,
            kind: LightKind::Point,
            color: crate::renderer::Color3(255, 214, 170, 255),
            intensity: 2.5,
            range: 5.0,
            cone_angle_deg: 0.0,
            inner_cone_angle_deg: 0.0,
            mask_channel: None,
            shadow_near: None,
            source_radius: 0.03,
            in_level_bake: true,
        };
        // The sky's sun beside it: suns light only a placed effect the sun
        // reaches (`Placed::sunlit`), the pond's mist here.
        let sun = Light { kind: LightKind::Directional, direction: Vec3::new(0.3, -0.6, 0.7).normalize(), intensity: 3.0, color: crate::renderer::Color3(255, 244, 228, 255), range: 0.0, ..lamp };
        let lights = [lamp, sun];
        let ambient = |_: Vec3| Vec3::new(0.05, 0.05, 0.06);
        let v3 = |v: Vec3| json!([v.x, v.y, v.z]);
        let mut cases = Vec::new();
        // (name, assets, exposure, head (player frame), events: (frame, burst | placed), times)
        type Ev = (usize, &'static str, Vec3, Vec3, Option<(Vec3, Vec3)>, [f32; 4], u64, u64, Option<Placed>);
        let shot = |frame: usize, asset: &'static str, seed: u64, heat: f32| -> Ev {
            (frame, asset, Vec3::new(0.1, 1.45, -0.6), Vec3::new(0.2, 0.05, -1.0).normalize(), Some((Vec3::new(0.14, 1.47, -0.4), Vec3::new(1.0, 0.3, 0.2).normalize())), [heat, 0.0, 0.0, 0.0], 3, seed, None)
        };
        let specs: Vec<(&str, Vec<&str>, f32, Vec3, Vec<Ev>, Vec<f64>)> = vec![
            ("sink_one_shot", vec![SINK], 2.5, Vec3::new(0.3, 1.6, 0.4), vec![shot(0, "fixture_sink", 11, 0.6)], vec![5.0, 5.0 + 1.0 / 36.0, 5.0 + 2.0 / 36.0, 5.0 + 3.0 / 36.0, 5.3, 5.9, 6.4, 7.1]),
            (
                "sink_automatic",
                vec![SINK],
                2.5,
                Vec3::new(0.3, 1.6, 0.4),
                (0..6).map(|k| shot(k * 3, "fixture_sink", 100 + k as u64, 0.15 * k as f32)).collect(),
                (0..24).map(|k| 9.0 + k as f64 / 36.0).collect(),
            ),
            (
                "placed_stream",
                vec![VENT],
                1.2,
                Vec3::new(0.0, 1.6, 2.0),
                vec![(0, "test_vent", Vec3::new(0.2, 0.0, -0.5), Vec3::Y, None, [0.0; 4], 77, 0, Some(Placed { scale: 1.5, rate: 0.8, tint: [1.0, 0.9, 0.8], sunlit: 0.0, footprint: [0.0; 2] }))],
                vec![0.0, 0.5, 30.0, 30.0 + 1.0 / 36.0, 700.25],
            ),
            ("muzzle_556_a2", vec![], 4.0, Vec3::new(0.0, 1.6, 0.0), vec![shot(0, "muzzle_556_a2", 5, 0.3)], vec![2.0, 2.0 + 1.0 / 72.0, 2.3, 3.0]),
            ("muzzle_9mm_bare", vec![], 9.0, Vec3::new(0.0, 1.6, 0.0), vec![shot(0, "muzzle_9mm_bare", 6, 0.0)], vec![2.0, 2.0 + 1.0 / 36.0, 2.5]),
            ("muzzle_suppressed", vec![], 1.4, Vec3::new(0.0, 1.6, 0.0), vec![shot(0, "muzzle_suppressed", 7, 1.0)], vec![2.0, 2.0 + 1.0 / 36.0, 2.6, 3.4]),
            (
                "steam_vent",
                vec![],
                2.0,
                Vec3::new(0.0, 1.6, 2.5),
                vec![(0, "steam_vent", Vec3::new(0.0, 0.0, 0.0), Vec3::Y, None, [0.0; 4], 7, 0, Some(Placed::default()))],
                vec![0.0, 1.0 / 36.0, 4.0, 30.0, 30.0 + 1.0 / 36.0],
            ),
            (
                "pond_mist",
                vec![],
                1.2,
                Vec3::new(0.0, 1.6, 4.0),
                vec![(0, "pond_mist", Vec3::new(0.0, -0.3, 0.0), Vec3::Y, None, [0.0; 4], 7, 0, Some(Placed { sunlit: 0.8, footprint: [6.0, 3.0], ..Placed::default() }))],
                vec![0.0, 1.0 / 36.0, 12.0, 40.0],
            ),
        ];
        for (name, extra, exposure, head, events, times) in specs {
            let mut texts: Vec<String> = extra.iter().map(|t| t.to_string()).collect();
            texts.extend([&rifle, &pistol, &quiet, &steam, &mist].into_iter().filter(|t| t.contains(&format!("\"name\": \"{name}\""))).cloned());
            if texts.is_empty() {
                continue; // (an effect not in this game/effects)
            }
            let mut lib = EffectLibrary::new();
            for t in &texts {
                lib.add(&EffectAsset::parse(t).unwrap()).unwrap();
            }
            let at = Surroundings { head, offset, yaw_inv: Quat::from_rotation_y(-yaw), lights: &lights, reaches: &every, ambient: &ambient, exposure };
            let mut set = BurstSet::default();
            let mut frames = Vec::new();
            for (k, time) in times.iter().enumerate() {
                for e in events.iter().filter(|e| e.0 == k) {
                    let id = lib.id(e.1).unwrap();
                    match e.8 {
                        Some(p) => set.place(&lib, id, e.2, e.3, e.6, p),
                        None => set.add(&lib, EffectBurst { port: e.4, inputs: e.5, velocity: Vec3::new(0.0, 0.2, 0.4), up: Vec3::new(0.1, 1.0, 0.0).normalize(), ..EffectBurst::new(id, e.2, e.3, e.6, e.7) }),
                    }
                }
                let f = frame(&lib, &mut set, *time, &at);
                assert_eq!(set.stats.thinned, 0, "the fixtures do not reach the governor");
                frames.push(json!({
                    "time": time, "over": f.over, "screened": f.screened,
                    "instances": f.instances.iter().map(|i| bytemuck::cast::<EffectInstance, [f32; 28]>(*i).to_vec()).collect::<Vec<_>>(),
                }));
            }
            cases.push(json!({
                "name": name,
                "assets": texts,
                "surroundings": { "head": v3(head), "offset": v3(offset), "yaw": yaw, "exposure": exposure, "ambient": [0.05, 0.05, 0.06],
                    "lights": lights.iter().map(|l| json!({ "position": v3(l.position), "direction": v3(l.direction), "kind": format!("{:?}", l.kind), "color": [l.color.0, l.color.1, l.color.2, l.color.3],
                        "intensity": l.intensity, "range": l.range, "source_radius": l.source_radius })).collect::<Vec<_>>() },
                "events": events.iter().map(|e| json!({
                    "frame": e.0, "asset": e.1, "position": v3(e.2), "direction": v3(e.3), "port": e.4.map(|(p, d)| json!([v3(p), v3(d)])),
                    "inputs": e.5, "source": e.6.to_string(), "seed": e.7.to_string(), "velocity": [0.0, 0.2, 0.4], "up": v3(Vec3::new(0.1, 1.0, 0.0).normalize()),
                    "placed": e.8.map(|p| json!({ "scale": p.scale, "rate": p.rate, "tint": p.tint, "sunlit": p.sunlit, "footprint": p.footprint })),
                })).collect::<Vec<_>>(),
                "frames": frames,
            }));
        }
        let text = serde_json::to_string(&json!({
            "about": "space_soup effects::stack::tests::the_editors_stack_fixtures_are_current: WRITE_FIXTURES=1 cargo test --release --lib the_editors_stack_fixtures",
            "books": crate::renderer::effects::authoring::books_json(),
            "cases": cases,
        }))
        .unwrap()
            + "\n";
        if std::env::var("WRITE_FIXTURES").is_ok() {
            std::fs::write(&path, &text).unwrap();
            eprintln!("wrote {} ({} bytes)", path.display(), text.len());
        } else if let Ok(old) = std::fs::read_to_string(&path) {
            assert!(old == text, "the editor's fixtures are not what the engine makes now: WRITE_FIXTURES=1 cargo test --release --lib the_editors_stack_fixtures, then npx vitest run effectStack");
        }
    }

    /// A PLACED ASSET stands for as long as the level is loaded: its stream
    /// keeps about rate x life alive at any hour, it is never retired or
    /// taken over by a shot of the same asset, and its object's scale, rate
    /// and tint are its sprites' size, its density and its colour.
    #[test]
    fn a_placed_asset_streams_for_as_long_as_the_level_is_loaded() {
        let mut lib = EffectLibrary::new();
        let id = lib.add(&EffectAsset::parse(VENT).unwrap()).unwrap();
        let at = seen_from(Vec3::new(0.0, 1.6, 3.0), 1.0);
        let run = |placed: Placed, time: f64| {
            let mut set = BurstSet::default();
            set.place(&lib, id, Vec3::ZERO, Vec3::Y, 7, placed);
            // A shot of the same asset from the same source changes nothing of it.
            set.add(&lib, EffectBurst::new(id, Vec3::new(5.0, 0.0, 0.0), Vec3::Y, 7, 1));
            frame(&lib, &mut set, 0.0, &at);
            let f = frame(&lib, &mut set, time, &at);
            assert_eq!(set.live.iter().filter(|l| l.standing.is_some()).count(), 1, "still standing at {time} s");
            f.instances.into_iter().filter(|i| i.centre[0].abs() < 2.0).collect::<Vec<_>>()
        };
        for time in [3.0, 700.0, 3600.0] {
            let n = run(Placed::default(), time).len();
            assert!((10..=20).contains(&n), "{n} puffs at {time} s (rate 10 x life 1-2 s)");
        }
        let one = run(Placed::default(), 30.0);
        let big = run(Placed { scale: 2.0, ..Default::default() }, 30.0);
        assert_eq!(one.len(), big.len());
        for (a, b) in one.iter().zip(&big) {
            assert_eq!(a.centre[..3], b.centre[..3], "scale sizes sprites, it does not move them");
            assert!((b.centre[3] - 2.0 * a.centre[3]).abs() < 1e-5);
        }
        let dense = run(Placed { rate: 2.0, ..Default::default() }, 30.0);
        assert!(dense.len() as f32 > 1.6 * one.len() as f32, "{} against {}", dense.len(), one.len());
        let red = run(Placed { tint: [1.0, 0.5, 0.25], ..Default::default() }, 30.0);
        for (a, b) in one.iter().zip(&red) {
            assert_eq!(a.colour[0], b.colour[0]);
            assert!((b.colour[1] - 0.5 * a.colour[1]).abs() < 1e-6 && (b.colour[2] - 0.25 * a.colour[2]).abs() < 1e-6);
        }
    }

    /// THE GOVERNOR COVERS PLACED EMITTERS AND SPLASHES: with a row of smokes
    /// and a row of splashes over the budget, the far ones are thinned
    /// before the near, a tenth a frame; a fire's sheets are never thinned;
    /// what it sheds is counted; every particle still drawn is the same bytes
    /// it is in the whole frame; and with room again all of it comes back.
    #[test]
    fn the_governor_thins_far_placed_emitters_and_splashes_first() {
        use crate::renderer::effects::{emitter_key, governor_wants, simulate_governed, splash_key, EffectKind, Shares, Splash};
        let head = Vec3::new(0.0, 1.6, 0.0);
        let at = seen_from(head, 1.0);
        let lib = EffectLibrary::new();
        let placed = |kind: EffectKind, id: String, position: Vec3| EffectEmitter {
            id,
            kind,
            position,
            direction: Vec3::Y,
            extent: [Vec3::X, Vec3::Y, Vec3::Z],
            scale: 1.0,
            rate: 1.0,
            tint: [1.0; 3],
            ceiling: None,
            variation: Default::default(),
        };
        let mut emitters: Vec<EffectEmitter> = (0..16).map(|k| placed(EffectKind::Smoke, format!("s{k}"), Vec3::new(0.5, 0.0, -2.0 - 1.5 * k as f32))).collect();
        emitters.push(placed(EffectKind::Fire, "fire".into(), Vec3::new(-0.8, 0.0, -30.0)));
        let splashes: Vec<Splash> = (0..8)
            .map(|k| Splash { seed: 900 + k, ..Splash::new(Vec3::new(-0.5, 0.0, -3.0 - 3.0 * k as f32), 9.6, Vec3::new(0.0, -6.3, 0.0), 0.4, 30.0) })
            .collect();
        let mut set = BurstSet { budget: FrameBudget { fill: 1e9, instances: u32::MAX }, ..Default::default() };
        let step = |set: &mut BurstSet, time: f64, governed: bool| {
            let wants = governor_wants(&emitters, &|_| true, &splashes, &set.splash_measured, head, &[]);
            let drawn = set.frame_with(&lib, time, &[], head, &wants);
            let shares = Shares { emitter: &|e| set.share_of(emitter_key(e)), splash: &|s| set.share_of(splash_key(s)), sight: &[] };
            let f = simulate_governed(&emitters, &|_| true, &splashes, Some(&BurstFrame { lib: &lib, set, drawn: &drawn }), governed.then_some(&shares), time, &at);
            set.measured(&f.splash_costs);
            set.end_frame();
            f
        };
        // A budget with room for everything but half the smoke: the splashes
        // (importance 0.7) outrank the smokes (0.6), so smoke is what goes.
        for k in 0..12 {
            step(&mut set, 9.7 + k as f64 / 36.0, true);
        }
        let wants = governor_wants(&emitters, &|_| true, &splashes, &set.splash_measured, head, &[]);
        let all: f32 = wants.iter().map(|w| w.count).sum();
        let smoke: f32 = wants.iter().filter(|w| w.kind == WantKind::Emitter && w.sheddable).map(|w| w.count).sum();
        set.budget.instances = (all - 0.5 * smoke) as u32;
        let mut sizes = Vec::new();
        let mut last = EffectFrame::default();
        for k in 0..16 {
            last = step(&mut set, 10.0 + k as f64 / 36.0, true);
            sizes.push(last.instances.len());
        }
        let s = set.stats;
        assert!(s.instances_wanted > GOVERN_AIM * set.budget.instances as f32, "the test must ask for more than there is: {s:?}");
        assert!(s.instances <= set.budget.instances as f32 * 1.05, "{s:?}");
        assert!(s.emitters_thinned > 0 && s.splashes_thinned == 0, "{s:?}");
        assert!(last.bursts.emitter_shed > 0 && last.bursts.splash_shed == 0, "{:?}", last.bursts);
        // FAR FIRST.
        let share = |e: &EffectEmitter| set.share_of(emitter_key(e));
        assert_eq!(share(&emitters[0]), 1.0, "the nearest smoke is whole");
        assert!(share(&emitters[15]) < 0.5, "the farthest is thinned: {}", share(&emitters[15]));
        assert!(emitters[..16].windows(2).all(|w| share(&w[0]) >= share(&w[1])), "{:?}", emitters.iter().map(share).collect::<Vec<_>>());
        assert_eq!(share(&emitters[16]), 1.0, "a fire's sheets are never thinned");
        // Tighter still: the splashes are thinned too, the far first.
        set.budget.instances = (0.03 * all) as u32;
        for k in 16..32 {
            last = step(&mut set, 10.0 + k as f64 / 36.0, true);
        }
        let wet = |s: &Splash| set.share_of(splash_key(s));
        assert!(set.stats.splashes_thinned > 0 && last.bursts.splash_shed > 0, "{:?} {:?}", set.stats, last.bursts);
        assert!(wet(&splashes[7]) < wet(&splashes[0]), "{:?}", splashes.iter().map(wet).collect::<Vec<_>>());
        assert!(set.stats.instances <= set.budget.instances as f32 * 1.05, "{:?}", set.stats);
        // A tenth a frame.
        assert!(sizes[2..].windows(2).all(|w| w[0] as f32 - w[1] as f32 <= 0.2 * sizes[1] as f32), "{sizes:?}");
        // SPARSER, NOT SMALLER OR DIMMER.
        let whole = step(&mut set.clone(), 10.0 + 32.0 / 36.0, false);
        let thinned = step(&mut set.clone(), 10.0 + 32.0 / 36.0, true);
        assert!(whole.instances.len() > thinned.instances.len());
        let key = |i: &EffectInstance| bytemuck::bytes_of(i).to_vec();
        let all: std::collections::HashSet<Vec<u8>> = whole.instances.iter().map(key).collect();
        assert!(thinned.instances.iter().all(|i| all.contains(&key(i))), "a thinned frame's particles are the whole frame's, fewer");
        // With room again, all of it comes back.
        set.budget = FrameBudget::default();
        for k in 32..56 {
            last = step(&mut set, 10.0 + k as f64 / 36.0, true);
        }
        assert_eq!((set.stats.emitters_thinned, set.stats.splashes_thinned, last.bursts.emitter_shed, last.bursts.splash_shed), (0, 0, 0, 0));
    }

    /// THE GOVERNOR: with more bursts in view than the frame's budget, the
    /// lasting particles of the least important, then the farthest, are
    /// thinned first, a tenth a frame, sparser and not smaller; what it shed
    /// is counted; and with room again they come back.
    #[test]
    fn the_governor_thins_the_far_and_small_first_and_counts_it() {
        let Some(lib) = shipped() else { return };
        let head = Vec3::new(0.0, 1.5, 0.0);
        let at = seen_from(head, 1.0);
        // A frame with room for a third of what is asked.
        let mut set = BurstSet { budget: FrameBudget { fill: 0.035, instances: 100_000 }, ..Default::default() };
        // Sixty weapons in a row down the hall, the nearest a metre and a half off.
        let n = 60u64;
        for k in 0..n {
            let position = head + Vec3::new(0.2, -0.1, -1.5 - 0.45 * k as f32);
            set.add(&lib, EffectBurst { position, port: None, source: 10 + k, inputs: [1.0, 0.0, 0.0, 0.0], ..shot(&lib, "muzzle_556_a2", 77 + k) });
        }
        let mut last = EffectFrame::default();
        let mut sizes = Vec::new();
        for k in 0..14 {
            last = frame(&lib, &mut set, 2.0 + k as f64 / 36.0, &at);
            sizes.push(last.over);
        }
        let s = set.stats;
        assert!(s.fill_wanted > 2.0 * set.budget.fill, "the test must ask for more than there is: {}", s.fill_wanted);
        assert!(s.fill <= set.budget.fill * 1.05, "held to the budget: {} of {}", s.fill, set.budget.fill);
        assert!(s.thinned > 0 && last.bursts.shed > 0, "{s:?}, {} shed", last.bursts.shed);
        // FAR FIRST: every burst thinned is farther than every one whole.
        let far = |l: &LiveBurst| l.burst.position.distance(head);
        let nearest_thinned = set.live.iter().filter(|l| l.share < 0.5).map(far).fold(f32::INFINITY, f32::min);
        let farthest_whole = set.live.iter().filter(|l| l.share >= 1.0).map(far).fold(0.0, f32::max);
        assert!(farthest_whole > 0.0 && nearest_thinned > farthest_whole, "whole out to {farthest_whole} m, thinned from {nearest_thinned} m");
        // Thinned by a tenth a frame: no frame loses its puffs at once.
        assert!(sizes[1..].windows(2).all(|w| w[0] as f32 - w[1] as f32 <= 0.2 * sizes[1] as f32), "{sizes:?}");
        // Never drawn before, they start where the governor wants them:
        // nothing on screen went at once.
        // SPARSER, NOT SMALLER OR DIMMER: every particle still drawn is the
        // same bytes as with no governor at all.
        let mut free = BurstSet { budget: FrameBudget { fill: 1e9, instances: u32::MAX }, ..Default::default() };
        free.live = set.live.clone();
        free.live.iter_mut().for_each(|l| l.share = 1.0);
        let time = 2.0 + 13.0 / 36.0;
        let drawn = vec![true; free.live.len()];
        let whole = simulate_all(&[], &|_| true, &[], Some(&BurstFrame { lib: &lib, set: &free, drawn: &drawn }), time, &at);
        assert!(whole.over > last.over);
        let key = |i: &EffectInstance| bytemuck::bytes_of(i).to_vec();
        let all: std::collections::HashSet<Vec<u8>> = whole.instances.iter().map(key).collect();
        assert!(last.instances.iter().all(|i| all.contains(&key(i))), "a thinned frame's particles are the whole frame's, fewer");
        // With room again, they come back.
        set.budget = FrameBudget::default();
        for k in 14..30 {
            last = frame(&lib, &mut set, 2.0 + k as f64 / 36.0, &at);
        }
        assert_eq!((set.stats.thinned, last.bursts.shed), (0, 0));
    }
    /// THE STRESS SCENE, on the CPU: `units` posts 8 m apart in a square,
    /// each with four rifles firing automatic, three campfires (bed,
    /// flames, smoke, sparks), two more smokes, a box of dust and a pool
    /// struck four times a second. Run from four seconds before, so every
    /// weapon is at its cap and every pool is churned, then timed over two
    /// seconds of frames at the app's 36 a second: culling, the governor,
    /// the simulation and the draw list, as the frame runs them.
    /// `cargo test --release --lib stress_many_effects_at_once -- --ignored
    /// --nocapture`.
    #[test]
    #[ignore]
    fn stress_many_effects_at_once() {
        use crate::renderer::effects::{emitter_seen, splash_seen, strike, variants, EffectKind, Splash};
        let Some(lib) = shipped() else { return };
        let rifle = lib.id("muzzle_556_a2").unwrap();
        let fresh = |kind: EffectKind, id: String, position: Vec3| EffectEmitter {
            id,
            kind,
            position,
            direction: Vec3::Y,
            extent: [Vec3::X * 1.5, Vec3::Y * 1.25, Vec3::Z * 1.5],
            scale: 1.0,
            rate: 1.0,
            tint: [1.0; 3],
            ceiling: None,
            variation: Default::default(),
        };
        println!(
            "| scale | governor | facing | emitters seen | bursts live / seen | splashes seen | instances (worst) | over + screened | draws | est. fill wanted -> held (budget {}) | drawn fill, worst | instances wanted -> held (budget {}) | thinned: bursts / emitters / splashes | shed a frame: bursts / emitters / splashes | CPU ms a frame (median) |",
            FRAME_FILL_BUDGET, FRAME_INSTANCE_BUDGET
        );
        println!("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|");
        let mut worst_at_16 = (0.0f32, 0usize, 0.0f32);
        for (units, governed) in [(1usize, true), (4, true), (16, false), (16, true)] {
            let side = (units as f32).sqrt().round() as usize;
            let posts: Vec<Vec3> = (0..units).map(|u| Vec3::new((u % side) as f32 * 8.0, 0.0, (u / side) as f32 * 8.0)).collect();
            let mut emitters = Vec::new();
            for (u, post) in posts.iter().enumerate() {
                for f in 0..3 {
                    let at = *post + Vec3::new(f as f32 * 1.8 - 1.8, 0.0, 2.0);
                    emitters.push(EffectEmitter { scale: 1.5, ..fresh(EffectKind::Coals, format!("c{u}_{f}"), at) });
                    emitters.push(EffectEmitter { scale: 1.5, ..fresh(EffectKind::Fire, format!("f{u}_{f}"), at) });
                    emitters.push(fresh(EffectKind::Smoke, format!("s{u}_{f}"), at + Vec3::Y * 0.5));
                    emitters.push(EffectEmitter { scale: 1.5, ..fresh(EffectKind::Embers, format!("e{u}_{f}"), at) });
                }
                for k in 0..2 {
                    emitters.push(fresh(EffectKind::Smoke, format!("x{u}_{k}"), *post + Vec3::new(k as f32 * 3.0 - 1.5, 0.0, -2.5)));
                }
                emitters.push(EffectEmitter { rate: 0.6, ..fresh(EffectKind::Dust, format!("d{u}"), *post + Vec3::new(0.0, 1.25, 0.0)) });
            }
            let middle = posts.iter().copied().sum::<Vec3>() / units as f32;
            let head = Vec3::new(-5.0, 1.6, middle.z);
            for (facing, look) in [("toward", Vec3::X), ("away", Vec3::NEG_X)] {
                let vp = glam::Mat4::perspective_rh(1.75, 1.0, 0.03, 1000.0) * glam::Mat4::look_at_rh(head, head + look, Vec3::Y);
                let views = [crate::renderer::shadow::frustum_planes(vp)];
                let at = seen_from(head, 3.0);
                let mut set = BurstSet::default();
                let mut splashes: Vec<Splash> = Vec::new();
                let (mut shots, mut struck) = (0u64, 0u64);
                let (mut times, mut frames): (Vec<f64>, u32) = (Vec::new(), 0);
                let mut sums = [0.0f64; 16];
                let (mut worst_fill, mut worst_instances, mut shed) = (0.0f32, 0usize, [0.0f64; 3]);
                if !governed {
                    set.budget = FrameBudget { fill: 1e9, instances: u32::MAX };
                }
                for step in 0..(6 * 36) {
                    let time = 100.0 + step as f64 / 36.0;
                    // Each rifle's shots since the last frame, at 12.5 a second.
                    while (shots as f64) < (time - 100.0) * 12.5 {
                        for (u, post) in posts.iter().enumerate() {
                            for w in 0..4u64 {
                                let muzzle = *post + Vec3::new(w as f32 * 0.9 - 1.35, 1.4, 0.0);
                                let heat = (shots as f32 / 12.0).min(1.0);
                                set.add(&lib, EffectBurst { inputs: [heat, 0.0, 0.0, 0.0], ..EffectBurst::new(rifle, muzzle, Vec3::X, (u as u64) << 8 | w, 1000 + shots * 64 + (u as u64) * 4 + w) });
                            }
                        }
                        shots += 1;
                    }
                    while (struck as f64) < (time - 100.0) * 4.0 {
                        for (u, post) in posts.iter().enumerate() {
                            let (size, mass, speed) = [(0.015, 0.035, 4.4), (0.07, 0.9, 4.4), (0.05, 3.0, 2.2), (0.4, 30.0, 6.3)][(struck as usize + u) % 4];
                            let spot = *post + Vec3::new((struck % 5) as f32 * 0.5 - 1.0, 0.0, -5.0 + (struck % 3) as f32 * 0.4);
                            strike(&mut splashes, Splash { seed: struck * 31 + u as u64, ..Splash::new(spot, time, Vec3::new(0.0, -speed, 0.0), size, mass) }, 48);
                        }
                        struck += 1;
                    }
                    let start = std::time::Instant::now();
                    let seen: Vec<bool> = emitters.iter().map(|e| emitter_seen(e, &views, head, &[], &[])).collect();
                    let wet: Vec<Splash> = splashes.iter().copied().filter(|s| splash_seen(s, time, &views)).collect();
                    // THE WHOLE FRAME GOVERNED, as `render_frame` runs it:
                    // the placed emitters and splashes weighed with the bursts.
                    let wants = crate::renderer::effects::governor_wants(&emitters, &|i| seen[i], &wet, &set.splash_measured, head, &[]);
                    let drawn = set.frame_with(&lib, time, &views, head, &wants);
                    let shares = crate::renderer::effects::Shares {
                        emitter: &|e| set.share_of(crate::renderer::effects::emitter_key(e)),
                        splash: &|s| set.share_of(crate::renderer::effects::splash_key(s)),
                        sight: &[],
                    };
                    let f = crate::renderer::effects::simulate_governed(
                        &emitters,
                        &|i| seen[i],
                        &wet,
                        Some(&BurstFrame { lib: &lib, set: &set, drawn: &drawn }),
                        if governed { Some(&shares) } else { None },
                        time,
                        &at,
                    );
                    let (runs, stats) = variants::runs(&f, true);
                    std::hint::black_box(&runs);
                    let took = start.elapsed().as_secs_f64();
                    set.measured(&f.splash_costs);
                    set.end_frame();
                    // What the frame drew, measured from its particles.
                    let drawn_fill: f32 = f.instances.iter().map(|i| crate::renderer::effects::instance_fill(i, head)).sum();
                    if step >= 4 * 36 {
                        worst_fill = worst_fill.max(drawn_fill);
                        worst_instances = worst_instances.max(f.instances.len());
                        shed[0] += f.bursts.shed as f64;
                        shed[1] += f.bursts.emitter_shed as f64;
                        shed[2] += f.bursts.splash_shed as f64;
                    }
                    if step >= 4 * 36 {
                        times.push(took);
                        frames += 1;
                        for (k, v) in [
                            seen.iter().filter(|s| **s).count() as f64,
                            set.stats.live as f64,
                            set.stats.seen as f64,
                            wet.len() as f64,
                            f.instances.len() as f64,
                            f.over as f64,
                            f.screened as f64,
                            stats.draws as f64,
                            set.stats.fill_wanted as f64,
                            set.stats.fill as f64,
                            set.stats.instances_wanted as f64,
                            set.stats.instances as f64,
                            set.stats.thinned as f64,
                            set.stats.emitters_thinned as f64,
                            set.stats.splashes_thinned as f64,
                            drawn_fill as f64,
                        ]
                        .into_iter()
                        .enumerate()
                        {
                            sums[k] += v;
                        }
                    }
                }
                let n = frames as f64;
                times.sort_by(f64::total_cmp);
                println!(
                    "| {units}x | {} | {facing} | {:.0} of {} | {:.0} / {:.0} | {:.0} of {} | {:.0} ({}) | {:.0} + {:.0} | {:.1} | {:.2} -> {:.2} | {:.2}, {:.2} | {:.0} -> {:.0} | {:.0} / {:.0} / {:.0} | {:.0} / {:.0} / {:.0} | {:.3} |",
                    if governed { "on" } else { "off" },
                    sums[0] / n,
                    emitters.len(),
                    sums[1] / n,
                    sums[2] / n,
                    sums[3] / n,
                    splashes.len(),
                    sums[4] / n,
                    worst_instances,
                    sums[5] / n,
                    sums[6] / n,
                    sums[7] / n,
                    sums[8] / n,
                    sums[9] / n,
                    sums[15] / n,
                    worst_fill,
                    sums[10] / n,
                    sums[11] / n,
                    sums[12] / n,
                    sums[13] / n,
                    sums[14] / n,
                    shed[0] / n,
                    shed[1] / n,
                    shed[2] / n,
                    times[times.len() / 2] * 1e3
                );
                if units == 16 && governed && facing == "toward" {
                    worst_at_16 = (sums[9] as f32 / n as f32, worst_instances, sums[11] as f32 / n as f32);
                }
            }
        }
        // AT 16x, GOVERNED: the frame's estimated fill and instances are held
        // to the budget (the frame's own fixed parts -- a fire's sheets, a
        // flash -- besides, which it cannot shed).
        let (fill, _, instances) = worst_at_16;
        assert!(fill <= FRAME_FILL_BUDGET * 1.05, "16x: estimated fill {fill} over the budget {FRAME_FILL_BUDGET}");
        assert!(instances <= FRAME_INSTANCE_BUDGET as f32 * 1.05, "16x: {instances} instances over the budget {FRAME_INSTANCE_BUDGET}");
    }

}
