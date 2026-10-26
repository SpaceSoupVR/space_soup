use super::{Preset, EffectKind, COALS_GLOW, COALS_HALF, COALS_LAYER, DUST_AMBIENT_SEEN, DUST_LIGHT_SIZE, EMBER_LAYER, EMBER_POPS, EMBER_RADIANCE, FIRE_BED, FIRE_FIRST, FIRE_FRAMES, FIRE_LEVEL, FIRE_RADIANCE, MOTE_LAYER, SMOKE_FRAMES, SPLASH_DROPS, SPLASH_PETALS, SPLASH_SECONDS, DROP_PHASE_G, DROP_STREAK};
use crate::renderer::effect_asset::{
    Adapt, Budget, EffectAsset, EmitterDef, InitModule, LifeValue, LightMode, LogSquaredValue, MotionLayer, PlacementMode, Play, RangeValue, RenderModule, Scalability, SeedPolicy, SpawnModule,
    UpdateModule, Value, WaterPart, At, About, ASSET_VERSION,
};
use glam::Vec3;

#[derive(Clone, Debug, PartialEq)]
pub struct Template {
    pub name: String,
    pub(super) preset: Preset,
    pub(super) form: Form,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum Form {

    Sheets,

    Bed,

    Slots(Slots),

    WaterEntry,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Slots {

    pub density: bool,

    pub boxed: bool,

    pub air: bool,

    pub log_size: bool,
    pub ceiling: Ceiling,
    pub shade: SlotShade,

    pub pops: bool,

    pub own_light: bool,

    pub held: bool,

    pub screened: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum Ceiling {
    Open,

    Spread,

    Stop,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum SlotShade {

    Sixway { peak: f32, rise: f32, power: f32 },

    Scatter { ambient: f32 },

    Ember,
}

const BOOK_LAYERS: [(&str, u32, u32); 5] = [("smoke", 0, SMOKE_FRAMES), ("flame", FIRE_FIRST, FIRE_FRAMES), ("ember", EMBER_LAYER, 1), ("mote", MOTE_LAYER, 1), ("coals", COALS_LAYER, 1)];

pub const WATER_PARTS: [(WaterPart, u32, f32); 6] = [
    (WaterPart::Drops, SPLASH_DROPS as u32 + 3, 0.006),
    (WaterPart::Crown, SPLASH_PETALS as u32, 0.32),
    (WaterPart::Spray, 6, 0.62),
    (WaterPart::Froth, 9, 0.94),
    (WaterPart::Jet, 1, 0.17),
    (WaterPart::Mist, 8, 1.5),
];

pub const WATER_KEPT: u32 = 48;

impl EffectKind {

    pub fn template_name(self) -> &'static str {
        match self {
            Self::Fire => "fire",
            Self::Smoke => "smoke",
            Self::Embers => "embers",
            Self::Dust => "dust",
            Self::Coals => "coals",
        }
    }

    pub const ALL: [EffectKind; 5] = [Self::Fire, Self::Smoke, Self::Embers, Self::Dust, Self::Coals];
}

fn num(v: f32) -> Value {
    Value::Number(v)
}
fn range(a: f32, b: f32) -> Value {
    Value::Range(RangeValue { range: [a, b] })
}
fn life(a: f32, b: f32) -> InitModule {
    InitModule::Life { value: LifeValue::Seconds(if a == b { num(a) } else { range(a, b) }) }
}
fn placement(mode: PlacementMode, streak_seconds: f32) -> RenderModule {
    RenderModule::Placement { mode, streak_seconds }
}
fn motion(layer: MotionLayer) -> RenderModule {
    RenderModule::Motion { layer, reason: None }
}
fn emitter(name: &str, spawn: Vec<SpawnModule>, initialize: Vec<InitModule>, update: Vec<UpdateModule>, render: Vec<RenderModule>) -> EmitterDef {
    EmitterDef { name: name.into(), at: At::Origin, spawn, initialize, update, render, events: Vec::new() }
}

fn placed(name: &str, emitters: Vec<EmitterDef>, importance: f32, fill: f32, instances: u32) -> EffectAsset {
    EffectAsset {
        name: name.into(),
        version: ASSET_VERSION,
        inputs: Vec::new(),
        energy: None,
        emitters,
        scalability: Scalability { cull_distance: 120.0, min_pixels: 2.0, importance, max_live: 1, near_fade: None, reference_distance: Some(2.5) },
        budget: Budget { fill_megapixel_layers: fill, instances, lights: 0 },
        over_budget: None,
        seed_policy: SeedPolicy::PerSpawn,
    }
}

pub fn asset(kind: EffectKind) -> EffectAsset {
    use std::f32::consts::PI;
    let name = kind.template_name();
    match kind {

        EffectKind::Fire => placed(
            name,
            vec![emitter(
                "sheets",
                vec![SpawnModule::Beat { per_second: 9.0 }],
                vec![life(0.9, 1.3), InitModule::SpotsLowDiscrepancy { radius: FIRE_BED }, InitModule::Size { value: num(0.145) }, InitModule::Spin { rate: 0.0, start: 0.07 }],
                vec![
                    UpdateModule::SizeOverLife { to: Some(0.13), value: None },
                    UpdateModule::FlameSheets { radiance: FIRE_RADIANCE },
                    UpdateModule::AdaptOwnLight { radius: FIRE_BED, level: FIRE_LEVEL },
                ],
                vec![
                    RenderModule::SpriteFlame { book: "flame".into() },
                    placement(PlacementMode::Upright, 0.0),
                    RenderModule::Flipbook { play: Play::Age },
                    RenderModule::SoftDepth { metres: 0.05 },
                    motion(MotionLayer::Soft),
                ],
            )],
            1.0,
            0.4,
            16,
        ),
        EffectKind::Smoke => placed(
            name,
            vec![emitter(
                "puffs",
                vec![SpawnModule::Rate { per_second: 6.0, for_seconds: None }],
                vec![
                    life(4.0, 6.0),
                    InitModule::ShapeDisc { radius: 0.06, fills_object: false },
                    InitModule::ShapeCone { angle_deg: 10.0, about: About::Direction },
                    InitModule::Speed { value: range(0.15, 0.28) },
                    InitModule::Size { value: num(0.1) },
                    InitModule::Spin { rate: 0.2, start: PI },
                ],
                vec![
                    UpdateModule::Ballistic { accel: [0.0, 0.3, 0.0], drag: 0.6 },
                    UpdateModule::Swirl { radius: 0.14, turn: 0.7 },
                    UpdateModule::SizeOverLife { to: Some(0.55), value: None },
                    UpdateModule::CeilingSpread {},
                    UpdateModule::PuffFade { peak: 0.55, rise: 0.1, power: 1.4 },
                    UpdateModule::LightCpu { mode: LightMode::Sixway, phase_g: 0.35, albedo: [0.3; 3], ambient: 1.0 },
                ],
                vec![
                    RenderModule::SpriteSixway { book: "smoke".into() },
                    placement(PlacementMode::Facing, 0.0),
                    RenderModule::Flipbook { play: Play::Age },
                    RenderModule::SoftDepth { metres: 0.3 },
                    motion(MotionLayer::Soft),
                ],
            )],
            0.6,
            1.5,
            40,
        ),
        EffectKind::Embers => placed(
            name,
            vec![emitter(
                "sparks",
                vec![SpawnModule::Rate { per_second: 5.0, for_seconds: None }],
                vec![
                    life(1.2, 2.4),
                    InitModule::ShapeDisc { radius: 0.06, fills_object: false },
                    InitModule::ShapeCone { angle_deg: 28.0, about: About::Direction },
                    InitModule::Speed { value: range(0.9, 1.8) },
                    InitModule::Size { value: num(0.005) },
                ],
                vec![
                    UpdateModule::Ballistic { accel: [0.0, -1.1, 0.0], drag: 1.3 },
                    UpdateModule::Swirl { radius: 0.06, turn: 3.0 },
                    UpdateModule::SizeOverLife { to: Some(0.003), value: None },
                    UpdateModule::CeilingStop {},
                    UpdateModule::EmberGlow { radiance: EMBER_RADIANCE },
                    UpdateModule::AdaptOwnLight { radius: FIRE_BED, level: FIRE_LEVEL },
                    UpdateModule::Pops { per_second: EMBER_POPS },
                ],
                vec![
                    RenderModule::SpriteEmissive { book: "ember".into(), radiance: EMBER_RADIANCE, colour: [1.0, 0.45, 0.12], heat_ramp: false, adapt: Adapt::OwnLight },
                    placement(PlacementMode::Streak, 0.035),
                    RenderModule::SoftDepth { metres: 0.02 },
                    motion(MotionLayer::Bright),
                ],
            )],
            0.4,
            0.05,
            40,
        ),

        EffectKind::Coals => placed(
            name,
            vec![emitter(
                "bed",
                vec![SpawnModule::Burst { count: num(1.0), chance: 1.0 }],
                vec![life(1.0, 1.0), InitModule::ShapePoint {}, InitModule::Size { value: num(COALS_HALF) }],
                vec![UpdateModule::CoalBed { glow: COALS_GLOW }, UpdateModule::AdaptOwnLight { radius: FIRE_BED, level: FIRE_LEVEL }],
                vec![RenderModule::SpriteBed { book: "coals".into() }, placement(PlacementMode::Flat, 0.0), motion(MotionLayer::None).reasoned("a bed lies still")],
            )],
            0.8,
            0.2,
            1,
        ),

        EffectKind::Dust => placed(
            name,
            vec![emitter(
                "motes",
                vec![SpawnModule::Density { per_cubic_metre: 90.0 }],
                vec![
                    life(3.5, 6.5),
                    InitModule::ShapeBox {},
                    InitModule::ShapeCone { angle_deg: 180.0, about: About::Direction },
                    InitModule::Speed { value: range(0.0, 0.004) },
                    InitModule::Size { value: Value::LogSquared(LogSquaredValue { log_squared: [DUST_LIGHT_SIZE.0, DUST_LIGHT_SIZE.1] }) },
                ],
                vec![
                    UpdateModule::Ballistic { accel: [0.0, -0.0012, 0.0], drag: 0.4 },
                    UpdateModule::AirDrift { strength: 1.0 },
                    UpdateModule::Jitter { reach: 0.006, pace: 0.8 },
                    UpdateModule::Glint {},
                    UpdateModule::LightCpu { mode: LightMode::Scatter, phase_g: 0.7, albedo: [0.8; 3], ambient: DUST_AMBIENT_SEEN },
                ],
                vec![RenderModule::SpriteScatter { book: "mote".into() }, placement(PlacementMode::Point, 0.0), RenderModule::SoftDepth { metres: 0.02 }, motion(MotionLayer::Bright)],
            )],
            0.2,
            0.05,
            1,
        ),
    }
}

trait Reasoned {
    fn reasoned(self, why: &str) -> Self;
}
impl Reasoned for RenderModule {
    fn reasoned(self, why: &str) -> Self {
        match self {
            RenderModule::Motion { layer, .. } => RenderModule::Motion { layer, reason: Some(why.into()) },
            other => other,
        }
    }
}

pub fn water_entry_asset() -> EffectAsset {
    let part = |name: &str, part: WaterPart, book: &str, mode: PlacementMode, soft: f32, layer: MotionLayer| {
        let (_, count, half) = WATER_PARTS.iter().copied().find(|p| p.0 == part).expect("every part is listed");
        let mut update = vec![UpdateModule::WaterEntry { part }];
        let sprite = if part == WaterPart::Drops {

            update.push(UpdateModule::LightCpu { mode: LightMode::Scatter, phase_g: DROP_PHASE_G, albedo: [1.0; 3], ambient: 1.0 });
            RenderModule::SpriteScatter { book: book.into() }
        } else {
            RenderModule::SpritePlain { book: book.into() }
        };
        emitter(
            name,
            vec![SpawnModule::Burst { count: num(count as f32), chance: 1.0 }],
            vec![life(SPLASH_SECONDS, SPLASH_SECONDS), InitModule::ShapePoint {}, InitModule::Size { value: num(half) }],
            update,
            vec![sprite, placement(mode, if mode == PlacementMode::Streak { DROP_STREAK } else { 0.0 }), RenderModule::SoftDepth { metres: soft }, motion(layer)],
        )
    };
    EffectAsset {
        name: "water_entry".into(),
        version: ASSET_VERSION,
        inputs: vec!["size".into(), "mass".into(), "speed".into()],
        energy: None,
        emitters: vec![
            part("drops", WaterPart::Drops, "ember", PlacementMode::Streak, 0.01, MotionLayer::Bright),
            part("crown", WaterPart::Crown, "crown", PlacementMode::Upright, 0.03, MotionLayer::Soft),
            part("spray", WaterPart::Spray, "spray", PlacementMode::Upright, 0.05, MotionLayer::Soft),
            part("froth", WaterPart::Froth, "froth", PlacementMode::Upright, 0.05, MotionLayer::Soft),
            part("jet", WaterPart::Jet, "jet", PlacementMode::Upright, 0.03, MotionLayer::Soft),
            part("mist", WaterPart::Mist, "smoke", PlacementMode::Facing, 0.05, MotionLayer::Soft),
        ],
        scalability: Scalability { cull_distance: 250.0, min_pixels: 1.0, importance: 0.7, max_live: WATER_KEPT, near_fade: None, reference_distance: Some(2.5) },
        budget: Budget { fill_megapixel_layers: 1.5, instances: 400, lights: 0 },
        over_budget: Some(
            "what a strike throws follows what struck, and its stack is costed at the largest thing (a 120 kg body dropped in 2.5 m away) with every strike the renderer keeps alive at once; \
             strikes are culled by range and view, thin their drops with distance, and the frame's governor sheds the pieces of the far and small ones first \
             (`effects::simulate_governed`)"
                .into(),
        ),
        seed_policy: SeedPolicy::PerSpawn,
    }
}

pub fn assets() -> Vec<EffectAsset> {
    EffectKind::ALL.iter().map(|k| asset(*k)).chain([water_entry_asset()]).collect()
}

pub(super) fn template(kind: EffectKind) -> &'static Template {
    static BUILT: std::sync::OnceLock<Vec<Template>> = std::sync::OnceLock::new();
    let all = BUILT.get_or_init(|| EffectKind::ALL.iter().map(|k| Template::compile(&asset(*k)).unwrap_or_else(|e| panic!("the built-in {} template: {e}", k.template_name()))).collect());
    &all[EffectKind::ALL.iter().position(|k| *k == kind).expect("every kind is listed")]
}

impl Template {

    pub fn compile(a: &EffectAsset) -> Result<Self, String> {
        let name = a.name.clone();
        let blank = Preset {
            rate: 0.0,
            life: (1.0, 1.0),
            speed: (0.0, 0.0),
            spread_deg: 0.0,
            bed: 0.0,
            size: (0.0, 0.0),
            drag: 0.0,
            accel: Vec3::ZERO,
            swirl: (0.0, 0.0),
            spin: (0.0, 0.0),
            streak: 0.0,
            soft: 0.0,
            frames: (0, 1),
            phase_g: 0.0,
            albedo: 0.0,
        };
        let water = |e: &EmitterDef| e.update.iter().any(|m| matches!(m, UpdateModule::WaterEntry { .. }));
        if !a.emitters.is_empty() && a.emitters.iter().all(water) {
            for e in &a.emitters {
                let Some(UpdateModule::WaterEntry { part }) = e.update.iter().find(|m| matches!(m, UpdateModule::WaterEntry { .. })) else { unreachable!() };
                let most = WATER_PARTS.iter().find(|p| p.0 == *part).map_or(0, |p| p.1);
                let stated = e.spawn.iter().find_map(|m| if let SpawnModule::Burst { count, .. } = m { Some(count.bounds().1 as u32) } else { None });
                if stated != Some(most) {
                    return Err(format!("`{}`: a strike throws at most {most} of its {part:?}; the stack's burst says {stated:?}", e.name));
                }
            }
            return Ok(Self { name, preset: blank, form: Form::WaterEntry });
        }
        let [e] = a.emitters.as_slice() else {
            return Err(format!("{} emitters: a placed template is one emitter (or a strike into water, every emitter a water_entry part)", a.emitters.len()));
        };
        let pair = |v: &Value, what: &str| match v {
            Value::Number(n) => Ok((*n, *n)),
            Value::Range(RangeValue { range: [a, b] }) => Ok((*a, *b)),
            _ => Err(format!("{what}: a placed emitter's is a number or a range")),
        };
        let fixed = |what: &str, stated: f32, drawn: f32| if stated == drawn { Ok(()) } else { Err(format!("{what} {stated}: the engine draws {drawn}, and a stack may not say otherwise")) };
        let mut p = blank;
        let (mut density, mut beat, mut burst) = (false, false, false);
        for m in &e.spawn {
            match m {
                SpawnModule::Rate { per_second, for_seconds: None } => p.rate = *per_second,
                SpawnModule::Density { per_cubic_metre } => (p.rate, density) = (*per_cubic_metre, true),
                SpawnModule::Beat { per_second } => (p.rate, beat) = (*per_second, true),
                SpawnModule::Burst { count: Value::Number(n), .. } if *n == 1.0 => burst = true,
                other => return Err(format!("spawn {other:?}: a placed emitter spawns by rate, density or beat (a bed is a burst of one)")),
            }
        }
        let (mut boxed, mut spots, mut log_size) = (false, false, false);
        for m in &e.initialize {
            match m {
                InitModule::Life { value: LifeValue::Seconds(v) } => p.life = pair(v, "life")?,
                InitModule::Size { value: Value::LogSquared(LogSquaredValue { log_squared: [lo, hi] }) } => (p.size, log_size) = ((*lo, *hi), true),
                InitModule::Size { value: Value::Number(s) } => p.size = (*s, *s),
                InitModule::Speed { value } => p.speed = pair(value, "speed")?,
                InitModule::Spin { rate, start } => p.spin = (*rate, *start),
                InitModule::ShapeDisc { radius, .. } => p.bed = *radius,
                InitModule::SpotsLowDiscrepancy { radius } => (p.bed, spots) = (*radius, true),
                InitModule::ShapeBox {} => boxed = true,
                InitModule::ShapePoint {} => {}
                InitModule::ShapeCone { angle_deg, about: About::Direction } => p.spread_deg = *angle_deg,
                other => return Err(format!("initialize {other:?}: not one a placed emitter's loop runs")),
            }
        }
        let (mut air, mut jitter, mut swirl) = (false, false, false);
        let mut ceiling = Ceiling::Open;
        let mut fade = None;
        let mut light = None;
        let (mut held, mut own_light, mut ember, mut sheets, mut bed, mut pops) = (false, false, false, false, false, false);
        for m in &e.update {
            match m {
                UpdateModule::Ballistic { accel, drag } => (p.accel, p.drag) = (Vec3::from(*accel), *drag),
                UpdateModule::Swirl { radius, turn } => (p.swirl, swirl) = ((*radius, *turn), true),
                UpdateModule::AirDrift { strength } => {
                    fixed("air_drift strength", *strength, 1.0)?;
                    air = true;
                }
                UpdateModule::Jitter { reach, pace } => (p.swirl, jitter) = ((*reach, *pace), true),
                UpdateModule::CeilingSpread {} => ceiling = Ceiling::Spread,
                UpdateModule::CeilingStop {} => ceiling = Ceiling::Stop,
                UpdateModule::SizeOverLife { to: Some(to), value: None } => p.size.1 = *to,
                UpdateModule::PuffFade { peak, rise, power } => fade = Some((*peak, *rise, *power)),
                UpdateModule::Glint {} => held = true,
                UpdateModule::LightCpu { mode, phase_g, albedo, ambient } => {
                    if albedo[1] != albedo[0] || albedo[2] != albedo[0] {
                        return Err("light_cpu: a placed emitter's albedo is one grey (its tint is the scene's)".into());
                    }
                    (p.phase_g, p.albedo, light) = (*phase_g, albedo[0], Some((*mode, *ambient)));
                }
                UpdateModule::AdaptOwnLight { radius, level } => {
                    fixed("adapt_own_light radius", *radius, FIRE_BED)?;
                    fixed("adapt_own_light level", *level, FIRE_LEVEL)?;
                    own_light = true;
                }
                UpdateModule::EmberGlow { radiance } => {
                    fixed("ember_glow radiance", *radiance, EMBER_RADIANCE)?;
                    ember = true;
                }
                UpdateModule::FlameSheets { radiance } => {
                    fixed("flame_sheets radiance", *radiance, FIRE_RADIANCE)?;
                    sheets = true;
                }
                UpdateModule::CoalBed { glow } => {
                    fixed("coal_bed glow", *glow, COALS_GLOW)?;
                    bed = true;
                }
                UpdateModule::Pops { per_second } => {
                    fixed("pops per_second", *per_second, EMBER_POPS)?;
                    pops = true;
                }
                other => return Err(format!("update {other:?}: not one a placed emitter's loop runs")),
            }
        }
        if air != jitter || (air && swirl) {
            return Err("carried by the room's air (air_drift with jitter) or by a swirl, one of them".into());
        }
        #[derive(PartialEq)]
        enum Sprite {
            Sixway,
            Scatter,
            Emissive,
            Flame,
            Bed,
        }
        let mut sprite = None;
        let mut point = false;
        for m in &e.render {
            let book = match m {
                RenderModule::SpriteSixway { book } => Some((Sprite::Sixway, book)),
                RenderModule::SpriteScatter { book } => Some((Sprite::Scatter, book)),
                RenderModule::SpriteEmissive { book, radiance, colour, heat_ramp: false, adapt: Adapt::OwnLight } => {
                    fixed("sprite_emissive radiance", *radiance, EMBER_RADIANCE)?;
                    if *colour != [1.0, 0.45, 0.12] {
                        return Err("sprite_emissive colour: a spark's is the engine's [1, 0.45, 0.12] (its tint is the scene's)".into());
                    }
                    Some((Sprite::Emissive, book))
                }
                RenderModule::SpriteFlame { book } => Some((Sprite::Flame, book)),
                RenderModule::SpriteBed { book } => Some((Sprite::Bed, book)),
                RenderModule::Placement { mode, streak_seconds } => {
                    p.streak = if *mode == PlacementMode::Streak { *streak_seconds } else { 0.0 };
                    point = *mode == PlacementMode::Point;
                    None
                }
                RenderModule::SoftDepth { metres } => {
                    p.soft = *metres;
                    None
                }
                RenderModule::Flipbook { play: Play::Age } | RenderModule::Motion { .. } => None,
                other => return Err(format!("render {other:?}: not one a placed emitter's loop draws")),
            };
            if let Some((kind, book)) = book {
                let Some(layers) = BOOK_LAYERS.iter().find(|b| b.0 == book.as_str()) else {
                    return Err(format!("the book `{book}` is not one a placed emitter draws ({})", BOOK_LAYERS.map(|b| b.0).join(", ")));
                };
                p.frames = (layers.1, layers.2);
                sprite = Some(kind);
            }
        }
        let form = match sprite {
            Some(Sprite::Flame) if sheets && beat && spots => Form::Sheets,
            Some(Sprite::Bed) if bed && burst => Form::Bed,
            Some(kind) if !sheets && !bed && !burst && !beat && !spots => {
                let shade = match (kind, light, fade) {
                    (Sprite::Sixway, Some((LightMode::Sixway, _)), Some((peak, rise, power))) => SlotShade::Sixway { peak, rise, power },
                    (Sprite::Scatter, Some((LightMode::Scatter, ambient)), None) if point => SlotShade::Scatter { ambient },
                    (Sprite::Emissive, None, None) if ember => SlotShade::Ember,
                    _ => return Err("a placed emitter is lit one of three ways: sprite_sixway with light_cpu (sixway) and puff_fade; sprite_scatter (a point) with light_cpu (scatter); sprite_emissive with ember_glow".into()),
                };
                if density != boxed || (held && !boxed) || (pops && !ember) {
                    return Err("density fills a shape_box; glint is a boxed mote's; pops are an ember_glow's".into());
                }
                Form::Slots(Slots { density, boxed, air, log_size, ceiling, shade, pops, own_light, held, screened: !matches!(shade, SlotShade::Sixway { .. }) })
            }
            _ => return Err("flame_sheets goes with beat, spots_low_discrepancy and sprite_flame; coal_bed with a burst of one and sprite_bed; everything else with rate or density".into()),
        };
        Ok(Self { name, preset: p, form })
    }
}

#[cfg(test)]
mod tests {
    use super::super::*;
    use super::*;
    use crate::renderer::effect_asset::Report;
    use glam::Quat;

    fn legacy_preset(kind: EffectKind) -> Preset {
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

    fn legacy(emitters: &[EffectEmitter], time: f64, at: &Surroundings) -> EffectFrame {
        let seen = |_: usize| true;
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
            let p = legacy_preset(e.kind);
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
            // (The drift's clocks once a frame, as `simulate_governed` reads
            // them since 2026-10-22.)
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
        over.sort_by(|a, b| b.0.total_cmp(&a.0));
        let mut frame = EffectFrame { instances: Vec::new(), over: over.len() as u32, screened: screened.len() as u32, screened_heat: 0, bursts: Default::default(), splash_costs: Vec::new() };
        frame.instances.extend(over.into_iter().map(|(_, i)| i));
        frame.instances.extend(screened);
        frame
    }

    fn every(_: &EffectEmitter, _: usize) -> bool {
        true
    }
    fn dim(p: Vec3) -> Vec3 {
        Vec3::new(0.1, 0.12, 0.09) * (1.0 + 0.1 * p.y)
    }
    fn placed_at(kind: EffectKind, id: &str, position: Vec3) -> EffectEmitter {
        EffectEmitter {
            id: id.into(),
            kind,
            position,
            direction: Vec3::Y,
            extent: [Vec3::X * 1.5, Vec3::Y, Vec3::Z * 1.2],
            scale: 1.0,
            rate: 1.0,
            tint: [1.0; 3],
            ceiling: None,
            variation: Variation::default(),
        }
    }
    fn lamp(position: Vec3) -> Light {
        Light {
            position,
            direction: Vec3::NEG_Y,
            kind: LightKind::Point,
            color: crate::renderer::Color3(255, 230, 180, 255),
            intensity: 2.0,
            range: 9.0,
            cone_angle_deg: 0.0,
            inner_cone_angle_deg: 0.0,
            mask_channel: None,
            shadow_near: None,
            source_radius: 0.0,
            in_level_bake: true,
        }
    }

    #[test]
    fn every_template_checks_and_compiles_to_the_preset_it_was() {
        let books = books();
        for (name, first, frames) in BOOK_LAYERS {
            let b = books.iter().find(|b| b.name == name).expect("a template's book exists");
            assert_eq!((b.first_layer, b.frames), (first, frames), "{name}");
        }
        for kind in EffectKind::ALL {
            let a = asset(kind);
            let report: Report = a.check(books).unwrap_or_else(|e| panic!("{}: {}", a.name, e.iter().map(std::string::ToString::to_string).collect::<Vec<_>>().join("; ")));
            println!(
                "{:<8} {:>3} instances, fill {:.3} megapixel-layers at {} m (budget {} / {})",
                a.name, report.instances_steady, report.fill_steady, report.reference_distance, a.budget.instances, a.budget.fill_megapixel_layers
            );

            assert_eq!(Template::compile(&a).unwrap().preset, legacy_preset(kind), "{}", a.name);
            let again = EffectAsset::parse(&a.to_json()).unwrap();
            assert_eq!(again, a, "{}", a.name);
            assert_eq!(Template::compile(&again).unwrap(), *template(kind));
            assert_eq!(matches!(template(kind).form, Form::Slots(s) if s.screened), kind.screened());
        }
        let water = water_entry_asset();
        let report = water.check(books).unwrap_or_else(|e| panic!("water_entry: {}", e.iter().map(std::string::ToString::to_string).collect::<Vec<_>>().join("; ")));
        println!("water_entry {} instances at most, fill {:.3} megapixel-layers at {} m at most", report.instances_per_burst, report.fill_per_burst, report.reference_distance);
        let (why, over) = report.over_budget.expect("a strike's stack is over a burst's budget, and says why");
        println!("  over budget ({} lines): {why}", over.len());
        assert_eq!(Template::compile(&water).unwrap().form, Form::WaterEntry);
        assert_eq!(assets().len(), 6);
    }

    #[test]
    fn the_stacks_make_todays_particles_bit_for_bit() {
        let varied = Variation { intensity: 1.7, size_variation: 2.1, life_variation: 0.4, speed_variation: 2.5, temperature_variation: 0.6, flicker: 2.0, flicker_rate: 1.8, turbulence: 2.6, seed: 77 };
        let mut scenes: Vec<Vec<EffectEmitter>> = Vec::new();
        for kind in EffectKind::ALL {
            let e = placed_at(kind, "a", Vec3::new(0.3, 0.0, -1.0));
            scenes.push(vec![e.clone()]);
            scenes.push(vec![EffectEmitter { ceiling: Some(1.1), scale: 1.6, rate: 0.7, tint: [0.9, 0.8, 1.1], variation: varied, direction: Vec3::new(0.2, 1.0, 0.1), ..e.clone() }]);
            scenes.push(vec![EffectEmitter { rate: 3.0, scale: 0.5, id: "another".into(), ..e }]);
        }

        let at_fire = Vec3::new(1.0, 0.0, -2.0);
        scenes.push(
            [EffectKind::Coals, EffectKind::Fire, EffectKind::Smoke, EffectKind::Embers, EffectKind::Dust]
                .iter()
                .enumerate()
                .map(|(i, k)| EffectEmitter { scale: 1.5, ceiling: (i % 2 == 0).then_some(2.4), ..placed_at(*k, &format!("camp{i}"), at_fire + Vec3::Y * if *k == EffectKind::Smoke { 0.5 } else { 0.0 }) })
                .collect(),
        );
        let lights = [lamp(Vec3::new(0.5, 2.0, -1.5)), fire_light(&placed_at(EffectKind::Fire, "camp1", at_fire), 3.0, Vec3::ZERO, Quat::IDENTITY).expect("a fire lights")];
        let views = [
            (Vec3::new(0.0, 1.6, 1.0), Vec3::ZERO, Quat::IDENTITY, 1.0, &lights[..0]),
            (Vec3::new(0.4, 1.5, 0.6), Vec3::new(2.0, 0.0, -3.0), Quat::from_rotation_y(0.7), 3.2, &lights[..]),
        ];
        let (mut compared, mut bytes) = (0usize, 0usize);
        for scene in &scenes {
            for (head, offset, yaw_inv, exposure, lights) in views {
                let at = Surroundings { head, offset, yaw_inv, lights, reaches: &every, ambient: &dim, exposure, room: None };
                for time in [0.0, 0.37, 5.21, 20.0, 1234.567, 86_400.25] {
                    let (was, is) = (legacy(scene, time, &at), simulate(scene, time, &at));
                    assert_eq!((was.over, was.screened), (is.over, is.screened), "{:?} at {time}", scene[0].kind);
                    let (a, b) = (bytemuck::cast_slice::<_, u8>(&was.instances), bytemuck::cast_slice::<_, u8>(&is.instances));
                    assert!(a == b, "{:?} at {time}: the stack's particles differ from the loop's that was", scene[0].kind);
                    compared += is.instances.len();
                    bytes += b.len();
                }
            }
        }
        println!("{} frames, {compared} instances, {bytes} bytes the same", scenes.len() * views.len() * 6);
        assert!(compared > 5000, "the comparison drew {compared} particles");
    }

    #[test]
    fn a_stacks_numbers_are_the_loops_and_what_it_cannot_run_is_refused() {
        let mut thick = asset(EffectKind::Smoke);
        for m in &mut thick.emitters[0].update {
            if let UpdateModule::PuffFade { peak, .. } = m {
                *peak = 0.8;
            }
            if let UpdateModule::Ballistic { drag, .. } = m {
                *drag = 0.9;
            }
        }
        let t = Template::compile(&thick).unwrap();
        assert_eq!(t.preset.drag, 0.9);
        assert!(matches!(t.form, Form::Slots(Slots { shade: SlotShade::Sixway { peak, .. }, .. }) if peak == 0.8));

        let mut hot = asset(EffectKind::Fire);
        hot.emitters[0].update[1] = UpdateModule::FlameSheets { radiance: 90.0 };
        assert!(Template::compile(&hot).unwrap_err().contains("flame_sheets radiance"));

        let mut wrong = asset(EffectKind::Smoke);
        wrong.emitters[0].update.retain(|m| !matches!(m, UpdateModule::PuffFade { .. }));
        assert!(Template::compile(&wrong).is_err());

        let mut wet = water_entry_asset();
        wet.emitters[0].spawn = vec![SpawnModule::Burst { count: Value::Number(500.0), chance: 1.0 }];
        assert!(Template::compile(&wet).unwrap_err().contains("at most"));
    }

    fn part_of(i: &EffectInstance, screened: bool) -> WaterPart {
        let layer = i.params[1] as u32;
        if screened {
            WaterPart::Drops
        } else if layer >= FROTH_LAYER {
            WaterPart::Froth
        } else if layer >= SPRAY_LAYER {
            WaterPart::Spray
        } else if layer >= JET_LAYER {
            WaterPart::Jet
        } else if layer >= CROWN_LAYER {
            WaterPart::Crown
        } else {
            WaterPart::Mist
        }
    }

    #[test]
    fn a_strike_into_water_stays_inside_its_stacks_counts() {
        let at = Surroundings { head: Vec3::new(0.0, 1.6, 2.5), offset: Vec3::ZERO, yaw_inv: Quat::IDENTITY, lights: &[], reaches: &every, ambient: &dim, exposure: 1.0, room: None };
        let mut most = [(0u32, 0.0f32); 6];
        let (mut most_instances, mut most_fill, mut strikes) = (0usize, 0.0f32, 0);
        let shares: Vec<f32> = books().iter().map(|b| b.cutout_share).collect();
        let share_of = |part: WaterPart| {
            let name = match part {
                WaterPart::Drops => "ember",
                WaterPart::Crown => "crown",
                WaterPart::Spray => "spray",
                WaterPart::Froth => "froth",
                WaterPart::Jet => "jet",
                WaterPart::Mist => "smoke",
            };
            shares[books().iter().position(|b| b.name == name).unwrap()]
        };
        for (size, mass) in [(0.003, 0.0005), (0.0045, 0.008), (0.015, 0.035), (0.05, 3.0), (0.07, 0.9), (0.15, 8.0), (0.4, 30.0), (0.6, 120.0)] {
            for velocity in [Vec3::new(0.0, -1.0, 0.0), Vec3::new(0.0, -4.4, 0.0), Vec3::new(-2.3, -1.1, 0.0), Vec3::new(0.0, -9.0, 0.0), Vec3::new(247.0, -247.0, 0.0), Vec3::new(0.0, -800.0, 0.0)] {

                if mass > 1.0 && velocity.length() > 12.0 {
                    continue;
                }
                for (calm, held, depth) in [(1.0, 0.0, 30.0), (0.3, 0.0, 2.0), (1.0, 1.0, 0.2)] {
                    let s = Splash { calm, held, depth, seed: 3 + strikes, ..Splash::new(Vec3::ZERO, 0.0, velocity, size, mass) };
                    strikes += 1;
                    for step in 0..40 {
                        let f = simulate_with(&[], &[s], 0.01 + f64::from(step) * f64::from(SPLASH_SECONDS) / 40.0, &at);
                        let mut counts = [0u32; 6];
                        let mut fill = 0.0;
                        for (k, i) in f.instances.iter().enumerate() {
                            let part = part_of(i, k >= f.over as usize);
                            let slot = WATER_PARTS.iter().position(|p| p.0 == part).unwrap();
                            counts[slot] += 1;
                            most[slot].1 = most[slot].1.max(i.centre[3]);

                            let away = (Vec3::new(i.centre[0], i.centre[1], i.centre[2]) - at.head).length().max(0.35);
                            let across = (2.0 * i.centre[3] / away * crate::renderer::effect_asset::REFERENCE_FOCAL_PIXELS).max(1.5);
                            let tall = if part == WaterPart::Drops { 1.0 } else { i.params[2].abs().max(1.0) };
                            fill += across * across * tall * share_of(part) * 2.0 / 1e6;
                        }
                        for (slot, n) in counts.iter().enumerate() {
                            most[slot].0 = most[slot].0.max(*n);
                        }
                        most_instances = most_instances.max(f.instances.len());
                        most_fill = most_fill.max(fill);
                    }
                }
            }
        }
        println!("{strikes} strikes: at most {most_instances} instances, fill {most_fill:.3} megapixel-layers from 2.5 m back");
        for ((part, count, half), (seen, seen_half)) in WATER_PARTS.iter().zip(most) {
            println!("  {part:?}: at most {seen} of {count}, half size up to {seen_half:.4} m (the stack's {half})");
        }
        for ((part, count, half), (seen, seen_half)) in WATER_PARTS.iter().zip(most) {
            assert!(seen <= *count, "{part:?}: {seen} drawn, the stack says at most {count}");
            assert!(seen_half <= *half, "{part:?}: {seen_half} m, the stack says at most {half}");
        }
        let report = water_entry_asset().check(books()).expect("its stack checks, over its budget with its reason");
        assert!(most_instances as u32 <= report.instances_per_burst);
        assert!(most_fill <= report.fill_per_burst, "{most_fill} drawn, the stack's cost says at most {}", report.fill_per_burst);
    }

    #[test]
    #[ignore]
    fn export_the_templates() {
        let out = std::path::PathBuf::from(std::env::var("OUT").unwrap_or_else(|_| "/tmp".into()));
        std::fs::create_dir_all(&out).unwrap();
        for a in assets() {
            let path = out.join(format!("{}.json", a.name));
            std::fs::write(&path, a.to_json() + "\n").unwrap();
            eprintln!("wrote {}", path.display());
        }
    }

}
