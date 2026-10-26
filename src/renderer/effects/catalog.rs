//! THE MODULES AN EFFECT IS BUILT FROM, as an author sees them: by stage, with
//! physical names and units, defaults the format accepts, and what each costs
//! and scales with. The set is the format's (`effect_asset`: `SpawnModule`,
//! `InitModule`, `UpdateModule`, `RenderModule`, `EventModule`); a module is offered
//! only in its own stage, and the validator (`authoring::check_json`) is the
//! judge of the rest.
//!
//! An author never sees slots, periods, uniform indices or instance fields.
//! The required modules -- a spawn, a life, a sprite, a placement and a
//! motion -- are in every new emitter and can be changed but not removed:
//! every effect writes `SpaceWarp` motion vectors, and `none` takes a reason.
//!
//! Editors work on the asset as JSON (`serde_json::Value`), so a field the
//! format adds later survives an edit it does not understand.

use serde_json::{json, Value};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Stage {
    Spawn,
    Initialize,
    Update,
    Render,
    Events,
}

impl Stage {
    pub const ALL: [Stage; 5] = [Stage::Spawn, Stage::Initialize, Stage::Update, Stage::Render, Stage::Events];

    /// The emitter field the stage's modules are listed in.
    pub fn key(self) -> &'static str {
        match self {
            Stage::Spawn => "spawn",
            Stage::Initialize => "initialize",
            Stage::Update => "update",
            Stage::Render => "render",
            Stage::Events => "events",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Stage::Spawn => "Spawn \u{2014} when and how many",
            Stage::Initialize => "Initialize \u{2014} each particle at birth",
            Stage::Update => "Update \u{2014} over its life",
            Stage::Render => "Render \u{2014} how it is drawn",
            Stage::Events => "Events \u{2014} what it does to the world",
        }
    }

    pub fn of_key(key: &str) -> Option<Stage> {
        Stage::ALL.into_iter().find(|s| s.key() == key)
    }
}

/// How a parameter is edited.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Kind {
    Number,
    Int,
    Bool,
    Enum(&'static [&'static str]),
    /// A number, or how one is drawn: range, uniform, lognormal, squared,
    /// `log_squared`, or a curve over its life or its seed.
    Value,
    /// `one_frame`, or a [`Kind::Value`] in seconds.
    Life,
    /// Keys `[x, y]` over an input, 0-1.
    Curve,
    /// Stops `[t, r, g, b, a]` over its life.
    Gradient,
    Vec3,
    /// Linear rgb.
    Colour,
    /// A sprite book, of these channels when given.
    Book(Option<&'static str>),
    /// One of the asset's inputs.
    Input,
    Text,
}

/// Which budget a parameter moves in step with, so its slider can end where
/// the budget does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Couples {
    Instances,
    Fill,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Param {
    pub name: &'static str,
    pub kind: Kind,
    pub unit: &'static str,
    pub min: f32,
    pub max: f32,
    pub step: f32,
    /// JSON text of what a new module writes.
    pub default: &'static str,
    /// Left out of a new module; absent means the format's default.
    pub optional: bool,
    /// One of the module's `one_of` parameters is written, not both.
    pub one_of: bool,
    pub couples: Option<Couples>,
}

const fn p(name: &'static str, kind: Kind, default: &'static str) -> Param {
    Param { name, kind, unit: "", min: 0.0, max: 1.0, step: 0.01, default, optional: false, one_of: false, couples: None }
}

impl Param {
    const fn unit(mut self, unit: &'static str) -> Self {
        self.unit = unit;
        self
    }
    const fn range(mut self, min: f32, max: f32, step: f32) -> Self {
        self.min = min;
        self.max = max;
        self.step = step;
        self
    }
    const fn optional(mut self) -> Self {
        self.optional = true;
        self
    }
    const fn one_of(mut self) -> Self {
        self.one_of = true;
        self
    }
    const fn couples(mut self, c: Couples) -> Self {
        self.couples = Some(c);
        self
    }

    pub fn default_value(&self) -> Value {
        serde_json::from_str(self.default).expect("a catalogue default is JSON")
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Module {
    /// Unique in the catalogue: `by_input_update` is `by_input` in update.
    pub key: &'static str,
    pub stage: Stage,
    /// The `module` the asset writes.
    pub module: &'static str,
    pub label: &'static str,
    pub text: &'static str,
    pub cost: &'static str,
    pub scales: &'static str,
    pub required: bool,
    /// Spawns particles; an emitter keeps at least one.
    pub source: bool,
    /// Draws them; an emitter keeps at least one.
    pub sprite: bool,
    /// The built-in templates' fused modules: an asset using one runs only as
    /// a scene's kind, so they are not offered.
    pub template: bool,
    pub params: &'static [Param],
}

const fn m(key: &'static str, stage: Stage, label: &'static str, params: &'static [Param]) -> Module {
    Module { key, stage, module: key, label, text: "", cost: "CPU", scales: "-", required: false, source: false, sprite: false, template: false, params }
}

impl Module {
    const fn text(mut self, text: &'static str) -> Self {
        self.text = text;
        self
    }
    const fn cost(mut self, cost: &'static str, scales: &'static str) -> Self {
        self.cost = cost;
        self.scales = scales;
        self
    }
    const fn writes(mut self, module: &'static str) -> Self {
        self.module = module;
        self
    }
    const fn required(mut self) -> Self {
        self.required = true;
        self
    }
    const fn source(mut self) -> Self {
        self.source = true;
        self
    }
    const fn sprite(mut self) -> Self {
        self.sprite = true;
        self
    }
    const fn template(mut self) -> Self {
        self.template = true;
        self.cost = "";
        self
    }
}

use Kind::*;
use Stage::*;

pub const MODULES: &[Module] = &[
    // SPAWN
    m("rate", Spawn, "Rate", &[
        p("per_second", Number, "6").unit("/s").range(0.1, 200.0, 0.1).couples(Couples::Instances),
        p("for_seconds", Number, "1").unit("s").range(0.05, 60.0, 0.05).optional(),
    ])
    .source()
    .text("Particles a second. In a burst's effect, for so many seconds after the event (a smoking barrel); placed, for as long as the level is loaded.")
    .cost("CPU", "instances = rate \u{d7} longest life"),
    m("burst", Spawn, "Burst", &[
        p("count", Value, "4").range(0.0, 200.0, 1.0).couples(Couples::Instances),
        p("chance", Number, "1").range(0.0, 1.0, 0.01).optional(),
    ])
    .source()
    .text("So many at the event, each burst with this chance of any at all.")
    .cost("CPU", "instances = count (\u{d7} max_live for what outlives the event)"),
    m("by_input", Spawn, "By input", &[
        p("input", Input, "\"\""),
        p("target", Enum(&["count", "rate", "chance"]), "\"rate\""),
        p("curve", Curve, "[[0, 0.5], [1, 1]]"),
    ])
    .text("One of the effect's inputs (heat, energy\u{2026}) scales the count, the rate or the chance, by a curve of the input.")
    .cost("CPU", "the curve's highest point multiplies instances"),
    m("density", Spawn, "Density (template)", &[p("per_cubic_metre", Number, "90").unit("/m\u{b3}").range(1.0, 500.0, 1.0)]).template(),
    m("beat", Spawn, "Beat (template)", &[p("per_second", Number, "9").unit("/s").range(0.1, 30.0, 0.1)]).template(),
    // INITIALIZE
    m("life", Initialize, "Life", &[p("value", Life, "{\"range\": [1, 2]}").unit("s").range(0.01, 30.0, 0.05).couples(Couples::Instances)])
        .required()
        .text("How long each particle lasts: seconds (a number or a range), or the one app frame of its event (a muzzle flash).")
        .cost("CPU", "instances of a rate = rate \u{d7} the longest life"),
    m("size", Initialize, "Size", &[p("value", Value, "0.05").unit("m").range(0.001, 2.0, 0.005).couples(Couples::Fill)])
        .text("The sprite's half size at birth.")
        .cost("fill", "fill \u{d7} size\u{b2}"),
    m("speed", Initialize, "Speed", &[p("value", Value, "{\"range\": [0.3, 0.6]}").unit("m/s").range(0.0, 60.0, 0.05)])
        .text("Launch speed along its direction.")
        .cost("CPU", "how far it travels (its cull bound)"),
    m("spin", Initialize, "Spin", &[
        p("rate", Number, "0.3").unit("rad/s").range(0.0, 10.0, 0.05),
        p("start", Number, "3.14").unit("rad").range(0.0, 3.15, 0.05),
    ])
    .text("Turning: radians a second either way, and how far it may start turned."),
    m("shape_point", Initialize, "Shape: point", &[]).text("Born at the emitter."),
    m("shape_disc", Initialize, "Shape: disc", &[
        p("radius", Number, "0.05").unit("m").range(0.0, 20.0, 0.01),
        p("fills_object", Bool, "false").optional(),
    ])
    .text("Born on a disc across the emitter's direction. Fills object: placed in a level, the disc spans its object's box instead (a pond's mist sized by the pond), as many more particles as the box is wider.")
    .cost("CPU", "fills object: particles \u{d7} the box's area \u{f7} the disc's"),
    m("shape_cone", Initialize, "Direction: cone", &[
        p("angle_deg", Number, "15").unit("\u{b0}").range(0.0, 180.0, 1.0),
        p("about", Enum(&["direction", "up"]), "\"direction\"").optional(),
    ])
    .text("Launched within this angle of the emitter's direction, or of the world's up (what rises whichever way the emitter points)."),
    m("shape_lobes", Initialize, "Shape: muzzle lobes", &[
        p("lobes", Int, "5").range(0.0, 12.0, 1.0),
        p("open_deg", Number, "40").unit("\u{b0}").range(0.0, 120.0, 1.0),
        p("rake_deg", Number, "0").unit("\u{b0}").range(-45.0, 45.0, 1.0).optional(),
        p("skip_down", Bool, "false").optional(),
        p("side_pairs", Int, "0").range(0.0, 4.0, 1.0).optional(),
        p("length", Value, "{\"lognormal\": [0.08, 0.45]}").unit("m").range(0.001, 1.0, 0.005),
    ])
    .text("A muzzle device's ports: jets round the bore, opened from it; none downward for a closed bottom; side pairs for a brake. Each particle is one lobe; past the last, the core."),
    m("delay", Initialize, "Delay", &[p("frames", Int, "1").range(0.0, 8.0, 1.0)])
        .text("Its particles appear this many app frames after the event (an afterglow)."),
    m("shape_box", Initialize, "Shape: box (template)", &[]).template(),
    m("spots_low_discrepancy", Initialize, "Spots (template)", &[p("radius", Number, "0.15").unit("m").range(0.0, 5.0, 0.01)]).template(),
    // UPDATE
    m("ballistic", Update, "Ballistic", &[
        p("accel", Vec3, "[0, 0.3, 0]").unit("m/s\u{b2}").range(-20.0, 20.0, 0.05),
        p("drag", Number, "1").unit("1/s").range(0.0, 30.0, 0.1),
    ])
    .text("Constant acceleration (buoyancy up, gravity down) and linear drag, exact at any age.")
    .cost("CPU", "how far it travels"),
    m("swirl", Update, "Swirl", &[
        p("radius", Number, "0.1").unit("m").range(0.0, 2.0, 0.01),
        p("turn", Number, "0.8").unit("rad/s").range(-10.0, 10.0, 0.05),
    ])
    .text("Carried round a circle growing to this radius over its life."),
    m("air_drift", Update, "Air drift", &[p("strength", Number, "1").unit("\u{d7}").range(0.0, 4.0, 0.05).optional()])
        .text("Carried by the room's shared air: neighbours (and the dust, and a weapon's smoke) move together."),
    m("jitter", Update, "Jitter", &[
        p("reach", Number, "0.02").unit("m").range(0.0, 0.5, 0.002),
        p("pace", Number, "1").unit("/s").range(0.0, 10.0, 0.05),
    ])
    .text("A wander of its own that never retraces itself."),
    m("ceiling_spread", Update, "Ceiling spread", &[])
        .text("Rising stops under the ceiling found over a placed emitter, and spreads (bursts find no ceiling).")
        .cost("CPU, one ray at load", "-"),
    m("ceiling_stop", Update, "Ceiling stop", &[]).text("Stops against that ceiling.").cost("CPU, one ray at load", "-"),
    m("size_over_life", Update, "Size over life", &[
        p("to", Number, "0.3").unit("m").range(0.001, 5.0, 0.01).optional().one_of().couples(Couples::Fill),
        p("value", Value, "{\"curve\": [[0, 1], [1, 3]]}").unit("\u{d7}").range(0.0, 10.0, 0.05).optional().one_of(),
    ])
    .text("Grows (eased) to `to` metres, or is multiplied by a value (a curve over its life).")
    .cost("fill", "fill \u{d7} mean size\u{b2}"),
    m("opacity_over_life", Update, "Opacity over life", &[p("value", Value, "{\"curve\": [[0, 0], [0.15, 0.3], [1, 0]]}").range(0.0, 1.0, 0.01)])
        .text("Opacity (drawn over) or strength (screened), 0\u{2013}1, usually a curve over its life."),
    m("puff_fade", Update, "Puff fade", &[
        p("peak", Number, "0.3").range(0.01, 1.0, 0.01),
        p("rise", Number, "0.15").unit(" of life").range(0.01, 0.99, 0.01),
        p("power", Number, "1.5").range(0.1, 5.0, 0.05),
    ])
    .text("A puff's opacity in closed form: up to `peak` over the first `rise` of its life, thinning as (1 \u{2212} t)^power."),
    m("colour_over_life", Update, "Colour over life", &[p("gradient", Gradient, "[[0, 1, 1, 1, 1], [1, 1, 1, 1, 1]]")])
        .text("Multiplies its colour and opacity: a gradient over its life."),
    m("by_input_update", Update, "By input", &[
        p("input", Input, "\"\""),
        p("target", Enum(&["opacity", "size", "brightness"]), "\"opacity\""),
        p("curve", Curve, "[[0, 0.6], [1, 1]]"),
    ])
    .writes("by_input")
    .text("One of the effect's inputs scales opacity, size or brightness, by a curve of the input.")
    .cost("CPU", "size: fill \u{d7} the curve's highest\u{b2}"),
    m("light_cpu", Update, "Lit (CPU)", &[
        p("mode", Enum(&["sixway", "scatter"]), "\"sixway\""),
        p("phase_g", Number, "0.4").unit(" g").range(-0.95, 0.95, 0.01).optional(),
        p("albedo", Colour, "[0.8, 0.8, 0.8]"),
        p("ambient", Number, "1").unit("\u{d7}").range(0.0, 1.0, 0.01).optional(),
    ])
    .text("Lit once a particle a frame by the room's baked light and the lamps that reach it: six-way through a puff's own volume, or scattered toward the eye (Henyey-Greenstein). Comes after everything that moves or sizes it.")
    .cost("CPU", "CPU \u{d7} lamps in the room"),
    m("adapt_own_light", Update, "Seen as the eye adapts to it", &[
        p("radius", Number, "0.15").unit("m").range(0.005, 2.0, 0.005),
        p("level", Number, "4").range(0.1, 10.0, 0.1),
    ])
    .text("An emissive emitter seen as the eye adapted to it sees it: `radius` of source, held at `level` on the tone curve. Required when an emissive sprite adapts to its own light."),
    m("glint", Update, "Glint (template)", &[]).template(),
    m("ember_glow", Update, "Ember glow (template)", &[p("radiance", Number, "60").range(0.0, 500.0, 1.0)]).template(),
    m("flame_sheets", Update, "Flame sheets (template)", &[p("radiance", Number, "40").range(0.0, 500.0, 1.0)]).template(),
    m("coal_bed", Update, "Coal bed (template)", &[p("glow", Number, "0.6").range(0.0, 5.0, 0.05)]).template(),
    m("pops", Update, "Pops (template)", &[p("per_second", Number, "0.3").unit("/s").range(0.0, 10.0, 0.05)]).template(),
    m("water_entry", Update, "Water entry (template)", &[p("part", Enum(&["drops", "crown", "spray", "froth", "jet", "mist"]), "\"drops\"")]).template(),
    // RENDER
    m("sprite_plain", Render, "Sprite: plain", &[p("book", Book(None), "\"smoke\"")])
        .sprite()
        .text("Alpha from its book, one lit colour, drawn over what is behind (sorted).")
        .cost("fill, plain variant", "fill \u{d7} the plain variant's instructions"),
    m("sprite_sixway", Render, "Sprite: six-way smoke", &[p("book", Book(Some("sixway")), "\"smoke\"")])
        .sprite()
        .text("Two layers a frame: the light passed through it from six sides. Needs Lit (CPU) six-way.")
        .cost("fill, six-way variant", "fill \u{d7} the six-way variant's instructions"),
    m("sprite_scatter", Render, "Sprite: scattered light", &[p("book", Book(None), "\"mote\"")])
        .sprite()
        .text("The light it scatters, screened (only adds): dust, drops. Needs Lit (CPU) scatter.")
        .cost("fill, screened variant", "fill"),
    m("sprite_emissive", Render, "Sprite: emissive", &[
        p("book", Book(None), "\"ember\""),
        p("radiance", Number, "30").range(0.1, 2000.0, 0.5),
        p("colour", Colour, "[1, 1, 1]").optional(),
        p("heat_ramp", Bool, "false").optional(),
        p("adapt", Enum(&["room", "own_light"]), "\"room\"").optional(),
    ])
    .sprite()
    .text("Light, screened through the tone curve: radiance at its hottest, coloured, or through the flame's heat ramp from its book's heat.")
    .cost("fill, screened variant", "fill"),
    m("sprite_fog", Render, "Mist layer (hovering tufts)", &[
        p("density", Number, "0.6").unit("/m").range(0.01, 20.0, 0.01),
        p("height", Number, "1.2").unit("m").range(0.05, 20.0, 0.05),
        p("falloff", Number, "0.4").unit("m").range(0.02, 20.0, 0.01),
        p("edge", Number, "1.4").unit("m").range(0.0, 10.0, 0.05).optional(),
        p("wind", Vec3, "[0.12, 0, 0.07]").unit("m/s").range(-5.0, 5.0, 0.01).optional(),
    ])
    .sprite()
    .text("Not particles: one layer of mist hovering in tufts over a box -- its disc across (Fills object: the placed object's box; size it to the water), standing on its origin, height up. Soft billowing tufts in a band 3/8 of the height up, falloff metres either side of its middle. Density is per metre inside a tuft. It thins to nothing over edge metres in from its rim, drifts with the wind (its SpaceWarp motion), and the hour thickens it at dawn. Needs Lit (CPU) scatter. Takes no placement.")
    .cost("fill: one layer over its box, the fog variant", "the box's pixels in view"),
    m("placement", Render, "Placement", &[
        p("mode", Enum(&["facing", "upright", "streak", "flat", "point"]), "\"facing\""),
        p("streak_seconds", Number, "0").unit("s").range(0.0, 0.2, 0.005).optional(),
    ])
    .required()
    .text("Facing the eye, upright (a flame), streaked along its motion (sparks), flat, or a point (motes, 1.5 px).")
    .cost("vertex only", "-"),
    m("flipbook", Render, "Flipbook", &[p("play", Enum(&["age", "seed"]), "\"age\"").optional()])
        .text("Which frame of its book: by age (blended to the next), or one by its seed.")
        .cost("+1 fetch when blending", "-"),
    m("soft_depth", Render, "Soft fade into surfaces", &[p("metres", Number, "0.1").unit("m").range(0.0, 1.0, 0.01)])
        .text("Fades into what it meets over this many metres, so a sprite never ends in a hard line at a wall.")
        .cost("1 depth fetch", "-"),
    m("motion", Render, "SpaceWarp motion", &[
        p("layer", Enum(&["soft", "bright", "none"]), "\"soft\""),
        p("reason", Text, "\"\"").optional(),
    ])
    .required()
    .text("How SpaceWarp moves its pixels: bright (sharp, bright layers), soft (puffs), or none with a written reason. Every effect writes motion vectors.")
    .cost("motion pass", "motion pass fill"),
    m("sprite_flame", Render, "Sprite: flame (template)", &[p("book", Book(None), "\"flame\"")]).sprite().template(),
    m("sprite_bed", Render, "Sprite: coals (template)", &[p("book", Book(None), "\"coals\"")]).sprite().template(),
    // EVENTS
    m("light_pulse", Events, "Light pulse", &[
        p("intensity", Number, "2").range(0.0, 100.0, 0.1),
        p("range", Number, "4").unit("m").range(0.5, 8.0, 0.1),
        p("frames", Int, "1").range(1.0, 4.0, 1.0).optional(),
        p("gate", Enum(&["visible_only", "always"]), "\"visible_only\"").optional(),
        p("colour", Colour, "[1, 0.8, 0.5]").optional(),
    ])
    .text("A live, unshadowed point light for a few app frames at the event, left out where it could raise no pixel by two levels (every daylight shot).")
    .cost("gated: one live light", "every lit pixel within range pays the surface-light loop on its frames"),
];

/// The catalogue entry of a module written in a stage.
pub fn module(stage: Stage, name: &str) -> Option<&'static Module> {
    MODULES.iter().find(|m| m.stage == stage && m.module == name)
}

pub fn by_key(key: &str) -> Option<&'static Module> {
    MODULES.iter().find(|m| m.key == key)
}

/// The modules an author may add to a stage: the templates' fused ones only
/// when the asset already runs as a template.
pub fn for_stage(stage: Stage, templates: bool) -> impl Iterator<Item = &'static Module> {
    MODULES.iter().filter(move |m| m.stage == stage && (templates || !m.template))
}

/// A module with its defaults, as "add module" writes it: optional
/// parameters left out, and the first of a `one_of` pair written.
pub fn new_module(m: &Module) -> Value {
    let mut out = serde_json::Map::new();
    out.insert("module".into(), json!(m.module));
    let mut chose = false;
    for p in m.params {
        if p.one_of {
            if !chose {
                chose = true;
                out.insert(p.name.into(), p.default_value());
            }
            continue;
        }
        if !p.optional {
            out.insert(p.name.into(), p.default_value());
        }
    }
    Value::Object(out)
}

fn module_of(key: &str) -> Value {
    new_module(by_key(key).expect("a catalogue key"))
}

/// A new emitter: every required module, a gentle rising lit puff, valid as
/// it is made.
pub fn new_emitter(name: &str) -> Value {
    json!({
        "name": name,
        "spawn": [module_of("rate")],
        "initialize": [module_of("life"), module_of("shape_cone"), module_of("speed"), module_of("size")],
        "update": [module_of("ballistic"), module_of("size_over_life"), module_of("puff_fade"), module_of("light_cpu")],
        "render": [module_of("sprite_sixway"), module_of("placement"), module_of("flipbook"), module_of("soft_depth"), module_of("motion")],
        "events": [],
    })
}

/// A blank effect: one emitter, and the scalability and budget an asset needs.
pub fn new_asset(name: &str) -> Value {
    json!({
        "name": clean_name(name),
        "version": super::super::effect_asset::ASSET_VERSION,
        "emitters": [new_emitter("puffs")],
        "scalability": { "cull_distance": 60, "min_pixels": 2, "importance": 0.5, "max_live": 1 },
        "budget": { "fill_megapixel_layers": 1.5, "instances": 100, "lights": 0 },
    })
}

/// A name the format accepts: lower case, digits and `_`.
pub fn clean_name(s: &str) -> String {
    let mut out = String::new();
    for c in s.trim().chars().flat_map(char::to_lowercase) {
        if c.is_ascii_lowercase() || c.is_ascii_digit() {
            out.push(c);
        } else if !out.ends_with('_') {
            out.push('_');
        }
    }
    let out = out.trim_matches('_').to_string();
    if out.is_empty() { "new_effect".into() } else { out }
}

/// Whether the asset spawns on events (a burst) rather than standing.
pub fn is_burst(asset: &Value) -> bool {
    asset["emitters"].as_array().into_iter().flatten().any(|e| e["spawn"].as_array().into_iter().flatten().any(|m| m["module"] == "burst"))
}

/// Whether module `i` of `stage` may be removed from `emitter`: a spawn
/// source, a life, a sprite, a placement and the motion stay (a mist layer
/// takes no placement).
pub fn removable(stage: Stage, emitter: &Value, i: usize) -> bool {
    let list = emitter[stage.key()].as_array();
    let Some(name) = list.and_then(|l| l.get(i)).and_then(|m| m["module"].as_str()) else { return false };
    let Some(m) = module(stage, name) else { return true };
    let count = |what: fn(&Module) -> bool| list.into_iter().flatten().filter(|x| x["module"].as_str().and_then(|n| module(stage, n)).is_some_and(what)).count();
    if m.key == "placement" && emitter["render"].as_array().into_iter().flatten().any(|x| x["module"] == "sprite_fog") {
        return true;
    }
    if m.required {
        return false;
    }
    if m.source {
        return count(|m| m.source) > 1;
    }
    if m.sprite {
        return count(|m| m.sprite) > 1;
    }
    true
}

/// One row of the budget panel: green, or red with what pushes it over.
#[derive(Clone, Debug, PartialEq)]
pub struct BudgetRow {
    pub key: &'static str,
    pub label: String,
    pub ok: bool,
    pub text: String,
    pub scales: &'static str,
}

fn num(v: &Value) -> f64 {
    v.as_f64().unwrap_or(f64::NAN)
}

fn fmt(v: f64, d: usize) -> String {
    if v.is_finite() { format!("{v:.d$}") } else { "\u{2013}".into() }
}

/// The budget panel's rows, from the validator's verdict (`check_json`).
pub fn budget_rows(asset: &Value, verdict: &Value) -> Vec<BudgetRow> {
    let r = &verdict["report"];
    if !r.is_object() {
        return Vec::new();
    }
    let limits = &verdict["limits"];
    let budget = &asset["budget"];
    let or_inf = |v: &Value| v.as_f64().unwrap_or(f64::INFINITY);
    let heaviest = |field: &str| {
        r["emitters"]
            .as_array()
            .and_then(|e| e.iter().max_by(|a, b| num(&a[field]).total_cmp(&num(&b[field]))))
            .and_then(|e| e["name"].as_str())
            .map(|n| format!(" (most: {n})"))
            .unwrap_or_default()
    };
    let inst_limit = or_inf(&budget["instances"]).min(or_inf(&limits["instances"]));
    let fill_limit = or_inf(&budget["fill_megapixel_layers"]).min(or_inf(&limits["fill_steady"]));
    let (inst, fill) = (num(&r["instances_steady"]), num(&r["fill_steady"]));
    let reference = num(&r["reference_distance"]);
    let mut rows = vec![BudgetRow {
        key: "instances",
        label: "Live particles (at its cap)".into(),
        ok: inst <= inst_limit,
        text: format!("{} of {} budgeted (limit {}){}", fmt(inst, 0), fmt(num(&budget["instances"]), 0), fmt(num(&limits["instances"]), 0), heaviest("instances")),
        scales: "count, or rate \u{d7} longest life; \u{d7} max_live for what outlives its event",
    }];
    let seen_side = r["seen_from_side"].as_bool().unwrap_or(false);
    rows.push(BudgetRow {
        key: "fill",
        label: if seen_side { format!("Fill seen from {} m off its side", fmt(reference, 1)) } else { format!("Fill at {} m", fmt(reference, 1)) },
        ok: fill <= fill_limit,
        text: format!(
            "{} megapixel-layers of {} budgeted (limit {}){}{}",
            fmt(fill, 3),
            fmt(num(&budget["fill_megapixel_layers"]), 2),
            fmt(num(&limits["fill_steady"]), 2),
            if seen_side { ": a worst-case hint; in the level the governor weighs what each frame's eyes see" } else { "" },
            heaviest("fill"),
        ),
        scales: if seen_side { "its box's pixels in view \u{d7} layers" } else { "particles \u{d7} size\u{b2} \u{d7} 1/distance\u{b2} \u{d7} the book's cutout" },
    });
    if r["burst"].as_bool().unwrap_or(false) {
        let (one, limit) = (num(&r["fill_per_burst"]), num(&limits["fill_burst"]));
        rows.push(BudgetRow {
            key: "fill_burst",
            label: "Fill of one burst".into(),
            ok: !(one > limit),
            text: format!("{} of {} megapixel-layers; {} particles", fmt(one, 3), fmt(limit, 2), fmt(num(&r["instances_per_burst"]), 0)),
            scales: "one event's count \u{d7} size\u{b2}",
        });
    }
    let variants = verdict["variants"].as_array().cloned().unwrap_or_default();
    if let Some(worst) = variants.iter().max_by(|a, b| num(&a["instructions"]).total_cmp(&num(&b["instructions"]))) {
        let mut names: Vec<&str> = variants.iter().filter_map(|v| v["variant"].as_str()).collect();
        names.dedup();
        let (i, limit) = (num(&worst["instructions"]), num(&worst["limit"]));
        rows.push(BudgetRow {
            key: "shader",
            label: "Shader variant".into(),
            ok: i <= limit,
            text: format!("{}: {} of {} instructions (estimated from the shader's size, not yet measured)", names.join(", "), fmt(i, 0), fmt(limit, 0)),
            scales: "paid by every pixel it fills",
        });
    }
    let (bytes, limit) = (num(&r["texture_bytes"]), num(&limits["texture_bytes"]));
    rows.push(BudgetRow {
        key: "texture",
        label: "Textures".into(),
        ok: !(bytes > limit),
        text: format!("{} MB of books (a level's limit {} MB, shared books once)", fmt(bytes / 1_048_576.0, 2), fmt(limit / 1_048_576.0, 0)),
        scales: "the books it names",
    });
    let (lights, budget_lights, limit_lights) = (num(&r["lights"]), budget["lights"].as_f64().unwrap_or(0.0), limits["lights"].as_f64().unwrap_or(1.0));
    rows.push(BudgetRow {
        key: "lights",
        label: "Live lights".into(),
        ok: lights <= budget_lights && lights <= limit_lights,
        text: format!("{} of {} budgeted (limit {})", fmt(lights, 0), fmt(budget_lights, 0), fmt(limit_lights, 0)),
        scales: "every lit pixel within its range, on its frames",
    });
    rows
}

/// How far a coupled slider may go before a budget is crossed, as a multiple
/// of its value now: a count, rate or life moves particles and fill in step,
/// a size moves fill as its square. At least 1.
pub fn headroom(couples: Couples, asset: &Value, verdict: &Value) -> f32 {
    let r = &verdict["report"];
    if !r.is_object() {
        return f32::INFINITY;
    }
    let (limits, budget) = (&verdict["limits"], &asset["budget"]);
    let or_inf = |v: &Value| v.as_f64().unwrap_or(f64::INFINITY);
    let inst = or_inf(&budget["instances"]).min(or_inf(&limits["instances"])) / num(&r["instances_steady"]).max(1.0);
    let fill = or_inf(&budget["fill_megapixel_layers"]).min(or_inf(&limits["fill_steady"])) / num(&r["fill_steady"]).max(1e-6);
    let burst = if r["burst"].as_bool().unwrap_or(false) { or_inf(&limits["fill_burst"]) / num(&r["fill_per_burst"]).max(1e-6) } else { f64::INFINITY };
    let ratio = match couples {
        Couples::Fill => fill.min(burst).sqrt(),
        Couples::Instances => inst.min(fill).min(burst),
    };
    ratio.max(1.0) as f32
}

/// Whether the asset may be saved, and why: the validator said ok (an
/// over-budget one only with its written reason) and it runs as an asset.
#[derive(Clone, Debug, PartialEq)]
pub struct SaveState {
    pub ok: bool,
    pub why: String,
    /// Over a budget: the reason field is offered.
    pub over_budget: bool,
}

pub fn save_state(verdict: &Value) -> SaveState {
    let issues = verdict["issues"].as_array().cloned().unwrap_or_default();
    let budget = issues.iter().filter(|i| i["budget"] == true).count();
    let other = issues.len() - budget;
    let no = |why: String, over_budget: bool| SaveState { ok: false, why, over_budget };
    if verdict["ok"] != true {
        if other > 0 {
            return no(format!("{other} problem{} the engine refuses", if other > 1 { "s" } else { "" }), budget > 0);
        }
        if budget > 0 {
            return no("Over budget: lower it, or write why it may exceed it".into(), true);
        }
        return no("Refused".into(), false);
    }
    if verdict["runs"] != "asset" {
        return no("Uses the built-in templates' fused modules: runs only as a scene's kind".into(), false);
    }
    match verdict["report"]["over_budget"]["reason"].as_str() {
        Some(reason) => SaveState { ok: true, why: format!("Over budget, because: {reason}"), over_budget: true },
        None => SaveState { ok: true, why: "Inside every budget".into(), over_budget: false },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::renderer::effect_asset::EffectAsset;
    use crate::renderer::effects::authoring::{check_json, templates_json};

    #[test]
    fn a_new_effect_passes_the_validator_and_runs_as_an_asset() {
        let a = new_asset("My Effect!");
        assert_eq!(a["name"], "my_effect");
        let v = check_json(&a.to_string());
        assert_eq!(v["ok"], true, "{v}");
        assert_eq!(v["runs"], "asset", "{v}");
        assert!(budget_rows(&a, &v).iter().all(|r| r.ok), "{:?}", budget_rows(&a, &v));
        assert!(save_state(&v).ok);
        assert!(!is_burst(&a));
    }

    #[test]
    fn every_module_added_with_its_defaults_is_one_the_format_reads() {
        for m in MODULES {
            let mut a = new_asset("probe");
            a["emitters"][0][m.stage.key()].as_array_mut().unwrap().push(new_module(m));
            if let Err(e) = EffectAsset::parse(&a.to_string()) {
                panic!("{} ({}): {e}", m.key, m.stage.key());
            }
            // Every optional parameter, written with its default, is read too.
            let mut full = new_module(m);
            for p in m.params {
                full[p.name] = p.default_value();
            }
            if m.params.iter().filter(|p| p.one_of).count() > 1 {
                continue;
            }
            let mut a = new_asset("probe");
            a["emitters"][0][m.stage.key()].as_array_mut().unwrap().push(full);
            if let Err(e) = EffectAsset::parse(&a.to_string()) {
                panic!("{} with every parameter: {e}", m.key);
            }
        }
    }

    #[test]
    fn every_module_the_shipped_effects_and_templates_use_is_in_the_catalogue() {
        let game = crate::test_game_dir().join("effects");
        let mut assets: Vec<Value> = std::fs::read_dir(&game)
            .into_iter()
            .flatten()
            .flatten()
            .filter(|e| e.path().extension().is_some_and(|x| x == "json"))
            .map(|e| serde_json::from_str(&std::fs::read_to_string(e.path()).unwrap()).unwrap())
            .collect();
        assets.extend(templates_json().as_array().unwrap().iter().map(|t| t["asset"].clone()));
        assert!(assets.len() > 6, "the shipped effects were read");
        for a in &assets {
            for e in a["emitters"].as_array().unwrap() {
                for stage in Stage::ALL {
                    for x in e[stage.key()].as_array().into_iter().flatten() {
                        let name = x["module"].as_str().unwrap();
                        assert!(module(stage, name).is_some(), "{} uses {name} in {}, which the catalogue lacks", a["name"], stage.key());
                    }
                }
            }
        }
    }

    #[test]
    fn the_required_modules_stay_and_a_second_source_or_sprite_may_go() {
        let mut e = new_emitter("e");
        assert!(!removable(Stage::Spawn, &e, 0), "the only spawn");
        assert!(!removable(Stage::Initialize, &e, 0), "life");
        assert!(!removable(Stage::Render, &e, 0), "the only sprite");
        assert!(!removable(Stage::Render, &e, 4), "motion");
        assert!(removable(Stage::Update, &e, 0), "ballistic");
        e["spawn"].as_array_mut().unwrap().push(module_of("burst"));
        assert!(removable(Stage::Spawn, &e, 0));
        e["render"].as_array_mut().unwrap().push(module_of("sprite_fog"));
        assert!(removable(Stage::Render, &e, 1), "a mist layer takes no placement");
    }

    #[test]
    fn a_budget_over_is_red_and_saved_only_with_its_reason() {
        let mut a = new_asset("busy");
        a["emitters"][0]["spawn"][0]["per_second"] = json!(150);
        a["emitters"][0]["initialize"][0]["value"] = json!(10);
        let v = check_json(&a.to_string());
        let rows = budget_rows(&a, &v);
        let s = save_state(&v);
        assert!(!s.ok && s.over_budget, "{s:?} {v}");
        a["over_budget"] = json!("the boiler room's hero vent");
        let v = check_json(&a.to_string());
        assert!(save_state(&v).ok, "{v}");
        assert!(budget_rows(&a, &v).iter().any(|r| !r.ok), "{rows:?}");
        assert!(headroom(Couples::Instances, &a, &v) >= 1.0);
    }
}
