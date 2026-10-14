//! WHAT THE EDITOR ASKS THE ENGINE while an effect is authored: the
//! validator's verdict and costs for an asset (`EffectAsset::check`, the very
//! code a level load runs), whether the engine can run it and how
//! (`CompiledEffect::compile` for an authored asset, `templates::Template`
//! for a kind's), each emitter's shader variant and its size by the proxy,
//! the books there are, the built-in templates, the limits, and a scene's
//! placed effects against the scene's budgets. As JSON, for the editor's
//! backend (`src/bin/effect_check.rs`), so no rule is written twice.
//!
//! WHAT EACH COST SCALES WITH (stated in every report):
//! - instances: a burst's count, or a stream's rate x its longest life; at a
//!   source's cap, times `max_live` for what outlives its event;
//! - fill: instances x the mean drawn area at the reference view
//!   ((2 x mean half size / distance x 604 px)^2, both eyes, times the
//!   book's cutout share) -- so x size^2, x rate, x 1/distance^2;
//! - shader: the variant's instructions (by the proxy x 1.87 until a headset
//!   logs them), the same for every particle it draws;
//! - lights: one live light per `light_pulse`, for its frames.

use super::stack::CompiledEffect;
use super::templates;
use super::variants::FxVariant;
use super::{books, EffectKind};
use crate::renderer::effect_asset::{self as asset, Channels, EffectAsset, Issue, Report};
use serde_json::{json, Value};

/// Proxy units to instructions, until the effects' own PIPESTATS are logged
/// (the scene readers' ratio; `variants::tests::PROXY_TO_INSTRUCTIONS`).
pub const PROXY_TO_INSTRUCTIONS: f32 = 1.87;
/// The most instructions an effect's variant may be (plan 5.1), provisional.
pub const LIMIT_INSTRUCTIONS: f32 = 1200.0;

/// A SCENE'S PLACED EFFECTS: the most fill of placed effects within
/// [`SCENE_NEIGHBOURHOOD`] of any one of them (a room's worth, what one view
/// can hold at once), megapixel-layers at each one's reference view. The
/// frame's governor holds the same number at run time
/// (`stack::FRAME_FILL_BUDGET`); this refuses a level that would lean on it
/// everywhere.
pub const LIMIT_SCENE_FILL: f32 = super::stack::FRAME_FILL_BUDGET;
pub const SCENE_NEIGHBOURHOOD: f32 = 8.0;
/// Placed effects' live particles in a whole level: the editor's existing
/// guidance (`deviceBudget.js`, "Effect particles").
pub const LIMIT_SCENE_INSTANCES: u32 = 4000;

fn issues_json(issues: &[Issue]) -> Value {
    Value::Array(issues.iter().map(|i| json!({ "budget": i.budget, "emitter": i.emitter, "message": i.message })).collect())
}

fn channels(c: Channels) -> &'static str {
    match c {
        Channels::Sixway => "sixway",
        Channels::HeatCover => "heat_cover",
        Channels::Alpha => "alpha",
        Channels::Bed => "bed",
    }
}

/// EACH VARIANT'S SIZE BY THE PROXY (`shader_inlining`, the inlined
/// function sizes a fragment entry reaches), in `FxVariant::ALL`'s order.
/// The proxy is a test-only tool, so the numbers are pinned here and
/// `the_variant_table_is_the_proxys` fails the moment a shader edit moves one.
pub const VARIANT_PROXY: [usize; 8] = [436, 114, 160, 215, 142, 130, 105, 126];

/// A variant's size by the proxy, and as instructions estimated.
pub fn variant_cost(v: FxVariant) -> (usize, f32) {
    let proxy = VARIANT_PROXY[FxVariant::ALL.iter().position(|x| *x == v).expect("every variant is listed")];
    (proxy, proxy as f32 * PROXY_TO_INSTRUCTIONS)
}

fn report_json(r: &Report) -> Value {
    json!({
        "emitters": r.emitters.iter().map(|e| json!({
            "name": e.name, "instances": e.instances, "longest_life": e.longest_life, "fill": e.fill, "stream": e.stream,
        })).collect::<Vec<_>>(),
        "instances_per_burst": r.instances_per_burst,
        "fill_per_burst": r.fill_per_burst,
        "instances_steady": r.instances_steady,
        "fill_steady": r.fill_steady,
        "lights": r.lights,
        "texture_bytes": r.texture_bytes,
        "reference_distance": r.reference_distance,
        "burst": r.burst,
        "over_budget": r.over_budget.as_ref().map(|(reason, what)| json!({ "reason": reason, "exceeds": what })),
    })
}

/// THE VERDICT ON AN ASSET, as the editor shows it and its backend saves by:
/// `ok` (the validator passed: it may be saved), its `report` (costs) or its
/// `issues` (why not), how the engine runs it (`runs`: "asset" -- raised by an
/// event or placed by a scene object's `effect.asset`; "template" -- the
/// placed emitters' loop, as a kind; or null with `runs_why`), each
/// emitter's shader variant and its size, and the limits it was held to.
pub fn check_json(text: &str) -> Value {
    let a = match EffectAsset::parse(text) {
        Ok(a) => a,
        Err(e) => return json!({ "ok": false, "parse_error": e, "issues": [{ "budget": false, "emitter": null, "message": e }] }),
    };
    let books = books();
    let (ok, report, issues) = match a.check(books) {
        Ok(r) => (true, report_json(&r), json!([])),
        Err(i) => (false, Value::Null, issues_json(&i)),
    };
    // How the engine runs it.
    let (runs, why, variants) = match CompiledEffect::compile(&a, books) {
        Ok(c) => {
            let variants: Vec<Value> = c
                .programs
                .iter()
                .map(|p| {
                    let v = match p.shade {
                        super::stack::Shade::Plain => FxVariant::Plain,
                        super::stack::Shade::Sixway => FxVariant::Sixway,
                        super::stack::Shade::Scatter | super::stack::Shade::Emissive { .. } => FxVariant::ScreenPlain,
                        super::stack::Shade::EmissiveHeat { .. } => FxVariant::ScreenHeat,
                    };
                    let (proxy, instructions) = variant_cost(v);
                    json!({
                        "emitter": p.name, "variant": v.label(), "screened": v.screened(), "proxy": proxy,
                        "instructions": instructions, "limit": LIMIT_INSTRUCTIONS, "motion": format!("{:?}", p.motion).to_lowercase(),
                        "book": p.book.name, "placement": format!("{:?}", p.placement).to_lowercase(),
                    })
                })
                .collect();
            (json!("asset"), Value::Null, json!(variants))
        }
        Err(e) => match templates::Template::compile(&a) {
            Ok(_) => (json!("template"), Value::Null, json!([])),
            Err(t) => (Value::Null, json!(format!("{}; as a template: {t}", e.iter().map(|i| i.to_string()).collect::<Vec<_>>().join("; "))), json!([])),
        },
    };
    json!({ "ok": ok, "name": a.name, "report": report, "issues": issues, "runs": runs, "runs_why": why, "variants": variants, "limits": limits_json() })
}

/// The hard limits, as the validator holds them.
pub fn limits_json() -> Value {
    json!({
        "instances": asset::LIMIT_INSTANCES,
        "fill_burst": asset::LIMIT_FILL_BURST,
        "fill_steady": asset::LIMIT_FILL_STEADY,
        "lights": asset::LIMIT_LIGHTS,
        "texture_bytes": asset::LIMIT_TEXTURE_BYTES,
        "instructions": LIMIT_INSTRUCTIONS,
        "near_eye": asset::NEAR_EYE,
        "reference_held": asset::REFERENCE_HELD,
        "reference_placed": asset::REFERENCE_PLACED,
        "reference_focal_pixels": asset::REFERENCE_FOCAL_PIXELS,
        "reference_eyes": asset::REFERENCE_EYES,
        "max_curve_keys": asset::MAX_CURVE_KEYS,
        "max_inputs": asset::MAX_INPUTS,
        "scene_fill": LIMIT_SCENE_FILL,
        "scene_neighbourhood": SCENE_NEIGHBOURHOOD,
        "scene_instances": LIMIT_SCENE_INSTANCES,
        "frame_fill": super::stack::FRAME_FILL_BUDGET,
        "frame_instances": super::stack::FRAME_INSTANCE_BUDGET,
    })
}

/// The books an asset can name.
pub fn books_json() -> Value {
    Value::Array(
        books()
            .iter()
            .map(|b| {
                json!({
                    "name": b.name, "first_layer": b.first_layer, "frames": b.frames, "variants": b.variants,
                    "channels": channels(b.channels), "cutout_share": b.cutout_share, "bytes": b.bytes,
                })
            })
            .collect(),
    )
}

/// The built-in templates, each as its stack and the verdict on it.
pub fn templates_json() -> Value {
    Value::Array(
        templates::assets()
            .into_iter()
            .map(|a| {
                let text = a.to_json();
                json!({ "asset": serde_json::from_str::<Value>(&text).expect("an asset is JSON"), "check": check_json(&text) })
            })
            .collect(),
    )
}

/// One placed effect as the scene budget weighs it.
#[derive(Clone, Debug, PartialEq)]
pub struct PlacedCost {
    pub object: String,
    pub position: [f32; 3],
    pub fill: f32,
    pub instances: u32,
}

/// What a scene's placed effects come to against its budgets.
#[derive(Clone, Debug, PartialEq)]
pub struct SceneCost {
    pub placed: Vec<PlacedCost>,
    /// The most fill within [`SCENE_NEIGHBOURHOOD`] of one placed effect, and
    /// which.
    pub worst_fill: f32,
    pub worst_at: Option<String>,
    pub instances: u32,
    /// Empty when inside every budget.
    pub over: Vec<String>,
}

/// What one placed effect costs: its stack's report at its cap, its fill x
/// `scale`^2 x `rate` and its instances x `rate`; a box of dust (`density`,
/// per cubic metre) its `volume`'s worth of motes at the 1.5 pixel floor.
pub fn placed_cost(object: &str, position: [f32; 3], report: &Report, density: Option<f32>, scale: f32, rate: f32, volume: f32) -> PlacedCost {
    let (scale, rate) = (scale.max(0.0), rate.max(0.0));
    match density {
        Some(d) => {
            let n = (d * volume * rate).ceil() as u32;
            PlacedCost { object: object.into(), position, fill: n as f32 * 1.5 * 1.5 * asset::REFERENCE_EYES / 1e6, instances: n }
        }
        None => PlacedCost {
            object: object.into(),
            position,
            fill: report.fill_steady * scale * scale * rate,
            instances: (report.instances_steady as f32 * rate).ceil() as u32,
        },
    }
}

/// THE SCENE'S BUDGETS: the worst neighbourhood's fill against
/// [`LIMIT_SCENE_FILL`] and the level's instances against
/// [`LIMIT_SCENE_INSTANCES`]. Scales with placed effects within 8 m of one
/// another; the rest of a level costs nothing at once (it is culled).
pub fn scene_cost(placed: Vec<PlacedCost>) -> SceneCost {
    let mut worst = (0.0f32, None);
    for p in &placed {
        let here: f32 = placed
            .iter()
            .filter(|q| {
                let d = [q.position[0] - p.position[0], q.position[1] - p.position[1], q.position[2] - p.position[2]];
                d[0] * d[0] + d[1] * d[1] + d[2] * d[2] <= SCENE_NEIGHBOURHOOD * SCENE_NEIGHBOURHOOD
            })
            .map(|q| q.fill)
            .sum();
        if here > worst.0 {
            worst = (here, Some(p.object.clone()));
        }
    }
    let instances: u32 = placed.iter().map(|p| p.instances).sum();
    let mut over = Vec::new();
    if worst.0 > LIMIT_SCENE_FILL {
        over.push(format!(
            "placed effects within {SCENE_NEIGHBOURHOOD} m of `{}` fill {:.2} megapixel-layers; a scene's limit is {LIMIT_SCENE_FILL}",
            worst.1.clone().unwrap_or_default(),
            worst.0
        ));
    }
    if instances > LIMIT_SCENE_INSTANCES {
        over.push(format!("placed effects keep {instances} particles alive; a scene's limit is {LIMIT_SCENE_INSTANCES}"));
    }
    SceneCost { placed, worst_fill: worst.0, worst_at: worst.1, instances, over }
}

/// A kind's template report (and its density, for dust).
pub fn kind_report(kind: EffectKind) -> (Report, Option<f32>) {
    let a = templates::asset(kind);
    let density = a.emitters.iter().flat_map(|e| &e.spawn).find_map(|m| if let asset::SpawnModule::Density { per_cubic_metre } = m { Some(*per_cubic_metre) } else { None });
    (a.check(books()).expect("the built-in templates check"), density)
}

/// THE SCENE CHECK the editor and a level load both run: `items` are a
/// scene's placed effects, each `{ "object", "kind" | "asset", "position":
/// [x,y,z] (world), "scale", "rate", "volume" }`; `assets` the effect
/// assets by name (their JSON). An asset that is missing or refused is an
/// issue of its own.
pub fn scene_json(items: &Value, assets: &std::collections::HashMap<String, String>) -> Value {
    let mut placed = Vec::new();
    let mut issues = Vec::new();
    for it in items.as_array().into_iter().flatten() {
        let object = it["object"].as_str().unwrap_or("?").to_string();
        let f = |k: &str, d: f32| it[k].as_f64().map_or(d, |v| v as f32);
        let p = &it["position"];
        let position = [p[0].as_f64().unwrap_or(0.0) as f32, p[1].as_f64().unwrap_or(0.0) as f32, p[2].as_f64().unwrap_or(0.0) as f32];
        let (report, density) = if let Some(name) = it["asset"].as_str().filter(|s| !s.is_empty()) {
            match assets.get(name).map(|t| EffectAsset::parse(t).and_then(|a| a.check(books()).map_err(|i| i.iter().map(|i| i.to_string()).collect::<Vec<_>>().join("; ")))) {
                Some(Ok(r)) => (r, None),
                Some(Err(e)) => {
                    issues.push(format!("`{object}` names the effect `{name}`, which is refused: {e}"));
                    continue;
                }
                None => {
                    issues.push(format!("`{object}` names the effect `{name}`, which is not in game/effects"));
                    continue;
                }
            }
        } else {
            match it["kind"].as_str().and_then(EffectKind::from_name) {
                Some(k) => kind_report(k),
                None => continue,
            }
        };
        placed.push(placed_cost(&object, position, &report, density, f("scale", 1.0), f("rate", 1.0), f("volume", 0.0)));
    }
    let cost = scene_cost(placed);
    json!({
        "ok": cost.over.is_empty() && issues.is_empty(),
        "worst_fill": cost.worst_fill,
        "worst_at": cost.worst_at,
        "instances": cost.instances,
        "over": cost.over,
        "issues": issues,
        "placed": cost.placed.iter().map(|p| json!({ "object": p.object, "fill": p.fill, "instances": p.instances })).collect::<Vec<_>>(),
        "limits": { "fill": LIMIT_SCENE_FILL, "neighbourhood": SCENE_NEIGHBOURHOOD, "instances": LIMIT_SCENE_INSTANCES },
    })
}

/// ONE REQUEST of the backend's long-lived checker (`effect_check serve`):
/// `{"op": "check", "text": <asset JSON text>}`, `{"op": "books"}`,
/// `{"op": "templates"}`, `{"op": "limits"}`, `{"op": "scene", "items": [..],
/// "assets": {name: text}}`. Unknown ops answer with an error.
pub fn answer(request: &Value) -> Value {
    match request["op"].as_str() {
        Some("check") => match request["text"].as_str() {
            Some(t) => check_json(t),
            None => check_json(&request["asset"].to_string()),
        },
        Some("books") => books_json(),
        Some("templates") => templates_json(),
        Some("limits") => limits_json(),
        Some("scene") => {
            let assets: std::collections::HashMap<String, String> =
                request["assets"].as_object().into_iter().flatten().map(|(k, v)| (k.clone(), v.as_str().map(str::to_string).unwrap_or_else(|| v.to_string()))).collect();
            scene_json(&request["items"], &assets)
        }
        Some("ping") => json!({ "ok": true, "version": env!("CARGO_PKG_VERSION") }),
        other => json!({ "error": format!("unknown op {other:?}") }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The editor's verdict is the validator's: an asset it checks is ok
    /// with its costs and variant; a typo is refused, named; one over budget
    /// is refused unless it says why.
    #[test]
    fn the_editors_verdict_is_the_validators() {
        let vent = super::super::stack::tests::VENT;
        let v = check_json(vent);
        assert_eq!(v["ok"], true, "{v}");
        assert_eq!(v["runs"], "asset");
        assert_eq!(v["variants"][0]["variant"], "effects_over_sixway");
        assert!(v["variants"][0]["instructions"].as_f64().unwrap() < LIMIT_INSTRUCTIONS as f64, "{v}");
        assert_eq!(v["report"]["instances_steady"], 20);
        let typo = check_json(&vent.replace("\"drag\"", "\"drga\""));
        assert_eq!(typo["ok"], false);
        assert!(typo["issues"][0]["message"].as_str().unwrap().contains("drga"), "{typo}");
        let over = vent.replace("\"per_second\": 10", "\"per_second\": 100");
        let refused = check_json(&over);
        assert_eq!(refused["ok"], false, "{refused}");
        assert!(refused["issues"].as_array().unwrap().iter().any(|i| i["budget"] == true), "{refused}");
        let excused = check_json(&over.replace("\"version\": 1,", "\"version\": 1, \"over_budget\": \"the hero vent of the boiler room\","));
        assert_eq!(excused["ok"], true, "{excused}");
        assert_eq!(excused["report"]["over_budget"]["reason"], "the hero vent of the boiler room");
        // Every template answers, and runs as a template or an asset.
        let t = templates_json();
        assert_eq!(t.as_array().unwrap().len(), 6);
        assert!(t.as_array().unwrap().iter().all(|t| t["check"]["ok"] == true), "{t}");
    }

    /// The pinned variant sizes are the proxy's, today.
    #[test]
    fn the_variant_table_is_the_proxys() {
        let live: Vec<usize> = FxVariant::ALL
            .iter()
            .map(|v| crate::renderer::shader_inlining::inlined_sizes(&v.shader(), v.entry()).iter().map(|f| f.own * f.copies).sum())
            .collect();
        assert_eq!(live, VARIANT_PROXY.to_vec(), "update VARIANT_PROXY: the effects' shader changed");
    }

    /// A scene's placed effects are held to the neighbourhood fill and the
    /// level's particles: many smokes in one room are refused, the same
    /// spread out are not.
    #[test]
    fn a_scene_is_held_to_its_fill_where_effects_crowd() {
        let smoke = |k: usize, spacing: f32| json!({ "object": format!("s{k}"), "kind": "smoke", "position": [k as f32 * spacing, 0.0, 0.0], "scale": 1.0, "rate": 1.0 });
        let crowded = json!((0..10).map(|k| smoke(k, 0.5)).collect::<Vec<_>>());
        let spread = json!((0..10).map(|k| smoke(k, 20.0)).collect::<Vec<_>>());
        let none = std::collections::HashMap::new();
        let a = scene_json(&crowded, &none);
        let b = scene_json(&spread, &none);
        assert_eq!(a["ok"], false, "{a}");
        assert!(a["over"][0].as_str().unwrap().contains("megapixel-layers"), "{a}");
        assert_eq!(b["ok"], true, "{b}");
        assert!((b["worst_fill"].as_f64().unwrap() - a["worst_fill"].as_f64().unwrap() / 10.0).abs() < 1e-3);
        let missing = scene_json(&json!([{ "object": "v", "asset": "no_such_effect", "position": [0, 0, 0] }]), &none);
        assert_eq!(missing["ok"], false);
    }
}
