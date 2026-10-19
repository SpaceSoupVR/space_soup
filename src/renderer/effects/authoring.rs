use super::stack::CompiledEffect;
use super::templates;
use super::variants::FxVariant;
use super::{books, EffectKind};
use crate::renderer::effect_asset::{self as asset, Channels, EffectAsset, Issue, Report};
use serde_json::{json, Value};

pub const PROXY_TO_INSTRUCTIONS: f32 = 1.87;

pub const LIMIT_INSTRUCTIONS: f32 = 1200.0;

pub const LIMIT_SCENE_FILL: f32 = super::stack::FRAME_FILL_BUDGET;
pub const SCENE_NEIGHBOURHOOD: f32 = 8.0;

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

pub const VARIANT_PROXY: [usize; 9] = [436, 114, 160, 215, 142, 130, 105, 126, 462];

pub fn variant_cost(v: FxVariant) -> (usize, f32) {
    let proxy = VARIANT_PROXY[FxVariant::ALL.iter().position(|x| *x == v).expect("every variant is listed")];
    (proxy, proxy as f32 * PROXY_TO_INSTRUCTIONS)
}

fn report_json(r: &Report) -> Value {
    json!({
        "emitters": r.emitters.iter().map(|e| json!({
            "name": e.name, "instances": e.instances, "longest_life": e.longest_life, "fill": e.fill, "stream": e.stream,
            "area": e.area, "flat": e.flat, "points": e.points, "fills_disc": e.fills_disc, "volume": e.volume,
        })).collect::<Vec<_>>(),
        "instances_per_burst": r.instances_per_burst,
        "fill_per_burst": r.fill_per_burst,
        "instances_steady": r.instances_steady,
        "fill_steady": r.fill_steady,
        "lights": r.lights,
        "texture_bytes": r.texture_bytes,
        "reference_distance": r.reference_distance,
        "burst": r.burst,
        "seen_from_side": r.seen_from_side,
        "extent": { "lateral": r.extent.lateral, "below": r.extent.below, "above": r.extent.above },
        "area_facing": r.area_facing,
        "area_flat": r.area_flat,
        "points": r.points,
        "volumes": r.volumes,
        "fills_disc": fills_disc(r),
        "over_budget": r.over_budget.as_ref().map(|(reason, what)| json!({ "reason": reason, "exceeds": what })),
    })
}

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
                        super::stack::Shade::Fog { .. } => FxVariant::Fog,
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
        "reference_tan": super::seen::REFERENCE_TAN,
        "standing_eye": super::seen::STANDING_EYE,
        "max_curve_keys": asset::MAX_CURVE_KEYS,
        "max_inputs": asset::MAX_INPUTS,
        "scene_fill": LIMIT_SCENE_FILL,
        "scene_neighbourhood": SCENE_NEIGHBOURHOOD,
        "scene_instances": LIMIT_SCENE_INSTANCES,
        "frame_fill": super::stack::FRAME_FILL_BUDGET,
        "frame_instances": super::stack::FRAME_INSTANCE_BUDGET,
    })
}

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

#[derive(Clone, Debug, PartialEq)]
pub struct PlacedCost {
    pub object: String,
    pub position: [f32; 3],
    pub fill: f32,
    pub instances: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SceneCost {
    pub placed: Vec<PlacedCost>,

    pub worst_fill: f32,
    pub worst_at: Option<String>,
    pub instances: u32,

    pub over: Vec<String>,
}

pub fn fills_disc(r: &Report) -> f32 {
    r.emitters.iter().map(|e| e.fills_disc).fold(0.0, f32::max)
}

pub fn placed_cost(object: &str, position: [f32; 3], report: &Report, density: Option<f32>, scale: f32, rate: f32, volume: f32, footprint: [f32; 2]) -> PlacedCost {
    let (scale, rate) = (scale.max(0.0), rate.max(0.0));
    if let Some(d) = density {
        let n = (d * volume * rate).ceil() as u32;
        return PlacedCost { object: object.into(), position, fill: n as f32 * 1.5 * 1.5 * asset::REFERENCE_EYES / 1e6, instances: n };
    }
    let disc = fills_disc(report);
    let stretched = disc > 0.0 && footprint[0] > 0.0 && footprint[1] > 0.0;
    let more = if stretched { footprint[0] * footprint[1] / (disc * disc) } else { 1.0 };
    let fill = if report.seen_from_side {
        let e = report.extent;
        let half = if stretched { [footprint[0] + e.lateral - disc, footprint[1] + e.lateral - disc] } else { [e.lateral; 2] };
        let (lo, hi, eyes) = super::seen::stated_view_of(half, e.below, e.above, report.reference_distance);
        let grown = scale * scale * rate * more;
        super::seen::seen_fill(&eyes, lo, hi, report.area_facing * grown, report.area_flat * grown, report.points as f32 * rate * more, report.volumes as f32)
    } else {
        report.fill_steady * scale * scale * rate
    };
    PlacedCost { object: object.into(), position, fill, instances: (report.instances_steady as f32 * rate * more).ceil() as u32 }
}

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

pub fn kind_report(kind: EffectKind) -> (Report, Option<f32>) {
    let a = templates::asset(kind);
    let density = a.emitters.iter().flat_map(|e| &e.spawn).find_map(|m| if let asset::SpawnModule::Density { per_cubic_metre } = m { Some(*per_cubic_metre) } else { None });
    (a.check(books()).expect("the built-in templates check"), density)
}

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
        let fp = &it["footprint"];
        let footprint = [fp[0].as_f64().unwrap_or(0.0) as f32, fp[1].as_f64().unwrap_or(0.0) as f32];
        placed.push(placed_cost(&object, position, &report, density, f("scale", 1.0), f("rate", 1.0), f("volume", 0.0), footprint));
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

        let t = templates_json();
        assert_eq!(t.as_array().unwrap().len(), 6);
        assert!(t.as_array().unwrap().iter().all(|t| t["check"]["ok"] == true), "{t}");
    }

    #[test]
    fn the_variant_table_is_the_proxys() {
        let live: Vec<usize> = FxVariant::ALL
            .iter()
            .map(|v| crate::renderer::shader_inlining::inlined_sizes(&v.shader(), v.entry()).iter().map(|f| f.own * f.copies).sum())
            .collect();
        assert_eq!(live, VARIANT_PROXY.to_vec(), "update VARIANT_PROXY: the effects' shader changed");
    }

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
