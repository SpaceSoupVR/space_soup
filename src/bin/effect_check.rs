//! THE ENGINE'S EFFECT VALIDATOR, for the editor's backend
//! (`scene_editor_web/effect_check.py`): the same `EffectAsset::check` and
//! compiler a level load runs, answering in JSON
//! (`space_soup::renderer::effects::authoring`).
//!
//! - `effect_check serve`: one JSON request a line on stdin, one JSON answer
//!   a line on stdout, until stdin closes (`authoring::answer`: ops `check`,
//!   `books`, `templates`, `limits`, `scene`, `ping`). The books' cutouts
//!   are worked out once, at the first request that needs them.
//! - `effect_check check <file.json>`: one asset's verdict; exit 1 if refused.
//! - `effect_check books | templates | limits`: as `serve` answers them.
//!
//! Build: `cargo build --release --bin effect_check` in space_soup.
use space_soup::renderer::effects::authoring;
use std::io::{BufRead, Write};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("serve") => {
            let stdin = std::io::stdin();
            let mut out = std::io::stdout().lock();
            for line in stdin.lock().lines() {
                let Ok(line) = line else { break };
                if line.trim().is_empty() {
                    continue;
                }
                let answer = match serde_json::from_str::<serde_json::Value>(&line) {
                    Ok(request) => authoring::answer(&request),
                    Err(e) => serde_json::json!({ "error": format!("not JSON: {e}") }),
                };
                if writeln!(out, "{answer}").and_then(|_| out.flush()).is_err() {
                    break;
                }
            }
        }
        Some("check") => {
            let path = args.get(1).expect("effect_check check <file.json>");
            let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path}: {e}"));
            let v = authoring::check_json(&text);
            println!("{}", serde_json::to_string_pretty(&v).unwrap());
            std::process::exit(if v["ok"] == true { 0 } else { 1 });
        }
        Some(op @ ("books" | "templates" | "limits")) => {
            println!("{}", serde_json::to_string_pretty(&authoring::answer(&serde_json::json!({ "op": op }))).unwrap());
        }
        _ => {
            eprintln!("usage: effect_check serve | check <file.json> | books | templates | limits");
            std::process::exit(2);
        }
    }
}
