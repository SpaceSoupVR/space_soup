//! `MODEL_ASTC`: encode the game's models' material maps as ASTC, ahead, so
//! the headset uploads them as they are -- 8 bits a texel against 32 -- with
//! no mips built at load (`mesh::texture`'s `astc_texture`).
//!
//! ```text
//! model_astc <game folder | model .glb | folder>... [--astcenc PATH] [--quality thorough|medium|fast] [--force]
//! ```
//!
//! A game folder means every `.glb` under its `models/`. Each material's base
//! colour and emissive maps, as the game loads them (`mesh::material_images`),
//! with every mip the RGBA upload builds (`mesh::model_mips`), are encoded by
//! ARM's `astcenc` in 4x4 sRGB blocks to `<model>.astc/<material>_<map>.ssastc`
//! (`mesh::model_astc_path`), each with the fingerprint of the images it was
//! made from. A map of one texel (a factor, no texture) or of a size that is
//! not whole blocks is left to the RGBA path. An encoding whose fingerprint
//! is the images' is skipped unless `--force`. Dev tooling: the game ships
//! the `.ssastc` files, and nothing on the headset needs `astcenc`.

use space_soup::renderer::brush_astc::BrushAstc;
use space_soup::renderer::mesh::{material_images, model_astc_dir, model_astc_path, model_image_fingerprint, model_mips, ModelMap};
use space_soup::renderer::terrain_pipeline::{astcenc_level, AstcArray, ASTC_BLOCK};
use std::path::{Path, PathBuf};

fn up_to_date(path: &Path, source: u64) -> bool {
    std::fs::read(path).ok().and_then(|b| BrushAstc::from_bytes(&b).ok()).is_some_and(|e| e.source == source)
}

fn encode_model(glb: &Path, astcenc: &Path, quality: &str, force: bool) -> Result<(), String> {
    // A model the importer refuses is one the game cannot load either: noted,
    // not a failure, so one such file does not stop a deploy's encoding.
    let (doc, _buffers, images) = match gltf::import(glb) {
        Ok(i) => i,
        Err(e) => {
            println!("{}: skipped, it does not import ({e})", glb.display());
            return Ok(());
        }
    };
    let dir = model_astc_dir(glb);
    let scratch = std::env::temp_dir().join(format!("model_astc_{}", std::process::id()));
    std::fs::create_dir_all(&scratch).map_err(|e| e.to_string())?;
    let mut seen: Vec<Option<usize>> = Vec::new();
    let (mut wrote, mut kept) = (0, 0);
    for mesh in doc.meshes() {
        for prim in mesh.primitives() {
            let material = prim.material();
            if seen.contains(&material.index()) {
                continue;
            }
            seen.push(material.index());
            let maps = material_images(&material, &images);
            for (map, px) in [(ModelMap::Base, &maps.base), (ModelMap::Emissive, &maps.emissive)] {
                let (w, h) = (px.1, px.2);
                if w * h <= 1 || w % u32::from(ASTC_BLOCK[0]) != 0 || h % u32::from(ASTC_BLOCK[1]) != 0 {
                    continue;
                }
                let path = model_astc_path(&dir, material.index(), map);
                let source = model_image_fingerprint(map, px);
                if !force && up_to_date(&path, source) {
                    kept += 1;
                    continue;
                }
                let started = std::time::Instant::now();
                let chain = model_mips(px);
                let data = chain.iter().map(|l| astcenc_level(astcenc, quality, true, &l.rgba, l.width, l.height, &scratch)).collect::<Result<Vec<_>, _>>()?;
                let file = BrushAstc { source, array: AstcArray { block: ASTC_BLOCK, srgb: true, width: w, height: h, levels: chain.len() as u32, data: vec![data] } };
                let bytes = file.to_bytes();
                // The reader is the check: what is written must read back as written.
                if BrushAstc::from_bytes(&bytes).as_ref() != Ok(&file) {
                    return Err(format!("{} does not read back", path.display()));
                }
                std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
                std::fs::write(&path, &bytes).map_err(|e| format!("write {}: {e}", path.display()))?;
                let rgba: usize = chain.iter().map(|l| l.rgba.len()).sum();
                println!("{}: {w}x{h}, {} mips, {:.2} MB ({:.2} MB as RGBA8), {:.1} s", path.display(), chain.len(), bytes.len() as f64 / 1e6, rgba as f64 / 1e6, started.elapsed().as_secs_f32());
                wrote += 1;
            }
        }
    }
    let _ = std::fs::remove_dir_all(&scratch);
    if wrote == 0 {
        println!("{}: up to date ({kept} encoding(s))", glb.display());
    }
    Ok(())
}

fn glbs_under(dir: &Path, out: &mut Vec<PathBuf>) {
    let mut entries: Vec<PathBuf> = std::fs::read_dir(dir).into_iter().flatten().flatten().map(|e| e.path()).collect();
    entries.sort();
    for p in entries {
        if p.is_dir() {
            if p.extension().is_none_or(|x| x != "astc") {
                glbs_under(&p, out);
            }
        } else if p.extension().is_some_and(|x| x == "glb") {
            out.push(p);
        }
    }
}

fn main() {
    let mut models: Vec<PathBuf> = Vec::new();
    let mut astcenc = PathBuf::from("astcenc");
    let mut quality = "thorough".to_string();
    let mut force = false;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--astcenc" => astcenc = args.next().map(PathBuf::from).expect("--astcenc PATH"),
            "--quality" => quality = args.next().expect("--quality thorough|medium|fast"),
            "--force" => force = true,
            _ => {
                let path = PathBuf::from(a);
                if path.join("models").is_dir() {
                    glbs_under(&path.join("models"), &mut models);
                } else if path.is_dir() {
                    glbs_under(&path, &mut models);
                } else {
                    models.push(path);
                }
            }
        }
    }
    if models.is_empty() {
        eprintln!("usage: model_astc <game folder | model .glb | folder>... [--astcenc PATH] [--quality thorough|medium|fast] [--force]");
        std::process::exit(2);
    }
    let mut failed = false;
    for glb in &models {
        if let Err(e) = encode_model(glb, &astcenc, &quality, force) {
            eprintln!("model_astc: {e}");
            failed = true;
        }
    }
    if failed {
        std::process::exit(1);
    }
}
