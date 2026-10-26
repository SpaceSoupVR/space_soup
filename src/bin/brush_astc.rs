//! BRUSH_ASTC: encode the game's brush materials as ASTC, ahead, so the
//! headset uploads a level's material arrays as they are -- 8 bits a texel
//! against 32 -- instead of building every layer's mips at load
//! (`BrushMaterials::from_astc`, `XrRenderer::set_brush_materials_from`).
//!
//! ```text
//! brush_astc <game folder | material folder>... [--astcenc PATH] [--quality thorough|medium|fast] [--force]
//! ```
//!
//! A game folder means every material in its `materials/`. Each map a
//! material has is written to `<material>/astc/<map>_<w>x<h>.ssastc` at
//! every colour size its library holds -- a scene's arrays take their first
//! material's size and resample the rest to it -- from the images as the
//! game loads them (`brush_astc::load_material_maps`), every mip made by
//! `brush_pipeline::brush_layer_mips`, the filtering the RGBA upload uses,
//! then encoded by ARM's `astcenc` in 4x4 blocks: colour as sRGB, the rest
//! linear. A map with nothing to encode (absent, so one colour) is skipped.
//! A material whose sources -- and the game's `material_overrides.json` --
//! are older than its encodings is skipped unless `--force`. Dev tooling:
//! the game ships the `.ssastc` files, and nothing on the headset needs
//! `astcenc`.

use space_soup::renderer::brush_astc::{brush_astc_path, load_material_maps, solid_colour, source_fingerprint, BrushAstc};
use space_soup::renderer::brush_pipeline::{brush_layer_mips, BrushMap};
use space_soup::renderer::terrain_pipeline::{astcenc_level, AstcArray, ASTC_BLOCK};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

fn newest_source(material: &Path, game: &Path) -> Option<SystemTime> {
    let own = std::fs::read_dir(material)
        .ok()?
        .flatten()
        .filter(|e| e.path().extension().is_some_and(|x| x == "jpg" || x == "png" || x == "json"))
        .filter_map(|e| e.metadata().ok()?.modified().ok());
    let overrides = std::fs::metadata(game.join("material_overrides.json")).and_then(|m| m.modified()).ok();
    own.chain(overrides).max()
}

fn fresh(path: &Path, source: Option<SystemTime>) -> bool {
    let Some(source) = source else { return false };
    std::fs::metadata(path).and_then(|m| m.modified()).is_ok_and(|t| t >= source)
}

/// Every colour size in the library a material sits in, that ASTC blocks
/// tile: the sizes a scene's arrays can take.
fn library_sizes(library: &Path) -> Vec<(u32, u32)> {
    let mut sizes: Vec<(u32, u32)> = std::fs::read_dir(library)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| image::image_dimensions(e.path().join("color.jpg")).ok())
        .filter(|(w, h)| w % ASTC_BLOCK[0] as u32 == 0 && h % ASTC_BLOCK[1] as u32 == 0)
        .collect();
    sizes.sort();
    sizes.dedup();
    sizes
}

fn encode_material(material: &Path, astcenc: &Path, quality: &str, force: bool) -> Result<(), String> {
    let (Some(library), Some(id)) = (material.parent(), material.file_name().map(|n| n.to_string_lossy().into_owned())) else {
        return Err(format!("{} is not a material folder", material.display()));
    };
    let game = library.parent().unwrap_or(library);
    let sizes = library_sizes(library);
    if sizes.is_empty() {
        return Err(format!("{}: no colour map whose size is whole 4x4 blocks", library.display()));
    }
    let source = newest_source(material, game);
    let maps = load_material_maps(game, &id);
    let normal = maps[BrushMap::Normal as usize].as_ref();
    let todo: Vec<(BrushMap, (u32, u32))> = BrushMap::ALL
        .into_iter()
        .filter(|map| solid_colour(*map, maps[*map as usize].as_ref(), normal).is_none())
        .flat_map(|map| sizes.iter().map(move |s| (map, *s)))
        .filter(|(map, size)| force || !fresh(&brush_astc_path(material, *map, *size), source))
        .collect();
    if todo.is_empty() {
        println!("{id}: up to date");
        return Ok(());
    }
    std::fs::create_dir_all(material.join("astc")).map_err(|e| e.to_string())?;
    let scratch = std::env::temp_dir().join(format!("brush_astc_{}", std::process::id()));
    std::fs::create_dir_all(&scratch).map_err(|e| e.to_string())?;
    for (map, (w, h)) in todo {
        let started = std::time::Instant::now();
        let image = maps[map as usize].as_ref();
        let chain = brush_layer_mips(map, image, normal, w, h);
        let data = chain.iter().map(|l| astcenc_level(astcenc, quality, map.srgb(), &l.rgba, l.width, l.height, &scratch)).collect::<Result<Vec<_>, _>>()?;
        let file = BrushAstc { source: source_fingerprint(map, image, normal), array: AstcArray { block: ASTC_BLOCK, srgb: map.srgb(), width: w, height: h, levels: chain.len() as u32, data: vec![data] } };
        let bytes = file.to_bytes();
        let path = brush_astc_path(material, map, (w, h));
        // The reader is the check: what is written must read back as written.
        if BrushAstc::from_bytes(&bytes).as_ref() != Ok(&file) {
            return Err(format!("{} does not read back", path.display()));
        }
        std::fs::write(&path, &bytes).map_err(|e| format!("write {}: {e}", path.display()))?;
        let rgba: usize = chain.iter().map(|l| l.rgba.len()).sum();
        println!("{}: {w}x{h}, {} mips, {:.2} MB ({:.2} MB as RGBA8), {:.1} s", path.display(), chain.len(), bytes.len() as f64 / 1e6, rgba as f64 / 1e6, started.elapsed().as_secs_f32());
    }
    let _ = std::fs::remove_dir_all(&scratch);
    Ok(())
}

fn main() {
    let mut materials: Vec<PathBuf> = Vec::new();
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
                let library = path.join("materials");
                if library.is_dir() {
                    let mut all: Vec<PathBuf> = std::fs::read_dir(&library).into_iter().flatten().flatten().map(|e| e.path()).filter(|p| p.is_dir()).collect();
                    all.sort();
                    materials.extend(all);
                } else {
                    materials.push(path);
                }
            }
        }
    }
    if materials.is_empty() {
        eprintln!("usage: brush_astc <game folder | material folder>... [--astcenc PATH] [--quality thorough|medium|fast] [--force]");
        std::process::exit(2);
    }
    let mut failed = false;
    for material in &materials {
        if let Err(e) = encode_material(material, &astcenc, &quality, force) {
            eprintln!("brush_astc: {e}");
            failed = true;
        }
    }
    if failed {
        std::process::exit(1);
    }
}
