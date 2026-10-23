//! TERRAIN_ASTC: encode a ground texture set's layer arrays as ASTC, ahead,
//! so the headset uploads them as they are -- 8 bits a texel against 32 --
//! instead of decoding and mipping 16 JPEGs at load
//! (`TerrainLayerArrays::from_astc`).
//!
//! ```text
//! terrain_astc <texture set folder>... [--astcenc PATH] [--quality thorough|medium|fast] [--force]
//! ```
//!
//! Each set's four maps are written to `<set>/astc/<map>.ssastc`. Every layer
//! and mip is made by `terrain_pipeline::layer_mips`, the filtering the RGBA
//! upload uses (sRGB-correct colour mips, normal-variance roughness), then
//! encoded by ARM's `astcenc` in 4x4 blocks: colour as sRGB, the rest
//! linear. A set whose sources are older than its encoding is skipped
//! unless `--force`. Dev tooling: the game ships the `.ssastc` files, and
//! nothing on the headset needs `astcenc`.

use space_soup::renderer::terrain_pipeline::{astc_path, astcenc_level, filled_layers, layer_mips, load_terrain_ao, load_terrain_layers, load_terrain_normals, load_terrain_rough, AstcArray, LayerMap, ASTC_BLOCK};
use std::path::{Path, PathBuf};

const BLOCK: [u8; 2] = ASTC_BLOCK;

fn newest_source(set: &Path) -> Option<std::time::SystemTime> {
    std::fs::read_dir(set)
        .ok()?
        .flatten()
        .filter(|e| e.path().extension().is_some_and(|x| x == "jpg" || x == "png" || x == "json"))
        .filter_map(|e| e.metadata().ok()?.modified().ok())
        .max()
}

fn stale(set: &Path) -> bool {
    let Some(source) = newest_source(set) else { return true };
    LayerMap::ALL.iter().any(|m| std::fs::metadata(astc_path(set, *m)).and_then(|m| m.modified()).map_or(true, |t| t < source))
}

fn encode_set(set: &Path, astcenc: &Path, quality: &str) -> Result<(), String> {
    let layers = load_terrain_layers(set);
    if layers.iter().all(|l| l.is_none()) {
        return Err(format!("{} has no layer colours ({})", set.display(), space_soup::renderer::terrain_pipeline::TERRAIN_LAYER_FILES.join(", ")));
    }
    let (normals, rough, ao) = (load_terrain_normals(set), load_terrain_rough(set), load_terrain_ao(set));
    let filled = filled_layers(&layers);
    let (w, h) = (filled[0].width, filled[0].height);
    if w % BLOCK[0] as u32 != 0 || h % BLOCK[1] as u32 != 0 {
        return Err(format!("{}: {w}x{h} is not a whole number of {}x{} blocks", set.display(), BLOCK[0], BLOCK[1]));
    }
    std::fs::create_dir_all(set.join("astc")).map_err(|e| e.to_string())?;
    let scratch = std::env::temp_dir().join(format!("terrain_astc_{}", std::process::id()));
    std::fs::create_dir_all(&scratch).map_err(|e| e.to_string())?;
    for map in LayerMap::ALL {
        let started = std::time::Instant::now();
        let mut data = Vec::with_capacity(4);
        let mut levels = 0;
        for slot in 0..4 {
            let chain = layer_mips(map, slot, &filled, &normals, &rough, &ao);
            levels = chain.len() as u32;
            data.push(chain.iter().map(|(rgba, lw, lh)| astcenc_level(astcenc, quality, map.srgb(), rgba, *lw, *lh, &scratch)).collect::<Result<Vec<_>, _>>()?);
        }
        let array = AstcArray { block: BLOCK, srgb: map.srgb(), width: w, height: h, levels, data };
        let bytes = array.to_bytes();
        let path = astc_path(set, map);
        // The reader is the check: what is written must read back as written.
        if AstcArray::from_bytes(&bytes).as_ref() != Ok(&array) {
            return Err(format!("{} does not read back", path.display()));
        }
        std::fs::write(&path, &bytes).map_err(|e| format!("write {}: {e}", path.display()))?;
        println!("{}: {w}x{h}, 4 layers, {levels} mips, {:.1} MB ({:.1} MB as RGBA8), {:.1} s", path.display(), bytes.len() as f64 / 1e6, (w * h * 4 * 4) as f64 * 4.0 / 3.0 / 1e6, started.elapsed().as_secs_f32());
    }
    let _ = std::fs::remove_dir_all(&scratch);
    Ok(())
}

fn main() {
    let mut sets: Vec<PathBuf> = Vec::new();
    let mut astcenc = PathBuf::from("astcenc");
    let mut quality = "thorough".to_string();
    let mut force = false;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--astcenc" => astcenc = args.next().map(PathBuf::from).expect("--astcenc PATH"),
            "--quality" => quality = args.next().expect("--quality thorough|medium|fast"),
            "--force" => force = true,
            _ => sets.push(PathBuf::from(a)),
        }
    }
    if sets.is_empty() {
        eprintln!("usage: terrain_astc <texture set folder>... [--astcenc PATH] [--quality thorough|medium|fast] [--force]");
        std::process::exit(2);
    }
    let mut failed = false;
    for set in &sets {
        if !force && !stale(set) {
            println!("{}: up to date", set.display());
            continue;
        }
        if let Err(e) = encode_set(set, &astcenc, &quality) {
            eprintln!("terrain_astc: {e}");
            failed = true;
        }
    }
    if failed {
        std::process::exit(1);
    }
}
