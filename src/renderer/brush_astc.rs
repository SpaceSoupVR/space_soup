//! BRUSH MATERIALS AS ASTC: a level's static geometry's material arrays
//! uploaded as blocks encoded ahead (`bin/brush_astc`) -- 8 bits a texel
//! against 32, and no mips built on the headset -- where every material in
//! use has an encoding made from the very images the RGBA path would upload
//! ([`BrushMaterials::from_astc`]).
//!
//! An encoding is one map of one material at one array size,
//! `materials/<id>/astc/<map>_<w>x<h>.ssastc` ([`brush_astc_path`]): a
//! scene's arrays take their first colour map's size and resample the rest
//! to it ([`brush_array_size`]), so a library holding two sizes encodes each
//! material at both. Each file carries a fingerprint of the images it was
//! made from ([`source_fingerprint`]). A file whose fingerprint is not that
//! of the images the level loaded is stale whatever its date, and the whole
//! scene takes the RGBA path: never a compressed copy of something else.
//!
//! A layer holding one colour (a map the material does not have, a layer no
//! material uses) needs no file. It is written as ASTC's constant-colour
//! block ([`solid_block`]) at the colour the RGBA path's mips of it hold
//! ([`solid_levels`]), which decodes to exactly that colour.

use super::brush_pipeline::{brush_array_size, brush_layer_mips, BrushMap, BrushMaterials, MAX_BRUSH_MATERIALS};
use super::terrain_pipeline::{mip_levels_for, AstcArray, TerrainImage, ASTC_BLOCK};
use std::path::{Path, PathBuf};
use wgpu::*;

/// The file one map of one material keeps its encoding at `size` in.
pub fn brush_astc_path(material: &Path, map: BrushMap, size: (u32, u32)) -> PathBuf {
    material.join("astc").join(format!("{}_{}x{}.ssastc", map.name(), size.0, size.1))
}

const BRUSH_ASTC_MAGIC: &[u8; 8] = b"SSBRUSH1";

/// Bump when [`brush_layer_mips`] filters differently: every encoding made
/// before then is a compressed copy of mips the RGBA path no longer builds,
/// and folding this into the fingerprint makes them stale.
const BRUSH_FILTER_VERSION: u64 = 1;

/// One map of one material: its array layer's every mip as ASTC blocks
/// (an [`AstcArray`] of one layer), and the fingerprint of the images they
/// were made from.
#[derive(Debug, Clone, PartialEq)]
pub struct BrushAstc {
    pub source: u64,
    pub array: AstcArray,
}

impl BrushAstc {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = BRUSH_ASTC_MAGIC.to_vec();
        out.extend_from_slice(&self.source.to_le_bytes());
        out.extend_from_slice(&self.array.to_bytes());
        out
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        let head = bytes.get(..16).ok_or("too short for its header")?;
        if &head[..8] != BRUSH_ASTC_MAGIC {
            return Err("not an SSBRUSH1 file".into());
        }
        let source = u64::from_le_bytes(head[8..16].try_into().unwrap());
        let array = AstcArray::from_bytes(&bytes[16..])?;
        if array.data.len() != 1 {
            return Err(format!("{} layers, where one material has one", array.data.len()));
        }
        Ok(Self { source, array })
    }
}

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

fn fold_image(h: &mut u64, img: Option<&TerrainImage>) {
    let mut fold = |word: u64| *h = (*h ^ word).wrapping_mul(FNV_PRIME);
    match img {
        None => fold(u64::MAX),
        Some(i) => {
            fold((u64::from(i.width) << 32) | u64::from(i.height));
            for c in i.rgba.chunks(8) {
                let mut word = [0u8; 8];
                word[..c.len()].copy_from_slice(c);
                fold(u64::from_le_bytes(word));
            }
        }
    }
}

/// What one map's layer is made from, as one number: the image as loaded
/// (before it is fitted to the array) and, for roughness, the normal map
/// that widens its mips. The encoder stores it and the loader recomputes it
/// from the images the level loaded, so an encoding is used only for the
/// content it was made from -- a re-exported texture, an edited roughness
/// range or a changed decoder all read as stale. FNV-1a over 8-byte words:
/// a few milliseconds a 1K map, against the mips it saves building.
pub fn source_fingerprint(map: BrushMap, image: Option<&TerrainImage>, normal: Option<&TerrainImage>) -> u64 {
    let mut h = FNV_OFFSET ^ (BRUSH_FILTER_VERSION << 8) ^ map as u64;
    fold_image(&mut h, image);
    if map == BrushMap::Rough {
        fold_image(&mut h, normal);
    }
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
    h ^ (h >> 33)
}

/// The one colour a layer holds when its map is a single colour wherever
/// it is read -- absent (the map's fallback) or one texel (the app's white
/// for a material with no colour map) -- so no encoding is needed. None
/// when the layer needs one. Roughness with a normal map never is: the
/// normals' variance widens its mips texel by texel.
pub fn solid_colour(map: BrushMap, image: Option<&TerrainImage>, normal: Option<&TerrainImage>) -> Option<[u8; 4]> {
    if map == BrushMap::Rough && normal.is_some() {
        return None;
    }
    match image {
        None => {
            let [r, g, b] = map.fallback();
            Some([r, g, b, 255])
        }
        Some(i) if i.width == 1 && i.height == 1 && i.rgba.len() >= 4 => Some([i.rgba[0], i.rgba[1], i.rgba[2], i.rgba[3]]),
        Some(_) => None,
    }
}

/// Each of `levels` mips' colour for a layer of one colour, as the RGBA
/// path's mips hold it: [`brush_layer_mips`] of a one-texel-wide strip,
/// whose every texel averages four of the same samples in the same order
/// as any size's would, so it is that filtering's rounding exactly.
pub fn solid_levels(map: BrushMap, colour: [u8; 4], levels: u32) -> Vec<[u8; 4]> {
    let tall = 1u32 << levels.saturating_sub(1);
    let strip = TerrainImage { width: 1, height: tall, rgba: colour.repeat(tall as usize) };
    brush_layer_mips(map, Some(&strip), None, 1, tall).iter().map(|l| [l.rgba[0], l.rgba[1], l.rgba[2], l.rgba[3]]).collect()
}

/// ASTC's constant-colour ("void extent") block: every texel of it decodes
/// to `rgba`, in sRGB and linear arrays alike. Header 0x1FC with the LDR
/// bit clear and no extent given, then each channel as UNORM16 -- `c` as
/// `c * 257`, whose top byte (what an sRGB decode reads) is `c` and whose
/// value over 65535 is `c / 255`.
pub fn solid_block(rgba: [u8; 4]) -> [u8; 16] {
    let mut block = [0u8; 16];
    block[..8].copy_from_slice(&[0xFC, 0xFD, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF]);
    for (i, c) in rgba.iter().enumerate() {
        block[8 + 2 * i..10 + 2 * i].copy_from_slice(&(u16::from(*c) * 257).to_le_bytes());
    }
    block
}

/// One map's array as blocks, `[layer][level]`, for every layer the arrays
/// have: encoded files where a material has the map, constant-colour blocks
/// where its layer is one colour. `materials` are the layers' folders.
/// Err says why the scene cannot have them -- a missing, unreadable, misfit
/// or stale encoding -- and the scene then takes the RGBA path.
pub fn gather_map(map: BrushMap, materials: &[Option<PathBuf>], images: &[Option<TerrainImage>], normals: &[Option<TerrainImage>], (w, h): (u32, u32)) -> Result<Vec<Vec<Vec<u8>>>, String> {
    let levels = mip_levels_for(w, h);
    (0..MAX_BRUSH_MATERIALS)
        .map(|slot| {
            let image = images.get(slot).and_then(|x| x.as_ref());
            let normal = normals.get(slot).and_then(|x| x.as_ref());
            if let Some(colour) = solid_colour(map, image, normal) {
                return Ok(solid_levels(map, colour, levels)
                    .iter()
                    .enumerate()
                    .map(|(level, c)| solid_block(*c).repeat(AstcArray::level_bytes(ASTC_BLOCK, w, h, level as u32) / 16))
                    .collect());
            }
            let folder = materials.get(slot).and_then(|x| x.as_ref()).ok_or_else(|| format!("layer {slot}'s {} map has no folder", map.name()))?;
            let name = folder.file_name().map_or_else(|| folder.display().to_string(), |n| n.to_string_lossy().into_owned());
            let path = brush_astc_path(folder, map, (w, h));
            let bytes = std::fs::read(&path).map_err(|_| format!("{name} has no {} encoding at {w}x{h}", map.name()))?;
            let file = BrushAstc::from_bytes(&bytes).map_err(|e| format!("{}: {e}", path.display()))?;
            let a = file.array;
            if a.block != ASTC_BLOCK || a.srgb != map.srgb() || (a.width, a.height) != (w, h) || a.levels != levels {
                return Err(format!("{} does not fit a {w}x{h} array", path.display()));
            }
            if file.source != source_fingerprint(map, image, normal) {
                return Err(format!("{name}'s {} encoding is stale (made from other images)", map.name()));
            }
            Ok(a.data.into_iter().next().unwrap_or_default())
        })
        .collect()
}

/// One map's array from [`gather_map`]'s blocks.
pub(crate) fn astc_array_texture(device: &Device, queue: &Queue, map: BrushMap, layers: &[Vec<Vec<u8>>], (w, h): (u32, u32)) -> Texture {
    let channel = if map.srgb() { AstcChannel::UnormSrgb } else { AstcChannel::Unorm };
    let texture = device.create_texture(&TextureDescriptor {
        label: Some(map.label()),
        size: Extent3d { width: w, height: h, depth_or_array_layers: layers.len() as u32 },
        mip_level_count: mip_levels_for(w, h),
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Astc { block: AstcBlock::B4x4, channel },
        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let (bw, bh) = (u32::from(ASTC_BLOCK[0]), u32::from(ASTC_BLOCK[1]));
    for (slot, chain) in layers.iter().enumerate() {
        for (level, data) in chain.iter().enumerate() {
            let (lw, lh) = ((w >> level).max(1), (h >> level).max(1));
            let (cols, rows) = (lw.div_ceil(bw), lh.div_ceil(bh));
            queue.write_texture(
                TexelCopyTextureInfo { texture: &texture, mip_level: level as u32, origin: Origin3d { x: 0, y: 0, z: slot as u32 }, aspect: TextureAspect::All },
                data,
                TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(cols * 16), rows_per_image: Some(rows) },
                Extent3d { width: cols * bw, height: rows * bh, depth_or_array_layers: 1 },
            );
        }
    }
    texture
}

impl BrushMaterials {
    /// The arrays from encodings made ahead, for the same images
    /// [`Self::new`] takes (and `materials`, each layer's folder). Every map
    /// is read and checked before any texture is made, so a refusal leaves
    /// nothing half built; Err says why, and the caller uploads the images.
    #[allow(clippy::too_many_arguments, reason = "GPU setup passes each resource explicitly; bundling them is a separate refactor")]
    pub fn from_astc(
        device: &Device,
        queue: &Queue,
        layout: &BindGroupLayout,
        materials: &[Option<PathBuf>],
        colours: &[Option<TerrainImage>],
        normals: &[Option<TerrainImage>],
        roughs: &[Option<TerrainImage>],
        aos: &[Option<TerrainImage>],
    ) -> Result<Self, String> {
        if !device.features().contains(Features::TEXTURE_COMPRESSION_ASTC) {
            return Err("this GPU reads no ASTC".into());
        }
        let size = brush_array_size(colours);
        if !size.0.is_multiple_of(u32::from(ASTC_BLOCK[0])) || !size.1.is_multiple_of(u32::from(ASTC_BLOCK[1])) {
            return Err(format!("{}x{} is not a whole number of blocks", size.0, size.1));
        }
        let images = |map: BrushMap| match map {
            BrushMap::Colour => colours,
            BrushMap::Normal => normals,
            BrushMap::Rough => roughs,
            BrushMap::Ao => aos,
        };
        let mut blocks = Vec::with_capacity(4);
        for map in BrushMap::ALL {
            blocks.push(gather_map(map, materials, images(map), normals, size)?);
        }
        let textures = BrushMap::ALL.map(|map| astc_array_texture(device, queue, map, &blocks[map as usize], size));
        Ok(Self::with_arrays(device, layout, &textures, colours, roughs))
    }
}

/// One material's maps as the game loads them (`space_soup_app`'s
/// `brush_render::load_materials`): `materials/<id>/{color,normal,rough,ao}.jpg`,
/// a missing colour as one white texel, and the roughness read through the
/// author's range -- `"roughness": {"min", "max"}` in the game's
/// `material_overrides.json` by id, else the material's own
/// `material.json`. The encoder's images: should the game load them
/// otherwise, the fingerprints differ and the RGBA path is taken, not a
/// wrong encoding. In [`BrushMap::ALL`]'s order.
pub fn load_material_maps(game_dir: &Path, id: &str) -> [Option<TerrainImage>; 4] {
    let dir = game_dir.join("materials").join(id);
    let load = |map: BrushMap| {
        let path = dir.join(format!("{}.jpg", map.name()));
        path.exists().then(|| TerrainImage::load(&path)).flatten()
    };
    let colour = load(BrushMap::Colour).unwrap_or(TerrainImage { width: 1, height: 1, rgba: vec![255, 255, 255, 255] });
    let mut rough = load(BrushMap::Rough);
    if let (Some(img), Some((lo, hi))) = (rough.as_mut(), roughness_range(game_dir, id)) {
        for px in img.rgba.chunks_exact_mut(4) {
            for c in &mut px[..3] {
                let r = lo + (f32::from(*c) / 255.0) * (hi - lo);
                *c = (r.clamp(0.0, 1.0) * 255.0).round() as u8;
            }
        }
    }
    [Some(colour), load(BrushMap::Normal), rough, load(BrushMap::Ao)]
}

/// What material `id`'s roughness map's 0 and 1 stand for, read as the game
/// reads it (`space_soup_app`'s `brush_render::roughness_range`).
pub fn roughness_range(game_dir: &Path, id: &str) -> Option<(f32, f32)> {
    let read = |path: PathBuf| -> Option<serde_json::Value> { serde_json::from_str(&std::fs::read_to_string(path).ok()?).ok() };
    let range = |v: &serde_json::Value| -> Option<(f32, f32)> {
        let r = v.get("roughness")?;
        let lo = r.get("min")?.as_f64()? as f32;
        let hi = r.get("max")?.as_f64()? as f32;
        (lo.is_finite() && hi.is_finite()).then_some((lo.clamp(0.0, 1.0), hi.clamp(0.0, 1.0)))
    };
    read(game_dir.join("material_overrides.json"))
        .and_then(|o| o.get(id).and_then(range))
        .or_else(|| read(game_dir.join("materials").join(id).join("material.json")).and_then(|m| range(&m)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::renderer::brush_pipeline::brush_material_bind_group_layout;

    fn image(w: u32, h: u32, seed: u8) -> TerrainImage {
        let rgba = (0..w * h).flat_map(|i| [(i as u8).wrapping_mul(7) ^ seed, (i as u8).wrapping_mul(13), seed, 255]).collect();
        TerrainImage { width: w, height: h, rgba }
    }

    fn encoded(map: BrushMap, image: Option<&TerrainImage>, normal: Option<&TerrainImage>, (w, h): (u32, u32)) -> BrushAstc {
        let levels = mip_levels_for(w, h);
        let data = vec![(0..levels).map(|l| solid_block([l as u8, 0, 0, 255]).repeat(AstcArray::level_bytes(ASTC_BLOCK, w, h, l) / 16)).collect()];
        BrushAstc { source: source_fingerprint(map, image, normal), array: AstcArray { block: ASTC_BLOCK, srgb: map.srgb(), width: w, height: h, levels, data } }
    }

    fn write(folder: &Path, map: BrushMap, file: &BrushAstc) {
        let path = brush_astc_path(folder, map, (file.array.width, file.array.height));
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, file.to_bytes()).unwrap();
    }

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("ss_brush_astc_{name}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn an_encoding_reads_back_as_written_and_a_wrong_one_is_refused() {
        let img = image(16, 8, 3);
        let file = encoded(BrushMap::Colour, Some(&img), None, (16, 8));
        assert_eq!(BrushAstc::from_bytes(&file.to_bytes()), Ok(file.clone()));
        let mut bytes = file.to_bytes();
        bytes[0] = b'X';
        assert!(BrushAstc::from_bytes(&bytes).is_err(), "not the brush magic");
        assert!(BrushAstc::from_bytes(&file.array.to_bytes()).is_err(), "a terrain array is not a brush map");
        let mut two = file.clone();
        two.array.data.push(two.array.data[0].clone());
        assert!(BrushAstc::from_bytes(&two.to_bytes()).is_err(), "two layers");
        let mut cut = file.to_bytes();
        cut.truncate(cut.len() - 5);
        assert!(BrushAstc::from_bytes(&cut).is_err(), "cut short");
    }

    #[test]
    fn the_fingerprint_follows_the_images_and_roughness_follows_its_normals() {
        let (a, n) = (image(8, 8, 1), image(8, 8, 2));
        let base = source_fingerprint(BrushMap::Colour, Some(&a), None);
        assert_eq!(base, source_fingerprint(BrushMap::Colour, Some(&a.clone()), None));
        let mut b = a.clone();
        b.rgba[17] ^= 1;
        assert_ne!(base, source_fingerprint(BrushMap::Colour, Some(&b), None), "one bit of one texel");
        assert_ne!(base, source_fingerprint(BrushMap::Normal, Some(&a), None), "the same image as another map");
        assert_eq!(source_fingerprint(BrushMap::Colour, Some(&a), Some(&n)), base, "colour reads no normals");
        assert_ne!(source_fingerprint(BrushMap::Rough, Some(&a), Some(&n)), source_fingerprint(BrushMap::Rough, Some(&a), None), "roughness does");
    }

    /// A layer of one colour is written as constant-colour blocks with no
    /// file; that is exact only if the RGBA path's mips of it hold that
    /// colour level by level, which this checks against the real filtering.
    #[test]
    fn a_solid_layer_is_the_rgba_paths_mips_of_it() {
        let (w, h) = (16, 8);
        let levels = mip_levels_for(w, h);
        for map in BrushMap::ALL {
            for c in [0u8, 1, 37, 128, 140, 200, 254, 255] {
                let one = TerrainImage { width: 1, height: 1, rgba: vec![c, c / 2, 255 - c, 255] };
                for img in [None, Some(&one)] {
                    let colour = solid_colour(map, img, None).expect("one colour");
                    let rgba = brush_layer_mips(map, img, None, w, h);
                    assert_eq!(rgba.len() as u32, levels);
                    for (level, want) in solid_levels(map, colour, levels).iter().enumerate() {
                        assert!(rgba[level].rgba.chunks(4).all(|t| t == want), "{map:?} level {level}: {:?} vs {want:?}", &rgba[level].rgba[..4]);
                    }
                }
            }
        }
        assert_eq!(solid_colour(BrushMap::Normal, None, None), Some([128, 128, 255, 255]));
        assert_eq!(solid_colour(BrushMap::Rough, None, Some(&image(4, 4, 0))), None, "a normal map widens roughness texel by texel");
        assert_eq!(solid_colour(BrushMap::Colour, Some(&image(4, 4, 0)), None), None);
    }

    #[test]
    fn a_scene_takes_the_encodings_only_when_every_material_has_fresh_ones() {
        let lib = scratch("choice");
        let (a, b) = (lib.join("A"), lib.join("B"));
        let size = (16, 8);
        let colour_a = image(16, 8, 1);
        let colour_b = image(16, 16, 2);
        let normal_a = image(16, 8, 9);
        let folders = vec![Some(a.clone()), Some(b.clone()), Some(lib.join("default"))];
        let white = TerrainImage { width: 1, height: 1, rgba: vec![255; 4] };
        let colours = vec![Some(colour_a.clone()), Some(colour_b.clone()), Some(white)];
        let none = vec![None, None, None];

        let err = gather_map(BrushMap::Colour, &folders, &colours, &none, size).unwrap_err();
        assert!(err.contains("A has no color encoding at 16x8"), "{err}");
        write(&a, BrushMap::Colour, &encoded(BrushMap::Colour, Some(&colour_a), None, size));
        let err = gather_map(BrushMap::Colour, &folders, &colours, &none, size).unwrap_err();
        assert!(err.contains("B has no color encoding"), "{err}");
        // B's own size is not the array's: it needs one made at A's.
        write(&b, BrushMap::Colour, &encoded(BrushMap::Colour, Some(&colour_b), None, (16, 16)));
        assert!(gather_map(BrushMap::Colour, &folders, &colours, &none, size).is_err());
        write(&b, BrushMap::Colour, &encoded(BrushMap::Colour, Some(&colour_b), None, size));
        let layers = gather_map(BrushMap::Colour, &folders, &colours, &none, size).expect("every colour encoded");
        assert_eq!(layers.len(), MAX_BRUSH_MATERIALS, "every layer, the unused ones as one colour");
        assert_eq!(layers[2][0][..16], solid_block([255; 4]), "the default material's white needs no file");
        // No normals anywhere: every layer one colour, nothing read.
        assert!(gather_map(BrushMap::Normal, &folders, &none, &none, size).is_ok());
        assert!(gather_map(BrushMap::Rough, &folders, &none, &none, size).is_ok());

        // A normal map: its own encoding, and roughness's, which it widens.
        let normals = vec![Some(normal_a.clone()), None, None];
        assert!(gather_map(BrushMap::Normal, &folders, &normals, &normals, size).is_err());
        assert!(gather_map(BrushMap::Rough, &folders, &none, &normals, size).is_err());
        write(&a, BrushMap::Normal, &encoded(BrushMap::Normal, Some(&normal_a), None, size));
        write(&a, BrushMap::Rough, &encoded(BrushMap::Rough, None, Some(&normal_a), size));
        assert!(gather_map(BrushMap::Normal, &folders, &normals, &normals, size).is_ok());
        assert!(gather_map(BrushMap::Rough, &folders, &none, &normals, size).is_ok());

        // The images the level loaded moved on: stale, whatever the dates.
        let mut edited = colours.clone();
        edited[1].as_mut().unwrap().rgba[40] ^= 4;
        let err = gather_map(BrushMap::Colour, &folders, &edited, &none, size).unwrap_err();
        assert!(err.contains("B's color encoding is stale"), "{err}");
        // A file that is not an encoding.
        std::fs::write(brush_astc_path(&a, BrushMap::Colour, size), b"junk").unwrap();
        assert!(gather_map(BrushMap::Colour, &folders, &colours, &none, size).is_err());
        let _ = std::fs::remove_dir_all(&lib);
    }

    fn astc_device() -> Option<(Device, Queue)> {
        let instance = Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&RequestAdapterOptions::default())).ok()?;
        if !adapter.features().contains(Features::TEXTURE_COMPRESSION_ASTC) {
            eprintln!("this GPU reads no ASTC: skipped");
            return None;
        }
        pollster::block_on(adapter.request_device(&DeviceDescriptor { required_features: Features::TEXTURE_COMPRESSION_ASTC, ..Default::default() })).ok()
    }

    /// Every texel of one layer's one level as the GPU reads it (linear for
    /// an sRGB array), by a compute pass's textureLoad.
    fn read_layer(device: &Device, queue: &Queue, texture: &Texture, layer: u32, level: u32) -> Vec<[f32; 4]> {
        let size = texture.size();
        let (w, h) = ((size.width >> level).max(1), (size.height >> level).max(1));
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("brush_astc_read"),
            source: ShaderSource::Wgsl(
                "@group(0) @binding(0) var t: texture_2d_array<f32>;
                 @group(0) @binding(1) var<storage, read_write> out: array<vec4<f32>>;
                 @group(0) @binding(2) var<uniform> p: vec4<u32>;
                 @compute @workgroup_size(8, 8) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                     if (id.x >= p.z || id.y >= p.w) { return; }
                     out[id.y * p.z + id.x] = textureLoad(t, vec2<i32>(id.xy), i32(p.x), i32(p.y));
                 }"
                .into(),
            ),
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor { label: None, layout: None, module: &module, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
        let bytes = u64::from(w * h * 16);
        let out = device.create_buffer(&BufferDescriptor { label: None, size: bytes, usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC, mapped_at_creation: false });
        let back = device.create_buffer(&BufferDescriptor { label: None, size: bytes, usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST, mapped_at_creation: false });
        let params = wgpu::util::DeviceExt::create_buffer_init(device, &wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&[layer, level, w, h]), usage: BufferUsages::UNIFORM });
        let view = texture.create_view(&TextureViewDescriptor { dimension: Some(TextureViewDimension::D2Array), ..Default::default() });
        let group = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry { binding: 0, resource: BindingResource::TextureView(&view) },
                BindGroupEntry { binding: 1, resource: out.as_entire_binding() },
                BindGroupEntry { binding: 2, resource: params.as_entire_binding() },
            ],
        });
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(w.div_ceil(8), h.div_ceil(8), 1);
        }
        encoder.copy_buffer_to_buffer(&out, 0, &back, 0, bytes);
        queue.submit(Some(encoder.finish()));
        let slice = back.slice(..);
        slice.map_async(MapMode::Read, |_| {});
        device.poll(PollType::Wait { submission_index: None, timeout: None }).ok();
        let data = slice.get_mapped_range().unwrap();
        bytemuck::cast_slice::<u8, [f32; 4]>(&data).to_vec()
    }

    /// The upload is the risk: a wrong row pitch or copy size for a block
    /// format is a validation error, and on the headset those are silent.
    #[test]
    fn an_encoded_brush_set_uploads_with_no_validation_error_where_the_gpu_reads_astc() {
        let lib = scratch("upload");
        let a = lib.join("A");
        let size = (64, 32);
        let (colour, normal, rough) = (image(64, 32, 1), image(64, 32, 2), image(64, 32, 3));
        write(&a, BrushMap::Colour, &encoded(BrushMap::Colour, Some(&colour), None, size));
        write(&a, BrushMap::Normal, &encoded(BrushMap::Normal, Some(&normal), None, size));
        write(&a, BrushMap::Rough, &encoded(BrushMap::Rough, Some(&rough), Some(&normal), size));
        let folders = vec![Some(a.clone())];
        let (colours, normals, roughs, aos) = (vec![Some(colour.clone())], vec![Some(normal)], vec![Some(rough)], vec![None]);

        let instance = Instance::default();
        let Ok(adapter) = pollster::block_on(instance.request_adapter(&RequestAdapterOptions::default())) else {
            eprintln!("no GPU: skipped");
            return;
        };
        let (plain, plain_queue) = pollster::block_on(adapter.request_device(&DeviceDescriptor::default())).unwrap();
        let layout = brush_material_bind_group_layout(&plain);
        assert!(BrushMaterials::from_astc(&plain, &plain_queue, &layout, &folders, &colours, &normals, &roughs, &aos).is_err(), "no ASTC asked of the device, none used");
        let Some((device, queue)) = astc_device() else {
            let _ = std::fs::remove_dir_all(&lib);
            return;
        };
        let layout = brush_material_bind_group_layout(&device);
        let scope = device.push_error_scope(ErrorFilter::Validation);
        let materials = BrushMaterials::from_astc(&device, &queue, &layout, &folders, &colours, &normals, &roughs, &aos).expect("the set loads");
        queue.submit(None);
        let error = pollster::block_on(scope.pop());
        assert!(error.is_none(), "{error:?}");
        assert!(!materials.reflective);
        // A size that is not whole blocks is refused, not padded.
        let odd = vec![Some(image(30, 32, 1))];
        assert!(BrushMaterials::from_astc(&device, &queue, &layout, &folders, &odd, &normals, &roughs, &aos).err().is_some_and(|e| e.contains("whole number of blocks")));
        let _ = std::fs::remove_dir_all(&lib);
    }

    /// The constant-colour block must read as the RGBA layer it stands in
    /// for, in an sRGB array and a linear one, at the top mip and the last:
    /// the same byte, so within half a step of 255. (Apple's GPU decodes a
    /// linear block's UNORM16 at its own precision, up to a twentieth of a
    /// step off c/255; exact on sRGB.)
    #[test]
    fn a_solid_block_reads_as_the_rgba_layer_it_replaces() {
        let Some((device, queue)) = astc_device() else { return };
        let size = (16, 8);
        let one = |c: [u8; 4]| Some(TerrainImage { width: 1, height: 1, rgba: c.to_vec() });
        let images = vec![one([200, 30, 7, 255]), None, one([0, 255, 128, 255]), one([1, 2, 3, 255])];
        let scope = device.push_error_scope(ErrorFilter::Validation);
        for map in [BrushMap::Colour, BrushMap::Normal] {
            let blocks = gather_map(map, &[], &images, &[], size).expect("all one colour");
            let astc = astc_array_texture(&device, &queue, map, &blocks, size);
            let rgba = BrushMaterials::array_texture(&device, &queue, map, &images, &[], size.0, size.1);
            for layer in [0, 1, 2, 3, 20] {
                for level in [0, mip_levels_for(size.0, size.1) - 1] {
                    let (a, b) = (read_layer(&device, &queue, &astc, layer, level), read_layer(&device, &queue, &rgba, layer, level));
                    for (x, y) in a.iter().zip(&b) {
                        assert!(x.iter().zip(y).all(|(p, q)| (p - q).abs() < 0.5 / 255.0), "{map:?} layer {layer} level {level}: ASTC {x:?}, RGBA {y:?}");
                    }
                }
            }
        }
        assert!(pollster::block_on(scope.pop()).is_none());
    }

    /// A real material encoded by `bin/brush_astc` against its RGBA upload:
    /// the difference must be ASTC's own error, not a flipped channel, an
    /// upside-down layer or other images. Set `BRUSH_ASTC_MATERIAL` to an
    /// encoded material's folder (`<game>/materials/<id>`); skipped without.
    #[test]
    fn an_encoded_library_material_reads_as_its_rgba_upload() {
        let Some(folder) = std::env::var_os("BRUSH_ASTC_MATERIAL").map(PathBuf::from) else {
            eprintln!("BRUSH_ASTC_MATERIAL unset: skipped");
            return;
        };
        let Some((device, queue)) = astc_device() else { return };
        let game = folder.parent().and_then(Path::parent).unwrap();
        let id = folder.file_name().unwrap().to_string_lossy().into_owned();
        let [colour, normal, rough, ao] = load_material_maps(game, &id);
        let maps = [vec![colour], vec![normal], vec![rough], vec![ao]];
        let size = brush_array_size(&maps[0]);
        let levels = mip_levels_for(size.0, size.1);
        for map in BrushMap::ALL {
            let images = &maps[map as usize];
            let blocks = gather_map(map, &[Some(folder.clone())], images, &maps[1], size).unwrap_or_else(|e| panic!("{e}"));
            let astc = astc_array_texture(&device, &queue, map, &blocks, size);
            let rgba = BrushMaterials::array_texture(&device, &queue, map, images, &maps[1], size.0, size.1);
            for level in [0, levels / 2, levels - 1] {
                let (a, b) = (read_layer(&device, &queue, &astc, 0, level), read_layer(&device, &queue, &rgba, 0, level));
                let n = (a.len() * 3) as f64;
                let sq: f64 = a.iter().zip(&b).flat_map(|(x, y)| (0..3).map(move |c| f64::from(x[c] - y[c]).powi(2))).sum();
                let psnr = 10.0 * (1.0 / (sq / n).max(1e-12)).log10();
                eprintln!("{id} {} level {level}: PSNR {psnr:.1} dB", map.name());
                assert!(psnr > 30.0, "{id} {} level {level}: PSNR {psnr:.1} dB is not ASTC's error", map.name());
            }
        }
    }
}
