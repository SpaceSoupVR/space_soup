use anyhow::{Context, Result};
use glam::{Mat4, Vec3};
use std::io::Write;
use std::path::Path;

use crate::renderer::{Color3, Cuboid};

struct Triangle {
    v: [Vec3; 3],
    color: [f32; 4],
}

pub fn gltf_to_cuboid_glb(input: &Path, output: &Path, voxel_size: f32) -> Result<Vec<Cuboid>> {
    let tris = import_gltf(input)?;
    let cuboids = voxelize(&tris, voxel_size);
    export_glb(&cuboids, output)?;
    log::info!(
        "voxelize: {} cuboids written to {}",
        cuboids.len(),
        output.display()
    );
    Ok(cuboids)
}

pub fn gltf_to_cuboids(input: &Path, voxel_size: f32) -> Result<Vec<Cuboid>> {
    let tris = import_gltf(input)?;
    Ok(voxelize(&tris, voxel_size))
}

pub fn cuboids_to_glb(cuboids: &[Cuboid], output: &Path) -> Result<()> {
    export_glb(cuboids, output)
}

fn import_gltf(path: &Path) -> Result<Vec<Triangle>> {
    let (doc, buffers, _images) =
        gltf::import(path).with_context(|| format!("failed to open {}", path.display()))?;

    let mut tris = Vec::new();

    for scene in doc.scenes() {
        for node in scene.nodes() {
            collect_node(&node, Mat4::IDENTITY, &buffers, &mut tris);
        }
    }

    log::info!("voxelize: imported {} triangles", tris.len());
    Ok(tris)
}

fn collect_node(
    node: &gltf::Node,
    parent: Mat4,
    buffers: &[gltf::buffer::Data],
    out: &mut Vec<Triangle>,
) {
    let local = Mat4::from_cols_array_2d(&node.transform().matrix());
    let world = parent * local;

    if let Some(mesh) = node.mesh() {
        for prim in mesh.primitives() {
            let reader = prim.reader(|buf| Some(&buffers[buf.index()]));

            let positions: Vec<Vec3> = match reader.read_positions() {
                Some(it) => it.map(|p| world.transform_point3(Vec3::from(p))).collect(),
                None => continue,
            };

            let base = prim.material().pbr_metallic_roughness().base_color_factor();

            let colors: Vec<[f32; 4]> = match reader.read_colors(0) {
                Some(c) => c.into_rgba_f32().collect(),
                None => vec![base; positions.len()],
            };

            let indices: Vec<u32> = match reader.read_indices() {
                Some(it) => it.into_u32().collect(),
                None => (0..positions.len() as u32).collect(),
            };

            for chunk in indices.chunks_exact(3) {
                let (i0, i1, i2) = (chunk[0] as usize, chunk[1] as usize, chunk[2] as usize);
                if i0 >= positions.len() || i1 >= positions.len() || i2 >= positions.len() {
                    continue;
                }
                let color = avg_color4(colors[i0], colors[i1], colors[i2]);
                out.push(Triangle {
                    v: [positions[i0], positions[i1], positions[i2]],
                    color,
                });
            }
        }
    }

    for child in node.children() {
        collect_node(&child, world, buffers, out);
    }
}

fn avg_color4(a: [f32; 4], b: [f32; 4], c: [f32; 4]) -> [f32; 4] {
    [
        (a[0] + b[0] + c[0]) / 3.0,
        (a[1] + b[1] + c[1]) / 3.0,
        (a[2] + b[2] + c[2]) / 3.0,
        (a[3] + b[3] + c[3]) / 3.0,
    ]
}

fn linear_to_srgb_u8(v: f32) -> u8 {
    let s = if v <= 0.0031308 {
        v * 12.92
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    };
    (s.clamp(0.0, 1.0) * 255.0) as u8
}

fn voxelize(tris: &[Triangle], voxel_size: f32) -> Vec<Cuboid> {
    if tris.is_empty() {
        return Vec::new();
    }

    let mut min = Vec3::splat(f32::MAX);
    let mut max = Vec3::splat(f32::MIN);
    for t in tris {
        for v in &t.v {
            min = min.min(*v);
            max = max.max(*v);
        }
    }
    min -= Vec3::splat(voxel_size * 0.5);
    max += Vec3::splat(voxel_size * 0.5);

    let nx = ((max.x - min.x) / voxel_size).ceil() as usize + 1;
    let ny = ((max.y - min.y) / voxel_size).ceil() as usize + 1;
    let nz = ((max.z - min.z) / voxel_size).ceil() as usize + 1;

    log::info!("voxelize: grid {}x{}x{}", nx, ny, nz);

    let hs = Vec3::splat(voxel_size * 0.5);
    let mut cuboids = Vec::new();

    for xi in 0..nx {
        for zi in 0..nz {
            let cx = min.x + (xi as f32 + 0.5) * voxel_size;
            let cz = min.z + (zi as f32 + 0.5) * voxel_size;

            let mut hits: Vec<(f32, [f32; 4])> =
                tris.iter().filter_map(|t| ray_tri_y(cx, cz, t)).collect();

            if hits.is_empty() {
                continue;
            }
            hits.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            let mut i = 0;
            while i + 1 < hits.len() {
                let y_enter = hits[i].0;
                let y_exit = hits[i + 1].0;
                let color = hits[i].1;

                let yi_start = ((y_enter - min.y) / voxel_size).floor() as usize;
                let yi_end = ((y_exit - min.y) / voxel_size).ceil() as usize;

                for yi in yi_start..=yi_end.min(ny - 1) {
                    let cy = min.y + (yi as f32 + 0.5) * voxel_size;
                    // Voxel cuboids are solid geometry, not glass — force
                    // opaque regardless of the source mesh's own alpha
                    // (cutout/blend materials, e.g. foliage textures, are
                    // common in source GLTFs and otherwise silently produce
                    // see-through blocks).
                    let c = Color3(
                        linear_to_srgb_u8(color[0]),
                        linear_to_srgb_u8(color[1]),
                        linear_to_srgb_u8(color[2]),
                        255,
                    );
                    cuboids.push(Cuboid::solid(Vec3::new(cx, cy, cz), hs, c));
                }
                i += 2;
            }
        }
    }

    log::info!("voxelize: {} cuboids", cuboids.len());
    cuboids
}

fn ray_tri_y(rx: f32, rz: f32, t: &Triangle) -> Option<(f32, [f32; 4])> {
    let [v0, v1, v2] = t.v;

    let d0x = v1.x - v0.x;
    let d0z = v1.z - v0.z;
    let d1x = v2.x - v0.x;
    let d1z = v2.z - v0.z;
    let dx = rx - v0.x;
    let dz = rz - v0.z;

    let denom = d0x * d1z - d0z * d1x;
    if denom.abs() < 1e-10 {
        return None;
    }

    let inv = 1.0 / denom;
    let u = (dx * d1z - dz * d1x) * inv;
    let v = (d0x * dz - d0z * dx) * inv;

    if u < 0.0 || v < 0.0 || u + v > 1.0 {
        return None;
    }

    let y = v0.y + u * (v1.y - v0.y) + v * (v2.y - v0.y);
    Some((y, t.color))
}

fn export_glb(cuboids: &[Cuboid], path: &Path) -> Result<()> {
    if cuboids.is_empty() {
        anyhow::bail!("no cuboids to export");
    }

    let n = cuboids.len();
    let pos_stride = 8 * 12usize;
    let col_stride = 8 * 16usize;
    let idx_stride = 36 * 2usize;

    let idx_pad = if (idx_stride % 4) != 0 {
        4 - (idx_stride % 4)
    } else {
        0
    };

    let mut bin: Vec<u8> = Vec::with_capacity(n * (pos_stride + col_stride + idx_stride + idx_pad));

    let pos_base = 0usize;
    let col_base = n * pos_stride;
    let idx_base = col_base + n * col_stride;

    for c in cuboids {
        let p = c.position;
        let h = c.half_size;
        let corners = box_corners(p, h);
        for corner in &corners {
            bin.extend_from_slice(&corner.x.to_le_bytes());
            bin.extend_from_slice(&corner.y.to_le_bytes());
            bin.extend_from_slice(&corner.z.to_le_bytes());
        }
    }

    for c in cuboids {
        let col = c.color.to_linear();
        for _ in 0..8 {
            bin.extend_from_slice(&col[0].to_le_bytes());
            bin.extend_from_slice(&col[1].to_le_bytes());
            bin.extend_from_slice(&col[2].to_le_bytes());
            bin.extend_from_slice(&col[3].to_le_bytes());
        }
    }

    for i in 0..n {
        let base = (i * 8) as u16;
        let faces: [[u16; 6]; 6] = [
            [0, 1, 2, 0, 2, 3],
            [5, 4, 7, 5, 7, 6],
            [4, 0, 3, 4, 3, 7],
            [1, 5, 6, 1, 6, 2],
            [3, 2, 6, 3, 6, 7],
            [4, 5, 1, 4, 1, 0],
        ];
        for face in &faces {
            for &idx in face {
                bin.extend_from_slice(&(base + idx).to_le_bytes());
            }
        }
        for _ in 0..idx_pad {
            bin.push(0u8);
        }
    }

    while bin.len() % 4 != 0 {
        bin.push(0u8);
    }
    let bin_len = bin.len() as u32;

    let mut accessors = Vec::new();
    let mut buffer_views = Vec::new();
    let mut meshes = Vec::new();
    let mut nodes = Vec::new();

    for i in 0..n {
        let pos_bv = i * 3;
        let col_bv = i * 3 + 1;
        let idx_bv = i * 3 + 2;
        let pos_acc = i * 3;
        let col_acc = i * 3 + 1;
        let idx_acc = i * 3 + 2;

        let pos_offset = (pos_base + i * pos_stride) as u32;
        let col_offset = (col_base + i * col_stride) as u32;
        let idx_offset = (idx_base + i * (idx_stride + idx_pad)) as u32;

        buffer_views.push(serde_json::json!({
            "buffer": 0, "byteOffset": pos_offset, "byteLength": pos_stride, "target": 34962
        }));
        buffer_views.push(serde_json::json!({
            "buffer": 0, "byteOffset": col_offset, "byteLength": col_stride, "target": 34962
        }));
        buffer_views.push(serde_json::json!({
            "buffer": 0, "byteOffset": idx_offset, "byteLength": idx_stride, "target": 34963
        }));

        let c = &cuboids[i];
        let p = c.position;
        let h = c.half_size;

        accessors.push(serde_json::json!({
            "bufferView": pos_bv, "byteOffset": 0, "componentType": 5126,
            "count": 8, "type": "VEC3",
            "min": [p.x - h.x, p.y - h.y, p.z - h.z],
            "max": [p.x + h.x, p.y + h.y, p.z + h.z]
        }));
        accessors.push(serde_json::json!({
            "bufferView": col_bv, "byteOffset": 0, "componentType": 5126,
            "count": 8, "type": "VEC4"
        }));
        accessors.push(serde_json::json!({
            "bufferView": idx_bv, "byteOffset": 0, "componentType": 5123,
            "count": 36, "type": "SCALAR"
        }));

        meshes.push(serde_json::json!({
            "primitives": [{
                "attributes": { "POSITION": pos_acc, "COLOR_0": col_acc },
                "indices": idx_acc,
                "mode": 4
            }]
        }));

        nodes.push(serde_json::json!({ "mesh": i, "name": format!("cuboid_{i}") }));
    }

    let node_indices: Vec<usize> = (0..n).collect();

    let gltf_json = serde_json::json!({
        "asset": { "version": "2.0", "generator": "space_soup voxelizer" },
        "scene": 0,
        "scenes": [{ "nodes": node_indices }],
        "nodes": nodes,
        "meshes": meshes,
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [{ "byteLength": bin_len }]
    });

    let json_str = gltf_json.to_string();
    let json_bytes = json_str.as_bytes();
    let json_len = json_bytes.len() as u32;
    let json_pad = if json_len % 4 != 0 {
        4 - json_len % 4
    } else {
        0
    };
    let json_chunk_len = json_len + json_pad;

    let total_len = 12 + 8 + json_chunk_len + 8 + bin_len;

    let mut file =
        std::fs::File::create(path).with_context(|| format!("cannot create {}", path.display()))?;

    file.write_all(b"glTF")?;
    file.write_all(&2u32.to_le_bytes())?;
    file.write_all(&total_len.to_le_bytes())?;

    file.write_all(&json_chunk_len.to_le_bytes())?;
    file.write_all(&0x4E4F534Au32.to_le_bytes())?;
    file.write_all(json_bytes)?;
    for _ in 0..json_pad {
        file.write_all(&[0x20])?;
    }

    file.write_all(&bin_len.to_le_bytes())?;
    file.write_all(&0x004E4942u32.to_le_bytes())?;
    file.write_all(&bin)?;

    Ok(())
}

fn box_corners(center: Vec3, half: Vec3) -> [Vec3; 8] {
    let p = center;
    let h = half;
    [
        Vec3::new(p.x - h.x, p.y - h.y, p.z - h.z),
        Vec3::new(p.x + h.x, p.y - h.y, p.z - h.z),
        Vec3::new(p.x + h.x, p.y + h.y, p.z - h.z),
        Vec3::new(p.x - h.x, p.y + h.y, p.z - h.z),
        Vec3::new(p.x - h.x, p.y - h.y, p.z + h.z),
        Vec3::new(p.x + h.x, p.y - h.y, p.z + h.z),
        Vec3::new(p.x + h.x, p.y + h.y, p.z + h.z),
        Vec3::new(p.x - h.x, p.y + h.y, p.z + h.z),
    ]
}
