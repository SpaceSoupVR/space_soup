//! THE GROUND'S SHAPE AS A TEXTURE: its step-1 vertex normals and heights,
//! read by the ground's shaders at the pixel's place on the ground, so a
//! COARSER triangle (`ground_lod`'s steps 2 to 16) is lit and textured as the
//! step-1 ground under it is.
//!
//! WHY (`docs/ground-lod-status-2026-10-13.md`, "what is left" 1). The ground
//! was lit by its INTERPOLATED VERTEX NORMALS and textured on the planes its
//! DRAWN TRIANGLE faced (`terrain_pipeline`'s `sample_frame`). A triangle of
//! step 2 spans four of step 1 and interpolates three normals where step 1
//! interpolated nine: within a pixel of the same surface, the light along a
//! plateau's rim still moved by more than 20 levels in 766 pixels of the
//! hilltop of stress_625. Both come from here now, for every triangle of the
//! lattice at any step:
//!
//! - the NORMAL: the step-1 triangle under the pixel, found by the cell and
//!   the cell's own diagonal, and its three vertex normals blended by the
//!   pixel's place in it -- exactly what the rasterizer blends for a step-1
//!   triangle, so step 1 draws the picture it drew;
//! - the FACE: that step-1 triangle's own plane, from its three heights --
//!   what `cross(dpdx, dpdy)` gives on a step-1 triangle.
//!
//! ONE TEXEL A SAMPLE, `Rgba32Float`: the vertex normal in x, y, z and the
//! height in w, whose lowest bit is the diagonal of the cell whose lowest
//! corner the sample is ([`DIAGONAL_BIT`]; a height a part in eight million
//! off). Read by FOUR loads, the cell's corners: blending them is then three
//! multiply-adds a corner. (Packed into 8 bytes, two gathers and an unpack
//! a corner, it was 193 proxy units on every ground reader -- the dry
//! readers 2,047 -> 2,240, the torch's puddle reader past its 2,600 --
//! where this is a third of that.) 16 bytes a sample: 1 MB on test_room,
//! 6.3 MB at 625 samples, 67 MB at 2,049 (its vertex buffer is 218 MB);
//! sectors cut both.
//!
//! WHICH TRIANGLES: the lattice's. A vertex of a cave's ground patch (after the
//! lattice in the vertex buffer) is marked by its colour's alpha, 0, and a
//! triangle touching one keeps its interpolated normal and its drawn face
//! ([`WGSL`]'s `shaped`), as before.

use glam::Vec3;

/// THE GROUND'S READERS TAKE THEIR NORMAL AND FACE FROM THE SHAPE (`true`).
/// `false`: the renderer binds no shape and they shade by the vertices as
/// before 2026-10-14 -- the A/B of its cost (`adb shell setprop
/// debug.spacesoup.groundshape 0` before a launch). The readers carry the
/// read either way (`mat.shape` 0 skips nothing but its effect).
pub static ENABLED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

/// Set in the lowest bit of a texel's height: the cell whose LOWEST corner
/// (smallest x and z) this sample is, is split along the diagonal between
/// its other two corners (`(ix + 1, iz)` to `(ix, iz + 1)`); clear for the
/// one through this sample.
pub const DIAGONAL_BIT: u32 = 1;

/// The ground's step-1 shape, a texel a sample, row `iz`, column `ix`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GroundShape {
    /// Texels a side: the lattice's samples, `cells + 1`.
    pub size: [u32; 2],
    /// Metres a cell, along x and z.
    pub cell: [f32; 2],
    /// Cells the ground's footprint uv runs over, along x and z: the uv of a
    /// sample is its index over these (the footprint can be a hair longer
    /// than the cells meshed, `Heightfield::meshed_cells`).
    pub uv_cells: [f32; 2],
    /// `[x, y, z, height]` a texel, row-major (the height's lowest bit
    /// [`DIAGONAL_BIT`]).
    pub texels: Vec<[f32; 4]>,
}

/// The blend weights of a cell's four corners `(00, 10, 01, 11)` -- x first --
/// at `t` (the place in the cell, 0..1 each way), for the cell's diagonal:
/// the step-1 triangle the place lies in, and the place's barycentric weights
/// in it. `through_first`: the diagonal from corner 00 to 11.
pub fn corner_weights(t: [f32; 2], through_first: bool) -> [f32; 4] {
    let [tx, ty] = t;
    if through_first {
        [1.0 - tx.max(ty), (tx - ty).max(0.0), (ty - tx).max(0.0), tx.min(ty)]
    } else {
        let w11 = (tx + ty - 1.0).max(0.0);
        [(1.0 - tx - ty).max(0.0), tx - w11, ty - w11, w11]
    }
}

impl GroundShape {
    /// The shape of a ground meshed as a lattice of `cells`, its samples the
    /// first `(cells[0] + 1) * (cells[1] + 1)` of `positions` (sample
    /// `(ix, iz)` at `iz * (cells[0] + 1) + ix`) with `normals` beside them,
    /// drawn as `triangles`. `uv_cells`: [`Self::uv_cells`]. `None` for a
    /// ground with no cell, or too few positions.
    pub fn build(cells: [u32; 2], uv_cells: [f32; 2], positions: &[Vec3], normals: &[Vec3], triangles: &[u32]) -> Option<Self> {
        let [nx, nz] = cells;
        let size = [nx + 1, nz + 1];
        let samples = (size[0] * size[1]) as usize;
        if nx == 0 || nz == 0 || positions.len() < samples || normals.len() < samples {
            return None;
        }
        let cell = [(positions[nx as usize].x - positions[0].x) / nx as f32, (positions[(nz * size[0]) as usize].z - positions[0].z) / nz as f32];
        // EACH CELL'S DIAGONAL, as its drawn triangles have it: a triangle of
        // the lattice within one cell holds one of the two diagonals' ends.
        // A cell with none drawn (a hole, ground under a floor) keeps the one
        // through its first sample; only a coarser step draws there, over
        // the same normals.
        let mut across = vec![false; samples];
        for t in triangles.chunks_exact(3) {
            if t.iter().any(|&v| v as usize >= samples) {
                continue;
            }
            let at = [t[0], t[1], t[2]].map(|v| (v % size[0], v / size[0]));
            let (x0, z0) = (at.iter().map(|a| a.0).min().unwrap(), at.iter().map(|a| a.1).min().unwrap());
            let (x1, z1) = (at.iter().map(|a| a.0).max().unwrap(), at.iter().map(|a| a.1).max().unwrap());
            if x1 != x0 + 1 || z1 != z0 + 1 {
                continue;
            }
            let has = |x: u32, z: u32| at.contains(&(x, z));
            if has(x0 + 1, z0) && has(x0, z0 + 1) {
                across[(z0 * size[0] + x0) as usize] = true;
            }
        }
        let texels = (0..samples)
            .map(|i| {
                let n = normals[i].try_normalize().unwrap_or(Vec3::Y);
                let h = f32::from_bits((positions[i].y.to_bits() & !DIAGONAL_BIT) | if across[i] { DIAGONAL_BIT } else { 0 });
                [n.x, n.y, n.z, h]
            })
            .collect();
        Some(Self { size, cell, uv_cells, texels })
    }

    /// The texels as bytes, for an `Rgba32Float` texture.
    pub fn bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.texels)
    }

    /// What the shader reads at footprint uv `uv`: the normal (normalised)
    /// and the face, both in the world. The CPU twin of [`WGSL`].
    pub fn at(&self, uv: [f32; 2]) -> (Vec3, Vec3) {
        let g = [uv[0] * self.uv_cells[0], uv[1] * self.uv_cells[1]];
        let c = [g[0].floor().clamp(0.0, (self.size[0] - 2) as f32), g[1].floor().clamp(0.0, (self.size[1] - 2) as f32)];
        let t = [(g[0] - c[0]).clamp(0.0, 1.0), (g[1] - c[1]).clamp(0.0, 1.0)];
        let (i, j) = (c[0] as u32, c[1] as u32);
        let texel = |x: u32, z: u32| self.texels[(z * self.size[0] + x) as usize];
        let corners = [texel(i, j), texel(i + 1, j), texel(i, j + 1), texel(i + 1, j + 1)];
        let through_first = corners[0][3].to_bits() & DIAGONAL_BIT == 0;
        let w = corner_weights(t, through_first);
        let n: Vec3 = (0..4).map(|k| Vec3::new(corners[k][0], corners[k][1], corners[k][2]) * w[k]).sum();
        let h = corners.map(|c| c[3]);
        let upper = if through_first { t[1] > t[0] } else { t[0] + t[1] > 1.0 };
        let sx = if upper { h[3] - h[2] } else { h[1] - h[0] };
        let sz = if through_first == upper { h[2] - h[0] } else { h[3] - h[1] };
        (n.normalize_or_zero(), Vec3::new(-sx / self.cell[0], 1.0, -sz / self.cell[1]).normalize())
    }
}

/// THE SHADER'S READ, for the ground's material group (binding 9; the uv's
/// cells and one over the cell sizes in `mat.shape`, [`crate::renderer::terrain_pipeline::
/// TerrainMaterialUniform::shape`]). `ground_shape_at(uv, lattice)`: the
/// normal and face at footprint uv `uv` in the WORLD, `shaped` false where
/// the material has no shape or the triangle touches a vertex off the lattice
/// (`lattice`, the vertex colour's alpha, under 1). Handed to
/// `terrain_shader`'s `format!` as an argument, so in single braces.
pub const WGSL: &str = r#"
@group(1) @binding(9) var ground_shape: texture_2d<f32>;

struct GroundShapeRead {
    n: vec3<f32>,
    face: vec3<f32>,
    shaped: bool,
}

// THE STEP-1 GROUND UNDER A PIXEL (`ground_shape`): the cell from the uv, the
// triangle from the cell's diagonal, the normals blended as the rasterizer
// blends a step-1 triangle's, and that triangle's own plane.
fn ground_shape_at(uv: vec2<f32>, lattice: f32) -> GroundShapeRead {
    var r: GroundShapeRead;
    r.shaped = (mat.shape.x > 0.0) & (lattice > 0.999);
    let g = uv * mat.shape.xy;
    let c = clamp(vec2<i32>(floor(g)), vec2<i32>(0), vec2<i32>(textureDimensions(ground_shape)) - 2);
    let t = clamp(g - vec2<f32>(c), vec2<f32>(0.0), vec2<f32>(1.0));
    let a = textureLoad(ground_shape, c, 0);
    let b = textureLoad(ground_shape, c + vec2<i32>(1, 0), 0);
    let d = textureLoad(ground_shape, c + vec2<i32>(0, 1), 0);
    let e = textureLoad(ground_shape, c + vec2<i32>(1, 1), 0);
    // The triangle's two edges along x and z, normal and height alike: its
    // normal is a corner's plus `t` along them, and its plane falls by their
    // heights' steps. `first`: the cell's diagonal runs from `a` to `e`.
    let first = (bitcast<u32>(a.w) & 1u) == 0u;
    let upper = select(t.x + t.y > 1.0, t.y > t.x, first);
    let along_x = select(b - a, e - d, upper);
    let along_z = select(e - b, d - a, first == upper);
    let base = select(a.xyz, b.xyz + d.xyz - e.xyz, upper & !first);
    r.n = base + along_x.xyz * t.x + along_z.xyz * t.y;
    r.face = normalize(vec3<f32>(-along_x.w * mat.shape.z, 1.0, -along_z.w * mat.shape.w));
    return r;
}
"#;

/// The ground reader's two lines that take the shape, and the same lines
/// without it: what [`unshaped`] edits.
pub const SHAPED_EDITS: &[(&str, &str)] = &[
    (
        "    let shape = ground_shape_at(in.shape_uv, in.col.a);\n    let n = normalize(select(in.normal, to_player_direction(shape.n), shape.shaped));\n",
        "    let n = normalize(in.normal);\n",
    ),
    ("    let f = sample_frame(in.tex_pos, n, step_x, step_y, to_player_direction(shape.face), shape.shaped);", "    let f = sample_frame(in.tex_pos, n, step_x, step_y, n, false);"),
];

/// A ground shader (any role or twin) WITHOUT the shape: lit by its
/// interpolated normals and textured by its drawn triangles, as before
/// 2026-10-14, and none of the read's code in it. `None` if the lines are not
/// there once each.
///
/// THE WEATHER'S TWINS ARE BUILT SO (`weather::with_weather`): its snow lifts
/// the vertices off the shape the texture holds, and its gentle twins were at
/// the instruction cache's edge before the read (the puddles' about 3,350 of
/// 3,387; the read is about 110 proxy units more). A chunk the weather draws
/// is therefore held at step 1 (`ground_lod::GroundLod::hold`), where the two
/// are one picture.
pub fn unshaped(src: &str) -> Option<String> {
    crate::renderer::ground_cuts::with_cut(src, SHAPED_EDITS)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A lattice of `cells`, 0.43 m cells, heights `h`, meshed as the
    /// mesher meshes it (the shorter diagonal), with area-weighted normals.
    fn lattice(cells: u32, h: &dyn Fn(f32, f32) -> f32) -> (Vec<Vec3>, Vec<Vec3>, Vec<u32>) {
        let (positions, triangles) = crate::renderer::ground_lod::tests::ground(cells, h, &|_, _| false);
        let mut normals = vec![Vec3::ZERO; positions.len()];
        for t in triangles.chunks_exact(3) {
            let (a, b, c) = (positions[t[0] as usize], positions[t[1] as usize], positions[t[2] as usize]);
            let f = (b - a).cross(c - a);
            for &v in t {
                normals[v as usize] += f;
            }
        }
        (positions, normals.into_iter().map(|n| n.normalize_or(Vec3::Y)).collect(), triangles)
    }

    /// AT STEP 1 THE READ IS THE RASTERIZER'S BLEND: at every place inside
    /// every triangle of a rough lattice, the shape's normal is the
    /// triangle's three vertex normals blended by the place's barycentric
    /// weights, and its face is the triangle's own plane.
    #[test]
    fn inside_every_step_1_triangle_the_read_is_its_vertex_blend_and_its_plane() {
        let h = |x: f32, z: f32| 3.0 * (x * 0.37).sin() * (z * 0.21).cos() + 0.8 * (x * 1.9 + z * 2.3).sin();
        let cells = 24;
        let (positions, normals, triangles) = lattice(cells, &h);
        let shape = GroundShape::build([cells, cells], [cells as f32; 2], &positions, &normals, &triangles).unwrap();
        let (lo, hi) = (positions[0], positions[positions.len() - 1]);
        let mut worst = (0.0f32, 0.0f32);
        let across = shape.texels.iter().filter(|t| t[3].to_bits() & DIAGONAL_BIT != 0).count();
        for t in triangles.chunks_exact(3) {
            let [a, b, c] = [t[0], t[1], t[2]].map(|v| positions[v as usize]);
            let [na, nb, nc] = [t[0], t[1], t[2]].map(|v| normals[v as usize]);
            for (u, v) in [(0.2, 0.3), (0.6, 0.1), (0.1, 0.8), (0.33, 0.33)] {
                let p = a * (1.0 - u - v) + b * u + c * v;
                let uv = [(p.x - lo.x) / (hi.x - lo.x), (p.z - lo.z) / (hi.z - lo.z)];
                let (n, face) = shape.at(uv);
                let want = (na * (1.0 - u - v) + nb * u + nc * v).normalize();
                let mut plane = (b - a).cross(c - a).normalize();
                if plane.y < 0.0 {
                    plane = -plane;
                }
                worst = (worst.0.max((n - want).length()), worst.1.max((face - plane).length()));
            }
        }
        assert!(across > 0, "every cell split one way: the test does not try the other diagonal");
        assert!(worst.0 < 1e-5, "normal off by {}", worst.0);
        assert!(worst.1 < 1e-4, "face off by {}", worst.1);
    }

    /// A COARSER TRIANGLE READS THE GROUND UNDER IT: a step-2 triangle's
    /// middle reads the step-1 triangle there, not the blend of its own
    /// three corners.
    #[test]
    fn a_coarser_triangle_reads_the_step_1_ground_under_it() {
        let h = |x: f32, z: f32| if x > -18.0 { 1.5 } else { 0.0 } + 0.1 * z + 0.3 * (z * 3.1).sin();
        let cells = 8;
        let (positions, normals, triangles) = lattice(cells, &h);
        let shape = GroundShape::build([cells, cells], [cells as f32; 2], &positions, &normals, &triangles).unwrap();
        // The middle of the 2 x 2 cells from sample (4, 0): sample (5, 1).
        let mid = 1 * (cells + 1) + 5;
        let uv = [5.0 / cells as f32, 1.0 / cells as f32];
        let (n, _) = shape.at(uv);
        assert!((n - normals[mid as usize]).length() < 1e-5, "{n} against the sample's own {}", normals[mid as usize]);
    }

    #[test]
    fn the_weights_are_barycentric_in_both_triangles_of_both_diagonals() {
        for first in [true, false] {
            for t in [[0.1, 0.2], [0.7, 0.2], [0.2, 0.7], [0.9, 0.95], [0.5, 0.5]] {
                let w = corner_weights(t, first);
                assert!((w.iter().sum::<f32>() - 1.0).abs() < 1e-6);
                assert!(w.iter().all(|&x| x >= 0.0));
                let x = w[1] + w[3];
                let z = w[2] + w[3];
                assert!((x - t[0]).abs() < 1e-6 && (z - t[1]).abs() < 1e-6, "{first} {t:?} -> {w:?}");
                // The corner off the triangle has no say.
                assert!(if first { w[1] == 0.0 || w[2] == 0.0 } else { w[0] == 0.0 || w[3] == 0.0 });
            }
        }
    }
}

#[cfg(test)]
mod varyings {
    /// EVERY GROUND READER'S VARYINGS FIT THE QUEST 3: no vertex output past
    /// location 14 (15 inter-stage locations; the host does not enforce it,
    /// the headset refuses the pipeline -- `effects::variants`' test of the
    /// same name). The shape added `shape_uv` at location 5. Every role,
    /// both builds, the slope twins and the weather's.
    #[test]
    fn every_ground_readers_varyings_fit_the_quest_3s_inter_stage_limit() {
        use crate::renderer::terrain_pipeline::{terrain_shader_built, TerrainRole};
        const QUEST3_MAX_INTER_STAGE_LOCATIONS: u32 = 15;
        let mut checked = 0;
        for role in [TerrainRole::Scene, TerrainRole::Read, TerrainRole::ProbePass] {
            for painted in [false, true] {
                let base = terrain_shader_built(role, painted);
                let mut twins = vec![base.clone()];
                twins.extend(crate::renderer::ground_twins::gentle_shader(&base));
                twins.extend(crate::renderer::ground_twins::steep_shader(&base));
                let weathered: Vec<String> = twins
                    .iter()
                    .flat_map(|t| crate::renderer::weather::WeatherKinds::ALL.iter().filter_map(move |k| crate::renderer::weather::with_weather_kinds(t, *k)))
                    .collect();
                twins.extend(weathered);
                for src in twins.iter().filter_map(|t| crate::renderer::terrain_pipeline::world_vertices(t)).chain(twins.iter().cloned()) {
                    let module = wgpu::naga::front::wgsl::parse_str(&src).unwrap_or_else(|e| panic!("{role:?}: {}", e.emit_to_string(&src)));
                    for ep in module.entry_points.iter().filter(|e| e.stage == wgpu::naga::ShaderStage::Vertex) {
                        let Some(result) = &ep.function.result else { continue };
                        let mut top = 0;
                        if let wgpu::naga::TypeInner::Struct { members, .. } = &module.types[result.ty].inner {
                            for m in members {
                                if let Some(wgpu::naga::Binding::Location { location, .. }) = &m.binding {
                                    top = top.max(*location);
                                }
                            }
                        }
                        assert!(top < QUEST3_MAX_INTER_STAGE_LOCATIONS, "{role:?} writes location {top}; the Quest 3 stops at 14");
                        checked += 1;
                    }
                }
            }
        }
        assert!(checked >= 60, "{checked} ground vertex stages checked");
    }
}

#[cfg(test)]
mod sizes {
    /// MEASUREMENT: what the shape read costs the dry gentle reader by the
    /// host proxy, function by function (`--ignored --nocapture`).
    #[test]
    #[ignore]
    fn shape_read_sizes() {
        use crate::renderer::terrain_pipeline::{terrain_shader_built, TerrainRole};
        let read = terrain_shader_built(TerrainRole::Read, true);
        let gentle = crate::renderer::lights::without_spot_shadows(crate::renderer::brush_pipeline::sun_reader_shader(
            crate::renderer::ground_twins::gentle_shader(&read).unwrap(),
            crate::renderer::brush_pipeline::FaceSun::Baked,
        ));
        let total = |src: &str| crate::renderer::shader_inlining::inlined_sizes(src, "fs_main").iter().map(|f| f.own * f.copies).sum::<usize>();
        // The face is dead in a gentle twin (its steep test is false): what
        // the compiler keeps of the read there.
        let faceless = |src: &str| {
            let face = "    r.face = normalize(vec3<f32>(-along_x.w * mat.shape.z, 1.0, -along_z.w * mat.shape.w));";
            assert!(src.contains(face));
            src.replace(face, "    r.face = vec3<f32>(0.0, 1.0, 0.0);")
                .replace("let along_x = select(b - a, e - d, upper);", "let along_x = vec4<f32>(select(b.xyz - a.xyz, e.xyz - d.xyz, upper), 0.0);")
                .replace("let along_z = select(e - b, d - a, first == upper);", "let along_z = vec4<f32>(select(e.xyz - b.xyz, d.xyz - a.xyz, first == upper), 0.0);")
        };
        let unshaped = |src: &str| src.replace("    let shape = ground_shape_at(in.shape_uv, in.col.a);\n", "    var shape: GroundShapeRead;\n");
        eprintln!("dry gentle: {} with the read; {} as the compiler keeps it (face dead); {} without it", total(&gentle), total(&faceless(&gentle)), total(&unshaped(&gentle)));
        for kinds in crate::renderer::weather::WeatherKinds::ALL {
            let w = crate::renderer::weather::with_weather_kinds(&gentle, kinds).unwrap();
            eprintln!("weather {:?} gentle: {} with; {} face dead; {} without", kinds, total(&w), total(&faceless(&w)), total(&unshaped(&w)));
            let t = crate::renderer::ground_twins::torch_weather_twin(w.clone());
            let lit = crate::renderer::brush_pipeline::sun_reader_shader(
                crate::renderer::ground_twins::gentle_shader(&read).unwrap(),
                crate::renderer::brush_pipeline::FaceSun::Baked,
            );
            let _ = (t, lit);
        }
        let mut all = crate::renderer::shader_inlining::inlined_sizes(&gentle, "fs_main");
        all.sort_by_key(|f| std::cmp::Reverse(f.own * f.copies));
        for f in all.iter().take(40) {
            eprintln!("{:>6} = {:>5} x {:>2}  {}", f.own * f.copies, f.own, f.copies, f.name);
        }
    }
}
