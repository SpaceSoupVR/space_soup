use glam::Vec3;

pub static ENABLED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

pub const DIAGONAL_BIT: u32 = 1;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct GroundShape {

    pub size: [u32; 2],

    pub cell: [f32; 2],

    pub uv_cells: [f32; 2],

    pub texels: Vec<[f32; 4]>,
}

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

    pub fn build(cells: [u32; 2], uv_cells: [f32; 2], positions: &[Vec3], normals: &[Vec3], triangles: &[u32]) -> Option<Self> {
        let [nx, nz] = cells;
        let size = [nx + 1, nz + 1];
        let samples = (size[0] * size[1]) as usize;
        if nx == 0 || nz == 0 || positions.len() < samples || normals.len() < samples {
            return None;
        }
        let cell = [(positions[nx as usize].x - positions[0].x) / nx as f32, (positions[(nz * size[0]) as usize].z - positions[0].z) / nz as f32];

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

                let y = positions[i].y;
                let y = if y.to_bits() & 0x7fff_ffff < f32::MIN_POSITIVE.to_bits() { f32::MIN_POSITIVE.copysign(y) } else { y };
                let h = f32::from_bits((y.to_bits() & !DIAGONAL_BIT) | if across[i] { DIAGONAL_BIT } else { 0 });
                [n.x, n.y, n.z, h]
            })
            .collect();
        Some(Self { size, cell, uv_cells, texels })
    }

    pub fn bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.texels)
    }

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

pub const WGSL: &str = r"
@group(1) @binding(9) var ground_shape: texture_2d<f32>;

struct GroundShapeRead {
    n: vec3<f32>,
    face: vec3<f32>,
    shaped: bool,
}

fn ground_shape_at(uv: vec2<f32>, corner: vec2<f32>, lattice: f32) -> GroundShapeRead {
    var r: GroundShapeRead;
    let k = vec2<u32>(round(corner * mat.shape.xy));
    let coarse = all((k & vec2<u32>(1u)) == vec2<u32>(0u));
    if ((mat.shape.x > 0.0) & (lattice > 0.999) & coarse) {
        r.shaped = true;
        let g = uv * mat.shape.xy;
        let c = clamp(vec2<i32>(floor(g)), vec2<i32>(0), vec2<i32>(textureDimensions(ground_shape)) - 2);
        let t = clamp(g - vec2<f32>(c), vec2<f32>(0.0), vec2<f32>(1.0));
        let a = textureLoad(ground_shape, c, 0);
        let b = textureLoad(ground_shape, c + vec2<i32>(1, 0), 0);
        let d = textureLoad(ground_shape, c + vec2<i32>(0, 1), 0);
        let e = textureLoad(ground_shape, c + vec2<i32>(1, 1), 0);

        let first = (bitcast<u32>(a.w) & 1u) == 0u;
        let upper = select(t.x + t.y > 1.0, t.y > t.x, first);
        let along_x = select(b - a, e - d, upper);
        let along_z = select(e - b, d - a, first == upper);
        let base = select(a.xyz, b.xyz + d.xyz - e.xyz, upper & !first);
        r.n = base + along_x.xyz * t.x + along_z.xyz * t.y;
        r.face = normalize(vec3<f32>(-along_x.w * mat.shape.z, 1.0, -along_z.w * mat.shape.w));
    }
    return r;
}
";

pub const SHAPED_EDITS: &[(&str, &str)] = &[
    (
        "    let shape = ground_shape_at(in.uv, in.corner, in.col.a);\n    let n = normalize(select(in.normal, to_player_direction(shape.n), shape.shaped));\n",
        "    let n = normalize(in.normal);\n",
    ),
    ("    let f = sample_frame(in.tex_pos, n, step_x, step_y, to_player_direction(shape.face), shape.shaped);", "    let f = sample_frame(in.tex_pos, n, step_x, step_y, n, false);"),
];

pub fn unshaped(src: &str) -> Option<String> {
    crate::renderer::ground_cuts::with_cut(src, SHAPED_EDITS)
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn a_height_of_zero_keeps_its_diagonal_bit_on_a_normal_number() {
        let cells = 8;
        let (positions, normals, triangles) = lattice(cells, &|x, z| if (x + z) > 0.0 { 0.0 } else { -0.0 });
        let shape = GroundShape::build([cells, cells], [cells as f32; 2], &positions, &normals, &triangles).unwrap();
        assert!(shape.texels.iter().all(|t| t[3].is_normal()), "a height carries the bit as a denormal");
        assert!(shape.texels.iter().all(|t| t[3].abs() < 1e-30));
        assert!(shape.texels.iter().any(|t| t[3].to_bits() & DIAGONAL_BIT != 0));
    }

    #[test]
    fn a_coarser_triangle_reads_the_step_1_ground_under_it() {
        let h = |x: f32, z: f32| if x > -18.0 { 1.5 } else { 0.0 } + 0.1 * z + 0.3 * (z * 3.1).sin();
        let cells = 8;
        let (positions, normals, triangles) = lattice(cells, &h);
        let shape = GroundShape::build([cells, cells], [cells as f32; 2], &positions, &normals, &triangles).unwrap();

        let mid = (cells + 1) + 5;
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

                assert!(if first { w[1] == 0.0 || w[2] == 0.0 } else { w[0] == 0.0 || w[3] == 0.0 });
            }
        }
    }
}

#[cfg(test)]
mod varyings {

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

                        if let wgpu::naga::TypeInner::Struct { members, .. } = &module.types[result.ty].inner {
                            for m in members {
                                if let Some(wgpu::naga::Binding::Location { sampling, .. }) = &m.binding {
                                    assert_ne!(*sampling, Some(wgpu::naga::Sampling::Centroid), "{role:?}: `{}` is interpolated at the centroid", m.name.as_deref().unwrap_or("?"));
                                }
                            }
                        }
                        assert!(!src.contains(", centroid") && !src.contains("(centroid"), "{role:?}: a ground reader asks for centroid");
                        checked += 1;
                    }
                }
            }
        }
        assert!(checked >= 60, "{checked} ground vertex stages checked");
    }
}

#[cfg(test)]
mod twins {

    #[test]
    fn a_step_1_run_is_drawn_by_a_pipeline_without_the_shape_read() {
        use crate::renderer::terrain_pipeline::{by_step, terrain_shader_built, twin_sources, TerrainRole};
        const CALL: &str = "ground_shape_at(in.";
        let mut checked = 0;
        for role in [TerrainRole::Scene, TerrainRole::Read, TerrainRole::ProbePass] {
            for painted in [false, true] {
                let base = terrain_shader_built(role, painted);
                let mut sources = vec![base.clone()];
                sources.extend(crate::renderer::ground_twins::gentle_shader(&base));
                sources.extend(crate::renderer::ground_twins::steep_shader(&base));
                if role == TerrainRole::ProbePass {
                    sources.extend(crate::renderer::ground_twins::dedup_pass_shader(&base));
                } else {
                    sources.push(crate::renderer::lights::without_spot_shadows(base.clone()));
                    sources.push(crate::renderer::brush_pipeline::sun_reader_shader(base.clone(), crate::renderer::brush_pipeline::FaceSun::Baked));
                }
                let world: Vec<String> = sources.iter().filter_map(|s| crate::renderer::terrain_pipeline::world_vertices(s)).collect();
                sources.extend(world);
                for src in sources {
                    assert!(src.contains(CALL), "{role:?}: a ground source without the read before the twins are made");
                    let (plain, shaped) = twin_sources(src);
                    assert!(!plain.contains(CALL), "{role:?}: the step-1 pipeline reads the shape");
                    assert!(shaped.is_some_and(|s| s.contains(CALL)), "{role:?}: no shaped pipeline for the coarser lists");
                    checked += 1;
                }
            }
        }
        for src in [
            crate::renderer::terrain_pipeline::outdoor_pass_shader(),
            crate::renderer::terrain_pipeline::nowhere_pass_shader(),
            crate::renderer::terrain_pipeline::open_pass_shader(),
        ]
        .into_iter()
        .flatten()
        {
            let (plain, shaped) = twin_sources(src);
            assert!(!plain.contains(CALL) && shaped.is_some_and(|s| s.contains(CALL)));
            checked += 1;
        }
        let weathered = crate::renderer::weather::with_weather(&terrain_shader_built(TerrainRole::Read, true)).unwrap();
        let (plain, shaped) = twin_sources(weathered);
        assert!(!plain.contains(CALL) && shaped.is_none(), "the weather's twin is unshaped and has no shaped one");
        assert!(checked >= 40, "{checked} sources");

        let parts = |first, len| by_step(first, len, 1000).collect::<Vec<_>>();
        assert_eq!(parts(0, 999), vec![(0, 999, false)]);
        assert_eq!(parts(990, 30), vec![(990, 10, false), (1000, 20, true)]);
        assert_eq!(parts(1000, 30), vec![(1000, 30, true)]);
        assert_eq!(by_step(5, 7, u32::MAX).collect::<Vec<_>>(), vec![(5, 7, false)], "no lists: all step 1");
    }
}

#[cfg(test)]
mod sizes {

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

        let faceless = |src: &str| {
            let face = "        r.face = normalize(vec3<f32>(-along_x.w * mat.shape.z, 1.0, -along_z.w * mat.shape.w));";
            assert!(src.contains(face));
            src.replace(face, "        r.face = vec3<f32>(0.0, 1.0, 0.0);")
                .replace("let along_x = select(b - a, e - d, upper);", "let along_x = vec4<f32>(select(b.xyz - a.xyz, e.xyz - d.xyz, upper), 0.0);")
                .replace("let along_z = select(e - b, d - a, first == upper);", "let along_z = vec4<f32>(select(e.xyz - b.xyz, d.xyz - a.xyz, first == upper), 0.0);")
        };
        let unshaped = |src: &str| src.replace("    let shape = ground_shape_at(in.uv, in.corner, in.col.a);\n", "    var shape: GroundShapeRead;\n");
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
