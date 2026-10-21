use glam::{Mat4, Vec3};
use wgpu::*;

use super::layered_mesh_pipeline::{layered_mesh_shader, LayeredMeshPipeline, LayeredVertex};

pub const EXTRAPOLATION: f32 = 0.5;

const GROUND_FROM: f32 = 0.6;
const GROUND_FULL: f32 = 0.85;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CaveClass {

    Rock,

    GroundFront,

    GroundBack,

    Mixed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CavePath {

    Rock,

    Ground,
}

fn closest_to_origin(a: Vec3, b: Vec3, c: Vec3) -> Vec3 {

    let ab = b - a;
    let ac = c - a;
    let ap = -a;
    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return a;
    }
    let bp = -b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        return b;
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        return a + ab * (d1 / (d1 - d3));
    }
    let cp = -c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 {
        return c;
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        return a + ac * (d2 / (d2 - d6));
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        return b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)));
    }
    let denom = 1.0 / (va + vb + vc);
    a + ab * (vb * denom) + ac * (vc * denom)
}

fn widened<T: Copy + std::ops::Add<Output = T> + std::ops::Sub<Output = T> + std::ops::Mul<f32, Output = T>>(
    v: [T; 3],
    k: f32,
) -> [T; 3] {
    let c = (v[0] + v[1] + v[2]) * (1.0 / 3.0);
    let s = 1.0 + 3.0 * k;
    v.map(|x| c + (x - c) * s)
}

pub fn classify(normals: [Vec3; 3], ground: [f32; 3]) -> CaveClass {
    let n = widened(normals, EXTRAPOLATION);
    let g = widened(ground, EXTRAPOLATION);

    let longest = n.iter().map(|v| v.length()).fold(0.0f32, f32::max);
    let nearest = closest_to_origin(n[0], n[1], n[2]).length();

    let y_min = n.iter().map(|v| v.y).fold(f32::INFINITY, f32::min);
    let y_max = n.iter().map(|v| v.y).fold(f32::NEG_INFINITY, f32::max);
    let no_ground_copy = g.iter().all(|&z| z <= 0.0);
    if no_ground_copy && nearest > 1e-6 && y_min.abs().max(y_max.abs()) < GROUND_FROM * nearest {
        return CaveClass::Rock;
    }
    if longest > 0.0 && y_min > 0.0 && y_min >= GROUND_FULL * longest {
        return CaveClass::GroundFront;
    }
    if longest > 0.0 && y_max < 0.0 && -y_max >= GROUND_FULL * longest {
        return CaveClass::GroundBack;
    }
    CaveClass::Mixed
}

pub fn partition(vertices: &[LayeredVertex], indices: &[u32]) -> (Vec<u32>, [u32; 4]) {
    let mut classes: [Vec<u32>; 4] = Default::default();
    for tri in indices.chunks_exact(3) {
        let v = |i: u32| vertices.get(i as usize);
        let class = match (v(tri[0]), v(tri[1]), v(tri[2])) {
            (Some(a), Some(b), Some(c)) => classify(
                [Vec3::from(a.normal), Vec3::from(b.normal), Vec3::from(c.normal)],
                [a.ground[2], b.ground[2], c.ground[2]],
            ),
            _ => CaveClass::Mixed,
        };
        let k = match class {
            CaveClass::Rock => 0,
            CaveClass::GroundFront => 1,
            CaveClass::GroundBack => 2,
            CaveClass::Mixed => 3,
        };
        classes[k].extend_from_slice(tri);
    }
    let mut out = Vec::with_capacity(indices.len());
    let mut ends = [0u32; 4];
    for (k, c) in classes.iter().enumerate() {
        out.extend_from_slice(c);
        ends[k] = out.len() as u32;
    }

    out.extend_from_slice(&indices[indices.len() - indices.len() % 3..]);
    (out, ends)
}

pub fn keeps_vertical(m: Mat4) -> bool {
    let (x, y, z) = (m.x_axis.truncate(), m.y_axis.truncate(), m.z_axis.truncate());
    let s = y.y;
    let tol = 1e-5 * s.abs().max(1e-12);
    s > 0.0
        && x.y.abs() <= tol
        && z.y.abs() <= tol
        && y.x.abs() <= tol
        && y.z.abs() <= tol
        && (x.length() - s).abs() <= 1e-4 * s
        && (z.length() - s).abs() <= 1e-4 * s
        && m.determinant() > 0.0
}

pub fn placed_box(m: Mat4, b: [[f32; 3]; 2]) -> (Vec3, Vec3) {
    (0..8).fold((Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY)), |(lo, hi), i| {
        let c = Vec3::new(b[(i & 1)][0], b[(i >> 1) & 1][1], b[(i >> 2) & 1][2]);
        let p = m.transform_point3(c);
        (lo.min(p), hi.max(p))
    })
}

fn substitute(src: String, from: &str, to: &str) -> Option<String> {
    src.contains(from).then(|| src.replacen(from, to, 1))
}

const GROUND_BRANCH: &str = "    if (as_ground_w > 0.0) {";
const ROCK_LIT: &str = "    var lit = shade_with_sky(in.world_pos, shaded_n, sky);";
const ROCK_BOUNCE: &str =
    "    lit = lit + sky_irradiance(up_p) * (GROUND_ALBEDO * downward * smoothstep(0.1, 0.45, sky) * (1.0 - gb));";
const GROUND_MIX: &str = "        colour = mix(colour, as_ground, as_ground_w);";

pub fn twin_source(path: CavePath) -> Option<String> {
    let src = layered_mesh_shader();
    match path {

        CavePath::Rock => substitute(src, GROUND_BRANCH, "    if (false) {"),

        CavePath::Ground => {
            let src = substitute(src, ROCK_LIT, "    var lit = vec3<f32>(0.0);")?;
            let src = substitute(src, ROCK_BOUNCE, "")?;
            substitute(src, GROUND_MIX, "        colour = as_ground;")
        }
    }
}

pub fn occluder_source() -> String {
    format!("{}\n@fragment fn fs_depth() {{}}\n", layered_mesh_shader())
}

pub struct CaveTwins {
    rock: LayeredMeshPipeline,
    ground_front: LayeredMeshPipeline,
    ground_back: LayeredMeshPipeline,
    full_cull_front: LayeredMeshPipeline,
    full_cull_back: LayeredMeshPipeline,
    pub occluder: RenderPipeline,
}

impl CaveTwins {

    pub fn new(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        material_layout: &BindGroupLayout,
        samples: u32,
    ) -> Option<Self> {
        let rock_src = twin_source(CavePath::Rock)?;
        let ground_src = twin_source(CavePath::Ground)?;
        let mono = crate::renderer::multiview::ViewMode::Mono;
        let build = |src: &str, cull: Option<Face>, label: &str| {
            LayeredMeshPipeline::build(
                device, format, uniform_layout, material_layout, FrontFace::Ccw, samples, mono, src.to_string(), cull,
                label,
            )
        };
        let full = layered_mesh_shader();
        let rock = build(&rock_src, None, "cave_twin_rock");
        let ground_front = build(&ground_src, Some(Face::Back), "cave_twin_ground");
        let ground_back = build(&ground_src, Some(Face::Front), "cave_twin_ground_back");
        let full_cull_front = build(&full, Some(Face::Front), "cave_twin_full_front");
        let full_cull_back = build(&full, Some(Face::Back), "cave_twin_full_back");
        let occluder = Self::occluder_pipeline(device, uniform_layout, material_layout, &rock.model_layout);
        Some(Self { rock, ground_front, ground_back, full_cull_front, full_cull_back, occluder })
    }

    fn occluder_pipeline(
        device: &Device,
        uniform_layout: &BindGroupLayout,
        material_layout: &BindGroupLayout,
        model_layout: &BindGroupLayout,
    ) -> RenderPipeline {
        use crate::renderer::brush_pipeline::probe_pass::{FORMAT, REACH_FORMAT};
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("cave_occluder"),
            source: ShaderSource::Wgsl(crate::renderer::multiview::ViewMode::Mono.shader(occluder_source()).into()),
        });
        let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("cave_occluder_layout"),
            bind_group_layouts: &[Some(uniform_layout), Some(material_layout), Some(model_layout)],
            immediate_size: 0,
        });
        let nothing = |format| Some(ColorTargetState { format, blend: None, write_mask: ColorWrites::empty() });
        device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("cave_occluder"),
            layout: Some(&layout),
            vertex: VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: PipelineCompilationOptions::default(),
                buffers: &[Some(LayeredVertex::layout())],
            },
            fragment: Some(FragmentState {
                module: &shader,
                entry_point: Some("fs_depth"),
                compilation_options: PipelineCompilationOptions::default(),
                targets: &[nothing(FORMAT), nothing(REACH_FORMAT)],
            }),
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                cull_mode: None,
                front_face: FrontFace::Ccw,
                polygon_mode: PolygonMode::Fill,
                ..Default::default()
            },
            depth_stencil: Some(DepthStencilState {
                format: TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(CompareFunction::Less),
                stencil: StencilState::default(),
                bias: DepthBiasState::default(),
            }),
            multisample: MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        })
    }

    pub fn draw<'a>(&'a self, pass: &mut RenderPass<'a>, full: &'a RenderPipeline, ends: [u32; 4]) {
        let [rock, front, back, mixed] = ends;
        let draws: [(&RenderPipeline, std::ops::Range<u32>); 6] = [
            (&self.rock.pipeline, 0..rock),
            (&self.ground_front.pipeline, rock..front),
            (&self.full_cull_front.pipeline, rock..front),
            (&self.ground_back.pipeline, front..back),
            (&self.full_cull_back.pipeline, front..back),
            (full, back..mixed),
        ];
        for (pipeline, range) in draws {
            if !range.is_empty() {
                pass.set_pipeline(pipeline);
                pass.draw_indexed(range, 0, 0..1);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(x: f32, y: f32, z: f32) -> Vec3 {
        Vec3::new(x, y, z).normalize()
    }

    #[test]
    fn a_wall_is_rock_a_floor_is_ground_and_a_bend_is_mixed() {

        assert_eq!(classify([v(1.0, 0.1, 0.0), v(1.0, 0.0, 0.1), v(1.0, -0.1, 0.0)], [0.0; 3]), CaveClass::Rock);

        assert_eq!(classify([v(1.0, 0.1, 0.0), v(1.0, 0.0, 0.1), v(1.0, -0.1, 0.0)], [0.0, 0.2, 0.0]), CaveClass::Mixed);

        assert_eq!(classify([v(0.0, 1.0, 0.05), v(0.05, 1.0, 0.0), v(0.0, 1.0, -0.05)], [0.0; 3]), CaveClass::GroundFront);
        assert_eq!(classify([v(0.0, -1.0, 0.05), v(0.05, -1.0, 0.0), v(0.0, -1.0, -0.05)], [0.0; 3]), CaveClass::GroundBack);

        assert_eq!(classify([v(1.0, 0.2, 0.0), v(0.3, 1.0, 0.0), v(1.0, 0.2, 0.3)], [0.0; 3]), CaveClass::Mixed);

        assert_eq!(classify([v(1.0, 0.0, 0.0), v(-1.0, 0.0, 0.0), v(0.0, 0.0, 1.0)], [0.0; 3]), CaveClass::Mixed);
    }

    #[test]
    fn every_pixel_of_a_class_takes_its_twins_path() {
        let smooth = |x: f32| {
            let t = ((x - GROUND_FROM) / (GROUND_FULL - GROUND_FROM)).clamp(0.0, 1.0);
            t * t * (3.0 - 2.0 * t)
        };
        let mut seed = 7u32;
        let mut rnd = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0
        };
        let mut counted = [0usize; 4];
        for _ in 0..20000 {
            let base = Vec3::new(rnd(), rnd() * 1.5, rnd()).normalize();
            let jitter = 0.3 * rnd().abs();
            let n = [0, 1, 2].map(|_| (base + Vec3::new(rnd(), rnd(), rnd()) * jitter).normalize());
            let g = [0, 1, 2].map(|_| if rnd() > 0.8 { rnd().max(0.0) } else { 0.0 });
            let class = classify(n, g);
            counted[class as usize] += 1;
            for a in 0..=12 {
                for b in 0..=(12 - a) {
                    let k = EXTRAPOLATION;
                    let wa = -k + (1.0 + 3.0 * k) * a as f32 / 12.0;
                    let wb = -k + (1.0 + 3.0 * k) * b as f32 / 12.0;
                    let wc = 1.0 - wa - wb;
                    if wc < -k - 1e-6 {
                        continue;
                    }
                    let p = (n[0] * wa + n[1] * wb + n[2] * wc).normalize();
                    let gb = (g[0] * wa + g[1] * wb + g[2] * wc).clamp(0.0, 1.0);
                    let w = |y: f32| gb.max(smooth(y));
                    match class {
                        CaveClass::Rock => assert!(w(p.y) == 0.0 && w(-p.y) == 0.0, "{n:?} {g:?}"),
                        CaveClass::GroundFront => assert!(w(p.y) == 1.0, "{n:?}"),
                        CaveClass::GroundBack => assert!(w(-p.y) == 1.0, "{n:?}"),
                        CaveClass::Mixed => {}
                    }
                }
            }
        }
        assert!(counted.iter().all(|&c| c > 100), "every class exercised: {counted:?}");
    }

    #[test]
    fn the_partition_keeps_every_triangle_and_its_winding() {
        let vert = |n: Vec3| LayeredVertex {
            position: [0.0; 3],
            normal: n.into(),
            weights: [1.0, 0.0, 0.0, 0.0],
            aux: [1.0, 0.0, 0.0, 1.0],
            ground: [0.0; 4],
        };
        let vs = vec![vert(Vec3::X), vert(Vec3::Z), vert(Vec3::Y), vert(-Vec3::Y), vert(v(1.0, 0.7, 0.0))];
        let idx = vec![2, 2, 2, 0, 1, 0, 4, 2, 0, 3, 3, 3, 1, 0, 1];
        let (out, ends) = partition(&vs, &idx);
        assert_eq!(ends, [6, 9, 12, 15]);
        let tris = |i: &[u32]| {
            let mut t: Vec<[u32; 3]> = i.chunks(3).map(|c| [c[0], c[1], c[2]]).collect();
            t.sort();
            t
        };
        assert_eq!(tris(&out), tris(&idx));
        assert_eq!(&out[6..9], &[2, 2, 2]);
        assert_eq!(&out[9..12], &[3, 3, 3]);
    }

    #[test]
    fn a_placed_box_holds_every_placed_corner() {
        let m = Mat4::from_scale_rotation_translation(Vec3::splat(2.0), glam::Quat::from_rotation_y(0.5), Vec3::new(10.0, 1.0, -3.0));
        let b = [[-1.0, -2.0, -3.0], [4.0, 5.0, 6.0]];
        let (lo, hi) = placed_box(m, b);
        for i in 0..8 {
            let c = Vec3::new(b[i & 1][0], b[(i >> 1) & 1][1], b[(i >> 2) & 1][2]);
            let p = m.transform_point3(c);
            assert!(p.cmpge(lo - 1e-4).all() && p.cmple(hi + 1e-4).all());
        }
        assert!((lo.y - (1.0 - 4.0)).abs() < 1e-4 && (hi.y - (1.0 + 10.0)).abs() < 1e-4);
    }

    #[test]
    fn only_a_turn_about_the_vertical_keeps_the_classes() {
        assert!(keeps_vertical(Mat4::IDENTITY));
        assert!(keeps_vertical(Mat4::from_scale_rotation_translation(
            Vec3::splat(2.0),
            glam::Quat::from_rotation_y(1.1),
            Vec3::new(3.0, -1.0, 9.0)
        )));
        assert!(!keeps_vertical(Mat4::from_rotation_x(0.3)));
        assert!(!keeps_vertical(Mat4::from_scale(Vec3::new(1.0, 2.0, 1.0))));
        assert!(!keeps_vertical(Mat4::from_scale(Vec3::new(-1.0, 1.0, 1.0))));
    }

    #[test]
    fn the_twins_are_the_cave_shader_less_one_path() {
        let rock = twin_source(CavePath::Rock).expect("the rock twin's substitution matches");
        let ground = twin_source(CavePath::Ground).expect("the ground twin's substitutions match");
        let full = layered_mesh_shader();
        assert!(!rock.contains(GROUND_BRANCH) && full.contains(GROUND_BRANCH));
        assert!(!ground.contains(ROCK_LIT) && !ground.contains(ROCK_BOUNCE) && !ground.contains(GROUND_MIX));
        for (name, src) in [("rock", rock), ("ground", ground), ("occluder", occluder_source())] {
            let src = crate::renderer::multiview::ViewMode::Mono.shader(src);
            let module = naga::front::wgsl::parse_str(&src).unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(&src)));
            naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
                .validate(&module)
                .unwrap_or_else(|e| panic!("{name}: {e:?}"));
        }
    }

    #[test]
    fn opaque_scene_shaders_write_alpha_one() {
        use crate::renderer::terrain_pipeline::{terrain_shader_for, TerrainRole};
        let sources = [
            ("cave", layered_mesh_shader()),
            ("cave rock", twin_source(CavePath::Rock).unwrap()),
            ("cave ground", twin_source(CavePath::Ground).unwrap()),
            ("terrain scene", terrain_shader_for(TerrainRole::Scene)),
            ("terrain read", terrain_shader_for(TerrainRole::Read)),
        ];
        for (name, src) in sources {
            let start = src.find("@fragment fn fs_main(").unwrap_or_else(|| panic!("{name}: no fs_main"));
            let body = &src[start..];
            let body = &body[..body.find("\n}\n").unwrap_or(body.len())];
            let returns: Vec<&str> = body.lines().filter(|l| l.contains("return ")).collect();
            assert!(!returns.is_empty(), "{name}: no return");
            for r in returns {
                assert!(r.trim_end().ends_with(", 1.0);"), "{name}: `{}` may write alpha below 1", r.trim());
            }
        }
    }

    #[test]
    fn the_twins_and_the_occluder_build() {
        let Some((device, _queue)) = crate::renderer::terrain_pipeline::tests::headless_gpu() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let lights = crate::renderer::lights::LightsUniform::new(&device);
        let (_shadows, uniforms) = crate::renderer::uniforms::test_support::scene_uniforms(&device, &lights);
        let material_layout = crate::renderer::terrain_pipeline::material_bind_group_layout(&device);
        let scope = device.push_error_scope(ErrorFilter::Validation);
        let twins = CaveTwins::new(&device, TextureFormat::Rgba8Unorm, &uniforms.layout, &material_layout, 4);
        assert!(twins.is_some(), "a twin's substitution no longer matches");
        if let Some(err) = pollster::block_on(scope.pop()) {
            panic!("cave twins failed validation: {err}");
        }
    }

    #[test]
    #[ignore]
    fn shipped_cave_classes() {
        let path = &crate::test_game_dir().join("models/caves/cave.glb");
        let (doc, buffers, _) = gltf::import(path).expect("the shipped cave");
        let mut area = [0.0f64; 5];
        let mut count = [0usize; 5];
        for mesh in doc.meshes() {
            for prim in mesh.primitives() {
                let r = prim.reader(|b| Some(&buffers[b.index()]));
                let pos: Vec<Vec3> = r.read_positions().unwrap().map(Vec3::from).collect();
                let nor: Vec<Vec3> = r.read_normals().unwrap().map(Vec3::from).collect();
                let g: Vec<f32> = r
                    .read_tex_coords(1)
                    .map(|t| t.into_f32().map(|u| u[0]).collect())
                    .unwrap_or_else(|| vec![0.0; pos.len()]);
                let idx: Vec<u32> = r.read_indices().unwrap().into_u32().collect();
                for t in idx.chunks_exact(3) {
                    let [a, b, c] = [t[0], t[1], t[2]].map(|i| i as usize);
                    let mut k = classify([nor[a], nor[b], nor[c]], [g[a], g[b], g[c]]) as usize;
                    if k == 3 && [g[a], g[b], g[c]].iter().any(|&z| z > 0.0) {
                        k = 4;
                    }
                    area[k] += 0.5 * (pos[b] - pos[a]).cross(pos[c] - pos[a]).length() as f64;
                    count[k] += 1;
                }
            }
        }
        let total: f64 = area.iter().sum();
        for (k, name) in ["rock", "ground front", "ground back", "mixed", "mixed, some copy"].iter().enumerate() {
            println!("{name:>13}: {:6} triangles, {:5.1}% of the area", count[k], 100.0 * area[k] / total);
        }
    }

    #[test]
    #[ignore]
    fn cave_twin_sizes() {
        for (name, src) in [
            ("full", layered_mesh_shader()),
            ("rock", twin_source(CavePath::Rock).unwrap()),
            ("ground", twin_source(CavePath::Ground).unwrap()),
        ] {
            let src = crate::renderer::multiview::ViewMode::Mono.shader(src);
            println!("{}", crate::renderer::shader_inlining::report(name, &src, "fs_main", 6));
        }
    }
}
