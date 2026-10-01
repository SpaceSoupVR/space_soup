//! THIN PARTS: the wires, chain links and rims of a model that can be finer
//! than a pixel, kept from sparkling.
//!
//! A hanging lamp's cage is wire a few millimetres thick. From five metres it
//! is a third of an eye pixel wide, and 4x MSAA catches a third of a pixel on
//! zero, one or two of its four samples depending on exactly where the wire
//! falls -- so a step of the head by a millimetre redraws it as dots, then
//! dashes, then a line. Measured on the headset's own eye images (2026-10-01,
//! `tools/reference/highlight_stability.py`, head moved 1 mm at a time): the
//! reflections, floors and walls held still, and the bright cage wires of
//! every lamp beyond a couple of metres were the shimmer that was left.
//!
//! The fix is the one Persson gave for phone wires ("Phone-wire AA", 2011):
//! never draw a thin round part narrower than about two pixels, and fade it by
//! the ratio of its true width to the width drawn. Its light per unit length
//! is then exactly the true wire's, and two pixels of width meet the four
//! samples steadily enough that a millimetre of head movement no longer
//! decides whether the wire is there.
//!
//! What counts as thin is found from the mesh at load: the radius of
//! curvature at each vertex (chord over the change of normal between welded
//! neighbours -- exactly the radius on a circle), and a triangle is thin when
//! all three corners are and the triangle itself is no wider than a few such
//! radii. A tube, a chain link and the rolled rim of a shade pass; a flat
//! plate whose corners sit on its thin rim does not, because the plate's own
//! triangles are wide. The thin triangles are drawn separately (see
//! `mesh_pipeline::MeshPipeline::thin_pipeline`) after everything opaque and
//! the sky, so a faded wire blends over what is really behind it.

use bytemuck::{Pod, Zeroable};
use std::collections::HashMap;

use super::vertex::MeshVertex;

/// Thicker than this (a radius, metres) is never a thin part. At the far end
/// of a 20 m room an eye pixel is about 4 cm, so a 2 cm-wide part is already
/// half a pixel there; anything thicker has its samples to itself.
pub const THIN_RADIUS_MAX: f32 = 0.01;

/// A thin triangle is no wider (its least altitude) than this many radii of
/// its corners. A tube's facets are about one radius wide; a plate's are as
/// wide as the plate.
pub const THIN_ALTITUDE_FACTOR: f32 = 4.0;

/// Positions closer than this (metres) are one point of the surface: meshes
/// are split at every UV seam and, with a lightmap, at every corner.
const WELD: f32 = 1.0e-5;

/// Per vertex, what the thin pass needs to widen it: the surface's normal
/// there, averaged over every face that meets at the point (so the copies of a
/// split vertex move together and the tube stays closed), and the part's
/// radius there; 0 where the vertex is on no thin part.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct ThinVertex {
    pub normal: [f32; 3],
    pub radius: f32,
}

impl ThinVertex {
    pub const ATTRIBS: [wgpu::VertexAttribute; 1] = wgpu::vertex_attr_array![5 => Float32x4];

    pub fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<ThinVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

/// A primitive's triangles, the thin ones apart.
#[derive(Clone, Debug)]
pub struct ThinSplit {
    /// One per vertex of the primitive, in its order.
    pub vertices: Vec<ThinVertex>,
    /// The triangles drawn as ever.
    pub solid: Vec<u32>,
    /// The thin ones, drawn widened and faded.
    pub thin: Vec<u32>,
}

/// The split on the GPU.
#[derive(Clone)]
pub struct ThinParts {
    pub vertex_buffer: wgpu::Buffer,
    pub solid_index_buffer: wgpu::Buffer,
    pub solid_count: u32,
    pub thin_index_buffer: wgpu::Buffer,
    pub thin_count: u32,
}

impl ThinParts {
    pub fn upload(device: &wgpu::Device, split: &ThinSplit) -> Self {
        use wgpu::util::DeviceExt;
        let buffer = |label: &str, contents: &[u8], usage: wgpu::BufferUsages| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents,
                usage,
            })
        };
        // An empty solid list still needs a buffer to bind: one dummy index,
        // never drawn (count 0).
        let solid: &[u32] = if split.solid.is_empty() { &[0] } else { &split.solid };
        Self {
            vertex_buffer: buffer("thin_vb", bytemuck::cast_slice(&split.vertices), wgpu::BufferUsages::VERTEX),
            solid_index_buffer: buffer("thin_solid_ib", bytemuck::cast_slice(solid), wgpu::BufferUsages::INDEX),
            solid_count: split.solid.len() as u32,
            thin_index_buffer: buffer("thin_ib", bytemuck::cast_slice(&split.thin), wgpu::BufferUsages::INDEX),
            thin_count: split.thin.len() as u32,
        }
    }
}

/// The thin parts of one primitive, or `None` when it has none.
pub fn split_thin_parts(vertices: &[MeshVertex], indices: &[u32]) -> Option<ThinSplit> {
    let tris: Vec<[usize; 3]> = indices
        .chunks_exact(3)
        .map(|t| [t[0] as usize, t[1] as usize, t[2] as usize])
        .filter(|t| t.iter().all(|&i| i < vertices.len()))
        .collect();
    if tris.is_empty() {
        return None;
    }
    let pos = |i: usize| glam::Vec3::from(vertices[i].position);

    // Weld: one id per point of the surface.
    let mut ids: HashMap<[i64; 3], usize> = HashMap::new();
    let weld: Vec<usize> = (0..vertices.len())
        .map(|i| {
            let p = pos(i) / WELD;
            let key = [p.x.round() as i64, p.y.round() as i64, p.z.round() as i64];
            let next = ids.len();
            *ids.entry(key).or_insert(next)
        })
        .collect();
    let points = ids.len();
    let mut point = vec![glam::Vec3::ZERO; points];
    for i in 0..vertices.len() {
        point[weld[i]] = pos(i);
    }

    // The normal at each point: the faces round it, weighted by area -- the
    // SHAPE's, not the shading's, since a smooth-shaded plate's corner normals
    // lean out at 45 degrees and hide how thin its rim is. Turned to agree
    // with the authored normals there: winding says which way a face points
    // only in a right-handed frame, and a mirrored node (negative scale) or a
    // left-handed one reverses it. Unturned, the widening pulled a wire IN.
    let mut normal = vec![glam::Vec3::ZERO; points];
    let mut authored = vec![glam::Vec3::ZERO; points];
    for t in &tris {
        let [a, b, c] = t.map(|i| pos(i));
        let n = (b - a).cross(c - a);
        for &i in t {
            normal[weld[i]] += n;
        }
    }
    for (i, v) in vertices.iter().enumerate() {
        authored[weld[i]] += glam::Vec3::from(v.normal);
    }
    for (n, a) in normal.iter_mut().zip(&authored) {
        if n.dot(*a) < 0.0 {
            *n = -*n;
        }
    }
    // A point where opposite faces meet (a double-sided sheet) has no
    // direction to widen in: it is not round, whatever its curvature says.
    let flat: Vec<bool> = normal.iter().map(|n| n.length_squared() < 1e-24).collect();
    for n in normal.iter_mut() {
        *n = n.normalize_or_zero();
    }

    // The radius of curvature at each point: chord over the change of normal,
    // least over its edges. On a circle of radius R both are 2 sin(half the
    // angle) times R and 1, so the ratio is R exactly, however coarse the
    // facets. Along a straight tube the normal does not change: no bound.
    let mut radius = vec![f32::INFINITY; points];
    for t in &tris {
        for k in 0..3 {
            let (i, j) = (weld[t[k]], weld[t[(k + 1) % 3]]);
            if i == j || flat[i] || flat[j] {
                continue;
            }
            let turn = (normal[i] - normal[j]).length();
            if turn < 1e-3 {
                continue;
            }
            let r = (point[i] - point[j]).length() / turn;
            radius[i] = radius[i].min(r);
            radius[j] = radius[j].min(r);
        }
    }
    let thin_point = |i: usize| !flat[i] && radius[i] < THIN_RADIUS_MAX;

    let (mut solid, mut thin) = (Vec::new(), Vec::new());
    for t in &tris {
        let w = t.map(|i| weld[i]);
        let is_thin = w.iter().all(|&i| thin_point(i)) && {
            let [a, b, c] = w.map(|i| point[i]);
            let twice_area = (b - a).cross(c - a).length();
            let longest = (b - a).length().max((c - b).length()).max((a - c).length());
            // The least altitude is twice the area over the longest side.
            let least_altitude = twice_area / longest.max(1e-12);
            let widest = w.iter().map(|&i| radius[i]).fold(0.0f32, f32::max);
            least_altitude <= THIN_ALTITUDE_FACTOR * widest
        };
        let out = if is_thin { &mut thin } else { &mut solid };
        out.extend(t.iter().map(|&i| i as u32));
    }
    if thin.is_empty() {
        return None;
    }
    let vertices = (0..vertices.len())
        .map(|i| {
            let p = weld[i];
            ThinVertex {
                normal: normal[p].into(),
                radius: if thin_point(p) { radius[p] } else { 0.0 },
            }
        })
        .collect();
    Some(ThinSplit { vertices, solid, thin })
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    fn vertex(p: Vec3, n: Vec3) -> MeshVertex {
        MeshVertex { position: p.into(), normal: n.into(), uv: [0.0; 2], uv2: [0.0; 2], emissive: 0 }
    }

    /// A closed-sided tube along y: `sides` facets round, `rings` rings.
    /// `split`: every triangle gets its own three vertices, as a lightmapped
    /// mesh's do.
    fn tube(radius: f32, length: f32, sides: usize, rings: usize, split: bool) -> (Vec<MeshVertex>, Vec<u32>) {
        let at = |s: usize, r: usize| {
            let a = s as f32 / sides as f32 * std::f32::consts::TAU;
            let n = Vec3::new(a.cos(), 0.0, a.sin());
            (n * radius + Vec3::Y * (r as f32 / (rings - 1) as f32 * length), n)
        };
        let mut v = Vec::new();
        let mut idx = Vec::new();
        for r in 0..rings {
            for s in 0..sides {
                let (p, n) = at(s, r);
                v.push(vertex(p, n));
            }
        }
        for r in 0..rings - 1 {
            for s in 0..sides {
                let i = |s: usize, r: usize| (r * sides + s % sides) as u32;
                idx.extend([i(s, r), i(s, r + 1), i(s + 1, r), i(s + 1, r), i(s, r + 1), i(s + 1, r + 1)]);
            }
        }
        if split {
            let sv: Vec<MeshVertex> = idx.iter().map(|&i| v[i as usize]).collect();
            let si = (0..sv.len() as u32).collect();
            return (sv, si);
        }
        (v, idx)
    }

    #[test]
    fn a_wire_is_thin_and_its_radius_is_measured() {
        for split in [false, true] {
            let (v, i) = tube(0.0015, 0.3, 8, 5, split);
            let s = split_thin_parts(&v, &i).expect("a 1.5 mm wire is thin");
            assert!(s.solid.is_empty(), "every facet of a wire is thin (split {split})");
            assert_eq!(s.thin.len(), i.len());
            for t in &s.vertices {
                assert!((t.radius - 0.0015).abs() < 0.0015 * 0.02, "radius {} (split {split})", t.radius);
                let n = Vec3::from(t.normal);
                assert!((n.length() - 1.0).abs() < 1e-4 && n.y.abs() < 1e-4, "welded normal {n:?} is radial");
            }
        }
    }

    #[test]
    fn a_thick_pipe_and_a_round_bulb_are_not_thin() {
        let (v, i) = tube(0.03, 0.3, 12, 4, false);
        assert!(split_thin_parts(&v, &i).is_none(), "a 3 cm pipe has its samples to itself");
        // A bulb: a 4.5 cm sphere.
        let (mut v, mut idx) = (Vec::new(), Vec::new());
        let (lat, lon) = (12, 16);
        for a in 0..=lat {
            for b in 0..lon {
                let th = a as f32 / lat as f32 * std::f32::consts::PI;
                let ph = b as f32 / lon as f32 * std::f32::consts::TAU;
                let n = Vec3::new(th.sin() * ph.cos(), th.cos(), th.sin() * ph.sin());
                v.push(vertex(n * 0.045, n));
            }
        }
        for a in 0..lat {
            for b in 0..lon {
                let i = |a: usize, b: usize| (a * lon + b % lon) as u32;
                idx.extend([i(a, b), i(a + 1, b), i(a, b + 1), i(a, b + 1), i(a + 1, b), i(a + 1, b + 1)]);
            }
        }
        assert!(split_thin_parts(&v, &idx).is_none(), "a bulb is round but not thin");
    }

    #[test]
    fn a_thin_plate_keeps_its_faces_and_gives_up_its_rim() {
        // A 30 cm square plate 3 mm thick, its normals averaged at the corners
        // as an exporter's smooth shading leaves them: every corner sits on
        // the thin rim, but the faces are 30 cm wide and must stay solid.
        let (h, t) = (0.15f32, 0.0015f32);
        let corners: Vec<Vec3> = [(-h, -t, -h), (h, -t, -h), (h, -t, h), (-h, -t, h), (-h, t, -h), (h, t, -h), (h, t, h), (-h, t, h)]
            .iter()
            .map(|&(x, y, z)| Vec3::new(x, y, z))
            .collect();
        let v: Vec<MeshVertex> = corners.iter().map(|&p| vertex(p, p.normalize())).collect();
        let quads = [[0, 1, 2, 3], [7, 6, 5, 4], [0, 4, 5, 1], [1, 5, 6, 2], [2, 6, 7, 3], [3, 7, 4, 0]];
        let mut idx = Vec::new();
        for q in quads {
            idx.extend([q[0], q[1], q[2], q[0], q[2], q[3]]);
        }
        let s = split_thin_parts(&v, &idx).expect("the rim is thin");
        assert_eq!(s.solid.len(), 12, "both 30 cm faces stay solid");
        assert_eq!(s.thin.len(), 24, "the four rim sides are thin");
    }

    #[test]
    fn a_chain_link_is_thin() {
        // A torus: 6 mm round the middle, 1.5 mm thick.
        let (major, minor, a_n, b_n) = (0.006f32, 0.0015f32, 16, 8);
        let mut v = Vec::new();
        let mut idx = Vec::new();
        for a in 0..a_n {
            let pa = a as f32 / a_n as f32 * std::f32::consts::TAU;
            let centre = Vec3::new(pa.cos(), 0.0, pa.sin()) * major;
            for b in 0..b_n {
                let pb = b as f32 / b_n as f32 * std::f32::consts::TAU;
                let n = Vec3::new(pa.cos() * pb.cos(), pb.sin(), pa.sin() * pb.cos());
                v.push(vertex(centre + n * minor, n));
            }
        }
        for a in 0..a_n {
            for b in 0..b_n {
                let i = |a: usize, b: usize| ((a % a_n) * b_n + b % b_n) as u32;
                idx.extend([i(a, b), i(a + 1, b), i(a, b + 1), i(a, b + 1), i(a + 1, b), i(a + 1, b + 1)]);
            }
        }
        let s = split_thin_parts(&v, &idx).expect("a chain link is thin");
        assert!(s.solid.is_empty());
        let worst = s.vertices.iter().map(|t| (t.radius - minor).abs() / minor).fold(0.0f32, f32::max);
        assert!(worst < 0.05, "the link's radius is its wire's, within {:.0}%", worst * 100.0);
    }
}

/// The real fixtures' thin parts, from the files the game ships. Skipped
/// where the game directory is not beside this crate (it is published alone).
/// `THIN_OBJ_DIR=<dir>` also writes each lamp's thin and solid triangles as
/// OBJ, to look at.
#[cfg(test)]
mod shipped_lamps {
    use super::*;
    use glam::{Mat4, Vec3};

    fn load(path: &std::path::Path) -> Vec<(Vec<MeshVertex>, Vec<u32>)> {
        let (doc, buffers, _) = gltf::import(path).expect("the lamp loads");
        let mut out = Vec::new();
        fn walk(node: gltf::Node, parent: Mat4, buffers: &[gltf::buffer::Data], out: &mut Vec<(Vec<MeshVertex>, Vec<u32>)>) {
            let world = parent * Mat4::from_cols_array_2d(&node.transform().matrix());
            if let Some(mesh) = node.mesh() {
                for prim in mesh.primitives() {
                    let transmission = prim.material().transmission().map(|t| t.transmission_factor()).unwrap_or(0.0);
                    if transmission > 0.0 {
                        continue;
                    }
                    let r = prim.reader(|b| Some(&buffers[b.index()]));
                    let p: Vec<Vec3> = r.read_positions().unwrap().map(|v| world.transform_point3(Vec3::from(v))).collect();
                    let n: Vec<Vec3> = r.read_normals().unwrap().map(|v| world.transform_vector3(Vec3::from(v)).normalize()).collect();
                    let idx: Vec<u32> = r.read_indices().map(|i| i.into_u32().collect()).unwrap_or_else(|| (0..p.len() as u32).collect());
                    let v = p.iter().zip(&n).map(|(p, n)| MeshVertex { position: (*p).into(), normal: (*n).into(), uv: [0.0; 2], uv2: [0.0; 2], emissive: 0 }).collect();
                    out.push((v, idx));
                }
            }
            for c in node.children() {
                walk(c, world, buffers, out);
            }
        }
        for scene in doc.scenes() {
            for node in scene.nodes() {
                walk(node, Mat4::IDENTITY, &buffers, &mut out);
            }
        }
        out
    }

    fn area(v: &[MeshVertex], idx: &[u32]) -> f32 {
        idx.chunks_exact(3)
            .map(|t| {
                let [a, b, c] = [t[0], t[1], t[2]].map(|i| Vec3::from(v[i as usize].position));
                (b - a).cross(c - a).length() * 0.5
            })
            .sum()
    }

    #[test]
    fn the_lamps_thin_parts_are_a_small_share_of_them() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../game/models/lights");
        for name in ["hanging_industrial_lamp/hanging_industrial_lamp_1k.gltf", "industrial_wall_sconce/industrial_wall_sconce_1k.gltf"] {
            let path = root.join(name);
            if !path.exists() {
                eprintln!("THIN: {} not here, skipped", path.display());
                continue;
            }
            let (mut tri, mut thin_tri, mut a_all, mut a_thin) = (0usize, 0usize, 0.0f32, 0.0f32);
            let mut obj_thin = String::new();
            let mut obj_solid = String::new();
            let (mut base_t, mut base_s) = (1usize, 1usize);
            for (v, idx) in load(&path) {
                tri += idx.len() / 3;
                a_all += area(&v, &idx);
                if let Some(s) = split_thin_parts(&v, &idx) {
                    thin_tri += s.thin.len() / 3;
                    a_thin += area(&v, &s.thin);
                    for (list, obj, base) in [(&s.thin, &mut obj_thin, &mut base_t), (&s.solid, &mut obj_solid, &mut base_s)] {
                        for t in list.chunks_exact(3) {
                            for &i in t {
                                let p = v[i as usize].position;
                                obj.push_str(&format!("v {} {} {}\n", p[0], p[1], p[2]));
                            }
                            obj.push_str(&format!("f {} {} {}\n", *base, *base + 1, *base + 2));
                            *base += 3;
                        }
                    }
                }
            }
            eprintln!("THIN: {name}: {thin_tri} of {tri} triangles thin, {:.1}% of the area", 100.0 * a_thin / a_all);
            if let Ok(dir) = std::env::var("THIN_OBJ_DIR") {
                let stem = name.split('/').next().unwrap();
                std::fs::write(format!("{dir}/{stem}_thin.obj"), &obj_thin).unwrap();
                std::fs::write(format!("{dir}/{stem}_solid.obj"), &obj_solid).unwrap();
            }
            assert!(thin_tri > 0, "{name}: its cage is thin");
            assert!(a_thin < 0.25 * a_all, "{name}: a lamp is mostly not wire");
        }
    }
}
