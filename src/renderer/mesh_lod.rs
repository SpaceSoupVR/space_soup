//! MODEL LODS (the colleague's plan part 1): coarser index lists over a
//! model's own vertices, each with the geometric error it may show, so a draw
//! can take the coarsest whose error projects under half a pixel.
//!
//! Quality first: a LOD is only taken where its error is smaller than what
//! the eye's pixels can resolve, so a switch never pops. Every LOD keeps the
//! model's vertex buffer and indexes a subset of its vertices (half-edge
//! collapse: a vertex is folded onto a neighbour that already exists). The
//! vertices the coarse list keeps carry their exact position, normal, UVs and
//! lightmap chart. Vertices on an open border or a seam (one position shared
//! by several vertices: a UV split, a lightmap chart edge, a hard normal)
//! never move, so no crack opens and no chart bleeds. A fold that would turn
//! any of the triangles it changes over by more than about 80 degrees is
//! refused.
//!
//! The error is conservative: a fold costs the square root of its quadric,
//! the sum of squared distances to every original plane the vertex carried,
//! which is never less than the largest of those distances. A LOD's error is
//! the largest fold it took.

use glam::{DVec3, Vec3};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

/// One coarser index list and the most it may differ from the model, metres
/// in the model's own space.
#[derive(Clone, Debug)]
pub struct Lod {
    pub indices: Vec<u32>,
    pub error: f32,
}

/// Each LOD keeps at most this share of the triangles of the one before.
pub const LOD_RATIO: f32 = 0.5;
/// No more LODs than this.
pub const MAX_LODS: usize = 4;
/// A model with fewer triangles has no LODs: there is nothing to buy.
pub const MIN_TRIANGLES: usize = 256;
/// A LOD that keeps more than this share of the one before is not kept: the
/// mesh is too locked down (seams everywhere) to be worth another list.
const KEEP_AT_MOST: f32 = 0.85;
/// A fold may not turn a changed triangle's normal past this (cos ~80 deg).
const FLIP_COS: f64 = 0.17;

#[derive(Clone, Copy, Default)]
struct Quadric([f64; 10]);

impl Quadric {
    fn plane(n: DVec3, d: f64) -> Self {
        let (a, b, c) = (n.x, n.y, n.z);
        Self([a * a, a * b, a * c, a * d, b * b, b * c, b * d, c * c, c * d, d * d])
    }

    fn add(&mut self, o: &Quadric) {
        for (x, y) in self.0.iter_mut().zip(o.0.iter()) {
            *x += y;
        }
    }

    fn error(&self, p: DVec3) -> f64 {
        let q = &self.0;
        let (x, y, z) = (p.x, p.y, p.z);
        (q[0] * x * x + 2.0 * q[1] * x * y + 2.0 * q[2] * x * z + 2.0 * q[3] * x
            + q[4] * y * y + 2.0 * q[5] * y * z + 2.0 * q[6] * y
            + q[7] * z * z + 2.0 * q[8] * z
            + q[9])
            .max(0.0)
    }
}

#[derive(PartialEq)]
struct Fold {
    cost: f64,
    from: u32,
    to: u32,
    stamp: u32,
}

impl Eq for Fold {}

impl Ord for Fold {
    fn cmp(&self, o: &Self) -> Ordering {
        o.cost.partial_cmp(&self.cost).unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for Fold {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}

struct Mesh {
    pos: Vec<DVec3>,
    tris: Vec<[u32; 3]>,
    alive: Vec<bool>,
    /// Each vertex's live triangles (by index; dead ones dropped lazily).
    of: Vec<Vec<u32>>,
    quadric: Vec<Quadric>,
    locked: Vec<bool>,
    stamp: Vec<u32>,
    live: usize,
}

fn normal(a: DVec3, b: DVec3, c: DVec3) -> DVec3 {
    (b - a).cross(c - a)
}

impl Mesh {
    fn new(positions: &[Vec3], indices: &[u32]) -> Self {
        let pos: Vec<DVec3> = positions.iter().map(glam::Vec3::as_dvec3).collect();
        let n = pos.len();
        let mut tris = Vec::with_capacity(indices.len() / 3);
        for t in indices.chunks_exact(3) {
            if t[0] != t[1] && t[1] != t[2] && t[0] != t[2] && t.iter().all(|&i| (i as usize) < n) {
                tris.push([t[0], t[1], t[2]]);
            }
        }
        let mut of = vec![Vec::new(); n];
        let mut quadric = vec![Quadric::default(); n];
        for (k, t) in tris.iter().enumerate() {
            let nn = normal(pos[t[0] as usize], pos[t[1] as usize], pos[t[2] as usize]);
            let len = nn.length();
            let q = if len > 1e-18 {
                let u = nn / len;
                Quadric::plane(u, -u.dot(pos[t[0] as usize]))
            } else {
                Quadric::default()
            };
            for &v in t {
                of[v as usize].push(k as u32);
                quadric[v as usize].add(&q);
            }
        }
        // Locked: a seam (a position more than one vertex shares) or an open
        // border (an edge only one triangle uses).
        let mut locked = vec![false; n];
        let mut at: HashMap<[u64; 3], u32> = HashMap::new();
        for (i, p) in positions.iter().enumerate() {
            let key = [u64::from(p.x.to_bits()), u64::from(p.y.to_bits()), u64::from(p.z.to_bits())];
            if let Some(&j) = at.get(&key) {
                locked[i] = true;
                locked[j as usize] = true;
            } else {
                at.insert(key, i as u32);
            }
        }
        let mut edges: HashMap<(u32, u32), u32> = HashMap::new();
        for t in &tris {
            for e in 0..3 {
                let (a, b) = (t[e], t[(e + 1) % 3]);
                *edges.entry((a.min(b), a.max(b))).or_insert(0) += 1;
            }
        }
        for (&(a, b), &count) in &edges {
            if count != 2 {
                locked[a as usize] = true;
                locked[b as usize] = true;
            }
        }
        let live = tris.len();
        Self { alive: vec![true; tris.len()], pos, tris, of, quadric, locked, stamp: vec![0; n], live }
    }

    fn neighbours(&self, v: u32) -> Vec<u32> {
        let mut out: Vec<u32> = Vec::new();
        for &t in &self.of[v as usize] {
            if !self.alive[t as usize] {
                continue;
            }
            for &w in &self.tris[t as usize] {
                if w != v && !out.contains(&w) {
                    out.push(w);
                }
            }
        }
        out
    }

    /// The cost of folding `from` onto `to`, or None if the fold would turn a
    /// triangle over.
    fn cost(&self, from: u32, to: u32) -> Option<f64> {
        let target = self.pos[to as usize];
        for &t in &self.of[from as usize] {
            if !self.alive[t as usize] {
                continue;
            }
            let tri = self.tris[t as usize];
            if tri.contains(&to) {
                continue;
            }
            let p = tri.map(|i| self.pos[i as usize]);
            let before = normal(p[0], p[1], p[2]);
            let moved = tri.map(|i| if i == from { target } else { self.pos[i as usize] });
            let after = normal(moved[0], moved[1], moved[2]);
            let (lb, la) = (before.length(), after.length());
            if la < 1e-18 || (lb > 1e-18 && before.dot(after) < FLIP_COS * lb * la) {
                return None;
            }
        }
        Some(self.quadric[from as usize].error(target).sqrt())
    }

    fn push_folds(&self, v: u32, heap: &mut BinaryHeap<Fold>) {
        if self.locked[v as usize] {
            return;
        }
        for to in self.neighbours(v) {
            if let Some(cost) = self.cost(v, to) {
                heap.push(Fold { cost, from: v, to, stamp: self.stamp[v as usize] });
            }
        }
    }

    fn fold(&mut self, from: u32, to: u32) {
        let ts = std::mem::take(&mut self.of[from as usize]);
        for t in ts {
            if !self.alive[t as usize] {
                continue;
            }
            let tri = &mut self.tris[t as usize];
            if tri.contains(&to) {
                self.alive[t as usize] = false;
                self.live -= 1;
                continue;
            }
            for i in tri.iter_mut() {
                if *i == from {
                    *i = to;
                }
            }
            self.of[to as usize].push(t);
        }
        let q = self.quadric[from as usize];
        self.quadric[to as usize].add(&q);
        self.locked[from as usize] = true;
    }

    fn indices(&self) -> Vec<u32> {
        self.tris.iter().zip(&self.alive).filter(|(_, a)| **a).flat_map(|(t, _)| t.iter().copied()).collect()
    }
}

/// The LOD chain of a triangle list over `positions`: each list keeps at most
/// `LOD_RATIO` of the triangles of the one before, coarsest last. Empty for a
/// small or unfoldable model.
pub fn lod_chain(positions: &[Vec3], indices: &[u32]) -> Vec<Lod> {
    let mut mesh = Mesh::new(positions, indices);
    if mesh.live < MIN_TRIANGLES {
        return Vec::new();
    }
    let mut heap = BinaryHeap::new();
    for v in 0..positions.len() as u32 {
        mesh.push_folds(v, &mut heap);
    }
    let mut chain = Vec::new();
    let mut worst = 0.0f64;
    let mut before = mesh.live;
    while chain.len() < MAX_LODS {
        let target = ((before as f32) * LOD_RATIO) as usize;
        while mesh.live > target {
            let Some(f) = heap.pop() else { break };
            if mesh.locked[f.from as usize] || f.stamp != mesh.stamp[f.from as usize] {
                continue;
            }
            // The neighbourhood may have moved since this fold was costed.
            let Some(cost) = mesh.cost(f.from, f.to) else { continue };
            if cost > f.cost * 1.0001 + 1e-12 {
                heap.push(Fold { cost, ..f });
                continue;
            }
            worst = worst.max(cost);
            let touched = mesh.neighbours(f.from);
            mesh.fold(f.from, f.to);
            for v in touched {
                mesh.stamp[v as usize] = mesh.stamp[v as usize].wrapping_add(1);
                mesh.push_folds(v, &mut heap);
            }
        }
        if mesh.live as f32 > before as f32 * KEEP_AT_MOST || mesh.live == 0 {
            break;
        }
        chain.push(Lod { indices: mesh.indices(), error: worst as f32 });
        before = mesh.live;
        if heap.is_empty() {
            break;
        }
    }
    chain
}

/// Which list to draw: the coarsest whose error, seen from `distance` metres
/// at `pixels_per_radian`, stays under `max_pixels`. `scale` is the model's
/// largest scale (errors are in its own space). None is the full mesh.
pub fn pick(lods: &[f32], scale: f32, distance: f32, pixels_per_radian: f32, max_pixels: f32) -> Option<usize> {
    let d = distance.max(1e-3);
    lods.iter().rposition(|&e| e * scale / d * pixels_per_radian <= max_pixels)
}

/// A LOD's error is taken this many times larger when choosing: the
/// quadric bounds how far a kept vertex sits off the old planes, not every
/// point of a new face, so the half-pixel rule keeps a margin.
pub const ERROR_MARGIN: f32 = 2.0;

/// One LOD on the GPU: its index buffer (over the model's own vertex buffer),
/// index count and error.
pub struct GpuLod {
    pub buffer: wgpu::Buffer,
    pub count: u32,
    pub error: f32,
}

/// THE RENDERER'S LODS, by the index buffer of the primitive they belong
/// to: built off the render thread the first time a primitive is drawn
/// (`lods`), uploaded a few a frame (`poll`), so loading a level costs
/// nothing more and a model draws whole until its LODs are in.
pub struct LodCache {
    entries: HashMap<wgpu::Buffer, Vec<GpuLod>>,
    pending: std::collections::HashSet<wgpu::Buffer>,
    requests: std::sync::mpsc::Sender<(wgpu::Buffer, Vec<Vec3>, Vec<u32>)>,
    results: std::sync::mpsc::Receiver<(wgpu::Buffer, Vec<Lod>)>,
}

/// LOD index buffers made per frame at most.
pub const UPLOADS_PER_FRAME: usize = 4;

impl Default for LodCache {
    fn default() -> Self {
        let (req_tx, req_rx) = std::sync::mpsc::channel::<(wgpu::Buffer, Vec<Vec3>, Vec<u32>)>();
        let (res_tx, res_rx) = std::sync::mpsc::channel();
        std::thread::Builder::new()
            .name("mesh_lods".into())
            .spawn(move || {
                while let Ok((key, pos, idx)) = req_rx.recv() {
                    let t = std::time::Instant::now();
                    let chain = lod_chain(&pos, &idx);
                    if !chain.is_empty() {
                        log::info!(
                            "MESHLOD {} triangles -> {:?} in {:.1} ms, errors {:?} m",
                            idx.len() / 3,
                            chain.iter().map(|l| l.indices.len() / 3).collect::<Vec<_>>(),
                            t.elapsed().as_secs_f32() * 1000.0,
                            chain.iter().map(|l| l.error).collect::<Vec<_>>(),
                        );
                    }
                    if res_tx.send((key, chain)).is_err() {
                        break;
                    }
                }
            })
            .ok();
        Self { entries: HashMap::new(), pending: Default::default(), requests: req_tx, results: res_rx }
    }
}

impl LodCache {
    /// Upload the LODs the worker has finished, a few a frame.
    pub fn poll(&mut self, device: &wgpu::Device) {
        use wgpu::util::DeviceExt;
        for _ in 0..UPLOADS_PER_FRAME {
            let Ok((key, chain)) = self.results.try_recv() else { break };
            self.pending.remove(&key);
            let gpu = chain
                .into_iter()
                .map(|l| GpuLod {
                    buffer: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("mesh_lod"),
                        contents: bytemuck::cast_slice(&l.indices),
                        usage: wgpu::BufferUsages::INDEX,
                    }),
                    count: l.indices.len() as u32,
                    error: l.error,
                })
                .collect();
            self.entries.insert(key, gpu);
        }
    }

    /// The LODs of the primitive drawn from `indices` over `positions`; the
    /// first call asks the worker for them and answers none.
    pub fn lods(&mut self, key: &wgpu::Buffer, positions: impl FnOnce() -> Vec<Vec3>, indices: &[u32]) -> &[GpuLod] {
        if !self.entries.contains_key(key) && !self.pending.contains(key) {
            if indices.len() / 3 >= MIN_TRIANGLES && self.requests.send((key.clone(), positions(), indices.to_vec())).is_ok() {
                self.pending.insert(key.clone());
            } else {
                self.entries.insert(key.clone(), Vec::new());
            }
        }
        self.entries.get(key).map_or(&[], Vec::as_slice)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A sphere of `rings` x `segments`, welded (no seams) so it can fold.
    fn sphere(rings: u32, segments: u32) -> (Vec<Vec3>, Vec<u32>) {
        let mut pos = vec![Vec3::Y];
        for r in 1..rings {
            let th = std::f32::consts::PI * r as f32 / rings as f32;
            for s in 0..segments {
                let ph = std::f32::consts::TAU * s as f32 / segments as f32;
                pos.push(Vec3::new(th.sin() * ph.cos(), th.cos(), th.sin() * ph.sin()));
            }
        }
        pos.push(Vec3::NEG_Y);
        let ring = |r: u32, s: u32| 1 + (r - 1) * segments + s % segments;
        let mut idx = Vec::new();
        for s in 0..segments {
            idx.extend([0, ring(1, s + 1), ring(1, s)]);
        }
        for r in 1..rings - 1 {
            for s in 0..segments {
                let (a, b, c, d) = (ring(r, s), ring(r, s + 1), ring(r + 1, s), ring(r + 1, s + 1));
                idx.extend([a, b, c, b, d, c]);
            }
        }
        let last = pos.len() as u32 - 1;
        for s in 0..segments {
            idx.extend([last, ring(rings - 1, s), ring(rings - 1, s + 1)]);
        }
        (pos, idx)
    }

    #[test]
    fn a_sphere_halves_per_lod_and_its_error_bounds_how_far_it_moved() {
        let (pos, idx) = sphere(32, 48);
        let chain = lod_chain(&pos, &idx);
        assert!(chain.len() >= 3, "{} lods", chain.len());
        let mut before = idx.len();
        for lod in &chain {
            assert!(lod.indices.len() <= before * 55 / 100, "{} of {before}", lod.indices.len());
            before = lod.indices.len();
            // Every kept triangle's vertices lie on the unit sphere (they are
            // original vertices), and its plane stays within the error of it.
            for t in lod.indices.chunks_exact(3) {
                let c = (pos[t[0] as usize] + pos[t[1] as usize] + pos[t[2] as usize]) / 3.0;
                assert!(1.0 - c.length() <= lod.error + 1e-4, "a face sinks {} past error {}", 1.0 - c.length(), lod.error);
            }
        }
        let errors: Vec<f32> = chain.iter().map(|l| l.error).collect();
        assert!(errors.windows(2).all(|w| w[0] <= w[1]), "{errors:?}");
        assert!(errors[0] < 0.05, "the first lod moves the sphere little: {errors:?}");
    }

    #[test]
    fn seams_and_borders_never_move() {
        // A flat grid with an open border: every border vertex stays.
        let n = 24u32;
        let pos: Vec<Vec3> = (0..=n).flat_map(|j| (0..=n).map(move |i| Vec3::new(i as f32, 0.0, j as f32))).collect();
        let mut idx = Vec::new();
        for j in 0..n {
            for i in 0..n {
                let a = j * (n + 1) + i;
                idx.extend([a, a + n + 1, a + 1, a + 1, a + n + 1, a + n + 2]);
            }
        }
        let chain = lod_chain(&pos, &idx);
        assert!(!chain.is_empty());
        let coarse = &chain[chain.len() - 1];
        assert!(coarse.error < 1e-3, "a plane folds with no error: {}", coarse.error);
        for k in 0..=n {
            for v in [k, n * (n + 1) + k, k * (n + 1), k * (n + 1) + n] {
                assert!(coarse.indices.contains(&v), "border vertex {v} was folded away");
            }
        }
    }

    #[test]
    #[ignore]
    fn lod_cost() {
        let (pos, idx) = sphere(160, 320);
        let t = std::time::Instant::now();
        let chain = lod_chain(&pos, &idx);
        eprintln!(
            "lod chain of {} triangles: {:?} in {:.1} ms",
            idx.len() / 3,
            chain.iter().map(|l| (l.indices.len() / 3, l.error)).collect::<Vec<_>>(),
            t.elapsed().as_secs_f64() * 1e3
        );
    }

    #[test]
    fn a_small_model_has_no_lods() {
        let (pos, idx) = sphere(6, 8);
        assert!(lod_chain(&pos, &idx).is_empty());
    }

    #[test]
    fn the_coarsest_list_under_half_a_pixel_is_picked() {
        let errors = [0.001, 0.004, 0.02];
        // 800 px a radian: at 1 m an error of 1 mm is 0.8 px.
        assert_eq!(pick(&errors, 1.0, 1.0, 800.0, 0.5), None);
        assert_eq!(pick(&errors, 1.0, 2.0, 800.0, 0.5), Some(0));
        assert_eq!(pick(&errors, 1.0, 10.0, 800.0, 0.5), Some(1));
        assert_eq!(pick(&errors, 1.0, 100.0, 800.0, 0.5), Some(2));
        assert_eq!(pick(&errors, 4.0, 10.0, 800.0, 0.5), Some(0), "a model scaled up shows its error larger");
    }
}
