//! THE GROUND IN CHUNKS OF A FIXED SIZE, CULLED THROUGH A QUADTREE, AND DRAWN
//! COARSER WITH DISTANCE (`Levers::ground_lod`, OFF).
//!
//! `docs/terrain-lod-and-sectors-design-2026-10-13.md`, section 1, steps 1
//! and 2 of its section 4. Status: `docs/ground-lod-status-2026-10-13.md`.
//!
//! STEP 1, always: the ground's triangles are cut into chunks of
//! [`CHUNK_CELLS`] cells a side (13.75 m on test_room's 0.43 m cells) -- 8 a
//! side on test_room as before, 20 at 625 samples, 64 at 2,049 -- laid in the
//! index buffer in MORTON order of the chunk grid, so a square block of
//! chunks is one stretch of the buffer. Over them stands a quadtree (each
//! node the box of its children): a frame asks it for the chunks its frusta
//! touch ([`GroundLod::select`]) and never walks the level.
//!
//! STEP 2, when the lists were built ([`LISTS`]) and the lever is on: a node
//! of the tree at level `L` (a square of `2^L` chunks a side) can be drawn as
//! ONE run of triangles at step `2^L` (1, 2, 4, 8, 16 samples) over the SAME
//! vertices -- a coarse level indexes a subset of the level's one vertex
//! buffer, so a vertex's height, normal and splat uv are the same at every
//! level. (AT the vertices. Between them a coarser triangle interpolates
//! fewer normals, and the ground is lit by its vertices' normals: measured
//! offline, lighting along rims changes even where the surface stays within
//! a pixel. The normal has to come from a texture before a coarser step is
//! the same picture -- `docs/ground-lod-status-2026-10-13.md`.)
//! Per chunk that is the design's five lists (2,048 / 512 / 128 / 32 /
//! 8 triangles); they are stored BY NODE because a run the slope split sorts
//! (`ground_twins::SlopeSplit`) is then a whole block of far chunks: at most
//! three draws for 16 x 16 chunks at step 16, where a list a chunk would be
//! 768.
//!
//! WHICH LEVEL: a node is drawn whole when the head is no nearer its box
//! than its `split_under` distance, else its four children are asked. That
//! distance is the design's table and error rule
//! ([`BAND_START`], [`FORCED`], [`PIXEL_RAD`], [`HOLE_FLOOR`]) made
//! RESTRICTED at load: raised until a node cannot be whole beside a
//! neighbour two levels finer ([`GroundLod::build`]), so neighbours never
//! differ by more than one level and nothing has to be propagated in a
//! frame. It follows the head's POSITION only, never where it looks: a node
//! out of every frustum has the level it would be drawn at, so turning the
//! head changes no triangle.
//!
//! BORDERS: the finer of two neighbours gives way. A node's list is its
//! `core` (everything but its outer ring of quads) and four side strips, each
//! in two forms: `plain` (the ring's own triangles) and `stitch` (the same
//! strip with every other vertex of the outer edge left out, which is the
//! coarser neighbour's edge exactly: both index the same vertices, so there
//! is no T-junction and no "nearly equal" position). A node with no coarser
//! neighbour draws its `whole` list -- for a chunk at step 1 the very
//! triangles the lever-off frame draws.
//!
//! A chunk with a hole or a cave's ground patch is drawn at step 1 within
//! [`HOLE_FLOOR`], its holes open and its patch in; past that, in a coarser
//! node, the holes are CLOSED and the patch is left out. Such a chunk has no
//! strips (its triangles are not the lattice's): beside a coarser neighbour
//! it draws its `whole` list and, where its border has a vertex the
//! neighbour skips, the sliver between the two edges -- the gap itself, on
//! shared vertices.

use crate::renderer::shadow::{aabb_in_frustum, CasterChunk};
use glam::{Vec3, Vec4};

/// Cells along a side of a chunk: test_room's own chunk (256 cells in 8).
pub const CHUNK_CELLS: u32 = 32;
/// The coarsest level drawn: step 16.
pub const MAX_LEVEL: usize = 4;
/// A pixel of the headset, radians: the error a coarser level may show.
pub const PIXEL_RAD: f32 = 1.2e-3;
/// THE ERROR A COARSER LEVEL MAY SHOW, radians, as a level's load builds its
/// tree: [`PIXEL_RAD`] unless changed (f32 bits). MEASUREMENT: how far a
/// coarser ground may stand from step 1 before the picture changes, now that
/// its light does not (`ground_shape`): the offline frame's `GROUND_LOD_MRAD`,
/// the scale bench's `BUDGET_LOD_MRAD`, the headset's `adb shell setprop
/// debug.spacesoup.groundlodmrad 2.4` before a launch.
pub static TOLERANCE: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0x3a9d_4952);
/// The cell the distances below were worked out for (test_room's), metres.
pub const REFERENCE_CELL: f32 = 110.0 / 256.0;
/// From how far a node at each level MAY be drawn whole, metres from the
/// head to the nearest point of its box, on [`REFERENCE_CELL`] cells: step 2
/// from 30 m, 4 from 60, 8 from 120, 16 from 240.
pub const BAND_START: [f32; MAX_LEVEL + 1] = [0.0, 30.0, 60.0, 120.0, 240.0];
/// Past this a node at each level is drawn whole WHATEVER its error: step 1
/// ends at 120 m and step 2 at 240 (the design's "finest step allowed").
pub const FORCED: [f32; MAX_LEVEL + 1] = [0.0, 120.0, 240.0, f32::INFINITY, f32::INFINITY];
/// Within this a chunk with a hole or a cave's ground patch is at step 1.
pub const HOLE_FLOOR: f32 = 60.0;
/// A node once split joins again only this much farther out.
pub const HYSTERESIS: f32 = 0.10;
/// How far a posed box is let out, metres (the turn's arithmetic).
pub const BOX_SLACK: f32 = 1e-4;

/// WHETHER A LEVEL'S LOAD BUILDS THE COARSER LISTS. Off: the index buffer is
/// step 1's alone and the lever can do nothing. On (the default since
/// 2026-10-14; `adb shell setprop debug.spacesoup.groundlod 0` before a
/// launch turns it off; the offline frame and the budget harness set it from
/// `GROUND_LOD` / `BUDGET_GROUND_LOD`): the lists are in the buffer (about
/// 1.4 times step 1's indices with sliver borders) and `Levers::ground_lod`
/// draws by them.
pub static LISTS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

/// THE SLOPE SPLIT LAYS A BLOCK OF CHUNKS BUCKET BY BUCKET (`true`): every
/// gentle triangle of a block of [`BLOCK_CHUNKS`] x [`BLOCK_CHUNKS`] step-1
/// chunks, chunk by chunk, then every mixed one, then every steep one (each
/// by its ground class), where each chunk was its own gentle, mixed and steep
/// runs. A block in view is then one run a reader -- three draws in the scene
/// pass for 16 chunks, where they were up to 48 -- and a chunk is drawn as
/// its pieces ([`GroundLod::frame`]). `false`: each chunk its own runs, as
/// until 2026-10-14. Read when the split is made (a level's load):
/// `adb shell setprop debug.spacesoup.groundblocks 0` before a launch.
pub static BLOCK_ORDER: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
/// A NODE BESIDE A COARSER ONE IS DRAWN WHOLE, ITS BORDER CLOSED BY SLIVERS
/// (`true`): for each vertex of its edge the neighbour skips, the thin
/// triangle between it and the neighbour's edge (`sliver_lists`, as chunks
/// with holes always were) -- not by its core and four strips, one of them
/// stitched. On shared vertices, so watertight the same way. A node is then
/// its whole list, laid with its block's (`BLOCK_ORDER`), and a sliver run
/// a stitched side: the ring of step-1 chunks round a coarser node was five
/// runs a chunk, each split three ways by slope (the 625 hilltop drew 935
/// scene draws with the lever on against 128 off). And the lists hold no
/// cores or strips: about half the memory. Read when a level's tree is built.
pub static SLIVER_BORDERS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

/// Chunks a side of a block ([`BLOCK_ORDER`]): 4, 55 m on test_room's cells.
/// A block number that says: a sliver run, laid as ONE bucket -- the full
/// reader's, slope and ground class alike -- so the stitched sides along one
/// edge, which lie one after another, are one draw. They are a few thin
/// triangles each.
pub const SLIVER_RUN: u32 = u32::MAX - 1;
pub const BLOCK_CHUNKS: u32 = 4;

/// MEASUREMENT (the offline frame's `GROUND_ORDER=rows`): the chunks laid in
/// the buffer row by row, as they were until 2026-10-13, not in Morton order
/// -- the control that shows what the order alone does to a picture (the
/// offline frame draws the ground as one run, and its pixels at shared edges
/// follow the order the triangles come in).
pub static ROW_ORDER: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

const NONE: u32 = u32::MAX;
/// Sides of a node: west (low x), east, south (low z), north.
pub const SIDES: usize = 4;

/// What the lists are built from: the ground's step-1 lattice.
pub struct GridInput<'a> {
    /// Cells along x and z.
    pub cells: [u32; 2],
    /// Every vertex in the WORLD, as drawn: the lattice's first, sample
    /// (`ix`, `iz`) at `iz * (cells[0] + 1) + ix`, then any others (a cave's
    /// ground patch).
    pub positions: &'a [Vec3],
    /// The step-1 triangles that are drawn, in the mesher's order.
    pub triangles: &'a [u32],
    /// Whether cell (`ix`, `iz`) is a hole; `None` for a ground without any.
    pub open: Option<&'a dyn Fn(u32, u32) -> bool>,
    /// Cells a side of a chunk ([`CHUNK_CELLS`]).
    pub chunk_cells: u32,
    /// Whether to build the coarser lists ([`LISTS`]).
    pub lists: bool,
}

/// One run of the index buffer the slope split sorts and a pass draws.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Unit {
    pub first: u32,
    pub count: u32,
    /// The node it is a list of.
    pub node: u32,
}

/// A node of the quadtree: at level 0 a chunk, at level `L` a square of
/// `2^L` chunks a side.
#[derive(Clone, Debug)]
pub struct Node {
    pub level: u8,
    /// Its column and row in its level's grid.
    pub at: [u32; 2],
    /// Its cells: `x0, z0, x1, z1`.
    pub cells: [u32; 4],
    /// Its box in the WORLD, over every step-1 vertex in it (and a patch's):
    /// a coarser list's vertices are some of those, so it holds every level.
    pub lo: Vec3,
    pub hi: Vec3,
    /// How far its surface at its own step stands from the step-1 samples,
    /// metres ([`GroundLod::error_of`]).
    pub error: f32,
    /// Nearer than this (head to box) it is split into its children.
    pub split_under: f32,
    /// Whether it holds a hole cell or a triangle of a cave's ground patch.
    pub holes: bool,
    /// Whether it holds a chunk held at step 1 ([`GroundLod::hold`]): never
    /// drawn whole above level 0.
    pub held: bool,
    /// Its border toward a coarser neighbour is closed by slivers beside its
    /// `whole` list, not by strips: a chunk with holes or a patch, or a node
    /// too narrow to have a ring.
    pub sliver: bool,
    /// Unit ids, `NONE` where there is none.
    whole: u32,
    core: u32,
    plain: [u32; SIDES],
    stitch: [u32; SIDES],
}

#[derive(Clone, Copy, Debug, Default)]
struct LevelGrid {
    nx: u32,
    nz: u32,
    first: u32,
}

/// The ground's chunks, the tree over them and (when built) the coarser
/// lists, with the index buffer they all lie in.
#[derive(Clone, Debug, Default)]
pub struct GroundLod {
    /// The ground's index buffer: every chunk's step-1 triangles in Morton
    /// order of the chunks (the first [`Self::level0_len`] indices: all a
    /// lever-off frame draws), then the lists.
    pub indices: Vec<u32>,
    pub units: Vec<Unit>,
    pub nodes: Vec<Node>,
    grids: Vec<LevelGrid>,
    level0_len: u32,
    /// The coarsest level with a list, 0 when the lists were not built.
    top_lod: usize,
    /// Metres a cell.
    pub cell: f32,
    /// THE SLOPE SPLIT'S PIECES (`ground_twins::SlopeSplit`, blocks): a
    /// unit laid in a block is drawn as its pieces ([`Self::frame`]).
    pub pieces: Pieces,
    /// Per block of [`Self::pieces`], by its first index: `(first, count,
    /// units)` -- its whole range and how many units lie in it. What lets a
    /// pass that draws the ground whole draw a block as one range
    /// ([`Wholes`]). Made by [`Self::set_pieces`].
    block_spans: Vec<(u32, u32, u32)>,
    /// The step-1 lattice's cells along x and z ([`GridInput::cells`]): its
    /// samples are the first vertices, row by row. `[0, 0]` for a ground that
    /// is no lattice ([`Self::single`]).
    pub lattice: [u32; 2],
    /// `Levers::ground_lod_strict`: no node whole nearer than its own error
    /// allows, at any distance ([`Self::set_strict`]).
    strict: bool,
    /// The error a coarser level may show, radians ([`TOLERANCE`] at the
    /// build).
    pub tolerance: f32,
    // The frame's answers, and what a node did last time it was asked.
    frame: u32,
    stamp: Vec<u32>,
    split_now: Vec<bool>,
    was_split: Vec<bool>,
}

/// What a frame draws of the ground.
#[derive(Clone, Debug, Default)]
pub struct Selection {
    /// Unit ids, ascending -- so ascending by first index.
    pub units: Vec<u32>,
    /// The nodes drawn, each with the sides stitched (bit `k` for side `k`).
    pub nodes: Vec<(u32, u8)>,
    /// Nodes asked (boxes tested, split decisions made).
    pub visited: u32,
    /// The largest error of a drawn node over its distance, radians: how far
    /// the drawn ground may stand under the step-1 ground the horizon was
    /// built from ([`lifted`]). 0 with the lever off.
    pub error_rad: f32,
    /// Whether the coarser lists were chosen from.
    pub lod: bool,
}

/// Interleave the low 16 bits of `x` and `z`: the order chunks lie in.
pub fn morton(x: u32, z: u32) -> u32 {
    fn spread(v: u32) -> u32 {
        let mut v = v & 0xffff;
        v = (v | (v << 8)) & 0x00ff_00ff;
        v = (v | (v << 4)) & 0x0f0f_0f0f;
        v = (v | (v << 2)) & 0x3333_3333;
        (v | (v << 1)) & 0x5555_5555
    }
    spread(x) | (spread(z) << 1)
}

/// A box's top raised by what a coarser ground in front of it may have let
/// show: a ridge drawn `error_rad` of its distance too low hides what stands
/// behind it as if that stood `error_rad` of ITS distance higher. What the
/// horizon is asked with while the lever is on.
pub fn lifted(error_rad: f32, eye: Vec3, lo: Vec3, hi: Vec3) -> Vec3 {
    if error_rad <= 0.0 {
        return hi;
    }
    let far = (eye - lo).abs().max((eye - hi).abs()).length();
    hi + Vec3::Y * (error_rad * far)
}

/// The mesher's rule for a quad's diagonal (`terrain::splits_b_to_c`): the
/// shorter one.
fn splits_b_to_c(ha: f32, hb: f32, hc: f32, hd: f32) -> bool {
    (hb - hc).abs() <= (ha - hd).abs()
}

/// A node's lattice: the sample lines it draws, in cells.
struct Lattice {
    xs: Vec<u32>,
    zs: Vec<u32>,
    stride: u32,
}

impl Lattice {
    fn new(cells: [u32; 4], step: u32, stride: u32) -> Self {
        let line = |a: u32, b: u32| {
            let mut v: Vec<u32> = (a..=b).step_by(step as usize).collect();
            if v.last() != Some(&b) {
                v.push(b);
            }
            v
        };
        Self { xs: line(cells[0], cells[2]), zs: line(cells[1], cells[3]), stride }
    }
    fn quads(&self) -> (usize, usize) {
        (self.xs.len() - 1, self.zs.len() - 1)
    }
    fn vertex(&self, i: usize, j: usize) -> u32 {
        self.zs[j] * self.stride + self.xs[i]
    }
    /// Lattice point (`along`, `depth`) of side `k` -- `depth` 0 on the
    /// node's edge, 1 one line in -- as lattice indices.
    fn side_point(&self, k: usize, t: usize, r: usize) -> (usize, usize) {
        let (nx, nz) = self.quads();
        match k {
            0 => (r, t),
            1 => (nx - r, t),
            2 => (t, r),
            _ => (t, nz - r),
        }
    }
    /// How many quads lie along side `k`.
    fn side_len(&self, k: usize) -> usize {
        let (nx, nz) = self.quads();
        if k < 2 { nz } else { nx }
    }
    /// The quad at place `t` along side `k`.
    fn side_quad(&self, k: usize, t: usize) -> (usize, usize) {
        let (nx, nz) = self.quads();
        match k {
            0 => (0, t),
            1 => (nx - 1, t),
            2 => (t, 0),
            _ => (t, nz - 1),
        }
    }
    /// One triangle of lattice points, wound as the mesher winds.
    fn tri(&self, out: &mut Vec<u32>, p: [(usize, usize); 3]) {
        let at = |q: (usize, usize)| (self.xs[q.0] as i64, self.zs[q.1] as i64);
        let (a, b, c) = (at(p[0]), at(p[1]), at(p[2]));
        let cross = (b.0 - a.0) * (c.1 - a.1) - (b.1 - a.1) * (c.0 - a.0);
        let v = |q: (usize, usize)| self.vertex(q.0, q.1);
        // The mesher's `[a, c, b]` of a cell turns this way.
        let t = if cross < 0 { [p[0], p[1], p[2]] } else { [p[0], p[2], p[1]] };
        // First (the provoking vertex) a sample even along x and z where
        // there is one: what tells the ground reader a triangle is no
        // step-1 triangle (`ground_shape::WGSL`). Turned, not reflected.
        let even = |q: (usize, usize)| self.xs[q.0] % 2 == 0 && self.zs[q.1] % 2 == 0;
        let r = (0..3).find(|k| even(t[*k])).unwrap_or(0);
        out.extend_from_slice(&[v(t[r]), v(t[(r + 1) % 3]), v(t[(r + 2) % 3])]);
    }
}

/// A node's lists, before they are laid in the buffer.
#[derive(Default)]
struct Lists {
    core: Vec<u32>,
    plain: [Vec<u32>; SIDES],
    stitch: [Vec<u32>; SIDES],
}

/// The core and the eight strips of a node whose lattice has a ring: `quad`
/// appends quad (`i`, `j`)'s own triangles, `present` says whether it has
/// any.
fn ring_lists(lat: &Lattice, quad: &dyn Fn(usize, usize, &mut Vec<u32>), present: &dyn Fn(usize, usize) -> bool) -> Lists {
    let (nx, nz) = lat.quads();
    let mut lists = Lists::default();
    for j in 1..nz - 1 {
        for i in 1..nx - 1 {
            quad(i, j, &mut lists.core);
        }
    }
    for k in 0..SIDES {
        let n = lat.side_len(k);
        let outer = |t: usize| lat.side_point(k, t, 0);
        let inner = |t: usize| lat.side_point(k, t, 1);
        let here = |t: usize| {
            let (i, j) = lat.side_quad(k, t);
            present(i, j)
        };
        // PLAIN: the ring's own triangles, the two corner quads cut from the
        // node's corner to the core's, so the four strips tile the ring.
        let plain = &mut lists.plain[k];
        if here(0) {
            lat.tri(plain, [outer(0), outer(1), inner(1)]);
        }
        for t in 1..n - 1 {
            let (i, j) = lat.side_quad(k, t);
            quad(i, j, plain);
        }
        if here(n - 1) {
            lat.tri(plain, [outer(n - 1), outer(n), inner(n - 1)]);
        }
        // STITCHED: the same strip between the inner line and the COARSER
        // neighbour's edge -- every other vertex of the outer line, and its
        // last. Walked along both lines at once, the one whose next vertex
        // comes first stepping.
        let stitch = &mut lists.stitch[k];
        let edge: Vec<usize> = (0..=n).filter(|t| t % 2 == 0 || *t == n).collect();
        let line: Vec<usize> = (1..n).collect();
        let (mut a, mut b) = (0usize, 0usize);
        loop {
            let (more_edge, more_line) = (a + 1 < edge.len(), b + 1 < line.len());
            if !more_edge && !more_line {
                break;
            }
            let (from, to) = if more_edge && (!more_line || edge[a + 1] <= line[b + 1]) {
                let span = (edge[a].min(line[b]), edge[a + 1].max(line[b]));
                if (span.0..span.1).any(here) {
                    lat.tri(stitch, [outer(edge[a]), outer(edge[a + 1]), inner(line[b])]);
                }
                a += 1;
                span
            } else {
                let span = (edge[a].min(line[b]), edge[a].max(line[b + 1]));
                if (span.0..span.1).any(here) {
                    lat.tri(stitch, [outer(edge[a]), inner(line[b + 1]), inner(line[b])]);
                }
                b += 1;
                span
            };
            debug_assert!(from < to);
        }
    }
    lists
}

/// For a node drawn by its `whole` list: on each side, the slivers between
/// its edge and a coarser neighbour's -- for every vertex of the edge the
/// neighbour skips and a triangle of `whole` uses, the triangle of it and
/// its two neighbours along the edge, both ways round (it stands on edge: a
/// view sees one of them).
fn sliver_lists(lat: &Lattice, whole: &[u32]) -> [Vec<u32>; SIDES] {
    let mut edges: std::collections::HashSet<(u32, u32)> = std::collections::HashSet::new();
    for t in whole.chunks_exact(3) {
        for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
            edges.insert((a.min(b), a.max(b)));
        }
    }
    let has = |a: u32, b: u32| edges.contains(&(a.min(b), a.max(b)));
    std::array::from_fn(|k| {
        let n = lat.side_len(k);
        let v = |t: usize| {
            let (i, j) = lat.side_point(k, t, 0);
            lat.vertex(i, j)
        };
        let mut out = Vec::new();
        let mut t = 0;
        while t + 2 <= n {
            let (a, m, b) = (v(t), v(t + 1), v(t + 2));
            if has(a, m) || has(m, b) {
                out.extend_from_slice(&[a, m, b, a, b, m]);
            }
            t += 2;
        }
        out
    })
}

impl GroundLod {
    /// Whether the coarser lists are in the buffer.
    pub fn has_lists(&self) -> bool {
        self.top_lod > 0
    }

    /// How many of the buffer's first indices are the step-1 ground: what
    /// the horizon's occluder and anything else that wants "the ground's
    /// triangles" reads.
    pub fn level0_len(&self) -> u32 {
        self.level0_len
    }

    /// Chunks along x and z.
    pub fn chunks(&self) -> [u32; 2] {
        self.grids.first().map_or([0, 0], |g| [g.nx, g.nz])
    }

    fn id(&self, level: usize, i: u32, j: u32) -> Option<usize> {
        let g = self.grids.get(level)?;
        (i < g.nx && j < g.nz).then(|| (g.first + j * g.nx + i) as usize)
    }

    fn parent(&self, n: usize) -> Option<usize> {
        let node = &self.nodes[n];
        self.id(node.level as usize + 1, node.at[0] >> 1, node.at[1] >> 1)
    }

    fn children(&self, n: usize) -> [Option<usize>; 4] {
        let node = &self.nodes[n];
        if node.level == 0 {
            return [None; 4];
        }
        let (l, i, j) = (node.level as usize - 1, node.at[0] * 2, node.at[1] * 2);
        [self.id(l, i, j), self.id(l, i + 1, j), self.id(l, i, j + 1), self.id(l, i + 1, j + 1)]
    }

    /// The node one level up across side `k` of node `n`, if `n` lies on
    /// that side of its own parent: the one neighbour that can be coarser.
    fn coarser_neighbour(&self, n: usize, k: usize) -> Option<usize> {
        let node = &self.nodes[n];
        let (l, [i, j]) = (node.level as usize + 1, node.at);
        let (pi, pj) = (i >> 1, j >> 1);
        match k {
            0 if i & 1 == 0 && pi > 0 => self.id(l, pi - 1, pj),
            1 if i & 1 == 1 => self.id(l, pi + 1, pj),
            2 if j & 1 == 0 && pj > 0 => self.id(l, pi, pj - 1),
            3 if j & 1 == 1 => self.id(l, pi, pj + 1),
            _ => None,
        }
    }

    /// A ground that is no grid of samples, as ONE chunk: all of it drawn or
    /// none.
    pub fn single(positions: &[Vec3], triangles: &[u32]) -> Self {
        let (mut lo, mut hi) = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
        for v in triangles {
            lo = lo.min(positions[*v as usize]);
            hi = hi.max(positions[*v as usize]);
        }
        let node = Node {
            level: 0,
            at: [0, 0],
            cells: [0; 4],
            lo,
            hi,
            error: 0.0,
            split_under: 0.0,
            holes: false,
            held: false,
            sliver: false,
            whole: if triangles.is_empty() { NONE } else { 0 },
            core: NONE,
            plain: [NONE; SIDES],
            stitch: [NONE; SIDES],
        };
        Self {
            indices: triangles.to_vec(),
            units: if triangles.is_empty() { Vec::new() } else { vec![Unit { first: 0, count: triangles.len() as u32, node: 0 }] },
            nodes: vec![node],
            grids: vec![LevelGrid { nx: 1, nz: 1, first: 0 }],
            level0_len: triangles.len() as u32,
            stamp: vec![0],
            split_now: vec![false],
            was_split: vec![false],
            ..Self::default()
        }
    }

    /// THE CHUNKS, THE TREE AND THE LISTS of one ground. Linear in its
    /// samples; once a level.
    pub fn build(input: &GridInput) -> Self {
        Self::build_with(input, SLIVER_BORDERS.load(std::sync::atomic::Ordering::Relaxed))
    }

    /// [`Self::build`], every node's border toward a coarser neighbour
    /// closed by its slivers (`slivers`, [`SLIVER_BORDERS`]) or by stitched
    /// strips.
    pub fn build_with(input: &GridInput, slivers: bool) -> Self {
        let [cx, cz] = input.cells;
        let stride = cx + 1;
        let grid_vertices = stride * (cz + 1);
        let cc = input.chunk_cells.max(1);
        if cx == 0 || cz == 0 || input.positions.len() < grid_vertices as usize {
            return Self::default();
        }
        let pos = |v: u32| input.positions[v as usize];
        let origin = pos(0);
        let cell_x = (pos(cx).x - origin.x) / cx as f32;
        let cell_z = (pos(stride * cz).z - origin.z) / cz as f32;
        let cell = cell_x.abs().max(cell_z.abs());
        let chunks = [cx.div_ceil(cc), cz.div_ceil(cc)];
        let top_tree = {
            let widest = chunks[0].max(chunks[1]);
            let mut l = 0usize;
            while (1u32 << l) < widest {
                l += 1;
            }
            if input.lists { l.max(MAX_LEVEL) } else { l }
        };
        let top_lod = if input.lists { MAX_LEVEL } else { 0 };

        // ---- The nodes, level by level, row-major in each.
        let mut grids = Vec::with_capacity(top_tree + 1);
        let mut nodes: Vec<Node> = Vec::new();
        for l in 0..=top_tree {
            let (nx, nz) = (chunks[0].div_ceil(1 << l), chunks[1].div_ceil(1 << l));
            grids.push(LevelGrid { nx, nz, first: nodes.len() as u32 });
            let side = cc << l;
            for j in 0..nz {
                for i in 0..nx {
                    nodes.push(Node {
                        level: l as u8,
                        at: [i, j],
                        cells: [i * side, j * side, ((i + 1) * side).min(cx), ((j + 1) * side).min(cz)],
                        lo: Vec3::splat(f32::INFINITY),
                        hi: Vec3::splat(f32::NEG_INFINITY),
                        error: 0.0,
                        split_under: 0.0,
                        holes: false,
                        held: false,
                        sliver: false,
                        whole: NONE,
                        core: NONE,
                        plain: [NONE; SIDES],
                        stitch: [NONE; SIDES],
                    });
                }
            }
        }
        let tolerance = f32::from_bits(TOLERANCE.load(std::sync::atomic::Ordering::Relaxed));
        let tolerance = if tolerance.is_finite() && tolerance > 0.0 { tolerance } else { PIXEL_RAD };
        let mut lod = Self { grids, nodes, top_lod, cell, lattice: input.cells, tolerance, ..Self::default() };

        // ---- Step 1: every drawn triangle to its chunk, in the mesher's
        // order within it. A lattice triangle by its cell; any other (a
        // cave's ground patch) by its centre.
        let n0 = (chunks[0] * chunks[1]) as usize;
        let mut whole0: Vec<Vec<u32>> = vec![Vec::new(); n0];
        // Per cell: 0 nothing drawn and no hole (buried under a floor), 1 otherwise.
        let mut cell_drawn = vec![0u8; (cx * cz) as usize];
        let mut patched = vec![false; n0];
        // Per lattice vertex: whether a drawn triangle uses it. A sample under
        // a hole or under a building's floor is no part of the ground seen,
        // and how far a coarser step stands from it is no error.
        let mut live = vec![false; grid_vertices as usize];
        let cell_of = |t: &[u32]| -> Option<(u32, u32)> {
            if t.iter().any(|v| *v >= grid_vertices) {
                return None;
            }
            let xs = [t[0] % stride, t[1] % stride, t[2] % stride];
            let zs = [t[0] / stride, t[1] / stride, t[2] / stride];
            let (x0, x1) = (xs.iter().min().copied()?, xs.iter().max().copied()?);
            let (z0, z1) = (zs.iter().min().copied()?, zs.iter().max().copied()?);
            (x1 - x0 <= 1 && z1 - z0 <= 1 && x0 < cx && z0 < cz).then_some((x0, z0))
        };
        for t in input.triangles.chunks_exact(3) {
            for v in t {
                if let Some(l) = live.get_mut(*v as usize) {
                    *l = true;
                }
            }
            let chunk = match cell_of(t) {
                Some((x, z)) => {
                    cell_drawn[(z * cx + x) as usize] = 1;
                    (z / cc) * chunks[0] + x / cc
                }
                None => {
                    let c = (pos(t[0]) + pos(t[1]) + pos(t[2])) / 3.0;
                    let along = |p: f32, o: f32, size: f32, n: u32| {
                        let at = if size.abs() > 1e-9 { (p - o) / (size * cc as f32) } else { 0.0 };
                        (at.max(0.0) as u32).min(n - 1)
                    };
                    let chunk = along(c.z, origin.z, cell_z, chunks[1]) * chunks[0] + along(c.x, origin.x, cell_x, chunks[0]);
                    patched[chunk as usize] = true;
                    chunk
                }
            };
            // A lattice triangle with a sample odd along x or z FIRST (the
            // provoking vertex; every cell's triangle has one): the ground
            // reader knows a step-1 triangle by it and reads no shape there
            // (`ground_shape::WGSL`). Turned, not reflected: the same winding.
            let odd = |v: u32| v < grid_vertices && ((v % stride) % 2 == 1 || (v / stride) % 2 == 1);
            let r = (0..3).find(|k| odd(t[*k])).unwrap_or(0);
            whole0[chunk as usize].extend_from_slice(&[t[r], t[(r + 1) % 3], t[(r + 2) % 3]]);
        }
        if let Some(open) = input.open {
            for z in 0..cz {
                for x in 0..cx {
                    if open(x, z) {
                        cell_drawn[(z * cx + x) as usize] = 1;
                        lod.nodes[((z / cc) * chunks[0] + x / cc) as usize].holes = true;
                    }
                }
            }
        }
        for (k, p) in patched.iter().enumerate() {
            lod.nodes[k].holes |= *p;
        }

        // ---- Boxes: a chunk's over its lattice and its triangles, a
        // node's over its children.
        for k in 0..n0 {
            let [x0, z0, x1, z1] = lod.nodes[k].cells;
            let (mut lo, mut hi) = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
            for z in z0..=z1 {
                for x in x0..=x1 {
                    let p = pos(z * stride + x);
                    lo = lo.min(p);
                    hi = hi.max(p);
                }
            }
            for v in &whole0[k] {
                lo = lo.min(pos(*v));
                hi = hi.max(pos(*v));
            }
            (lod.nodes[k].lo, lod.nodes[k].hi) = (lo, hi);
        }
        for n in n0..lod.nodes.len() {
            for c in lod.children(n).into_iter().flatten() {
                let (lo, hi, holes) = (lod.nodes[c].lo, lod.nodes[c].hi, lod.nodes[c].holes);
                lod.nodes[n].lo = lod.nodes[n].lo.min(lo);
                lod.nodes[n].hi = lod.nodes[n].hi.max(hi);
                lod.nodes[n].holes |= holes;
            }
        }

        // ---- The buffer. First the chunks at step 1, in Morton order.
        let rows = ROW_ORDER.load(std::sync::atomic::Ordering::Relaxed);
        let by_morton = |lod: &Self, l: usize| {
            let g = lod.grids[l];
            let mut order: Vec<usize> = (g.first as usize..(g.first + g.nx * g.nz) as usize).collect();
            if !rows {
                order.sort_by_key(|n| morton(lod.nodes[*n].at[0], lod.nodes[*n].at[1]));
            }
            order
        };
        let mut indices: Vec<u32> = Vec::with_capacity(input.triangles.len());
        let mut units: Vec<Unit> = Vec::new();
        let push = |indices: &mut Vec<u32>, units: &mut Vec<Unit>, node: usize, list: &[u32]| -> u32 {
            if list.is_empty() {
                return NONE;
            }
            units.push(Unit { first: indices.len() as u32, count: list.len() as u32, node: node as u32 });
            indices.extend_from_slice(list);
            units.len() as u32 - 1
        };
        for n in by_morton(&lod, 0) {
            lod.nodes[n].whole = push(&mut indices, &mut units, n, &whole0[n]);
        }
        lod.level0_len = indices.len() as u32;

        // ---- Step 2: the lists.
        if input.lists {
            let height = |v: u32| pos(v).y;
            for l in 0..=top_lod.min(top_tree) {
                let step = 1u32 << l;
                // A level's whole lists first, in Morton order, so a block of
                // its nodes is one stretch of the buffer (`BLOCK_ORDER`);
                // then its borders.
                let mut wholes: Vec<Vec<u32>> = Vec::new();
                // The level's slivers, laid after it side by side (below).
                let mut pending: Vec<(usize, [Vec<u32>; SIDES])> = Vec::new();
                for pass in 0..2 {
                for (at, n) in by_morton(&lod, l).into_iter().enumerate() {
                    let lat = Lattice::new(lod.nodes[n].cells, step, stride);
                    let (nx, nz) = lat.quads();
                    // A coarse quad's own two triangles, by the shorter diagonal.
                    let coarse_quad = |i: usize, j: usize, out: &mut Vec<u32>| {
                        let (a, b, c, d) = (lat.vertex(i, j), lat.vertex(i + 1, j), lat.vertex(i, j + 1), lat.vertex(i + 1, j + 1));
                        if splits_b_to_c(height(a), height(b), height(c), height(d)) {
                            out.extend_from_slice(&[a, c, b, b, c, d]);
                        } else {
                            out.extend_from_slice(&[a, c, d, a, d, b]);
                        }
                    };
                    // Drawn unless every cell under it is buried.
                    let coarse_present = |i: usize, j: usize| {
                        (lat.zs[j]..lat.zs[j + 1]).any(|z| (lat.xs[i]..lat.xs[i + 1]).any(|x| cell_drawn[(z * cx + x) as usize] != 0))
                    };
                    if pass == 0 {
                        let whole: Vec<u32> = if l == 0 {
                            Vec::new() // in the buffer already
                        } else {
                            let mut out = Vec::with_capacity(nx * nz * 6);
                            for j in 0..nz {
                                for i in 0..nx {
                                    if coarse_present(i, j) {
                                        coarse_quad(i, j, &mut out);
                                    }
                                }
                            }
                            out
                        };
                        if l > 0 {
                            lod.nodes[n].whole = push(&mut indices, &mut units, n, &whole);
                            lod.nodes[n].error = Self::error_of(&lat, &pos, &live);
                        }
                        wholes.push(whole);
                        continue;
                    }
                    let whole = std::mem::take(&mut wholes[at]);
                    // The top level has no coarser neighbour: no strips.
                    if l == top_lod {
                        continue;
                    }
                    let ringless = nx < 2 || nz < 2;
                    if slivers || ringless || (l == 0 && lod.nodes[n].holes) {
                        lod.nodes[n].sliver = true;
                        let of: &[u32] = if l == 0 { &whole0[n] } else { &whole };
                        pending.push((n, sliver_lists(&lat, of)));
                        continue;
                    }
                    let lists = if l == 0 {
                        // A chunk's quads are its own drawn triangles, cell by cell.
                        let [x0, z0, ..] = lod.nodes[n].cells;
                        let mut of_cell: Vec<([u32; 6], u8)> = vec![([0; 6], 0); nx * nz];
                        for t in whole0[n].chunks_exact(3) {
                            if let Some((x, z)) = cell_of(t) {
                                let slot = &mut of_cell[(z - z0) as usize * nx + (x - x0) as usize];
                                if slot.1 < 2 {
                                    slot.0[slot.1 as usize * 3..slot.1 as usize * 3 + 3].copy_from_slice(t);
                                    slot.1 += 1;
                                }
                            }
                        }
                        let quad = |i: usize, j: usize, out: &mut Vec<u32>| {
                            let (tris, count) = &of_cell[j * nx + i];
                            out.extend_from_slice(&tris[..*count as usize * 3]);
                        };
                        let present = |i: usize, j: usize| of_cell[j * nx + i].1 > 0;
                        ring_lists(&lat, &quad, &present)
                    } else {
                        let quad = |i: usize, j: usize, out: &mut Vec<u32>| {
                            if coarse_present(i, j) {
                                coarse_quad(i, j, out);
                            }
                        };
                        ring_lists(&lat, &quad, &coarse_present)
                    };
                    lod.nodes[n].core = push(&mut indices, &mut units, n, &lists.core);
                    for k in 0..SIDES {
                        lod.nodes[n].plain[k] = push(&mut indices, &mut units, n, &lists.plain[k]);
                    }
                    for k in 0..SIDES {
                        lod.nodes[n].stitch[k] = push(&mut indices, &mut units, n, &lists.stitch[k]);
                    }
                }
                }
                // THE SLIVERS SIDE BY SIDE, each side's along the line its
                // nodes stand on: a west or east side's by column (x, then
                // z), a south or north side's by row -- so the stitched sides
                // along one straight edge of a coarser region are one
                // stretch of the buffer, and one draw (`BLOCK_ORDER`'s
                // sliver runs are one bucket each).
                for k in 0..SIDES {
                    let mut order: Vec<usize> = (0..pending.len()).collect();
                    order.sort_by_key(|&i| {
                        let at = lod.nodes[pending[i].0].at;
                        if k < 2 { (at[0], at[1]) } else { (at[1], at[0]) }
                    });
                    for i in order {
                        let n = pending[i].0;
                        let list = std::mem::take(&mut pending[i].1[k]);
                        lod.nodes[n].stitch[k] = push(&mut indices, &mut units, n, &list);
                    }
                }
            }
            lod.split_distances();
        }
        lod.indices = indices;
        lod.units = units;
        let n = lod.nodes.len();
        lod.stamp = vec![0; n];
        lod.split_now = vec![false; n];
        lod.was_split = vec![false; n];
        lod
    }

    /// How far a node's surface on `lat` stands from the step-1 samples under
    /// it: the largest distance from a sample to the PLANE of the coarse
    /// triangle over it -- the vertical gap times the triangle's own lean --
    /// over the samples a drawn step-1 triangle uses (`live`).
    /// (The vertical gap alone calls a cliff whose edge a coarser step moves
    /// 40 cm sideways "five metres wrong": on the shipped ground it put half
    /// the nodes at their distance cap for an error nobody could see.)
    fn error_of(lat: &Lattice, pos: &dyn Fn(u32) -> Vec3, live: &[bool]) -> f32 {
        let (nx, nz) = lat.quads();
        let at = |x: u32, z: u32| pos(z * lat.stride + x);
        let mut worst = 0.0f32;
        for j in 0..nz {
            for i in 0..nx {
                let (xa, xb, za, zc) = (lat.xs[i], lat.xs[i + 1], lat.zs[j], lat.zs[j + 1]);
                if xb - xa <= 1 && zc - za <= 1 {
                    continue;
                }
                let (a, b, c, d) = (at(xa, za), at(xb, za), at(xa, zc), at(xb, zc));
                let b_to_c = splits_b_to_c(a.y, b.y, c.y, d.y);
                // The quad's two triangles as planes: a point and a unit normal.
                let plane = |p: Vec3, q: Vec3, r: Vec3| (p, (q - p).cross(r - p).normalize_or_zero());
                let (first, second) = if b_to_c { (plane(a, c, b), plane(b, c, d)) } else { (plane(a, c, d), plane(a, d, b)) };
                for z in za..=zc {
                    for x in xa..=xb {
                        let (u, v) = ((x - xa) as f32 / (xb - xa) as f32, (z - za) as f32 / (zc - za) as f32);
                        let in_first = if b_to_c { u + v <= 1.0 } else { v >= u };
                        let (p, n) = if in_first { first } else { second };
                        if live[(z * lat.stride + x) as usize] {
                            worst = worst.max((at(x, z) - p).dot(n).abs());
                        }
                    }
                }
            }
        }
        worst
    }

    /// THE RULE'S OTHER FORM (`Levers::ground_lod_strict`): with `strict`, a
    /// node is never drawn whole nearer than its error allows -- the
    /// design's distance caps ([`FORCED`]: step 1 ends at 120 m, step 2 at
    /// 240 whatever they hide) are not applied. Every node's distance is
    /// worked out again (the nodes, once; nothing when it is as it was).
    pub fn set_strict(&mut self, strict: bool) {
        if self.strict != strict && self.has_lists() {
            self.strict = strict;
            self.split_distances();
        }
    }

    /// EACH NODE'S `split_under`: the design's rule, then raised until the
    /// tree is restricted whatever the head does.
    ///
    /// A node `N` at level `L` must be split whenever a level `L - 1` node
    /// `P` across one of its edges is: `P` is split only while the head is
    /// within `(1 + HYSTERESIS) * split_under(P)` of its box, and from any
    /// point of `P`'s box the shared edge is no more than `P`'s width away
    /// (and `N`'s box no more than `P`'s height range past that), so
    /// `split_under(N) >= (1 + HYSTERESIS) * split_under(P) + width(P) +
    /// height(P)` is enough. Levels ascending, so each `P` is final first.
    fn split_distances(&mut self) {
        let scale = (self.cell / REFERENCE_CELL).max(1e-3);
        for n in 0..self.nodes.len() {
            let node = &self.nodes[n];
            let l = node.level as usize;
            self.nodes[n].split_under = if l == 0 {
                0.0
            } else if l > self.top_lod || node.held {
                f32::INFINITY
            } else {
                let by_error = (node.error / self.tolerance.max(1e-6)).min(if self.strict { f32::INFINITY } else { FORCED[l] * scale });
                (BAND_START[l] * scale).max(by_error).max(if node.holes { HOLE_FLOOR } else { 0.0 })
            };
        }
        for l in 2..=self.top_lod.min(self.grids.len() - 1) {
            let g = self.grids[l];
            for n in g.first as usize..(g.first + g.nx * g.nz) as usize {
                let [i, j] = self.nodes[n].at;
                let (i, j) = (i as i64 * 2, j as i64 * 2);
                // The eight level L-1 nodes across its four edges, and
                // whether each lies across an edge along z (so its width
                // toward us is its x extent).
                let across = [
                    (i - 1, j, true),
                    (i - 1, j + 1, true),
                    (i + 2, j, true),
                    (i + 2, j + 1, true),
                    (i, j - 1, false),
                    (i + 1, j - 1, false),
                    (i, j + 2, false),
                    (i + 1, j + 2, false),
                ];
                let mut under = self.nodes[n].split_under;
                for (pi, pj, along_z) in across {
                    if pi < 0 || pj < 0 {
                        continue;
                    }
                    let Some(p) = self.id(l - 1, pi as u32, pj as u32) else { continue };
                    let p = &self.nodes[p];
                    let width = if along_z { p.hi.x - p.lo.x } else { p.hi.z - p.lo.z };
                    under = under.max((1.0 + HYSTERESIS) * p.split_under + width + (p.hi.y - p.lo.y));
                }
                self.nodes[n].split_under = under;
            }
        }
    }

    /// Whether node `n` is split this frame: asked once, kept for the frame,
    /// and remembered for the next (the hysteresis).
    fn split(&mut self, n: usize, head: Vec3) -> bool {
        if self.stamp[n] == self.frame {
            return self.split_now[n];
        }
        let node = &self.nodes[n];
        let split = if node.level == 0 {
            false
        } else if node.level as usize > self.top_lod {
            true
        } else {
            let d = (head.clamp(node.lo, node.hi) - head).length();
            d < node.split_under * if self.was_split[n] { 1.0 + HYSTERESIS } else { 1.0 }
        };
        self.stamp[n] = self.frame;
        self.split_now[n] = split;
        self.was_split[n] = split;
        split
    }

    /// Whether node `n` is what the ground is drawn by where it lies: itself
    /// whole, and every node over it split. Asked of a node in no frustum
    /// too -- its level does not depend on being seen.
    fn whole_here(&mut self, n: usize, head: Vec3) -> bool {
        if self.split(n, head) {
            return false;
        }
        let mut up = self.parent(n);
        while let Some(p) = up {
            if !self.split(p, head) {
                return false;
            }
            up = self.parent(p);
        }
        true
    }

    /// WHAT A FRAME DRAWS OF THE GROUND: the units of the nodes any of
    /// `frusta` (planes in the WORLD; none = everything) touches -- every
    /// chunk at step 1 with `lod` off or no lists, else each node at the
    /// level the head's place gives it, its sides toward coarser neighbours
    /// stitched. One choice for both eyes and every pass.
    ///
    /// Cost: the nodes the frusta touch, down to the level drawn -- not the
    /// level's size.
    pub fn select(&mut self, head: Vec3, frusta: &[[Vec4; 6]], lod: bool) -> Selection {
        let lod = lod && self.has_lists();
        let mut out = Selection { lod, ..Selection::default() };
        let Some(top) = self.grids.last().copied() else { return out };
        self.frame = self.frame.wrapping_add(1).max(1);
        let mut stack: Vec<usize> = (top.first as usize..(top.first + top.nx * top.nz) as usize).rev().collect();
        while let Some(n) = stack.pop() {
            out.visited += 1;
            let (lo, hi, level) = (self.nodes[n].lo, self.nodes[n].hi, self.nodes[n].level);
            if !frusta.is_empty() && !frusta.iter().any(|f| aabb_in_frustum(f, lo, hi)) {
                continue;
            }
            let descend = if lod { self.split(n, head) } else { level > 0 };
            if descend {
                let children = self.children(n);
                stack.extend(children.into_iter().rev().flatten());
                continue;
            }
            let mut mask = 0u8;
            if lod && (level as usize) < self.top_lod {
                for k in 0..SIDES {
                    if let Some(q) = self.coarser_neighbour(n, k) {
                        if self.whole_here(q, head) {
                            mask |= 1 << k;
                        }
                    }
                }
            }
            let node = &self.nodes[n];
            let mut take = |u: u32| {
                if u != NONE {
                    out.units.push(u);
                }
            };
            if mask == 0 {
                take(node.whole);
            } else if node.sliver {
                take(node.whole);
                (0..SIDES).filter(|k| mask & (1 << k) != 0).for_each(|k| take(node.stitch[k]));
            } else {
                take(node.core);
                (0..SIDES).for_each(|k| take(if mask & (1 << k) != 0 { node.stitch[k] } else { node.plain[k] }));
            }
            if lod && level > 0 {
                let d = (head.clamp(lo, hi) - head).length().max(1.0);
                out.error_rad = out.error_rad.max(node.error / d);
            }
            out.nodes.push((n as u32, mask));
        }
        out.units.sort_unstable();
        out
    }

    /// A unit's box in the WORLD: its node's.
    pub fn world_box(&self, unit: u32) -> (Vec3, Vec3) {
        let node = &self.nodes[self.units[unit as usize].node as usize];
        (node.lo, node.hi)
    }

    /// EVERY unit as a chunk, its box in the WORLD: what the slope split is
    /// built over, and what the weather and the horizon are told of at load.
    /// The same list every frame.
    pub fn all_units(&self) -> Vec<CasterChunk> {
        self.units
            .iter()
            .map(|u| {
                let node = &self.nodes[u.node as usize];
                CasterChunk { first_index: u.first, index_count: u.count, min: node.lo, max: node.hi }
            })
            .collect()
    }

    /// The units a frame chose, as the chunks the passes cull and draw: each
    /// with its box in the PLAYER's frame (from its node's box in the world:
    /// a chunk is a square on the grid, so the box of its turned corners is
    /// the box of all of it) and its first index moved by `index_start`.
    /// Cost: the units chosen.
    pub fn posed(&self, units: &[u32], offset: Vec3, yaw: f32, index_start: u32) -> Vec<CasterChunk> {
        let yaw_inv = glam::Quat::from_rotation_y(-yaw);
        let mut last: Option<(u32, Vec3, Vec3)> = None;
        units
            .iter()
            .map(|&u| {
                let unit = self.units[u as usize];
                let (min, max) = match last {
                    Some((node, lo, hi)) if node == unit.node => (lo, hi),
                    _ => {
                        let node = &self.nodes[unit.node as usize];
                        let (mut lo, mut hi) = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
                        for c in 0..8 {
                            let corner = Vec3::new(
                                if c & 1 == 0 { node.lo.x } else { node.hi.x },
                                if c & 2 == 0 { node.lo.y } else { node.hi.y },
                                if c & 4 == 0 { node.lo.z } else { node.hi.z },
                            );
                            let p = yaw_inv * (corner - offset);
                            lo = lo.min(p);
                            hi = hi.max(p);
                        }
                        let (lo, hi) = (lo - Vec3::splat(BOX_SLACK), hi + Vec3::splat(BOX_SLACK));
                        last = Some((unit.node, lo, hi));
                        (lo, hi)
                    }
                };
                CasterChunk { first_index: unit.first + index_start, index_count: unit.count, min, max }
            })
            .collect()
    }

    /// The step of the level node `n` is drawn at.
    pub fn step_of(&self, n: u32) -> u32 {
        1 << self.nodes[n as usize].level
    }
}

/// THE PIECES A BLOCK'S CHUNKS ARE DRAWN AS (`ground_twins::SlopeSplit`'s
/// blocks, [`BLOCK_ORDER`]). Empty: none.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Pieces {
    /// Per unit: where its pieces start in `runs` and how many -- 0 for a
    /// unit that lies whole in its own range.
    pub of: Vec<(u32, u8)>,
    /// `[first, count, id, bucket]` of each piece: its index range, its place
    /// in the slope split's tables (past the units'), and which of the
    /// unit's nine buckets it is.
    pub runs: Vec<[u32; 4]>,
    /// Per unit: the first index of its block (`u32::MAX`: none).
    pub block: Vec<u32>,
}

/// THE FRAME'S GROUND FOR THE PASSES: what [`GroundLod::select`] chose, as
/// chunks in the player's frame, with each one's unit (its place in the
/// slope split's tables and the horizon's bounds) and its box in the world.
#[derive(Clone, Debug, Default)]
pub struct FrameGround {
    pub chunks: Vec<CasterChunk>,
    /// Per chunk: its unit.
    pub units: Vec<u32>,
    /// Per chunk: its node's box in the world.
    pub world: Vec<(Vec3, Vec3)>,
    pub error_rad: f32,
    pub visited: u32,
    pub lod: bool,
    /// The same ground for the passes that draw it WHOLE (the SpaceWarp
    /// motion pass, the sun's and the spots' shadows): see [`Wholes`].
    pub wholes: Wholes,
}

/// THE GROUND FOR A PASS THAT DRAWS IT WHOLE: no reader a bucket, so a block
/// whose every unit the frame chose is ONE range, wherever its pieces lie.
///
/// Laid bucket by bucket (`BLOCK_ORDER`), a chunk is up to nine pieces, and a
/// pass that drew the frame's chunks by their pieces drew a block the frustum
/// or the horizon cut as nine draws a run of its chunks: stress_2049's
/// hilltop motion pass 131 draws before the blocks, 964 with them and the
/// lever on (2026-10-14). [`Self::runs`] draws a chosen block as its range
/// when the pass sees at least three quarters of its units (the rest the GPU
/// clips, or the depth test drops behind the ridge that hid them) and as the
/// pieces of the units it sees otherwise. A block only some of whose units
/// were chosen (the rest at another step) is always its pieces: its range
/// would draw the same ground twice.
#[derive(Clone, Debug, Default)]
pub struct Wholes {
    /// Per group (a block's chosen units, or a unit in no block):
    /// `[first, count, pieces_end, members_end]` -- its range drawn as one
    /// (`count` 0: it may not be), and where its pieces in the frame's
    /// `chunks` and its units in `members` end (each group starts where the
    /// one before ended).
    pub groups: Vec<[u32; 4]>,
    /// Per unit chosen: the index in the frame's `chunks` of its first
    /// piece, whose box (every piece of a unit has the unit's) a pass culls.
    pub members: Vec<u32>,
}

impl Wholes {
    /// THE RUNS A WHOLE-GROUND PASS DRAWS: `visible(k)` of the frame's chunk
    /// `k` (in the pass's frustum, not behind the horizon), the runs that
    /// touch in the buffer joined. Cost: the units chosen.
    pub fn runs(&self, chunks: &[crate::renderer::shadow::CasterChunk], visible: impl Fn(usize) -> bool) -> Vec<(u32, u32)> {
        if self.groups.is_empty() {
            // No groups made (a frame not from `GroundLod::frame`): every chunk.
            return merged(chunks.iter().enumerate().filter(|(k, _)| visible(*k)).map(|(_, c)| (c.first_index, c.index_count)));
        }
        let mut out: Vec<(u32, u32)> = Vec::new();
        let (mut p, mut m) = (0usize, 0usize);
        for &[first, count, p_end, m_end] in &self.groups {
            let members = &self.members[m..m_end as usize];
            let seen = members.iter().filter(|k| visible(**k as usize)).count();
            if seen > 0 {
                if count > 0 && seen * 4 >= members.len() * 3 {
                    out.push((first, count));
                } else {
                    out.extend((p..p_end as usize).filter(|k| visible(*k)).map(|k| (chunks[k].first_index, chunks[k].index_count)));
                }
            }
            (p, m) = (p_end as usize, m_end as usize);
        }
        merged(out.into_iter())
    }
}

impl GroundLod {
    /// [`Self::select`] and [`Self::posed`] in one: the call a frame makes.
    /// `frusta` in the WORLD. A unit the slope split laid in a block
    /// ([`Self::pieces`]) comes as its pieces, each with its own place in the
    /// split's tables and its unit's box; all of them by first index.
    pub fn frame(&mut self, head: Vec3, frusta: &[[Vec4; 6]], lod: bool, offset: Vec3, yaw: f32, index_start: u32) -> FrameGround {
        let chosen = self.select(head, frusta, lod);
        let mut chunks = self.posed(&chosen.units, offset, yaw, index_start);
        let mut world: Vec<(Vec3, Vec3)> = chosen.units.iter().map(|u| self.world_box(*u)).collect();
        let mut units = chosen.units;
        let mut wholes = Wholes::default();
        if !self.pieces.runs.is_empty() && units.iter().any(|u| self.pieces.of.get(*u as usize).is_some_and(|p| p.1 > 0)) {
            // A block's chosen chunks, bucket by bucket and chunk by chunk
            // within one: the order its pieces lie in, so by first index
            // without a sort. Cost: the units chosen.
            let mut out_c: Vec<CasterChunk> = Vec::with_capacity(units.len() * 2);
            let mut out_u: Vec<u32> = Vec::with_capacity(units.len() * 2);
            let mut out_w: Vec<(Vec3, Vec3)> = Vec::with_capacity(units.len() * 2);
            let mut group: Vec<usize> = Vec::with_capacity(16);
            let pieces = &self.pieces;
            let spans = &self.block_spans;
            let mut first_piece: Vec<u32> = Vec::with_capacity(16);
            let mut flush = |group: &mut Vec<usize>, out_c: &mut Vec<CasterChunk>, out_u: &mut Vec<u32>, out_w: &mut Vec<(Vec3, Vec3)>, wholes: &mut Wholes| {
                if group.is_empty() {
                    return;
                }
                first_piece.clear();
                first_piece.resize(group.len(), u32::MAX);
                for slope in 0..9u32 {
                    for (j, &i) in group.iter().enumerate() {
                        let (at, n) = pieces.of[units[i] as usize];
                        for [first, count, id, s] in &pieces.runs[at as usize..at as usize + n as usize] {
                            if *s == slope {
                                if first_piece[j] == u32::MAX {
                                    first_piece[j] = out_c.len() as u32;
                                }
                                out_c.push(CasterChunk { first_index: first + index_start, index_count: *count, ..chunks[i] });
                                out_u.push(*id);
                                out_w.push(world[i]);
                            }
                        }
                    }
                }
                // The block's range, when the frame chose every unit in it.
                let block = pieces.block[units[group[0]] as usize];
                let whole = spans
                    .binary_search_by_key(&block, |s| s.0)
                    .ok()
                    .map(|k| spans[k])
                    .filter(|s| s.2 as usize == group.len())
                    .map_or((0, 0), |s| (s.0 + index_start, s.1));
                wholes.members.extend_from_slice(&first_piece);
                wholes.groups.push([whole.0, whole.1, out_c.len() as u32, wholes.members.len() as u32]);
                group.clear();
            };
            let mut block = u32::MAX;
            for i in 0..units.len() {
                let u = units[i] as usize;
                let b = match pieces.of.get(u) {
                    Some(&(_, n)) if n > 0 => pieces.block[u],
                    _ => u32::MAX,
                };
                if b != block || b == u32::MAX {
                    flush(&mut group, &mut out_c, &mut out_u, &mut out_w, &mut wholes);
                }
                block = b;
                if b == u32::MAX {
                    wholes.members.push(out_c.len() as u32);
                    out_c.push(chunks[i]);
                    out_u.push(units[i]);
                    out_w.push(world[i]);
                    wholes.groups.push([chunks[i].first_index, chunks[i].index_count, out_c.len() as u32, wholes.members.len() as u32]);
                } else {
                    group.push(i);
                }
            }
            flush(&mut group, &mut out_c, &mut out_u, &mut out_w, &mut wholes);
            debug_assert!(out_c.windows(2).all(|w| w[0].first_index < w[1].first_index));
            (chunks, units, world) = (out_c, out_u, out_w);
        } else {
            for (k, c) in chunks.iter().enumerate() {
                wholes.members.push(k as u32);
                wholes.groups.push([c.first_index, c.index_count, k as u32 + 1, k as u32 + 1]);
            }
        }
        FrameGround { chunks, world, units, error_rad: chosen.error_rad, visited: chosen.visited, lod: chosen.lod, wholes }
    }

    /// Per unit, the block the slope split may lay it in with its
    /// neighbours ([`BLOCK_CHUNKS`] a side of step-1 chunks, consecutive in
    /// the buffer by the Morton order), `u32::MAX` for none: a coarser
    /// list's run, or with [`BLOCK_ORDER`] off.
    pub fn blocks(&self) -> Vec<u32> {
        let on = BLOCK_ORDER.load(std::sync::atomic::Ordering::Relaxed) && !ROW_ORDER.load(std::sync::atomic::Ordering::Relaxed);
        self.units
            .iter()
            .enumerate()
            .map(|(id, u)| {
                let node = &self.nodes[u.node as usize];
                if on && node.sliver && node.stitch.contains(&(id as u32)) {
                    return SLIVER_RUN;
                }
                // A level's whole lists by its own blocks (`level` in the top bits).
                let whole = node.whole != NONE && self.units.get(node.whole as usize).is_some_and(|w| w.first == u.first);
                if on && whole && self.nodes.len() > 1 {
                    ((node.level as u32) << 28) | (morton(node.at[0], node.at[1]) / (BLOCK_CHUNKS * BLOCK_CHUNKS))
                } else {
                    u32::MAX
                }
            })
            .collect()
    }

    /// THE CHUNKS HELD AT STEP 1, whatever their distance: `held(unit)` of
    /// each step-1 chunk's unit. Every node over one is never drawn whole,
    /// and the tree stays restricted round it. For the chunks the weather
    /// draws: its twins light the ground by its own vertices
    /// (`ground_shape::unshaped`), which only step 1 draws as the shape. A
    /// cost that follows the weather's areas, not the level. Works the
    /// distances out again only when the held chunks change.
    pub fn hold(&mut self, held: &dyn Fn(u32) -> bool) {
        let n0 = self.grids.first().map_or(0, |g| (g.nx * g.nz) as usize);
        let mut now = vec![false; self.nodes.len()];
        for k in 0..n0.min(self.nodes.len()) {
            let w = self.nodes[k].whole;
            now[k] = w != NONE && held(w);
        }
        for n in n0..self.nodes.len() {
            now[n] = self.children(n).into_iter().flatten().any(|c| now[c]);
        }
        if now.iter().zip(&self.nodes).any(|(h, node)| *h != node.held) {
            for (node, h) in self.nodes.iter_mut().zip(now) {
                node.held = h;
            }
            if self.has_lists() {
                self.split_distances();
            }
        }
    }

    /// The pieces the slope split laid this ground's units in
    /// ([`Self::pieces`]): `None` takes them away.
    pub fn set_pieces(&mut self, pieces: Option<Pieces>) {
        self.pieces = pieces.unwrap_or_default();
        // Each block's range and units: its units' ranges (laying moved the
        // triangles, not the block's extent). Cost: the units.
        let mut spans: Vec<(u32, u32, u32)> = Vec::new();
        for (u, (_, n)) in self.pieces.of.iter().enumerate() {
            let (Some(&b), Some(unit)) = (self.pieces.block.get(u), self.units.get(u)) else { continue };
            if *n == 0 || b == u32::MAX {
                continue;
            }
            match spans.binary_search_by_key(&b, |s| s.0) {
                Ok(k) => {
                    spans[k].1 += unit.count;
                    spans[k].2 += 1;
                }
                Err(k) => spans.insert(k, (b, unit.count, 1)),
            }
        }
        self.block_spans = spans;
    }
}

/// THE RUNS A PASS DRAWS for one reader: `(first, count)` of each chunk's
/// range for it, in order, with runs that touch in the buffer joined -- a
/// block of far chunks that all take one reader is one draw.
pub fn merged(runs: impl Iterator<Item = (u32, u32)>) -> Vec<(u32, u32)> {
    let mut out: Vec<(u32, u32)> = Vec::new();
    for (first, count) in runs {
        if count == 0 {
            continue;
        }
        match out.last_mut() {
            Some(last) if last.0 + last.1 == first => last.1 += count,
            _ => out.push((first, count)),
        }
    }
    out
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::collections::HashMap;

    /// A ground of `cells` a side on test_room's cell, with hills that
    /// every step sees differently.
    pub fn hills(cells: u32) -> (Vec<Vec3>, Vec<u32>) {
        ground(cells, &|x, z| 6.0 * (x * 0.031).sin() * (z * 0.027).cos() + 0.4 * (x * 0.9).sin() * (z * 1.1).sin(), &|_, _| false)
    }

    pub fn ground(cells: u32, height: &dyn Fn(f32, f32) -> f32, open: &dyn Fn(u32, u32) -> bool) -> (Vec<Vec3>, Vec<u32>) {
        let stride = cells + 1;
        let mut positions = Vec::new();
        for z in 0..=cells {
            for x in 0..=cells {
                let (wx, wz) = (x as f32 * REFERENCE_CELL - 20.0, z as f32 * REFERENCE_CELL + 7.0);
                positions.push(Vec3::new(wx, height(wx, wz), wz));
            }
        }
        let mut triangles = Vec::new();
        for z in 0..cells {
            for x in 0..cells {
                if open(x, z) {
                    continue;
                }
                let (a, b, c, d) = (z * stride + x, z * stride + x + 1, (z + 1) * stride + x, (z + 1) * stride + x + 1);
                let h = |v: u32| positions[v as usize].y;
                if splits_b_to_c(h(a), h(b), h(c), h(d)) {
                    triangles.extend_from_slice(&[a, c, b, b, c, d]);
                } else {
                    triangles.extend_from_slice(&[a, c, d, a, d, b]);
                }
            }
        }
        (positions, triangles)
    }

    /// [`built`] with stitched strips, not slivers ([`SLIVER_BORDERS`] off).
    fn built_strips(cells: u32, chunk_cells: u32) -> (Vec<Vec3>, Vec<u32>, GroundLod) {
        let (positions, triangles) = hills(cells);
        let lod = GroundLod::build_with(&GridInput { cells: [cells, cells], positions: &positions, triangles: &triangles, open: None, chunk_cells, lists: true }, false);
        (positions, triangles, lod)
    }

    fn built(cells: u32, chunk_cells: u32, lists: bool) -> (Vec<Vec3>, Vec<u32>, GroundLod) {
        let (positions, triangles) = hills(cells);
        let lod = GroundLod::build(&GridInput { cells: [cells, cells], positions: &positions, triangles: &triangles, open: None, chunk_cells, lists });
        (positions, triangles, lod)
    }

    fn sorted_triangles(idx: &[u32]) -> Vec<[u32; 3]> {
        let mut t: Vec<[u32; 3]> = idx
            .chunks_exact(3)
            .map(|t| {
                let k = (0..3).min_by_key(|k| t[*k]).unwrap();
                [t[k], t[(k + 1) % 3], t[(k + 2) % 3]]
            })
            .collect();
        t.sort_unstable();
        t
    }

    fn drawn(lod: &GroundLod, sel: &Selection) -> Vec<u32> {
        sel.units.iter().flat_map(|u| {
            let unit = lod.units[*u as usize];
            lod.indices[unit.first as usize..(unit.first + unit.count) as usize].to_vec()
        }).collect()
    }

    /// The directed edges of `idx` with no partner the other way. A sliver
    /// (three vertices in a line on the map, in the buffer both ways round)
    /// counts once, the way round that closes what is open beside it.
    fn open_edges(positions: &[Vec3], idx: &[u32]) -> Vec<(u32, u32)> {
        let mut edges: HashMap<(u32, u32), i32> = HashMap::new();
        let mut slivers: std::collections::BTreeSet<[u32; 3]> = std::collections::BTreeSet::new();
        for t in idx.chunks_exact(3) {
            let p = |k: usize| positions[t[k] as usize];
            let area = (p(1).x - p(0).x) * (p(2).z - p(0).z) - (p(1).z - p(0).z) * (p(2).x - p(0).x);
            if area == 0.0 {
                let mut s = [t[0], t[1], t[2]];
                s.sort_unstable();
                slivers.insert(s);
                continue;
            }
            for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                *edges.entry((a, b)).or_default() += 1;
            }
        }
        for [a, b, c] in slivers {
            let closes = |t: [u32; 3]| {
                [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])].iter().filter(|(x, y)| edges.get(&(*y, *x)).copied().unwrap_or(0) > edges.get(&(*x, *y)).copied().unwrap_or(0)).count()
            };
            let t = if closes([a, b, c]) >= closes([a, c, b]) { [a, b, c] } else { [a, c, b] };
            for (x, y) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                *edges.entry((x, y)).or_default() += 1;
            }
        }
        // An edge is closed when its reverse is there as often.
        let mut open: Vec<(u32, u32)> = edges.iter().filter(|((a, b), n)| edges.get(&(*b, *a)).copied().unwrap_or(0) != **n).map(|(e, _)| *e).collect();
        open.sort_unstable();
        open
    }

    /// The open edges of a ground must all lie on its outline.
    fn assert_watertight(positions: &[Vec3], cells: u32, idx: &[u32], what: &str) {
        let stride = cells + 1;
        let on_outline = |v: u32| {
            let (x, z) = (v % stride, v / stride);
            x == 0 || z == 0 || x == cells || z == cells
        };
        let open = open_edges(positions, idx);
        let inside: Vec<_> = open.iter().filter(|(a, b)| !(on_outline(*a) && on_outline(*b))).collect();
        assert!(inside.is_empty(), "{what}: {} open edge(s) inside the ground, first {:?}", inside.len(), &inside[..inside.len().min(6)]);
    }

    /// Area on the map of a triangle list, each triangle counted positive.
    fn map_area(positions: &[Vec3], idx: &[u32]) -> f64 {
        idx.chunks_exact(3)
            .map(|t| {
                let p = |k: usize| positions[t[k] as usize];
                (((p(1).x - p(0).x) as f64 * (p(2).z - p(0).z) as f64 - (p(1).z - p(0).z) as f64 * (p(2).x - p(0).x) as f64) * 0.5).abs()
            })
            .sum()
    }

    #[test]
    fn morton_order_keeps_a_square_of_chunks_together() {
        // The four children of any node are four neighbours in the order.
        for l in 1..5u32 {
            let side = 1 << l;
            for (i, j) in [(0u32, 0u32), (3, 5), (7, 2)] {
                let mut codes: Vec<u32> = (0..side).flat_map(|a| (0..side).map(move |b| (a, b))).map(|(a, b)| morton(i * side + a, j * side + b)).collect();
                codes.sort_unstable();
                assert_eq!(codes[codes.len() - 1] - codes[0], side * side - 1, "a node at level {l} is one stretch");
            }
        }
    }

    /// STEP 1: test_room's size is 8 x 8 chunks of 32 cells, every chunk the
    /// mesher's own triangles in the mesher's order, the buffer in Morton
    /// order, and with no lists nothing else in it.
    #[test]
    fn the_chunks_are_32_cells_whatever_the_levels_size_and_hold_every_triangle_once() {
        for (cells, a_side) in [(256u32, 8u32), (624, 20), (1024, 32)] {
            let (_, triangles, lod) = built(cells, CHUNK_CELLS, false);
            assert_eq!(lod.chunks(), [a_side, a_side], "{cells} cells");
            assert!(!lod.has_lists());
            assert_eq!(lod.indices.len(), triangles.len());
            assert_eq!(lod.level0_len() as usize, triangles.len());
            assert_eq!(lod.units.len(), (a_side * a_side) as usize);
            assert_eq!(sorted_triangles(&lod.indices), sorted_triangles(&triangles), "every triangle once");
            // Morton order, and each chunk's triangles in the mesher's order.
            let codes: Vec<u32> = lod.units.iter().map(|u| {
                let at = lod.nodes[u.node as usize].at;
                morton(at[0], at[1])
            }).collect();
            assert!(codes.windows(2).all(|w| w[0] < w[1]), "the chunks lie in Morton order");
            let stride = cells + 1;
            for u in &lod.units {
                let node = &lod.nodes[u.node as usize];
                let cell = |t: &[u32]| (t.iter().map(|v| v / stride).min().unwrap(), t.iter().map(|v| v % stride).min().unwrap());
                let run = &lod.indices[u.first as usize..(u.first + u.count) as usize];
                let cells_in: Vec<(u32, u32)> = run.chunks_exact(3).map(cell).collect();
                assert!(cells_in.windows(2).all(|w| w[0] <= w[1]), "row-major within a chunk, as the mesher left them");
                assert!(cells_in.iter().all(|(z, x)| *x >= node.cells[0] && *x < node.cells[2] && *z >= node.cells[1] && *z < node.cells[3]));
            }
        }
    }

    /// The tree's cull answers as asking every chunk does, and asks far
    /// fewer nodes than the level has chunks.
    #[test]
    fn the_quadtree_cull_is_the_cull_of_every_chunk_and_does_not_walk_the_level() {
        let (_, _, mut lod) = built(2048, CHUNK_CELLS, false);
        assert_eq!(lod.chunks(), [64, 64]);
        let eye = Vec3::new(100.0, 8.0, 200.0);
        for (at, far) in [(Vec3::new(140.0, 2.0, 260.0), 1000.0f32), (Vec3::new(100.0, -30.0, 201.0), 1000.0), (Vec3::new(-300.0, 0.0, 100.0), 60.0)] {
            let view = glam::Mat4::look_at_rh(eye, at, Vec3::Y);
            let proj = glam::Mat4::perspective_rh(1.6, 1.0, 0.05, far);
            let planes = crate::renderer::shadow::frustum_planes(proj * view);
            let chosen = lod.select(eye, &[planes], false);
            let asked_all: Vec<u32> = (0..lod.units.len() as u32)
                .filter(|u| {
                    let (lo, hi) = lod.world_box(*u);
                    aabb_in_frustum(&planes, lo, hi)
                })
                .collect();
            assert_eq!(chosen.units, asked_all);
            assert!(!chosen.units.is_empty());
            assert!((chosen.visited as usize) < chosen.units.len() * 3 + 64, "{} nodes asked for {} chunks", chosen.visited, chosen.units.len());
        }
        // A view of a corner of the level asks a small part of it.
        let view = glam::Mat4::look_at_rh(eye, Vec3::new(-300.0, 0.0, 100.0), Vec3::Y);
        let planes = crate::renderer::shadow::frustum_planes(glam::Mat4::perspective_rh(1.0, 1.0, 0.05, 60.0) * view);
        let near = lod.select(eye, &[planes], false);
        assert!(near.visited < 200, "{} of 5,461 nodes asked for a 60 m view", near.visited);
    }

    /// With the lever off the lists change nothing that is drawn: the same
    /// units, the same indices, as a ground built without them.
    #[test]
    fn with_the_lever_off_the_lists_are_not_drawn() {
        let (_, triangles, mut plain) = built(256, CHUNK_CELLS, false);
        let (_, _, mut with) = built(256, CHUNK_CELLS, true);
        assert!(with.has_lists() && with.indices.len() > plain.indices.len());
        assert_eq!(with.level0_len() as usize, triangles.len());
        assert_eq!(&with.indices[..triangles.len()], &plain.indices[..]);
        let head = Vec3::new(10.0, 3.0, 40.0);
        let (a, b) = (plain.select(head, &[], false), with.select(head, &[], false));
        assert_eq!(a.units, b.units);
        assert_eq!(drawn(&plain, &a), drawn(&with, &b));
        assert_eq!(b.error_rad, 0.0);
    }

    /// EACH CHUNK'S TRIANGLES AT EACH LEVEL: 2,048 / 512 / 128 / 32 / 8.
    #[test]
    fn a_chunk_has_2048_512_128_32_and_8_triangles_at_the_five_steps() {
        // With slivers: every node's whole list, and its slivers -- a node
        // beside a coarser one keeps its whole list; the buffer is about 1.4
        // times step 1's indices.
        let (_, _, slivered) = built(1024, CHUNK_CELLS, true);
        for (l, per_chunk) in [(0usize, 2048u32), (1, 512), (2, 128), (3, 32), (4, 8)] {
            let g = slivered.grids[l];
            for n in g.first..g.first + g.nx * g.nz {
                let node = &slivered.nodes[n as usize];
                assert_eq!(slivered.units[node.whole as usize].count / 3, per_chunk << (2 * l));
                assert!(node.sliver || l == MAX_LEVEL);
                assert_eq!(node.core, NONE);
            }
        }
        let ratio = slivered.indices.len() as f64 / slivered.level0_len() as f64;
        eprintln!("the buffer with slivers: {ratio:.3} times step 1's indices");
        assert!((1.3..1.5).contains(&ratio), "{ratio}");
        let (_, _, lod) = built_strips(1024, CHUNK_CELLS);
        for (l, per_chunk) in [(0usize, 2048u32), (1, 512), (2, 128), (3, 32), (4, 8)] {
            let g = lod.grids[l];
            for n in g.first..g.first + g.nx * g.nz {
                let node = &lod.nodes[n as usize];
                let whole = lod.units[node.whole as usize];
                let chunks = 1u32 << (2 * l);
                assert_eq!(whole.count / 3, per_chunk * chunks, "level {l}: a node of {chunks} chunk(s)");
                assert_eq!(whole.count / 3, 2048, "every node is 32 x 32 quads at its own step");
                if l < MAX_LEVEL {
                    // The core and the four plain strips are the node, once.
                    let of = |u: u32| lod.units[u as usize].count / 3;
                    assert_eq!(of(node.core) + node.plain.iter().map(|u| of(*u)).sum::<u32>(), 2048);
                    assert_eq!(of(node.core), 30 * 30 * 2);
                    // A stitched strip: 16 + 30 triangles for the plain one's 62.
                    assert!(node.plain.iter().all(|u| of(*u) == 62) && node.stitch.iter().all(|u| of(*u) == 46));
                } else {
                    assert_eq!(node.core, NONE, "the coarsest level has no coarser neighbour");
                }
            }
        }
        // The buffer: 2.78 times step 1's indices.
        let ratio = lod.indices.len() as f64 / lod.level0_len() as f64;
        assert!((2.7..2.9).contains(&ratio), "{ratio}");
    }

    /// The core with its plain strips covers what the whole list covers, and
    /// so does it with any of the sixteen choices of stitched sides.
    #[test]
    fn a_nodes_core_and_strips_cover_it_exactly_with_any_sides_stitched() {
        let (positions, _, lod) = built_strips(256, 8);
        for l in 0..MAX_LEVEL {
            let g = lod.grids[l];
            let node = &lod.nodes[(g.first + g.nx + 1) as usize];
            let list = |u: u32| {
                let unit = lod.units[u as usize];
                lod.indices[unit.first as usize..(unit.first + unit.count) as usize].to_vec()
            };
            let whole = map_area(&positions, &list(node.whole));
            for mask in 0..16u8 {
                let mut idx = list(node.core);
                for k in 0..SIDES {
                    idx.extend(list(if mask & (1 << k) != 0 { node.stitch[k] } else { node.plain[k] }));
                }
                let area = map_area(&positions, &idx);
                assert!((area - whole).abs() < 1e-3 * whole, "level {l} mask {mask:04b}: {area} against {whole}");
                // All wound one way: no triangle of it faces down.
                for t in idx.chunks_exact(3) {
                    let p = |k: usize| positions[t[k] as usize];
                    let n = (p(1) - p(0)).cross(p(2) - p(0));
                    assert!(n.y > 0.0, "level {l} mask {mask:04b}: a triangle wound the other way");
                }
                // Inside the node every edge is shared: its own outline is all that is open.
                let open = open_edges(&positions, &idx);
                let stride = 257u32;
                let on_edge = |v: u32| {
                    let (x, z) = (v % stride, v / stride);
                    x == node.cells[0] || x == node.cells[2] || z == node.cells[1] || z == node.cells[3]
                };
                assert!(open.iter().all(|(a, b)| on_edge(*a) && on_edge(*b)), "level {l} mask {mask:04b}: an open edge inside the node");
            }
        }
    }

    /// EVERY BORDER OF EVERY LEVEL PAIR IS WATERTIGHT, all four sides: a
    /// node at level L with that side stitched, beside the node one level
    /// up, share every edge along the border -- the same two vertices, once
    /// each way. And the plain strip there does NOT (the T-junctions the
    /// stitch is for).
    #[test]
    fn every_border_of_every_level_pair_and_side_is_watertight() {
        let (positions, _, lod) = built_strips(256, 8);
        let stride = 257u32;
        let list = |u: u32| {
            let unit = lod.units[u as usize];
            lod.indices[unit.first as usize..(unit.first + unit.count) as usize].to_vec()
        };
        let mut pairs = 0;
        for l in 0..MAX_LEVEL {
            let g = lod.grids[l];
            for n in g.first as usize..(g.first + g.nx * g.nz) as usize {
                for k in 0..SIDES {
                    let Some(q) = lod.coarser_neighbour(n, k) else { continue };
                    let (fine, coarse) = (&lod.nodes[n], &lod.nodes[q]);
                    // The border line: the fine node's edge on side k.
                    let on_border = |v: u32| {
                        let (x, z) = (v % stride, v / stride);
                        match k {
                            0 => x == fine.cells[0] && z >= fine.cells[1] && z <= fine.cells[3],
                            1 => x == fine.cells[2] && z >= fine.cells[1] && z <= fine.cells[3],
                            2 => z == fine.cells[1] && x >= fine.cells[0] && x <= fine.cells[2],
                            _ => z == fine.cells[3] && x >= fine.cells[0] && x <= fine.cells[2],
                        }
                    };
                    let border_edges = |idx: &[u32]| {
                        let mut e: Vec<(u32, u32)> = idx
                            .chunks_exact(3)
                            .flat_map(|t| [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])])
                            .filter(|(a, b)| on_border(*a) && on_border(*b))
                            .collect();
                        e.sort_unstable();
                        e
                    };
                    let reversed = |e: &[(u32, u32)]| {
                        let mut r: Vec<(u32, u32)> = e.iter().map(|(a, b)| (*b, *a)).collect();
                        r.sort_unstable();
                        r
                    };
                    let stitched = border_edges(&list(fine.stitch[k]));
                    // The coarse node as it may be drawn: whole, or by its
                    // core and strips with any of ITS sides stitched (the
                    // side facing us is never one of those: we are finer).
                    let mut coarse_forms = vec![list(coarse.whole)];
                    if coarse.core != NONE {
                        let facing = [1usize, 0, 3, 2][k];
                        let mut idx = list(coarse.core);
                        (0..SIDES).for_each(|s| idx.extend(list(if s == facing { coarse.plain[s] } else { coarse.stitch[s] })));
                        coarse_forms.push(idx);
                    }
                    for form in &coarse_forms {
                        let theirs = border_edges(form);
                        assert!(!stitched.is_empty());
                        assert_eq!(stitched, reversed(&theirs), "level {l} to {} side {k}: the border's edges", l + 1);
                    }
                    let plain = border_edges(&list(fine.plain[k]));
                    assert_ne!(plain, reversed(&border_edges(&coarse_forms[0])), "the plain strip has the vertices the neighbour skips");
                    assert_eq!(plain.len(), stitched.len() * 2);
                    // And the strip itself is closed toward the node's core.
                    let _ = &positions;
                    pairs += 1;
                }
            }
        }
        assert!(pairs > 200, "{pairs} borders checked over four level pairs and four sides");
    }

    /// THE WHOLE GROUND AS DRAWN FROM ANYWHERE IS ONE SHEET: from many head
    /// positions (and as the head walks, so the hysteresis is in play) every
    /// edge inside the ground is shared by exactly two triangles, neighbours
    /// differ by at most one level, and the map is covered exactly once.
    #[test]
    fn from_anywhere_the_ground_is_one_sheet_and_neighbours_differ_by_one_level() {
        for (cells, chunk) in [(1024u32, CHUNK_CELLS), (624, CHUNK_CELLS), (300, 8)] {
            let (positions, triangles, mut lod) = built(cells, chunk, true);
            let whole_area = map_area(&positions, &triangles);
            let side = cells as f32 * REFERENCE_CELL;
            let mut seed = 0x2545_f491u32;
            let mut rnd = || {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                seed as f32 / u32::MAX as f32
            };
            let mut head = Vec3::new(-20.0 + side * 0.5, 4.0, 7.0 + side * 0.5);
            let mut levels_seen = [false; MAX_LEVEL + 1];
            for step in 0..60 {
                // A jump now and then, else a walk of a few metres.
                head = if step == 21 {
                    // From far off: the coarsest level, whatever the hills' error.
                    Vec3::new(-20.0 + side * 0.5 + 6000.0, 50.0, 7.0 + side * 0.5)
                } else if step == 22 {
                    Vec3::new(-20.0 + side * 0.5 + 1000.0, 50.0, 7.0 + side * 0.5)
                } else if step % 7 == 0 {
                    Vec3::new(-20.0 + side * (rnd() * 1.4 - 0.2), rnd() * 40.0 - 5.0, 7.0 + side * (rnd() * 1.4 - 0.2))
                } else {
                    head + Vec3::new(rnd() * 8.0 - 4.0, 0.0, rnd() * 8.0 - 4.0)
                };
                let chosen = lod.select(head, &[], true);
                let idx = drawn(&lod, &chosen);
                assert_watertight(&positions, cells, &idx, &format!("{cells} cells, head {head}"));
                let area = map_area(&positions, &idx);
                assert!((area - whole_area).abs() < 1e-3 * whole_area, "{cells} cells, head {head}: covers {area} of {whole_area}");
                // Neighbours: every chunk's level, then each against the four beside it.
                let [nx, nz] = lod.chunks();
                let mut level_of = vec![u8::MAX; (nx * nz) as usize];
                for (n, _) in &chosen.nodes {
                    let node = &lod.nodes[*n as usize];
                    levels_seen[node.level as usize] = true;
                    let span = 1u32 << node.level;
                    for j in node.at[1] * span..((node.at[1] + 1) * span).min(nz) {
                        for i in node.at[0] * span..((node.at[0] + 1) * span).min(nx) {
                            assert_eq!(level_of[(j * nx + i) as usize], u8::MAX, "a chunk drawn twice");
                            level_of[(j * nx + i) as usize] = node.level;
                        }
                    }
                }
                assert!(level_of.iter().all(|l| *l != u8::MAX), "every chunk drawn");
                for j in 0..nz {
                    for i in 0..nx {
                        let here = level_of[(j * nx + i) as usize] as i32;
                        for (di, dj) in [(1u32, 0u32), (0, 1)] {
                            if i + di < nx && j + dj < nz {
                                let there = level_of[((j + dj) * nx + i + di) as usize] as i32;
                                assert!((here - there).abs() <= 1, "{cells} cells, head {head}: chunks ({i},{j}) and its neighbour at levels {here} and {there}");
                            }
                        }
                    }
                }
            }
            if cells == 1024 {
                assert!(levels_seen.iter().all(|s| *s), "all five levels drawn somewhere: {levels_seen:?}");
            }
        }
    }

    /// The level does not depend on where the head LOOKS: a view draws the
    /// part of the same sheet its frustum touches, sides stitched the same.
    #[test]
    fn turning_the_head_changes_no_level_and_no_stitch() {
        let (_, _, mut lod) = built(1024, CHUNK_CELLS, true);
        let head = Vec3::new(190.0, 3.0, 220.0);
        let all: HashMap<u32, u8> = lod.select(head, &[], true).nodes.into_iter().collect();
        for turn in 0..12 {
            let a = turn as f32 * 0.55;
            let view = glam::Mat4::look_at_rh(head, head + Vec3::new(a.cos(), -0.1, a.sin()), Vec3::Y);
            let planes = crate::renderer::shadow::frustum_planes(glam::Mat4::perspective_rh(1.7, 1.0, 0.05, 1000.0) * view);
            let seen = lod.select(head, &[planes], true);
            assert!(seen.nodes.len() < all.len());
            for (n, mask) in &seen.nodes {
                assert_eq!(all.get(n), Some(mask), "turn {turn}: node {n} drawn otherwise than the whole sheet has it");
            }
        }
    }

    /// HYSTERESIS: a head swaying a few centimetres either side of where a
    /// node splits flips it once, not every sway.
    #[test]
    fn a_swaying_head_does_not_flip_a_level() {
        let (_, _, mut lod) = built(1024, CHUNK_CELLS, true);
        // A level-1 node well inside, and a head on the line where it splits.
        let g = lod.grids[1];
        let n = (g.first + g.nx * 5 + 5) as usize;
        let node = lod.nodes[n].clone();
        let from = Vec3::new(node.hi.x, (node.lo.y + node.hi.y) * 0.5, (node.lo.z + node.hi.z) * 0.5);
        let at = |d: f32| from + Vec3::X * d;
        // Whole, or part of a node over it: its chunks are not drawn at step 1.
        let chunk = lod.id(0, node.at[0] * 2, node.at[1] * 2).unwrap() as u32;
        let whole = |lod: &mut GroundLod, d: f32| !lod.select(at(d), &[], true).nodes.iter().any(|(k, _)| *k == chunk);
        let _ = n;
        let edge = node.split_under;
        assert!(whole(&mut lod, edge + 20.0), "far out the node is drawn whole");
        let mut flips = 0;
        let mut was = true;
        for sway in 0..200 {
            let d = edge + 0.05 * ((sway as f32) * 0.7).sin();
            let now = whole(&mut lod, d);
            flips += (now != was) as u32;
            was = now;
        }
        assert!(flips <= 1, "{flips} flips in 200 sways of 5 cm about the split line");
        // It joins again only past the band's 10%.
        assert!(!whole(&mut lod, edge * 1.05), "still split 5% out");
        assert!(whole(&mut lod, edge * 1.11), "whole again past 10%");
        // Without the memory the same sway would flip it every time it crossed.
        let crossings = (0..200).map(|s| 0.05 * ((s as f32) * 0.7).sin() < 0.0).collect::<Vec<_>>().windows(2).filter(|w| w[0] != w[1]).count();
        assert!(crossings > 20);
    }

    /// HOLES AND A CAVE'S PATCH: within 60 m such a chunk is at step 1 with
    /// its holes open and its patch in; farther, in a coarser node, the hole
    /// is closed and the patch is not drawn. And beside a coarser neighbour
    /// its border is closed.
    #[test]
    fn a_chunk_with_a_hole_or_a_patch_is_at_step_1_within_60_m_and_closed_beyond() {
        let cells = 1024u32;
        let stride = cells + 1;
        // A flat ground (so nothing but the floor holds it fine) with a hole
        // of 3 x 2 cells on a chunk's east edge, and a patch triangle in it.
        let open = |x: u32, z: u32| (509..512).contains(&x) && (300..302).contains(&z);
        let (mut positions, mut triangles) = ground(cells, &|_, _| 2.0, &open);
        let corner = |x: u32, z: u32| z * stride + x;
        positions.push(positions[corner(510, 301) as usize] - Vec3::Y * 0.5);
        let extra = positions.len() as u32 - 1;
        // The patch: welded to the lattice at three samples of the hole's rim, one of them on the chunk's edge.
        triangles.extend_from_slice(&[corner(512, 300), extra, corner(512, 301), corner(512, 301), extra, corner(512, 302)]);
        let mut lod = GroundLod::build(&GridInput { cells: [cells, cells], positions: &positions, triangles: &triangles, open: Some(&open), chunk_cells: CHUNK_CELLS, lists: true });
        let chunk = lod.id(0, 15, 9).unwrap();
        assert!(lod.nodes[chunk].holes && lod.nodes[chunk].sliver);
        let (lo, hi) = (lod.nodes[chunk].lo, lod.nodes[chunk].hi);
        let patch_drawn = |lod: &GroundLod, sel: &Selection| drawn(lod, sel).contains(&extra);
        let level_of_chunk = |lod: &GroundLod, sel: &Selection| {
            sel.nodes.iter().map(|(n, _)| &lod.nodes[*n as usize]).find(|n| {
                let span = 1u32 << n.level;
                n.at[0] * span <= 15 && 15 < (n.at[0] + 1) * span && n.at[1] * span <= 9 && 9 < (n.at[1] + 1) * span
            }).map(|n| n.level).unwrap()
        };
        let centre = (lo + hi) * 0.5;
        for d in [0.0f32, 20.0, 45.0, 58.0] {
            let head = Vec3::new(hi.x + d, 3.0, centre.z);
            let sel = lod.select(head, &[], true);
            assert_eq!(level_of_chunk(&lod, &sel), 0, "{d} m from the chunk");
            assert!(patch_drawn(&lod, &sel));
            let idx = drawn(&lod, &sel);
            // The hole is open: no triangle on its cells but the patch's.
            assert!(!idx.chunks_exact(3).any(|t| t.iter().all(|v| *v < extra) && t.iter().map(|v| v % stride).min() == Some(510) && t.iter().map(|v| v / stride).min() == Some(300)));
        }
        // Flat ground has no error: past the floor (and its hysteresis) the node over it is whole.
        let far = lod.select(Vec3::new(hi.x + 300.0, 3.0, centre.z), &[], true);
        assert!(level_of_chunk(&lod, &far) >= 1);
        assert!(!patch_drawn(&lod, &far), "the patch is not drawn at a coarser step");
        let idx = drawn(&lod, &far);
        assert_watertight(&positions, cells, &idx, "the hole closed in the coarser node");
        // A flat ground apart from this: every other chunk's node has the design's own distances.
        let g1 = lod.grids[1];
        let plain = &lod.nodes[(g1.first + 3) as usize];
        assert!((plain.split_under - BAND_START[1]).abs() < 1e-3, "{}", plain.split_under);
        // The chunk beside a coarser neighbour: from the west, 40 m off, its
        // own node is split (the floor) while the node east of it, without a
        // hole, is whole from 30 m: the border on the hole's side must close.
        let mut closed_a_stitched_border = false;
        for d in [31.0f32, 34.0, 38.0, 44.0, 50.0] {
            let head = Vec3::new(lo.x - d, 3.0, centre.z);
            let sel = lod.select(head, &[], true);
            let idx = drawn(&lod, &sel);
            // One sheet but for the hole's own rim (and the patch's free edges in it).
            let open_now = open_edges(&positions, &idx);
            let near_hole = |v: u32| v == extra || ((508..=513).contains(&(v % stride)) && (299..=303).contains(&(v / stride)));
            let on_outline = |v: u32| v % stride == 0 || v / stride == 0 || v % stride == cells || v / stride == cells;
            assert!(open_now.iter().all(|(a, b)| (near_hole(*a) && near_hole(*b)) || (on_outline(*a) && on_outline(*b))), "{d} m west: an open edge away from the hole");
            closed_a_stitched_border |= sel.nodes.iter().any(|(n, mask)| *n as usize == chunk && *mask != 0);
        }
        assert!(closed_a_stitched_border, "the chunk with the hole was drawn beside a coarser neighbour");
    }

    /// The error a level stores is what it draws wrong: zero for a plane at
    /// any step, the bump's height for a bump every coarser step skips.
    #[test]
    fn a_nodes_error_is_how_far_its_step_stands_from_the_samples() {
        let (positions, triangles) = ground(256, &|x, z| 0.1 * x - 0.05 * z, &|_, _| false);
        let lod = GroundLod::build(&GridInput { cells: [256, 256], positions: &positions, triangles: &triangles, open: None, chunk_cells: CHUNK_CELLS, lists: true });
        assert!(lod.nodes.iter().all(|n| n.error < 1e-3), "a plane is exact at every step");
        // One sample raised 0.5 m at an odd place: every node over it is 0.5 m out.
        let (mut positions, _) = ground(256, &|_, _| 0.0, &|_, _| false);
        positions[(77 * 257 + 45) as usize].y = 0.5;
        let (_, triangles) = ground(256, &|_, _| 0.0, &|_, _| false);
        let lod = GroundLod::build(&GridInput { cells: [256, 256], positions: &positions, triangles: &triangles, open: None, chunk_cells: CHUNK_CELLS, lists: true });
        for l in 1..=3usize {
            let n = lod.id(l, (45 / 32) >> l, (77 / 32) >> l).unwrap();
            assert!((lod.nodes[n].error - 0.5).abs() < 1e-4, "level {l}: {}", lod.nodes[n].error);
            // And it holds the node fine until the bump is under a pixel -- or the design's cap.
            let want = (0.5 / PIXEL_RAD).min(FORCED[l]).max(BAND_START[l]);
            assert!(lod.nodes[n].split_under >= want - 1e-3, "level {l}: split under {} m, the rule asks {want}", lod.nodes[n].split_under);
        }
        let elsewhere = lod.id(1, 3, 3).unwrap();
        assert_eq!(lod.nodes[elsewhere].error, 0.0);
    }

    /// THE STRICT RULE (`Levers::ground_lod_strict`): nothing is drawn
    /// farther from the step-1 samples than a pixel at its distance, from
    /// anywhere; the design's distance caps do let it (rough ground past
    /// 120 m), which is what they are for. Still one sheet either way.
    #[test]
    fn the_strict_rule_keeps_every_drawn_node_within_a_pixel() {
        // Rough ground: every other sample 0.6 m up, which step 2 skips.
        let rough = |x: f32, z: f32| {
            let (i, j) = (((x + 20.0) / REFERENCE_CELL).round() as i32, ((z - 7.0) / REFERENCE_CELL).round() as i32);
            2.0 * (x * 0.02).sin() + if (i + j) % 2 == 0 { 0.6 } else { 0.0 }
        };
        let build = || {
            let (positions, triangles) = ground(512, &rough, &|_, _| false);
            let lod = GroundLod::build(&GridInput { cells: [512, 512], positions: &positions, triangles: &triangles, open: None, chunk_cells: CHUNK_CELLS, lists: true });
            (positions, lod)
        };
        let (positions, mut lod) = build();
        let side = 512.0 * REFERENCE_CELL;
        let heads: Vec<Vec3> = (0..8).map(|k| Vec3::new(-20.0 + side * (k as f32 * 0.37).fract(), 3.0 + k as f32, 7.0 + side * (k as f32 * 0.61).fract())).collect();
        let mut capped_worst = 0.0f32;
        let mut triangles = [0usize; 2];
        for (which, strict) in [false, true].into_iter().enumerate() {
            lod.set_strict(strict);
            for head in &heads {
                let chosen = lod.select(*head, &[], true);
                let idx = drawn(&lod, &chosen);
                triangles[which] += idx.len() / 3;
                assert_watertight(&positions, 512, &idx, &format!("strict {strict}, head {head}"));
                if strict {
                    // (The hysteresis holds a node whole to 10% inside its line.)
                    assert!(chosen.error_rad <= PIXEL_RAD * (1.0 + HYSTERESIS) + 1e-6, "head {head}: {} mrad drawn", chosen.error_rad * 1000.0);
                } else {
                    capped_worst = capped_worst.max(chosen.error_rad);
                }
            }
        }
        assert!(capped_worst > 2.0 * PIXEL_RAD, "the caps drew {} mrad at most on these hills", capped_worst * 1000.0);
        assert!(triangles[1] > triangles[0], "strict draws more: {triangles:?}");
        // Back again: the distances are the design's as first built.
        lod.set_strict(false);
        let (_, fresh) = build();
        assert!(lod.nodes.iter().zip(&fresh.nodes).all(|(a, b)| a.split_under == b.split_under));
    }

    /// The frame's chunks are the chosen units with boxes that hold them at
    /// any turn of the rig, and runs that touch join into one draw.
    #[test]
    fn the_frames_chunks_hold_their_triangles_at_any_turn_and_touching_runs_join() {
        let (positions, _, mut lod) = built(256, CHUNK_CELLS, true);
        let head = Vec3::new(30.0, 3.0, 60.0);
        for (yaw, offset) in [(0.0f32, Vec3::ZERO), (0.9, Vec3::new(4.0, 1.0, -7.0)), (-2.4, Vec3::new(-30.0, 0.0, 12.0))] {
            let frame = lod.frame(head, &[], true, offset, yaw, 100);
            assert_eq!(frame.chunks.len(), frame.units.len());
            let yaw_inv = glam::Quat::from_rotation_y(-yaw);
            for (c, u) in frame.chunks.iter().zip(&frame.units) {
                let unit = lod.units[*u as usize];
                assert_eq!((c.first_index, c.index_count), (unit.first + 100, unit.count));
                for v in &lod.indices[unit.first as usize..(unit.first + unit.count) as usize] {
                    let p = yaw_inv * (positions[*v as usize] - offset);
                    assert!(p.cmpge(c.min).all() && p.cmple(c.max).all(), "a vertex outside its chunk's box");
                }
            }
            assert!(frame.chunks.windows(2).all(|w| w[0].first_index < w[1].first_index), "ascending, for the passes' search");
        }
        assert_eq!(merged([(0, 6), (6, 3), (12, 3), (15, 0), (15, 9)].into_iter()), vec![(0, 9), (12, 12)]);
    }

    /// The horizon's lift: nothing with the lever off, a pixel of the far
    /// corner's distance for a pixel of error.
    /// A HELD CHUNK (the weather's, `GroundLod::hold`) is at step 1 from
    /// anywhere, its neighbours differ from it by one level at most, and
    /// letting it go gives the tree back as it was.
    /// THE PROVOKING VERTEX SAYS WHICH STEP A TRIANGLE IS (`ground_shape`):
    /// every step-1 lattice triangle starts at a sample odd along x or z,
    /// every coarser one at a sample even along both -- the flat `corner`
    /// the ground reader reads the shape by -- and turning them kept every
    /// triangle's winding.
    #[test]
    fn a_triangles_first_vertex_tells_step_1_from_coarser() {
        let cells = 96;
        let (positions, triangles, lod) = built(cells, 32, true);
        let stride = cells + 1;
        let even = |v: u32| (v % stride) % 2 == 0 && (v / stride) % 2 == 0;
        let up = |t: &[u32]| {
            let (a, b, c) = (positions[t[0] as usize], positions[t[1] as usize], positions[t[2] as usize]);
            (b - a).cross(c - a).y
        };
        let sign = up(&triangles[0..3]).signum();
        let n0 = lod.level0_len() as usize;
        for (i, t) in lod.indices.chunks_exact(3).enumerate() {
            assert_eq!(up(t).signum(), sign, "triangle {i} turned over");
            if i * 3 < n0 {
                assert!(!even(t[0]), "step-1 triangle {i} starts at an even sample");
            }
        }
        // Each coarser node's whole list: every triangle starts even.
        let mut coarse = 0;
        for node in lod.nodes.iter().filter(|n| n.level > 0 && n.whole != NONE) {
            let u = lod.units[node.whole as usize];
            for t in lod.indices[u.first as usize..(u.first + u.count) as usize].chunks_exact(3) {
                assert!(even(t[0]), "a step-{} triangle starts odd", 1 << node.level);
                coarse += 1;
            }
        }
        assert!(coarse > 100, "{coarse} coarser triangles");
    }

    #[test]
    fn a_held_chunk_is_at_step_1_from_anywhere() {
        let (positions, triangles, mut lod) = built(1024, CHUNK_CELLS, true);
        let free: Vec<f32> = lod.nodes.iter().map(|n| n.split_under).collect();
        let held_unit = lod.nodes[5].whole;
        lod.hold(&|u| u == held_unit);
        assert!(lod.nodes.iter().filter(|n| n.held).count() >= 2);
        for head in [Vec3::new(2000.0, 50.0, 2000.0), Vec3::new(-500.0, 10.0, 200.0), Vec3::new(200.0, 30.0, 220.0)] {
            let chosen = lod.select(head, &[], true);
            assert!(chosen.nodes.iter().any(|(n, _)| *n == 5), "from {head} the held chunk is drawn itself");
            let idx = drawn(&lod, &chosen);
            assert_watertight(&positions, 1024, &idx, "held");
        }
        lod.hold(&|_| false);
        let back: Vec<f32> = lod.nodes.iter().map(|n| n.split_under).collect();
        assert_eq!(back, free);
        let _ = triangles;
    }

    #[test]
    fn a_box_is_lifted_by_the_error_at_its_far_corner() {
        let (lo, hi) = (Vec3::new(90.0, 0.0, -5.0), Vec3::new(100.0, 4.0, 5.0));
        assert_eq!(lifted(0.0, Vec3::ZERO, lo, hi), hi);
        let up = lifted(PIXEL_RAD, Vec3::ZERO, lo, hi);
        assert!((up.y - (4.0 + PIXEL_RAD * Vec3::new(100.0, 4.0, 5.0).length())).abs() < 1e-5);
        assert_eq!((up.x, up.z), (hi.x, hi.z));
    }
}
