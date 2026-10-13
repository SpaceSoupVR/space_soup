//! THE GROUND THE GROUND HIDES (`Levers::ground_horizon`).
//!
//! A hill hides the ground, the water and the cave behind it, and all of it
//! was drawn: in the view's frustum, so past the frustum test, rasterized in
//! the reflection pass, the scene pass and the motion pass of each eye, and
//! shading nothing (Quest 3, 2026-10-13: about 12 ns a triangle a pass, 2 ms of
//! the view from behind the cave's hill).
//!
//! The test is a HORIZON: the drawn ground is one sheet over x and z, so a
//! sight line that is above it at one place and under it further on has gone
//! through it between the two -- through its upper side, which is the side
//! that is drawn (the ground is culled by its back) and which writes depth.
//! Whatever lies further along that line is behind a drawn pixel.
//!
//! WHAT MAKES IT EXACT RATHER THAN LIKELY:
//!
//! - The sheet is taken from the triangles the renderer is handed
//!   ([`GroundOccluder::from_triangles`]), not from the heightfield they came
//!   from: ground hidden under a floor, a cave's mouth and the patch round it
//!   are as they are drawn. A cell of the grid counts as ground only where the
//!   triangles cover all of it exactly once, facing up.
//! - A CAVE IS A HOLE. A cell not wholly covered is no ground, and a sight
//!   line that passes over one has proved nothing yet: it must be seen above
//!   the ground again, past the hole, before going under it counts. So a line
//!   into a cave's mouth is never called hidden by the hill it is under, nor
//!   one through the cave and out of its far mouth, nor one from an eye that
//!   stands inside.
//! - Every bound is taken the safe way: the lowest ground of everything a
//!   sector of sight lines can cross to say "under", the highest to say
//!   "above", and both by the whole of [`Horizon::reach`], so one horizon
//!   stands for every eye within that far of where it was taken -- both eyes of
//!   a frame, and the next frames of a head that has hardly moved.
//!
//! Held to the drawn triangles themselves, ray by ray, in this file's tests
//! and on the shipped level in `quest_app::terrain_render`.

use glam::{Vec2, Vec3};

/// Sectors the full turn is cut into: two degrees each.
pub const BINS: usize = 180;

/// How far from where a horizon was taken an eye may be and still use it,
/// metres. Half the distance between the eyes and some head movement.
pub const REACH: f32 = 0.10;

/// How near, in cells of the grid, the steps along a sight line are taken.
const STEP_CELLS: f32 = 2.0;

/// Coverage of a cell, as a share of its area, within this of one is whole.
const WHOLE: f32 = 2e-3;

/// `[lowest, highest]` of the ground over a cell; a cell that is not wholly
/// ground is `HOLE`.
type Span = [f32; 2];
const HOLE: Span = [f32::NEG_INFINITY, f32::INFINITY];

struct Level {
    nx: u32,
    nz: u32,
    spans: Vec<Span>,
}

/// The drawn ground as what it hides: over a grid in the WORLD's x and z, the
/// lowest and highest it reaches in each cell, and whether the cell is ground
/// at all. See the module.
pub struct GroundOccluder {
    origin: Vec2,
    cell: Vec2,
    levels: Vec<Level>,
    /// The cells of the finest level that are ground, for the log.
    pub solid_cells: u32,
}

/// `poly` cut to `keep(p) >= 0`, heights carried along the cut edges.
fn clip(poly: &[Vec3], out: &mut Vec<Vec3>, keep: impl Fn(Vec3) -> f32) {
    out.clear();
    for i in 0..poly.len() {
        let (a, b) = (poly[i], poly[(i + 1) % poly.len()]);
        let (ka, kb) = (keep(a), keep(b));
        if ka >= 0.0 {
            out.push(a);
        }
        if (ka >= 0.0) != (kb >= 0.0) {
            out.push(a + (b - a) * (ka / (ka - kb)));
        }
    }
}

/// The area a polygon covers on the ground plane.
fn area_xz(poly: &[Vec3]) -> f32 {
    let mut twice = 0.0f64;
    for i in 0..poly.len() {
        let (a, b) = (poly[i], poly[(i + 1) % poly.len()]);
        twice += a.x as f64 * b.z as f64 - b.x as f64 * a.z as f64;
    }
    (twice.abs() * 0.5) as f32
}

impl GroundOccluder {
    /// From the ground's triangles as they are drawn, in the WORLD, over a
    /// grid of `cells` across their footprint (the heightfield's own cells
    /// make every cell two triangles; any other count is as safe and hides
    /// less). `None` for no triangles or a footprint without area.
    pub fn from_triangles(positions: &[[f32; 3]], indices: &[u32], cells: [u32; 2]) -> Option<Self> {
        let (nx, nz) = (cells[0].max(1), cells[1].max(1));
        let mut lo = Vec2::splat(f32::INFINITY);
        let mut hi = Vec2::splat(f32::NEG_INFINITY);
        for &i in indices {
            let p = positions.get(i as usize)?;
            lo = lo.min(Vec2::new(p[0], p[2]));
            hi = hi.max(Vec2::new(p[0], p[2]));
        }
        if !(hi.x > lo.x && hi.y > lo.y) {
            return None;
        }
        let cell = (hi - lo) / Vec2::new(nx as f32, nz as f32);
        let n = (nx * nz) as usize;
        let mut covered = vec![0.0f32; n];
        let mut spans = vec![[f32::INFINITY, f32::NEG_INFINITY]; n];
        // A cell a triangle that does not face up reaches into is never ground.
        let mut broken = vec![false; n];
        let (mut a, mut b) = (Vec::with_capacity(8), Vec::with_capacity(8));
        for tri in indices.chunks_exact(3) {
            let p = [0, 1, 2].map(|k| Vec3::from(positions[tri[k] as usize]));
            let up = (p[1] - p[0]).cross(p[2] - p[0]).y;
            let tlo = p[0].min(p[1]).min(p[2]);
            let thi = p[0].max(p[1]).max(p[2]);
            let cell_of = |v: f32, o: f32, c: f32, n: u32| (((v - o) / c).floor().max(0.0) as u32).min(n - 1);
            let (x0, x1) = (cell_of(tlo.x, lo.x, cell.x, nx), cell_of(thi.x, lo.x, cell.x, nx));
            let (z0, z1) = (cell_of(tlo.z, lo.y, cell.y, nz), cell_of(thi.z, lo.y, cell.y, nz));
            for iz in z0..=z1 {
                for ix in x0..=x1 {
                    let k = (iz * nx + ix) as usize;
                    let (cx0, cz0) = (lo.x + ix as f32 * cell.x, lo.y + iz as f32 * cell.y);
                    let (cx1, cz1) = (cx0 + cell.x, cz0 + cell.y);
                    a.clear();
                    a.extend_from_slice(&p);
                    clip(&a, &mut b, |v| v.x - cx0);
                    clip(&b, &mut a, |v| cx1 - v.x);
                    clip(&a, &mut b, |v| v.z - cz0);
                    clip(&b, &mut a, |v| cz1 - v.z);
                    if a.len() < 3 {
                        continue;
                    }
                    let area = area_xz(&a);
                    if area <= 0.0 {
                        continue;
                    }
                    if !(up > 0.0) {
                        broken[k] = true;
                    }
                    covered[k] += area;
                    for v in &a {
                        spans[k][0] = spans[k][0].min(v.y);
                        spans[k][1] = spans[k][1].max(v.y);
                    }
                }
            }
        }
        let whole = cell.x * cell.y;
        let mut solid_cells = 0;
        for k in 0..n {
            let ok = !broken[k] && (covered[k] / whole - 1.0).abs() <= WHOLE && spans[k][0].is_finite() && spans[k][1].is_finite();
            if ok {
                solid_cells += 1;
            } else {
                spans[k] = HOLE;
            }
        }
        let mut levels = vec![Level { nx, nz, spans }];
        while levels.last().is_some_and(|l| l.nx > 1 || l.nz > 1) {
            let below = levels.last().unwrap();
            let (mx, mz) = (below.nx.div_ceil(2), below.nz.div_ceil(2));
            let mut spans = vec![[f32::INFINITY, f32::NEG_INFINITY]; (mx * mz) as usize];
            for iz in 0..below.nz {
                for ix in 0..below.nx {
                    let s = below.spans[(iz * below.nx + ix) as usize];
                    let d = &mut spans[((iz / 2) * mx + ix / 2) as usize];
                    *d = [d[0].min(s[0]), d[1].max(s[1])];
                }
            }
            levels.push(Level { nx: mx, nz: mz, spans });
        }
        Some(Self { origin: lo, cell, levels, solid_cells })
    }

    /// The grid's footprint in the world's x and z.
    pub fn footprint(&self) -> (Vec2, Vec2) {
        let l = &self.levels[0];
        (self.origin, self.origin + self.cell * Vec2::new(l.nx as f32, l.nz as f32))
    }

    /// The lowest and the highest the ground reaches anywhere in the box
    /// `lo..hi` of x and z -- `HOLE` when any of it is not ground, or lies
    /// off the grid. Never a narrower span than the truth.
    fn span(&self, lo: Vec2, hi: Vec2) -> Span {
        let ratio = ((hi - lo) / self.cell).max_element().max(1.0);
        let k = ratio.ceil() as u32;
        let level = ((32 - (k - 1).leading_zeros()) as usize).min(self.levels.len() - 1);
        let l = &self.levels[level];
        let size = self.cell * (1u32 << level) as f32;
        let a = ((lo - self.origin) / size).floor();
        let b = ((hi - self.origin) / size).floor();
        if a.x < 0.0 || a.y < 0.0 || b.x >= l.nx as f32 || b.y >= l.nz as f32 {
            return HOLE;
        }
        let mut out = [f32::INFINITY, f32::NEG_INFINITY];
        for iz in a.y as u32..=b.y as u32 {
            for ix in a.x as u32..=b.x as u32 {
                let s = l.spans[(iz * l.nx + ix) as usize];
                out = [out[0].min(s[0]), out[1].max(s[1])];
            }
        }
        out
    }
}

/// From `from` metres on, a sight line whose slope lies between `low` and
/// `high` has gone through the drawn ground.
#[derive(Clone, Copy, Debug)]
struct Stop {
    from: f32,
    low: f32,
    high: f32,
}

/// The unit directions of the sectors' edges, `BINS + 1` of them from -pi.
fn edges() -> &'static [Vec2] {
    static EDGES: std::sync::OnceLock<Vec<Vec2>> = std::sync::OnceLock::new();
    EDGES.get_or_init(|| {
        (0..=BINS).map(|k| Vec2::from_angle(-std::f32::consts::PI + k as f32 * std::f32::consts::TAU / BINS as f32)).collect()
    })
}

/// What the drawn ground hides from every eye within `reach` of `centre`.
/// Each sector is walked when a box first asks for it. See the module.
pub struct Horizon {
    centre: Vec3,
    reach: f32,
    radii: Vec<f32>,
    bins: Vec<Option<Vec<Stop>>>,
    /// Sectors walked so far, for the log.
    pub walked: u32,
}

impl Horizon {
    pub fn new(ground: &GroundOccluder, centre: Vec3, reach: f32) -> Self {
        let (lo, hi) = ground.footprint();
        let c = Vec2::new(centre.x, centre.z);
        let far = [lo, hi, Vec2::new(lo.x, hi.y), Vec2::new(hi.x, lo.y)].iter().map(|p| p.distance(c)).fold(0.0, f32::max);
        let step = ground.cell.max_element() * STEP_CELLS;
        let grow = 1.0 + std::f32::consts::TAU / BINS as f32;
        let mut radii = vec![0.0f32];
        while *radii.last().unwrap() < far && radii.len() < 4096 {
            let r = *radii.last().unwrap();
            radii.push((r + step).max(r * grow));
        }
        Self { centre, reach, radii, bins: vec![None; BINS], walked: 0 }
    }

    pub fn centre(&self) -> Vec3 {
        self.centre
    }

    /// Whether this horizon stands for an eye at `eye`.
    pub fn covers(&self, eye: Vec3) -> bool {
        eye.distance(self.centre) <= self.reach
    }

    /// One sector, walked outwards. `above` is the lowest slope a sight line
    /// may have and be known to stand above the ground somewhere behind it on
    /// this unbroken run of ground (infinite: not known of any); a line under
    /// the ground ahead with a slope over that has gone through it.
    fn walk(&self, ground: &GroundOccluder, bin: usize) -> Vec<Stop> {
        let e = edges();
        let (d0, d1) = (e[bin], e[bin + 1]);
        // The outer corners pushed out to the tangents, so the four corners'
        // box holds the arc between them.
        let widen = 1.0 / (std::f32::consts::PI / BINS as f32).cos();
        let c = Vec2::new(self.centre.x, self.centre.z);
        let pad = Vec2::splat(self.reach + 1e-3);
        let mut stops: Vec<Stop> = Vec::new();
        let mut above = f32::INFINITY;
        for i in 0..self.radii.len() - 1 {
            let (ra, rb) = (self.radii[i], self.radii[i + 1]);
            let p = [c + d0 * ra, c + d1 * ra, c + d0 * rb * widen, c + d1 * rb * widen];
            let lo = p[0].min(p[1]).min(p[2]).min(p[3]) - pad;
            let hi = p[0].max(p[1]).max(p[2]).max(p[3]) + pad;
            let [lowest, highest] = ground.span(lo, hi);
            if lowest == f32::NEG_INFINITY {
                // A hole, or the grid's edge: what was above the ground before
                // it is not known to be above the ground after it.
                above = f32::INFINITY;
                continue;
            }
            if i >= 1 {
                // Under the ground somewhere in this step: at its near end
                // when the ground is above the eye, at its far end when below.
                let h = lowest - self.reach - self.centre.y;
                let under = if h > 0.0 { h / ra } else { h / rb };
                if under > above {
                    let (low, high) = match stops.last() {
                        Some(s) if above < s.high && under > s.low => (s.low.min(above), s.high.max(under)),
                        Some(s) if under <= s.high => (s.low, s.high),
                        _ => (above, under),
                    };
                    if stops.last().map_or(true, |s| s.low != low || s.high != high) {
                        stops.push(Stop { from: rb, low, high });
                    }
                }
            }
            // Above the ground somewhere in this step, for the steps after it.
            let h = highest + self.reach - self.centre.y;
            let over = if h > 0.0 { h / rb } else if ra > 0.0 { h / ra } else { f32::NEG_INFINITY };
            above = above.min(over);
        }
        stops
    }

    /// Whether everything in the WORLD box `lo..hi` is behind drawn ground
    /// for every eye this horizon covers. Never true of anything such an eye
    /// could see a point of.
    pub fn hides(&mut self, ground: &GroundOccluder, lo: Vec3, hi: Vec3) -> bool {
        let c = Vec2::new(self.centre.x, self.centre.z);
        let (blo, bhi) = (Vec2::new(lo.x, lo.z), Vec2::new(hi.x, hi.z));
        let nearest = c.clamp(blo, bhi).distance(c);
        if self.radii.len() < 3 || nearest <= self.radii[1] + self.reach {
            return false;
        }
        let corners = [blo, bhi, Vec2::new(blo.x, bhi.y), Vec2::new(bhi.x, blo.y)];
        let farthest = corners.iter().map(|p| p.distance(c)).fold(0.0, f32::max);
        let (top, bottom) = (hi.y - self.centre.y, lo.y - self.centre.y);
        let steepest = if top >= 0.0 { top / nearest } else { top / farthest };
        let flattest = if bottom >= 0.0 { bottom / farthest } else { bottom / nearest };
        // The box's sectors: its corners' bearings about its middle's.
        let mid = (blo + bhi) * 0.5 - c;
        let bearing = mid.y.atan2(mid.x);
        let (mut left, mut right) = (0.0f32, 0.0f32);
        for p in corners {
            let d = p - c;
            let mut turn = d.y.atan2(d.x) - bearing;
            if turn > std::f32::consts::PI {
                turn -= std::f32::consts::TAU;
            } else if turn < -std::f32::consts::PI {
                turn += std::f32::consts::TAU;
            }
            left = left.min(turn);
            right = right.max(turn);
        }
        const SLACK: f32 = 1e-3;
        let width = std::f32::consts::TAU / BINS as f32;
        let first = ((bearing + left - SLACK + std::f32::consts::PI) / width).floor() as i32;
        let last = ((bearing + right + SLACK + std::f32::consts::PI) / width).floor() as i32;
        for k in first..=last {
            let bin = k.rem_euclid(BINS as i32) as usize;
            if self.bins[bin].is_none() {
                self.bins[bin] = Some(self.walk(ground, bin));
                self.walked += 1;
            }
            let stops = self.bins[bin].as_ref().unwrap();
            let hidden = stops.iter().rev().find(|s| s.from <= nearest).is_some_and(|s| steepest < s.high && flattest > s.low);
            if !hidden {
                return false;
            }
        }
        true
    }
}

/// A frame's horizon: last frame's while it stands for both eyes, else a new
/// one from between them. `eyes` in the WORLD.
pub fn for_eyes(held: Option<Horizon>, ground: &GroundOccluder, eyes: &[Vec3]) -> Option<Horizon> {
    if eyes.is_empty() {
        return None;
    }
    match held {
        Some(h) if eyes.iter().all(|e| h.covers(*e)) => Some(h),
        _ => {
            let centre = eyes.iter().copied().sum::<Vec3>() / eyes.len() as f32;
            let h = Horizon::new(ground, centre, REACH);
            eyes.iter().all(|e| h.covers(*e)).then_some(h)
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A heightfield's triangles over `n` x `n` cells of `cell` metres from
    /// the origin, both of a cell's triangles left out where `open` says so.
    pub fn field(n: u32, cell: f32, height: impl Fn(f32, f32) -> f32, open: impl Fn(u32, u32) -> bool) -> (Vec<[f32; 3]>, Vec<u32>) {
        let mut positions = Vec::new();
        for iz in 0..=n {
            for ix in 0..=n {
                let (x, z) = (ix as f32 * cell, iz as f32 * cell);
                positions.push([x, height(x, z), z]);
            }
        }
        let mut indices = Vec::new();
        let at = |ix: u32, iz: u32| iz * (n + 1) + ix;
        for iz in 0..n {
            for ix in 0..n {
                if open(ix, iz) {
                    continue;
                }
                // Counter-clockwise from above: the upper side is the front.
                indices.extend_from_slice(&[at(ix, iz), at(ix, iz + 1), at(ix + 1, iz), at(ix + 1, iz), at(ix, iz + 1), at(ix + 1, iz + 1)]);
            }
        }
        (positions, indices)
    }

    /// Whether the segment from `eye` to `to` goes through the upper side of
    /// one of the triangles before it arrives: the drawn ground, asked ray by
    /// ray. What `hides` is held to.
    pub fn blocked(positions: &[[f32; 3]], indices: &[u32], eye: Vec3, to: Vec3) -> bool {
        let dir = to - eye;
        indices.chunks_exact(3).any(|t| {
            let [a, b, c] = [0, 1, 2].map(|k| Vec3::from(positions[t[k] as usize]));
            let n = (b - a).cross(c - a);
            let facing = n.dot(dir);
            if facing >= 0.0 {
                return false;
            }
            let t_hit = n.dot(a - eye) / facing;
            if !(t_hit > 1e-4 && t_hit < 1.0 - 1e-4) {
                return false;
            }
            let p = eye + dir * t_hit;
            let inside = |u: Vec3, v: Vec3| (v - u).cross(p - u).dot(n) >= 0.0;
            inside(a, b) && inside(b, c) && inside(c, a)
        })
    }

    /// A ridge 8 m high along x = 20, on level ground 40 m square.
    fn ridge(x: f32, _z: f32) -> f32 {
        (8.0 - (x - 20.0).abs() * 2.0).max(0.0)
    }

    fn horizon(positions: &[[f32; 3]], indices: &[u32], n: u32, eye: Vec3) -> (GroundOccluder, Horizon) {
        let g = GroundOccluder::from_triangles(positions, indices, [n, n]).unwrap();
        let h = Horizon::new(&g, eye, REACH);
        (g, h)
    }

    #[test]
    fn a_cell_is_ground_only_where_the_triangles_cover_all_of_it() {
        let (p, i) = field(8, 1.0, |_, _| 0.0, |ix, iz| ix == 3 && iz == 4);
        let g = GroundOccluder::from_triangles(&p, &i, [8, 8]).unwrap();
        assert_eq!(g.solid_cells, 63);
        assert_eq!(g.span(Vec2::new(3.2, 4.2), Vec2::new(3.8, 4.8)), HOLE);
        assert_eq!(g.span(Vec2::new(2.2, 4.2), Vec2::new(2.8, 4.8)), [0.0, 0.0]);
        // A box that reaches the hole, or off the grid, is not ground.
        assert_eq!(g.span(Vec2::new(2.2, 4.2), Vec2::new(3.1, 4.8)), HOLE);
        assert_eq!(g.span(Vec2::new(-0.5, 1.0), Vec2::new(0.5, 2.0)), HOLE);
        // One triangle of a cell is not the cell.
        let half = &i[..i.len() - 3];
        assert_eq!(GroundOccluder::from_triangles(&p, half, [8, 8]).unwrap().solid_cells, 62);
        // A grid that is not the triangles' own is still only ground where it
        // is covered, with the heights of everything in each cell.
        let (p, i) = field(8, 1.0, |x, _| x, |ix, _| ix == 7);
        let g = GroundOccluder::from_triangles(&p, &i, [3, 3]).unwrap();
        assert_eq!(g.solid_cells, 9);
        let s = g.span(Vec2::new(0.1, 0.1), Vec2::new(0.2, 0.2));
        assert!((s[0] - 0.0).abs() < 1e-5 && (s[1] - 7.0 / 3.0).abs() < 1e-4, "{s:?}");
    }

    #[test]
    fn a_ridge_hides_what_stands_behind_it_and_nothing_else() {
        let (p, i) = field(80, 0.5, ridge, |_, _| false);
        let eye = Vec3::new(8.0, 1.6, 20.0);
        let (g, mut h) = horizon(&p, &i, 80, eye);
        // Behind the ridge, on the ground: hidden.
        assert!(h.hides(&g, Vec3::new(28.0, 0.0, 16.0), Vec3::new(34.0, 1.0, 24.0)));
        // The same box tall enough to stand over the ridge's line: not.
        assert!(!h.hides(&g, Vec3::new(28.0, 0.0, 16.0), Vec3::new(34.0, 12.0, 24.0)));
        // In front of the ridge, and the ridge's own near face: not.
        assert!(!h.hides(&g, Vec3::new(12.0, 0.0, 16.0), Vec3::new(14.0, 1.0, 24.0)));
        assert!(!h.hides(&g, Vec3::new(16.0, 0.0, 16.0), Vec3::new(19.0, 6.0, 24.0)));
        // A box the eye stands in or beside: not.
        assert!(!h.hides(&g, Vec3::new(7.0, 0.0, 19.0), Vec3::new(9.0, 1.0, 21.0)));
        // From the ridge's top everything shows.
        let (g, mut h) = horizon(&p, &i, 80, Vec3::new(20.0, 9.6, 20.0));
        assert!(!h.hides(&g, Vec3::new(28.0, 0.0, 16.0), Vec3::new(34.0, 1.0, 24.0)));
    }

    #[test]
    fn a_cave_through_the_ridge_shows_what_is_behind_it() {
        // Mouths cut in both faces of the ridge round z = 20; the tunnel
        // between them runs under ground that is still drawn.
        let mouth = |ix: u32, iz: u32| {
            let (x, z) = (ix as f32 * 0.5, iz as f32 * 0.5);
            (19.0..21.0).contains(&z) && ((16.0..17.5).contains(&x) || (22.5..24.0).contains(&x))
        };
        let (p, i) = field(80, 0.5, ridge, mouth);
        let eye = Vec3::new(8.0, 1.0, 20.0);
        let (g, mut h) = horizon(&p, &i, 80, eye);
        // Straight through the cave: seen.
        assert!(!h.hides(&g, Vec3::new(28.0, 0.0, 19.5), Vec3::new(30.0, 1.0, 20.5)));
        // Behind the ridge away from the cave: hidden as before.
        assert!(h.hides(&g, Vec3::new(28.0, 0.0, 30.0), Vec3::new(34.0, 1.0, 36.0)));
        assert!(h.hides(&g, Vec3::new(28.0, 0.0, 4.0), Vec3::new(34.0, 1.0, 10.0)));
        // A box that the cave's sectors only touch is seen whole.
        assert!(!h.hides(&g, Vec3::new(28.0, 0.0, 12.0), Vec3::new(34.0, 1.0, 20.0)));
        // From inside the cave, under the ridge: out of either mouth is seen,
        // and so is the far side's ground past the mouth.
        let inside = Vec3::new(20.0, 0.8, 20.0);
        assert!(!blocked(&p, &i, inside, Vec3::new(30.0, 0.5, 20.0)), "the fixture: the far mouth is open");
        let (g, mut h) = horizon(&p, &i, 80, inside);
        assert!(!h.hides(&g, Vec3::new(28.0, 0.0, 19.5), Vec3::new(30.0, 1.0, 20.5)));
        assert!(!h.hides(&g, Vec3::new(8.0, 0.0, 19.5), Vec3::new(10.0, 1.0, 20.5)));
    }

    #[test]
    fn one_horizon_stands_for_the_eyes_near_it_and_no_others() {
        let (p, i) = field(16, 1.0, |_, _| 0.0, |_, _| false);
        let g = GroundOccluder::from_triangles(&p, &i, [16, 16]).unwrap();
        let eyes = [Vec3::new(8.0, 1.6, 8.0), Vec3::new(8.064, 1.6, 8.0)];
        let h = for_eyes(None, &g, &eyes).unwrap();
        let centre = h.centre();
        assert!(eyes.iter().all(|e| h.covers(*e)));
        // A head that has hardly moved keeps it; one that has gets a new one.
        let h = for_eyes(Some(h), &g, &eyes.map(|e| e + Vec3::X * 0.03)).unwrap();
        assert_eq!(h.centre(), centre);
        let h = for_eyes(Some(h), &g, &eyes.map(|e| e + Vec3::X * 0.2)).unwrap();
        assert_ne!(h.centre(), centre);
        // Eyes too far apart for one horizon get none: nothing is hidden.
        assert!(for_eyes(None, &g, &[Vec3::ZERO, Vec3::X]).is_none());
    }

    /// EVERY BOX CALLED HIDDEN IS HIDDEN, RAY BY RAY: hills, pits and holes
    /// from a fixed seed, eyes on and over the ground (and under it, in the
    /// holes), boxes everywhere; each box `hides` answers for is asked of the
    /// triangles themselves from eyes across the horizon's reach to points
    /// across the box. And it does answer for a fair share of them, so the
    /// test is of something.
    #[test]
    fn nothing_called_hidden_can_be_seen() {
        let mut seed = 0x2545_f491u32;
        let mut rnd = move || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            (seed >> 8) as f32 / (1u32 << 24) as f32
        };
        let (mut hidden, mut asked) = (0, 0);
        for world in 0..6 {
            let hills: Vec<(f32, f32, f32, f32)> = (0..7).map(|_| (rnd() * 32.0, rnd() * 32.0, 2.0 + rnd() * 5.0, rnd() * 12.0 - 3.0)).collect();
            let height = |x: f32, z: f32| hills.iter().map(|&(hx, hz, w, h)| h * (-((x - hx).powi(2) + (z - hz).powi(2)) / (w * w)).exp()).sum::<f32>();
            let holes: Vec<(u32, u32)> = (0..(world * 3)).map(|_| ((rnd() * 62.0) as u32, (rnd() * 62.0) as u32)).collect();
            let (p, i) = field(64, 0.5, height, |ix, iz| holes.iter().any(|&(hx, hz)| ix.abs_diff(hx) <= 1 && iz.abs_diff(hz) <= 1));
            let g = GroundOccluder::from_triangles(&p, &i, [64, 64]).unwrap();
            for _ in 0..24 {
                let (ex, ez) = (2.0 + rnd() * 28.0, 2.0 + rnd() * 28.0);
                let eye = Vec3::new(ex, height(ex, ez) + rnd() * 3.0 - 0.5, ez);
                let mut h = Horizon::new(&g, eye, REACH);
                for _ in 0..40 {
                    let lo = Vec3::new(rnd() * 30.0, -4.0 + rnd() * 6.0, rnd() * 30.0);
                    let hi = lo + Vec3::new(0.2 + rnd() * 4.0, rnd() * 3.0, 0.2 + rnd() * 4.0);
                    asked += 1;
                    if !h.hides(&g, lo, hi) {
                        continue;
                    }
                    hidden += 1;
                    for _ in 0..12 {
                        let from = eye + (Vec3::new(rnd(), rnd(), rnd()) * 2.0 - 1.0).normalize_or_zero() * REACH * rnd();
                        let to = lo + (hi - lo) * Vec3::new(rnd(), rnd(), rnd());
                        assert!(blocked(&p, &i, from, to), "world {world}: {to:?} is seen from {from:?} (box {lo:?}..{hi:?}, horizon at {eye:?})");
                    }
                }
            }
        }
        assert!(hidden * 12 > asked, "only {hidden} of {asked} boxes hidden: the test proves little");
    }
}
