use glam::{Vec2, Vec3};

pub static CHUNKS_A_SIDE: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

pub const BINS: usize = 180;

pub const REACH: f32 = 0.10;

pub const SLIVER: f32 = 4e-3;

const STEP_CELLS: f32 = 2.0;

const WHOLE: f32 = 2e-3;

type Span = [f32; 2];
const HOLE: Span = [f32::NEG_INFINITY, f32::INFINITY];

struct Level {
    nx: u32,
    nz: u32,
    spans: Vec<Span>,
}

pub struct GroundOccluder {
    origin: Vec2,
    cell: Vec2,
    levels: Vec<Level>,

    pub solid_cells: u32,
}

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

fn area_xz(poly: &[Vec3]) -> f32 {
    let mut twice = 0.0f64;
    for i in 0..poly.len() {
        let (a, b) = (poly[i], poly[(i + 1) % poly.len()]);
        twice += a.x as f64 * b.z as f64 - b.x as f64 * a.z as f64;
    }
    (twice.abs() * 0.5) as f32
}

impl GroundOccluder {

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
        let solid: Vec<bool> = (0..n)
            .map(|k| !broken[k] && (covered[k] / whole - 1.0).abs() <= WHOLE && spans[k][0].is_finite() && spans[k][1].is_finite())
            .collect();

        let mut seen = vec![false; n];
        let mut stack: Vec<usize> = Vec::new();
        let mut members: Vec<usize> = Vec::new();
        for start in 0..n {
            if solid[start] {
                solid_cells += 1;
                continue;
            }
            if seen[start] {
                continue;
            }
            let mut top = f32::NEG_INFINITY;
            members.clear();
            stack.push(start);
            seen[start] = true;
            while let Some(k) = stack.pop() {
                members.push(k);
                let (ix, iz) = ((k as u32 % nx) as i64, (k as u32 / nx) as i64);
                for (dx, dz) in [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)] {
                    let (jx, jz) = (ix + dx, iz + dz);
                    if jx < 0 || jz < 0 || jx >= nx as i64 || jz >= nz as i64 {
                        top = f32::INFINITY;
                        continue;
                    }
                    let j = (jz as u32 * nx + jx as u32) as usize;
                    if solid[j] {
                        top = top.max(spans[j][1]);
                    } else if !seen[j] {
                        seen[j] = true;
                        stack.push(j);
                    }
                }
            }
            for &k in &members {
                spans[k] = [f32::NEG_INFINITY, top];
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

    pub fn footprint(&self) -> (Vec2, Vec2) {
        let l = &self.levels[0];
        (self.origin, self.origin + self.cell * Vec2::new(l.nx as f32, l.nz as f32))
    }

    fn span(&self, lo: Vec2, hi: Vec2) -> Span {

        let ratio = (((hi - lo) / self.cell).max_element() / 3.0).max(1.0);
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

type Slopes = (f32, f32);
const NONE: Slopes = (f32::INFINITY, f32::NEG_INFINITY);

fn within(s: Slopes, flattest: f32, steepest: f32) -> bool {
    flattest > s.0 && steepest < s.1
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Stop {
    from: f32,
    through: Slopes,
    sunk: Slopes,
}

fn edges() -> &'static [Vec2] {
    static EDGES: std::sync::OnceLock<Vec<Vec2>> = std::sync::OnceLock::new();
    EDGES.get_or_init(|| {
        (0..=BINS).map(|k| Vec2::from_angle(-std::f32::consts::PI + k as f32 * std::f32::consts::TAU / BINS as f32)).collect()
    })
}

pub struct Horizon {
    centre: Vec3,
    reach: f32,
    radii: Vec<f32>,
    bins: Vec<Option<Vec<Stop>>>,

    plugged: bool,

    mouths: Vec<(Vec2, Vec2)>,

    pub walked: u32,
}

impl Horizon {

    pub fn new(ground: &GroundOccluder, centre: Vec3, reach: f32, caves: &[(Vec3, Vec3)]) -> Self {
        let (lo, hi) = ground.footprint();
        let c = Vec2::new(centre.x, centre.z);

        let plugged = Self::plugged_at(ground, centre, reach, caves);
        let far = [lo, hi, Vec2::new(lo.x, hi.y), Vec2::new(hi.x, lo.y)].iter().map(|p| p.distance(c)).fold(0.0, f32::max);
        let step = ground.cell.max_element() * STEP_CELLS;
        let grow = 1.0 + std::f32::consts::TAU / BINS as f32;
        let mut radii = vec![0.0f32];
        while *radii.last().unwrap() < far && radii.len() < 4096 {
            let r = *radii.last().unwrap();
            radii.push((r + step).max(r * grow));
        }
        let grow = Vec2::splat(reach + ground.cell.max_element());
        let mouths = caves.iter().map(|(l, h)| (Vec2::new(l.x, l.z) - grow, Vec2::new(h.x, h.z) + grow)).collect();
        Self { centre, reach, radii, bins: vec![None; BINS], plugged, mouths, walked: 0 }
    }

    pub fn plugged(&self) -> bool {
        self.plugged
    }

    fn plugged_at(ground: &GroundOccluder, centre: Vec3, reach: f32, caves: &[(Vec3, Vec3)]) -> bool {
        let c = Vec2::new(centre.x, centre.z);

        let about = Vec2::splat(reach);
        let over_ground = centre.y - reach > ground.span(c - about, c + about)[1];
        let in_a_cave = caves.iter().any(|(l, h)| centre.cmpge(*l - reach).all() && centre.cmple(*h + reach).all());
        !caves.is_empty() && (over_ground || in_a_cave)
    }

    pub fn centre(&self) -> Vec3 {
        self.centre
    }

    pub fn covers(&self, eye: Vec3) -> bool {
        eye.distance(self.centre) <= self.reach
    }

    fn walk(&self, ground: &GroundOccluder, bin: usize) -> Vec<Stop> {
        let e = edges();
        let (d0, d1) = (e[bin], e[bin + 1]);

        let widen = 1.0 / (std::f32::consts::PI / BINS as f32).cos();
        let c = Vec2::new(self.centre.x, self.centre.z);
        let pad = Vec2::splat(self.reach + 1e-3);
        let mut stops: Vec<Stop> = Vec::new();
        let mut above = f32::INFINITY;
        let (mut through, mut sunk) = (NONE, NONE);
        for i in 0..self.radii.len() - 1 {
            let (ra, rb) = (self.radii[i], self.radii[i + 1]);
            let p = [c + d0 * ra, c + d1 * ra, c + d0 * rb * widen, c + d1 * rb * widen];
            let lo = p[0].min(p[1]).min(p[2]).min(p[3]) - pad;
            let hi = p[0].max(p[1]).max(p[2]).max(p[3]) + pad;
            let [lowest, highest] = ground.span(lo, hi);

            let h = highest + self.reach - self.centre.y;
            let clears = if h > 0.0 { if ra > 0.0 { h / ra } else { f32::INFINITY } } else { h / rb };
            let from;
            if lowest == f32::NEG_INFINITY {

                above = clears;

                let a_mouth = highest == f32::INFINITY || self.mouths.iter().any(|(l, h)| lo.cmple(*h).all() && hi.cmpge(*l).all());
                if sunk != NONE && a_mouth {
                    sunk.0 = sunk.0.max(clears);
                    if !(sunk.0 < sunk.1) {
                        sunk = NONE;
                    }
                }

                from = ra;
            } else {
                if i >= 1 {

                    let d = lowest - self.reach - self.centre.y;
                    let under = if d > 0.0 { d / ra } else { d / rb };
                    if under > above {
                        through = if through == NONE {
                            (above, under)
                        } else if above < through.1 && under > through.0 {
                            (through.0.min(above), through.1.max(under))
                        } else if under > through.1 {
                            (above, under)
                        } else {
                            through
                        };
                    }
                    if self.plugged {
                        sunk = if sunk == NONE || under > sunk.0 {
                            (f32::NEG_INFINITY, if sunk == NONE { under } else { sunk.1.max(under) })
                        } else {
                            sunk
                        };
                    }
                }

                let over = if h > 0.0 { h / rb } else if ra > 0.0 { h / ra } else { f32::NEG_INFINITY };
                above = above.min(over);
                from = rb;
            }
            let stop = Stop { from, through, sunk };
            if stops.last().map_or(through != NONE || sunk != NONE, |s| s.through != through || s.sunk != sunk) {
                stops.push(stop);
            }
        }
        stops
    }

    pub fn hides(&mut self, ground: &GroundOccluder, lo: Vec3, hi: Vec3) -> bool {
        self.hidden(ground, lo, hi, false)
    }

    pub fn hides_ground(&mut self, ground: &GroundOccluder, lo: Vec3, hi: Vec3) -> bool {
        self.hidden(ground, lo, hi, true)
    }

    fn hidden(&mut self, ground: &GroundOccluder, lo: Vec3, hi: Vec3, on_the_sheet: bool) -> bool {
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

        let steepest = steepest + SLIVER * (1.0 + steepest * steepest);
        let flattest = flattest - SLIVER * (1.0 + flattest * flattest);

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

            let Some(first) = stops.iter().rposition(|s| s.from <= nearest) else { return false };
            for (k, s) in stops[first..].iter().enumerate() {
                if k > 0 && s.from > farthest {
                    break;
                }
                let hidden = within(s.through, flattest, steepest)
                    || (on_the_sheet
                        && (within(s.sunk, flattest, steepest)
                            || (s.through.0 < s.sunk.1
                                && s.sunk.0 < s.through.1
                                && within((s.through.0.min(s.sunk.0), s.through.1.max(s.sunk.1)), flattest, steepest))));
                if !hidden {
                    return false;
                }
                if within(s.through, flattest, steepest) {
                    break;
                }
            }
        }
        true
    }
}

pub fn for_eyes(held: Option<Horizon>, ground: &GroundOccluder, eyes: &[Vec3], caves: &[(Vec3, Vec3)]) -> Option<Horizon> {
    if eyes.is_empty() {
        return None;
    }
    let centre = eyes.iter().copied().sum::<Vec3>() / eyes.len() as f32;
    match held {

        Some(h) if eyes.iter().all(|e| h.covers(*e)) && h.plugged == Horizon::plugged_at(ground, h.centre, h.reach, caves) => Some(h),
        _ => {
            let h = Horizon::new(ground, centre, REACH, caves);
            eyes.iter().all(|e| h.covers(*e)).then_some(h)
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

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

                indices.extend_from_slice(&[at(ix, iz), at(ix, iz + 1), at(ix + 1, iz), at(ix + 1, iz), at(ix, iz + 1), at(ix + 1, iz + 1)]);
            }
        }
        (positions, indices)
    }

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

    fn ridge(x: f32, _z: f32) -> f32 {
        (8.0 - (x - 20.0).abs() * 2.0).max(0.0)
    }

    fn horizon(positions: &[[f32; 3]], indices: &[u32], n: u32, eye: Vec3) -> (GroundOccluder, Horizon) {
        let g = GroundOccluder::from_triangles(positions, indices, [n, n]).unwrap();
        let h = Horizon::new(&g, eye, REACH, &[]);
        (g, h)
    }

    #[test]
    fn a_cell_is_ground_only_where_the_triangles_cover_all_of_it() {
        let (p, i) = field(8, 1.0, |_, _| 0.0, |ix, iz| ix == 3 && iz == 4);
        let g = GroundOccluder::from_triangles(&p, &i, [8, 8]).unwrap();
        assert_eq!(g.solid_cells, 63);

        assert_eq!(g.span(Vec2::new(3.2, 4.2), Vec2::new(3.8, 4.8)), [f32::NEG_INFINITY, 0.0]);
        assert_eq!(g.span(Vec2::new(2.2, 4.2), Vec2::new(2.8, 4.8)), [0.0, 0.0]);

        assert_eq!(g.span(Vec2::new(2.2, 4.2), Vec2::new(3.1, 4.8)), [f32::NEG_INFINITY, 0.0]);
        assert_eq!(g.span(Vec2::new(-0.5, 1.0), Vec2::new(0.5, 2.0)), HOLE);

        let (p2, i2) = field(8, 1.0, |x, _| x, |ix, iz| (ix == 3 || ix == 4) && iz == 4 || ix == 0 && iz == 0);
        let g2 = GroundOccluder::from_triangles(&p2, &i2, [8, 8]).unwrap();
        assert_eq!(g2.span(Vec2::new(3.2, 4.2), Vec2::new(3.8, 4.8)), [f32::NEG_INFINITY, 6.0]);
        assert_eq!(g2.span(Vec2::new(0.2, 0.2), Vec2::new(0.8, 0.8)), HOLE);

        let half = &i[..i.len() - 3];
        assert_eq!(GroundOccluder::from_triangles(&p, half, [8, 8]).unwrap().solid_cells, 62);

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

        assert!(h.hides(&g, Vec3::new(28.0, 0.0, 16.0), Vec3::new(34.0, 1.0, 24.0)));

        assert!(!h.hides(&g, Vec3::new(28.0, 0.0, 16.0), Vec3::new(34.0, 12.0, 24.0)));

        assert!(!h.hides(&g, Vec3::new(12.0, 0.0, 16.0), Vec3::new(14.0, 1.0, 24.0)));
        assert!(!h.hides(&g, Vec3::new(16.0, 0.0, 16.0), Vec3::new(19.0, 6.0, 24.0)));

        assert!(!h.hides(&g, Vec3::new(7.0, 0.0, 19.0), Vec3::new(9.0, 1.0, 21.0)));

        let (g, mut h) = horizon(&p, &i, 80, Vec3::new(20.0, 9.6, 20.0));
        assert!(!h.hides(&g, Vec3::new(28.0, 0.0, 16.0), Vec3::new(34.0, 1.0, 24.0)));
    }

    #[test]
    fn a_cave_through_the_ridge_shows_what_is_behind_it() {

        let mouth = |ix: u32, iz: u32| {
            let (x, z) = (ix as f32 * 0.5, iz as f32 * 0.5);
            (19.0..21.0).contains(&z) && ((16.0..17.5).contains(&x) || (22.5..24.0).contains(&x))
        };
        let (p, i) = field(80, 0.5, ridge, mouth);
        let eye = Vec3::new(8.0, 1.0, 20.0);
        let (g, mut h) = horizon(&p, &i, 80, eye);

        assert!(!h.hides(&g, Vec3::new(28.0, 0.0, 19.5), Vec3::new(30.0, 1.0, 20.5)));

        assert!(h.hides(&g, Vec3::new(28.0, 0.0, 30.0), Vec3::new(34.0, 1.0, 36.0)));
        assert!(h.hides(&g, Vec3::new(28.0, 0.0, 4.0), Vec3::new(34.0, 1.0, 10.0)));

        assert!(!h.hides(&g, Vec3::new(28.0, 0.0, 12.0), Vec3::new(34.0, 1.0, 20.0)));

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
        let h = for_eyes(None, &g, &eyes, &[]).unwrap();
        let centre = h.centre();
        assert!(eyes.iter().all(|e| h.covers(*e)));

        let h = for_eyes(Some(h), &g, &eyes.map(|e| e + Vec3::X * 0.03), &[]).unwrap();
        assert_eq!(h.centre(), centre);
        let h = for_eyes(Some(h), &g, &eyes.map(|e| e + Vec3::X * 0.2), &[]).unwrap();
        assert_ne!(h.centre(), centre);

        assert!(for_eyes(None, &g, &[Vec3::ZERO, Vec3::X], &[]).is_none());
    }

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
                let mut h = Horizon::new(&g, eye, REACH, &[]);
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

    fn ridge_with_a_cave() -> ((Vec<[f32; 3]>, Vec<u32>), Vec<[Vec3; 3]>, (Vec3, Vec3)) {
        let mouth = |ix: u32, iz: u32| {
            let (x, z) = (ix as f32 * 0.5, iz as f32 * 0.5);
            (19.0..21.0).contains(&z) && ((16.0..17.5).contains(&x) || (22.5..24.0).contains(&x))
        };
        let ground = field(80, 0.5, ridge, mouth);
        let mut rock: Vec<[Vec3; 3]> = Vec::new();
        let mut quad = |a: Vec3, b: Vec3, c: Vec3, d: Vec3| {
            rock.push([a, b, c]);
            rock.push([a, c, d]);
        };
        let v = Vec3::new;
        quad(v(16.0, 0.0, 19.0), v(24.0, 0.0, 19.0), v(24.0, 0.0, 21.0), v(16.0, 0.0, 21.0));
        quad(v(17.0, 2.0, 19.0), v(23.0, 2.0, 19.0), v(23.0, 2.0, 21.0), v(17.0, 2.0, 21.0));
        for z in [19.0, 21.0] {
            quad(v(16.0, 0.0, z), v(24.0, 0.0, z), v(23.0, 2.0, z), v(17.0, 2.0, z));
        }
        quad(v(17.0, 2.0, 19.0), v(17.5, 3.0, 19.0), v(17.5, 3.0, 21.0), v(17.0, 2.0, 21.0));
        quad(v(23.0, 2.0, 19.0), v(22.5, 3.0, 19.0), v(22.5, 3.0, 21.0), v(23.0, 2.0, 21.0));
        (ground, rock, (v(16.0, 0.0, 19.0), v(24.0, 3.0, 21.0)))
    }

    fn meets(rock: &[[Vec3; 3]], eye: Vec3, to: Vec3) -> bool {
        let dir = to - eye;
        rock.iter().any(|&[a, b, c]| {
            let n = (b - a).cross(c - a);
            let facing = n.dot(dir);
            if facing.abs() < 1e-12 {
                return false;
            }
            let t = n.dot(a - eye) / facing;
            if !(t > 1e-4 && t < 1.0 - 1e-4) {
                return false;
            }
            let p = eye + dir * t;
            let slack = -1e-4 * n.length();
            let inside = |u: Vec3, w: Vec3| (w - u).cross(p - u).dot(n) >= slack * (w - u).length();
            inside(a, b) && inside(b, c) && inside(c, a)
        })
    }

    #[test]
    fn from_inside_a_plugged_cave_only_its_mouths_show_the_ground() {
        let ((p, i), _, cave) = ridge_with_a_cave();
        let g = GroundOccluder::from_triangles(&p, &i, [80, 80]).unwrap();
        let inside = Vec3::new(20.0, 0.8, 20.0);
        let side = (Vec3::new(28.0, 0.0, 30.0), Vec3::new(34.0, 1.0, 36.0));
        let ahead = (Vec3::new(26.0, 0.0, 19.5), Vec3::new(30.0, 1.0, 20.5));

        let mut h = Horizon::new(&g, inside, REACH, &[cave]);
        assert!(h.plugged());
        assert!(h.hides_ground(&g, side.0, side.1));
        assert!(!h.hides_ground(&g, ahead.0, ahead.1));
        assert!(!h.hides_ground(&g, Vec3::new(8.0, 0.0, 19.5), Vec3::new(12.0, 1.0, 20.5)));

        assert!(!h.hides(&g, side.0, side.1));

        let mut open = Horizon::new(&g, inside, REACH, &[]);
        assert!(!open.plugged() && !open.hides_ground(&g, side.0, side.1));

        let mut buried = Horizon::new(&g, Vec3::new(20.0, 3.0, 30.0), REACH, &[cave]);
        assert!(!buried.plugged() && !buried.hides_ground(&g, side.0, side.1));

        let mut out = Horizon::new(&g, Vec3::new(8.0, 1.0, 20.0), REACH, &[cave]);
        assert!(out.plugged());
        assert!(out.hides_ground(&g, Vec3::new(28.0, 0.0, 4.0), Vec3::new(34.0, 1.0, 10.0)));
        assert!(!out.hides_ground(&g, Vec3::new(28.0, 0.0, 19.5), Vec3::new(30.0, 1.0, 20.5)));
        assert!(!out.hides_ground(&g, Vec3::new(28.0, 0.0, 12.0), Vec3::new(34.0, 1.0, 20.0)));
    }

    #[test]
    fn no_ground_called_hidden_from_a_plugged_cave_can_be_seen() {
        let ((p, i), rock, cave) = ridge_with_a_cave();
        let g = GroundOccluder::from_triangles(&p, &i, [80, 80]).unwrap();
        let mut seed = 0x9e37_79b9u32;
        let mut rnd = move || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            (seed >> 8) as f32 / (1u32 << 24) as f32
        };
        let tris: Vec<([Vec3; 3], Vec3)> = i
            .chunks_exact(3)
            .map(|t| {
                let c = [0, 1, 2].map(|k| Vec3::from(p[t[k] as usize]));
                (c, (c[0] + c[1] + c[2]) / 3.0)
            })
            .collect();
        let (mut hidden, mut by_the_cave, mut asked, mut rays) = (0, 0, 0, 0);
        for e in 0..60 {
            let eye = match e % 3 {

                0 => Vec3::new(17.6 + rnd() * 4.8, 0.2 + rnd() * 1.6, 19.2 + rnd() * 1.6),

                1 => Vec3::new(if rnd() < 0.5 { 4.0 + rnd() * 11.0 } else { 25.0 + rnd() * 11.0 }, 0.4 + rnd() * 2.0, 17.0 + rnd() * 6.0),
                _ => Vec3::new(2.0 + rnd() * 36.0, ridge(0.0, 0.0) + 9.0 * rnd() + 8.2 * (rnd() < 0.3) as u8 as f32, 2.0 + rnd() * 36.0),
            };
            let mut h = Horizon::new(&g, eye, REACH, &[cave]);
            if !h.plugged() {
                continue;
            }
            for _ in 0..60 {
                let lo = Vec3::new(rnd() * 36.0, -0.5, rnd() * 36.0);
                let hi = lo + Vec3::new(0.5 + rnd() * 4.0, 0.6 + rnd() * 9.0, 0.5 + rnd() * 4.0);
                asked += 1;
                if !h.hides_ground(&g, lo, hi) {
                    continue;
                }
                hidden += 1;
                by_the_cave += !h.hides(&g, lo, hi) as u32;
                for (corners, mid) in tris.iter().filter(|(_, m)| m.cmpge(lo).all() && m.cmple(hi).all()).step_by(3) {
                    let from = eye + (Vec3::new(rnd(), rnd(), rnd()) * 2.0 - 1.0).normalize_or_zero() * REACH * rnd();
                    let n = (corners[1] - corners[0]).cross(corners[2] - corners[0]);
                    if n.dot(from - *mid) <= 0.0 {
                        continue;
                    }
                    rays += 1;
                    assert!(
                        blocked(&p, &i, from, *mid) || meets(&rock, from, *mid),
                        "the ground at {mid:?} is seen from {from:?} (box {lo:?}..{hi:?}, horizon at {eye:?})"
                    );
                }
            }
        }
        assert!(hidden * 10 > asked && by_the_cave * 4 > hidden && rays > 2000, "{hidden} of {asked} boxes hidden, {by_the_cave} by the cave alone, {rays} rays: the test proves little");
    }
}
