use crate::renderer::uniforms::{ProbePortal, ProbeUpload, MAX_PORTALS, MAX_PROBES};
use glam::Vec3;

pub const SLACK: f32 = 0.01;

const PORTAL_MARGIN: f32 = {
    let fade = if crate::renderer::lights::PROBE_PORTAL_FADE > crate::renderer::lights::PROBE_PORTAL_SIDE_FADE {
        crate::renderer::lights::PROBE_PORTAL_FADE
    } else {
        crate::renderer::lights::PROBE_PORTAL_SIDE_FADE
    };
    if fade > 0.02 { fade } else { 0.02 }
};

struct BoxIndex {
    boxes: Vec<(Vec3, Vec3)>,

    wide: Vec<u32>,
    origin: [f32; 2],
    cell: f32,
    cells_x: usize,
    cells_z: usize,
    cells: Vec<Vec<u32>>,
    #[cfg(test)]
    tests: std::cell::Cell<u64>,
}

impl BoxIndex {

    const WIDE_CELLS: usize = 64;

    fn new(boxes: &[(Vec3, Vec3)]) -> Self {
        let finite = |b: &(Vec3, Vec3)| b.0.is_finite() && b.1.is_finite();
        let (mut lo, mut hi) = ([f32::INFINITY; 2], [f32::NEG_INFINITY; 2]);
        for (l, h) in boxes.iter().filter(|b| finite(b)) {
            lo = [lo[0].min(l.x), lo[1].min(l.z)];
            hi = [hi[0].max(h.x), hi[1].max(h.z)];
        }
        let mut index = Self {
            boxes: boxes.to_vec(),
            wide: Vec::new(),
            origin: lo,
            cell: 1.0,
            cells_x: 0,
            cells_z: 0,
            cells: Vec::new(),
            #[cfg(test)]
            tests: Default::default(),
        };
        if !(lo[0] <= hi[0] && lo[1] <= hi[1]) {
            index.wide = (0..boxes.len() as u32).collect();
            return index;
        }

        let side = (hi[0] - lo[0]).max(hi[1] - lo[1]).max(1.0);
        let per_axis = ((boxes.len() as f32).sqrt().ceil() as usize * 2).clamp(1, 256);
        index.cell = (side / per_axis as f32).max(2.0);
        index.cells_x = (((hi[0] - lo[0]) / index.cell).floor() as usize + 1).max(1);
        index.cells_z = (((hi[1] - lo[1]) / index.cell).floor() as usize + 1).max(1);
        index.cells = vec![Vec::new(); index.cells_x * index.cells_z];
        for (i, b) in boxes.iter().enumerate() {
            if !finite(b) {
                index.wide.push(i as u32);
                continue;
            }
            let (x, z) = (index.span(b.0.x, b.1.x, 0), index.span(b.0.z, b.1.z, 1));
            if (x.1 - x.0 + 1) * (z.1 - z.0 + 1) > Self::WIDE_CELLS {
                index.wide.push(i as u32);
                continue;
            }
            for cz in z.0..=z.1 {
                for cx in x.0..=x.1 {
                    index.cells[cz * index.cells_x + cx].push(i as u32);
                }
            }
        }
        index
    }

    fn span(&self, lo: f32, hi: f32, k: usize) -> (usize, usize) {
        let n = if k == 0 { self.cells_x } else { self.cells_z };
        let at = |v: f32| (((v - self.origin[k]) / self.cell).floor().max(0.0) as usize).min(n - 1);
        (at(lo), at(hi))
    }

    fn touches(&self, lo: Vec3, hi: Vec3) -> bool {
        let touch = |i: &u32| {
            #[cfg(test)]
            self.tests.set(self.tests.get() + 1);
            let (l, h) = &self.boxes[*i as usize];
            lo.cmple(*h).all() && hi.cmpge(*l).all()
        };
        if self.wide.iter().any(touch) {
            return true;
        }
        if self.cells.is_empty() || !(lo.is_finite() && hi.is_finite()) {

            return false;
        }

        let far_x = self.origin[0] + self.cell * self.cells_x as f32;
        let far_z = self.origin[1] + self.cell * self.cells_z as f32;
        if hi.x < self.origin[0] || hi.z < self.origin[1] || lo.x > far_x || lo.z > far_z {
            return false;
        }
        let (x, z) = (self.span(lo.x, hi.x, 0), self.span(lo.z, hi.z, 1));
        (z.0..=z.1).any(|cz| (x.0..=x.1).any(|cx| self.cells[cz * self.cells_x + cx].iter().any(touch)))
    }
}

pub struct OutdoorFrame {

    outdoors: (Vec3, Vec3),

    indoors: Vec<(Vec3, Vec3)>,
}

impl OutdoorFrame {

    pub fn new(probes: &ProbeUpload) -> Option<Self> {
        if probes.no_trace || probes.outdoor_volume < 0.0 {
            return None;
        }
        let v3 = |a: [f32; 4]| Vec3::new(a[0], a[1], a[2]);
        let margin = Vec3::splat(crate::renderer::lights::probe_box_margin() + SLACK);
        let mut outdoors: Option<(Vec3, Vec3)> = None;
        let mut indoors = Vec::new();
        for slot in 0..(probes.count as usize).min(MAX_PROBES) {
            let b = &probes.boxes[slot];
            let (lo, hi) = (v3(b[1]), v3(b[2]));
            if b[2][3] == probes.outdoor_volume {

                outdoors = Some(match outdoors {
                    Some((l, h)) => (l.max(lo), h.min(hi)),
                    None => (lo, hi),
                });
            } else {
                indoors.push((lo - margin, hi + margin));
            }
        }
        let (lo, hi) = outdoors?;
        let carve = Vec3::splat(PORTAL_MARGIN + SLACK);
        for p in 0..(probes.portal_count as usize).min(MAX_PORTALS) {
            indoors.push((v3(probes.portals[p][0]) - carve, v3(probes.portals[p][1]) + carve));
        }
        Some(Self { outdoors: (lo + Vec3::splat(SLACK), hi - Vec3::splat(SLACK)), indoors })
    }

    pub fn holds(&self, lo: Vec3, hi: Vec3) -> bool {
        lo.cmpge(self.outdoors.0).all()
            && hi.cmple(self.outdoors.1).all()
            && !self.indoors.iter().any(|(l, h)| lo.cmple(*h).all() && hi.cmpge(*l).all())
    }
}

pub struct NowhereFrame {
    reach: Vec<(Vec3, Vec3)>,
}

impl NowhereFrame {
    pub fn new(probes: &ProbeUpload) -> Self {
        let v3 = |a: [f32; 4]| Vec3::new(a[0], a[1], a[2]);
        let margin = Vec3::splat(crate::renderer::lights::probe_box_margin() + SLACK);
        let fade = Vec3::splat(PORTAL_MARGIN + SLACK);
        let boxes = (0..(probes.count as usize).min(MAX_PROBES)).map(|slot| {
            let b = &probes.boxes[slot];
            (v3(b[1]) - margin, v3(b[2]) + margin)
        });
        let doorways = (0..(probes.portal_count as usize).min(MAX_PORTALS))
            .map(|p| (v3(probes.portals[p][0]) - fade, v3(probes.portals[p][1]) + fade));
        Self { reach: boxes.chain(doorways).collect() }
    }

    pub fn holds(&self, lo: Vec3, hi: Vec3) -> bool {
        !self.reach.iter().any(|(l, h)| lo.cmple(*h).all() && hi.cmpge(*l).all())
    }
}

pub struct LevelGround {
    outdoor: Option<OutdoorFrame>,
    nowhere: NowhereFrame,

    reach_bounds: (Vec3, Vec3),

    pub key: u64,

    index: std::cell::OnceCell<(BoxIndex, Option<BoxIndex>)>,
}

pub const GROUND_FULL: usize = 0;
pub const GROUND_OUTDOORS: usize = 1;
pub const GROUND_NOWHERE: usize = 2;

impl LevelGround {

    pub fn new(volumes: &[(u32, Vec3, Vec3, Vec3)], rooms: &[u32], outdoor_room: Option<u32>, portals: &[ProbePortal]) -> Self {
        let margin = Vec3::splat(crate::renderer::lights::probe_box_margin() + SLACK);
        let fade = Vec3::splat(PORTAL_MARGIN + SLACK);
        let mut outdoors: Option<(Vec3, Vec3)> = None;
        let mut indoors = Vec::new();
        let mut reach = Vec::new();
        let mut key: u64 = 1469598103934665603;
        let mut mix = |v: Vec3| {
            for x in v.to_array() {
                key = (key ^ u64::from(x.to_bits())).wrapping_mul(1099511628211);
            }
        };
        for &(probe, _, lo, hi) in volumes {
            mix(lo);
            mix(hi);
            reach.push((lo - margin, hi + margin));
            let room = rooms.get(probe as usize).copied();
            mix(Vec3::splat(room.map_or(-1.0, |r| r as f32)));
            if room.is_some() && room == outdoor_room {
                outdoors = Some(match outdoors {
                    Some((l, h)) => (l.max(lo), h.min(hi)),
                    None => (lo, hi),
                });
            } else {
                indoors.push((lo - margin, hi + margin));
            }
        }
        for p in portals {
            mix(p.min);
            mix(p.max);
            indoors.push((p.min - fade, p.max + fade));
            reach.push((p.min - fade, p.max + fade));
        }
        let outdoor = outdoors.map(|(lo, hi)| OutdoorFrame { outdoors: (lo + Vec3::splat(SLACK), hi - Vec3::splat(SLACK)), indoors });
        let reach_bounds = Self::bounds_of(&reach);
        Self { outdoor, nowhere: NowhereFrame { reach }, reach_bounds, key: key | 1, index: Default::default() }
    }

    fn bounds_of(reach: &[(Vec3, Vec3)]) -> (Vec3, Vec3) {
        reach.iter().fold((Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY)), |(lo, hi), (l, h)| (lo.min(*l), hi.max(*h)))
    }

    pub fn of_table(probes: &ProbeUpload) -> Self {
        let nowhere = NowhereFrame::new(probes);
        let reach_bounds = Self::bounds_of(&nowhere.reach);
        Self { outdoor: OutdoorFrame::new(probes), nowhere, reach_bounds, key: 1, index: Default::default() }
    }

    pub fn class_of(&self, lo: Vec3, hi: Vec3) -> usize {
        let outside = !(lo.cmple(self.reach_bounds.1).all() && hi.cmpge(self.reach_bounds.0).all());
        if outside {
            return GROUND_NOWHERE;
        }

        let (reach, indoors) = self.index.get_or_init(|| (BoxIndex::new(&self.nowhere.reach), self.outdoor.as_ref().map(|f| BoxIndex::new(&f.indoors))));
        if !reach.touches(lo, hi) {
            GROUND_NOWHERE
        } else if self.outdoor.as_ref().zip(indoors.as_ref()).is_some_and(|(f, indoors)| {
            lo.cmpge(f.outdoors.0).all() && hi.cmple(f.outdoors.1).all() && !indoors.touches(lo, hi)
        }) {
            GROUND_OUTDOORS
        } else {
            GROUND_FULL
        }
    }

    #[cfg(test)]
    fn class_of_asking_every_box(&self, lo: Vec3, hi: Vec3) -> usize {
        let outside = !(lo.cmple(self.reach_bounds.1).all() && hi.cmpge(self.reach_bounds.0).all());
        if outside || self.nowhere.holds(lo, hi) {
            GROUND_NOWHERE
        } else if self.outdoor.as_ref().is_some_and(|f| f.holds(lo, hi)) {
            GROUND_OUTDOORS
        } else {
            GROUND_FULL
        }
    }

    pub fn class_of_posed(&self, offset: Vec3, yaw: f32, corners: [[f32; 3]; 3]) -> usize {
        let (s, c) = yaw.sin_cos();
        let mut lo = Vec3::splat(f32::INFINITY);
        let mut hi = Vec3::splat(f32::NEG_INFINITY);
        for p in corners {
            let w = Vec3::new(c * p[0] + s * p[2], p[1], -s * p[0] + c * p[2]) + offset;
            lo = lo.min(w);
            hi = hi.max(w);
        }
        self.class_of(lo, hi)
    }
}

pub fn world_box(offset: Vec3, yaw: f32, lo: Vec3, hi: Vec3) -> (Vec3, Vec3) {
    let (s, c) = yaw.sin_cos();
    let mut out = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
    for k in 0..8 {
        let p = Vec3::new(if k & 1 == 0 { lo.x } else { hi.x }, if k & 2 == 0 { lo.y } else { hi.y }, if k & 4 == 0 { lo.z } else { hi.z });
        let w = Vec3::new(c * p.x + s * p.z, p.y, -s * p.x + c * p.z) + offset;
        out = (out.0.min(w), out.1.max(w));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_boxes_by_where_they_stand_answer_as_asking_them_all() {
        let mut volumes: Vec<(u32, Vec3, Vec3, Vec3)> = Vec::new();
        let mut rooms: Vec<u32> = Vec::new();
        let mut portals: Vec<ProbePortal> = Vec::new();
        let outdoor = 16 * 16;
        for b in 0..256u32 {
            let (i, j) = ((b % 16) as f32, (b / 16) as f32);
            let centre = Vec3::new(i * 40.0 - 300.0 + (b as f32 * 0.7).sin() * 14.0, 1.5 + (b % 4) as f32, j * 40.0 - 300.0 + (b as f32 * 1.3).cos() * 14.0);
            let half = Vec3::new(6.0 + (b % 5) as f32 * 3.0, 2.5, 5.0 + (b % 3) as f32 * 4.0);
            for cell in 0..(2 + b % 4) {
                volumes.push((volumes.len() as u32, centre, centre - half, centre + half));
                let _ = cell;
                rooms.push(b);
            }
            portals.push(ProbePortal { min: centre + Vec3::new(half.x - 0.3, -2.5, -0.8), max: centre + Vec3::new(half.x + 0.3, -0.3, 0.8), axis: 0, low: b, high: outdoor, wall: None });
        }
        let (olo, ohi) = (Vec3::new(-330.0, -12.0, -330.0), Vec3::new(330.0, 40.0, 330.0));
        volumes.push((volumes.len() as u32, (olo + ohi) * 0.5, olo, ohi));
        rooms.push(outdoor);
        let level = LevelGround::new(&volumes, &rooms, Some(outdoor), &portals);
        let (mut seen, mut asked) = ([0u32; 3], 0u64);
        for k in 0..60_000u32 {

            let p = Vec3::new((k % 251) as f32 * 2.9 - 364.0, ((k / 7) % 13) as f32 * 0.9 - 3.0, (k / 251) as f32 * 3.05 - 364.0);
            let (lo, hi) = (p, p + Vec3::new(0.43, 0.2 + (k % 3) as f32, 0.43));
            let class = level.class_of(lo, hi);
            assert_eq!(class, level.class_of_asking_every_box(lo, hi), "the box {lo}..{hi}");
            seen[class] += 1;
            asked += (volumes.len() + portals.len()) as u64;
        }
        assert!(seen.iter().all(|&n| n > 1500), "every class on this level: {seen:?}");

        for (lo, hi) in [(Vec3::new(-100.0, -5.0, -60.0), Vec3::new(10.0, 9.0, 50.0)), (olo - Vec3::splat(50.0), ohi + Vec3::splat(50.0)), (Vec3::new(400.0, 0.0, 400.0), Vec3::new(420.0, 1.0, 420.0))] {
            assert_eq!(level.class_of(lo, hi), level.class_of_asking_every_box(lo, hi));
        }
        let (reach, indoors) = level.index.get().expect("made by the first triangle");
        let tests = reach.tests.get() + indoors.as_ref().map_or(0, |i| i.tests.get());
        assert!(tests * 20 < asked, "{tests} box tests through the grid; asking every box, at least {asked}");
    }

    fn level() -> ProbeUpload {
        let mut p = ProbeUpload::default();
        p.count = 3;

        for (slot, centre) in [(0usize, [-2.0, 1.5, 0.0]), (1, [2.0, 1.5, 0.0])] {
            p.boxes[slot] = [[centre[0], centre[1], centre[2], slot as f32], [-5.0, 0.0, -4.0, 0.0], [5.0, 3.0, 4.0, 3.0]];
        }
        p.boxes[2] = [[0.0, 20.0, 0.0, 2.0], [-100.0, -10.0, -100.0, 0.0], [100.0, 60.0, 100.0, 7.0]];
        p.portal_count = 1;
        p.portals[0] = [[5.0, 0.0, -1.0, 0.0], [5.4, 2.2, 1.0, 3.0], [7.0, 0.0, 0.0, 0.0]];
        p.outdoor_volume = 7.0;
        p
    }

    #[test]
    fn ground_past_every_box_is_nowhere_and_the_shader_finds_nothing_there() {
        let mut p = level();

        p.portal_count = 2;
        p.portals[1] = [[120.0, 0.0, -1.0, 0.0], [120.4, 2.2, 1.0, 3.0], [7.0, 0.0, 0.0, 0.0]];
        let f = NowhereFrame::new(&p);
        assert!(f.holds(Vec3::new(101.0, -1.0, 20.0), Vec3::new(110.0, 2.0, 36.0)));
        assert!(!f.holds(Vec3::new(20.0, -1.0, 20.0), Vec3::new(36.0, 2.0, 36.0)), "in the outdoor volume");
        assert!(!f.holds(Vec3::new(99.0, -1.0, 20.0), Vec3::new(110.0, 2.0, 36.0)), "straddling its edge");
        let margin = crate::renderer::lights::probe_box_margin();
        assert!(!f.holds(Vec3::new(100.0 + margin * 0.5, -1.0, 0.0), Vec3::new(104.0, 2.0, 4.0)), "inside the shader's margin");
        assert!(!f.holds(Vec3::new(119.0, 0.0, 1.2), Vec3::new(119.8, 1.0, 1.4)), "inside the far doorway's fades");
        assert!(f.holds(Vec3::new(119.0, 0.0, 1.6), Vec3::new(119.4, 1.0, 3.0)));

        assert!(NowhereFrame::new(&ProbeUpload::default()).holds(Vec3::splat(-5.0), Vec3::splat(5.0)));

        let in_a_box = |w: Vec3| {
            (0..p.count as usize).any(|slot| {
                let b = &p.boxes[slot];
                let (lo, hi) = (Vec3::new(b[1][0], b[1][1], b[1][2]), Vec3::new(b[2][0], b[2][1], b[2][2]));
                !(w.cmplt(lo - margin).any() || w.cmpgt(hi + margin).any())
            })
        };
        let by_a_doorway = |w: Vec3| {
            use crate::renderer::lights::{PROBE_PORTAL_FADE, PROBE_PORTAL_SIDE_FADE};
            (0..p.portal_count as usize).any(|i| {
                let lo = Vec3::new(p.portals[i][0][0], p.portals[i][0][1], p.portals[i][0][2]);
                let hi = Vec3::new(p.portals[i][1][0], p.portals[i][1][1], p.portals[i][1][2]);
                let axis = p.portals[i][0][3] as usize;
                let along = Vec3::new(f32::from(u8::from(axis == 0)), f32::from(u8::from(axis == 1)), f32::from(u8::from(axis == 2)));
                let side = ((lo - w).max(w - hi).max(Vec3::ZERO) * (Vec3::ONE - along)).length();
                let t = w.dot(along);
                let fades = !(side >= PROBE_PORTAL_SIDE_FADE || t <= lo.dot(along) - PROBE_PORTAL_FADE || t >= hi.dot(along) + PROBE_PORTAL_FADE);
                let carve = w.cmpge(lo - 0.02).all() && w.cmple(hi + 0.02).all();
                fades || carve
            })
        };
        let mut held = 0;
        let mut seed = 29u64;
        let mut rnd = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f32) / (1u64 << 31) as f32
        };
        for _ in 0..6000 {
            let lo = Vec3::new(rnd() * 60.0 + 80.0, rnd() * 8.0 - 12.0, rnd() * 60.0 - 30.0);
            let hi = lo + Vec3::new(rnd() * 6.0 + 0.1, rnd() * 2.0 + 0.1, rnd() * 6.0 + 0.1);
            if !f.holds(lo, hi) {
                continue;
            }
            held += 1;
            for _ in 0..20 {
                let w = lo + (hi - lo) * Vec3::new(rnd(), rnd(), rnd());
                assert!(!in_a_box(w) && !by_a_doorway(w), "{w} of {lo}..{hi}");
            }
        }
        assert!(held > 200, "only {held} boxes held: the sample no longer tests the rule");
    }

    #[test]
    fn what_the_level_decides_holds_in_every_frame() {

        let v = |x: f32, y: f32, z: f32| Vec3::new(x, y, z);
        let volumes = vec![
            (0u32, v(-2.0, 1.5, 0.0), v(-5.0, 0.0, -4.0), v(5.0, 3.0, 4.0)),
            (1, v(2.0, 1.5, 0.0), v(-5.0, 0.0, -4.0), v(5.0, 3.0, 4.0)),
            (2, v(0.0, 20.0, 0.0), v(-40.0, -10.0, -40.0), v(40.0, 60.0, 40.0)),
            (3, v(10.0, 20.0, 0.0), v(-40.0, -10.0, -40.0), v(40.0, 60.0, 40.0)),
            (4, v(20.0, 1.0, 20.0), v(18.0, 0.0, 18.0), v(22.0, 2.5, 22.0)),
        ];
        let rooms = vec![3u32, 3, 7, 7, 4];
        let portals = vec![
            ProbePortal { min: v(5.0, 0.0, -1.0), max: v(5.4, 2.2, 1.0), axis: 0, low: 3, high: 7, wall: None },
            ProbePortal { min: v(17.6, 0.0, 19.5), max: v(18.0, 2.0, 20.5), axis: 0, low: 7, high: 4, wall: None },
        ];
        let level = LevelGround::new(&volumes, &rooms, Some(7), &portals);
        assert_ne!(level.key, 0);
        assert_ne!(level.key, LevelGround::new(&volumes[..4], &rooms, Some(7), &portals).key);
        assert_eq!(level.class_of(v(50.0, -1.0, 0.0), v(51.0, 0.0, 1.0)), GROUND_NOWHERE);
        assert_eq!(level.class_of(v(30.0, -1.0, 0.0), v(31.0, 0.0, 1.0)), GROUND_OUTDOORS);
        assert_eq!(level.class_of(v(39.5, -1.0, 0.0), v(40.5, 0.0, 1.0)), GROUND_FULL, "across the volume's edge");
        assert_eq!(level.class_of(v(4.0, -0.5, 0.0), v(6.0, 0.1, 1.0)), GROUND_FULL, "in the hall and its doorway");
        assert_eq!(level.class_of(v(17.0, 0.0, 19.8), v(17.5, 0.2, 20.2)), GROUND_FULL, "by the hut's doorway");

        let no_volume = LevelGround::new(&volumes, &rooms, None, &portals);
        assert_eq!(no_volume.class_of(v(30.0, -1.0, 0.0), v(31.0, 0.0, 1.0)), GROUND_FULL);
        assert_eq!(no_volume.class_of(v(50.0, -1.0, 0.0), v(51.0, 0.0, 1.0)), GROUND_NOWHERE);

        let (offset, yaw) = (v(3.0, 0.4, -2.0), 0.7f32);
        let posed = |w: Vec3| {
            let d = w - offset;
            let (s, c) = yaw.sin_cos();
            [c * d.x - s * d.z, d.y, s * d.x + c * d.z]
        };
        for (lo, hi) in [(v(50.0, -1.0, 0.0), v(51.0, 0.0, 1.0)), (v(30.0, -1.0, 0.0), v(31.0, 0.0, 1.0)), (v(4.0, -0.5, 0.0), v(6.0, 0.1, 1.0))] {
            let tri = [posed(lo), posed(v(hi.x, lo.y, lo.z)), posed(hi)];
            assert_eq!(level.class_of_posed(offset, yaw, tri), level.class_of(lo, hi));
        }

        let mut seed = 5u64;
        let mut rnd = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f32) / (1u64 << 31) as f32
        };
        let (mut nowhere, mut outdoors, mut with_twin) = (0, 0, 0);
        for frame in 0..64u32 {

            let mut p = ProbeUpload::default();
            let mut slot = 0usize;
            for (i, &(probe, centre, lo, hi)) in volumes.iter().enumerate() {
                if frame >> i & 1 == 1 {
                    p.set(slot, probe, centre, lo, hi);
                    p.set_volume(slot, rooms[probe as usize]);
                    slot += 1;
                }
            }
            p.count = slot as u32;
            let kept: Vec<ProbePortal> = portals.iter().enumerate().filter(|(i, _)| frame >> (5 + i) & 1 == 1).map(|(_, d)| *d).collect();
            p.portal_count = kept.len() as u32;
            for (i, d) in kept.iter().enumerate() {
                p.portals[i] = [[d.min.x, d.min.y, d.min.z, d.axis as f32], [d.max.x, d.max.y, d.max.z, d.low as f32], [d.high as f32, 0.0, 0.0, 0.0]];
            }
            p.outdoor_volume = 7.0;
            let frame_nowhere = NowhereFrame::new(&p);
            let frame_outdoors = OutdoorFrame::new(&p);
            with_twin += u32::from(frame_outdoors.is_some());
            for _ in 0..400 {
                let lo = v(rnd() * 100.0 - 50.0, rnd() * 4.0 - 2.0, rnd() * 100.0 - 50.0);
                let hi = lo + v(rnd() * 1.5 + 0.05, rnd() * 0.5 + 0.01, rnd() * 1.5 + 0.05);
                match level.class_of(lo, hi) {
                    GROUND_NOWHERE => {
                        nowhere += 1;
                        assert!(frame_nowhere.holds(lo, hi), "frame {frame}: {lo}..{hi} is nowhere in the level and somewhere in a frame");
                    }
                    GROUND_OUTDOORS => {
                        outdoors += 1;
                        if let Some(f) = &frame_outdoors {
                            assert!(f.holds(lo, hi), "frame {frame}: {lo}..{hi} is out of doors in the level and not in a frame");
                        }
                    }
                    _ => {}
                }
            }
        }
        assert!(nowhere > 1000 && outdoors > 1000 && with_twin >= 16, "{nowhere} nowhere, {outdoors} out of doors, {with_twin} frames with a twin");
    }

    #[test]
    fn the_trace_word_is_off_open_or_neither() {
        let mut p = level();
        assert_eq!(p.trace_word(), 0.0);
        p.open_ground = true;
        assert_eq!(p.trace_word(), -1.0);
        p.no_trace = true;
        assert_eq!(p.trace_word(), 1.0);
        p.open_ground = false;
        assert_eq!(p.trace_word(), 1.0);
        assert_eq!(ProbeUpload::default().trace_word(), 0.0);
    }

    #[test]
    fn open_ground_is_outdoors_and_ground_by_the_hall_is_not() {
        let f = OutdoorFrame::new(&level()).unwrap();

        assert!(f.holds(Vec3::new(20.0, -1.0, 20.0), Vec3::new(36.0, 2.0, 36.0)));

        assert!(!f.holds(Vec3::new(-8.0, -1.0, -8.0), Vec3::new(8.0, 0.5, 8.0)));
        assert!(!f.holds(Vec3::new(4.9, -1.0, 10.0 - 6.5), Vec3::new(21.0, 0.5, 10.0)));
        let margin = crate::renderer::lights::probe_box_margin();
        assert!(!f.holds(Vec3::new(5.0 + margin * 0.5, -1.0, 2.0), Vec3::new(9.0, 0.5, 3.9)));
        assert!(f.holds(Vec3::new(5.0 + margin + 0.05, -1.0, 2.0), Vec3::new(9.0, 0.5, 3.9)));

        assert!(f.holds(Vec3::new(-3.0, -6.0, -3.0), Vec3::new(3.0, -1.0, 3.0)));
    }

    #[test]
    fn ground_in_a_doorway_is_not_outdoors() {
        let f = OutdoorFrame::new(&level()).unwrap();

        assert!(!f.holds(Vec3::new(5.2, -0.2, -0.5), Vec3::new(6.0, 0.05, 0.5)));

        assert!(!f.holds(Vec3::new(5.2, -0.2, 1.2), Vec3::new(6.0, 0.05, 3.0)));
        assert!(f.holds(Vec3::new(5.2, -0.2, 1.6), Vec3::new(6.0, 0.05, 3.0)));

        assert!(!f.holds(Vec3::new(5.7, -0.2, -0.5), Vec3::new(6.5, 0.05, 0.5)));
        assert!(f.holds(Vec3::new(6.0, -0.2, -0.5), Vec3::new(6.8, 0.05, 0.5)));
    }

    #[test]
    fn ground_past_the_outdoor_volume_is_not() {
        let f = OutdoorFrame::new(&level()).unwrap();
        assert!(!f.holds(Vec3::new(90.0, 0.0, 90.0), Vec3::new(110.0, 2.0, 99.0)));
        assert!(!f.holds(Vec3::new(20.0, -12.0, 20.0), Vec3::new(30.0, 0.0, 30.0)));

        assert!(!f.holds(Vec3::new(90.0, 0.0, 90.0), Vec3::new(100.0, 2.0, 99.0)));
    }

    #[test]
    fn no_frame_without_the_trace_an_outdoor_volume_or_its_photograph() {
        let mut p = level();
        p.no_trace = true;
        assert!(OutdoorFrame::new(&p).is_none());
        let mut p = level();
        p.outdoor_volume = -1.0;
        assert!(OutdoorFrame::new(&p).is_none());

        let mut p = level();
        p.count = 2;
        assert!(OutdoorFrame::new(&p).is_none());
        assert!(OutdoorFrame::new(&ProbeUpload::default()).is_none());
    }

    #[test]
    fn the_world_box_holds_the_turned_chunk() {
        let (lo, hi) = (Vec3::new(1.0, 0.0, 2.0), Vec3::new(5.0, 3.0, 4.0));
        let offset = Vec3::new(10.0, 1.0, -7.0);
        assert_eq!(world_box(offset, 0.0, lo, hi), (lo + offset, hi + offset));
        for yaw in [0.3f32, 1.1, -2.4, 3.0] {
            let (wl, wh) = world_box(offset, yaw, lo, hi);
            let (s, c) = yaw.sin_cos();
            for t in 0..=10 {
                let k = t as f32 / 10.0;
                let p = lo + (hi - lo) * Vec3::new(k, 1.0 - k, (k * 3.0).fract());
                let w = Vec3::new(c * p.x + s * p.z, p.y, -s * p.x + c * p.z) + offset;
                assert!(w.cmpge(wl - 1e-4).all() && w.cmple(wh + 1e-4).all(), "{yaw}: {w} outside {wl}..{wh}");
            }
        }
    }

    #[test]
    fn every_point_of_a_held_box_chooses_the_outdoor_room() {
        let p = level();
        let f = OutdoorFrame::new(&p).unwrap();
        let margin = crate::renderer::lights::probe_box_margin();
        let choose = |w: Vec3| -> f32 {
            let (mut best_volume, mut room) = (1e30f32, -1.0f32);
            for slot in 0..p.count as usize {
                let b = &p.boxes[slot];
                let (lo, hi) = (Vec3::new(b[1][0], b[1][1], b[1][2]), Vec3::new(b[2][0], b[2][1], b[2][2]));
                if w.cmplt(lo - margin).any() || w.cmpgt(hi + margin).any() {
                    continue;
                }
                let d = hi - lo;
                if d.x * d.y * d.z < best_volume * 0.999 {
                    best_volume = d.x * d.y * d.z;
                    room = b[2][3];
                }
            }
            room
        };
        let in_doorway = |w: Vec3| {
            (0..p.portal_count as usize).any(|i| {
                let lo = Vec3::new(p.portals[i][0][0], p.portals[i][0][1], p.portals[i][0][2]) - 0.02;
                let hi = Vec3::new(p.portals[i][1][0], p.portals[i][1][1], p.portals[i][1][2]) + 0.02;
                w.cmpge(lo).all() && w.cmple(hi).all()
            })
        };

        let through_a_doorway = |w: Vec3| {
            use crate::renderer::lights::{PROBE_PORTAL_FADE, PROBE_PORTAL_SIDE_FADE};
            (0..p.portal_count as usize).any(|i| {
                let lo = Vec3::new(p.portals[i][0][0], p.portals[i][0][1], p.portals[i][0][2]);
                let hi = Vec3::new(p.portals[i][1][0], p.portals[i][1][1], p.portals[i][1][2]);
                let axis = p.portals[i][0][3] as usize;
                let along = Vec3::new(f32::from(u8::from(axis == 0)), f32::from(u8::from(axis == 1)), f32::from(u8::from(axis == 2)));
                let side = ((lo - w).max(w - hi).max(Vec3::ZERO) * (Vec3::ONE - along)).length();
                let t = w.dot(along);
                !(side >= PROBE_PORTAL_SIDE_FADE || t <= lo.dot(along) - PROBE_PORTAL_FADE || t >= hi.dot(along) + PROBE_PORTAL_FADE)
            })
        };
        let mut held = 0;
        let mut seed = 11u64;
        let mut rnd = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f32) / (1u64 << 31) as f32
        };
        for _ in 0..4000 {
            let lo = Vec3::new(rnd() * 30.0 - 15.0, rnd() * 4.0 - 2.0, rnd() * 30.0 - 15.0);
            let hi = lo + Vec3::new(rnd() * 6.0 + 0.1, rnd() * 2.0 + 0.1, rnd() * 6.0 + 0.1);
            if !f.holds(lo, hi) {
                continue;
            }
            held += 1;
            for _ in 0..20 {
                let w = lo + (hi - lo) * Vec3::new(rnd(), rnd(), rnd());
                assert_eq!(choose(w), 7.0, "{w} of {lo}..{hi}");
                assert!(!in_doorway(w), "{w} of {lo}..{hi}");
                assert!(!through_a_doorway(w), "{w} of {lo}..{hi}");
            }
        }
        assert!(held > 200, "only {held} boxes held: the sample no longer tests the rule");
    }
}
