//! Which chunks of ground the reflection pass may draw with its OUTDOOR TWIN
//! (`terrain_pipeline::outdoor_pass_shader`, `Levers::outdoor_pass`) or its
//! NOWHERE TWIN (`terrain_pipeline::nowhere_pass_shader`,
//! `Levers::nowhere_pass`; see [`NowhereFrame`]).
//!
//! The ground's reflection pass is the brushes' whole machinery -- choose a
//! photograph, walk the rooms and their doorways, meet what stands in them,
//! record the rims and outlines for the fix-up -- compiled into a shader that,
//! on open ground, uses none of it: a surface in the outdoor volume starts its
//! trace there and leaves at once, and its reflection is the ground map's
//! trace, the buildings' outsides and the sky. That machinery is why the pass
//! is 8,200-9,200 instructions at 26 registers (37% of its waves in flight),
//! and it was the heaviest pipeline of every outdoor view (5.5 ms of the
//! field's 21.6, per-draw capture 2026-10-12;
//! docs/shader-optimization-2026-10-12.md).
//!
//! The twin has the outdoor answer alone. It is the same picture exactly where
//! the full pass would take that path for EVERY texel of a draw, and that is
//! decidable for a whole chunk from its bounds and the frame's probe table:
//!
//! - the trace is on (`ProbeUpload::no_trace` off) and there is an outdoor
//!   volume with a resident photograph;
//! - the chunk lies inside that volume's box, so every point is in it and the
//!   trace's origin -- the point clamped into the box -- is the point;
//! - it touches no other resident photograph's box (with the shader's margin,
//!   `lights::probe_box_margin`), so `probe_choose` can only choose the
//!   outdoor volume's room;
//! - it comes no nearer a doorway than the doorway's fades reach, so no point
//!   is "in a doorway" and none blends a photograph through one: a surface too
//!   rough to trace takes the outdoor volume's photographs and nothing else
//!   (`probe_outdoor_untraced`).
//!
//! The pass is single-sampled, so a texel's position is inside its triangle,
//! and inside the chunk's bounds to rounding; [`SLACK`] covers the rounding
//! and the two frames' transforms.
//!
//! Its own module because the XR renderer compiles for the headset alone.

use crate::renderer::uniforms::{ProbeUpload, MAX_PORTALS, MAX_PROBES};
use glam::Vec3;

/// How far inside the outdoor box, and clear of every room and doorway, a
/// chunk must be: a centimetre, against rounding in the shader's own
/// player-to-world transform (single precision, tens of metres from the
/// origin: micrometres).
pub const SLACK: f32 = 0.01;

/// How far from a doorway's opening the shader still acts on it: its carve
/// margin (`probe_portal_holding`, 0.02 m) for a surface "in a doorway", and
/// the fades of `probe_through_portals` -- along the doorway's axis and across
/// it -- inside which a surface too rough to trace blends in the far room's
/// photograph. The larger, every way: a box round the region the shader tests.
const PORTAL_MARGIN: f32 = {
    let fade = if crate::renderer::lights::PROBE_PORTAL_FADE > crate::renderer::lights::PROBE_PORTAL_SIDE_FADE {
        crate::renderer::lights::PROBE_PORTAL_FADE
    } else {
        crate::renderer::lights::PROBE_PORTAL_SIDE_FADE
    };
    if fade > 0.02 { fade } else { 0.02 }
};

/// What a frame's probe table lets be decided of a box of ground. World frame,
/// as the table is.
pub struct OutdoorFrame {
    /// The outdoor volume's box, shrunk by [`SLACK`].
    outdoors: (Vec3, Vec3),
    /// Every other resident photograph's box, grown by the shader's margin
    /// and [`SLACK`], and every doorway's opening, grown by its fades' reach
    /// and [`SLACK`].
    indoors: Vec<(Vec3, Vec3)>,
}

impl OutdoorFrame {
    /// `None` when no ground can be drawn by the outdoor twin this frame: the
    /// trace is off, the level has no outdoor volume, or none of its
    /// photographs is resident.
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
                // Every cell of a room carries the room's box; were two ever
                // to differ, what is inside both is inside each.
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

    /// Whether every point of the world-frame box `lo..hi` starts its
    /// reflection out of doors: inside the outdoor volume and clear of every
    /// room and doorway.
    pub fn holds(&self, lo: Vec3, hi: Vec3) -> bool {
        lo.cmpge(self.outdoors.0).all()
            && hi.cmple(self.outdoors.1).all()
            && !self.indoors.iter().any(|(l, h)| lo.cmple(*h).all() && hi.cmpge(*l).all())
    }
}

/// What a frame's probe table reaches: every resident photograph's box (the
/// outdoor volume's among them), grown by the shader's margin and [`SLACK`],
/// and every doorway's opening, grown by its fades' reach and [`SLACK`].
/// Ground touching none of them is NOWHERE to the reflection lookup -- in no
/// room to start a trace from, with no photograph to choose and no doorway to
/// blend one through -- and the lookup's answer for it is nothing
/// (`PROBE_NOWHERE` in the lights block, `terrain_pipeline::nowhere_pass_shader`,
/// `Levers::nowhere_pass`). World frame, as the table is.
///
/// A photograph that is not resident is not in the table, and the shader does
/// not find it either: the two agree by reading the same table.
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

    /// Whether no point of the world-frame box `lo..hi` is in any
    /// photograph's box or within any doorway's reach.
    pub fn holds(&self, lo: Vec3, hi: Vec3) -> bool {
        !self.reach.iter().any(|(l, h)| lo.cmple(*h).all() && hi.cmpge(*l).all())
    }
}

/// The world-frame box holding a player-frame box: the shader's
/// `to_world_space` -- a turn by `yaw` about the vertical, then `offset` --
/// of its eight corners.
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

    /// A level: the outdoor volume (7) round everything, a hall (3) and a
    /// doorway in its east wall.
    fn level() -> ProbeUpload {
        let mut p = ProbeUpload::default();
        p.count = 3;
        // The hall, two cells sharing its box.
        for (slot, centre) in [(0usize, [-2.0, 1.5, 0.0]), (1, [2.0, 1.5, 0.0])] {
            p.boxes[slot] = [[centre[0], centre[1], centre[2], slot as f32], [-5.0, 0.0, -4.0, 0.0], [5.0, 3.0, 4.0, 3.0]];
        }
        p.boxes[2] = [[0.0, 20.0, 0.0, 2.0], [-100.0, -10.0, -100.0, 0.0], [100.0, 60.0, 100.0, 7.0]];
        p.portal_count = 1;
        p.portals[0] = [[5.0, 0.0, -1.0, 0.0], [5.4, 2.2, 1.0, 3.0], [7.0, 0.0, 0.0, 0.0]];
        p.outdoor_volume = 7.0;
        p
    }

    /// Ground beyond the outdoor volume is nowhere; ground in it, in a room,
    /// or within a doorway's reach is not -- and every point of a box that
    /// holds is one the shader's own tests find nothing for: no box with its
    /// margin, no doorway's carve, no doorway's fades.
    #[test]
    fn ground_past_every_box_is_nowhere_and_the_shader_finds_nothing_there() {
        let mut p = level();
        // A doorway standing clear of the volume, as a level may have one.
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
        // No table: all of the ground is nowhere.
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
                let along = Vec3::new((axis == 0) as u8 as f32, (axis == 1) as u8 as f32, (axis == 2) as u8 as f32);
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
    fn open_ground_is_outdoors_and_ground_by_the_hall_is_not() {
        let f = OutdoorFrame::new(&level()).unwrap();
        // A chunk of the field.
        assert!(f.holds(Vec3::new(20.0, -1.0, 20.0), Vec3::new(36.0, 2.0, 36.0)));
        // One the hall stands on, one that only reaches under its wall, and
        // one that stops a hair short of it: inside the shader's margin.
        assert!(!f.holds(Vec3::new(-8.0, -1.0, -8.0), Vec3::new(8.0, 0.5, 8.0)));
        assert!(!f.holds(Vec3::new(4.9, -1.0, 10.0 - 6.5), Vec3::new(21.0, 0.5, 10.0)));
        let margin = crate::renderer::lights::probe_box_margin();
        assert!(!f.holds(Vec3::new(5.0 + margin * 0.5, -1.0, 2.0), Vec3::new(9.0, 0.5, 3.9)));
        assert!(f.holds(Vec3::new(5.0 + margin + 0.05, -1.0, 2.0), Vec3::new(9.0, 0.5, 3.9)));
        // Under the hall's floor but clear of its box by more than the margin.
        assert!(f.holds(Vec3::new(-3.0, -6.0, -3.0), Vec3::new(3.0, -1.0, 3.0)));
    }

    #[test]
    fn ground_in_a_doorway_is_not_outdoors() {
        let f = OutdoorFrame::new(&level()).unwrap();
        // The threshold: outside the hall's box, inside the doorway's carve.
        assert!(!f.holds(Vec3::new(5.2, -0.2, -0.5), Vec3::new(6.0, 0.05, 0.5)));
        // Beside the doorway: inside its side fade, where rough ground still
        // blends the hall's photograph in; and past it.
        assert!(!f.holds(Vec3::new(5.2, -0.2, 1.2), Vec3::new(6.0, 0.05, 3.0)));
        assert!(f.holds(Vec3::new(5.2, -0.2, 1.6), Vec3::new(6.0, 0.05, 3.0)));
        // In front of it, inside the fade along its axis; and past that.
        assert!(!f.holds(Vec3::new(5.7, -0.2, -0.5), Vec3::new(6.5, 0.05, 0.5)));
        assert!(f.holds(Vec3::new(6.0, -0.2, -0.5), Vec3::new(6.8, 0.05, 0.5)));
    }

    #[test]
    fn ground_past_the_outdoor_volume_is_not() {
        let f = OutdoorFrame::new(&level()).unwrap();
        assert!(!f.holds(Vec3::new(90.0, 0.0, 90.0), Vec3::new(110.0, 2.0, 99.0)));
        assert!(!f.holds(Vec3::new(20.0, -12.0, 20.0), Vec3::new(30.0, 0.0, 30.0)));
        // On its face: not inside by the slack.
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
        // The outdoor photograph not resident: only the hall's two.
        let mut p = level();
        p.count = 2;
        assert!(OutdoorFrame::new(&p).is_none());
        assert!(OutdoorFrame::new(&ProbeUpload::default()).is_none());
    }

    /// The box of a turned chunk holds every corner the shader's own
    /// transform gives, and with no turn is the chunk moved by the offset.
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

    /// THE SHADER'S OWN CHOICE, for every point of a box the frame holds: the
    /// tightest box containing the point (`probe_choose`, with its margin) is
    /// the outdoor volume's, no doorway's carve holds it, and no doorway's
    /// fades reach it
    /// (`probe_portal_holding`) -- the two things the twin's answer rests on.
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
        // `probe_through_portals`' own test of whether a doorway acts on a point.
        let through_a_doorway = |w: Vec3| {
            use crate::renderer::lights::{PROBE_PORTAL_FADE, PROBE_PORTAL_SIDE_FADE};
            (0..p.portal_count as usize).any(|i| {
                let lo = Vec3::new(p.portals[i][0][0], p.portals[i][0][1], p.portals[i][0][2]);
                let hi = Vec3::new(p.portals[i][1][0], p.portals[i][1][1], p.portals[i][1][2]);
                let axis = p.portals[i][0][3] as usize;
                let along = Vec3::new((axis == 0) as u8 as f32, (axis == 1) as u8 as f32, (axis == 2) as u8 as f32);
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
