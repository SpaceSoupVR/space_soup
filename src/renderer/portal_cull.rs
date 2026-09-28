//! WHAT CAN BE SEEN OUTSIDE A CLOSED ROOM, AND THROUGH WHICH DOORWAYS.
//!
//! From inside a room whose walls, floor and ceiling are solid everywhere but
//! its doorways, anything outside the building -- the terrain, the lawn -- can
//! only be seen through those doorways, and through doorways seen through
//! them. The terrain was nonetheless drawn in full from every room: 32 chunk
//! draws an eye in the marble hall, every fragment rejected by the depth
//! prepass, 0.57 ms an eye of vertex and binning work for nothing (Quest 3,
//! per-draw trace, 2026-09-28).
//!
//! [`outdoor_frusta`] walks the doorways out from the room holding the eye,
//! narrowing the view to each opening's outline on screen, and returns the
//! narrowed frusta at which the walk leaves the closed rooms. Something
//! outside the building is drawn only where it meets one of them -- which,
//! for a closed room, is exactly where it can be seen, so the picture does not
//! change.
//!
//! WHEN IT ANSWERS "NO CULLING" (`None`): the eye in no room, or in a room that
//! is not CLOSED -- a carve that breaks out of its shell, a courtyard -- or in a
//! doorway, between rooms. A room counts as closed only when the level says so
//! ([`CullRoom::closed`]; the app decides it from the brushes: a room carve
//! lying strictly inside the solid it was cut from).
//!
//! CONSERVATIVE throughout: a doorway straddling the eye's plane passes the
//! whole window on, and a walk that reaches `MAX_DOORWAYS` takes whatever it
//! still sees as outside, rather than guess. Drawing too much only costs
//! time. A room MAY be revisited -- through another doorway, which is a real
//! line of sight round a loop -- but never back through the doorway just
//! entered by: that looks back the way the view came.

use glam::{Mat4, Vec3, Vec4};

use crate::renderer::uniforms::ProbePortal;

/// A room, by the number the doorways use for it. See [`outdoor_frusta`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CullRoom {
    pub id: u32,
    pub min: Vec3,
    pub max: Vec3,
    /// Solid all round except at its doorways. Only a closed room can hide
    /// the outside; any other room is walked no further, and whatever its
    /// doorway shows counts as outside.
    pub closed: bool,
}

/// How many doorways deep the walk goes before it stops narrowing and takes
/// whatever is left as visible.
pub const MAX_DOORWAYS: usize = 4;

/// A window on screen, in NDC: `[x0, y0, x1, y1]`.
type Window = [f32; 4];

const FULL: Window = [-1.0, -1.0, 1.0, 1.0];

/// The frusta through which anything outside the closed rooms can be seen from
/// `eye`, or `None` when the eye is not inside a closed room and nothing can
/// be culled this way.
///
/// `eye` and the rooms and doorways are in WORLD space; `clip_from_world`
/// projects them. The planes returned are for `clip_from_frame`, the same
/// camera in whatever frame the caller's geometry is in (the XR renderer's
/// player frame), so they can be handed straight to `shadow::chunk_seen`.
pub fn outdoor_frusta(
    eye: Vec3,
    clip_from_world: Mat4,
    clip_from_frame: Mat4,
    rooms: &[CullRoom],
    portals: &[ProbePortal],
) -> Option<Vec<[Vec4; 6]>> {
    // THE TIGHTEST ROOM HOLDING THE EYE: a room nested in a larger volume --
    // the hall inside the outdoor volume -- is the one it is standing in.
    let start = rooms
        .iter()
        .filter(|r| (r.min.cmplt(eye) & eye.cmplt(r.max)).all())
        .min_by(|a, b| volume(a).total_cmp(&volume(b)))?;
    if !start.closed {
        return None;
    }
    let mut out = Vec::new();
    walk(start.id, None, FULL, 0, clip_from_world, clip_from_frame, rooms, portals, &mut out);
    Some(out)
}

fn volume(r: &CullRoom) -> f32 {
    let d = r.max - r.min;
    d.x * d.y * d.z
}

/// Every doorway of `room` bar the one it was `entered_by`, seen within
/// `window`: into a closed room, walked on; anywhere else, recorded.
#[allow(clippy::too_many_arguments)]
fn walk(
    room: u32,
    entered_by: Option<usize>,
    window: Window,
    depth: usize,
    clip_from_world: Mat4,
    clip_from_frame: Mat4,
    rooms: &[CullRoom],
    portals: &[ProbePortal],
    out: &mut Vec<[Vec4; 6]>,
) {
    for (i, p) in portals.iter().enumerate() {
        if (p.low != room && p.high != room) || Some(i) == entered_by {
            continue;
        }
        let Some(through) = doorway_window(p, clip_from_world).and_then(|w| intersect(window, w)) else {
            continue;
        };
        let other = if p.low == room { p.high } else { p.low };
        let closed = rooms.iter().any(|r| r.id == other && r.closed);
        if closed && depth + 1 < MAX_DOORWAYS {
            walk(other, Some(i), through, depth + 1, clip_from_world, clip_from_frame, rooms, portals, out);
        } else {
            // Outdoors, an open room, or too deep to follow: what the opening
            // shows may be outside.
            out.push(crate::renderer::shadow::frustum_planes(window_matrix(through) * clip_from_frame));
        }
    }
}

/// Where doorway `p` lands on screen, as an NDC window; `None` when it is
/// wholly off screen or behind the eye.
///
/// A doorway beside the eye can reach across its plane -- some corners in
/// front, some behind -- and corners behind the eye cannot be projected. So
/// the box is CLIPPED against a plane just in front of the eye first: the
/// corners in front are projected, and every edge running from in front to
/// behind contributes the point where it crosses that plane, which projects
/// far out in its own direction. The outline of all of them is the outline of
/// the clipped box (a convex hull's bounding rectangle is its vertices').
fn doorway_window(p: &ProbePortal, clip_from_world: Mat4) -> Option<Window> {
    const NEAR_W: f32 = 1e-3;
    let corners: [Vec4; 8] = std::array::from_fn(|i| {
        clip_from_world
            * Vec3::new(
                if i & 1 == 0 { p.min.x } else { p.max.x },
                if i & 2 == 0 { p.min.y } else { p.max.y },
                if i & 4 == 0 { p.min.z } else { p.max.z },
            )
            .extend(1.0)
    });
    let mut w = [f32::INFINITY, f32::INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY];
    let mut take = |c: Vec4| {
        let (x, y) = (c.x / c.w, c.y / c.w);
        w = [w[0].min(x), w[1].min(y), w[2].max(x), w[3].max(y)];
    };
    let mut any_in_front = false;
    for (i, c) in corners.iter().enumerate() {
        if c.w > NEAR_W {
            any_in_front = true;
            take(*c);
            // The three edges from this corner: along x, y and z (flip one
            // bit of the index). Where one runs behind the eye, where it
            // crosses the near plane.
            for bit in [1, 2, 4] {
                let other = corners[i ^ bit];
                if other.w <= NEAR_W {
                    let t = (c.w - NEAR_W) / (c.w - other.w);
                    take(c + (other - c) * t);
                }
            }
        }
    }
    if !any_in_front {
        return None;
    }
    intersect(w, FULL)
}

fn intersect(a: Window, b: Window) -> Option<Window> {
    let w = [a[0].max(b[0]), a[1].max(b[1]), a[2].min(b[2]), a[3].min(b[3])];
    (w[0] < w[2] && w[1] < w[3]).then_some(w)
}

/// The clip-space map taking NDC window `w` to the whole screen: applied after
/// a camera's projection, the frustum it bounds is the camera's, narrowed to
/// the window.
fn window_matrix(w: Window) -> Mat4 {
    let (sx, sy) = (2.0 / (w[2] - w[0]), 2.0 / (w[3] - w[1]));
    let (tx, ty) = (-(w[0] + w[2]) / (w[2] - w[0]), -(w[1] + w[3]) / (w[3] - w[1]));
    Mat4::from_cols(
        Vec4::new(sx, 0.0, 0.0, 0.0),
        Vec4::new(0.0, sy, 0.0, 0.0),
        Vec4::new(0.0, 0.0, 1.0, 0.0),
        Vec4::new(tx, ty, 0.0, 1.0),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::renderer::shadow::chunk_seen;
    use crate::renderer::shadow::CasterChunk;

    /// A hall 6 x 3 x 20 m (room 0) with a front door (to room 9, outdoors)
    /// in its z = 4 wall, and a side door into a hallway (room 1, closed) in
    /// its x = 3 wall; the hallway has no other way out.
    fn level() -> (Vec<CullRoom>, Vec<ProbePortal>) {
        let rooms = vec![
            CullRoom { id: 0, min: Vec3::new(-3.0, 0.0, -16.0), max: Vec3::new(3.0, 3.0, 4.0), closed: true },
            CullRoom { id: 1, min: Vec3::new(3.3, 0.0, -5.0), max: Vec3::new(10.0, 2.6, -1.0), closed: true },
            CullRoom { id: 9, min: Vec3::splat(-100.0), max: Vec3::splat(100.0), closed: false },
        ];
        let door = |min: Vec3, max: Vec3, axis: u32, low: u32, high: u32| ProbePortal { min, max, axis, low, high, wall: None };
        let portals = vec![
            // Front door, across z = 4: the hall (low) to outdoors (high).
            door(Vec3::new(-0.8, 0.0, 3.9), Vec3::new(0.8, 2.2, 4.4), 2, 0, 9),
            // Side door, across x = 3: the hall to the hallway.
            door(Vec3::new(2.9, 0.0, -3.8), Vec3::new(3.4, 2.2, -2.2), 0, 0, 1),
        ];
        (rooms, portals)
    }

    fn camera(eye: Vec3, at: Vec3) -> Mat4 {
        Mat4::perspective_rh(1.8, 1.0, 0.05, 200.0) * Mat4::look_at_rh(eye, at, Vec3::Y)
    }

    fn chunk(min: Vec3, max: Vec3) -> CasterChunk {
        CasterChunk { first_index: 0, index_count: 3, min, max }
    }

    /// Facing the front door from inside the hall: the lawn straight out of
    /// it is drawn; the lawn off to the side of the building, behind the
    /// hall's wall, is not.
    #[test]
    fn only_what_the_front_door_shows_is_drawn() {
        let (rooms, portals) = level();
        let cam = camera(Vec3::new(0.0, 1.6, -6.0), Vec3::new(0.0, 1.6, 10.0));
        let frusta = outdoor_frusta(Vec3::new(0.0, 1.6, -6.0), cam, cam, &rooms, &portals).expect("in a closed room");
        assert!(!frusta.is_empty(), "the front door is in view");
        let out_the_door = chunk(Vec3::new(-2.0, 0.0, 10.0), Vec3::new(2.0, 1.0, 14.0));
        assert!(chunk_seen(&out_the_door, &frusta), "the lawn through the door must be drawn");
        let beside_the_hall = chunk(Vec3::new(12.0, 0.0, 6.0), Vec3::new(20.0, 1.0, 14.0));
        assert!(!chunk_seen(&beside_the_hall, &frusta), "the lawn behind the wall must not be");
    }

    /// Facing the back wall, with neither door in view: nothing outside can
    /// be seen at all.
    #[test]
    fn with_no_doorway_in_view_nothing_outside_is_drawn() {
        let (rooms, portals) = level();
        let cam = camera(Vec3::new(0.0, 1.6, -6.0), Vec3::new(0.0, 1.6, -20.0));
        let frusta = outdoor_frusta(Vec3::new(0.0, 1.6, -6.0), cam, cam, &rooms, &portals).unwrap();
        assert!(frusta.is_empty(), "{} frusta", frusta.len());
    }

    /// Through the side door into the closed hallway, which has no way out:
    /// still nothing outside. Walking on through a closed room is what makes
    /// the culling work beyond the first doorway.
    #[test]
    fn a_closed_room_beyond_a_doorway_hides_the_outside_too() {
        let (rooms, portals) = level();
        let eye = Vec3::new(0.0, 1.6, -3.0);
        let cam = camera(eye, Vec3::new(10.0, 1.6, -3.0));
        let frusta = outdoor_frusta(eye, cam, cam, &rooms, &portals).unwrap();
        assert!(frusta.is_empty(), "the hallway is closed: {} frusta", frusta.len());
        // Open the hallway, and the lawn beyond its far end can be seen.
        let mut open = rooms.clone();
        open[1].closed = false;
        let frusta = outdoor_frusta(eye, cam, cam, &open, &portals).unwrap();
        assert!(chunk_seen(&chunk(Vec3::new(12.0, 0.0, -3.5), Vec3::new(14.0, 1.0, -2.5)), &frusta));
    }

    /// Outdoors, in an open room, or standing in a doorway: no culling.
    #[test]
    fn outside_a_closed_room_nothing_is_culled() {
        let (rooms, portals) = level();
        let cam = camera(Vec3::new(0.0, 1.6, 20.0), Vec3::new(0.0, 1.6, 0.0));
        assert!(outdoor_frusta(Vec3::new(0.0, 1.6, 20.0), cam, cam, &rooms, &portals).is_none());
        let mut open = rooms.clone();
        open[0].closed = false;
        let cam = camera(Vec3::new(0.0, 1.6, -6.0), Vec3::new(0.0, 1.6, 10.0));
        assert!(outdoor_frusta(Vec3::new(0.0, 1.6, -6.0), cam, cam, &open, &portals).is_none());
        // In the doorway itself: inside no room's interior.
        assert!(outdoor_frusta(Vec3::new(0.0, 1.6, 4.2), cam, cam, &rooms, &portals).is_none());
    }

    /// The planes come out in the CALLER's frame: the same camera expressed
    /// in a player frame (moved and turned) culls the same chunk the same way.
    #[test]
    fn the_planes_are_in_the_frame_the_caller_draws_in() {
        let (rooms, portals) = level();
        let eye = Vec3::new(0.0, 1.6, -6.0);
        let cam_world = camera(eye, Vec3::new(0.0, 1.6, 10.0));
        // A player frame: yaw by 90 degrees and an offset, as the XR renderer
        // has. Geometry in that frame is `world_to_player * world`.
        let world_to_player = Mat4::from_rotation_y(1.2) * Mat4::from_translation(Vec3::new(-5.0, 0.0, 3.0));
        let clip_from_frame = cam_world * world_to_player.inverse();
        let frusta = outdoor_frusta(eye, cam_world, clip_from_frame, &rooms, &portals).unwrap();
        let world_chunk = chunk(Vec3::new(-2.0, 0.0, 10.0), Vec3::new(2.0, 1.0, 14.0));
        // The chunk's bounds in the player frame (its corners' bounding box).
        let corners: Vec<Vec3> = (0..8)
            .map(|i| {
                let c = Vec3::new(
                    if i & 1 == 0 { world_chunk.min.x } else { world_chunk.max.x },
                    if i & 2 == 0 { world_chunk.min.y } else { world_chunk.max.y },
                    if i & 4 == 0 { world_chunk.min.z } else { world_chunk.max.z },
                );
                world_to_player.transform_point3(c)
            })
            .collect();
        let min = corners.iter().fold(Vec3::splat(f32::INFINITY), |a, &b| a.min(b));
        let max = corners.iter().fold(Vec3::splat(f32::NEG_INFINITY), |a, &b| a.max(b));
        assert!(chunk_seen(&chunk(min, max), &frusta));
    }
}
