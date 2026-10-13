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

/// WHAT AN EYE IN A CLOSED ROOM CAN SEE OF THE MODELS (`Levers::model_rooms`):
/// those that reach into the room itself, and those one of its doorways
/// shows. Everything else stands behind the room's walls -- a lamp two rooms
/// away, in the eye's frustum, was drawn in full: its thousands of wire
/// triangles through the scene pass and the motion pass for no pixel (Quest 3,
/// 2026-10-13, torch_doorway_out: five lamps' thin parts, 39,000 triangles an
/// eye, 5,000 fragments between them).
///
/// One doorway deep: what a doorway shows is not narrowed further by the
/// rooms beyond it. See [`room_view`].
#[derive(Clone, Debug)]
pub struct RoomView {
    /// The room's box, WORLD.
    room: (Vec3, Vec3),
    /// The view narrowed to each of its doorways in sight, in the caller's
    /// frame.
    doorways: Vec<[Vec4; 6]>,
}

impl RoomView {
    /// Whether a model may be seen: its bounding sphere, centre in the WORLD
    /// (`world_centre`) and in the caller's frame (`bound`, xyz and radius),
    /// reaches into the room or into what a doorway shows.
    pub fn shows(&self, world_centre: Vec3, bound: Vec4, margin: f32) -> bool {
        let r = bound.w + margin;
        let nearest = world_centre.clamp(self.room.0, self.room.1);
        nearest.distance_squared(world_centre) <= r * r
            || self.doorways.iter().any(|p| crate::renderer::shadow::sphere_in_frustum(p, bound.truncate().extend(r)))
    }

    /// How many of the room's doorways are in sight.
    pub fn doorways(&self) -> usize {
        self.doorways.len()
    }
}

/// The closed room `eye` is in and what its doorways show, or `None` where
/// nothing can be culled this way (as [`outdoor_frusta`]: the eye in no room,
/// in an open one, in a doorway). Frames as [`outdoor_frusta`].
pub fn room_view(
    eye: Vec3,
    clip_from_world: Mat4,
    clip_from_frame: Mat4,
    rooms: &[CullRoom],
    portals: &[ProbePortal],
) -> Option<RoomView> {
    let start = rooms
        .iter()
        .filter(|r| (r.min.cmplt(eye) & eye.cmplt(r.max)).all())
        .min_by(|a, b| volume(a).total_cmp(&volume(b)))?;
    if !start.closed {
        return None;
    }
    let doorways = portals
        .iter()
        .filter(|p| p.low == start.id || p.high == start.id)
        .filter_map(|p| doorway_window(p, clip_from_world))
        .map(|w| crate::renderer::shadow::frustum_planes(window_matrix(w) * clip_from_frame))
        .collect();
    Some(RoomView { room: (start.min, start.max), doorways })
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

    /// A model is drawn from inside the hall where it stands in the hall, in
    /// a doorway of it, or where a doorway shows it -- not behind a wall, in
    /// the frustum or out of it.
    #[test]
    fn a_closed_rooms_walls_hide_the_models_behind_them() {
        let (rooms, portals) = level();
        let eye = Vec3::new(0.0, 1.6, -6.0);
        let cam = camera(eye, Vec3::new(0.0, 1.6, 10.0));
        let view = room_view(eye, cam, cam, &rooms, &portals).expect("in a closed room");
        let shows = |c: Vec3, r: f32| view.shows(c, c.extend(r), 0.25);
        assert!(shows(Vec3::new(1.0, 2.5, 0.0), 0.3), "a lamp in the hall");
        assert!(shows(Vec3::new(-2.0, 1.0, -15.0), 0.3), "one behind the eye, in the hall: the frustum's to cull, not the room's");
        assert!(shows(Vec3::new(0.0, 1.0, 4.15), 1.2), "the front door's leaf, in its doorway");
        assert!(shows(Vec3::new(0.0, 1.0, 12.0), 0.5), "a lamp on the lawn, straight out of the door");
        assert!(!shows(Vec3::new(8.0, 1.0, 12.0), 0.5), "one on the lawn behind the hall's front wall");
        assert!(!shows(Vec3::new(-9.0, 2.0, 0.0), 0.5), "one behind the hall's far side wall");
        // The side door is behind the frustum's edge here: the hallway's lamp
        // is hidden; turned to face that door, it shows.
        let in_hallway = Vec3::new(4.7, 0.9, -1.5);
        let turned = camera(eye, Vec3::new(10.0, 1.6, -3.0));
        let view = room_view(eye, turned, turned, &rooms, &portals).unwrap();
        assert!(view.shows(in_hallway, in_hallway.extend(0.4), 0.25), "the hallway's lamp through the side door");
        assert_eq!(view.doorways(), 1, "the front door is out of this view");
        // A sphere that only just reaches the room's box counts as in it.
        assert!(view.shows(Vec3::new(-3.5, 1.0, 0.0), Vec3::new(-3.5, 1.0, 0.0).extend(0.3), 0.25));
        // Outdoors, or in a doorway: no view, nothing culled.
        assert!(room_view(Vec3::new(0.0, 1.6, 10.0), cam, cam, &rooms, &portals).is_none());
        assert!(room_view(Vec3::new(0.0, 1.6, 4.2), cam, cam, &rooms, &portals).is_none());
        // The frame the caller draws in: the planes follow `clip_from_frame`.
        let shift = Mat4::from_translation(Vec3::new(50.0, 0.0, 0.0));
        let view = room_view(eye, cam, cam * shift.inverse(), &rooms, &portals).unwrap();
        let lawn = Vec3::new(0.0, 1.0, 12.0);
        assert!(view.shows(lawn, shift.transform_point3(lawn).extend(0.5), 0.25));
        let hidden = Vec3::new(8.0, 1.0, 12.0);
        assert!(!view.shows(hidden, shift.transform_point3(hidden).extend(0.5), 0.25));
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
