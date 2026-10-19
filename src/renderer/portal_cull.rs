use glam::{Mat4, Vec3, Vec4};

use crate::renderer::uniforms::ProbePortal;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CullRoom {
    pub id: u32,
    pub min: Vec3,
    pub max: Vec3,

    pub closed: bool,
}

pub const MAX_DOORWAYS: usize = 4;

type Window = [f32; 4];

const FULL: Window = [-1.0, -1.0, 1.0, 1.0];

pub fn outdoor_frusta(
    eye: Vec3,
    clip_from_world: Mat4,
    clip_from_frame: Mat4,
    rooms: &[CullRoom],
    portals: &[ProbePortal],
) -> Option<Vec<[Vec4; 6]>> {

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

#[derive(Clone, Debug)]
pub struct RoomView {

    room: (Vec3, Vec3),

    doorways: Vec<[Vec4; 6]>,
}

impl RoomView {

    pub fn shows(&self, world_centre: Vec3, bound: Vec4, margin: f32) -> bool {
        let r = bound.w + margin;
        let nearest = world_centre.clamp(self.room.0, self.room.1);
        nearest.distance_squared(world_centre) <= r * r
            || self.doorways.iter().any(|p| crate::renderer::shadow::sphere_in_frustum(p, bound.truncate().extend(r)))
    }

    pub fn doorways(&self) -> usize {
        self.doorways.len()
    }
}

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

            out.push(crate::renderer::shadow::frustum_planes(window_matrix(through) * clip_from_frame));
        }
    }
}

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

    fn level() -> (Vec<CullRoom>, Vec<ProbePortal>) {
        let rooms = vec![
            CullRoom { id: 0, min: Vec3::new(-3.0, 0.0, -16.0), max: Vec3::new(3.0, 3.0, 4.0), closed: true },
            CullRoom { id: 1, min: Vec3::new(3.3, 0.0, -5.0), max: Vec3::new(10.0, 2.6, -1.0), closed: true },
            CullRoom { id: 9, min: Vec3::splat(-100.0), max: Vec3::splat(100.0), closed: false },
        ];
        let door = |min: Vec3, max: Vec3, axis: u32, low: u32, high: u32| ProbePortal { min, max, axis, low, high, wall: None };
        let portals = vec![

            door(Vec3::new(-0.8, 0.0, 3.9), Vec3::new(0.8, 2.2, 4.4), 2, 0, 9),

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

    #[test]
    fn with_no_doorway_in_view_nothing_outside_is_drawn() {
        let (rooms, portals) = level();
        let cam = camera(Vec3::new(0.0, 1.6, -6.0), Vec3::new(0.0, 1.6, -20.0));
        let frusta = outdoor_frusta(Vec3::new(0.0, 1.6, -6.0), cam, cam, &rooms, &portals).unwrap();
        assert!(frusta.is_empty(), "{} frusta", frusta.len());
    }

    #[test]
    fn a_closed_room_beyond_a_doorway_hides_the_outside_too() {
        let (rooms, portals) = level();
        let eye = Vec3::new(0.0, 1.6, -3.0);
        let cam = camera(eye, Vec3::new(10.0, 1.6, -3.0));
        let frusta = outdoor_frusta(eye, cam, cam, &rooms, &portals).unwrap();
        assert!(frusta.is_empty(), "the hallway is closed: {} frusta", frusta.len());

        let mut open = rooms.clone();
        open[1].closed = false;
        let frusta = outdoor_frusta(eye, cam, cam, &open, &portals).unwrap();
        assert!(chunk_seen(&chunk(Vec3::new(12.0, 0.0, -3.5), Vec3::new(14.0, 1.0, -2.5)), &frusta));
    }

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

        let in_hallway = Vec3::new(4.7, 0.9, -1.5);
        let turned = camera(eye, Vec3::new(10.0, 1.6, -3.0));
        let view = room_view(eye, turned, turned, &rooms, &portals).unwrap();
        assert!(view.shows(in_hallway, in_hallway.extend(0.4), 0.25), "the hallway's lamp through the side door");
        assert_eq!(view.doorways(), 1, "the front door is out of this view");

        assert!(view.shows(Vec3::new(-3.5, 1.0, 0.0), Vec3::new(-3.5, 1.0, 0.0).extend(0.3), 0.25));

        assert!(room_view(Vec3::new(0.0, 1.6, 10.0), cam, cam, &rooms, &portals).is_none());
        assert!(room_view(Vec3::new(0.0, 1.6, 4.2), cam, cam, &rooms, &portals).is_none());

        let shift = Mat4::from_translation(Vec3::new(50.0, 0.0, 0.0));
        let view = room_view(eye, cam, cam * shift.inverse(), &rooms, &portals).unwrap();
        let lawn = Vec3::new(0.0, 1.0, 12.0);
        assert!(view.shows(lawn, shift.transform_point3(lawn).extend(0.5), 0.25));
        let hidden = Vec3::new(8.0, 1.0, 12.0);
        assert!(!view.shows(hidden, shift.transform_point3(hidden).extend(0.5), 0.25));
    }

    #[test]
    fn outside_a_closed_room_nothing_is_culled() {
        let (rooms, portals) = level();
        let cam = camera(Vec3::new(0.0, 1.6, 20.0), Vec3::new(0.0, 1.6, 0.0));
        assert!(outdoor_frusta(Vec3::new(0.0, 1.6, 20.0), cam, cam, &rooms, &portals).is_none());
        let mut open = rooms.clone();
        open[0].closed = false;
        let cam = camera(Vec3::new(0.0, 1.6, -6.0), Vec3::new(0.0, 1.6, 10.0));
        assert!(outdoor_frusta(Vec3::new(0.0, 1.6, -6.0), cam, cam, &open, &portals).is_none());

        assert!(outdoor_frusta(Vec3::new(0.0, 1.6, 4.2), cam, cam, &rooms, &portals).is_none());
    }

    #[test]
    fn the_planes_are_in_the_frame_the_caller_draws_in() {
        let (rooms, portals) = level();
        let eye = Vec3::new(0.0, 1.6, -6.0);
        let cam_world = camera(eye, Vec3::new(0.0, 1.6, 10.0));

        let world_to_player = Mat4::from_rotation_y(1.2) * Mat4::from_translation(Vec3::new(-5.0, 0.0, 3.0));
        let clip_from_frame = cam_world * world_to_player.inverse();
        let frusta = outdoor_frusta(eye, cam_world, clip_from_frame, &rooms, &portals).unwrap();
        let world_chunk = chunk(Vec3::new(-2.0, 0.0, 10.0), Vec3::new(2.0, 1.0, 14.0));

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
