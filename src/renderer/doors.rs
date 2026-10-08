//! DOORS AS THE RENDERER SEES THEM: each leaf where it hangs this frame, and
//! whether it seals its doorway.
//!
//! A door is never baked (`space_soup_engine::door`): the lightmaps, the lamp
//! masks and the probe photographs hold the level with its doorways open. So
//! the renderer adds what a door changes, live:
//!
//! - its SHADOWS: the leaf is a mesh drawn like any other model (lit by its
//!   room's baked light and the lamps), drawn into the sun's moving-objects
//!   map and the torch's spot slot as every model is, and -- the part a baked
//!   level needs -- into a moving casters' tile for each lamp reaching it
//!   (`shadow::moving_caster_tiles`), whose shadow takes away the light a
//!   closed leaf stops: the spill of the lamps beyond it through its doorway,
//!   narrowing to a wedge as it closes;
//! - its CULLING: a shut doorway is a wall to `portal_cull`, so what lies
//!   past it -- the outdoors seen through the next room -- is not drawn
//!   (`shut_portals`).
//!
//! The app hands the doors over each frame (`XrRenderer::set_doors`), in the
//! WORLD, as the server last placed them or as the local hand is pushing them.

use glam::{Quat, Vec3};

use crate::renderer::uniforms::ProbePortal;

/// One door leaf this frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DoorView {
    /// The leaf's eight corners where it stands now, world.
    pub corners: [Vec3; 8],
    /// The leaf's centre when closed, world: which doorway it hangs in.
    pub closed_centre: Vec3,
    /// Within `space_soup_engine::door::SHUT_DEG` of closed.
    pub shut: bool,
}

impl DoorView {
    /// The corners in the player's frame, where the shadow tiles are fitted:
    /// `yaw_inv * (world - offset)`, as every bit of geometry is sent.
    pub fn corners_in(&self, offset: Vec3, yaw: f32) -> [Vec3; 8] {
        let yaw_inv = Quat::from_rotation_y(-yaw);
        self.corners.map(|c| yaw_inv * (c - offset))
    }
}

/// How far outside a doorway's box a leaf's closed centre may sit and still
/// hang in it, metres: a leaf hung on the wall's face stands proud of a carve
/// cut exactly through the wall.
const IN_DOORWAY: f32 = 0.1;

/// WHICH DOORWAYS ARE SHUT: `portals[i]` is shut when at least one door hangs
/// in it and every door hanging in it is shut -- a double door with one leaf
/// ajar is open. A doorway with no door is never shut.
pub fn shut_portals(portals: &[ProbePortal], doors: &[DoorView]) -> Vec<bool> {
    portals
        .iter()
        .map(|p| {
            let (lo, hi) = (p.min - Vec3::splat(IN_DOORWAY), p.max + Vec3::splat(IN_DOORWAY));
            let mut hung = doors.iter().filter(|d| (lo.cmple(d.closed_centre) & d.closed_centre.cmple(hi)).all()).peekable();
            hung.peek().is_some() && hung.all(|d| d.shut)
        })
        .collect()
}

/// How far in front of a leaf's seen face it reads its room's light, metres:
/// past the doorway's blend between the rooms either side (`room_light`
/// ramps over about a quarter metre beyond a doorway's box, and the box holds
/// the wall's thickness).
pub const LIGHT_READ_OUT: f32 = 0.6;

/// WHERE A LEAF READS ITS ROOM'S LIGHT. A model takes the baked light of the
/// room it stands in (`room_light::room_light_at` at its centre); a shut leaf
/// stands IN its doorway, where that light is the blend of the rooms either
/// side -- so the face toward a dark room glowed with the bright one's light.
/// The eye sees one face of a leaf (it is a board), so the leaf reads its
/// light `LIGHT_READ_OUT` in front of the face toward `eye`: the room that
/// face looks into. `centre` and `eye` are world; a centre no door stands at
/// is returned as it is.
pub fn light_point(doors: &[DoorView], centre: Vec3, eye: Vec3) -> Vec3 {
    let Some(d) = doors.iter().find(|d| (d.corners.iter().sum::<Vec3>() / 8.0 - centre).length() < 0.05) else {
        return centre;
    };
    // The leaf's thin axis: the shortest of the three edges from a corner.
    let c = d.corners;
    let normal = [c[1] - c[0], c[2] - c[0], c[4] - c[0]]
        .into_iter()
        .min_by(|a, b| a.length_squared().total_cmp(&b.length_squared()))
        .unwrap_or(Vec3::X)
        .normalize_or_zero();
    let side = if normal.dot(eye - centre) >= 0.0 { 1.0 } else { -1.0 };
    centre + normal * (side * LIGHT_READ_OUT)
}

/// The doorways portal culling may walk through: every one but the shut.
pub fn open_portals(portals: &[ProbePortal], shut: &[bool]) -> Vec<ProbePortal> {
    portals.iter().zip(shut.iter().chain(std::iter::repeat(&false))).filter(|(_, s)| !**s).map(|(p, _)| *p).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn portal() -> ProbePortal {
        ProbePortal {
            min: Vec3::new(2.5, 0.0, -3.8),
            max: Vec3::new(3.1, 2.2, -2.2),
            axis: 0,
            low: 0,
            high: 1,
            wall: Some((2.7, 3.0)),
        }
    }

    fn leaf(z: f32, shut: bool) -> DoorView {
        DoorView { corners: [Vec3::ZERO; 8], closed_centre: Vec3::new(2.85, 1.1, z), shut }
    }

    #[test]
    fn a_doorway_is_shut_only_when_every_leaf_in_it_is() {
        let p = [portal()];
        assert_eq!(shut_portals(&p, &[leaf(-3.39, true), leaf(-2.61, true)]), vec![true]);
        assert_eq!(shut_portals(&p, &[leaf(-3.39, true), leaf(-2.61, false)]), vec![false], "one leaf ajar opens it");
        assert_eq!(shut_portals(&p, &[]), vec![false], "no door, no seal");
        // A door elsewhere does not seal this doorway.
        let far = DoorView { closed_centre: Vec3::new(10.15, 1.1, -3.39), ..leaf(0.0, true) };
        assert_eq!(shut_portals(&p, &[far]), vec![false]);
        assert!(open_portals(&p, &[true]).is_empty());
        assert_eq!(open_portals(&p, &[false]).len(), 1);
    }

    #[test]
    fn the_corners_reach_the_shaders_in_the_players_frame() {
        let d = DoorView { corners: [Vec3::new(1.0, 1.0, 0.0); 8], ..leaf(0.0, true) };
        let c = d.corners_in(Vec3::new(1.0, 0.0, 0.0), std::f32::consts::FRAC_PI_2)[0];
        assert!((c - Vec3::new(0.0, 1.0, 0.0)).length() < 1e-6);
        let c = d.corners_in(Vec3::ZERO, std::f32::consts::FRAC_PI_2)[0];
        // A quarter turn of the rig: the world's +x lies along the player's +z.
        assert!((c - Vec3::new(0.0, 1.0, 1.0)).length() < 1e-5, "{c}");
    }

    /// A shut leaf in a doorway across x = 2.85 reads its light in the room
    /// on the eye's side, whichever side that is; anything else as it was.
    #[test]
    fn a_leaf_reads_the_light_of_the_room_its_seen_face_looks_into() {
        let centre = Vec3::new(2.85, 1.1, -3.39);
        let half = Vec3::new(0.02, 1.09, 0.38);
        let corners = std::array::from_fn(|i| {
            centre
                + Vec3::new(
                    if i & 1 == 0 { -half.x } else { half.x },
                    if i & 2 == 0 { -half.y } else { half.y },
                    if i & 4 == 0 { -half.z } else { half.z },
                )
        });
        let d = [DoorView { corners, closed_centre: centre, shut: true }];
        let hall = light_point(&d, centre, Vec3::new(-1.0, 1.6, -3.0));
        assert!((hall - (centre - Vec3::X * LIGHT_READ_OUT)).length() < 1e-5, "{hall}");
        let hallway = light_point(&d, centre, Vec3::new(6.0, 1.6, -3.0));
        assert!((hallway - (centre + Vec3::X * LIGHT_READ_OUT)).length() < 1e-5, "{hallway}");
        let elsewhere = Vec3::new(0.0, 1.0, 0.0);
        assert_eq!(light_point(&d, elsewhere, Vec3::ZERO), elsewhere);
    }

    /// PORTAL CULLING WITH A DOOR: from the hallway (closed) looking back
    /// through its doorway, across the hall (closed) and out of a doorway in
    /// the hall's far wall, the lawn beyond is drawn while the hallway's door
    /// stands open, and nothing outside is drawn once both its leaves are shut.
    #[test]
    fn a_shut_door_hides_what_its_doorway_showed() {
        use crate::renderer::portal_cull::{outdoor_frusta, CullRoom};
        use crate::renderer::shadow::{chunk_seen, CasterChunk};
        use glam::Mat4;
        let rooms = vec![
            CullRoom { id: 0, min: Vec3::new(-3.0, 0.0, -16.0), max: Vec3::new(3.0, 3.0, 4.0), closed: true },
            CullRoom { id: 1, min: Vec3::new(3.0, 0.0, -4.2), max: Vec3::new(10.0, 2.6, -1.8), closed: true },
            CullRoom { id: 9, min: Vec3::splat(-100.0), max: Vec3::splat(100.0), closed: false },
        ];
        let doorway = |min: Vec3, max: Vec3, axis: u32, low: u32, high: u32| ProbePortal { min, max, axis, low, high, wall: None };
        let portals = vec![
            doorway(Vec3::new(-3.4, 0.0, -3.8), Vec3::new(-2.9, 2.2, -2.2), 0, 9, 0),
            doorway(Vec3::new(2.5, 0.0, -3.8), Vec3::new(3.1, 2.2, -2.2), 0, 0, 1),
        ];
        // From the hallway, back through its doorway and across the hall to
        // the lawn beyond the hall's west door, in line with it.
        let eye = Vec3::new(5.5, 1.6, -3.0);
        let at = Vec3::new(-10.0, 1.2, -3.0);
        let cam = Mat4::perspective_rh(1.8, 1.0, 0.05, 200.0) * Mat4::look_at_rh(eye, at, Vec3::Y);
        let lawn = CasterChunk { first_index: 0, index_count: 3, min: Vec3::new(-12.0, 0.0, -4.0), max: Vec3::new(-8.0, 1.0, -2.0) };
        let leaves = [leaf(-3.39, false), leaf(-2.61, false)];
        let open = open_portals(&portals, &shut_portals(&portals, &leaves));
        let frusta = outdoor_frusta(eye, cam, cam, &rooms, &open).expect("in a closed room");
        assert!(chunk_seen(&lawn, &frusta), "through the open door the lawn is in view");
        let leaves = [leaf(-3.39, true), leaf(-2.61, true)];
        let open = open_portals(&portals, &shut_portals(&portals, &leaves));
        let frusta = outdoor_frusta(eye, cam, cam, &rooms, &open).expect("in a closed room");
        assert!(frusta.is_empty(), "behind a shut door nothing outside is drawn: {} frusta", frusta.len());
    }
}
