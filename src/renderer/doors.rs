use glam::{Quat, Vec3};

use crate::renderer::uniforms::{ProbePortal, ProbeProxy};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DoorView {

    pub corners: [Vec3; 8],

    pub closed_centre: Vec3,

    pub shut: bool,

    pub rotation: Quat,
    pub closed_rotation: Quat,
}

impl DoorView {

    pub fn corners_in(&self, offset: Vec3, yaw: f32) -> [Vec3; 8] {
        let yaw_inv = Quat::from_rotation_y(-yaw);
        self.corners.map(|c| yaw_inv * (c - offset))
    }
}

const IN_DOORWAY: f32 = 0.1;

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

pub const LIGHT_READ_OUT: f32 = 0.6;

pub fn light_point(doors: &[DoorView], centre: Vec3, eye: Vec3) -> Vec3 {
    let Some(d) = doors.iter().find(|d| (d.corners.iter().sum::<Vec3>() / 8.0 - centre).length() < 0.05) else {
        return centre;
    };

    let c = d.corners;
    let normal = [c[1] - c[0], c[2] - c[0], c[4] - c[0]]
        .into_iter()
        .min_by(|a, b| a.length_squared().total_cmp(&b.length_squared()))
        .unwrap_or(Vec3::X)
        .normalize_or_zero();
    let side = if normal.dot(eye - centre) >= 0.0 { 1.0 } else { -1.0 };
    centre + normal * (side * LIGHT_READ_OUT)
}

pub fn posed_proxies(rest: &[ProbeProxy], doors: &[DoorView]) -> Vec<ProbeProxy> {
    rest.iter()
        .map(|p| {
            let mut p = *p;
            if let Some(d) = doors.iter().find(|d| !p.solid && (p.centre - d.closed_centre).length() < 0.05) {
                let turn = d.rotation * d.closed_rotation.inverse();
                let now = d.corners.iter().sum::<Vec3>() / 8.0;
                p.centre = now + turn * (p.centre - d.closed_centre);
                p.rotation = (turn * p.rotation).normalize();
            }
            p
        })
        .collect()
}

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
        DoorView { corners: [Vec3::ZERO; 8], closed_centre: Vec3::new(2.85, 1.1, z), shut, rotation: Quat::IDENTITY, closed_rotation: Quat::IDENTITY }
    }

    #[test]
    fn a_doorway_is_shut_only_when_every_leaf_in_it_is() {
        let p = [portal()];
        assert_eq!(shut_portals(&p, &[leaf(-3.39, true), leaf(-2.61, true)]), vec![true]);
        assert_eq!(shut_portals(&p, &[leaf(-3.39, true), leaf(-2.61, false)]), vec![false], "one leaf ajar opens it");
        assert_eq!(shut_portals(&p, &[]), vec![false], "no door, no seal");

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

        assert!((c - Vec3::new(0.0, 1.0, 1.0)).length() < 1e-5, "{c}");
    }

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
        let d = [DoorView { corners, closed_centre: centre, shut: true, rotation: Quat::IDENTITY, closed_rotation: Quat::IDENTITY }];
        let hall = light_point(&d, centre, Vec3::new(-1.0, 1.6, -3.0));
        assert!((hall - (centre - Vec3::X * LIGHT_READ_OUT)).length() < 1e-5, "{hall}");
        let hallway = light_point(&d, centre, Vec3::new(6.0, 1.6, -3.0));
        assert!((hallway - (centre + Vec3::X * LIGHT_READ_OUT)).length() < 1e-5, "{hallway}");
        let elsewhere = Vec3::new(0.0, 1.0, 0.0);
        assert_eq!(light_point(&d, elsewhere, Vec3::ZERO), elsewhere);
    }

    #[test]
    fn a_doors_reflection_proxy_turns_with_its_leaf() {
        let closed = Vec3::new(2.85, 1.1, -3.39);
        let hinge = Vec3::new(2.85, 1.1, -3.77);
        let turn = Quat::from_rotation_y(0.5);
        let now = hinge + turn * (closed - hinge);
        let leaf = DoorView {
            corners: [now; 8],
            closed_centre: closed,
            shut: false,
            rotation: turn,
            closed_rotation: Quat::IDENTITY,
        };
        let proxy = ProbeProxy {
            centre: closed,
            half_size: Vec3::new(0.02, 1.09, 0.38),
            rotation: Quat::IDENTITY,
            volume: 1,
            solid: false,
            field: Some(2),
            cards: None,
        };
        let pillar = ProbeProxy { centre: Vec3::new(0.0, 1.5, -6.0), solid: true, field: None, ..proxy };
        let posed = posed_proxies(&[proxy, pillar], &[leaf]);
        assert!((posed[0].centre - now).length() < 1e-5, "{}", posed[0].centre);
        assert!(posed[0].rotation.angle_between(turn) < 1e-5);
        assert_eq!(posed[1], pillar);
    }

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
