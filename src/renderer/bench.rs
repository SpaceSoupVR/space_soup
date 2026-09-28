//! A BENCHMARK VIEWPOINT: the camera pinned to a named place in the level, so
//! a frame's cost can be measured with the headset lying on a desk.
//!
//! # Why
//!
//! Every cost measured before this was measured from wherever the person in
//! the headset happened to stand and look, so two numbers from two sessions
//! described two different frames, and a change had to be tried on a head.
//! Pinned, a named view renders the same frame every run: an A/B across two
//! builds compares like with like, and nobody has to wear the headset for it.
//! The views are the offline harness's coordinates, so the frame measured on
//! the device is also one that can be looked at on the development machine.
//!
//! # How it is pinned
//!
//! The level is drawn in the PLAYER'S frame -- world = `offset + yaw *
//! tracked` -- so a world pose is reached in two parts. The rig is moved
//! (offset and yaw) so that a fixed tracked head lands on the viewpoint, and
//! the tracked head itself is replaced by that fixed pose, pitched to look at
//! the target. Everything downstream -- culling, the probes streamed in, the
//! lights chosen, exposure -- then sees one consistent place.
//!
//! Each eye keeps where it sits relative to the real head, so the stereo pair
//! and each eye's field of view are the headset's own. Tracking can be lost
//! with the headset on a desk (a dark room, the cameras facing the table); the
//! pose is pinned anyway, so the eyes fall back to the average spacing rather
//! than the frame being skipped.

use glam::{Quat, Vec3};
use serde::{Deserialize, Serialize};

/// Where the fixed tracked head stands in the rig: at a standing eye height
/// over the rig's origin, which is then where a standing player's feet are.
pub const TRACKED_EYE_HEIGHT: f32 = 1.6;

/// The spacing given to the eyes when the runtime's own poses mean nothing.
/// The adult average; the headset's is used whenever it is known.
pub const FALLBACK_IPD: f32 = 0.063;

/// A named viewpoint, as the lever file carries it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BenchPose {
    /// What the results call this view. Letters, digits, `_` and `-`: it is
    /// written into the comma-separated lever summary on every `PERF` line.
    pub name: String,
    /// Where the eyes are, in world metres.
    pub eye: [f32; 3],
    /// What they look at, in world metres.
    pub at: [f32; 3],
}

impl BenchPose {
    /// Why this pose cannot be used, if it cannot.
    pub fn problem(&self) -> Option<String> {
        if self.name.is_empty() || !self.name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-') {
            return Some(format!("bench name {:?} must be letters, digits, '_' or '-'", self.name));
        }
        if !self.eye.iter().chain(self.at.iter()).all(|v| v.is_finite()) {
            return Some(format!("bench '{}' has a coordinate that is not a number", self.name));
        }
        if (Vec3::from(self.at) - Vec3::from(self.eye)).length_squared() < 1e-6 {
            return Some(format!("bench '{}' looks at its own eye", self.name));
        }
        None
    }
}

/// A `BenchPose` taken apart into what the app and the renderer each set.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BenchRig {
    /// The rig's origin in the world: `Locomotion::player_offset`.
    pub offset: Vec3,
    /// The rig's turn about the vertical: `Locomotion::player_yaw`.
    pub yaw: f32,
    /// The tracked head, in the rig's own (stage) space.
    pub head_position: Vec3,
    /// Its orientation there: pitch only, since the yaw is the rig's.
    pub head_rotation: Quat,
}

impl BenchRig {
    pub fn for_pose(pose: &BenchPose) -> Self {
        let eye = Vec3::from(pose.eye);
        let dir = (Vec3::from(pose.at) - eye).normalize_or(Vec3::NEG_Z);
        // `from_rotation_y(yaw) * -Z` is `(-sin yaw, 0, -cos yaw)`. Straight up
        // or down has no heading at all; any yaw is then right, so zero.
        let yaw = if dir.x * dir.x + dir.z * dir.z > 1e-10 { (-dir.x).atan2(-dir.z) } else { 0.0 };
        // `from_rotation_x(pitch) * -Z` is `(0, sin pitch, -cos pitch)`.
        let pitch = dir.y.clamp(-1.0, 1.0).asin();
        let head_position = Vec3::new(0.0, TRACKED_EYE_HEIGHT, 0.0);
        Self {
            // The yaw turns about the vertical, which leaves the head's height
            // where it is: the head lands exactly on `eye`.
            offset: eye - head_position,
            yaw,
            head_position,
            head_rotation: Quat::from_rotation_x(pitch),
        }
    }
}

/// The two eyes of the real head, moved onto the pinned one.
///
/// Each eye keeps its offset and its turn relative to the real head -- the
/// midpoint of the pair, turned halfway between the two eyes -- so the
/// separation and any canting are the headset's. (Halfway rather than the left
/// eye's turn: on a canted display the left eye's turn would swing the pair
/// off the pinned head's right axis. The Quest 3's eyes are parallel, where
/// the two agree.) When `located` is false, or the poses are
/// not usable numbers, the eyes are set `FALLBACK_IPD` apart, straight ahead.
pub fn pin_eyes(eyes: [(Vec3, Quat); 2], located: bool, head_position: Vec3, head_rotation: Quat) -> [(Vec3, Quat); 2] {
    let head_rotation = head_rotation.normalize();
    let usable = located
        && eyes.iter().all(|(p, q)| p.is_finite() && q.is_finite() && (q.length() - 1.0).abs() < 0.1);
    let relative: [(Vec3, Quat); 2] = if usable {
        let centre = (eyes[0].0 + eyes[1].0) * 0.5;
        let inverse = eyes[0].1.normalize().slerp(eyes[1].1.normalize(), 0.5).inverse();
        [0, 1].map(|i| (inverse * (eyes[i].0 - centre), (inverse * eyes[i].1.normalize()).normalize()))
    } else {
        let half = FALLBACK_IPD * 0.5;
        [(Vec3::new(-half, 0.0, 0.0), Quat::IDENTITY), (Vec3::new(half, 0.0, 0.0), Quat::IDENTITY)]
    };
    relative.map(|(p, q)| (head_position + head_rotation * p, (head_rotation * q).normalize()))
}

/// `pin_eyes` over the runtime's own views, in place: the adapter the app and
/// the renderer both call, so the two cannot pin differently.
#[cfg(target_os = "android")]
pub fn pin_xr_views(views: &mut [openxr::View], located: bool, rig: &BenchRig) {
    if views.len() < 2 {
        return;
    }
    let get = |v: &openxr::View| {
        let (p, o) = (v.pose.position, v.pose.orientation);
        (Vec3::new(p.x, p.y, p.z), Quat::from_xyzw(o.x, o.y, o.z, o.w))
    };
    let pinned = pin_eyes([get(&views[0]), get(&views[1])], located, rig.head_position, rig.head_rotation);
    for (view, (p, q)) in views.iter_mut().zip(pinned) {
        view.pose.position = openxr::Vector3f { x: p.x, y: p.y, z: p.z };
        view.pose.orientation = openxr::Quaternionf { x: q.x, y: q.y, z: q.z, w: q.w };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pose(eye: [f32; 3], at: [f32; 3]) -> BenchPose {
        BenchPose { name: "t".into(), eye, at }
    }

    /// Through the same transform the app puts every tracked pose through:
    /// `Locomotion::apply_to_head`.
    fn to_world(rig: &BenchRig, p: Vec3, q: Quat) -> (Vec3, Quat) {
        let yaw = Quat::from_rotation_y(rig.yaw);
        (rig.offset + yaw * p, yaw * q)
    }

    #[test]
    fn the_pinned_head_stands_at_the_eye_and_looks_at_the_target() {
        for (eye, at) in [
            ([-1.3, 1.6, -14.5], [0.4, 2.4, 3.7]),
            ([3.4, 1.6, -3.0], [9.0, 1.4, -3.0]),
            ([0.2, 1.6, -2.0], [0.0, 0.2, 3.7]),
            ([0.0, 1.6, 2.5], [0.2, 0.0, -6.0]),
            ([1.0, 0.3, 1.0], [-4.0, 3.0, 2.0]),
        ] {
            let rig = BenchRig::for_pose(&pose(eye, at));
            let (p, q) = to_world(&rig, rig.head_position, rig.head_rotation);
            assert!((p - Vec3::from(eye)).length() < 1e-4, "head at {p} for eye {eye:?}");
            let want = (Vec3::from(at) - Vec3::from(eye)).normalize();
            let got = q * Vec3::NEG_Z;
            assert!(got.dot(want) > 0.99999, "looking along {got}, wanted {want}");
            // Level: no roll, so the view's right axis stays horizontal.
            assert!((q * Vec3::X).y.abs() < 1e-5, "the view is rolled for {eye:?} -> {at:?}");
        }
    }

    #[test]
    fn the_eyes_keep_the_headsets_separation_and_turn() {
        // A real head somewhere on the desk, turned and tilted, eyes 64 mm
        // apart and each toed out a little, as a canted display would be.
        let head = Quat::from_euler(glam::EulerRot::YXZ, 1.1, -0.4, 0.2);
        let centre = Vec3::new(0.4, 0.9, -0.3);
        let toe = Quat::from_rotation_y(0.05);
        let eyes = [
            (centre + head * Vec3::new(-0.032, 0.0, 0.0), head * toe),
            (centre + head * Vec3::new(0.032, 0.0, 0.0), head * toe.inverse()),
        ];
        let rig = BenchRig::for_pose(&pose([0.3, 1.6, -3.0], [0.0, 0.9, -7.0]));
        let pinned = pin_eyes(eyes, true, rig.head_position, rig.head_rotation);
        let gap = pinned[1].0 - pinned[0].0;
        assert!((gap.length() - 0.064).abs() < 1e-5, "separation {}", gap.length());
        assert!((gap.normalize().dot(rig.head_rotation * Vec3::X) - 1.0).abs() < 1e-5, "eyes not along the head's right");
        let mid = (pinned[0].0 + pinned[1].0) * 0.5;
        assert!((mid - rig.head_position).length() < 1e-5, "the pair is not centred on the pinned head");
        // Each eye keeps its own toe relative to the pinned head.
        let left = rig.head_rotation.inverse() * pinned[0].1;
        assert!(left.angle_between(toe) < 1e-4, "the left eye lost its turn");
    }

    #[test]
    fn a_head_that_is_not_tracked_still_gets_a_usable_pair() {
        let nan = (Vec3::splat(f32::NAN), Quat::from_xyzw(0.0, 0.0, 0.0, 0.0));
        let rig = BenchRig::for_pose(&pose([0.0, 1.6, 0.0], [0.0, 1.6, -1.0]));
        for (eyes, located) in [([nan, nan], true), ([nan, nan], false), ([(Vec3::ZERO, Quat::IDENTITY); 2], false)] {
            let pinned = pin_eyes(eyes, located, rig.head_position, rig.head_rotation);
            for (p, q) in pinned {
                assert!(p.is_finite() && q.is_finite() && (q.length() - 1.0).abs() < 1e-5, "{p} {q}");
            }
            assert!(((pinned[1].0 - pinned[0].0).length() - FALLBACK_IPD).abs() < 1e-6);
        }
    }

    #[test]
    fn straight_down_is_a_valid_pose() {
        let rig = BenchRig::for_pose(&pose([0.0, 2.0, 0.0], [0.0, 0.0, 0.0]));
        assert!(rig.yaw.is_finite() && rig.head_rotation.is_finite());
        let (_, q) = to_world(&rig, rig.head_position, rig.head_rotation);
        assert!((q * Vec3::NEG_Z).dot(Vec3::NEG_Y) > 0.99999);
    }

    #[test]
    fn a_pose_that_cannot_be_used_says_why() {
        assert!(pose([0.0; 3], [0.0, 0.0, -1.0]).problem().is_none());
        assert!(BenchPose { name: "hall back".into(), ..pose([0.0; 3], [1.0; 3]) }.problem().is_some());
        assert!(BenchPose { name: "a,b".into(), ..pose([0.0; 3], [1.0; 3]) }.problem().is_some());
        assert!(BenchPose { name: String::new(), ..pose([0.0; 3], [1.0; 3]) }.problem().is_some());
        assert!(pose([0.0; 3], [0.0; 3]).problem().is_some(), "a view with no direction");
        assert!(pose([f32::NAN, 0.0, 0.0], [1.0; 3]).problem().is_some());
    }
}
