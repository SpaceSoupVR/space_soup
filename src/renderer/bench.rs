use glam::{Quat, Vec3};
use serde::{Deserialize, Serialize};

pub const TRACKED_EYE_HEIGHT: f32 = 1.6;

pub const SWAY_PERIOD_SECONDS: f32 = 4.0;

pub const MAX_SWAY: f32 = 1.0;

pub const FALLBACK_IPD: f32 = 0.063;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BenchPose {

    pub name: String,

    pub eye: [f32; 3],

    pub at: [f32; 3],

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rig_yaw: Option<f32>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sway: Option<f32>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub flashlight: Option<BenchFlashlight>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BenchFlashlight {

    pub at: [f32; 3],

    pub aim: [f32; 3],

    #[serde(default)]
    pub torch: bool,
}

impl BenchPose {

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
        if self.rig_yaw.is_some_and(|y| !y.is_finite()) {
            return Some(format!("bench '{}' has a rig_yaw that is not a number", self.name));
        }
        if self.sway.is_some_and(|s| !(0.0..=MAX_SWAY).contains(&s)) {
            return Some(format!("bench '{}' sways by {:?} m: 0 to {MAX_SWAY} m", self.name, self.sway));
        }
        if let Some(f) = &self.flashlight {
            if !f.at.iter().chain(f.aim.iter()).all(|v| v.is_finite()) {
                return Some(format!("bench '{}' has a flashlight coordinate that is not a number", self.name));
            }
            if (Vec3::from(f.aim) - Vec3::from(f.at)).length_squared() < 1e-6 {
                return Some(format!("bench '{}' aims its flashlight at its own glass", self.name));
            }
        }
        None
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BenchRig {

    pub offset: Vec3,

    pub yaw: f32,

    pub head_position: Vec3,

    pub head_rotation: Quat,
}

impl BenchRig {
    pub fn for_pose(pose: &BenchPose) -> Self {
        Self::for_pose_at(pose, 0.0)
    }

    pub fn for_pose_at(pose: &BenchPose, seconds: f32) -> Self {
        let eye = Vec3::from(pose.eye) + sway_offset(pose, seconds);
        let dir = (Vec3::from(pose.at) - eye).normalize_or(Vec3::NEG_Z);

        let yaw = if dir.x * dir.x + dir.z * dir.z > 1e-10 { (-dir.x).atan2(-dir.z) } else { 0.0 };

        let pitch = dir.y.clamp(-1.0, 1.0).asin();
        let head_position = Vec3::new(0.0, TRACKED_EYE_HEIGHT, 0.0);

        let rig = pose.rig_yaw.map_or(yaw, f32::to_radians);
        Self {

            offset: eye - head_position,
            yaw: rig,
            head_position,
            head_rotation: Quat::from_rotation_y(yaw - rig) * Quat::from_rotation_x(pitch),
        }
    }
}

fn sway_offset(pose: &BenchPose, seconds: f32) -> Vec3 {
    let Some(amplitude) = pose.sway else { return Vec3::ZERO };
    let dir = Vec3::from(pose.at) - Vec3::from(pose.eye);
    let right = dir.cross(Vec3::Y).try_normalize().unwrap_or(Vec3::X);
    right * amplitude * (std::f32::consts::TAU * seconds / SWAY_PERIOD_SECONDS).sin()
}

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
        BenchPose { name: "t".into(), eye, at, rig_yaw: None, sway: None, flashlight: None }
    }

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

            assert!((q * Vec3::X).y.abs() < 1e-5, "the view is rolled for {eye:?} -> {at:?}");
        }
    }

    #[test]
    fn a_turned_rig_still_stands_at_the_eye_and_looks_at_the_target() {
        let (eye, at) = ([-1.0, 1.6, -14.84], [-1.0, 1.45, -15.7]);
        for turn in [0.0f32, 45.0, -90.0, 180.0] {
            let rig = BenchRig::for_pose(&BenchPose { rig_yaw: Some(turn), ..pose(eye, at) });
            assert!((rig.yaw - turn.to_radians()).abs() < 1e-6, "rig turned {} for {turn}", rig.yaw.to_degrees());
            let (p, q) = to_world(&rig, rig.head_position, rig.head_rotation);
            assert!((p - Vec3::from(eye)).length() < 1e-4, "head at {p} at turn {turn}");
            let want = (Vec3::from(at) - Vec3::from(eye)).normalize();
            assert!((q * Vec3::NEG_Z).dot(want) > 0.99999, "at turn {turn} looking along {}", q * Vec3::NEG_Z);
            assert!((q * Vec3::X).y.abs() < 1e-5, "the view is rolled at turn {turn}");
        }
    }

    #[test]
    fn a_swaying_head_glides_sideways_and_keeps_looking_at_the_target() {
        let (eye, at) = ([0.3, 1.6, -3.0], [0.0, 0.9, -7.0]);
        let swaying = BenchPose { sway: Some(0.1), ..pose(eye, at) };
        let right = (Vec3::from(at) - Vec3::from(eye)).cross(Vec3::Y).normalize();
        for (t, out) in [(0.0, 0.0), (0.25, 0.1), (0.5, 0.0), (0.75, -0.1), (1.0, 0.0)] {
            let rig = BenchRig::for_pose_at(&swaying, t * SWAY_PERIOD_SECONDS);
            let (p, q) = to_world(&rig, rig.head_position, rig.head_rotation);
            let want = Vec3::from(eye) + right * out;
            assert!((p - want).length() < 1e-4, "at {t} of a sway the head is at {p}, not {want}");
            let look = (Vec3::from(at) - p).normalize();
            assert!((q * Vec3::NEG_Z).dot(look) > 0.99999, "at {t} of a sway the head looks away");
            assert!((q * Vec3::X).y.abs() < 1e-5, "the swaying view is rolled at {t}");
        }
        assert_eq!(BenchRig::for_pose_at(&pose(eye, at), 1.3), BenchRig::for_pose(&pose(eye, at)), "a still pose moved");
        assert!(BenchPose { sway: Some(-0.1), ..pose(eye, at) }.problem().is_some());
        assert!(BenchPose { sway: Some(f32::NAN), ..pose(eye, at) }.problem().is_some());
    }

    #[test]
    fn the_eyes_keep_the_headsets_separation_and_turn() {

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

    #[test]
    fn a_bench_flashlight_reads_from_the_lever_file_and_is_checked() {
        let lit: BenchPose = serde_json::from_str(
            r#"{"name": "t", "eye": [0, 1.6, 0], "at": [0, 1, -3],
                "flashlight": {"at": [0.2, 1.2, -0.2], "aim": [0, 0.5, -3], "torch": true}}"#,
        )
        .expect("a view with a flashlight parses");
        let f = lit.flashlight.as_ref().expect("the flashlight is kept");
        assert!(f.torch && f.at == [0.2, 1.2, -0.2]);
        assert!(lit.problem().is_none());
        let no_torch: BenchFlashlight = serde_json::from_str(r#"{"at": [0, 1, 0], "aim": [0, 0, -1]}"#).unwrap();
        assert!(!no_torch.torch, "the torch is drawn only when asked for");
        let pointless = BenchPose {
            flashlight: Some(BenchFlashlight { at: [1.0; 3], aim: [1.0; 3], torch: false }),
            ..pose([0.0; 3], [0.0, 0.0, -1.0])
        };
        assert!(pointless.problem().is_some(), "a flashlight aimed at its own glass");
        assert!(serde_json::from_str::<BenchFlashlight>(r#"{"at": [0, 1, 0], "aim": [0, 0, -1], "beam": 3}"#).is_err());
    }
}
