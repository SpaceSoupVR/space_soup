use glam::{Quat, Vec3};
use serde::{Deserialize, Serialize};

pub const TRACKED_EYE_HEIGHT: f32 = 1.6;

pub const SWAY_PERIOD_SECONDS: f32 = 4.0;

pub const MAX_SWAY: f32 = 1.0;

pub const WALK_SPEED: f32 = 1.4;

pub const TURN_SECONDS: f32 = 2.0;

pub const LOOK_AHEAD: f32 = 1.0;

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
    pub path: Option<Vec<[f32; 3]>>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speed: Option<f32>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spin: Option<f32>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub flashlight: Option<BenchFlashlight>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub doors: Option<Vec<f32>>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub door_swing: Option<f32>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backward: Option<bool>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pitch: Option<f32>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub body: Option<bool>,
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

    pub fn door_angles_at(&self, t: f32) -> Option<Vec<f32>> {
        let doors = self.doors.as_ref()?;
        let k = match self.door_swing.filter(|s| *s > 0.0) {
            Some(period) => 0.5 - 0.5 * (std::f32::consts::TAU * t / period).cos(),
            None => 1.0,
        };
        Some(doors.iter().map(|a| a * k).collect())
    }

    pub fn moves(&self) -> bool {
        self.sway.is_some() || self.path.is_some() || self.spin.is_some()
    }

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
        if let Some(path) = &self.path {
            if path.is_empty() || !path.iter().flatten().all(|v| v.is_finite()) {
                return Some(format!("bench '{}' walks a path that is empty or has a coordinate that is not a number", self.name));
            }
            if walk_legs(self).iter().any(|(a, b)| (*b - *a).length() < 1e-3) {
                return Some(format!("bench '{}' walks a path with two points in one place", self.name));
            }
        }
        if self.speed.is_some_and(|s| !(s > 0.0 && s <= 10.0)) {
            return Some(format!("bench '{}' walks at {:?} m/s: over 0, to 10", self.name, self.speed));
        }
        if self.spin.is_some_and(|s| !s.is_finite() || s.abs() > 360.0) {
            return Some(format!("bench '{}' spins at {:?} degrees a second: to 360", self.name, self.spin));
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
        let (eye, dir) = view_at(pose, seconds);

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

fn walk_legs(pose: &BenchPose) -> Vec<(Vec3, Vec3)> {
    let mut points = vec![Vec3::from(pose.eye)];
    points.extend(pose.path.iter().flatten().map(|p| Vec3::from(*p)));
    points.windows(2).map(|w| (w[0], w[1])).collect()
}

fn along(legs: &[(Vec3, Vec3)], metres: f32) -> (Vec3, Vec3) {
    let mut left = metres.max(0.0);
    for (a, b) in legs {
        let len = (*b - *a).length();
        if left <= len {
            return (a.lerp(*b, left / len.max(1e-6)), (*b - *a) / len.max(1e-6));
        }
        left -= len;
    }
    let (a, b) = legs[legs.len() - 1];
    (b, (b - a).normalize_or(Vec3::NEG_Z))
}

pub fn view_at(pose: &BenchPose, seconds: f32) -> (Vec3, Vec3) {
    let (eye, dir) = view_at_level(pose, seconds);

    let dir = if pose.backward == Some(true) { Vec3::new(-dir.x, dir.y, -dir.z) } else { dir };
    let dir = match pose.pitch {
        Some(deg) if deg.is_finite() => {
            let level = Vec3::new(dir.x, 0.0, dir.z).normalize_or(Vec3::NEG_Z);
            let p = deg.clamp(-89.0, 89.0).to_radians();
            level * p.cos() + Vec3::Y * p.sin()
        }
        _ => dir,
    };
    (eye, dir)
}

fn view_at_level(pose: &BenchPose, seconds: f32) -> (Vec3, Vec3) {
    let still_eye = Vec3::from(pose.eye);
    let still = (Vec3::from(pose.at) - still_eye).normalize_or(Vec3::NEG_Z);
    if pose.path.is_some() {
        let legs = walk_legs(pose);
        let length: f32 = legs.iter().map(|(a, b)| (*b - *a).length()).sum();
        let speed = pose.speed.unwrap_or(WALK_SPEED);
        let walk = length / speed;

        let t = seconds.max(0.0) % (2.0 * (walk + TURN_SECONDS));
        let (out, t) = if t < walk + TURN_SECONDS { (true, t) } else { (false, t - walk - TURN_SECONDS) };

        let facing = |metres: f32, forward: bool| {
            let (here, heading) = along(&legs, metres);
            let ahead = along(&legs, metres + if forward { LOOK_AHEAD } else { -LOOK_AHEAD }).0;
            let to = ahead - here;
            let level = Vec3::new(to.x, 0.0, to.z);
            let fallback = Vec3::new(heading.x, 0.0, heading.z) * if forward { 1.0 } else { -1.0 };
            (if level.length() > 0.05 { level } else { fallback }).normalize_or(Vec3::NEG_Z)
        };
        if t < walk {
            let metres = if out { t * speed } else { length - t * speed };
            return (along(&legs, metres).0, facing(metres, out));
        }

        let end = if out { length } else { 0.0 };
        let (from, to) = (facing(end, out), facing(end, !out));
        let yaw = |d: Vec3| (-d.x).atan2(-d.z);
        let mut delta = (yaw(to) - yaw(from)).rem_euclid(std::f32::consts::TAU);
        if delta > std::f32::consts::PI + 1e-3 {
            delta -= std::f32::consts::TAU;
        }
        let turned = yaw(from) + delta * (t - walk) / TURN_SECONDS;
        return (along(&legs, end).0, Vec3::new(-turned.sin(), 0.0, -turned.cos()));
    }
    if let Some(spin) = pose.spin {
        let turn = Quat::from_rotation_y(spin.to_radians() * seconds);
        return (still_eye, turn * still);
    }
    let eye = still_eye + sway_offset(pose, seconds);
    (eye, (Vec3::from(pose.at) - eye).normalize_or(Vec3::NEG_Z))
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

    #[test]
    fn a_walk_can_be_walked_backward_looking_down() {
        let fwd = BenchPose { path: Some(vec![[0.0, 1.6, -10.0]]), speed: Some(1.0), ..pose([0.0, 1.6, 0.0], [0.0, 1.6, -5.0]) };
        let back = BenchPose { backward: Some(true), pitch: Some(-60.0), ..fwd.clone() };
        let (e0, d0) = view_at(&fwd, 2.0);
        let (e1, d1) = view_at(&back, 2.0);
        assert!((e0 - e1).length() < 1e-6, "the same place on the walk");
        assert!(d0.z < -0.9 && d1.z > 0.4, "facing back: {d0} {d1}");
        assert!((d1.y - (-60f32).to_radians().sin()).abs() < 1e-5, "pitched down: {d1}");
    }

    #[test]
    fn a_views_doors_are_held_or_swung() {
        let held = BenchPose { doors: Some(vec![80.0, -60.0]), ..pose([0.0; 3], [1.0; 3]) };
        assert_eq!(held.door_angles_at(3.7), Some(vec![80.0, -60.0]));
        let swung = BenchPose { door_swing: Some(4.0), ..held.clone() };
        assert_eq!(swung.door_angles_at(0.0), Some(vec![0.0, 0.0]));
        let half = swung.door_angles_at(2.0).unwrap();
        assert!((half[0] - 80.0).abs() < 1e-3 && (half[1] + 60.0).abs() < 1e-3);
        assert!(pose([0.0; 3], [1.0; 3]).door_angles_at(1.0).is_none());
        let parsed: BenchPose = serde_json::from_str(r#"{"name":"d","eye":[0,1,0],"at":[1,1,0],"doors":[80,80],"door_swing":2.5}"#).unwrap();
        assert_eq!(parsed.door_swing, Some(2.5));
    }

    fn pose(eye: [f32; 3], at: [f32; 3]) -> BenchPose {
        BenchPose { name: "t".into(), eye, at, rig_yaw: None, sway: None, path: None, speed: None, spin: None, flashlight: None, doors: None, door_swing: None, backward: None, pitch: None, body: None }
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
    fn a_walker_goes_out_and_back_along_its_path_and_the_rig_goes_with_it() {
        let walker = BenchPose { path: Some(vec![[10.0, 1.6, 0.0], [10.0, 2.6, 10.0]]), speed: Some(2.0), ..pose([0.0, 1.6, 0.0], [5.0, 1.6, 0.0]) };
        assert_eq!(walker.problem(), None);
        assert!(walker.moves() && !pose([0.0; 3], [1.0; 3]).moves());

        let length = 10.0 + (101.0f32).sqrt();
        let walk = length / 2.0;
        let (eye, dir) = view_at(&walker, 0.0);
        assert!((eye - Vec3::new(0.0, 1.6, 0.0)).length() < 1e-5 && (dir - Vec3::X).length() < 1e-4, "{eye} {dir}");

        let (eye, dir) = view_at(&walker, 2.5);
        assert!((eye - Vec3::new(5.0, 1.6, 0.0)).length() < 1e-4 && (dir - Vec3::X).length() < 1e-4);

        let (eye, dir) = view_at(&walker, 5.0 + 2.5);
        assert!(eye.x == 10.0 && eye.y > 1.6 && eye.z > 4.0 && (dir - Vec3::Z).length() < 1e-4, "{eye} {dir}");

        let (eye, dir) = view_at(&walker, walk + TURN_SECONDS * 0.5);
        assert!((eye - Vec3::new(10.0, 2.6, 10.0)).length() < 1e-3 && dir.z.abs() < 1e-3 && dir.y == 0.0, "{eye} {dir}");

        let (eye, dir) = view_at(&walker, walk + TURN_SECONDS + 1.0);
        assert!(eye.z < 10.0 && eye.z > 7.5 && (dir + Vec3::Z).length() < 1e-3, "{eye} {dir}");

        let (eye, _) = view_at(&walker, 2.0 * (walk + TURN_SECONDS));
        assert!((eye - Vec3::new(0.0, 1.6, 0.0)).length() < 1e-2);

        let mut last = view_at(&walker, 0.0);
        let mut k = 1;
        while (k as f32) / 90.0 < 2.0 * (walk + TURN_SECONDS) {
            let now = view_at(&walker, k as f32 / 90.0);
            assert!((now.0 - last.0).length() < 0.03, "a step of {} m at {k}", (now.0 - last.0).length());
            assert!(now.1.angle_between(last.1).to_degrees() < 3.0, "a turn of {} degrees at {k}", now.1.angle_between(last.1).to_degrees());
            assert!((now.1.length() - 1.0).abs() < 1e-4);
            last = now;
            k += 1;
        }

        let rig = BenchRig::for_pose_at(&walker, 2.5);
        assert!((rig.offset - Vec3::new(5.0, 0.0, 0.0)).length() < 1e-4);
        assert!((Quat::from_rotation_y(rig.yaw) * Vec3::NEG_Z - Vec3::X).length() < 1e-4);

        assert!(BenchPose { path: Some(vec![]), ..pose([0.0; 3], [1.0; 3]) }.problem().is_some());
        assert!(BenchPose { path: Some(vec![[0.0; 3]]), ..pose([0.0; 3], [1.0; 3]) }.problem().is_some());
        assert!(BenchPose { path: Some(vec![[1.0, 0.0, 0.0]]), speed: Some(0.0), ..pose([0.0; 3], [1.0; 3]) }.problem().is_some());
    }

    #[test]
    fn a_spin_turns_the_view_on_the_spot() {
        let spinner = BenchPose { spin: Some(90.0), ..pose([3.0, 1.6, 4.0], [3.0, 0.6, 3.0]) };
        assert_eq!(spinner.problem(), None);
        assert!(spinner.moves());
        let (eye0, dir0) = view_at(&spinner, 0.0);
        let (eye1, dir1) = view_at(&spinner, 1.0);
        assert_eq!(eye0, eye1);
        assert!((dir0.y - dir1.y).abs() < 1e-6, "its pitch is kept");

        let flat = |d: Vec3| Vec3::new(d.x, 0.0, d.z).normalize();
        assert!((flat(dir1) - Quat::from_rotation_y(std::f32::consts::FRAC_PI_2) * flat(dir0)).length() < 1e-4);
        let (a, b) = (BenchRig::for_pose_at(&spinner, 0.0), BenchRig::for_pose_at(&spinner, 1.0));
        assert_eq!(a.offset, b.offset);
        assert!(((b.yaw - a.yaw).rem_euclid(std::f32::consts::TAU) - std::f32::consts::FRAC_PI_2).abs() < 1e-4);
        assert!(BenchPose { spin: Some(f32::NAN), ..pose([0.0; 3], [1.0; 3]) }.problem().is_some());
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
