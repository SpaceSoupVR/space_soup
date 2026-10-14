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

/// How long one full sway takes, side to side and back. See `BenchPose::sway`.
pub const SWAY_PERIOD_SECONDS: f32 = 4.0;

/// The widest sway a pose may ask for, metres each way.
pub const MAX_SWAY: f32 = 1.0;

/// A walking pace, metres a second. See `BenchPose::path`.
pub const WALK_SPEED: f32 = 1.4;

/// How long a walk's turn-round at each end takes.
pub const TURN_SECONDS: f32 = 2.0;

/// How far ahead along its way a walking eye looks, metres.
pub const LOOK_AHEAD: f32 = 1.0;

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
    /// The rig's turn, degrees, where it should not simply face `at`: the
    /// same view reached with the rig turned -- as a snap turn leaves it --
    /// and the head turned back by the difference. Absent: the rig faces the
    /// view, as it always has. Everything drawn in the player's frame turns
    /// with the rig, so the same view at two turns isolates a frame bug.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rig_yaw: Option<f32>,
    /// A head that sways, metres each way: the eye glides along the view's
    /// own level right axis and back, once every [`SWAY_PERIOD_SECONDS`],
    /// still looking at `at` -- the drift of a head held nearly still, for
    /// watching what moves with it: reflections on polished stone, and the
    /// frames SpaceWarp makes between the rendered ones, which a still view
    /// never shows (the user, 2026-10-01: reflections still jitter). Absent:
    /// still.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sway: Option<f32>,
    /// A WALK: the eye carried from `eye` through these points, world metres,
    /// at [`Self::speed`], looking a metre ahead along its way; at the last it
    /// turns round on the spot ([`TURN_SECONDS`]) and walks back, and again at
    /// `eye`. The rig goes with it, so every frame is a new offset and, at a
    /// bend or an end, a new yaw: what a player walking does to everything
    /// the app poses for the rig. `at` is then only the still picture's.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<Vec<[f32; 3]>>,
    /// Metres a second along `path`. Absent: [`WALK_SPEED`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speed: Option<f32>,
    /// A TURN ON THE SPOT: the view turned about the vertical at `eye`, this
    /// many degrees a second (a held stick turns the rig so). Its pitch is
    /// the still view's.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spin: Option<f32>,
    /// The player's flashlight, held still for this view: its glass at `at`,
    /// aimed at `aim`, world metres, the torch drawn with it when `torch`.
    /// Absent: none, whatever the player last switched on -- a benchmark
    /// measures what it names. See quest_app's `flashlight`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub flashlight: Option<BenchFlashlight>,
    /// THE DOORS for this view, degrees from shut, in the scene's order (as
    /// `DOORS=a,b,c,d` in the offline harness): held at these angles, or with
    /// [`Self::door_swing`] swung from shut to them and back. Absent: as the
    /// player and the server left them. See [`Self::door_angles_at`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub doors: Option<Vec<f32>>,
    /// Seconds for each door to swing from shut to its [`Self::doors`] angle
    /// and back, smoothly (a player opening a door and letting it close):
    /// a door that MOVES every frame, against the same angles held still.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub door_swing: Option<f32>,
}

/// A flashlight a bench view holds still. See [`BenchPose::flashlight`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BenchFlashlight {
    /// Where the glass is, world metres.
    pub at: [f32; 3],
    /// What the beam is aimed at, world metres.
    pub aim: [f32; 3],
    /// Whether the torch itself is drawn, as well as its light.
    #[serde(default)]
    pub torch: bool,
}

impl BenchPose {
    /// The doors' angles, degrees, `t` seconds into the view; none when it
    /// names none. See [`Self::doors`].
    pub fn door_angles_at(&self, t: f32) -> Option<Vec<f32>> {
        let doors = self.doors.as_ref()?;
        let k = match self.door_swing.filter(|s| *s > 0.0) {
            Some(period) => 0.5 - 0.5 * (std::f32::consts::TAU * t / period).cos(),
            None => 1.0,
        };
        Some(doors.iter().map(|a| a * k).collect())
    }

    /// Whether the rig is somewhere else from one frame to the next.
    pub fn moves(&self) -> bool {
        self.sway.is_some() || self.path.is_some() || self.spin.is_some()
    }

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

/// A `BenchPose` taken apart into what the app and the renderer each set.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BenchRig {
    /// The rig's origin in the world: `Locomotion::player_offset`.
    pub offset: Vec3,
    /// The rig's turn about the vertical: `Locomotion::player_yaw`.
    pub yaw: f32,
    /// The tracked head, in the rig's own (stage) space.
    pub head_position: Vec3,
    /// Its orientation there: pitch only, since the yaw is the rig's -- or,
    /// with `BenchPose::rig_yaw`, the rest of the view's turn and the pitch.
    pub head_rotation: Quat,
}

impl BenchRig {
    pub fn for_pose(pose: &BenchPose) -> Self {
        Self::for_pose_at(pose, 0.0)
    }

    /// The rig `seconds` into a swaying pose's sway (`BenchPose::sway`); a
    /// still pose's at any time.
    pub fn for_pose_at(pose: &BenchPose, seconds: f32) -> Self {
        let (eye, dir) = view_at(pose, seconds);
        // `from_rotation_y(yaw) * -Z` is `(-sin yaw, 0, -cos yaw)`. Straight up
        // or down has no heading at all; any yaw is then right, so zero.
        let yaw = if dir.x * dir.x + dir.z * dir.z > 1e-10 { (-dir.x).atan2(-dir.z) } else { 0.0 };
        // `from_rotation_x(pitch) * -Z` is `(0, sin pitch, -cos pitch)`.
        let pitch = dir.y.clamp(-1.0, 1.0).asin();
        let head_position = Vec3::new(0.0, TRACKED_EYE_HEIGHT, 0.0);
        // A rig turned otherwise: the head turns back by the difference, so the
        // eyes still face `at`.
        let rig = pose.rig_yaw.map_or(yaw, f32::to_radians);
        Self {
            // The yaw turns about the vertical, which leaves the head's height
            // where it is: the head lands exactly on `eye`.
            offset: eye - head_position,
            yaw: rig,
            head_position,
            head_rotation: Quat::from_rotation_y(yaw - rig) * Quat::from_rotation_x(pitch),
        }
    }
}

/// The legs of a pose's walk, each from one point to the next, `eye` first.
fn walk_legs(pose: &BenchPose) -> Vec<(Vec3, Vec3)> {
    let mut points = vec![Vec3::from(pose.eye)];
    points.extend(pose.path.iter().flatten().map(|p| Vec3::from(*p)));
    points.windows(2).map(|w| (w[0], w[1])).collect()
}

/// Where along its legs a walker `metres` from the start is, and the leg's
/// heading there; held at either end.
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

/// A pose's eye and where it looks, `seconds` in: the still view's; a
/// swaying one's (`BenchPose::sway`); a walker's along its path
/// (`BenchPose::path`); or the still view turned (`BenchPose::spin`).
pub fn view_at(pose: &BenchPose, seconds: f32) -> (Vec3, Vec3) {
    let still_eye = Vec3::from(pose.eye);
    let still = (Vec3::from(pose.at) - still_eye).normalize_or(Vec3::NEG_Z);
    if pose.path.is_some() {
        let legs = walk_legs(pose);
        let length: f32 = legs.iter().map(|(a, b)| (*b - *a).length()).sum();
        let speed = pose.speed.unwrap_or(WALK_SPEED);
        let walk = length / speed;
        // Out, turn, back, turn.
        let t = seconds.max(0.0) % (2.0 * (walk + TURN_SECONDS));
        let (out, t) = if t < walk + TURN_SECONDS { (true, t) } else { (false, t - walk - TURN_SECONDS) };
        // Looking a metre ahead along the way rounds each bend into a turn.
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
        // Turning round at the end it has reached, the short way by yaw.
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

/// Where a swaying pose's eye has glided to, `seconds` in: along the level
/// right axis of its unswayed view.
fn sway_offset(pose: &BenchPose, seconds: f32) -> Vec3 {
    let Some(amplitude) = pose.sway else { return Vec3::ZERO };
    let dir = Vec3::from(pose.at) - Vec3::from(pose.eye);
    let right = dir.cross(Vec3::Y).try_normalize().unwrap_or(Vec3::X);
    right * amplitude * (std::f32::consts::TAU * seconds / SWAY_PERIOD_SECONDS).sin()
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

    /// A view's doors held open, or swung shut-open-shut over its period.
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
        BenchPose { name: "t".into(), eye, at, rig_yaw: None, sway: None, path: None, speed: None, spin: None, flashlight: None, doors: None, door_swing: None }
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

    /// A rig turned as a snap turn leaves it sees the same view: same eye,
    /// same target, no roll -- only the player's frame has turned.
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

    /// A swaying head glides side to side along the view's right axis, a
    /// full sway every period, and keeps looking at the target, level; still
    /// at the start, half way and at the end of each sway.
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

    /// A walker goes from its eye through its points at its speed, looking
    /// along its way; turns round at the end; comes back; and the rig is
    /// under it all the way, never jumping.
    #[test]
    fn a_walker_goes_out_and_back_along_its_path_and_the_rig_goes_with_it() {
        let walker = BenchPose { path: Some(vec![[10.0, 1.6, 0.0], [10.0, 2.6, 10.0]]), speed: Some(2.0), ..pose([0.0, 1.6, 0.0], [5.0, 1.6, 0.0]) };
        assert_eq!(walker.problem(), None);
        assert!(walker.moves() && !pose([0.0; 3], [1.0; 3]).moves());
        // 20.05 m of path at 2 m/s.
        let length = 10.0 + (101.0f32).sqrt();
        let walk = length / 2.0;
        let (eye, dir) = view_at(&walker, 0.0);
        assert!((eye - Vec3::new(0.0, 1.6, 0.0)).length() < 1e-5 && (dir - Vec3::X).length() < 1e-4, "{eye} {dir}");
        // Half-way down the first leg, heading along it.
        let (eye, dir) = view_at(&walker, 2.5);
        assert!((eye - Vec3::new(5.0, 1.6, 0.0)).length() < 1e-4 && (dir - Vec3::X).length() < 1e-4);
        // On the second leg, climbing, the view level along it.
        let (eye, dir) = view_at(&walker, 5.0 + 2.5);
        assert!(eye.x == 10.0 && eye.y > 1.6 && eye.z > 4.0 && (dir - Vec3::Z).length() < 1e-4, "{eye} {dir}");
        // At the end, turned half-way round after half the turn.
        let (eye, dir) = view_at(&walker, walk + TURN_SECONDS * 0.5);
        assert!((eye - Vec3::new(10.0, 2.6, 10.0)).length() < 1e-3 && dir.z.abs() < 1e-3 && dir.y == 0.0, "{eye} {dir}");
        // Walking back, facing back.
        let (eye, dir) = view_at(&walker, walk + TURN_SECONDS + 1.0);
        assert!(eye.z < 10.0 && eye.z > 7.5 && (dir + Vec3::Z).length() < 1e-3, "{eye} {dir}");
        // A whole cycle on, where it began.
        let (eye, _) = view_at(&walker, 2.0 * (walk + TURN_SECONDS));
        assert!((eye - Vec3::new(0.0, 1.6, 0.0)).length() < 1e-2);
        // Never a jump: a 90th of a second moves the head under 3 cm and
        // turns the view under 3 degrees, the whole way round.
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
        // The rig stands under the head and faces its way.
        let rig = BenchRig::for_pose_at(&walker, 2.5);
        assert!((rig.offset - Vec3::new(5.0, 0.0, 0.0)).length() < 1e-4);
        assert!((Quat::from_rotation_y(rig.yaw) * Vec3::NEG_Z - Vec3::X).length() < 1e-4);
        // What cannot be walked says so.
        assert!(BenchPose { path: Some(vec![]), ..pose([0.0; 3], [1.0; 3]) }.problem().is_some());
        assert!(BenchPose { path: Some(vec![[0.0; 3]]), ..pose([0.0; 3], [1.0; 3]) }.problem().is_some());
        assert!(BenchPose { path: Some(vec![[1.0, 0.0, 0.0]]), speed: Some(0.0), ..pose([0.0; 3], [1.0; 3]) }.problem().is_some());
    }

    /// A spin turns the still view about the vertical where it stands.
    #[test]
    fn a_spin_turns_the_view_on_the_spot() {
        let spinner = BenchPose { spin: Some(90.0), ..pose([3.0, 1.6, 4.0], [3.0, 0.6, 3.0]) };
        assert_eq!(spinner.problem(), None);
        assert!(spinner.moves());
        let (eye0, dir0) = view_at(&spinner, 0.0);
        let (eye1, dir1) = view_at(&spinner, 1.0);
        assert_eq!(eye0, eye1);
        assert!((dir0.y - dir1.y).abs() < 1e-6, "its pitch is kept");
        // A quarter turn to the left (counter-clockwise from above) in a second.
        let flat = |d: Vec3| Vec3::new(d.x, 0.0, d.z).normalize();
        assert!((flat(dir1) - Quat::from_rotation_y(std::f32::consts::FRAC_PI_2) * flat(dir0)).length() < 1e-4);
        let (a, b) = (BenchRig::for_pose_at(&spinner, 0.0), BenchRig::for_pose_at(&spinner, 1.0));
        assert_eq!(a.offset, b.offset);
        assert!(((b.yaw - a.yaw).rem_euclid(std::f32::consts::TAU) - std::f32::consts::FRAC_PI_2).abs() < 1e-4);
        assert!(BenchPose { spin: Some(f32::NAN), ..pose([0.0; 3], [1.0; 3]) }.problem().is_some());
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
