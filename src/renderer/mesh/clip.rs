//! Keyframed skeletal animation clips: a glTF animation kept as its actual
//! keys (not collapsed to one pose like `GltfAnimationPose`), sampled at any
//! time -- a character's idle, walk, attack.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use glam::{Quat, Vec3};

/// A joint's local transform: translation, rotation, scale.
pub type Trs = (Vec3, Quat, Vec3);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Interp {
    Linear,
    Step,
}

#[derive(Clone, Debug)]
pub struct Track<T> {
    pub times: Vec<f32>,
    pub values: Vec<T>,
    pub interp: Interp,
}

impl<T: Copy> Track<T> {
    /// The value at `t`: before the first key the first, after the last
    /// the last, in between `mix`ed (or held, for Step).
    fn sample(&self, t: f32, mix: impl Fn(T, T, f32) -> T) -> Option<T> {
        let n = self.times.len().min(self.values.len());
        if n == 0 {
            return None;
        }
        let i = self.times[..n].partition_point(|k| *k <= t);
        if i == 0 {
            return Some(self.values[0]);
        }
        if i >= n {
            return Some(self.values[n - 1]);
        }
        let (a, b) = (self.times[i - 1], self.times[i]);
        if self.interp == Interp::Step || b <= a {
            return Some(self.values[i - 1]);
        }
        Some(mix(self.values[i - 1], self.values[i], (t - a) / (b - a)))
    }
}

/// One joint's keys in a clip (any of them may be absent: that part stays
/// as the pose under it).
#[derive(Clone, Debug, Default)]
pub struct JointTrack {
    pub translation: Option<Track<Vec3>>,
    pub rotation: Option<Track<Quat>>,
    pub scale: Option<Track<Vec3>>,
}

#[derive(Clone, Debug)]
pub struct GltfClip {
    pub name: String,
    pub duration: f32,
    /// Per joint of the skin it plays on (same order); empty = not animated.
    pub joints: Vec<JointTrack>,
}

impl GltfClip {
    /// Every joint's local transform at `t` seconds (clamped to the clip),
    /// starting from `base` for whatever the clip doesn't key.
    pub fn sample(&self, base: &[Trs], t: f32) -> Vec<Trs> {
        base.iter()
            .enumerate()
            .map(|(j, &(bt, br, bs))| match self.joints.get(j) {
                None => (bt, br, bs),
                Some(tr) => (
                    tr.translation.as_ref().and_then(|k| k.sample(t, |a, b, f| a.lerp(b, f))).unwrap_or(bt),
                    tr.rotation.as_ref().and_then(|k| k.sample(t, |a, b, f| a.slerp(b, f))).unwrap_or(br),
                    tr.scale.as_ref().and_then(|k| k.sample(t, |a, b, f| a.lerp(b, f))).unwrap_or(bs),
                ),
            })
            .collect()
    }

    /// Does this clip key joint `j` at all?
    pub fn animates(&self, j: usize) -> bool {
        self.joints.get(j).is_some_and(|t| t.translation.is_some() || t.rotation.is_some() || t.scale.is_some())
    }
}

/// Read one glTF animation into a clip for a skin of `joint_count` joints;
/// `joint_of` maps a node index to its joint (None = not on this skin).
#[allow(dead_code)]
pub(super) fn read_clip(anim: &gltf::Animation, buffers: &[gltf::buffer::Data], joint_count: usize, joint_of: impl Fn(usize) -> Option<usize>) -> GltfClip {
    let mut joints = vec![JointTrack::default(); joint_count];
    let mut duration = 0.0f32;
    for channel in anim.channels() {
        let Some(j) = joint_of(channel.target().node().index()) else { continue };
        let reader = channel.reader(|b| Some(&buffers[b.index()]));
        let Some(times) = reader.read_inputs().map(|t| t.collect::<Vec<f32>>()) else { continue };
        duration = duration.max(times.last().copied().unwrap_or(0.0));
        let interp = channel.sampler().interpolation();
        let cubic = interp == gltf::animation::Interpolation::CubicSpline;
        let interp = if interp == gltf::animation::Interpolation::Step { Interp::Step } else { Interp::Linear };
        // Cubic splines store (in-tangent, value, out-tangent) per key: keep
        // the values.
        fn values<T: Copy>(all: Vec<T>, cubic: bool) -> Vec<T> {
            if cubic { all.chunks(3).filter_map(|c| c.get(1).copied()).collect() } else { all }
        }
        match reader.read_outputs() {
            Some(gltf::animation::util::ReadOutputs::Translations(t)) => {
                joints[j].translation = Some(Track { times: times.clone(), values: values(t.map(Vec3::from).collect(), cubic), interp });
            }
            Some(gltf::animation::util::ReadOutputs::Rotations(r)) => {
                let v = r.into_f32().map(|q| Quat::from_xyzw(q[0], q[1], q[2], q[3]).normalize()).collect();
                joints[j].rotation = Some(Track { times: times.clone(), values: values(v, cubic), interp });
            }
            Some(gltf::animation::util::ReadOutputs::Scales(s)) => {
                joints[j].scale = Some(Track { times: times.clone(), values: values(s.map(Vec3::from).collect(), cubic), interp });
            }
            _ => {}
        }
    }
    GltfClip { name: anim.name().unwrap_or("").to_string(), duration, joints }
}

/// A bone name without an exporter's trailing number ("Hips_06" -> "hips").
fn plain_name(name: &str) -> String {
    let n = match name.rsplit_once('_') {
        Some((head, tail)) if !head.is_empty() && tail.chars().all(|c| c.is_ascii_digit()) => head,
        _ => name,
    };
    n.to_ascii_lowercase()
}

/// The clips in another file, for a skin with these joints (matched by
/// bone name, ignoring case and exporter numbering). Only rotations carry
/// over, plus translations of bones the same length in both (so a clip made
/// for a bigger or smaller rig doesn't stretch this one).
pub fn load_clips_for(path: &Path, joint_names: &[String], local_bind: &[Trs]) -> Result<Vec<GltfClip>> {
    let (doc, buffers, _) = gltf::import(path).with_context(|| format!("failed to open {}", path.display()))?;
    let exact: HashMap<&str, usize> = joint_names.iter().enumerate().map(|(i, n)| (n.as_str(), i)).collect();
    let plain: HashMap<String, usize> = joint_names.iter().enumerate().map(|(i, n)| (plain_name(n), i)).collect();
    let map: HashMap<usize, usize> = doc
        .nodes()
        .filter_map(|n| {
            let name = n.name()?;
            exact.get(name).copied().or_else(|| plain.get(&plain_name(name)).copied()).map(|j| (n.index(), j))
        })
        .collect();
    let same_length: HashMap<usize, bool> = doc
        .nodes()
        .filter_map(|n| {
            let j = *map.get(&n.index())?;
            let (t, _, _) = n.transform().decomposed();
            let (mine, theirs) = (local_bind.get(j)?.0.length(), Vec3::from(t).length());
            Some((j, (mine - theirs).abs() <= mine.max(theirs) * 0.15 + 1e-4))
        })
        .collect();
    let clips = doc
        .animations()
        .map(|anim| {
            let mut clip = read_clip(&anim, &buffers, joint_names.len(), |node| map.get(&node).copied());
            for (j, tr) in clip.joints.iter_mut().enumerate() {
                tr.scale = None;
                if !same_length.get(&j).copied().unwrap_or(false) {
                    tr.translation = None;
                }
            }
            clip
        })
        .collect();
    Ok(clips)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn samples_between_keys_and_clamps() {
        let tr = JointTrack {
            translation: Some(Track { times: vec![0.0, 1.0], values: vec![Vec3::ZERO, Vec3::X * 2.0], interp: Interp::Linear }),
            rotation: None,
            scale: None,
        };
        let clip = GltfClip { name: "a".into(), duration: 1.0, joints: vec![tr] };
        let base = [(Vec3::Y, Quat::IDENTITY, Vec3::ONE)];
        assert!((clip.sample(&base, 0.5)[0].0 - Vec3::X).length() < 1e-5);
        assert!((clip.sample(&base, 5.0)[0].0 - Vec3::X * 2.0).length() < 1e-5);
        assert!((clip.sample(&base, -1.0)[0].0).length() < 1e-5);
    }

    #[test]
    fn plain_names_drop_exporter_numbers() {
        assert_eq!(plain_name("Hips_06"), "hips");
        assert_eq!(plain_name("Thigh.R_0145"), "thigh.r");
        assert_eq!(plain_name("Spine"), "spine");
    }
}
