use glam::{Mat4, Quat, Vec3};
use std::sync::Arc;
use wgpu::util::DeviceExt;

use super::texture::LoadedTexture;
use super::vertex::SkinnedMeshVertex;

#[derive(Clone)]
pub struct SkinnedMeshPrimitive {
    pub vertices: Vec<SkinnedMeshVertex>,
    pub indices: Vec<u32>,
    pub texture: Arc<LoadedTexture>,
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
}

impl SkinnedMeshPrimitive {
    /// Drop triangles belonging to `excluded_joints`, or sitting above
    /// `cutoff_height` in the bind pose.
    ///
    /// WHY A HEIGHT CUTOFF AND NOT JUST JOINTS
    ///
    /// Joint exclusion drops a triangle when a vertex's DOMINANT joint is in
    /// the set. That cannot remove the top of the neck, because those vertices
    /// are often weighted mostly to Chest or Spine -- joints that must stay,
    /// since they carry the torso the wearer is supposed to see when they look
    /// down. So hiding Head, its descendants and Neck still leaves an open tube
    /// of neck geometry whose rim sits near eye level, and the wearer looks
    /// straight into the inside of their own throat.
    ///
    /// No set of joints fixes that; the geometry simply is not partitioned the
    /// way the joints are. A plane is. Everything above the collar goes,
    /// whatever it happens to be weighted to.
    ///
    /// `bind_positions` are the vertices already through their bind-pose joint
    /// matrices, so the cutoff is a height on the MODEL, in model units, and
    /// does not move when the head turns.
    pub(crate) fn excluding_joints_and_above(
        &self,
        device: &wgpu::Device,
        excluded_joints: &[usize],
        bind_positions: &[Vec3],
        up: Vec3,
        cutoff_height: Option<f32>,
    ) -> Option<Self> {
        let above = |vi: u32| match cutoff_height {
            Some(c) => bind_positions
                .get(vi as usize)
                .is_some_and(|p| p.dot(up) > c),
            None => false,
        };
        let indices: Vec<u32> = self
            .indices
            .chunks_exact(3)
            .filter(|tri| {
                !tri.iter().any(|&vi| {
                    excluded_joints.contains(&self.vertices[vi as usize].dominant_joint())
                        || above(vi)
                })
            })
            .flatten()
            .copied()
            .collect();
        if indices.is_empty() {
            return None;
        }
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("skinned_mesh_ib_excluding_joints"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        Some(Self {
            vertices: self.vertices.clone(),
            index_buffer,
            indices,
            texture: self.texture.clone(),
            vertex_buffer: self.vertex_buffer.clone(),
        })
    }
}

pub const MAX_SKIN_JOINTS: usize = 96;

#[derive(Clone)]
pub struct GltfAnimationPose {
    pub name: String,
    pub joint_transforms: Vec<Option<(Vec3, Quat, Vec3)>>,
}

/// A blend at or below this contributes nothing, so the clip is not holding the
/// joint and must not shadow a later one. Well above f32 noise on a controller
/// axis, well below any blend a player could mean.
pub const IDLE_BLEND: f32 = 1e-4;

/// How a clip combines with the ones before it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ClipBlendMode {
    /// Takes the joint outright. The first active Override clip wins it, and
    /// later Override clips on that joint are ignored.
    #[default]
    Override,
    /// Adds its displacement from the bind pose on top of whatever the Override
    /// layer produced. This is what lets recoil ride a cycling bolt: the cycle
    /// says where the bolt is, recoil nudges it from there.
    Additive,
}

/// Pose every joint from several clips at once, in local space.
///
/// Highest-priority ACTIVE clip wins each joint: `targets` arrives in priority
/// order (the scene's `part_animations` order), and the first clip that both
/// drives this joint and is actually blended in takes it.
///
/// The idle check is the load-bearing part. Without it a clip claimed its joints
/// at *any* blend, including 0.0 -- returning `lerp(bind, target, 0)`, the bind
/// pose -- and returned early, so no later clip was ever consulted. A part
/// appearing in two clips was therefore pinned at rest by whichever clip came
/// first in the array, permanently and with nothing logged: add a `fire_cycle`
/// that moves the bolt while `charging_handle` already does, and the bolt simply
/// never moves. Authoring one part into several clips is normal and supported --
/// different actions need different combinations of the same parts -- so a clip
/// that is not contributing has to step aside.
///
/// Free-standing rather than a method because it is pure math over pose data;
/// GltfSkin owns a wgpu::Buffer, and requiring a GPU to test a lerp is how this
/// went unnoticed.
///
/// Clips are applied in two passes: the Override layer decides where a joint is,
/// then Additive layers nudge it from there. See ClipBlendMode.
pub fn blend_joint_local(
    joint_local_bind: &[(Vec3, Quat, Vec3)],
    animations: &[GltfAnimationPose],
    targets: &[(usize, f32, ClipBlendMode)],
) -> Vec<(Vec3, Quat, Vec3)> {
    (0..joint_local_bind.len())
        .map(|ji| {
            let bind = joint_local_bind[ji];
            let target_for = |clip: usize| {
                animations
                    .get(clip)
                    .and_then(|a| a.joint_transforms.get(ji).copied().flatten())
            };

            // Base layer: the first active Override clip that drives this joint.
            let mut pose = bind;
            for &(clip, blend, mode) in targets {
                let b = blend.clamp(0.0, 1.0);
                if b <= IDLE_BLEND || mode != ClipBlendMode::Override {
                    continue;
                }
                let Some(target) = target_for(clip) else { continue };
                pose = (
                    bind.0.lerp(target.0, b),
                    bind.1.slerp(target.1, b),
                    bind.2.lerp(target.2, b),
                );
                break;
            }

            // Additive layers accumulate their offset FROM BIND on top of that.
            // Offsets rather than absolute poses, so two additive clips compose
            // instead of the last one erasing the others -- and so an additive
            // clip means the same thing regardless of what the base is doing.
            for &(clip, blend, mode) in targets {
                let b = blend.clamp(0.0, 1.0);
                if b <= IDLE_BLEND || mode != ClipBlendMode::Additive {
                    continue;
                }
                let Some(target) = target_for(clip) else { continue };
                pose.0 += (target.0 - bind.0) * b;
                // Rotation composes rather than adds: the delta from bind, scaled
                // by the blend, applied to what we have.
                let delta = target.1 * bind.1.inverse();
                pose.1 = Quat::IDENTITY.slerp(delta, b) * pose.1;
                // Scale is a ratio, so it multiplies. A bind scale component of 0
                // has no meaningful ratio, so that axis is left alone.
                for i in 0..3 {
                    if bind.2[i].abs() > f32::EPSILON {
                        let ratio = target.2[i] / bind.2[i];
                        pose.2[i] *= 1.0 + (ratio - 1.0) * b;
                    }
                }
            }
            pose
        })
        .collect()
}

#[derive(Clone)]
pub struct GltfSkin {
    pub joint_names: Vec<String>,

    pub inv_bind_mats: Vec<Mat4>,
    pub joint_parents: Vec<Option<usize>>,
    pub joint_local_bind: Vec<(Vec3, Quat, Vec3)>,
    pub animations: Vec<GltfAnimationPose>,

    pub joint_buffer: wgpu::Buffer,
    /// The joint matrices of the update before the last: the pose this skin
    /// had a frame ago, for SpaceWarp's motion vectors (`space_warp`).
    pub prev_joint_buffer: wgpu::Buffer,
    /// What `joint_buffer` holds now, so the next update can move it into
    /// `prev_joint_buffer`. Shared by clones, as the buffers are.
    pub last_joints: std::sync::Arc<std::sync::Mutex<Option<Vec<[f32; 16]>>>>,
    /// The motion pass's bind group (both palettes), made on first use.
    pub motion_bind_group: std::sync::Arc<std::sync::OnceLock<wgpu::BindGroup>>,

    pub joint_bind_group: Option<wgpu::BindGroup>,
    pub primitives: Vec<SkinnedMeshPrimitive>,

    /// Total height of the skinned mesh in its bind pose, in model units.
    ///
    /// The character's STATURE -- sole to crown -- and the only measurement on
    /// a humanoid rig that means the same thing on every model. Joint positions
    /// do not: "Head" sits at the skull base on one rig and mid-skull on
    /// another, and a rig may carry no eye joints at all, so anything derived
    /// from a single joint is a guess about that rig's conventions.
    ///
    /// Exists so the first-person eye position can be placed anthropometrically
    /// when the rig does not say where the eyes are. Measured along the model's
    /// own up axis at load, from the same bind-pose vertex walk that produces
    /// `bounding_radius`, so it costs nothing extra.
    pub bind_stature: f32,
}

impl GltfSkin {
    pub fn update_joint_matrices(&self, queue: &wgpu::Queue, skinned_mats: &[Mat4]) {
        let mut buf = [[0f32; 16]; MAX_SKIN_JOINTS];
        for (i, mat) in skinned_mats.iter().enumerate().take(MAX_SKIN_JOINTS) {
            buf[i] = mat.to_cols_array();
        }
        // The pose it replaces becomes the previous one -- the first update
        // has none, and is its own.
        let mut last = self.last_joints.lock().unwrap_or_else(|e| e.into_inner());
        let prev: &[[f32; 16]] = last.as_deref().unwrap_or(&buf);
        queue.write_buffer(&self.prev_joint_buffer, 0, bytemuck::cast_slice(prev));
        queue.write_buffer(&self.joint_buffer, 0, bytemuck::cast_slice(&buf));
        *last = Some(buf.to_vec());
    }

    /// Both palettes, as the motion pass binds them (`space_warp`).
    pub fn motion_bind_group(&self, device: &wgpu::Device, layout: &wgpu::BindGroupLayout) -> &wgpu::BindGroup {
        self.motion_bind_group.get_or_init(|| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("skin_motion_joints"),
                layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.joint_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: self.prev_joint_buffer.as_entire_binding() },
                ],
            })
        })
    }

    /// A joint buffer the size every skin's is.
    pub fn joint_buffer(device: &wgpu::Device, label: &str) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: (MAX_SKIN_JOINTS * 64) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    pub fn blended_local_pose(
        &self,
        from: usize,
        to: usize,
        blend: impl Fn(usize) -> f32,
    ) -> Vec<(Vec3, Quat, Vec3)> {
        let get = |clip: usize, ji: usize| -> (Vec3, Quat, Vec3) {
            self.animations
                .get(clip)
                .and_then(|a| a.joint_transforms.get(ji).copied().flatten())
                .unwrap_or(self.joint_local_bind[ji])
        };
        (0..self.joint_names.len())
            .map(|ji| {
                let (ft, fr, fs) = get(from, ji);
                let (tt, tr, ts) = get(to, ji);
                let b = blend(ji).clamp(0.0, 1.0);
                (ft.lerp(tt, b), fr.slerp(tr, b), fs.lerp(ts, b))
            })
            .collect()
    }

    pub fn generic_joint_name(name: &str) -> &str {
        name.rsplit_once("_r_")
            .or_else(|| name.rsplit_once("_l_"))
            .map(|(_, suffix)| suffix)
            .unwrap_or(name)
    }

    pub fn skin_matrices_blended(&self, clip: usize, blend: f32) -> Vec<Mat4> {
        self.skin_matrices_blended_multi(&[(clip, blend, ClipBlendMode::Override)])
    }

    pub fn skin_matrices_blended_multi(&self, targets: &[(usize, f32, ClipBlendMode)]) -> Vec<Mat4> {
        let local = blend_joint_local(&self.joint_local_bind, &self.animations, targets);
        let world = self.hierarchical_transforms(&local);
        self.inv_bind_mats
            .iter()
            .enumerate()
            .map(|(ji, inv_bind)| world[ji] * *inv_bind)
            .collect()
    }

    pub fn animation_index(&self, name: &str) -> Option<usize> {
        self.animations.iter().position(|a| a.name == name)
    }

    pub fn pull_geometry(&self, clip: usize) -> Option<(Vec3, Vec3, f32)> {
        let rest = self.hierarchical_transforms(&self.joint_local_bind);
        let posed_local = self.blended_local_pose(usize::MAX, clip, |_| 1.0);
        let posed = self.hierarchical_transforms(&posed_local);
        let mut best: Option<(usize, f32)> = None;
        for ji in 0..self.joint_names.len() {
            let d = (posed[ji].w_axis.truncate() - rest[ji].w_axis.truncate()).length();
            if d > best.map(|(_, bd)| bd).unwrap_or(1e-4) {
                best = Some((ji, d));
            }
        }
        let (ji, travel) = best?;
        let rest_p = rest[ji].w_axis.truncate();
        let posed_p = posed[ji].w_axis.truncate();
        Some((rest_p, (posed_p - rest_p) / travel, travel))
    }

    pub fn hierarchical_transforms(&self, local: &[(Vec3, Quat, Vec3)]) -> Vec<Mat4> {
        let mut out = vec![Mat4::IDENTITY; local.len()];
        for ji in 0..local.len() {
            let (t, r, s) = local[ji];
            let local_mat = Mat4::from_scale_rotation_translation(s, r, t);
            out[ji] = match self.joint_parents[ji] {
                Some(pi) => out[pi] * local_mat,
                None => local_mat,
            };
        }
        out
    }
}
