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
    pub(crate) fn excluding_joints(&self, device: &wgpu::Device, excluded_joints: &[usize]) -> Option<Self> {
        let indices: Vec<u32> = self
            .indices
            .chunks_exact(3)
            .filter(|tri| {
                !tri.iter()
                    .any(|&vi| excluded_joints.contains(&self.vertices[vi as usize].dominant_joint()))
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

#[derive(Clone)]
pub struct GltfSkin {
    pub joint_names: Vec<String>,

    pub inv_bind_mats: Vec<Mat4>,
    pub joint_parents: Vec<Option<usize>>,
    pub joint_local_bind: Vec<(Vec3, Quat, Vec3)>,
    pub animations: Vec<GltfAnimationPose>,

    pub joint_buffer: wgpu::Buffer,

    pub joint_bind_group: Option<wgpu::BindGroup>,
    pub primitives: Vec<SkinnedMeshPrimitive>,
}

impl GltfSkin {
    pub fn update_joint_matrices(&self, queue: &wgpu::Queue, skinned_mats: &[Mat4]) {
        let mut buf = [[0f32; 16]; MAX_SKIN_JOINTS];
        for (i, mat) in skinned_mats.iter().enumerate().take(MAX_SKIN_JOINTS) {
            buf[i] = mat.to_cols_array();
        }
        queue.write_buffer(&self.joint_buffer, 0, bytemuck::cast_slice(&buf));
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
        self.skin_matrices_blended_multi(&[(clip, blend)])
    }

    pub fn skin_matrices_blended_multi(&self, targets: &[(usize, f32)]) -> Vec<Mat4> {
        let local: Vec<(Vec3, Quat, Vec3)> = (0..self.joint_names.len())
            .map(|ji| {
                let bind = self.joint_local_bind[ji];
                for &(clip, blend) in targets {
                    let Some(target) = self
                        .animations
                        .get(clip)
                        .and_then(|a| a.joint_transforms.get(ji).copied().flatten())
                    else {
                        continue;
                    };
                    let b = blend.clamp(0.0, 1.0);
                    return (bind.0.lerp(target.0, b), bind.1.slerp(target.1, b), bind.2.lerp(target.2, b));
                }
                bind
            })
            .collect();
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
