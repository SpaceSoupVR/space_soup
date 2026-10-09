use anyhow::{Context, Result};
use glam::{Mat4, Quat, Vec3};
use std::collections::HashMap;
use std::path::Path;

mod clip;
mod node;
mod skin;
mod texture;
mod vertex;

pub use clip::{load_clips_for, GltfClip, Interp, JointTrack, Track, Trs};
pub use skin::{blend_joint_local, ClipBlendMode, GltfAnimationPose, GltfSkin, SkinnedMeshPrimitive, MAX_SKIN_JOINTS};
pub use texture::{create_texture_from_rgba, LoadedTexture};
pub use vertex::{MeshPrimitive, MeshVertex, SkinnedMeshVertex};

use node::{ancestor_joint_and_baked_local, collect_node};

#[derive(Clone)]
pub struct GltfMesh {
    pub primitives: Vec<MeshPrimitive>,

    pub skin: Option<GltfSkin>,
    pub position: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
    pub bounding_radius: f32,
}

impl GltfMesh {
    pub fn model_matrix(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.position)
    }

    pub fn is_skinned(&self) -> bool {
        self.skin.is_some()
    }

    pub fn joint_names(&self) -> &[String] {
        self.skin
            .as_ref()
            .map(|s| s.joint_names.as_slice())
            .unwrap_or(&[])
    }

    pub fn create_skin_bind_group(
        &mut self,
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
    ) {
        if let Some(skin) = &mut self.skin {
            skin.joint_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("skin_joints_bg"),
                layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: skin.joint_buffer.as_entire_binding(),
                }],
            }));
        }
    }

    pub fn update_joint_matrices(&self, queue: &wgpu::Queue, skinned_mats: &[Mat4]) {
        if let Some(skin) = &self.skin {
            skin.update_joint_matrices(queue, skinned_mats);
        }
    }

    pub fn clone_with_independent_skin(&self, device: &wgpu::Device) -> Self {
        let mut m = self.clone();
        if let Some(skin) = &self.skin {
            m.skin = Some(GltfSkin {
                joint_names: skin.joint_names.clone(),
                inv_bind_mats: skin.inv_bind_mats.clone(),
                joint_parents: skin.joint_parents.clone(),
                joint_local_bind: skin.joint_local_bind.clone(),
                animations: skin.animations.clone(),
                clips: skin.clips.clone(),
                joint_buffer: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("skin_joint_buf"),
                    size: (MAX_SKIN_JOINTS * 64) as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
                joint_bind_group: None,
                primitives: skin.primitives.clone(),
            });
        }
        m
    }

    pub fn clone_with_independent_skin_excluding_joints(
        &self,
        device: &wgpu::Device,
        excluded_joints: &[usize],
    ) -> Self {
        let mut m = self.clone_with_independent_skin(device);
        if let Some(skin) = &mut m.skin {
            skin.primitives = skin
                .primitives
                .iter()
                .filter_map(|prim| prim.excluding_joints(device, excluded_joints))
                .collect();
        }
        m
    }

    /// A single textured quad in the XZ-facing plane (+Z normal), centered on
    /// the origin — the mesh a world-space UI panel renders as. `u`/`v` map
    /// left→right / top→bottom of `texture`.
    pub fn textured_quad(
        device: &wgpu::Device,
        texture: std::sync::Arc<LoadedTexture>,
        width_m: f32,
        height_m: f32,
    ) -> Self {
        use wgpu::util::DeviceExt;
        let (hw, hh) = (width_m * 0.5, height_m * 0.5);
        let n = [0.0, 0.0, 1.0];
        let vertices = vec![
            MeshVertex { position: [-hw, hh, 0.0], normal: n, uv: [0.0, 0.0], uv2: [0.0, 0.0] },
            MeshVertex { position: [hw, hh, 0.0], normal: n, uv: [1.0, 0.0], uv2: [0.0, 0.0] },
            MeshVertex { position: [hw, -hh, 0.0], normal: n, uv: [1.0, 1.0], uv2: [0.0, 0.0] },
            MeshVertex { position: [-hw, -hh, 0.0], normal: n, uv: [0.0, 1.0], uv2: [0.0, 0.0] },
        ];
        let indices: Vec<u32> = vec![0, 2, 1, 0, 3, 2];
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("panel_quad_vb"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("panel_quad_ib"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        Self {
            primitives: vec![MeshPrimitive {
                vertices,
                indices,
                texture,
                vertex_buffer,
                index_buffer,
            }],
            skin: None,
            position: glam::Vec3::ZERO,
            rotation: glam::Quat::IDENTITY,
            scale: glam::Vec3::ONE,
            bounding_radius: (hw * hw + hh * hh).sqrt(),
        }
    }

    pub fn load(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layout: &wgpu::BindGroupLayout,
        path: &Path,
    ) -> Result<Self> {
        Self::load_impl(device, queue, layout, path, false)
    }

    /// Like [`Self::load`], but every material renders opaque — for avatars
    /// and other closed meshes whose BLEND-mode materials would otherwise
    /// read as see-through.
    pub fn load_all_opaque(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layout: &wgpu::BindGroupLayout,
        path: &Path,
    ) -> Result<Self> {
        Self::load_impl(device, queue, layout, path, true)
    }

    fn load_impl(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layout: &wgpu::BindGroupLayout,
        path: &Path,
        force_all_opaque: bool,
    ) -> Result<Self> {
        let (doc, buffers, images) =
            gltf::import(path).with_context(|| format!("failed to open {}", path.display()))?;

        let all_nodes: Vec<gltf::Node> = doc.nodes().collect();
        let mut parent_of_node: HashMap<usize, usize> = HashMap::new();
        for node in doc.nodes() {
            for child in node.children() {
                parent_of_node.insert(child.index(), node.index());
            }
        }

        let mut joint_names: Vec<String> = Vec::new();
        let mut inv_bind_mats: Vec<Mat4> = Vec::new();
        let mut joint_parents: Vec<Option<usize>> = Vec::new();
        let mut joint_local_bind: Vec<(Vec3, Quat, Vec3)> = Vec::new();
        let mut node_index_to_joint: HashMap<usize, usize> = HashMap::new();

        for skin in doc.skins() {
            let joint_nodes: Vec<gltf::Node> = skin.joints().collect();
            let start = joint_names.len();
            joint_names.extend(joint_nodes.iter().map(|j| j.name().unwrap_or("").to_string()));
            for (ji, node) in joint_nodes.iter().enumerate() {
                node_index_to_joint.insert(node.index(), start + ji);
            }

            let skin_inv_binds: Vec<Mat4> = if let Some(acc) = skin.inverse_bind_matrices() {
                let view = acc.view().context("skin ibm: no buffer view")?;
                let buf_data: &[u8] = &buffers[view.buffer().index()];
                let bstart = view.offset() + acc.offset();
                let stride = view.stride().unwrap_or(64);
                (0..acc.count())
                    .map(|i| {
                        let off = bstart + i * stride;
                        let arr: [f32; 16] = bytemuck::pod_read_unaligned(&buf_data[off..off + 64]);
                        Mat4::from_cols_array(&arr)
                    })
                    .collect()
            } else {
                vec![Mat4::IDENTITY; joint_nodes.len()]
            };
            inv_bind_mats.extend(skin_inv_binds);

            for node in &joint_nodes {
                let (parent_joint, t, r, s) =
                    ancestor_joint_and_baked_local(node, &all_nodes, &parent_of_node, &node_index_to_joint);
                joint_parents.push(parent_joint);
                joint_local_bind.push((t, r, s));
            }
            break;
        }

        let mut orphan_target_nodes: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
        for anim in doc.animations() {
            for channel in anim.channels() {
                let node_idx = channel.target().node().index();
                if !node_index_to_joint.contains_key(&node_idx) {
                    orphan_target_nodes.insert(node_idx);
                }
            }
        }

        if !orphan_target_nodes.is_empty() {
            let mut synthetic_nodes = orphan_target_nodes.clone();
            for node in &all_nodes {
                if node.mesh().is_none() || node.skin().is_some() {
                    continue;
                }
                let mut idx = Some(node.index());
                let mut covered = false;
                while let Some(i) = idx {
                    if node_index_to_joint.contains_key(&i) || synthetic_nodes.contains(&i) {
                        covered = true;
                        break;
                    }
                    idx = parent_of_node.get(&i).copied();
                }
                if !covered {
                    synthetic_nodes.insert(node.index());
                }
            }

            for &node_idx in &synthetic_nodes {
                let node = &all_nodes[node_idx];
                let ji = joint_names.len();
                joint_names.push(node.name().unwrap_or("").to_string());
                inv_bind_mats.push(Mat4::IDENTITY);
                node_index_to_joint.insert(node_idx, ji);
            }
            for &node_idx in &synthetic_nodes {
                let node = &all_nodes[node_idx];
                let (parent_joint, t, r, s) =
                    ancestor_joint_and_baked_local(node, &all_nodes, &parent_of_node, &node_index_to_joint);
                joint_parents.push(parent_joint);
                joint_local_bind.push((t, r, s));
            }

            if joint_names.len() > MAX_SKIN_JOINTS {
                log::warn!(
                    "GltfMesh: {} has {} joints (real + synthetic), exceeding MAX_SKIN_JOINTS={} -- extra joints will not animate correctly",
                    path.display(),
                    joint_names.len(),
                    MAX_SKIN_JOINTS,
                );
            }
        }

        let joint_count = joint_names.len();
        let animations: Vec<GltfAnimationPose> = if joint_count == 0 {
            Vec::new()
        } else {
            doc.animations()
                .map(|anim| {
                    let mut partial: Vec<Option<(Option<Vec3>, Option<Quat>, Option<Vec3>)>> =
                        vec![None; joint_count];
                    for channel in anim.channels() {
                        let Some(&ji) = node_index_to_joint.get(&channel.target().node().index())
                        else {
                            continue;
                        };
                        let reader = channel.reader(|b| Some(&buffers[b.index()]));
                        let entry = partial[ji].get_or_insert((None, None, None));
                        match reader.read_outputs() {
                            Some(gltf::animation::util::ReadOutputs::Translations(t)) => {
                                if let Some(v) = t.last() {
                                    entry.0 = Some(Vec3::from(v));
                                }
                            }
                            Some(gltf::animation::util::ReadOutputs::Rotations(r)) => {
                                if let Some(v) = r.into_f32().last() {
                                    entry.1 = Some(Quat::from_xyzw(v[0], v[1], v[2], v[3]));
                                }
                            }
                            Some(gltf::animation::util::ReadOutputs::Scales(s)) => {
                                if let Some(v) = s.last() {
                                    entry.2 = Some(Vec3::from(v));
                                }
                            }
                            _ => {}
                        }
                    }
                    let joint_transforms = partial
                        .into_iter()
                        .enumerate()
                        .map(|(ji, entry)| {
                            entry.map(|(t, r, s)| {
                                (
                                    t.unwrap_or(joint_local_bind[ji].0),
                                    r.unwrap_or(joint_local_bind[ji].1),
                                    s.unwrap_or(joint_local_bind[ji].2),
                                )
                            })
                        })
                        .collect();
                    GltfAnimationPose {
                        name: anim.name().unwrap_or("").to_string(),
                        joint_transforms,
                    }
                })
                .collect()
        };

        let clips: Vec<GltfClip> = if joint_count == 0 {
            Vec::new()
        } else {
            doc.animations().map(|anim| clip::read_clip(&anim, &buffers, joint_count, |n| node_index_to_joint.get(&n).copied())).collect()
        };

        let mut static_prims: Vec<MeshPrimitive> = Vec::new();
        let mut skinned_prims: Vec<SkinnedMeshPrimitive> = Vec::new();

        for scene in doc.scenes() {
            for node in scene.nodes() {
                collect_node(
                    &node,
                    Mat4::IDENTITY,
                    None,
                    &buffers,
                    &images,
                    device,
                    queue,
                    layout,
                    false,
                    force_all_opaque,
                    &node_index_to_joint,
                    &mut static_prims,
                    &mut skinned_prims,
                );
            }
        }

        if static_prims.is_empty() && skinned_prims.is_empty() {
            anyhow::bail!("no renderable primitives found in {}", path.display());
        }

        log::info!(
            "GltfMesh: loaded {} static + {} skinned primitives from {} ({} joints: {:?})",
            static_prims.len(),
            skinned_prims.len(),
            path.display(),
            joint_count,
            &joint_names,
        );

        let mut skin_opt: Option<GltfSkin> = if joint_count == 0 {
            None
        } else {
            let joint_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("skin_joint_buf"),
                size: (MAX_SKIN_JOINTS * 64) as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            Some(GltfSkin {
                joint_names,
                inv_bind_mats,
                joint_parents,
                joint_local_bind,
                animations,
                clips: std::sync::Arc::new(clips),
                joint_buffer,
                joint_bind_group: None,
                primitives: Vec::new(),
            })
        };

        let bounding_radius = if let Some(skin) = &mut skin_opt {
            skin.primitives = skinned_prims;
            let bind_transforms = skin.hierarchical_transforms(&skin.joint_local_bind);
            let bind_pose_mats: Vec<Mat4> = skin
                .inv_bind_mats
                .iter()
                .enumerate()
                .map(|(ji, inv_bind)| bind_transforms[ji] * *inv_bind)
                .collect();
            skin.update_joint_matrices(queue, &bind_pose_mats);

            let skinned_radius = skin
                .primitives
                .iter()
                .flat_map(|p| p.vertices.iter())
                .map(|v| {
                    let world = bind_pose_mats
                        .get(v.dominant_joint())
                        .copied()
                        .unwrap_or(Mat4::IDENTITY);
                    world.transform_point3(Vec3::from(v.position)).length()
                })
                .fold(0.0_f32, f32::max);
            let static_radius = static_prims
                .iter()
                .flat_map(|p| p.vertices.iter())
                .map(|v| Vec3::from(v.position).length())
                .fold(0.0_f32, f32::max);
            skinned_radius.max(static_radius)
        } else {
            static_prims
                .iter()
                .flat_map(|p| p.vertices.iter())
                .map(|v| Vec3::from(v.position).length())
                .fold(0.0_f32, f32::max)
        };

        Ok(Self {
            primitives: static_prims,
            skin: skin_opt,
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
            bounding_radius,
        })
    }

    pub fn load_static_bind_pose(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layout: &wgpu::BindGroupLayout,
        path: &Path,
    ) -> Result<Self> {
        let (doc, buffers, images) =
            gltf::import(path).with_context(|| format!("failed to open {}", path.display()))?;

        let mut static_prims: Vec<MeshPrimitive> = Vec::new();
        let mut skinned_prims: Vec<SkinnedMeshPrimitive> = Vec::new();
        let empty_node_to_joint = HashMap::new();

        for scene in doc.scenes() {
            for node in scene.nodes() {
                collect_node(
                    &node,
                    Mat4::IDENTITY,
                    None,
                    &buffers,
                    &images,
                    device,
                    queue,
                    layout,
                    true,
                    false,
                    &empty_node_to_joint,
                    &mut static_prims,
                    &mut skinned_prims,
                );
            }
        }

        if static_prims.is_empty() {
            anyhow::bail!("no renderable primitives found in {}", path.display());
        }

        log::info!(
            "GltfMesh: loaded {} static primitives (bind pose, skin ignored) from {}",
            static_prims.len(),
            path.display(),
        );

        let bounding_radius = static_prims
            .iter()
            .flat_map(|p| p.vertices.iter())
            .map(|v| Vec3::from(v.position).length())
            .fold(0.0_f32, f32::max);

        Ok(Self {
            primitives: static_prims,
            skin: None,
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
            bounding_radius,
        })
    }

    /// Like [`Self::load_all_opaque`], but keeps every mesh-bearing node's
    /// geometry un-baked and independently addressable instead of flattening
    /// the whole scene into one static mesh -- for props (weapons, tools...)
    /// whose own sub-parts (a slide, a magazine, a trigger) need to move
    /// independently later.
    ///
    /// Implemented by treating every mesh-bearing node as its own synthetic
    /// skin joint with an identity inverse-bind matrix (the same mechanism
    /// `load_impl` already uses for orphan animation-target nodes, just
    /// applied to every mesh node instead of only those) -- so the returned
    /// mesh is skinned, its vertices stay in each node's own local space, and
    /// animating a part is just writing that one joint's world matrix via
    /// the regular `update_joint_matrices` -- no new rendering path needed.
    /// Because each joint's inverse bind is identity, a "skin matrix" for
    /// joint `i` *is* that joint's world transform, nothing more to compose.
    ///
    /// Returns the mesh plus each joint's (name, bind-pose world transform)
    /// in joint order, matching `update_joint_matrices`'s own joint order --
    /// the caller composes `root * bind_world * own_animated_offset` per
    /// part and passes the results straight through.
    pub fn load_parts(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layout: &wgpu::BindGroupLayout,
        path: &Path,
    ) -> Result<(Self, Vec<(String, Mat4)>)> {
        let (doc, buffers, images) =
            gltf::import(path).with_context(|| format!("failed to open {}", path.display()))?;

        let all_nodes: Vec<gltf::Node> = doc.nodes().collect();
        let mut parent_of_node: HashMap<usize, usize> = HashMap::new();
        for node in doc.nodes() {
            for child in node.children() {
                parent_of_node.insert(child.index(), node.index());
            }
        }

        // Every mesh-bearing node becomes its own synthetic joint, in a
        // stable (sorted) order -- registered up front, same two-pass
        // approach `load_impl` uses for orphan animation nodes, so a
        // child's ancestor-joint lookup always finds its parent already
        // mapped regardless of node declaration order.
        let mesh_nodes: std::collections::BTreeSet<usize> = all_nodes
            .iter()
            .filter(|n| n.mesh().is_some())
            .map(|n| n.index())
            .collect();

        let mut joint_names: Vec<String> = Vec::new();
        let mut inv_bind_mats: Vec<Mat4> = Vec::new();
        let mut joint_parents: Vec<Option<usize>> = Vec::new();
        let mut joint_local_bind: Vec<(Vec3, Quat, Vec3)> = Vec::new();
        let mut node_index_to_joint: HashMap<usize, usize> = HashMap::new();

        for &node_idx in &mesh_nodes {
            let node = &all_nodes[node_idx];
            let ji = joint_names.len();
            joint_names.push(node.name().unwrap_or("").to_string());
            inv_bind_mats.push(Mat4::IDENTITY);
            node_index_to_joint.insert(node_idx, ji);
        }
        for &node_idx in &mesh_nodes {
            let node = &all_nodes[node_idx];
            let (parent_joint, t, r, s) =
                ancestor_joint_and_baked_local(node, &all_nodes, &parent_of_node, &node_index_to_joint);
            joint_parents.push(parent_joint);
            joint_local_bind.push((t, r, s));
        }

        if joint_names.len() > MAX_SKIN_JOINTS {
            log::warn!(
                "GltfMesh::load_parts: {} has {} mesh parts, exceeding MAX_SKIN_JOINTS={} -- extra parts will not animate correctly",
                path.display(),
                joint_names.len(),
                MAX_SKIN_JOINTS,
            );
        }

        let mut static_prims: Vec<MeshPrimitive> = Vec::new();
        let mut skinned_prims: Vec<SkinnedMeshPrimitive> = Vec::new();
        for scene in doc.scenes() {
            for node in scene.nodes() {
                collect_node(
                    &node,
                    Mat4::IDENTITY,
                    None,
                    &buffers,
                    &images,
                    device,
                    queue,
                    layout,
                    false,
                    false,
                    &node_index_to_joint,
                    &mut static_prims,
                    &mut skinned_prims,
                );
            }
        }
        // Every mesh node was registered as a synthetic joint above, so
        // `collect_node` never bakes and never falls back to a static
        // primitive -- this would only be empty if the file had no
        // mesh-bearing nodes at all.
        if skinned_prims.is_empty() {
            anyhow::bail!("no renderable mesh-bearing nodes found in {}", path.display());
        }

        log::info!(
            "GltfMesh::load_parts: loaded {} parts from {}: {:?}",
            joint_names.len(),
            path.display(),
            &joint_names,
        );

        let joint_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("skin_joint_buf"),
            size: (MAX_SKIN_JOINTS * 64) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let skin = GltfSkin {
            joint_names: joint_names.clone(),
            inv_bind_mats,
            joint_parents,
            joint_local_bind: joint_local_bind.clone(),
            animations: Vec::new(),
            clips: Default::default(),
            joint_buffer,
            joint_bind_group: None,
            primitives: skinned_prims,
        };

        let bind_world = skin.hierarchical_transforms(&joint_local_bind);
        skin.update_joint_matrices(queue, &bind_world);

        let bounding_radius = skin
            .primitives
            .iter()
            .flat_map(|p| p.vertices.iter())
            .map(|v| {
                let world = bind_world.get(v.dominant_joint()).copied().unwrap_or(Mat4::IDENTITY);
                world.transform_point3(Vec3::from(v.position)).length()
            })
            .fold(0.0_f32, f32::max);

        let mesh = Self {
            primitives: static_prims,
            skin: Some(skin),
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
            bounding_radius,
        };
        let parts = joint_names.into_iter().zip(bind_world).collect();
        Ok((mesh, parts))
    }
}

#[cfg(test)]
mod tests {
    include!("mesh/tests.rs");
}
