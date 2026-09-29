use anyhow::{Context, Result};
use glam::{Mat4, Quat, Vec3};
use std::collections::HashMap;
use std::path::Path;

mod node;
mod skin;
mod texture;
mod vertex;

pub use skin::{blend_joint_local, ClipBlendMode, GltfAnimationPose, GltfSkin, SkinnedMeshPrimitive, MAX_SKIN_JOINTS};
pub use texture::{
    create_lightmap_texture, create_lightmap_texture_full, create_lightmap_texture_with_sun, create_mesh_material_texture, LightmapLight,
    create_texture_from_rgba, LoadedTexture,
    LIGHTMAP_MIP_LEVELS, NEUTRAL_BOUNCE_DIRECTION, NEUTRAL_SUN_MASK, SUN_MASK_MIP_LEVELS,
};
pub use vertex::{MeshPrimitive, MeshVertex, SkinnedMeshVertex};

use node::{ancestor_joint_and_baked_local, collect_node};

/// A second UV set for a mesh's baked lighting, supplied by the caller.
///
/// WHY THE CALLER AND NOT THIS CRATE
///
/// Deciding where a triangle's lighting lives in an atlas is a decision the
/// BAKER has to make identically, and the baker is not in this crate -- this
/// one is a standalone renderer and does not know what a scene is. So the
/// layout lives with the game engine that owns both ends of it, exactly as the
/// brush atlas already does, and what arrives here is only the answer: three
/// uv2 per triangle. Nothing in this file knows what a chart is.
///
/// `None`, or a primitive with no entry, is the ordinary case: the mesh keeps
/// whatever uv2 it was authored with and binds whatever lightmap it was given.
#[derive(Debug, Default, Clone)]
pub struct MeshLightmapUv {
    /// Keyed by (glTF node index, primitive index within that node's mesh).
    ///
    /// Each value holds one uv2 per triangle CORNER, in the primitive's own
    /// index order -- so its length is the primitive's index count. Keyed by
    /// node index rather than by traversal order because that is a property of
    /// the file, and the producer of this map walked the document separately.
    pub per_primitive: HashMap<(usize, usize), Vec<[f32; 2]>>,
}

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
                joint_buffer: GltfSkin::joint_buffer(device, "skin_joint_buf"),
                prev_joint_buffer: GltfSkin::joint_buffer(device, "skin_prev_joint_buf"),
                last_joints: Default::default(),
                motion_bind_group: Default::default(),
                joint_bind_group: None,
                primitives: skin.primitives.clone(),
                bind_stature: skin.bind_stature,
            });
        }
        m
    }

    pub fn clone_with_independent_skin_excluding_joints(
        &self,
        device: &wgpu::Device,
        excluded_joints: &[usize],
    ) -> Self {
        self.clone_with_independent_skin_excluding(device, excluded_joints, Vec3::Y, None)
    }

    /// As above, and additionally dropping everything above `cutoff_height`.
    ///
    /// The cutoff is what removes the wearer's own neck. Joint exclusion alone
    /// cannot: neck vertices are typically weighted mostly to Chest or Spine,
    /// which have to stay because they are the torso the wearer looks down at.
    /// See `SkinnedMeshPrimitive::excluding_joints_and_above`.
    pub fn clone_with_independent_skin_excluding(
        &self,
        device: &wgpu::Device,
        excluded_joints: &[usize],
        up: Vec3,
        cutoff_height: Option<f32>,
    ) -> Self {
        let mut m = self.clone_with_independent_skin(device);
        if let Some(skin) = &mut m.skin {
            // Bind-pose positions, so the cutoff is a fixed height on the model
            // and does not swing about as the head turns.
            let bind_transforms = skin.hierarchical_transforms(&skin.joint_local_bind);
            let bind_pose_mats: Vec<Mat4> = skin
                .inv_bind_mats
                .iter()
                .enumerate()
                .map(|(ji, inv_bind)| bind_transforms[ji] * *inv_bind)
                .collect();
            let before: usize = skin.primitives.iter().map(|p| p.indices.len()).sum();
            skin.primitives = skin
                .primitives
                .iter()
                .filter_map(|prim| {
                    let bind_positions: Vec<Vec3> = prim
                        .vertices
                        .iter()
                        .map(|v| {
                            bind_pose_mats
                                .get(v.dominant_joint())
                                .copied()
                                .unwrap_or(Mat4::IDENTITY)
                                .transform_point3(Vec3::from(v.position))
                        })
                        .collect();
                    prim.excluding_joints_and_above(
                        device,
                        excluded_joints,
                        &bind_positions,
                        up,
                        cutoff_height,
                    )
                })
                .collect();
            let after: usize = skin.primitives.iter().map(|p| p.indices.len()).sum();
            // The one number that distinguishes "the cull is wrong" from "the
            // cull did nothing". Every previous theory about the wearer's own
            // body was argued from the geometry rather than measured, and each
            // was wrong in a way this line would have shown immediately.
            log::info!(
                "AVATARCULL joints={:?} cutoff={:?} along {:?}: {} -> {} indices ({} tris removed)",
                excluded_joints.len(),
                cutoff_height,
                up,
                before,
                after,
                (before - after) / 3,
            );
        }
        m
    }

    pub fn load(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layout: &wgpu::BindGroupLayout,
        path: &Path,
    ) -> Result<Self> {
        Self::load_with_lightmap_uv(device, queue, layout, path, None)
    }

    /// As [`Self::load`], with a caller-supplied lightmap UV set.
    ///
    /// A primitive that has an entry in `lightmap_uv` is DE-INDEXED: adjacent
    /// triangles land in unrelated parts of the atlas, so a vertex shared
    /// between them has no single uv2 -- every edge is a chart seam. That costs
    /// three vertices per triangle instead of roughly one, which is why the
    /// caller decides which meshes are worth it rather than this crate deciding
    /// for every mesh it ever loads.
    pub fn load_with_lightmap_uv(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layout: &wgpu::BindGroupLayout,
        path: &Path,
        lightmap: Option<&MeshLightmapUv>,
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
                    &node_index_to_joint,
                    lightmap,
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
                joint_buffer,
                prev_joint_buffer: GltfSkin::joint_buffer(device, "skin_prev_joint_buf"),
                last_joints: Default::default(),
                motion_bind_group: Default::default(),
                joint_bind_group: None,
                primitives: Vec::new(),
                // Filled in by the bind-pose vertex walk below, which cannot
                // run until the primitives are attached.
                bind_stature: 0.0,
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

            // One walk, two answers: the radius the culler wants and the
            // stature the first-person eye placement wants.
            let mut lo = f32::INFINITY;
            let mut hi = f32::NEG_INFINITY;
            let skinned_radius = skin
                .primitives
                .iter()
                .flat_map(|p| p.vertices.iter())
                .map(|v| {
                    let world = bind_pose_mats
                        .get(v.dominant_joint())
                        .copied()
                        .unwrap_or(Mat4::IDENTITY);
                    let p = world.transform_point3(Vec3::from(v.position));
                    // Along the MODEL's up axis, which for a glTF skin is +Y by
                    // specification -- a Z-up authoring tool has to bake its own
                    // correction into the export, so the file is always Y-up by
                    // the time it reaches here.
                    lo = lo.min(p.y);
                    hi = hi.max(p.y);
                    p.length()
                })
                .fold(0.0_f32, f32::max);
            skin.bind_stature = if hi > lo { hi - lo } else { 0.0 };
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
                    &empty_node_to_joint,
                    None,
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
}

#[cfg(test)]
pub(crate) mod tests {
    include!("mesh/tests.rs");
}
