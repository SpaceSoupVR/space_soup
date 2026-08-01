use anyhow::{Context, Result};
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Quat, Vec3};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct MeshVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
    pub uv2: [f32; 2],
}

impl MeshVertex {
    pub const ATTRIBS: [wgpu::VertexAttribute; 4] =
        wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x2, 3 => Float32x2];

    pub fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct SkinnedMeshVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
    pub joint_ids: [u32; 4],
    pub joint_weights: [f32; 4],
}

impl SkinnedMeshVertex {
    pub const ATTRIBS: [wgpu::VertexAttribute; 5] = wgpu::vertex_attr_array![
        0 => Float32x3,
        1 => Float32x3,
        2 => Float32x2,
        3 => Uint32x4,
        4 => Float32x4,
    ];

    pub fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }

    pub fn dominant_joint(&self) -> usize {
        let mut best = 0;
        for i in 1..4 {
            if self.joint_weights[i] > self.joint_weights[best] {
                best = i;
            }
        }
        self.joint_ids[best] as usize
    }
}

#[derive(Clone)]
pub struct SkinnedMeshPrimitive {
    pub vertices: Vec<SkinnedMeshVertex>,
    pub indices: Vec<u32>,
    pub texture: Arc<LoadedTexture>,
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
}

impl SkinnedMeshPrimitive {
    fn excluding_joints(&self, device: &wgpu::Device, excluded_joints: &[usize]) -> Option<Self> {
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

#[derive(Clone)]
pub struct MeshPrimitive {
    pub vertices: Vec<MeshVertex>,
    pub indices: Vec<u32>,
    pub texture: Arc<LoadedTexture>,
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
}

pub struct LoadedTexture {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    _sampler: wgpu::Sampler,
    pub bind_group: wgpu::BindGroup,
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

    pub fn load(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layout: &wgpu::BindGroupLayout,
        path: &Path,
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
}

fn ancestor_joint_and_baked_local(
    node: &gltf::Node,
    all_nodes: &[gltf::Node],
    parent_of_node: &HashMap<usize, usize>,
    joint_of_node: &HashMap<usize, usize>,
) -> (Option<usize>, Vec3, Quat, Vec3) {
    let (t, r, s) = node.transform().decomposed();
    let mut local =
        Mat4::from_scale_rotation_translation(Vec3::from(s), Quat::from_array(r), Vec3::from(t));

    let parent_joint = parent_of_node
        .get(&node.index())
        .and_then(|pidx| joint_of_node.get(pidx).copied());

    if parent_joint.is_none() {
        let mut ancestor_idx = parent_of_node.get(&node.index()).copied();
        let mut ancestor_mat = Mat4::IDENTITY;
        while let Some(idx) = ancestor_idx {
            if joint_of_node.contains_key(&idx) {
                break;
            }
            let (at, ar, asc) = all_nodes[idx].transform().decomposed();
            let anc_local = Mat4::from_scale_rotation_translation(
                Vec3::from(asc),
                Quat::from_array(ar),
                Vec3::from(at),
            );
            ancestor_mat = anc_local * ancestor_mat;
            ancestor_idx = parent_of_node.get(&idx).copied();
        }
        local = ancestor_mat * local;
    }

    let (s2, r2, t2) = local.to_scale_rotation_translation();
    (parent_joint, t2, r2, s2)
}

#[allow(clippy::too_many_arguments)]
fn collect_node(
    node: &gltf::Node,
    parent: Mat4,
    current_joint: Option<usize>,
    buffers: &[gltf::buffer::Data],
    images: &[gltf::image::Data],
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
    force_static: bool,
    node_to_joint: &HashMap<usize, usize>,
    static_out: &mut Vec<MeshPrimitive>,
    skinned_out: &mut Vec<SkinnedMeshPrimitive>,
) {
    let local = Mat4::from_cols_array_2d(&node.transform().matrix());
    let world = parent * local;

    let this_joint = if force_static { None } else { node_to_joint.get(&node.index()).copied() };
    let real_skin = !force_static && node.skin().is_some();
    let synthetic_joint = if real_skin { None } else { this_joint.or(current_joint) };
    let bake = !(real_skin || synthetic_joint.is_some());

    if let Some(mesh) = node.mesh() {
        for prim in mesh.primitives() {
            let reader = prim.reader(|buf| Some(&buffers[buf.index()]));

            let positions: Vec<Vec3> = match reader.read_positions() {
                Some(p) => {
                    if bake {
                        p.map(|v| world.transform_point3(Vec3::from(v))).collect()
                    } else {
                        p.map(Vec3::from).collect()
                    }
                }
                None => continue,
            };
            if positions.is_empty() {
                continue;
            }

            let normals: Vec<Vec3> = match reader.read_normals() {
                Some(n) => {
                    if bake {
                        n.map(|v| world.transform_vector3(Vec3::from(v)).normalize_or_zero())
                            .collect()
                    } else {
                        n.map(|v| Vec3::from(v).normalize_or_zero()).collect()
                    }
                }
                None => vec![Vec3::Y; positions.len()],
            };

            let uvs: Vec<[f32; 2]> = match reader.read_tex_coords(0) {
                Some(uv) => uv.into_f32().collect(),
                None => vec![[0.0, 0.0]; positions.len()],
            };

            let uv2s: Vec<[f32; 2]> = match reader.read_tex_coords(1) {
                Some(uv) => uv.into_f32().collect(),
                None => vec![[0.0, 0.0]; positions.len()],
            };

            let indices: Vec<u32> = match reader.read_indices() {
                Some(i) => i.into_u32().collect(),
                None => (0..positions.len() as u32).collect(),
            };

            let texture = load_primitive_texture(&prim, images, device, queue, layout);
            let texture = Arc::new(texture);

            if bake {
                let vertices: Vec<MeshVertex> = (0..positions.len())
                    .map(|i| MeshVertex {
                        position: positions[i].into(),
                        normal: normals.get(i).copied().unwrap_or(Vec3::Y).into(),
                        uv: uvs.get(i).copied().unwrap_or([0.0, 0.0]),
                        uv2: uv2s.get(i).copied().unwrap_or([0.0, 0.0]),
                    })
                    .collect();

                let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("mesh_vb"),
                    contents: bytemuck::cast_slice(&vertices),
                    usage: wgpu::BufferUsages::VERTEX,
                });
                let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("mesh_ib"),
                    contents: bytemuck::cast_slice(&indices),
                    usage: wgpu::BufferUsages::INDEX,
                });
                static_out.push(MeshPrimitive {
                    vertices,
                    indices,
                    texture,
                    vertex_buffer,
                    index_buffer,
                });
            } else {
                let (joint_ids, joint_weights): (Vec<[u32; 4]>, Vec<[f32; 4]>) = if real_skin {
                    let joint_ids = match reader.read_joints(0) {
                        Some(gltf::mesh::util::ReadJoints::U8(it)) => it
                            .map(|j| [j[0] as u32, j[1] as u32, j[2] as u32, j[3] as u32])
                            .collect(),
                        Some(gltf::mesh::util::ReadJoints::U16(it)) => it
                            .map(|j| [j[0] as u32, j[1] as u32, j[2] as u32, j[3] as u32])
                            .collect(),
                        None => vec![[0, 0, 0, 0]; positions.len()],
                    };
                    let joint_weights = match reader.read_weights(0) {
                        Some(w) => w.into_f32().collect(),
                        None => vec![[1.0, 0.0, 0.0, 0.0]; positions.len()],
                    };
                    (joint_ids, joint_weights)
                } else {
                    let ji = synthetic_joint.unwrap_or(0) as u32;
                    (
                        vec![[ji, 0, 0, 0]; positions.len()],
                        vec![[1.0, 0.0, 0.0, 0.0]; positions.len()],
                    )
                };

                let vertices: Vec<SkinnedMeshVertex> = (0..positions.len())
                    .map(|i| SkinnedMeshVertex {
                        position: positions[i].into(),
                        normal: normals.get(i).copied().unwrap_or(Vec3::Y).into(),
                        uv: uvs.get(i).copied().unwrap_or([0.0, 0.0]),
                        joint_ids: joint_ids.get(i).copied().unwrap_or([0; 4]),
                        joint_weights: joint_weights.get(i).copied().unwrap_or([1.0, 0.0, 0.0, 0.0]),
                    })
                    .collect();

                let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("skinned_mesh_vb"),
                    contents: bytemuck::cast_slice(&vertices),
                    usage: wgpu::BufferUsages::VERTEX,
                });
                let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("skinned_mesh_ib"),
                    contents: bytemuck::cast_slice(&indices),
                    usage: wgpu::BufferUsages::INDEX,
                });
                skinned_out.push(SkinnedMeshPrimitive {
                    vertices,
                    indices,
                    texture,
                    vertex_buffer,
                    index_buffer,
                });
            }
        }
    }

    let child_parent = if this_joint.is_some() { Mat4::IDENTITY } else { world };
    for child in node.children() {
        collect_node(
            &child,
            child_parent,
            synthetic_joint,
            buffers,
            images,
            device,
            queue,
            layout,
            force_static,
            node_to_joint,
            static_out,
            skinned_out,
        );
    }
}

fn load_primitive_texture(
    prim: &gltf::Primitive,
    images: &[gltf::image::Data],
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
) -> LoadedTexture {
    let material = prim.material();
    let pbr = material.pbr_metallic_roughness();
    let force_opaque = material.alpha_mode() != gltf::material::AlphaMode::Blend;

    if let Some(info) = pbr.base_color_texture() {
        let image = &images[info.texture().source().index()];
        return upload_texture_rgba(device, queue, layout, image, force_opaque);
    }

    let c = pbr.base_color_factor();
    let rgba = [
        (c[0] * 255.0) as u8,
        (c[1] * 255.0) as u8,
        (c[2] * 255.0) as u8,
        if force_opaque { 255 } else { (c[3] * 255.0) as u8 },
    ];
    upload_solid_texture(device, queue, layout, rgba)
}

fn upload_texture_rgba(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
    image: &gltf::image::Data,
    force_opaque: bool,
) -> LoadedTexture {
    use gltf::image::Format;

    let (width, height) = (image.width, image.height);

    let rgba: Vec<u8> = match image.format {
        Format::R8G8B8A8 => {
            if force_opaque {
                let mut out = image.pixels.clone();
                out.chunks_exact_mut(4).for_each(|px| px[3] = 255);
                out
            } else {
                image.pixels.clone()
            }
        }
        Format::R8G8B8 => {
            let mut out = Vec::with_capacity((width * height * 4) as usize);
            for px in image.pixels.chunks_exact(3) {
                out.extend_from_slice(&[px[0], px[1], px[2], 255]);
            }
            out
        }
        Format::R8 => {
            let mut out = Vec::with_capacity((width * height * 4) as usize);
            for &v in &image.pixels {
                out.extend_from_slice(&[v, v, v, 255]);
            }
            out
        }
        _ => {
            log::warn!(
                "Unsupported glTF image format {:?}, using gray fallback",
                image.format
            );
            [180u8, 180, 180, 255].repeat((width * height) as usize)
        }
    };

    create_texture_from_rgba(device, queue, layout, &rgba, width, height)
}

fn upload_solid_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
    rgba: [u8; 4],
) -> LoadedTexture {
    create_texture_from_rgba(device, queue, layout, &rgba, 1, 1)
}

pub fn create_texture_from_rgba(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
    rgba: &[u8],
    width: u32,
    height: u32,
) -> LoadedTexture {
    let size = wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("gltf_texture"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        rgba,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4 * width),
            rows_per_image: Some(height),
        },
        size,
    );

    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("gltf_sampler"),
        address_mode_u: wgpu::AddressMode::Repeat,
        address_mode_v: wgpu::AddressMode::Repeat,
        address_mode_w: wgpu::AddressMode::Repeat,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("gltf_texture_bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
        ],
    });

    LoadedTexture {
        texture,
        view,
        _sampler: sampler,
        bind_group,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn workspace_root() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..")
    }

    #[test]
    fn ancestor_bake_handles_m4a1_wrapper_chain() {
        let path = workspace_root().join("game/models/m4a1.glb");
        let (doc, _buffers, _images) = gltf::import(&path).expect("m4a1.glb should load");

        let all_nodes: Vec<gltf::Node> = doc.nodes().collect();
        let mut parent_of_node: HashMap<usize, usize> = HashMap::new();
        for node in doc.nodes() {
            for child in node.children() {
                parent_of_node.insert(child.index(), node.index());
            }
        }

        let charging_handle = doc
            .nodes()
            .find(|n| n.name() == Some("m4a1 charging handle_10"))
            .expect("m4a1.glb should have a node named 'm4a1 charging handle_10'");

        let joint_of_node: HashMap<usize, usize> = HashMap::new();
        let (parent_joint, t2, r2, s2) = ancestor_joint_and_baked_local(
            &charging_handle,
            &all_nodes,
            &parent_of_node,
            &joint_of_node,
        );
        assert_eq!(parent_joint, None, "no ancestor of this node is itself a joint");

        let (t, r, s) = charging_handle.transform().decomposed();
        let mut expected =
            Mat4::from_scale_rotation_translation(Vec3::from(s), Quat::from_array(r), Vec3::from(t));
        let mut idx = parent_of_node.get(&charging_handle.index()).copied();
        let mut acc = Mat4::IDENTITY;
        while let Some(i) = idx {
            let (at, ar, asc) = all_nodes[i].transform().decomposed();
            acc = Mat4::from_scale_rotation_translation(Vec3::from(asc), Quat::from_array(ar), Vec3::from(at)) * acc;
            idx = parent_of_node.get(&i).copied();
        }
        expected = acc * expected;
        let (es, er, et) = expected.to_scale_rotation_translation();
        assert!(et.distance(t2) < 1e-4, "expected translation {et:?}, got {t2:?}");
        assert!(es.distance(s2) < 1e-4, "expected scale {es:?}, got {s2:?}");
        assert!(er.angle_between(r2) < 1e-3, "expected rotation {er:?}, got {r2:?}");
    }

    #[test]
    fn m4a1_orphan_node_promotion_end_to_end() {
        let Some((device, queue, layout)) = headless_gpu() else {
            eprintln!("skipping: no GPU adapter available in this environment");
            return;
        };

        let original_path = workspace_root().join("game/models/m4a1.glb");

        let baseline = GltfMesh::load(&device, &queue, &layout, &original_path)
            .expect("baseline m4a1.glb should load");
        assert!(baseline.skin.is_none(), "file with no orphan animations must stay unskinned");
        assert!(!baseline.primitives.is_empty(), "baseline should have static primitives");
        assert!(baseline.bounding_radius > 0.0);

        let test_path = std::env::temp_dir().join(format!("m4a1_test_{}.glb", std::process::id()));
        std::fs::copy(&original_path, &test_path).expect("copy fixture");

        let script_dir = workspace_root().join("scene_editor_web");
        let py = format!(
            "import sys; sys.path.insert(0, {script_dir:?}); from gltf_animation import write_joint_animation_clip; \
             from pathlib import Path; \
             write_joint_animation_clip(Path({test_path:?}), 'charging_handle_pull', {{'m4a1 charging handle_10': [ \
             {{'t': 0.0, 'position': [0,0,0], 'rotation': [0,0,0,1], 'scale': [1,1,1]}}, \
             {{'t': 0.3, 'position': [0,0,-0.05], 'rotation': [0,0,0,1], 'scale': [1,1,1]}}]}})",
        );
        let status = std::process::Command::new("python3")
            .arg("-c")
            .arg(&py)
            .status()
            .expect("run python3 to write the test clip");
        assert!(status.success(), "write_joint_animation_clip failed");

        let animated = GltfMesh::load(&device, &queue, &layout, &test_path).expect("animated m4a1 should load");
        std::fs::remove_file(&test_path).ok();
        let bin_glob = test_path.with_file_name(format!(
            "{}_anim_charging_handle_pull.bin",
            test_path.file_stem().unwrap().to_string_lossy()
        ));
        std::fs::remove_file(&bin_glob).ok();

        assert!(animated.primitives.is_empty(), "all geometry must be routed through the skin pipeline once any node is promoted");
        let skin = animated.skin.expect("orphan animation should produce a skin");
        assert!(skin.joint_names.len() >= 19, "expected at least the 19 named m4a1 parts as joints, got {}", skin.joint_names.len());
        assert!(!skin.primitives.is_empty(), "no geometry should be silently dropped");
        assert!(animated.bounding_radius > 0.0, "bounding_radius must not regress to 0 for a fully-skinned mesh");

        let ji = skin
            .joint_names
            .iter()
            .position(|n| n == "m4a1 charging handle_10")
            .expect("charging handle should be a joint");
        let pose = &skin.animations[0];
        assert_eq!(pose.name, "charging_handle_pull");
        let (t, _r, _s) = pose.joint_transforms[ji].expect("charging handle should have a sampled pose");
        assert!(
            t.distance(Vec3::new(0.0, 0.0, -0.05)) < 1e-4,
            "expected end-keyframe (pulled) position (0,0,-0.05), got {t:?}"
        );

        let at_rest = skin.skin_matrices_blended(0, 0.0);
        let at_full = skin.skin_matrices_blended(0, 1.0);
        let rest_off = at_rest[ji].w_axis.truncate();
        let full_off = at_full[ji].w_axis.truncate();
        assert!(
            full_off.distance(rest_off) > 0.01,
            "full blend should move the charging handle joint away from rest ({rest_off:?} -> {full_off:?})"
        );

        let filler_count = pose.joint_transforms.iter().filter(|p| p.is_none()).count();
        assert!(filler_count >= 18, "every other joint should be an inert filler with no animation entry, got {filler_count} inert of {}", skin.joint_names.len());
    }

    fn headless_gpu() -> Option<(wgpu::Device, wgpu::Queue, wgpu::BindGroupLayout)> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok()?;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            ..Default::default()
        }))
        .ok()?;
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("test_mesh_texture_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        Some((device, queue, layout))
    }

    #[test]
    fn real_skinned_fixtures_still_load_correctly() {
        let Some((device, queue, layout)) = headless_gpu() else {
            eprintln!("skipping: no GPU adapter available in this environment");
            return;
        };
        let root = workspace_root();
        for rel in [
            "game/models/left_hand.glb",
            "game/models/right_hand.glb",
            "game/models/boy/boy.glb",
            "game/models/ar15/870022.gltf",
        ] {
            let path = root.join(rel);
            let mesh = GltfMesh::load(&device, &queue, &layout, &path)
                .unwrap_or_else(|e| panic!("{rel} should still load: {e}"));
            let skin = mesh.skin.as_ref().unwrap_or_else(|| panic!("{rel} should be skinned"));
            assert!(!skin.joint_names.is_empty(), "{rel}: expected real joints");
            assert!(!skin.primitives.is_empty(), "{rel}: expected skinned primitives, none dropped");
            assert!(mesh.primitives.is_empty(), "{rel}: fully-skinned fixture should have zero static primitives");
            assert!(mesh.bounding_radius > 0.0, "{rel}: bounding_radius should not be zero");
            println!(
                "{rel}: {} joints, {} skinned primitives, radius {:.3}",
                skin.joint_names.len(),
                skin.primitives.len(),
                mesh.bounding_radius
            );
        }
    }
}

