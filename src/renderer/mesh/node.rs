use glam::{Mat4, Quat, Vec3};
use std::collections::HashMap;
use std::sync::Arc;
use wgpu::util::DeviceExt;

use super::skin::SkinnedMeshPrimitive;
use super::texture::load_primitive_texture;
use super::vertex::{LayeredPrimitive, MeshPrimitive, MeshVertex, SkinnedMeshVertex};
use crate::renderer::layered_mesh_pipeline::LayeredVertex;

pub(crate) fn ancestor_joint_and_baked_local(
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
pub(crate) fn collect_node(
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

            // Layer weights, when the file both asks for layered shading and
            // carries them. Read here rather than in the `bake` arm below
            // because the reader borrows the primitive, and a cave is always
            // static anyway -- nothing has ever rigged one.
            let layered_weights: Option<Vec<[f32; 4]>> = if wants_layered_shading(&mesh) {
                reader.read_colors(0).map(|c| c.into_rgba_f32().collect())
            } else {
                None
            };

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
                let layered = layered_weights.map(|weights| {
                    let lv: Vec<LayeredVertex> = (0..positions.len())
                        .map(|i| LayeredVertex {
                            position: positions[i].into(),
                            normal: normals.get(i).copied().unwrap_or(Vec3::Y).into(),
                            // A vertex past the end of a short COLOR_0 gets
                            // layer 0 rather than nothing: the shader reads
                            // all-zero weights as layer 0 too, so the two
                            // agree instead of one of them rendering black.
                            weights: weights.get(i).copied().unwrap_or([1.0, 0.0, 0.0, 0.0]),
                        })
                        .collect();
                    let vertex_buffer =
                        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("layered_mesh_vb"),
                            contents: bytemuck::cast_slice(&lv),
                            usage: wgpu::BufferUsages::VERTEX,
                        });
                    LayeredPrimitive { vertices: lv, vertex_buffer }
                });

                static_out.push(MeshPrimitive {
                    vertices,
                    indices,
                    texture,
                    vertex_buffer,
                    index_buffer,
                    layered,
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

/// Whether a mesh asked to be shaded from its per-vertex layer weights.
///
/// Declared by the FILE, in `extras`, rather than by the scene object that
/// happens to reference it. A cave bake is a cave bake wherever it is placed,
/// and putting the flag in the scene would mean carrying it through the engine
/// schema, the wire protocol and the client to say something the mesh already
/// knows.
///
/// Not inferred from "has COLOR_0 and no baseColorTexture", which would silently
/// render an artist's vertex-coloured model as cave rock.
fn wants_layered_shading(mesh: &gltf::Mesh) -> bool {
    let Some(raw) = mesh.extras().as_ref() else {
        return false;
    };
    let Ok(v) = serde_json::from_str::<serde_json::Value>(raw.get()) else {
        return false;
    };
    v.get("space_soup")
        .and_then(|s| s.get("shading"))
        .and_then(|s| s.as_str())
        == Some("layered")
}
