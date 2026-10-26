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

    lightmap: Option<&super::MeshLightmapUv>,
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

            let layered_weights: Option<Vec<[f32; 4]>> = if wants_layered_shading(&mesh) {
                reader.read_colors(0).map(|c| c.into_rgba_f32().collect())
            } else {
                None
            };

            let layered_aux: Vec<[f32; 4]> = if layered_weights.is_some() {
                reader.read_colors(1).map(|c| c.into_rgba_f32().collect()).unwrap_or_default()
            } else {
                Vec::new()
            };

            let emissive = {
                let m = prim.material();
                let f = m.emissive_factor();

                let strength = 1.0f32;
                MeshVertex::pack_emissive([f[0] * strength, f[1] * strength, f[2] * strength])
            };

            if bake {

                let mut split_corners: Vec<usize> = Vec::new();
                let split = lightmap
                    .and_then(|lm| lm.per_primitive.get(&(node.index(), prim.index())))

                    .filter(|corner_uv| corner_uv.len() == indices.len())
                    .map(|corner_uv| {
                        let mut v = Vec::with_capacity(indices.len());
                        let mut from = Vec::with_capacity(indices.len());
                        for (c, &idx) in indices.iter().enumerate() {
                            let i = idx as usize;
                            from.push(i);

                            let t = c - c % 3;
                            let rect = match corner_uv.get(t..t + 3) {
                                Some(&[a, b, d]) => MeshVertex::uv2_rect_of([a, b, d]),
                                _ => MeshVertex::WHOLE_ATLAS,
                            };
                            v.push(MeshVertex {
                                position: positions.get(i).copied().unwrap_or(Vec3::ZERO).into(),
                                normal: normals.get(i).copied().unwrap_or(Vec3::Y).into(),
                                uv: uvs.get(i).copied().unwrap_or([0.0, 0.0]),
                                uv2: corner_uv[c],
                                emissive,
                                uv2_rect: rect,
                            });
                        }
                        split_corners = from;
                        v
                    });
                let indices: Vec<u32> = match &split {
                    Some(v) => (0..v.len() as u32).collect(),
                    None => indices,
                };
                let vertices: Vec<MeshVertex> = split.unwrap_or_else(|| {
                    (0..positions.len())
                        .map(|i| MeshVertex {
                            position: positions[i].into(),
                            normal: normals.get(i).copied().unwrap_or(Vec3::Y).into(),
                            uv: uvs.get(i).copied().unwrap_or([0.0, 0.0]),
                            uv2: uv2s.get(i).copied().unwrap_or([0.0, 0.0]),
                            emissive,
                            uv2_rect: MeshVertex::WHOLE_ATLAS,
                        })
                        .collect()
                });

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

                    let source: Vec<usize> = if split_corners.is_empty() {
                        (0..positions.len()).collect()
                    } else {
                        split_corners.clone()
                    };
                    let lv: Vec<LayeredVertex> = source
                        .iter()
                        .map(|&i| LayeredVertex {
                            position: positions.get(i).copied().unwrap_or(Vec3::ZERO).into(),
                            normal: normals.get(i).copied().unwrap_or(Vec3::Y).into(),

                            weights: weights.get(i).copied().unwrap_or([1.0, 0.0, 0.0, 0.0]),
                            aux: layered_aux.get(i).copied().unwrap_or([1.0, 0.0, 0.0, 1.0]),

                            ground: {
                                let uv = uvs.get(i).copied().unwrap_or([0.0, 0.0]);
                                let b = uv2s.get(i).map_or(0.0, |u| u[0]);
                                [uv[0], uv[1], b, 0.0]
                            },
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

                let casts_shadow = casts_shadow(
                    prim.material()
                        .transmission()
                        .map_or(0.0, |t| t.transmission_factor()),
                );

                let blended = prim.material().alpha_mode() == gltf::material::AlphaMode::Blend
                    || prim.material().transmission().map_or(0.0, |t| t.transmission_factor()) > 0.0;

                let thin = if layered.is_none() && !blended {
                    super::thin_parts::split_thin_parts(&vertices, &indices)
                        .map(|split| super::thin_parts::ThinParts::upload(device, &split))
                } else {
                    None
                };
                static_out.push(MeshPrimitive {
                    vertices,
                    indices,
                    texture,
                    vertex_buffer,
                    index_buffer,
                    layered,
                    casts_shadow,
                    blended,
                    thin,
                });
            } else {
                let (joint_ids, joint_weights): (Vec<[u32; 4]>, Vec<[f32; 4]>) = if real_skin {
                    let joint_ids = match reader.read_joints(0) {
                        Some(gltf::mesh::util::ReadJoints::U8(it)) => it
                            .map(|j| [u32::from(j[0]), u32::from(j[1]), u32::from(j[2]), u32::from(j[3])])
                            .collect(),
                        Some(gltf::mesh::util::ReadJoints::U16(it)) => it
                            .map(|j| [u32::from(j[0]), u32::from(j[1]), u32::from(j[2]), u32::from(j[3])])
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

                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::STORAGE,
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
            lightmap,
            static_out,
            skinned_out,
        );
    }
}

pub(crate) fn casts_shadow(transmission: f32) -> bool {
    transmission <= 0.5
}

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

#[cfg(test)]
mod shadow_caster_tests {
    use super::casts_shadow;

    #[test]
    fn clear_glass_does_not_cast_a_shadow() {
        assert!(!casts_shadow(1.0));
    }

    #[test]
    fn an_ordinary_material_casts_exactly_as_before() {
        assert!(casts_shadow(0.0));
        assert!(casts_shadow(0.1));
    }

    #[test]
    fn partial_transmission_rounds_to_the_nearer_answer() {
        assert!(casts_shadow(0.4), "blocks 60% of the light -- it casts");
        assert!(!casts_shadow(0.6), "blocks 40% of the light -- it does not");
        assert!(casts_shadow(0.5), "exactly half stays a caster, so the default side is the safe one");
    }
}
