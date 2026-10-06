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
    // The caller-supplied lightmap UV set. `None` means the mesh keeps
    // whatever uv2 it was authored with, which for almost every asset is none
    // at all -- and a 1x1 lightmap does not care where it is sampled.
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

            // Layer weights, when the file both asks for layered shading and
            // carries them. Read here rather than in the `bake` arm below
            // because the reader borrows the primitive, and a cave is always
            // static anyway -- nothing has ever rigged one.
            let layered_weights: Option<Vec<[f32; 4]>> = if wants_layered_shading(&mesh) {
                reader.read_colors(0).map(|c| c.into_rgba_f32().collect())
            } else {
                None
            };

            // The material's emissive colour, as the ARTIST authored it.
            //
            // This is what makes a lamp read correctly: the bulb's material is
            // emissive and the housing's is not, so switching the light on lights
            // the bulb rather than the whole fixture. An asset with no emissive
            // material is unaffected -- glTF's default emissiveFactor is black.
            //
            // KHR_materials_emissive_strength is folded in here rather than
            // stored separately: it is a multiplier on the same colour, and
            // keeping them apart would mean carrying a second per-vertex value
            // to say something the first one can already express.
            let emissive = {
                let m = prim.material();
                let f = m.emissive_factor();
                // KHR_materials_emissive_strength is not exposed by this gltf
                // version, so the factor is taken as authored. An asset that
                // needs a brighter bulb than 1.0 says so through the light's
                // own intensity, which drives this per object anyway.
                let strength = 1.0f32;
                MeshVertex::pack_emissive([f[0] * strength, f[1] * strength, f[2] * strength])
            };

            if bake {
                // A LIGHTMAPPED MESH IS DE-INDEXED.
                //
                // Adjacent triangles land in unrelated parts of the atlas, so a
                // vertex shared between them has no single uv2 -- every edge is
                // a chart seam. Splitting is what an unwrapper does at a seam;
                // here it is every corner, which for a prop costs three
                // vertices per triangle instead of roughly one.
                //
                // The GENERATED uv2 wins over any the asset happens to carry.
                // The baker cannot read an artist's second UV set -- it derives
                // the layout from the geometry -- so honouring an authored one
                // here would mean the renderer sampling a chart the baker never
                // wrote to, which is a black model rather than an error.
                // Which source vertex each split corner came from, so any
                // other per-vertex stream can be split the same way. Empty when
                // the mesh was not split.
                let mut split_corners: Vec<usize> = Vec::new();
                let split = lightmap
                    .and_then(|lm| lm.per_primitive.get(&(node.index(), prim.index())))
                    // One uv2 per index, or the map is describing different
                    // geometry from the one loaded here -- a stale bake, a
                    // re-exported asset -- and using it would scatter this
                    // mesh's lighting. Falling back is the safe answer and
                    // shows up as a flat model rather than a shredded one.
                    .filter(|corner_uv| corner_uv.len() == indices.len())
                    .map(|corner_uv| {
                        let mut v = Vec::with_capacity(indices.len());
                        let mut from = Vec::with_capacity(indices.len());
                        for (c, &idx) in indices.iter().enumerate() {
                            let i = idx as usize;
                            from.push(i);
                            // The triangle's own chart, which its pixels'
                            // `uv2` is clamped into (see `uv2_rect`).
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
                    // The layered buffer is drawn with the SAME index buffer as
                    // the shaded one, so it has to be split wherever that was.
                    // Leaving it indexed by the original vertex count while the
                    // indices count corners reads far off the end of it -- and
                    // the failure is silent garbage on the GPU rather than a
                    // bounds check.
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

                // See `MeshPrimitive::casts_shadow`. Half, because that is the
                // honest rounding of a continuous transmission onto a binary
                // shadow map: a surface that lets more light through than it
                // stops is better modelled as not stopping it. A material with
                // no transmission -- which is nearly all of them -- is
                // unaffected and casts exactly as it always did.
                let casts_shadow = casts_shadow(
                    prim.material()
                        .transmission()
                        .map(|t| t.transmission_factor())
                        .unwrap_or(0.0),
                );
                // What `texture::load_primitive_texture` draws see-through.
                let blended = prim.material().alpha_mode() == gltf::material::AlphaMode::Blend
                    || prim.material().transmission().map(|t| t.transmission_factor()).unwrap_or(0.0) > 0.0;
                // See `thin_parts`. Not for a cave (drawn by the layered
                // pipeline) nor glass (drawn see-through, after the thin pass).
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
                    // STORAGE too: read by the pass that poses it once a frame
                    // (`skin_compute`).
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

/// Whether a surface transmitting `transmission` of the light belongs in a
/// shadow map. See `MeshPrimitive::casts_shadow`.
///
/// Half, because that is the honest rounding of a continuous transmission onto
/// a BINARY shadow map: a surface that lets more light through than it stops is
/// better modelled as not stopping it. A material with no transmission -- which
/// is nearly all of them -- is unaffected and casts exactly as it always did.
///
/// Its own function so the threshold can be tested without a glTF fixture, and
/// so there is one place to change if translucent shadows ever arrive.
pub(crate) fn casts_shadow(transmission: f32) -> bool {
    transmission <= 0.5
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


#[cfg(test)]
mod shadow_caster_tests {
    use super::casts_shadow;

    /// The hanging lamp's envelope: `KHR_materials_transmission` 1.0, invisible
    /// to the eye, and as a shadow caster it sealed its own bulb in -- measured
    /// at 100% of the bulb's downward light blocked, against 2% for the opaque
    /// housing around it. Both rooms in `test_room` rendered black because of
    /// it, and the lamp still glowed, which made it read as a lighting problem
    /// rather than a shadow one.
    #[test]
    fn clear_glass_does_not_cast_a_shadow() {
        assert!(!casts_shadow(1.0));
    }

    /// Everything else is unchanged. glTF's default transmission is 0, so this
    /// is every material in every asset that has never heard of the extension.
    #[test]
    fn an_ordinary_material_casts_exactly_as_before() {
        assert!(casts_shadow(0.0));
        assert!(casts_shadow(0.1));
    }

    /// A binary shadow map has to round, and it rounds at the halfway point:
    /// a surface stopping most of the light still casts.
    #[test]
    fn partial_transmission_rounds_to_the_nearer_answer() {
        assert!(casts_shadow(0.4), "blocks 60% of the light -- it casts");
        assert!(!casts_shadow(0.6), "blocks 40% of the light -- it does not");
        assert!(casts_shadow(0.5), "exactly half stays a caster, so the default side is the safe one");
    }
}
