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
}

impl MeshVertex {
    pub const ATTRIBS: [wgpu::VertexAttribute; 3] =
        wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x2];

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
}

#[derive(Clone)]
pub struct SkinnedMeshPrimitive {
    pub index_count: u32,
    pub texture: Arc<LoadedTexture>,
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
}

pub const MAX_SKIN_JOINTS: usize = 64;

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

    pub fn load(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layout: &wgpu::BindGroupLayout,
        path: &Path,
    ) -> Result<Self> {
        let (doc, buffers, images) =
            gltf::import(path).with_context(|| format!("failed to open {}", path.display()))?;

        let mut skin_opt: Option<GltfSkin> = None;
        for skin in doc.skins() {
            let joint_nodes: Vec<gltf::Node> = skin.joints().collect();
            let joint_names: Vec<String> = joint_nodes
                .iter()
                .map(|j| j.name().unwrap_or("").to_string())
                .collect();

            let joint_count = joint_names.len();
            let inv_bind_mats: Vec<Mat4> = if let Some(acc) = skin.inverse_bind_matrices() {
                let view = acc.view().context("skin ibm: no buffer view")?;
                let buf_data: &[u8] = &buffers[view.buffer().index()];
                let start = view.offset() + acc.offset();
                let stride = view.stride().unwrap_or(64);
                (0..acc.count())
                    .map(|i| {
                        let off = start + i * stride;
                        let arr: [f32; 16] = bytemuck::pod_read_unaligned(&buf_data[off..off + 64]);
                        Mat4::from_cols_array(&arr)
                    })
                    .collect()
            } else {
                vec![Mat4::IDENTITY; joint_count]
            };

            let node_index_to_joint: HashMap<usize, usize> = joint_nodes
                .iter()
                .enumerate()
                .map(|(ji, node)| (node.index(), ji))
                .collect();
            let mut parent_of_node: HashMap<usize, usize> = HashMap::new();
            for node in doc.nodes() {
                for child in node.children() {
                    parent_of_node.insert(child.index(), node.index());
                }
            }
            let joint_parents: Vec<Option<usize>> = joint_nodes
                .iter()
                .map(|node| {
                    parent_of_node
                        .get(&node.index())
                        .and_then(|pidx| node_index_to_joint.get(pidx).copied())
                })
                .collect();
            let joint_local_bind: Vec<(Vec3, Quat, Vec3)> = joint_nodes
                .iter()
                .map(|node| {
                    let (t, r, s) = node.transform().decomposed();
                    (Vec3::from(t), Quat::from_array(r), Vec3::from(s))
                })
                .collect();

            let animations: Vec<GltfAnimationPose> = doc
                .animations()
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
                            Some(gltf::animation::util::ReadOutputs::Translations(mut t)) => {
                                if let Some(v) = t.next() {
                                    entry.0 = Some(Vec3::from(v));
                                }
                            }
                            Some(gltf::animation::util::ReadOutputs::Rotations(r)) => {
                                if let Some(v) = r.into_f32().next() {
                                    entry.1 = Some(Quat::from_xyzw(v[0], v[1], v[2], v[3]));
                                }
                            }
                            Some(gltf::animation::util::ReadOutputs::Scales(mut s)) => {
                                if let Some(v) = s.next() {
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
                .collect();

            log::info!(
                "GltfMesh: skin has {} joints: {:?}, animation clips: {:?}",
                joint_count,
                &joint_names,
                animations
                    .iter()
                    .map(|a| a.name.as_str())
                    .collect::<Vec<_>>(),
            );

            let joint_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("skin_joint_buf"),
                size: (MAX_SKIN_JOINTS * 64) as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            skin_opt = Some(GltfSkin {
                joint_names,
                inv_bind_mats,
                joint_parents,
                joint_local_bind,
                animations,
                joint_buffer,
                joint_bind_group: None,
                primitives: Vec::new(),
            });
            break;
        }

        let is_skinned = skin_opt.is_some();

        let mut static_prims: Vec<MeshPrimitive> = Vec::new();
        let mut skinned_prims: Vec<SkinnedMeshPrimitive> = Vec::new();

        for scene in doc.scenes() {
            for node in scene.nodes() {
                collect_node(
                    &node,
                    Mat4::IDENTITY,
                    &buffers,
                    &images,
                    device,
                    queue,
                    layout,
                    is_skinned,
                    &mut static_prims,
                    &mut skinned_prims,
                );
            }
        }

        if static_prims.is_empty() && skinned_prims.is_empty() {
            anyhow::bail!("no renderable primitives found in {}", path.display());
        }

        log::info!(
            "GltfMesh: loaded {} static + {} skinned primitives from {}",
            static_prims.len(),
            skinned_prims.len(),
            path.display(),
        );

        let bounding_radius = static_prims
            .iter()
            .flat_map(|p| p.vertices.iter())
            .map(|v| Vec3::from(v.position).length())
            .fold(0.0_f32, f32::max);

        if let Some(skin) = &mut skin_opt {
            skin.primitives = skinned_prims;
        }

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

        for scene in doc.scenes() {
            for node in scene.nodes() {
                collect_node(
                    &node,
                    Mat4::IDENTITY,
                    &buffers,
                    &images,
                    device,
                    queue,
                    layout,
                    false,
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

#[allow(clippy::too_many_arguments)]
fn collect_node(
    node: &gltf::Node,
    parent: Mat4,
    buffers: &[gltf::buffer::Data],
    images: &[gltf::image::Data],
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
    skinned: bool,
    static_out: &mut Vec<MeshPrimitive>,
    skinned_out: &mut Vec<SkinnedMeshPrimitive>,
) {
    let local = Mat4::from_cols_array_2d(&node.transform().matrix());
    let world = parent * local;

    if let Some(mesh) = node.mesh() {
        for prim in mesh.primitives() {
            let reader = prim.reader(|buf| Some(&buffers[buf.index()]));

            let positions: Vec<Vec3> = match reader.read_positions() {
                Some(p) => {
                    if skinned {
                        p.map(Vec3::from).collect()
                    } else {
                        p.map(|v| world.transform_point3(Vec3::from(v))).collect()
                    }
                }
                None => continue,
            };
            if positions.is_empty() {
                continue;
            }

            let normals: Vec<Vec3> = match reader.read_normals() {
                Some(n) => {
                    if skinned {
                        n.map(|v| Vec3::from(v).normalize_or_zero()).collect()
                    } else {
                        n.map(|v| world.transform_vector3(Vec3::from(v)).normalize_or_zero())
                            .collect()
                    }
                }
                None => vec![Vec3::Y; positions.len()],
            };

            let uvs: Vec<[f32; 2]> = match reader.read_tex_coords(0) {
                Some(uv) => uv.into_f32().collect(),
                None => vec![[0.0, 0.0]; positions.len()],
            };

            let indices: Vec<u32> = match reader.read_indices() {
                Some(i) => i.into_u32().collect(),
                None => (0..positions.len() as u32).collect(),
            };

            let texture = load_primitive_texture(&prim, images, device, queue, layout);
            let texture = Arc::new(texture);

            if skinned {
                let joint_ids: Vec<[u32; 4]> = match reader.read_joints(0) {
                    Some(gltf::mesh::util::ReadJoints::U8(it)) => it
                        .map(|j| [j[0] as u32, j[1] as u32, j[2] as u32, j[3] as u32])
                        .collect(),
                    Some(gltf::mesh::util::ReadJoints::U16(it)) => it
                        .map(|j| [j[0] as u32, j[1] as u32, j[2] as u32, j[3] as u32])
                        .collect(),
                    None => vec![[0, 0, 0, 0]; positions.len()],
                };

                let weights: Vec<[f32; 4]> = match reader.read_weights(0) {
                    Some(w) => w.into_f32().collect(),
                    None => vec![[1.0, 0.0, 0.0, 0.0]; positions.len()],
                };

                let vertices: Vec<SkinnedMeshVertex> = (0..positions.len())
                    .map(|i| SkinnedMeshVertex {
                        position: positions[i].into(),
                        normal: normals.get(i).copied().unwrap_or(Vec3::Y).into(),
                        uv: uvs.get(i).copied().unwrap_or([0.0, 0.0]),
                        joint_ids: joint_ids.get(i).copied().unwrap_or([0; 4]),
                        joint_weights: weights.get(i).copied().unwrap_or([1.0, 0.0, 0.0, 0.0]),
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
                    index_count: indices.len() as u32,
                    texture,
                    vertex_buffer,
                    index_buffer,
                });
            } else {
                let vertices: Vec<MeshVertex> = (0..positions.len())
                    .map(|i| MeshVertex {
                        position: positions[i].into(),
                        normal: normals.get(i).copied().unwrap_or(Vec3::Y).into(),
                        uv: uvs.get(i).copied().unwrap_or([0.0, 0.0]),
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
            }
        }
    }

    for child in node.children() {
        collect_node(
            &child,
            world,
            buffers,
            images,
            device,
            queue,
            layout,
            skinned,
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
    let pbr = prim.material().pbr_metallic_roughness();

    if let Some(info) = pbr.base_color_texture() {
        let image = &images[info.texture().source().index()];
        return upload_texture_rgba(device, queue, layout, image);
    }

    let c = pbr.base_color_factor();
    let rgba = [
        (c[0] * 255.0) as u8,
        (c[1] * 255.0) as u8,
        (c[2] * 255.0) as u8,
        (c[3] * 255.0) as u8,
    ];
    upload_solid_texture(device, queue, layout, rgba)
}

fn upload_texture_rgba(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
    image: &gltf::image::Data,
) -> LoadedTexture {
    use gltf::image::Format;

    let (width, height) = (image.width, image.height);

    let rgba: Vec<u8> = match image.format {
        Format::R8G8B8A8 => image.pixels.clone(),
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
            vec![180u8; (width * height * 4) as usize]
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

pub(crate) fn create_texture_from_rgba(
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
        bind_group,
    }
}
