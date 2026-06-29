use bytemuck::{Pod, Zeroable};
use glam::{Vec3, Mat4, Quat};
use std::path::Path;
use std::sync::Arc;
use anyhow::{Result, Context};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct MeshVertex {
    pub position: [f32; 3],
    pub normal:   [f32; 3],
    pub uv:       [f32; 2],
}

impl MeshVertex {
    pub const ATTRIBS: [wgpu::VertexAttribute; 3] =
        wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x2];

    pub fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode:    wgpu::VertexStepMode::Vertex,
            attributes:   &Self::ATTRIBS,
        }
    }
}

/// `vertex_buffer`/`index_buffer` are built once, here, at load time, and
/// reused every frame thereafter — this geometry never changes after
/// load, so re-uploading it every frame (the old behavior, via
/// `device.create_buffer_init` inside `render_internal`) was pure wasted
/// CPU+GPU work, the dominant cost in scenes with several mesh objects.
pub struct MeshPrimitive {
    pub vertices: Vec<MeshVertex>,
    pub indices:  Vec<u32>,
    pub texture:  Arc<LoadedTexture>,
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer:  wgpu::Buffer,
}

pub struct LoadedTexture {
    pub texture:    wgpu::Texture,
    pub view:       wgpu::TextureView,
    pub bind_group: wgpu::BindGroup,
}

pub struct GltfMesh {
    pub primitives: Vec<MeshPrimitive>,
    pub position:   Vec3,
    pub rotation:   Quat,
    pub scale:      Vec3,
}

impl GltfMesh {
    pub fn model_matrix(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.position)
    }

    pub fn load(
        device: &wgpu::Device,
        queue:  &wgpu::Queue,
        layout: &wgpu::BindGroupLayout,
        path:   &Path,
    ) -> Result<Self> {
        let (doc, buffers, images) = gltf::import(path)
            .with_context(|| format!("failed to open {}", path.display()))?;

        let mut primitives = Vec::new();

        for scene in doc.scenes() {
            for node in scene.nodes() {
                collect_node(
                    &node, Mat4::IDENTITY, &buffers, &images,
                    device, queue, layout, &mut primitives,
                );
            }
        }

        if primitives.is_empty() {
            anyhow::bail!("no renderable primitives found in {}", path.display());
        }

        log::info!(
            "GltfMesh: loaded {} primitives from {}",
            primitives.len(), path.display(),
        );

        Ok(Self {
            primitives,
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale:    Vec3::ONE,
        })
    }
}

fn collect_node(
    node:    &gltf::Node,
    parent:  Mat4,
    buffers: &[gltf::buffer::Data],
    images:  &[gltf::image::Data],
    device:  &wgpu::Device,
    queue:   &wgpu::Queue,
    layout:  &wgpu::BindGroupLayout,
    out:     &mut Vec<MeshPrimitive>,
) {
    let local = Mat4::from_cols_array_2d(&node.transform().matrix());
    let world = parent * local;

    if let Some(mesh) = node.mesh() {
        for prim in mesh.primitives() {
            let reader = prim.reader(|buf| Some(&buffers[buf.index()]));

            let positions: Vec<Vec3> = match reader.read_positions() {
                Some(p) => p.map(|v| world.transform_point3(Vec3::from(v))).collect(),
                None => continue,
            };
            if positions.is_empty() { continue; }

            let normals: Vec<Vec3> = match reader.read_normals() {
                Some(n) => n.map(|v| world.transform_vector3(Vec3::from(v)).normalize_or_zero()).collect(),
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

            let vertices: Vec<MeshVertex> = (0..positions.len())
                .map(|i| MeshVertex {
                    position: positions[i].into(),
                    normal:   normals.get(i).copied().unwrap_or(Vec3::Y).into(),
                    uv:       uvs.get(i).copied().unwrap_or([0.0, 0.0]),
                })
                .collect();

            let texture = load_primitive_texture(&prim, images, device, queue, layout);

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

            out.push(MeshPrimitive {
                vertices, indices, texture: Arc::new(texture),
                vertex_buffer, index_buffer,
            });
        }
    }

    for child in node.children() {
        collect_node(&child, world, buffers, images, device, queue, layout, out);
    }
}

fn load_primitive_texture(
    prim:   &gltf::Primitive,
    images: &[gltf::image::Data],
    device: &wgpu::Device,
    queue:  &wgpu::Queue,
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
    queue:  &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
    image:  &gltf::image::Data,
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
            log::warn!("Unsupported glTF image format {:?}, using gray fallback", image.format);
            vec![180u8; (width * height * 4) as usize]
        }
    };

    create_texture_from_rgba(device, queue, layout, &rgba, width, height)
}

fn upload_solid_texture(
    device: &wgpu::Device,
    queue:  &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
    rgba:   [u8; 4],
) -> LoadedTexture {
    create_texture_from_rgba(device, queue, layout, &rgba, 1, 1)
}

fn create_texture_from_rgba(
    device: &wgpu::Device,
    queue:  &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
    rgba:   &[u8],
    width:  u32,
    height: u32,
) -> LoadedTexture {
    let size = wgpu::Extent3d { width, height, depth_or_array_layers: 1 };

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label:           Some("gltf_texture"),
        size,
        mip_level_count: 1,
        sample_count:    1,
        dimension:       wgpu::TextureDimension::D2,
        format:          wgpu::TextureFormat::Rgba8UnormSrgb,
        usage:           wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats:    &[],
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
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&view) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&sampler) },
        ],
    });

    LoadedTexture { texture, view, bind_group }
}