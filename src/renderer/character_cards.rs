use glam::{Mat4, Vec3, Vec4};
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingResource, BindingType, Buffer, BufferBindingType,
    BufferDescriptor, BufferUsages, CommandEncoder, Device, Queue, RenderPipeline,
    ShaderModuleDescriptor, ShaderSource, ShaderStages, Texture, TextureView,
};

use super::brush_pipeline::probe_pass::{MirrorMips, MIRROR_MIPS};
use super::mesh::{SkinnedMeshVertex, MAX_SKIN_JOINTS};
use super::mesh_pipeline::SkinnedMeshPipeline;
use super::proxy_cards::CARD_FACES;
use super::uniforms::{CapsuleUpload, CAPSULES_PER_GROUP};

pub const CARD_BOX_MARGIN: f32 = 0.06;

pub const CARD_MIPS: u32 = MIRROR_MIPS;

const CARD_STRIDE: u64 = 256;

pub fn simplify(
    vertices: &[SkinnedMeshVertex],
    indices: &[u32],
    cell: f32,
) -> (Vec<SkinnedMeshVertex>, Vec<u32>) {
    use std::collections::{HashMap, HashSet};

    let mut parent: Vec<u32> = (0..vertices.len() as u32).collect();
    fn root(parent: &mut [u32], mut i: u32) -> u32 {
        while parent[i as usize] != i {
            parent[i as usize] = parent[parent[i as usize] as usize];
            i = parent[i as usize];
        }
        i
    }
    for t in indices.chunks_exact(3) {
        let a = root(&mut parent, t[0]);
        for &other in &t[1..] {
            let b = root(&mut parent, other);
            parent[b as usize] = a;
        }
    }
    let cell_of = |v: &SkinnedMeshVertex| {
        let q = (Vec3::from(v.position) / cell).floor();
        (q.x as i32, q.y as i32, q.z as i32)
    };
    let mut cell_mean: HashMap<(i32, i32, i32), (Vec3, f32)> = HashMap::new();
    for v in vertices {
        let e = cell_mean.entry(cell_of(v)).or_insert((Vec3::ZERO, 0.0));
        e.0 += Vec3::from(v.position);
        e.1 += 1.0;
    }
    let mut clusters: HashMap<((i32, i32, i32), u32, u32), u32> = HashMap::new();
    let mut out: Vec<SkinnedMeshVertex> = Vec::new();
    let mut uv_sums: Vec<([f32; 2], f32)> = Vec::new();
    let mut remap = Vec::with_capacity(vertices.len());
    for (i, v) in vertices.iter().enumerate() {
        let strongest = (0..4).fold(0, |b, k| {
            if v.joint_weights[k] > v.joint_weights[b] {
                k
            } else {
                b
            }
        });
        let c = cell_of(v);
        let key = (c, v.joint_ids[strongest], root(&mut parent, i as u32));
        let at = *clusters.entry(key).or_insert_with(|| {
            let (sum, n) = cell_mean[&c];
            out.push(SkinnedMeshVertex {
                position: (sum / n).to_array(),
                ..*v
            });
            uv_sums.push(([0.0; 2], 0.0));
            (out.len() - 1) as u32
        });
        let s = &mut uv_sums[at as usize];
        s.0 = [s.0[0] + v.uv[0], s.0[1] + v.uv[1]];
        s.1 += 1.0;
        remap.push(at);
    }
    for (v, (sum, n)) in out.iter_mut().zip(&uv_sums) {
        v.uv = [sum[0] / n, sum[1] / n];
    }
    let mut kept = HashSet::new();
    let mut tris = Vec::new();
    for t in indices.chunks_exact(3) {
        let [a, b, c] = [
            remap[t[0] as usize],
            remap[t[1] as usize],
            remap[t[2] as usize],
        ];
        if a == b || b == c || a == c {
            continue;
        }
        let mut key = [a, b, c];
        key.sort_unstable();
        if kept.insert(key) {
            tris.extend([a, b, c]);
        }
    }
    (out, tris)
}

#[derive(Clone, Copy)]
pub struct CardPart<'a> {
    pub model: &'a BindGroup,
    pub texture: &'a BindGroup,
    pub joints: &'a BindGroup,

    pub joint_buffer: &'a Buffer,

    pub source: &'a Buffer,
    pub vertices: &'a [SkinnedMeshVertex],
    pub indices: &'a [u32],
}

struct CardMesh {
    vertices: Buffer,
    indices: Buffer,
    count: u32,
    vertex_count: u32,

    posed: Option<(Buffer, crate::renderer::skin_compute::Posed)>,
}

pub fn card_turn(yaw: f32) -> glam::Quat {
    glam::Quat::from_rotation_y(yaw)
}

pub fn card_box(capsules: &CapsuleUpload, g: usize, yaw: f32) -> Option<(Vec3, Vec3)> {
    if g >= capsules.group_count as usize {
        return None;
    }
    let turn = card_turn(yaw);
    let count = capsules.groups[g * 2 + 1][3] as usize;
    let mut lo = Vec3::splat(f32::MAX);
    let mut hi = Vec3::splat(f32::MIN);
    for k in 0..count.min(CAPSULES_PER_GROUP) {
        let i = g * CAPSULES_PER_GROUP + k;
        let (a, b) = (capsules.capsules[i * 2], capsules.capsules[i * 2 + 1]);
        let r = Vec3::splat(a[3]);
        for end in [Vec3::new(a[0], a[1], a[2]), Vec3::new(b[0], b[1], b[2])] {
            let end = turn * end;
            lo = lo.min(end - r);
            hi = hi.max(end + r);
        }
    }
    if count == 0 || !lo.is_finite() || !hi.is_finite() {
        return None;
    }
    let (lo, hi) = (lo - CARD_BOX_MARGIN, hi + CARD_BOX_MARGIN);
    Some((turn.inverse() * ((lo + hi) * 0.5), ((hi - lo) * 0.5).max(Vec3::splat(0.05))))
}

pub fn card_matrix(k: usize, centre: Vec3, half: Vec3, yaw: f32) -> Mat4 {
    let turn = card_turn(yaw);
    square_card_matrix(k, turn * centre, half) * Mat4::from_quat(turn)
}

fn square_card_matrix(k: usize, centre: Vec3, half: Vec3) -> Mat4 {
    let (a, from_plus) = (k / 2, k % 2 == 0);
    let (a1, a2) = ((a + 1) % 3, (a + 2) % 3);
    let axis = |i: usize| Vec3::AXES[i];
    let row = |v: Vec3, w: f32| Vec4::new(v.x, v.y, v.z, w);
    let x = row(axis(a1) / half[a1], -centre[a1] / half[a1]);
    let y = row(-axis(a2) / half[a2], centre[a2] / half[a2]);
    let sign = if from_plus { -1.0 } else { 1.0 };
    let z = row(
        axis(a) * (sign * 0.5 / half[a]),
        0.5 - sign * 0.5 * centre[a] / half[a],
    );
    Mat4::from_cols(x, y, z, Vec4::W).transpose()
}

fn shader() -> String {
    format!(
        r#"
struct Card {{ view_proj: mat4x4<f32> }}
@group(0) @binding(0) var<uniform> card: Card;

struct ModelUniform {{ model: mat4x4<f32>, params: vec4<f32>, room: array<vec4<f32>, 9> }}
@group(1) @binding(0) var<uniform> model_u: ModelUniform;

@group(2) @binding(0) var tex: texture_2d<f32>;
@group(2) @binding(1) var samp: sampler;

struct JointMatrices {{ mats: array<mat4x4<f32>, {MAX_SKIN_JOINTS}> }}
@group(3) @binding(0) var<uniform> joints: JointMatrices;

struct VIn {{
    @location(0) position:      vec3<f32>,
    @location(1) normal:        vec3<f32>,
    @location(2) uv:            vec2<f32>,
    @location(3) joint_ids:     vec4<u32>,
    @location(4) joint_weights: vec4<f32>,
}}

struct VOut {{
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
}}

@vertex
fn vs_main(v: VIn) -> VOut {{
    let p = vec4<f32>(v.position, 1.0);
    let skinned_p =
        (joints.mats[v.joint_ids.x] * p) * v.joint_weights.x +
        (joints.mats[v.joint_ids.y] * p) * v.joint_weights.y +
        (joints.mats[v.joint_ids.z] * p) * v.joint_weights.z +
        (joints.mats[v.joint_ids.w] * p) * v.joint_weights.w;
    var out: VOut;
    out.clip = card.view_proj * (model_u.model * skinned_p);
    out.uv = v.uv;
    return out;
}}

struct PosedIn {{
    @location(0) skinned_p: vec4<f32>,
    @location(2) uv:        vec2<f32>,
}}

@vertex
fn vs_posed(v: PosedIn) -> VOut {{
    var out: VOut;
    out.clip = card.view_proj * (model_u.model * v.skinned_p);
    out.uv = v.uv;
    return out;
}}

@fragment
fn fs_main(in: VOut) -> @location(0) vec4<f32> {{
    let t = textureSample(tex, samp, in.uv);
    if (t.a < 0.5) {{
        discard;
    }}
    return vec4<f32>(t.rgb, 1.0);
}}
"#
    )
}

pub struct CharacterCards {
    resolution: u32,
    target: Texture,

    levels: Vec<TextureView>,
    _depth: Texture,
    depth_view: TextureView,
    pipeline: RenderPipeline,

    posed_pipeline: RenderPipeline,
    matrices: Buffer,
    matrices_group: BindGroup,

    meshes: std::sync::Mutex<std::collections::HashMap<Buffer, CardMesh>>,
}

impl CharacterCards {

    pub fn new(device: &Device, resolution: u32, skinned: &SkinnedMeshPipeline) -> Self {
        Self::with_layouts(
            device,
            resolution,
            [
                &skinned.model_layout,
                &skinned.texture_layout,
                &skinned.skin_joint_layout,
            ],
        )
    }

    pub fn with_layouts(
        device: &Device,
        resolution: u32,
        [model, texture, joints]: [&BindGroupLayout; 3],
    ) -> Self {
        let width = resolution * CARD_FACES as u32;
        let level_count = CARD_MIPS.min(resolution.max(1).ilog2() + 1);
        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("character_cards"),
            size: wgpu::Extent3d {
                width,
                height: resolution,
                depth_or_array_layers: 1,
            },
            mip_level_count: level_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let levels = (0..level_count)
            .map(|l| {
                target.create_view(&wgpu::TextureViewDescriptor {
                    base_mip_level: l,
                    mip_level_count: Some(1),
                    ..Default::default()
                })
            })
            .collect();
        let depth = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("character_cards_depth"),
            size: wgpu::Extent3d {
                width,
                height: resolution,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth_view = depth.create_view(&Default::default());
        let card_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("character_cards_matrices"),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::VERTEX,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: std::num::NonZeroU64::new(64),
                },
                count: None,
            }],
        });
        let matrices = device.create_buffer(&BufferDescriptor {
            label: Some("character_cards_matrices"),
            size: CARD_STRIDE * CARD_FACES as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let matrices_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("character_cards_matrices"),
            layout: &card_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &matrices,
                    offset: 0,
                    size: std::num::NonZeroU64::new(64),
                }),
            }],
        });
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("character_cards"),
            source: ShaderSource::Wgsl(shader().into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("character_cards"),
            bind_group_layouts: &[Some(&card_layout), Some(model), Some(texture), Some(joints)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("character_cards"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[Some(SkinnedMeshVertex::layout())],
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),

            primitive: wgpu::PrimitiveState {
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });

        let posed_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("character_cards_posed"),
            bind_group_layouts: &[Some(&card_layout), Some(model), Some(texture)],
            immediate_size: 0,
        });
        const UV: [wgpu::VertexAttribute; 1] = wgpu::vertex_attr_array![2 => Float32x2];
        let uv_from_source = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<SkinnedMeshVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[wgpu::VertexAttribute { offset: 24, ..UV[0] }],
        };
        let posed_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("character_cards_posed"),
            layout: Some(&posed_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_posed"),
                compilation_options: Default::default(),
                buffers: &[Some(crate::renderer::skin_compute::posed_layout()), Some(uv_from_source)],
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });
        Self {
            resolution,
            target,
            levels,
            _depth: depth,
            depth_view,
            pipeline,
            posed_pipeline,
            matrices,
            matrices_group,
            meshes: Default::default(),
        }
    }

    pub fn resolution(&self) -> u32 {
        self.resolution
    }

    pub fn read_back(
        &self,
        device: &Device,
        queue: &Queue,
    ) -> Result<(u32, u32, Vec<[f32; 4]>), String> {
        let (width, height) = (self.resolution * CARD_FACES as u32, self.resolution);
        let row = (width * 8).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
            * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let buffer = device.create_buffer(&BufferDescriptor {
            label: Some("character_cards_read"),
            size: u64::from(row) * u64::from(height),
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("character_cards_read"),
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.target,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(row),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(Some(encoder.finish()));
        buffer.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        let data = buffer
            .slice(..)
            .get_mapped_range()
            .map_err(|e| format!("{e:?}"))?;
        let mut texels = Vec::with_capacity((width * height) as usize);
        for y in 0..height as usize {
            let line: &[u16] = bytemuck::cast_slice(
                &data[y * row as usize..y * row as usize + width as usize * 8],
            );
            texels.extend(
                line.chunks_exact(4).map(|c| {
                    std::array::from_fn(|i| crate::renderer::ground_map::f16_to_f32(c[i]))
                }),
            );
        }
        Ok((width, height, texels))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn record(
        &self,
        device: &Device,
        queue: &Queue,
        encoder: &mut CommandEncoder,
        mips: &MirrorMips,
        parts: &[CardPart],
        (centre, half): (Vec3, Vec3),
        yaw: f32,
        atlas: &Texture,
        row: u32,
        timer: Option<(&crate::renderer::pass_timers::PassTimers, usize)>,
        skin: Option<&crate::renderer::skin_compute::SkinCompute>,
    ) {
        let mut meshes = self.meshes.lock().unwrap_or_else(|e| e.into_inner());
        meshes.retain(|source, _| parts.iter().any(|part| part.source == source));
        if parts.iter().any(|part| !meshes.contains_key(part.source)) {

            let (lo, hi) = parts.iter().flat_map(|part| part.vertices).fold(
                (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)),
                |(lo, hi), v| {
                    let p = Vec3::from(v.position);
                    (lo.min(p), hi.max(p))
                },
            );
            let cell = ((hi - lo).max_element() / self.resolution as f32).max(1e-4);
            for part in parts {
                meshes.entry(part.source.clone()).or_insert_with(|| {
                let (vertices, indices) = simplify(part.vertices, part.indices, cell);
                log::info!(
                    "character cards: a body primitive of {} vertices, {} triangles drawn on its cards as {} and {}",
                    part.vertices.len(),
                    part.indices.len() / 3,
                    vertices.len(),
                    indices.len() / 3,
                );
                CardMesh {
                    vertices: wgpu::util::DeviceExt::create_buffer_init(device, &wgpu::util::BufferInitDescriptor {
                        label: Some("character_card_mesh_vb"),
                        contents: bytemuck::cast_slice(&vertices),

                        usage: BufferUsages::VERTEX | BufferUsages::STORAGE,
                    }),
                    indices: wgpu::util::DeviceExt::create_buffer_init(device, &wgpu::util::BufferInitDescriptor {
                        label: Some("character_card_mesh_ib"),
                        contents: bytemuck::cast_slice(&indices),
                        usage: BufferUsages::INDEX,
                    }),
                    count: indices.len() as u32,
                    vertex_count: vertices.len() as u32,
                    posed: None,
                }
                });
            }
        }

        if let Some(skin) = skin {
            for part in parts {
                if let Some(mesh) = meshes.get_mut(part.source) {
                    if mesh.posed.as_ref().is_none_or(|(joints, _)| joints != part.joint_buffer) {
                        let posed = skin.posed_for(device, &mesh.vertices, mesh.vertex_count, part.joint_buffer);
                        mesh.posed = Some((part.joint_buffer.clone(), posed));
                    }
                }
            }
            skin.dispatch(
                encoder,
                parts.iter().filter_map(|part| meshes.get(part.source)).filter_map(|mesh| mesh.posed.as_ref().map(|(_, p)| p)),
            );
        }
        let mut bytes = vec![0u8; (CARD_STRIDE * CARD_FACES as u64) as usize];
        for k in 0..CARD_FACES {
            let m = card_matrix(k, centre, half, yaw).to_cols_array();
            let at = k * CARD_STRIDE as usize;
            bytes[at..at + 64].copy_from_slice(bytemuck::cast_slice(&m));
        }
        queue.write_buffer(&self.matrices, 0, &bytes);
        let res = self.resolution;
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("character_cards"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.levels[0],
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: timer.and_then(|(timers, slot)| timers.writes(slot)),
                ..Default::default()
            });
            for part in parts {
                let Some(mesh) = meshes.get(part.source) else {
                    continue;
                };
                match (skin, &mesh.posed) {
                    (Some(_), Some((_, posed))) => {
                        pass.set_pipeline(&self.posed_pipeline);
                        pass.set_vertex_buffer(0, posed.positions.slice(..));
                        pass.set_vertex_buffer(1, mesh.vertices.slice(..));
                    }
                    _ => {
                        pass.set_pipeline(&self.pipeline);
                        pass.set_bind_group(3, part.joints, &[]);
                        pass.set_vertex_buffer(0, mesh.vertices.slice(..));
                    }
                }
                pass.set_bind_group(1, part.model, &[]);
                pass.set_bind_group(2, part.texture, &[]);
                pass.set_index_buffer(mesh.indices.slice(..), wgpu::IndexFormat::Uint32);
                for k in 0..CARD_FACES as u32 {
                    pass.set_viewport((k * res) as f32, 0.0, res as f32, res as f32, 0.0, 1.0);
                    pass.set_bind_group(0, &self.matrices_group, &[k * CARD_STRIDE as u32]);
                    pass.draw_indexed(0..mesh.count, 0, 0..1);
                }
            }
        }
        mips.record(
            device,
            encoder,
            &self.levels,
            (res * CARD_FACES as u32, res),
            timer.map(|(timers, slot)| (timers, slot + 1)),
        );
        for (l, _) in self.levels.iter().enumerate() {
            let l = l as u32;
            encoder.copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &self.target,
                    mip_level: l,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyTextureInfo {
                    texture: atlas,
                    mip_level: l,
                    origin: wgpu::Origin3d {
                        x: 0,
                        y: (row * res) >> l,
                        z: 0,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::Extent3d {
                    width: (res * CARD_FACES as u32) >> l,
                    height: (res >> l).max(1),
                    depth_or_array_layers: 1,
                },
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_card_sees_the_box_as_the_lookup_reads_it() {
        let (centre, half) = (Vec3::new(1.0, 2.0, -3.0), Vec3::new(0.4, 0.9, 0.3));
        for (yaw, k) in [0.0f32, std::f32::consts::FRAC_PI_4, -1.3, 3.0]
            .into_iter()
            .flat_map(|yaw| (0..CARD_FACES).map(move |k| (yaw, k)))
        {

            let turn = card_turn(yaw);
            let centre_seen = turn.inverse() * centre;
            let (a, plus) = (k / 2, k % 2 == 0);
            let (a1, a2) = ((a + 1) % 3, (a + 2) % 3);
            let m = card_matrix(k, centre_seen, half, yaw);

            let mut on_face = centre;
            on_face[a1] += (0.25 - 0.5) * 2.0 * half[a1];
            on_face[a2] += (0.75 - 0.5) * 2.0 * half[a2];
            on_face[a] += if plus { half[a] } else { -half[a] };
            let mut far = on_face;
            far[a] = 2.0 * centre[a] - on_face[a];
            let (c, f) = (
                m.project_point3(turn.inverse() * on_face),
                m.project_point3(turn.inverse() * far),
            );
            let uv = |p: Vec3| ((p.x + 1.0) * 0.5, (1.0 - p.y) * 0.5);
            let (u, v) = uv(c);
            assert!(
                (u - 0.25).abs() < 1e-5 && (v - 0.75).abs() < 1e-5,
                "card {k}, rig turned {yaw}: uv {:?}",
                (u, v)
            );
            assert!(
                c.z.abs() < 1e-5 && (f.z - 1.0).abs() < 1e-5,
                "card {k}, rig turned {yaw}: depths {} and {}",
                c.z,
                f.z
            );
        }
    }

    fn box_body() -> (Vec<SkinnedMeshVertex>, Vec<u32>) {
        let (mut verts, mut index) = (Vec::new(), Vec::new());
        add_box(Vec3::new(-0.2, 0.0, -0.1), Vec3::new(0.2, 1.6, 0.1), &mut verts, &mut index);
        (verts, index)
    }

    fn legged_body() -> (Vec<SkinnedMeshVertex>, Vec<u32>) {
        let (mut verts, mut index) = (Vec::new(), Vec::new());
        add_box(Vec3::new(-0.2, 0.6, -0.1), Vec3::new(0.2, 1.6, 0.1), &mut verts, &mut index);
        add_box(Vec3::new(-0.08, 0.0, -0.1), Vec3::new(0.08, 0.6, 0.1), &mut verts, &mut index);
        (verts, index)
    }

    fn add_box(lo: Vec3, hi: Vec3, verts: &mut Vec<SkinnedMeshVertex>, index: &mut Vec<u32>) {
        for (a, sign) in [
            (2usize, 1.0f32),
            (2, -1.0),
            (0, 1.0),
            (0, -1.0),
            (1, 1.0),
            (1, -1.0),
        ] {
            let texel = match (a, sign > 0.0) {
                (2, true) => 0.0,
                (2, false) => 1.0,
                (0, _) => 2.0,
                _ => 3.0,
            };
            let (a1, a2) = ((a + 1) % 3, (a + 2) % 3);
            let base = verts.len() as u32;
            for (i, j) in [(0, 0), (1, 0), (1, 1), (0, 1)] {
                let mut p = Vec3::ZERO;
                p[a] = if sign > 0.0 { hi[a] } else { lo[a] };
                p[a1] = if i == 0 { lo[a1] } else { hi[a1] };
                p[a2] = if j == 0 { lo[a2] } else { hi[a2] };
                let mut n = [0.0; 3];
                n[a] = sign;
                verts.push(SkinnedMeshVertex {
                    position: p.to_array(),
                    normal: n,
                    uv: [(texel + 0.5) / 4.0, 0.5],
                    joint_ids: [0; 4],
                    joint_weights: [1.0, 0.0, 0.0, 0.0],
                });
            }
            index.extend([base, base + 1, base + 2, base, base + 2, base + 3]);
        }
    }

    #[test]
    fn a_reflection_meeting_the_player_shows_the_side_it_meets() {
        use crate::renderer::brush_pipeline::probe_pass::MirrorMips;
        use crate::renderer::uniforms::Uniforms;
        use wgpu::util::DeviceExt;
        let Some((device, queue)) = crate::renderer::terrain_pipeline::tests::headless_gpu() else {
            eprintln!("no GPU adapter; skipped");
            return;
        };
        let entry =
            |binding: u32, visibility: ShaderStages, ty: BindingType| BindGroupLayoutEntry {
                binding,
                visibility,
                ty,
                count: None,
            };
        let uniform = BindingType::Buffer {
            ty: BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        };
        let model_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: None,
            entries: &[entry(
                0,
                ShaderStages::VERTEX | ShaderStages::FRAGMENT,
                uniform,
            )],
        });
        let texture_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                entry(
                    0,
                    ShaderStages::FRAGMENT,
                    BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                ),
                entry(
                    1,
                    ShaderStages::FRAGMENT,
                    BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                ),
            ],
        });
        let joint_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: None,
            entries: &[entry(0, ShaderStages::VERTEX, uniform)],
        });
        let cards = CharacterCards::with_layouts(
            &device,
            64,
            [&model_layout, &texture_layout, &joint_layout],
        );
        let mips = MirrorMips::new(&device);
        let atlas = crate::renderer::proxy_cards::atlas_with_characters(&device, &queue, &[], 1);
        assert_eq!(
            (atlas.resolution, atlas.character_rows.clone()),
            (64, vec![0])
        );

        for yaw in [0.0f32, std::f32::consts::FRAC_PI_4, -std::f32::consts::FRAC_PI_2, 2.5] {
            let mut model = vec![0.0f32; 56];
            model[..16].copy_from_slice(&Mat4::from_quat(card_turn(yaw).inverse()).to_cols_array());
            let model_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&model),
                usage: BufferUsages::UNIFORM,
            });
            let joints: Vec<f32> = (0..MAX_SKIN_JOINTS)
                .flat_map(|_| Mat4::IDENTITY.to_cols_array())
                .collect();
            let joint_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&joints),
                usage: BufferUsages::UNIFORM,
            });
            let colours: [[u8; 4]; 4] = [
                [255, 0, 0, 255],
                [0, 255, 0, 255],
                [0, 0, 255, 255],
                [255, 255, 255, 255],
            ];
            let tex = device.create_texture_with_data(
                &queue,
                &wgpu::TextureDescriptor {
                    label: None,
                    size: wgpu::Extent3d {
                        width: 4,
                        height: 1,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                },
                wgpu::util::TextureDataOrder::LayerMajor,
                bytemuck::cast_slice(&colours),
            );
            let nearest = device.create_sampler(&Default::default());
            let tex_view = tex.create_view(&Default::default());
            let model_bg = device.create_bind_group(&BindGroupDescriptor {
                label: None,
                layout: &model_layout,
                entries: &[BindGroupEntry {
                    binding: 0,
                    resource: model_buf.as_entire_binding(),
                }],
            });
            let tex_bg = device.create_bind_group(&BindGroupDescriptor {
                label: None,
                layout: &texture_layout,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::TextureView(&tex_view),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: BindingResource::Sampler(&nearest),
                    },
                ],
            });
            let joint_bg = device.create_bind_group(&BindGroupDescriptor {
                label: None,
                layout: &joint_layout,
                entries: &[BindGroupEntry {
                    binding: 0,
                    resource: joint_buf.as_entire_binding(),
                }],
            });
            let (verts, index) = legged_body();
            let vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&verts),
                usage: BufferUsages::VERTEX,
            });

            let mut caps = CapsuleUpload::default();
            caps.group_count = 1;
            caps.groups[0] = [0.0, 0.8, 0.0, 0.8];
            caps.groups[1] = [0.5, 0.5, 0.5, 1.0];
            caps.capsules[0] = [0.0, 0.2, 0.0, 0.2];
            caps.capsules[1] = [0.0, 1.4, 0.0, 0.0];
            let (centre, half) = card_box(&caps, 0, yaw).unwrap();
            let row = atlas.character_rows[0];
            let mut enc = device.create_command_encoder(&Default::default());
            let part = CardPart {
                model: &model_bg,
                texture: &tex_bg,
                joints: &joint_bg,
                joint_buffer: &joint_buf,
                source: &vb,
                vertices: &verts,
                indices: &index,
            };
            cards.record(
                &device,
                &queue,
                &mut enc,
                &mips,
                &[part],
                (centre, half),
                yaw,
                &atlas.texture,
                row,
                None,
                None,
            );
            queue.submit([enc.finish()]);

            let (w, h, texels) = cards.read_back(&device, &queue).unwrap();
            assert_eq!((w, h), (6 * 64, 64));
            let at = |card: u32, x: u32, y: u32| texels[(y * w + card * 64 + x) as usize];
            for (card, colour) in [
                (0, [0.0, 0.0, 1.0, 1.0]),
                (1, [0.0, 0.0, 1.0, 1.0]),
                (2, [1.0; 4]),
                (3, [1.0; 4]),
                (4, [1.0, 0.0, 0.0, 1.0]),
                (5, [0.0, 1.0, 0.0, 1.0]),
            ] {
                assert_eq!(at(card, 32, 32), colour, "card {card}'s middle, rig turned {yaw}");
                assert_eq!(at(card, 0, 0), [0.0; 4], "card {card}'s corner, rig turned {yaw}");
            }

            let mut u: Uniforms = bytemuck::Zeroable::zeroed();
            u.capsules = caps.capsules;
            u.capsule_groups = caps.groups;
            u.capsule_params = [1.0, -1.0, -1.0, 0.0];
            u.player_frame = [0.0, 0.0, 0.0, yaw];
            u.character_cards = [
                [centre.x, centre.y, centre.z, row as f32 + 1.0],
                [half.x, half.y, half.z, 0.0],
            ];
            let code = format!(
                "{}\n{}",
                crate::renderer::lights::wgsl_lights_block(0, 1),
                r#"
    @group(1) @binding(0) var<storage, read> rays: array<vec4<f32>>;
    @group(1) @binding(1) var<storage, read_write> out: array<vec4<f32>>;
    @compute @workgroup_size(1)
    fn look_main(@builtin(global_invocation_id) id: vec3<u32>) {
        pixel_footprint = rays[id.x * 2u].w;

        out[id.x] = capsule_reflection(rays[id.x * 2u].xyz, normalize(rays[id.x * 2u + 1u].xyz), 0.0, vec3<f32>(3.14159265), vec4<f32>(0.0));
    }
"#
            );
            let module = device.create_shader_module(ShaderModuleDescriptor {
                label: None,
                source: ShaderSource::Wgsl(code.into()),
            });
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: None,
                module: &module,
                entry_point: Some("look_main"),
                compilation_options: Default::default(),
                cache: None,
            });

            let rays: [(Vec3, Vec3, f32); 9] = [
                (Vec3::new(0.0, 0.8, 2.0), Vec3::NEG_Z, 0.0),
                (Vec3::new(0.0, 0.8, -2.0), Vec3::Z, 0.0),
                (Vec3::new(2.0, 0.8, 0.0), Vec3::NEG_X, 0.0),

                (Vec3::new(0.3, 0.8, 2.0), Vec3::NEG_Z, 0.0),

                (Vec3::new(0.0, 1.63, 2.0), Vec3::NEG_Z, 0.0),

                (Vec3::new(2.0, 0.8, 2.0), Vec3::new(-1.0, 0.0, -1.0), 0.0),

                (Vec3::new(0.2, 0.8, 2.0), Vec3::NEG_Z, 0.05),

                (Vec3::new(0.15, 1.3, 1.0), Vec3::new(0.0, -1.0, -1.0), 0.0),

                (Vec3::new(0.0, 1.3, 1.0), Vec3::new(0.0, -1.0, -1.0), 0.0),
            ];

            let to_player = card_turn(yaw).inverse();
            let packed: Vec<[f32; 4]> = rays
                .iter()
                .map(|(p, d, f)| (to_player * *p, to_player * *d, f))
                .flat_map(|(p, d, f)| [[p.x, p.y, p.z, *f], [d.x, d.y, d.z, 0.0]])
                .collect();
            let camera = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::bytes_of(&u),
                usage: BufferUsages::UNIFORM,
            });
            let ray_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&packed),
                usage: BufferUsages::STORAGE,
            });
            let size = (rays.len() * 16) as u64;
            let out_buf = device.create_buffer(&BufferDescriptor {
                label: None,
                size,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let read = device.create_buffer(&BufferDescriptor {
                label: None,
                size,
                usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let (_, samp) = crate::renderer::uniforms::default_probe_cube(&device);

            let lights_uniform = crate::renderer::lights::LightsUniform::new(&device);
            let shadow_map = crate::renderer::shadow::ShadowMap::with_dimension(&device, 64);
            let g0 = device.create_bind_group(&BindGroupDescriptor {
                label: None,
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: camera.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: lights_uniform.buffer().as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 3,
                        resource: BindingResource::Sampler(shadow_map.sampler()),
                    },
                    BindGroupEntry {
                        binding: 4,
                        resource: BindingResource::TextureView(shadow_map.spot_depth_view()),
                    },
                    BindGroupEntry {
                        binding: 6,
                        resource: BindingResource::Sampler(&samp),
                    },
                    BindGroupEntry {
                        binding: 12,
                        resource: BindingResource::TextureView(&atlas.view),
                    },
                ],
            });
            let g1 = device.create_bind_group(&BindGroupDescriptor {
                label: None,
                layout: &pipeline.get_bind_group_layout(1),
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: ray_buf.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: out_buf.as_entire_binding(),
                    },
                ],
            });
            let mut enc = device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &g0, &[]);
                pass.set_bind_group(1, &g1, &[]);
                pass.dispatch_workgroups(rays.len() as u32, 1, 1);
            }
            enc.copy_buffer_to_buffer(&out_buf, 0, &read, 0, size);
            queue.submit([enc.finish()]);
            read.slice(..).map_async(wgpu::MapMode::Read, |_| {});
            let _ = device.poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            });
            let got: Vec<[f32; 4]> =
                bytemuck::cast_slice(&read.slice(..).get_mapped_range().unwrap()).to_vec();
            for (i, g) in got.iter().enumerate() {
                eprintln!("rig turned {yaw}, ray {i}: {g:?}");
            }
            let near =
                |a: [f32; 4], b: [f32; 4], tol: f32| a.iter().zip(b).all(|(x, y)| (x - y).abs() <= tol);
            assert!(
                near(got[0], [1.0, 0.0, 0.0, 1.0], 0.01),
                "front, rig turned {yaw}: {:?}",
                got[0]
            );
            assert!(
                near(got[1], [0.0, 1.0, 0.0, 1.0], 0.01),
                "back, rig turned {yaw}: {:?}",
                got[1]
            );
            assert!(
                near(got[2], [0.0, 0.0, 1.0, 1.0], 0.01),
                "side, rig turned {yaw}: {:?}",
                got[2]
            );
            assert!(
                near(got[3], [0.0; 4], 0.01),
                "beside the body, rig turned {yaw}: {:?}",
                got[3]
            );
            assert!(near(got[4], [0.0; 4], 0.01), "over its head, rig turned {yaw}: {:?}", got[4]);
            assert!(
                near(got[5], [0.5, 0.0, 0.5, 1.0], 0.02),
                "between two axes, rig turned {yaw}: {:?}",
                got[5]
            );
            let edge = got[6];
            assert!(
                edge[3] > 0.3 && edge[3] < 0.7,
                "a blurred outline, rig turned {yaw}: {edge:?}"
            );
            assert!(
                (edge[0] / edge[3] - 1.0).abs() < 0.02 && edge[1] < 0.01 && edge[2] < 0.01,
                "the front's colour, rig turned {yaw}: {edge:?}"
            );
            assert!(
                near(got[7], [0.0; 4], 0.01),
                "past the legs on a slant, rig turned {yaw}: {:?}",
                got[7]
            );
            assert!(
                got[8][3] > 0.97 && got[8][0] > 0.97,
                "through the legs on a slant, rig turned {yaw}: {:?}",
                got[8]
            );
        }
    }

    #[test]
    fn simplifying_keeps_islands_and_joints_apart() {
        let (verts, index) = box_body();
        let (v, i) = simplify(&verts, &index, 0.05);
        assert_eq!((v.len(), i.len()), (24, 36), "every face kept, none merged into another");
        for w in &v {
            let twins = v.iter().filter(|o| o.position == w.position).count();
            assert_eq!(twins, 3, "a corner is three faces' vertices at one place: {:?}", w.position);
        }

        let quad = |joint: u32, x: f32| -> Vec<SkinnedMeshVertex> {
            [[x, 0.0, 0.0], [x + 0.2, 0.0, 0.0], [x + 0.2, 0.2, 0.0], [x, 0.2, 0.0]]
                .map(|p| SkinnedMeshVertex { position: p, normal: [0.0, 0.0, 1.0], uv: [0.0; 2], joint_ids: [joint, 0, 0, 0], joint_weights: [1.0, 0.0, 0.0, 0.0] })
                .to_vec()
        };
        let mut both = quad(0, 0.0);
        both.extend(quad(1, 0.0));
        let tris = vec![0, 1, 2, 0, 2, 3, 4, 5, 6, 4, 6, 7];
        let (v, i) = simplify(&both, &tris, 0.05);
        assert_eq!(v.len(), 8, "no vertex shared between the joints");
        assert_eq!(i.len(), 12);

        let fan: Vec<SkinnedMeshVertex> = (0..18)
            .map(|k| {
                let a = k as f32 * 0.3;
                SkinnedMeshVertex { position: [0.01 + 0.005 * a.cos(), 0.01 + 0.005 * a.sin(), 0.0], normal: [0.0; 3], uv: [0.0; 2], joint_ids: [0; 4], joint_weights: [1.0, 0.0, 0.0, 0.0] }
            })
            .collect();
        let slivers: Vec<u32> = (1..17).flat_map(|k| [0, k, k + 1]).collect();
        let (v, i) = simplify(&fan, &slivers, 0.05);
        assert_eq!((v.len(), i.len()), (1, 0), "smaller than a cell: gone");
    }

    #[test]
    fn the_box_holds_the_body_with_its_margin() {
        let mut caps = CapsuleUpload::default();
        assert!(card_box(&caps, 0, 0.0).is_none());
        caps.group_count = 1;
        caps.groups[1] = [0.5, 0.5, 0.5, 2.0];
        caps.capsules[0] = [0.0, 0.2, 0.0, 0.1];
        caps.capsules[1] = [0.0, 1.6, 0.0, 0.0];
        caps.capsules[2] = [0.3, 1.3, 0.1, 0.05];
        caps.capsules[3] = [0.7, 1.3, 0.1, 0.0];
        let (c, h) = card_box(&caps, 0, 0.0).unwrap();
        let (lo, hi) = (c - h, c + h);
        let m = CARD_BOX_MARGIN;
        assert!(
            (lo - Vec3::new(-0.1 - m, 0.1 - m, -0.1 - m))
                .abs()
                .max_element()
                < 1e-5,
            "{lo}"
        );
        assert!(
            (hi - Vec3::new(0.75 + m, 1.7 + m, 0.15 + m))
                .abs()
                .max_element()
                < 1e-5,
            "{hi}"
        );

        let quarter = std::f32::consts::FRAC_PI_2;
        let (c, h) = card_box(&caps, 0, quarter).unwrap();
        let world_c = card_turn(quarter) * c;
        let (lo, hi) = (world_c - h, world_c + h);
        assert!(
            (lo - Vec3::new(-0.1 - m, 0.1 - m, -0.75 - m)).abs().max_element() < 1e-5
                && (hi - Vec3::new(0.15 + m, 1.7 + m, 0.1 + m)).abs().max_element() < 1e-5,
            "turned a quarter: {lo} to {hi}"
        );
    }

    #[test]
    fn a_body_posed_once_draws_the_cards_it_drew_skinned() {
        use crate::renderer::brush_pipeline::probe_pass::MirrorMips;
        use wgpu::util::DeviceExt;
        let Some((device, queue)) = crate::renderer::terrain_pipeline::tests::headless_gpu() else {
            eprintln!("no GPU adapter; skipped");
            return;
        };
        let entry = |binding: u32, visibility: ShaderStages, ty: BindingType| BindGroupLayoutEntry { binding, visibility, ty, count: None };
        let uniform = BindingType::Buffer { ty: BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None };
        let model_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: None,
            entries: &[entry(0, ShaderStages::VERTEX | ShaderStages::FRAGMENT, uniform)],
        });
        let texture_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                entry(
                    0,
                    ShaderStages::FRAGMENT,
                    BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                ),
                entry(1, ShaderStages::FRAGMENT, BindingType::Sampler(wgpu::SamplerBindingType::Filtering)),
            ],
        });
        let joint_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: None,
            entries: &[entry(0, ShaderStages::VERTEX, uniform)],
        });
        let mips = MirrorMips::new(&device);
        let atlas = crate::renderer::proxy_cards::atlas_with_characters(&device, &queue, &[], 1);
        let mut model = vec![0.0f32; 56];
        model[..16].copy_from_slice(&Mat4::IDENTITY.to_cols_array());
        let model_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&model),
            usage: BufferUsages::UNIFORM,
        });

        let mut joints: Vec<f32> = (0..MAX_SKIN_JOINTS).flat_map(|_| Mat4::IDENTITY.to_cols_array()).collect();
        joints[..16].copy_from_slice(
            &(Mat4::from_translation(Vec3::new(0.05, 0.1, 0.0)) * Mat4::from_rotation_y(0.4)).to_cols_array(),
        );
        joints[16..32].copy_from_slice(&Mat4::from_rotation_x(0.3).to_cols_array());
        let joint_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&joints),
            usage: BufferUsages::UNIFORM,
        });
        let colours: [[u8; 4]; 4] = [[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255], [255, 255, 255, 255]];
        let tex = device.create_texture_with_data(
            &queue,
            &wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d { width: 4, height: 1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            bytemuck::cast_slice(&colours),
        );
        let nearest = device.create_sampler(&Default::default());
        let tex_view = tex.create_view(&Default::default());
        let model_bg = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &model_layout,
            entries: &[BindGroupEntry { binding: 0, resource: model_buf.as_entire_binding() }],
        });
        let tex_bg = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &texture_layout,
            entries: &[
                BindGroupEntry { binding: 0, resource: BindingResource::TextureView(&tex_view) },
                BindGroupEntry { binding: 1, resource: BindingResource::Sampler(&nearest) },
            ],
        });
        let joint_bg = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &joint_layout,
            entries: &[BindGroupEntry { binding: 0, resource: joint_buf.as_entire_binding() }],
        });
        let (mut verts, index) = legged_body();
        for v in verts.iter_mut().filter(|v| v.position[1] < 0.6) {
            v.joint_ids = [1, 0, 0, 0];
        }
        let vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&verts),
            usage: BufferUsages::VERTEX,
        });
        let mut caps = CapsuleUpload::default();
        caps.group_count = 1;
        caps.groups[0] = [0.0, 0.8, 0.0, 0.8];
        caps.groups[1] = [0.5, 0.5, 0.5, 1.0];
        caps.capsules[0] = [0.0, 0.2, 0.0, 0.3];
        caps.capsules[1] = [0.0, 1.4, 0.0, 0.0];
        let (centre, half) = card_box(&caps, 0, 0.0).unwrap();
        let part = CardPart {
            model: &model_bg,
            texture: &tex_bg,
            joints: &joint_bg,
            joint_buffer: &joint_buf,
            source: &vb,
            vertices: &verts,
            indices: &index,
        };
        let skin = crate::renderer::skin_compute::SkinCompute::new(&device);
        let draw = |posed: bool| {
            let cards = CharacterCards::with_layouts(&device, 64, [&model_layout, &texture_layout, &joint_layout]);
            let mut enc = device.create_command_encoder(&Default::default());
            let skin = posed.then_some(&skin);
            cards.record(&device, &queue, &mut enc, &mips, &[part], (centre, half), 0.0, &atlas.texture, atlas.character_rows[0], None, skin);
            queue.submit([enc.finish()]);
            cards.read_back(&device, &queue).unwrap()
        };
        let (w, h, skinned) = draw(false);
        let (_, _, posed) = draw(true);
        let covered = skinned.iter().filter(|t| t[3] > 0.5).count();
        assert!(covered > (w * h / 20) as usize, "the body must cover the cards: {covered} of {}", w * h);
        let worst = skinned
            .iter()
            .zip(&posed)
            .map(|(a, b)| (0..4).map(|c| (a[c] - b[c]).abs()).fold(0.0f32, f32::max))
            .fold(0.0f32, f32::max);
        let differing = skinned.iter().zip(&posed).filter(|(a, b)| (0..4).any(|c| (a[c] - b[c]).abs() > 1e-3)).count();
        eprintln!("posed against skinned: {differing} of {} texels differ, worst {worst:.2e}", w * h);
        assert!(differing <= (w * h / 1000) as usize, "{differing} texels differ, worst {worst}");
    }
}
