//! THE PLAYER ON CARDS, EVERY FRAME: six pictures of their body, one looking in
//! through each face of a box round it, from which a reflection that meets the
//! player takes their real colours and outline instead of a capsule's one mean
//! colour (user, 2026-09-30: "build the character probe"). The capsules stay
//! as the cheap test for whether a reflected ray passes a body at all; where
//! one does, the cards say what it shows. See `character_card_look` and
//! `capsule_reflection` in the lights block.
//!
//! The fixtures' cards are baked (`proxy_cards`); a body moves, so its cards
//! are drawn each frame -- six orthographic views of the player's own
//! mirror-only mesh, a few thousand pixels in all -- into a small target,
//! given their blur levels by the floor mirror's mip pass (`MirrorMips`), and
//! copied into the rows the card atlas keeps for them
//! (`proxy_cards::atlas_with_characters`). The probe pass already samples that
//! atlas and is at its sampled-texture limit, so the cards could not have a
//! texture of their own.
//!
//! WHAT A CARD HOLDS: the body's ALBEDO, premultiplied by coverage, with the
//! coverage in alpha -- not its lit colour. A capsule's reflection is lit by
//! the light arriving at the reflecting surface, and the cards are read the
//! same way, so the same light falls on both and switching between them
//! changes the shape and the colours and nothing else. Coverage averages
//! through the blur levels, so a body seen far off, or in a rough surface,
//! fades at its outline as the footprint says rather than as a capsule
//! blurred by a fixed 15 cm.
//!
//! THE CARDS' FRAME is the fixtures' convention, so one lookup reads both
//! kinds of row alike: card `2a` looks in through the box's `+a` face and
//! `2a + 1` through its `-a` face; `u` runs along axis `a + 1` and `v` along
//! `a + 2`, `v = 0` the top row; the depth is 0 at the face a card looks
//! through. The box is square to the player's frame, round the body's
//! capsules -- which card sees the face does not matter, only that six do.

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

/// How far past the body's capsules the box reaches: clothes and hands stand
/// a little outside them.
pub const CARD_BOX_MARGIN: f32 = 0.06;

/// Blur levels a character's cards carry above the first, as `MirrorMips`
/// makes them: the most a reflection reads them at (`CHARACTER_CARD_MAX_LOD`
/// in the lights block). Level 4 of a 64-texel card is a 16th of the body
/// across, as soft as any reflection of it needs.
pub const CARD_MIPS: u32 = MIRROR_MIPS;

/// Bytes between one card's matrix and the next in the uniform: the
/// dynamic-offset alignment every device allows.
const CARD_STRIDE: u64 = 256;

/// THE BODY AS THE CARDS NEED IT: its vertices clustered `cell` apart in its
/// bind pose, and the triangles that collapse, or repeat one already kept,
/// dropped. Two vertices merge only if they share a cell, their strongest
/// joint -- else a limb would be tied to what it passes -- and their UV island
/// (the triangles they are connected through), else a triangle would stretch
/// its texture across the atlas from one island to another: a box's corners
/// merged gave its side the front's colour. Every cluster in a cell stands at
/// the mean of ALL the cell's vertices, so the two sides of a UV seam, kept
/// apart, still meet; a cluster's UV is its own members' mean.
///
/// The avatar is 24,634 vertices and 33,894 triangles: drawn six times into
/// cards 64 texels across, nearly every triangle was smaller than a texel,
/// and the six views cost 0.56 ms a frame on the headset (2026-10-01). At a
/// texel of its height it is about a third of that, and the cards cannot
/// show the difference.
pub fn simplify(
    vertices: &[SkinnedMeshVertex],
    indices: &[u32],
    cell: f32,
) -> (Vec<SkinnedMeshVertex>, Vec<u32>) {
    use std::collections::{HashMap, HashSet};
    // The UV islands: the vertices one triangle or a chain of them joins.
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

/// One primitive of a body as the cards draw it: the bind groups the lit
/// pass draws it with, and its vertices as loaded.
#[derive(Clone, Copy)]
pub struct CardPart<'a> {
    pub model: &'a BindGroup,
    pub texture: &'a BindGroup,
    pub joints: &'a BindGroup,
    /// The primitive's own vertex buffer: which simplified copy is its.
    pub source: &'a Buffer,
    pub vertices: &'a [SkinnedMeshVertex],
    pub indices: &'a [u32],
}

/// A primitive's [`simplify`]d copy on the GPU, and its index count.
struct CardMesh {
    vertices: Buffer,
    indices: Buffer,
    count: u32,
}

/// The box round character `g`'s capsules, padded by [`CARD_BOX_MARGIN`], as
/// its centre and half size; `None` for a character with none.
pub fn card_box(capsules: &CapsuleUpload, g: usize) -> Option<(Vec3, Vec3)> {
    if g >= capsules.group_count as usize {
        return None;
    }
    let count = capsules.groups[g * 2 + 1][3] as usize;
    let mut lo = Vec3::splat(f32::MAX);
    let mut hi = Vec3::splat(f32::MIN);
    for k in 0..count.min(CAPSULES_PER_GROUP) {
        let i = g * CAPSULES_PER_GROUP + k;
        let (a, b) = (capsules.capsules[i * 2], capsules.capsules[i * 2 + 1]);
        let r = Vec3::splat(a[3]);
        for end in [Vec3::new(a[0], a[1], a[2]), Vec3::new(b[0], b[1], b[2])] {
            lo = lo.min(end - r);
            hi = hi.max(end + r);
        }
    }
    if count == 0 || !lo.is_finite() || !hi.is_finite() {
        return None;
    }
    let (lo, hi) = (lo - CARD_BOX_MARGIN, hi + CARD_BOX_MARGIN);
    Some(((lo + hi) * 0.5, ((hi - lo) * 0.5).max(Vec3::splat(0.05))))
}

/// Card `k`'s view of the box `centre`, `half`: a point to the card's clip
/// space -- x = 2u - 1, y = 1 - 2v, z = the depth from the face it looks
/// through. See the module notes.
pub fn card_matrix(k: usize, centre: Vec3, half: Vec3) -> Mat4 {
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

/// The six views' shader: the body posed exactly as the lit pass poses it,
/// each fragment its albedo with full coverage. The body's own texture's
/// cut-outs are cut out.
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

/// One character's cards: the target their six views are drawn into, its
/// blur levels, and the pass that draws them.
pub struct CharacterCards {
    resolution: u32,
    target: Texture,
    /// Level 0 drawn into and read by the blur; then the blur levels.
    levels: Vec<TextureView>,
    _depth: Texture,
    depth_view: TextureView,
    pipeline: RenderPipeline,
    matrices: Buffer,
    matrices_group: BindGroup,
    /// Each body primitive's simplified copy, by the primitive's own vertex
    /// buffer: made the first frame it is drawn, dropped the first it is not.
    meshes: std::sync::Mutex<std::collections::HashMap<Buffer, CardMesh>>,
}

impl CharacterCards {
    /// Cards `resolution` texels across, drawn with `skinned`'s model,
    /// texture and joint layouts -- the bind groups every skinned draw
    /// already has.
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

    /// [`Self::new`] with the model, texture and joint layouts given: the
    /// shader reads a model's matrix, its colour texture and sampler (its
    /// first two bindings) and its joints.
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
            // Both sides: a card may look at a limb from inside its outline.
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
            matrices,
            matrices_group,
            meshes: Default::default(),
        }
    }

    pub fn resolution(&self) -> u32 {
        self.resolution
    }

    /// THE CARDS AS LAST DRAWN, their first level read back: `(width, height,
    /// texels)`, linear RGBA row after row, the six cards side by side. Waits
    /// on the GPU -- for a test or a diagnosis, never a frame.
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

    /// Draws the six cards of the body `parts` in the box `centre`, `half`,
    /// from each part's [`simplify`]d copy; blurs
    /// them; and copies every level into `atlas`'s `row`
    /// (`proxy_cards::CardAtlas::character_rows`). `timer`: a pass timer's
    /// slot for the drawing and the next for the blur levels.
    #[allow(clippy::too_many_arguments)]
    pub fn record(
        &self,
        device: &Device,
        queue: &Queue,
        encoder: &mut CommandEncoder,
        mips: &MirrorMips,
        parts: &[CardPart],
        (centre, half): (Vec3, Vec3),
        atlas: &Texture,
        row: u32,
        timer: Option<(&crate::renderer::pass_timers::PassTimers, usize)>,
    ) {
        let mut meshes = self.meshes.lock().unwrap_or_else(|e| e.into_inner());
        meshes.retain(|source, _| parts.iter().any(|part| part.source == source));
        if parts.iter().any(|part| !meshes.contains_key(part.source)) {
            // A texel of the card the body fills, in its bind pose's own
            // units: the whole body's longest side over the card's texels --
            // the same for every part, small ones included.
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
                        usage: BufferUsages::VERTEX,
                    }),
                    indices: wgpu::util::DeviceExt::create_buffer_init(device, &wgpu::util::BufferInitDescriptor {
                        label: Some("character_card_mesh_ib"),
                        contents: bytemuck::cast_slice(&indices),
                        usage: BufferUsages::INDEX,
                    }),
                    count: indices.len() as u32,
                }
                });
            }
        }
        let mut bytes = vec![0u8; (CARD_STRIDE * CARD_FACES as u64) as usize];
        for k in 0..CARD_FACES {
            let m = card_matrix(k, centre, half).to_cols_array();
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
            pass.set_pipeline(&self.pipeline);
            for part in parts {
                let Some(mesh) = meshes.get(part.source) else {
                    continue;
                };
                pass.set_bind_group(1, part.model, &[]);
                pass.set_bind_group(2, part.texture, &[]);
                pass.set_bind_group(3, part.joints, &[]);
                pass.set_vertex_buffer(0, mesh.vertices.slice(..));
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

    /// The cards' frame is the fixtures': card 2a looks in through the +a
    /// face, u along a + 1, v along a + 2 from the top, depth 0 at the face
    /// it looks through -- what `probe_card_colour` and
    /// `character_card_look` both read by.
    #[test]
    fn each_card_sees_the_box_as_the_lookup_reads_it() {
        let (centre, half) = (Vec3::new(1.0, 2.0, -3.0), Vec3::new(0.4, 0.9, 0.3));
        for k in 0..CARD_FACES {
            let (a, plus) = (k / 2, k % 2 == 0);
            let (a1, a2) = ((a + 1) % 3, (a + 2) % 3);
            let m = card_matrix(k, centre, half);
            // A point at (u, v) = (0.25, 0.75) of the card, on the face it
            // looks through and on the far one.
            let mut on_face = centre;
            on_face[a1] += (0.25 - 0.5) * 2.0 * half[a1];
            on_face[a2] += (0.75 - 0.5) * 2.0 * half[a2];
            on_face[a] += if plus { half[a] } else { -half[a] };
            let mut far = on_face;
            far[a] = 2.0 * centre[a] - on_face[a];
            let (c, f) = (m.project_point3(on_face), m.project_point3(far));
            let uv = |p: Vec3| ((p.x + 1.0) * 0.5, (1.0 - p.y) * 0.5);
            let (u, v) = uv(c);
            assert!(
                (u - 0.25).abs() < 1e-5 && (v - 0.75).abs() < 1e-5,
                "card {k}: uv {:?}",
                (u, v)
            );
            assert!(
                c.z.abs() < 1e-5 && (f.z - 1.0).abs() < 1e-5,
                "card {k}: depths {} and {}",
                c.z,
                f.z
            );
        }
    }

    /// A box-shaped body -- 0.4 m wide, 1.6 m tall, 0.2 m deep -- its front
    /// (+z) red, back green, sides blue, top and bottom white, all from one
    /// four-texel texture.
    fn box_body() -> (Vec<SkinnedMeshVertex>, Vec<u32>) {
        let (lo, hi) = (Vec3::new(-0.2, 0.0, -0.1), Vec3::new(0.2, 1.6, 0.1));
        let mut verts = Vec::new();
        let mut index = Vec::new();
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
        (verts, index)
    }

    /// END TO END ON THE GPU: the box body drawn onto its cards by `record`
    /// -- its blur levels made and copied into an atlas's character rows --
    /// and read back by `capsule_reflection` in the lights block, with one
    /// capsule standing for the body. A ray meeting a side takes that side's
    /// colour; a ray the capsule's soft edge reaches but the body does not
    /// takes nothing; a ray between two axes mixes their two views; and a
    /// footprint wider than a texel reads the outline as blurred.
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

        let mut model = vec![0.0f32; 56];
        model[..16].copy_from_slice(&Mat4::IDENTITY.to_cols_array());
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
        let (verts, index) = box_body();
        let vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&verts),
            usage: BufferUsages::VERTEX,
        });

        // One capsule down the body's middle, fatter than it front to back.
        let mut caps = CapsuleUpload::default();
        caps.group_count = 1;
        caps.groups[0] = [0.0, 0.8, 0.0, 0.8];
        caps.groups[1] = [0.5, 0.5, 0.5, 1.0];
        caps.capsules[0] = [0.0, 0.2, 0.0, 0.2];
        caps.capsules[1] = [0.0, 1.4, 0.0, 0.0];
        let (centre, half) = card_box(&caps, 0).unwrap();
        let row = atlas.character_rows[0];
        let mut enc = device.create_command_encoder(&Default::default());
        let part = CardPart {
            model: &model_bg,
            texture: &tex_bg,
            joints: &joint_bg,
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
            &atlas.texture,
            row,
            None,
        );
        queue.submit([enc.finish()]);
        // The cards themselves: each one's middle the side it looks at, its
        // corners empty -- the body is narrower than its box.
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
            assert_eq!(at(card, 32, 32), colour, "card {card}'s middle");
            assert_eq!(at(card, 0, 0), [0.0; 4], "card {card}'s corner");
        }

        let mut u: Uniforms = bytemuck::Zeroable::zeroed();
        u.capsules = caps.capsules;
        u.capsule_groups = caps.groups;
        u.capsule_params = [1.0, -1.0, -1.0, 0.0];
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
    // Lit by pi: the reflection is then the albedo it shows.
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
        // (from, direction, pixel footprint)
        let rays: [(Vec3, Vec3, f32); 7] = [
            (Vec3::new(0.0, 0.8, 2.0), Vec3::NEG_Z, 0.0),
            (Vec3::new(0.0, 0.8, -2.0), Vec3::Z, 0.0),
            (Vec3::new(2.0, 0.8, 0.0), Vec3::NEG_X, 0.0),
            // Inside the capsule's soft edge, outside the body.
            (Vec3::new(0.3, 0.8, 2.0), Vec3::NEG_Z, 0.0),
            // Over the body's head, inside its box and its capsule's edge.
            (Vec3::new(0.0, 1.63, 2.0), Vec3::NEG_Z, 0.0),
            // Between two axes: the front's view and the side's, half each.
            (Vec3::new(2.0, 0.8, 2.0), Vec3::new(-1.0, 0.0, -1.0), 0.0),
            // At the body's edge with a footprint of a few texels.
            (Vec3::new(0.2, 0.8, 2.0), Vec3::NEG_Z, 0.05),
        ];
        let packed: Vec<[f32; 4]> = rays
            .iter()
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
        let g0 = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: camera.as_entire_binding(),
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
            eprintln!("ray {i}: {g:?}");
        }
        let near =
            |a: [f32; 4], b: [f32; 4], tol: f32| a.iter().zip(b).all(|(x, y)| (x - y).abs() <= tol);
        assert!(
            near(got[0], [1.0, 0.0, 0.0, 1.0], 0.01),
            "front: {:?}",
            got[0]
        );
        assert!(
            near(got[1], [0.0, 1.0, 0.0, 1.0], 0.01),
            "back: {:?}",
            got[1]
        );
        assert!(
            near(got[2], [0.0, 0.0, 1.0, 1.0], 0.01),
            "side: {:?}",
            got[2]
        );
        assert!(
            near(got[3], [0.0; 4], 0.01),
            "beside the body: {:?}",
            got[3]
        );
        assert!(near(got[4], [0.0; 4], 0.01), "over its head: {:?}", got[4]);
        assert!(
            near(got[5], [0.5, 0.0, 0.5, 1.0], 0.02),
            "between two axes: {:?}",
            got[5]
        );
        let edge = got[6];
        assert!(
            edge[3] > 0.3 && edge[3] < 0.7,
            "a blurred outline: {edge:?}"
        );
        assert!(
            (edge[0] / edge[3] - 1.0).abs() < 0.02 && edge[1] < 0.01 && edge[2] < 0.01,
            "the front's colour: {edge:?}"
        );
    }

    /// What clustering must keep: a box's faces, each its own UV island,
    /// stay apart (each corner one vertex a face, all three at one place);
    /// two faces bound to different joints share nothing even where they
    /// touch; and a fan of triangles finer than a cell collapses to nothing.
    #[test]
    fn simplifying_keeps_islands_and_joints_apart() {
        let (verts, index) = box_body();
        let (v, i) = simplify(&verts, &index, 0.05);
        assert_eq!((v.len(), i.len()), (24, 36), "every face kept, none merged into another");
        for w in &v {
            let twins = v.iter().filter(|o| o.position == w.position).count();
            assert_eq!(twins, 3, "a corner is three faces' vertices at one place: {:?}", w.position);
        }
        // Two quads at one place, one on joint 0 and one on joint 1, each
        // fine enough to collapse to a point were they one island.
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
        // A fan of 16 slivers inside one cell.
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

    /// The box holds every capsule with the margin, and a character with no
    /// capsules has none.
    #[test]
    fn the_box_holds_the_body_with_its_margin() {
        let mut caps = CapsuleUpload::default();
        assert!(card_box(&caps, 0).is_none());
        caps.group_count = 1;
        caps.groups[1] = [0.5, 0.5, 0.5, 2.0];
        caps.capsules[0] = [0.0, 0.2, 0.0, 0.1];
        caps.capsules[1] = [0.0, 1.6, 0.0, 0.0];
        caps.capsules[2] = [0.3, 1.3, 0.1, 0.05];
        caps.capsules[3] = [0.7, 1.3, 0.1, 0.0];
        let (c, h) = card_box(&caps, 0).unwrap();
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
    }
}
