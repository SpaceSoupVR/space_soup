//! SKIN ONCE A FRAME.
//!
//! The player's body is drawn many times a frame outside the eye passes: into
//! the moving-objects map's sun tile and its near tile and the characters'
//! tiles (`shadow::ShadowMap::record_moving`), into the spot atlas when a spot
//! reaches it, and onto the six reflection cards (`character_cards`). Each of
//! those draws skinned every vertex again -- four joint matrices read at
//! indices the vertex supplies, which the driver keeps out of constant memory
//! -- and a tile GPU runs a draw's vertex stage twice, once to bin it and once
//! to render it. Measured on the Quest 3 (exp62, 2026-10-02): the moving-objects
//! pass 0.62 ms a frame and the cards' 0.44, both mostly binning.
//!
//! Here each body primitive is posed ONCE, by a compute pass, into a buffer of
//! positions that every one of those draws reads as a rigid mesh would: 16
//! bytes a vertex to fetch, where the skinned draw fetched 64 and four
//! matrices. `Levers::skin_once`.
//!
//! The arithmetic is the skinned shaders' term for term -- `(M[j] * p) * w`
//! summed in the same order, kept as the whole `vec4` -- so the body that casts
//! the shadow is the body that is drawn (`shadow::SHADOW_SHADER`'s `vs_skinned`
//! says why that matters: a shadow from a differently posed body reads as the
//! shadow lagging).

use std::collections::HashMap;

use wgpu::*;

use super::mesh::MAX_SKIN_JOINTS;

/// Bytes a posed vertex takes: its skinned position, `w` included, in the
/// model's frame -- the model matrix still applies, as to any rigid mesh.
pub const POSED_STRIDE: u64 = 16;

/// Words of `SkinnedMeshVertex` (position, normal, uv, joint ids, weights).
const SOURCE_WORDS: u32 = 16;

/// Threads a workgroup.
const GROUP: u32 = 64;

/// The vertex layout a posed buffer is drawn with: its position at location 0.
pub fn posed_layout() -> VertexBufferLayout<'static> {
    const ATTRS: [VertexAttribute; 1] = vertex_attr_array![0 => Float32x4];
    VertexBufferLayout { array_stride: POSED_STRIDE, step_mode: VertexStepMode::Vertex, attributes: &ATTRS }
}

/// The compute pipeline that poses a skinned primitive.
pub struct SkinCompute {
    pipeline: ComputePipeline,
    layout: BindGroupLayout,
}

/// One primitive's posed copy: written by [`SkinCompute::dispatch`], drawn as a
/// rigid mesh with the primitive's own index buffer and model bind group.
#[derive(Clone)]
pub struct Posed {
    pub positions: Buffer,
    bind_group: BindGroup,
    vertices: u32,
}

impl SkinCompute {
    pub fn new(device: &Device) -> Self {
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("skin_compute"),
            source: ShaderSource::Wgsl(shader().into()),
        });
        let buffer = |binding: u32, ty: BufferBindingType| BindGroupLayoutEntry {
            binding,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("skin_compute_bgl"),
            entries: &[
                buffer(0, BufferBindingType::Uniform),
                buffer(1, BufferBindingType::Storage { read_only: true }),
                buffer(2, BufferBindingType::Storage { read_only: false }),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("skin_compute_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("skin_compute"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("cs_skin"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        Self { pipeline, layout }
    }

    /// A posed copy of the skinned primitive whose vertex buffer is `source`
    /// (`vertices` of them, made with `STORAGE` usage), posed by `joints`.
    pub fn posed_for(&self, device: &Device, source: &Buffer, vertices: u32, joints: &Buffer) -> Posed {
        let positions = device.create_buffer(&BufferDescriptor {
            label: Some("skin_posed_vb"),
            size: (vertices as u64 * POSED_STRIDE).max(POSED_STRIDE),
            // COPY_SRC: read back by tests and diagnosis.
            usage: BufferUsages::STORAGE | BufferUsages::VERTEX | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("skin_compute_bg"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry { binding: 0, resource: joints.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: source.as_entire_binding() },
                BindGroupEntry { binding: 2, resource: positions.as_entire_binding() },
            ],
        });
        Posed { positions, bind_group, vertices }
    }

    /// Poses every one of `posed` from its joints as they are now: one pass,
    /// recorded before anything that draws them.
    pub fn dispatch<'p>(&self, encoder: &mut CommandEncoder, posed: impl IntoIterator<Item = &'p Posed>) {
        let mut posed = posed.into_iter().peekable();
        if posed.peek().is_none() {
            return;
        }
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor { label: Some("skin_compute"), timestamp_writes: None });
        pass.set_pipeline(&self.pipeline);
        for p in posed {
            pass.set_bind_group(0, &p.bind_group, &[]);
            pass.dispatch_workgroups(p.vertices.div_ceil(GROUP), 1, 1);
        }
    }
}

/// Each skinned primitive's posed copy, by its vertex buffer and the joint
/// buffer that poses it: made the first frame it is drawn, dropped the first
/// it is not.
#[derive(Default)]
pub struct PosedCache {
    posed: HashMap<(Buffer, Buffer), Posed>,
}

impl PosedCache {
    /// The posed copy of `source` (`vertices` long) by `joints`, made if new.
    pub fn posed(&mut self, skin: &SkinCompute, device: &Device, source: &Buffer, vertices: u32, joints: &Buffer) -> Posed {
        self.posed
            .entry((source.clone(), joints.clone()))
            .or_insert_with(|| skin.posed_for(device, source, vertices, joints))
            .clone()
    }

    /// Drops every copy not among `live`.
    pub fn keep_only(&mut self, live: &[(Buffer, Buffer)]) {
        self.posed.retain(|key, _| live.contains(key));
    }
}

fn shader() -> String {
    format!(
        r#"
struct JointMatrices {{ mats: array<mat4x4<f32>, {MAX_SKIN_JOINTS}> }}
@group(0) @binding(0) var<uniform> joints: JointMatrices;
// The skinned primitive's own vertex buffer, `SkinnedMeshVertex` {SOURCE_WORDS} words
// a vertex: position 0-2, normal 3-5, uv 6-7, joint ids 8-11, weights 12-15.
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<vec4<f32>>;

@compute @workgroup_size({GROUP})
fn cs_skin(@builtin(global_invocation_id) id: vec3<u32>) {{
    let i = id.x;
    if (i >= arrayLength(&dst)) {{
        return;
    }}
    let b = i * {SOURCE_WORDS}u;
    let p = vec4<f32>(bitcast<f32>(src[b]), bitcast<f32>(src[b + 1u]), bitcast<f32>(src[b + 2u]), 1.0);
    let j = vec4<u32>(src[b + 8u], src[b + 9u], src[b + 10u], src[b + 11u]);
    let w = vec4<f32>(
        bitcast<f32>(src[b + 12u]), bitcast<f32>(src[b + 13u]),
        bitcast<f32>(src[b + 14u]), bitcast<f32>(src[b + 15u]),
    );
    // Term for term the skinned shaders' sum (`shadow::vs_skinned`).
    dst[i] =
        (joints.mats[j.x] * p) * w.x +
        (joints.mats[j.y] * p) * w.y +
        (joints.mats[j.z] * p) * w.z +
        (joints.mats[j.w] * p) * w.w;
}}
"#
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use wgpu::util::DeviceExt;

    /// The pose a GPU computes is the pose the skinned shaders compute: one
    /// vertex weighted across two joints, posed on a real adapter and read back.
    #[test]
    fn a_posed_vertex_is_the_skinned_vertex() {
        let Some((device, queue)) = crate::renderer::terrain_pipeline::tests::headless_gpu() else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        let skin = SkinCompute::new(&device);
        let a = glam::Mat4::from_translation(glam::Vec3::new(1.0, 2.0, 3.0));
        let b = glam::Mat4::from_rotation_y(0.7) * glam::Mat4::from_scale(glam::Vec3::splat(2.0));
        let mut mats = vec![[0.0f32; 16]; MAX_SKIN_JOINTS];
        mats[3] = a.to_cols_array();
        mats[7] = b.to_cols_array();
        let joints = device.create_buffer_init(&util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&mats),
            usage: BufferUsages::UNIFORM,
        });
        let v = crate::renderer::mesh::SkinnedMeshVertex {
            position: [0.5, -1.0, 2.0],
            normal: [0.0, 1.0, 0.0],
            uv: [0.0, 0.0],
            joint_ids: [3, 7, 0, 0],
            joint_weights: [0.25, 0.75, 0.0, 0.0],
        };
        let source = device.create_buffer_init(&util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::bytes_of(&v),
            usage: BufferUsages::VERTEX | BufferUsages::STORAGE,
        });
        let posed = skin.posed_for(&device, &source, 1, &joints);
        let readback = device.create_buffer(&BufferDescriptor {
            label: None,
            size: POSED_STRIDE,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor::default());
        skin.dispatch(&mut encoder, [&posed]);
        encoder.copy_buffer_to_buffer(&posed.positions, 0, &readback, 0, POSED_STRIDE);
        queue.submit(Some(encoder.finish()));
        readback.slice(..).map_async(MapMode::Read, |_| {});
        let _ = device.poll(PollType::Wait { submission_index: None, timeout: None });
        let got: Vec<f32> = bytemuck::cast_slice(&readback.slice(..).get_mapped_range().unwrap()).to_vec();
        let p = glam::Vec4::new(0.5, -1.0, 2.0, 1.0);
        let want = (a * p) * 0.25 + (b * p) * 0.75;
        for k in 0..4 {
            assert!((got[k] - want[k]).abs() < 1e-5, "component {k}: {} vs {}", got[k], want[k]);
        }
    }
}
