use bytemuck::{Pod, Zeroable};
use glam::Mat4;
use wgpu::*;

use super::lights::LightsUniform;

/// Camera matrices for up to two views.
///
/// Always two, even for single-view passes. Multiview shaders index this by
/// `@builtin(view_index)`; single-view shaders read slot 0. Keeping one shape
/// means one bind group layout serves both, so the scope's single-view world
/// pass and the stereo eye pass can share pipelines' layouts without divergence.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Uniforms {
    pub view_proj: [[[f32; 4]; 4]; 2],
}

pub struct UniformBuffer {
    pub buffer: Buffer,
    pub layout: BindGroupLayout,
    pub bind_group: BindGroup,
}

impl UniformBuffer {
    pub fn new(device: &Device, lights: &LightsUniform) -> Self {
        let buffer = device.create_buffer(&BufferDescriptor {
            label: Some("uniform_buf"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("uniform_bgl"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::VERTEX,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("uniform_bg"),
            layout: &layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: lights.buffer().as_entire_binding(),
                },
            ],
        });

        Self {
            buffer,
            layout,
            bind_group,
        }
    }

    /// Single-view upload: both slots get the same matrix, so a multiview
    /// pipeline pointed at this buffer still renders something sane rather than
    /// reading uninitialised memory for the second eye.
    pub fn upload(&self, queue: &Queue, view_proj: Mat4) {
        self.upload_stereo(queue, [view_proj, view_proj]);
    }

    /// Stereo upload: one matrix per eye, indexed in-shader by `view_index`.
    pub fn upload_stereo(&self, queue: &Queue, view_proj: [Mat4; 2]) {
        let u = Uniforms {
            view_proj: [
                view_proj[0].to_cols_array_2d(),
                view_proj[1].to_cols_array_2d(),
            ],
        };
        queue.write_buffer(&self.buffer, 0, bytemuck::bytes_of(&u));
    }
}


/// WGSL fragment declaring the camera uniform, shared by every pipeline that
/// consumes it so single-view and multiview variants cannot drift apart.
///
/// `views == 2` emits a `@builtin(view_index)` parameter on the vertex entry
/// point and selects the matrix per eye — that is single-pass stereo. `views ==
/// 1` keeps the plain signature and reads slot 0, which is what the offscreen
/// passes (scope world view, mirror reflection) need: they render into
/// single-layer targets, where a multiview pipeline is invalid.
pub fn camera_uniform_wgsl(group: u32, binding: u32) -> String {
    format!(
        "struct Uniforms {{ view_proj: array<mat4x4<f32>, 2> }}\n\
         @group({group}) @binding({binding}) var<uniform> u: Uniforms;\n"
    )
}

/// Extra vertex-entry parameter for a multiview shader; empty for single view.
pub fn view_index_param(views: u32) -> &'static str {
    if views == 2 {
        ", @builtin(view_index) view_idx: i32"
    } else {
        ""
    }
}

/// Expression selecting this invocation's view-projection matrix.
pub fn view_proj_expr(views: u32) -> &'static str {
    if views == 2 {
        "u.view_proj[view_idx]"
    } else {
        "u.view_proj[0]"
    }
}
