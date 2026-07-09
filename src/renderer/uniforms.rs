use bytemuck::{Pod, Zeroable};
use glam::Mat4;
use wgpu::*;

use super::lights::LightsUniform;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Uniforms {
    pub view_proj: [[f32; 4]; 4],
}

/// Camera and lights share one bind group (binding 0 = camera, binding 1 =
/// lights) rather than each getting their own group — wgpu's default
/// `max_bind_groups` limit of 4 leaves no headroom for a 5th group once the
/// skinned mesh pipeline's model/texture/joint groups are accounted for.
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

    pub fn upload(&self, queue: &Queue, view_proj: Mat4) {
        let u = Uniforms {
            view_proj: view_proj.to_cols_array_2d(),
        };
        queue.write_buffer(&self.buffer, 0, bytemuck::bytes_of(&u));
    }
}
