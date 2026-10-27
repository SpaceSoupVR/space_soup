//! The ocular: a quad over the rear lens that shows the scope's render by ray
//! angle, cuts the scope shadow, and draws the reticle.
//!
//! Drawn inside the scene pass after the opaque world and the sky, depth tested
//! against the weapon it sits on but never writing depth. Output is
//! premultiplied, so a window (1x) lens can add its reticle and shadow over
//! the eye's own view without rendering the world twice.

use glam::Vec3;
use wgpu::{BindGroup, BindGroupLayout, Buffer, Device, Sampler, TextureFormat, TextureView};

use super::ScopeRender;
use crate::renderer::multiview::ViewMode;

/// The lens composite for one sample count and view mode.
pub struct LensPipeline {
    pub pipeline: wgpu::RenderPipeline,
    pub texture_layout: BindGroupLayout,
    params_layout: BindGroupLayout,
    sampler: Sampler,
}

/// One optic's lens uniform. `extra.z` is the eye this buffer's eye 0 stands
/// for: 0 for the stereo pass and the left eye, 1 for the right eye's mono pass.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LensParams {
    /// Per eye: xyz position, w eye-box occupancy.
    pub eyes: [[f32; 4]; 2],
    /// Per eye: xy direction the eye moved off the axis.
    pub offsets: [[f32; 4]; 2],
    /// Right, up and forward of the optic, then the ocular centre (w radius).
    pub frame: [[f32; 4]; 4],
    /// x power, y tan(half true fov), z edge feather (px), w 1 for a window.
    pub optic: [f32; 4],
    /// rgb reticle emission, w style.
    pub reticle: [f32; 4],
    /// x reticle scale, y 1 when shadow darkens, z eye base.
    pub extra: [f32; 4],
}

const LENS_WGSL: &str = include_str!("lens.wgsl");

impl LensParams {
    /// The uniform for `scope` seen from `eyes` (render space), for the
    /// buffer whose eye 0 stands for `eye_base`.
    pub fn for_scope(scope: &ScopeRender, eyes: [Vec3; 2], eye_base: u32) -> Self {
        let basis = scope.basis();
        let offset = |e: usize| {
            let [x, y] = basis.offset_direction(scope.ocular, eyes[e]);
            [x, y, 0.0, 0.0]
        };
        let c = scope.ocular;
        let r = &scope.reticle;
        Self {
            eyes: std::array::from_fn(|e| eyes[e].extend(scope.occupancy[e]).to_array()),
            offsets: [offset(0), offset(1)],
            frame: [
                basis.right.extend(0.0).to_array(),
                basis.up.extend(0.0).to_array(),
                basis.forward.extend(0.0).to_array(),
                [c.x, c.y, c.z, scope.ocular_radius_m],
            ],
            optic: [
                scope.magnification,
                (scope.true_fov_y_rad.max(1e-3) * 0.5).tan(),
                scope.edge_feather_px,
                if scope.is_window() { 1.0 } else { 0.0 },
            ],
            reticle: [r.emission[0], r.emission[1], r.emission[2], r.style as f32],
            extra: [r.scale, if scope.shadow_darkens { 1.0 } else { 0.0 }, eye_base as f32, 0.0],
        }
    }
}

impl LensPipeline {
    /// `camera_layout` is the scene's group 0; `samples` and `view` match the scene pass it draws in.
    pub fn new(device: &Device, format: TextureFormat, camera_layout: &BindGroupLayout, samples: u32, view: ViewMode) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("scope_lens_shader"),
            source: wgpu::ShaderSource::Wgsl(view.shader(LENS_WGSL.to_string()).into()),
        });
        let texture_layout = Self::texture_layout(device, "scope_lens_texture_bgl");
        let params_layout = Self::params_layout(device);
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("scope_lens_layout"),
            bind_group_layouts: &[Some(camera_layout), Some(&texture_layout), Some(&params_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("scope_lens"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: 12,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3],
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            // A flat lens seen from one side: culling would blank it by approach angle.
            primitive: wgpu::PrimitiveState { cull_mode: None, ..Default::default() },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: TextureFormat::Depth32Float,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState { count: samples, ..Default::default() },
            multiview_mask: view.mask(),
            cache: None,
        });
        Self { pipeline, texture_layout, params_layout, sampler: super::ScopeTarget::clamped_sampler(device, true) }
    }

    /// A filterable 2D texture and its sampler: what the lens and the mip blit read.
    pub fn texture_layout(device: &Device, label: &str) -> BindGroupLayout {
        let entry = |binding, ty| wgpu::BindGroupLayoutEntry { binding, visibility: wgpu::ShaderStages::FRAGMENT, ty, count: None };
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(label),
            entries: &[
                entry(0, wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                }),
                entry(1, wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering)),
            ],
        })
    }

    pub fn texture_group(device: &Device, layout: &BindGroupLayout, view: &TextureView, sampler: &Sampler) -> BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scope_texture"),
            layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(sampler) },
            ],
        })
    }

    /// The lens's view of a target: `view` must span the whole mip chain.
    pub fn target_group(&self, device: &Device, view: &TextureView) -> BindGroup {
        Self::texture_group(device, &self.texture_layout, view, &self.sampler)
    }

    /// A lens uniform buffer and its group, written each frame with `queue.write_buffer`.
    pub fn params_group(&self, device: &Device) -> (Buffer, BindGroup) {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scope_lens_params"),
            size: std::mem::size_of::<LensParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scope_lens_params"),
            layout: &self.params_layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: buffer.as_entire_binding() }],
        });
        (buffer, group)
    }

    fn params_layout(device: &Device) -> BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("scope_lens_params_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        })
    }
}

#[cfg(test)]
pub(crate) mod tests;
