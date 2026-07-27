use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use wgpu::*;

use super::lights::LightsUniform;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Uniforms {
    pub view_proj: [[f32; 4]; 4],
    /// Sun (directional) light-space view-projection, for the sun shadow map.
    pub sun_view_proj: [[f32; 4]; 4],
    /// Spot/flashlight light-space view-projection, for the spot shadow map.
    pub spot_view_proj: [[f32; 4]; 4],
    /// World-space camera position (xyz); w unused. Drives specular.
    pub camera_pos: [f32; 4],
    /// x = sun shadow enabled (1/0), y = spot shadow enabled (1/0),
    /// z = flashlight light index (as f32, valid when y > 0.5), w reserved.
    pub shadow_params: [f32; 4],
}

/// Per-frame shadow inputs, bundled to keep `upload` call sites readable.
pub struct ShadowUpload {
    pub sun_view_proj: Mat4,
    pub spot_view_proj: Mat4,
    pub sun_enabled: bool,
    pub spot_enabled: bool,
    /// Index into the lights array of the shadow-casting flashlight spot light.
    pub flashlight_index: u32,
}

impl ShadowUpload {
    /// No shadows (identity matrices, both disabled) — used by paths that don't
    /// render a shadow pass yet (e.g. the XR renderer).
    pub fn disabled() -> Self {
        Self {
            sun_view_proj: Mat4::IDENTITY,
            spot_view_proj: Mat4::IDENTITY,
            sun_enabled: false,
            spot_enabled: false,
            flashlight_index: 0,
        }
    }
}

/// Camera, lights, and both shadow maps share one bind group — wgpu's default
/// `max_bind_groups` limit of 4 leaves no room for extra groups once the
/// skinned mesh pipeline's model/texture/joint groups are accounted for.
/// Bindings: 0 = camera uniforms (vertex + fragment), 1 = lights (fragment),
/// 2 = sun shadow depth texture, 3 = shadow comparison sampler,
/// 4 = spot shadow depth texture.
pub struct UniformBuffer {
    pub buffer: Buffer,
    pub layout: BindGroupLayout,
    pub bind_group: BindGroup,
}

impl UniformBuffer {
    pub fn new(
        device: &Device,
        lights: &LightsUniform,
        sun_shadow_view: &TextureView,
        spot_shadow_view: &TextureView,
        shadow_sampler: &Sampler,
    ) -> Self {
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
                    visibility: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
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
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Depth,
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler(SamplerBindingType::Comparison),
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 4,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Depth,
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
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
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(sun_shadow_view),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::Sampler(shadow_sampler),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: BindingResource::TextureView(spot_shadow_view),
                },
            ],
        });

        Self {
            buffer,
            layout,
            bind_group,
        }
    }

    pub fn upload(&self, queue: &Queue, view_proj: Mat4, camera_pos: Vec3, shadow: &ShadowUpload) {
        let u = Uniforms {
            view_proj: view_proj.to_cols_array_2d(),
            sun_view_proj: shadow.sun_view_proj.to_cols_array_2d(),
            spot_view_proj: shadow.spot_view_proj.to_cols_array_2d(),
            camera_pos: [camera_pos.x, camera_pos.y, camera_pos.z, 1.0],
            shadow_params: [
                if shadow.sun_enabled { 1.0 } else { 0.0 },
                if shadow.spot_enabled { 1.0 } else { 0.0 },
                shadow.flashlight_index as f32,
                0.0,
            ],
        };
        queue.write_buffer(&self.buffer, 0, bytemuck::bytes_of(&u));
    }
}
