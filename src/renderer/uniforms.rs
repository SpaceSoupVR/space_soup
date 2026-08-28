use bytemuck::{Pod, Zeroable};
use super::shadow::MAX_SPOT_SHADOWS;
use glam::{Mat4, Vec3};
use wgpu::*;

use super::lights::LightsUniform;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Uniforms {
    pub view_proj: [[f32; 4]; 4],
    /// The inverse, so the sky pass can turn a pixel back into a view ray.
    ///
    /// Inverted on the CPU once per eye rather than in the shader: a 4x4
    /// inverse per fragment is absurd, and the sky is a full-screen pass.
    pub inv_view_proj: [[f32; 4]; 4],
    /// Sun (directional) light-space view-projection, for the sun shadow map.
    pub sun_view_proj: [[f32; 4]; 4],
    /// One light-space view-projection per shadow-casting spot.
    ///
    /// An ARRAY because a level has more than one lamp. It used to be a single
    /// matrix and the first spot in the scene silently claimed it, so a room
    /// with two identical fixtures had one casting a shadow and one not -- which
    /// reads as a broken light rather than an exhausted budget.
    pub spot_view_proj: [[[f32; 4]; 4]; MAX_SPOT_SHADOWS],
    /// World-space camera position (xyz); w unused. Drives specular.
    pub camera_pos: [f32; 4],
    /// x = sun shadow enabled (1/0), y = how many spot shadow layers are live,
    /// z = reserved, w reserved.
    ///
    /// Which light uses which layer is carried on the LIGHT (`params.w`) rather
    /// than here, because it is a property of the light and not of the camera --
    /// and the previous single "flashlight index" could only ever name one.
    pub shadow_params: [f32; 4],
    /// x = sky intensity. yzw reserved.
    pub sky_params: [f32; 4],
    /// Nine RGB spherical-harmonic coefficients of the sky's irradiance.
    ///
    /// `vec4` per coefficient because std140 rounds an array element up to 16
    /// bytes anyway -- packing them as vec3 would be the same size and would
    /// need the shader to index a padded array by hand.
    ///
    /// MUST stay in step with the `Camera` struct in `wgsl_lights_block`:
    /// bytemuck checks size and alignment, not names, and a mismatch surfaces
    /// as `invalid field accessor` pointing at the WGSL line that reads it.
    pub sky_sh: [[f32; 4]; 9],
}

/// Per-frame sky inputs.
#[derive(Clone)]
pub struct SkyUpload {
    pub intensity: f32,
    pub sh: [[f32; 4]; 9],
}

impl SkyUpload {
    /// A scene with no sky: the flat ambient the engine always had, expressed
    /// as a constant-band SH so there is one lighting path rather than a branch.
    pub fn none() -> Self {
        Self::from(&crate::renderer::sky::SkyIrradiance::flat(crate::renderer::sky::AMBIENT))
    }

    pub fn from(irr: &crate::renderer::sky::SkyIrradiance) -> Self {
        let mut sh = [[0.0f32; 4]; 9];
        for i in 0..9 {
            sh[i] = [irr.sh[i][0], irr.sh[i][1], irr.sh[i][2], 0.0];
        }
        Self { intensity: 1.0, sh }
    }
}

/// Per-frame shadow inputs, bundled to keep `upload` call sites readable.
pub struct ShadowUpload {
    pub sun_view_proj: Mat4,
    /// One per shadow-casting spot, in shadow-layer order.
    pub spot_view_proj: [Mat4; MAX_SPOT_SHADOWS],
    pub sun_enabled: bool,
    /// How many spot shadow layers this frame actually filled.
    ///
    /// Replaces a bool plus a single "flashlight index": with an array of
    /// layers, WHICH light uses WHICH layer belongs on the light, and the
    /// camera only needs to know how many are live.
    pub spot_count: u32,
}

impl ShadowUpload {
    /// No shadows (identity matrices, both disabled) — used by paths that don't
    /// render a shadow pass yet (e.g. the XR renderer).
    pub fn disabled() -> Self {
        Self {
            sun_view_proj: Mat4::IDENTITY,
            spot_view_proj: [Mat4::IDENTITY; MAX_SPOT_SHADOWS],
            sun_enabled: false,
            spot_count: 0,
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
                        // D2Array: one texture, one layer per shadow-casting
                        // spot, sampled by index. Must match the
                        // `texture_depth_2d_array` the shader declares, or
                        // pipeline creation fails -- which is the good outcome,
                        // since the alternative is reading the wrong lamp's
                        // depth and drawing a shadow from a light that is not
                        // there.
                        view_dimension: TextureViewDimension::D2Array,
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
        self.upload_with_sky(queue, view_proj, camera_pos, shadow, &SkyUpload::none());
    }

    pub fn upload_with_sky(
        &self,
        queue: &Queue,
        view_proj: Mat4,
        camera_pos: Vec3,
        shadow: &ShadowUpload,
        sky: &SkyUpload,
    ) {
        let u = Uniforms {
            view_proj: view_proj.to_cols_array_2d(),
            inv_view_proj: view_proj.inverse().to_cols_array_2d(),
            sun_view_proj: shadow.sun_view_proj.to_cols_array_2d(),
            spot_view_proj: std::array::from_fn(|i| shadow.spot_view_proj[i].to_cols_array_2d()),
            camera_pos: [camera_pos.x, camera_pos.y, camera_pos.z, 1.0],
            shadow_params: [
                if shadow.sun_enabled { 1.0 } else { 0.0 },
                shadow.spot_count as f32,
                0.0,
                0.0,
            ],
            sky_params: [sky.intensity, 0.0, 0.0, 0.0],
            sky_sh: sky.sh,
        };
        queue.write_buffer(&self.buffer, 0, bytemuck::bytes_of(&u));
    }
}

#[cfg(test)]
pub(crate) mod test_support {
    use super::*;
    use crate::renderer::shadow::ShadowMap;

    /// Where the eye sits in a render test.
    ///
    /// Straight out in front of the quad every harness here draws, so a
    /// specular highlight lands symmetrically and cannot be mistaken for one
    /// material winning over another. Only specular reads it.
    pub const TEST_EYE: glam::Vec3 = glam::Vec3::new(0.0, 0.0, 5.0);

    /// Group 0 for a render test, wired exactly as the renderer wires it.
    ///
    /// A real `ShadowMap`, only tiny: the shadow textures have to exist and be
    /// bound with the right sample types or nothing validates, and building a
    /// stand-in here would mean the tests stopped noticing when the real layout
    /// changed. 64 squared costs nothing and proves the same thing.
    ///
    /// The `ShadowMap` comes back with the buffer because it owns the textures
    /// the bind group points at; dropping it would leave the group dangling.
    pub fn scene_uniforms(device: &Device, lights: &LightsUniform) -> (ShadowMap, UniformBuffer) {
        let shadows = ShadowMap::with_dimension(device, 64);
        let uniforms = UniformBuffer::new(
            device,
            lights,
            shadows.sun_depth_view(),
            shadows.spot_depth_view(),
            shadows.sampler(),
        );
        (shadows, uniforms)
    }
}
