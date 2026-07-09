use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use wgpu::*;

use super::Color3;

/// Matches the fixed-size `array<Light, MAX_LIGHTS>` declared in the mesh/solid
/// fragment shaders — keep these in sync.
pub const MAX_LIGHTS: usize = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LightKind {
    Point,
    Spot,
}

/// A light to be drawn this frame, in whatever world space `cuboids`/`meshes`
/// are already in for this call — the renderer has no opinion on game logic,
/// it just shades with what it's handed.
#[derive(Debug, Clone, Copy)]
pub struct Light {
    pub position: Vec3,
    /// Aim direction for `Spot` lights; ignored for `Point`.
    pub direction: Vec3,
    pub kind: LightKind,
    pub color: Color3,
    pub intensity: f32,
    pub range: f32,
    pub cone_angle_deg: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GpuLight {
    position: [f32; 4],
    direction: [f32; 4],
    color_intensity: [f32; 4],
    /// x = range, y = cos(outer half-angle), z = kind (0 = point, 1 = spot), w unused.
    params: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GpuLights {
    /// x = active light count; yzw pad the field to the array's 16-byte stride.
    count: [u32; 4],
    lights: [GpuLight; MAX_LIGHTS],
}

/// Owns just the GPU buffer — the bind group itself lives alongside the
/// camera uniform in `uniforms::UniformBuffer` (one shared group, two
/// bindings), since wgpu's default `max_bind_groups` limit of 4 leaves no
/// room for lights as their own group once model/texture/joint groups are
/// already spoken for on the skinned mesh pipeline.
pub struct LightsUniform {
    buffer: Buffer,
}

impl LightsUniform {
    pub fn new(device: &Device) -> Self {
        let buffer = device.create_buffer(&BufferDescriptor {
            label: Some("lights_uniform"),
            size: std::mem::size_of::<GpuLights>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self { buffer }
    }

    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    pub fn upload(&self, queue: &Queue, lights: &[Light]) {
        let mut gpu = GpuLights {
            count: [lights.len().min(MAX_LIGHTS) as u32, 0, 0, 0],
            lights: [GpuLight::zeroed(); MAX_LIGHTS],
        };
        for (slot, l) in gpu.lights.iter_mut().zip(lights.iter().take(MAX_LIGHTS)) {
            let color = l.color.to_linear();
            let cos_outer = (l.cone_angle_deg.to_radians() * 0.5).cos();
            *slot = GpuLight {
                position: [l.position.x, l.position.y, l.position.z, 0.0],
                direction: [l.direction.x, l.direction.y, l.direction.z, 0.0],
                color_intensity: [color[0], color[1], color[2], l.intensity],
                params: [
                    l.range,
                    cos_outer,
                    if l.kind == LightKind::Spot { 1.0 } else { 0.0 },
                    0.0,
                ],
            };
        }
        queue.write_buffer(&self.buffer, 0, bytemuck::bytes_of(&gpu));
    }
}

/// WGSL shared by every fragment shader that samples the lights uniform.
/// `group_index`/`binding_index` must match wherever the pipeline binds the
/// lights buffer (see `uniforms::UniformBuffer`, which shares one bind group
/// between the camera and lights uniforms).
pub fn wgsl_lights_block(group_index: u32, binding_index: u32) -> String {
    format!(
        r#"
struct Light {{
    position: vec4<f32>,
    direction: vec4<f32>,
    color_intensity: vec4<f32>,
    params: vec4<f32>,
}}
struct Lights {{
    count: vec4<u32>,
    lights: array<Light, {MAX_LIGHTS}>,
}}
@group({group_index}) @binding({binding_index}) var<uniform> lights: Lights;

const AMBIENT: f32 = 0.6;

fn light_contribution(l: Light, world_pos: vec3<f32>, n: vec3<f32>) -> vec3<f32> {{
    let to_light = l.position.xyz - world_pos;
    let dist = length(to_light);
    let l_dir = to_light / max(dist, 0.0001);
    let ndotl = max(dot(n, l_dir), 0.0);

    let d_over_r = dist / max(l.params.x, 0.0001);
    let window = clamp(1.0 - pow(d_over_r, 4.0), 0.0, 1.0);
    var atten = (window * window) / (dist * dist + 1.0);

    if (l.params.z > 0.5) {{
        let cos_outer = l.params.y;
        let cos_angle = dot(-l_dir, l.direction.xyz);
        let cone = clamp((cos_angle - cos_outer) / max(1.0 - cos_outer, 0.0001), 0.0, 1.0);
        atten = atten * cone * cone * (3.0 - 2.0 * cone);
    }}

    return l.color_intensity.rgb * l.color_intensity.a * ndotl * atten;
}}

fn shade(world_pos: vec3<f32>, n: vec3<f32>) -> vec3<f32> {{
    var lit = vec3<f32>(AMBIENT, AMBIENT, AMBIENT);
    for (var i: u32 = 0u; i < lights.count.x; i = i + 1u) {{
        lit = lit + light_contribution(lights.lights[i], world_pos, n);
    }}
    return lit;
}}
"#
    )
}
