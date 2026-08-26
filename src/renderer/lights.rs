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
    /// Infinitely-distant parallel light (sun). `position`/`range` are ignored;
    /// the beam travels along `direction`, so there is no distance attenuation.
    Directional,
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
            // Kind tag packed into params.z — must match the branch constants
            // in `wgsl_lights_block`: 0 = point, 1 = spot, 2 = directional.
            let kind = match l.kind {
                LightKind::Point => 0.0,
                LightKind::Spot => 1.0,
                LightKind::Directional => 2.0,
            };
            *slot = GpuLight {
                position: [l.position.x, l.position.y, l.position.z, 0.0],
                direction: [l.direction.x, l.direction.y, l.direction.z, 0.0],
                color_intensity: [color[0], color[1], color[2], l.intensity],
                params: [l.range, cos_outer, kind, 0.0],
            };
        }
        queue.write_buffer(&self.buffer, 0, bytemuck::bytes_of(&gpu));
    }
}

/// WGSL for the whole of group 0: the camera, the lights, and both shadow maps.
///
/// EVERY SHADING SHADER TAKES THIS BLOCK ENTIRE
///
/// It emits the camera uniform's struct and binding as well as the lights',
/// which the raytracing branch this was ported from did not -- there, each
/// shader declared its own copy of the camera struct and `shade` took the four
/// fields it needed as arguments, so a call read:
///
///     shade(world, n, view_dir, camera.sun_view_proj, camera.spot_view_proj,
///           camera.shadow_params)
///
/// repeated at seven call sites, each of which had to be edited in step with
/// the struct. Since a shader that calls `shade` needs the camera uniform
/// anyway, emitting both together makes the call `shade(world, n)` again and
/// leaves exactly one declaration of the layout to keep in step with
/// `uniforms::Uniforms`.
///
/// Shaders that only need `view_proj` -- wire, mirror, the ui2d passes -- keep
/// their own two-field struct. That is legal and deliberate: a uniform struct
/// may be a PREFIX of the buffer bound to it, so they are unaffected by
/// anything added here.
pub fn wgsl_lights_block(group_index: u32, binding_index: u32) -> String {
    let shadow_tex = binding_index + 1;
    let shadow_samp = binding_index + 2;
    let spot_tex = binding_index + 3;
    format!(
        r#"
struct Camera {{
    view_proj: mat4x4<f32>,
    sun_view_proj: mat4x4<f32>,
    spot_view_proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
    // x = sun shadow on, y = spot shadow on, z = which light is the flashlight.
    shadow_params: vec4<f32>,
}}
@group({group_index}) @binding(0) var<uniform> camera: Camera;

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
@group({group_index}) @binding({shadow_tex}) var sun_shadow_tex: texture_depth_2d;
@group({group_index}) @binding({shadow_samp}) var shadow_samp: sampler_comparison;
@group({group_index}) @binding({spot_tex}) var spot_shadow_tex: texture_depth_2d;

const AMBIENT: f32 = 0.6;
const SPEC_STRENGTH: f32 = 0.35;
const SHININESS: f32 = 32.0;

// Projects a world position into a light's clip space and returns
// (uv.x, uv.y, biased depth, valid). `valid` is 0 outside the light's frustum,
// where the caller must treat the fragment as fully lit rather than shadowed --
// otherwise everything beyond the shadow map's reach goes black.
fn shadow_coords(world_pos: vec3<f32>, light_view_proj: mat4x4<f32>) -> vec4<f32> {{
    let lp = light_view_proj * vec4<f32>(world_pos, 1.0);
    let ndc = lp.xyz / lp.w;
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    var valid = 1.0;
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || ndc.z > 1.0 || ndc.z < 0.0) {{
        valid = 0.0;
    }}
    let bias = 0.0015;
    return vec4<f32>(uv.x, uv.y, ndc.z - bias, valid);
}}

// 3x3 hardware PCF: 1 is lit, 0 is shadowed.
//
// `textureSampleCompareLevel` and not `textureSampleCompare`: the latter takes
// implicit derivatives and so may not be called after the per-fragment early
// return above, which is exactly the WGSL uniform-control-flow rule.
fn pcf(tex: texture_depth_2d, world_pos: vec3<f32>, light_view_proj: mat4x4<f32>) -> f32 {{
    let c = shadow_coords(world_pos, light_view_proj);
    if (c.w < 0.5) {{ return 1.0; }}
    let texel = 1.0 / vec2<f32>(textureDimensions(tex));
    var sum = 0.0;
    for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {{
        for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {{
            let off = vec2<f32>(f32(dx), f32(dy)) * texel;
            sum = sum + textureSampleCompareLevel(tex, shadow_samp, c.xy + off, c.z);
        }}
    }}
    return sum / 9.0;
}}

fn light_contribution(l: Light, world_pos: vec3<f32>, n: vec3<f32>, view_dir: vec3<f32>) -> vec3<f32> {{
    // params.z tags the kind: 0 point, 1 spot, 2 directional.
    let kind = l.params.z;

    var l_dir: vec3<f32>;
    var atten: f32;
    if (kind > 1.5) {{
        // The sun. Parallel rays travel ALONG `direction`, so the surface-to-
        // light vector is its negation, and there is no distance falloff --
        // applying one would make the sun dim with the scene's origin.
        l_dir = normalize(-l.direction.xyz);
        atten = 1.0;
    }} else {{
        let to_light = l.position.xyz - world_pos;
        let dist = length(to_light);
        l_dir = to_light / max(dist, 0.0001);

        let d_over_r = dist / max(l.params.x, 0.0001);
        let window = clamp(1.0 - pow(d_over_r, 4.0), 0.0, 1.0);
        atten = (window * window) / (dist * dist + 1.0);

        if (kind > 0.5) {{
            let cos_outer = l.params.y;
            let cos_angle = dot(-l_dir, l.direction.xyz);
            let cone = clamp((cos_angle - cos_outer) / max(1.0 - cos_outer, 0.0001), 0.0, 1.0);
            atten = atten * cone * cone * (3.0 - 2.0 * cone);
        }}
    }}

    let ndotl = max(dot(n, l_dir), 0.0);
    let radiance = l.color_intensity.rgb * l.color_intensity.a;
    var out = radiance * ndotl * atten;

    // Blinn-Phong specular, gated on ndotl so a surface facing away from the
    // light gets no highlight. Ungated, the half-vector still lines up on the
    // far side and rims every object with light coming from behind it.
    if (ndotl > 0.0) {{
        let h = normalize(l_dir + view_dir);
        let spec = pow(max(dot(n, h), 0.0), SHININESS) * SPEC_STRENGTH;
        out = out + radiance * spec * atten;
    }}
    return out;
}}

fn shade(world_pos: vec3<f32>, n: vec3<f32>) -> vec3<f32> {{
    let view_dir = normalize(camera.camera_pos.xyz - world_pos);
    let flash_idx = u32(camera.shadow_params.z);
    var lit = vec3<f32>(AMBIENT, AMBIENT, AMBIENT);
    for (var i: u32 = 0u; i < lights.count.x; i = i + 1u) {{
        let l = lights.lights[i];
        var c = light_contribution(l, world_pos, n, view_dir);
        // Only the sun casts the orthographic map, and only the flashlight the
        // perspective one. Every other light is unshadowed, which is the whole
        // reason a scene may have eight of them.
        if (l.params.z > 1.5 && camera.shadow_params.x > 0.5) {{
            c = c * pcf(sun_shadow_tex, world_pos, camera.sun_view_proj);
        }}
        if (camera.shadow_params.y > 0.5 && i == flash_idx) {{
            c = c * pcf(spot_shadow_tex, world_pos, camera.spot_view_proj);
        }}
        lit = lit + c;
    }}
    return lit;
}}
"#
    )
}
