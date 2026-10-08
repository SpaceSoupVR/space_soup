//! WEATHER ON SCREEN: rain and snow falling round the head, their splashes,
//! and what they leave on the ground -- wet, darkened, glossy ground, flat
//! mirror puddles that take the rain's ripples, and snow cover that lifts the
//! ground and sparkles.
//!
//! The simulation is the engine's (`space_soup_engine::weather`): a few
//! numbers an area move with time, and a map an area is worked out from them
//! a few times a second -- wetness, standing water, snow depth and the COVER
//! HEIGHT (the first thing straight down from the sky). This module uploads
//! those maps ([`WeatherMaps`]) and draws with them.
//!
//! ONE LIGHTING, NOT TWO
//!
//! Nothing here lights a surface its own way. The ground's weather is an edit
//! of the ground's own shader ([`with_weather`]): the wet, puddled or snowy
//! albedo, roughness and normal go INTO `shade_material_env`, so the sun's
//! baked mask and its shadows, the stationary lamps' masks, the probes and
//! their outdoor trace (a puddle's mirror is the ground's own reflection path
//! at a puddle's roughness), the cards and the torch all light them exactly
//! as they light dry ground. The falling rain and snow ([`WeatherPipeline`])
//! are lit by the sky, by the sun where the ground's baked sun mask says the
//! sun reaches -- read where each drop's ray to the sun meets the ground, so a
//! drop in a wall's shadow does not glow -- and by the player's torch.
//!
//! WHAT IT COSTS A LEVEL WITHOUT WEATHER: nothing. The ground's weather is a
//! TWIN of each ground shader, drawn only for the terrain chunks an area
//! touches (`XrRenderer::set_weather`); every other chunk, and every level
//! without weather, draws the shaders it always did. The particles are one
//! draw, made only while something falls near the head.
//!
//! OCCLUSION, FROM DATA ALREADY THERE: the cover height is found once at load
//! by casting down through the physics scene the game already has, so no rain
//! falls through a roof or into the cave, no drop splashes under one, and
//! nothing accumulates on ground under cover. No pass renders it.

use bytemuck::{Pod, Zeroable};
use wgpu::*;

/// The most areas drawn, as the engine's `weather::MAX_AREAS`.
pub const MAX_AREAS: usize = 4;
/// Rain streaks, snowflakes and splashes drawn at full intensity, near the
/// head. The rain fills a 16 x 16 x 9 m box round the head, the snow a
/// 10 x 10 x 7 m one (a flake past ~6 m is under a pixel and a half, and
/// faded by it), the splashes the ground 12 m round it. Heavy snow is about
/// 12 flakes a cubic metre here: thinner than the real thing's hundreds,
/// most of which no eye resolves.
pub const RAIN_PARTICLES: u32 = 4096;
pub const SNOW_PARTICLES: u32 = 8192;
pub const SPLASHES: u32 = 768;
/// Vertices a particle: two triangles.
pub const VERTICES_PER_PARTICLE: u32 = 6;
/// Fresh snow's albedo, linear (about 0.85-0.9 in the visible).
pub const SNOW_ALBEDO: [f32; 3] = [0.86, 0.88, 0.92];

/// The weather's uniform, group 2 of the ground's twins and of the particles.
/// Per area three vec4: `[min x, min z, 1 / extent x, 1 / extent z]`, `[uv
/// scale x, uv scale z (the area's share of the texture), kind (0 rain, 1
/// snow), how hard it falls now]`, `[wind x, wind z, layer, edge metres]`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct WeatherUniform {
    pub areas: [[f32; 4]; 3 * MAX_AREAS],
    /// x areas, y seconds (wrapped), z rain particles drawn, w snow
    /// particles drawn (splashes after both).
    pub params: [f32; 4],
    /// The terrain footprint, for its baked map: `[min x, min z, 1 / extent
    /// x, 1 / extent z]`; zero with no terrain.
    pub ground: [f32; 4],
    /// The head, WORLD frame.
    pub head: [f32; 4],
    /// The head's right and up, PLAYER frame (both eyes see one quad).
    pub right: [f32; 4],
    pub up: [f32; 4],
    /// The sky's light on a drop or a flake: the mean radiance a white
    /// diffuser takes from it, exposed. w: the exposure.
    pub sky: [f32; 4],
    /// The sun's: what a white diffuser square to it reflects, exposed.
    pub sun: [f32; 4],
    /// Toward the sun, WORLD frame; w 1 with a sun.
    pub sun_dir: [f32; 4],
    /// The player's torch, PLAYER frame: position and range (0: none),
    /// direction and the cone's outer cosine, colour times intensity times
    /// exposure and its inner cosine.
    pub torch_pos: [f32; 4],
    pub torch_dir: [f32; 4],
    pub torch_col: [f32; 4],
    /// The player frame: offset and yaw, as `Uniforms::player_frame`.
    pub frame: [f32; 4],
    /// x an eye pixel's size at unit depth; y the wind at the head, x, z.
    pub pixel: [f32; 4],
    pub wind: [f32; 4],
}

impl Default for WeatherUniform {
    fn default() -> Self {
        Self::zeroed()
    }
}

/// One area as the GPU sees it.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AreaParams {
    pub min: [f32; 2],
    pub extent: [f32; 2],
    pub snow: bool,
    pub falling: f32,
    pub wind: [f32; 2],
    pub edge: f32,
}

/// Group 2: the uniform, the maps (one layer an area), their sampler.
pub fn bind_group_layout(device: &Device) -> BindGroupLayout {
    let both = ShaderStages::VERTEX | ShaderStages::FRAGMENT;
    device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("weather_layout"),
        entries: &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: both,
                ty: BindingType::Buffer { ty: BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: both,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::D2Array,
                    multisampled: false,
                },
                count: None,
            },
            BindGroupLayoutEntry { binding: 2, visibility: both, ty: BindingType::Sampler(SamplerBindingType::Filtering), count: None },
        ],
    })
}

/// IEEE half from f32, round to nearest; enough for the maps' metres.
pub fn f16_bits(v: f32) -> u16 {
    let x = v.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let exp = ((x >> 23) & 0xff) as i32;
    let man = x & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if man != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = (man | 0x80_0000) >> (1 - e);
        return sign | ((m + 0x1000) >> 13) as u16;
    }
    let h = sign as u32 | ((e as u32) << 10) | (man >> 13);
    // Round half up on the dropped bits (carries into the exponent correctly).
    (h + ((man >> 12) & 1)) as u16
}

/// The maps on the GPU, the uniform and their bind group. Made when a level
/// has weather, so a level without pays nothing.
pub struct WeatherMaps {
    texture: Texture,
    pub size: (u32, u32),
    pub layers: u32,
    uniform: Buffer,
    pub bind_group: BindGroup,
    pub params: WeatherUniform,
}

impl WeatherMaps {
    /// Room for areas of these sizes in texels, one layer each.
    pub fn new(device: &Device, layout: &BindGroupLayout, sizes: &[(u32, u32)]) -> Self {
        let w = sizes.iter().map(|s| s.0).max().unwrap_or(1).max(1);
        let h = sizes.iter().map(|s| s.1).max().unwrap_or(1).max(1);
        let layers = (sizes.len() as u32).clamp(1, MAX_AREAS as u32);
        let texture = device.create_texture(&TextureDescriptor {
            label: Some("weather_maps"),
            size: Extent3d { width: w, height: h, depth_or_array_layers: layers },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&TextureViewDescriptor { dimension: Some(TextureViewDimension::D2Array), ..Default::default() });
        let sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("weather_sampler"),
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            ..Default::default()
        });
        let uniform = device.create_buffer(&BufferDescriptor {
            label: Some("weather_uniform"),
            size: std::mem::size_of::<WeatherUniform>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("weather_bind_group"),
            layout,
            entries: &[
                BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: BindingResource::TextureView(&view) },
                BindGroupEntry { binding: 2, resource: BindingResource::Sampler(&sampler) },
            ],
        });
        Self { texture, size: (w, h), layers, uniform, bind_group, params: WeatherUniform::default() }
    }

    /// Area `layer`'s map: `nx * nz` texels, rows along z, each wetness,
    /// standing water (m), snow (m), cover height (world y).
    pub fn upload_map(&self, queue: &Queue, layer: u32, nx: u32, nz: u32, texels: &[[f32; 4]]) {
        if layer >= self.layers || nx > self.size.0 || nz > self.size.1 || texels.len() < (nx * nz) as usize {
            log::warn!("weather map {layer} ({nx}x{nz}) does not fit {:?} x {}", self.size, self.layers);
            return;
        }
        let bytes: Vec<u8> = texels[..(nx * nz) as usize]
            .iter()
            .flat_map(|t| t.iter().flat_map(|&v| f16_bits(v).to_le_bytes()))
            .collect();
        queue.write_texture(
            TexelCopyTextureInfo {
                texture: &self.texture,
                mip_level: 0,
                origin: Origin3d { x: 0, y: 0, z: layer },
                aspect: TextureAspect::All,
            },
            &bytes,
            TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(nx * 8), rows_per_image: Some(nz) },
            Extent3d { width: nx, height: nz, depth_or_array_layers: 1 },
        );
    }

    /// Area `k`'s place, kind and what is falling there now, into the uniform
    /// [`Self::write`] sends; its map is `texels` across.
    pub fn set_area(&mut self, k: usize, a: &AreaParams, texels: (u32, u32)) {
        if k >= MAX_AREAS {
            return;
        }
        let ex = a.extent[0].max(1e-3);
        let ez = a.extent[1].max(1e-3);
        self.params.areas[3 * k] = [a.min[0], a.min[1], 1.0 / ex, 1.0 / ez];
        self.params.areas[3 * k + 1] = [
            texels.0 as f32 / self.size.0 as f32,
            texels.1 as f32 / self.size.1 as f32,
            if a.snow { 1.0 } else { 0.0 },
            a.falling.max(0.0),
        ];
        self.params.areas[3 * k + 2] = [a.wind[0], a.wind[1], k as f32, a.edge.max(1e-3)];
        self.params.params[0] = self.params.params[0].max(k as f32 + 1.0);
    }

    pub fn write(&self, queue: &Queue) {
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&self.params));
    }
}

/// How many of each particle to draw this frame: rain and snow as hard as it
/// falls in the strongest area within reach of the head, 0 when nothing
/// falls near. Splashes go with the rain.
pub fn particle_counts(areas: &[AreaParams], head: [f32; 3]) -> (u32, u32, u32) {
    let reach = 10.0;
    let mut rain = 0.0f32;
    let mut snow = 0.0f32;
    for a in areas {
        let near = head[0] > a.min[0] - reach
            && head[0] < a.min[0] + a.extent[0] + reach
            && head[2] > a.min[1] - reach
            && head[2] < a.min[1] + a.extent[1] + reach;
        if near {
            if a.snow {
                snow = snow.max(a.falling);
            } else {
                rain = rain.max(a.falling);
            }
        }
    }
    let n = |k: f32, full: u32| (k.clamp(0.0, 1.0) * full as f32).ceil() as u32;
    (n(rain, RAIN_PARTICLES), n(snow, SNOW_PARTICLES), n(rain, SPLASHES))
}

/// THE MAP'S WGSL, shared by the ground's twins and the particles: group 2,
/// the hash and noise, and `wx_at`, which finds the area a world point lies
/// in and reads its map there (explicit level, so it may be called anywhere,
/// in either stage).
pub fn map_block() -> String {
    r#"
// ---- WEATHER (`space_soup::renderer::weather`) ----
struct WxUniform {
    areas: array<vec4<f32>, 12>,
    params: vec4<f32>,
    ground: vec4<f32>,
    head: vec4<f32>,
    right: vec4<f32>,
    up: vec4<f32>,
    sky: vec4<f32>,
    sun: vec4<f32>,
    sun_dir: vec4<f32>,
    torch_pos: vec4<f32>,
    torch_dir: vec4<f32>,
    torch_col: vec4<f32>,
    frame: vec4<f32>,
    pixel: vec4<f32>,
    wind: vec4<f32>,
}
@group(2) @binding(0) var<uniform> wx: WxUniform;
@group(2) @binding(1) var wx_map: texture_2d_array<f32>;
@group(2) @binding(2) var wx_samp: sampler;

fn wx_hash(n: u32) -> u32 {
    var x = n;
    x = x ^ (x >> 16u);
    x = x * 0x7feb352du;
    x = x ^ (x >> 15u);
    x = x * 0x846ca68bu;
    x = x ^ (x >> 16u);
    return x;
}
fn wx_rand(n: u32) -> f32 {
    return f32(wx_hash(n) & 0xffffffu) / 16777216.0;
}
fn wx_cell(c: vec2<f32>) -> u32 {
    let i = vec2<i32>(floor(c));
    return wx_hash(bitcast<u32>(i.x) * 1973u + bitcast<u32>(i.y) * 9277u + 26699u);
}
// Value noise, 0..1.
// The four corners' cells from one: `wx_cell` of the corner one step along
// x is its key plus 1973, along z plus 9277 -- the same keys, so the same
// noise, without converting and keying each corner on its own (2026-10-08:
// the ground's weather twins were past the instruction cache).
fn wx_noise(q: vec2<f32>) -> f32 {
    let i = floor(q);
    let f = q - i;
    let s = f * f * (3.0 - 2.0 * f);
    let ii = vec2<i32>(i);
    let key = bitcast<u32>(ii.x) * 1973u + bitcast<u32>(ii.y) * 9277u + 26699u;
    let a = f32(wx_hash(key) & 0xffffu) / 65535.0;
    let b = f32(wx_hash(key + 1973u) & 0xffffu) / 65535.0;
    let c = f32(wx_hash(key + 9277u) & 0xffffu) / 65535.0;
    let d = f32(wx_hash(key + 11250u) & 0xffffu) / 65535.0;
    return mix(mix(a, b, s.x), mix(c, d, s.x), s.y);
}

// What weather lies at a world point: `m` the map (wetness, standing water m,
// snow m, cover height), `area` (falling, kind, wind x, wind z), `fade` its
// edge's.
struct WxAt {
    m: vec4<f32>,
    area: vec4<f32>,
    fade: f32,
}
fn wx_at(p: vec3<f32>) -> WxAt {
    var r: WxAt;
    r.m = vec4<f32>(0.0, 0.0, 0.0, -10000.0);
    r.area = vec4<f32>(0.0);
    r.fade = 0.0;
    // The LAST area the point lies in, as when each was read in turn; then
    // its map read once, not once an area in an unrolled loop.
    let count = u32(wx.params.x);
    var hit = 4u;
    var hit_uv = vec2<f32>(0.0);
    for (var k = 0u; k < 4u; k = k + 1u) {
        if (k >= count) {
            break;
        }
        let a0 = wx.areas[k * 3u];
        let uv = (p.xz - a0.xy) * a0.zw;
        if (all(uv >= vec2<f32>(0.0)) && all(uv <= vec2<f32>(1.0))) {
            hit = k;
            hit_uv = uv;
        }
    }
    if (hit < 4u) {
        let a0 = wx.areas[hit * 3u];
        let a1 = wx.areas[hit * 3u + 1u];
        let a2 = wx.areas[hit * 3u + 2u];
        r.m = textureSampleLevel(wx_map, wx_samp, hit_uv * a1.xy, i32(hit), 0.0);
        r.area = vec4<f32>(a1.w, a1.z, a2.x, a2.y);
        let inside = min(hit_uv, vec2<f32>(1.0) - hit_uv) / a0.zw;
        r.fade = smoothstep(0.0, a2.w, inside.x) * smoothstep(0.0, a2.w, inside.y);
    }
    return r;
}

// How far the sky reaches a point: 1 on (or above) the first thing the sky
// meets there, 0 half a metre and more under it. As `WeatherGround::exposure`.
fn wx_open(y: f32, cover: f32) -> f32 {
    return clamp((y - cover + 0.5) / 0.4, 0.0, 1.0);
}
"#
    .to_string()
}

/// The surface's half: what the weather makes of a surface at a point.
fn surface_block() -> String {
    format!(
        r#"
// RAIN RINGS on standing water: in each cell a drop lands every ~0.7 s at a
// jittered point, and its ring runs out, damped; two offset grids hide the
// cells. The slope of the water there, x and z.
fn wx_ripple_layer(q: vec2<f32>, t: f32, seed: u32) -> vec2<f32> {{
    let h = wx_cell(q + vec2<f32>(f32(seed), 0.0));
    let hx = f32(h & 0xffffu) / 65535.0;
    let hy = f32(h >> 16u) / 65535.0;
    let centre = floor(q) + vec2<f32>(0.5) + (vec2<f32>(hx, hy) - vec2<f32>(0.5)) * 0.5;
    let d = q - centre;
    let r = length(d);
    let age = fract(t * 1.4 + hx * 7.0 + hy * 3.0);
    let x = (r - age * 0.45) * 28.0;
    let wave = sin(x) * exp(-0.12 * x * x) * (1.0 - age) * (1.0 - age);
    return select(vec2<f32>(0.0), d / max(r, 1e-4) * wave, r > 1e-4);
}}
fn wx_ripples(xz: vec2<f32>, t: f32) -> vec2<f32> {{
    return 0.35 * (wx_ripple_layer(xz / 0.35, t, 0u) + wx_ripple_layer(xz / 0.35 + vec2<f32>(0.5), t + 0.37, 17u));
}}

struct WxSurface {{
    wet: f32,
    water: f32,
    snow: f32,
    sparkle: f32,
    // The water's slope (rings), world x and z; the snow's own slope, its
    // drifts; and a sparkling grain's normal, world.
    ripple: vec2<f32>,
    drift: vec2<f32>,
    grain: vec3<f32>,
}}
// At world point `p` with world normal `n`, seen from `view` (world, toward
// the eye), a pixel `footprint` metres across.
fn wx_surface(p: vec3<f32>, n: vec3<f32>, view: vec3<f32>, footprint: f32) -> WxSurface {{
    var s: WxSurface;
    let at = wx_at(p);
    let open = wx_open(p.y, at.m.w);
    let up = smoothstep(0.55, 0.9, n.y);
    var fine = 0.0;
    if (WX_WATER || WX_SNOW) {{
        fine = 0.65 * wx_noise(p.xz * 3.1) + 0.35 * wx_noise(p.xz * 9.7);
    }}
    s.wet = at.m.x * open;
    s.water = 0.0;
    s.snow = 0.0;
    s.ripple = vec2<f32>(0.0);
    s.drift = vec2<f32>(0.0);
    s.grain = vec3<f32>(0.0);
    s.sparkle = 0.0;
    if (WX_WATER) {{
        // A film of a few millimetres covers; its edge broken by the grain.
        s.water = smoothstep(0.0005, 0.004, at.m.y - 0.0015 * fine) * open * up;
        if (at.area.x > 0.0 && at.area.y < 0.5 && s.water > 0.0) {{
            s.ripple = wx_ripples(p.xz, wx.params.y) * min(at.area.x, 1.5) * open;
        }}
    }}
    if (WX_SNOW) {{
        // Two centimetres of snow hide the ground; thinner lies in patches.
        let depth = at.m.z * open * up;
        s.snow = smoothstep(0.0, 1.0, depth / 0.02 * (0.7 + 0.6 * fine) - 0.15);
        // DRIFTS: the wind leaves snow in long low swells and ripples, a few
        // degrees of slope, which is what shades a sunlit snowfield.
        let q = p.xz;
        s.drift = 0.07 * vec2<f32>(cos(q.x * 1.3 + 1.7 * sin(q.y * 0.7)), cos(q.y * 1.1 + 1.3 * sin(q.x * 0.9)))
            + 0.05 * vec2<f32>(cos(q.x * 4.1 + 2.3 * fine), sin(q.y * 3.7 - 2.1 * fine));
        // SPARKLE: an ice crystal's face is a mirror, and among millions some
        // face halfway between the sun and the eye. One grain in ~40, its face
        // within a few degrees of that halfway direction, flashes in the sun's
        // own highlight -- and not in shadow, where the sun's mask takes it.
        // Gone before a pixel covers a grain, so far snow does not fizz.
        let g = wx_cell(p.xz / 0.011 + vec2<f32>(p.y * 31.0, 0.0));
        let facing = normalize(normalize(view) + wx.sun_dir.xyz);
        let jitter = (vec2<f32>(f32((g >> 8u) & 0xffu), f32((g >> 16u) & 0xffu)) / 255.0 - vec2<f32>(0.5)) * 0.25;
        s.grain = normalize(facing + vec3<f32>(jitter.x, 0.0, jitter.y));
        s.sparkle = select(0.0, 1.0, (g & 0xffu) < 6u) * (1.0 - smoothstep(0.004, 0.012, footprint)) * s.snow * wx.sun_dir.w;
    }}
    return s;
}}

// Snow lifts the ground it lies on: a point at `p` (PLAYER frame) with
// normal `n`, raised by the snow's depth where it lies open and up.
fn wx_lift(p: vec3<f32>, n: vec3<f32>) -> vec3<f32> {{
    let w = to_world_space(p);
    let at = wx_at(w);
    return p + vec3<f32>(0.0, at.m.z * wx_open(w.y, at.m.w) * smoothstep(0.55, 0.9, n.y), 0.0);
}}
const WX_SNOW_ALBEDO: vec3<f32> = vec3<f32>({sa:?}, {sb:?}, {sc:?});
// Whether this twin draws standing water and snow (`WeatherKinds`): false
// only where none lies, so a twin without one is the same picture there.
const WX_WATER: bool = true;
const WX_SNOW: bool = true;
"#,
        sa = SNOW_ALBEDO[0],
        sb = SNOW_ALBEDO[1],
        sc = SNOW_ALBEDO[2],
    )
}

/// What the ground's fragment stage does with the weather, just before it
/// shades -- on its albedo, roughness, normal and occlusion, which then go
/// into the shading every dry pixel takes. Wet porous ground darkens to about
/// half and its highlight tightens (Lagarde, "Water drop 2"); standing water
/// is a flat, near-mirror film over the darkened ground, its slope the rain's
/// rings; snow is white and rough, smooths the bumps under it, and a grain in
/// seventy is a tilted mirror.
const TERRAIN_APPLY: &str = "    // WEATHER: see `weather::with_weather`.
    let wx_s = wx_surface(in.tex_pos, to_world_direction(n), to_world_space(cam_pos()) - in.tex_pos, pixel_footprint);
    let wx_porous = clamp((rough - 0.2) / 0.6, 0.0, 1.0);
    // Without water (or snow), each mix by its zero cover leaves its value as
    // it was, but for the normal's renormalising, which is kept.
    if (WX_WATER) {
        albedo = albedo * (1.0 - 0.35 * wx_s.wet * wx_porous) * (1.0 - 0.3 * wx_s.water);
    } else {
        albedo = albedo * (1.0 - 0.35 * wx_s.wet * wx_porous) * 1.0;
    }
    rough = mix(rough, mix(min(rough, 0.15), max(0.85 * rough, 0.65), wx_porous), wx_s.wet);
    if (WX_WATER) {
        rough = mix(rough, 0.035, wx_s.water);
        shaded_n = normalize(mix(shaded_n, to_player_direction(normalize(vec3<f32>(-wx_s.ripple.x, 1.0, -wx_s.ripple.y))), wx_s.water));
    } else {
        shaded_n = normalize(shaded_n);
    }
    if (WX_SNOW) {
        let wx_snow_n = normalize(mix(normalize(n + to_player_direction(vec3<f32>(-wx_s.drift.x, 0.0, -wx_s.drift.y))), to_player_direction(wx_s.grain), wx_s.sparkle));
        albedo = mix(albedo, WX_SNOW_ALBEDO, wx_s.snow);
        rough = mix(rough, mix(0.62, 0.06, wx_s.sparkle), wx_s.snow);
        shaded_n = normalize(mix(shaded_n, wx_snow_n, wx_s.snow));
        ao_map = mix(ao_map, 1.0, wx_s.snow);
    } else {
        shaded_n = normalize(shaded_n);
    }
";

/// THE GROUND'S WEATHER TWIN of a ground shader (any of its roles and twins):
/// the map's group, snow lifting its vertices, and [`TERRAIN_APPLY`] before
/// its shading. `None` if the shader no longer has the lines it edits -- a
/// test catches that (`every_ground_shader_takes_its_weather_twin`).
pub fn with_weather(src: &str) -> Option<String> {
    let mut s = src.to_string();
    let edits: [(&str, &str); 4] = [
        (
            "    out.clip      = cam_view_proj() * vec4<f32>(v.pos, 1.0);",
            "    let wx_pos = wx_lift(v.pos, v.norm);\n    out.clip      = cam_view_proj() * vec4<f32>(wx_pos, 1.0);",
        ),
        ("    out.world_pos = v.pos;", "    out.world_pos = wx_pos;"),
        ("    out.tex_pos   = to_world_space(v.pos);", "    out.tex_pos   = to_world_space(wx_pos);"),
        ("    let rough = specular_aa_roughness(", "    var rough = specular_aa_roughness("),
    ];
    for (from, to) in edits {
        if s.matches(from).count() != 1 {
            return None;
        }
        s = s.replacen(from, to, 1);
    }
    // Before the shading (the scene's readers) or the reflection's (the pass).
    let at = ["    let lit = shade_material_env(\n", "    probe_fragment = in.clip;\n"]
        .iter()
        .find_map(|m| (s.matches(m).count() == 1).then(|| s.find(m).unwrap()))?;
    s.insert_str(at, TERRAIN_APPLY);
    s.push_str(&map_block());
    s.push_str(&surface_block());
    Some(s)
}

/// WHAT A WEATHER TWIN DRAWS (`WX_WATER`, `WX_SNOW` in its surface): both
/// standing water and snow, or one of them where the other lies nowhere --
/// its cover is then zero at every pixel, and every mix by it leaves its value
/// as it was, so the twin without it is the same picture with less code
/// (PIPESTATS 2026-10-08: the snow's part 200 instructions, the water's 264).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeatherKinds {
    Both,
    /// Standing water and rain rings; no snow.
    Snowless,
    /// Snow; no standing water.
    Waterless,
}

impl WeatherKinds {
    pub const ALL: [WeatherKinds; 3] = [WeatherKinds::Both, WeatherKinds::Snowless, WeatherKinds::Waterless];

    /// The twin for ground where standing water may lie (`water`) and snow
    /// may (`snow`).
    pub fn of(water: bool, snow: bool) -> Self {
        match (water, snow) {
            (_, false) => WeatherKinds::Snowless,
            (false, true) => WeatherKinds::Waterless,
            (true, true) => WeatherKinds::Both,
        }
    }

    pub fn index(self) -> usize {
        self as usize
    }

    pub fn label(self) -> &'static str {
        match self {
            WeatherKinds::Both => "",
            WeatherKinds::Snowless => "_snowless",
            WeatherKinds::Waterless => "_waterless",
        }
    }
}

/// [`with_weather`] drawing only `kinds`.
pub fn with_weather_kinds(src: &str, kinds: WeatherKinds) -> Option<String> {
    let s = with_weather(src)?;
    let off = |s: String, name: &str| {
        let on = format!("const {name}: bool = true;");
        (s.matches(&on).count() == 1).then(|| s.replacen(&on, &format!("const {name}: bool = false;"), 1))
    };
    match kinds {
        WeatherKinds::Both => Some(s),
        WeatherKinds::Snowless => off(s, "WX_SNOW"),
        WeatherKinds::Waterless => off(s, "WX_WATER"),
    }
}

/// Whether an area's map (`WeatherMaps::upload_map`'s texels) holds standing
/// water and snow anywhere a pixel could show them: deeper than the least
/// `wx_surface` draws (water over half a millimetre; snow over 2.3 mm), with
/// room for the texture's half floats.
pub fn map_holds(texels: &[[f32; 4]]) -> (bool, bool) {
    let water = texels.iter().any(|t| t[1] > 0.0004);
    let snow = texels.iter().any(|t| t[2] > 0.002);
    (water, snow)
}

/// THE FALLING RAIN AND SNOW AND THE SPLASHES, one draw: instances
/// `[0, rain)` rain, `[rain, rain + snow)` snow, then splashes. Each is a
/// function of its index and the clock, laid in a box round the head and
/// fixed in the world (wrapped, not carried), so walking through the rain
/// passes drops. One is drawn only where an area of its kind falls, faded at
/// its edge, and only above the cover height -- under a roof it is gone.
fn particle_shader() -> String {
    format!(
        "{aces}{map}{body}",
        aces = crate::renderer::tonemap::wgsl_aces_block(),
        map = map_block(),
        body = r#"
var<private> view_slot: i32 = 0;
struct Camera { view_proj: array<mat4x4<f32>, 2> }
@group(0) @binding(0) var<uniform> camera: Camera;
// The ground's baked map (`TerrainMaterial`'s layer 0): sky visibility in
// red, the sun's signed distance in green, its penumbra in blue, alpha 0
// where it was baked.
@group(1) @binding(1) var wx_ground_samp: sampler;
@group(1) @binding(6) var wx_ground_tex: texture_2d_array<f32>;

struct PVOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    // Its sky light, its sun light (both exposed), and alpha.
    @location(1) @interpolate(flat) sky: vec4<f32>,
    @location(2) @interpolate(flat) sun: vec3<f32>,
    // Where its ray to the sun meets the ground (world x, z), and its shape:
    // 0 a streak, 1 a flake, 2 a splash.
    @location(3) @interpolate(flat) ground: vec3<f32>,
    @location(4) @interpolate(flat) torch: vec3<f32>,
}

fn wx_to_player(p: vec3<f32>) -> vec3<f32> {
    let s = sin(wx.frame.w);
    let c = cos(wx.frame.w);
    let q = p - wx.frame.xyz;
    return vec3<f32>(c * q.x - s * q.z, q.y, s * q.x + c * q.z);
}
fn wx_to_player_dir(d: vec3<f32>) -> vec3<f32> {
    let s = sin(wx.frame.w);
    let c = cos(wx.frame.w);
    return vec3<f32>(c * d.x - s * d.z, d.y, s * d.x + c * d.z);
}
// `c` wrapped into the window of width `w` starting at `lo`: fixed in the
// world, re-entering at the far side when it leaves.
fn wx_wrap(c: f32, lo: f32, w: f32) -> f32 {
    return c - w * floor((c - lo) / w);
}

const WX_GONE: vec4<f32> = vec4<f32>(2.0, 2.0, 2.0, 1.0);

@vertex fn vs_main(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> PVOut {
    var out: PVOut;
    out.clip = WX_GONE;
    out.uv = vec2<f32>(0.0);
    out.sky = vec4<f32>(0.0);
    out.sun = vec3<f32>(0.0);
    out.ground = vec3<f32>(0.0);
    out.torch = vec3<f32>(0.0);
    let rain_n = u32(wx.params.z);
    let snow_n = u32(wx.params.w);
    let shape = select(select(2u, 1u, ii < rain_n + snow_n), 0u, ii < rain_n);
    let t = wx.params.y;
    let r0 = wx_rand(ii * 7u + 1u);
    let r1 = wx_rand(ii * 7u + 2u);
    let r2 = wx_rand(ii * 7u + 3u);
    let r3 = wx_rand(ii * 7u + 4u);
    let r4 = wx_rand(ii * 7u + 5u);
    let r5 = wx_rand(ii * 7u + 6u);
    let r6 = wx_rand(ii * 7u + 7u);
    let head = wx.head.xyz;
    let wind = wx.wind.xy;
    var p: vec3<f32>;
    var vel: vec3<f32>;
    var alpha = 1.0;
    var splash_age = 0.0;
    if (shape == 0u) {
        // RAIN: 6.5-9 m/s, slanted by the wind, in a 16 x 9 x 16 m box.
        let fall = 6.5 + 2.5 * r3;
        let cx = r0 * 16.0 + wind.x * t;
        let cz = r1 * 16.0 + wind.y * t;
        let cy = r2 * 9.0 - fall * t;
        p = vec3<f32>(wx_wrap(cx, head.x - 8.0, 16.0), wx_wrap(cy, head.y - 3.5, 9.0), wx_wrap(cz, head.z - 8.0, 16.0));
        vel = vec3<f32>(wind.x, -fall, wind.y);
        alpha = 0.5;
    } else if (shape == 1u) {
        // SNOW: 0.9-1.4 m/s, wandering on the air and carried by the wind.
        let fall = 0.9 + 0.5 * r3;
        let wob = WX_WOBBLE_T;
        let cx = r0 * 10.0 + wind.x * t + wob.x;
        let cz = r1 * 10.0 + wind.y * t + wob.y;
        let cy = r2 * 7.0 - fall * t;
        p = vec3<f32>(wx_wrap(cx, head.x - 5.0, 10.0), wx_wrap(cy, head.y - 2.8, 7.0), wx_wrap(cz, head.z - 5.0, 10.0));
        vel = vec3<f32>(wind.x, -fall, wind.y);
        alpha = 0.9;
    } else {
        // A SPLASH where a drop strikes: each spot struck every 0.5-1.1 s,
        // its crown standing 0.12 s, on whatever the sky meets there.
        let period = 0.5 + 0.6 * r3;
        splash_age = fract(t / period + r4) * period;
        if (splash_age > 0.12) {
            return out;
        }
        p = vec3<f32>(wx_wrap(r0 * 12.0, head.x - 6.0, 12.0), 0.0, wx_wrap(r1 * 12.0, head.z - 6.0, 12.0));
        vel = vec3<f32>(0.0, 1.0, 0.0);
        alpha = 0.55 * (1.0 - splash_age / 0.12);
    }
    let at = wx_at(p);
    let kind = select(0.0, 1.0, shape == 1u);
    if (at.area.x <= 0.0 || abs(at.area.y - kind) > 0.5 || r6 * 1.5 > at.area.x || at.fade <= 0.0) {
        return out;
    }
    if (shape == 2u) {
        p.y = at.m.w;
    } else if (p.y < at.m.w) {
        // Under the roof, the ridge, the ground: what fell there was stopped.
        return out;
    }
    alpha = alpha * at.fade;

    // In the player's frame, as the camera has it.
    let pp = wx_to_player(p);
    let to_eye = normalize(wx_to_player(head) - pp);
    let depth = max((camera.view_proj[view_slot] * vec4<f32>(pp, 1.0)).w, 0.05);
    // A pixel and a half at the least: thinner falls between the samples and
    // twinkles. Drawn that wide, its light spread over it.
    let least = 0.75 * wx.pixel.x * depth;
    var corner = vec2<f32>(-1.0, -1.0);
    let k = vi % 6u;
    if (k == 1u || k == 4u) { corner = vec2<f32>(1.0, -1.0); }
    if (k == 2u || k == 3u) { corner = vec2<f32>(1.0, 1.0); }
    if (k == 5u) { corner = vec2<f32>(-1.0, 1.0); }
    var world: vec3<f32>;
    if (shape == 0u) {
        // A streak: how far a drop falls while the eye takes it in.
        let along = wx_to_player_dir(normalize(vel));
        let side = cross(along, to_eye);
        let across = select(wx.right.xyz, normalize(side), dot(side, side) > 1e-8);
        // About 30 ms of fall: what the eye smears a drop into.
        let half_len = max(0.5 * length(vel) * 0.03, 2.0 * least);
        let half_w = max(0.0012, least);
        // Thinner than its pixel and a half, its light spread over it -- but
        // seen, as the eye sees a streak finer than its acuity: by its
        // contrast over the length, not its width.
        alpha = alpha * sqrt(0.0012 / half_w);
        world = pp - along * half_len + across * (corner.x * half_w) + along * (corner.y * half_len);
    } else if (shape == 1u) {
        // Flakes clump: a heavy fall's are 5-15 mm across.
        let r = 0.003 + 0.005 * r5;
        let half = max(r, least);
        alpha = alpha * (r / half);
        world = pp + wx.right.xyz * (corner.x * half) + wx.up.xyz * (corner.y * half);
    } else {
        // Upright, facing the eye across the ground.
        let flat_eye = normalize(vec3<f32>(to_eye.x, 0.0, to_eye.z) + vec3<f32>(1e-4, 0.0, 0.0));
        let across = normalize(cross(vec3<f32>(0.0, 1.0, 0.0), flat_eye));
        let grow = 0.025 + 0.35 * splash_age;
        let half = max(grow, least);
        alpha = alpha * min(1.0, (grow * grow) / (half * half));
        world = pp + across * (corner.x * half) + vec3<f32>(0.0, 1.0, 0.0) * ((corner.y + 1.0) * half);
    }
    out.clip = camera.view_proj[view_slot] * vec4<f32>(world, 1.0);
    out.uv = corner;

    // LIGHT. The sky; the sun where the ground's baked mask has it reach the
    // point this one's sun ray meets the ground; the torch. A drop passes on
    // the light behind it bent round, and glints toward the sun. A flake is a
    // white diffuser lit from all round -- the snow under it gives back most
    // of what falls on it -- and an ice lattice, so it scatters forward too:
    // a flake against the sun glows. A plain diffuser read as a grey speck
    // against the sunlit snowfield it fell on.
    let sd = wx.sun_dir.xyz;
    var ground = p.xz;
    if (sd.y > 0.05) {
        ground = p.xz - sd.xz * (max(p.y - at.m.w, 0.0) / sd.y);
    }
    out.ground = vec3<f32>(ground, f32(shape));
    let view_w = normalize(p - head);
    var sky_k = 1.0;
    var sun_k = 0.55 + 0.6 * pow(max(dot(view_w, sd), 0.0), 4.0);
    if (shape != 1u) {
        sky_k = 0.6;
        sun_k = 0.04 + 0.8 * pow(max(dot(view_w, sd), 0.0), 12.0);
    }
    out.sky = vec4<f32>(wx.sky.rgb * sky_k, alpha);
    out.sun = wx.sun.rgb * (sun_k * wx.sun_dir.w);
    if (wx.torch_pos.w > 0.0) {
        let d = pp - wx.torch_pos.xyz;
        let dist2 = max(dot(d, d), 1e-4);
        let cone = smoothstep(wx.torch_dir.w, wx.torch_col.w, dot(d * inverseSqrt(dist2), wx.torch_dir.xyz));
        let reach = clamp(1.0 - dist2 / (wx.torch_pos.w * wx.torch_pos.w), 0.0, 1.0);
        out.torch = wx.torch_col.rgb * (cone * reach * select(0.5, 0.35, shape == 1u) / dist2);
    }
    return out;
}

@fragment fn fs_main(in: PVOut) -> @location(0) vec4<f32> {
    var a: f32;
    let shape = u32(in.ground.z + 0.5);
    if (shape == 0u) {
        a = (1.0 - in.uv.x * in.uv.x) * (1.0 - smoothstep(0.55, 1.0, abs(in.uv.y)));
    } else if (shape == 1u) {
        a = 1.0 - smoothstep(0.35, 1.0, length(in.uv));
    } else {
        // A crown: a thin arc of water thrown up and out.
        let q = in.uv - vec2<f32>(0.0, -1.6);
        a = (1.0 - smoothstep(0.0, 0.22, abs(length(q) - 1.35))) * step(-0.95, in.uv.y);
    }
    var sky_vis = 1.0;
    var sun_vis = 1.0;
    if (wx.ground.z > 0.0) {
        let g = textureSampleLevel(wx_ground_tex, wx_ground_samp, (in.ground.xy - wx.ground.xy) * wx.ground.zw, 0, 0.0);
        sky_vis = max(g.r, 0.5);
        let d = (g.g - 0.5) * (2.0 * WX_SUN_RANGE);
        let w = max(g.b * WX_SUN_RANGE, 0.5);
        sun_vis = select(1.0, smoothstep(-w, w, d), g.a < 0.5);
    }
    let lit = in.sky.rgb * sky_vis + in.sun * sun_vis + in.torch;
    let alpha = clamp(in.sky.a * a, 0.0, 1.0);
    return vec4<f32>(aces_fitted(lit) * alpha, alpha);
}
"#
        .replace("WX_SUN_RANGE", &format!("{:?}", super::brush_pipeline::SUN_MASK_DISTANCE_TEXELS))
        .replace("WX_WOBBLE_T", &snow_wobble("t")),
    )
}

/// A FLAKE'S WANDER on the air at the WGSL time `t` (the expression's text),
/// across x and z: one source for the particle shader and its motion twin,
/// which takes it at two times.
fn snow_wobble(t: &str) -> String {
    format!("0.35 * vec2<f32>(sin({t} * (0.6 + 0.5 * r4) + 6.2832 * r5), cos({t} * (0.5 + 0.4 * r5) + 6.2832 * r4))")
}

/// THE RAIN, SNOW AND SPLASHES' SPACEWARP MOTION (`space_warp::MotionKind::
/// Weather`): the particle shader itself, its vertex stage run twice -- as
/// drawn, and as the previous frame had each one -- and its cover again.
///
/// One source: the twin is [`particle_shader`]'s text with these changes,
/// each checked to have matched (so an edit there that moves one fails
/// here, at build time on the device and in the tests, rather than letting
/// the two drift):
/// - its vertex and fragment stages are plain functions (`wx_vertex`);
/// - the camera is SpaceWarp's (`wx_motion_view`: this frame's, or the
///   previous frame's for the previous place), at a motion pixel's size;
/// - the weather's group is group 1, the ground's (unused) group 3;
/// - the place it is drawn is moved back by how far it fell and drifted in
///   the last `cam.params.z` seconds of the weather's clock (`wx_moved`:
///   the fall and the wind, exactly, as the sim is stateless; a flake's
///   wander at both times), and a splash's crown is as grown as it was.
///
/// Moved back from where it is, not placed again at the earlier time: the
/// box a drop is wrapped into is the head's, and one that wrapped between
/// the two times would have moved across the box.
///
/// Its motion is written over what is behind it by its cover, as a bright
/// sharp layer (`space_warp::BRIGHT_LAYER_CONTRAST`), widened to a motion
/// pixel and a half as the particle shader widens it to an eye pixel and a
/// half, its alpha spread as there: a streak no motion pixel's centre falls
/// in still counts, by its share.
pub fn motion_shader() -> String {
    let mut s = particle_shader();
    let swap = |s: &mut String, from: &str, to: &str, n: usize| {
        assert_eq!(s.matches(from).count(), n, "weather motion twin: `{from}` moved in the particle shader");
        *s = s.replace(from, to);
    };
    swap(&mut s, "@group(0) @binding(0) var<uniform> camera: Camera;", "", 1);
    swap(&mut s, "camera.view_proj[view_slot]", "wx_motion_view()", 2);
    swap(&mut s, "@group(1) @binding(1) var wx_ground_samp", "@group(3) @binding(1) var wx_ground_samp", 1);
    swap(&mut s, "@group(1) @binding(6) var wx_ground_tex", "@group(3) @binding(6) var wx_ground_tex", 1);
    swap(&mut s, "@group(2) @binding(", "@group(1) @binding(", 3);
    swap(&mut s, "@vertex fn vs_main(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> PVOut", "fn wx_vertex(vi: u32, ii: u32) -> PVOut", 1);
    swap(&mut s, "@fragment fn fs_main(in: PVOut) -> @location(0) vec4<f32>", "fn wx_fragment(in: PVOut) -> vec4<f32>", 1);
    swap(&mut s, "0.75 * wx.pixel.x * depth", "0.75 * wx.pixel.x * wx_pixel_scale * depth", 1);
    swap(&mut s, "    let pp = wx_to_player(p);\n", "    let pp = wx_to_player(p - wx_moved(shape, vel, t, r4, r5));\n", 1);
    swap(&mut s, "let grow = 0.025 + 0.35 * splash_age;", "let grow = 0.025 + 0.35 * max(splash_age - wx_back, 0.0);", 1);
    // The cover: the fragment stage's own shape, taken from it.
    let from = s.find("    var a: f32;\n").expect("weather motion twin: the particle's cover moved");
    let to = from + s[from..].find("    var sky_vis").expect("weather motion twin: the particle's cover moved");
    let cover = s[from..to].to_string();
    format!(
        r#"{motion}
{s}
var<private> wx_back: f32 = 0.0;
var<private> wx_before: bool = false;
var<private> wx_pixel_scale: f32 = 1.0;

fn wx_motion_view() -> mat4x4<f32> {{
    if (wx_before) {{
        return cam.prev;
    }}
    return cam.curr;
}}

// How far a particle fell and drifted in the last `wx_back` seconds.
fn wx_moved(shape: u32, vel: vec3<f32>, t: f32, r4: f32, r5: f32) -> vec3<f32> {{
    if (shape == 2u) {{
        return vec3<f32>(0.0);
    }}
    var d = vel * wx_back;
    if (shape == 1u) {{
        let now = {wob_now};
        let before = {wob_before};
        d = d + vec3<f32>(now.x - before.x, 0.0, now.y - before.y);
    }}
    return d;
}}

struct WxMotionOut {{
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) sky: vec4<f32>,
    @location(2) @interpolate(flat) ground: vec3<f32>,
    @location(3) curr: vec4<f32>,
    @location(4) prev: vec4<f32>,
}}

@vertex fn vs_weather_motion(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> WxMotionOut {{
    wx_pixel_scale = cam.reflect.x;
    wx_back = cam.params.z;
    wx_before = true;
    let before = wx_vertex(vi, ii);
    wx_back = 0.0;
    wx_before = false;
    let now = wx_vertex(vi, ii);
    var out: WxMotionOut;
    out.clip = now.clip;
    out.uv = now.uv;
    out.sky = now.sky;
    out.ground = now.ground;
    out.curr = now.clip;
    out.prev = before.clip;
    return out;
}}

@fragment fn fs_weather_motion(in: WxMotionOut) -> @location(0) vec4<f32> {{
{cover}    let w = layer_weight(in.sky.a * a, {bright:?});
    return vec4<f32>(motion_of(in.curr, in.prev).xyz * w, w);
}}
"#,
        motion = crate::renderer::space_warp::motion_block(),
        wob_now = snow_wobble("t"),
        wob_before = snow_wobble("(t - wx_back)"),
        bright = crate::renderer::space_warp::BRIGHT_LAYER_CONTRAST,
    )
}

/// THE RAIN AND SNOW SEEN FROM AFAR: each area's column of falling water or
/// snow as one box, its back faces drawn so every pixel it covers is shaded
/// once from inside or out. The fragment finds where its ray enters the box
/// and where it leaves it -- or meets what the probe pass drew, the ground or
/// a wall -- and takes three samples between for how much of the column is
/// there: faded in from the area's edges, from the ground up to a height and
/// thinning above, with slow drifting shafts, and none within the particles'
/// reach of the head (they take over there). Over that, at the middle of the
/// ray, falling streaks or flakes in the view's own angles, each at least two
/// pixels and a half, scrolling down at the rain's or the snow's speed. A
/// scattering medium's alpha, `1 - exp(-sigma * depth)`, lit by the sky and
/// the sun as the particles are. No march in any scene shader: one draw an
/// area, on the pixels the box covers.
fn veil_shader() -> String {
    // The cube, wound counter-clockwise seen from outside.
    let mut cube = String::new();
    let faces: [([f32; 3], [[f32; 3]; 4]); 6] = [
        ([1.0, 0.0, 0.0], [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [1.0, 0.0, 1.0]]),
        ([-1.0, 0.0, 0.0], [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 1.0, 0.0]]),
        ([0.0, 1.0, 0.0], [[0.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 0.0]]),
        ([0.0, -1.0, 0.0], [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
        ([0.0, 0.0, 1.0], [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]]),
        ([0.0, 0.0, -1.0], [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
    ];
    for (n, q) in faces {
        let sub = |a: [f32; 3], b: [f32; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
        let (u, v) = (sub(q[1], q[0]), sub(q[2], q[0]));
        let c = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
        let out = c[0] * n[0] + c[1] * n[1] + c[2] * n[2] > 0.0;
        let order = if out { [0, 1, 2, 0, 2, 3] } else { [0, 2, 1, 0, 3, 2] };
        for i in order {
            let p = q[i];
            cube.push_str(&format!("vec3<f32>({:?}, {:?}, {:?}), ", p[0], p[1], p[2]));
        }
    }
    format!(
        "{aces}{map}const VEIL_CUBE = array<vec3<f32>, 36>({cube});\nconst VEIL_NEAR: f32 = {near:?};\nconst VEIL_FAR: f32 = {far:?};\n{body}",
        aces = crate::renderer::tonemap::wgsl_aces_block(),
        map = map_block(),
        near = crate::renderer::brush_pipeline::probe_pass::EYE_NEAR,
        far = crate::renderer::brush_pipeline::probe_pass::EYE_FAR,
        body = r#"
var<private> view_slot: i32 = 0;
struct Camera { view_proj: array<mat4x4<f32>, 2> }
@group(0) @binding(0) var<uniform> camera: Camera;
// What the probe pass drew: the ground and the walls the column ends at.
@group(3) @binding(1) var veil_depth: texture_depth_2d_array;

struct VeilOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) at: vec4<f32>,
    // The column's bottom and top, world y; its area.
    @location(2) @interpolate(flat) span: vec2<f32>,
    @location(3) @interpolate(flat) area: u32,
}

fn veil_to_player(p: vec3<f32>) -> vec3<f32> {
    let s = sin(wx.frame.w);
    let c = cos(wx.frame.w);
    let q = p - wx.frame.xyz;
    return vec3<f32>(c * q.x - s * q.z, q.y, s * q.x + c * q.z);
}

// Area `k`'s map at world x/z, clamped to its box.
fn veil_map(k: u32, xz: vec2<f32>) -> vec4<f32> {
    let a0 = wx.areas[k * 3u];
    let a1 = wx.areas[k * 3u + 1u];
    let uv = clamp((xz - a0.xy) * a0.zw, vec2<f32>(0.0), vec2<f32>(1.0));
    return textureSampleLevel(wx_map, wx_samp, uv * a1.xy, i32(k), 0.0);
}

// How high a column stands over its ground, metres: rain from a cloud
// above the view, thinning; snow lower and softer.
fn veil_height(snow: bool) -> f32 {
    return select(24.0, 22.0, snow);
}

@vertex fn vs_veil(@builtin(vertex_index) vi: u32, @builtin(instance_index) k: u32) -> VeilOut {
    var out: VeilOut;
    out.clip = vec4<f32>(2.0, 2.0, 2.0, 1.0);
    out.world = vec3<f32>(0.0);
    out.at = vec4<f32>(0.0);
    out.span = vec2<f32>(0.0);
    out.area = k;
    if (k >= u32(wx.params.x)) {
        return out;
    }
    let a0 = wx.areas[k * 3u];
    let a1 = wx.areas[k * 3u + 1u];
    if (a1.w <= 0.0) {
        return out;
    }
    let lo = a0.xy;
    let hi = a0.xy + vec2<f32>(1.0) / a0.zw;
    var top = -1.0e4;
    var bottom = 1.0e4;
    for (var i = 0u; i < 3u; i = i + 1u) {
        for (var j = 0u; j < 3u; j = j + 1u) {
            let c = veil_map(k, mix(lo, hi, vec2<f32>(f32(i), f32(j)) * 0.5)).w;
            top = max(top, c);
            bottom = min(bottom, c);
        }
    }
    top = top + veil_height(a1.z > 0.5);
    bottom = bottom - 2.0;
    let c = VEIL_CUBE[vi];
    let w = vec3<f32>(mix(lo.x, hi.x, c.x), mix(bottom, top, c.y), mix(lo.y, hi.y, c.z));
    let clip = camera.view_proj[view_slot] * vec4<f32>(veil_to_player(w), 1.0);
    out.clip = clip;
    out.at = clip;
    out.world = w;
    out.span = vec2<f32>(bottom, top);
    return out;
}

@fragment fn fs_veil(in: VeilOut) -> @location(0) vec4<f32> {
    let k = in.area;
    let a0 = wx.areas[k * 3u];
    let a1 = wx.areas[k * 3u + 1u];
    let a2 = wx.areas[k * 3u + 2u];
    let snow = a1.z > 0.5;
    let eye = wx.head.xyz;
    let to = in.world - eye;
    let t_back = max(length(to), 1e-3);
    let dir = to / t_back;
    // Where the ray enters the box (0 inside it).
    let lo = vec3<f32>(a0.x, in.span.x, a0.y);
    let hi = vec3<f32>(a0.x + 1.0 / a0.z, in.span.y, a0.y + 1.0 / a0.w);
    let inv = vec3<f32>(1.0) / select(dir, vec3<f32>(1e-6), abs(dir) < vec3<f32>(1e-6));
    let ta = (lo - eye) * inv;
    let tb = (hi - eye) * inv;
    let t_in = max(max(max(min(ta.x, tb.x), min(ta.y, tb.y)), min(ta.z, tb.z)), 0.0);
    // Where it meets what the probe pass drew, along the ray.
    var t_end = t_back;
    if (in.at.w > 0.0) {
        let ndc = in.at.xyz / in.at.w;
        let size = vec2<f32>(textureDimensions(veil_depth));
        let texel = clamp(vec2<i32>(vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5) * size), vec2<i32>(0), vec2<i32>(size) - vec2<i32>(1));
        let z = textureLoad(veil_depth, texel, view_slot, 0);
        if (z < 1.0) {
            let scene = VEIL_NEAR * VEIL_FAR / (VEIL_FAR - z * (VEIL_FAR - VEIL_NEAR));
            t_end = min(t_end, scene * t_back / in.at.w);
        }
    }
    // The particles' reach: they draw the near part.
    let hand = select(vec2<f32>(6.0, 11.0), vec2<f32>(3.5, 7.5), snow);
    let t0 = max(t_in, hand.x);
    if (t_end <= t0) {
        return vec4<f32>(0.0);
    }
    let time = wx.params.y;
    let height = veil_height(snow);
    let soft = max(2.0 * a2.w, 6.0);
    let dt = (t_end - t0) / 3.0;
    var depth = 0.0;
    for (var i = 0u; i < 3u; i = i + 1u) {
        let t = t0 + (f32(i) + 0.5) * dt;
        let p = eye + dir * t;
        let ground = veil_map(k, p.xz).w;
        let above = p.y - ground;
        let rise = smoothstep(-0.3, 0.3, above) * (1.0 - smoothstep(select(0.15, 0.35, snow) * height, 0.9 * height, above));
        let inside = min(p.xz - a0.xy, a0.xy + vec2<f32>(1.0) / a0.zw - p.xz);
        let edge = smoothstep(0.0, soft, min(inside.x, inside.y));
        let near = smoothstep(hand.x, hand.y, t);
        // Shafts: the fall is heavier here than there, drifting on the wind.
        let shaft = 0.45 + 1.1 * wx_noise(p.xz * 0.11 - a2.xy * (time * 0.09) + vec2<f32>(p.y * 0.02, 0.0));
        depth = depth + rise * edge * near * shaft * dt;
    }
    if (depth <= 0.0) {
        return vec4<f32>(0.0);
    }
    // THE FALL ITSELF, in the view's angles at the column's middle: columns
    // of streaks (or a scatter of flakes) a pixel and a quarter at least
    // either side, so the far rain shimmers rather than aliases.
    let centre = a0.xy + 0.5 / a0.zw;
    let reach = max(length(centre - eye.xz), 1.0);
    let flat_len = max(length(dir.xz), 1e-3);
    let across = atan2(dir.z, dir.x);
    let rise_v = dir.y / flat_len * reach;
    // Seen from under it the fall is end-on and shows no streaks.
    let side_on = 1.0 - smoothstep(0.45, 0.85, abs(dir.y));
    var fall = 1.0;
    if (snow) {
        // Two scatters of flakes at different sizes and speeds, each flake
        // anywhere in its cell and only some cells holding one, so no
        // lattice shows.
        var flakes = 0.0;
        for (var layer = 0u; layer < 2u; layer = layer + 1u) {
            let cell = max(0.5 / reach, 6.0 * wx.pixel.x) * (1.0 + 0.7 * f32(layer));
            let speed = 1.15 - 0.35 * f32(layer);
            let q = vec2<f32>(across / cell + 0.5 * f32(layer), (rise_v / reach + speed * time / reach) / cell);
            let h = wx_cell(floor(q) + vec2<f32>(0.0, 517.0 * f32(layer)));
            let jit = (vec2<f32>(f32(h & 0xffu), f32((h >> 8u) & 0xffu)) / 255.0 - vec2<f32>(0.5)) * 0.8;
            let d = length(fract(q) - vec2<f32>(0.5) - 0.5 * jit);
            flakes = flakes + (1.0 - smoothstep(0.08, 0.24, d)) * select(0.0, 1.0, ((h >> 16u) & 1u) == 0u);
        }
        fall = 0.7 + 1.2 * flakes * side_on;
    } else {
        // Two layers of columns, each a thin line at a jittered place in its
        // column, only some of them carrying a streak at a time.
        var streaks = 0.0;
        for (var layer = 0u; layer < 2u; layer = layer + 1u) {
            let width = max(0.05 / reach, 2.5 * wx.pixel.x) * (1.0 + 0.6 * f32(layer));
            let u = across / width + 0.37 * f32(layer);
            let col = floor(u);
            let h = wx_hash(bitcast<u32>(i32(col)) * 747796405u + 2891336453u + layer * 1013904223u);
            let r = f32(h & 0xffffu) / 65535.0;
            let r2 = f32(h >> 16u) / 65535.0;
            let line = 1.0 - smoothstep(0.1, 0.4, abs(fract(u) - 0.5 - 0.3 * (r2 - 0.5)));
            let len = 1.4 + 1.2 * r;
            let seg = fract((rise_v + (7.0 + 2.0 * r) * time) / len + r * 7.0);
            streaks = streaks + smoothstep(0.0, 0.08, seg) * (1.0 - smoothstep(0.25, 0.4, seg)) * line * step(0.3, r2);
        }
        fall = 0.6 + 1.1 * streaks * side_on;
    }
    let sigma = select(0.016, 0.09, snow) * clamp(a1.w, 0.0, 1.5);
    let alpha = clamp(1.0 - exp(-sigma * depth * fall), 0.0, 0.7);
    var light = wx.sky.rgb * select(0.75, 1.0, snow) + wx.sun.rgb * select(0.12, 0.5, snow) * wx.sun_dir.w;
    return vec4<f32>(aces_fitted(light) * alpha, alpha);
}
"#,
    )
}

/// The falling rain and snow's pipeline: group 0 the scene's camera, group 1
/// the ground's material (`terrain_pipeline::material_bind_group_layout`) for
/// its baked sky and sun, group 2 [`bind_group_layout`]. Premultiplied over,
/// colour only (the eye's alpha is SpaceWarp's), tested against the scene's
/// depth and writing none. Mono and stereo twins, as every scene-pass
/// pipeline.
pub struct WeatherPipeline {
    pub pipeline: RenderPipeline,
    /// The areas' columns seen from afar (`veil_shader`): group 3 the
    /// probe pass's read group, for its depth.
    pub veils: RenderPipeline,
}

impl WeatherPipeline {
    pub fn new(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        ground_layout: &BindGroupLayout,
        weather_layout: &BindGroupLayout,
        samples: u32,
        view: crate::renderer::multiview::ViewMode,
    ) -> Self {
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("weather_particles"),
            source: ShaderSource::Wgsl(view.shader(particle_shader()).into()),
        });
        let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("weather_particles_layout"),
            bind_group_layouts: &[Some(uniform_layout), Some(ground_layout), Some(weather_layout)],
            immediate_size: 0,
        });
        let over = BlendComponent { src_factor: BlendFactor::One, dst_factor: BlendFactor::OneMinusSrcAlpha, operation: BlendOperation::Add };
        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("weather_particles"),
            layout: Some(&layout),
            vertex: VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: PipelineCompilationOptions::default(),
                buffers: &[],
            },
            fragment: Some(FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: PipelineCompilationOptions::default(),
                targets: &[Some(ColorTargetState {
                    format,
                    blend: Some(BlendState { color: over, alpha: BlendComponent::OVER }),
                    write_mask: ColorWrites::COLOR,
                })],
            }),
            primitive: PrimitiveState { topology: PrimitiveTopology::TriangleList, cull_mode: None, ..Default::default() },
            depth_stencil: Some(DepthStencilState {
                format: TextureFormat::Depth32Float,
                depth_write_enabled: Some(false),
                depth_compare: Some(CompareFunction::LessEqual),
                stencil: StencilState::default(),
                bias: DepthBiasState::default(),
            }),
            multisample: MultisampleState { count: samples, ..Default::default() },
            multiview_mask: view.mask(),
            cache: None,
        });
        let veil_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("weather_veils"),
            source: ShaderSource::Wgsl(view.shader(veil_shader()).into()),
        });
        let probe_layout = crate::renderer::brush_pipeline::probe_pass::bind_group_layout(device);
        let veil_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("weather_veils_layout"),
            bind_group_layouts: &[Some(uniform_layout), Some(ground_layout), Some(weather_layout), Some(&probe_layout)],
            immediate_size: 0,
        });
        let veils = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("weather_veils"),
            layout: Some(&veil_layout),
            vertex: VertexState {
                module: &veil_module,
                entry_point: Some("vs_veil"),
                compilation_options: PipelineCompilationOptions::default(),
                buffers: &[],
            },
            fragment: Some(FragmentState {
                module: &veil_module,
                entry_point: Some("fs_veil"),
                compilation_options: PipelineCompilationOptions::default(),
                targets: &[Some(ColorTargetState {
                    format,
                    blend: Some(BlendState { color: over, alpha: BlendComponent::OVER }),
                    write_mask: ColorWrites::COLOR,
                })],
            }),
            // The box's far faces: each covered pixel once, inside or out.
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                front_face: FrontFace::Ccw,
                cull_mode: Some(Face::Front),
                ..Default::default()
            },
            depth_stencil: Some(DepthStencilState {
                format: TextureFormat::Depth32Float,
                depth_write_enabled: Some(false),
                depth_compare: Some(CompareFunction::Always),
                stencil: StencilState::default(),
                bias: DepthBiasState::default(),
            }),
            multisample: MultisampleState { count: samples, ..Default::default() },
            multiview_mask: view.mask(),
            cache: None,
        });
        Self { pipeline, veils }
    }

    /// The areas' columns seen from afar, after the particles: `probe` the
    /// probe pass's read group (its depth ends each column at the ground and
    /// the walls). One box an area that is falling.
    pub fn draw_veils(&self, pass: &mut RenderPass<'_>, camera: &BindGroup, ground: &BindGroup, weather: &BindGroup, probe: &BindGroup) {
        pass.set_pipeline(&self.veils);
        pass.set_bind_group(0, camera, &[]);
        pass.set_bind_group(1, ground, &[]);
        pass.set_bind_group(2, weather, &[]);
        pass.set_bind_group(3, probe, &[]);
        pass.draw(0..36, 0..MAX_AREAS as u32);
    }

    /// Draw `counts` (rain, snow, splashes) -- as [`particle_counts`] gave
    /// them and [`WeatherMaps::params`] carries them -- after everything
    /// opaque.
    pub fn draw(&self, pass: &mut RenderPass<'_>, camera: &BindGroup, ground: &BindGroup, weather: &BindGroup, counts: (u32, u32, u32)) {
        let total = counts.0 + counts.1 + counts.2;
        if total == 0 {
            return;
        }
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, camera, &[]);
        pass.set_bind_group(1, ground, &[]);
        pass.set_bind_group(2, weather, &[]);
        pass.draw(0..VERTICES_PER_PARTICLE, 0..total);
    }
}

/// What the frame knows that the particles' light needs, to fill
/// [`WeatherUniform`]: the head (world), the eye's right and up (player
/// frame), the player frame, the sky's and sun's light (unexposed), the torch.
#[derive(Clone, Copy, Debug, Default)]
pub struct ParticleView {
    pub head_world: [f32; 3],
    pub right: [f32; 3],
    pub up: [f32; 3],
    pub frame: [f32; 4],
    /// The mean radiance a white diffuser takes from the sky (irradiance over
    /// pi), linear, unexposed.
    pub sky: [f32; 3],
    /// What a white diffuser square to the sun reflects (its irradiance over
    /// pi), and the direction toward it, WORLD frame; `None` no sun.
    pub sun: Option<([f32; 3], [f32; 3])>,
    /// Player frame: position, direction, range, cos outer, cos inner,
    /// colour times intensity.
    pub torch: Option<([f32; 3], [f32; 3], f32, f32, f32, [f32; 3])>,
    pub exposure: f32,
    /// An eye pixel's size at unit depth: 2 tan(fov_y / 2) / height.
    pub pixel: f32,
    pub time: f64,
    /// The terrain footprint, min and max x/z; `None` without terrain.
    pub ground: Option<([f32; 2], [f32; 2])>,
}

impl WeatherMaps {
    /// The frame's view and light, and the particles to draw, into the
    /// uniform; then [`Self::write`].
    pub fn set_view(&mut self, v: &ParticleView, counts: (u32, u32, u32), wind: [f32; 2]) {
        let e = v.exposure.max(0.0);
        let p = &mut self.params;
        p.params[1] = (v.time % 4096.0) as f32;
        p.params[2] = counts.0 as f32;
        p.params[3] = counts.1 as f32;
        p.ground = match v.ground {
            Some((lo, hi)) => [lo[0], lo[1], 1.0 / (hi[0] - lo[0]).max(1e-3), 1.0 / (hi[1] - lo[1]).max(1e-3)],
            None => [0.0; 4],
        };
        p.head = [v.head_world[0], v.head_world[1], v.head_world[2], 0.0];
        p.right = [v.right[0], v.right[1], v.right[2], 0.0];
        p.up = [v.up[0], v.up[1], v.up[2], 0.0];
        p.sky = [v.sky[0] * e, v.sky[1] * e, v.sky[2] * e, e];
        match v.sun {
            Some((c, d)) => {
                p.sun = [c[0] * e, c[1] * e, c[2] * e, 0.0];
                p.sun_dir = [d[0], d[1], d[2], 1.0];
            }
            None => {
                p.sun = [0.0; 4];
                p.sun_dir = [0.0, 1.0, 0.0, 0.0];
            }
        }
        match v.torch {
            Some((pos, dir, range, outer, inner, col)) => {
                p.torch_pos = [pos[0], pos[1], pos[2], range];
                p.torch_dir = [dir[0], dir[1], dir[2], outer];
                p.torch_col = [col[0] * e, col[1] * e, col[2] * e, inner];
            }
            None => {
                p.torch_pos = [0.0; 4];
            }
        }
        p.frame = v.frame;
        p.pixel = [v.pixel, 0.0, 0.0, 0.0];
        p.wind = [wind[0], wind[1], 0.0, 0.0];
    }
}

/// The sky's and the sun's light on a drop or a flake, for
/// [`ParticleView`]: the sky as a white diffuser takes it, averaged over a
/// tumbling one's faces (half from above, half from round about), and the
/// sun's (`SkySun::light_rgb`, the same kind of quantity) and its direction.
pub fn particle_light(
    sky: &crate::renderer::sky::SkyIrradiance,
    sun: Option<&crate::renderer::sky::SkySun>,
) -> ([f32; 3], Option<([f32; 3], [f32; 3])>) {
    let up = sky.evaluate([0.0, 1.0, 0.0]);
    let round = [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0f32]].map(|d| sky.evaluate(d));
    let mean = |c: usize| 0.5 * up[c] + 0.125 * round.iter().map(|r| r[c]).sum::<f32>();
    ([mean(0), mean(1), mean(2)], sun.map(|s| (s.light_rgb, s.direction)))
}

/// A level's weather on the headset (`XrRenderer::set_weather`): its maps,
/// the ground's weather twins, the particles' pipeline, which terrain chunks
/// take the twins, and this frame's numbers.
pub struct WeatherScene {
    pub layout: BindGroupLayout,
    pub maps: WeatherMaps,
    pub twins: crate::renderer::terrain_pipeline::TerrainWeatherTwins,
    pub particles: WeatherPipeline,
    /// The terrain chunks drawn with the twins, by their first index into the
    /// terrain's own index buffer, sorted, each with the areas it touches as
    /// a bit mask.
    pub chunks: Vec<(u32, u32)>,
    /// The ground's gentle readers' weather twins, by [`WeatherKinds`]: see
    /// `ground_twins`. Empty when the ground has no steep tests to take out.
    pub gentle: Vec<[crate::renderer::terrain_pipeline::TerrainPipeline; 4]>,
    /// The weather twin's probe pass and its poolless twin with the repeated
    /// code written once (`ground_twins::dedup_passes`).
    pub dedup_passes: Option<[crate::renderer::terrain_pipeline::TerrainPipeline; 2]>,
    /// Per area, whether its map last uploaded holds standing water, and snow
    /// ([`map_holds`]); both until its first map.
    pub holds: Vec<(bool, bool)>,
    pub areas: Vec<AreaParams>,
    pub texels: Vec<(u32, u32)>,
    /// Rain, snow and splashes drawn this frame.
    pub counts: (u32, u32, u32),
    /// The player's torch this frame, PLAYER frame, for the particles.
    pub torch: Option<crate::renderer::Light>,
    /// The weather's clock, seconds.
    pub seconds: f64,
}

impl WeatherScene {
    /// Whether the terrain chunk starting at `first` (terrain-local) is one
    /// an area touches.
    pub fn weathered(&self, first: u32) -> bool {
        self.areas_of(first).is_some()
    }

    /// The areas the terrain chunk starting at `first` touches, as a bit
    /// mask; `None` for a chunk none does.
    pub fn areas_of(&self, first: u32) -> Option<u32> {
        self.chunks.binary_search_by_key(&first, |c| c.0).ok().map(|i| self.chunks[i].1)
    }

    /// The twin for the chunk starting at `first`: what its areas' maps hold.
    pub fn kinds_of(&self, first: u32) -> WeatherKinds {
        let mask = self.areas_of(first).unwrap_or(u32::MAX);
        let (mut water, mut snow) = (false, false);
        for k in 0..32usize {
            if mask & (1 << k) != 0 {
                let (w, s) = self.holds.get(k).copied().unwrap_or((true, true));
                water |= w;
                snow |= s;
            }
        }
        WeatherKinds::of(water, snow)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::renderer::terrain_pipeline::{terrain_shader_for, TerrainRole};

    fn validate(label: &str, src: &str) {
        use wgpu::naga;
        let module = naga::front::wgsl::parse_str(src).unwrap_or_else(|e| panic!("{label}: {}", e.emit_to_string(src)));
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
            .validate(&module)
            .unwrap_or_else(|e| panic!("{label}: {e:?}"));
    }

    /// Every ground shader the frame draws has a weather twin that parses
    /// and validates -- the reader and its twins, the probe pass and its
    /// poolless twin -- and the twin changes nothing but what it adds.
    #[test]
    fn every_ground_shader_takes_its_weather_twin() {
        let read = terrain_shader_for(TerrainRole::Read);
        let pass = terrain_shader_for(TerrainRole::ProbePass);
        let baked = crate::renderer::brush_pipeline::sun_reader_shader(read.clone(), crate::renderer::brush_pipeline::FaceSun::Baked);
        let spotless = crate::renderer::lights::without_spot_shadows;
        for (label, src) in [
            ("scene", terrain_shader_for(TerrainRole::Scene)),
            ("read", read.clone()),
            ("spotless read", spotless(read.clone())),
            ("baked read", baked.clone()),
            ("baked spotless read", spotless(baked)),
            ("pass", pass.clone()),
            ("poolless pass", crate::renderer::lights::without_pool_maps(pass)),
        ] {
            let twin = with_weather(&src).unwrap_or_else(|| panic!("{label}: the weather twin's edits no longer match"));
            assert!(twin.contains("wx_surface(in.tex_pos") && twin.contains("wx_lift(v.pos"), "{label}");
            validate(label, &twin);
        }
    }

    /// MEASUREMENT: how much bigger each ground shader's weather twin is, as
    /// naga IR expressions reachable from its fragment stage (every function
    /// it calls counted once) -- a proxy for the instructions the Adreno
    /// compiler emits, to be read beside the headset's PIPESTATS.
    #[test]
    #[ignore]
    fn weather_twin_size() {
        use wgpu::naga;
        fn reach(m: &naga::Module, block: &naga::Block, seen: &mut std::collections::HashSet<naga::Handle<naga::Function>>) {
            for st in block.iter() {
                match st {
                    naga::Statement::Call { function, .. } => {
                        if seen.insert(*function) {
                            reach(m, &m.functions[*function].body, seen);
                        }
                    }
                    naga::Statement::Block(b) => reach(m, b, seen),
                    naga::Statement::If { accept, reject, .. } => {
                        reach(m, accept, seen);
                        reach(m, reject, seen);
                    }
                    naga::Statement::Switch { cases, .. } => {
                        for c in cases {
                            reach(m, &c.body, seen);
                        }
                    }
                    naga::Statement::Loop { body, continuing, .. } => {
                        reach(m, body, seen);
                        reach(m, continuing, seen);
                    }
                    _ => {}
                }
            }
        }
        let size = |src: &str| {
            let m = naga::front::wgsl::parse_str(src).unwrap();
            let fs = m.entry_points.iter().find(|e| e.stage == naga::ShaderStage::Fragment).unwrap();
            let mut seen = std::collections::HashSet::new();
            reach(&m, &fs.function.body, &mut seen);
            fs.function.expressions.len() + seen.iter().map(|f| m.functions[*f].expressions.len()).sum::<usize>()
        };
        let read = terrain_shader_for(TerrainRole::Read);
        let pass = terrain_shader_for(TerrainRole::ProbePass);
        let baked = crate::renderer::brush_pipeline::sun_reader_shader(read.clone(), crate::renderer::brush_pipeline::FaceSun::Baked);
        let spotless = crate::renderer::lights::without_spot_shadows;
        for (label, src) in [
            ("read", read.clone()),
            ("baked spotless read", spotless(baked)),
            ("pass", pass.clone()),
        ] {
            let (dry, wet) = (size(&src), size(&with_weather(&src).unwrap()));
            eprintln!("{label}: {dry} -> {wet} expressions (+{}, +{:.1}%)", wet - dry, 100.0 * (wet - dry) as f32 / dry as f32);
        }
    }

    /// The rain, snow and splashes' SpaceWarp twin validates, and is the
    /// particle shader's own text: every change it makes is checked to have
    /// matched (`motion_shader` panics otherwise), and the draw shader is
    /// still the one it was -- the wander written once (`snow_wobble`).
    #[test]
    fn the_weather_motion_twin_validates_and_is_the_particle_shader() {
        validate("weather motion", &motion_shader());
        let draw = particle_shader();
        assert!(draw.contains(
            "let wob = 0.35 * vec2<f32>(sin(t * (0.6 + 0.5 * r4) + 6.2832 * r5), cos(t * (0.5 + 0.4 * r5) + 6.2832 * r4));"
        ));
        assert!(draw.contains("@vertex fn vs_main(") && !draw.contains("wx_moved"), "the draw shader carries none of the twin");
        let twin = motion_shader();
        for part in ["fn wx_vertex(vi: u32, ii: u32) -> PVOut", "wx_to_player(p - wx_moved(shape, vel, t, r4, r5))", "@vertex fn vs_weather_motion", "@fragment fn fs_weather_motion"] {
            assert!(twin.contains(part), "{part}");
        }
    }

    /// WHERE A DROP WAS: moved back from where it is by its fall and the
    /// wind (`wx_moved`), not placed again at the earlier time -- a drop that
    /// wrapped across the head's box between the frames would have crossed
    /// the box. A CPU twin of the rain's placement at two times.
    #[test]
    fn a_drop_moves_back_by_its_fall_not_across_its_box() {
        let wrap = |c: f32, lo: f32, w: f32| c - w * (((c - lo) / w).floor());
        let (r2, fall, head_y) = (0.999f32, 8.0f32, 1.6f32);
        let (t, back) = (10.0f32, 1.0 / 36.0);
        let y = |t: f32| wrap(r2 * 9.0 - fall * t, head_y - 3.5, 9.0);
        // Find a moment it wraps between the frames.
        let t = (0..10_000).map(|k| t + k as f32 * 0.001).find(|&t| y(t - back) < y(t)).expect("a wrap");
        let placed_again = y(t - back);
        let moved_back = y(t) + fall * back;
        assert!((placed_again - y(t)).abs() > 4.0, "placed again, it jumped across the box");
        assert!((moved_back - y(t) - fall * back).abs() < 1e-4);
    }

    #[test]
    fn the_particle_shader_validates_mono_and_stereo() {
        validate("particles", &particle_shader());
        validate("veils", &veil_shader());
        validate("veils stereo", &crate::renderer::multiview::ViewMode::Stereo.shader(veil_shader()));
        validate("particles stereo", &crate::renderer::multiview::ViewMode::Stereo.shader(particle_shader()));
    }

    #[test]
    fn halves_round_trip_the_maps_values() {
        let back = |h: u16| {
            let s = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
            let e = ((h >> 10) & 0x1f) as i32;
            let m = (h & 0x3ff) as f32;
            s * if e == 0 { m / 1024.0 * 2f32.powi(-14) } else { (1.0 + m / 1024.0) * 2f32.powi(e - 15) }
        };
        for v in [0.0f32, 1.0, 0.5, 0.03, 0.004, -3.4, 12.25, 0.15, -10000.0] {
            let r = back(f16_bits(v));
            assert!((r - v).abs() <= v.abs() * 1e-3 + 1e-6, "{v} -> {r}");
        }
    }

    #[test]
    fn particles_are_drawn_only_near_where_something_falls() {
        let rain = AreaParams { min: [0.0, 0.0], extent: [10.0, 10.0], falling: 1.0, edge: 2.0, ..Default::default() };
        assert_eq!(particle_counts(&[rain], [5.0, 1.6, 5.0]), (RAIN_PARTICLES, 0, SPLASHES));
        assert_eq!(particle_counts(&[rain], [50.0, 1.6, 5.0]), (0, 0, 0), "nothing far away");
        let snow = AreaParams { snow: true, falling: 0.5, ..rain };
        let (r, s, _) = particle_counts(&[snow], [5.0, 1.6, 5.0]);
        assert_eq!(r, 0);
        assert!(s > 0 && s < SNOW_PARTICLES);
        let stopped = AreaParams { falling: 0.0, ..rain };
        assert_eq!(particle_counts(&[stopped], [5.0, 1.6, 5.0]), (0, 0, 0));
    }

    /// GPU SMOKE: the twins and the particles build on a real device, with
    /// the layouts they are drawn with, and a map uploads.
    #[test]
    fn the_weather_pipelines_build_on_a_device() {
        let Some((device, queue)) = crate::renderer::terrain_pipeline::tests::headless_gpu() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let lights = crate::renderer::lights::LightsUniform::new(&device);
        let (_shadows, uniforms) = crate::renderer::uniforms::test_support::scene_uniforms(&device, &lights);
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let layout = bind_group_layout(&device);
        let probe_layout = crate::renderer::brush_pipeline::probe_pass::bind_group_layout(&device);
        let fixups = crate::renderer::probe_fixup::ProbeFixups::new(&device, &uniforms.layout, 1024);
        let _twins = crate::renderer::terrain_pipeline::TerrainPipeline::new_weather_twins(
            &device,
            TextureFormat::Rgba8UnormSrgb,
            &uniforms.layout,
            4,
            &probe_layout,
            &fixups,
            &layout,
        );
        for kinds in WeatherKinds::ALL {
            crate::renderer::ground_twins::gentle_readers(&device, TextureFormat::Rgba8UnormSrgb, &uniforms.layout, 4, &probe_layout, Some((&layout, kinds)))
                .expect("the ground's gentle weather twins");
        }
        crate::renderer::ground_twins::gentle_readers(&device, TextureFormat::Rgba8UnormSrgb, &uniforms.layout, 4, &probe_layout, None)
            .expect("the ground's gentle twins");
        let ground = crate::renderer::terrain_pipeline::material_bind_group_layout(&device);
        let _p = WeatherPipeline::new(&device, TextureFormat::Rgba8UnormSrgb, &uniforms.layout, &ground, &layout, 4, crate::renderer::multiview::ViewMode::Mono);
        let mut maps = WeatherMaps::new(&device, &layout, &[(64, 48), (32, 80)]);
        maps.upload_map(&queue, 1, 32, 80, &vec![[1.0, 0.01, 0.1, 0.0]; 32 * 80]);
        maps.set_area(1, &AreaParams { min: [0.0, 0.0], extent: [8.0, 20.0], snow: true, falling: 1.0, wind: [1.0, 0.0], edge: 2.0 }, (32, 80));
        maps.write(&queue);
        let err = pollster::block_on(scope.pop());
        assert!(err.is_none(), "the weather pipelines failed to build: {err:?}");
        assert_eq!(maps.size, (64, 80));
        assert_eq!(maps.params.params[0], 2.0);
        assert!((maps.params.areas[4][1] - 1.0).abs() < 1e-6 && (maps.params.areas[4][0] - 0.5).abs() < 1e-6);
    }
}
