use bytemuck::{Pod, Zeroable};
use wgpu::*;

pub const MAX_AREAS: usize = 4;

pub const RAIN_PARTICLES: u32 = 4096;
pub const SNOW_PARTICLES: u32 = 8192;
pub const SPLASHES: u32 = 768;

pub const VERTICES_PER_PARTICLE: u32 = 6;

pub const SNOW_ALBEDO: [f32; 3] = [0.86, 0.88, 0.92];

pub const WET_BACK: f32 = 0.435;
pub const WET_OUT: f32 = 0.554;

pub const FILM_ROUGH: f32 = 0.12;

pub const FILM_FROM: f32 = 0.6;

pub const FILM_LEVEL: f32 = 0.875;

pub const WET_SPECKLE: f32 = 0.6;

pub fn wet_albedo(dry: f32) -> f32 {
    WET_OUT * dry * (1.0 + WET_BACK * dry * (1.0 + WET_BACK * dry))
}

pub fn wet_soak(wet: f32, fine: f32) -> f32 {
    (wet * (1.0 + WET_SPECKLE) - WET_SPECKLE * fine).clamp(0.0, 1.0)
}

pub fn wet_film(film: f32, soak: f32) -> f32 {
    film * ((soak - FILM_FROM) / (1.0 - FILM_FROM)).clamp(0.0, 1.0)
}

pub fn film_levels(film: f32, up: f32) -> f32 {
    film * ((up - FILM_LEVEL) / (1.0 - FILM_LEVEL)).clamp(0.0, 1.0)
}

pub fn wet_ground(albedo: [f32; 3], rough: f32, soak: f32, porosity: f32, film: f32) -> ([f32; 3], f32) {
    let k = soak * porosity;
    let f = wet_film(film, soak);
    (albedo.map(|a| a + (wet_albedo(a) - a) * k), rough + (FILM_ROUGH - rough) * f)
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct WeatherUniform {
    pub areas: [[f32; 4]; 3 * MAX_AREAS],

    pub params: [f32; 4],

    pub ground: [f32; 4],

    pub head: [f32; 4],

    pub right: [f32; 4],
    pub up: [f32; 4],

    pub sky: [f32; 4],

    pub sun: [f32; 4],

    pub sun_dir: [f32; 4],

    pub torch_pos: [f32; 4],
    pub torch_dir: [f32; 4],
    pub torch_col: [f32; 4],

    pub frame: [f32; 4],

    pub pixel: [f32; 4],
    pub wind: [f32; 4],
}

impl Default for WeatherUniform {
    fn default() -> Self {
        Self::zeroed()
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AreaParams {
    pub min: [f32; 2],
    pub extent: [f32; 2],
    pub snow: bool,
    pub falling: f32,
    pub wind: [f32; 2],
    pub edge: f32,

    pub shore: bool,
}

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

    (h + ((man >> 12) & 1)) as u16
}

pub struct WeatherMaps {
    texture: Texture,
    pub size: (u32, u32),
    pub layers: u32,
    uniform: Buffer,
    pub bind_group: BindGroup,
    pub params: WeatherUniform,
}

impl WeatherMaps {

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
            if a.shore { 2.0 } else if a.snow { 1.0 } else { 0.0 },
            a.falling.max(0.0),
        ];
        self.params.areas[3 * k + 2] = [a.wind[0], a.wind[1], k as f32, a.edge.max(1e-3)];
        self.params.params[0] = self.params.params[0].max(k as f32 + 1.0);
    }

    pub fn write(&self, queue: &Queue) {
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&self.params));
    }
}

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

pub fn map_block() -> String {
    r#"
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

fn wx_corner(key: u32) -> f32 {
    var x = key * 0x7feb352du;
    x = (x ^ (x >> 15u)) * 0x846ca68bu;
    return f32(x >> 16u) / 65535.0;
}
fn wx_noise(q: vec2<f32>) -> f32 {
    let i = floor(q);
    let f = q - i;
    let s = f * f * (3.0 - 2.0 * f);
    let ii = vec2<i32>(i);
    let key = bitcast<u32>(ii.x) * 1973u + bitcast<u32>(ii.y) * 9277u + 26699u;
    let a = wx_corner(key);
    let b = wx_corner(key + 1973u);
    let c = wx_corner(key + 9277u);
    let d = wx_corner(key + 11250u);
    return mix(mix(a, b, s.x), mix(c, d, s.x), s.y);
}

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

fn wx_open(y: f32, cover: f32) -> f32 {
    return clamp((y - cover + 0.5) / 0.4, 0.0, 1.0);
}
"#
    .to_string()
}

fn surface_block() -> String {
    format!(
        r#"
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
    // The two layers through one copy of the layer: the first's offsets are 0.
    var sum = vec2<f32>(0.0);
    for (var k = 0u; k < 2u; k = k + 1u) {{
        let o = f32(k);
        sum = sum + wx_ripple_layer(xz / 0.35 + vec2<f32>(0.5 * o), t + 0.37 * o, 17u * k);
    }}
    return 0.35 * sum;
}}

struct WxSurface {{
    wet: f32,
    water: f32,
    snow: f32,
    sparkle: f32,

    ripple: vec2<f32>,
    drift: vec2<f32>,
    grain: vec3<f32>,
}}

fn wx_surface(p: vec3<f32>, n: vec3<f32>, view: vec3<f32>, footprint: f32) -> WxSurface {{
    var s: WxSurface;
    let at = wx_at(p);
    let open = wx_open(p.y, at.m.w);
    let up = smoothstep(0.55, 0.9, n.y);
    var fine = 0.0;
    for (var k = 0u; k < 2u; k = k + 1u) {{
        fine = fine + select(0.65, 0.35, k == 1u) * wx_noise(p.xz * select(3.1, 9.7, k == 1u));
    }}

    let damp = select(at.m.x, clamp((mat.sea_line + at.m.x - p.y) * {shore_fade:?}, 0.0, 1.0), at.area.y > 1.5);

    s.wet = clamp(damp * open * {soak_gain:?} - {speckle:?} * fine, 0.0, 1.0);
    s.water = 0.0;
    s.snow = 0.0;
    s.ripple = vec2<f32>(0.0);
    s.drift = vec2<f32>(0.0);
    s.grain = vec3<f32>(0.0);
    s.sparkle = 0.0;
    if (WX_WATER) {{

        s.water = smoothstep(0.0005, 0.004, at.m.y - 0.0015 * fine) * open * up;
        if (at.area.x > 0.0 && at.area.y < 0.5 && s.water > 0.0) {{
            s.ripple = wx_ripples(p.xz, wx.params.y) * min(at.area.x, 1.5) * open;
        }}
    }}
    if (WX_SNOW) {{

        let depth = at.m.z * open * up;
        s.snow = smoothstep(0.0, 1.0, depth / 0.02 * (0.7 + 0.6 * fine) - 0.15);

        let q = p.xz;
        s.drift = 0.07 * vec2<f32>(cos(q.x * 1.3 + 1.7 * sin(q.y * 0.7)), cos(q.y * 1.1 + 1.3 * sin(q.x * 0.9)))
            + 0.05 * vec2<f32>(cos(q.x * 4.1 + 2.3 * fine), sin(q.y * 3.7 - 2.1 * fine));

        let g = wx_cell(p.xz / 0.011 + vec2<f32>(p.y * 31.0, 0.0));
        let facing = normalize(normalize(view) + wx.sun_dir.xyz);
        let jitter = (vec2<f32>(f32((g >> 8u) & 0xffu), f32((g >> 16u) & 0xffu)) / 255.0 - vec2<f32>(0.5)) * 0.25;
        s.grain = normalize(facing + vec3<f32>(jitter.x, 0.0, jitter.y));
        s.sparkle = select(0.0, 1.0, (g & 0xffu) < 6u) * (1.0 - smoothstep(0.004, 0.012, footprint)) * s.snow * wx.sun_dir.w;
    }}
    return s;
}}

fn wx_lift(p: vec3<f32>, n: vec3<f32>) -> vec3<f32> {{
    let w = to_world_space(p);
    let at = wx_at(w);
    return p + vec3<f32>(0.0, at.m.z * wx_open(w.y, at.m.w) * smoothstep(0.55, 0.9, n.y), 0.0);
}}
const WX_SNOW_ALBEDO: vec3<f32> = vec3<f32>({sa:?}, {sb:?}, {sc:?});

const WX_WET_BACK: f32 = {wet_back:?};
const WX_WET_OUT: f32 = {wet_out:?};
const WX_FILM_ROUGH: f32 = {film_rough:?};
const WX_FILM_FROM: f32 = {film_from:?};
const WX_FILM_LEVEL: f32 = {film_level:?};

const WX_WATER: bool = true;
const WX_SNOW: bool = true;
"#,
        sa = SNOW_ALBEDO[0],
        sb = SNOW_ALBEDO[1],
        sc = SNOW_ALBEDO[2],
        shore_fade = 1.0 / crate::renderer::water_pipeline::SHORE_BAND,
        soak_gain = 1.0 + WET_SPECKLE,
        speckle = WET_SPECKLE,
        wet_back = WET_BACK,
        wet_out = WET_OUT,
        film_rough = FILM_ROUGH,
        film_from = FILM_FROM,
        film_level = FILM_LEVEL,
    )
}

const TERRAIN_APPLY: &str = "
    let wx_s = wx_surface(in.tex_pos, to_world_direction(n), to_world_space(cam_pos()) - in.tex_pos, pixel_footprint);

    let wx_in = wx_s.wet * dot(w, mat.porosity);
    let wx_back = WX_WET_BACK * albedo;
    albedo = albedo * ((1.0 - wx_in * (1.0 - WX_WET_OUT)) + (wx_in * WX_WET_OUT) * (wx_back * wx_back + wx_back));

    let wx_film = dot(w, mat.film) * clamp((wx_s.wet - WX_FILM_FROM) / (1.0 - WX_FILM_FROM), 0.0, 1.0) * clamp((in.tex_pos.y - mat.sea_line) * 20.0, 0.0, 1.0);
    rough = mix(rough, WX_FILM_ROUGH, wx_film);

    var wx_n = mix(shaded_n, n, wx_film * clamp((n.y - WX_FILM_LEVEL) / (1.0 - WX_FILM_LEVEL), 0.0, 1.0));
    if (WX_WATER) {
        albedo = albedo * (1.0 - 0.3 * wx_s.water);
        rough = mix(rough, 0.035, wx_s.water);
        wx_n = mix(wx_n, to_player_direction(normalize(vec3<f32>(-wx_s.ripple.x, 1.0, -wx_s.ripple.y))), wx_s.water);
    }
    if (WX_SNOW) {
        let wx_snow_n = normalize(mix(normalize(n + to_player_direction(vec3<f32>(-wx_s.drift.x, 0.0, -wx_s.drift.y))), to_player_direction(wx_s.grain), wx_s.sparkle));
        albedo = mix(albedo, WX_SNOW_ALBEDO, wx_s.snow);
        rough = mix(rough, mix(0.62, 0.06, wx_s.sparkle), wx_s.snow);
        wx_n = mix(wx_n, wx_snow_n, wx_s.snow);
        ao_map = mix(ao_map, 1.0, wx_s.snow);
    }
    shaded_n = normalize(wx_n);
";

pub fn with_weather(src: &str) -> Option<String> {

    let mut s = crate::renderer::ground_shape::unshaped(src)?;
    let edits: [(&str, &str); 5] = [

        (crate::renderer::terrain_pipeline::SEA_LINE_WGSL, ""),
        (
            "    out.clip      = cam_view_proj() * vec4<f32>(pos, 1.0);",
            "    let wx_pos = wx_lift(pos, v.norm);\n    out.clip      = cam_view_proj() * vec4<f32>(wx_pos, 1.0);",
        ),
        ("    out.world_pos = pos;", "    out.world_pos = wx_pos;"),
        ("    out.tex_pos   = to_world_space(pos);", "    out.tex_pos   = to_world_space(wx_pos);"),
        ("    let rough = specular_aa_roughness(", "    var rough = specular_aa_roughness("),
    ];
    for (from, to) in edits {
        if s.matches(from).count() != 1 {
            return None;
        }
        s = s.replacen(from, to, 1);
    }

    let at = ["    let lit = shade_material_env(\n", "    probe_fragment = in.clip;\n"]
        .iter()
        .find_map(|m| (s.matches(m).count() == 1).then(|| s.find(m).unwrap()))?;
    s.insert_str(at, TERRAIN_APPLY);
    s.push_str(&map_block());
    s.push_str(&surface_block());
    Some(s)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeatherKinds {
    Both,

    Snowless,

    Waterless,

    WetOnly,
}

impl WeatherKinds {
    pub const ALL: [WeatherKinds; 4] = [WeatherKinds::Both, WeatherKinds::Snowless, WeatherKinds::Waterless, WeatherKinds::WetOnly];

    pub fn of(water: bool, snow: bool) -> Self {
        match (water, snow) {
            (true, false) => WeatherKinds::Snowless,
            (false, false) => WeatherKinds::WetOnly,
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
            WeatherKinds::WetOnly => "_wet",
        }
    }
}

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
        WeatherKinds::WetOnly => off(s, "WX_WATER").and_then(|s| off(s, "WX_SNOW")),
    }
}

pub fn map_holds(texels: &[[f32; 4]]) -> (bool, bool) {
    let water = texels.iter().any(|t| t[1] > 0.0004);
    let snow = texels.iter().any(|t| t[2] > 0.002);
    (water, snow)
}

pub fn map_is_wet(texels: &[[f32; 4]]) -> bool {
    texels.iter().any(|t| t[0] > 0.02 || t[1] > 0.0004)
}

fn particle_shader() -> String {
    format!(
        "{aces}{map}{body}",
        aces = crate::renderer::tonemap::wgsl_aces_block(),
        map = map_block(),
        body = r#"
var<private> view_slot: i32 = 0;
struct Camera { view_proj: array<mat4x4<f32>, 2> }
@group(0) @binding(0) var<uniform> camera: Camera;

@group(1) @binding(1) var wx_ground_samp: sampler;
@group(1) @binding(6) var wx_ground_tex: texture_2d_array<f32>;

struct PVOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,

    @location(1) @interpolate(flat) sky: vec4<f32>,
    @location(2) @interpolate(flat) sun: vec3<f32>,

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

        let fall = 6.5 + 2.5 * r3;
        let cx = r0 * 16.0 + wind.x * t;
        let cz = r1 * 16.0 + wind.y * t;
        let cy = r2 * 9.0 - fall * t;
        p = vec3<f32>(wx_wrap(cx, head.x - 8.0, 16.0), wx_wrap(cy, head.y - 3.5, 9.0), wx_wrap(cz, head.z - 8.0, 16.0));
        vel = vec3<f32>(wind.x, -fall, wind.y);
        alpha = 0.5;
    } else if (shape == 1u) {

        let fall = 0.9 + 0.5 * r3;
        let wob = WX_WOBBLE_T;
        let cx = r0 * 10.0 + wind.x * t + wob.x;
        let cz = r1 * 10.0 + wind.y * t + wob.y;
        let cy = r2 * 7.0 - fall * t;
        p = vec3<f32>(wx_wrap(cx, head.x - 5.0, 10.0), wx_wrap(cy, head.y - 2.8, 7.0), wx_wrap(cz, head.z - 5.0, 10.0));
        vel = vec3<f32>(wind.x, -fall, wind.y);
        alpha = 0.9;
    } else {

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

        return out;
    }
    alpha = alpha * at.fade;

    let pp = wx_to_player(p);
    let to_eye = normalize(wx_to_player(head) - pp);
    let depth = max((camera.view_proj[view_slot] * vec4<f32>(pp, 1.0)).w, 0.05);

    let least = 0.75 * wx.pixel.x * depth;
    var corner = vec2<f32>(-1.0, -1.0);
    let k = vi % 6u;
    if (k == 1u || k == 4u) { corner = vec2<f32>(1.0, -1.0); }
    if (k == 2u || k == 3u) { corner = vec2<f32>(1.0, 1.0); }
    if (k == 5u) { corner = vec2<f32>(-1.0, 1.0); }
    var world: vec3<f32>;
    if (shape == 0u) {

        let along = wx_to_player_dir(normalize(vel));
        let side = cross(along, to_eye);
        let across = select(wx.right.xyz, normalize(side), dot(side, side) > 1e-8);

        let half_len = max(0.5 * length(vel) * 0.03, 2.0 * least);
        let half_w = max(0.0012, least);

        alpha = alpha * sqrt(0.0012 / half_w);
        world = pp - along * half_len + across * (corner.x * half_w) + along * (corner.y * half_len);
    } else if (shape == 1u) {

        let r = 0.003 + 0.005 * r5;
        let half = max(r, least);
        alpha = alpha * (r / half);
        world = pp + wx.right.xyz * (corner.x * half) + wx.up.xyz * (corner.y * half);
    } else {

        let flat_eye = normalize(vec3<f32>(to_eye.x, 0.0, to_eye.z) + vec3<f32>(1e-4, 0.0, 0.0));
        let across = normalize(cross(vec3<f32>(0.0, 1.0, 0.0), flat_eye));
        let grow = 0.025 + 0.35 * splash_age;
        let half = max(grow, least);
        alpha = alpha * min(1.0, (grow * grow) / (half * half));
        world = pp + across * (corner.x * half) + vec3<f32>(0.0, 1.0, 0.0) * ((corner.y + 1.0) * half);
    }
    out.clip = camera.view_proj[view_slot] * vec4<f32>(world, 1.0);
    out.uv = corner;

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

fn snow_wobble(t: &str) -> String {
    format!("0.35 * vec2<f32>(sin({t} * (0.6 + 0.5 * r4) + 6.2832 * r5), cos({t} * (0.5 + 0.4 * r5) + 6.2832 * r4))")
}

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

fn veil_shader() -> String {

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

@group(3) @binding(1) var veil_depth: texture_depth_2d_array;

struct VeilOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) at: vec4<f32>,

    @location(2) @interpolate(flat) span: vec2<f32>,
    @location(3) @interpolate(flat) area: u32,
}

fn veil_to_player(p: vec3<f32>) -> vec3<f32> {
    let s = sin(wx.frame.w);
    let c = cos(wx.frame.w);
    let q = p - wx.frame.xyz;
    return vec3<f32>(c * q.x - s * q.z, q.y, s * q.x + c * q.z);
}

fn veil_map(k: u32, xz: vec2<f32>) -> vec4<f32> {
    let a0 = wx.areas[k * 3u];
    let a1 = wx.areas[k * 3u + 1u];
    let uv = clamp((xz - a0.xy) * a0.zw, vec2<f32>(0.0), vec2<f32>(1.0));
    return textureSampleLevel(wx_map, wx_samp, uv * a1.xy, i32(k), 0.0);
}

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

struct VeilFrag {
    @location(0) colour: vec4<f32>,
    @builtin(frag_depth) depth: f32,
}
fn veil_frag(colour: vec4<f32>, depth: f32) -> VeilFrag {
    var f: VeilFrag;
    f.colour = colour;
    f.depth = depth;
    return f;
}

@fragment fn fs_veil(in: VeilOut) -> VeilFrag {
    let k = in.area;
    let a0 = wx.areas[k * 3u];
    let a1 = wx.areas[k * 3u + 1u];
    let a2 = wx.areas[k * 3u + 2u];
    let snow = a1.z > 0.5;
    let eye = wx.head.xyz;
    let to = in.world - eye;
    let t_back = max(length(to), 1e-3);
    let dir = to / t_back;

    let lo = vec3<f32>(a0.x, in.span.x, a0.y);
    let hi = vec3<f32>(a0.x + 1.0 / a0.z, in.span.y, a0.y + 1.0 / a0.w);
    let inv = vec3<f32>(1.0) / select(dir, vec3<f32>(1e-6), abs(dir) < vec3<f32>(1e-6));
    let ta = (lo - eye) * inv;
    let tb = (hi - eye) * inv;
    let t_in = max(max(max(min(ta.x, tb.x), min(ta.y, tb.y)), min(ta.z, tb.z)), 0.0);

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

    let hand = select(vec2<f32>(6.0, 11.0), vec2<f32>(3.5, 7.5), snow);
    let t0 = max(t_in, hand.x);

    let begin = camera.view_proj[view_slot] * vec4<f32>(veil_to_player(eye + dir * t0), 1.0);
    let begin_depth = clamp(begin.z / max(begin.w, 1e-6), 0.0, 1.0);
    if (t_end <= t0) {
        return veil_frag(vec4<f32>(0.0), begin_depth);
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

        let shaft = 0.45 + 1.1 * wx_noise(p.xz * 0.11 - a2.xy * (time * 0.09) + vec2<f32>(p.y * 0.02, 0.0));
        depth = depth + rise * edge * near * shaft * dt;
    }
    if (depth <= 0.0) {
        return veil_frag(vec4<f32>(0.0), begin_depth);
    }

    let centre = a0.xy + 0.5 / a0.zw;
    let reach = max(length(centre - eye.xz), 1.0);
    let flat_len = max(length(dir.xz), 1e-3);
    let across = atan2(dir.z, dir.x);
    let rise_v = dir.y / flat_len * reach;

    let side_on = 1.0 - smoothstep(0.45, 0.85, abs(dir.y));
    var fall = 1.0;
    if (snow) {

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
    return veil_frag(vec4<f32>(aces_fitted(light) * alpha, alpha), begin_depth);
}
"#,
    )
}

pub struct WeatherPipeline {
    pub pipeline: RenderPipeline,

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

            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                front_face: FrontFace::Ccw,
                cull_mode: Some(Face::Front),
                ..Default::default()
            },
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
        Self { pipeline, veils }
    }

    pub fn draw_veils(&self, pass: &mut RenderPass<'_>, camera: &BindGroup, ground: &BindGroup, weather: &BindGroup, probe: &BindGroup) {
        pass.set_pipeline(&self.veils);
        pass.set_bind_group(0, camera, &[]);
        pass.set_bind_group(1, ground, &[]);
        pass.set_bind_group(2, weather, &[]);
        pass.set_bind_group(3, probe, &[]);
        pass.draw(0..36, 0..MAX_AREAS as u32);
    }

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

#[derive(Clone, Copy, Debug, Default)]
pub struct ParticleView {
    pub head_world: [f32; 3],
    pub right: [f32; 3],
    pub up: [f32; 3],
    pub frame: [f32; 4],

    pub sky: [f32; 3],

    pub sun: Option<([f32; 3], [f32; 3])>,

    pub torch: Option<([f32; 3], [f32; 3], f32, f32, f32, [f32; 3])>,
    pub exposure: f32,

    pub pixel: f32,
    pub time: f64,

    pub ground: Option<([f32; 2], [f32; 2])>,
}

impl WeatherMaps {

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

pub fn particle_light(
    sky: &crate::renderer::sky::SkyIrradiance,
    sun: Option<&crate::renderer::sky::SkySun>,
) -> ([f32; 3], Option<([f32; 3], [f32; 3])>) {
    let up = sky.evaluate([0.0, 1.0, 0.0]);
    let round = [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0f32]].map(|d| sky.evaluate(d));
    let mean = |c: usize| 0.5 * up[c] + 0.125 * round.iter().map(|r| r[c]).sum::<f32>();
    ([mean(0), mean(1), mean(2)], sun.map(|s| (s.light_rgb, s.direction)))
}

pub const CHUNK_POOLS: u32 = 16;

pub struct WeatherScene {
    pub layout: BindGroupLayout,
    pub maps: WeatherMaps,
    pub twins: crate::renderer::terrain_pipeline::TerrainWeatherTwins,
    pub particles: WeatherPipeline,

    pub chunks: Vec<(u32, u32)>,

    pub gentle: Vec<[crate::renderer::terrain_pipeline::TerrainPipeline; 4]>,

    pub wet: Option<[[crate::renderer::terrain_pipeline::TerrainPipeline; 4]; 2]>,

    pub dedup_passes: Option<[crate::renderer::terrain_pipeline::TerrainPipeline; 2]>,

    pub holds: Vec<(bool, bool)>,

    pub films: Vec<bool>,
    pub areas: Vec<AreaParams>,
    pub texels: Vec<(u32, u32)>,

    pub counts: (u32, u32, u32),

    pub torch: Option<crate::renderer::Light>,

    pub seconds: f64,
}

impl WeatherScene {

    pub fn weathered(&self, first: u32) -> bool {
        self.areas_of(first).is_some()
    }

    pub fn areas_of(&self, first: u32) -> Option<u32> {
        self.chunks.binary_search_by_key(&first, |c| c.0).ok().map(|i| self.chunks[i].1)
    }

    pub fn films_of(&self, first: u32) -> bool {
        let mask = self.areas_of(first).unwrap_or(u32::MAX);
        (0..CHUNK_POOLS as usize).filter(|k| mask & (1 << k) != 0).any(|k| {
            self.areas.get(k).is_some_and(|a| a.shore)
                || self.films.get(k).copied().unwrap_or(true)
                || self.holds.get(k).is_none_or(|h| h.0)
        })
    }

    pub fn kinds_of(&self, first: u32) -> WeatherKinds {
        let mask = self.areas_of(first).unwrap_or(u32::MAX);
        let (mut water, mut snow) = (false, false);
        for k in 0..CHUNK_POOLS as usize {
            if mask & (1 << k) != 0 {
                let (w, s) = self.holds.get(k).copied().unwrap_or((true, true));
                water |= w && mask & (1 << (k + CHUNK_POOLS as usize)) != 0;
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
            assert!(twin.contains("wx_surface(in.tex_pos") && twin.contains("wx_lift(pos"), "{label}");
            validate(label, &twin);
        }
    }

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

    #[test]
    fn wet_ground_darkens_by_the_wet_surface_law() {
        let closed = |a: f32| WET_OUT * a / (1.0 - WET_BACK * a);
        for a in [0.02f32, 0.08, 0.2, 0.38, 0.6] {
            assert!((wet_albedo(a) / closed(a) - 1.0).abs() < 0.02, "albedo {a}: {} against {}", wet_albedo(a), closed(a));
        }

        assert!((WET_BACK - (1.0 - 1.0 / (1.33f32 * 1.33))).abs() < 2e-3 && (WET_OUT - 0.98 * (1.0 - WET_BACK)).abs() < 2e-3);
        let ratio = |a: f32| wet_albedo(a) / a;
        assert!(ratio(0.08) < ratio(0.2) && ratio(0.2) < ratio(0.38) && ratio(0.38) < ratio(0.6));
        assert!((0.55..0.60).contains(&ratio(0.08)) && (0.64..0.69).contains(&ratio(0.38)), "soil {}, sand {}", ratio(0.08), ratio(0.38));

        let (wet, _) = wet_ground([0.10, 0.20, 0.05], 0.8, 1.0, 1.0, 0.0);
        assert!(wet[1] / wet[0] > 2.0 && wet[1] / wet[2] > 4.0 && wet[1] < 0.2 * 0.62, "{wet:?}");
        let line = crate::renderer::terrain_pipeline::SEA_LINE_WGSL;
        for (written, law) in [("0.435 * albedo", WET_BACK), ("sea_in * 0.446)", 1.0 - WET_OUT), ("(sea_in * 0.554)", WET_OUT)] {
            let n: f32 = written.trim_matches(|c: char| !c.is_ascii_digit() && c != '.').split(' ').find_map(|t| t.parse().ok()).unwrap();
            assert!(line.contains(written) && (n - law).abs() < 1e-6, "the sea line's {written} is not the law's {law}");
        }
    }

    #[test]
    fn a_film_glosses_what_can_hold_one() {
        use crate::renderer::terrain_pipeline::{WET_FILM, WET_POROSITY};
        let soaked = |layer: usize, rough: f32| wet_ground([0.3; 3], rough, 1.0, WET_POROSITY[layer], WET_FILM[layer]).1;
        let (lawn, rock, sand) = (soaked(0, 0.78), soaked(1, 0.7), soaked(3, 0.85));
        assert!(lawn > 0.65 && lawn < 0.78, "a soaked lawn's roughness {lawn}");
        assert!(rock < 0.3 && sand < 0.22 && sand > FILM_ROUGH, "rock {rock}, sand {sand}");

        let (albedo, rough) = wet_ground([0.3; 3], 0.85, FILM_FROM, 1.0, 0.9);
        assert!(rough == 0.85 && albedo[0] < 0.3);
        assert_eq!(wet_ground([0.3; 3], 0.85, 0.0, 1.0, 0.9), ([0.3; 3], 0.85));

        assert_eq!(film_levels(0.75, 1.0), 0.75);
        assert_eq!(film_levels(0.75, 30f32.to_radians().cos()), 0.0);
        let at_15 = film_levels(1.0, 15f32.to_radians().cos());
        assert!(at_15 > 0.6 && at_15 < 0.85, "{at_15}");
        let src = with_weather(&crate::renderer::terrain_pipeline::terrain_shader_for(crate::renderer::terrain_pipeline::TerrainRole::Read)).expect("the ground's weather twin");
        assert!(src.contains("wx_film * clamp((n.y - WX_FILM_LEVEL) / (1.0 - WX_FILM_LEVEL), 0.0, 1.0)"), "the film's levelling left the ground's shader");
        assert!(src.contains(&format!("const WX_FILM_LEVEL: f32 = {FILM_LEVEL:?};")));

        let dark = |layer: usize| 0.3 - wet_ground([0.3; 3], 0.8, 1.0, WET_POROSITY[layer], 0.0).0[0];
        assert!((dark(1) / dark(3) - 0.35).abs() < 1e-3);
    }

    #[test]
    fn ground_wets_in_spots_and_soaks_whole() {
        for fine in [0.0f32, 0.3, 0.7, 1.0] {
            assert_eq!(wet_soak(0.0, fine), 0.0);
            assert_eq!(wet_soak(1.0, fine), 1.0);
        }
        assert!(wet_soak(0.2, 0.1) > 0.2 && wet_soak(0.2, 0.9) == 0.0);
        assert!(wet_soak(0.5, 0.0) > wet_soak(0.5, 1.0) && wet_soak(0.5, 1.0) > 0.0);

        assert!((0..=20).map(|k| wet_soak(k as f32 / 20.0, 0.6)).collect::<Vec<_>>().windows(2).all(|p| p[1] >= p[0]));
    }

    #[test]
    fn a_map_is_wet_from_its_first_damp_texel() {
        let dry = [0.0, 0.0, 0.0, 1.0];
        assert!(!map_is_wet(&[]));
        assert!(!map_is_wet(&[dry; 16]));
        assert!(!map_is_wet(&[[0.0, 0.0, 0.8, 1.0]; 16]), "snow alone is not a film");
        assert!(!map_is_wet(&[[0.01, 0.0003, 0.0, 1.0]; 16]), "under what the law shows");
        let mut one = [dry; 16];
        one[11] = [0.3, 0.0, 0.0, 1.0];
        assert!(map_is_wet(&one), "one damp texel");
        one[11] = [0.0, 0.002, 0.0, 1.0];
        assert!(map_is_wet(&one), "one texel of standing water");

        assert!(map_holds(&one).0 && map_is_wet(&one));
    }

    #[test]
    fn a_chunk_draws_the_lightest_twin_that_shows_what_it_holds() {
        assert_eq!(WeatherKinds::of(false, false), WeatherKinds::WetOnly);
        assert_eq!(WeatherKinds::of(true, false), WeatherKinds::Snowless);
        assert_eq!(WeatherKinds::of(false, true), WeatherKinds::Waterless);
        assert_eq!(WeatherKinds::of(true, true), WeatherKinds::Both);
        assert_eq!(WeatherKinds::ALL.map(|k| k.index()), [0, 1, 2, 3]);
        let kinds = |holds: &[(bool, bool)], mask: u32| {
            let (mut water, mut snow) = (false, false);
            for k in 0..CHUNK_POOLS as usize {
                if mask & (1 << k) != 0 {
                    let (w, s) = holds.get(k).copied().unwrap_or((true, true));
                    water |= w && mask & (1 << (k + CHUNK_POOLS as usize)) != 0;
                    snow |= s;
                }
            }
            WeatherKinds::of(water, snow)
        };

        let holds = [(false, false), (true, false)];
        assert_eq!(kinds(&holds, 0b10), WeatherKinds::WetOnly, "a slope of the rain area");
        assert_eq!(kinds(&holds, 0b10 | 0b10 << CHUNK_POOLS), WeatherKinds::Snowless, "its hollows");
        assert_eq!(kinds(&holds, 0b01), WeatherKinds::WetOnly, "the beach");

        let wet = with_weather_kinds(&terrain_shader_for(TerrainRole::Read), WeatherKinds::WetOnly).unwrap();
        assert!(wet.contains("const WX_WATER: bool = false;") && wet.contains("const WX_SNOW: bool = false;"));
        assert!(!wet.contains("mat.sea_line - in.tex_pos.y"), "the twin reads the sea's soaking from the map");
        validate("wet-only read", &wet);

        let size = |s: &str| crate::renderer::shader_inlining::inlined_sizes(s, "fs_main").iter().map(|f| f.own * f.copies).sum::<usize>();
        let read = terrain_shader_for(TerrainRole::Read);
        let steep = crate::renderer::ground_twins::steep_shader(&read).expect("the ground's steep twin");
        let baked = |s: String| crate::renderer::brush_pipeline::sun_reader_shader(s, crate::renderer::brush_pipeline::FaceSun::Baked);
        let spotless = crate::renderer::lights::without_spot_shadows;
        for (name, src) in [("mixed", read), ("steep", steep)] {
            for (twin, src) in [("", src.clone()), ("_spotless", spotless(src.clone())), ("_baked", baked(src.clone())), ("_baked_spotless", spotless(baked(src)))] {
                let wet = with_weather_kinds(&src, WeatherKinds::WetOnly).unwrap_or_else(|| panic!("no wet-only twin of the {name}{twin} reader"));
                validate(&format!("wet-only {name}{twin}"), &wet);
                let full = with_weather(&src).unwrap();
                // 250: the ripples' two layers now share one copy (`wx_ripples`), so
                // the code this twin drops is ~50 smaller than when this was 300.
                assert!(size(&wet) + 250 < size(&full), "wet-only {name}{twin}: {} against the full twin's {}", size(&wet), size(&full));
            }
        }
    }

    #[test]
    #[ignore]
    fn wetness_reader_sizes() {
        use crate::renderer::ground_twins::{gentle_shader, steep_shader};
        use crate::renderer::terrain_pipeline::terrain_shader_built;
        let total = |s: &str| crate::renderer::shader_inlining::inlined_sizes(s, "fs_main").iter().map(|f| f.own * f.copies).sum::<usize>();
        let lit = |src: String| {
            crate::renderer::lights::without_spot_shadows(crate::renderer::brush_pipeline::sun_reader_shader(src, crate::renderer::brush_pipeline::FaceSun::Baked))
        };
        let read = terrain_shader_built(TerrainRole::Read, true);
        let gentle = lit(gentle_shader(&read).unwrap());
        let steep = lit(steep_shader(&read).unwrap());
        let mixed = lit(read);
        eprintln!("dry gentle {}  dry steep {}  dry mixed {}", total(&gentle), total(&steep), total(&mixed));
        for kinds in WeatherKinds::ALL {
            eprintln!(
                "weather{:<10} gentle {}  mixed {}  steep {}",
                kinds.label(),
                total(&with_weather_kinds(&gentle, kinds).unwrap()),
                total(&with_weather_kinds(&mixed, kinds).unwrap()),
                total(&with_weather_kinds(&steep, kinds).unwrap()),
            );
        }
        for (label, src) in [
            ("pass nowhere", crate::renderer::terrain_pipeline::nowhere_pass_shader().unwrap()),
            ("pass outdoors", crate::renderer::terrain_pipeline::outdoor_pass_shader().unwrap()),
            ("pass full", terrain_shader_for(TerrainRole::ProbePass)),
        ] {
            let kinds: Vec<String> = WeatherKinds::ALL.iter().map(|k| format!("{:?} {}", k, total(&with_weather_kinds(&src, *k).unwrap()))).collect();
            eprintln!("{label}: dry {}  {}", total(&src), kinds.join("  "));
        }
    }

    #[test]
    fn the_torch_puddle_reader_walks_its_sky() {
        use crate::renderer::ground_twins::gentle_shader;
        use crate::renderer::terrain_pipeline::terrain_shader_built;
        let total = |s: &str| crate::renderer::shader_inlining::inlined_sizes(s, "fs_main").iter().map(|f| f.own * f.copies).sum::<usize>();
        let baked = |src: String| crate::renderer::brush_pipeline::sun_reader_shader(src, crate::renderer::brush_pipeline::FaceSun::Baked);
        let gentle = baked(gentle_shader(&terrain_shader_built(TerrainRole::Read, true)).unwrap());
        for kinds in WeatherKinds::ALL {
            let twin = with_weather_kinds(&gentle, kinds).unwrap();
            let walked = crate::renderer::ground_twins::torch_weather_twin(twin.clone());
            assert_ne!(walked, twin, "{kinds:?}: the sky was not walked");
            validate(&format!("torch walked {kinds:?}"), &walked);
            assert!(total(&walked) + 80 < total(&twin), "{kinds:?}: {} walked against {}", total(&walked), total(&twin));
        }
        let puddles = crate::renderer::ground_twins::torch_weather_twin(with_weather_kinds(&gentle, WeatherKinds::ALL[1]).unwrap());
        assert!(total(&puddles) <= 2600, "the torch's puddle reader is {} by the proxy", total(&puddles));
    }

    #[test]
    #[ignore]
    fn torch_wetness_reader_sizes() {
        use crate::renderer::ground_twins::gentle_shader;
        use crate::renderer::terrain_pipeline::terrain_shader_built;
        let total = |s: &str| crate::renderer::shader_inlining::inlined_sizes(s, "fs_main").iter().map(|f| f.own * f.copies).sum::<usize>();
        let baked = |src: String| crate::renderer::brush_pipeline::sun_reader_shader(src, crate::renderer::brush_pipeline::FaceSun::Baked);
        let read = terrain_shader_built(TerrainRole::Read, true);
        let gentle = baked(gentle_shader(&read).unwrap());
        let spotless = crate::renderer::lights::without_spot_shadows(gentle.clone());
        eprintln!("dry gentle: spotless {}  torch {}", total(&spotless), total(&gentle));
        for kinds in WeatherKinds::ALL {
            eprintln!(
                "weather{:<10} gentle: spotless {}  torch {}",
                kinds.label(),
                total(&with_weather_kinds(&spotless, kinds).unwrap()),
                total(&with_weather_kinds(&gentle, kinds).unwrap()),
            );
        }
        for kinds in WeatherKinds::ALL {
            let torch = with_weather_kinds(&gentle, kinds).unwrap();
            eprintln!(
                "weather{:<10} gentle: torch, sky walked {}",
                kinds.label(),
                total(&crate::renderer::ground_twins::torch_weather_twin(torch)),
            );
        }
        if std::env::var("REPORT").is_ok() {
            let k = WeatherKinds::ALL[1];
            eprintln!("{}", crate::renderer::shader_inlining::report("snowless torch", &with_weather_kinds(&gentle, k).unwrap(), "fs_main", 25));
            eprintln!("{}", crate::renderer::shader_inlining::report("snowless spotless", &with_weather_kinds(&spotless, k).unwrap(), "fs_main", 25));
        }
    }

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

    #[test]
    fn a_drop_moves_back_by_its_fall_not_across_its_box() {
        let wrap = |c: f32, lo: f32, w: f32| c - w * (((c - lo) / w).floor());
        let (r2, fall, head_y) = (0.999f32, 8.0f32, 1.6f32);
        let (t, back) = (10.0f32, 1.0 / 36.0);
        let y = |t: f32| wrap(r2 * 9.0 - fall * t, head_y - 3.5, 9.0);

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

    #[test]
    fn a_far_column_is_hidden_by_anything_drawn_in_front_of_it() {
        let Some((device, queue)) = crate::renderer::terrain_pipeline::tests::headless_gpu() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        const SIZE: u32 = 16;
        let format = TextureFormat::Rgba8Unorm;
        let lights = crate::renderer::lights::LightsUniform::new(&device);
        let (_shadows, uniforms) = crate::renderer::uniforms::test_support::scene_uniforms(&device, &lights);
        let eye = glam::Vec3::new(0.0, 2.0, 0.0);
        let view_proj = glam::Mat4::perspective_rh(1.2, 1.0, 0.1, 200.0)
            * glam::Mat4::look_at_rh(eye, glam::Vec3::new(0.0, 2.0, -40.0), glam::Vec3::Y);
        uniforms.upload(&queue, view_proj, eye, &crate::renderer::uniforms::ShadowUpload::disabled());
        let layout = bind_group_layout(&device);
        let ground_layout = crate::renderer::terrain_pipeline::material_bind_group_layout(&device);
        let material = crate::renderer::terrain_pipeline::TerrainMaterial::fallback(&device, &queue, &ground_layout);
        let probe_layout = crate::renderer::brush_pipeline::probe_pass::bind_group_layout(&device);
        let probe = crate::renderer::brush_pipeline::probe_pass::Target::new(&device, &probe_layout, SIZE, SIZE, 1);
        let pipeline = WeatherPipeline::new(&device, format, &uniforms.layout, &ground_layout, &layout, 1, crate::renderer::multiview::ViewMode::Mono);
        let mut maps = WeatherMaps::new(&device, &layout, &[(8, 8)]);
        maps.upload_map(&queue, 0, 8, 8, &[[0.0; 4]; 64]);
        maps.set_area(0, &AreaParams { min: [-30.0, -60.0], extent: [60.0, 30.0], snow: false, falling: 1.0, wind: [0.0, 0.0], edge: 3.0, shore: false }, (8, 8));
        let view = ParticleView { head_world: eye.to_array(), sky: [1.0; 3], exposure: 1.0, pixel: 0.01, ..Default::default() };
        maps.set_view(&view, (0, 0, 0), [0.0, 0.0]);
        maps.write(&queue);

        let centre_after = |frame_depth: f32| -> [u8; 4] {
            let make = |label: &str, format: TextureFormat| {
                device.create_texture(&TextureDescriptor {
                    label: Some(label),
                    size: Extent3d { width: SIZE, height: SIZE, depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: TextureDimension::D2,
                    format,
                    usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
                    view_formats: &[],
                })
            };
            let target = make("veil_test_target", format);
            let depth = make("veil_test_depth", TextureFormat::Depth32Float);
            let (target_view, depth_view) = (target.create_view(&Default::default()), depth.create_view(&Default::default()));
            let readback = device.create_buffer(&BufferDescriptor {
                label: Some("veil_test_readback"),
                size: (256 * SIZE) as u64,
                usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut encoder = device.create_command_encoder(&Default::default());

            encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("veil_test_probe_depth"),
                color_attachments: &[],
                depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                    view: &probe.depth_view,
                    depth_ops: Some(Operations { load: LoadOp::Clear(1.0), store: StoreOp::Store }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            {

                let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                    label: Some("veil_test_pass"),
                    color_attachments: &[Some(RenderPassColorAttachment {
                        view: &target_view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: Operations { load: LoadOp::Clear(Color::BLACK), store: StoreOp::Store },
                    })],
                    depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                        view: &depth_view,
                        depth_ops: Some(Operations { load: LoadOp::Clear(frame_depth), store: StoreOp::Store }),
                        stencil_ops: None,
                    }),
                    ..Default::default()
                });
                pipeline.draw_veils(&mut pass, &uniforms.bind_group, &material.bind_group, &maps.bind_group, &probe.bind_group);
            }
            encoder.copy_texture_to_buffer(
                TexelCopyTextureInfo { texture: &target, mip_level: 0, origin: Origin3d::ZERO, aspect: TextureAspect::All },
                TexelCopyBufferInfo {
                    buffer: &readback,
                    layout: TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(256), rows_per_image: Some(SIZE) },
                },
                Extent3d { width: SIZE, height: SIZE, depth_or_array_layers: 1 },
            );
            queue.submit(Some(encoder.finish()));
            let slice = readback.slice(..);
            slice.map_async(MapMode::Read, |_| {});
            device.poll(PollType::Wait { submission_index: None, timeout: None }).ok();
            let data = slice.get_mapped_range().unwrap();
            let at = (SIZE / 2) as usize * 256 + (SIZE / 2) as usize * 4;
            [data[at], data[at + 1], data[at + 2], data[at + 3]]
        };
        let open = centre_after(1.0);
        assert!(open[3] > 0 && open[0] > 0, "the column should show with nothing in front of it: {open:?}");

        let hidden = centre_after(0.2);
        assert_eq!(hidden, [0, 0, 0, 255], "the column drew over something in front of it (the clear is opaque black)");
    }

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
        crate::renderer::ground_twins::wet_readers(&device, TextureFormat::Rgba8UnormSrgb, &uniforms.layout, 4, &probe_layout, &layout)
            .expect("the ground's wet-only mixed and steep twins");
        let ground = crate::renderer::terrain_pipeline::material_bind_group_layout(&device);
        let _p = WeatherPipeline::new(&device, TextureFormat::Rgba8UnormSrgb, &uniforms.layout, &ground, &layout, 4, crate::renderer::multiview::ViewMode::Mono);
        let mut maps = WeatherMaps::new(&device, &layout, &[(64, 48), (32, 80)]);
        maps.upload_map(&queue, 1, 32, 80, &vec![[1.0, 0.01, 0.1, 0.0]; 32 * 80]);
        maps.set_area(1, &AreaParams { min: [0.0, 0.0], extent: [8.0, 20.0], snow: true, falling: 1.0, wind: [1.0, 0.0], edge: 2.0, shore: false }, (32, 80));
        maps.write(&queue);
        let err = pollster::block_on(scope.pop());
        assert!(err.is_none(), "the weather pipelines failed to build: {err:?}");
        assert_eq!(maps.size, (64, 80));
        assert_eq!(maps.params.params[0], 2.0);
        assert!((maps.params.areas[4][1] - 1.0).abs() < 1e-6 && (maps.params.areas[4][0] - 0.5).abs() < 1e-6);
    }
}
