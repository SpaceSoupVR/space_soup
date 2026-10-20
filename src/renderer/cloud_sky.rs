//! THE VOLUMETRIC CLOUDS ON THE GPU (2026-10-20): the layers of
//! `space_soup_sky::cloud_layers` marched through, a slice of a sky map a
//! frame, and their shadow on the ground.
//!
//! The sky is at infinity: a cloud 1.5 km up shifts 0.002 deg between the
//! eyes. So its light is worked out once per direction for both eyes, into
//! a `MAP_WIDTH` x `MAP_HEIGHT` map over the upper hemisphere (azimuth by
//! elevation, rows packed toward the horizon), `MAP_SLICES` frames to a whole
//! map, and the sky shader reads it with two fetches: the last whole map
//! blending into the newest as the next is built, so nothing ever steps.
//! Its cost is fixed by the map, not by the eyes or the screen
//! (`docs/engine/sky-realism-research.md` 4.5).
//!
//! The ground's shadow map (`UniformBuffer::cloud_shadow`, read per pixel by
//! `cloud_sun_shadow`) is drawn from the same density every frame, so the
//! shadows sweeping over the ground are the clouds overhead.

pub use space_soup_sky::{cloud_layers, clouds};
use space_soup_sky::cloud_layers::{self as cl, CloudLayer, CloudNoise, LAYERS, NOISE_SIZE};
use space_soup_sky::clouds::{self as c, CloudField, CLOUD_TILE};
use wgpu::*;

/// The sky map: 0.35 deg a texel round the horizon.
pub const MAP_WIDTH: u32 = 1024;
pub const MAP_HEIGHT: u32 = 384;
/// Frames to a whole map: 16 rows, 16 k rays a frame; a map every 0.33 s
/// at 72 Hz, while a 12 m/s wind moves a cloud 1.5 km up 0.15 deg.
pub const MAP_SLICES: u32 = 24;
/// Rows are packed toward the horizon: elevation = 90 deg * v^MAP_PACK.
pub const MAP_PACK: f32 = 1.5;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CloudSkyUniform {
    /// World x, altitude, world z of the eye the map is seen from; w the
    /// scale the map's light is stored at (`CloudSky::encode` sets it).
    pub eye: [f32; 4],
    /// Toward the light (the sun, or the moon after it), unit.
    pub light_dir: [f32; 4],
    /// The light's irradiance; w the map's rows.
    pub light: [f32; 4],
    /// The sky's radiance on a cloud from above; w the map's width.
    pub amb_sky: [f32; 4],
    /// The ground's from below; w the shadow map's mean.
    pub amb_ground: [f32; 4],
    /// The shadow map's centre x, z, extent, and the ground's altitude.
    pub shadow: [f32; 4],
    pub layers: [[[f32; 4]; 8]; LAYERS],
}

impl CloudSkyUniform {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        layers: &[CloudLayer; LAYERS],
        eye: [f32; 3],
        light_dir: [f32; 3],
        light: [f32; 3],
        amb_sky: [f32; 3],
        amb_ground: [f32; 3],
        shadow_centre: [f32; 2],
        shadow_extent: f32,
        shadow_mean: f32,
    ) -> Self {
        let len = (light_dir[0] * light_dir[0] + light_dir[1] * light_dir[1] + light_dir[2] * light_dir[2]).sqrt().max(1e-6);
        Self {
            eye: [eye[0], eye[1], eye[2], 0.0],
            light_dir: [light_dir[0] / len, light_dir[1] / len, light_dir[2] / len, 0.0],
            light: [light[0], light[1], light[2], MAP_HEIGHT as f32],
            amb_sky: [amb_sky[0], amb_sky[1], amb_sky[2], MAP_WIDTH as f32],
            amb_ground: [amb_ground[0], amb_ground[1], amb_ground[2], shadow_mean],
            shadow: [shadow_centre[0], shadow_centre[1], shadow_extent, eye[1]],
            layers: std::array::from_fn(|k| layers[k].gpu()),
        }
    }
}

/// The direction a map coordinate looks along (`cloud_map_dir`'s twin).
pub fn map_dir(uv: [f32; 2]) -> [f32; 3] {
    let el = std::f32::consts::FRAC_PI_2 * uv[1].clamp(0.0, 1.0).powf(MAP_PACK);
    let az = (uv[0] - 0.5) * std::f32::consts::TAU;
    [az.sin() * el.cos(), el.sin(), az.cos() * el.cos()]
}

/// The map coordinate a direction is held at (the sky shader's lookup).
pub fn map_uv(d: [f32; 3]) -> [f32; 2] {
    let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt().max(1e-6);
    let el = (d[1] / len).clamp(0.0, 1.0).asin();
    [d[0].atan2(d[2]) / std::f32::consts::TAU + 0.5, (el / std::f32::consts::FRAC_PI_2).powf(1.0 / MAP_PACK)]
}

pub struct CloudSky {
    noise: Texture,
    weather: Texture,
    maps: Texture,
    map_layers: [TextureView; 3],
    array_view: TextureView,
    sampler: Sampler,
    uniform: Buffer,
    layout: BindGroupLayout,
    bind_group: BindGroup,
    map_pipeline: RenderPipeline,
    shadow_pipeline: RenderPipeline,
    /// The map shown last, the one shown now, and the one being built.
    prev: u32,
    cur: u32,
    build: u32,
    next_row: u32,
    /// No map drawn yet: the first is drawn whole.
    fresh: bool,
    /// Each map's light is stored over the light it was drawn under: a
    /// night's radiances (1e-8) fell among half floats' subnormals and the
    /// clouds came out banded and fringed red.
    scales: [f32; 3],
}

fn weather_texture(device: &Device, queue: &Queue, field: &CloudField) -> Texture {
    let t = device.create_texture(&TextureDescriptor {
        label: Some("cloud_weather"),
        size: Extent3d { width: CLOUD_TILE, height: CLOUD_TILE, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        TexelCopyTextureInfo { texture: &t, mip_level: 0, origin: Origin3d::ZERO, aspect: TextureAspect::All },
        &field.rgba_bytes(),
        TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(CLOUD_TILE * 4), rows_per_image: Some(CLOUD_TILE) },
        Extent3d { width: CLOUD_TILE, height: CLOUD_TILE, depth_or_array_layers: 1 },
    );
    t
}

impl CloudSky {
    pub fn new(device: &Device, queue: &Queue, field: &CloudField, noise: &CloudNoise) -> Self {
        let n = NOISE_SIZE;
        let noise_tex = device.create_texture(&TextureDescriptor {
            label: Some("cloud_noise"),
            size: Extent3d { width: n, height: n, depth_or_array_layers: n },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D3,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            TexelCopyTextureInfo { texture: &noise_tex, mip_level: 0, origin: Origin3d::ZERO, aspect: TextureAspect::All },
            &noise.rgba_bytes(),
            TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(n * 4), rows_per_image: Some(n) },
            Extent3d { width: n, height: n, depth_or_array_layers: n },
        );
        let weather = weather_texture(device, queue, field);
        let maps = device.create_texture(&TextureDescriptor {
            label: Some("cloud_sky_maps"),
            size: Extent3d { width: MAP_WIDTH, height: MAP_HEIGHT, depth_or_array_layers: 3 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let map_layers = std::array::from_fn(|i| {
            maps.create_view(&TextureViewDescriptor {
                label: Some("cloud_sky_map"),
                dimension: Some(TextureViewDimension::D2),
                base_array_layer: i as u32,
                array_layer_count: Some(1),
                ..Default::default()
            })
        });
        let array_view = maps.create_view(&TextureViewDescriptor {
            label: Some("cloud_sky_maps"),
            dimension: Some(TextureViewDimension::D2Array),
            ..Default::default()
        });
        let sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("cloud_noise_sampler"),
            address_mode_u: AddressMode::Repeat,
            address_mode_v: AddressMode::Repeat,
            address_mode_w: AddressMode::Repeat,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            ..Default::default()
        });
        let uniform = device.create_buffer(&BufferDescriptor {
            label: Some("cloud_sky_uniform"),
            size: std::mem::size_of::<CloudSkyUniform>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let tex = |binding, dim| BindGroupLayoutEntry {
            binding,
            visibility: ShaderStages::FRAGMENT,
            ty: BindingType::Texture { sample_type: TextureSampleType::Float { filterable: true }, view_dimension: dim, multisampled: false },
            count: None,
        };
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("cloud_sky_bgl"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Buffer { ty: BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                tex(1, TextureViewDimension::D3),
                tex(2, TextureViewDimension::D2),
                BindGroupLayoutEntry { binding: 3, visibility: ShaderStages::FRAGMENT, ty: BindingType::Sampler(SamplerBindingType::Filtering), count: None },
            ],
        });
        // Clear sky until a map is drawn: a map never drawn reads black.
        let mut enc = device.create_command_encoder(&CommandEncoderDescriptor { label: Some("cloud_sky_clear") });
        for view in &map_layers {
            enc.begin_render_pass(&RenderPassDescriptor {
                label: Some("cloud_sky_clear"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: Operations { load: LoadOp::Clear(Color { r: 0.0, g: 0.0, b: 0.0, a: 1.0 }), store: StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                multiview_mask: None,
                occlusion_query_set: None,
            });
        }
        queue.submit(Some(enc.finish()));
        let bind_group = Self::bind(device, &layout, &uniform, &noise_tex, &weather, &sampler);
        let shader = device.create_shader_module(ShaderModuleDescriptor { label: Some("cloud_sky"), source: ShaderSource::Wgsl(cloud_sky_shader().into()) });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("cloud_sky_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = |label: &str, entry: &str, format: TextureFormat| {
            device.create_render_pipeline(&RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                vertex: VertexState { module: &shader, entry_point: Some("vs_full"), compilation_options: Default::default(), buffers: &[] },
                fragment: Some(FragmentState {
                    module: &shader,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    targets: &[Some(ColorTargetState { format, blend: None, write_mask: ColorWrites::ALL })],
                }),
                primitive: PrimitiveState::default(),
                depth_stencil: None,
                multisample: MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        let map_pipeline = pipeline("cloud_sky_map", "fs_map", TextureFormat::Rgba16Float);
        let shadow_pipeline = pipeline("cloud_sky_shadow", "fs_shadow", TextureFormat::R8Unorm);
        Self {
            noise: noise_tex,
            weather,
            maps,
            map_layers,
            array_view,
            sampler,
            uniform,
            layout,
            bind_group,
            map_pipeline,
            shadow_pipeline,
            prev: 0,
            cur: 0,
            build: 1,
            next_row: 0,
            fresh: true,
            scales: [1.0; 3],
        }
    }

    fn bind(device: &Device, layout: &BindGroupLayout, uniform: &Buffer, noise: &Texture, weather: &Texture, sampler: &Sampler) -> BindGroup {
        let nv = noise.create_view(&TextureViewDescriptor { dimension: Some(TextureViewDimension::D3), ..Default::default() });
        let wv = weather.create_view(&Default::default());
        device.create_bind_group(&BindGroupDescriptor {
            label: Some("cloud_sky_bg"),
            layout,
            entries: &[
                BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: BindingResource::TextureView(&nv) },
                BindGroupEntry { binding: 2, resource: BindingResource::TextureView(&wv) },
                BindGroupEntry { binding: 3, resource: BindingResource::Sampler(sampler) },
            ],
        })
    }

    /// A new weather field (a new sky's seed).
    pub fn set_weather(&mut self, device: &Device, queue: &Queue, field: &CloudField) {
        self.weather = weather_texture(device, queue, field);
        self.bind_group = Self::bind(device, &self.layout, &self.uniform, &self.noise, &self.weather, &self.sampler);
        self.fresh = true;
    }

    /// The maps the sky shader reads (`texture_2d_array`).
    pub fn maps_view(&self) -> &TextureView {
        &self.array_view
    }

    /// The layers the sky shader blends between and how far: the last whole
    /// map, the newest, and the share of the next one built.
    pub fn display(&self) -> [f32; 3] {
        [self.prev as f32, self.cur as f32, self.next_row as f32 / MAP_HEIGHT as f32]
    }

    /// The radiance one stored unit of the shown maps stands for: the last
    /// whole map's, the newest's.
    pub fn display_scales(&self) -> [f32; 2] {
        [1.0 / self.scales[self.prev as usize], 1.0 / self.scales[self.cur as usize]]
    }

    /// ONE FRAME'S CLOUDS: the next slice of the sky map (the whole map when
    /// `whole`, or when none is drawn yet), and the ground's shadow into
    /// `shadow` when given.
    pub fn encode(&mut self, encoder: &mut CommandEncoder, queue: &Queue, u: &CloudSkyUniform, shadow: Option<&TextureView>, whole: bool) {
        let rows = MAP_HEIGHT / MAP_SLICES;
        let (target, from, to) = if self.fresh || whole {
            (self.cur, 0, MAP_HEIGHT)
        } else {
            (self.build, self.next_row, (self.next_row + rows).min(MAP_HEIGHT))
        };
        // A map keeps the scale its first slice was drawn at.
        if from == 0 {
            let peak = (0..3).map(|c| u.light[c] + u.amb_sky[c]).fold(0.0f32, f32::max);
            self.scales[target as usize] = 1.0 / peak.max(1e-20);
        }
        let mut u = *u;
        u.eye[3] = self.scales[target as usize];
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&u));
        {
            let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("cloud_sky_map"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &self.map_layers[target as usize],
                    depth_slice: None,
                    resolve_target: None,
                    ops: Operations { load: LoadOp::Load, store: StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                multiview_mask: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.map_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.set_scissor_rect(0, from, MAP_WIDTH, to - from);
            pass.draw(0..3, 0..1);
        }
        if self.fresh || whole {
            self.fresh = false;
            self.prev = self.cur;
            self.next_row = 0;
        } else {
            self.next_row = to;
            if to >= MAP_HEIGHT {
                let (p, c) = (self.cur, self.build);
                self.prev = p;
                self.cur = c;
                self.build = 3 - p - c;
                self.next_row = 0;
            }
        }
        if let Some(view) = shadow {
            let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("cloud_sky_shadow"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: Operations { load: LoadOp::Clear(Color::WHITE), store: StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                multiview_mask: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.shadow_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
    }
}

/// The marcher: `cloud_layers`' density, height profiles, light and cirrus
/// twinned line for line, then the map and shadow entry points.
pub fn cloud_sky_shader() -> String {
    let f = |v: f32| format!("{v:?}");
    let steps = |s: [u32; 3]| format!("array<u32, 3>({}u, {}u, {}u)", s[0], s[1], s[2]);
    CLOUD_SKY_WGSL
        .replace("{EARTH}", &format!("{:.1}", c::EARTH_RADIUS_M))
        .replace("{DETAIL_RATIO}", &f(cl::DETAIL_RATIO))
        .replace("{DETAIL_EROSION}", &f(cl::DETAIL_EROSION))
        .replace("{EDGE_GAIN}", &f(cl::EDGE_GAIN))
        .replace("{MAX_MARCH}", &f(cl::MAX_MARCH_M))
        .replace("{CIRRUS_TAU}", &f(cl::CIRRUS_TAU))
        .replace("{SUN_RADIUS}", &f(cl::SHADOW_DISC_RAD))
        .replace("{CIRRUS_STRETCH}", &f(cl::CIRRUS_STRETCH))
        .replace("{EXTINCTION}", &f(c::EXTINCTION_PER_M))
        .replace("{VIEW_STEPS}", &steps(cl::VIEW_STEPS))
        .replace("{SHADOW_STEPS}", &steps(cl::SHADOW_STEPS))
        .replace("{OCT_A}", &f(c::OCTAVE_A))
        .replace("{OCT_B}", &f(c::OCTAVE_B))
        .replace("{OCT_C}", &f(c::OCTAVE_C))
        .replace("{FORWARD_G}", &f(c::PHASE_FORWARD_G))
        .replace("{BACK_G}", &f(c::PHASE_BACK_G))
        .replace("{BACK_SHARE}", &f(c::PHASE_BACK_SHARE))
        .replace("{HAZE}", &format!("{:e}", c::HAZE_PER_M))
        .replace("{POWDER_K}", &f(c::POWDER_K))
        .replace("{PACK}", &f(MAP_PACK))
        .replace("{KIND_CUMULUS}", &f(cl::KIND_CUMULUS))
        .replace("{KIND_STRATO}", &f(cl::KIND_STRATOCUMULUS))
        .replace("{KIND_CIRRUS}", &f(cl::KIND_CIRRUS))
        .replace("{KIND_ALTO}", &f(cl::KIND_ALTOCUMULUS))
        .replace("{HIGH_SHADE_FILL}", &f(cl::HIGH_SHADE_FILL))
        .replace("{SHADOW_SIZE}", &f(crate::renderer::uniforms::CLOUD_SHADOW_SIZE as f32))
}

const CLOUD_SKY_WGSL: &str = r#"
struct Layer {
    v: array<vec4<f32>, 8>,
}
struct CloudSkyUniform {
    eye: vec4<f32>,
    light_dir: vec4<f32>,
    light: vec4<f32>,
    amb_sky: vec4<f32>,
    amb_ground: vec4<f32>,
    shadow: vec4<f32>,
    layers: array<Layer, 4>,
}
@group(0) @binding(0) var<uniform> u: CloudSkyUniform;
@group(0) @binding(1) var noise_tex: texture_3d<f32>;
@group(0) @binding(2) var weather_tex: texture_2d<f32>;
@group(0) @binding(3) var samp: sampler;

const PI: f32 = 3.14159265;
const EARTH: f32 = {EARTH};
const DETAIL_RATIO: f32 = {DETAIL_RATIO};
const DETAIL_EROSION: f32 = {DETAIL_EROSION};
const EDGE_GAIN: f32 = {EDGE_GAIN};
const MAX_MARCH: f32 = {MAX_MARCH};
const CIRRUS_TAU: f32 = {CIRRUS_TAU};
const SUN_RADIUS: f32 = {SUN_RADIUS};
const CIRRUS_STRETCH: f32 = {CIRRUS_STRETCH};
const EXTINCTION: f32 = {EXTINCTION};
const OCT_A: f32 = {OCT_A};
const OCT_B: f32 = {OCT_B};
const OCT_C: f32 = {OCT_C};
const FORWARD_G: f32 = {FORWARD_G};
const BACK_G: f32 = {BACK_G};
const BACK_SHARE: f32 = {BACK_SHARE};
const HAZE: f32 = {HAZE};
const POWDER_K: f32 = {POWDER_K};
const PACK: f32 = {PACK};
const KIND_CUMULUS: f32 = {KIND_CUMULUS};
const KIND_STRATO: f32 = {KIND_STRATO};
const KIND_CIRRUS: f32 = {KIND_CIRRUS};
const KIND_ALTO: f32 = {KIND_ALTO};
const HIGH_SHADE_FILL: f32 = {HIGH_SHADE_FILL};

@vertex fn vs_full(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
}

fn lv(k: u32, i: u32) -> vec4<f32> {
    return u.layers[k].v[i];
}

fn threshold(k: u32, c: f32) -> f32 {
    let x = clamp(c, 0.0, 1.0) * 15.0;
    let i = min(u32(floor(x)), 14u);
    let f = x - f32(i);
    let a = lv(k, 4u + i / 4u)[i % 4u];
    let j = i + 1u;
    let b = lv(k, 4u + j / 4u)[j % 4u];
    return a + (b - a) * f;
}

fn local_coverage(k: u32, xz: vec2<f32>) -> f32 {
    let w = textureSampleLevel(weather_tex, samp, xz * lv(k, 1u).z + lv(k, 3u).xy, 0.0).r;
    return clamp(lv(k, 0u).z + (w - 0.5) * lv(k, 1u).w, 0.0, 1.0);
}

fn height_profile(kind: f32, h: f32, cover: f32) -> f32 {
    if (kind == KIND_CUMULUS) {
        let top = 0.3 + 0.7 * cover;
        return smoothstep(0.0, 0.06, h) * (1.0 - smoothstep(0.5 * top, top, h));
    } else if (kind == KIND_STRATO) {
        return smoothstep(0.0, 0.15, h) * (1.0 - smoothstep(0.5, 0.95, h));
    }
    return smoothstep(0.0, 0.2, h) * (1.0 - smoothstep(0.45, 0.9, h));
}

fn density(k: u32, xz: vec2<f32>, alt: f32, detail: bool) -> f32 {
    let a = lv(k, 0u);
    let h = (alt - a.x) / max(a.y - a.x, 1.0);
    if (h < 0.0 || h > 1.0) {
        return 0.0;
    }
    let cover = local_coverage(k, xz);
    if (cover <= 0.0) {
        return 0.0;
    }
    let si = lv(k, 1u).x;
    let s = vec3<f32>(xz.x * si, alt * si, xz.y * si) + lv(k, 2u).xyz;
    let n = textureSampleLevel(noise_tex, samp, s, 0.0);
    let low = 0.625 * n.g + 0.25 * n.b + 0.125 * n.a;
    let base = clamp((n.r - (low - 1.0)) / (2.0 - low), 0.0, 1.0);
    let shape = base * height_profile(lv(k, 1u).y, h, cover);
    let th = threshold(k, cover);
    var d = clamp((shape - th) / max(1.0 - th, 1e-3), 0.0, 1.0);
    if (d <= 0.0) {
        return 0.0;
    }
    if (detail) {
        let q = textureSampleLevel(noise_tex, samp, s * DETAIL_RATIO, 0.0);
        let fine = 0.625 * q.g + 0.25 * q.b + 0.125 * q.a;
        let e = fine + (1.0 - 2.0 * fine) * smoothstep(0.1, 0.6, h);
        let cut = DETAIL_EROSION * e;
        d = clamp((d - cut) / (1.0 - cut), 0.0, 1.0);
    }
    return min(d * EDGE_GAIN, 1.0) * a.w;
}

fn cirrus_tau(k: u32, xz: vec2<f32>) -> f32 {
    let wd = lv(k, 3u).zw;
    let along = xz.x * wd.x + xz.y * wd.y;
    let across = -xz.x * wd.y + xz.y * wd.x;
    let s = lv(k, 1u).z * 10.0;
    let uv = vec2<f32>(along * s / CIRRUS_STRETCH, across * s) + lv(k, 3u).xy;
    let f = textureSampleLevel(weather_tex, samp, uv, 0.0);
    let g = textureSampleLevel(weather_tex, samp, uv * 3.0 + vec2<f32>(0.37, 0.71), 0.0);
    let n = 0.45 * f.g + 0.3 * f.b + 0.25 * g.b;
    let tuft = textureSampleLevel(weather_tex, samp, xz * lv(k, 1u).z * 2.0 + lv(k, 3u).yx, 0.0).r;
    let c = lv(k, 0u).z;
    let d = clamp((n - (1.0 - c)) / max(c, 1e-3), 0.0, 1.0) * smoothstep(0.35, 0.7, tuft);
    return d * d * CIRRUS_TAU * min(lv(k, 0u).w / EXTINCTION, 2.0);
}

// From altitude eye_alt along mu to the shell at alt (above it); < 0 when
// the ray never meets it.
fn shell_distance(eye_alt: f32, alt: f32, mu: f32) -> f32 {
    if (alt <= eye_alt) {
        return 0.0;
    }
    let r = EARTH + eye_alt;
    let c = (alt - eye_alt) * (2.0 * EARTH + alt + eye_alt);
    let rm = r * mu;
    let disc = rm * rm + c;
    if (disc < 0.0 || mu <= -0.2) {
        return -1.0;
    }
    return c / max(sqrt(disc) + rm, 1e-3);
}

fn altitude_along(eye_alt: f32, mu: f32, t: f32) -> f32 {
    let r = EARTH + eye_alt;
    return sqrt(r * r + t * t + 2.0 * r * t * mu) - EARTH;
}

fn layer_on(k: u32) -> bool {
    let a = lv(k, 0u);
    return a.z > 0.0 && a.w > 0.0;
}

// A high thin layer's depth toward the light as the ground feels it: its
// coverage, not its cells (`cloud_layers::high_layer_tau`).
fn high_layer_tau(k: u32, xz: vec2<f32>, slant: f32) -> f32 {
    let a = lv(k, 0u);
    if (lv(k, 1u).y == KIND_CIRRUS) {
        return a.z * HIGH_SHADE_FILL * CIRRUS_TAU * min(a.w / EXTINCTION, 2.0) * slant;
    }
    return local_coverage(k, xz) * HIGH_SHADE_FILL * a.w * (a.y - a.x) * slant;
}

// The column's direct transmittance toward dir from xz at alt, every layer
// (`cloud_layers::transmittance_toward`).
fn transmittance_toward(xz: vec2<f32>, alt: f32, dir: vec3<f32>) -> f32 {
    let mu = dir.y;
    if (mu <= 0.02) {
        return 1.0;
    }
    var steps = {SHADOW_STEPS};
    var tau = 0.0;
    for (var k = 0u; k < 4u; k = k + 1u) {
        if (!layer_on(k)) {
            continue;
        }
        let a = lv(k, 0u);
        let kind = lv(k, 1u).y;
        if (kind == KIND_CIRRUS || kind == KIND_ALTO) {
            let t = shell_distance(alt, a.x, mu);
            if (t < 0.0) {
                continue;
            }
            tau = tau + high_layer_tau(k, xz + dir.xz * t, min(1.0 / mu, 8.0));
            continue;
        }
        let t0 = shell_distance(alt, a.x, mu);
        let t1 = shell_distance(alt, a.y, mu);
        if (t0 < 0.0 || t1 < 0.0) {
            continue;
        }
        let n = steps[min(k, 2u)];
        let dt = min(t1 - t0, MAX_MARCH) / f32(n);
        for (var i = 0u; i < n; i = i + 1u) {
            let t = t0 + (f32(i) + 0.5) * dt;
            tau = tau + density(k, xz + dir.xz * t, altitude_along(alt, mu, t), false) * dt;
        }
    }
    return exp(-tau);
}

fn hg(g: f32, c: f32) -> f32 {
    let g2 = g * g;
    return (1.0 - g2) / (4.0 * PI * pow(max(1.0 + g2 - 2.0 * g * c, 1e-6), 1.5));
}

fn phase(s: f32, c: f32) -> f32 {
    return (1.0 - BACK_SHARE) * hg(FORWARD_G * s, c) + BACK_SHARE * hg(BACK_G * s, c);
}

// The light's optical depth from a point in layer k toward it: taps out
// through the layer, nearer ones closer together.
fn light_tau(k: u32, xz: vec2<f32>, alt: f32) -> f32 {
    let l = u.light_dir.xyz;
    let a = lv(k, 0u);
    let thick = max(a.y - a.x, 100.0);
    var tau = 0.0;
    var last = 0.0;
    var reach = array<f32, 4>(0.05, 0.15, 0.35, 0.8);
    let taps = select(2u, 4u, lv(k, 1u).y == KIND_CUMULUS);
    for (var i = 0u; i < taps; i = i + 1u) {
        let r = reach[i + 4u - taps] * thick;
        let p = vec3<f32>(xz.x, alt, xz.y) + l * r;
        tau = tau + density(k, p.xz, p.y, false) * (r - last);
        last = r;
    }
    return tau;
}

// The light a sample in layer k sends toward the eye per unit extinction.
fn in_scatter(k: u32, alt: f32, tau_l: f32, cv: f32, d: f32) -> vec3<f32> {
    let a = lv(k, 0u);
    let h = clamp((alt - a.x) / max(a.y - a.x, 1.0), 0.0, 1.0);
    var oct = phase(1.0, cv) * exp(-tau_l)
        + OCT_A * phase(OCT_C, cv) * exp(-OCT_B * tau_l)
        + OCT_A * OCT_A * phase(OCT_C * OCT_C, cv) * exp(-OCT_B * OCT_B * tau_l);
    // Beer-powder, mild and only away from the sun: thin outer parts have
    // built up less scattered light.
    let powder = 1.0 - exp(-POWDER_K * (2.0 * tau_l + 40.0 * d));
    oct = oct * mix(1.0, powder, 0.35 * (1.0 - cv));
    // The light scattered deep inside, which the octaves leave out.
    let deep = max(u.light_dir.y, 0.0) * exp(-0.25 * tau_l) * 0.06;
    let amb = u.amb_sky.rgb * (0.45 + 0.55 * h) + u.amb_ground.rgb * (0.35 * (1.0 - h));
    return u.light.rgb * (oct + deep) + amb;
}

// Where in its step each sample sits. Per-texel jitter (even interleaved
// gradient noise) left a dot pattern the display's magnification showed;
// the steps' middle, with enough steps, leaves no grain and no bands.
fn cloud_hash(p: vec2<f32>) -> f32 {
    return 0.5;
}

fn map_dir(uv: vec2<f32>) -> vec3<f32> {
    let el = 0.5 * PI * pow(clamp(uv.y, 0.0, 1.0), PACK);
    let az = (uv.x - 0.5) * 2.0 * PI;
    return vec3<f32>(sin(az) * cos(el), sin(el), cos(az) * cos(el));
}

// THE SKY MAP: the clouds' light toward the eye along this texel's
// direction (rgb) and how much of the sky beyond shows (a), haze included.
@fragment fn fs_map(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let size = vec2<f32>(u.amb_sky.w, u.light.w);
    let dir = map_dir(pos.xy / size);
    let mu = dir.y;
    let alt = u.eye.y;
    let eye = u.eye.xz;
    let cv = dot(dir, u.light_dir.xyz);
    let jitter = cloud_hash(pos.xy);
    var steps = {VIEW_STEPS};
    var trans = 1.0;
    var seen = 0.0;
    var rgb = vec3<f32>(0.0);
    for (var k = 0u; k < 3u; k = k + 1u) {
        if (!layer_on(k) || trans < 0.01) {
            continue;
        }
        let a = lv(k, 0u);
        let t0 = shell_distance(alt, a.x, mu);
        let t1 = shell_distance(alt, a.y, mu);
        if (t0 < 0.0 || t1 < 0.0) {
            continue;
        }
        let n = steps[k];
        let dt = min(t1 - t0, MAX_MARCH) / f32(n);
        for (var i = 0u; i < n; i = i + 1u) {
            let t = t0 + (f32(i) + jitter) * dt;
            let p_alt = altitude_along(alt, mu, t);
            let xz = eye + dir.xz * t;
            let d = density(k, xz, p_alt, true);
            if (d > 0.0) {
                let s = in_scatter(k, p_alt, light_tau(k, xz, p_alt), cv, d);
                let ts = exp(-d * dt);
                let w = trans * (1.0 - ts) * exp(-t * HAZE);
                rgb = rgb + w * s;
                seen = seen + w;
                trans = trans * ts;
                if (trans < 0.01) {
                    break;
                }
            }
        }
    }
    // Cirrus: ice, a thin sheet, its light thrown forward.
    if (layer_on(3u) && trans > 0.01) {
        let t = shell_distance(alt, lv(3u, 0u).x, mu);
        if (t >= 0.0) {
            let tau = cirrus_tau(3u, eye + dir.xz * t) * min(1.0 / max(mu, 1e-3), 8.0);
            let alpha = 1.0 - exp(-tau);
            let s = u.light.rgb * (0.7 * hg(0.85, cv) + 0.3 * hg(0.2, cv)) + u.amb_sky.rgb * 0.8;
            let w = trans * alpha * exp(-t * HAZE);
            rgb = rgb + w * s;
            seen = seen + w;
        }
    }
    return vec4<f32>(rgb * u.eye.w, 1.0 - seen);
}

// THE GROUND'S SHADOW: the light every layer lets through toward the light
// from this texel's patch of ground. The outer ring holds the map's mean,
// for the ground past it (`sky::cloud_map_border`).
@fragment fn fs_shadow(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let size = {SHADOW_SIZE};
    if (pos.x < 1.0 || pos.y < 1.0 || pos.x > size - 1.0 || pos.y > size - 1.0) {
        return vec4<f32>(u.amb_ground.w);
    }
    let step = u.shadow.z / size;
    let xz = u.shadow.xy - 0.5 * u.shadow.z + pos.xy * step;
    // Four rays across the light's disc: the penumbra
    // (`cloud_layers::transmittance_penumbra`).
    let l = u.light_dir.xyz;
    let up = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(l.y) < 0.9);
    let a = normalize(cross(l, up));
    let b = cross(l, a);
    var sum = 0.0;
    for (var k = 0u; k < 4u; k = k + 1u) {
        let ang = (f32(k) + 0.5) * 0.5 * PI;
        let d = normalize(l + SUN_RADIUS * 0.7 * (cos(ang) * a + sin(ang) * b));
        sum = sum + transmittance_toward(xz, u.shadow.w, d);
    }
    return vec4<f32>(0.25 * sum);
}
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use space_soup_sky::clouds::CloudParams;

    fn read_back(device: &Device, queue: &Queue, texture: &Texture, layer: u32, width: u32, height: u32, bytes_per_texel: u32) -> Vec<u8> {
        let row = (width * bytes_per_texel).div_ceil(256) * 256;
        let buf = device.create_buffer(&BufferDescriptor { label: None, size: (row * height) as u64, usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ, mapped_at_creation: false });
        let mut enc = device.create_command_encoder(&Default::default());
        enc.copy_texture_to_buffer(
            TexelCopyTextureInfo { texture, mip_level: 0, origin: Origin3d { x: 0, y: 0, z: layer }, aspect: TextureAspect::All },
            TexelCopyBufferInfo { buffer: &buf, layout: TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(height) } },
            Extent3d { width, height, depth_or_array_layers: 1 },
        );
        queue.submit(Some(enc.finish()));
        let slice = buf.slice(..);
        slice.map_async(MapMode::Read, |_| {});
        device.poll(PollType::Wait { submission_index: None, timeout: None }).ok();
        let data = slice.get_mapped_range().unwrap();
        (0..height).flat_map(|y| data[(y * row) as usize..(y * row + width * bytes_per_texel) as usize].to_vec()).collect()
    }

    fn half(b: &[u8]) -> f32 {
        let h = u16::from_le_bytes([b[0], b[1]]);
        let (s, e, m) = ((h >> 15) as u32, ((h >> 10) & 0x1f) as i32, (h & 0x3ff) as f32);
        let v = if e == 0 { m / 1024.0 * 2f32.powi(-14) } else { (1.0 + m / 1024.0) * 2f32.powi(e - 15) };
        if s == 1 { -v } else { v }
    }

    /// THE GPU'S CLOUDS ARE THE CPU'S: the shadow map's texels against the
    /// twin's columns, and a whole sky map with clouds in it, sky between
    /// them, and none past the horizon.
    #[test]
    fn the_gpu_clouds_match_their_twin() {
        let Some((device, queue)) = crate::renderer::terrain_pipeline::tests::headless_gpu() else { return };
        let scope = device.push_error_scope(ErrorFilter::Validation);
        let field = CloudField::build(7);
        let noise = CloudNoise::shared();
        let params = CloudParams { coverage: 0.5, ..Default::default() };
        let layers = space_soup_sky::cloud_layers::CloudModel::new(&params, &noise).layers(300.0);
        let mut sky = CloudSky::new(&device, &queue, &field, &noise);
        let size = crate::renderer::uniforms::CLOUD_SHADOW_SIZE;
        let shadow = device.create_texture(&TextureDescriptor {
            label: None,
            size: Extent3d { width: size, height: size, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::R8Unorm,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let sun = [0.35f32, 0.85, 0.4];
        let (centre, extent) = ([120.0f32, -40.0], 4096.0);
        let u = CloudSkyUniform::new(&layers, [10.0, 2.0, 5.0], sun, [3.0, 2.9, 2.7], [0.4, 0.5, 0.7], [0.2, 0.2, 0.15], centre, extent, 0.5);
        let mut enc = device.create_command_encoder(&Default::default());
        sky.encode(&mut enc, &queue, &u, Some(&shadow.create_view(&Default::default())), true);
        queue.submit(Some(enc.finish()));
        if let Some(e) = pollster::block_on(scope.pop()) {
            panic!("the cloud passes do not validate: {e}");
        }
        let texels = read_back(&device, &queue, &shadow, 0, size, size, 1);
        let dir = { let l = (sun[0] * sun[0] + sun[1] * sun[1] + sun[2] * sun[2]).sqrt(); sun.map(|v| v / l) };
        let step = extent / size as f32;
        let (mut worst, mut sum, mut n, mut shaded) = (0.0f32, 0.0f32, 0, 0);
        for j in 1..size - 1 {
            for i in 1..size - 1 {
                let xz = [centre[0] - 0.5 * extent + (i as f32 + 0.5) * step, centre[1] - 0.5 * extent + (j as f32 + 0.5) * step];
                let cpu = space_soup_sky::cloud_layers::transmittance_penumbra(&layers, &noise, &field, xz, 2.0, dir);
                let gpu = texels[(j * size + i) as usize] as f32 / 255.0;
                worst = worst.max((cpu - gpu).abs());
                sum += (cpu - gpu).abs();
                n += 1;
                shaded += (gpu < 0.5) as u32;
            }
        }
        eprintln!("shadow vs twin: mean |diff| {:.4}, worst {worst:.3}; {shaded} of {n} shaded", sum / n as f32);
        assert!(sum / (n as f32) < 0.02 && worst < 0.25, "the GPU shadow is not the twin's: mean {}, worst {worst}", sum / n as f32);
        assert!(shaded > n / 20 && shaded < n * 19 / 20, "{shaded} of {n} shaded");
        assert_eq!(texels[0], 128, "the border holds the mean");

        let map = read_back(&device, &queue, &sky.maps, sky.display()[1] as u32, MAP_WIDTH, MAP_HEIGHT, 8);
        let alpha: Vec<f32> = (0..(MAP_WIDTH * MAP_HEIGHT) as usize).map(|i| half(&map[i * 8 + 6..i * 8 + 8])).collect();
        let at = |x: u32, y: u32| alpha[(y * MAP_WIDTH + x) as usize];
        let rows = |y0: u32, y1: u32| {
            let v: Vec<f32> = (y0..y1).flat_map(|y| (0..MAP_WIDTH).step_by(4).map(move |x| at(x, y))).collect();
            (v.iter().filter(|&&a| a < 0.5).count() as f32 / v.len() as f32, v.iter().filter(|&&a| a > 0.95).count() as f32 / v.len() as f32)
        };
        let (cloudy, clear) = rows(MAP_HEIGHT / 3, MAP_HEIGHT);
        eprintln!("sky map above 20 deg: {:.0}% cloud, {:.0}% clear", cloudy * 100.0, clear * 100.0);
        assert!(cloudy > 0.05 && clear > 0.05, "the sky map is {cloudy} cloud, {clear} clear");
        let rgb_ok = (0..(MAP_WIDTH * MAP_HEIGHT) as usize).step_by(97).all(|i| (0..3).all(|c| { let v = half(&map[i * 8 + 2 * c..i * 8 + 2 * c + 2]); v.is_finite() && v >= 0.0 }));
        assert!(rgb_ok, "the sky map holds a negative or non-finite light");
        // The map's direction and lookup are each other's inverse.
        for &d in &[[0.3f32, 0.2, -0.9], [0.0, 0.99, 0.1], [-0.7, 0.05, 0.7]] {
            let back = map_dir(map_uv(d));
            let l = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            assert!((0..3).all(|k| (back[k] - d[k] / l).abs() < 1e-4), "{d:?} -> {back:?}");
        }
    }
}
