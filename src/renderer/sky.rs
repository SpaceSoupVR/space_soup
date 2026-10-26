pub use space_soup_sky::{
    decode_radiance, project_irradiance, sky_lighting, uv_to_direction, Panorama, SkyIrradiance,
    SkySun,
};
pub use space_soup_sky::time_of_day::{AtlasLayers, LayerWeights, SkySnapshot, TimeOfDayParams, TimeOfDaySky};

use wgpu::*;

use super::lights::wgsl_lights_block;

pub const AMBIENT: f32 = 0.6;

pub fn direction_to_uv(d: [f32; 3]) -> [f32; 2] {
    let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt().max(1e-6);
    let (x, y, z) = (d[0] / len, d[1] / len, d[2] / len);
    let u = (x.atan2(z) + std::f32::consts::PI) / (2.0 * std::f32::consts::PI);
    let v = y.clamp(-1.0, 1.0).acos() / std::f32::consts::PI;
    [u, v]
}

pub fn f32_to_f16(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let mut exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mant = bits & 0x7f_ffff;
    if exp >= 0x1f {
        return sign | 0x7c00;
    }
    if exp <= 0 {
        return sign;
    }
    if exp > 0x1e {
        exp = 0x1e;
    }
    sign | ((exp as u16) << 10) | ((mant >> 13) as u16)
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SkyExtra {

    pub mode: [f32; 4],

    pub sun: [f32; 4],

    pub sun_radiance: [f32; 4],

    pub moon: [f32; 4],

    pub moon_radiance: [f32; 4],

    pub moon_sunward: [f32; 4],

    pub stars: [f32; 4],

    pub extinction: [f32; 4],

    pub celestial: [[f32; 4]; 3],
}

impl SkyExtra {

    pub fn from_snapshot(snap: &SkySnapshot, seconds: f32, zenith_depth: [f32; 3]) -> Self {
        let d = &snap.discs;
        let v4 = |v: [f32; 3], w: f32| [v[0], v[1], v[2], w];

        let mean_sky = 1.0 / snap.panorama_scale.max(1e-30);
        let px = (0.05f32).to_radians();
        let brightest = d.star_zero_point * 10f32.powf(0.6) / (2.0 * std::f32::consts::PI * (0.5 * px).powi(2));
        let stars_show = d.star_count > 0.0 && brightest > 1e-3 * mean_sky;
        Self {
            mode: [1.0, 1.0 / snap.panorama_scale.max(1e-30), seconds, if stars_show { 1.0 } else { 0.0 }],
            sun: v4(d.sun_dir, space_soup_sky::time_of_day::SUN_ANGULAR_RADIUS),
            sun_radiance: v4(d.sun_radiance, 0.6),
            moon: v4(d.moon_dir, space_soup_sky::time_of_day::MOON_ANGULAR_RADIUS),
            moon_radiance: v4(d.moon_radiance, d.earthshine),
            moon_sunward: v4(d.moon_sunward, 0.0),
            stars: [d.star_zero_point, d.limiting_magnitude, d.star_count, 1.0],
            extinction: v4(zenith_depth, 0.0),
            celestial: [v4(d.celestial_rows[0], 0.0), v4(d.celestial_rows[1], 0.0), v4(d.celestial_rows[2], 0.0)],
        }
    }
}

pub struct Sky {
    pub bind_group: BindGroup,

    pub irradiance: SkyIrradiance,

    pub sun: Option<SkySun>,

    pub reflection: Option<ReflectionSky>,

    pub extra: SkyExtra,
    extra_buffer: Buffer,
    texture: Texture,
    sampler: Sampler,
}

#[derive(Clone)]
pub struct ReflectionSky {
    pub pano: Panorama,
    pub rotation_deg: f32,
    pub intensity: f32,
}

impl ReflectionSky {

    pub fn new(pano: &Panorama, rotation_deg: f32, intensity: f32) -> Self {
        let pano = match space_soup_sky::extract_sun(pano, rotation_deg, intensity) {
            Some((_, rest)) => rest,
            None => pano.clone(),
        };
        Self { pano, rotation_deg, intensity }
    }

    pub fn radiance(&self, d: glam::Vec3) -> [f32; 3] {
        let (rc, rs) = (self.rotation_deg.to_radians().cos(), self.rotation_deg.to_radians().sin());

        let d = d.normalize_or_zero();
        let p = [rc * d.x - rs * d.z, d.y, rs * d.x + rc * d.z];
        let [u, v] = direction_to_uv(p);
        let (w, h) = (self.pano.width, self.pano.height);
        let fx = u * w as f32 - 0.5;
        let fy = (v * h as f32 - 0.5).clamp(0.0, (h - 1) as f32);
        let x0 = fx.floor();
        let tx = fx - x0;
        let y0 = fy.floor();
        let ty = fy - y0;
        let wrap = |x: f32| (x.rem_euclid(w as f32) as u32).min(w - 1);
        let (xa, xb) = (wrap(x0), wrap(x0 + 1.0));
        let (ya, yb) = (y0 as u32, (y0 as u32 + 1).min(h - 1));
        let mut out = [0.0f32; 3];
        let (a, b, c, e) = (self.pano.texel(xa, ya), self.pano.texel(xb, ya), self.pano.texel(xa, yb), self.pano.texel(xb, yb));
        for k in 0..3 {
            let top = a[k] + (b[k] - a[k]) * tx;
            let bottom = c[k] + (e[k] - c[k]) * tx;
            out[k] = (top + (bottom - top) * ty) * self.intensity;
        }
        out
    }

    pub fn cube_faces(&self, res: u32) -> Vec<u8> {
        let res = res.max(1);
        let mut out = Vec::with_capacity((res * res * 6 * 8) as usize);
        for face in 0..6usize {
            for y in 0..res {
                for x in 0..res {
                    let mut acc = [0.0f32; 3];
                    for (sx, sy) in [(0.25f32, 0.25f32), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)] {
                        let d = crate::renderer::probe_prefilter::texel_direction(
                            face,
                            (x as f32 + sx) / res as f32,
                            (y as f32 + sy) / res as f32,
                        );
                        let r = self.radiance(d);
                        for k in 0..3 {
                            acc[k] += r[k] * 0.25;
                        }
                    }
                    for v in [acc[0], acc[1], acc[2], 1.0] {
                        out.extend_from_slice(&f32_to_f16(v).to_le_bytes());
                    }
                }
            }
        }
        out
    }
}

pub fn sky_bind_group_layout(device: &Device) -> BindGroupLayout {
    device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("sky_bgl"),
        entries: &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Sampler(SamplerBindingType::Filtering),
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}

impl Sky {
    pub fn new(
        device: &Device,
        queue: &Queue,
        layout: &BindGroupLayout,
        pano: &Panorama,
        rotation_deg: f32,
        intensity: f32,
    ) -> Self {
        let texture = device.create_texture(&TextureDescriptor {
            label: Some("sky_panorama"),
            size: Extent3d {
                width: pano.width,
                height: pano.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });

        write_panorama(queue, &texture, pano);

        let sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("sky_sampler"),

            address_mode_u: AddressMode::Repeat,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            ..Default::default()
        });

        let light_pano = pano.for_lighting(1024);
        let (irradiance, sun, without_sun) = match space_soup_sky::extract_sun(&light_pano, rotation_deg, intensity) {
            Some((sun, rest)) => (project_irradiance(&rest, rotation_deg, intensity), Some(sun), rest),
            None => (project_irradiance(&light_pano, rotation_deg, intensity), None, light_pano.clone()),
        };
        let extra = SkyExtra::default();
        let extra_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("sky_extra"),
            size: std::mem::size_of::<SkyExtra>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&extra_buffer, 0, bytemuck::bytes_of(&extra));
        let bind_group = Self::bind(device, layout, &texture, &sampler, &extra_buffer);

        Self {
            bind_group,
            irradiance,
            sun,
            reflection: Some(ReflectionSky { pano: without_sun, rotation_deg, intensity }),
            extra,
            extra_buffer,
            texture,
            sampler,
        }
    }

    fn bind(device: &Device, layout: &BindGroupLayout, texture: &Texture, sampler: &Sampler, extra: &Buffer) -> BindGroup {
        let view = texture.create_view(&TextureViewDescriptor::default());
        device.create_bind_group(&BindGroupDescriptor {
            label: Some("sky_bg"),
            layout,
            entries: &[
                BindGroupEntry { binding: 0, resource: BindingResource::TextureView(&view) },
                BindGroupEntry { binding: 1, resource: BindingResource::Sampler(sampler) },
                BindGroupEntry { binding: 2, resource: extra.as_entire_binding() },
            ],
        })
    }

    pub fn apply_snapshot(&mut self, device: &Device, queue: &Queue, layout: &BindGroupLayout, snap: &SkySnapshot, seconds: f32, zenith_depth: [f32; 3]) {
        let pano = &snap.panorama;
        let size = self.texture.size();
        if size.width != pano.width || size.height != pano.height {
            self.texture = device.create_texture(&TextureDescriptor {
                label: Some("sky_panorama"),
                size: Extent3d { width: pano.width, height: pano.height, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba16Float,
                usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
                view_formats: &[],
            });
            self.bind_group = Self::bind(device, layout, &self.texture, &self.sampler, &self.extra_buffer);
        }
        write_panorama(queue, &self.texture, pano);
        self.irradiance = snap.irradiance;
        self.sun = snap.light;
        self.reflection = Some(ReflectionSky { pano: snap.panorama_engine(), rotation_deg: 0.0, intensity: 1.0 });
        self.extra = SkyExtra::from_snapshot(snap, seconds, zenith_depth);
        queue.write_buffer(&self.extra_buffer, 0, bytemuck::bytes_of(&self.extra));
    }

    pub fn set_clock(&mut self, queue: &Queue, seconds: f32) {
        if self.extra.mode[0] > 0.5 {
            self.extra.mode[2] = seconds;
            queue.write_buffer(&self.extra_buffer, 0, bytemuck::bytes_of(&self.extra.mode));
        }
    }

    pub fn none(device: &Device, queue: &Queue, layout: &BindGroupLayout, ambient: f32) -> Self {
        let mut s = Self::new(
            device,
            queue,
            layout,
            &Panorama::solid([0.0, 0.0, 0.0], 1, 1),
            0.0,
            1.0,
        );
        s.irradiance = SkyIrradiance::flat(ambient);
        s.sun = None;

        s.reflection = None;
        s
    }
}

fn write_panorama(queue: &Queue, texture: &Texture, pano: &Panorama) {
    let mut half = Vec::with_capacity((pano.width * pano.height * 4) as usize);
    for i in 0..(pano.width * pano.height) as usize {
        half.push(f32_to_f16(pano.rgb[i * 3]));
        half.push(f32_to_f16(pano.rgb[i * 3 + 1]));
        half.push(f32_to_f16(pano.rgb[i * 3 + 2]));
        half.push(f32_to_f16(1.0));
    }
    queue.write_texture(
        TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: Origin3d::ZERO,
            aspect: TextureAspect::All,
        },
        bytemuck::cast_slice(&half),
        TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(pano.width * 8),
            rows_per_image: Some(pano.height),
        },
        Extent3d {
            width: pano.width,
            height: pano.height,
            depth_or_array_layers: 1,
        },
    );
}

pub struct SkyPipeline {
    pub pipeline: RenderPipeline,
    pub layout: BindGroupLayout,
}

impl SkyPipeline {
    pub fn new(

        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        samples: u32,
    ) -> Self {
        Self::new_with_view(device, format, uniform_layout, samples, crate::renderer::multiview::ViewMode::Mono)
    }

    pub fn new_stereo(

        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        samples: u32,
    ) -> Self {
        Self::new_with_view(device, format, uniform_layout, samples, crate::renderer::multiview::ViewMode::Stereo)
    }

    fn new_with_view(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        samples: u32,
        view: crate::renderer::multiview::ViewMode,
    ) -> Self {
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("sky_shader"),
            source: ShaderSource::Wgsl(view.shader(sky_shader()).into()),
        });
        let layout = sky_bind_group_layout(device);
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("sky_layout"),
            bind_group_layouts: &[Some(uniform_layout), Some(&layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("sky_pipeline"),
            layout: Some(&pipeline_layout),
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
                    blend: None,
                    write_mask: ColorWrites::COLOR,
                })],
            }),
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                cull_mode: None,
                front_face: FrontFace::Ccw,
                polygon_mode: PolygonMode::Fill,
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

        Self { pipeline, layout }
    }
}

pub const SKY_CRACK_DEBUG: bool = false;

fn sky_shader() -> String {
    format!(
        r"
{lights_block}

@group(1) @binding(0) var sky_tex: texture_2d<f32>;
@group(1) @binding(1) var sky_samp: sampler;

struct SkyExtra {{
    mode: vec4<f32>,
    sun: vec4<f32>,
    sun_radiance: vec4<f32>,
    moon: vec4<f32>,
    moon_radiance: vec4<f32>,
    moon_sunward: vec4<f32>,
    stars: vec4<f32>,
    extinction: vec4<f32>,
    celestial: array<vec4<f32>, 3>,
}}
@group(1) @binding(2) var<uniform> sky_extra: SkyExtra;

struct VOut {{
    @builtin(position) clip: vec4<f32>,
    @location(0) ndc: vec2<f32>,
}}

@vertex fn vs_main(@builtin(vertex_index) vi: u32) -> VOut {{
    var p = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    var out: VOut;

    out.clip = vec4<f32>(p[vi], 1.0, 1.0);
    out.ndc = p[vi];
    return out;
}}

const SKY_PI: f32 = 3.14159265;

const STAR_CELLS: f32 = {star_cells:.1};

const MOON_SEEN: f32 = 2.0;

fn star_hash(x: u32, y: u32, z: u32) -> vec4<f32> {{
    var h = x * 1597334673u ^ y * 3812015801u ^ z * 2798796415u;
    var o = vec4<u32>(0u);
    for (var k = 0u; k < 4u; k = k + 1u) {{
        h = h ^ (h >> 16u);
        h = h * 2246822519u;
        h = h ^ (h >> 13u);
        h = h * 3266489917u;
        h = h ^ (h >> 16u);
        o[k] = h;
        h = h + 0x9e3779b9u;
    }}
    return vec4<f32>(o >> vec4<u32>(8u)) / 16777216.0;
}}

fn star_face_dir(face: u32, uv: vec2<f32>) -> vec3<f32> {{
    switch face {{
        case 0u: {{ return normalize(vec3<f32>(1.0, uv.x, uv.y)); }}
        case 1u: {{ return normalize(vec3<f32>(-1.0, uv.x, uv.y)); }}
        case 2u: {{ return normalize(vec3<f32>(uv.x, 1.0, uv.y)); }}
        case 3u: {{ return normalize(vec3<f32>(uv.x, -1.0, uv.y)); }}
        case 4u: {{ return normalize(vec3<f32>(uv.x, uv.y, 1.0)); }}
        default: {{ return normalize(vec3<f32>(uv.x, uv.y, -1.0)); }}
    }}
}}

fn sky_stars(d: vec3<f32>, px: f32) -> vec3<f32> {{
    let q = vec3<f32>(dot(sky_extra.celestial[0].xyz, d), dot(sky_extra.celestial[1].xyz, d), dot(sky_extra.celestial[2].xyz, d));
    let a = abs(q);
    var face = 0u;
    var uv = vec2<f32>(0.0);
    if (a.x >= a.y && a.x >= a.z) {{
        face = select(1u, 0u, q.x > 0.0);
        uv = q.yz / a.x;
    }} else if (a.y >= a.z) {{
        face = select(3u, 2u, q.y > 0.0);
        uv = q.xz / a.y;
    }} else {{
        face = select(5u, 4u, q.z > 0.0);
        uv = q.xy / a.z;
    }}
    let w = atan(uv) * (4.0 / SKY_PI);
    let f = (w * 0.5 + 0.5) * STAR_CELLS;
    let base = floor(f - 0.5);
    let sigma = max(0.5 * px, 1.5e-4);
    let airmass = 1.0 / (d.y + 0.025 * exp(-11.0 * d.y));
    let dimmed = exp(-sky_extra.extinction.xyz * airmass);
    let twinkle_depth = clamp((airmass - 2.5) / 10.0, 0.0, 0.6) * sky_extra.stars.w;
    let cell_angle = (SKY_PI * 0.5) / STAR_CELLS;
    var sum = vec3<f32>(0.0);
    for (var k = 0; k < 4; k = k + 1) {{
        let cell = base + vec2<f32>(f32(k & 1), f32(k >> 1u));
        if (any(cell < vec2<f32>(0.0)) || any(cell >= vec2<f32>(STAR_CELLS))) {{
            continue;
        }}
        let h = star_hash(u32(cell.x), u32(cell.y), face);

        let c = tan(((cell + 0.5) / STAR_CELLS * 2.0 - 1.0) * (SKY_PI * 0.25));
        let r2 = 1.0 + dot(c, c);
        let omega = cell_angle * cell_angle * (1.0 + c.x * c.x) * (1.0 + c.y * c.y) / (r2 * sqrt(r2));
        let p = sky_extra.stars.z * omega / (4.0 * SKY_PI);
        if (h.x >= p) {{
            continue;
        }}

        let m = max(sky_extra.stars.y + log2(max(h.x / p, 1e-6)) * (0.30103 / 0.47), -1.5);
        let at = star_face_dir(face, tan(((cell + h.yz) / STAR_CELLS * 2.0 - 1.0) * (SKY_PI * 0.25)));
        let off = length(cross(q, at));
        let flux = sky_extra.stars.x * exp2(-1.3287712 * m);
        let spread = flux / (2.0 * SKY_PI * sigma * sigma) * exp(-off * off / (2.0 * sigma * sigma));

        let t = h.w;
        let colour = mix(vec3<f32>(1.18, 0.95, 0.72), vec3<f32>(0.82, 0.92, 1.25), t * t * (3.0 - 2.0 * t));
        let tw = 1.0 + twinkle_depth * (0.6 * sin(sky_extra.mode.z * (11.0 + 7.0 * h.y) + 40.0 * h.z) + 0.4 * sin(sky_extra.mode.z * (23.0 + 9.0 * h.z) + 17.0 * h.y));
        sum = sum + spread * colour * tw;
    }}
    return sum * dimmed;
}}

fn moon_albedo(p: vec3<f32>) -> f32 {{
    var maria = array<vec4<f32>, 11>(
        vec4<f32>(33.0, -16.0, 11.0, 0.5),
        vec4<f32>(18.0, -57.0, 19.0, 0.45),
        vec4<f32>(28.0, 17.0, 7.0, 0.5),
        vec4<f32>(8.5, 31.0, 8.0, 0.5),
        vec4<f32>(17.0, 59.0, 5.0, 0.55),
        vec4<f32>(-8.0, 51.0, 6.5, 0.45),
        vec4<f32>(-21.0, -17.0, 7.0, 0.4),
        vec4<f32>(-24.0, -39.0, 4.0, 0.45),
        vec4<f32>(56.0, 1.0, 6.0, 0.35),
        vec4<f32>(-15.0, 35.0, 4.0, 0.45),
        vec4<f32>(0.0, -30.0, 9.0, 0.3),
    );
    var a = 1.0;
    for (var i = 0; i < 11; i = i + 1) {{
        let m = maria[i];
        let la = m.x * (SKY_PI / 180.0);
        let lo = m.y * (SKY_PI / 180.0);
        let c = vec3<f32>(cos(la) * sin(lo), sin(la), cos(la) * cos(lo));
        let ang = acos(clamp(dot(p, c), -1.0, 1.0)) * (180.0 / SKY_PI);
        a = a * (1.0 - m.w * (1.0 - smoothstep(m.z * 0.55, m.z, ang)));
    }}

    let ty = vec3<f32>(cos(-0.75) * sin(-0.19), sin(-0.75), cos(-0.75) * cos(-0.19));
    a = a + 0.25 * (1.0 - smoothstep(1.0, 4.0, acos(clamp(dot(p, ty), -1.0, 1.0)) * (180.0 / SKY_PI)));

    return a * 1.18;
}}

fn sky_moon(d: vec3<f32>, px: f32) -> vec4<f32> {{
    let m = sky_extra.moon.xyz;
    let rho = sky_extra.moon.w;
    let along = dot(d, m);
    if (along < cos(rho + 3.0 * px) || m.y < -0.02) {{
        return vec4<f32>(0.0);
    }}
    let pole = sky_extra.celestial[2].xyz;
    let up = normalize(pole - m * dot(pole, m));
    let right = cross(m, up);
    let xy = vec2<f32>(dot(d, right), dot(d, up)) / sin(rho);
    let r = length(xy);
    let e = px / rho;
    let cover = 1.0 - smoothstep(1.0 - e, 1.0 + e, r);
    let mu = sqrt(max(1.0 - min(r * r, 1.0), 0.0));
    let n = right * xy.x + up * xy.y - m * mu;
    let mu0 = dot(n, sky_extra.moon_sunward.xyz);
    let lit = smoothstep(-e, e, mu0) * 2.0 * max(mu0, 0.0) / (max(mu0, 0.0) + mu + 1e-4);
    let albedo = moon_albedo(vec3<f32>(xy.x, xy.y, mu));
    let surface = sky_extra.moon_radiance.rgb * albedo * (lit + sky_extra.moon_radiance.w);

    let rgb = sky_extra.moon_radiance.rgb;
    let peak = max(max(rgb.r, rgb.g), rgb.b) * 1.3 * max(camera.post_params.x, 0.0);
    let seen = min(1.0, MOON_SEEN / max(peak, 1e-12));
    return vec4<f32>(surface * cover * seen, cover);
}}

fn sky_sun_disc(d: vec3<f32>, px: f32) -> vec3<f32> {{
    let s = sky_extra.sun.xyz;
    let rho = sky_extra.sun.w;
    let along = dot(d, s);
    if (along < cos(rho + 3.0 * px)) {{
        return vec3<f32>(0.0);
    }}
    let r = length(cross(d, s)) / sin(rho);
    let e = px / rho;
    let cover = 1.0 - smoothstep(1.0 - e, 1.0 + e, r);
    let mu = sqrt(max(1.0 - min(r * r, 1.0), 0.0));
    return sky_extra.sun_radiance.rgb * (1.0 - sky_extra.sun_radiance.w * (1.0 - mu)) * cover;
}}

@fragment fn fs_main(in: VOut) -> @location(0) vec4<f32> {{

    let near = cam_inv_view_proj() * vec4<f32>(in.ndc, 0.0, 1.0);
    let far  = cam_inv_view_proj() * vec4<f32>(in.ndc, 1.0, 1.0);
    let dir = normalize(far.xyz / far.w - near.xyz / near.w);

    let world_dir = to_world_direction(dir);

    let px = max(length(dpdx(world_dir)), length(dpdy(world_dir)));
    let uv = sky_uv(world_dir);
    var radiance = textureSample(sky_tex, sky_samp, uv).rgb * camera.sky_params.x;

    if (sky_extra.mode.x > 0.5) {{
        radiance = radiance * sky_extra.mode.y + sky_sun_disc(world_dir, px);
        let moon = sky_moon(world_dir, px);
        radiance = radiance + moon.rgb;
        if (sky_extra.mode.w > 0.5 && world_dir.y > 0.0) {{
            radiance = radiance + sky_stars(world_dir, px) * (1.0 - moon.a);
        }}
    }}

    {sky_tail}
}}
",
        lights_block = wgsl_lights_block(0, 1),
        star_cells = STAR_CELLS,
        sky_tail = if SKY_CRACK_DEBUG {
            "return vec4<f32>(1.0, 0.0, 1.0, 1.0);"
        } else {
            "return vec4<f32>(tonemap(radiance), 1.0);"
        },
    )
}

pub const STAR_CELLS: f32 = 160.0;

pub fn star_count_in_cells(star_count: f32) -> f32 {
    let n = STAR_CELLS as u32;
    let cell_angle = std::f32::consts::FRAC_PI_2 / STAR_CELLS;
    let mut sum = 0.0f64;
    for y in 0..n {
        for x in 0..n {
            let c = |i: u32| ((i as f32 + 0.5) / STAR_CELLS * 2.0 - 1.0) * std::f32::consts::FRAC_PI_4;
            let (cx, cy) = (c(x).tan(), c(y).tan());
            let r2 = 1.0 + cx * cx + cy * cy;
            let omega = cell_angle * cell_angle * (1.0 + cx * cx) * (1.0 + cy * cy) / (r2 * r2.sqrt());
            sum += f64::from((star_count * omega / (4.0 * std::f32::consts::PI)).min(1.0));
        }
    }
    (sum * 6.0) as f32
}

#[cfg(test)]
pub fn sky_shader_src() -> String {
    sky_shader()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_star_field_holds_the_catalogues_count() {
        for n in [100.0f32, 2_000.0, 9_100.0, 20_000.0] {
            let got = star_count_in_cells(n);
            assert!((got / n - 1.0).abs() < 0.01, "{n} stars asked, the field holds {got}");
        }

        let crowded = star_count_in_cells(1.0e7);
        assert!(crowded < 6.0 * STAR_CELLS * STAR_CELLS + 1.0, "{crowded}");
        assert!(sky_shader().contains(&format!("const STAR_CELLS: f32 = {STAR_CELLS:.1};")));
    }

    #[test]
    fn the_crack_diagnostic_is_off() {
        assert!(!SKY_CRACK_DEBUG, "SKY_CRACK_DEBUG is on: the sky is painted magenta");
    }

    fn workspace_root() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..")
    }

    fn lobby_sky() -> Option<Panorama> {
        let p = workspace_root()
            .join("game/skies/kloofendal_48d_partly_cloudy_puresky/sky.hdr");
        let bytes = std::fs::read(p).ok()?;
        decode_radiance(&bytes).ok()
    }

    #[test]
    fn the_shipped_panorama_decodes() {
        let Some(pano) = lobby_sky() else {
            eprintln!("skipping: the lobby's sky is not installed");
            return;
        };
        assert_eq!(pano.width, pano.height * 2);
        assert!(pano.width >= 1024);
        assert_eq!(pano.rgb.len(), (pano.width * pano.height * 3) as usize);
        assert!(pano.rgb.iter().all(|v| v.is_finite() && *v >= 0.0));
    }

    #[test]
    fn a_panorama_carries_values_above_one() {

        let Some(pano) = lobby_sky() else {
            eprintln!("skipping: the lobby's sky is not installed");
            return;
        };
        let peak = pano.rgb.iter().copied().fold(0.0f32, f32::max);
        assert!(peak > 100.0, "peak radiance {peak} is too low to be HDR");
    }

    #[test]
    fn the_suns_texel_is_where_the_direction_mapping_says_it_is() {

        let Some(pano) = lobby_sky() else {
            eprintln!("skipping: the lobby's sky is not installed");
            return;
        };
        let mut best = (f32::MIN, 0u32, 0u32);
        for y in 0..pano.height {
            for x in 0..pano.width {
                let t = pano.texel(x, y);
                let l = 0.2126 * t[0] + 0.7152 * t[1] + 0.0722 * t[2];
                if l > best.0 {
                    best = (l, x, y);
                }
            }
        }

        let (elev, azim) = (48.0f32.to_radians(), 34.3f32.to_radians());
        let d = [
            elev.cos() * azim.sin(),
            elev.sin(),
            elev.cos() * azim.cos(),
        ];
        let uv = direction_to_uv(d);
        let (px, py) = (
            (uv[0] * pano.width as f32) as u32,
            (uv[1] * pano.height as f32) as u32,
        );
        assert!(
            px.abs_diff(best.1) <= 2 && py.abs_diff(best.2) <= 2,
            "the sun's direction maps to ({px}, {py}) but the brightest texel is at ({}, {})",
            best.1,
            best.2,
        );
    }

    #[test]
    fn the_shipped_sky_lights_the_ground_from_above_and_from_its_sun() {

        let Some(pano) = lobby_sky() else {
            eprintln!("skipping: the lobby's sky is not installed");
            return;
        };
        let sh = project_irradiance(&pano, 0.0, 1.0);

        let up = sh.evaluate([0.0, 1.0, 0.0]);
        let down = sh.evaluate([0.0, -1.0, 0.0]);
        assert!(
            up[0] > down[0] * 1.5,
            "the sky is not brighter above than below: up {up:?}, down {down:?}",
        );

        let a = 34.3f32.to_radians();
        let toward = sh.evaluate([a.sin(), 0.0, a.cos()]);
        let away = sh.evaluate([-a.sin(), 0.0, -a.cos()]);
        assert!(
            toward[0] > away[0],
            "facing the sun ({toward:?}) is not brighter than facing away ({away:?})",
        );

        assert!(
            up[0] > 0.05 && up[0] < 20.0,
            "upward irradiance {up:?} is not a plausible daylight value",
        );
    }

    #[test]
    fn the_uv_mapping_round_trips() {
        for &d in &[
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.3, 0.5, -0.8],
        ] {
            let uv = direction_to_uv(d);
            let back = uv_to_direction(uv[0], uv[1]);
            let n = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            for c in 0..3 {
                assert!(
                    (back[c] - d[c] / n).abs() < 1e-3,
                    "{d:?} -> {uv:?} -> {back:?}",
                );
            }
        }
    }

    #[test]
    fn a_uniform_sky_gives_uniform_irradiance() {

        let pano = Panorama::solid([0.5, 0.5, 0.5], 64, 32);
        let sh = project_irradiance(&pano, 0.0, 1.0);
        for n in [[0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.3, -0.4, 0.9]] {
            let e = sh.evaluate(n);
            for c in 0..3 {
                assert!(
                    (e[c] - 0.5).abs() < 0.02,
                    "normal {n:?} got {e:?}, expected a flat 0.5",
                );
            }
        }
    }

    #[test]
    fn the_solid_angle_weighting_is_not_optional() {

        let mut pano = Panorama::solid([0.0, 0.0, 0.0], 64, 32);
        for y in 0..2 {
            for x in 0..64 {
                let i = ((y * 64 + x) * 3) as usize;
                pano.rgb[i] = 10.0;
                pano.rgb[i + 1] = 10.0;
                pano.rgb[i + 2] = 10.0;
            }
        }
        let up = project_irradiance(&pano, 0.0, 1.0).evaluate([0.0, 1.0, 0.0]);
        assert!(
            up[0] < 1.0,
            "a thin cap at the pole contributed {up:?}, so it is not being \
             weighted by its solid angle",
        );
        assert!(up[0] > 0.0, "the cap contributed nothing at all: {up:?}");
    }

    #[test]
    fn a_sky_bright_on_one_side_lights_that_side() {

        let mut pano = Panorama::solid([0.0, 0.0, 0.0], 64, 32);
        for y in 0..32 {
            for x in 0..32 {
                let i = ((y * 64 + x) * 3) as usize;
                pano.rgb[i] = 4.0;
                pano.rgb[i + 1] = 4.0;
                pano.rgb[i + 2] = 4.0;
            }
        }
        let sh = project_irradiance(&pano, 0.0, 1.0);

        let lit = sh.evaluate([-1.0, 0.0, 0.0]);
        let dark = sh.evaluate([1.0, 0.0, 0.0]);
        assert!(
            lit[0] > dark[0] * 2.0,
            "the lit side got {lit:?} and the dark side {dark:?}",
        );
    }

    #[test]
    fn rotating_the_sky_moves_the_light_with_it() {

        let mut pano = Panorama::solid([0.0, 0.0, 0.0], 64, 32);
        for y in 0..32 {
            for x in 0..16 {
                let i = ((y * 64 + x) * 3) as usize;
                pano.rgb[i] = 8.0;
                pano.rgb[i + 1] = 8.0;
                pano.rgb[i + 2] = 8.0;
            }
        }
        let probe = [-1.0, 0.0, 0.0];
        let straight = project_irradiance(&pano, 0.0, 1.0).evaluate(probe);
        let turned = project_irradiance(&pano, 180.0, 1.0).evaluate(probe);
        assert!(
            (straight[0] - turned[0]).abs() > straight[0] * 0.3,
            "turning the sky by 180 degrees barely changed the light: \
             {straight:?} vs {turned:?}",
        );
    }

    #[test]
    fn intensity_scales_the_light_it_casts() {
        let pano = Panorama::solid([0.4, 0.4, 0.4], 32, 16);
        let one = project_irradiance(&pano, 0.0, 1.0).evaluate([0.0, 1.0, 0.0]);
        let two = project_irradiance(&pano, 0.0, 2.0).evaluate([0.0, 1.0, 0.0]);
        assert!((two[0] - one[0] * 2.0).abs() < 0.01, "{one:?} vs {two:?}");
    }

    #[test]
    fn no_sky_evaluates_to_exactly_the_old_flat_ambient() {

        let flat = SkyIrradiance::flat(0.6);
        for n in [[0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [-0.5, 0.2, 0.8]] {
            let e = flat.evaluate(n);
            for c in 0..3 {
                assert!((e[c] - 0.6).abs() < 1e-4, "normal {n:?} got {e:?}, wanted 0.6");
            }
        }
    }

    #[test]
    fn half_precision_survives_the_range_a_sky_uses() {
        for v in [0.0f32, 0.5, 1.0, 12.5, 250.0, 6000.0] {
            let h = f32_to_f16(v);

            let sign = u32::from(h >> 15);
            let exp = i32::from((h >> 10) & 0x1f);
            let mant = u32::from(h & 0x3ff);
            let back = if exp == 0 {
                0.0
            } else {
                f32::from_bits((sign << 31) | (((exp - 15 + 127) as u32) << 23) | (mant << 13))
            };
            let err = if v == 0.0 { back } else { (back - v).abs() / v };
            assert!(err < 0.01, "{v} became {back}");
        }
    }

    #[test]
    fn a_malformed_file_is_refused_rather_than_guessed_at() {
        assert!(decode_radiance(b"not an hdr at all").is_err());
        assert!(decode_radiance(b"#?RADIANCE\n\n+Y 4 +X 4\n").is_err());
    }
}

#[cfg(test)]
mod render_tests {
    use super::*;
    use crate::renderer::lights::LightsUniform;
    use crate::renderer::uniforms::{PlayerUpload, PostUpload, ShadowUpload, SkyUpload};

    const SIZE: u32 = 9;

    fn split_panorama() -> Panorama {
        let (w, h) = (64u32, 32u32);
        let mut rgb = Vec::with_capacity((w * h * 3) as usize);
        for _y in 0..h {
            for x in 0..w {

                if x < w / 2 {
                    rgb.extend_from_slice(&[3.0, 0.0, 0.0]);
                } else {
                    rgb.extend_from_slice(&[0.0, 0.0, 3.0]);
                }
            }
        }
        Panorama { width: w, height: h, rgb }
    }

    fn look(dir: [f32; 3], pano: &Panorama) -> Option<[u8; 4]> {
        look_turned(dir, pano, 0.0)
    }

    fn look_turned(dir: [f32; 3], pano: &Panorama, yaw: f32) -> Option<[u8; 4]> {
        render_centre(dir, yaw, 1.0, PostUpload::default(), |device, queue, layout| Sky::new(device, queue, layout, pano, 0.0, 1.0))
    }

    fn look_at_snapshot(dir: [f32; 3], snap: &SkySnapshot, zenith: [f32; 3], fov: f32, exposure: f32) -> Option<[u8; 4]> {
        render_centre(dir, 0.0, fov, PostUpload { exposure, ..Default::default() }, |device, queue, layout| {
            let mut sky = Sky::none(device, queue, layout, AMBIENT);
            sky.apply_snapshot(device, queue, layout, snap, 0.0, zenith);
            sky
        })
    }

    fn render_centre(
        dir: [f32; 3],
        yaw: f32,
        fov: f32,
        post: PostUpload,
        make_sky: impl FnOnce(&Device, &Queue, &BindGroupLayout) -> Sky,
    ) -> Option<[u8; 4]> {
        let (device, queue) = crate::renderer::terrain_pipeline::tests::headless_gpu()?;
        let format = TextureFormat::Rgba8Unorm;

        let lights = LightsUniform::new(&device);
        let (_shadows, uniforms) =
            crate::renderer::uniforms::test_support::scene_uniforms(&device, &lights);
        lights.upload(&queue, &[]);

        let eye = glam::Vec3::ZERO;
        let d = glam::Vec3::from(dir).normalize();
        let up = if d.y.abs() > 0.99 { glam::Vec3::Z } else { glam::Vec3::Y };
        let view_proj = glam::Mat4::perspective_rh(fov, 1.0, 0.1, 100.0)
            * glam::Mat4::look_at_rh(eye, eye + d, up);
        uniforms.upload_scene(
            &queue,
            view_proj,
            eye,
            &ShadowUpload::disabled(),
            &SkyUpload { intensity: 1.0, sh: [[0.0; 4]; 9] },
            &post,
            &PlayerUpload { yaw, ..Default::default() },
        );

        let pipeline = SkyPipeline::new(&device, format, &uniforms.layout, 1);
        let sky = make_sky(&device, &queue, &pipeline.layout);

        let desc = |fmt, usage| TextureDescriptor {
            label: None,
            size: Extent3d { width: SIZE, height: SIZE, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: fmt,
            usage,
            view_formats: &[],
        };
        let target = device.create_texture(&desc(
            format,
            TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
        ));
        let depth = device.create_texture(&desc(
            TextureFormat::Depth32Float,
            TextureUsages::RENDER_ATTACHMENT,
        ));
        let target_view = target.create_view(&Default::default());
        let depth_view = depth.create_view(&Default::default());
        let readback = device.create_buffer(&BufferDescriptor {
            label: None,
            size: u64::from(256 * SIZE),
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("sky_test_pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &target_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: Operations { load: LoadOp::Clear(Color::GREEN), store: StoreOp::Store },
                })],
                depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(Operations { load: LoadOp::Clear(1.0), store: StoreOp::Store }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                multiview_mask: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&pipeline.pipeline);
            pass.set_bind_group(0, &uniforms.bind_group, &[]);
            pass.set_bind_group(1, &sky.bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        encoder.copy_texture_to_buffer(
            TexelCopyTextureInfo {
                texture: &target,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            TexelCopyBufferInfo {
                buffer: &readback,
                layout: TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(256),
                    rows_per_image: Some(SIZE),
                },
            },
            Extent3d { width: SIZE, height: SIZE, depth_or_array_layers: 1 },
        );
        queue.submit(Some(encoder.finish()));

        let slice = readback.slice(..);
        slice.map_async(MapMode::Read, |_| {});
        device.poll(PollType::Wait { submission_index: None, timeout: None }).ok();
        let data = slice.get_mapped_range().unwrap();
        let at = (SIZE / 2) as usize * 256 + (SIZE / 2) as usize * 4;
        Some([data[at], data[at + 1], data[at + 2], data[at + 3]])
    }

    macro_rules! shot {
        ($e:expr) => {
            match $e {
                Some(px) => px,
                None => {
                    eprintln!("skipping: no GPU adapter available");
                    return;
                }
            }
        };
    }

    #[test]
    fn the_shader_samples_the_direction_the_cpu_mapping_says_it_should() {

        let pano = split_panorama();

        for dir in [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.3, 0.0, 1.0], [0.3, 0.0, -1.0]] {
            let px = shot!(look(dir, &pano));
            let uv = direction_to_uv(dir);
            let expected = pano.texel(
                ((uv[0] * pano.width as f32) as u32).min(pano.width - 1),
                ((uv[1] * pano.height as f32) as u32).min(pano.height - 1),
            );
            let want_red = expected[0] > expected[2];
            assert!(
                (px[0] > px[2]) == want_red,
                "looking {dir:?}: the CPU mapping says {expected:?} but the \
                 shader drew {px:?}",
            );
        }
    }

    #[test]
    fn the_sky_stays_with_the_world_when_the_player_turns() {
        let pano = split_panorama();
        for yaw in [std::f32::consts::FRAC_PI_2, std::f32::consts::PI, -std::f32::consts::FRAC_PI_4] {
            for world in [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.3, 0.0, 1.0], [0.3, 0.0, -1.0]] {
                let player = glam::Quat::from_rotation_y(-yaw) * glam::Vec3::from(world);
                let px = shot!(look_turned(player.to_array(), &pano, yaw));
                let uv = direction_to_uv(world);
                let expected = pano.texel(
                    ((uv[0] * pano.width as f32) as u32).min(pano.width - 1),
                    ((uv[1] * pano.height as f32) as u32).min(pano.height - 1),
                );
                assert!(
                    (px[0] > px[2]) == (expected[0] > expected[2]),
                    "turned {yaw} rad, looking {world:?} in the world: the sky there is \
                     {expected:?} but the shader drew {px:?}",
                );
            }
        }
    }

    #[test]
    fn the_sky_is_tonemapped_rather_than_clipped() {

        let px = shot!(look([-1.0, 0.0, 0.0], &split_panorama()));
        assert!(px[0] > 240, "red should be near the top of the range, got {px:?}");
        assert!(
            px[1] > 14 && px[1] < 32,
            "expected the ACES cross-channel green (~22), got {px:?}",
        );
        assert!(
            px[2] > 1 && px[2] < 14,
            "expected the ACES cross-channel blue (~6), got {px:?}",
        );
    }

    #[test]
    fn the_sky_only_fills_where_nothing_was_drawn() {

        let Some((device, queue)) = crate::renderer::terrain_pipeline::tests::headless_gpu() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let format = TextureFormat::Rgba8Unorm;
        let lights = LightsUniform::new(&device);
        let (_shadows, uniforms) =
            crate::renderer::uniforms::test_support::scene_uniforms(&device, &lights);
        lights.upload(&queue, &[]);
        uniforms.upload_with_sky(
            &queue,
            glam::Mat4::IDENTITY,
            glam::Vec3::ZERO,
            &ShadowUpload::disabled(),
            &SkyUpload { intensity: 1.0, sh: [[0.0; 4]; 9] },
        );
        let pipeline = SkyPipeline::new(&device, format, &uniforms.layout, 1);
        let pano = split_panorama();
        let sky = Sky::new(&device, &queue, &pipeline.layout, &pano, 0.0, 1.0);

        let desc = |fmt, usage| TextureDescriptor {
            label: None,
            size: Extent3d { width: SIZE, height: SIZE, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: fmt,
            usage,
            view_formats: &[],
        };
        let target = device.create_texture(&desc(
            format,
            TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
        ));
        let depth = device.create_texture(&desc(
            TextureFormat::Depth32Float,
            TextureUsages::RENDER_ATTACHMENT,
        ));
        let tv = target.create_view(&Default::default());
        let dv = depth.create_view(&Default::default());
        let readback = device.create_buffer(&BufferDescriptor {
            label: None,
            size: u64::from(256 * SIZE),
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("occluded_sky"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &tv,
                    depth_slice: None,
                    resolve_target: None,
                    ops: Operations { load: LoadOp::Clear(Color::GREEN), store: StoreOp::Store },
                })],
                depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                    view: &dv,

                    depth_ops: Some(Operations { load: LoadOp::Clear(0.0), store: StoreOp::Store }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                multiview_mask: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&pipeline.pipeline);
            pass.set_bind_group(0, &uniforms.bind_group, &[]);
            pass.set_bind_group(1, &sky.bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        encoder.copy_texture_to_buffer(
            TexelCopyTextureInfo {
                texture: &target, mip_level: 0, origin: Origin3d::ZERO, aspect: TextureAspect::All,
            },
            TexelCopyBufferInfo {
                buffer: &readback,
                layout: TexelCopyBufferLayout {
                    offset: 0, bytes_per_row: Some(256), rows_per_image: Some(SIZE),
                },
            },
            Extent3d { width: SIZE, height: SIZE, depth_or_array_layers: 1 },
        );
        queue.submit(Some(encoder.finish()));
        let slice = readback.slice(..);
        slice.map_async(MapMode::Read, |_| {});
        device.poll(PollType::Wait { submission_index: None, timeout: None }).ok();
        let data = slice.get_mapped_range().unwrap();
        let at = (SIZE / 2) as usize * 256 + (SIZE / 2) as usize * 4;
        let px = [data[at], data[at + 1], data[at + 2], data[at + 3]];
        assert!(
            px[1] > 200 && px[0] < 40 && px[2] < 40,
            "the sky drew over geometry that was already in front of it: {px:?}",
        );
    }

    #[test]
    fn the_time_of_day_sky_draws_noon_and_the_moon() {
        let tod = TimeOfDaySky::new(TimeOfDayParams::default());
        let zenith = tod.atmosphere.eye_transmittance([0.0, 1.0, 0.0]).map(|t| -t.max(1e-6).ln());
        let noon = tod.snapshot(12.0, 0.0);

        let px = shot!(look_at_snapshot([0.0, 1.0, 0.0], &noon, zenith, 0.5, 4.0));
        assert!(px[2] > px[0] + 30 && px[2] > px[1] && px[2] > 60, "noon's zenith is not blue: {px:?}");
        let night = tod.snapshot(23.5, 0.0);
        let moon = night.moon.direction;
        assert!(moon[1] > 0.2, "the moon should be up at 23:30 in the defaults: {moon:?}");

        let disc = shot!(look_at_snapshot(moon, &night, zenith, 0.02, 4.0));
        let m = glam::Vec3::from(moon);
        let beside = (m + m.cross(glam::Vec3::Y).normalize() * 0.05).normalize().to_array();
        let sky = shot!(look_at_snapshot(beside, &night, zenith, 0.02, 4.0));
        assert!(disc[0].max(disc[1]).max(disc[2]) > 60, "the moon's disc is dark: {disc:?}");
        assert!(sky[0].max(sky[1]).max(sky[2]) < 10, "the sky beside the moon is lit: {sky:?}");
    }

    #[test]
    fn the_pipeline_validates_multisampled_too() {

        let Some((device, queue)) = crate::renderer::terrain_pipeline::tests::headless_gpu() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let _ = &queue;
        let lights = LightsUniform::new(&device);
        let (_shadows, uniforms) =
            crate::renderer::uniforms::test_support::scene_uniforms(&device, &lights);
        let err_scope_1 = device.push_error_scope(ErrorFilter::Validation);
        let _one = SkyPipeline::new(&device, TextureFormat::Rgba8UnormSrgb, &uniforms.layout, 1);
        let _four = SkyPipeline::new(&device, TextureFormat::Rgba8UnormSrgb, &uniforms.layout, 4);
        if let Some(e) = pollster::block_on(err_scope_1.pop()) {
            panic!("the sky pipeline failed validation: {e}");
        }
    }
}
