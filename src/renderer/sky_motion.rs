//! SPACEWARP'S SKY (2026-10-21): the motion of every pixel nothing nearer
//! covers. The sky is at infinity, so it moves on screen with every turn of
//! the head, and a cloud moves on its wind as well. The motion pass used to
//! leave the sky at its clear value, zero: a sky stuck to the head while it
//! turned, in every synthesised frame.
//!
//! One full-screen triangle at the far plane, after the scene's draws and
//! before the particles (which blend their motion over it), passing only
//! where the depth is still the clear value. A pixel's direction is moved
//! back by the cloud clock's step on the wind of the height the newest cloud
//! map holds for it (`cloud_sky::cloud_flow_wgsl`, the sky shader's own
//! lookup), weighted by how much cloud covers it: a clear sky moves with the
//! head alone.

use crate::renderer::space_warp::{motion_block, MotionCamera, BRIGHT_LAYER_CONTRAST, MOTION_FORMAT};

/// Bytes between the eyes' uniforms (both bindings fit in one).
pub const EYE_STRIDE: u64 = 512;
/// Where the flow's uniform starts within an eye's.
const FLOW_OFFSET: u64 = 256;

/// The sky pass's own inputs, beside the eye's camera (`MotionCamera`, in
/// world space: curr and prev take world points to clip, eye the world eye).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SkyFlow {
    /// Clip to world, this frame: a pixel's direction.
    pub inv_curr: [[f32; 4]; 4],
    /// The motion target's width and height.
    pub size: [f32; 4],
    /// The newest map's layer, the cloud clock's step (s), the eye's
    /// altitude, 1 with clouds up (`sky::SkyMotionSource::misc`).
    pub misc: [f32; 4],
    /// Each cloud layer's height and wind, lowest first.
    pub winds: [[f32; 4]; 4],
}

pub fn shader() -> String {
    format!(
        r#"
{motion}
{flow}

struct SkyFlow {{
    inv_curr: mat4x4<f32>,
    size: vec4<f32>,
    misc: vec4<f32>,
    winds: array<vec4<f32>, 4>,
}}
@group(0) @binding(1) var<uniform> flow: SkyFlow;
@group(0) @binding(2) var cloud_maps: texture_2d_array<f32>;
@group(0) @binding(3) var cloud_alts: texture_2d_array<f32>;
@group(0) @binding(4) var cloud_samp: sampler;

@vertex fn vs_sky(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {{
    let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(p * 2.0 - 1.0, 1.0, 1.0);
}}

@fragment fn fs_sky(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {{
    let ndc = vec2<f32>(2.0 * pos.x / flow.size.x - 1.0, 1.0 - 2.0 * pos.y / flow.size.y);
    let far = flow.inv_curr * vec4<f32>(ndc, 1.0, 1.0);
    let dir = normalize(far.xyz / far.w - cam.eye.xyz);
    var back = dir;
    if (flow.misc.w > 0.5 && dir.y > 0.0) {{
        let uv = cloud_map_uv(dir);
        let layer = i32(flow.misc.x);
        let here = textureSampleLevel(cloud_alts, cloud_samp, uv, layer, 0.0);
        let alt = here.r;
        let cover = 1.0 - textureSampleLevel(cloud_maps, cloud_samp, uv, layer, 0.0).a;
        if (alt > 0.0 && cover > 0.0) {{
            // The wind the map stores for what it shows here (`fs_map`).
            let v = here.ba;
            let p = dir * cloud_flow_distance(flow.misc.z, alt, dir.y) - vec3<f32>(v.x, 0.0, v.y) * flow.misc.y;
            back = normalize(mix(dir, normalize(p), layer_weight(cover, {contrast:?})));
        }}
    }}
    return motion_of(cam.curr * vec4<f32>(dir, 0.0), cam.prev * vec4<f32>(back, 0.0));
}}
"#,
        motion = motion_block(),
        flow = crate::renderer::cloud_sky::cloud_flow_wgsl(),
        contrast = BRIGHT_LAYER_CONTRAST,
    )
}

/// The pass, its two eyes' uniforms, and the bind group over the sky's maps
/// (made again when the sky binds new ones).
pub struct SkyMotion {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    uniform: wgpu::Buffer,
    sampler: wgpu::Sampler,
    group: std::cell::RefCell<Option<(u64, wgpu::BindGroup)>>,
}

/// One eye's sky draw for `space_warp::record`.
pub struct SkyDraw<'a> {
    pub pipeline: &'a wgpu::RenderPipeline,
    pub group: &'a wgpu::BindGroup,
    pub offsets: [u32; 2],
}

impl SkyMotion {
    pub fn new(device: &wgpu::Device, depth_format: wgpu::TextureFormat) -> Self {
        let buffer = |binding, size: usize| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: true, min_binding_size: wgpu::BufferSize::new(size as u64) },
            count: None,
        };
        let maps = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2Array,
                multisampled: false,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("space_warp_sky"),
            entries: &[
                buffer(0, std::mem::size_of::<MotionCamera>()),
                buffer(1, std::mem::size_of::<SkyFlow>()),
                maps(2),
                maps(3),
                wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("space_warp_sky"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("space_warp_sky"), source: wgpu::ShaderSource::Wgsl(shader().into()) });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("space_warp_sky"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState { module: &module, entry_point: Some("vs_sky"), compilation_options: Default::default(), buffers: &[] },
            primitive: wgpu::PrimitiveState { cull_mode: None, ..Default::default() },
            // At the far plane, only where nothing nearer was drawn.
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth_format,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_sky"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState { format: MOTION_FORMAT, blend: None, write_mask: wgpu::ColorWrites::ALL })],
            }),
            multiview_mask: None,
            cache: None,
        });
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("space_warp_sky"),
            size: 2 * EYE_STRIDE,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("space_warp_sky"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        Self { pipeline, layout, uniform, sampler, group: std::cell::RefCell::new(None) }
    }

    /// Eye `eye`'s camera (world space) and flow, for this frame.
    pub fn write(&self, queue: &wgpu::Queue, eye: usize, cam: &MotionCamera, flow: &SkyFlow) {
        let base = eye as u64 * EYE_STRIDE;
        queue.write_buffer(&self.uniform, base, bytemuck::bytes_of(cam));
        queue.write_buffer(&self.uniform, base + FLOW_OFFSET, bytemuck::bytes_of(flow));
    }

    /// The bind group over `src`'s maps, made again only when they change.
    pub fn group(&self, device: &wgpu::Device, src: &crate::renderer::sky::SkyMotionSource) -> wgpu::BindGroup {
        let mut slot = self.group.borrow_mut();
        if let Some((key, g)) = slot.as_ref() {
            if *key == src.key {
                return g.clone();
            }
        }
        let binding = |offset: u64, size: usize| {
            wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: &self.uniform, offset, size: wgpu::BufferSize::new(size as u64) })
        };
        let g = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("space_warp_sky"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: binding(0, std::mem::size_of::<MotionCamera>()) },
                wgpu::BindGroupEntry { binding: 1, resource: binding(FLOW_OFFSET, std::mem::size_of::<SkyFlow>()) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&src.maps) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&src.alts) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&self.sampler) },
            ],
        });
        *slot = Some((src.key, g.clone()));
        g
    }

    pub fn draw<'a>(&'a self, group: &'a wgpu::BindGroup, eye: usize) -> SkyDraw<'a> {
        let base = (eye as u64 * EYE_STRIDE) as u32;
        SkyDraw { pipeline: &self.pipeline, group, offsets: [base, base] }
    }
}

/// The flow for one eye: `world_to_clip` this frame, the target's size, and
/// what the sky gives (`Sky::motion_source`).
pub fn flow_for(world_to_clip: glam::Mat4, size: (u32, u32), src: &crate::renderer::sky::SkyMotionSource) -> SkyFlow {
    SkyFlow {
        inv_curr: world_to_clip.inverse().to_cols_array_2d(),
        size: [size.0 as f32, size.1 as f32, 0.0, 0.0],
        misc: src.misc,
        winds: src.winds,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::renderer::cloud_sky::{self as cs, CloudSky, CloudSkyUniform, MAP_HEIGHT, MAP_WIDTH};
    use crate::renderer::space_warp::{layer_motion_weight, record, MotionPipelines, NEAR_Z, FAR_Z};
    use glam::{Mat4, Quat, Vec2, Vec3, Vec4};

    const W: u32 = 48;

    fn half(b: &[u8]) -> f32 {
        let h = u16::from_le_bytes([b[0], b[1]]);
        let (s, e, m) = ((h >> 15) as u32, ((h >> 10) & 0x1f) as i32, (h & 0x3ff) as f32);
        let v = if e == 0 { m / 1024.0 * 2f32.powi(-14) } else { (1.0 + m / 1024.0) * 2f32.powi(e - 15) };
        if s == 1 { -v } else { v }
    }

    fn read(device: &wgpu::Device, queue: &wgpu::Queue, texture: &wgpu::Texture, layer: u32, w: u32, h: u32, bpp: u32) -> Vec<u8> {
        let row = (w * bpp).div_ceil(256) * 256;
        let buf = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (row * h) as u64, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
        let mut enc = device.create_command_encoder(&Default::default());
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo { texture, mip_level: 0, origin: wgpu::Origin3d { x: 0, y: 0, z: layer }, aspect: wgpu::TextureAspect::All },
            wgpu::TexelCopyBufferInfo { buffer: &buf, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(h) } },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        queue.submit(Some(enc.finish()));
        buf.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        let data = buf.slice(..).get_mapped_range().unwrap();
        (0..h).flat_map(|y| data[(y * row) as usize..(y * row + w * bpp) as usize].to_vec()).collect()
    }

    fn camera(eye: Vec3, yaw: f32, pitch: f32) -> Mat4 {
        Mat4::perspective_rh(1.5, 1.0, NEAR_Z, FAR_Z) * Mat4::from_rotation_translation(Quat::from_rotation_y(yaw) * Quat::from_rotation_x(pitch), eye).inverse()
    }

    /// Pixel (i, j)'s direction, as the shader reconstructs it.
    fn pixel_dir(curr: Mat4, eye: Vec3, i: u32, j: u32) -> Vec3 {
        let ndc = Vec2::new(2.0 * (i as f32 + 0.5) / W as f32 - 1.0, 1.0 - 2.0 * (j as f32 + 0.5) / W as f32);
        (curr.inverse().project_point3(ndc.extend(1.0)) - eye).normalize()
    }

    fn ndc_of(m: Mat4, d: Vec3) -> Vec2 {
        let c = m * Vec4::new(d.x, d.y, d.z, 0.0);
        Vec2::new(c.x / c.w, c.y / c.w)
    }

    /// The sky pass alone (no draws) into a W x W target: each pixel's motion.
    fn run(device: &wgpu::Device, queue: &wgpu::Queue, pipelines: &MotionPipelines, src: &crate::renderer::sky::SkyMotionSource, curr: Mat4, prev: Mat4, eye: Vec3) -> Vec<Vec2> {
        let cam = MotionCamera { curr: curr.to_cols_array_2d(), prev: prev.to_cols_array_2d(), params: [1.0, 1.0, 0.0, 0.0], eye: [eye.x, eye.y, eye.z, 1.0], reflect: [0.0; 4] };
        pipelines.sky.write(queue, 0, &cam, &flow_for(curr, (W, W), src));
        let group = pipelines.sky.group(device, src);
        let target = |format, usage| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d { width: W, height: W, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
        };
        let motion = target(MOTION_FORMAT, wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC);
        let depth = target(wgpu::TextureFormat::Depth32Float, wgpu::TextureUsages::RENDER_ATTACHMENT);
        let cams = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: 256, usage: wgpu::BufferUsages::UNIFORM, mapped_at_creation: false });
        let cameras = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipelines.camera_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: &cams, offset: 0, size: wgpu::BufferSize::new(std::mem::size_of::<MotionCamera>() as u64) }),
            }],
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        record(&mut encoder, pipelines, &cameras, &motion.create_view(&Default::default()), &depth.create_view(&Default::default()), &[], &[], Some(pipelines.sky.draw(&group, 0)), false, 1.0);
        queue.submit(Some(encoder.finish()));
        let bytes = read(device, queue, &motion, 0, W, W, 8);
        (0..(W * W) as usize).map(|p| Vec2::new(half(&bytes[p * 8..]), half(&bytes[p * 8 + 2..]))).collect()
    }

    fn dummies(device: &wgpu::Device, queue: &wgpu::Queue) -> crate::renderer::sky::SkyMotionSource {
        use wgpu::util::DeviceExt;
        let one = |format, bytes: &[u8]| {
            device
                .create_texture_with_data(
                    queue,
                    &wgpu::TextureDescriptor {
                        label: None,
                        size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format,
                        usage: wgpu::TextureUsages::TEXTURE_BINDING,
                        view_formats: &[],
                    },
                    wgpu::util::TextureDataOrder::LayerMajor,
                    bytes,
                )
                .create_view(&wgpu::TextureViewDescriptor { dimension: Some(wgpu::TextureViewDimension::D2Array), ..Default::default() })
        };
        let clear: Vec<u8> = [0u16, 0, 0, 0x3c00].iter().flat_map(|v| v.to_le_bytes()).collect();
        crate::renderer::sky::SkyMotionSource {
            key: 1,
            maps: one(wgpu::TextureFormat::Rgba16Float, &clear),
            alts: one(wgpu::TextureFormat::Rgba16Float, &[0, 0, 0x00, 0x3c, 0, 0, 0, 0]),
            misc: [0.0; 4],
            winds: [[0.0; 4]; 4],
        }
    }

    /// A clear sky moves on screen as points at infinity do when the head
    /// turns: every pixel, not the zero it was left at.
    #[test]
    fn the_clear_sky_moves_with_the_head_as_a_point_at_infinity_would() {
        let Some((device, queue)) = crate::renderer::terrain_pipeline::tests::headless_gpu() else { return };
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let pipelines = MotionPipelines::new(&device, wgpu::TextureFormat::Depth32Float);
        let eye = Vec3::new(3.0, 1.6, -2.0);
        let (curr, prev) = (camera(eye, 0.03, 0.4), camera(eye, 0.0, 0.38));
        let got = run(&device, &queue, &pipelines, &dummies(&device, &queue), curr, prev, eye);
        if let Some(e) = pollster::block_on(scope.pop()) {
            panic!("the sky motion pass does not validate: {e}");
        }
        let mut worst = 0.0f32;
        for j in 0..W {
            for i in 0..W {
                let d = pixel_dir(curr, eye, i, j);
                let want = ndc_of(curr, d) - ndc_of(prev, d);
                worst = worst.max((got[(j * W + i) as usize] - want).abs().max_element());
                assert!(want.length() > 0.01, "the turn moves nothing at ({i}, {j})");
            }
        }
        eprintln!("clear sky: worst |shader - reference| {worst:.5}");
        assert!(worst < 2e-3, "the sky's motion is not a far point's: {worst}");
    }

    /// With the head still, a cloud's pixel moves as its cloud does on its
    /// height's wind over the clock's step, and a gap in the clouds does not.
    #[test]
    fn a_cloud_moves_on_its_wind_and_the_gaps_do_not() {
        let Some((device, queue)) = crate::renderer::terrain_pipeline::tests::headless_gpu() else { return };
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        use space_soup_sky::cloud_layers::{CloudModel, CloudNoise};
        use space_soup_sky::cloud_population::CloudPopulation;
        use space_soup_sky::clouds::{CloudField, CloudParams};
        let field = CloudField::build(7);
        let noise = CloudNoise::shared();
        let params = CloudParams { coverage: 0.5, ..Default::default() };
        let t = 20_000.0;
        let mut population = CloudPopulation::new();
        population.update(&params, &field, t);
        let mut layers = CloudModel::new(&params, &noise).layers(t);
        population.place(&mut layers, t);
        let eye = Vec3::new(10.0, 2.0, 5.0);
        let mut sky = CloudSky::new(&device, &queue, &field, &noise);
        sky.set_population(&device, &queue, &population);
        sky.set_winds(cs::flow_winds(&params, &layers));
        let u = CloudSkyUniform::new(&layers, eye.to_array(), [0.35, 0.85, 0.4], [3.0, 2.9, 2.7], [0.4, 0.5, 0.7], [0.2, 0.2, 0.15], [0.0, 0.0], 4096.0, 0.5);
        let mut enc = device.create_command_encoder(&Default::default());
        sky.encode(&mut enc, &queue, &u, None, true);
        queue.submit(Some(enc.finish()));
        let layer = sky.display()[1];
        let winds = cs::flow_winds(&params, &layers);
        let dt = 20.0;
        let src = crate::renderer::sky::SkyMotionSource {
            key: 2,
            maps: sky.maps_view().clone(),
            alts: sky.alts_view().clone(),
            misc: [layer, dt, eye.y, 1.0],
            winds,
        };
        let pipelines = MotionPipelines::new(&device, wgpu::TextureFormat::Depth32Float);
        let curr = camera(eye, 0.4, 1.0);
        let got = run(&device, &queue, &pipelines, &src, curr, curr, eye);
        if let Some(e) = pollster::block_on(scope.pop()) {
            panic!("the sky motion pass does not validate: {e}");
        }
        let (maps, alts) = sky.textures();
        let map = read(&device, &queue, maps, layer as u32, MAP_WIDTH, MAP_HEIGHT, 8);
        let alt = read(&device, &queue, alts, layer as u32, MAP_WIDTH, MAP_HEIGHT, 8);
        let texel = |uv: [f32; 2], f: &dyn Fn(usize) -> f32| {
            let x = ((uv[0] * MAP_WIDTH as f32) as u32).min(MAP_WIDTH - 1);
            let y = ((uv[1] * MAP_HEIGHT as f32) as u32).min(MAP_HEIGHT - 1);
            f((y * MAP_WIDTH + x) as usize)
        };
        let (mut moving, mut close, mut gaps, mut still) = (0, 0, 0, 0);
        for j in 0..W {
            for i in 0..W {
                let d = pixel_dir(curr, eye, i, j);
                if d.y <= 0.05 {
                    continue;
                }
                let uv = cs::map_uv(d.to_array());
                let a = texel(uv, &|k| half(&map[k * 8 + 6..]));
                let h = texel(uv, &|k| half(&alt[k * 8..]));
                let g = got[(j * W + i) as usize];
                if a > 0.995 {
                    gaps += 1;
                    still += (g.length() < 1e-3) as u32;
                } else if a < 0.1 && h > 0.0 {
                    let v = [texel(uv, &|k| half(&alt[k * 8 + 4..])), texel(uv, &|k| half(&alt[k * 8 + 6..]))];
                    let p = d * cs::flow_distance(eye.y, h, d.y) - Vec3::new(v[0], 0.0, v[1]) * dt;
                    let back = d.lerp(p.normalize(), layer_motion_weight(1.0 - a, crate::renderer::space_warp::BRIGHT_LAYER_CONTRAST)).normalize();
                    let want = ndc_of(curr, d) - ndc_of(curr, back);
                    moving += 1;
                    close += ((g - want).length() < 0.15 * want.length() + 2e-3) as u32;
                }
            }
        }
        eprintln!("{moving} cloud pixels, {close} as the reference; {gaps} gaps, {still} still");
        assert!(moving > 50 && gaps > 20, "the view holds {moving} cloud and {gaps} gap pixels");
        assert!(close * 10 >= moving * 9, "only {close} of {moving} cloud pixels move as their cloud");
        assert!(still * 10 >= gaps * 9, "only {still} of {gaps} gaps are still");
    }
}
