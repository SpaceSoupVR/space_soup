use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingResource, BindingType, CommandEncoder, ComputePipeline, Device,
    ShaderModuleDescriptor, ShaderSource, ShaderStages, StorageTextureAccess, TextureSampleType,
    TextureViewDimension,
};

use super::brush_pipeline::probe_pass;

const GROUP: u32 = 8;

pub fn compute_wgsl() -> String {
    format!(
        r#"
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var src_depth: texture_depth_2d;
@group(0) @binding(2) var dst: texture_storage_2d<rgba16float, write>;

@group(0) @binding(3) var src_linear: sampler;
@group(0) @binding(4) var src_point: sampler;

@group(0) @binding(5) var src_code: texture_2d<f32>;

// THE REFLECTIONS' HISTORY (2026-10-23): last frame's result, carried to
// this texel by its depth, kept within what this frame's same-surface
// neighbours span (`History`).
@group(0) @binding(6) var history: texture_2d<f32>;
@group(0) @binding(7) var history_out: texture_storage_2d<rgba16float, write>;
struct BlurHistory {{
    // This frame's NDC to player space, and to last frame's clip.
    unproject: mat4x4<f32>,
    reproject: mat4x4<f32>,
    // The eye in player space (xyz); last frame's drawn share of the target
    // (w, the dynamic resolution's scale).
    eye: vec4<f32>,
    // On (x); the share of the history kept at the eye (y) and from (w)
    // metres on (z).
    keep: vec4<f32>,
}}
@group(0) @binding(8) var<uniform> hist: BlurHistory;

const SAME_SURFACE_SLACK: f32 = 4e-7;

fn settle(p: vec2<i32>, now: vec4<f32>, uv: vec2<f32>, d0: f32, lo: vec4<f32>, hi: vec4<f32>) -> vec4<f32> {{
    var out = now;
    if (hist.keep.x > 0.0) {{
        let ndc = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, d0, 1.0);
        let then = hist.reproject * ndc;
        let at = vec2<f32>(then.x, -then.y) / max(then.w, 1e-6) * 0.5 + 0.5;
        if (then.w > 0.0 && all(at >= vec2<f32>(0.0)) && all(at <= vec2<f32>(hist.eye.w))) {{
            let here = hist.unproject * ndc;
            let far = distance(here.xyz / here.w, hist.eye.xyz);
            let keep = mix(hist.keep.y, hist.keep.w, smoothstep(0.0, hist.keep.z, far));
            let old = clamp(textureSampleLevel(history, src_linear, at, 0.0), lo, hi);
            out = mix(now, old, keep);
        }}
    }}
    textureStore(dst, p, out);
    textureStore(history_out, p, out);
    return out;
}}

fn span(lo: ptr<function, vec4<f32>>, hi: ptr<function, vec4<f32>>, p: vec2<i32>, o: vec2<i32>, top: vec2<i32>, ok: bool) {{
    if (ok) {{
        let c = textureLoad(src, clamp(p + o, vec2<i32>(0), top), 0);
        *lo = min(*lo, c);
        *hi = max(*hi, c);
    }}
}}

fn same_surface(d: f32, d0: f32, off: vec2<f32>, slope: vec2<f32>) -> bool {{
    return abs(d - d0) <= 2.0 * dot(off, slope) + SAME_SURFACE_SLACK;
}}

fn neighbour(p: vec2<i32>, o: vec2<i32>, top: vec2<i32>, ok: bool, w: f32) -> vec4<f32> {{
    if (!ok) {{
        return vec4<f32>(0.0);
    }}
    return textureLoad(src, clamp(p + o, vec2<i32>(0), top), 0) * w;
}}

@compute @workgroup_size({GROUP}, {GROUP}, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
    let size = vec2<i32>(textureDimensions(src));
    let p = vec2<i32>(id.xy);
    if (p.x >= size.x || p.y >= size.y) {{
        return;
    }}
    let c0 = textureLoad(src, p, 0);

    if (c0.a <= 0.0) {{
        textureStore(dst, p, c0);
        textureStore(history_out, p, c0);
        return;
    }}
    let texel = 1.0 / vec2<f32>(size);
    let centre = (vec2<f32>(p) + vec2<f32>(0.5)) * texel;

    let lo = vec2<f32>(-0.5);
    let ul = textureGather(src_depth, src_point, centre + lo * texel);
    let ur = textureGather(src_depth, src_point, centre + vec2<f32>(0.5, -0.5) * texel);
    let dl = textureGather(src_depth, src_point, centre + vec2<f32>(-0.5, 0.5) * texel);
    let dr = textureGather(src_depth, src_point, centre + vec2<f32>(0.5) * texel);
    let d0 = ul.y;
    if (d0 >= 1.0) {{
        textureStore(dst, p, c0);
        textureStore(history_out, p, c0);
        return;
    }}

    let kul = textureGather(1, src_code, src_point, centre + lo * texel);
    let kur = textureGather(1, src_code, src_point, centre + vec2<f32>(0.5, -0.5) * texel);
    let kdl = textureGather(1, src_code, src_point, centre + vec2<f32>(-0.5, 0.5) * texel);
    let kdr = textureGather(1, src_code, src_point, centre + vec2<f32>(0.5) * texel);
    let k0 = kul.y;
    let f_l = probe_face_same(kul.x, k0);
    let f_r = probe_face_same(kur.y, k0);
    let f_u = probe_face_same(kul.z, k0);
    let f_d = probe_face_same(kdl.y, k0);

    let none = 3.4e38;
    let steps = select(
        vec4<f32>(none),
        abs(vec4<f32>(ul.x, ur.y, ul.z, dl.y) - vec4<f32>(d0)),
        vec4<bool>(f_l, f_r, f_u, f_d),
    );
    let gentler = vec2<f32>(min(steps.x, steps.y), min(steps.z, steps.w));
    let slope = select(gentler, vec2<f32>(0.0), gentler >= vec2<f32>(none));
    let side = vec2<f32>(1.0, 0.0);
    let corner = vec2<f32>(1.0);
    let up = vec2<f32>(0.0, 1.0);
    let ok_l = f_l && same_surface(ul.x, d0, side, slope);
    let ok_r = f_r && same_surface(ur.y, d0, side, slope);
    let ok_u = f_u && same_surface(ul.z, d0, up, slope);
    let ok_d = f_d && same_surface(dl.y, d0, up, slope);
    let ok_ul = probe_face_same(kul.w, k0) && same_surface(ul.w, d0, corner, slope);
    let ok_ur = probe_face_same(kur.z, k0) && same_surface(ur.z, d0, corner, slope);
    let ok_dl = probe_face_same(kdl.x, k0) && same_surface(dl.x, d0, corner, slope);
    let ok_dr = probe_face_same(kdr.y, k0) && same_surface(dr.y, d0, corner, slope);
    var sum: vec4<f32>;
    if (ok_l && ok_r && ok_u && ok_d && ok_ul && ok_ur && ok_dl && ok_dr) {{

        sum = textureSampleLevel(src, src_linear, centre + lo * texel, 0.0)
            + textureSampleLevel(src, src_linear, centre + vec2<f32>(0.5, -0.5) * texel, 0.0)
            + textureSampleLevel(src, src_linear, centre + vec2<f32>(-0.5, 0.5) * texel, 0.0)
            + textureSampleLevel(src, src_linear, centre + vec2<f32>(0.5) * texel, 0.0);
    }} else {{

        let top = size - vec2<i32>(1);
        sum = c0 * 4.0
            + neighbour(p, vec2<i32>(-1, 0), top, ok_l, 2.0)
            + neighbour(p, vec2<i32>(1, 0), top, ok_r, 2.0)
            + neighbour(p, vec2<i32>(0, -1), top, ok_u, 2.0)
            + neighbour(p, vec2<i32>(0, 1), top, ok_d, 2.0)
            + neighbour(p, vec2<i32>(-1, -1), top, ok_ul, 1.0)
            + neighbour(p, vec2<i32>(1, -1), top, ok_ur, 1.0)
            + neighbour(p, vec2<i32>(-1, 1), top, ok_dl, 1.0)
            + neighbour(p, vec2<i32>(1, 1), top, ok_dr, 1.0);
    }}

    let now = vec4<f32>(sum.rgb / max(sum.a, 1e-6) * c0.a, c0.a);
    var least = c0;
    var most = c0;
    if (hist.keep.x > 0.0) {{
        let top = size - vec2<i32>(1);
        span(&least, &most, p, vec2<i32>(-1, 0), top, ok_l);
        span(&least, &most, p, vec2<i32>(1, 0), top, ok_r);
        span(&least, &most, p, vec2<i32>(0, -1), top, ok_u);
        span(&least, &most, p, vec2<i32>(0, 1), top, ok_d);
        span(&least, &most, p, vec2<i32>(-1, -1), top, ok_ul);
        span(&least, &most, p, vec2<i32>(1, -1), top, ok_ur);
        span(&least, &most, p, vec2<i32>(-1, 1), top, ok_dl);
        span(&least, &most, p, vec2<i32>(1, 1), top, ok_dr);
    }}
    settle(p, now, centre, d0, least, most);
}}
{}"#,
        probe_pass::FACE_CODE_WGSL,
    )
}

/// The share of last frame's reflections a texel keeps at the eye, and from
/// `HISTORY_FAR_M` on (`History`). Up close a reflection is large and steady,
/// and a reflection's own parallax (it moves unlike the surface it lies on)
/// would show as lag; far off, a lamp cage's sub-pixel wires came out whole
/// or not at all in each texel and danced as the head moved (user,
/// 2026-10-23: "the lighting and reflections dance" from mid distance).
pub const HISTORY_NEAR: f32 = 0.5;
pub const HISTORY_FAR: f32 = 0.85;
pub const HISTORY_FAR_M: f32 = 8.0;

/// `BlurHistory` in the blur's WGSL.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct HistoryUniform {
    pub unproject: [[f32; 4]; 4],
    pub reproject: [[f32; 4]; 4],
    pub eye: [f32; 4],
    pub keep: [f32; 4],
}

impl HistoryUniform {
    /// This frame's view-projection and player frame against last frame's
    /// (`None`: no history, the blur alone), the eye in player space, and
    /// last frame's dynamic resolution scale.
    pub fn new(view_proj: glam::Mat4, world_to_player: glam::Mat4, prev: Option<(glam::Mat4, glam::Mat4, f32)>, eye: glam::Vec3) -> Self {
        let unproject = view_proj.inverse();
        match prev {
            Some((prev_view_proj, prev_world_to_player, prev_scale)) => Self {
                unproject: unproject.to_cols_array_2d(),
                reproject: (prev_view_proj * prev_world_to_player * world_to_player.inverse() * unproject).to_cols_array_2d(),
                eye: [eye.x, eye.y, eye.z, prev_scale],
                keep: [1.0, HISTORY_NEAR, HISTORY_FAR_M, HISTORY_FAR],
            },
            None => Self { unproject: unproject.to_cols_array_2d(), ..Self::default() },
        }
    }
}

/// THE REFLECTIONS' HISTORY: two textures the size of the probe pass, read
/// and written in turn, and what carries them from frame to frame.
pub struct History {
    _textures: [wgpu::Texture; 2],
    views: [wgpu::TextureView; 2],
    pub uniform: wgpu::Buffer,
}

impl History {
    pub fn new(device: &Device, width: u32, height: u32) -> Self {
        let textures: [wgpu::Texture; 2] = std::array::from_fn(|_| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some("probe_blur_history"),
                size: wgpu::Extent3d { width: width.max(1), height: height.max(1), depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: probe_pass::FORMAT,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
                view_formats: &[],
            })
        });
        let views = std::array::from_fn(|k| textures[k].create_view(&Default::default()));
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("probe_blur_history_uniform"),
            size: std::mem::size_of::<HistoryUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self { _textures: textures, views, uniform }
    }
}

/// An eye's blur: `groups[k]` reads history `k` and writes the other, so a
/// frame uses `groups[frame & 1]`.
pub struct BlurGroups {
    pub groups: [BindGroup; 2],
    pub history: History,
}

pub struct ProbeBlur {
    layout: BindGroupLayout,
    pipeline: ComputePipeline,

    samplers: [wgpu::Sampler; 2],
}

impl ProbeBlur {
    pub fn new(device: &Device) -> Self {
        let layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("probe_blur_layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Depth,
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::WriteOnly,
                        format: probe_pass::FORMAT,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 4,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 5,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 6,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Float { filterable: true },
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 7,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::WriteOnly,
                        format: probe_pass::FORMAT,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 8,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let sampler = |label: &str, filter: wgpu::FilterMode| {
            device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some(label),
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                mag_filter: filter,
                min_filter: filter,
                ..Default::default()
            })
        };
        let samplers = [
            sampler("probe_blur_linear", wgpu::FilterMode::Linear),
            sampler("probe_blur_point", wgpu::FilterMode::Nearest),
        ];
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("probe_blur"),
            source: ShaderSource::Wgsl(compute_wgsl().into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("probe_blur_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("probe_blur"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self {
            layout,
            pipeline,
            samplers,
        }
    }

    pub fn bind_group(&self, device: &Device, target: &probe_pass::Target) -> Option<BlurGroups> {
        let soft = target.soft.as_ref()?;
        let history = History::new(device, target.width, target.height);
        let groups = std::array::from_fn(|k| {
            self.bind_views(device, &target.color_view, &target.depth_view, &soft.write_view, &target.reach_single, &history, k)
        });
        Some(BlurGroups { groups, history })
    }

    /// What carries an eye's reflections from last frame to this one
    /// (`HistoryUniform::new`).
    pub fn write_history(&self, queue: &wgpu::Queue, groups: &BlurGroups, u: &HistoryUniform) {
        queue.write_buffer(&groups.history.uniform, 0, bytemuck::bytes_of(u));
    }

    #[allow(clippy::too_many_arguments)]
    fn bind_views(
        &self,
        device: &Device,
        colour: &wgpu::TextureView,
        depth: &wgpu::TextureView,
        out: &wgpu::TextureView,
        code: &wgpu::TextureView,
        history: &History,
        read: usize,
    ) -> BindGroup {
        device.create_bind_group(&BindGroupDescriptor {
            label: Some("probe_blur"),
            layout: &self.layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(colour),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(depth),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(out),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::Sampler(&self.samplers[0]),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: BindingResource::Sampler(&self.samplers[1]),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: BindingResource::TextureView(code),
                },
                BindGroupEntry {
                    binding: 6,
                    resource: BindingResource::TextureView(&history.views[read]),
                },
                BindGroupEntry {
                    binding: 7,
                    resource: BindingResource::TextureView(&history.views[1 - read]),
                },
                BindGroupEntry {
                    binding: 8,
                    resource: history.uniform.as_entire_binding(),
                },
            ],
        })
    }

    pub fn dispatch(
        &self,
        encoder: &mut CommandEncoder,
        bind_group: &BindGroup,
        (width, height): (u32, u32),
        timestamp_writes: Option<wgpu::ComputePassTimestampWrites<'_>>,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("probe_blur"),
            timestamp_writes,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(width.div_ceil(GROUP), height.div_ceil(GROUP), 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn half(v: f32) -> u16 {
        match v {
            0.0 => 0x0000,
            0.25 => 0x3400,
            0.5 => 0x3800,
            1.0 => 0x3C00,
            529.0 => 0x6022,
            533.0 => 0x602A,
            _ => panic!("no f16 written for {v}"),
        }
    }

    fn from_half(h: u16) -> f32 {
        let e = ((h >> 10) & 0x1f) as i32;
        let m = (h & 0x3ff) as f32;
        let v = if e == 0 {
            m * 2f32.powi(-24)
        } else {
            (1.0 + m / 1024.0) * 2f32.powi(e - 15)
        };
        if h & 0x8000 != 0 {
            -v
        } else {
            v
        }
    }

    /// A point of the world lands, through the history's matrices, where
    /// last frame's eye saw it, though the head and the player both moved
    /// (the player frame turns and slides with locomotion).
    #[test]
    fn the_history_finds_a_point_where_last_frame_saw_it() {
        use glam::{Mat4, Quat, Vec3, Vec4};
        let proj = Mat4::perspective_rh(1.6, 0.9, 0.05, 500.0);
        let view = |eye: Vec3, at: Vec3| Mat4::look_at_rh(eye, at, Vec3::Y);
        let frame = |yaw: f32, offset: Vec3| Mat4::from_quat(Quat::from_rotation_y(yaw).inverse()) * Mat4::from_translation(-offset);
        let (vp0, w2p0) = (proj * view(Vec3::new(0.0, 1.6, 0.0), Vec3::new(0.1, 1.5, -5.0)), frame(0.0, Vec3::ZERO));
        let (vp1, w2p1) = (proj * view(Vec3::new(0.03, 1.61, -0.02), Vec3::new(0.2, 1.5, -5.0)), frame(0.3, Vec3::new(0.5, 0.0, -0.4)));
        let world = Vec3::new(1.2, 0.0, -7.0);
        let clip_now = vp1 * (w2p1 * world.extend(1.0));
        let ndc = clip_now / clip_now.w;
        let u = HistoryUniform::new(vp1, w2p1, Some((vp0, w2p0, 1.0)), Vec3::ZERO);
        let then = Mat4::from_cols_array_2d(&u.reproject) * Vec4::new(ndc.x, ndc.y, ndc.z, 1.0);
        let want = vp0 * (w2p0 * world.extend(1.0));
        assert!(((then / then.w).truncate() - (want / want.w).truncate()).length() < 1e-4, "{then:?} against {want:?}");
        let here = Mat4::from_cols_array_2d(&u.unproject) * Vec4::new(ndc.x, ndc.y, ndc.z, 1.0);
        assert!(((here / here.w).truncate() - (w2p1 * world.extend(1.0)).truncate()).length() < 1e-3, "player space: {here:?}");
        assert_eq!(HistoryUniform::new(vp1, w2p1, None, Vec3::ZERO).keep[0], 0.0, "no last frame, no history");
    }

    /// THE HISTORY AVERAGES A HIGHLIGHT THAT FLICKERS AND KEEPS NO GHOST
    /// (2026-10-23): a bright patch stepping a texel back and forth each
    /// frame, as a sub-pixel wire's reflection did under head motion, changes
    /// a texel at its edge far less from frame to frame with the history
    /// than without; and when the patch is gone, the texel is dark at once,
    /// since this frame's neighbours no longer span the old light.
    #[test]
    fn the_history_steadies_a_flickering_highlight_and_leaves_no_ghost() {
        use wgpu::util::DeviceExt;
        let Some((device, queue)) = crate::renderer::terrain_pipeline::tests::headless_gpu() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        const N: u32 = 16;
        let image = |bright: &dyn Fn(u32, u32) -> bool| {
            let texels: Vec<u16> = (0..N)
                .flat_map(|y| (0..N).map(move |x| (x, y)))
                .flat_map(|(x, y)| if bright(x, y) { [0.8, 0.8, 0.8, 1.0] } else { [0.1, 0.1, 0.1, 1.0] })
                .map(crate::renderer::sky::f32_to_f16)
                .collect();
            device
                .create_texture_with_data(
                    &queue,
                    &wgpu::TextureDescriptor {
                        label: Some("history_src"),
                        size: wgpu::Extent3d { width: N, height: N, depth_or_array_layers: 1 },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: probe_pass::FORMAT,
                        usage: wgpu::TextureUsages::TEXTURE_BINDING,
                        view_formats: &[],
                    },
                    wgpu::util::TextureDataOrder::LayerMajor,
                    bytemuck::cast_slice(&texels),
                )
                .create_view(&Default::default())
        };
        let left = image(&|x, y| (7..9).contains(&x) && (7..9).contains(&y));
        let right = image(&|x, y| (8..10).contains(&x) && (7..9).contains(&y));
        let dark = image(&|_, _| false);
        let plain = |label: &str, format: wgpu::TextureFormat, usage: wgpu::TextureUsages| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d { width: N, height: N, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
        };
        let code = plain("history_code", probe_pass::REACH_FORMAT, wgpu::TextureUsages::TEXTURE_BINDING);
        let depth = plain("history_depth", wgpu::TextureFormat::Depth32Float, wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING);
        let dst = plain("history_dst", probe_pass::FORMAT, wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC);
        let (code_view, depth_view, dst_view) = (code.create_view(&Default::default()), depth.create_view(&Default::default()), dst.create_view(&Default::default()));
        // One plane at depth 0.5 over the whole target.
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("history_depth"),
            source: ShaderSource::Wgsl(
                "@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {\n    let p = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));\n    return vec4<f32>(p[i], 0.5, 1.0);\n}\n".into(),
            ),
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("history_depth"),
            layout: None,
            vertex: wgpu::VertexState { module: &module, entry_point: Some("vs"), compilation_options: Default::default(), buffers: &[] },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Always),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            fragment: None,
            multiview_mask: None,
            cache: None,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("history_depth"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            pass.set_pipeline(&pipeline);
            pass.draw(0..3, 0..1);
        }
        queue.submit([encoder.finish()]);

        let blur = ProbeBlur::new(&device);
        let history = History::new(&device, N, N);
        let row = 256u32;
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("history_readback"),
            size: (row * N) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        // One frame of `src`, reading history `k`; the red of texel (x, y) back.
        let frame = |src: &wgpu::TextureView, k: usize, carried: bool, x: u32, y: u32| -> f32 {
            let prev = carried.then_some((glam::Mat4::IDENTITY, glam::Mat4::IDENTITY, 1.0));
            let u = HistoryUniform::new(glam::Mat4::IDENTITY, glam::Mat4::IDENTITY, prev, glam::Vec3::new(0.0, 0.0, -100.0));
            queue.write_buffer(&history.uniform, 0, bytemuck::bytes_of(&u));
            let group = blur.bind_views(&device, src, &depth_view, &dst_view, &code_view, &history, k);
            let mut encoder = device.create_command_encoder(&Default::default());
            blur.dispatch(&mut encoder, &group, (N, N), None);
            encoder.copy_texture_to_buffer(
                dst.as_image_copy(),
                wgpu::TexelCopyBufferInfo { buffer: &readback, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: None } },
                wgpu::Extent3d { width: N, height: N, depth_or_array_layers: 1 },
            );
            queue.submit([encoder.finish()]);
            readback.slice(..).map_async(wgpu::MapMode::Read, |_| {});
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            let v = {
                let bytes = readback.slice(..).get_mapped_range().unwrap();
                let o = (y * row + x * 8) as usize;
                from_half(u16::from_le_bytes([bytes[o], bytes[o + 1]]))
            };
            readback.unmap();
            v
        };
        let flicker = |carried: bool| {
            let mut k = 0;
            let mut last = (0.0, 0.0);
            for i in 0..24 {
                let a = frame(if i % 2 == 0 { &left } else { &right }, k, carried, 7, 8);
                k = 1 - k;
                last = if i % 2 == 0 { (a, last.1) } else { (last.0, a) };
            }
            (last, k)
        };
        let ((a0, b0), _) = flicker(false);
        let ((a1, b1), k) = flicker(true);
        eprintln!("a texel at the patch's edge, frame to frame: alone {a0:.3} / {b0:.3}, with its history {a1:.3} / {b1:.3}");
        assert!((a1 - b1).abs() < 0.4 * (a0 - b0).abs(), "the history did not steady the flicker: {a1} / {b1} against {a0} / {b0}");
        let gone = frame(&dark, k, true, 8, 8);
        let alone = frame(&dark, 1 - k, false, 8, 8);
        eprintln!("the patch's middle once it is gone: {gone:.4} (alone {alone:.4})");
        assert!((gone - alone).abs() < 2e-3, "a ghost of the patch stayed: {gone} against {alone}");
    }

    #[test]
    fn reflections_blur_along_their_surface_and_keep_their_coverage() {
        use wgpu::util::DeviceExt;
        let Some((device, queue)) = crate::renderer::terrain_pipeline::tests::headless_gpu() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        const N: u32 = 16;
        let make = |label: &str, format: wgpu::TextureFormat, usage: wgpu::TextureUsages| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: N,
                    height: N,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
        };

        let colour_at = |x: u32, y: u32| -> [f32; 4] {
            match (x, y) {
                (3, 8) | (7, 7) => [0.25, 0.25, 0.25, 1.0],
                (1, 5) => [0.0; 4],
                (4..8, 12..) => [0.25, 0.25, 0.25, 1.0],
                (x, _) if x < 8 => [0.5, 0.5, 0.5, 1.0],
                _ => [1.0; 4],
            }
        };
        let texels: Vec<u16> = (0..N)
            .flat_map(|y| (0..N).flat_map(move |x| colour_at(x, y)))
            .map(half)
            .collect();
        let src = device.create_texture_with_data(
            &queue,
            &wgpu::TextureDescriptor {
                label: Some("blur_src"),
                size: wgpu::Extent3d {
                    width: N,
                    height: N,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: probe_pass::FORMAT,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            bytemuck::cast_slice(&texels),
        );

        let code_at = |x: u32, y: u32| -> [f32; 2] {
            match (x, y) {
                (0..4, 12..) => [0.0, 529.0],
                (4..8, 12..) => [0.0, 533.0],
                _ => [0.0; 2],
            }
        };
        let codes: Vec<u16> = (0..N)
            .flat_map(|y| (0..N).flat_map(move |x| code_at(x, y)))
            .map(half)
            .collect();
        let code = device.create_texture_with_data(
            &queue,
            &wgpu::TextureDescriptor {
                label: Some("blur_code"),
                size: wgpu::Extent3d {
                    width: N,
                    height: N,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: probe_pass::REACH_FORMAT,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            bytemuck::cast_slice(&codes),
        );
        let depth = make(
            "blur_depth",
            wgpu::TextureFormat::Depth32Float,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        );
        let dst = make(
            "blur_dst",
            probe_pass::FORMAT,
            wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
        );

        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("blur_test_depth"),
            source: ShaderSource::Wgsl(
                r#"
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    var p = array<vec4<f32>, 12>(
        vec4<f32>(-1.0, -1.0, 0.4, 1.0), vec4<f32>(0.0, -1.0, 0.6, 1.0), vec4<f32>(0.0, 1.0, 0.6, 1.0),
        vec4<f32>(-1.0, -1.0, 0.4, 1.0), vec4<f32>(0.0, 1.0, 0.6, 1.0), vec4<f32>(-1.0, 1.0, 0.4, 1.0),
        vec4<f32>(0.0, -1.0, 0.9, 1.0), vec4<f32>(1.0, -1.0, 0.9, 1.0), vec4<f32>(1.0, 1.0, 0.9, 1.0),
        vec4<f32>(0.0, -1.0, 0.9, 1.0), vec4<f32>(1.0, 1.0, 0.9, 1.0), vec4<f32>(0.0, 1.0, 0.9, 1.0),
    );
    return p[i];
}
"#
                .into(),
            ),
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("blur_test_depth"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Always),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            fragment: None,
            multiview_mask: None,
            cache: None,
        });
        let blur = ProbeBlur::new(&device);
        let (src_view, depth_view, dst_view) = (
            src.create_view(&Default::default()),
            depth.create_view(&Default::default()),
            dst.create_view(&Default::default()),
        );
        let code_view = code.create_view(&Default::default());
        let history = History::new(&device, N, N);
        queue.write_buffer(&history.uniform, 0, bytemuck::bytes_of(&HistoryUniform::default()));
        let group = blur.bind_views(&device, &src_view, &depth_view, &dst_view, &code_view, &history, 0);
        let row = 256u32;
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("blur_readback"),
            size: (row * N) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("blur_test_depth"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            pass.set_pipeline(&pipeline);
            pass.draw(0..12, 0..1);
        }
        blur.dispatch(&mut encoder, &group, (N, N), None);
        encoder.copy_texture_to_buffer(
            dst.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(row),
                    rows_per_image: None,
                },
            },
            wgpu::Extent3d {
                width: N,
                height: N,
                depth_or_array_layers: 1,
            },
        );
        queue.submit([encoder.finish()]);
        readback.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let bytes = readback.slice(..).get_mapped_range().unwrap().to_vec();
        let at = |x: u32, y: u32| -> [f32; 4] {
            let o = (y * row + x * 8) as usize;
            std::array::from_fn(|c| {
                from_half(u16::from_le_bytes([bytes[o + 2 * c], bytes[o + 2 * c + 1]]))
            })
        };
        eprintln!(
            "dark {:?}, dark at the far half {:?}, beside the far half {:?}, empty {:?}, beside it {:?}, far {:?}",
            at(3, 8),
            at(7, 7),
            at(7, 2),
            at(1, 5),
            at(1, 4),
            at(12, 12),
        );
        assert_eq!(
            at(3, 8),
            [0.4375, 0.4375, 0.4375, 1.0],
            "the dark texel was not averaged 1-2-1 along its plane"
        );
        let corner = at(7, 7);
        assert!(
            (corner[0] - 5.0 / 12.0).abs() < 1e-3 && corner[3] == 1.0,
            "the dark texel by the far half took from it: {corner:?}",
        );
        assert_eq!(
            at(7, 2),
            [0.5, 0.5, 0.5, 1.0],
            "the far surface bled into the near one"
        );
        assert_eq!(at(1, 5), [0.0; 4], "an empty texel took a reflection");
        assert_eq!(
            at(1, 4),
            [0.5, 0.5, 0.5, 1.0],
            "a texel beside an empty one lost colour or coverage to it"
        );
        assert_eq!(at(12, 12), [1.0; 4]);
        eprintln!("corner: front {:?}, side {:?}", at(3, 13), at(4, 13));
        assert_eq!(
            at(3, 13),
            [0.5, 0.5, 0.5, 1.0],
            "one face of the corner took the other's reflection"
        );
        assert_eq!(
            at(4, 13),
            [0.25, 0.25, 0.25, 1.0],
            "the other face of the corner took the first's reflection"
        );
    }

    #[test]
    fn the_blur_shader_validates() {
        use wgpu::naga;
        let src = compute_wgsl();
        let module = naga::front::wgsl::parse_str(&src)
            .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&src)));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("{e:?}"));
    }
}
