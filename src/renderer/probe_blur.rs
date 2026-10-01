//! A LITTLE BLUR ON EVERY REFLECTION.
//!
//! The probe pass shades each reflection once per half-resolution texel, from
//! one ray. Where what the rays meet changes within a texel -- a lamp's rim, a
//! bulb's highlight, the edge of a fixture's reflection in a polished floor --
//! the texel shows all of one or all of the other, and which one changes as
//! the head moves: the reflections crawl and sparkle at their edges while the
//! surfaces holding them stand still. The user, 2026-09-30: "what if we apply a
//! little blur to ALL reflections ... it will help hide some of the artifact
//! details we're still struggling with, like some jitter of light reflections,
//! jitter of the light fixture reflections especially at its edges".
//!
//! So after the pass and its fix-up, each texel's reflection is averaged with
//! its eight neighbours', 1-2-1 each way: the reconstruction filter one ray a
//! texel never had. Only neighbours on the SAME SURFACE -- a neighbour's depth
//! must continue this texel's plane, within twice its slope -- so a reflection
//! never bleeds across a silhouette, as the reader's own test keeps it from
//! doing (`probe_pass::READER_WGSL`). And only the COLOUR: each texel keeps its
//! own coverage, so a rough tile beside a polished one (coverage 0, see
//! `probe_pass::PASS_LIGHTING`) neither lends the polished one its emptiness
//! nor takes a reflection it would throw away.
//!
//! Averaged as stored -- compressed by `1 / (1 + luminance)` -- so a lone
//! bright texel is one voice among nine rather than a smear: Karis's weight,
//! as the upsample and the mips use.
//!
//! WHAT IS NOT BLURRED: the surfaces. A texture blurred is a texture out of
//! focus, which in a headset reads as poor eyesight, not as realism. Real
//! materials blur what they REFLECT, by their roughness, and the probe pass
//! already does that; this is the half-texel more every reflection gets
//! whatever its roughness -- the micro-roughness even polished stone has --
//! and its job is the texel grid.

use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingResource, BindingType, CommandEncoder, ComputePipeline, Device,
    ShaderModuleDescriptor, ShaderSource, ShaderStages, StorageTextureAccess, TextureSampleType,
    TextureViewDimension,
};

use super::brush_pipeline::probe_pass;

/// Threads a side of one workgroup.
const GROUP: u32 = 8;

/// The filter, a thread a texel.
pub fn compute_wgsl() -> String {
    format!(
        r#"
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var src_depth: texture_depth_2d;
@group(0) @binding(2) var dst: texture_storage_2d<rgba16float, write>;
// Clamped at the edges: a texel's neighbours past the image are the edge's.
@group(0) @binding(3) var src_linear: sampler;
@group(0) @binding(4) var src_point: sampler;

// How far a depth may stray from the plane and still be the same surface, at
// its steepest: a few steps of a 32-bit float just under 1, where the depths
// of anything near are.
const SAME_SURFACE_SLACK: f32 = 4e-7;

// THE CHEAP READS, measured on the headset at GPU level 5 (2026-09-30): a
// texel's nine colours and nine depths read one by one cost 0.26-0.39 ms an
// eye, a pass bound by its fetches; staged in workgroup memory, 0.81 -- on
// this GPU the texture cache beats it. So the nine depths come in four
// gathers, and where all nine are one surface -- everywhere but along an
// outline -- four bilinear reads at the texel's corners ARE the 1-2-1 kernel:
// each averages the four texels round its corner, and the four together
// weigh the centre four times, the sides twice and the corners once.
fn same_surface(d: f32, d0: f32, off: vec2<f32>, slope: vec2<f32>) -> bool {{
    return abs(d - d0) <= 2.0 * dot(off, slope) + SAME_SURFACE_SLACK;
}}

// Neighbour `o` of `p`, weighed by `w`, if it is on `p`'s surface.
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
    // Nothing reflected here: as it is (an empty texel is clear too).
    if (c0.a <= 0.0) {{
        textureStore(dst, p, c0);
        return;
    }}
    let texel = 1.0 / vec2<f32>(size);
    let centre = (vec2<f32>(p) + vec2<f32>(0.5)) * texel;
    // A gather at a texel corner returns the four texels round it as
    // (-x,+y) (+x,+y) (+x,-y) (-x,-y): `ul.y` is this texel, `ul.x` its left,
    // `ul.z` the one above, `ul.w` above-left; `ur.y` its right, `ur.z`
    // above-right; `dl.x` below-left, `dl.y` below; `dr.y` below-right.
    let lo = vec2<f32>(-0.5);
    let ul = textureGather(src_depth, src_point, centre + lo * texel);
    let ur = textureGather(src_depth, src_point, centre + vec2<f32>(0.5, -0.5) * texel);
    let dl = textureGather(src_depth, src_point, centre + vec2<f32>(-0.5, 0.5) * texel);
    let dr = textureGather(src_depth, src_point, centre + vec2<f32>(0.5) * texel);
    let d0 = ul.y;
    if (d0 >= 1.0) {{
        textureStore(dst, p, c0);
        return;
    }}
    // The plane's slope each way: the gentler side's step, so a silhouette on
    // one side does not widen the test on the other.
    let slope = vec2<f32>(min(abs(ul.x - d0), abs(ur.y - d0)), min(abs(ul.z - d0), abs(dl.y - d0)));
    let side = vec2<f32>(1.0, 0.0);
    let corner = vec2<f32>(1.0);
    let up = vec2<f32>(0.0, 1.0);
    let ok_l = same_surface(ul.x, d0, side, slope);
    let ok_r = same_surface(ur.y, d0, side, slope);
    let ok_u = same_surface(ul.z, d0, up, slope);
    let ok_d = same_surface(dl.y, d0, up, slope);
    let ok_ul = same_surface(ul.w, d0, corner, slope);
    let ok_ur = same_surface(ur.z, d0, corner, slope);
    let ok_dl = same_surface(dl.x, d0, corner, slope);
    let ok_dr = same_surface(dr.y, d0, corner, slope);
    var sum: vec4<f32>;
    if (ok_l && ok_r && ok_u && ok_d && ok_ul && ok_ur && ok_dl && ok_dr) {{
        // Premultiplied, so the filter's alpha is the coverage it weighed by.
        sum = textureSampleLevel(src, src_linear, centre + lo * texel, 0.0)
            + textureSampleLevel(src, src_linear, centre + vec2<f32>(0.5, -0.5) * texel, 0.0)
            + textureSampleLevel(src, src_linear, centre + vec2<f32>(-0.5, 0.5) * texel, 0.0)
            + textureSampleLevel(src, src_linear, centre + vec2<f32>(0.5) * texel, 0.0);
    }} else {{
        // Along an outline: each neighbour on this surface, one by one.
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
    // The coverage-weighted mean colour, at this texel's own coverage.
    textureStore(dst, p, vec4<f32>(sum.rgb / max(sum.a, 1e-6) * c0.a, c0.a));
}}
"#
    )
}

/// The blur's pipeline, and the layout of what one dispatch reads and writes.
pub struct ProbeBlur {
    layout: BindGroupLayout,
    pipeline: ComputePipeline,
    /// Linear and point, clamped at the edges. See `compute_wgsl`.
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

    /// What the blur reads and writes for one single-eye probe pass target:
    /// its colour and depth, into its blurred colour. `None` for a target
    /// without one (the two-eye target, which no fix-up or blur follows).
    pub fn bind_group(&self, device: &Device, target: &probe_pass::Target) -> Option<BindGroup> {
        let soft = target.soft.as_ref()?;
        Some(self.bind_views(
            device,
            &target.color_view,
            &target.depth_view,
            &soft.write_view,
        ))
    }

    /// The blur's bindings from the three views: colour and depth read, the
    /// blurred colour written.
    fn bind_views(
        &self,
        device: &Device,
        colour: &wgpu::TextureView,
        depth: &wgpu::TextureView,
        out: &wgpu::TextureView,
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
            ],
        })
    }

    /// Blurs one eye's reflections, after its probe pass and fix-up, before
    /// the scene pass reads them through `Target::soft`'s bind group.
    /// `timestamp_writes`: the pass timer's `blur_l`/`blur_r` slot.
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

    /// f16 bits for the few exact values the test writes.
    fn half(v: f32) -> u16 {
        match v {
            0.0 => 0x0000,
            0.25 => 0x3400,
            0.5 => 0x3800,
            1.0 => 0x3C00,
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

    /// THE BLUR ON ITS OWN, on a 16x16 target -- four workgroups -- holding
    /// two surfaces: the left half a slanted plane (its depth steps 0.025 a
    /// texel), the right half a flat one far behind it. A dark texel on the
    /// left, its neighbours across a workgroup's edge, is averaged with them
    /// 1-2-1 each way to exactly 7/16; one at the far half's edge, in the
    /// corner of its group, takes nothing from the far half (5/12); a texel
    /// beside the far half takes nothing from it; an empty texel stays empty,
    /// and its neighbour keeps both its colour and its coverage rather than
    /// averaging the emptiness in.
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
        // The colour: 0.5 on the left, 1 on the right, premultiplied; one
        // dark texel, one empty one.
        let colour_at = |x: u32, y: u32| -> [f32; 4] {
            match (x, y) {
                (3, 8) | (7, 7) => [0.25, 0.25, 0.25, 1.0],
                (1, 5) => [0.0; 4],
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
        // The depths, drawn: the left half from 0.4 to 0.6 across it, the
        // right half at 0.9.
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
        let group = blur.bind_views(&device, &src_view, &depth_view, &dst_view);
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
