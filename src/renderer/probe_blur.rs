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

const SAME_SURFACE_SLACK: f32 = 4e-7;

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

    textureStore(dst, p, vec4<f32>(sum.rgb / max(sum.a, 1e-6) * c0.a, c0.a));
}}
{}"#,
        probe_pass::FACE_CODE_WGSL,
    )
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

    pub fn bind_group(&self, device: &Device, target: &probe_pass::Target) -> Option<BindGroup> {
        let soft = target.soft.as_ref()?;
        Some(self.bind_views(
            device,
            &target.color_view,
            &target.depth_view,
            &soft.write_view,
            &target.reach_single,
        ))
    }

    fn bind_views(
        &self,
        device: &Device,
        colour: &wgpu::TextureView,
        depth: &wgpu::TextureView,
        out: &wgpu::TextureView,
        code: &wgpu::TextureView,
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
        let group = blur.bind_views(&device, &src_view, &depth_view, &dst_view, &code_view);
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
