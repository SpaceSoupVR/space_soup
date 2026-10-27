//! GPU tests that build the lens and read its pixels back: building a pipeline
//! only proves the WGSL parses, not that the sight picture is right.

use glam::Vec3;
use wgpu::{Device, Queue, Texture, TextureFormat};

use super::{LensParams, LensPipeline};
use crate::renderer::multiview::ViewMode;
use crate::renderer::pipeline::tests::headless_gpu;
use crate::renderer::scope::{MipBlit, ScopeQuality, ScopeRender, ScopeReticle, ScopeTarget};

const FORMAT: TextureFormat = TextureFormat::Rgba8Unorm;
const SIZE: u32 = 64;

/// A pipeline that paints a 1-texel checkerboard over its whole target.
pub(crate) fn checker_pipeline(device: &Device, format: TextureFormat) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("checker"),
        source: wgpu::ShaderSource::Wgsl(
            r"
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let x = f32((i << 1u) & 2u);
    let y = f32(i & 2u);
    return vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
}
@fragment fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let v = f32((u32(pos.x) + u32(pos.y)) % 2u);
    return vec4<f32>(v, v, v, 1.0);
}"
            .into(),
        ),
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("checker"),
        layout: None,
        vertex: wgpu::VertexState { module: &shader, entry_point: Some("vs"), buffers: &[], compilation_options: Default::default() },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs"),
            targets: &[Some(wgpu::ColorTargetState { format, blend: None, write_mask: wgpu::ColorWrites::ALL })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}

/// The red channel of every texel of one mip level.
pub(crate) fn read_level(device: &Device, queue: &Queue, texture: &Texture, level: u32) -> Vec<f64> {
    read_rgba(device, queue, texture, level).chunks_exact(4).map(|p| f64::from(p[0])).collect()
}

fn read_rgba(device: &Device, queue: &Queue, texture: &Texture, level: u32) -> Vec<u8> {
    let dim = texture.width() >> level;
    let bpr = (dim * 4).div_ceil(256) * 256;
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("scope_readback"),
        size: u64::from(bpr * dim),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo { texture, mip_level: level, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(bpr), rows_per_image: Some(dim) },
        },
        wgpu::Extent3d { width: dim, height: dim, depth_or_array_layers: 1 },
    );
    queue.submit(Some(encoder.finish()));
    let slice = buffer.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).ok();
    let mapped = slice.get_mapped_range().unwrap();
    let rows: Vec<u8> = mapped.chunks(bpr as usize).flat_map(|row| row[..(dim * 4) as usize].to_vec()).collect();
    drop(mapped);
    buffer.unmap();
    rows
}

fn camera(device: &Device) -> (wgpu::BindGroupLayout, wgpu::BindGroup) {
    use wgpu::util::DeviceExt;
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        }],
    });
    // Identity: world positions are clip positions.
    let identity = [glam::Mat4::IDENTITY.to_cols_array(); 2];
    let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&identity),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &layout,
        entries: &[wgpu::BindGroupEntry { binding: 0, resource: buffer.as_entire_binding() }],
    });
    (layout, group)
}

/// A lens in clip space: an eye at z = -1 looking down +z at an ocular of
/// radius 0.4 at z = 0.5.
fn scope(magnification: f32, style: u32, emission: [f32; 3]) -> ScopeRender {
    ScopeRender {
        objective: Vec3::new(0.0, 0.0, 0.7),
        ocular: Vec3::new(0.0, 0.0, 0.5),
        ocular_radius_m: 0.4,
        magnification,
        true_fov_y_rad: 0.5 / magnification,
        occupancy: [1.0; 2],
        shadow_darkens: true,
        edge_feather_px: 1.0,
        reticle: ScopeReticle { style, emission, scale: 1.0 },
        quality: ScopeQuality { max_size: 256, supersample: 1.0 },
    }
}

fn eye() -> Vec3 {
    Vec3::new(0.0, 0.0, -1.0)
}

/// Composites `scope` over a black frame, its picture filled with `fill`, and
/// returns the frame's RGBA bytes.
fn composite(device: &Device, queue: &Queue, scope: &ScopeRender, fill: wgpu::Color) -> Vec<u8> {
    use wgpu::util::DeviceExt;
    let (camera_layout, camera_group) = camera(device);
    let lens = LensPipeline::new(device, FORMAT, &camera_layout, 1, ViewMode::Mono);
    let blit = MipBlit::new(device, FORMAT);
    let picture = ScopeTarget::new(device, FORMAT, 128, 1, &blit);
    let frame = ScopeTarget::new(device, FORMAT, SIZE, 1, &blit);
    let (params, params_group) = lens.params_group(device);
    queue.write_buffer(&params, 0, bytemuck::bytes_of(&LensParams::for_scope(scope, [eye(); 2], 0)));
    let quad = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&scope.lens_quad()),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let texture_group = lens.target_group(device, &picture.sample_view);
    let clear = |view| wgpu::RenderPassColorAttachment {
        view,
        depth_slice: None,
        resolve_target: None,
        ops: wgpu::Operations { load: wgpu::LoadOp::Clear(fill), store: wgpu::StoreOp::Store },
    };
    let mut encoder = device.create_command_encoder(&Default::default());
    drop(encoder.begin_render_pass(&wgpu::RenderPassDescriptor { color_attachments: &[Some(clear(&picture.color_view))], ..Default::default() }));
    picture.generate_mips(&mut encoder, &blit);
    {
        let black = wgpu::RenderPassColorAttachment { ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store }, ..clear(&frame.color_view) };
        let depth = wgpu::RenderPassDepthStencilAttachment { depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }), ..frame.depth_attachment() };
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            color_attachments: &[Some(black)],
            depth_stencil_attachment: Some(depth),
            ..Default::default()
        });
        pass.set_pipeline(&lens.pipeline);
        pass.set_bind_group(0, &camera_group, &[]);
        pass.set_bind_group(1, &texture_group, &[]);
        pass.set_bind_group(2, &params_group, &[]);
        pass.set_vertex_buffer(0, quad.slice(..));
        pass.draw(0..6, 0..1);
    }
    queue.submit(Some(encoder.finish()));
    read_rgba(device, queue, &frame.color, 0)
}

fn pixel(rgba: &[u8], x: u32, y: u32) -> [u8; 4] {
    let i = ((y * SIZE + x) * 4) as usize;
    [rgba[i], rgba[i + 1], rgba[i + 2], rgba[i + 3]]
}

const GREEN: wgpu::Color = wgpu::Color { r: 0.0, g: 1.0, b: 0.0, a: 1.0 };

#[test]
fn the_lens_builds_mono_and_multisampled() {
    let Some((device, _queue)) = headless_gpu() else { return };
    let (layout, _) = camera(&device);
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let _mono = LensPipeline::new(&device, FORMAT, &layout, 1, ViewMode::Mono);
    let _msaa = LensPipeline::new(&device, FORMAT, &layout, 4, ViewMode::Mono);
    let err = pollster::block_on(scope.pop());
    assert!(err.is_none(), "lens pipeline refused: {err:?}");
}

#[test]
fn the_glass_shows_the_scope_picture_and_nothing_outside_it() {
    let Some((device, queue)) = headless_gpu() else { return };
    let rgba = composite(&device, &queue, &scope(4.0, 0, [0.0; 3]), GREEN);
    let centre = pixel(&rgba, SIZE / 2 + 4, SIZE / 2 + 4);
    assert!(centre[1] > 200 && centre[0] < 30, "the disc shows the picture: {centre:?}");
    assert_eq!(pixel(&rgba, 1, 1), [0, 0, 0, 255], "outside the disc the frame is untouched");
}

#[test]
fn a_crosshair_is_drawn_black_across_the_centre() {
    let Some((device, queue)) = headless_gpu() else { return };
    let rgba = composite(&device, &queue, &scope(4.0, 3, [0.0; 3]), GREEN);
    let on_line = pixel(&rgba, SIZE / 2 + 6, SIZE / 2);
    let off_line = pixel(&rgba, SIZE / 2 + 6, SIZE / 2 + 6);
    assert!(on_line[1] < off_line[1] / 2, "the horizontal line darkens its row: {on_line:?} vs {off_line:?}");
}

#[test]
fn a_window_draws_only_its_lit_dot_over_the_eyes_view() {
    let Some((device, queue)) = headless_gpu() else { return };
    let mut red_dot = scope(1.0, 1, [1.0, 0.0, 0.0]);
    red_dot.shadow_darkens = false;
    let rgba = composite(&device, &queue, &red_dot, GREEN);
    let dot = pixel(&rgba, SIZE / 2, SIZE / 2);
    assert!(dot[0] > 80 && dot[1] < 30, "the dot is lit red over the view: {dot:?}");
    let glass = pixel(&rgba, SIZE / 2 + 8, SIZE / 2 + 8);
    assert!(glass[0] < 10 && glass[1] < 10, "a window never draws the scope picture: {glass:?}");
}

#[test]
fn misalignment_shadows_the_far_side_of_the_glass() {
    let Some((device, queue)) = headless_gpu() else { return };
    // The optic sits a little left of the eye, so the eye has moved right of
    // its axis: the left of the glass goes dark.
    let mut s = scope(4.0, 0, [0.0; 3]);
    s.occupancy = [0.5; 2];
    s.ocular.x -= 0.01;
    s.objective.x -= 0.01;
    let params = LensParams::for_scope(&s, [eye(); 2], 0);
    // Looking down +z in clip space the optic's right is -x.
    assert!(params.offsets[0][0] < -0.99, "the eye's offset reaches the lens: {:?}", params.offsets[0]);
    let rgba = composite(&device, &queue, &s, GREEN);
    let (left, right) = (pixel(&rgba, SIZE / 2 - 9, SIZE / 2 + 2), pixel(&rgba, SIZE / 2 + 9, SIZE / 2 + 2));
    assert!(right[1] > left[1] + 60, "shadow falls opposite the eye's move: left {left:?} right {right:?}");
}
