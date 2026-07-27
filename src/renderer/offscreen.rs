//! Windowless render target + GPU readback, for rendering frames without a
//! surface: the CloudXR server render loop (render a `GameRuntime` scene on the
//! DigitalOcean GPU droplet, hand the RGBA frame to the video encoder), plus
//! editor thumbnails, tools, and tests.
//!
//! Handles the 256-byte `bytes_per_row` copy alignment so any width works and
//! the returned buffer is tightly packed RGBA (row padding stripped).

use wgpu::*;

pub struct OffscreenTarget {
    color: Texture,
    view: TextureView,
    buffer: Buffer,
    width: u32,
    height: u32,
    padded_bpr: u32,
}

impl OffscreenTarget {
    /// Creates an `Rgba8UnormSrgb` color target of `width`x`height` plus a
    /// mappable readback buffer sized for the padded row stride.
    pub fn new(device: &Device, width: u32, height: u32) -> Self {
        let color = device.create_texture(&TextureDescriptor {
            label: Some("offscreen_color"),
            size: Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8UnormSrgb,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = color.create_view(&TextureViewDescriptor::default());

        let unpadded_bpr = width * 4;
        let align = COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_bpr = unpadded_bpr.div_ceil(align) * align;

        let buffer = device.create_buffer(&BufferDescriptor {
            label: Some("offscreen_readback"),
            size: (padded_bpr * height) as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            color,
            view,
            buffer,
            width,
            height,
            padded_bpr,
        }
    }

    /// The color view to pass to `Renderer::render_*` as the render target.
    pub fn view(&self) -> &TextureView {
        &self.view
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    /// Copies the last-rendered color into the readback buffer and returns it
    /// as tightly packed RGBA (`width*height*4` bytes, row padding removed).
    /// Blocks until the copy completes.
    pub fn read_rgba(&self, device: &Device, queue: &Queue) -> Vec<u8> {
        let mut encoder =
            device.create_command_encoder(&CommandEncoderDescriptor { label: Some("offscreen_copy") });
        encoder.copy_texture_to_buffer(
            TexelCopyTextureInfo {
                texture: &self.color,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            TexelCopyBufferInfo {
                buffer: &self.buffer,
                layout: TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(self.padded_bpr),
                    rows_per_image: Some(self.height),
                },
            },
            Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(Some(encoder.finish()));

        let slice = self.buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = device.poll(PollType::Wait);
        rx.recv().unwrap().expect("map offscreen readback buffer");

        let data = slice.get_mapped_range();
        let row_bytes = (self.width * 4) as usize;
        let mut out = Vec::with_capacity(row_bytes * self.height as usize);
        for row in data.chunks(self.padded_bpr as usize) {
            out.extend_from_slice(&row[..row_bytes]);
        }
        drop(data);
        self.buffer.unmap();
        out
    }
}
