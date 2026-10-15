//! The frame-rate readout's quad layer: its own small swapchain, written by
//! upload when the numbers change and handed to the compositor beside the eye
//! layer every frame while it is on. What it shows: `renderer::fps_hud`.

use ash::vk::{self, Handle};
use openxr as xr;

use crate::renderer::fps_hud;

pub(super) struct FpsLayer {
    swapchain: xr::Swapchain<xr::Vulkan>,
    images: Vec<wgpu::Texture>,
    meter: fps_hud::FpsMeter,
    /// Whether an image has been released yet: until then there is nothing
    /// to submit.
    ready: bool,
}

impl FpsLayer {
    pub(super) fn new(session: &xr::Session<xr::Vulkan>, device: &wgpu::Device) -> Result<Self, Box<dyn std::error::Error>> {
        let swapchain = session.create_swapchain(&xr::SwapchainCreateInfo {
            create_flags: xr::SwapchainCreateFlags::EMPTY,
            // TRANSFER_DST: the picture is uploaded, not drawn. COLOR_ATTACHMENT:
            // the image is handed back in the layout the runtime expects by an
            // empty pass (see `update`). SAMPLED: the compositor reads it.
            usage_flags: xr::SwapchainUsageFlags::TRANSFER_DST
                | xr::SwapchainUsageFlags::COLOR_ATTACHMENT
                | xr::SwapchainUsageFlags::SAMPLED,
            format: vk::Format::R8G8B8A8_SRGB.as_raw() as _,
            sample_count: 1,
            width: fps_hud::WIDTH_PX,
            height: fps_hud::HEIGHT_PX,
            face_count: 1,
            array_size: 1,
            mip_count: 1,
        })?;
        let images = swapchain
            .enumerate_images()?
            .into_iter()
            .map(|raw| unsafe {
                super::vulkan_interop::import_vk_image_as_wgpu_with(
                    device,
                    vk::Image::from_raw(raw),
                    wgpu::TextureFormat::Rgba8UnormSrgb,
                    (fps_hud::WIDTH_PX, fps_hud::HEIGHT_PX, 1),
                    wgpu::TextureUses::COPY_DST | wgpu::TextureUses::COLOR_TARGET,
                    wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::RENDER_ATTACHMENT,
                )
            })
            .collect();
        Ok(Self { swapchain, images, meter: Default::default(), ready: false })
    }

    /// One frame made; on the frame that closes a window, the picture is
    /// redrawn into the next swapchain image.
    pub(super) fn frame(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        gpu_ms: Option<f64>,
        cpu_ms: Option<f64>,
        target_fps: f64,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let Some(reading) = self.meter.frame(std::time::Instant::now(), gpu_ms, cpu_ms, target_fps) else {
            return Ok(());
        };
        let index = self.swapchain.acquire_image()? as usize;
        self.swapchain.wait_image(xr::Duration::INFINITE)?;
        let texture = &self.images[index];
        queue.write_texture(
            wgpu::TexelCopyTextureInfo { texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            &fps_hud::picture(&fps_hud::lines(&reading)),
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(fps_hud::WIDTH_PX * 4), rows_per_image: None },
            wgpu::Extent3d { width: fps_hud::WIDTH_PX, height: fps_hud::HEIGHT_PX, depth_or_array_layers: 1 },
        );
        // An EMPTY pass that keeps what the upload wrote: it leaves the image
        // in COLOR_ATTACHMENT_OPTIMAL, the layout a colour swapchain image
        // must be released in. The upload alone would leave TRANSFER_DST.
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("fps_layer") });
        drop(encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("fps_layer_hand_back"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
            })],
            ..Default::default()
        }));
        queue.submit(Some(encoder.finish()));
        self.swapchain.release_image()?;
        self.ready = true;
        Ok(())
    }

    /// The layer to submit after the eyes', in `view_space` (the head's), or
    /// `None` before its first picture.
    pub(super) fn layer<'a>(&'a self, view_space: &'a xr::Space) -> Option<xr::CompositionLayerQuad<'a, xr::Vulkan>> {
        if !self.ready {
            return None;
        }
        let [x, y, z] = fps_hud::POSITION;
        Some(
            xr::CompositionLayerQuad::new()
                .layer_flags(xr::CompositionLayerFlags::BLEND_TEXTURE_SOURCE_ALPHA)
                .space(view_space)
                .eye_visibility(xr::EyeVisibility::BOTH)
                .sub_image(
                    xr::SwapchainSubImage::new()
                        .swapchain(&self.swapchain)
                        .image_array_index(0)
                        .image_rect(xr::Rect2Di {
                            offset: xr::Offset2Di { x: 0, y: 0 },
                            extent: xr::Extent2Di { width: fps_hud::WIDTH_PX as i32, height: fps_hud::HEIGHT_PX as i32 },
                        }),
                )
                .pose(xr::Posef { orientation: xr::Quaternionf::IDENTITY, position: xr::Vector3f { x, y, z } })
                .size(xr::Extent2Df { width: fps_hud::WIDTH_M, height: fps_hud::HEIGHT_M }),
        )
    }
}
