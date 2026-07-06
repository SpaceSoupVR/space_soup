use wgpu::{
    CommandEncoder, DepthStencilState, Device, LoadOp, MultisampleState, Operations, Queue,
    RenderPassColorAttachment, RenderPassDescriptor, StoreOp, TextureFormat, TextureView,
};

use crate::ui2d::{Area, Atlas, Item, Ui2dRenderer};

pub struct Overlay {
    atlas: Atlas,
    renderer: Ui2dRenderer,
    width: f32,
    height: f32,
    scale_factor: f32,
    items: Vec<(Area, Item)>,
}

impl Overlay {
    pub fn new(
        device: &Device,
        texture_format: TextureFormat,
        width: u32,
        height: u32,
        scale_factor: f32,
    ) -> Self {
        let multisample = MultisampleState::default();
        let depth_stencil: Option<DepthStencilState> = None;
        let renderer = Ui2dRenderer::new(device, &texture_format, multisample, depth_stencil);

        Self {
            atlas: Atlas::default(),
            renderer,
            width: width as f32,
            height: height as f32,
            scale_factor,
            items: Vec::new(),
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.width = width as f32;
        self.height = height as f32;
    }

    pub fn set_scale_factor(&mut self, scale_factor: f32) {
        if self.scale_factor != scale_factor {
            self.scale_factor = scale_factor;
            self.atlas.trim();
        }
    }

    pub fn set_items(&mut self, items: Vec<(Area, Item)>) {
        self.items = items;
    }

    pub fn render(
        &mut self,
        device: &Device,
        queue: &Queue,
        encoder: &mut CommandEncoder,
        view: &TextureView,
    ) {
        if self.items.is_empty() {
            return;
        }

        let items = std::mem::take(&mut self.items);
        self.renderer.prepare(
            device,
            queue,
            self.width,
            self.height,
            self.scale_factor,
            &mut self.atlas,
            items,
        );

        let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("ui2d_overlay_pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Load,
                    store: StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            ..Default::default()
        });

        self.renderer.render(&mut pass);
    }
}
