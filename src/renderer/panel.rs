use glam::{Mat4, Quat, Vec3};
use wgpu::{BindGroup, Device, Queue, Sampler, Texture, TextureFormat, TextureView};

use crate::renderer::mesh::MeshVertex;
use crate::renderer::mesh_pipeline::{MeshPipeline, ModelUniform};
use crate::ui2d::{Area, Atlas, Item, Ui2dRenderer};

pub(crate) fn quad_geometry(width_m: f32, height_m: f32) -> (Vec<MeshVertex>, Vec<u32>) {
    let hw = width_m * 0.5;
    let hh = height_m * 0.5;

    let verts = vec![
        MeshVertex {
            position: [-hw, -hh, 0.0],
            normal: [0.0, 0.0, 1.0],
            uv: [0.0, 1.0],
        },
        MeshVertex {
            position: [hw, -hh, 0.0],
            normal: [0.0, 0.0, 1.0],
            uv: [1.0, 1.0],
        },
        MeshVertex {
            position: [hw, hh, 0.0],
            normal: [0.0, 0.0, 1.0],
            uv: [1.0, 0.0],
        },
        MeshVertex {
            position: [-hw, hh, 0.0],
            normal: [0.0, 0.0, 1.0],
            uv: [0.0, 0.0],
        },
    ];
    let indices = vec![0, 1, 2, 0, 2, 3];
    (verts, indices)
}

pub struct WorldPanel {
    pub position: Vec3,
    pub rotation: Quat,

    pub width_m: f32,
    pub height_m: f32,

    texture: Texture,
    view: TextureView,
    bind_group: BindGroup,
    sampler: Sampler,

    ui_renderer: Ui2dRenderer,
    atlas: Atlas,

    quad_vertices: Vec<MeshVertex>,
    quad_indices: Vec<u32>,

    pub model: ModelUniform,

    width_px: u32,
    height_px: u32,

    dirty: bool,
}

impl WorldPanel {
    pub fn new(
        device: &Device,
        _queue: &Queue,
        texture_format: TextureFormat,
        mesh_pipeline: &MeshPipeline,
        width_px: u32,
        height_px: u32,
        width_m: f32,
        height_m: f32,
    ) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("world_panel_texture"),
            size: wgpu::Extent3d {
                width: width_px,
                height: height_px,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: texture_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("world_panel_bind_group"),
            layout: &mesh_pipeline.texture_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        let model = mesh_pipeline.create_model_uniform(device);

        let ui_renderer = Ui2dRenderer::new(
            device,
            &texture_format,
            wgpu::MultisampleState::default(),
            None,
        );

        let (quad_vertices, quad_indices) = quad_geometry(width_m, height_m);

        Self {
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            width_m,
            height_m,
            texture,
            view,
            bind_group,
            sampler,
            ui_renderer,
            atlas: Atlas::default(),
            quad_vertices,
            quad_indices,
            model,
            width_px,
            height_px,
            dirty: true,
        }
    }

    pub fn upload_model(&self, queue: &Queue) {
        self.model.upload(queue, self.model_matrix());
    }

    pub fn model_matrix(&self) -> Mat4 {
        Mat4::from_rotation_translation(self.rotation, self.position)
    }

    pub fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    pub fn bind_group(&self) -> &BindGroup {
        &self.bind_group
    }

    pub fn vertices(&self) -> &[MeshVertex] {
        &self.quad_vertices
    }

    pub fn indices(&self) -> &[u32] {
        &self.quad_indices
    }

    pub fn draw(&mut self, device: &Device, queue: &Queue, items: Vec<(Area, Item)>) {
        if !self.dirty {
            return;
        }
        self.dirty = false;

        self.ui_renderer.prepare(
            device,
            queue,
            self.width_px as f32,
            self.height_px as f32,
            1.0,
            &mut self.atlas,
            items,
        );

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("world_panel_encoder"),
        });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("world_panel_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 0.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });
            self.ui_renderer.render(&mut pass);
        }

        queue.submit(Some(encoder.finish()));
    }
}
