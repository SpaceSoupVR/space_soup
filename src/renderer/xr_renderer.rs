use ash::vk::{self, Handle};
use log::info;
use openxr as xr;

use crate::renderer::{
    camera::Camera,
    cuboid::{build_solid_mesh, build_wire_mesh, Cuboid},
    lights::{Light, LightsUniform},
    mesh_pipeline::{MeshPipeline, ModelUniform, SkinnedMeshPipeline},
    mirror::{self, MirrorPipeline, MirrorSurface, MirrorTarget},
    pipeline::{SolidPipeline, WirePipeline},
    uniforms::UniformBuffer,
    MeshInstance,
};
use crate::xr::{VkContext, XrContext};
use wgpu::util::DeviceExt;

struct EyeTarget {
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
}

type MeshDraw<'a> = (
    &'a wgpu::BindGroup,
    &'a wgpu::BindGroup,
    &'a wgpu::Buffer,
    &'a wgpu::Buffer,
    u32,
);

type SkinnedDraw<'a> = (
    &'a wgpu::BindGroup,
    &'a wgpu::BindGroup,
    &'a wgpu::BindGroup,
    &'a wgpu::Buffer,
    &'a wgpu::Buffer,
    u32,
);

fn push_mesh_draws<'a>(
    instance: &'a MeshInstance,
    mesh_draws: &mut Vec<MeshDraw<'a>>,
    double_sided_draws: &mut Vec<MeshDraw<'a>>,
    skinned_draws: &mut Vec<SkinnedDraw<'a>>,
) {
    if let Some(skin) = &instance.mesh.skin {
        if let Some(joint_bg) = &skin.joint_bind_group {
            for prim in &skin.primitives {
                skinned_draws.push((
                    &instance.model.bind_group,
                    &prim.texture.bind_group,
                    joint_bg,
                    &prim.vertex_buffer,
                    &prim.index_buffer,
                    prim.index_count,
                ));
            }
        }
    } else {
        for prim in &instance.mesh.primitives {
            let target = if prim.double_sided {
                &mut *double_sided_draws
            } else {
                &mut *mesh_draws
            };
            target.push((
                &instance.model.bind_group,
                &prim.texture.bind_group,
                &prim.vertex_buffer,
                &prim.index_buffer,
                prim.indices.len() as u32,
            ));
        }
    }
}

pub struct XrRenderer {
    pub swapchain: xr::Swapchain<xr::Vulkan>,
    pub width: u32,
    pub height: u32,
    wgpu_device: wgpu::Device,
    wgpu_queue: wgpu::Queue,
    solid_pipeline: SolidPipeline,
    wire_pipeline: WirePipeline,
    mesh_pipeline: MeshPipeline,
    skinned_mesh_pipeline: SkinnedMeshPipeline,
    // Mirror-variant pipelines (flipped winding — see MeshPipeline::new_mirror)
    // used only when rendering into the mirror's own offscreen texture.
    mirror_solid_pipeline: SolidPipeline,
    mirror_mesh_pipeline: MeshPipeline,
    mirror_pipeline: MirrorPipeline,
    mirror_targets: [MirrorTarget; 2],
    mirror_model_uniform: ModelUniform,
    mirror_reflected_vp_uniform: ModelUniform,
    uniform_buf: UniformBuffer,
    lights_uniform: LightsUniform,
    depth_view: wgpu::TextureView,
    eye_targets: Vec<[EyeTarget; 2]>,
}

impl XrRenderer {
    pub fn new(
        vk: &VkContext,
        xr_ctx: &XrContext,
        session: &xr::Session<xr::Vulkan>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let view_configs = xr_ctx.instance.enumerate_view_configuration_views(
            xr_ctx.system,
            xr::ViewConfigurationType::PRIMARY_STEREO,
        )?;
        let width = view_configs[0].recommended_image_rect_width;
        let height = view_configs[0].recommended_image_rect_height;
        info!("XrRenderer: {width}x{height}");

        let vk_format = vk::Format::R8G8B8A8_SRGB;
        let wgpu_format = wgpu::TextureFormat::Rgba8UnormSrgb;

        let swapchain = session.create_swapchain(&xr::SwapchainCreateInfo {
            create_flags: xr::SwapchainCreateFlags::EMPTY,
            usage_flags: xr::SwapchainUsageFlags::COLOR_ATTACHMENT
                | xr::SwapchainUsageFlags::SAMPLED,
            format: vk_format.as_raw() as _,
            sample_count: 1,
            width,
            height,
            face_count: 1,
            array_size: 2,
            mip_count: 1,
        })?;

        let raw_images: Vec<vk::Image> = swapchain
            .enumerate_images()?
            .into_iter()
            .map(vk::Image::from_raw)
            .collect();
        info!("XrRenderer: {} swapchain images", raw_images.len());

        let (wgpu_device, wgpu_queue) = unsafe { build_wgpu_from_vulkan(vk)? };

        let lights_uniform = LightsUniform::new(&wgpu_device);
        let uniform_buf = UniformBuffer::new(&wgpu_device, &lights_uniform);
        let solid_pipeline = SolidPipeline::new(&wgpu_device, wgpu_format, &uniform_buf.layout);
        let wire_pipeline = WirePipeline::new(&wgpu_device, wgpu_format, &uniform_buf.layout);
        let mesh_pipeline = MeshPipeline::new(&wgpu_device, wgpu_format, &uniform_buf.layout);
        let skinned_mesh_pipeline =
            SkinnedMeshPipeline::new(&wgpu_device, wgpu_format, &uniform_buf.layout);
        let mirror_solid_pipeline =
            SolidPipeline::new_mirror(&wgpu_device, wgpu_format, &uniform_buf.layout);
        let mirror_mesh_pipeline =
            MeshPipeline::new_mirror(&wgpu_device, wgpu_format, &uniform_buf.layout);
        let mirror_pipeline = MirrorPipeline::new(&wgpu_device, wgpu_format, &uniform_buf.layout);
        let mirror_targets: [MirrorTarget; 2] =
            std::array::from_fn(|_| mirror_pipeline.create_target(&wgpu_device, wgpu_format, width, height));
        let mirror_model_uniform = mirror_pipeline.create_model_uniform(&wgpu_device);
        let mirror_reflected_vp_uniform = mirror_pipeline.create_reflected_vp_uniform(&wgpu_device);

        let depth_tex = wgpu_device.create_texture(&wgpu::TextureDescriptor {
            label: Some("xr_depth"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth_view = depth_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let mut eye_targets: Vec<[EyeTarget; 2]> = Vec::new();
        for &raw_image in &raw_images {
            let targets = std::array::from_fn(|eye| {
                let wgpu_tex = unsafe {
                    import_vk_image_as_wgpu(&wgpu_device, raw_image, wgpu_format, width, height, 2)
                };
                let view = wgpu_tex.create_view(&wgpu::TextureViewDescriptor {
                    format: Some(wgpu_format),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_array_layer: eye as u32,
                    array_layer_count: Some(1),
                    ..Default::default()
                });
                EyeTarget {
                    _texture: wgpu_tex,
                    view,
                }
            });
            eye_targets.push(targets);
        }

        Ok(Self {
            swapchain,
            width,
            height,
            wgpu_device,
            wgpu_queue,
            solid_pipeline,
            wire_pipeline,
            mesh_pipeline,
            skinned_mesh_pipeline,
            mirror_solid_pipeline,
            mirror_mesh_pipeline,
            mirror_pipeline,
            mirror_targets,
            mirror_model_uniform,
            mirror_reflected_vp_uniform,
            uniform_buf,
            lights_uniform,
            depth_view,
            eye_targets,
        })
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.wgpu_device
    }
    pub fn queue(&self) -> &wgpu::Queue {
        &self.wgpu_queue
    }
    pub fn mesh_texture_layout(&self) -> &wgpu::BindGroupLayout {
        &self.mesh_pipeline.texture_layout
    }
    pub fn skin_joint_layout(&self) -> &wgpu::BindGroupLayout {
        &self.skinned_mesh_pipeline.skin_joint_layout
    }
    pub fn create_model_uniform(&self) -> crate::renderer::mesh_pipeline::ModelUniform {
        self.mesh_pipeline.create_model_uniform(&self.wgpu_device)
    }
    pub fn create_skinned_model_uniform(&self) -> crate::renderer::mesh_pipeline::ModelUniform {
        self.skinned_mesh_pipeline
            .create_model_uniform(&self.wgpu_device)
    }

    pub fn render_frame(
        &mut self,
        session: &xr::Session<xr::Vulkan>,
        stage: &xr::Space,
        time: xr::Time,
        cuboids: &[Cuboid],
    ) -> Result<Vec<xr::CompositionLayerProjectionView<xr::Vulkan>>, Box<dyn std::error::Error>>
    {
        self.render_frame_with_meshes(session, stage, time, cuboids, &[], &[], &[], None)
    }

    pub fn render_frame_with_meshes(
        &mut self,
        session: &xr::Session<xr::Vulkan>,
        stage: &xr::Space,
        time: xr::Time,
        cuboids: &[Cuboid],
        meshes: &[MeshInstance],
        mirror_only_meshes: &[MeshInstance],
        lights: &[Light],
        mirror: Option<MirrorSurface>,
    ) -> Result<Vec<xr::CompositionLayerProjectionView<xr::Vulkan>>, Box<dyn std::error::Error>>
    {
        let image_index = self.swapchain.acquire_image()? as usize;
        self.swapchain.wait_image(xr::Duration::INFINITE)?;
        self.lights_uniform.upload(&self.wgpu_queue, lights);

        let (_, eye_views) =
            session.locate_views(xr::ViewConfigurationType::PRIMARY_STEREO, time, stage)?;

        let (solid_verts, solid_idx) = build_solid_mesh(cuboids);
        let (wire_verts, wire_idx) = build_wire_mesh(cuboids);

        let solid_vb = self
            .wgpu_device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("solid_vb"),
                contents: bytemuck::cast_slice(&solid_verts),
                usage: wgpu::BufferUsages::VERTEX,
            });
        let solid_ib = self
            .wgpu_device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("solid_ib"),
                contents: bytemuck::cast_slice(&solid_idx),
                usage: wgpu::BufferUsages::INDEX,
            });
        let wire_vb = self
            .wgpu_device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("wire_vb"),
                contents: bytemuck::cast_slice(&wire_verts),
                usage: wgpu::BufferUsages::VERTEX,
            });
        let wire_ib = self
            .wgpu_device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("wire_ib"),
                contents: bytemuck::cast_slice(&wire_idx),
                usage: wgpu::BufferUsages::INDEX,
            });

        let mut mesh_draws: Vec<MeshDraw> = Vec::new();
        let mut double_sided_draws: Vec<MeshDraw> = Vec::new();
        let mut skinned_draws: Vec<SkinnedDraw> = Vec::new();
        for instance in meshes {
            instance
                .model
                .upload(&self.wgpu_queue, instance.mesh.model_matrix());
            push_mesh_draws(
                instance,
                &mut mesh_draws,
                &mut double_sided_draws,
                &mut skinned_draws,
            );
        }

        // Extra content that only shows up in the mirror's reflection, not
        // the direct view — the local player's own avatar body, so you
        // don't see your own untracked torso/legs hanging in your face, but
        // it's still there for a mirror (or another player) to actually see.
        let mut mirror_only_mesh_draws: Vec<MeshDraw> = Vec::new();
        let mut mirror_only_double_sided_draws: Vec<MeshDraw> = Vec::new();
        let mut mirror_only_skinned_draws: Vec<SkinnedDraw> = Vec::new();
        for instance in mirror_only_meshes {
            instance
                .model
                .upload(&self.wgpu_queue, instance.mesh.model_matrix());
            push_mesh_draws(
                instance,
                &mut mirror_only_mesh_draws,
                &mut mirror_only_double_sided_draws,
                &mut mirror_only_skinned_draws,
            );
        }

        // Mirror quad geometry/model matrix don't depend on the eye — built
        // once here, reused for both eyes' main-pass draw below.
        let mirror_quad = mirror.map(|m| {
            let (verts, idx) = mirror::build_mirror_quad(m.half_size.x, m.half_size.y);
            let vb = self
                .wgpu_device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("mirror_quad_vb"),
                    contents: bytemuck::cast_slice(&verts),
                    usage: wgpu::BufferUsages::VERTEX,
                });
            let ib = self
                .wgpu_device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("mirror_quad_ib"),
                    contents: bytemuck::cast_slice(&idx),
                    usage: wgpu::BufferUsages::INDEX,
                });
            let model = glam::Mat4::from_rotation_translation(m.rotation, m.position);
            self.mirror_model_uniform.upload(&self.wgpu_queue, model);
            (vb, ib, idx.len() as u32)
        });

        for eye in 0..2usize {
            let ev = &eye_views[eye];
            let view = Camera::xr_view(ev.pose);
            let proj = Camera::xr_projection(ev.fov, 0.01, 1000.0);

            // --- Mirror pass: render the whole scene again from the
            // reflected viewpoint into this eye's offscreen mirror texture,
            // *before* the main pass overwrites the shared camera uniform
            // with the real (unreflected) view-projection it needs.
            if let Some(m) = &mirror {
                let reflect = mirror::reflection_matrix(m.position, m.normal());
                let mirror_view = view * reflect;

                // Bend the near clip plane to exactly coincide with the
                // mirror surface (Lengyel's oblique near-plane clipping),
                // so anything on the wrong side — between the reflected
                // camera and the mirror — structurally can't render into
                // it, regardless of what geometry happens to be nearby.
                let world_plane = mirror::world_plane_equation(m.position, m.normal());
                let eye_plane = mirror::plane_to_eye_space(mirror_view.inverse(), world_plane);
                let mirror_proj = mirror::oblique_near_clip(proj, eye_plane);
                let mirror_view_proj = mirror_proj * mirror_view;

                self.uniform_buf.upload(&self.wgpu_queue, mirror_view_proj);
                // Same matrix, stashed for the mirror quad's own vertex
                // shader to use later in this eye's main pass — that's what
                // makes the reflection UV pixel-accurate instead of an
                // approximation from this quad's real-camera screen position.
                self.mirror_reflected_vp_uniform.upload(&self.wgpu_queue, mirror_view_proj);

                let mut encoder = self.wgpu_device.create_command_encoder(
                    &wgpu::CommandEncoderDescriptor { label: Some("mirror_eye") },
                );
                {
                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("mirror_pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &self.mirror_targets[eye].color_view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color {
                                    r: 0.02,
                                    g: 0.02,
                                    b: 0.05,
                                    a: 1.0,
                                }),
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: &self.mirror_targets[eye].depth_view,
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Clear(1.0),
                                store: wgpu::StoreOp::Store,
                            }),
                            stencil_ops: None,
                        }),
                        ..Default::default()
                    });

                    if !solid_verts.is_empty() {
                        pass.set_pipeline(&self.mirror_solid_pipeline.pipeline);
                        pass.set_bind_group(0, &self.uniform_buf.bind_group, &[]);
                        pass.set_vertex_buffer(0, solid_vb.slice(..));
                        pass.set_index_buffer(solid_ib.slice(..), wgpu::IndexFormat::Uint32);
                        pass.draw_indexed(0..solid_idx.len() as u32, 0, 0..1);
                    }
                    let all_mesh_draws = mesh_draws.iter().chain(mirror_only_mesh_draws.iter());
                    let all_double_sided_draws = double_sided_draws
                        .iter()
                        .chain(mirror_only_double_sided_draws.iter());
                    let all_skinned_draws =
                        skinned_draws.iter().chain(mirror_only_skinned_draws.iter());

                    if !double_sided_draws.is_empty() || !mirror_only_double_sided_draws.is_empty() {
                        pass.set_pipeline(&self.mirror_mesh_pipeline.pipeline_double_sided);
                        pass.set_bind_group(0, &self.uniform_buf.bind_group, &[]);
                        for (model_bg, tex_bg, vb, ib, count) in all_double_sided_draws {
                            pass.set_bind_group(1, *model_bg, &[]);
                            pass.set_bind_group(2, *tex_bg, &[]);
                            pass.set_vertex_buffer(0, vb.slice(..));
                            pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                            pass.draw_indexed(0..*count, 0, 0..1);
                        }
                    }

                    if !mesh_draws.is_empty() || !mirror_only_mesh_draws.is_empty() {
                        pass.set_pipeline(&self.mirror_mesh_pipeline.pipeline);
                        pass.set_bind_group(0, &self.uniform_buf.bind_group, &[]);
                        for (model_bg, tex_bg, vb, ib, count) in all_mesh_draws {
                            pass.set_bind_group(1, *model_bg, &[]);
                            pass.set_bind_group(2, *tex_bg, &[]);
                            pass.set_vertex_buffer(0, vb.slice(..));
                            pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                            pass.draw_indexed(0..*count, 0, 0..1);
                        }
                    }
                    if !skinned_draws.is_empty() || !mirror_only_skinned_draws.is_empty() {
                        pass.set_pipeline(&self.skinned_mesh_pipeline.pipeline);
                        pass.set_bind_group(0, &self.uniform_buf.bind_group, &[]);
                        for (model_bg, tex_bg, joint_bg, vb, ib, count) in all_skinned_draws {
                            pass.set_bind_group(1, *model_bg, &[]);
                            pass.set_bind_group(2, *tex_bg, &[]);
                            pass.set_bind_group(3, *joint_bg, &[]);
                            pass.set_vertex_buffer(0, vb.slice(..));
                            pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                            pass.draw_indexed(0..*count, 0, 0..1);
                        }
                    }
                    // Wireframe boxes deliberately excluded from the mirror
                    // reflection — thin unlit lines add little and this
                    // keeps the extra pass cheaper.
                }
                self.wgpu_queue.submit(Some(encoder.finish()));
            }

            self.uniform_buf.upload(&self.wgpu_queue, proj * view);

            let color_view = &self.eye_targets[image_index][eye].view;
            let mut encoder = self
                .wgpu_device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("eye") });

            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("eye_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: color_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.02,
                                g: 0.02,
                                b: 0.05,
                                a: 1.0,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &self.depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    ..Default::default()
                });

                if !solid_verts.is_empty() {
                    pass.set_pipeline(&self.solid_pipeline.pipeline);
                    pass.set_bind_group(0, &self.uniform_buf.bind_group, &[]);
                    pass.set_vertex_buffer(0, solid_vb.slice(..));
                    pass.set_index_buffer(solid_ib.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(0..solid_idx.len() as u32, 0, 0..1);
                }
                if !wire_verts.is_empty() {
                    pass.set_pipeline(&self.wire_pipeline.pipeline);
                    pass.set_bind_group(0, &self.uniform_buf.bind_group, &[]);
                    pass.set_vertex_buffer(0, wire_vb.slice(..));
                    pass.set_index_buffer(wire_ib.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(0..wire_idx.len() as u32, 0, 0..1);
                }
                if !mesh_draws.is_empty() {
                    pass.set_pipeline(&self.mesh_pipeline.pipeline);
                    pass.set_bind_group(0, &self.uniform_buf.bind_group, &[]);
                    for (model_bg, tex_bg, vb, ib, count) in &mesh_draws {
                        pass.set_bind_group(1, *model_bg, &[]);
                        pass.set_bind_group(2, *tex_bg, &[]);
                        pass.set_vertex_buffer(0, vb.slice(..));
                        pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                        pass.draw_indexed(0..*count, 0, 0..1);
                    }
                }
                if !double_sided_draws.is_empty() {
                    pass.set_pipeline(&self.mesh_pipeline.pipeline_double_sided);
                    pass.set_bind_group(0, &self.uniform_buf.bind_group, &[]);
                    for (model_bg, tex_bg, vb, ib, count) in &double_sided_draws {
                        pass.set_bind_group(1, *model_bg, &[]);
                        pass.set_bind_group(2, *tex_bg, &[]);
                        pass.set_vertex_buffer(0, vb.slice(..));
                        pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                        pass.draw_indexed(0..*count, 0, 0..1);
                    }
                }
                if !skinned_draws.is_empty() {
                    pass.set_pipeline(&self.skinned_mesh_pipeline.pipeline);
                    pass.set_bind_group(0, &self.uniform_buf.bind_group, &[]);
                    for (model_bg, tex_bg, joint_bg, vb, ib, count) in &skinned_draws {
                        pass.set_bind_group(1, *model_bg, &[]);
                        pass.set_bind_group(2, *tex_bg, &[]);
                        pass.set_bind_group(3, *joint_bg, &[]);
                        pass.set_vertex_buffer(0, vb.slice(..));
                        pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                        pass.draw_indexed(0..*count, 0, 0..1);
                    }
                }
                if let Some((vb, ib, count)) = &mirror_quad {
                    pass.set_pipeline(&self.mirror_pipeline.pipeline);
                    pass.set_bind_group(0, &self.uniform_buf.bind_group, &[]);
                    pass.set_bind_group(1, &self.mirror_model_uniform.bind_group, &[]);
                    pass.set_bind_group(2, &self.mirror_targets[eye].texture_bind_group, &[]);
                    pass.set_bind_group(3, &self.mirror_reflected_vp_uniform.bind_group, &[]);
                    pass.set_vertex_buffer(0, vb.slice(..));
                    pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(0..*count, 0, 0..1);
                }
            }

            self.wgpu_queue.submit(Some(encoder.finish()));
        }

        self.wgpu_device.poll(wgpu::PollType::Wait);
        self.swapchain.release_image()?;

        let proj_views = eye_views
            .iter()
            .enumerate()
            .map(|(i, ev)| {
                xr::CompositionLayerProjectionView::new()
                    .pose(ev.pose)
                    .fov(ev.fov)
                    .sub_image(
                        xr::SwapchainSubImage::new()
                            .swapchain(&self.swapchain)
                            .image_array_index(i as u32)
                            .image_rect(xr::Rect2Di {
                                offset: xr::Offset2Di { x: 0, y: 0 },
                                extent: xr::Extent2Di {
                                    width: self.width as i32,
                                    height: self.height as i32,
                                },
                            }),
                    )
            })
            .collect();

        Ok(proj_views)
    }

    pub fn cleanup(&self) {}
}

unsafe fn build_wgpu_from_vulkan(
    vk: &VkContext,
) -> Result<(wgpu::Device, wgpu::Queue), Box<dyn std::error::Error>> {
    use wgpu::hal::vulkan as hvk;

    let shared_instance = hvk::Instance::from_raw(
        ash::Entry::linked(),
        vk.instance.clone(),
        vk::make_api_version(0, 1, 1, 0),
        0,
        None,
        vec![],
        wgpu::InstanceFlags::empty(),
        false,
        None,
    )?;

    let exposed = shared_instance
        .expose_adapter(vk.physical_device)
        .ok_or("wgpu: failed to expose physical device")?;

    let open_device = exposed.adapter.device_from_raw(
        vk.device.clone(),
        None,
        &[],
        wgpu::Features::empty(),
        &wgpu::MemoryHints::default(),
        vk.queue_family_index,
        0,
    )?;

    let wgpu_instance = wgpu::Instance::from_hal::<hvk::Api>(shared_instance);
    let wgpu_adapter = wgpu_instance.create_adapter_from_hal(exposed);

    let (device, queue) = wgpu_adapter.create_device_from_hal(
        open_device,
        &wgpu::DeviceDescriptor {
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits {
                max_texture_dimension_2d: 4096,
                ..wgpu::Limits::downlevel_defaults()
            },
            ..Default::default()
        },
    )?;

    Ok((device, queue))
}

unsafe fn import_vk_image_as_wgpu(
    device: &wgpu::Device,
    image: vk::Image,
    wgpu_format: wgpu::TextureFormat,
    width: u32,
    height: u32,
    array_layers: u32,
) -> wgpu::Texture {
    use wgpu::hal::vulkan as hvk;

    let hal_texture = hvk::Device::texture_from_raw(
        image,
        &wgpu::hal::TextureDescriptor {
            label: Some("xr_swapchain_image"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: array_layers,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu_format,
            usage: wgpu::TextureUses::COLOR_TARGET | wgpu::TextureUses::RESOURCE,
            memory_flags: wgpu::hal::MemoryFlags::empty(),
            view_formats: vec![],
        },
        None,
    );

    device.create_texture_from_hal::<hvk::Api>(
        hal_texture,
        &wgpu::TextureDescriptor {
            label: Some("xr_swapchain_tex"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: array_layers,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        },
    )
}
