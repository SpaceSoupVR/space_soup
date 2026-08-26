use openxr as xr;
use wgpu::util::DeviceExt;

use crate::renderer::{
    brush_pipeline::BrushVertex,
    camera::Camera,
    cuboid::{build_solid_mesh_with_ranges, build_wire_mesh, Cuboid, SolidVertex},
    lights::Light,
    mirror::{self, MirrorSurface},
    particle::{self, Beam, Particle},
    MeshInstance,
};

use super::{ShadowQuality, XrRenderer};

type MeshDraw<'a> = (
    &'a wgpu::BindGroup,
    &'a wgpu::BindGroup,
    &'a wgpu::BindGroup,
    &'a wgpu::Buffer,
    &'a wgpu::Buffer,
    u32,
);

/// A layered draw needs no texture bind group: every layered mesh in a scene
/// shares the one terrain material array, which is bound once for the batch.
type LayeredDraw<'a> = (
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
    lightmap_bg: &'a wgpu::BindGroup,
    mesh_draws: &mut Vec<MeshDraw<'a>>,
    skinned_draws: &mut Vec<SkinnedDraw<'a>>,
    layered_draws: &mut Vec<LayeredDraw<'a>>,
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
                    prim.indices.len() as u32,
                ));
            }
        }
    } else {
        for prim in &instance.mesh.primitives {
            // One or the other, never both: drawing a cave through the mesh
            // pipeline as well would put untextured geometry in exactly the
            // same place, z-fighting with itself.
            if let Some(layered) = &prim.layered {
                layered_draws.push((
                    &instance.model.bind_group,
                    &layered.vertex_buffer,
                    &prim.index_buffer,
                    prim.indices.len() as u32,
                ));
                continue;
            }
            mesh_draws.push((
                &instance.model.bind_group,
                &prim.texture.bind_group,
                lightmap_bg,
                &prim.vertex_buffer,
                &prim.index_buffer,
                prim.indices.len() as u32,
            ));
        }
    }
}

impl XrRenderer {
    pub fn render_frame(
        &mut self,
        session: &xr::Session<xr::Vulkan>,
        stage: &xr::Space,
        time: xr::Time,
        cuboids: &[Cuboid],
    ) -> Result<Vec<xr::CompositionLayerProjectionView<xr::Vulkan>>, Box<dyn std::error::Error>>
    {
        self.render_frame_with_meshes(
            session, stage, time, cuboids, &[], &[], &[], &[], &[], None, None, None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn render_frame_with_meshes(
        &mut self,
        session: &xr::Session<xr::Vulkan>,
        stage: &xr::Space,
        time: xr::Time,
        cuboids: &[Cuboid],
        meshes: &[MeshInstance],
        mirror_only_meshes: &[MeshInstance],
        lights: &[Light],
        particles: &[Particle],
        beams: &[Beam],
        // Static ground geometry, already in SolidVertex form. Rides the cuboid
        // solid pipeline rather than getting one of its own: it wants exactly
        // the same shading, shadowing and depth behaviour, and a second pipeline
        // would be a second place for those to drift.
        terrain: Option<(&[SolidVertex], &[u32])>,
        // Level geometry the client meshed from brushes, already in the same
        // player-local space as the cuboids, and carrying a material per vertex.
        // Its own pipeline and its own buffer: a brush vertex is not a solid
        // vertex any more, because a wall's material and tangent frame have
        // nowhere to live in one.
        //
        // Drawn textured in the eye pass AND in the mirror pass, which is one
        // better than terrain manages -- terrain is splat-shaded when looked at
        // and flat in reflections. SSR skips brushes for the same reason it
        // skips flat cuboids: that pass only runs for reflective ranges.
        brushes: Option<(&[BrushVertex], &[u32])>,
        mirror: Option<MirrorSurface>,
    ) -> Result<Vec<xr::CompositionLayerProjectionView<xr::Vulkan>>, Box<dyn std::error::Error>>
    {
        let image_index = self.swapchain.acquire_image()? as usize;
        self.swapchain.wait_image(xr::Duration::INFINITE)?;
        let cpu_start = std::time::Instant::now();
        self.lights_uniform.upload(&self.wgpu_queue, lights);

        let (_, eye_views) =
            session.locate_views(xr::ViewConfigurationType::PRIMARY_STEREO, time, stage)?;

        let head_rot = {
            let o = eye_views[0].pose.orientation;
            glam::Quat::from_xyzw(o.x, o.y, o.z, o.w)
        };
        let cam_right = head_rot * glam::Vec3::X;
        let cam_up = head_rot * glam::Vec3::Y;
        let view_dir = head_rot * glam::Vec3::NEG_Z;

        let (mut solid_verts, mut solid_idx, mut solid_ranges) =
            build_solid_mesh_with_ranges(cuboids);

        // Terrain appends into the SAME buffers as the cuboids -- the terrain
        // pipeline takes SolidVertex too, so one vertex buffer serves both and
        // the geometry path is unchanged.
        //
        // It is recorded twice on purpose. It stays in `solid_ranges` so the
        // mirror and SSR passes keep drawing it (a reflection that omits the
        // ground is far worse than one that shades it flatly), and it is ALSO
        // recorded in `terrain_range` so the main eye pass can skip it there and
        // redraw it through TerrainPipeline. The consequence is deliberate and
        // worth naming: terrain is splat-shaded when looked at directly and
        // flat-shaded in reflections. Unifying that means teaching the mirror
        // and SSR pipelines the terrain material, which is a bigger change than
        // this one and buys much less.
        let mut terrain_range: Option<(u32, u32)> = None;
        if let Some((terrain_verts, terrain_idx)) = terrain {
            if !terrain_verts.is_empty() && !terrain_idx.is_empty() {
                let base = solid_verts.len() as u32;
                let index_start = solid_idx.len() as u32;
                solid_verts.extend_from_slice(terrain_verts);
                solid_idx.extend(terrain_idx.iter().map(|i| i + base));
                solid_ranges.push((None, index_start, terrain_idx.len() as u32, 0.0));
                terrain_range = Some((index_start, terrain_idx.len() as u32));
            }
        }
        let (solid_verts, solid_idx, solid_ranges) = (solid_verts, solid_idx, solid_ranges);

        // Empty when the scene has no brushes, or when every one of them has
        // been shot away -- both are ordinary, and both mean no draw rather
        // than a zero-length one.
        let brush_geometry = brushes.filter(|(v, i)| !v.is_empty() && !i.is_empty());
        let brush_buffers = brush_geometry.map(|(v, i)| {
            (
                self.wgpu_device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("brush_vb"),
                        contents: bytemuck::cast_slice(v),
                        usage: wgpu::BufferUsages::VERTEX,
                    }),
                self.wgpu_device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("brush_ib"),
                        contents: bytemuck::cast_slice(i),
                        usage: wgpu::BufferUsages::INDEX,
                    }),
                i.len() as u32,
            )
        });
        let (wire_verts, wire_idx) = build_wire_mesh(cuboids);
        let (particle_verts, particle_idx) =
            particle::build_particle_mesh(particles, beams, cam_right, cam_up, view_dir);

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
        let particle_vb = self
            .wgpu_device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("particle_vb"),
                contents: bytemuck::cast_slice(&particle_verts),
                usage: wgpu::BufferUsages::VERTEX,
            });
        let particle_ib = self
            .wgpu_device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("particle_ib"),
                contents: bytemuck::cast_slice(&particle_idx),
                usage: wgpu::BufferUsages::INDEX,
            });

        let mut mesh_draws: Vec<MeshDraw> = Vec::new();
        let mut skinned_draws: Vec<SkinnedDraw> = Vec::new();
        let mut layered_draws: Vec<LayeredDraw> = Vec::new();
        // Depth-pass casters. The ordinary vertex buffer even for a layered
        // mesh -- a depth pass reads position and nothing else, so a baked cave
        // casts here with no pipeline of its own.
        let mut shadow_casters: Vec<crate::renderer::shadow::ShadowMeshDraw> = Vec::new();
        for instance in meshes {
            instance
                .model
                .upload(&self.wgpu_queue, instance.mesh.model_matrix());
            let lightmap_bg = self.mesh_lightmap_bg(instance.lightmap_key);
            push_mesh_draws(
                instance, lightmap_bg, &mut mesh_draws, &mut skinned_draws, &mut layered_draws,
            );
            if instance.mesh.skin.is_none() {
                for prim in &instance.mesh.primitives {
                    shadow_casters.push((
                        &prim.vertex_buffer,
                        &prim.index_buffer,
                        prim.indices.len() as u32,
                        &instance.model.bind_group,
                    ));
                }
            }
        }

        let mut mirror_only_mesh_draws: Vec<MeshDraw> = Vec::new();
        let mut mirror_only_skinned_draws: Vec<SkinnedDraw> = Vec::new();
        let mut mirror_only_layered_draws: Vec<LayeredDraw> = Vec::new();
        for instance in mirror_only_meshes {
            instance
                .model
                .upload(&self.wgpu_queue, instance.mesh.model_matrix());
            let lightmap_bg = self.mesh_lightmap_bg(instance.lightmap_key);
            push_mesh_draws(
                instance,
                lightmap_bg,
                &mut mirror_only_mesh_draws,
                &mut mirror_only_skinned_draws,
                &mut mirror_only_layered_draws,
            );
        }

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

        // The sky's ambient, projected when the scene was loaded rather than
        // now -- see XrRenderer::set_sky.
        let sky_upload =
            crate::renderer::uniforms::SkyUpload::from(&self.sky.irradiance);

        // SHADOWS: once per frame, not once per eye.
        //
        // A shadow map is built in the LIGHT's space, so it is identical for
        // both eyes -- rendering it inside the loop below would double the most
        // expensive pass in the frame for a bit-identical second copy.
        let head = glam::Vec3::new(
            eye_views[0].pose.position.x,
            eye_views[0].pose.position.y,
            eye_views[0].pose.position.z,
        );
        let want_sun = self.shadow_quality != ShadowQuality::Off;
        let want_spot = self.shadow_quality == ShadowQuality::SunAndSpot;

        let sun = want_sun
            .then(|| lights.iter().find(|l| l.kind == crate::renderer::LightKind::Directional))
            .flatten();
        let spot_index = want_spot
            .then(|| lights.iter().position(|l| l.kind == crate::renderer::LightKind::Spot))
            .flatten();

        // The box follows the head and is pushed forward, so its limited
        // resolution is spent on what the player is looking at. Half of a
        // head-centred box would always be behind them.
        let forward = (glam::Quat::from_xyzw(
            eye_views[0].pose.orientation.x,
            eye_views[0].pose.orientation.y,
            eye_views[0].pose.orientation.z,
            eye_views[0].pose.orientation.w,
        ) * glam::Vec3::NEG_Z)
            .normalize_or_zero();
        let sun_radius = 30.0_f32;
        let focus = head + forward * (sun_radius * 0.5);

        let shadow = crate::renderer::uniforms::ShadowUpload {
            sun_view_proj: sun
                .map(|l| {
                    crate::renderer::shadow::directional_light_matrix(l.direction, focus, sun_radius)
                })
                .unwrap_or(glam::Mat4::IDENTITY),
            spot_view_proj: spot_index
                .map(|i| {
                    let l = &lights[i];
                    crate::renderer::shadow::spot_light_matrix(
                        l.position, l.direction, l.cone_angle_deg, l.range,
                    )
                })
                .unwrap_or(glam::Mat4::IDENTITY),
            sun_enabled: sun.is_some(),
            spot_enabled: spot_index.is_some(),
            flashlight_index: spot_index.unwrap_or(0) as u32,
        };

        if shadow.sun_enabled || shadow.spot_enabled {
            let solid_caster = (!solid_idx.is_empty())
                .then_some((&solid_vb, &solid_ib, solid_idx.len() as u32));
            let brush_caster = brush_buffers
                .as_ref()
                .map(|(vb, ib, count)| (vb, ib, *count));
            let mut encoder = self
                .wgpu_device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("shadow_encoder"),
                });
            if shadow.sun_enabled {
                self.shadow_map.upload_light(
                    &self.wgpu_queue,
                    crate::renderer::shadow::ShadowKind::Sun,
                    shadow.sun_view_proj,
                );
                self.shadow_map.record(
                    &mut encoder,
                    crate::renderer::shadow::ShadowKind::Sun,
                    solid_caster,
                    brush_caster,
                    &shadow_casters,
                );
            }
            if shadow.spot_enabled {
                self.shadow_map.upload_light(
                    &self.wgpu_queue,
                    crate::renderer::shadow::ShadowKind::Spot,
                    shadow.spot_view_proj,
                );
                self.shadow_map.record(
                    &mut encoder,
                    crate::renderer::shadow::ShadowKind::Spot,
                    solid_caster,
                    brush_caster,
                    &shadow_casters,
                );
            }
            self.wgpu_queue.submit(Some(encoder.finish()));
        }

        // Whether the eye pass will composite anything that has to depth-test
        // against the world. Only these two read the scene depth, and only they
        // make a multisampled depth buffer worth storing.
        let needs_scene_depth = mirror_quad.is_some()
            || solid_ranges.iter().any(|(_, _, _, r)| *r > 0.0);

        for eye in 0..2usize {
            let ev = &eye_views[eye];
            let view = Camera::xr_view(ev.pose);
            let proj = Camera::xr_projection(ev.fov, 0.03, 1000.0);

            if let Some(m) = &mirror {
                let reflect = mirror::reflection_matrix(m.position, m.normal());
                let mirror_view = view * reflect;

                let world_plane = mirror::world_plane_equation(m.position, m.normal());
                let eye_plane = mirror::plane_to_eye_space(mirror_view.inverse(), world_plane);
                let mirror_proj = mirror::oblique_near_clip(proj, eye_plane);
                let mirror_view_proj = Camera::gl_to_wgpu_ndc(mirror_proj) * mirror_view;

                // The reflected eye, so specular highlights land where the
                // reflection says they should rather than where the real eye is.
                let mirror_eye = mirror_view.inverse().transform_point3(glam::Vec3::ZERO);
                self.uniform_buf.upload_with_sky(
                    &self.wgpu_queue, mirror_view_proj, mirror_eye, &shadow, &sky_upload,
                );
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
                        for (lightmap_key, index_start, count, _reflectivity) in &solid_ranges {
                            pass.set_bind_group(1, self.cuboid_lightmap_bg(lightmap_key.as_deref()), &[]);
                            pass.draw_indexed(*index_start..*index_start + *count, 0, 0..1);
                        }
                    }
                    if let Some((vb, ib, count)) = &brush_buffers {
                        // The mirror variant, because a reflected world reverses
                        // every winding -- the same reason the solid pipeline
                        // has one. Textured here too: a room whose walls are
                        // concrete in front of the mirror and flat grey inside
                        // it is worse than no mirror.
                        pass.set_pipeline(&self.brush_mirror_pipeline.pipeline);
                        pass.set_bind_group(0, &self.uniform_buf.bind_group, &[]);
                        pass.set_bind_group(1, &self.brush_materials.bind_group, &[]);
                        pass.set_vertex_buffer(0, vb.slice(..));
                        pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                        pass.draw_indexed(0..*count, 0, 0..1);
                    }
                    if !layered_draws.is_empty() || !mirror_only_layered_draws.is_empty() {
                        // Mirror variant, because a reflected world reverses
                        // every winding. It matters more here than elsewhere:
                        // this shader flips the normal on a back face, so the
                        // wrong front-face rule inverts the lighting of every
                        // cave surface in the reflection rather than merely
                        // culling the wrong side.
                        pass.set_pipeline(&self.layered_mesh_mirror_pipeline.pipeline);
                        pass.set_bind_group(0, &self.uniform_buf.bind_group, &[]);
                        pass.set_bind_group(1, &self.terrain_material.bind_group, &[]);
                        for (model_bg, vb, ib, count) in
                            layered_draws.iter().chain(mirror_only_layered_draws.iter())
                        {
                            pass.set_bind_group(2, *model_bg, &[]);
                            pass.set_vertex_buffer(0, vb.slice(..));
                            pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                            pass.draw_indexed(0..*count, 0, 0..1);
                        }
                    }
                    let all_mesh_draws = mesh_draws.iter().chain(mirror_only_mesh_draws.iter());
                    let all_skinned_draws =
                        skinned_draws.iter().chain(mirror_only_skinned_draws.iter());

                    if !mesh_draws.is_empty() || !mirror_only_mesh_draws.is_empty() {
                        pass.set_pipeline(&self.mirror_mesh_pipeline.pipeline);
                        pass.set_bind_group(0, &self.uniform_buf.bind_group, &[]);
                        for (model_bg, tex_bg, lightmap_bg, vb, ib, count) in all_mesh_draws {
                            pass.set_bind_group(1, *model_bg, &[]);
                            pass.set_bind_group(2, *tex_bg, &[]);
                            pass.set_bind_group(3, *lightmap_bg, &[]);
                            pass.set_vertex_buffer(0, vb.slice(..));
                            pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                            pass.draw_indexed(0..*count, 0, 0..1);
                        }
                    }
                    if !skinned_draws.is_empty() || !mirror_only_skinned_draws.is_empty() {
                        // The 1x twin: this pass renders into a single-sampled
                        // mirror target.
                        pass.set_pipeline(&self.skinned_mesh_mirror_pipeline.pipeline);
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
                    // The sky in the reflection too. A mirror showing a black
                    // void where the sky is looks worse than no mirror, and the
                    // ray is reconstructed from whatever view_proj was uploaded
                    // -- which for this pass is the reflected one, so it needs
                    // no special case beyond the 1x pipeline.
                    {
                        pass.set_pipeline(&self.sky_mirror_pipeline.pipeline);
                        pass.set_bind_group(0, &self.uniform_buf.bind_group, &[]);
                        pass.set_bind_group(1, &self.sky.bind_group, &[]);
                        pass.draw(0..3, 0..1);
                    }
                }
                self.wgpu_queue.submit(Some(encoder.finish()));
            }

            let eye_view_proj = Camera::gl_to_wgpu_ndc(proj) * view;
            let cam_pos = glam::Vec3::new(ev.pose.position.x, ev.pose.position.y, ev.pose.position.z);
            self.uniform_buf.upload_with_sky(
                &self.wgpu_queue, eye_view_proj, cam_pos, &shadow, &sky_upload,
            );
            self.ssr_camera_uniform.upload(&self.wgpu_queue, eye_view_proj, cam_pos);

            {
                let mut encoder = self.wgpu_device.create_command_encoder(
                    &wgpu::CommandEncoderDescriptor { label: Some("ssr_scene") },
                );
                {
                    // MULTISAMPLING, AND THE TWO STORE OPS THAT DECIDE ITS COST.
                    //
                    // Colour: when the target is multisampled the pass draws
                    // into the MSAA attachment and RESOLVES into the ordinary
                    // one. `StoreOp::Discard` on the multisampled side still
                    // performs the resolve -- it discards the samples, which is
                    // exactly what we want: on a tile GPU they never leave tile
                    // memory and only the resolved image is written out.
                    //
                    // Depth: cannot be resolved, so a multisampled pass that
                    // has to KEEP its depth pays four times the write. The only
                    // thing that reads it is the blit's `frag_depth`, which
                    // exists so reflective solids and the mirror quad composite
                    // correctly in the eye pass. When the frame has neither --
                    // which is every scene shipped so far -- the depth is
                    // discarded and the expensive half of MSAA never happens.
                    let target = &self.scene_targets[eye];
                    let (color_view, resolve_target, color_store) =
                        match target.msaa_color_view.as_ref() {
                            Some(msaa) => (
                                msaa,
                                Some(&target.color_view),
                                wgpu::StoreOp::Discard,
                            ),
                            None => (&target.color_view, None, wgpu::StoreOp::Store),
                        };
                    let depth_store = if needs_scene_depth {
                        wgpu::StoreOp::Store
                    } else {
                        wgpu::StoreOp::Discard
                    };

                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("ssr_scene_pass"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: color_view,
                            resolve_target,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color {
                                    r: 0.02,
                                    g: 0.02,
                                    b: 0.05,
                                    a: 1.0,
                                }),
                                store: color_store,
                            },
                        })],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: &target.depth_view,
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Clear(1.0),
                                store: depth_store,
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
                        for (lightmap_key, index_start, count, _reflectivity) in &solid_ranges {
                            // Terrain is in this list for the reflection passes;
                            // here it gets its own pipeline instead.
                            if terrain_range == Some((*index_start, *count)) {
                                continue;
                            }
                            pass.set_bind_group(1, self.cuboid_lightmap_bg(lightmap_key.as_deref()), &[]);
                            pass.draw_indexed(*index_start..*index_start + *count, 0, 0..1);
                        }
                    }
                    if let Some((index_start, count)) = terrain_range {
                        pass.set_pipeline(&self.terrain_pipeline.pipeline);
                        pass.set_bind_group(0, &self.uniform_buf.bind_group, &[]);
                        pass.set_bind_group(1, &self.terrain_material.bind_group, &[]);
                        pass.set_vertex_buffer(0, solid_vb.slice(..));
                        pass.set_index_buffer(solid_ib.slice(..), wgpu::IndexFormat::Uint32);
                        pass.draw_indexed(index_start..index_start + count, 0, 0..1);
                    }
                    if let Some((vb, ib, count)) = &brush_buffers {
                        // One draw for the whole level, however many materials
                        // it uses: the material is a vertex attribute and every
                        // colour map is a layer of one array.
                        pass.set_pipeline(&self.brush_pipeline.pipeline);
                        pass.set_bind_group(0, &self.uniform_buf.bind_group, &[]);
                        pass.set_bind_group(1, &self.brush_materials.bind_group, &[]);
                        pass.set_vertex_buffer(0, vb.slice(..));
                        pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                        pass.draw_indexed(0..*count, 0, 0..1);
                    }
                    if !layered_draws.is_empty() {
                        // One material bind for the batch: every cave in a scene
                        // blends the same four terrain layers, so the only thing
                        // that changes between draws is the model matrix.
                        pass.set_pipeline(&self.layered_mesh_pipeline.pipeline);
                        pass.set_bind_group(0, &self.uniform_buf.bind_group, &[]);
                        pass.set_bind_group(1, &self.terrain_material.bind_group, &[]);
                        for (model_bg, vb, ib, count) in &layered_draws {
                            pass.set_bind_group(2, *model_bg, &[]);
                            pass.set_vertex_buffer(0, vb.slice(..));
                            pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                            pass.draw_indexed(0..*count, 0, 0..1);
                        }
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
                        for (model_bg, tex_bg, lightmap_bg, vb, ib, count) in &mesh_draws {
                            pass.set_bind_group(1, *model_bg, &[]);
                            pass.set_bind_group(2, *tex_bg, &[]);
                            pass.set_bind_group(3, *lightmap_bg, &[]);
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
                    // THE SKY, after every opaque and before anything blended.
                    //
                    // It sits at the far plane and writes no depth, so early-Z
                    // rejects every pixel the level already covered -- drawing
                    // it first would shade all of them and throw the work away,
                    // which on a fill-limited tile GPU is the whole cost of the
                    // pass for nothing. Before the particles because those are
                    // blended and have to land on top of it.
                    {
                        pass.set_pipeline(&self.sky_pipeline.pipeline);
                        pass.set_bind_group(0, &self.uniform_buf.bind_group, &[]);
                        pass.set_bind_group(1, &self.sky.bind_group, &[]);
                        pass.draw(0..3, 0..1);
                    }
                    if !particle_verts.is_empty() {
                        pass.set_pipeline(&self.particle_pipeline.pipeline);
                        pass.set_bind_group(0, &self.uniform_buf.bind_group, &[]);
                        pass.set_vertex_buffer(0, particle_vb.slice(..));
                        pass.set_index_buffer(particle_ib.slice(..), wgpu::IndexFormat::Uint32);
                        pass.draw_indexed(0..particle_idx.len() as u32, 0, 0..1);
                    }
                }
                self.wgpu_queue.submit(Some(encoder.finish()));
            }

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

                pass.set_pipeline(&self.ssr_pipelines.blit_pipeline);
                pass.set_bind_group(0, &self.scene_targets[eye].bind_group, &[]);
                pass.draw(0..3, 0..1);

                if solid_ranges.iter().any(|(_, _, _, r)| *r > 0.0) {
                    pass.set_pipeline(&self.ssr_solid_pipeline.pipeline);
                    pass.set_bind_group(0, &self.uniform_buf.bind_group, &[]);
                    pass.set_bind_group(2, &self.ssr_camera_uniform.bind_group, &[]);
                    pass.set_bind_group(3, &self.scene_targets[eye].bind_group, &[]);
                    pass.set_vertex_buffer(0, solid_vb.slice(..));
                    pass.set_index_buffer(solid_ib.slice(..), wgpu::IndexFormat::Uint32);
                    for (lightmap_key, index_start, count, reflectivity) in &solid_ranges {
                        if *reflectivity <= 0.0 {
                            continue;
                        }
                        pass.set_bind_group(1, self.cuboid_lightmap_bg(lightmap_key.as_deref()), &[]);
                        pass.draw_indexed(*index_start..*index_start + *count, 0, 0..1);
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

        let cpu_time = cpu_start.elapsed();
        let gpu_wait_start = std::time::Instant::now();
        self.wgpu_device.poll(wgpu::PollType::Wait);
        let gpu_time = gpu_wait_start.elapsed();
        self.frame_stats.record(cpu_time, gpu_time, std::time::Instant::now());
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
}
