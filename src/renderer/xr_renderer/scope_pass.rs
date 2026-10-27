//! The held optics' world views: the level drawn again from each objective
//! lens, with the eye pass's own mono pipelines, before the eyes are drawn.

use glam::Vec3;

use super::render_frame::{LayeredDraw, MeshDraw, SkinnedDraw};
use super::XrRenderer;
use crate::renderer::baked_cave::BakedCaveDraw;
use crate::renderer::scope::EyeSample;
use crate::renderer::uniforms::{PlayerUpload, PostUpload, ProbeUpload, ShadowUpload, SkyUpload};

/// Bound radius below which a draw can be a held object: one the scope camera
/// sits inside is the optic's own body, so the view leaves it out.
const HELD_BODY_M: f32 = 1.5;

/// The cuboids' buffers and where the ground sits in them.
pub(super) struct SolidWorld<'a> {
    pub vertices: &'a wgpu::Buffer,
    pub indices: &'a wgpu::Buffer,
    pub ranges: &'a [(Option<String>, u32, u32, f32)],
    pub terrain: Option<(u32, u32)>,
    /// The ground's finest level of detail, when it carries several.
    pub ground_step1: Option<u32>,
}

/// The frame's draw lists, as the eye pass built them.
pub(super) struct ScopeWorld<'a> {
    pub solid: Option<SolidWorld<'a>>,
    pub brushes: Option<&'a (wgpu::Buffer, wgpu::Buffer, u32)>,
    pub layered: &'a [LayeredDraw<'a>],
    pub baked_caves: &'a [BakedCaveDraw<'a>],
    pub meshes: &'a [MeshDraw<'a>],
    pub skinned: &'a [SkinnedDraw<'a>],
    pub particles: Option<(&'a wgpu::Buffer, &'a wgpu::Buffer, u32)>,
    pub water: bool,
}

/// The frame's lighting uploads, shared with the eyes.
pub(super) struct ScopeLight<'a> {
    pub shadow: &'a ShadowUpload,
    pub sky: &'a SkyUpload,
    pub post: &'a PostUpload,
    pub player: &'a PlayerUpload,
    pub probes: Option<&'a ProbeUpload>,
}

impl XrRenderer {
    /// Size the optics' targets to their lenses and write their uniforms.
    /// Before the draw lists are built: those borrow `self` for the frame.
    pub(super) fn prepare_scopes(&mut self, eye_views: &[openxr::View]) {
        if self.scopes.is_empty() || eye_views.len() < 2 {
            return;
        }
        let sample = |v: &openxr::View| EyeSample {
            position: Vec3::new(v.pose.position.x, v.pose.position.y, v.pose.position.z),
            vertical_fov_rad: (v.fov.angle_up - v.fov.angle_down).abs(),
        };
        let eyes = [sample(&eye_views[0]), sample(&eye_views[1])];
        self.scopes.prepare(&self.wgpu_device, &self.wgpu_queue, eyes, self.height);
    }

    /// Draw each optic's world view, one submit apiece: the scene uniform is
    /// rewritten for every camera, and a write lands only at the next submit.
    pub(super) fn render_scope_views(&self, world: &ScopeWorld, light: &ScopeLight) {
        let post = PostUpload { reflection_share: false, ..*light.post };
        for (slot, view_proj, camera) in self.scopes.world_views(crate::renderer::space_warp::FAR_Z) {
            let Some(target) = self.scopes.target(slot) else { continue };
            self.uniform_buf.upload_scene_with_probes(
                &self.wgpu_queue, view_proj, camera, light.shadow, light.sky, &post, light.player, light.probes,
            );
            let mut encoder = self.wgpu_device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("scope") });
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("scope_world"),
                    color_attachments: &[Some(target.color_attachment())],
                    depth_stencil_attachment: Some(target.depth_attachment()),
                    ..Default::default()
                });
                self.draw_scope_solids(&mut pass, world);
                self.draw_scope_models(&mut pass, world, camera);
                self.draw_scope_water_and_sky(&mut pass, world);
            }
            self.scopes.finish_world(&mut encoder, slot);
            self.wgpu_queue.submit(Some(encoder.finish()));
        }
    }

    fn draw_scope_solids<'p>(&'p self, pass: &mut wgpu::RenderPass<'p>, world: &ScopeWorld<'p>) {
        if let Some(solid) = &world.solid {
            pass.set_pipeline(self.sp_solid(false));
            pass.set_bind_group(0, &self.uniform_buf.bind_group, &[]);
            pass.set_vertex_buffer(0, solid.vertices.slice(..));
            pass.set_index_buffer(solid.indices.slice(..), wgpu::IndexFormat::Uint32);
            for (lightmap, start, count, _) in solid.ranges.iter().filter(|r| solid.terrain != Some((r.1, r.2))) {
                pass.set_bind_group(1, self.cuboid_lightmap_bg(lightmap.as_deref()), &[]);
                pass.draw_indexed(*start..*start + *count, 0, 0..1);
            }
            if let Some((start, count)) = solid.terrain {
                pass.set_pipeline(self.sp_terrain(false));
                pass.set_bind_group(1, &self.terrain_material.bind_group, &[]);
                let count = solid.ground_step1.map_or(count, |step| step.min(count));
                pass.draw_indexed(start..start + count, 0, 0..1);
            }
        }
        if let Some((vertices, indices, count)) = world.brushes {
            pass.set_pipeline(self.sp_brush(false));
            pass.set_bind_group(0, &self.uniform_buf.bind_group, &[]);
            pass.set_bind_group(1, &self.brush_materials.bind_group, &[]);
            pass.set_bind_group(2, self.brush_lightmap_bg(), &[]);
            pass.set_vertex_buffer(0, vertices.slice(..));
            pass.set_index_buffer(indices.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..*count, 0, 0..1);
        }
        if !world.layered.is_empty() {
            pass.set_pipeline(self.sp_layered(false));
            pass.set_bind_group(0, &self.uniform_buf.bind_group, &[]);
            pass.set_bind_group(1, &self.terrain_material.bind_group, &[]);
            for (model, vertices, indices, count, _, _) in world.layered {
                pass.set_bind_group(2, *model, &[]);
                pass.set_vertex_buffer(0, vertices.slice(..));
                pass.set_index_buffer(indices.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..*count, 0, 0..1);
            }
        }
        let spotless = self.spotless_frame.load(std::sync::atomic::Ordering::Relaxed);
        self.baked_cave_pipelines.draw_scene(
            pass, &self.uniform_buf.bind_group, &self.terrain_material.bind_group, world.baked_caves, None, spotless, false, None,
        );
    }

    fn draw_scope_models<'p>(&'p self, pass: &mut wgpu::RenderPass<'p>, world: &ScopeWorld<'p>, camera: Vec3) {
        let own_body = |d: &MeshDraw| d.bound.w < HELD_BODY_M && camera.distance(d.bound.truncate()) < d.bound.w;
        for blended in [false, true] {
            let pipeline = if blended { self.sp_mesh(false) } else { self.sp_mesh_opaque(false).unwrap_or(self.sp_mesh(false)) };
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &self.uniform_buf.bind_group, &[]);
            for d in world.meshes.iter().filter(|d| d.blended == blended && !own_body(d)) {
                pass.set_bind_group(1, d.model, &[]);
                pass.set_bind_group(2, d.texture, &[]);
                pass.set_bind_group(3, d.lightmap, &[]);
                pass.set_vertex_buffer(0, d.vertices.slice(..));
                pass.set_index_buffer(d.indices.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..d.count, 0, 0..1);
            }
        }
        for blended in [false, true] {
            let pipeline = if blended { self.sp_skinned(false) } else { self.sp_skinned_opaque(false).unwrap_or(self.sp_skinned(false)) };
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &self.uniform_buf.bind_group, &[]);
            for (model, texture, joints, vertices, indices, count, _) in world.skinned.iter().filter(|d| d.6 == blended) {
                pass.set_bind_group(1, *model, &[]);
                pass.set_bind_group(2, *texture, &[]);
                pass.set_bind_group(3, *joints, &[]);
                pass.set_vertex_buffer(0, vertices.slice(..));
                pass.set_index_buffer(indices.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..*count, 0, 0..1);
            }
        }
    }

    fn draw_scope_water_and_sky<'p>(&'p self, pass: &mut wgpu::RenderPass<'p>, world: &ScopeWorld<'p>) {
        if world.water {
            pass.set_bind_group(0, &self.uniform_buf.bind_group, &[]);
            for body in &self.water_bodies {
                pass.set_pipeline(self.sp_water(false, crate::renderer::water_pipeline::WaterMud::of(&body.uniform)));
                pass.set_bind_group(1, &body.bind_groups[body.waves.current()], &[]);
                pass.set_vertex_buffer(0, body.vertex_buffer.slice(..));
                pass.set_index_buffer(body.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..body.index_count, 0, 0..1);
            }
        }
        pass.set_pipeline(self.sp_sky(false));
        pass.set_bind_group(0, &self.uniform_buf.bind_group, &[]);
        pass.set_bind_group(1, &self.sky.bind_group, &[]);
        pass.draw(0..3, 0..1);
        if let Some((vertices, indices, count)) = world.particles.filter(|p| p.2 > 0) {
            pass.set_pipeline(self.sp_particle(false));
            pass.set_bind_group(0, &self.uniform_buf.bind_group, &[]);
            pass.set_vertex_buffer(0, vertices.slice(..));
            pass.set_index_buffer(indices.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..count, 0, 0..1);
        }
    }
}
