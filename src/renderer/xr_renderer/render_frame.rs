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
            session, stage, time, cuboids, &[], &[], &[], &[], &[], None, &[], None, None,
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
        // Spatially-coherent runs of `terrain`'s index buffer, with bounds, so
        // each shadow pass can draw only the ground its light actually reaches.
        // Empty means "not partitioned", and the whole terrain is drawn -- an
        // unculled caster is slow, a missing one is a bug.
        terrain_chunks: &[crate::renderer::shadow::CasterChunk],
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
        // Before the swapchain image is held: a level load marks it dirty, and
        // building it takes a moment once. See `ensure_ground_map`.
        self.ensure_ground_map();
        let image_index = self.swapchain.acquire_image()? as usize;
        self.swapchain.wait_image(xr::Duration::INFINITE)?;
        let cpu_start = std::time::Instant::now();

        // TRACKING MAY NOT BE READY YET, AND THAT IS NOT AN ERROR.
        //
        // `locate_views` reports through its FLAGS whether the poses it hands
        // back mean anything. Those were discarded, so on a cold start -- before
        // tracking settles, or with the headset off the head -- the renderer
        // drew with garbage poses and `xrEndFrame` refused them with
        // `XR_ERROR_POSE_INVALID`. That propagated all the way out of the frame
        // loop and the app EXITED.
        //
        // So the symptom was "it never loads": launching from the library died
        // during startup, while a launch a few seconds after a deploy survived
        // because tracking had settled by then (2026-09-19).
        //
        // An unlocated frame is an ordinary event. Return no views, and the
        // caller submits no layer for this frame -- which OpenXR allows, and
        // which the compositor covers with the previous frame.
        let (view_flags, mut eye_views) =
            session.locate_views(xr::ViewConfigurationType::PRIMARY_STEREO, time, stage)?;
        let located = view_flags.contains(xr::ViewStateFlags::ORIENTATION_VALID)
            && view_flags.contains(xr::ViewStateFlags::POSITION_VALID);
        if located && eye_views.len() >= 2 {
            self.last_fov = Some([eye_views[0].fov, eye_views[1].fov]);
        }
        // The eye images' density maps for this frame's level -- the lever,
        // with the A/B schedule's switch on top, exactly as `fx` below takes
        // them -- before anything further down borrows the renderer. See
        // `foveation`.
        let foveation_phase = if crate::renderer::perf_ab::ENABLED || self.levers.ab_cycle {
            crate::renderer::perf_ab::Phase::cycle_phase(self.perf_windows)
        } else {
            crate::renderer::perf_ab::Phase::Baseline
        };
        let foveation = self.levers.clone().with_phase(foveation_phase).foveation;
        self.apply_foveation(foveation);
        // Where the headset really is, for the compositor when the head is
        // pinned: see `proj_views` below.
        let tracked_poses: Option<Vec<xr::Posef>> = located.then(|| eye_views.iter().map(|v| v.pose).collect());
        // A PINNED HEAD (a benchmark viewpoint, see `bench`) draws whether or
        // not the runtime could locate the views: the poses are replaced, and
        // a headset on a desk can lose tracking in a dark room. It still needs
        // each eye's field of view, so it waits for one located frame.
        match (self.pinned_head, self.last_fov) {
            (Some((position, rotation)), Some(fov)) if eye_views.len() >= 2 => {
                let rig = crate::renderer::bench::BenchRig {
                    offset: glam::Vec3::ZERO,
                    yaw: 0.0,
                    head_position: position,
                    head_rotation: rotation,
                };
                crate::renderer::bench::pin_xr_views(&mut eye_views, located, &rig);
                if !located {
                    eye_views[0].fov = fov[0];
                    eye_views[1].fov = fov[1];
                }
            }
            _ if located => {}
            _ => {
                self.swapchain.release_image()?;
                return Ok(Vec::new());
            }
        }

        // DIAGNOSIS: draw and submit every view from a pose the head is NOT
        // at, so the compositor has to correct the difference with our depth
        // and motion vectors -- head motion, on a headset lying still on a
        // desk. 128: 5 cm to the right; 256: swaying +-5 cm at half a hertz.
        // See `Levers::space_warp_debug`.
        let sway_dbg = self.levers.space_warp_debug;
        if sway_dbg & (128 | 256) != 0 {
            let dx = if sway_dbg & 256 != 0 {
                0.05 * (self.started_at.elapsed().as_secs_f32() * std::f32::consts::PI).sin()
            } else {
                0.05
            };
            for v in eye_views.iter_mut() {
                v.pose.position.x += dx;
            }
        }

        // APPLICATION SPACEWARP: this frame's motion and depth images, when
        // the lever asks for them -- only now, with the views located, so no
        // early return above can leave them held. See `space_warp`.
        let warp_world_to_player = glam::Mat4::from_quat(glam::Quat::from_rotation_y(self.player.yaw).inverse())
            * glam::Mat4::from_translation(-self.player.offset);
        let warp_view_proj: [glam::Mat4; 2] = std::array::from_fn(|i| {
            let v = &eye_views[i.min(eye_views.len() - 1)];
            Camera::gl_to_wgpu_ndc(Camera::xr_projection(v.fov, crate::renderer::space_warp::NEAR_Z, crate::renderer::space_warp::FAR_Z))
                * Camera::xr_view(v.pose)
        });
        if let Some(sw) = self.space_warp.as_mut() {
            sw.acquired = None;
            if self.levers.space_warp {
                let m = sw.motion.acquire_image()? as usize;
                sw.motion.wait_image(xr::Duration::INFINITE)?;
                let d = sw.depth.acquire_image()? as usize;
                sw.depth.wait_image(xr::Duration::INFINITE)?;
                sw.acquired = Some((m, d));
            }
        }

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
        // Terrain indices were rebased into the shared solid buffer, so the
        // chunk offsets have to move with them. Using the raw offsets here
        // would draw whatever happened to sit at that position in the combined
        // buffer -- cuboids, or nothing.
        let solid_chunks: Vec<crate::renderer::shadow::CasterChunk> = match terrain_range {
            Some((index_start, _)) => terrain_chunks
                .iter()
                .map(|c| crate::renderer::shadow::CasterChunk {
                    first_index: c.first_index + index_start,
                    ..*c
                })
                .collect(),
            None => Vec::new(),
        };
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
            // Back into WORLD space to sample the occlusion map. Mesh positions
            // are in the player's frame -- the inverse of the same
            // `yaw_inv * (world - offset)` every other bit of geometry gets --
            // and the baked map is indexed by world footprint, so sampling it
            // with a player-frame position would make an avatar's brightness
            // depend on where the player happened to be standing.
            let world = glam::Quat::from_rotation_y(self.player.yaw)
                * instance.mesh.position
                + self.player.offset;
            let sky_vis = self.sky_visibility_at(world.x, world.z);
            instance.model.upload_full(
                &self.wgpu_queue,
                instance.mesh.model_matrix(),
                sky_vis,
                instance.emissive_drive,
            );
            let lightmap_bg = self.mesh_lightmap_bg(instance.lightmap_key);
            push_mesh_draws(
                instance, lightmap_bg, &mut mesh_draws, &mut skinned_draws, &mut layered_draws,
            );
            if instance.mesh.skin.is_none() {
                for prim in instance.mesh.primitives.iter().filter(|p| p.casts_shadow) {
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
        // Skinned casters, from the primitives the lit pass already gathered.
        // Derived rather than collected separately so a character can never be
        // drawn in one pass and missing from the other -- which would look like
        // a shadow that belongs to nobody.
        let mut skinned_casters: Vec<crate::renderer::shadow::ShadowSkinnedDraw> = skinned_draws
            .iter()
            .map(|(model_bg, _tex, joint_bg, vb, ib, count)| {
                (*vb, *ib, *count, *model_bg, *joint_bg)
            })
            .collect();

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
            // HIDDEN FROM ITS OWNER, NOT FROM THE SUN. A mirror-only mesh is
            // the player's own head: left out of their view so they do not see
            // the inside of it, but it is still there, and it still casts. Left
            // out of the casters, the player's shadow had no head (headset,
            // 2026-09-23).
            if instance.mesh.skin.is_none() {
                for prim in instance.mesh.primitives.iter().filter(|p| p.casts_shadow) {
                    shadow_casters.push((
                        &prim.vertex_buffer,
                        &prim.index_buffer,
                        prim.indices.len() as u32,
                        &instance.model.bind_group,
                    ));
                }
            }
        }
        skinned_casters.extend(
            mirror_only_skinned_draws
                .iter()
                .map(|(model_bg, _tex, joint_bg, vb, ib, count)| (*vb, *ib, *count, *model_bg, *joint_bg)),
        );

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

        // EYE ADAPTATION: meter what the player is looking at, from the probe
        // of the room they stand in, and ease the exposure toward it. The
        // head and gaze go back to WORLD space, where the probes were baked.
        let post = {
            let now = std::time::Instant::now();
            let dt = self
                .last_frame_at
                .replace(Some(now))
                .map(|t| now.duration_since(t).as_secs_f32().min(0.25))
                .unwrap_or(0.0);
            let yaw = glam::Quat::from_rotation_y(self.player.yaw);
            let o = eye_views[0].pose.orientation;
            let gaze = yaw * (glam::Quat::from_xyzw(o.x, o.y, o.z, o.w) * glam::Vec3::NEG_Z);
            let head_world = yaw * head + self.player.offset;
            let mut eye = self.eye.borrow_mut();
            let metered = eye.meter(head_world, gaze);
            let auto = eye.update(metered, dt);
            if self.shadow_diag_frames.get() % 120 == 0 {
                log::info!("EXPOSURE meter {metered:.4} -> x{auto:.2} (auto {})", self.auto_exposure);
            }
            crate::renderer::uniforms::PostUpload {
                exposure: self.post.exposure * if self.auto_exposure { auto } else { 1.0 },
                terrain_detail_distance: self.levers.terrain_detail_distance,
                ..self.post
            }
        };
        // The `perf_ab` NoShadows phase switches both off for its window.
        // The A/B schedule, compiled in (`perf_ab::ENABLED`) or asked for from
        // the headset (`Levers::ab_cycle`). Baseline otherwise.
        let ab_phase = if crate::renderer::perf_ab::ENABLED || self.levers.ab_cycle {
            crate::renderer::perf_ab::Phase::cycle_phase(self.perf_windows)
        } else {
            crate::renderer::perf_ab::Phase::Baseline
        };
        // EVERYTHING THIS FRAME SWITCHES: the lever file, with the schedule's
        // one extra switch on top. See `levers`. Every feature below reads
        // this, never the phase, so a lever and a phase cannot disagree.
        let fx = self.levers.clone().with_phase(ab_phase);
        let no_shadow_phase = !fx.shadows;
        let want_sun = self.shadow_quality != ShadowQuality::Off && !no_shadow_phase;
        let want_spot = self.shadow_quality == ShadowQuality::SunAndSpot && !no_shadow_phase;

        // Trim to the budget by INFLUENCE before anything else looks at the
        // list. Once, and here, because the shadow layers below are indices
        // INTO this list -- reordering after they are chosen would point each
        // spot at another light's shadow map.
        // THE SKY'S SUN, joined to the frame's lights unless the scene brought
        // its own. See `Sky::sun` and `lights::sky_sun_light`: it is a MIXED
        // light -- baked into the brushes, shaded live on everything else.
        let player_to_world = glam::Mat4::from_rotation_translation(
            glam::Quat::from_rotation_y(self.player.yaw),
            self.player.offset,
        );
        let sky_sun = crate::renderer::lights::sky_sun_light(
            self.sky.sun.as_ref(),
            lights,
            glam::Quat::from_rotation_y(self.player.yaw).inverse(),
        );
        let with_sky_sun: Vec<Light>;
        let lights: &[Light] = match sky_sun {
            Some(sun) => {
                with_sky_sun = lights.iter().copied().chain(std::iter::once(sun)).collect();
                &with_sky_sun
            }
            None => lights,
        };
        // `Levers::stationary_lights` off leaves the stationary lamps out --
        // measurement only: their light is in no lightmap.
        let without_stationary: Vec<Light>;
        let lights: &[Light] = if fx.stationary_lights {
            lights
        } else {
            without_stationary = lights.iter().copied().filter(|l| l.mask_channel.is_none()).collect();
            &without_stationary
        };
        let ranked_idx = crate::renderer::lights::rank_for_budget_indices(
            lights,
            crate::renderer::lights::MAX_LIGHTS,
        );
        let source_lights = lights;
        let ranked: Vec<Light> = ranked_idx.iter().map(|&i| source_lights[i]).collect();
        let lights: &[Light] = if !fx.direct_lights { &[] } else { &ranked };

        let sun = want_sun
            .then(|| lights.iter().find(|l| l.kind == crate::renderer::LightKind::Directional))
            .flatten();
        // Spots take shadow layers in the same influence order. It used to be
        // scene order, so which of two identical lamps cast a shadow came down
        // to which was authored first.
        //
        // HELD FROM FRAME TO FRAME, not re-chosen. Influence is measured from
        // the PLAYER, so simply taking the best four every frame means the set
        // changes as they walk -- and with more spots in a room than slots,
        // shadows appear and disappear with movement. On the headset that read
        // as one side of the avatar's hand unshadowed when it should not have
        // been, and the wall spotlight in the back corner casting none
        // (2026-09-18). Each frame's choice was individually right; the defect
        // only existed ACROSS frames. See `lights::spot_shadow_slots`.
        //
        // The incumbents are named by their index in the list the CALLER handed
        // in, which is the only name that survives the ranking above.
        let spot_indices: Vec<usize> = if want_spot && !lights.is_empty() {
            let spot_scores: Vec<(usize, f32)> = ranked_idx
                .iter()
                .copied()
                .filter(|&src| source_lights[src].kind == crate::renderer::LightKind::Spot)
                // A STATIONARY lamp's shadows are baked into its mask channel;
                // a shadow map slot spent on it would draw the same shadow a
                // second time, every frame.
                .filter(|&src| source_lights[src].mask_channel.is_none())
                .map(|src| (src, crate::renderer::lights::influence_score(&source_lights[src])))
                .collect();
            let chosen = crate::renderer::lights::spot_shadow_slots(
                &spot_scores,
                &self.shadow_spot_incumbents.borrow(),
                crate::renderer::shadow::MAX_SPOT_SHADOWS,
                crate::renderer::lights::SHADOW_SLOT_MARGIN,
            );
            self.shadow_spot_incumbents.borrow_mut().clone_from(&chosen);
            // Back to positions in the RANKED list, which is what the uniform
            // and the shadow layers are indexed by.
            chosen
                .iter()
                .filter_map(|&src| ranked_idx.iter().position(|&r| r == src))
                .collect()
        } else {
            Vec::new()
        };

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

        // Uploaded HERE rather than earlier in the frame, because each light has
        // to be told which shadow layer it casts into -- and that is not known
        // until the spots have been assigned layers just above.
        // The baked lamps ride behind the live ones for the surfaces that
        // have no lightmap. See `lights::append_baked`. Dropped with the live
        // ones when `perf_ab` measures a frame without direct light.
        let frame_lights = if !fx.direct_lights {
            Vec::new()
        } else {
            crate::renderer::lights::append_baked(lights, &self.baked_lights, crate::renderer::lights::MAX_LIGHTS)
        };
        self.lights_uniform.set_culling(fx.light_culling);
        self.lights_uniform.upload_frame_split(
            &self.wgpu_queue,
            &frame_lights,
            lights.len(),
            &spot_indices,
            sky_sun.is_some(),
        );

        // THE SKY SUN'S SHADOW IS DRAWN ONCE, not every frame.
        //
        // Nothing it shadows by moves: the sun is fixed and so is the level.
        // So the map is built in WORLD space over the level's bounds and only
        // redrawn when the geometry changes, and each frame merely re-expresses
        // it in the player's frame -- the render space is the world moved by
        // the rig, and `player_to_world` undoes exactly that. The head-following
        // box this replaces for the sky sun redrew every brush every frame and
        // spent half its resolution behind the player.
        //
        // Static casters only -- the level and the ground. A moving object
        // does not cast the sun's shadow yet; the baked level never shows it
        // anyway, because the sun on brushes comes from the lightmap.
        let static_sun = sun.filter(|_| sky_sun.is_some()).map(|l| {
            let world_dir = glam::Quat::from_rotation_y(self.player.yaw) * l.direction;
            let signature = (
                brushes.map(|(v, i)| (v.len(), i.len())).unwrap_or((0, 0)),
                terrain.map(|(v, i)| (v.len(), i.len())).unwrap_or((0, 0)),
                world_dir.to_array().map(f32::to_bits),
            );
            let mut cache = self.static_sun_shadow.borrow_mut();
            let stale = cache.as_ref().is_none_or(|c| c.signature != signature);
            if stale {
                let world_view_proj = crate::renderer::lights::static_sun_matrix(
                    world_dir,
                    brushes.map(|(v, _)| v.iter().map(|b| b.position)),
                    terrain.map(|(v, _)| v.iter().map(|t| t.position)),
                    player_to_world,
                );
                *cache = Some(crate::renderer::lights::StaticSunShadow { signature, world_view_proj });
            }
            (cache.as_ref().unwrap().world_view_proj * player_to_world, stale)
        });
        // The moving-objects sun map: a small box under the player's head,
        // built in WORLD space and snapped to its own texel grid there, then
        // carried into the player's frame like the static map. Snapping in the
        // player's frame instead would re-grid the map every time they walked
        // or turned, and every moving shadow edge would crawl.
        let dynamic_sun = static_sun.and(sun).filter(|_| fx.sun_dynamic).map(|l| {
            let world_dir = glam::Quat::from_rotation_y(self.player.yaw) * l.direction;
            let head_world = player_to_world.transform_point3(head);
            crate::renderer::lights::dynamic_sun_matrix(world_dir, head_world) * player_to_world
        });

        let shadow = crate::renderer::uniforms::ShadowUpload {
            sun_view_proj: match static_sun {
                Some((m, _)) => m,
                None => sun
                    .map(|l| {
                        crate::renderer::shadow::directional_light_matrix(l.direction, focus, sun_radius)
                    })
                    .unwrap_or(glam::Mat4::IDENTITY),
            },
            spot_view_proj: {
                let mut m = [glam::Mat4::IDENTITY; crate::renderer::shadow::MAX_SPOT_SHADOWS];
                for (layer, &i) in spot_indices.iter().enumerate() {
                    let l = &lights[i];
                    m[layer] = crate::renderer::shadow::spot_light_matrix(
                        l.position, l.direction, l.cone_angle_deg, l.range,
                    );
                }
                m
            },
            sun_enabled: sun.is_some(),
            spot_count: spot_indices.len() as u32,
            sun_dynamic_view_proj: dynamic_sun.unwrap_or(glam::Mat4::IDENTITY),
            sun_dynamic_enabled: dynamic_sun.is_some(),
        };

        // A static sun map is recorded only when it went stale.
        let record_sun = shadow.sun_enabled && static_sun.is_none_or(|(_, stale)| stale);
        if record_sun || shadow.sun_dynamic_enabled || shadow.spot_count > 0 {
            let solid_caster = (!solid_idx.is_empty())
                .then_some((&solid_vb, &solid_ib, solid_idx.len() as u32));
            let brush_caster = brush_buffers
                .as_ref()
                .map(|(vb, ib, count)| (vb, ib, *count));
            // A missing shadow has four separate causes -- no pass ran, the
            // caster was not in the list, the list was empty, or the depth
            // landed somewhere else -- and they are indistinguishable by
            // looking at the wall. This separates the first three.
            let mut drawn = 0u32;
            self.shadow_diag_frames.set(self.shadow_diag_frames.get() + 1);
            let diag = self.shadow_diag_frames.get() % 120 == 0;
            // WHICH spots hold the four slots, reported the moment the set
            // changes rather than on the 120-frame cadence.
            //
            // `test_room` has five lights and `MAX_SPOT_SHADOWS` is four, so
            // one light is always without a shadow and which one can change as
            // the player moves. A periodic log cannot distinguish "stable" from
            // "swapping between samples", and a swap is exactly the thing worth
            // knowing about: it makes a shadow appear and disappear wholesale.
            // Logging only on CHANGE means silence is itself the measurement.
            {
                let mut last = self.shadow_slot_log.borrow_mut();
                if *last != spot_indices {
                    log::info!(
                        "SHADOWSLOTS changed: {:?} -> {:?} (of {} lights, {} slots)",
                        *last,
                        spot_indices,
                        source_lights.len(),
                        crate::renderer::shadow::MAX_SPOT_SHADOWS,
                    );
                    last.clone_from(&spot_indices);
                }
            }
            if diag {
                log::info!(
                    "SHADOWDIAG sun={} spots={} solid_idx={} brush_idx={} rigid_casters={} skinned_casters={}",
                    shadow.sun_enabled,
                    shadow.spot_count,
                    solid_caster.map(|(_, _, n)| n).unwrap_or(0),
                    brush_caster.map(|(_, _, n)| n).unwrap_or(0),
                    shadow_casters.len(),
                    skinned_casters.len(),
                );
            }
            let mut encoder = self
                .wgpu_device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("shadow_encoder"),
                });
            if record_sun {
                // The ground only, for a static map: the solid buffer also
                // holds cuboids, which can move. `record` draws only the
                // terrain chunks when it is given any, so without them the
                // buffer is left out rather than frozen into the map.
                let solid_caster = if static_sun.is_some() && solid_chunks.is_empty() {
                    None
                } else {
                    solid_caster
                };
                let no_meshes: [crate::renderer::shadow::ShadowMeshDraw; 0] = [];
                let no_skinned: [crate::renderer::shadow::ShadowSkinnedDraw; 0] = [];
                let (mesh_casters, skinned) = if static_sun.is_some() {
                    (&no_meshes[..], &no_skinned[..])
                } else {
                    (&shadow_casters[..], &skinned_casters[..])
                };
                self.shadow_map.upload_light(
                    &self.wgpu_queue,
                    crate::renderer::shadow::ShadowKind::Sun,
                    shadow.sun_view_proj,
                );
                drawn += self.shadow_map.record(
                    &mut encoder,
                    crate::renderer::shadow::ShadowKind::Sun,
                    solid_caster,
                    brush_caster,
                    mesh_casters,
                    skinned,
                    &solid_chunks,
                    shadow.sun_view_proj,
                );
                if static_sun.is_some() {
                    log::info!("SUNSHADOW static map recorded ({drawn} indices)");
                }
            }
            // The sun's shadow of MOVING things, every frame, into its own
            // small map. Meshes and skinned characters only: the level is in
            // the brushes' baked mask and in the static map already.
            if shadow.sun_dynamic_enabled {
                self.shadow_map.upload_light(
                    &self.wgpu_queue,
                    crate::renderer::shadow::ShadowKind::SunDynamic,
                    shadow.sun_dynamic_view_proj,
                );
                drawn += self.shadow_map.record(
                    &mut encoder,
                    crate::renderer::shadow::ShadowKind::SunDynamic,
                    None,
                    None,
                    &shadow_casters,
                    &skinned_casters,
                    &[],
                    shadow.sun_dynamic_view_proj,
                );
            }
            // ONE pass for every spot, filling its own tile of the shared
            // atlas. This used to be a pass each, and that was the cost: on a
            // tile GPU a pass is a tile load/store cycle whatever is in it, and
            // culling 95.5% of the caster geometry gave back only 1.8 ms of the
            // 3.2 ms three spots cost.
            for layer in 0..shadow.spot_count as usize {
                self.shadow_map.upload_light(
                    &self.wgpu_queue,
                    crate::renderer::shadow::ShadowKind::Spot(layer),
                    shadow.spot_view_proj[layer],
                );
            }
            if shadow.spot_count > 0 {
                drawn += self.shadow_map.record_spots(
                    &mut encoder,
                    shadow.spot_count as usize,
                    &shadow.spot_view_proj,
                    solid_caster,
                    brush_caster,
                    &shadow_casters,
                    &skinned_casters,
                    &solid_chunks,
                );
            }
            if diag {
                // The number that says whether culling is doing anything: the
                // baseline is every caster redrawn once per pass, and `drawn`
                // is what actually reached a depth buffer.
                // Spots are now ONE pass between them, not one each.
                let passes = u32::from(shadow.spot_count > 0) + u32::from(record_sun);
                let uncalled = (solid_caster.map(|(_, _, n)| n).unwrap_or(0)
                    + brush_caster.map(|(_, _, n)| n).unwrap_or(0))
                    * passes;
                log::info!(
                    "SHADOWDIAG culled: drew {drawn} of {uncalled} indices over {passes} pass(es), {} chunks",
                    solid_chunks.len(),
                );
            }
            self.wgpu_queue.submit(Some(encoder.finish()));
        }

        // Whether the eye pass will composite anything that has to depth-test
        // against the world. Only these two read the scene depth, and only they
        // make a multisampled depth buffer worth storing.
        // A reflective BRUSH needs the offscreen copy for the same reason a
        // reflective cuboid does: it samples the finished scene. Gated on the
        // materials actually being smooth, so a level built from rock and grass
        // keeps the cheaper straight-to-swapchain path -- measured at about
        // 0.9 ms.
        let reflective_brushes = self.brush_materials.reflective && brush_geometry.is_some();
        // The A/B schedule. `Baseline` whenever `perf_ab::ENABLED` is off.
        // THROUGH the policy, so the renderer cannot drift from what
        // `scene_pass_plan`'s tests pin. With screen-space reflections off on
        // the headset, a reflective material no longer forces the offscreen
        // copy; only a planar mirror still does.
        let needs_scene_depth = crate::renderer::scene_pass_plan::needs_scene_readback(
            mirror_quad.is_some(),
            reflective_brushes || solid_ranges.iter().any(|(_, _, _, r)| *r > 0.0),
            // The runtime switch, not the constant: the constant is only the
            // default it starts from. See `set_screen_space_reflections`.
            self.screen_space_reflections,
        );
        // Forcing the direct path is the ONE override of this decision, and it
        // goes through the same value the scene and eye passes both read, so
        // they cannot disagree about it even while it is being overridden.
        let needs_scene_depth = needs_scene_depth && !fx.direct_path;
        // ONE decision, read by both the scene pass and the eye pass.
        //
        // They have to agree. If the scene pass draws straight into the
        // swapchain and the eye pass still runs, its first act is to blit the
        // offscreen texture -- now a frame stale and never written this frame --
        // over the top of everything just rendered. That is a black or frozen
        // eye, and it is the kind of drift a later edit to one site introduces
        // silently. Deriving both from the same value is what makes it
        // unrepresentable.
        // A multiview scene pass draws into its own layered target, so the
        // eye pass copies each eye across even with nothing to sample back.
        let stereo_frame = self.multiview_scene && self.stereo_pipelines.is_some();
        let plan = crate::renderer::scene_pass_plan::ScenePassPlan::for_frame_with(needs_scene_depth, stereo_frame);

        // ONCE PER FRAME, not per eye: takes in whatever probes the stream's
        // worker finished, and opens the frame in which layers the eyes use
        // cannot be evicted. See `probe_stream::LayerPool`.
        if let Some(stream) = self.probe_stream.borrow_mut().as_mut() {
            stream.begin_frame();
        }

        // SPACEWARP: which meshes the motion pass draws, and every draw's two
        // clip transforms for both eyes, into the ring once. A mesh's previous
        // clip is the previous camera times its previous model matrix -- both
        // in the previous player frame, so nothing else is needed. See
        // `space_warp`.
        let warp_meshes: Vec<(&MeshInstance, glam::Mat4, glam::Mat4)> = match self.space_warp.as_ref() {
            Some(sw) if sw.acquired.is_some() => meshes
                .iter()
                .take(crate::renderer::space_warp::MAX_SLOTS as usize / 2 - 1)
                .map(|m| {
                    let model = m.mesh.model_matrix();
                    let prev = sw.prev_models.get(&m.model.buffer).copied().unwrap_or(model);
                    (m, model, prev)
                })
                .collect(),
            _ => Vec::new(),
        };
        let warp_per_eye = 1 + warp_meshes.len() as u32;
        if let Some(sw) = self.space_warp.as_ref().filter(|sw| sw.acquired.is_some()) {
            use crate::renderer::space_warp::{previous_clip, MotionCamera, SLOT_STRIDE};
            let stride = SLOT_STRIDE as usize;
            let size = std::mem::size_of::<MotionCamera>();
            let mut bytes = vec![0u8; stride * 2 * warp_per_eye as usize];
            for eye in 0..2usize {
                let curr = warp_view_proj[eye];
                let prev_view_proj = sw.prev.map_or(curr, |(vps, _)| vps[eye]);
                let world_prev = sw.prev.map_or(curr, |(vps, w2p)| previous_clip(vps[eye], w2p, warp_world_to_player));
                let dbg = self.levers.space_warp_debug;
                let params = [
                    if dbg & 2 != 0 { -1.0 } else { 1.0 },
                    if dbg & 4 != 0 { 0.0 } else { 1.0 },
                    0.0,
                    0.0,
                ];
                let mut put = |slot: usize, c: glam::Mat4, p: glam::Mat4| {
                    let cam = MotionCamera { curr: c.to_cols_array_2d(), prev: p.to_cols_array_2d(), params };
                    bytes[slot * stride..slot * stride + size].copy_from_slice(bytemuck::bytes_of(&cam));
                };
                let base = eye * warp_per_eye as usize;
                put(base, curr, world_prev);
                for (i, (_, model, prev)) in warp_meshes.iter().enumerate() {
                    put(base + 1 + i, curr * *model, prev_view_proj * *prev);
                }
            }
            self.wgpu_queue.write_buffer(&sw.cameras, 0, &bytes);
        }

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
                self.uniform_buf.upload_scene(
                    &self.wgpu_queue, mirror_view_proj, mirror_eye, &shadow, &sky_upload,
                    &post, &self.player,
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
                            depth_slice: None,
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
                        pass.set_bind_group(2, self.brush_lightmap_bg(), &[]);
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
            // BOTH EYES' CAMERAS. A stereo scene pass cannot re-upload the
            // uniform between eyes -- it draws them in one go -- so both have
            // to be resident and the shader picks its own with
            // `@builtin(view_index)`. Index 0 is the left eye, matching the
            // OpenXR view array, the swapchain's layer order and the order
            // `view_index` counts in.
            let both_view_proj: [glam::Mat4; 2] = std::array::from_fn(|i| {
                let v = &eye_views[i];
                Camera::gl_to_wgpu_ndc(Camera::xr_projection(v.fov, 0.03, 1000.0))
                    * Camera::xr_view(v.pose)
            });
            let both_cam_pos: [glam::Vec3; 2] = std::array::from_fn(|i| {
                let p = eye_views[i].pose.position;
                glam::Vec3::new(p.x, p.y, p.z)
            });

            // WHICH PROBES OCCUPY THE SHADER'S SLOTS THIS FRAME.
            //
            // The cube array holds every probe the level baked; the loop can
            // only afford `MAX_PROBES` of them. Which ones matter depends on
            // where the player is standing, so it is decided here rather than
            // at load.
            //
            // IN WORLD SPACE, because probe boxes are. Everything else in this
            // frame is in the PLAYER's frame -- `yaw_inv * (world - offset)` --
            // so both the camera and its frustum have to be lifted back out of
            // it before they can be compared against a box that was baked
            // against walls that do not move. Comparing the two frames directly
            // is the bug that made probe boxes slide as the player walked.
            let resident = if self.probe_volumes.is_empty() {
                None
            } else {
                let yaw = glam::Quat::from_rotation_y(self.player.yaw);
                let player_world = yaw * cam_pos + self.player.offset;
                let world_to_player = glam::Mat4::from_quat(yaw.inverse())
                    * glam::Mat4::from_translation(-self.player.offset);
                let planes = crate::renderer::shadow::frustum_planes(eye_view_proj * world_to_player);
                let mut upload = crate::renderer::uniforms::select_resident_probes(
                    &self.probe_volumes,
                    player_world,
                    |min, max| crate::renderer::shadow::aabb_in_frustum(&planes, min, max),
                );
                // Residency names PROBES; brightness and room follow the probe,
                // and only then does the stream turn each into a cube layer.
                upload.fill_brightness(&self.probe_brightness);
                upload.fill_volumes(&self.probe_rooms);
                // The doorways around the player, nearest first, that open
                // onto a room with a photograph in this frame.
                upload.set_portals(&self.probe_portals, player_world, &upload.volumes());
                // And what stands in those rooms, for the reflection trace.
                upload.set_proxies(&self.probe_proxies, player_world, &upload.volumes());
                upload.set_proxy_fields(&self.proxy_field_slots);
                // And the outdoors: which room it is, its sky, its ground.
                let sky_layer = self.probe_stream.borrow().as_ref().and_then(|s| s.sky_layer());
                let ground = if fx.ground_trace { self.ground_placement } else { None };
                upload.set_outdoors(self.probe_outdoor_volume, sky_layer, ground);
                let building_layer = self.probe_stream.borrow().as_ref().and_then(|s| s.building_layer());
                upload.set_buildings(&self.probe_buildings, building_layer);
                // perf_ab: every slot its own room (no two-photograph blend), or
                // no doorways. Measurement only; see `perf_ab::Phase`.
                if !fx.probe_blend {
                    for slot in 0..upload.count as usize {
                        upload.set_volume(slot, 1_000_000 + slot as u32);
                    }
                }
                if !fx.portals {
                    upload.portal_count = 0;
                }
                upload.no_trace = !fx.probe_trace;
                if !fx.reflection_proxies {
                    upload.proxy_count = 0;
                }
                Some(upload)
            };
            let resident = match (resident, self.probe_stream.borrow_mut().as_mut()) {
                (Some(mut upload), Some(stream)) => {
                    stream.resolve(&self.wgpu_queue, &mut upload);
                    Some(upload)
                }
                (resident, _) => resident,
            };

            let no_probes = crate::renderer::uniforms::ProbeUpload::default();
            // A STEREO SCENE PASS NEEDS BOTH CAMERAS RESIDENT AT ONCE; every
            // other pass in the frame is per eye and wants this eye's. They
            // cannot share one upload, so the stereo one is written here, the
            // scene pass is submitted on its own below, and the per-eye upload
            // follows it. See the note at that submit.
            let stereo = self.multiview_scene && self.stereo_pipelines.is_some();
            // WHAT THIS PASS CAN SEE, for terrain chunk culling further down.
            //
            // BOTH eyes when the pass is stereo. A multiview pass draws the two
            // eyes together, so culling against one eye's frustum would drop
            // geometry the other eye can see -- a hole in one eye only, which
            // is both horrible to look at and easy to miss on a monitor.
            //
            // Player frame, deliberately without the `world_to_player` the
            // probe residency test above needs: a chunk's bounds were built
            // from the same vertices the draw uses, so both are already in the
            // frame the view matrix expects. The probe boxes are the odd ones
            // out, being baked in world space.
            let mut terrain_drawn = 0u32;
            let mut terrain_culled = 0u32;
            let cull_planes: Vec<[glam::Vec4; 6]> = if stereo {
                both_view_proj
                    .iter()
                    .map(|vp| crate::renderer::shadow::frustum_planes(*vp))
                    .collect()
            } else {
                vec![crate::renderer::shadow::frustum_planes(eye_view_proj)]
            };
            // DOORWAY CULLING of what lies outside the building. From inside a
            // closed room the terrain is visible only through doorways, so it
            // is drawn only where one of these frusta -- the view narrowed to a
            // doorway, walked on through closed rooms -- reaches it. `None`
            // (no culling) unless EVERY eye this pass draws is inside a closed
            // room. See `portal_cull`.
            let outdoor_frusta: Option<Vec<[glam::Vec4; 6]>> = if fx.portal_culling && !self.cull_rooms.is_empty() {
                let yaw = glam::Quat::from_rotation_y(self.player.yaw);
                let world_to_player =
                    glam::Mat4::from_quat(yaw.inverse()) * glam::Mat4::from_translation(-self.player.offset);
                let eyes: Vec<(glam::Vec3, glam::Mat4)> = if stereo {
                    both_cam_pos.iter().copied().zip(both_view_proj.iter().copied()).collect()
                } else {
                    vec![(cam_pos, eye_view_proj)]
                };
                let mut all = Some(Vec::new());
                for (pos, vp) in eyes {
                    let eye_world = yaw * pos + self.player.offset;
                    match crate::renderer::portal_cull::outdoor_frusta(
                        eye_world,
                        vp * world_to_player,
                        vp,
                        &self.cull_rooms,
                        &self.probe_portals,
                    ) {
                        Some(f) => {
                            if let Some(a) = all.as_mut() {
                                a.extend(f);
                            }
                        }
                        None => all = None,
                    }
                }
                all
            } else {
                None
            };
            // A chunk standing INSIDE a closed room (ground a room was carved
            // into) is seen from within it, not through a doorway: always drawn.
            let player_to_world = glam::Mat4::from_translation(self.player.offset)
                * glam::Mat4::from_quat(glam::Quat::from_rotation_y(self.player.yaw));
            let in_closed_room = |c: &crate::renderer::shadow::CasterChunk| {
                let corners = (0..8).map(|i| {
                    player_to_world.transform_point3(glam::Vec3::new(
                        if i & 1 == 0 { c.min.x } else { c.max.x },
                        if i & 2 == 0 { c.min.y } else { c.max.y },
                        if i & 4 == 0 { c.min.z } else { c.max.z },
                    ))
                });
                let (lo, hi) = corners.fold(
                    (glam::Vec3::splat(f32::INFINITY), glam::Vec3::splat(f32::NEG_INFINITY)),
                    |(lo, hi), p| (lo.min(p), hi.max(p)),
                );
                self.cull_rooms.iter().any(|r| r.closed && lo.cmplt(r.max).all() && r.min.cmplt(hi).all())
            };
            // WHETHER A TERRAIN CHUNK IS DRAWN this eye: in the view, and, from
            // inside a closed room, seen through a doorway. One rule for the
            // scene pass and the probe pass, which must draw the same ground.
            let terrain_chunk_visible = |c: &crate::renderer::shadow::CasterChunk| {
                // THE TESTED HELPER, not a second copy of the rule. An inline
                // `any` here would be the shipped behaviour while the tests
                // exercised something else that merely looked the same.
                let through_a_doorway = match &outdoor_frusta {
                    None => true,
                    Some(f) => crate::renderer::shadow::chunk_seen(c, f) || in_closed_room(c),
                };
                crate::renderer::shadow::chunk_seen(c, &cull_planes) && through_a_doorway
            };
            let probes_arg =
                if !fx.probes { Some(&no_probes) } else { resident.as_ref() };
            if stereo && eye == 0 {
                self.uniform_buf.upload_scene_stereo(
                    &self.wgpu_queue, both_view_proj, both_cam_pos, &shadow, &sky_upload,
                    &post, &self.player, probes_arg,
                );
            } else {
                self.uniform_buf.upload_scene_with_probes(
                    &self.wgpu_queue, eye_view_proj, cam_pos, &shadow, &sky_upload, &post,
                    &self.player,
                    // `Some(empty)`, not `None`: `None` means "use the level's
                    // probes", which would leave them on and label it off.
                    probes_arg,
                );
            }
            self.ssr_camera_uniform.upload(&self.wgpu_queue, eye_view_proj, cam_pos);
            // THE BRUSHES' REFLECTIONS AT HALF RESOLUTION, in their own pass
            // before the scene pass reads them. See `brush_pipeline::probe_pass`.
            // The diagnostic views keep the per-pixel shader, which is the only
            // one that paints them. A stereo scene pass needs the two-eye pass,
            // which the device may have refused (`StereoProbePass`).
            let probe_pass = fx.half_res_reflections
                && fx.probes
                && (!stereo || self.stereo_probe.is_some())
                && self.debug_view == crate::renderer::brush_pipeline::DebugView::Off
                && brush_buffers.is_some();
            // Its pipelines and target: this eye's, or both eyes' at once.
            let (probe_pipeline, probe_reader, probe_target) = match (&self.stereo_probe, stereo) {
                (Some(sp), true) => (&sp.pass, &sp.reader, &sp.target),
                _ => (&self.brush_probe_pass_pipeline, &self.brush_probe_reader_pipeline, &self.probe_pass_targets[eye]),
            };
            // ITS SECONDARY LOOKUPS DEFERRED to a compute pass over just the
            // texels that need them, in the single-eye pass. See `probe_fixup`.
            let deferred_lookups = fx.deferred_reflection_lookups && !stereo;
            // THE GROUND'S REFLECTION IN THE PROBE PASS TOO, read back in the
            // scene pass as the brushes' is. See `TerrainPipeline::new_probe_pass`.
            let terrain_in_probe_pass =
                probe_pass && deferred_lookups && fx.terrain_probe_pass && terrain_range.is_some();
            let probe_pipeline = if deferred_lookups { &self.brush_probe_pass_deferred_pipeline } else { probe_pipeline };

            {
                let mut encoder = self.wgpu_device.create_command_encoder(
                    &wgpu::CommandEncoderDescriptor { label: Some("ssr_scene") },
                );
                // Once a frame when stereo, like the scene pass it feeds.
                if probe_pass && (!stereo || eye == 0) {
                    if let Some((vb, ib, count)) = &brush_buffers {
                        if deferred_lookups {
                            self.probe_fixups.clear(&mut encoder);
                        }
                        let t = probe_target;
                        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("probe_pass"),
                            // Both layers when stereo; see the scene pass.
                            multiview_mask: if stereo {
                                crate::renderer::multiview::STEREO_VIEW_MASK
                            } else {
                                None
                            },
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: &t.color_view,
                                depth_slice: None,
                                resolve_target: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                                    store: wgpu::StoreOp::Store,
                                },
                            })],
                            // Kept: the scene pass reads it to match its pixels
                            // to this pass's texels by depth.
                            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                                view: &t.depth_view,
                                depth_ops: Some(wgpu::Operations {
                                    load: wgpu::LoadOp::Clear(1.0),
                                    store: wgpu::StoreOp::Store,
                                }),
                                stencil_ops: None,
                            }),
                            // Its own slots, `probe_l`/`probe_r`, after the
                            // eight the other passes address by index.
                            timestamp_writes: self.pass_timers.as_ref().and_then(|t| t.writes(8 + eye)),
                            ..Default::default()
                        });
                        pass.set_pipeline(&probe_pipeline.pipeline);
                        pass.set_bind_group(0, &self.uniform_buf.bind_group, &[]);
                        pass.set_bind_group(1, &self.brush_materials.bind_group, &[]);
                        pass.set_bind_group(2, self.brush_lightmap_bg(), &[]);
                        if deferred_lookups {
                            pass.set_bind_group(3, self.probe_fixups.pass_bind_group(), &[]);
                        }
                        pass.set_vertex_buffer(0, vb.slice(..));
                        pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                        pass.draw_indexed(0..*count, 0, 0..1);
                        if let (true, Some((index_start, count))) = (terrain_in_probe_pass, terrain_range) {
                            pass.set_pipeline(&self.terrain_probe_pass_pipeline.pipeline);
                            pass.set_bind_group(0, &self.uniform_buf.bind_group, &[]);
                            pass.set_bind_group(1, &self.terrain_material.bind_group, &[]);
                            pass.set_bind_group(3, self.probe_fixups.pass_bind_group(), &[]);
                            pass.set_vertex_buffer(0, solid_vb.slice(..));
                            pass.set_index_buffer(solid_ib.slice(..), wgpu::IndexFormat::Uint32);
                            if solid_chunks.is_empty() {
                                pass.draw_indexed(index_start..index_start + count, 0, 0..1);
                            } else {
                                for c in solid_chunks.iter().filter(|c| terrain_chunk_visible(c)) {
                                    pass.draw_indexed(c.first_index..c.first_index + c.index_count, 0, 0..1);
                                }
                            }
                        }
                        drop(pass);
                        if deferred_lookups {
                            self.probe_fixups.dispatch(&mut encoder, &self.uniform_buf.bind_group, &self.probe_fixup_targets[eye]);
                        }
                    }
                }
                // ONCE PER FRAME WHEN STEREO, not once per eye. The pass
                // covers both layers, so running it again on the second eye
                // would draw the whole scene twice into the same attachment --
                // which is the cost this exists to remove.
                if !stereo || eye == 0 {
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
                    // WHICH FAMILY OF PIPELINES THIS PASS DRAWS WITH, and
                    // therefore whether it covers one eye or both. False unless
                    // the switch is on AND the device built the stereo set, so
                    // a machine without MULTIVIEW can never reach that path.
                    let target = &self.scene_targets[eye];
                    // STRAIGHT INTO THE SWAPCHAIN when nothing will read the
                    // scene back.
                    //
                    // The offscreen copy exists so reflective solids and the
                    // mirror quad can SAMPLE the rendered scene. Nothing else
                    // wants it -- and no shipped level has either -- so for
                    // every scene we actually run, the eye pass that followed
                    // was a full-screen blit of a texture we had just finished
                    // drawing.
                    //
                    // On a tile GPU that is not a small waste. It is a
                    // full-resolution store out of tile memory, then a second
                    // render pass that loads all of it back and writes it
                    // again, per eye, per frame -- and a render pass is the
                    // expensive unit here, which this renderer has already
                    // measured once: collapsing three shadow passes into one
                    // atlas pass returned far more than culling 95% of the
                    // geometry did.
                    //
                    // It is also what stands between us and foveated
                    // rendering. FFR attaches a fragment density map to the
                    // SWAPCHAIN image; shading into a private texture and
                    // blitting means the density map only ever covers the
                    // blit, and the pass doing the actual work is untouched.
                    // Drawing the scene into the swapchain directly is what
                    // makes FFR apply to the shading it is supposed to cheapen.
                    //
                    // Same format, same size, same sample count, so this is a
                    // change of destination and not of pipelines.
                    let swap_view = &self.eye_targets[image_index][eye].view;
                    // THE LAYERED VIEWS WHEN STEREO. `D2Array` over both eyes
                    // is what makes this a multiview pass at all -- wgpu infers
                    // the view count from the attachment, so an ordinary
                    // per-eye view here would draw one eye with pipelines built
                    // for two and be refused.
                    //
                    // The stereo path never writes straight to the swapchain:
                    // that is one layer of the OpenXR image and the eye pass
                    // blits each eye across afterwards.
                    let stereo_views = stereo
                        .then(|| self.stereo_scene.as_ref())
                        .flatten();
                    let (color_view, resolve_target, color_store) = if let Some(st) = stereo_views {
                        match st.array_msaa_color_view.as_ref() {
                            Some(msaa) => (
                                msaa,
                                Some(&st.array_color_view),
                                wgpu::StoreOp::Discard,
                            ),
                            None => (&st.array_color_view, None, wgpu::StoreOp::Store),
                        }
                    } else {
                        match (target.msaa_color_view.as_ref(), plan.samples_scene_back()) {
                            // Multisampled: the resolve goes wherever the
                            // finished image is wanted. The samples never leave
                            // tile memory either way.
                            (Some(msaa), true) => (
                                msaa,
                                Some(&target.color_view),
                                wgpu::StoreOp::Discard,
                            ),
                            (Some(msaa), false) => (
                                msaa,
                                Some(swap_view),
                                wgpu::StoreOp::Discard,
                            ),
                            (None, true) => (&target.color_view, None, wgpu::StoreOp::Store),
                            (None, false) => (swap_view, None, wgpu::StoreOp::Store),
                        }
                    };
                    // The depth attachment has to match: layered when the
                    // colour is, or the pass has two different view counts.
                    let scene_depth_view = match stereo_views {
                        Some(st) => &st.array_depth_view,
                        None => &target.depth_view,
                    };
                    let depth_store = if needs_scene_depth {
                        wgpu::StoreOp::Store
                    } else {
                        wgpu::StoreOp::Discard
                    };

                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("ssr_scene_pass"),
                        // Both layers when stereo. wgpu checks this against the
                        // attachment: a mask that is not `(1 << layers) - 1` is
                        // a SELECTIVE multiview pass and needs a feature we do
                        // not have. See `multiview::STEREO_VIEW_MASK`.
                        multiview_mask: if stereo {
                            crate::renderer::multiview::STEREO_VIEW_MASK
                        } else {
                            None
                        },
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: color_view,
                            depth_slice: None,
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
                            view: scene_depth_view,
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Clear(1.0),
                                store: depth_store,
                            }),
                            stencil_ops: None,
                        }),
                        // Slot per EYE: both eyes run this, and one shared pair would report only the second.
                        timestamp_writes: self.pass_timers.as_ref().and_then(|t| t.writes(eye * 2)),
                        ..Default::default()
                    });

                    if fx.half_viewport {
                        pass.set_viewport(0.0, 0.0, (self.width / 2) as f32, (self.height / 2) as f32, 0.0, 1.0);
                    }

                    // THE BRUSHES' DEPTH FIRST, so nothing behind a wall or
                    // under a floor is shaded by what follows -- the terrain,
                    // meshes, and the brushes' own shading at LessEqual. See
                    // `BrushPipeline::new_depth_prepass`. Not under the
                    // diagnostic views, which draw brushes with pipelines of
                    // their own.
                    let prepass = if stereo { self.stereo_depth_prepass.as_ref() } else { Some(&self.brush_depth_prepass) };
                    if let (true, Some(prepass), Some((vb, ib, count))) = (
                        fx.depth_prepass && self.debug_view == crate::renderer::brush_pipeline::DebugView::Off,
                        prepass,
                        &brush_buffers,
                    ) {
                        pass.set_pipeline(&prepass.pipeline);
                        pass.set_bind_group(0, &self.uniform_buf.bind_group, &[]);
                        pass.set_bind_group(1, &self.brush_materials.bind_group, &[]);
                        pass.set_vertex_buffer(0, vb.slice(..));
                        pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                        pass.draw_indexed(0..*count, 0, 0..1);
                    }

                    if !solid_verts.is_empty() {
                        pass.set_pipeline(self.sp_solid(stereo));
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
                        if terrain_in_probe_pass {
                            pass.set_pipeline(&self.terrain_probe_reader_pipeline.pipeline);
                            pass.set_bind_group(3, &probe_target.bind_group, &[]);
                        } else {
                            pass.set_pipeline(self.sp_terrain(stereo));
                        }
                        pass.set_bind_group(0, &self.uniform_buf.bind_group, &[]);
                        pass.set_bind_group(1, &self.terrain_material.bind_group, &[]);
                        pass.set_vertex_buffer(0, solid_vb.slice(..));
                        pass.set_index_buffer(solid_ib.slice(..), wgpu::IndexFormat::Uint32);
                        // PER CHUNK, so ground behind the player is not walked
                        // through the binning pass. `chunk_indices` reorders
                        // every triangle into buckets and the ranges partition
                        // the whole index buffer, so drawing the visible
                        // chunks draws exactly what one range draw did minus
                        // what nothing can see.
                        //
                        // HONEST EXPECTATION: this frame is FILL bound, and
                        // off-screen geometry already produces no fragments --
                        // the GPU clips it. What this saves is vertex work and
                        // tile binning, which is real on a tile GPU and is not
                        // the bottleneck. The shadow pass measured culling
                        // 95.5% of casters returning only 1.8 of 3.2 ms, so
                        // the prior here is "small", and `TERRAINCULL` below
                        // reports the number instead of anyone assuming one.
                        if solid_chunks.is_empty() {
                            pass.draw_indexed(index_start..index_start + count, 0, 0..1);
                        } else {
                            for c in &solid_chunks {
                                if terrain_chunk_visible(c) {
                                    pass.draw_indexed(
                                        c.first_index..c.first_index + c.index_count,
                                        0,
                                        0..1,
                                    );
                                    terrain_drawn += c.index_count;
                                } else {
                                    terrain_culled += c.index_count;
                                }
                            }
                        }
                        if self.shadow_diag_frames.get() % 120 == 0 {
                            let total = terrain_drawn + terrain_culled;
                            let pct = if total > 0 {
                                100.0 * terrain_culled as f32 / total as f32
                            } else {
                                0.0
                            };
                            log::info!(
                                "TERRAINCULL: drew {terrain_drawn} of {total} indices \
                                 ({pct:.1}% culled) over {} chunks, {} frustum(s)",
                                solid_chunks.len(),
                                cull_planes.len(),
                            );
                        }
                    }
                    if let Some((vb, ib, count)) = &brush_buffers {
                        // One draw for the whole level, however many materials
                        // it uses: the material is a vertex attribute and every
                        // colour map is a layer of one array.
                        if probe_pass {
                            pass.set_pipeline(&probe_reader.pipeline);
                            pass.set_bind_group(3, &probe_target.bind_group, &[]);
                        } else {
                            pass.set_pipeline(self.sp_brush(stereo));
                        }
                        pass.set_bind_group(0, &self.uniform_buf.bind_group, &[]);
                        pass.set_bind_group(1, &self.brush_materials.bind_group, &[]);
                        pass.set_bind_group(2, self.brush_lightmap_bg(), &[]);
                        pass.set_vertex_buffer(0, vb.slice(..));
                        pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                        pass.draw_indexed(0..*count, 0, 0..1);
                        // SEAL THE CRACKS, right after the front faces so depth
                        // rejects the back faces everywhere except the holes.
                        // The same buffers stay bound. See `SEAL_BRUSH_CRACKS`.
                        if crate::renderer::brush_pipeline::SEAL_BRUSH_CRACKS {
                            pass.set_pipeline(self.sp_brush_seal(stereo));
                            pass.set_bind_group(0, &self.uniform_buf.bind_group, &[]);
                            pass.draw_indexed(0..*count, 0, 0..1);
                        }
                    }
                    if !layered_draws.is_empty() {
                        // One material bind for the batch: every cave in a scene
                        // blends the same four terrain layers, so the only thing
                        // that changes between draws is the model matrix.
                        pass.set_pipeline(self.sp_layered(stereo));
                        pass.set_bind_group(0, &self.uniform_buf.bind_group, &[]);
                        pass.set_bind_group(1, &self.terrain_material.bind_group, &[]);
                        for (model_bg, vb, ib, count) in &layered_draws {
                            pass.set_bind_group(2, *model_bg, &[]);
                            pass.set_vertex_buffer(0, vb.slice(..));
                            pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                            pass.draw_indexed(0..*count, 0, 0..1);
                        }
                    }
                    // WATER LAST of the world geometry, and that ordering is
                    // the whole reason it looks right: it is transparent and
                    // does not write depth, so everything it is meant to be seen
                    // THROUGH -- the lake bed, the ground, a sunken crate -- has
                    // to already be in the buffer. Drawn before them it would
                    // blend against the sky and the bed would punch straight
                    // through it.
                    if !self.water_bodies.is_empty() {
                        pass.set_pipeline(self.sp_water(stereo));
                        pass.set_bind_group(0, &self.uniform_buf.bind_group, &[]);
                        for body in &self.water_bodies {
                            pass.set_bind_group(1, &body.bind_group, &[]);
                            pass.set_vertex_buffer(0, body.vertex_buffer.slice(..));
                            pass.set_index_buffer(
                                body.index_buffer.slice(..),
                                wgpu::IndexFormat::Uint32,
                            );
                            pass.draw_indexed(0..body.index_count, 0, 0..1);
                        }
                    }
                    if !wire_verts.is_empty() {
                        pass.set_pipeline(self.sp_wire(stereo));
                        pass.set_bind_group(0, &self.uniform_buf.bind_group, &[]);
                        pass.set_vertex_buffer(0, wire_vb.slice(..));
                        pass.set_index_buffer(wire_ib.slice(..), wgpu::IndexFormat::Uint32);
                        pass.draw_indexed(0..wire_idx.len() as u32, 0, 0..1);
                    }
                    if !mesh_draws.is_empty() {
                        pass.set_pipeline(self.sp_mesh(stereo));
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
                        pass.set_pipeline(self.sp_skinned(stereo));
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
                        pass.set_pipeline(self.sp_sky(stereo));
                        pass.set_bind_group(0, &self.uniform_buf.bind_group, &[]);
                        pass.set_bind_group(1, &self.sky.bind_group, &[]);
                        pass.draw(0..3, 0..1);
                    }
                    if !particle_verts.is_empty() {
                        pass.set_pipeline(self.sp_particle(stereo));
                        pass.set_bind_group(0, &self.uniform_buf.bind_group, &[]);
                        pass.set_vertex_buffer(0, particle_vb.slice(..));
                        pass.set_index_buffer(particle_ib.slice(..), wgpu::IndexFormat::Uint32);
                        pass.draw_indexed(0..particle_idx.len() as u32, 0, 0..1);
                    }
                }
                // THE STEREO SCENE PASS LANDS ON ITS OWN.
                //
                // It drew both eyes from a uniform holding both cameras, and
                // everything after it in this frame -- the reflective draws,
                // the mirror -- is per eye and reads slot 0. Writing the
                // per-eye uniform before this pass has been SUBMITTED would
                // replace the cameras it is about to draw with, because a
                // buffer write lands relative to submits and not to where it
                // sits in the source.
                //
                // So: submit, start a fresh encoder, and only then upload this
                // eye's own camera. One extra submit a frame, and the ordering
                // stops being something to remember.
                if stereo && eye == 0 {
                    self.wgpu_queue.submit(Some(encoder.finish()));
                    encoder = self.wgpu_device.create_command_encoder(
                        &wgpu::CommandEncoderDescriptor { label: Some("ssr_eye_after_stereo") },
                    );
                    self.uniform_buf.upload_scene_with_probes(
                        &self.wgpu_queue, eye_view_proj, cam_pos, &shadow, &sky_upload,
                        &post, &self.player, probes_arg,
                    );
                }
                // THE MIP CHAIN, in this encoder, after the pass that filled
                // mip 0 and before the submit. Anywhere else and a rough
                // reflection shows the previous frame's world -- which is far
                // harder to spot as wrong than a missing reflection would be.
                //
                // Only when something is going to read it: on the direct path
                // nothing samples the scene, and downsampling it would be four
                // full-screen passes producing nothing.
                if plan.samples_scene_back() {
                    // The depth copy FIRST: the march, the blit and the mip
                    // chain all read the single-sampled one, and it is written
                    // here rather than sampled multisampled per load. See
                    // `SceneTarget::resolved_depth_view`.
                    // ONE SLOT ACROSS BOTH, because on a tile GPU the gaps
                    // between nine small passes are part of what they cost.
                    let prep = 4 + eye;
                    self.ssr_pipelines.resolve_depth(
                        &mut encoder,
                        &self.scene_targets[eye],
                        self.pass_timers.as_ref().and_then(|t| t.span_start(prep)),
                    );
                    // The min-depth pyramid above it, for the march to descend.
                    self.ssr_pipelines.build_hi_z(
                        &mut encoder,
                        &self.scene_targets[eye],
                        self.pass_timers.as_ref().and_then(|t| t.span_end(prep)),
                    );
                    self.ssr_pipelines.generate_mips(&mut encoder, &self.scene_targets[eye]);

                // THE REFLECTION TRACE AND ITS RESOLVE.
                //
                // The same reflective geometry the eye pass draws, marched into
                // a buffer instead of blended straight onto the surface, and
                // then filtered ACROSS pixels. That filter is the only place
                // the hit/miss cliff can be removed: a fragment can see its own
                // ray and nothing else, and the comb along every silhouette is
                // confident hits sitting next to rays that correctly found
                // nothing. Confirmed on the headset 2026-09-18 -- orange
                // against greyscale with a ragged stepped edge.
                //
                // Its own depth attachment, cleared, holding only reflective
                // geometry. A reflective surface hidden behind a wall may write
                // here and that is harmless: the composite draws the same
                // geometry against the real depth buffer and never samples
                // where it wrote.
                // BRUSHES ONLY. Reflective SOLIDS still march inline -- the
                // composite swaps the brush pipeline and nothing else -- so
                // entering this for a solid-only scene would trace an empty
                // buffer and filter it for nothing.
                if self.buffered_reflections && reflective_brushes {
                    let target = &self.reflection_targets[eye];
                    // ONE SLOT COVERING BOTH the trace and the resolve, so what
                    // the buffered path costs is a single number against the
                    // inline path's march. Slots 6 and 7; appended, never
                    // inserted, because the others are addressed by index.
                    //
                    // ATTACHED TO THE PASS, not called beside it. These RETURN
                    // the timestamp writes; calling them as bare statements set
                    // the slot's "written" bit and recorded nothing, so it
                    // reported a confident 0.00 ms -- which is exactly what a
                    // pass that never ran would report (2026-09-18).
                    let refl_slot = 6 + eye;
                    {
                        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("ssr_reflection_trace"),
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: &target.trace_view,
                                depth_slice: None,
                                resolve_target: None,
                                ops: wgpu::Operations {
                                    // TRANSPARENT is confidence ZERO, which is
                                    // "nothing found here" -- the value the
                                    // resolve treats as a gap to fill. Clearing
                                    // to anything opaque would have every pixel
                                    // the geometry misses claim a confident
                                    // black reflection.
                                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                                    store: wgpu::StoreOp::Store,
                                },
                            })],
                            depth_stencil_attachment: Some(
                                wgpu::RenderPassDepthStencilAttachment {
                                    view: &target.depth_view,
                                    depth_ops: Some(wgpu::Operations {
                                        load: wgpu::LoadOp::Clear(1.0),
                                        // STORED, not discarded: the resolve
                                        // reads it back as a depth test so it
                                        // runs only on reflective pixels. That
                                        // costs one depth store and saves a
                                        // 25-tap filter across the whole frame.
                                        store: wgpu::StoreOp::Store,
                                    }),
                                    stencil_ops: None,
                                },
                            ),
                            timestamp_writes: self
                                .pass_timers
                                .as_ref()
                                .and_then(|t| t.span_start(refl_slot)),
                            occlusion_query_set: None,
                            multiview_mask: None,
                        });
                        if let Some((vb, ib, count)) = &brush_buffers {
                            if reflective_brushes {
                                pass.set_pipeline(&self.brush_ssr_trace_pipeline.pipeline);
                                pass.set_bind_group(0, &self.uniform_buf.bind_group, &[]);
                                pass.set_bind_group(1, &self.brush_materials.bind_group, &[]);
                                pass.set_bind_group(2, self.brush_lightmap_bg(), &[]);
                                pass.set_bind_group(3, &self.scene_targets[eye].bind_group, &[]);
                                pass.set_vertex_buffer(0, vb.slice(..));
                                pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                                pass.draw_indexed(0..*count, 0, 0..1);
                            }
                        }
                    }
                    self.ssr_pipelines.resolve_reflections(
                        &mut encoder,
                        target,
                        self.pass_timers.as_ref().and_then(|t| t.span_end(refl_slot)),
                    );
                }
                }
                self.wgpu_queue.submit(Some(encoder.finish()));
            }

            // SPACEWARP: this eye's motion vectors and depth, small, from the
            // same geometry -- the world, then every mesh by its own motion.
            // See `space_warp`.
            //
            // HERE, before the eye pass can be skipped. It sat after the
            // `continue` below until 2026-09-29: whenever the scene drew
            // straight into the eye image -- normal play -- no motion or depth
            // was ever written, the compositor read the swapchains' empty
            // memory as depth 0, the near plane, and threw every frame off
            // screen at the first head movement. Black frames, only in the
            // headset (a still desk headset needs no correction), and only
            // partly in the debug views, which do run the eye pass.
            if let Some(sw) = self.space_warp.as_ref() {
                if let Some((m, d)) = sw.acquired {
                    use crate::renderer::space_warp::{MotionDraw, MotionKind};
                    let base = eye as u32 * warp_per_eye;
                    let mut draws: Vec<MotionDraw> = Vec::new();
                    if let Some((vb, ib, n)) = brush_buffers.as_ref() {
                        draws.push(MotionDraw { kind: MotionKind::Brush, vertices: vb, indices: ib, first: 0, count: *n, slot: base, joints: None });
                    }
                    // The solid buffer: its cuboids whole, and of the ground
                    // only the chunks this eye's scene pass drew.
                    let solid = |first: u32, count: u32| MotionDraw {
                        kind: MotionKind::Solid,
                        vertices: &solid_vb,
                        indices: &solid_ib,
                        first,
                        count,
                        slot: base,
                        joints: None,
                    };
                    match terrain_range {
                        Some((start, _)) if !solid_chunks.is_empty() => {
                            draws.push(solid(0, start));
                            for c in solid_chunks.iter().filter(|c| terrain_chunk_visible(c)) {
                                draws.push(solid(c.first_index, c.index_count));
                            }
                        }
                        _ => draws.push(solid(0, solid_idx.len() as u32)),
                    }
                    for (i, (inst, _, _)) in warp_meshes.iter().enumerate() {
                        let slot = base + 1 + i as u32;
                        if let Some(skin) = &inst.mesh.skin {
                            let joints = skin.motion_bind_group(&self.wgpu_device, &sw.pipelines.joints_layout);
                            for prim in &skin.primitives {
                                draws.push(MotionDraw {
                                    kind: MotionKind::Skinned,
                                    vertices: &prim.vertex_buffer,
                                    indices: &prim.index_buffer,
                                    first: 0,
                                    count: prim.indices.len() as u32,
                                    slot,
                                    joints: Some(joints),
                                });
                            }
                        } else {
                            for prim in inst.mesh.primitives.iter().filter(|p| p.layered.is_none()) {
                                draws.push(MotionDraw {
                                    kind: MotionKind::Mesh,
                                    vertices: &prim.vertex_buffer,
                                    indices: &prim.index_buffer,
                                    first: 0,
                                    count: prim.indices.len() as u32,
                                    slot,
                                    joints: None,
                                });
                            }
                        }
                    }
                    if self.levers.space_warp_debug & 1 != 0 {
                        draws.clear();
                    }
                    let mut encoder = self
                        .wgpu_device
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("space_warp") });
                    crate::renderer::space_warp::record(
                        &mut encoder,
                        &sw.pipelines,
                        &sw.camera_group,
                        &sw.motion_targets[m][eye].view,
                        &sw.depth_targets[d][eye].view,
                        &draws,
                        self.levers.space_warp_debug & 2048 != 0,
                        if self.levers.space_warp_debug & 4096 != 0 { 0.0 } else { 1.0 },
                    );
                    self.wgpu_queue.submit(Some(encoder.finish()));
                    // DIAGNOSIS: what the pass left in the images. See
                    // `space_warp::Readback`.
                    if self.levers.space_warp_debug & 8192 != 0 && eye == 0 {
                        if let Some(rb) = sw.readback.as_ref() {
                            match rb.read(sw.depth_raw[d], sw.depth_has_stencil, sw.motion_raw[m], eye as u32) {
                                Ok(s) => log::info!("SWDUMP {s}"),
                                Err(e) => log::warn!("SWDUMP failed: {e:?}"),
                            }
                        }
                    }
                }
            }

            // The scene has already been drawn straight into this image
            // unless something needs to sample it back. See the scene pass.
            if !plan.runs_eye_pass() {
                continue;
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
                        depth_slice: None,
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
                    // Its own slot, so the blit-plus-SSR cost is separable from the scene draw.
                    timestamp_writes: self.pass_timers.as_ref().and_then(|t| t.writes(eye * 2 + 1)),
                    ..Default::default()
                });

                pass.set_pipeline(&self.ssr_pipelines.blit_pipeline);
                // Its OWN bind group: the blit primes the depth everything
                // after it tests against, and reads the scene pass's depth
                // directly rather than the march's copy. See `blit_bind_group`.
                pass.set_bind_group(0, &self.scene_targets[eye].blit_bind_group, &[]);
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

                // REFLECTIVE BRUSHES, over the blit.
                //
                // Drawn a second time rather than reflecting in the scene pass,
                // because a pass cannot sample the attachment it is writing.
                // The first draw is what lands in the buffer this one reads.
                if reflective_brushes {
                    if let Some((vb, ib, count)) = &brush_buffers {
                        // THE COMPOSITE reads the filtered reflection; the
                        // inline path marches here as it always has. The
                        // diagnostics stay on the inline path deliberately --
                        // the false colour is about where a RAY went, and the
                        // composite has no ray.
                        let buffered = self.buffered_reflections
                            && self.debug_view == crate::renderer::brush_pipeline::DebugView::Off;
                        pass.set_pipeline(match self.debug_view {
                            _ if buffered => &self.brush_ssr_composite_pipeline.pipeline,
                            crate::renderer::brush_pipeline::DebugView::Off => &self.brush_ssr_pipeline.pipeline,
                            crate::renderer::brush_pipeline::DebugView::Sources => &self.brush_ssr_sources_pipeline.pipeline,
                            crate::renderer::brush_pipeline::DebugView::Ssr => &self.brush_ssr_debug_pipeline.pipeline,
                        });
                        pass.set_bind_group(0, &self.uniform_buf.bind_group, &[]);
                        pass.set_bind_group(1, &self.brush_materials.bind_group, &[]);
                        pass.set_bind_group(2, self.brush_lightmap_bg(), &[]);
                        pass.set_bind_group(3, if buffered {
                            &self.reflection_composite_bg[eye]
                        } else {
                            &self.scene_targets[eye].bind_group
                        }, &[]);
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

        // SPACEWARP: the images go back, and this frame becomes the history.
        if let Some(sw) = self.space_warp.as_mut() {
            if sw.acquired.is_some() {
                sw.motion.release_image()?;
                sw.depth.release_image()?;
                // How far locomotion moved the tracking space since the last
                // frame: the runtime fills untouched pixels (the sky) from it,
                // and stops extrapolating across a jump. See `space_warp`.
                let dbg = self.levers.space_warp_debug;
                let delta = sw
                    .prev
                    .filter(|_| dbg & 8 == 0)
                    .map(|(_, prev_w2p)| crate::renderer::space_warp::app_space_delta(prev_w2p, warp_world_to_player))
                    .unwrap_or(xr::Posef::IDENTITY);
                for info in sw.info.iter_mut() {
                    info.app_space_delta_pose = delta;
                    info.near_z = crate::renderer::space_warp::NEAR_Z;
                    info.far_z = if dbg & 16 != 0 { f32::INFINITY } else { crate::renderer::space_warp::FAR_Z };
                    if dbg & 512 != 0 {
                        // Declared reversed: 1.0 near, 0.0 far.
                        info.near_z = crate::renderer::space_warp::FAR_Z;
                        info.far_z = crate::renderer::space_warp::NEAR_Z;
                    }
                    if dbg & 1024 != 0 {
                        // Every depth value 500 m away or more.
                        info.near_z = 500.0;
                        info.far_z = crate::renderer::space_warp::FAR_Z;
                    }
                }
            }
            sw.prev = Some((warp_view_proj, warp_world_to_player));
            sw.prev_models = meshes.iter().map(|m| (m.model.buffer.clone(), m.mesh.model_matrix())).collect();
        }

        // Resolve the pass timers in their own submission, AFTER every pass
        // this frame has been submitted -- a resolve recorded earlier reads the
        // queries before they are written and reports the previous frame.
        if let Some(timers) = &self.pass_timers {
            let mut encoder = self
                .wgpu_device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("pass_timer_resolve"),
                });
            timers.resolve(&mut encoder);
            self.wgpu_queue.submit(Some(encoder.finish()));
        }

        let cpu_time = cpu_start.elapsed();
        let gpu_wait_start = std::time::Instant::now();
        // NO WAIT FOR THE GPU, unless measuring. Blocking here until the
        // frame finished drawing left the GPU idle while the CPU prepared the
        // next one: the first headset benchmark measured frames ~3 ms longer
        // than the GPU's own time (2026-09-28), which alone would keep a
        // 12 ms GPU frame from 72 Hz. Nothing read back needs it -- the pass
        // timers wait for themselves, once a window -- and the runtime waits
        // for the queue before it composites. `gpu_sync` restores the wait,
        // whose length is then exactly the GPU's time: the A/B schedule's
        // measure.
        let wait = if self.levers.gpu_sync {
            wgpu::PollType::Wait { submission_index: None, timeout: None }
        } else {
            wgpu::PollType::Poll
        };
        let _ = self.wgpu_device.poll(wait);
        let gpu_time = gpu_wait_start.elapsed();
        let outcome = self.frame_stats.record(cpu_time, gpu_time, std::time::Instant::now());
        // The runtime's counters, every measured frame, so the window carries
        // their average rather than the one frame that closed it.
        if outcome.measured {
            if let Some(metrics) = &self.perf_metrics {
                self.perf_metric_window.add(&metrics.sample());
            }
        }
        // ONLY on the frames `PERF` prints. `read` maps a buffer and polls,
        // which stalls the render thread; doing that every frame would make the
        // instrument change the thing it is measuring.
        if let Some(stats) = outcome.closed {
            let cycling = crate::renderer::perf_ab::ENABLED || self.levers.ab_cycle;
            // Labelled with the phase these frames ran under. The window just
            // logged was ALL one phase, because the phase only advances here --
            // or `warmup`, when it straddled a change of levers.
            let ab = if self.perf_warmup {
                "warmup"
            } else if cycling {
                ab_phase.label()
            } else {
                "-"
            };
            // And with the levers in force, so a window measured with a
            // feature switched off from the headset is never read as shipped.
            let levers = self.levers.summary();
            // Labelled with the SSR state too, so a window measured while the
            // switch was flipped mid-session is never read as the other state.
            let ssr = if self.screen_space_reflections { "on" } else { "off" };
            let breakdown = self.pass_timers.as_ref().and_then(|t| t.read(&self.wgpu_device));
            match &breakdown {
                Some(breakdown) => log::info!(
                    "{} [ab={ab} ssr={ssr} levers={levers}]",
                    crate::renderer::pass_timers::format_breakdown(breakdown)
                ),
                None => log::info!("PASS: unavailable [ab={ab} ssr={ssr} levers={levers}]"),
            }
            // The runtime's view of the same window. See `xr::perf_metrics`.
            let counters = self.perf_metric_window.take();
            if self.perf_metrics.is_some() {
                log::info!("{} [ab={ab} ssr={ssr} levers={levers}]", crate::perf_metrics_log::format_line(&counters));
            }
            // Where the window closed, so a slow window from a play session
            // can be put on the map ("the stone room was laggy").
            {
                let p = eye_views[0].pose.position;
                let head = warp_world_to_player.inverse().transform_point3(glam::Vec3::new(p.x, p.y, p.z));
                log::info!(
                    "WHERE: head=({:.2},{:.2},{:.2}) yaw={:.0} debug={:?}",
                    head.x,
                    head.y,
                    head.z,
                    self.player.yaw.to_degrees(),
                    self.debug_view
                );
            }
            if let Some(log) = &self.perf_log {
                let cycle_len = crate::renderer::perf_ab::Phase::ALL.len() as u64;
                let record = crate::perf_record::WindowRecord {
                    window: self.perf_window_index,
                    t: self.started_at.elapsed().as_secs_f64(),
                    phase: ab.to_string(),
                    cycle_pass: if cycling { self.perf_windows / cycle_len } else { self.perf_windows },
                    cycle_len: if cycling { cycle_len } else { 1 },
                    warmup: self.perf_warmup,
                    levers: levers.clone(),
                    bench: self.levers.bench.as_ref().map(|b| b.name.clone()),
                    ssr: self.screen_space_reflections,
                    multiview: self.multiview_scene,
                    frames: stats.frames,
                    cpu_avg: stats.cpu_avg,
                    cpu_max: stats.cpu_max,
                    gpu_avg: stats.gpu_avg,
                    gpu_max: stats.gpu_max,
                    frame_ms: stats.frame_ms,
                    fps: stats.fps,
                    pass: breakdown.iter().flatten().map(|t| (t.label.clone(), t.ms)).collect(),
                    xr: counters.iter().map(|c| (c.short.clone(), c.value)).collect(),
                };
                log.write(record.to_line());
            }
            self.perf_window_index += 1;
            if self.perf_warmup {
                self.perf_warmup = false;
            } else {
                self.perf_windows += 1;
            }
        }
        self.swapchain.release_image()?;

        // PINNED, the frame is handed to the compositor as if drawn from where
        // the headset really is. Given the pinned pose, the compositor
        // reprojects the image to the real head -- which, on a desk, is
        // nowhere near the viewpoint -- and a screenshot comes back tilted,
        // with black corners where no image was (bench, 2026-09-27). What
        // the frame costs is the same either way.
        let submit_pose = |i: usize, ev: &xr::View| match (&self.pinned_head, &tracked_poses) {
            (Some(_), Some(tracked)) => tracked.get(i).copied().unwrap_or(ev.pose),
            _ => ev.pose,
        };
        // With SpaceWarp, each view carries its motion vectors and depth.
        // The info structs live in the renderer, which is neither moved nor
        // touched again before the caller hands these views to `xrEndFrame`.
        let warp_info: Option<&[xr::sys::CompositionLayerSpaceWarpInfoFB; 2]> =
            self.space_warp.as_ref().filter(|sw| sw.acquired.is_some()).map(|sw| &sw.info);
        let proj_views = eye_views
            .iter()
            .enumerate()
            .map(|(i, ev)| {
                let view = xr::CompositionLayerProjectionView::new()
                    .pose(submit_pose(i, ev))
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
                    );
                match warp_info.and_then(|info| info.get(i)) {
                    Some(info) => {
                        let mut raw = view.into_raw();
                        raw.next = info as *const _ as *const std::ffi::c_void;
                        unsafe { xr::CompositionLayerProjectionView::from_raw(raw) }
                    }
                    None => view,
                }
            })
            .collect();

        Ok(proj_views)
    }
}
