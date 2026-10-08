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


/// One static mesh primitive to draw: its model, texture and lightmap bind
/// groups, vertices, indices and index count, and -- for the eye pass -- its
/// thin parts apart (see `mesh::thin_parts`) and whether it is see-through.
struct MeshDraw<'a> {
    model: &'a wgpu::BindGroup,
    texture: &'a wgpu::BindGroup,
    lightmap: &'a wgpu::BindGroup,
    vertices: &'a wgpu::Buffer,
    indices: &'a wgpu::Buffer,
    count: u32,
    thin: Option<&'a crate::renderer::mesh::ThinParts>,
    blended: bool,
}

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
            mesh_draws.push(MeshDraw {
                model: &instance.model.bind_group,
                texture: &prim.texture.bind_group,
                lightmap: lightmap_bg,
                vertices: &prim.vertex_buffer,
                indices: &prim.index_buffer,
                count: prim.indices.len() as u32,
                thin: prim.thin.as_ref(),
                blended: prim.blended,
            });
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
        // THE TIME OF DAY first: everything below reads the sky, its light
        // and the brush atlas it relights. See `xr_renderer::time_of_day`.
        self.update_time_of_day();
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
        let foveation = self.frame_levers().foveation;
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
        // MEASUREMENT: both eyes from the left eye's view. See
        // `Levers::same_eyes`.
        if self.levers.same_eyes && eye_views.len() >= 2 {
            eye_views[1] = eye_views[0];
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
                // EACH CHUNK'S GENTLE TRIANGLES FIRST, worked out once per
                // terrain, so the scene pass can draw them with the ground's
                // slope twins. The same triangles in each chunk's range, so
                // every other pass is unchanged. See `ground_twins`.
                let threshold = self.terrain_settings.biplanar_start_deg;
                if self.terrain_gentle.is_some()
                    && !self.slope_split.as_ref().is_some_and(|s| s.is_for(terrain_verts, terrain_idx, terrain_chunks, threshold))
                {
                    let started = std::time::Instant::now();
                    let split = crate::renderer::ground_twins::SlopeSplit::new(terrain_verts, terrain_idx, terrain_chunks, threshold);
                    log::info!(
                        "SLOPES: {:.1}% of the ground's triangles are gentle (under {:.0} degrees), {:.1}% steep (over {:.0}) over {} chunk(s), {:.1} ms",
                        100.0 * split.gentle_total as f64 / terrain_idx.len().max(1) as f64,
                        threshold - crate::renderer::ground_twins::GENTLE_MARGIN_DEG,
                        100.0 * split.steep_total as f64 / terrain_idx.len().max(1) as f64,
                        threshold + crate::renderer::ground_twins::GENTLE_MARGIN_DEG,
                        split.gentle.len(),
                        started.elapsed().as_secs_f64() * 1000.0,
                    );
                    self.slope_split = Some(split);
                }
                let terrain_idx: &[u32] = match &self.slope_split {
                    Some(split) if self.terrain_gentle.is_some() => &split.indices,
                    _ => terrain_idx,
                };
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
        // THE BRUSHES BY WHAT THE SUN CAN DO AT THEM -- never reached, then
        // baked throughout, then the rest -- so the scene pass can draw each
        // part with the smallest reader that gives it the same picture. Every
        // other pass draws the whole range, where the order changes nothing.
        // See `SunFaces`.
        let partitioned = match (&self.sun_faces, brush_geometry) {
            (Some(faces), Some((v, i))) => Some(faces.partition(v, i)),
            _ => None,
        };
        let brush_sun_ends = partitioned.as_ref().map_or([0, 0], |(_, ends)| *ends);
        let brush_buffers = brush_geometry.map(|(v, i)| {
            let i: &[u32] = partitioned.as_ref().map_or(i, |(p, _)| p.as_slice());
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

        let head = glam::Vec3::new(
            eye_views[0].pose.position.x,
            eye_views[0].pose.position.y,
            eye_views[0].pose.position.z,
        );

        // The effects with the A/B schedule's phase on the lever file, as the
        // app asked when it placed the fires' lights (`frame_levers`).
        let effects_on = self.frame_levers().effects;
        // THE EYES IN THE WATER: for each, which body it is in and how --
        // above it, on the line, under -- against the moving surface's CPU
        // twin, every frame. Everything under the water is drawn from this;
        // with neither eye in it, none of it runs. See `underwater`.
        use crate::renderer::underwater::EyeWater;
        let under_levers = self.frame_levers();
        let eyes_in_water: [Option<(usize, EyeWater)>; 2] = std::array::from_fn(|i| {
            if !(under_levers.water && under_levers.underwater) {
                return None;
            }
            let v = &eye_views[i.min(eye_views.len() - 1)];
            let p = glam::Quat::from_rotation_y(self.player.yaw) * glam::Vec3::new(v.pose.position.x, v.pose.position.y, v.pose.position.z)
                + self.player.offset;
            self.water_bodies.iter().enumerate().find_map(|(b, body)| {
                let depth = body.still.at(glam::Vec2::new(p.x, p.z))?;
                let reach = crate::renderer::underwater::wave_reach(body.waves.significant_height, depth);
                let state = crate::renderer::underwater::eye_water(&body.uniform, depth, reach, p);
                (state != EyeWater::Above).then_some((b, state))
            })
        });
        let deepest = eyes_in_water.iter().flatten().map(|(_, s)| *s).max_by_key(|s| *s as u8).unwrap_or(EyeWater::Above);
        let film_age = self.surfacing.update(deepest, self.water_seconds);
        // The light under the surface, this frame; each eye's flag goes in
        // before its scene pass.
        let under_base = {
            let yaw = glam::Quat::from_rotation_y(self.player.yaw);
            let sun = crate::renderer::underwater::sun_from_lights(lights, yaw)
                .or_else(|| self.sky.sun.as_ref().map(|s| (glam::Vec3::from(s.direction), glam::Vec3::from(s.light_rgb))));
            let head = eye_views[0].pose.position;
            let head = glam::Vec3::new(head.x, head.y, head.z);
            let mut spheres: Vec<(glam::Vec3, f32)> =
                meshes.iter().map(|m| (m.mesh.position, m.mesh.bounding_radius * m.mesh.scale.max_element())).collect();
            spheres.sort_by(|a, b| (a.0 - head).length_squared().total_cmp(&(b.0 - head).length_squared()));
            spheres.truncate(crate::renderer::underwater::MAX_SPHERES);
            crate::renderer::underwater::UnderUniform::new(&crate::renderer::underwater::UnderFrame {
                size: (self.width, self.height),
                fov_y: {
                    let f = eye_views[0].fov;
                    f.angle_up - f.angle_down
                },
                sun,
                sky_down: glam::Vec3::from(self.sky.irradiance.evaluate([0.0, 1.0, 0.0])),
                waterline: false,
                film_age,
                spheres: &spheres,
            })
        };

        // EYE ADAPTATION: meter what the player is looking at, from the probe
        // of the room they stand in, and ease the exposure toward it. The
        // head and gaze go back to WORLD space, where the probes were baked.
        // Before the models are uploaded: a fixture's own light is scaled
        // against it (`tonemap::own_light_scale`).
        let mut post = {
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
            // A burning fire, which the photographs never saw.
            let fires = if effects_on {
                crate::renderer::effects::meter_samples(&self.effect_emitters, head_world, &self.room_descs)
            } else {
                Vec::new()
            };
            let mut eye = self.eye.borrow_mut();
            let mut metered = eye.meter_with(head_world, gaze, &fires);
            // Under the water the eye adapts to the water's own light, which
            // dims with depth. See `underwater::in_water_luminance`.
            if let Some((b, state)) = eyes_in_water[0].or(eyes_in_water[1]) {
                let u = &self.water_bodies[b].uniform;
                let water = crate::renderer::underwater::in_water_luminance(u, &under_base, u.extinction[3] - head_world.y);
                metered = crate::renderer::underwater::meter_in_water(metered, water, crate::renderer::underwater::eye_share(state));
            }
            let auto = eye.update(metered, dt);
            if self.shadow_diag_frames.get() % 120 == 0 {
                log::info!("EXPOSURE meter {metered:.4} -> x{auto:.2} (auto {})", self.auto_exposure);
            }
            // The eye at night: toward rod vision as it adapts to the dark.
            let night_vision = self.tod_night_vision(eye.adapted_luminance());
            crate::renderer::uniforms::PostUpload {
                exposure: self.post.exposure * if self.auto_exposure { auto } else { 1.0 },
                terrain_detail_distance: self.levers.terrain_detail_distance,
                night_vision,
                ..self.post
            }
        };

        // Between the eyes, in the player's frame, as the lights are.
        let mid_eye = {
            let at = |v: &xr::View| glam::Vec3::new(v.pose.position.x, v.pose.position.y, v.pose.position.z);
            0.5 * (at(&eye_views[0]) + at(&eye_views[eye_views.len() - 1]))
        };

        // THE EFFECTS: this frame's particles, worked out in the world at the
        // frame's display time, lit by this frame's lamps in their emitter's
        // room and by the room's baked light, and drawn after the glass. Here,
        // before the frame's draws borrow the renderer. See `effects`.
        let mut effects_drawn = effects_on && !(self.effect_emitters.is_empty() && self.splashes.is_empty()) && self.effects_gpu.is_some();
        if effects_drawn {
            let yaw = glam::Quat::from_rotation_y(self.player.yaw);
            let (offset, descs) = (self.player.offset, &self.room_descs);
            // The sky's sun as well, which joins the frame's lights further
            // down: splashes sparkle in it (no other effect takes a sun).
            let fx_lights: Vec<Light> = lights
                .iter()
                .copied()
                .chain(crate::renderer::lights::sky_sun_light(self.sky.sun.as_ref(), lights, yaw.inverse()))
                .collect();
            let lights = &fx_lights[..];
            let reaches = |e: &crate::renderer::effects::EffectEmitter, i: usize| {
                crate::renderer::effects::lamp_in_room(descs, e.position, yaw * lights[i].position + offset)
            };
            let ambient = |world: glam::Vec3| crate::renderer::effects::room_ambient(descs, world);
            let at = crate::renderer::effects::Surroundings {
                head: mid_eye,
                offset,
                yaw_inv: yaw.inverse(),
                lights,
                reaches: &reaches,
                ambient: &ambient,
                exposure: post.exposure,
            };
            // Splashes were born on the water's clock: onto this one.
            let now = time.as_nanos() as f64 * 1e-9;
            let splashes: Vec<crate::renderer::effects::Splash> = self
                .splashes
                .iter()
                .map(|s| crate::renderer::effects::Splash { born: now - (self.water_seconds - s.born), ..*s })
                .collect();
            // WHICH EMITTERS EITHER EYE SEES -- in a frustum and, from inside
            // a closed room, through open doorways -- and those seen within
            // the last EMITTER_HOLD, so one at a view's edge does not pop.
            // The rest are neither simulated nor drawn; being closed-form,
            // one comes back where its clock has it (its light, in the
            // frame's lights, never stopped). See `effects::emitter_seen`.
            let lv = self.frame_levers();
            let world_to_player = glam::Mat4::from_quat(yaw.inverse()) * glam::Mat4::from_translation(-offset);
            let views: Vec<[glam::Vec4; 6]> = eye_views
                .iter()
                .map(|ev| {
                    let vp = Camera::gl_to_wgpu_ndc(Camera::xr_projection(ev.fov, 0.03, 1000.0)) * Camera::xr_view(ev.pose);
                    crate::renderer::shadow::frustum_planes(vp * world_to_player)
                })
                .collect();
            let portals: Vec<crate::renderer::uniforms::ProbePortal> = if lv.door_culling {
                crate::renderer::doors::open_portals(&self.probe_portals, &self.shut_portals)
            } else {
                self.probe_portals.clone()
            };
            let rooms: &[crate::renderer::portal_cull::CullRoom] = if lv.portal_culling { &self.cull_rooms } else { &[] };
            let head_world = yaw * mid_eye + offset;
            self.effect_seen_until.resize(self.effect_emitters.len(), f64::NEG_INFINITY);
            for (e, until) in self.effect_emitters.iter().zip(self.effect_seen_until.iter_mut()) {
                if crate::renderer::effects::emitter_seen(e, &views, head_world, rooms, &portals) {
                    *until = now + crate::renderer::effects::EMITTER_HOLD;
                }
            }
            let held = &self.effect_seen_until;
            let seen = |i: usize| held.get(i).map_or(true, |&t| now <= t);
            if (0..self.effect_emitters.len()).any(|i| seen(i)) || !splashes.is_empty() {
                let frame = crate::renderer::effects::simulate_seen(&self.effect_emitters, &seen, &splashes, now, &at);
                if let Some(gpu) = self.effects_gpu.as_mut() {
                    gpu.upload(&self.wgpu_device, &self.wgpu_queue, &frame);
                }
            } else {
                effects_drawn = false;
            }
        }

        // THE WEATHER'S VIEW: the head, its axes, the sky's and sun's light,
        // the torch and how many particles fall near. See `weather`.
        let weather_on = self.frame_levers().weather && self.weather.is_some();
        if weather_on {
            let pixel = eye_views.first().map_or(0.0, |v| {
                let f = v.fov;
                (f.angle_up.tan() - f.angle_down.tan()) / self.height.max(1) as f32
            });
            let yaw = glam::Quat::from_rotation_y(self.player.yaw);
            let head_world = yaw * mid_eye + self.player.offset;
            let (sky, sun) = crate::renderer::weather::particle_light(&self.sky.irradiance, self.sky.sun.as_ref());
            let ground = self.terrain_footprint.map(|(lo, hi)| ([lo.x, lo.z], [hi.x, hi.z]));
            let frame = [self.player.offset.x, self.player.offset.y, self.player.offset.z, self.player.yaw];
            if let Some(w) = self.weather.as_mut() {
                let counts = crate::renderer::weather::particle_counts(&w.areas, head_world.to_array());
                let wind = w
                    .areas
                    .iter()
                    .find(|a| {
                        head_world.x >= a.min[0] && head_world.x <= a.min[0] + a.extent[0] && head_world.z >= a.min[1] && head_world.z <= a.min[1] + a.extent[1]
                    })
                    .map_or([0.0, 0.0], |a| a.wind);
                let torch = w.torch.map(|l| {
                    let (outer, inner) = l.cone_cosines();
                    let c = l.color.to_linear();
                    let i = l.intensity;
                    (l.position.to_array(), l.direction.to_array(), l.range, outer, inner, [c[0] * i, c[1] * i, c[2] * i])
                });
                w.counts = counts;
                w.maps.set_view(
                    &crate::renderer::weather::ParticleView {
                        head_world: head_world.to_array(),
                        right: cam_right.to_array(),
                        up: cam_up.to_array(),
                        frame,
                        sky,
                        sun,
                        torch,
                        exposure: post.exposure,
                        pixel,
                        time: w.seconds,
                        ground,
                    },
                    counts,
                    wind,
                );
                w.maps.write(&self.wgpu_queue);
            }
        }
        // The thin pass's kernel unit as a share of depth: an eye pixel's
        // size at unit depth (tangent span over pixels, both axes averaged)
        // times `levers.thin_parts`. See `mesh::thin_parts`.
        let thin_width = match eye_views.first() {
            Some(v) if self.levers.thin_parts > 0.0 => {
                let f = v.fov;
                let across = (f.angle_right.tan() - f.angle_left.tan()) / self.width.max(1) as f32;
                let up = (f.angle_up.tan() - f.angle_down.tan()) / self.height.max(1) as f32;
                self.levers.thin_parts * 0.5 * (across + up)
            }
            _ => 0.0,
        };
        let thin_pass = thin_width > 0.0;
        let mut mesh_draws: Vec<MeshDraw> = Vec::new();
        let mut skinned_draws: Vec<SkinnedDraw> = Vec::new();
        let mut layered_draws: Vec<LayeredDraw> = Vec::new();
        // Depth-pass casters. The ordinary vertex buffer even for a layered
        // mesh -- a depth pass reads position and nothing else, so a baked cave
        // casts here with no pipeline of its own.
        let mut shadow_casters: Vec<crate::renderer::shadow::ShadowMeshDraw> = Vec::new();
        // Each caster's bounding sphere, entry for entry, so a shadow tile can
        // leave out the models it cannot reach. See `shadow::ShadowMeshBound`.
        let mut shadow_bounds: Vec<crate::renderer::shadow::ShadowMeshBound> = Vec::new();
        // Which of `shadow_casters` are doors, for the lamps' moving casters'
        // tiles. See `MeshInstance::tile_caster`.
        let mut tile_meshes: Vec<usize> = Vec::new();
        // The glare sources whose fixtures are drawn as the eye adapted to
        // them sees them: their veils lie only behind them. See `build_glare`.
        let mut glare_adapted = vec![false; self.glare_sources.len()];
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
            // The lit room round it, turned into the player's frame its
            // normals are in. See `room_light`. A door leaf reads it in front
            // of the face the eye sees (`doors::light_point`).
            let light_at = if instance.tile_caster {
                let eye_world = glam::Quat::from_rotation_y(self.player.yaw) * mid_eye + self.player.offset;
                crate::renderer::doors::light_point(&self.doors, world, eye_world)
            } else {
                world
            };
            let room = crate::renderer::room_light::turned_to_player(
                &crate::renderer::room_light::room_light_at(&self.room_descs, light_at),
                self.player.yaw,
            );
            // A fixture's own light as the eye adapted to it sees it: as far
            // as its bulb is in view and big enough to look at. See
            // `tonemap::own_light_scale` and `tonemap::bulb_adaptation`.
            let own_scale = instance.own_light.as_ref().map_or(1.0, |l| {
                let source = self.glare_sources.iter().position(|s| (s.position - l.position).length() < 0.05);
                if let Some(k) = source {
                    glare_adapted[k] = self.levers.fixture_bulb_level > 0.0;
                }
                let in_view =
                    source.map_or(1.0, |k| crate::renderer::glare::bulb_in_view(&self.glare_sources[k], mid_eye));
                let adapted = crate::renderer::tonemap::bulb_adaptation(
                    crate::renderer::glare::LAMP_RADIUS,
                    (mid_eye - l.position).length(),
                    in_view,
                );
                crate::renderer::tonemap::own_light_scale(
                    post.exposure,
                    instance.emissive_drive,
                    self.levers.fixture_bulb_level,
                    adapted,
                )
            });
            instance.model.upload_lit_bulb_scaled(
                &self.wgpu_queue,
                instance.mesh.model_matrix(),
                sky_vis,
                instance.emissive_drive,
                &room,
                instance.own_light.as_ref(),
                thin_width,
                own_scale,
            );
            let lightmap_bg = self.mesh_lightmap_bg(instance.lightmap_key);
            push_mesh_draws(
                instance, lightmap_bg, &mut mesh_draws, &mut skinned_draws, &mut layered_draws,
            );
            if instance.mesh.skin.is_none() {
                let bound = crate::renderer::shadow::mesh_caster_bound(&instance.mesh);
                for prim in instance.mesh.primitives.iter().filter(|p| p.casts_shadow) {
                    if instance.tile_caster {
                        tile_meshes.push(shadow_casters.len());
                    }
                    shadow_casters.push((
                        &prim.vertex_buffer,
                        &prim.index_buffer,
                        prim.indices.len() as u32,
                        &instance.model.bind_group,
                    ));
                    shadow_bounds.push(bound);
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
            // Lit as the same mesh in view would be: it is the same body. The
            // open sky this gave them made a body in the floor brighter than
            // the one standing on it indoors.
            let world = glam::Quat::from_rotation_y(self.player.yaw) * instance
                .mesh.position
                + self.player.offset;
            let room = crate::renderer::room_light::turned_to_player(
                &crate::renderer::room_light::room_light_at(&self.room_descs, world),
                self.player.yaw,
            );
            instance.model
                .upload_lit_bulb(&self.wgpu_queue, instance.mesh.model_matrix(),
                self.sky_visibility_at(world.x, world.z),
                instance.emissive_drive,
                &room,
                instance.own_light.as_ref(),
                0.0,
            );
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
                let bound = crate::renderer::shadow::mesh_caster_bound(&instance.mesh);
                for prim in instance.mesh.primitives.iter().filter(|p| p.casts_shadow) {
                    shadow_casters.push((
                        &prim.vertex_buffer,
                        &prim.index_buffer,
                        prim.indices.len() as u32,
                        &instance.model.bind_group,
                    ));
                    shadow_bounds.push(bound);
                }
            }
        }
        skinned_casters.extend(
            mirror_only_skinned_draws
                .iter()
                .map(|(model_bg, _tex, joint_bg, vb, ib, count)| (*vb, *ib, *count, *model_bg, *joint_bg)),
        );
        // POSED ONCE (`Levers::skin_once`): each of those primitives posed by
        // one compute pass at the head of the shadow encoder, and drawn into
        // the per-frame shadow tiles as a rigid mesh of its posed positions --
        // where each tile skinned it again. The full sun map, drawn only when
        // it is not baked, still skins. See `skin_compute`.
        let mut posed_live: Vec<(wgpu::Buffer, wgpu::Buffer)> = Vec::new();
        let mut posed: Vec<(crate::renderer::skin_compute::Posed, &wgpu::Buffer, u32, &wgpu::BindGroup)> = Vec::new();
        let mut posed_cache = self.posed_cache.lock().unwrap_or_else(|e| e.into_inner());
        if self.levers.skin_once {
            for instance in meshes.iter().chain(mirror_only_meshes.iter()) {
                let Some(skin) = &instance.mesh.skin else {
                    continue;
                };
                if skin.joint_bind_group.is_none() {
                    continue;
                }
                for prim in &skin.primitives {
                    let p = posed_cache.posed(
                        &self.skin_compute,
                        &self.wgpu_device,
                        &prim.vertex_buffer,
                        prim.vertices.len() as u32,
                        &skin.joint_buffer,
                    );
                    posed_live.push((prim.vertex_buffer.clone(), skin.joint_buffer.clone()));
                    posed.push((p, &prim.index_buffer, prim.indices.len() as u32, &instance.model.bind_group));
                }
            }
        }
        posed_cache.keep_only(&posed_live);
        drop(posed_cache);
        let posed_casters: Vec<crate::renderer::shadow::ShadowMeshDraw> =
            posed.iter().map(|(p, ib, count, model)| (&p.positions, *ib, *count, *model)).collect();
        let no_skinned_casters: [crate::renderer::shadow::ShadowSkinnedDraw; 0] = [];
        let frame_skinned_casters: &[crate::renderer::shadow::ShadowSkinnedDraw] =
            if posed_casters.is_empty() { &skinned_casters } else { &no_skinned_casters };

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
        // The `perf_ab` NoShadows phase switches both off for its window.
        // The A/B schedule, compiled in (`perf_ab::ENABLED`) or asked for from
        // the headset (`Levers::ab_cycle`). Baseline otherwise.
        let ab_phase = self.ab_phase();
        // EVERYTHING THIS FRAME SWITCHES: the lever file, with the schedule's
        // one extra switch on top. See `levers`. Every feature below reads
        // this, never the phase, so a lever and a phase cannot disagree.
        let fx = self.levers.clone().with_phase(ab_phase);
        // GLARE: each lamp's veil for this frame, seen from the head at this
        // frame's exposure, drawn last in the scene pass. See `glare`.
        // A lamp behind a wall is found in the probe pass's depth, when the
        // pass runs this frame.
        let glare_tests_walls = self.probe_pass_runs(&fx, self.stereo_scene(), brush_buffers.is_some());
        let (glare_verts, glare_idx, glare_halos) = if fx.glare {
            let eye_at = |v: &xr::View| glam::Vec3::new(v.pose.position.x, v.pose.position.y, v.pose.position.z);
            // Every two seconds or so: each source's parts as this frame sees
            // them -- how far, how much shows, how bright, and the veil that
            // makes. The headset drew a pendant's veil far fainter than the
            // same numbers on the desk predict (2026-10-02).
            if self.shadow_diag_frames.get() % 120 == 0 {
                let eye = 0.5 * (eye_at(&eye_views[0]) + eye_at(&eye_views[1]));
                for (k, s) in self.glare_sources.iter().enumerate() {
                    let lum = s.radiance.dot(glam::Vec3::new(0.2126, 0.7152, 0.0722));
                    for l in crate::renderer::glare::glare_lobes(s, eye) {
                        let q = crate::renderer::glare::glare_quad(s, &l, eye, post.exposure, fx.glare_strength);
                        // What the characters' capsules leave of it, as
                        // `build_glare` takes it, and the first capsule that
                        // takes any.
                        let caps = &self.glare_capsules;
                        let shielded = 0.5
                            * (crate::renderer::glare::capsule_visibility(l.centre, eye_at(&eye_views[0]), caps)
                                + crate::renderer::glare::capsule_visibility(l.centre, eye_at(&eye_views[1]), caps));
                        let blocker = caps.iter().position(|c| {
                            crate::renderer::glare::capsule_visibility(l.centre, eye, std::slice::from_ref(c)) < 0.99
                        });
                        log::info!(
                            "GLAREDIAG source {k} at {:?} lum {lum:.2} d {:.2} share {:.3} radius {:.3} exposure {:.2} shielded {shielded:.3} of {} capsules{}: {}",
                            s.position.to_array().map(|v| (v * 100.0).round() / 100.0),
                            (eye - l.centre).length(),
                            l.share,
                            l.radius,
                            post.exposure,
                            caps.len(),
                            blocker.map_or(String::new(), |i| {
                                let (a, b, r) = caps[i];
                                format!(
                                    " (capsule {i} {:?}-{:?} r {r:.3}, eye {:?})",
                                    a.to_array().map(|v| (v * 100.0).round() / 100.0),
                                    b.to_array().map(|v| (v * 100.0).round() / 100.0),
                                    eye.to_array().map(|v| (v * 100.0).round() / 100.0),
                                )
                            }),
                            q.map_or("no veil".to_string(), |q| format!(
                                "a {:.3} peak {:.3} core {:.1} deg reach {:.1} deg",
                                q.a, q.peak, q.core_degrees, q.degrees
                            )),
                        );
                    }
                }
            }
            crate::renderer::glare::build_glare(
                &self.glare_sources,
                [eye_at(&eye_views[0]), eye_at(&eye_views[1])],
                cam_right,
                cam_up,
                post.exposure,
                fx.glare_strength,
                glare_tests_walls,
                &self.glare_capsules,
                &glare_adapted,
            )
        } else {
            (Vec::new(), Vec::new(), 0)
        };
        let glare_buffers = (!glare_idx.is_empty()).then(|| {
            (
                self.wgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("glare_vb"),
                    contents: bytemuck::cast_slice(&glare_verts),
                    usage: wgpu::BufferUsages::VERTEX,
                }),
                self.wgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("glare_ib"),
                    contents: bytemuck::cast_slice(&glare_idx),
                    usage: wgpu::BufferUsages::INDEX,
                }),
            )
        });
        // THE EFFECTS' VIEW: the head's axes, and whether the probe pass's
        // depth, which they fade into the walls by, is this frame's. Their
        // particles went up beside the exposure.
        if let (true, Some(gpu)) = (effects_drawn, self.effects_gpu.as_ref()) {
            // An eye pixel's size at unit depth, the least a mote is drawn.
            let pixel = eye_views.first().map_or(0.0, |v| {
                let f = v.fov;
                (f.angle_up.tan() - f.angle_down.tan()) / self.height.max(1) as f32
            });
            gpu.set_view(
                &self.wgpu_queue,
                &crate::renderer::effects::EffectsUniform {
                    right: cam_right.extend(0.0).to_array(),
                    up: cam_up.extend(0.0).to_array(),
                    head: mid_eye.extend(1.0).to_array(),
                    depth: [
                        crate::renderer::brush_pipeline::probe_pass::EYE_NEAR,
                        crate::renderer::brush_pipeline::probe_pass::EYE_FAR,
                        if glare_tests_walls { 1.0 } else { 0.0 },
                        pixel,
                    ],
                },
            );
        }
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
        // THE GAME'S OWN LIGHTS that reach nothing either eye sees -- a fire
        // in a hall behind the player, a torch bounce off-screen -- are left
        // out: every pixel's lamp loop visits every live light, and one fire
        // across the island cost the beach 1.2 ms (bench 2026-10-07_2212).
        // Baked lights stay: their order is their masks' and shadows'.
        let unseen_cut: Vec<Light>;
        let lights: &[Light] = {
            let views: Vec<[glam::Vec4; 6]> = eye_views
                .iter()
                .map(|ev| {
                    let mut planes = crate::renderer::shadow::frustum_planes(
                        Camera::gl_to_wgpu_ndc(Camera::xr_projection(ev.fov, 0.03, 1000.0)) * Camera::xr_view(ev.pose),
                    );
                    planes[5] = glam::Vec4::new(0.0, 0.0, 0.0, 1.0);
                    planes
                })
                .collect();
            let seen = |l: &Light| {
                l.in_level_bake
                    || l.kind == crate::renderer::LightKind::Directional
                    || views.iter().any(|v| crate::renderer::shadow::sphere_in_frustum(v, l.position.extend(l.range.max(0.0))))
            };
            if lights.iter().all(seen) {
                lights
            } else {
                unseen_cut = lights.iter().copied().filter(|l| seen(l)).collect();
                &unseen_cut
            }
        };
        let ranked_idx = crate::renderer::lights::rank_for_budget_indices(
            lights,
            crate::renderer::lights::MAX_LIGHTS,
        );
        let source_lights = lights;
        let ranked: Vec<Light> = ranked_idx.iter().map(|&i| source_lights[i]).collect();
        let lights: &[Light] = if !fx.direct_lights { &[] } else { &ranked };
        // With no directional light lit, the sunless reader is every brush's.
        let any_directional = lights.iter().any(|l| l.kind == crate::renderer::LightKind::Directional);

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
                // Nor on a light that casts none (a flashlight's bounce).
                .filter(|&src| source_lights[src].casts_shadow())
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
        self.lights_uniform.set_terminator_aa(fx.terminator_aa);
        self.lights_uniform.set_surface_lights_apart(fx.surface_light_loop);
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

        // THE PLAYER'S CRISP SHADOWS: the lamps lighting them most that hold
        // no spot slot each fill a characters-only tile, fitted round the
        // player's capsules. See `shadow::MAX_CHARACTER_SHADOWS`.
        // THE MOVING CASTERS' TILES: the lamps lighting the player most that
        // hold no spot slot, then the lamps reaching a door near the eye, each
        // filling a tile fitted round what it shadows of them -- the player's
        // capsules, the doors' leaves. See `shadow::moving_caster_tiles`.
        let door_casters: Vec<crate::renderer::shadow::DoorCaster> = if fx.door_shadows {
            self.doors
                .iter()
                .map(|d| crate::renderer::shadow::DoorCaster { corners: d.corners_in(self.player.offset, self.player.yaw) })
                .collect()
        } else {
            Vec::new()
        };
        let has_body = fx.capsules && self.player.capsules.group_count > 0;
        let character_tiles: Vec<(usize, glam::Mat4)> = if fx.shadows && fx.character_shadows && (has_body || !door_casters.is_empty()) {
            let b = self.player.capsules.groups[0];
            let (centre, radius) = (glam::Vec3::new(b[0], b[1], b[2]), b[3]);
            let lamps: Vec<crate::renderer::shadow::CharacterLamp> = lights
                .iter()
                .enumerate()
                .map(|(i, l)| crate::renderer::shadow::CharacterLamp {
                    position: l.position,
                    direction: l.direction,
                    cos_outer: if l.kind == crate::renderer::LightKind::Spot {
                        (l.cone_angle_deg.to_radians() * 0.5).cos()
                    } else {
                        -1.0
                    },
                    range: l.range,
                    intensity: l.intensity,
                    eligible: l.kind != crate::renderer::LightKind::Directional
                        && !spot_indices.contains(&i)
                        && l.casts_shadow(),
                })
                .collect();
            // The body as the tile is fitted to it: each capsule's two ends.
            let count = (self.player.capsules.groups[1][3] as usize).min(crate::renderer::uniforms::CAPSULES_PER_GROUP);
            let body: Vec<(glam::Vec3, f32)> = (0..count)
                .flat_map(|k| {
                    let (a, b) = (self.player.capsules.capsules[k * 2], self.player.capsules.capsules[k * 2 + 1]);
                    [(glam::Vec3::new(a[0], a[1], a[2]), a[3]), (glam::Vec3::new(b[0], b[1], b[2]), a[3])]
                })
                .collect();
            let tiles = crate::renderer::shadow::moving_caster_tiles(
                &lamps,
                has_body.then_some((centre, radius, body.as_slice())),
                &door_casters,
                head,
                &self.character_shadow_held.borrow(),
            );
            let now: Vec<glam::Vec3> = tiles.iter().map(|(i, _)| lights[*i].position).collect();
            // On CHANGE, as the spot slots are reported: silence means stable.
            if *self.character_shadow_held.borrow() != now {
                log::info!(
                    "CHARSHADOWS lamps {:?} at {:?} for the player at ({:.2}, {:.2}, {:.2}) r {:.2}, {} door(s)",
                    tiles.iter().map(|(i, _)| *i).collect::<Vec<_>>(),
                    now.iter().map(|p| [p.x, p.y, p.z].map(|v| (v * 100.0).round() / 100.0)).collect::<Vec<_>>(),
                    centre.x, centre.y, centre.z, radius,
                    door_casters.len(),
                );
            }
            *self.character_shadow_held.borrow_mut() = now;
            tiles
                .into_iter()
                .filter_map(|(i, spheres)| {
                    let l = &lights[i];
                    let spot = (l.kind == crate::renderer::LightKind::Spot)
                        .then(|| (l.direction, (l.cone_angle_deg.to_radians() * 0.5).cos()));
                    crate::renderer::shadow::character_light_matrix(l.position, spot, &spheres, l.range).map(|m| (i, m))
                })
                .collect()
        } else {
            Vec::new()
        };
        // Each tile's lamp names it by its shadow layer: the lights again,
        // now that the tiles are known (`character_shadow_tile`).
        if !character_tiles.is_empty() {
            self.lights_uniform.set_tile_lamps(&character_tiles.iter().map(|(i, _)| *i).collect::<Vec<_>>());
            self.lights_uniform.upload_frame_split(
                &self.wgpu_queue,
                &frame_lights,
                lights.len(),
                &spot_indices,
                sky_sun.is_some(),
            );
            self.lights_uniform.set_tile_lamps(&[]);
        }
        // THE DOORWAYS THE CULLING MAY SEE THROUGH: every one but those a shut
        // door seals. See `doors::shut_portals`.
        let open_portals: Vec<crate::renderer::uniforms::ProbePortal> = if fx.door_culling {
            crate::renderer::doors::open_portals(&self.probe_portals, &self.shut_portals)
        } else {
            self.probe_portals.clone()
        };
        // The player as this frame's uniforms carry it: which lights hold the
        // characters' tiles. A copy, because the frame holds borrows of the
        // renderer by now.
        let mut frame_player = self.player;
        if !fx.capsules {
            frame_player.capsules.group_count = 0;
        }
        // THE CHARACTERS' FLOOR MIRROR, where the probe pass runs to hold it
        // and there is someone to mirror. See `probe_pass::MIRROR_FORMAT`.
        // Only where the probe pass lays the mirror over the floor -- the
        // single-eye pass that defers its lookups -- and only when a mirrored
        // character can be in view: each character's bound, mirrored in the
        // floor, against both eyes' frusta. Looking ahead the mirrored bodies
        // are under the floor out of sight, and the pass, about a millisecond,
        // is not drawn (headset, 2026-09-30).
        let mirror_in_view = {
            use crate::renderer::brush_pipeline::probe_pass;
            let frusta: Vec<[glam::Vec4; 6]> = eye_views
                .iter()
                .map(|ev| {
                    crate::renderer::shadow::frustum_planes(
                        Camera::gl_to_wgpu_ndc(Camera::xr_projection(ev.fov, probe_pass::EYE_NEAR, probe_pass::EYE_FAR))
                            * Camera::xr_view(ev.pose),
                    )
                })
                .collect();
            // Capsule by capsule, not the whole body's bound: a body's bound
            // is a two-metre cube whose corners, mirrored under the floor,
            // reach into the bottom of almost any view, where its limbs --
            // straight below the player -- are far out of sight until the
            // player looks down.
            let mirrored = |v: [f32; 4]| glam::Vec3::new(v[0], 2.0 * probe_pass::FLOOR_MIRROR_PLANE - v[1], v[2]);
            (0..frame_player.capsules.group_count as usize).any(|g| {
                let count = frame_player.capsules.groups[g * 2 + 1][3] as usize;
                (0..count).any(|k| {
                    let i = g * crate::renderer::uniforms::CAPSULES_PER_GROUP + k;
                    let (a, b) = (frame_player.capsules.capsules[i * 2], frame_player.capsules.capsules[i * 2 + 1]);
                    let (a3, b3) = (mirrored(a), mirrored(b));
                    let r = glam::Vec3::splat(a[3]);
                    let (lo, hi) = (a3.min(b3) - r, a3.max(b3) + r);
                    frusta.iter().any(|planes| crate::renderer::shadow::aabb_in_frustum(planes, lo, hi))
                })
            })
        };
        frame_player.capsules.floor_mirror = fx.floor_mirror
            && fx.deferred_reflection_lookups
            && !self.stereo_scene()
            && self.probe_pass_runs(&fx, false, brush_buffers.is_some())
            && !(skinned_draws.is_empty() && mirror_only_skinned_draws.is_empty())
            && mirror_in_view;
        frame_player.capsules.shadow_lights =
            std::array::from_fn(|k| character_tiles.get(k).map_or(-1.0, |(i, _)| *i as f32));
        // THE PLAYER ON CARDS this frame: the box round their capsules and the
        // atlas row the reflections are told to read, drawn before the first
        // probe pass below. Their body is the mirror-only mesh, and their
        // capsules the first group (`set_capsules`). See `character_cards`.
        let card_frame = match (&self.character_cards, &self.card_atlas) {
            (Some(_), Some(atlas))
                if fx.character_cards && !mirror_only_skinned_draws.is_empty() =>
            {
                atlas.character_rows.first().copied().zip(
                    crate::renderer::character_cards::card_box(&frame_player.capsules, 0, frame_player.yaw),
                )
            }
            _ => None,
        };
        if let Some((row, (centre, half))) = card_frame {
            frame_player.capsules.cards[0] = [centre.x, centre.y, centre.z, row as f32 + 1.0];
            frame_player.capsules.cards[1] = [half.x, half.y, half.z, 0.0];
        }

        // Each spot's shadow is drawn with one matrix and read with another
        // when it has its own near plane -- see `shadow::spot_shadow_matrices`.
        let spot_matrices: Vec<crate::renderer::shadow::SpotShadowMatrices> = spot_indices
            .iter()
            .map(|&i| crate::renderer::shadow::spot_shadow_matrices(&lights[i], self.shadow_map.spot_tile_dim()))
            .collect();
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
                let mut m = [glam::Mat4::IDENTITY; crate::renderer::shadow::SHADOW_MATRICES];
                for (layer, mats) in spot_matrices.iter().enumerate() {
                    m[layer] = mats.lookup;
                }
                for (k, (_, tile)) in character_tiles.iter().enumerate() {
                    m[crate::renderer::shadow::MAX_SPOT_SHADOWS + k] = *tile;
                }
                m
            },
            sun_enabled: sun.is_some(),
            spot_count: spot_indices.len() as u32,
            sun_dynamic_view_proj: dynamic_sun.unwrap_or(glam::Mat4::IDENTITY),
            sun_dynamic_enabled: dynamic_sun.is_some(),
        };

        // No spot casts this frame: the scene draws with its spotless twins.
        // See `XrRenderer::spotless_frame`. Nor any lit surface's light: the
        // twins shade none apart (`lights::without_spot_shadows`).
        self.spotless_frame.store(
            self.levers.spotless_shaders
                && shadow.spot_count == 0
                && !lights.iter().any(crate::renderer::lights::Light::is_surface_light),
            std::sync::atomic::Ordering::Relaxed,
        );
        // A static sun map is recorded only when it went stale.
        let record_sun = shadow.sun_enabled && static_sun.is_none_or(|(_, stale)| stale);
        if record_sun || shadow.sun_dynamic_enabled || shadow.spot_count > 0 || !character_tiles.is_empty() {
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
            // The body posed for every tile below. See `skin_compute`.
            self.skin_compute.dispatch(&mut encoder, posed.iter().map(|(p, ..)| p));
            // No bounds, no culling: `Levers::shadow_mesh_cull`.
            let cull_bounds: &[crate::renderer::shadow::ShadowMeshBound] =
                if self.levers.shadow_mesh_cull { &shadow_bounds } else { &[] };
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
            // small map -- meshes and skinned characters only: the level is in
            // the brushes' baked mask and in the static map already -- and in
            // the same pass the characters' own tiles. See `SUN_ATLAS_TILES`.
            if shadow.sun_dynamic_enabled || !character_tiles.is_empty() {
                if shadow.sun_dynamic_enabled {
                    self.shadow_map.upload_light(
                        &self.wgpu_queue,
                        crate::renderer::shadow::ShadowKind::SunDynamic,
                        shadow.sun_dynamic_view_proj,
                    );
                }
                for (k, (_, tile)) in character_tiles.iter().enumerate() {
                    self.shadow_map.upload_light(&self.wgpu_queue, crate::renderer::shadow::ShadowKind::Character(k), *tile);
                }
                drawn += self.shadow_map.record_moving(
                    &mut encoder,
                    shadow.sun_dynamic_enabled,
                    shadow.sun_dynamic_view_proj,
                    &character_tiles.iter().map(|(_, m)| *m).collect::<Vec<_>>(),
                    &shadow_casters,
                    cull_bounds,
                    frame_skinned_casters,
                    &posed_casters,
                    &tile_meshes,
                );
            }
            // ONE pass for every spot, filling its own tile of the shared
            // atlas. This used to be a pass each, and that was the cost: on a
            // tile GPU a pass is a tile load/store cycle whatever is in it, and
            // culling 95.5% of the caster geometry gave back only 1.8 ms of the
            // 3.2 ms three spots cost.
            let spot_pass: Vec<glam::Mat4> = spot_matrices.iter().map(|m| m.pass).collect();
            for (layer, &m) in spot_pass.iter().enumerate() {
                self.shadow_map.upload_light(
                    &self.wgpu_queue,
                    crate::renderer::shadow::ShadowKind::Spot(layer),
                    m,
                );
            }
            if shadow.spot_count > 0 {
                drawn += self.shadow_map.record_spots(
                    &mut encoder,
                    shadow.spot_count as usize,
                    &spot_pass,
                    solid_caster,
                    brush_caster,
                    &shadow_casters,
                    cull_bounds,
                    frame_skinned_casters,
                    &posed_casters,
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
                // The models each tile takes, of all of them: what
                // `Levers::shadow_mesh_cull` leaves out.
                let all: u32 = shadow_casters.iter().map(|d| d.2).sum();
                let reach = |m: glam::Mat4| {
                    let (n, indices) = crate::renderer::shadow::mesh_casters_reaching(
                        &crate::renderer::shadow::frustum_planes(m),
                        &shadow_casters,
                        cull_bounds,
                    );
                    format!("{n} ({indices} indices)")
                };
                let spots: Vec<String> = spot_pass
                    .iter()
                    .take(shadow.spot_count as usize)
                    .map(|&m| reach(m))
                    .collect();
                log::info!(
                    "SHADOWDIAG mesh casters of {} ({all} indices), cull {}: sun tile {}, near tile {}, spots {:?}",
                    shadow_casters.len(),
                    self.levers.shadow_mesh_cull,
                    if shadow.sun_dynamic_enabled { reach(shadow.sun_dynamic_view_proj) } else { "-".into() },
                    if shadow.sun_dynamic_enabled {
                        reach(crate::renderer::shadow::sun_near_matrix(shadow.sun_dynamic_view_proj))
                    } else {
                        "-".into()
                    },
                    spots,
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
                .take(crate::renderer::space_warp::MAX_SLOTS as usize / 2 - 2)
                .map(|m| {
                    let model = m.mesh.model_matrix();
                    let prev = sw.prev_models.get(&m.model.buffer).copied().unwrap_or(model);
                    (m, model, prev)
                })
                .collect(),
            _ => Vec::new(),
        };
        // WHICH BODIES OF WATER EITHER EYE SEES this frame: only theirs are
        // the waves moved and the surfaces drawn. In the world, where their
        // bounds are, and through the doorways from inside a closed room, as
        // the ground is. See `water_pipeline::water_seen`.
        if fx.water && !self.water_bodies.is_empty() {
            let yaw = glam::Quat::from_rotation_y(self.player.yaw);
            let world_to_player =
                glam::Mat4::from_quat(yaw.inverse()) * glam::Mat4::from_translation(-self.player.offset);
            let eyes: Vec<(glam::Vec3, glam::Mat4)> = eye_views
                .iter()
                .map(|ev| {
                    let vp = Camera::gl_to_wgpu_ndc(Camera::xr_projection(ev.fov, 0.03, 1000.0))
                        * Camera::xr_view(ev.pose);
                    (glam::Vec3::new(ev.pose.position.x, ev.pose.position.y, ev.pose.position.z), vp * world_to_player)
                })
                .collect();
            let views: Vec<[glam::Vec4; 6]> =
                eyes.iter().map(|(_, clip)| crate::renderer::shadow::frustum_planes(*clip)).collect();
            let doorways: Option<Vec<[glam::Vec4; 6]>> = if fx.portal_culling && !self.cull_rooms.is_empty() {
                let mut all = Some(Vec::new());
                for (pos, clip) in &eyes {
                    let eye_world = yaw * *pos + self.player.offset;
                    match crate::renderer::portal_cull::outdoor_frusta(
                        eye_world,
                        *clip,
                        *clip,
                        &self.cull_rooms,
                        &open_portals,
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
            for body in &self.water_bodies {
                let seen = crate::renderer::water_pipeline::water_seen(body.bounds, &views, doorways.as_deref());
                // Back in sight: its next update writes both sets, so
                // SpaceWarp does not see the surface leap.
                if seen && !body.seen.get() {
                    body.waves.reprime();
                }
                body.seen.set(seen);
            }
        } else {
            for body in &self.water_bodies {
                body.seen.set(false);
            }
        }
        // The water's slot follows the meshes': one for every body, as its
        // vertices are in the WORLD and its cameras carry each frame's way
        // into the player's frame. See `space_warp::MotionKind::Water`.
        let warp_water = fx.water
            && self.space_warp.as_ref().is_some_and(|sw| sw.acquired.is_some())
            && self.water_bodies.iter().any(|b| b.seen.get() && b.motion_groups.is_some());
        let warp_per_eye = 1 + warp_meshes.len() as u32 + warp_water as u32;
        // REFLECTIONS MOVE AS WHAT THEY SHOW wherever this frame can say what
        // that is: a single-eye scene pass drawn straight into the eye image,
        // whose alpha holds each pixel's reflected share, with the probe pass
        // running, which holds how far each reflection reached. Anything else
        // -- the diagnostic views, a multiview or offscreen frame -- moves
        // every pixel with its surface, as before. `space_warp_debug` 16384
        // forces that too, to compare. See `space_warp`'s module docs.
        let warp_reflections = plan == crate::renderer::scene_pass_plan::ScenePassPlan::Direct
            && fx.half_res_reflections
            && fx.probes
            && self.debug_view == crate::renderer::brush_pipeline::DebugView::Off
            && brush_buffers.is_some()
            && self.levers.space_warp_debug & 16384 == 0;
        // The brushes write their reflected share into alpha only for this.
        post.reflection_share = warp_reflections && self.space_warp.as_ref().is_some_and(|sw| sw.acquired.is_some());
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
                let mut put = |slot: usize, c: glam::Mat4, p: glam::Mat4, eye: [f32; 4], reflect: [f32; 4]| {
                    let cam = MotionCamera { curr: c.to_cols_array_2d(), prev: p.to_cols_array_2d(), params, eye, reflect };
                    bytes[slot * stride..slot * stride + size].copy_from_slice(bytemuck::bytes_of(&cam));
                };
                let base = eye * warp_per_eye as usize;
                // The world's slot carries the eye and the pixel scale for the
                // brushes' reflections. See `space_warp::reflected_point`.
                let e = eye_views[eye.min(eye_views.len() - 1)].pose.position;
                let reflect = [
                    self.width as f32 / sw.size.0.max(1) as f32,
                    self.height as f32 / sw.size.1.max(1) as f32,
                    if warp_reflections { 1.0 } else { 0.0 },
                    if dbg & 32768 != 0 { 1.0 } else { 0.0 },
                ];
                put(base, curr, world_prev, [e.x, e.y, e.z, 1.0], reflect);
                for (i, (_, model, prev)) in warp_meshes.iter().enumerate() {
                    put(base + 1 + i, curr * *model, prev_view_proj * *prev, [0.0; 4], [0.0; 4]);
                }
                if warp_water {
                    let prev_water = sw.prev.map_or(curr * warp_world_to_player, |(vps, w2p)| vps[eye] * w2p);
                    let world_eye = warp_world_to_player.inverse().transform_point3(glam::Vec3::new(e.x, e.y, e.z));
                    put(
                        base + warp_per_eye as usize - 1,
                        curr * warp_world_to_player,
                        prev_water,
                        [world_eye.x, world_eye.y, world_eye.z, 1.0],
                        [self.water_step, 0.0, 0.0, 0.0],
                    );
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
                    &post, &frame_player,
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
                        // Whole, thin parts and all: the mirror has no thin
                        // pass, and its own resolution.
                        for d in all_mesh_draws {
                            pass.set_bind_group(1, d.model, &[]);
                            pass.set_bind_group(2, d.texture, &[]);
                            pass.set_bind_group(3, d.lightmap, &[]);
                            pass.set_vertex_buffer(0, d.vertices.slice(..));
                            pass.set_index_buffer(d.indices.slice(..), wgpu::IndexFormat::Uint32);
                            pass.draw_indexed(0..d.count, 0, 0..1);
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

            // THE CHARACTERS MIRRORED IN THE FLOOR, this eye's: the uniforms
            // their pass draws with, in the scene uniforms' twin -- the pass
            // is recorded in this eye's own encoder, before its probe pass,
            // bound to the twin. A submit of its own left the GPU idle while
            // the eye's encoder was still being recorded (0.6-0.85 ms a frame
            // for an avatar at half resolution), and copying its camera into
            // the scene's buffer and back cost a full barrier each way (2-3
            // ms): headset, 2026-09-30. See `probe_pass::MIRROR_FORMAT`.
            if frame_player.capsules.floor_mirror {
                use crate::renderer::brush_pipeline::probe_pass;
                let floor_view = view
                    * mirror::reflection_matrix(glam::Vec3::new(0.0, probe_pass::FLOOR_MIRROR_PLANE, 0.0), glam::Vec3::Y);
                let floor_proj = Camera::xr_projection(ev.fov, probe_pass::EYE_NEAR, probe_pass::EYE_FAR);
                let floor_view_proj = Camera::gl_to_wgpu_ndc(floor_proj) * floor_view;
                let floor_eye = floor_view.inverse().transform_point3(glam::Vec3::ZERO);
                // Lit as seen from the mirrored eye, and kept linear: the
                // untoned curve is a clamp, at an exposure that fits.
                let floor_post = crate::renderer::uniforms::PostUpload {
                    exposure: probe_pass::MIRROR_EXPOSURE,
                    tonemap: crate::renderer::tonemap::ToneMapping::None,
                    ..post
                };
                self.uniform_buf.write_scene_stereo_to(
                    &self.wgpu_queue,
                    &self.uniform_buf.twin_buffer,
                    [floor_view_proj; 2],
                    [floor_eye; 2],
                    &shadow,
                    &sky_upload,
                    &floor_post,
                    &frame_player,
                    None,
                );
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
                upload.set_proxies(&crate::renderer::doors::posed_proxies(&self.probe_proxies, &self.doors), player_world, &upload.volumes());
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
            let stereo = self.stereo_scene();
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
                        &open_portals,
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
                    &post, &frame_player, probes_arg,
                );
            } else {
                self.uniform_buf.upload_scene_with_probes(
                    &self.wgpu_queue, eye_view_proj, cam_pos, &shadow, &sky_upload, &post,
                    &frame_player,
                    // `Some(empty)`, not `None`: `None` means "use the level's
                    // probes", which would leave them on and label it off.
                    probes_arg,
                );
            }
            self.ssr_camera_uniform.upload(&self.wgpu_queue, eye_view_proj, cam_pos);
            // THE BRUSHES' REFLECTIONS AT HALF RESOLUTION, in their own pass
            // before the scene pass reads them. See `brush_pipeline::probe_pass`
            // and `probe_pass_runs`.
            let probe_pass = self.probe_pass_runs(&fx, stereo, brush_buffers.is_some());
            // Its pipelines and target: this eye's, or both eyes' at once.
            // MEASUREMENT: under `reader_edit`, the edited readers in the
            // shipped ones' places, chosen the same way.
            let edited = self.reader_edits.as_ref().map(|(_, brushes, _)| brushes);
            let (probe_pipeline, probe_reader, probe_sun_readers, probe_target) = match (&self.stereo_probe, stereo) {
                (Some(sp), true) => (&sp.pass, &sp.reader, Some([&sp.reader_sunless, &sp.reader_baked]), &sp.target),
                // No spot casting: the spotless twins. See `spotless_frame`.
                _ if self.spotless_frame.load(std::sync::atomic::Ordering::Relaxed) && self.scene_cut_pipeline.is_none() => (
                    &self.brush_probe_pass_pipeline,
                    edited.map_or(&self.spotless_readers[0], |e| &e[3]),
                    Some(edited.map_or([&self.spotless_readers[1], &self.spotless_readers[2]], |e| [&e[4], &e[5]])),
                    &self.probe_pass_targets[eye],
                ),
                _ => (
                    &self.brush_probe_pass_pipeline,
                    // MEASUREMENT: the `scene_cut` lever's reader in its place,
                    // for every brush.
                    self.scene_cut_pipeline.as_ref().map_or(edited.map_or(&self.brush_probe_reader_pipeline, |e| &e[0]), |(_, p)| p),
                    self.scene_cut_pipeline.is_none().then_some(edited.map_or(
                        [&self.brush_probe_reader_sunless_pipeline, &self.brush_probe_reader_baked_pipeline],
                        |e| [&e[1], &e[2]],
                    )),
                    &self.probe_pass_targets[eye],
                ),
            };
            // ITS SECONDARY LOOKUPS DEFERRED to a compute pass over just the
            // texels that need them, in the single-eye pass. See `probe_fixup`.
            let deferred_lookups = fx.deferred_reflection_lookups && !stereo;
            // A LITTLE BLUR ON EVERY REFLECTION, after the fix-up, in the
            // single-eye pass; the scene pass then reads the blurred colour.
            // See `probe_blur`.
            let blur = match (&self.probe_blur_groups[eye], &probe_target.soft) {
                (Some(group), Some(soft)) if probe_pass && fx.reflection_blur && !stereo => {
                    Some((group, soft))
                }
                _ => None,
            };
            let probe_read_group =
                blur.map_or(&probe_target.bind_group, |(_, soft)| &soft.bind_group);
            // THE GROUND'S REFLECTION IN THE PROBE PASS TOO, read back in the
            // scene pass as the brushes' is. See `TerrainPipeline::new_probe_pass`.
            let terrain_in_probe_pass =
                probe_pass && deferred_lookups && fx.terrain_probe_pass && terrain_range.is_some();
            // No surface lit by the torch: the poolless twins, the brushes' and
            // the ground's. See `lights::without_pool_maps`.
            let poolless = fx.poolless_shaders && !self.lights_uniform.reads_pool_maps();
            // MEASUREMENT: the `pass_cut` lever's pass in its place.
            let probe_pipeline = match &self.pass_cut_pipeline {
                Some((_, p)) if deferred_lookups => p,
                None if deferred_lookups && poolless => &self.brush_probe_pass_poolless_pipeline,
                _ if deferred_lookups => &self.brush_probe_pass_deferred_pipeline,
                _ => probe_pipeline,
            };

            // THIS EYE IN THE WATER, drawn only in the single-eye scene pass
            // with the probe pass's depth to measure the water by.
            let eye_in_water = eyes_in_water[eye].filter(|_| probe_pass && !stereo);
            if eye_in_water.is_some() || film_age.is_some() {
                let mut u = under_base;
                u.sky[3] = if matches!(eye_in_water, Some((_, EyeWater::Waterline))) { 1.0 } else { 0.0 };
                self.wgpu_queue.write_buffer(&self.underwater.buffer, 0, bytemuck::bytes_of(&u));
            }
            {
                let mut encoder = self.wgpu_device.create_command_encoder(
                    &wgpu::CommandEncoderDescriptor { label: Some("ssr_scene") },
                );
                // The water's waves, once a frame, before the scene pass
                // reads them. See `water_waves`.
                if eye == 0 && fx.water {
                    for body in self.water_bodies.iter().filter(|b| b.seen.get()) {
                        body.waves.update(&self.wgpu_queue, &mut encoder, self.water_seconds);
                    }
                }
                // The player's cards, once a frame, before anything reads them.
                if let (0, Some((row, card_box)), Some(cards), Some(atlas)) =
                    (eye, card_frame, &self.character_cards, &self.card_atlas)
                {
                    let parts: Vec<crate::renderer::character_cards::CardPart> = mirror_only_meshes
                        .iter()
                        .filter_map(|instance| {
                            instance.mesh.skin.as_ref().map(|skin| (instance, skin))
                        })
                        .filter_map(|(instance, skin)| {
                            skin.joint_bind_group
                                .as_ref()
                                .map(|joints| (instance, skin, joints))
                        })
                        .flat_map(|(instance, skin, joints)| {
                            skin.primitives.iter().map(move |prim| {
                                crate::renderer::character_cards::CardPart {
                                    model: &instance.model.bind_group,
                                    texture: &prim.texture.bind_group,
                                    joints,
                                    joint_buffer: &skin.joint_buffer,
                                    source: &prim.vertex_buffer,
                                    vertices: &prim.vertices,
                                    indices: &prim.indices,
                                }
                            })
                        })
                        .collect();
                    cards.record(
                        &self.wgpu_device,
                        &self.wgpu_queue,
                        &mut encoder,
                        &self.floor_mirror_mips,
                        &parts,
                        card_box,
                        frame_player.yaw,
                        &atlas.texture,
                        row,
                        self.pass_timers.as_ref().map(|t| (t, 18)),
                        self.levers.skin_once.then_some(&self.skin_compute),
                    );
                }
                // The torch's pool on each lit surface, once a frame, before
                // any reflection reads it: after this frame's lights and spot
                // shadows. Its own slots, `pools`/`pool_mips`. See `pool_cards`.
                if let (0, Some(pools), Some(atlas)) = (eye, &self.pool_cards, &self.card_atlas) {
                    if self.lights_uniform.reads_pool_maps() {
                        pools.record(
                            &self.wgpu_device,
                            &mut encoder,
                            &self.uniform_buf.bind_group,
                            &self.floor_mirror_mips,
                            &atlas.texture,
                            pools.first_row(atlas.pool_row),
                            self.pass_timers.as_ref().map(|t| (t, 20)),
                        );
                    }
                }
                // THE CHARACTERS MIRRORED IN THE FLOOR, into this eye's probe
                // pass target before its pass lays them over the floor: drawn
                // with the twin's uniforms, then blurred. Single-eye only: the
                // flag is off for a two-eye frame. See `probe_pass::MIRROR_FORMAT`.
                if frame_player.capsules.floor_mirror {
                    let target = &self.probe_pass_targets[eye];
                    {
                        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("floor_mirror"),
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: &target.mirror_levels[0][0],
                                depth_slice: None,
                                resolve_target: None,
                                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store },
                            })],
                            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                                view: &target.mirror_depth_views[0],
                                depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                                stencil_ops: None,
                            }),
                            // Its own slots, `mirror_l`/`mirror_r`; the blur
                            // levels' are `mips_l`/`mips_r`.
                            timestamp_writes: self.pass_timers.as_ref().and_then(|t| t.writes(12 + eye)),
                            ..Default::default()
                        });
                        pass.set_pipeline(&self.floor_mirror_skinned.pipeline);
                        pass.set_bind_group(0, &self.uniform_buf.twin_bind_group, &[]);
                        for (model_bg, tex_bg, joint_bg, vb, ib, count) in skinned_draws.iter().chain(mirror_only_skinned_draws.iter()) {
                            pass.set_bind_group(1, *model_bg, &[]);
                            pass.set_bind_group(2, *tex_bg, &[]);
                            pass.set_bind_group(3, *joint_bg, &[]);
                            pass.set_vertex_buffer(0, vb.slice(..));
                            pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                            pass.draw_indexed(0..*count, 0, 0..1);
                        }
                    }
                    self.floor_mirror_mips.record(
                        &self.wgpu_device,
                        &mut encoder,
                        &target.mirror_levels[0],
                        (target.width, target.height),
                        self.pass_timers.as_ref().map(|t| (t, 14 + eye)),
                    );
                }
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
                            color_attachments: &[
                                Some(wgpu::RenderPassColorAttachment {
                                    view: &t.color_view,
                                    depth_slice: None,
                                    resolve_target: None,
                                    ops: wgpu::Operations {
                                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                                        store: wgpu::StoreOp::Store,
                                    },
                                }),
                                // How far each reflection reached, for
                                // SpaceWarp; 0 (no reflection) where no brush is.
                                Some(wgpu::RenderPassColorAttachment {
                                    view: &t.reach_view,
                                    depth_slice: None,
                                    resolve_target: None,
                                    ops: wgpu::Operations {
                                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                                        store: wgpu::StoreOp::Store,
                                    },
                                }),
                            ],
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
                        // Its photograph slots, as push constants: THIS eye's
                        // table, the one its camera block was just written with.
                        // See `brush_pipeline::PUSH_SCAN`, `UniformBuffer::probe_push`.
                        if probe_pipeline.reads_immediates {
                            pass.set_immediates(0, bytemuck::cast_slice(&self.uniform_buf.probe_push(probes_arg)));
                        }
                        pass.set_bind_group(0, &self.uniform_buf.bind_group, &[]);
                        pass.set_bind_group(1, &self.brush_materials.bind_group, &[]);
                        pass.set_bind_group(2, self.brush_lightmap_bg(), &[]);
                        if deferred_lookups {
                            pass.set_bind_group(3, &self.probe_fixup_passes[eye], &[]);
                        }
                        pass.set_vertex_buffer(0, vb.slice(..));
                        pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                        pass.draw_indexed(0..*count, 0, 0..1);
                        if let (true, Some((index_start, count))) = (terrain_in_probe_pass, terrain_range) {
                            // MEASUREMENT: the `pass_cut` lever's ground in its
                            // place, or the `terrain_reader` lever's inlined one.
                            let terrain = match (&self.terrain_cut_pipeline, &self.terrain_inlined) {
                                (Some((_, p)), _) => p,
                                (None, Some([_, _, pass])) if poolless => pass,
                                (None, Some([_, pass, _])) => pass,
                                (None, None) => match (&self.terrain_dedup_passes, self.levers.pass_dedup) {
                                    // The same picture with less code: see `ground_twins::dedup_passes`.
                                    (Some([_, p]), true) if poolless => p,
                                    (Some([p, _]), true) => p,
                                    _ if poolless => &self.terrain_probe_pass_poolless_pipeline,
                                    _ => &self.terrain_probe_pass_pipeline,
                                },
                            };
                            pass.set_pipeline(&terrain.pipeline);
                            pass.set_bind_group(0, &self.uniform_buf.bind_group, &[]);
                            pass.set_bind_group(1, &self.terrain_material.bind_group, &[]);
                            pass.set_bind_group(3, &self.probe_fixup_passes[eye], &[]);
                            pass.set_vertex_buffer(0, solid_vb.slice(..));
                            pass.set_index_buffer(solid_ib.slice(..), wgpu::IndexFormat::Uint32);
                            // THE WEATHER'S CHUNKS with the ground's weather twin,
                            // after the rest; none while a measurement lever
                            // draws its own ground.
                            let weathered = |c: &crate::renderer::shadow::CasterChunk| {
                                weather_on
                                    && self.terrain_cut_pipeline.is_none()
                                    && self.terrain_inlined.is_none()
                                    && self.weather.as_ref().is_some_and(|w| w.weathered(c.first_index - index_start))
                            };
                            if solid_chunks.is_empty() {
                                pass.draw_indexed(index_start..index_start + count, 0, 0..1);
                            } else {
                                for c in solid_chunks.iter().filter(|c| terrain_chunk_visible(c) && !weathered(c)) {
                                    pass.draw_indexed(c.first_index..c.first_index + c.index_count, 0, 0..1);
                                }
                                if let Some(w) = self.weather.as_ref().filter(|_| solid_chunks.iter().any(|c| weathered(c))) {
                                    let dedup = w.dedup_passes.as_ref().filter(|_| self.levers.pass_dedup);
                                    pass.set_pipeline(match (dedup, poolless) {
                                        (Some([_, p]), true) => &p.pipeline,
                                        (Some([p, _]), false) => &p.pipeline,
                                        (None, true) => &w.twins.pass_poolless.pipeline,
                                        (None, false) => &w.twins.pass.pipeline,
                                    });
                                    pass.set_bind_group(2, &w.maps.bind_group, &[]);
                                    for c in solid_chunks.iter().filter(|c| terrain_chunk_visible(c) && weathered(c)) {
                                        pass.draw_indexed(c.first_index..c.first_index + c.index_count, 0, 0..1);
                                    }
                                }
                            }
                        }
                        drop(pass);
                        if deferred_lookups {
                            // Its own slots, `fix_l`/`fix_r`, after the probe pass's.
                            self.probe_fixups.dispatch(
                                &mut encoder,
                                &self.uniform_buf.bind_group,
                                &self.probe_fixup_targets[eye],
                                self.pass_timers.as_ref().and_then(|t| t.compute_writes(10 + eye)),
                            );
                        }
                        if let Some((group, _)) = blur {
                            // Its own slots, `blur_l`/`blur_r`, after every other.
                            self.probe_blur.dispatch(
                                &mut encoder,
                                group,
                                (t.width, t.height),
                                self.pass_timers
                                    .as_ref()
                                    .and_then(|t| t.compute_writes(16 + eye)),
                            );
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
                            // Its twin for this frame. See `terrain_reader`.
                            pass.set_pipeline(&self.terrain_reader().pipeline);
                            pass.set_bind_group(3, probe_read_group, &[]);
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
                        // THE WEATHER'S CHUNKS with the reader's weather twin
                        // (only where the reader reads the probe pass, which
                        // drew them with the pass's twin).
                        let weathered = |c: &crate::renderer::shadow::CasterChunk| {
                            weather_on
                                && terrain_in_probe_pass
                                && self.terrain_cut_pipeline.is_none()
                                && self.terrain_inlined.is_none()
                                && self.weather.as_ref().is_some_and(|w| w.weathered(c.first_index - index_start))
                        };
                        // EACH CHUNK'S GENTLE TRIANGLES -- the first
                        // `gentle_of` of its indices -- with the slope twins,
                        // after the rest. See `ground_twins`.
                        let gentle_split: Option<&[u32]> = match (&self.slope_split, &self.terrain_gentle) {
                            (Some(split), Some(_)) if terrain_in_probe_pass && self.slope_twins_on() => Some(&split.gentle),
                            _ => None,
                        };
                        let gentle_of = |k: usize| gentle_split.and_then(|g| g.get(k).copied()).unwrap_or(0);
                        // And its steep ones -- the last `steep_of` -- with the
                        // steep twin, when there is one.
                        let steep_split: Option<&[u32]> = match (&self.slope_split, &self.terrain_steep) {
                            (Some(split), Some(_)) if gentle_split.is_some() => Some(&split.steep),
                            _ => None,
                        };
                        let steep_of = |k: usize| steep_split.and_then(|g| g.get(k).copied()).unwrap_or(0);
                        let twin = self.terrain_twin();
                        if solid_chunks.is_empty() {
                            pass.draw_indexed(index_start..index_start + count, 0, 0..1);
                        } else {
                            for (k, c) in solid_chunks.iter().enumerate() {
                                if terrain_chunk_visible(c) && !weathered(c) {
                                    let g = gentle_of(k).min(c.index_count);
                                    let end = c.index_count - steep_of(k).min(c.index_count - g);
                                    if g < end {
                                        pass.draw_indexed(c.first_index + g..c.first_index + end, 0, 0..1);
                                    }
                                    terrain_drawn += c.index_count;
                                } else if !terrain_chunk_visible(c) {
                                    terrain_culled += c.index_count;
                                }
                            }
                            if let Some(gentle) = self.terrain_gentle.as_ref().filter(|_| gentle_split.is_some()) {
                                let mut bound = false;
                                for (k, c) in solid_chunks.iter().enumerate() {
                                    let g = gentle_of(k).min(c.index_count);
                                    if g > 0 && terrain_chunk_visible(c) && !weathered(c) {
                                        if !bound {
                                            pass.set_pipeline(&gentle[twin].pipeline);
                                            bound = true;
                                        }
                                        pass.draw_indexed(c.first_index..c.first_index + g, 0, 0..1);
                                    }
                                }
                            }
                            if let Some(steep) = self.terrain_steep.as_ref().filter(|_| steep_split.is_some()) {
                                let mut bound = false;
                                for (k, c) in solid_chunks.iter().enumerate() {
                                    let g = gentle_of(k).min(c.index_count);
                                    let st = steep_of(k).min(c.index_count - g);
                                    if st > 0 && terrain_chunk_visible(c) && !weathered(c) {
                                        if !bound {
                                            pass.set_pipeline(&steep[twin].pipeline);
                                            bound = true;
                                        }
                                        pass.draw_indexed(c.first_index + c.index_count - st..c.first_index + c.index_count, 0, 0..1);
                                    }
                                }
                            }
                            if let Some(w) = self.weather.as_ref().filter(|_| solid_chunks.iter().any(|c| weathered(c))) {
                                // The gentle triangles by the twin for what
                                // their areas hold, when there are twins.
                                let gentle_w = |k: usize| if w.gentle.is_empty() { 0 } else { gentle_of(k) };
                                pass.set_pipeline(&self.terrain_weather_reader(w).pipeline);
                                pass.set_bind_group(2, &w.maps.bind_group, &[]);
                                for (k, c) in solid_chunks.iter().enumerate().filter(|(_, c)| terrain_chunk_visible(c) && weathered(c)) {
                                    let g = gentle_w(k).min(c.index_count);
                                    if g < c.index_count {
                                        pass.draw_indexed(c.first_index + g..c.first_index + c.index_count, 0, 0..1);
                                    }
                                    terrain_drawn += c.index_count;
                                }
                                for kinds in crate::renderer::weather::WeatherKinds::ALL {
                                    let mut bound = false;
                                    for (k, c) in solid_chunks.iter().enumerate().filter(|(_, c)| terrain_chunk_visible(c) && weathered(c)) {
                                        let g = gentle_w(k).min(c.index_count);
                                        if g == 0 || w.kinds_of(c.first_index - index_start) != kinds {
                                            continue;
                                        }
                                        if !bound {
                                            pass.set_pipeline(&w.gentle[kinds.index()][twin].pipeline);
                                            bound = true;
                                        }
                                        pass.draw_indexed(c.first_index..c.first_index + g, 0, 0..1);
                                    }
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
                        // colour map is a layer of one array. THREE where the
                        // probe pass runs: the faces the sun never reaches --
                        // every face, with no directional light lit -- with
                        // the reader that has no sun in it; the faces whose
                        // baked mask always answers with the one that has no
                        // static sun map; then the rest. The same picture, and
                        // the scene readers back under the Quest's
                        // instruction-cache cliff. See `SunFaces`.
                        let [sunless_end, baked_end] = match (probe_pass, probe_sun_readers) {
                            (true, Some(_)) if !any_directional => [*count; 2],
                            (true, Some(_)) => brush_sun_ends.map(|end| end.min(*count)),
                            _ => [0, 0],
                        };
                        pass.set_vertex_buffer(0, vb.slice(..));
                        pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                        for (pipeline, range) in [
                            (probe_sun_readers.map(|[p, _]| &p.pipeline), 0..sunless_end),
                            (probe_sun_readers.map(|[_, p]| &p.pipeline), sunless_end..baked_end),
                            (
                                Some(if probe_pass { &probe_reader.pipeline } else { self.sp_brush(stereo) }),
                                baked_end..*count,
                            ),
                        ] {
                            let Some(pipeline) = pipeline.filter(|_| !range.is_empty()) else { continue };
                            pass.set_pipeline(pipeline);
                            if probe_pass {
                                pass.set_bind_group(3, probe_read_group, &[]);
                            }
                            pass.set_bind_group(0, &self.uniform_buf.bind_group, &[]);
                            pass.set_bind_group(1, &self.brush_materials.bind_group, &[]);
                            pass.set_bind_group(2, self.brush_lightmap_bg(), &[]);
                            pass.draw_indexed(range, 0, 0..1);
                        }
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
                        // The opaque parts. A primitive with thin parts draws
                        // the rest here and those in the thin pass, after the
                        // sky; glass waits for that too.
                        for d in mesh_draws.iter().filter(|d| !d.blended) {
                            let (ib, count) = match (d.thin, thin_pass) {
                                (Some(t), true) => (&t.solid_index_buffer, t.solid_count),
                                _ => (d.indices, d.count),
                            };
                            if count == 0 {
                                continue;
                            }
                            pass.set_bind_group(1, d.model, &[]);
                            pass.set_bind_group(2, d.texture, &[]);
                            pass.set_bind_group(3, d.lightmap, &[]);
                            pass.set_vertex_buffer(0, d.vertices.slice(..));
                            pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                            pass.draw_indexed(0..count, 0, 0..1);
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
                    // WATER LAST of the world geometry, and that ordering is
                    // the whole reason it looks right: it tints what is
                    // already in the buffer, so everything it is meant to be
                    // seen THROUGH -- the sea bed, a wading leg, a sunken
                    // crate -- has to be there first. Before the sky, which
                    // its depth then keeps off a sea that runs on past the
                    // terrain. See `water_pipeline`.
                    //
                    // UNDER IT: first the water between the eye and all of
                    // that, then the surface's underside -- before its top,
                    // which along a waterline keeps only what is seen from
                    // above, and which wholly under is not drawn. See
                    // `underwater`.
                    if let Some((b, state)) = eye_in_water {
                        let body = &self.water_bodies[b];
                        let water_group = &body.under_groups[body.waves.current()];
                        let u = &self.underwater;
                        u.pipes.draw_veil(&mut pass, state, [&self.uniform_buf.bind_group, water_group, &probe_target.bind_group, &u.group]);
                        u.pipes.draw_underside(
                            &mut pass,
                            [&self.uniform_buf.bind_group, water_group, &u.group],
                            &body.vertex_buffer,
                            &body.index_buffer,
                            body.index_count,
                        );
                    }
                    let wholly_under = matches!(eye_in_water, Some((_, EyeWater::Under)));
                    if fx.water && self.water_bodies.iter().any(|b| b.seen.get()) {
                        pass.set_pipeline(self.sp_water(stereo));
                        pass.set_bind_group(0, &self.uniform_buf.bind_group, &[]);
                        for (i, body) in self.water_bodies.iter().enumerate().filter(|(_, b)| b.seen.get()) {
                            let eyes_body = eye_in_water.is_some_and(|(b, _)| b == i);
                            if wholly_under && eyes_body {
                                continue;
                            }
                            // Along a waterline, its twin that leaves the
                            // view under the line to the underwater one.
                            pass.set_pipeline(if eyes_body { &self.water_pipeline.waterline } else { self.sp_water(stereo) });
                            pass.set_bind_group(1, &body.bind_groups[body.waves.current()], &[]);
                            pass.set_vertex_buffer(0, body.vertex_buffer.slice(..));
                            pass.set_index_buffer(body.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                            pass.draw_indexed(0..body.index_count, 0, 0..1);
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
                    // Not from wholly under the water: it shows only through
                    // the surface, whose underside draws it.
                    if !wholly_under {
                        pass.set_pipeline(self.sp_sky(stereo));
                        pass.set_bind_group(0, &self.uniform_buf.bind_group, &[]);
                        pass.set_bind_group(1, &self.sky.bind_group, &[]);
                        pass.draw(0..3, 0..1);
                    }
                    // THE LAMPS' VEILS' CORES, where nothing stands in front of
                    // the light: after everything opaque and the sky, which
                    // cut them, and before the thin wires and the glass, which
                    // blend over them by what they really cover. Their halos
                    // come last. See `glare`.
                    if let Some((glare_vb, glare_ib)) = glare_buffers.as_ref() {
                        if (glare_halos as usize) < glare_idx.len() {
                            pass.set_pipeline(&self.sp_glare(stereo).core);
                            pass.set_bind_group(0, &self.uniform_buf.bind_group, &[]);
                            pass.set_bind_group(1, probe_read_group, &[]);
                            pass.set_vertex_buffer(0, glare_vb.slice(..));
                            pass.set_index_buffer(glare_ib.slice(..), wgpu::IndexFormat::Uint32);
                            pass.draw_indexed(glare_halos..glare_idx.len() as u32, 0, 0..1);
                        }
                    }
                    // THE THIN PASS: every model's wires, chain links and rims,
                    // never narrower than `levers.thin_parts` eye pixels and
                    // faded by the share they really fill (see
                    // `mesh::thin_parts`). After everything opaque AND the sky,
                    // so a faded wire blends over what is truly behind it --
                    // drawn before the sky it blended over the clear colour.
                    if thin_pass {
                        pass.set_pipeline(self.sp_mesh_thin(stereo));
                        pass.set_bind_group(0, &self.uniform_buf.bind_group, &[]);
                        for d in mesh_draws.iter().filter(|d| !d.blended) {
                            let Some(t) = d.thin else { continue };
                            pass.set_bind_group(1, d.model, &[]);
                            pass.set_bind_group(2, d.texture, &[]);
                            pass.set_bind_group(3, d.lightmap, &[]);
                            pass.set_vertex_buffer(0, d.vertices.slice(..));
                            pass.set_vertex_buffer(1, t.vertex_buffer.slice(..));
                            pass.set_index_buffer(t.thin_index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                            pass.draw_indexed(0..t.thin_count, 0, 0..1);
                        }
                    }
                    // GLASS, after all of that: a see-through surface blends
                    // over what is behind it, so that has to be drawn first.
                    // Among the opaque parts, a lamp's clear globe (alpha 0,
                    // depth written) hid whatever of the lamp came after it.
                    if mesh_draws.iter().any(|d| d.blended) {
                        pass.set_pipeline(self.sp_mesh(stereo));
                        pass.set_bind_group(0, &self.uniform_buf.bind_group, &[]);
                        for d in mesh_draws.iter().filter(|d| d.blended) {
                            pass.set_bind_group(1, d.model, &[]);
                            pass.set_bind_group(2, d.texture, &[]);
                            pass.set_bind_group(3, d.lightmap, &[]);
                            pass.set_vertex_buffer(0, d.vertices.slice(..));
                            pass.set_index_buffer(d.indices.slice(..), wgpu::IndexFormat::Uint32);
                            pass.draw_indexed(0..d.count, 0, 0..1);
                        }
                    }
                    if !particle_verts.is_empty() {
                        pass.set_pipeline(self.sp_particle(stereo));
                        pass.set_bind_group(0, &self.uniform_buf.bind_group, &[]);
                        pass.set_vertex_buffer(0, particle_vb.slice(..));
                        pass.set_index_buffer(particle_ib.slice(..), wgpu::IndexFormat::Uint32);
                        pass.draw_indexed(0..particle_idx.len() as u32, 0, 0..1);
                    }
                    // THE EFFECTS, over everything opaque and the glass, and
                    // under the lamps' halos: the smoke back to front, then
                    // fire, embers and dust screened. Faded into the walls
                    // from the probe pass's depth when it ran. See `effects`.
                    if let (true, Some(gpu)) = (effects_drawn, self.effects_gpu.as_ref()) {
                        gpu.draw(&mut pass, self.sp_effects(stereo), &self.uniform_buf.bind_group, probe_read_group);
                    }
                    // The falling rain and snow and the splashes, last; lit by
                    // the ground's baked sky and sun. Mono passes only
                    // (multiview is parked, `frame-budget-plan` A3).
                    if let (true, false, Some(w)) = (weather_on, stereo, self.weather.as_ref()) {
                        w.particles.draw(&mut pass, &self.uniform_buf.bind_group, &self.terrain_material.bind_group, &w.maps.bind_group, w.counts);
                        // The areas' columns seen from afar, ended by the probe
                        // pass's depth. See `weather::veil_shader`.
                        w.particles.draw_veils(&mut pass, &self.uniform_buf.bind_group, &self.terrain_material.bind_group, &w.maps.bind_group, probe_read_group);
                    }
                    // The lamps' veils' halos, over everything the pass drew;
                    // each veil gone where a wall hides its bulb.
                    if let Some((glare_vb, glare_ib)) = glare_buffers.as_ref() {
                        pass.set_pipeline(&self.sp_glare(stereo).pipeline);
                        pass.set_bind_group(0, &self.uniform_buf.bind_group, &[]);
                        // The probe pass's depth, for the walls: read only
                        // when that pass ran (`glare_tests_walls`).
                        pass.set_bind_group(1, probe_read_group, &[]);
                        pass.set_vertex_buffer(0, glare_vb.slice(..));
                        pass.set_index_buffer(glare_ib.slice(..), wgpu::IndexFormat::Uint32);
                        pass.draw_indexed(0..glare_halos, 0, 0..1);
                    }
                    // THE WET FILM for a moment after surfacing, over all of
                    // it. See `underwater::FILM_SECONDS`.
                    if let (Some(_), true, false, Some(body)) = (film_age, probe_pass, stereo, self.water_bodies.first()) {
                        let u = &self.underwater;
                        u.pipes.draw_film(
                            &mut pass,
                            [&self.uniform_buf.bind_group, &body.under_groups[body.waves.current()], &probe_target.bind_group, &u.group],
                        );
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
                        &post, &frame_player, probes_arg,
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
                    let reflect_group = sw.reflect_groups.get(image_index).map(|g| &g[eye]).filter(|_| warp_reflections);
                    if let Some((vb, ib, n)) = brush_buffers.as_ref() {
                        let kind = if reflect_group.is_some() { MotionKind::BrushReflect } else { MotionKind::Brush };
                        draws.push(MotionDraw { kind, vertices: vb, indices: ib, first: 0, count: *n, slot: base, joints: reflect_group });
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
                    if warp_water {
                        for body in self.water_bodies.iter().filter(|b| b.seen.get()) {
                            let Some(groups) = &body.motion_groups else { continue };
                            draws.push(MotionDraw {
                                kind: MotionKind::Water,
                                vertices: &body.vertex_buffer,
                                indices: &body.index_buffer,
                                first: 0,
                                count: body.index_count,
                                slot: base + warp_per_eye - 1,
                                joints: Some(&groups[body.waves.current()]),
                            });
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
                    // THE EYE IMAGE GOES BACK TO BEING AN ATTACHMENT once the
                    // brushes have read it: OpenXR takes a colour swapchain
                    // image back only in COLOR_ATTACHMENT_OPTIMAL, and the read
                    // above left this layer as a sampled texture.
                    if reflect_group.is_some() {
                        encoder.transition_resources(
                            std::iter::empty(),
                            std::iter::once(wgpu::TextureTransition {
                                texture: &self.eye_targets[image_index][eye]._texture,
                                selector: Some(wgpu_types::TextureSelector { mips: 0..1, layers: eye as u32..eye as u32 + 1 }),
                                state: wgpu::TextureUses::COLOR_TARGET,
                            }),
                        );
                    }
                    self.wgpu_queue.submit(Some(encoder.finish()));
                    // DIAGNOSIS: what the pass left in the images. See
                    // `space_warp::Readback`.
                    if self.levers.space_warp_debug & 8192 != 0 && eye == 0 {
                        if let Some(rb) = sw.readback.as_ref() {
                            // 65536: and the images themselves, to the app's files.
                            let save = (self.levers.space_warp_debug & 65536 != 0)
                                .then(|| ndk_glue::native_activity().external_data_path().join("swdump.bin"));
                            match rb.read(sw.depth_raw[d], sw.depth_has_stencil, sw.motion_raw[m], eye as u32, save.as_deref()) {
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
                // A snap turn or a teleport: nothing may be synthesised from
                // this frame's vectors. See `space_warp::locomotion_jumped`.
                let jumped = sw
                    .prev
                    .is_some_and(|(_, prev_w2p)| crate::renderer::space_warp::locomotion_jumped(prev_w2p, warp_world_to_player));
                if jumped {
                    log::info!("SPACEWARP: locomotion jumped (snap turn or teleport); this frame skips synthesis");
                }
                for info in sw.info.iter_mut() {
                    info.layer_flags = if jumped {
                        xr::sys::CompositionLayerSpaceWarpInfoFlagsFB::FRAME_SKIP
                    } else {
                        xr::sys::CompositionLayerSpaceWarpInfoFlagsFB::EMPTY
                    };
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
            // What the runtime would have had the eyes drawn at, beside the
            // clock it chose. See `dynamic_resolution`.
            if self.recommended_resolution.is_some() && self.levers.dynamic_resolution {
                let answers = self.recommendation_window.take();
                log::info!("{} [ab={ab} ssr={ssr} levers={levers}]", answers.format_line((self.width, self.height)));
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
        // DIAGNOSIS: both eyes' finished images, once per new request. See
        // `Levers::eye_capture`.
        let request = self.levers.eye_capture;
        if request != 0 && request != self.eye_capture_served {
            self.eye_capture_served = request;
            match self.capture_eyes(image_index, request) {
                Ok(path) => log::info!("EYECAPTURE {request} -> {}", path.display()),
                Err(e) => log::warn!("EYECAPTURE {request} failed: {e}"),
            }
            // And the player's cards as this frame drew them: what their
            // reflections read. Needs no swapchain access, so it is served
            // whether or not the eyes could be.
            match self.capture_cards(request) {
                Ok(Some(path)) => log::info!("CARDCAPTURE {request} -> {}", path.display()),
                Ok(None) => {}
                Err(e) => log::warn!("CARDCAPTURE {request} failed: {e}"),
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

    /// THE PLAYER'S CARDS as last drawn, to the app's files:
    /// `cardcapture_<id>.bin`, the bytes `CARD`, width and height as
    /// little-endian u32s, then RGBA8 rows -- the six cards side by side,
    /// albedo times coverage in sRGB, coverage in alpha. `None` with no cards
    /// made. See `character_cards`.
    fn capture_cards(&self, id: u32) -> Result<Option<std::path::PathBuf>, String> {
        let Some(cards) = &self.character_cards else {
            return Ok(None);
        };
        let (width, height, texels) = cards.read_back(&self.wgpu_device, &self.wgpu_queue)?;
        let mut out = Vec::with_capacity(12 + 4 * texels.len());
        out.extend_from_slice(b"CARD");
        out.extend_from_slice(&width.to_le_bytes());
        out.extend_from_slice(&height.to_le_bytes());
        let srgb = |c: f32| {
            let c = c.clamp(0.0, 1.0);
            let v = if c <= 0.003_130_8 {
                12.92 * c
            } else {
                1.055 * c.powf(1.0 / 2.4) - 0.055
            };
            (v * 255.0).round() as u8
        };
        for t in &texels {
            out.extend_from_slice(&[
                srgb(t[0]),
                srgb(t[1]),
                srgb(t[2]),
                (t[3].clamp(0.0, 1.0) * 255.0).round() as u8,
            ]);
        }
        let path = ndk_glue::native_activity()
            .external_data_path()
            .join(format!("cardcapture_{id}.bin"));
        std::fs::write(&path, &out).map_err(|e| format!("{}: {e}", path.display()))?;
        Ok(Some(path))
    }

    /// BOTH EYES' FINISHED IMAGES of swapchain image `image_index`, as the
    /// compositor is about to get them, to the app's files: `eyecapture_<id>.bin`,
    /// the bytes `EYES`, width and height as little-endian u32s, then the left
    /// eye's rows and the right eye's, RGBA8 in sRGB. Waits on the GPU: a
    /// diagnosis, not a feature. The system's screenshot is one view, so a
    /// difference between the eyes -- which has happened here before -- shows
    /// only this way. `quest_app/bench.py --eye-capture` pulls and converts it.
    fn capture_eyes(&self, image_index: usize, id: u32) -> Result<std::path::PathBuf, String> {
        if !self.eye_capture_enabled {
            return Err("the eye images cannot be copied: set debug.spacesoup.eyecapture to 1 before the app starts".into());
        }
        let texture = &self.eye_targets[image_index][0]._texture;
        let (width, height) = (texture.width(), texture.height());
        let row = (width * 4).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let layer_bytes = u64::from(row) * u64::from(height);
        let buffer = self.wgpu_device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("eye_capture"),
            size: 2 * layer_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self.wgpu_device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("eye_capture") });
        for layer in 0..2u32 {
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d { x: 0, y: 0, z: layer },
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &buffer,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: u64::from(layer) * layer_bytes,
                        bytes_per_row: Some(row),
                        rows_per_image: Some(height),
                    },
                },
                wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            );
        }
        // OpenXR takes a colour swapchain image back only as an attachment.
        encoder.transition_resources(
            std::iter::empty(),
            std::iter::once(wgpu::TextureTransition {
                texture,
                selector: Some(wgpu_types::TextureSelector { mips: 0..1, layers: 0..2 }),
                state: wgpu::TextureUses::COLOR_TARGET,
            }),
        );
        self.wgpu_queue.submit(Some(encoder.finish()));
        buffer.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        let _ = self.wgpu_device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        let data = buffer.slice(..).get_mapped_range().map_err(|e| format!("{e:?}"))?;
        let mut out = Vec::with_capacity(12 + 8 * (width * height) as usize);
        out.extend_from_slice(b"EYES");
        out.extend_from_slice(&width.to_le_bytes());
        out.extend_from_slice(&height.to_le_bytes());
        for layer in 0..2u64 {
            for y in 0..u64::from(height) {
                let start = (layer * layer_bytes + y * u64::from(row)) as usize;
                out.extend_from_slice(&data[start..start + (width * 4) as usize]);
            }
        }
        let path = ndk_glue::native_activity().external_data_path().join(format!("eyecapture_{id}.bin"));
        std::fs::write(&path, &out).map_err(|e| format!("{}: {e}", path.display()))?;
        Ok(path)
    }
}
