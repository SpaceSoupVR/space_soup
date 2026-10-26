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

    #[allow(clippy::too_many_arguments, reason = "GPU setup passes each resource explicitly; bundling them is a separate refactor")]
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

        terrain: Option<(&[SolidVertex], &[u32])>,

        terrain_chunks: &[crate::renderer::shadow::CasterChunk],

        brushes: Option<(&[BrushVertex], &[u32])>,
        mirror: Option<MirrorSurface>,
    ) -> Result<Vec<xr::CompositionLayerProjectionView<xr::Vulkan>>, Box<dyn std::error::Error>>
    {

        self.update_time_of_day();

        self.ensure_ground_map();
        let image_index = self.swapchain.acquire_image()? as usize;
        self.swapchain.wait_image(xr::Duration::INFINITE)?;
        let cpu_start = std::time::Instant::now();

        let (view_flags, mut eye_views) =
            session.locate_views(xr::ViewConfigurationType::PRIMARY_STEREO, time, stage)?;
        let located = view_flags.contains(xr::ViewStateFlags::ORIENTATION_VALID)
            && view_flags.contains(xr::ViewStateFlags::POSITION_VALID);
        if located && eye_views.len() >= 2 {
            self.last_fov = Some([eye_views[0].fov, eye_views[1].fov]);
        }

        let foveation = self.frame_levers().foveation;
        self.apply_foveation(foveation);

        let tracked_poses: Option<Vec<xr::Posef>> = located.then(|| eye_views.iter().map(|v| v.pose).collect());

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

        if self.levers.same_eyes && eye_views.len() >= 2 {
            eye_views[1] = eye_views[0];
        }

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

        let mut terrain_range: Option<(u32, u32)> = None;
        if let Some((terrain_verts, terrain_idx)) = terrain {
            if !terrain_verts.is_empty() && !terrain_idx.is_empty() {

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

        let brush_geometry = brushes.filter(|(v, i)| !v.is_empty() && !i.is_empty());

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

        let effects_on = self.frame_levers().effects;

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

            let fires = if effects_on {
                crate::renderer::effects::meter_samples(&self.effect_emitters, head_world, &self.room_descs)
            } else {
                Vec::new()
            };
            let mut eye = self.eye.borrow_mut();
            let mut metered = eye.meter_with(head_world, gaze, &fires);

            if let Some((b, state)) = eyes_in_water[0].or(eyes_in_water[1]) {
                let u = &self.water_bodies[b].uniform;
                let water = crate::renderer::underwater::in_water_luminance(u, &under_base, u.extinction[3] - head_world.y);
                metered = crate::renderer::underwater::meter_in_water(metered, water, crate::renderer::underwater::eye_share(state));
            }
            let auto = eye.update(metered, dt);
            if self.shadow_diag_frames.get() % 120 == 0 {
                log::info!("EXPOSURE meter {metered:.4} -> x{auto:.2} (auto {})", self.auto_exposure);
            }

            let night_vision = self.tod_night_vision(eye.adapted_luminance());
            crate::renderer::uniforms::PostUpload {
                exposure: self.post.exposure * if self.auto_exposure { auto } else { 1.0 },
                terrain_detail_distance: self.levers.terrain_detail_distance,
                night_vision,
                ..self.post
            }
        };

        let mid_eye = {
            let at = |v: &xr::View| glam::Vec3::new(v.pose.position.x, v.pose.position.y, v.pose.position.z);
            0.5 * (at(&eye_views[0]) + at(&eye_views[eye_views.len() - 1]))
        };

        let mut effects_drawn = effects_on && !(self.effect_emitters.is_empty() && self.splashes.is_empty()) && self.effects_gpu.is_some();
        if effects_drawn {
            let yaw = glam::Quat::from_rotation_y(self.player.yaw);
            let (offset, descs) = (self.player.offset, &self.room_descs);

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

            let now = time.as_nanos() as f64 * 1e-9;
            let splashes: Vec<crate::renderer::effects::Splash> = self
                .splashes
                .iter()
                .map(|s| crate::renderer::effects::Splash { born: now - (self.water_seconds - s.born), ..*s })
                .collect();

            let lv = self.frame_levers();
            let world_to_player = glam::Mat4::from_quat(yaw.inverse()) * glam::Mat4::from_translation(-offset);
            let views: Vec<[glam::Vec4; 6]> = eye_views
                .iter()
                .map(|ev| {
                    let vp = Camera::gl_to_wgpu_ndc(Camera::xr_projection(ev.fov, crate::renderer::space_warp::NEAR_Z, crate::renderer::space_warp::FAR_Z)) * Camera::xr_view(ev.pose);
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

        let mut shadow_casters: Vec<crate::renderer::shadow::ShadowMeshDraw> = Vec::new();

        let mut shadow_bounds: Vec<crate::renderer::shadow::ShadowMeshBound> = Vec::new();

        let mut tile_meshes: Vec<usize> = Vec::new();

        let mut glare_adapted = vec![false; self.glare_sources.len()];
        for instance in meshes {

            let world = glam::Quat::from_rotation_y(self.player.yaw)
                * instance.mesh.position
                + self.player.offset;
            let sky_vis = self.sky_visibility_at(world.x, world.z);

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

        let mut skinned_casters: Vec<crate::renderer::shadow::ShadowSkinnedDraw> = skinned_draws
            .iter()
            .map(|(model_bg, _tex, joint_bg, vb, ib, count)| {
                (*vb, *ib, *count, *model_bg, *joint_bg)
            })
            .collect();

        let mut mirror_only_skinned_draws: Vec<SkinnedDraw> = Vec::new();
        let mut mirror_only_layered_draws: Vec<LayeredDraw> = Vec::new();
        for instance in mirror_only_meshes {

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

        let sky_upload =
            crate::renderer::uniforms::SkyUpload::from(&self.sky.irradiance);

        let ab_phase = self.ab_phase();

        let fx = self.levers.clone().with_phase(ab_phase);

        let glare_tests_walls = self.probe_pass_runs(&fx, self.stereo_scene(), brush_buffers.is_some());
        let (glare_verts, glare_idx, glare_halos) = if fx.glare {
            let eye_at = |v: &xr::View| glam::Vec3::new(v.pose.position.x, v.pose.position.y, v.pose.position.z);

            if self.shadow_diag_frames.get() % 120 == 0 {
                let eye = 0.5 * (eye_at(&eye_views[0]) + eye_at(&eye_views[1]));
                for (k, s) in self.glare_sources.iter().enumerate() {
                    let lum = s.radiance.dot(glam::Vec3::new(0.2126, 0.7152, 0.0722));
                    for l in crate::renderer::glare::glare_lobes(s, eye) {
                        let q = crate::renderer::glare::glare_quad(s, &l, eye, post.exposure, fx.glare_strength);

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

        if let (true, Some(gpu)) = (effects_drawn, self.effects_gpu.as_ref()) {

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

        let without_stationary: Vec<Light>;
        let lights: &[Light] = if fx.stationary_lights {
            lights
        } else {
            without_stationary = lights.iter().copied().filter(|l| l.mask_channel.is_none()).collect();
            &without_stationary
        };

        let unseen_cut: Vec<Light>;
        let lights: &[Light] = {
            let views: Vec<[glam::Vec4; 6]> = eye_views
                .iter()
                .map(|ev| {
                    let mut planes = crate::renderer::shadow::frustum_planes(
                        Camera::gl_to_wgpu_ndc(Camera::xr_projection(ev.fov, crate::renderer::space_warp::NEAR_Z, crate::renderer::space_warp::FAR_Z)) * Camera::xr_view(ev.pose),
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

        let any_directional = lights.iter().any(|l| l.kind == crate::renderer::LightKind::Directional);

        let sun = want_sun
            .then(|| lights.iter().find(|l| l.kind == crate::renderer::LightKind::Directional))
            .flatten();

        let spot_indices: Vec<usize> = if want_spot && !lights.is_empty() {
            let spot_scores: Vec<(usize, f32)> = ranked_idx
                .iter()
                .copied()
                .filter(|&src| source_lights[src].kind == crate::renderer::LightKind::Spot)

                .filter(|&src| source_lights[src].mask_channel.is_none())

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

            chosen
                .iter()
                .filter_map(|&src| ranked_idx.iter().position(|&r| r == src))
                .collect()
        } else {
            Vec::new()
        };

        let forward = (glam::Quat::from_xyzw(
            eye_views[0].pose.orientation.x,
            eye_views[0].pose.orientation.y,
            eye_views[0].pose.orientation.z,
            eye_views[0].pose.orientation.w,
        ) * glam::Vec3::NEG_Z)
            .normalize_or_zero();
        let sun_radius = 30.0_f32;
        let focus = head + forward * (sun_radius * 0.5);

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

        let dynamic_sun = static_sun.and(sun).filter(|_| fx.sun_dynamic).map(|l| {
            let world_dir = glam::Quat::from_rotation_y(self.player.yaw) * l.direction;
            let head_world = player_to_world.transform_point3(head);
            crate::renderer::lights::dynamic_sun_matrix(world_dir, head_world) * player_to_world
        });

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

        let open_portals: Vec<crate::renderer::uniforms::ProbePortal> = if fx.door_culling {
            crate::renderer::doors::open_portals(&self.probe_portals, &self.shut_portals)
        } else {
            self.probe_portals.clone()
        };

        let mut frame_player = self.player;
        if !fx.capsules {
            frame_player.capsules.group_count = 0;
        }

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

        self.spotless_frame.store(
            self.levers.spotless_shaders
                && shadow.spot_count == 0
                && !lights.iter().any(crate::renderer::lights::Light::is_surface_light),
            std::sync::atomic::Ordering::Relaxed,
        );

        let record_sun = shadow.sun_enabled && static_sun.is_none_or(|(_, stale)| stale);
        if record_sun || shadow.sun_dynamic_enabled || shadow.spot_count > 0 || !character_tiles.is_empty() {
            let solid_caster = (!solid_idx.is_empty())
                .then_some((&solid_vb, &solid_ib, solid_idx.len() as u32));
            let brush_caster = brush_buffers
                .as_ref()
                .map(|(vb, ib, count)| (vb, ib, *count));

            let mut drawn = 0u32;
            self.shadow_diag_frames.set(self.shadow_diag_frames.get() + 1);
            let diag = self.shadow_diag_frames.get() % 120 == 0;

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

            self.skin_compute.dispatch(&mut encoder, posed.iter().map(|(p, ..)| p));

            let cull_bounds: &[crate::renderer::shadow::ShadowMeshBound] =
                if self.levers.shadow_mesh_cull { &shadow_bounds } else { &[] };
            if record_sun {

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

                let passes = u32::from(shadow.spot_count > 0) + u32::from(record_sun);
                let uncalled = (solid_caster.map(|(_, _, n)| n).unwrap_or(0)
                    + brush_caster.map(|(_, _, n)| n).unwrap_or(0))
                    * passes;
                log::info!(
                    "SHADOWDIAG culled: drew {drawn} of {uncalled} indices over {passes} pass(es), {} chunks",
                    solid_chunks.len(),
                );

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

        let reflective_brushes = self.brush_materials.reflective && brush_geometry.is_some();

        let needs_scene_depth = crate::renderer::scene_pass_plan::needs_scene_readback(
            mirror_quad.is_some(),
            reflective_brushes || solid_ranges.iter().any(|(_, _, _, r)| *r > 0.0),

            self.screen_space_reflections,
        );

        let needs_scene_depth = needs_scene_depth && !fx.direct_path;

        let stereo_frame = self.multiview_scene && self.stereo_pipelines.is_some();
        let plan = crate::renderer::scene_pass_plan::ScenePassPlan::for_frame_with(needs_scene_depth, stereo_frame);

        if let Some(stream) = self.probe_stream.borrow_mut().as_mut() {
            stream.begin_frame();
        }

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

        let warp_planes: [[glam::Vec4; 6]; 2] =
            std::array::from_fn(|e| crate::renderer::shadow::frustum_planes(warp_view_proj[e]));
        let warp_cull = self.levers.space_warp_debug & 131072 == 0;
        let brush_motion_ranges: Option<[Vec<(u32, u32)>; 2]> =
            match (warp_cull && self.space_warp.as_ref().is_some_and(|sw| sw.acquired.is_some()), brush_geometry) {
                (true, Some((v, i))) => {
                    let i: &[u32] = partitioned.as_ref().map_or(i, |(p, _)| p.as_slice());
                    let runs = crate::renderer::space_warp::index_runs(i, |k| glam::Vec3::from(v[k as usize].position));
                    Some(std::array::from_fn(|e| crate::renderer::space_warp::visible_ranges(&runs, &warp_planes[e])))
                }
                _ => None,
            };

        let warp_clocks = (self.weather.as_ref().map_or(0.0, |w| w.seconds), time.as_nanos() as f64 * 1e-9);
        let warp_steps = self.space_warp.as_ref().and_then(|sw| sw.prev_clocks).map_or((0.0, 0.0), |(w, f)| {
            (((warp_clocks.0 - w).clamp(0.0, 0.1)) as f32, ((warp_clocks.1 - f).clamp(0.0, 0.1)) as f32)
        });

        if fx.water && !self.water_bodies.is_empty() {
            let yaw = glam::Quat::from_rotation_y(self.player.yaw);
            let world_to_player =
                glam::Mat4::from_quat(yaw.inverse()) * glam::Mat4::from_translation(-self.player.offset);
            let eyes: Vec<(glam::Vec3, glam::Mat4)> = eye_views
                .iter()
                .map(|ev| {
                    let vp = Camera::gl_to_wgpu_ndc(Camera::xr_projection(ev.fov, crate::renderer::space_warp::NEAR_Z, crate::renderer::space_warp::FAR_Z))
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

        let warp_water = fx.water
            && self.space_warp.as_ref().is_some_and(|sw| sw.acquired.is_some())
            && self.water_bodies.iter().any(|b| b.seen.get() && b.motion_groups.is_some());
        let warp_per_eye = 1 + warp_meshes.len() as u32 + warp_water as u32;

        let warp_reflections = plan == crate::renderer::scene_pass_plan::ScenePassPlan::Direct
            && fx.half_res_reflections
            && fx.probes
            && self.debug_view == crate::renderer::brush_pipeline::DebugView::Off
            && brush_buffers.is_some()
            && self.levers.space_warp_debug & 16384 == 0;

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
                    warp_steps.0,
                    warp_steps.1,
                ];
                let mut put = |slot: usize, c: glam::Mat4, p: glam::Mat4, eye: [f32; 4], reflect: [f32; 4]| {
                    let cam = MotionCamera { curr: c.to_cols_array_2d(), prev: p.to_cols_array_2d(), params, eye, reflect };
                    bytes[slot * stride..slot * stride + size].copy_from_slice(bytemuck::bytes_of(&cam));
                };
                let base = eye * warp_per_eye as usize;

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
            let proj = Camera::xr_projection(ev.fov, crate::renderer::space_warp::NEAR_Z, crate::renderer::space_warp::FAR_Z);

            if let Some(m) = &mirror {
                let reflect = mirror::reflection_matrix(m.position, m.normal());
                let mirror_view = view * reflect;

                let world_plane = mirror::world_plane_equation(m.position, m.normal());
                let eye_plane = mirror::plane_to_eye_space(mirror_view.inverse(), world_plane);
                let mirror_proj = mirror::oblique_near_clip(proj, eye_plane);
                let mirror_view_proj = Camera::gl_to_wgpu_ndc(mirror_proj) * mirror_view;

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

                        pass.set_pipeline(&self.brush_mirror_pipeline.pipeline);
                        pass.set_bind_group(0, &self.uniform_buf.bind_group, &[]);
                        pass.set_bind_group(1, &self.brush_materials.bind_group, &[]);
                        pass.set_bind_group(2, self.brush_lightmap_bg(), &[]);
                        pass.set_vertex_buffer(0, vb.slice(..));
                        pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                        pass.draw_indexed(0..*count, 0, 0..1);
                    }
                    if !layered_draws.is_empty() || !mirror_only_layered_draws.is_empty() {

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

                    {
                        pass.set_pipeline(&self.sky_mirror_pipeline.pipeline);
                        pass.set_bind_group(0, &self.uniform_buf.bind_group, &[]);
                        pass.set_bind_group(1, &self.sky.bind_group, &[]);
                        pass.draw(0..3, 0..1);
                    }
                }
                self.wgpu_queue.submit(Some(encoder.finish()));
            }

            if frame_player.capsules.floor_mirror {
                use crate::renderer::brush_pipeline::probe_pass;
                let floor_view = view
                    * mirror::reflection_matrix(glam::Vec3::new(0.0, probe_pass::FLOOR_MIRROR_PLANE, 0.0), glam::Vec3::Y);
                let floor_proj = Camera::xr_projection(ev.fov, probe_pass::EYE_NEAR, probe_pass::EYE_FAR);
                let floor_view_proj = Camera::gl_to_wgpu_ndc(floor_proj) * floor_view;
                let floor_eye = floor_view.inverse().transform_point3(glam::Vec3::ZERO);

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

            let both_view_proj: [glam::Mat4; 2] = std::array::from_fn(|i| {
                let v = &eye_views[i];
                Camera::gl_to_wgpu_ndc(Camera::xr_projection(v.fov, 0.03, 1000.0))
                    * Camera::xr_view(v.pose)
            });
            let both_cam_pos: [glam::Vec3; 2] = std::array::from_fn(|i| {
                let p = eye_views[i].pose.position;
                glam::Vec3::new(p.x, p.y, p.z)
            });

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

                upload.fill_brightness(&self.probe_brightness);
                upload.fill_volumes(&self.probe_rooms);

                upload.set_portals(&self.probe_portals, player_world, &upload.volumes());

                upload.set_proxies(&crate::renderer::doors::posed_proxies(&self.probe_proxies, &self.doors), player_world, &upload.volumes());
                upload.set_proxy_fields(&self.proxy_field_slots);

                let sky_layer = self.probe_stream.borrow().as_ref().and_then(|s| s.sky_layer());
                let ground = if fx.ground_trace { self.ground_placement } else { None };
                upload.set_outdoors(self.probe_outdoor_volume, sky_layer, ground);
                let building_layer = self.probe_stream.borrow().as_ref().and_then(|s| s.building_layer());
                upload.set_buildings(&self.probe_buildings, building_layer);

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

            let stereo = self.stereo_scene();

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

            let terrain_chunk_visible = |c: &crate::renderer::shadow::CasterChunk| {

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

                    probes_arg,
                );
            }
            self.ssr_camera_uniform.upload(&self.wgpu_queue, eye_view_proj, cam_pos);

            let probe_pass = self.probe_pass_runs(&fx, stereo, brush_buffers.is_some());

            let edited = self.reader_edits.as_ref().map(|(_, brushes, _)| brushes);
            let (probe_pipeline, probe_reader, probe_sun_readers, probe_target) = match (&self.stereo_probe, stereo) {
                (Some(sp), true) => (&sp.pass, &sp.reader, Some([&sp.reader_sunless, &sp.reader_baked]), &sp.target),

                _ if self.spotless_frame.load(std::sync::atomic::Ordering::Relaxed) && self.scene_cut_pipeline.is_none() => (
                    &self.brush_probe_pass_pipeline,
                    edited.map_or(&self.spotless_readers[0], |e| &e[3]),
                    Some(edited.map_or([&self.spotless_readers[1], &self.spotless_readers[2]], |e| [&e[4], &e[5]])),
                    &self.probe_pass_targets[eye],
                ),
                _ => (
                    &self.brush_probe_pass_pipeline,

                    self.scene_cut_pipeline.as_ref().map_or(edited.map_or(&self.brush_probe_reader_pipeline, |e| &e[0]), |(_, p)| p),
                    self.scene_cut_pipeline.is_none().then_some(edited.map_or(
                        [&self.brush_probe_reader_sunless_pipeline, &self.brush_probe_reader_baked_pipeline],
                        |e| [&e[1], &e[2]],
                    )),
                    &self.probe_pass_targets[eye],
                ),
            };

            let deferred_lookups = fx.deferred_reflection_lookups && !stereo;

            let blur = match (&self.probe_blur_groups[eye], &probe_target.soft) {
                (Some(group), Some(soft)) if probe_pass && fx.reflection_blur && !stereo => {
                    Some((group, soft))
                }
                _ => None,
            };
            let probe_read_group =
                blur.map_or(&probe_target.bind_group, |(_, soft)| &soft.bind_group);

            let terrain_in_probe_pass =
                probe_pass && deferred_lookups && fx.terrain_probe_pass && terrain_range.is_some();

            let poolless = fx.poolless_shaders && !self.lights_uniform.reads_pool_maps();

            let probe_pipeline = match &self.pass_cut_pipeline {
                Some((_, p)) if deferred_lookups => p,
                None if deferred_lookups && poolless => &self.brush_probe_pass_poolless_pipeline,
                _ if deferred_lookups => &self.brush_probe_pass_deferred_pipeline,
                _ => probe_pipeline,
            };

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

                if eye == 0 && fx.water {
                    for body in self.water_bodies.iter().filter(|b| b.seen.get()) {
                        body.waves.update(&self.wgpu_queue, &mut encoder, self.water_seconds);
                    }
                }

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

                if probe_pass && (!stereo || eye == 0) {
                    if let Some((vb, ib, count)) = &brush_buffers {
                        if deferred_lookups {
                            self.probe_fixups.clear(&mut encoder);
                        }
                        let t = probe_target;
                        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("probe_pass"),

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

                            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                                view: &t.depth_view,
                                depth_ops: Some(wgpu::Operations {
                                    load: wgpu::LoadOp::Clear(1.0),
                                    store: wgpu::StoreOp::Store,
                                }),
                                stencil_ops: None,
                            }),

                            timestamp_writes: self.pass_timers.as_ref().and_then(|t| t.writes(8 + eye)),
                            ..Default::default()
                        });
                        pass.set_pipeline(&probe_pipeline.pipeline);

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

                            let terrain = match (&self.terrain_cut_pipeline, &self.terrain_inlined) {
                                (Some((_, p)), _) => p,
                                (None, Some([_, _, pass])) if poolless => pass,
                                (None, Some([_, pass, _])) => pass,
                                (None, None) => match (&self.terrain_dedup_passes, self.levers.pass_dedup) {

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

                            self.probe_fixups.dispatch(
                                &mut encoder,
                                &self.uniform_buf.bind_group,
                                &self.probe_fixup_targets[eye],
                                self.pass_timers.as_ref().and_then(|t| t.compute_writes(10 + eye)),
                            );
                        }
                        if let Some((group, _)) = blur {

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

                if !stereo || eye == 0 {

                    let target = &self.scene_targets[eye];

                    let swap_view = &self.eye_targets[image_index][eye].view;

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

                        timestamp_writes: self.pass_timers.as_ref().and_then(|t| t.writes(eye * 2)),
                        ..Default::default()
                    });

                    if fx.half_viewport {
                        pass.set_viewport(0.0, 0.0, (self.width / 2) as f32, (self.height / 2) as f32, 0.0, 1.0);
                    }

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

                            if terrain_range == Some((*index_start, *count)) {
                                continue;
                            }
                            pass.set_bind_group(1, self.cuboid_lightmap_bg(lightmap_key.as_deref()), &[]);
                            pass.draw_indexed(*index_start..*index_start + *count, 0, 0..1);
                        }
                    }
                    if let Some((index_start, count)) = terrain_range {
                        if terrain_in_probe_pass {

                            pass.set_pipeline(&self.terrain_reader().pipeline);
                            pass.set_bind_group(3, probe_read_group, &[]);
                        } else {
                            pass.set_pipeline(self.sp_terrain(stereo));
                        }
                        pass.set_bind_group(0, &self.uniform_buf.bind_group, &[]);
                        pass.set_bind_group(1, &self.terrain_material.bind_group, &[]);
                        pass.set_vertex_buffer(0, solid_vb.slice(..));
                        pass.set_index_buffer(solid_ib.slice(..), wgpu::IndexFormat::Uint32);

                        let weathered = |c: &crate::renderer::shadow::CasterChunk| {
                            weather_on
                                && terrain_in_probe_pass
                                && self.terrain_cut_pipeline.is_none()
                                && self.terrain_inlined.is_none()
                                && self.weather.as_ref().is_some_and(|w| w.weathered(c.first_index - index_start))
                        };

                        let gentle_split: Option<&[u32]> = match (&self.slope_split, &self.terrain_gentle) {
                            (Some(split), Some(_)) if terrain_in_probe_pass && self.slope_twins_on() => Some(&split.gentle),
                            _ => None,
                        };
                        let gentle_of = |k: usize| gentle_split.and_then(|g| g.get(k).copied()).unwrap_or(0);

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

                        if crate::renderer::brush_pipeline::SEAL_BRUSH_CRACKS {
                            pass.set_pipeline(self.sp_brush_seal(stereo));
                            pass.set_bind_group(0, &self.uniform_buf.bind_group, &[]);
                            pass.draw_indexed(0..*count, 0, 0..1);
                        }
                    }
                    if !layered_draws.is_empty() {

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

                            pass.set_pipeline(if eyes_body { &self.water_pipeline.waterline } else { self.sp_water(stereo) });
                            pass.set_bind_group(1, &body.bind_groups[body.waves.current()], &[]);
                            pass.set_vertex_buffer(0, body.vertex_buffer.slice(..));
                            pass.set_index_buffer(body.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                            pass.draw_indexed(0..body.index_count, 0, 0..1);
                        }
                    }

                    if !wholly_under {
                        pass.set_pipeline(self.sp_sky(stereo));
                        pass.set_bind_group(0, &self.uniform_buf.bind_group, &[]);
                        pass.set_bind_group(1, &self.sky.bind_group, &[]);
                        pass.draw(0..3, 0..1);
                    }

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

                    if let (true, Some(gpu)) = (effects_drawn, self.effects_gpu.as_ref()) {
                        gpu.draw(&mut pass, self.sp_effects(stereo), &self.uniform_buf.bind_group, probe_read_group);
                    }

                    if let (true, false, Some(w)) = (weather_on, stereo, self.weather.as_ref()) {
                        w.particles.draw(&mut pass, &self.uniform_buf.bind_group, &self.terrain_material.bind_group, &w.maps.bind_group, w.counts);

                        w.particles.draw_veils(&mut pass, &self.uniform_buf.bind_group, &self.terrain_material.bind_group, &w.maps.bind_group, probe_read_group);
                    }

                    if let Some((glare_vb, glare_ib)) = glare_buffers.as_ref() {
                        pass.set_pipeline(&self.sp_glare(stereo).pipeline);
                        pass.set_bind_group(0, &self.uniform_buf.bind_group, &[]);

                        pass.set_bind_group(1, probe_read_group, &[]);
                        pass.set_vertex_buffer(0, glare_vb.slice(..));
                        pass.set_index_buffer(glare_ib.slice(..), wgpu::IndexFormat::Uint32);
                        pass.draw_indexed(0..glare_halos, 0, 0..1);
                    }

                    if let (Some(_), true, false, Some(body)) = (film_age, probe_pass, stereo, self.water_bodies.first()) {
                        let u = &self.underwater;
                        u.pipes.draw_film(
                            &mut pass,
                            [&self.uniform_buf.bind_group, &body.under_groups[body.waves.current()], &probe_target.bind_group, &u.group],
                        );
                    }
                }

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

                if plan.samples_scene_back() {

                    let prep = 4 + eye;
                    self.ssr_pipelines.resolve_depth(
                        &mut encoder,
                        &self.scene_targets[eye],
                        self.pass_timers.as_ref().and_then(|t| t.span_start(prep)),
                    );

                    self.ssr_pipelines.build_hi_z(
                        &mut encoder,
                        &self.scene_targets[eye],
                        self.pass_timers.as_ref().and_then(|t| t.span_end(prep)),
                    );
                    self.ssr_pipelines.generate_mips(&mut encoder, &self.scene_targets[eye]);

                if self.buffered_reflections && reflective_brushes {
                    let target = &self.reflection_targets[eye];

                    let refl_slot = 6 + eye;
                    {
                        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("ssr_reflection_trace"),
                            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                                view: &target.trace_view,
                                depth_slice: None,
                                resolve_target: None,
                                ops: wgpu::Operations {

                                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                                    store: wgpu::StoreOp::Store,
                                },
                            })],
                            depth_stencil_attachment: Some(
                                wgpu::RenderPassDepthStencilAttachment {
                                    view: &target.depth_view,
                                    depth_ops: Some(wgpu::Operations {
                                        load: wgpu::LoadOp::Clear(1.0),

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

            if let Some(sw) = self.space_warp.as_ref() {
                if let Some((m, d)) = sw.acquired {
                    use crate::renderer::space_warp::{MotionDraw, MotionKind};
                    let base = eye as u32 * warp_per_eye;
                    let mut draws: Vec<MotionDraw> = Vec::new();
                    let reflect_group = sw.reflect_groups.get(image_index).map(|g| &g[eye]).filter(|_| warp_reflections);
                    if let Some((vb, ib, n)) = brush_buffers.as_ref() {
                        let kind = if reflect_group.is_some() { MotionKind::BrushReflect } else { MotionKind::Brush };
                        let brush = |first: u32, count: u32| MotionDraw { kind, vertices: vb, indices: ib, first, count, slot: base, joints: reflect_group };
                        match brush_motion_ranges.as_ref() {
                            Some(ranges) => draws.extend(ranges[eye].iter().map(|&(first, count)| brush(first, count))),
                            None => draws.push(brush(0, *n)),
                        }
                    }

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
                        if warp_cull
                            && inst.mesh.skin.is_none()
                            && !crate::renderer::shadow::sphere_in_frustum(
                                &warp_planes[eye],
                                crate::renderer::shadow::mesh_caster_bound(&inst.mesh),
                            )
                        {
                            continue;
                        }
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

                            for prim in inst.mesh.primitives.iter() {
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

                    use crate::renderer::space_warp::MotionParticles;
                    let mut particles: Vec<MotionParticles> = Vec::new();
                    if self.levers.space_warp_debug & 262144 == 0 {
                        if let (true, Some(gpu)) = (effects_drawn, self.effects_gpu.as_ref()) {
                            let (instances, group, over, screened) = gpu.motion_parts();
                            for (kind, range) in [
                                (MotionKind::EffectsOver, 0..over),
                                (MotionKind::EffectsScreened, over..over + screened),
                            ] {
                                particles.push(MotionParticles {
                                    kind,
                                    instances: Some(instances),
                                    vertices: crate::renderer::effects::VERTICES_PER_PARTICLE,
                                    range,
                                    slot: base,
                                    groups: [Some(probe_read_group), Some(group)],
                                });
                            }
                        }
                        if let (true, false, Some(w)) = (weather_on, stereo, self.weather.as_ref()) {
                            particles.push(MotionParticles {
                                kind: MotionKind::Weather,
                                instances: None,
                                vertices: crate::renderer::weather::VERTICES_PER_PARTICLE,
                                range: 0..w.counts.0 + w.counts.1 + w.counts.2,
                                slot: base,
                                groups: [Some(&w.maps.bind_group), None],
                            });
                        }
                    }
                    if self.levers.space_warp_debug & 1 != 0 {
                        draws.clear();
                        particles.clear();
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
                        &particles,
                        self.levers.space_warp_debug & 2048 != 0,
                        if self.levers.space_warp_debug & 4096 != 0 { 0.0 } else { 1.0 },
                    );

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

                    if self.levers.space_warp_debug & 8192 != 0 && eye == 0 {
                        if let Some(rb) = sw.readback.as_ref() {

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

                    timestamp_writes: self.pass_timers.as_ref().and_then(|t| t.writes(eye * 2 + 1)),
                    ..Default::default()
                });

                pass.set_pipeline(&self.ssr_pipelines.blit_pipeline);

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

                if reflective_brushes {
                    if let Some((vb, ib, count)) = &brush_buffers {

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

        if let Some(sw) = self.space_warp.as_mut() {
            if sw.acquired.is_some() {
                sw.motion.release_image()?;
                sw.depth.release_image()?;

                let dbg = self.levers.space_warp_debug;
                let delta = sw
                    .prev
                    .filter(|_| dbg & 8 == 0)
                    .map(|(_, prev_w2p)| crate::renderer::space_warp::app_space_delta(prev_w2p, warp_world_to_player))
                    .unwrap_or(xr::Posef::IDENTITY);

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

                        info.near_z = crate::renderer::space_warp::FAR_Z;
                        info.far_z = crate::renderer::space_warp::NEAR_Z;
                    }
                    if dbg & 1024 != 0 {

                        info.near_z = 500.0;
                        info.far_z = crate::renderer::space_warp::FAR_Z;
                    }
                }
            }
            sw.prev = Some((warp_view_proj, warp_world_to_player));
            sw.prev_clocks = Some(warp_clocks);
            sw.prev_models = meshes.iter().map(|m| (m.model.buffer.clone(), m.mesh.model_matrix())).collect();
        }

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

        let wait = if self.levers.gpu_sync {
            wgpu::PollType::Wait { submission_index: None, timeout: None }
        } else {
            wgpu::PollType::Poll
        };
        let _ = self.wgpu_device.poll(wait);
        let gpu_time = gpu_wait_start.elapsed();
        let outcome = self.frame_stats.record(cpu_time, gpu_time, std::time::Instant::now());

        if outcome.measured {
            if let Some(metrics) = &self.perf_metrics {
                self.perf_metric_window.add(&metrics.sample());
            }
        }

        if let Some(stats) = outcome.closed {
            let cycling = crate::renderer::perf_ab::ENABLED || self.levers.ab_cycle;

            let ab = if self.perf_warmup {
                "warmup"
            } else if cycling {
                ab_phase.label()
            } else {
                "-"
            };

            let levers = self.levers.summary();

            let ssr = if self.screen_space_reflections { "on" } else { "off" };
            let breakdown = self.pass_timers.as_ref().and_then(|t| t.read(&self.wgpu_device));
            match &breakdown {
                Some(breakdown) => log::info!(
                    "{} [ab={ab} ssr={ssr} levers={levers}]",
                    crate::renderer::pass_timers::format_breakdown(breakdown)
                ),
                None => log::info!("PASS: unavailable [ab={ab} ssr={ssr} levers={levers}]"),
            }

            let counters = self.perf_metric_window.take();
            if self.perf_metrics.is_some() {
                log::info!("{} [ab={ab} ssr={ssr} levers={levers}]", crate::perf_metrics_log::format_line(&counters));
            }

            if self.recommended_resolution.is_some() && self.levers.dynamic_resolution {
                let answers = self.recommendation_window.take();
                log::info!("{} [ab={ab} ssr={ssr} levers={levers}]", answers.format_line((self.width, self.height)));
            }

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

        let request = self.levers.eye_capture;
        if request != 0 && request != self.eye_capture_served {
            self.eye_capture_served = request;
            match self.capture_eyes(image_index, request) {
                Ok(path) => log::info!("EYECAPTURE {request} -> {}", path.display()),
                Err(e) => log::warn!("EYECAPTURE {request} failed: {e}"),
            }

            match self.capture_cards(request) {
                Ok(Some(path)) => log::info!("CARDCAPTURE {request} -> {}", path.display()),
                Ok(None) => {}
                Err(e) => log::warn!("CARDCAPTURE {request} failed: {e}"),
            }
        }
        self.swapchain.release_image()?;

        let submit_pose = |i: usize, ev: &xr::View| match (&self.pinned_head, &tracked_poses) {
            (Some(_), Some(tracked)) => tracked.get(i).copied().unwrap_or(ev.pose),
            _ => ev.pose,
        };

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
