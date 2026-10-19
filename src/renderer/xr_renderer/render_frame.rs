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

    bound: glam::Vec4,
}

type LayeredDraw<'a> = (
    &'a wgpu::BindGroup,
    &'a wgpu::Buffer,
    &'a wgpu::Buffer,
    u32,

    Option<[u32; 4]>,

    (glam::Vec3, glam::Vec3),
);

type SkinnedDraw<'a> = (
    &'a wgpu::BindGroup,
    &'a wgpu::BindGroup,
    &'a wgpu::BindGroup,
    &'a wgpu::Buffer,
    &'a wgpu::Buffer,
    u32,

    bool,
);

struct DrawLogFrame(bool);

impl DrawLogFrame {
    fn begin() -> Self {
        let on = crate::renderer::levers::DRAW_LOG_PENDING.swap(false, std::sync::atomic::Ordering::Relaxed);
        if on {
            log::warn!("DRAWLOG frame start");
            wgpu::hal::vulkan::DRAW_LOG.store(true, std::sync::atomic::Ordering::Relaxed);
        }
        Self(on)
    }
}

impl Drop for DrawLogFrame {
    fn drop(&mut self) {
        if self.0 {
            wgpu::hal::vulkan::DRAW_LOG.store(false, std::sync::atomic::Ordering::Relaxed);
            log::warn!("DRAWLOG frame end");
        }
    }
}

fn push_mesh_draws<'a>(
    instance: &'a MeshInstance,
    lightmap_bg: &'a wgpu::BindGroup,
    mesh_draws: &mut Vec<MeshDraw<'a>>,
    skinned_draws: &mut Vec<SkinnedDraw<'a>>,
    layered_draws: &mut Vec<LayeredDraw<'a>>,
    baked_cave_draws: &mut Vec<crate::renderer::baked_cave::BakedCaveDraw<'a>>,
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
                    prim.blended,
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
                    crate::renderer::cave_twins::keeps_vertical(instance.mesh.model_matrix())
                        .then_some(layered.twin_ends),
                    crate::renderer::cave_twins::placed_box(instance.mesh.model_matrix(), layered.bounds),
                ));
                continue;
            }

            if let Some(cave) = &prim.baked_cave {
                let placed = instance.mesh.model_matrix();
                baked_cave_draws.push(crate::renderer::baked_cave::BakedCaveDraw {
                    model: &instance.model.bind_group,
                    lightmap: lightmap_bg,
                    vertices: &cave.vertex_buffer,
                    indices: &prim.index_buffer,
                    index_count: prim.indices.len() as u32,
                    chunks: &cave.chunks,
                    emissive: cave.emissive,
                    placed,
                    bounds: crate::renderer::cave_twins::placed_box(placed, cave.bounds),
                });
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
                bound: crate::renderer::shadow::mesh_caster_bound(&instance.mesh),
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

        terrain: Option<(&[SolidVertex], &[u32])>,

        terrain_chunks: &[crate::renderer::shadow::CasterChunk],

        brushes: Option<(&[BrushVertex], &[u32])>,
        mirror: Option<MirrorSurface>,
    ) -> Result<Vec<xr::CompositionLayerProjectionView<xr::Vulkan>>, Box<dyn std::error::Error>>
    {

        let _draw_log = DrawLogFrame::begin();

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

        let geometry_clock = std::time::Instant::now();
        let (mut solid_verts, mut solid_idx, mut solid_ranges) =
            build_solid_mesh_with_ranges(cuboids);

        let mut kept_ground: Option<(&[SolidVertex], &[u32])> = None;

        let mut terrain_range: Option<(u32, u32)> = None;
        if let Some((terrain_verts, terrain_idx)) = terrain {
            if !terrain_verts.is_empty() && !terrain_idx.is_empty() {

                self.prepare_ground(terrain_verts, terrain_idx, terrain_chunks, "in a frame");
                let terrain_idx: &[u32] = match &self.slope_split {
                    Some(split) if self.terrain_gentle.is_some() => &split.indices,
                    _ => terrain_idx,
                };
                if self.levers.ground_buffers {

                    terrain_range = crate::renderer::static_solid::ranges_after_ground(&mut solid_ranges, terrain_idx.len() as u32);
                    kept_ground = Some((terrain_verts, terrain_idx));
                } else {
                    let base = solid_verts.len() as u32;
                    let index_start = solid_idx.len() as u32;
                    solid_verts.extend_from_slice(terrain_verts);
                    solid_idx.extend(terrain_idx.iter().map(|i| i + base));
                    solid_ranges.push((None, index_start, terrain_idx.len() as u32, 0.0));
                    terrain_range = Some((index_start, terrain_idx.len() as u32));
                }
            }
        }

        let mut uploaded: u64 = 0;
        let (solid_vb, solid_ib, solid_index_count, solid_any) = if self.levers.ground_buffers {
            let (gv, gi) = kept_ground.unwrap_or((&[], &[]));
            let written = crate::renderer::static_solid::StaticSolid::update(
                &mut self.static_solid,
                &self.wgpu_device,
                &self.wgpu_queue,
                kept_ground,

                if self.ground_world { (glam::Vec3::ZERO, 0.0) } else { (self.player.offset, self.player.yaw) },
                &solid_verts,
                &solid_idx,
            );
            uploaded += written.bytes();
            let kept = self.static_solid.as_ref().expect("update leaves the buffers");
            (
                kept.vertices.clone(),
                kept.indices.clone(),
                (gi.len() + solid_idx.len()) as u32,
                gv.len() + solid_verts.len() > 0,
            )
        } else {
            self.static_solid = None;
            uploaded += std::mem::size_of_val(solid_verts.as_slice()) as u64 + std::mem::size_of_val(solid_idx.as_slice()) as u64;
            (
                self.wgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("solid_vb"),
                    contents: bytemuck::cast_slice(&solid_verts),
                    usage: wgpu::BufferUsages::VERTEX,
                }),
                self.wgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("solid_ib"),
                    contents: bytemuck::cast_slice(&solid_idx),
                    usage: wgpu::BufferUsages::INDEX,
                }),
                solid_idx.len() as u32,
                !solid_verts.is_empty(),
            )
        };

        let ground_partitioned = terrain_range.is_some() && !terrain_chunks.is_empty();
        let solid_ranges = solid_ranges;

        let brush_geometry = brushes.filter(|(v, i)| !v.is_empty() && !i.is_empty());

        let partitioned = match (&self.sun_faces, brush_geometry) {
            (Some(faces), Some((v, i))) if !self.levers.brush_buffers => Some(faces.partition(v, i)),
            _ => None,
        };
        let mut brush_sun_ends = partitioned.as_ref().map_or([0, 0], |(_, ends)| *ends);
        let brush_buffers = match brush_geometry {
            Some((v, i)) if self.levers.brush_buffers => {

                let pose = if self.brush_world { (glam::Vec3::ZERO, 0.0) } else { (self.player.offset, self.player.yaw) };
                uploaded += crate::renderer::static_brush::StaticBrushes::update(
                    &mut self.static_brushes,
                    &self.wgpu_device,
                    v,
                    i,
                    pose,
                    self.sun_faces.as_ref(),
                );
                let kept = self.static_brushes.as_ref().expect("update leaves the buffers");
                brush_sun_ends = kept.sun_ends;
                Some((kept.vertices.clone(), kept.indices.clone(), kept.index_count))
            }
            Some((v, i)) => {
                self.static_brushes = None;
                let i: &[u32] = partitioned.as_ref().map_or(i, |(p, _)| p.as_slice());
                uploaded += std::mem::size_of_val(v) as u64 + std::mem::size_of_val(i) as u64;
                Some((
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
                ))
            }
            None => {
                self.static_brushes = None;
                None
            }
        };
        let brush_sun_ends = brush_sun_ends;
        let (wire_verts, wire_idx) = build_wire_mesh(cuboids);
        let (particle_verts, particle_idx) =
            particle::build_particle_mesh(particles, beams, cam_right, cam_up, view_dir);

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
        uploaded += std::mem::size_of_val(wire_verts.as_slice()) as u64
            + std::mem::size_of_val(wire_idx.as_slice()) as u64
            + std::mem::size_of_val(particle_verts.as_slice()) as u64
            + std::mem::size_of_val(particle_idx.as_slice()) as u64;
        self.note_cpu("geometry_ms", geometry_clock.elapsed().as_secs_f64() * 1e3);
        self.note_cpu("upload_mb", uploaded as f64 / 1_048_576.0);

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

            let fires: Vec<_> = fires.into_iter().chain(self.lights_uniform.pool_light()).collect();
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

        let mut effects_drawn =
            effects_on && !(self.effect_emitters.is_empty() && self.splashes.is_empty() && self.bursts.live.is_empty()) && self.effects_gpu.is_some();
        self.last_exposure = post.exposure;
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

            let relit: Vec<f32> = match self.tod.runtime {
                Some(_) => self.tod.probe_means.iter().zip(&self.probe_brightness).map(|(m, now)| if m.0 > 0.0 { now / m.0 } else { 1.0 }).collect(),
                None => Vec::new(),
            };
            let ground = self.ground_heights.as_ref();
            let ambient = |world: glam::Vec3| crate::renderer::effects::room_ambient_now(descs, world, &relit, ground);
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

            let splashes: Vec<crate::renderer::effects::Splash> =
                splashes.into_iter().filter(|s| crate::renderer::effects::splash_seen(s, now, &views)).collect();
            self.effect_seen_until.resize(self.effect_emitters.len(), f64::NEG_INFINITY);

            let mut sight: Vec<Vec<crate::renderer::effects::seen::Eye>> = Vec::with_capacity(self.effect_emitters.len());
            for (e, until) in self.effect_emitters.iter().zip(self.effect_seen_until.iter_mut()) {
                let eyes = crate::renderer::effects::emitter_sight(e, &views, head_world, rooms, &portals);
                if !eyes.is_empty() {
                    *until = now + crate::renderer::effects::EMITTER_HOLD;
                }
                sight.push(eyes);
            }
            let held = &self.effect_seen_until;
            let seen = |i: usize| held.get(i).map_or(true, |&t| now <= t);

            let wants = crate::renderer::effects::governor_wants(&self.effect_emitters, &seen, &splashes, &self.bursts.splash_measured, head_world, &sight);

            self.bursts.mist_hour = self.time_of_day_hour();
            let bursts_drawn = self.bursts.frame_with(&self.effect_library, now, &views, head_world, &wants);
            if (0..self.effect_emitters.len()).any(|i| seen(i)) || !splashes.is_empty() || bursts_drawn.contains(&true) {
                let bursts = crate::renderer::effects::stack::BurstFrame { lib: &self.effect_library, set: &self.bursts, drawn: &bursts_drawn };
                let set = &self.bursts;
                let shares = crate::renderer::effects::Shares {
                    emitter: &|e| set.share_of(crate::renderer::effects::emitter_key(e)),
                    splash: &|s| set.share_of(crate::renderer::effects::splash_key(s)),
                    sight: &sight,
                };
                let frame = crate::renderer::effects::simulate_governed(&self.effect_emitters, &seen, &splashes, Some(&bursts), Some(&shares), now, &at);
                self.burst_count = frame.bursts;
                self.bursts.measured(&frame.splash_costs);
                let (s, c) = (&self.bursts.stats, &frame.bursts);
                if self.shadow_diag_frames.get() % 180 == 0 && (s.fill_wanted > s.fill + 1e-4 || s.instances_wanted > s.instances + 0.5) {
                    log::info!(
                        "EFFECTS GOVERNOR fill {:.2} of {:.2} wanted, instances {:.0} of {:.0} | thinned bursts {} emitters {} splashes {} | shed bursts {} emitters {} splashes {} | frames governed {}",
                        s.fill, s.fill_wanted, s.instances, s.instances_wanted, s.thinned, s.emitters_thinned, s.splashes_thinned, c.shed, c.emitter_shed, c.splash_shed, s.frames_governed
                    );
                }
                if let Some(gpu) = self.effects_gpu.as_mut() {
                    (gpu.use_variants, gpu.overdraw) = (lv.effect_variants, lv.effects_overdraw);
                    gpu.upload(&self.wgpu_device, &self.wgpu_queue, &frame);
                }
            } else {
                effects_drawn = false;
            }
            self.bursts.end_frame();
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
        let mut baked_cave_draws: Vec<crate::renderer::baked_cave::BakedCaveDraw> = Vec::new();

        let mut shadow_casters: Vec<crate::renderer::shadow::ShadowMeshDraw> = Vec::new();

        let mut shadow_bounds: Vec<crate::renderer::shadow::ShadowMeshBound> = Vec::new();

        let mut level_casters: Vec<bool> = Vec::new();

        let mut caster_solid_only: Vec<Option<(&wgpu::Buffer, u32)>> = Vec::new();

        let mut tile_meshes: Vec<usize> = Vec::new();

        let mut glare_adapted = vec![false; self.glare_sources.len()];
        for instance in meshes {

            let world = glam::Quat::from_rotation_y(self.player.yaw)
                * instance.mesh.position
                + self.player.offset;
            let sky_vis = self.sky_visibility_at(world.x, world.z);

            let eye_world = glam::Quat::from_rotation_y(self.player.yaw) * mid_eye + self.player.offset;
            let light_at = if instance.tile_caster {
                crate::renderer::doors::light_point(&self.doors, world, eye_world)
            } else {

                crate::renderer::doors::doorway_light_point(&self.doors, world, eye_world)
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
                &mut baked_cave_draws,
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
                    caster_solid_only.push(prim.thin.as_ref().map(|t| (&t.solid_index_buffer, t.solid_count)));
                    shadow_bounds.push(bound);
                    level_casters.push(!instance.moves);
                }
            }
        }

        let mut mirror_only_mesh_draws: Vec<MeshDraw> = Vec::new();

        let mut skinned_casters: Vec<crate::renderer::shadow::ShadowSkinnedDraw> = skinned_draws
            .iter()
            .map(|(model_bg, _tex, joint_bg, vb, ib, count, _)| {
                (*vb, *ib, *count, *model_bg, *joint_bg)
            })
            .collect();

        let skinned_mirror_from = skinned_casters.len();
        let mut mirror_only_skinned_draws: Vec<SkinnedDraw> = Vec::new();
        let mut mirror_only_layered_draws: Vec<LayeredDraw> = Vec::new();
        let mut mirror_only_baked_caves: Vec<crate::renderer::baked_cave::BakedCaveDraw> = Vec::new();
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
                &mut mirror_only_baked_caves,
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
                    caster_solid_only.push(prim.thin.as_ref().map(|t| (&t.solid_index_buffer, t.solid_count)));
                    shadow_bounds.push(bound);
                    level_casters.push(false);
                }
            }
        }
        skinned_casters.extend(
            mirror_only_skinned_draws
                .iter()
                .map(|(model_bg, _tex, joint_bg, vb, ib, count, _)| (*vb, *ib, *count, *model_bg, *joint_bg)),
        );

        let mut posed_live: Vec<(wgpu::Buffer, wgpu::Buffer)> = Vec::new();
        let mut posed: Vec<(crate::renderer::skin_compute::Posed, &wgpu::Buffer, u32, &wgpu::BindGroup)> = Vec::new();
        let mut posed_cache = self.posed_cache.lock().unwrap_or_else(|e| e.into_inner());
        let mut posed_mirror_from = usize::MAX;
        if self.levers.skin_once {
            for (k, instance) in meshes.iter().chain(mirror_only_meshes.iter()).enumerate() {
                if k == meshes.len() {
                    posed_mirror_from = posed.len();
                }
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
        let posed_mirror_from = posed_mirror_from.min(posed.len());
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

                    brushes.map(|(v, _)| {
                        let world = self.brush_world;
                        v.iter().map(move |b| if world { warp_world_to_player.transform_point3(glam::Vec3::from(b.position)).to_array() } else { b.position })
                    }),

                    terrain.map(|(v, _)| {
                        let world = self.ground_world;
                        v.iter().map(move |t| if world { warp_world_to_player.transform_point3(glam::Vec3::from(t.position)).to_array() } else { t.position })
                    }),
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
        let character_tiles: Vec<(usize, glam::Mat4)> = if fx.shadows
            && fx.character_shadows
            && self.levers.spot_caster_drop & 64 == 0
            && (has_body || !door_casters.is_empty())
        {
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

        let ground_head = {
            let n = eye_views.len().max(1) as f32;
            let mid = eye_views.iter().fold(glam::Vec3::ZERO, |sum, v| sum + glam::Vec3::new(v.pose.position.x, v.pose.position.y, v.pose.position.z)) / n;
            glam::Quat::from_rotation_y(self.player.yaw) * mid + self.player.offset
        };
        let ground_frame: Option<crate::renderer::ground_lod::FrameGround> = match (terrain_range, self.ground_lod.borrow_mut().as_mut()) {
            (Some((index_start, _)), Some(lod)) if ground_partitioned && lod.units.len() == terrain_chunks.len() => {
                let world_planes = |m: glam::Mat4| crate::renderer::shadow::frustum_planes(m * warp_world_to_player);
                let frusta: Vec<[glam::Vec4; 6]> = eye_views
                    .iter()
                    .map(|v| world_planes(Camera::gl_to_wgpu_ndc(Camera::xr_projection(v.fov, 0.03, 1000.0)) * Camera::xr_view(v.pose)))
                    .chain(spot_matrices.iter().map(|m| world_planes(m.pass)))
                    .chain((shadow.sun_enabled && static_sun.is_none()).then(|| world_planes(shadow.sun_view_proj)))
                    .collect();
                lod.set_strict(self.levers.ground_lod_strict);
                Some(lod.frame(ground_head, &frusta, self.levers.ground_lod, self.player.offset, self.player.yaw, index_start))
            }
            _ => None,
        };

        let ground_selected = ground_frame.is_some();

        let ground_wholes: crate::renderer::ground_lod::Wholes = ground_frame.as_ref().map(|f| f.wholes.clone()).unwrap_or_default();

        let (solid_chunks, ground_units, ground_boxes, ground_error): (Vec<crate::renderer::shadow::CasterChunk>, Vec<u32>, Vec<(glam::Vec3, glam::Vec3)>, f32) =
            match (ground_frame, terrain_range) {
                (Some(f), _) => (f.chunks, f.units, f.world, f.error_rad),

                (None, Some((index_start, _))) => (
                    terrain_chunks.iter().map(|c| crate::renderer::shadow::CasterChunk { first_index: c.first_index + index_start, ..*c }).collect(),
                    Vec::new(),
                    Vec::new(),
                    0.0,
                ),
                (None, None) => (Vec::new(), Vec::new(), Vec::new(), 0.0),
            };
        let unit_of = |k: usize| ground_units.get(k).map_or(k, |u| *u as usize);

        let ground_step1 = self.ground_lod.borrow().as_ref().map(|l| l.level0_len()).filter(|n| *n > 0);

        let ground_coarse_from: u32 = match (terrain_range, ground_step1) {
            (Some((start, _)), Some(n)) => start + n,
            _ => u32::MAX,
        };

        self.spotless_frame.store(
            self.levers.spotless_shaders
                && shadow.spot_count == 0
                && !lights.iter().any(crate::renderer::lights::Light::is_surface_light),
            std::sync::atomic::Ordering::Relaxed,
        );

        let record_sun = shadow.sun_enabled && static_sun.is_none_or(|(_, stale)| stale);
        if record_sun || shadow.sun_dynamic_enabled || shadow.spot_count > 0 || !character_tiles.is_empty() {

            let solid_caster = (solid_index_count > 0 && !(ground_partitioned && solid_chunks.is_empty()))
                .then_some((&solid_vb, &solid_ib, solid_index_count));

            let world_ground_cast = self.ground_world && terrain_range.is_some();
            let world_brush_cast = self.brush_world && brush_buffers.is_some();
            if world_ground_cast || world_brush_cast {
                self.ground_model.upload(&self.wgpu_queue, warp_world_to_player);
            }
            self.shadow_map.set_solid_world(world_ground_cast.then_some(&self.ground_model.bind_group), false);
            self.shadow_map.set_brush_world(world_brush_cast.then_some(&self.ground_model.bind_group));
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

                let sun_chunks: Option<Vec<crate::renderer::shadow::CasterChunk>> = match (static_sun.is_some() && ground_selected, terrain_range, self.ground_lod.borrow_mut().as_mut()) {
                    (true, Some((index_start, _)), Some(lod)) => Some(
                        lod.frame(
                            ground_head,
                            &[crate::renderer::shadow::frustum_planes(shadow.sun_view_proj * warp_world_to_player)],
                            false,
                            self.player.offset,
                            self.player.yaw,
                            index_start,
                        )
                        .chunks,
                    ),
                    _ => None,
                };
                let sun_ground: &[crate::renderer::shadow::CasterChunk] = sun_chunks.as_deref().unwrap_or(&solid_chunks);
                let solid_caster = if sun_ground.is_empty() && (static_sun.is_some() || ground_partitioned) {
                    None
                } else {
                    (solid_index_count > 0).then_some((&solid_vb, &solid_ib, solid_index_count))
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
                    sun_ground,
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

                let moving_bounds: Vec<crate::renderer::shadow::ShadowMeshBound> = level_casters
                    .iter()
                    .enumerate()
                    .map(|(i, &level)| {
                        if level {
                            glam::Vec4::new(0.0, -1.0e9, 0.0, 0.0)
                        } else {
                            cull_bounds.get(i).copied().unwrap_or(glam::Vec4::new(0.0, 0.0, 0.0, f32::INFINITY))
                        }
                    })
                    .collect();
                drawn += self.shadow_map.record_moving(
                    &mut encoder,
                    shadow.sun_dynamic_enabled,
                    shadow.sun_dynamic_view_proj,
                    &character_tiles.iter().map(|(_, m)| *m).collect::<Vec<_>>(),
                    &shadow_casters,
                    &moving_bounds,
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

                let drop = self.levers.spot_caster_drop;
                let keep = |bit: u32| drop & bit == 0;
                self.shadow_map.set_tile_scissor(!keep(128));
                let spot_posed: Vec<crate::renderer::shadow::ShadowMeshDraw> = posed_casters
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| if *i < posed_mirror_from { keep(8) } else { keep(16) })
                    .map(|(_, d)| *d)
                    .collect();
                let spot_skinned: Vec<crate::renderer::shadow::ShadowSkinnedDraw> = frame_skinned_casters
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| if *i < skinned_mirror_from { keep(8) } else { keep(16) })
                    .map(|(_, d)| *d)
                    .collect();
                let no_meshes: [crate::renderer::shadow::ShadowMeshDraw; 0] = [];
                let spot_meshes: Vec<crate::renderer::shadow::ShadowMeshDraw> = shadow_casters
                    .iter()
                    .enumerate()
                    .map(|(i, &(vb, ib, n, model))| match caster_solid_only.get(i).copied().flatten() {
                        Some((solid, count)) if !keep(32) => (vb, solid, count, model),
                        _ => (vb, ib, n, model),
                    })
                    .collect();

                let pillarless: Option<(wgpu::Buffer, u32)> = brush_geometry.filter(|_| !keep(256)).map(|(v, i)| {
                    let to_world = if self.brush_world { glam::Mat4::IDENTITY } else { warp_world_to_player.inverse() };
                    let (lo, hi) = (glam::Vec3::new(-0.46, -0.01, -7.46), glam::Vec3::new(0.46, 3.11, -6.54));
                    let inside = |k: u32| {
                        let p = to_world.transform_point3(glam::Vec3::from(v[k as usize].position));
                        p.cmpge(lo).all() && p.cmple(hi).all()
                    };
                    let kept: Vec<u32> =
                        i.chunks_exact(3).filter(|t| !t.iter().all(|&k| inside(k))).flatten().copied().collect();
                    let ib = self.wgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("brush_ib_pillarless"),
                        contents: bytemuck::cast_slice(&kept),
                        usage: wgpu::BufferUsages::INDEX,
                    });
                    (ib, kept.len() as u32)
                });
                let spot_brushes = match (&pillarless, brush_caster) {
                    (Some((ib, n)), Some((vb, _, _))) => Some((vb, ib, *n)),
                    _ => brush_caster,
                };
                drawn += self.shadow_map.record_spots(
                    &mut encoder,
                    shadow.spot_count as usize,
                    &spot_pass,
                    solid_caster.filter(|_| keep(1)),
                    spot_brushes.filter(|_| keep(2)),
                    if keep(4) { &spot_meshes } else { &no_meshes },
                    if keep(4) { cull_bounds } else { &[] },
                    &spot_skinned,
                    &spot_posed,
                    if keep(1) { &solid_chunks } else { &[] },
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

        let warp_mesh_room = crate::renderer::space_warp::MAX_SLOTS as usize / 2 - 3;
        if meshes.len() > warp_mesh_room && self.space_warp.as_ref().is_some_and(|sw| sw.acquired.is_some()) && self.shadow_diag_frames.get() % 36 == 0 {
            log::error!(
                "SPACEWARP: {} meshes drawn and {warp_mesh_room} motion slots an eye: {} mesh(es) write no motion vectors this frame",
                meshes.len(),
                meshes.len() - warp_mesh_room,
            );
        }
        let warp_meshes: Vec<(&MeshInstance, glam::Mat4, glam::Mat4)> = match self.space_warp.as_ref() {
            Some(sw) if sw.acquired.is_some() => meshes
                .iter()

                .take(warp_mesh_room)
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

                    let planes = |e: usize| match self.brush_world {
                        true => crate::renderer::shadow::frustum_planes(warp_view_proj[e] * warp_world_to_player),
                        false => warp_planes[e],
                    };
                    match self.static_brushes.as_ref().filter(|_| self.levers.brush_buffers) {

                        Some(kept) => Some(std::array::from_fn(|e| crate::renderer::space_warp::visible_ranges(&kept.runs, &planes(e)))),
                        None => {
                            let i: &[u32] = partitioned.as_ref().map_or(i, |(p, _)| p.as_slice());
                            let runs = crate::renderer::space_warp::index_runs(i, |k| glam::Vec3::from(v[k as usize].position));
                            Some(std::array::from_fn(|e| crate::renderer::space_warp::visible_ranges(&runs, &planes(e))))
                        }
                    }
                }
                _ => None,
            };

        let warp_clocks = (self.weather.as_ref().map_or(0.0, |w| w.seconds), time.as_nanos() as f64 * 1e-9);
        let warp_steps = self.space_warp.as_ref().and_then(|sw| sw.prev_clocks).map_or((0.0, 0.0), |(w, f)| {
            (((warp_clocks.0 - w).clamp(0.0, 0.1)) as f32, ((warp_clocks.1 - f).clamp(0.0, 0.1)) as f32)
        });

        let mut horizon_runs: Vec<Vec<bool>> = Vec::new();

        let horizon_boxes: Option<Vec<(glam::Vec3, glam::Vec3)>> = match (&self.ground_occluder, terrain_range) {
            _ if !self.levers.ground_horizon => None,
            (Some(_), Some(_)) if ground_selected => Some(ground_boxes.clone()),
            (Some((_, bounds)), Some((index_start, _)))
                if bounds.len() == solid_chunks.len() && bounds.iter().zip(&solid_chunks).all(|((first, _), c)| first + index_start == c.first_index) =>
            {
                Some(bounds.iter().map(|(_, b)| *b).collect())
            }
            _ => None,
        };
        let (horizon_hidden, horizon_water): (Vec<bool>, Vec<bool>) = match (&self.ground_occluder, horizon_boxes) {
            (Some((ground, _)), Some(boxes)) => {
                let yaw = glam::Quat::from_rotation_y(self.player.yaw);
                let to_world = |p: glam::Vec3| yaw * p + self.player.offset;
                let eyes: Vec<glam::Vec3> = eye_views
                    .iter()
                    .map(|ev| to_world(glam::Vec3::new(ev.pose.position.x, ev.pose.position.y, ev.pose.position.z)))
                    .collect();
                let caves: Vec<(glam::Vec3, glam::Vec3)> = if self.levers.ground_horizon_caves && layered_draws.is_empty() {
                    baked_cave_draws
                        .iter()
                        .map(|d| {
                            let corners = (0..8).map(|i| {
                                to_world(glam::Vec3::new(
                                    if i & 1 == 0 { d.bounds.0.x } else { d.bounds.1.x },
                                    if i & 2 == 0 { d.bounds.0.y } else { d.bounds.1.y },
                                    if i & 4 == 0 { d.bounds.0.z } else { d.bounds.1.z },
                                ))
                            });
                            corners.fold(
                                (glam::Vec3::splat(f32::INFINITY), glam::Vec3::splat(f32::NEG_INFINITY)),
                                |(lo, hi), p| (lo.min(p), hi.max(p)),
                            )
                        })
                        .collect()
                } else {
                    Vec::new()
                };

                let lift_eye = eyes.first().copied().unwrap_or(glam::Vec3::ZERO);
                let lift = |lo: glam::Vec3, hi: glam::Vec3| crate::renderer::ground_lod::lifted(ground_error, lift_eye, lo, hi);

                let mut held = self.ground_horizon.borrow_mut();
                *held = crate::renderer::ground_horizon::for_eyes(held.take(), ground, &eyes, &caves);
                match held.as_mut() {
                    Some(h) => {
                        if self.levers.water_runs && fx.water {
                            const CLEAR: f32 = 2.0;
                            horizon_runs = self
                                .water_bodies
                                .iter()
                                .map(|b| {
                                    b.runs
                                        .iter()
                                        .map(|r| {
                                            let (lo, hi) = crate::renderer::water_runs::bounds_for(r, &eyes);
                                            let in_a_caves_box = caves.iter().any(|(clo, chi)| {
                                                lo.cmple(*chi + CLEAR).all() && hi.cmpge(*clo - CLEAR).all()
                                            });
                                            let hi = lift(lo, hi);
                                            if in_a_caves_box { h.hides(ground, lo, hi) } else { h.hides_ground(ground, lo, hi) }
                                        })
                                        .collect()
                                })
                                .collect();
                        }
                        (
                            boxes.iter().map(|(lo, hi)| h.hides_ground(ground, *lo, lift(*lo, *hi))).collect(),

                            self.water_bodies
                                .iter()
                                .map(|b| {
                                    let (lo, hi) = crate::renderer::water_runs::lifted(b.bounds, &eyes);
                                    h.hides(ground, lo, lift(lo, hi))
                                })
                                .collect(),
                        )
                    }
                    None => (Vec::new(), Vec::new()),
                }
            }
            _ => (Vec::new(), Vec::new()),
        };

        let mut planar_shares: Vec<Option<f32>> = Vec::new();
        let mut planar_rects_all: Vec<[Option<[f32; 4]>; 2]> = Vec::new();
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
            let world_eyes: Vec<glam::Vec3> = eyes.iter().map(|(pos, _)| yaw * *pos + self.player.offset).collect();
            for (i, body) in self.water_bodies.iter().enumerate() {
                let seen = crate::renderer::water_pipeline::water_seen(body.bounds, &views, doorways.as_deref())
                    && !horizon_water.get(i).copied().unwrap_or(false);

                let ranges = if !seen {
                    Vec::new()
                } else if self.levers.water_runs {
                    let behind = horizon_runs.get(i);
                    crate::renderer::water_runs::seen_ranges(&body.runs, |k, r| {
                        !behind.and_then(|b| b.get(k)).copied().unwrap_or(false)
                            && crate::renderer::water_pipeline::water_seen(
                                crate::renderer::water_runs::bounds_for(r, &world_eyes),
                                &views,
                                doorways.as_deref(),
                            )
                    })
                } else {
                    vec![0..body.index_count]
                };
                let seen = seen && !ranges.is_empty();
                *body.ranges.borrow_mut() = ranges;

                if seen && !body.seen.get() {
                    body.waves.reprime();
                }
                body.seen.set(seen);
            }

            if self.levers.planar_water {
                for body in &self.water_bodies {
                    let (lo, hi) = crate::renderer::planar_water::surface_box(body.bounds, body.uniform.extinction[3]);
                    let rects: [Option<[f32; 4]>; 2] =
                        std::array::from_fn(|i| eyes.get(i).and_then(|(_, clip)| crate::renderer::planar_water::screen_rect(*clip, lo, hi)));
                    let above = world_eyes.iter().all(|e| e.y > body.uniform.extinction[3]);
                    let share = (body.uniform.mirror[0] > 0.5 && body.seen.get() && above)
                        .then(|| rects.iter().flatten().map(|r| crate::renderer::planar_water::share(*r)).fold(0.0, f32::max));
                    planar_shares.push(share);
                    planar_rects_all.push(rects);
                }
            }
        } else {
            for body in &self.water_bodies {
                body.seen.set(false);
            }
        }

        let planar_allowed = self.levers.planar_water
            && fx.water
            && !self.stereo_scene()
            && self.water_cut_pipeline.is_none()
            && eyes_in_water.iter().all(Option::is_none);
        let planar_body = {
            let mut choice = self.planar_choice.get();
            let body = choice.choose(if planar_allowed { &planar_shares } else { &[] });
            self.planar_choice.set(choice);
            body
        };
        let planar_rects: [Option<[f32; 4]>; 2] =
            planar_body.and_then(|b| planar_rects_all.get(b).copied()).unwrap_or([None; 2]);

        let warp_water = fx.water
            && self.space_warp.as_ref().is_some_and(|sw| sw.acquired.is_some())
            && self.water_bodies.iter().any(|b| b.seen.get() && b.motion_groups.is_some());

        let warp_world = warp_water || (self.ground_world && self.space_warp.as_ref().is_some_and(|sw| sw.acquired.is_some()));

        let warp_brush_world = self.brush_world
            && brush_buffers.is_some()
            && self.space_warp.as_ref().is_some_and(|sw| sw.acquired.is_some());
        let warp_brush_slot = 1 + warp_meshes.len() as u32;
        let warp_per_eye = 1 + warp_meshes.len() as u32 + warp_brush_world as u32 + warp_world as u32;

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
                if warp_brush_world {
                    let (c, p, world_eye) = crate::renderer::space_warp::world_slot(curr, sw.prev.map(|(vps, w2p)| (vps[eye], w2p)), warp_world_to_player, glam::Vec3::new(e.x, e.y, e.z));
                    put(base + warp_brush_slot as usize, c, p, [world_eye.x, world_eye.y, world_eye.z, 1.0], reflect);
                }
                if warp_world {
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

                    if solid_any {
                        pass.set_pipeline(&self.mirror_solid_pipeline.pipeline);
                        pass.set_bind_group(0, &self.uniform_buf.bind_group, &[]);
                        pass.set_vertex_buffer(0, solid_vb.slice(..));
                        pass.set_index_buffer(solid_ib.slice(..), wgpu::IndexFormat::Uint32);
                        for (lightmap_key, index_start, count, _reflectivity) in &solid_ranges {

                            let world = (self.ground_world && terrain_range == Some((*index_start, *count)))
                                .then_some(self.mirror_solid_world.as_ref())
                                .flatten();
                            if self.ground_world && terrain_range == Some((*index_start, *count)) && world.is_none() {
                                continue;
                            }
                            if let Some(world) = world {
                                pass.set_pipeline(&world.pipeline);
                                pass.set_bind_group(0, &self.uniform_buf.bind_group, &[]);
                            }
                            pass.set_bind_group(1, self.cuboid_lightmap_bg(lightmap_key.as_deref()), &[]);

                            let count = match ground_step1 {
                                Some(step1) if terrain_range == Some((*index_start, *count)) => step1.min(*count),
                                _ => *count,
                            };
                            pass.draw_indexed(*index_start..*index_start + count, 0, 0..1);
                            if world.is_some() {
                                pass.set_pipeline(&self.mirror_solid_pipeline.pipeline);
                                pass.set_bind_group(0, &self.uniform_buf.bind_group, &[]);
                            }
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
                        for (model_bg, vb, ib, count, _, _) in
                            layered_draws.iter().chain(mirror_only_layered_draws.iter())
                        {
                            pass.set_bind_group(2, *model_bg, &[]);
                            pass.set_vertex_buffer(0, vb.slice(..));
                            pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                            pass.draw_indexed(0..*count, 0, 0..1);
                        }
                    }
                    self.baked_cave_pipelines.draw_mirror(
                        &mut pass,
                        &self.uniform_buf.bind_group,
                        &self.terrain_material.bind_group,
                        baked_cave_draws.iter().chain(mirror_only_baked_caves.iter()),
                    );
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
                        for (model_bg, tex_bg, joint_bg, vb, ib, count, _) in all_skinned_draws {
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

            let planar_eye = planar_body.zip(planar_rects[eye]).map(|(b, rect)| {
                let body = &self.water_bodies[b];
                let surface = body.uniform.extinction[3] - self.player.offset.y;
                let (mirror_vp, mirror_eye) = crate::renderer::planar_water::mirror_camera(view, proj, surface);
                let planar_post = crate::renderer::uniforms::PostUpload {
                    exposure: post.exposure * crate::renderer::planar_water::HEADROOM,
                    tonemap: crate::renderer::tonemap::ToneMapping::None,
                    night_vision: 0.0,
                    ..post
                };
                self.uniform_buf.write_scene_stereo_to(
                    &self.wgpu_queue,
                    &self.uniform_buf.planar_buffer,
                    [mirror_vp; 2],
                    [mirror_eye; 2],
                    &shadow,
                    &sky_upload,
                    &planar_post,
                    &frame_player,
                    None,
                );
                let lane = crate::renderer::planar_water::uniform_lane(true, true, post.exposure, false);
                self.wgpu_queue.write_buffer(
                    &body.uniform_buffer,
                    std::mem::offset_of!(crate::renderer::water_pipeline::WaterUniform, mirror) as u64,
                    bytemuck::cast_slice(&lane),
                );
                (b, mirror_vp, mirror_eye, rect)
            });

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

                upload.open_ground = self.levers.open_reflections;
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

            let model_rooms: Option<Vec<crate::renderer::portal_cull::RoomView>> =
                if self.levers.model_rooms && fx.portal_culling && !self.cull_rooms.is_empty() {
                    let world_to_player = player_to_world.inverse();
                    let eyes: Vec<(glam::Vec3, glam::Mat4)> = if stereo {
                        both_cam_pos.iter().copied().zip(both_view_proj.iter().copied()).collect()
                    } else {
                        vec![(cam_pos, eye_view_proj)]
                    };
                    eyes.into_iter()
                        .map(|(pos, vp)| {
                            crate::renderer::portal_cull::room_view(
                                player_to_world.transform_point3(pos),
                                vp * world_to_player,
                                vp,
                                &self.cull_rooms,
                                &open_portals,
                            )
                        })
                        .collect()
                } else {
                    None
                };
            let room_shows = |bound: glam::Vec4| {
                model_rooms.as_ref().map_or(true, |views| {
                    let world = player_to_world.transform_point3(bound.truncate());
                    views.iter().any(|v| v.shows(world, bound, crate::renderer::scene_cull::MARGIN))
                })
            };
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

            let chunk_order: Vec<usize> = {
                let mut order: Vec<usize> = (0..solid_chunks.len()).collect();
                if self.levers.ground_near_first || self.levers.ground_near_chunks > 0 {
                    let near = |k: usize| crate::renderer::scene_cull::distance_squared_to_box(cam_pos, solid_chunks[k].min, solid_chunks[k].max);
                    order.sort_by(|&a, &b| near(a).total_cmp(&near(b)));
                }
                order
            };
            let ordered_chunks = || chunk_order.iter().map(|&k| (k, &solid_chunks[k]));

            let chunk_visible: Vec<bool> = solid_chunks
                .iter()
                .enumerate()
                .map(|(k, c)| {

                    let through_a_doorway = match &outdoor_frusta {
                        None => true,
                        Some(f) => crate::renderer::shadow::chunk_seen(c, f) || in_closed_room(c),
                    };
                    crate::renderer::shadow::chunk_seen(c, &cull_planes) && through_a_doorway && !horizon_hidden.get(k).copied().unwrap_or(false)
                })
                .collect();
            let terrain_chunk_visible = |c: &crate::renderer::shadow::CasterChunk| {
                solid_chunks.binary_search_by_key(&c.first_index, |s| s.first_index).is_ok_and(|k| chunk_visible[k])
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

                if let Some((_, mirror_vp, _, rect)) = planar_eye {
                    use crate::renderer::planar_water;
                    let target = &self.planar_targets[eye];
                    let planes = [crate::renderer::shadow::frustum_planes(mirror_vp)];
                    let (sx, sy, sw, sh) = planar_water::scissor(rect, self.planar_size.0, self.planar_size.1);
                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("planar_water"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &target.color_view,
                            depth_slice: None,
                            resolve_target: None,
                            ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                        })],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: &target.depth_view,
                            depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Discard }),
                            stencil_ops: None,
                        }),
                        ..Default::default()
                    });
                    if sw > 0 && sh > 0 {
                        pass.set_scissor_rect(sx, sy, sw, sh);
                        let group = &self.uniform_buf.planar_bind_group;

                        if let Some((index_start, count)) = terrain_range {
                            pass.set_pipeline(&self.terrain_mirror.pipeline);
                            pass.set_bind_group(0, group, &[]);
                            pass.set_bind_group(1, &self.terrain_material.bind_group, &[]);
                            pass.set_vertex_buffer(0, solid_vb.slice(..));
                            pass.set_index_buffer(solid_ib.slice(..), wgpu::IndexFormat::Uint32);
                            if ground_partitioned {

                                let mut shaped_bound = false;
                                for c in solid_chunks.iter().filter(|c| crate::renderer::shadow::chunk_seen(c, &planes)) {
                                    let coarse = c.first_index >= ground_coarse_from;
                                    if coarse != shaped_bound {
                                        pass.set_pipeline(self.terrain_mirror.at(coarse));
                                        shaped_bound = coarse;
                                    }
                                    pass.draw_indexed(c.first_index..c.first_index + c.index_count, 0, 0..1);
                                }
                            } else {
                                let count = ground_step1.map_or(count, |step1| step1.min(count));
                                pass.draw_indexed(index_start..index_start + count, 0, 0..1);
                            }
                        }

                        if solid_any {
                            pass.set_pipeline(&self.mirror_solid_pipeline.pipeline);
                            pass.set_bind_group(0, group, &[]);
                            pass.set_vertex_buffer(0, solid_vb.slice(..));
                            pass.set_index_buffer(solid_ib.slice(..), wgpu::IndexFormat::Uint32);
                            for (lightmap_key, index_start, count, _) in &solid_ranges {
                                if terrain_range == Some((*index_start, *count)) {
                                    continue;
                                }
                                pass.set_bind_group(1, self.cuboid_lightmap_bg(lightmap_key.as_deref()), &[]);
                                pass.draw_indexed(*index_start..*index_start + *count, 0, 0..1);
                            }
                        }
                        if let Some((vb, ib, count)) = &brush_buffers {
                            pass.set_pipeline(&self.brush_mirror_pipeline.pipeline);
                            pass.set_bind_group(0, group, &[]);
                            pass.set_bind_group(1, &self.brush_materials.bind_group, &[]);
                            pass.set_bind_group(2, self.brush_lightmap_bg(), &[]);
                            pass.set_vertex_buffer(0, vb.slice(..));
                            pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                            pass.draw_indexed(0..*count, 0, 0..1);
                        }

                        let layered: Vec<&LayeredDraw> = layered_draws
                            .iter()
                            .chain(mirror_only_layered_draws.iter())
                            .filter(|d| crate::renderer::scene_cull::sees_box(&planes, d.5))
                            .collect();
                        if !layered.is_empty() {
                            pass.set_pipeline(&self.layered_mesh_mirror_pipeline.pipeline);
                            pass.set_bind_group(0, group, &[]);
                            pass.set_bind_group(1, &self.terrain_material.bind_group, &[]);
                            for (model_bg, vb, ib, count, _, _) in layered {
                                pass.set_bind_group(2, *model_bg, &[]);
                                pass.set_vertex_buffer(0, vb.slice(..));
                                pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                                pass.draw_indexed(0..*count, 0, 0..1);
                            }
                        }
                        self.baked_cave_pipelines.draw_mirror(
                            &mut pass,
                            group,
                            &self.terrain_material.bind_group,
                            baked_cave_draws
                                .iter()
                                .chain(mirror_only_baked_caves.iter())
                                .filter(|d| crate::renderer::scene_cull::sees_box(&planes, d.bounds)),
                        );

                        let near_model = |d: &&MeshDraw| {
                            crate::renderer::scene_cull::sees(&planes, d.bound)
                                && (d.bound.truncate() - cam_pos).length() - d.bound.w < planar_water::MODEL_REACH
                        };
                        let models: Vec<&MeshDraw> = mesh_draws.iter().chain(mirror_only_mesh_draws.iter()).filter(near_model).collect();
                        if !models.is_empty() {
                            pass.set_pipeline(&self.mirror_mesh_pipeline.pipeline);
                            pass.set_bind_group(0, group, &[]);
                            for d in models {
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
                            pass.set_bind_group(0, group, &[]);
                            for (model_bg, tex_bg, joint_bg, vb, ib, count, _) in skinned_draws.iter().chain(mirror_only_skinned_draws.iter()) {
                                pass.set_bind_group(1, *model_bg, &[]);
                                pass.set_bind_group(2, *tex_bg, &[]);
                                pass.set_bind_group(3, *joint_bg, &[]);
                                pass.set_vertex_buffer(0, vb.slice(..));
                                pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                                pass.draw_indexed(0..*count, 0, 0..1);
                            }
                        }

                        pass.set_pipeline(&self.sky_mirror_pipeline.pipeline);
                        pass.set_bind_group(0, group, &[]);
                        pass.set_bind_group(1, &self.sky.bind_group, &[]);
                        pass.draw(0..3, 0..1);
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
                        for (model_bg, tex_bg, joint_bg, vb, ib, count, _) in skinned_draws.iter().chain(mirror_only_skinned_draws.iter()) {
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

                        if let (Some(t), false, true) = (&self.cave_twins, stereo, self.levers.cave_occluder) {
                            let inside = |(lo, hi): &(glam::Vec3, glam::Vec3)| cam_pos.cmpge(*lo).all() && cam_pos.cmple(*hi).all();
                            if layered_draws.iter().any(|d| inside(&d.5)) {
                                pass.set_pipeline(&t.occluder);
                                pass.set_bind_group(0, &self.uniform_buf.bind_group, &[]);
                                pass.set_bind_group(1, &self.terrain_material.bind_group, &[]);
                                for (model_bg, vb, ib, count, _, _) in layered_draws.iter().filter(|d| inside(&d.5)) {
                                    pass.set_bind_group(2, *model_bg, &[]);
                                    pass.set_vertex_buffer(0, vb.slice(..));
                                    pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                                    pass.draw_indexed(0..*count, 0, 0..1);
                                }
                            }
                        }

                        if !stereo && self.levers.cave_occluder {
                            self.baked_cave_pipelines.draw_occluders(
                                &mut pass, &self.uniform_buf.bind_group, &baked_cave_draws, cam_pos,

                                (self.levers.cave_chunk_cull && self.levers.scene_cull).then_some(&cull_planes[..]),
                            );
                        }

                        if !stereo && self.levers.probe_prepass {
                            pass.set_pipeline(&self.brush_probe_depth_prepass.pipeline);
                            pass.set_bind_group(0, &self.uniform_buf.bind_group, &[]);
                            pass.set_bind_group(1, &self.brush_materials.bind_group, &[]);
                            pass.set_vertex_buffer(0, vb.slice(..));
                            pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                            pass.draw_indexed(0..*count, 0, 0..1);
                        }
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

                            let outdoor_twins = self
                                .terrain_outdoor_passes
                                .as_ref()
                                .filter(|_| self.levers.outdoor_pass && self.terrain_cut_pipeline.is_none() && self.terrain_inlined.is_none());
                            let outdoor_frame = outdoor_twins.and_then(|_| {
                                crate::renderer::outdoor_pass::OutdoorFrame::new(self.uniform_buf.probes_or_level(probes_arg))
                            });

                            let nowhere_twins = self
                                .terrain_nowhere_passes
                                .as_ref()
                                .filter(|_| self.levers.nowhere_pass && self.terrain_cut_pipeline.is_none() && self.terrain_inlined.is_none());

                            let open_ground = {
                                let table = self.uniform_buf.probes_or_level(probes_arg);
                                table.open_ground && !table.no_trace
                            };
                            let nowhere_twins = match self.terrain_open_passes.as_ref() {
                                Some(open) if open_ground && self.terrain_cut_pipeline.is_none() && self.terrain_inlined.is_none() => Some(open),

                                None if open_ground => None,
                                _ => nowhere_twins,
                            };
                            let nowhere_frame = nowhere_twins
                                .map(|_| crate::renderer::outdoor_pass::NowhereFrame::new(self.uniform_buf.probes_or_level(probes_arg)));
                            let nowhere = |c: &crate::renderer::shadow::CasterChunk| {
                                nowhere_frame.as_ref().is_some_and(|f| {
                                    let (lo, hi) =
                                        crate::renderer::outdoor_pass::world_box(self.player.offset, self.player.yaw, c.min, c.max);
                                    f.holds(lo, hi)
                                })
                            };

                            let outdoors = |c: &crate::renderer::shadow::CasterChunk| {
                                outdoor_frame.as_ref().is_some_and(|f| {
                                    let (lo, hi) =
                                        crate::renderer::outdoor_pass::world_box(self.player.offset, self.player.yaw, c.min, c.max);
                                    f.holds(lo, hi)
                                })
                            };

                            let class_split = self.slope_split.as_ref().filter(|s| {
                                self.levers.ground_classes
                                    && self.terrain_gentle.is_some()
                                    && s.has_classes()
                                    && self.terrain_cut_pipeline.is_none()
                                    && self.terrain_inlined.is_none()
                            });
                            let outdoor_twins = outdoor_twins.filter(|_| outdoor_frame.is_some());
                            let (outdoors_drawn, nowhere_drawn) = (outdoor_twins.is_some(), nowhere_twins.is_some());
                            use crate::renderer::outdoor_pass::{GROUND_FULL, GROUND_NOWHERE, GROUND_OUTDOORS};
                            let runs = |k: usize, c: &crate::renderer::shadow::CasterChunk, class: usize| -> ([(u32, u32); 4], usize) {
                                let whole = ([(c.first_index, c.index_count), (0, 0), (0, 0), (0, 0)], 1);
                                let none = ([(0, 0); 4], 0);
                                let chunks_class = if nowhere(c) {
                                    GROUND_NOWHERE
                                } else if outdoors(c) {
                                    GROUND_OUTDOORS
                                } else {
                                    GROUND_FULL
                                };
                                match (chunks_class, class_split) {
                                    (GROUND_FULL, Some(split)) => {
                                        let pick = [
                                            class == GROUND_FULL,
                                            if outdoors_drawn { class == GROUND_OUTDOORS } else { class == GROUND_FULL },
                                            if nowhere_drawn { class == GROUND_NOWHERE } else { class == GROUND_FULL },
                                        ];
                                        let (mut r, n) = split.class_ranges(unit_of(k), pick);
                                        r[..n].iter_mut().for_each(|run| run.0 += c.first_index);
                                        (r, n)
                                    }
                                    (found, _) if found == class => whole,
                                    _ => none,
                                }
                            };
                            if !ground_partitioned {
                                pass.draw_indexed(index_start..index_start + count, 0, 0..1);
                            } else {

                                for wet in [false, true] {
                                    let weather = self.weather.as_ref().filter(|_| wet && solid_chunks.iter().any(|c| weathered(c)));
                                    if wet {
                                        let Some(w) = weather else { continue };
                                        let dedup = w.dedup_passes.as_ref().filter(|_| self.levers.pass_dedup);
                                        pass.set_pipeline(match (dedup, poolless) {
                                            (Some([_, p]), true) => &p.pipeline,
                                            (Some([p, _]), false) => &p.pipeline,
                                            (None, true) => &w.twins.pass_poolless.pipeline,
                                            (None, false) => &w.twins.pass.pipeline,
                                        });
                                        pass.set_bind_group(2, &w.maps.bind_group, &[]);
                                    }
                                    let which = wet as usize;

                                    let wet_open = self.terrain_open_passes.as_ref().filter(|_| wet && self.levers.wet_reflections && nowhere_twins.is_some());

                                    let wet_sky = self.terrain_puddle_pass.as_ref().filter(|_| wet && self.levers.wet_sky && nowhere_twins.is_some());

                                    let mirror = wet_sky.or(wet_open.map(|t| &t[1]));
                                    let pick: Option<u8> = match (mirror.is_some(), self.levers.puddle_reflections, wet_sky.is_some()) {
                                        (true, true, _) => Some(0),
                                        (true, false, true) => Some(1),
                                        _ => None,
                                    };
                                    let mirrors = |c: &crate::renderer::shadow::CasterChunk| {
                                        use crate::renderer::weather::WeatherKinds::{Both, Snowless};
                                        let first = c.first_index - index_start;
                                        self.weather.as_ref().is_some_and(|w| match pick {
                                            Some(0) => matches!(w.kinds_of(first), Both | Snowless),
                                            _ => w.films_of(first),
                                        })
                                    };
                                    let nowhere = nowhere_twins.map(|t| &t[which]);
                                    for (class, twin, only) in [
                                        (GROUND_FULL, None, None),
                                        (GROUND_NOWHERE, mirror.or(nowhere), pick.map(|_| true)),
                                        (GROUND_NOWHERE, nowhere.filter(|_| pick.is_some()), Some(false)),
                                        (GROUND_OUTDOORS, outdoor_twins.map(|t| &t[which]), None),
                                    ] {
                                        if class != GROUND_FULL && twin.is_none() {
                                            continue;
                                        }

                                        let drawn_by = twin.or(if wet { None } else { Some(terrain) });
                                        let mut bound: Option<bool> = if twin.is_none() { Some(false) } else { None };

                                        let class_runs = crate::renderer::ground_lod::merged(
                                            ordered_chunks()
                                                .filter(|(_, c)| terrain_chunk_visible(c) && weathered(c) == wet && !only.is_some_and(|want| mirrors(c) != want))
                                                .flat_map(|(k, c)| {
                                                    let (r, n) = runs(k, c, class);
                                                    r.into_iter().take(n)
                                                }),
                                        );
                                        for (first, len) in class_runs {
                                            for (first, len, coarse) in crate::renderer::terrain_pipeline::by_step(first, len, ground_coarse_from) {
                                                if let (true, Some(p)) = (bound != Some(coarse), drawn_by) {
                                                    pass.set_pipeline(p.at(coarse));
                                                    bound = Some(coarse);
                                                }
                                                pass.draw_indexed(first..first + len, 0, 0..1);
                                            }
                                        }
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

                    if let (Some([plain, texels]), false) = (self.sp_body_prepass(stereo), skinned_draws.is_empty()) {
                        pass.set_bind_group(0, &self.uniform_buf.bind_group, &[]);
                        for see_through in [false, true] {
                            pass.set_pipeline(if see_through { texels } else { plain });
                            for (model_bg, tex_bg, joint_bg, vb, ib, count, _) in skinned_draws.iter().filter(|d| d.6 == see_through) {
                                pass.set_bind_group(1, *model_bg, &[]);
                                pass.set_bind_group(2, *tex_bg, &[]);
                                pass.set_bind_group(3, *joint_bg, &[]);
                                pass.set_vertex_buffer(0, vb.slice(..));
                                pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                                pass.draw_indexed(0..*count, 0, 0..1);
                            }
                        }
                    }

                    if self.levers.cave_first && baked_cave_draws.iter().any(|d| d.holds(cam_pos)) {
                        self.baked_cave_pipelines.draw_scene(
                            &mut pass,
                            &self.uniform_buf.bind_group,
                            &self.terrain_material.bind_group,
                            &baked_cave_draws,
                            self.levers.scene_cull.then_some(&cull_planes[..]),
                            self.spotless_frame.load(std::sync::atomic::Ordering::Relaxed),
                            stereo,
                            Some((cam_pos, true)),
                        );
                    }
                    if solid_any {

                        let cuboids_opaque = cuboids
                            .iter()
                            .all(|c| matches!(c.style, crate::renderer::cuboid::CuboidStyle::Wireframe) || c.color.3 == 255);
                        let solid_opaque = self.sp_solid_opaque(stereo).filter(|_| cuboids_opaque);
                        pass.set_pipeline(solid_opaque.unwrap_or(self.sp_solid(stereo)));
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
                        let gentle_of = |k: usize| gentle_split.and_then(|g| g.get(unit_of(k)).copied()).unwrap_or(0);

                        let steep_split: Option<&[u32]> = match (&self.slope_split, &self.terrain_steep) {
                            (Some(split), Some(_)) if gentle_split.is_some() => Some(&split.steep),
                            _ => None,
                        };
                        let steep_of = |k: usize| steep_split.and_then(|g| g.get(unit_of(k)).copied()).unwrap_or(0);
                        let twin = self.terrain_twin();
                        if !ground_partitioned {
                            pass.draw_indexed(index_start..index_start + count, 0, 0..1);
                        } else {

                            let near_walk = if self.levers.ground_near_first { usize::MAX } else { self.levers.ground_near_chunks as usize };
                            if near_walk > 0 || self.levers.ground_steep_first {

                                let mixed = if terrain_in_probe_pass { self.terrain_reader() } else { self.sp_terrain_twins(stereo) };
                                let gentle = self.terrain_gentle.as_ref().filter(|_| gentle_split.is_some());
                                let steep = self.terrain_steep.as_ref().filter(|_| steep_split.is_some());

                                let mut bound = (0usize, false);
                                let reader = |which: usize, coarse: bool| match (which, gentle, steep) {
                                    (1, Some(gentle), _) => gentle[twin].at(coarse),
                                    (2, _, Some(steep)) => steep[twin].at(coarse),
                                    _ => mixed.at(coarse),
                                };

                                let ranges = |k: usize, c: &crate::renderer::shadow::CasterChunk| {
                                    let g = if gentle.is_some() { gentle_of(k).min(c.index_count) } else { 0 };
                                    let st = if steep.is_some() { steep_of(k).min(c.index_count - g) } else { 0 };
                                    let end = c.index_count - st;
                                    [(1usize, 0..g), (0, g..end), (2, end..c.index_count)]
                                };
                                let mut far: Vec<(usize, &crate::renderer::shadow::CasterChunk)> = Vec::new();
                                let mut walked = 0usize;
                                for (k, c) in ordered_chunks() {
                                    if !terrain_chunk_visible(c) {
                                        terrain_culled += c.index_count;
                                        continue;
                                    }
                                    if weathered(c) {
                                        continue;
                                    }
                                    terrain_drawn += c.index_count;
                                    if walked >= near_walk {
                                        far.push((k, c));
                                        continue;
                                    }
                                    walked += 1;
                                    let coarse = c.first_index >= ground_coarse_from;
                                    for (which, range) in ranges(k, c) {
                                        if range.is_empty() {
                                            continue;
                                        }
                                        if bound != (which, coarse) {
                                            pass.set_pipeline(reader(which, coarse));
                                            bound = (which, coarse);
                                        }
                                        pass.draw_indexed(c.first_index + range.start..c.first_index + range.end, 0, 0..1);
                                    }
                                }

                                for which in crate::renderer::ground_twins::reader_order(self.levers.ground_steep_first) {
                                    let reader_runs = crate::renderer::ground_lod::merged(far.iter().map(|(k, c)| {
                                        let range = ranges(*k, c)[match which { 1 => 0, 0 => 1, _ => 2 }].1.clone();
                                        (c.first_index + range.start, range.end - range.start)
                                    }));
                                    for (first, len) in reader_runs {
                                        for (first, len, coarse) in crate::renderer::terrain_pipeline::by_step(first, len, ground_coarse_from) {
                                            if bound != (which, coarse) {
                                                pass.set_pipeline(reader(which, coarse));
                                                bound = (which, coarse);
                                            }
                                            pass.draw_indexed(first..first + len, 0, 0..1);
                                        }
                                    }
                                }
                            } else {

                                let mixed = if terrain_in_probe_pass { self.terrain_reader() } else { self.sp_terrain_twins(stereo) };
                                let mut shaped_bound = false;
                                for (k, c) in solid_chunks.iter().enumerate() {
                                    if terrain_chunk_visible(c) && !weathered(c) {
                                        let g = gentle_of(k).min(c.index_count);
                                        let end = c.index_count - steep_of(k).min(c.index_count - g);
                                        if g < end {
                                            let coarse = c.first_index >= ground_coarse_from;
                                            if coarse != shaped_bound {
                                                pass.set_pipeline(mixed.at(coarse));
                                                shaped_bound = coarse;
                                            }
                                            pass.draw_indexed(c.first_index + g..c.first_index + end, 0, 0..1);
                                        }
                                        terrain_drawn += c.index_count;
                                    } else if !terrain_chunk_visible(c) {
                                        terrain_culled += c.index_count;
                                    }
                                }
                                if let Some(gentle) = self.terrain_gentle.as_ref().filter(|_| gentle_split.is_some()) {
                                    let mut bound: Option<bool> = None;
                                    for (k, c) in solid_chunks.iter().enumerate() {
                                        let g = gentle_of(k).min(c.index_count);
                                        if g > 0 && terrain_chunk_visible(c) && !weathered(c) {
                                            let coarse = c.first_index >= ground_coarse_from;
                                            if bound != Some(coarse) {
                                                pass.set_pipeline(gentle[twin].at(coarse));
                                                bound = Some(coarse);
                                            }
                                            pass.draw_indexed(c.first_index..c.first_index + g, 0, 0..1);
                                        }
                                    }
                                }
                                if let Some(steep) = self.terrain_steep.as_ref().filter(|_| steep_split.is_some()) {
                                    let mut bound: Option<bool> = None;
                                    for (k, c) in solid_chunks.iter().enumerate() {
                                        let g = gentle_of(k).min(c.index_count);
                                        let st = steep_of(k).min(c.index_count - g);
                                        if st > 0 && terrain_chunk_visible(c) && !weathered(c) {
                                            let coarse = c.first_index >= ground_coarse_from;
                                            if bound != Some(coarse) {
                                                pass.set_pipeline(steep[twin].at(coarse));
                                                bound = Some(coarse);
                                            }
                                            pass.draw_indexed(c.first_index + c.index_count - st..c.first_index + c.index_count, 0, 0..1);
                                        }
                                    }
                                }
                            }
                            if let Some(w) = self.weather.as_ref().filter(|_| solid_chunks.iter().any(|c| weathered(c))) {

                                let gentle_w = |k: usize| if w.gentle.is_empty() { 0 } else { gentle_of(k) };

                                let wet = w.wet.as_ref().filter(|_| self.levers.wet_steep_twins && !w.gentle.is_empty());
                                let wet_only = |c: &crate::renderer::shadow::CasterChunk| {
                                    wet.is_some() && w.kinds_of(c.first_index - index_start) == crate::renderer::weather::WeatherKinds::WetOnly
                                };
                                pass.set_pipeline(&self.terrain_weather_reader(w).pipeline);
                                pass.set_bind_group(2, &w.maps.bind_group, &[]);
                                for (k, c) in ordered_chunks().filter(|(_, c)| terrain_chunk_visible(c) && weathered(c)) {
                                    let g = gentle_w(k).min(c.index_count);
                                    if g < c.index_count && !wet_only(c) {
                                        pass.draw_indexed(c.first_index + g..c.first_index + c.index_count, 0, 0..1);
                                    }
                                    terrain_drawn += c.index_count;
                                }
                                if let Some(wet) = wet {

                                    for which in 0..2 {
                                        let mut bound = false;
                                        for (k, c) in ordered_chunks().filter(|(_, c)| terrain_chunk_visible(c) && weathered(c) && wet_only(c)) {
                                            let g = gentle_w(k).min(c.index_count);
                                            let end = c.index_count - steep_of(k).min(c.index_count - g);
                                            let range = if which == 0 { g..end } else { end..c.index_count };
                                            if range.is_empty() {
                                                continue;
                                            }
                                            if !bound {
                                                pass.set_pipeline(&wet[which][twin].pipeline);
                                                bound = true;
                                            }
                                            pass.draw_indexed(c.first_index + range.start..c.first_index + range.end, 0, 0..1);
                                        }
                                    }
                                }
                                for kinds in crate::renderer::weather::WeatherKinds::ALL {
                                    let mut bound = false;
                                    for (k, c) in ordered_chunks().filter(|(_, c)| terrain_chunk_visible(c) && weathered(c)) {
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
                                 ({pct:.1}% culled) over {} chunks, {} frustum(s); {} chunk(s) and {} water behind the ground",
                                solid_chunks.len(),
                                cull_planes.len(),
                                horizon_hidden.iter().filter(|h| **h).count(),
                                horizon_water.iter().filter(|h| **h).count(),
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

                    let scene_cull = self.levers.scene_cull;
                    let mesh_seen =
                        |d: &MeshDraw| !scene_cull || (crate::renderer::scene_cull::sees(&cull_planes, d.bound) && room_shows(d.bound));
                    if !layered_draws.is_empty() {

                        pass.set_pipeline(self.sp_layered(stereo));
                        pass.set_bind_group(0, &self.uniform_buf.bind_group, &[]);
                        pass.set_bind_group(1, &self.terrain_material.bind_group, &[]);

                        let twins = self.cave_twins.as_ref().filter(|_| !stereo && self.levers.cave_twins);
                        for (model_bg, vb, ib, count, ends, bounds) in &layered_draws {
                            if self.levers.scene_cull && !crate::renderer::scene_cull::sees_box(&cull_planes, *bounds) {
                                continue;
                            }
                            pass.set_bind_group(2, *model_bg, &[]);
                            pass.set_vertex_buffer(0, vb.slice(..));
                            pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                            match (twins, ends) {
                                (Some(t), Some(ends)) if ends[3] == *count => {
                                    t.draw(&mut pass, self.sp_layered(stereo), *ends)
                                }
                                _ => {
                                    pass.set_pipeline(self.sp_layered(stereo));
                                    pass.draw_indexed(0..*count, 0, 0..1);
                                }
                            }
                        }
                    }

                    if !baked_cave_draws.is_empty() {
                        self.baked_cave_pipelines.draw_scene(
                            &mut pass,
                            &self.uniform_buf.bind_group,
                            &self.terrain_material.bind_group,
                            &baked_cave_draws,
                            self.levers.scene_cull.then_some(&cull_planes[..]),
                            self.spotless_frame.load(std::sync::atomic::Ordering::Relaxed),
                            stereo,
                            self.levers.cave_first.then_some((cam_pos, false)),
                        );
                    }
                    if !wire_verts.is_empty() {
                        pass.set_pipeline(self.sp_wire(stereo));
                        pass.set_bind_group(0, &self.uniform_buf.bind_group, &[]);
                        pass.set_vertex_buffer(0, wire_vb.slice(..));
                        pass.set_index_buffer(wire_ib.slice(..), wgpu::IndexFormat::Uint32);
                        pass.draw_indexed(0..wire_idx.len() as u32, 0, 0..1);
                    }
                    if !mesh_draws.is_empty() {

                        pass.set_pipeline(self.sp_mesh_opaque(stereo).unwrap_or(self.sp_mesh(stereo)));
                        pass.set_bind_group(0, &self.uniform_buf.bind_group, &[]);

                        for d in mesh_draws.iter().filter(|d| !d.blended && mesh_seen(d)) {
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
                        pass.set_bind_group(0, &self.uniform_buf.bind_group, &[]);

                        let blended_body = self.sp_skinned(stereo);
                        let runs: [Option<(Option<bool>, &wgpu::RenderPipeline)>; 2] = match self.sp_skinned_opaque(stereo) {
                            Some(opaque) => [Some((Some(false), opaque)), Some((Some(true), blended_body))],
                            None => [Some((None, blended_body)), None],
                        };
                        for (which, pipeline) in runs.into_iter().flatten() {
                            let mut set = false;
                            for (model_bg, tex_bg, joint_bg, vb, ib, count, blended) in &skinned_draws {
                                if which.is_some_and(|w| w != *blended) {
                                    continue;
                                }
                                if !set {
                                    pass.set_pipeline(pipeline);
                                    set = true;
                                }
                                pass.set_bind_group(1, *model_bg, &[]);
                                pass.set_bind_group(2, *tex_bg, &[]);
                                pass.set_bind_group(3, *joint_bg, &[]);
                                pass.set_vertex_buffer(0, vb.slice(..));
                                pass.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
                                pass.draw_indexed(0..*count, 0, 0..1);
                            }
                        }
                    }

                    let sky_first = self.levers.sky_before_water && eye_in_water.is_none();
                    if sky_first {
                        pass.set_pipeline(self.sp_sky(stereo));
                        pass.set_bind_group(0, &self.uniform_buf.bind_group, &[]);
                        pass.set_bind_group(1, &self.sky.bind_group, &[]);
                        pass.draw(0..3, 0..1);
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

                            if planar_eye.is_some_and(|p| p.0 == i) {
                                pass.set_pipeline(self.water_pipeline.mirrored_for_rings(&body.uniform.rings[0], &body.uniform.ring_drops));
                                pass.set_bind_group(2, &self.planar_groups[eye], &[]);
                            } else {
                                pass.set_pipeline(if eyes_body { &self.water_pipeline.waterline } else { self.sp_water(stereo) });
                            }
                            pass.set_bind_group(1, &body.bind_groups[body.waves.current()], &[]);
                            pass.set_vertex_buffer(0, body.vertex_buffer.slice(..));
                            pass.set_index_buffer(body.index_buffer.slice(..), wgpu::IndexFormat::Uint32);

                            if eyes_body {
                                pass.draw_indexed(0..body.index_count, 0, 0..1);
                            } else {
                                for range in body.ranges.borrow().iter() {
                                    pass.draw_indexed(range.clone(), 0, 0..1);
                                }
                            }
                        }
                    }

                    if !wholly_under && !sky_first {
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
                        for d in mesh_draws.iter().filter(|d| !d.blended && mesh_seen(d)) {
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
                        for d in mesh_draws.iter().filter(|d| d.blended && mesh_seen(d)) {
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

                        let slot = if warp_brush_world { base + warp_brush_slot } else { base };
                        let brush = |first: u32, count: u32| MotionDraw { kind, vertices: vb, indices: ib, first, count, slot, joints: reflect_group };
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
                        Some((start, count)) if ground_partitioned => {

                            for (first, n) in [(0, start), (start + count, solid_index_count.saturating_sub(start + count))] {
                                if n > 0 {
                                    draws.push(solid(first, n));
                                }
                            }

                            let slot = if self.ground_world { base + warp_per_eye - 1 } else { base };

                            let seen = ground_wholes.runs(&solid_chunks, |k| terrain_chunk_visible(&solid_chunks[k]));
                            for (first, count) in seen {
                                draws.push(MotionDraw { slot, ..solid(first, count) });
                            }
                        }
                        _ => draws.push(solid(0, solid_index_count)),
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

                        if inst.mesh.skin.is_none() && !room_shows(crate::renderer::shadow::mesh_caster_bound(&inst.mesh)) {
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

                                let runs = match &prim.baked_cave {
                                    Some(cave) if warp_cull && self.levers.cave_chunk_cull => {
                                        crate::renderer::baked_cave::visible_ranges(
                                            &cave.chunks,
                                            inst.mesh.model_matrix(),
                                            Some(std::slice::from_ref(&warp_planes[eye])),
                                            None,
                                        )
                                    }
                                    _ => vec![0..prim.indices.len() as u32],
                                };
                                for run in runs {
                                    draws.push(MotionDraw {
                                        kind: MotionKind::Mesh,
                                        vertices: &prim.vertex_buffer,
                                        indices: &prim.index_buffer,
                                        first: run.start,
                                        count: run.end - run.start,
                                        slot,
                                        joints: None,
                                    });
                                }
                            }
                        }
                    }
                    if warp_water {
                        for body in self.water_bodies.iter().filter(|b| b.seen.get()) {
                            let Some(groups) = &body.motion_groups else { continue };

                            for range in body.ranges.borrow().iter() {
                                draws.push(MotionDraw {
                                    kind: MotionKind::Water,
                                    vertices: &body.vertex_buffer,
                                    indices: &body.index_buffer,
                                    first: range.start,
                                    count: range.end - range.start,
                                    slot: base + warp_per_eye - 1,
                                    joints: Some(&groups[body.waves.current()]),
                                });
                            }
                        }
                    }

                    use crate::renderer::space_warp::MotionParticles;
                    let mut particles: Vec<MotionParticles> = Vec::new();
                    if self.levers.space_warp_debug & 262144 == 0 {
                        if let (true, Some(gpu)) = (effects_drawn, self.effects_gpu.as_ref()) {
                            let (instances, group, runs) = gpu.motion_parts();
                            for run in runs {
                                particles.push(MotionParticles {
                                    kind: MotionKind::Effects(run.variant),
                                    instances: Some(instances),
                                    vertices: crate::renderer::effects::VERTICES_PER_PARTICLE,
                                    range: run.first..run.first + run.count,
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
            for (part, value) in &self.cpu_frame {
                *self.cpu_window.entry(part).or_default() += value;
            }
        }
        self.cpu_frame.clear();

        if outcome.measured {
            if let Some(metrics) = &self.perf_metrics {
                self.perf_metric_window.add(&metrics.sample());
            }
        }

        if let Some(stats) = outcome.closed {
            let cpu_parts: std::collections::BTreeMap<String, f64> = std::mem::take(&mut self.cpu_window)
                .into_iter()
                .map(|(part, sum)| (part.to_string(), sum / stats.frames.max(1) as f64))
                .collect();
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
                    cpu: cpu_parts,
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
