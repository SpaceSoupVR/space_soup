use std::sync::{mpsc, Arc};

use super::XrRenderer;
use crate::renderer::probe_stream::{FixedFaces, ProbeDesc, ProbeSource};
use crate::renderer::time_of_day::{
    building_daylight_faces, relit_probe_source, ClockOverride, Prepared, ProbeDaylight, TimeOfDayRuntime, NIGHT_MAX_EXPOSURE,
    PROBE_RELIGHT_ANGLE_COS, PROBE_RELIGHT_RATIO,
};
use space_soup_sky::time_of_day::{AtlasLayers, TimeOfDayParams};
use space_soup_sky::{Panorama, SkyIrradiance, SkySun};

#[derive(Default)]
pub(super) struct TodState {
    pub runtime: Option<TimeOfDayRuntime>,

    pub scene_params: Option<TimeOfDayParams>,

    pub photographed: Option<(Panorama, f32, f32)>,

    pub baked_sky: Option<SkyIrradiance>,
    pub baked_sun: Option<SkySun>,

    pub layers: Option<Arc<AtlasLayers>>,

    pub shipped_atlas: Option<(Vec<f32>, u32, u32)>,

    pub sunless: Option<(wgpu::BindGroup, wgpu::Texture)>,

    pub off_bake: bool,

    pub sun_faces_baked: Option<crate::renderer::brush_pipeline::SunFaces>,

    pub candelas: f32,

    pub lever: ClockOverride,
    pub last_tick: Option<std::time::Instant>,

    pub ground_from: Option<(f32, [f32; 3])>,
    pub ground_build: Option<mpsc::Receiver<crate::renderer::ground_map::PreparedGround>>,

    pub daylight: f32,

    pub probe_lamps: Option<ProbeSource>,
    pub building_lamps: Vec<Vec<u8>>,

    pub probe_daylight: ProbeDaylight,

    pub probe_means: Vec<(f32, f32)>,

    pub buildings: Vec<(Arc<Vec<u8>>, Option<Arc<Vec<u8>>>)>,
    pub probe_resolution: u32,

    pub probe_from: Option<(f32, [f32; 3])>,

    /// The photographs as baked; `room_descs` is them with each room's
    /// light relit to the time of day (`room_light::relit`).
    pub room_full: Vec<ProbeDesc>,
    /// Each photograph's room light by the lamps alone, beside `room_full`.
    pub room_lamps: Vec<Option<crate::renderer::room_light::RoomLight>>,

    /// When the last snapshot was applied: the next one blends in over the
    /// time between them (`Sky::apply_snapshot_blended`).
    pub last_apply: Option<std::time::Instant>,
    /// The daylight the blend starts from, beside `daylight` it goes to.
    pub daylight_from: f32,
    /// The stops the brush atlas on the GPU is stored brighter by
    /// (`time_of_day::atlas_stops`); 0 for the shipped atlas.
    pub atlas_stops: f32,
}

const GROUND_RATIO: f32 = 1.5;
const GROUND_ANGLE_COS: f32 = 0.9986;

impl XrRenderer {

    pub fn set_time_of_day(&mut self, params: Option<TimeOfDayParams>) {
        self.tod.scene_params = params;
        self.tod.runtime = None;
        self.refresh_time_of_day_mode();
    }

    pub fn set_daylight_layers(&mut self, layers: Option<AtlasLayers>) {
        self.tod.layers = layers.map(Arc::new);
        if self.tod.runtime.is_some() {
            self.tod.runtime = None;
            self.refresh_time_of_day_mode();
        }
    }

    pub fn time_of_day_hour(&self) -> Option<f32> {
        self.tod.runtime.as_ref().map(TimeOfDayRuntime::hour)
    }

    fn refresh_time_of_day_mode(&mut self) {
        let lever = ClockOverride { hour: self.levers.time_of_day_hour, day_minutes: self.levers.time_of_day_minutes };
        self.tod.lever = lever;
        let wanted = self.tod.scene_params.or_else(|| {
            (lever.hour.is_some() || lever.day_minutes.is_some()).then(|| {
                let p = TimeOfDayParams::default();
                match self.tod.baked_sun {
                    Some(s) => p.with_sun_azimuth_of(s.direction),
                    None => p,
                }
            })
        });
        match (wanted, self.tod.runtime.is_some()) {
            (Some(params), false) => self.start_time_of_day(params),
            (None, true) => self.stop_time_of_day(),
            _ => {}
        }
    }

    fn start_time_of_day(&mut self, params: TimeOfDayParams) {
        let started = std::time::Instant::now();
        let baked_sky = self.tod.baked_sky.unwrap_or(self.sky.irradiance);
        let (runtime, first) = TimeOfDayRuntime::new(params, baked_sky, self.tod.baked_sun, self.tod.layers.clone());
        self.tod.candelas = first.snapshot.candelas_per_engine;
        self.tod.runtime = Some(runtime);
        self.eye.borrow_mut().set_max_exposure(NIGHT_MAX_EXPOSURE);
        self.set_sun_off_bake(true);
        let (daylight, dir) = (first.daylight, first.snapshot.light.map_or([0.0, -1.0, 0.0], |l| l.direction));
        self.apply_time_of_day(first);

        if self.probe_stream.get_mut().is_some() {
            self.relight_probes(daylight, Some(dir));
        }
        log::info!(
            "TIMEOFDAY on: {:.2} h, a day in {} min, latitude {}, day {}, layers {}; first sky in {} ms",
            params.start_hour,
            params.day_minutes,
            params.latitude_deg,
            params.day_of_year,
            self.tod.layers.is_some(),
            started.elapsed().as_millis()
        );
    }

    fn stop_time_of_day(&mut self) {
        self.tod.runtime = None;
        self.tod.last_apply = None;
        if let Some((pano, rotation, intensity)) = self.tod.photographed.take() {
            let keep = (pano.clone(), rotation, intensity);
            self.sky = crate::renderer::sky::Sky::new(&self.wgpu_device, &self.wgpu_queue, &self.sky_pipeline.layout, &pano, rotation, intensity);
            self.tod.photographed = Some(keep);
        }
        if let (Some((atlas, w, h)), Some(lm)) = (self.tod.shipped_atlas.as_ref(), self.brush_lightmap.as_ref()) {
            crate::renderer::mesh::write_lightmap_mips(&self.wgpu_queue, &lm.texture, &crate::renderer::mesh::lightmap_mips_f16(atlas, *w, *h));
        }
        self.tod.atlas_stops = 0.0;
        {
            let mut eye = self.eye.borrow_mut();
            eye.set_max_exposure(crate::renderer::exposure::MAX_EXPOSURE);
            eye.relight(self.sky.irradiance, 1.0);
        }
        self.relight_probes(1.0, None);
        self.tod_relight_rooms(1.0);
        self.set_sun_off_bake(false);
        self.ground_dirty = true;
        self.tod.ground_from = None;
        log::info!("TIMEOFDAY off: the photographed sky again");
    }

    fn set_sun_off_bake(&mut self, off: bool) {
        if self.tod.off_bake == off {
            return;
        }
        self.tod.off_bake = off;
        if off {
            self.tod.sun_faces_baked = self.sun_faces.take();
            if self.tod.sunless.is_none() {
                self.tod.sunless = self.brush_lightmap.as_ref().and_then(|lm| {
                    lm.sunless_bind_group(&self.wgpu_device, &self.wgpu_queue, &self.brush_pipeline.lightmap_layout)
                });
            }
        } else if self.tod.sun_faces_baked.is_some() {
            self.sun_faces = self.tod.sun_faces_baked.take();
        }

        if self.terrain_sky_occlusion.is_some() {
            self.terrain_sun_baked = !off && self.terrain_sky_occlusion.as_ref().is_some_and(|s| s.sun_baked_everywhere());
            self.rebuild_terrain_material();
        }
    }

    pub(super) fn tod_brush_lightmap_bg(&self) -> Option<&wgpu::BindGroup> {
        self.tod.off_bake.then(|| self.tod.sunless.as_ref().map(|(bg, _)| bg)).flatten()
    }

    pub(super) fn tod_brush_atlas_changed(&mut self, shipped: Option<(Vec<f32>, u32, u32)>) {
        self.tod.shipped_atlas = shipped;
        self.tod.atlas_stops = 0.0;
        self.tod.sunless = None;
        if self.tod.off_bake {
            self.tod.sun_faces_baked = self.sun_faces.take();
            self.tod.sunless = self.brush_lightmap.as_ref().and_then(|lm| {
                lm.sunless_bind_group(&self.wgpu_device, &self.wgpu_queue, &self.brush_pipeline.lightmap_layout)
            });
        }
    }

    pub(super) fn tod_sky_changed(&mut self, photographed: Option<(Panorama, f32, f32)>) {
        self.tod.photographed = photographed;
        self.tod.baked_sky = Some(self.sky.irradiance);
        self.tod.baked_sun = self.sky.sun;
        if self.tod.runtime.is_some() {

            self.tod.runtime = None;
            self.tod.scene_params = None;
            self.set_sun_off_bake(false);
            self.eye.borrow_mut().set_max_exposure(crate::renderer::exposure::MAX_EXPOSURE);
        }
    }

    pub(super) fn update_time_of_day(&mut self) {
        let lever = ClockOverride { hour: self.levers.time_of_day_hour, day_minutes: self.levers.time_of_day_minutes };
        if lever != self.tod.lever {
            let was_forced = self.tod.lever.hour.is_some() || self.tod.lever.day_minutes.is_some();
            self.tod.lever = lever;

            if self.tod.scene_params.is_none() && was_forced && lever == ClockOverride::default() {
                self.stop_time_of_day();
                return;
            }
            if self.tod.runtime.is_none() {
                self.refresh_time_of_day_mode();
            }
        }
        let now = std::time::Instant::now();
        let dt = self.tod.last_tick.replace(now).map_or(0.0, |t| now.duration_since(t).as_secs_f32().min(0.25));
        let Some(rt) = self.tod.runtime.as_mut() else { return };
        let prepared = rt.advance(dt, lever);
        let seconds = rt.seconds() as f32;
        // Where the sun, moon and stars are THIS frame (cheap trig): they
        // glide rather than stepping at each snapshot.
        let astro = space_soup_sky::time_of_day::Astronomy::at(&rt.params, rt.hour(), rt.params.days_at(rt.seconds()));
        let t0 = std::time::Instant::now();
        let applied = prepared.is_some();
        if let Some(p) = prepared {
            self.apply_time_of_day(p);
        }
        let t1 = std::time::Instant::now();
        self.sky.tick(&self.wgpu_queue, seconds, dt, Some(&astro));
        // The models' room light follows the blend too (`room_light::relit`).
        let t = self.sky.blend_progress();
        let daylight_now = self.tod.daylight_from + (self.tod.daylight - self.tod.daylight_from) * t;
        self.tod_relight_rooms(daylight_now);
        // Relit on the GPU, the probes follow the same blend
        // (`ProbeStream::relight`, at the frame's `probe_daylight`), and so
        // does the brightness the reflections are matched by.
        if self.probe_stream.get_mut().as_ref().is_some_and(|s| s.smooth()) {
            self.tod.probe_daylight.set(daylight_now);
            self.tod_probe_brightness();
        }
        let t2 = std::time::Instant::now();
        let ground = self.tod.ground_build.is_some();
        self.poll_ground_rebuild();
        let t3 = std::time::Instant::now();
        // The time of day's share of the render thread, where it is felt:
        // the headset hung for a tenth of a second every few seconds while
        // the clock ran (user, 2026-10-20).
        let ms = |a: std::time::Instant, b: std::time::Instant| b.duration_since(a).as_secs_f32() * 1000.0;
        if ms(t0, t3) > 3.0 {
            log::info!(
                "TODTIME {:.1} ms: apply {:.1}{} tick {:.1} ground {:.1}{}",
                ms(t0, t3),
                ms(t0, t1),
                if applied { "" } else { " (none)" },
                ms(t1, t2),
                ms(t2, t3),
                if ground && self.tod.ground_build.is_none() { " (uploaded)" } else { "" },
            );
        }
    }

    fn apply_time_of_day(&mut self, p: Prepared) {
        let Some(rt) = self.tod.runtime.as_ref() else { return };
        let (seconds, zenith) = (rt.seconds() as f32, rt.zenith_depth);
        // Blended in over the time since the last snapshot (none for the first).
        let now = std::time::Instant::now();
        let blend = self.tod.last_apply.replace(now).map_or(0.0, |t| now.duration_since(t).as_secs_f32().clamp(0.25, 4.0));
        self.tod.daylight_from = if blend > 0.0 {
            let t = self.sky.blend_progress();
            self.tod.daylight_from + (self.tod.daylight - self.tod.daylight_from) * t
        } else {
            p.daylight
        };
        let a0 = std::time::Instant::now();
        self.sky.apply_snapshot_blended(&self.wgpu_device, &self.wgpu_queue, &self.sky_pipeline.layout, &p.snapshot, seconds, zenith, blend);
        self.tod.daylight = p.daylight;
        let a1 = std::time::Instant::now();
        self.eye.borrow_mut().relight(self.sky.irradiance, p.daylight);
        let a2 = std::time::Instant::now();
        let atlas_bytes: usize = p.atlas.as_ref().map_or(0, |c| c.iter().map(|m| m.0.len()).sum());
        if let (Some(chain), Some(lm)) = (p.atlas.as_ref(), self.brush_lightmap.as_ref()) {
            crate::renderer::mesh::write_lightmap_mips(&self.wgpu_queue, &lm.texture, chain);
            self.tod.atlas_stops = p.atlas_stops;
        }
        let a3 = std::time::Instant::now();
        let ms = |a: std::time::Instant, b: std::time::Instant| b.duration_since(a).as_secs_f32() * 1000.0;
        log::info!(
            "TODTIME apply: sky {:.1} eye {:.1} atlas {:.1} ms ({} kB)",
            ms(a0, a1),
            ms(a1, a2),
            ms(a2, a3),
            atlas_bytes / 1024,
        );
        if self.shadow_diag_frames.get() % 120 == 0 || p.atlas.is_some() {
            log::info!(
                "TIMEOFDAY {:.3} h: {} up {:.1} deg, daylight x{:.3e}, sky x{:.3e} sun x{:.3e}{}",
                p.snapshot.hour,
                if p.snapshot.light_is_moon { "moon" } else { "sun" },
                p.snapshot.light.map_or(-90.0, |l| l.direction[1].asin().to_degrees()),
                p.daylight,
                p.weights.sky[1],
                p.weights.sun[1],
                if p.atlas.is_some() { ", atlas relit" } else { "" },
            );
        }

        let dir = p.snapshot.light.map_or([0.0, -1.0, 0.0], |l| l.direction);

        let probes_stale = self.tod.probe_from.is_some_and(|(d, at)| {
            let ratio = (p.daylight.max(1e-9) / d.max(1e-9)).max(d.max(1e-9) / p.daylight.max(1e-9));
            ratio > PROBE_RELIGHT_RATIO || at[0] * dir[0] + at[1] * dir[1] + at[2] * dir[2] < PROBE_RELIGHT_ANGLE_COS
        });

        let busy = self.probe_stream.get_mut().as_ref().is_some_and(|s| s.refreshing());
        if probes_stale && !busy {
            self.relight_probes(p.daylight, Some(dir));
        }
        let stale = self.tod.ground_from.is_none_or(|(d, at)| {
            let ratio = (p.daylight.max(1e-9) / d.max(1e-9)).max(d.max(1e-9) / p.daylight.max(1e-9));
            ratio > GROUND_RATIO || at[0] * dir[0] + at[1] * dir[1] + at[2] * dir[2] < GROUND_ANGLE_COS
        });
        if stale && self.tod.ground_build.is_none() {
            self.start_ground_rebuild(p.daylight, dir);
        }
    }

    fn start_ground_rebuild(&mut self, daylight: f32, dir: [f32; 3]) {
        let Some(rx) = self.ground_map_worker("tod_ground_map") else { return };
        self.tod.ground_from = Some((daylight, dir));
        self.tod.ground_build = Some(rx);
    }

    pub(super) fn ground_map_worker(&self, name: &str) -> Option<mpsc::Receiver<crate::renderer::ground_map::PreparedGround>> {
        let h = self.ground_heights.as_ref()?;
        let copy = |i: &crate::renderer::terrain_pipeline::TerrainImage| crate::renderer::terrain_pipeline::TerrainImage {
            width: i.width,
            height: i.height,
            rgba: i.rgba.clone(),
        };
        let heights = crate::renderer::ground_map::HeightGrid { min: h.min, max: h.max, width: h.width, height: h.height, heights: h.heights.clone() };
        let sky = self.sky.irradiance;
        let sun = self.sky.sun;
        let occ = self.terrain_sky_occlusion.as_ref().map(copy);
        let layers: Vec<Option<crate::renderer::terrain_pipeline::TerrainImage>> = self.terrain_layers.iter().map(|l| l.as_ref().map(copy)).collect();
        let splat = self.terrain_splat.as_ref().map(copy);
        let settings = self.terrain_settings;
        let sea_line = self.sea_line();
        let (tx, rx) = mpsc::channel();
        std::thread::Builder::new()
            .name(name.into())
            .spawn(move || {
                let map = crate::renderer::ground_map::build(
                    &crate::renderer::ground_map::GroundInputs {
                        heights: &heights,
                        sky: &sky,
                        sun: sun.as_ref(),
                        sky_occlusion: occ.as_ref(),
                        layers: &layers,
                        splat: splat.as_ref(),
                        settings: &settings,
                        sea_line,
                    },
                    crate::renderer::ground_map::GROUND_MAP_SIZE,
                );
                let _ = tx.send(crate::renderer::ground_map::prepare(map));
            })
            .ok()?;
        Some(rx)
    }

    fn poll_ground_rebuild(&mut self) {
        let Some(rx) = self.tod.ground_build.as_ref() else { return };
        match rx.try_recv() {
            Ok(ground) => {
                self.tod.ground_build = None;
                let view = crate::renderer::ground_map::upload_prepared(&self.wgpu_device, &self.wgpu_queue, &ground);
                let map = &ground.map;
                let extent = map.max - map.min;
                self.ground_placement = Some(([map.min.x, map.min.y, 1.0 / extent.x, 1.0 / extent.y], map.top));
                self.uniform_buf.set_ground_map(view);
                self.rebind_scene_group();
                // The sun it was lit along, against the static map's
                // (`SUNSHADOW`): what reflections of the ground show.
                let d = self.tod.ground_from.map_or([0.0, -1.0, 0.0], |(_, d)| d);
                log::info!(
                    "TIMEOFDAY ground map relit along sun {:.3}/{:.3} deg",
                    d[1].clamp(-1.0, 1.0).asin().to_degrees(),
                    d[0].atan2(d[2]).to_degrees()
                );
            }
            Err(mpsc::TryRecvError::Disconnected) => self.tod.ground_build = None,
            Err(mpsc::TryRecvError::Empty) => {}
        }
    }

    /// The probes by the lamps alone: their photographs, the buildings' and
    /// each photograph's room light (`room_lamps`, one per probe, in order).
    pub fn set_probe_lamps(
        &mut self,
        lamps: Option<ProbeSource>,
        building_lamps: Vec<Vec<u8>>,
        room_lamps: Vec<Option<crate::renderer::room_light::RoomLight>>,
    ) {
        self.tod.probe_lamps = lamps;
        self.tod.building_lamps = building_lamps;
        self.tod.room_lamps = room_lamps;
    }

    /// The photographs the models take their room light from, baked.
    pub(super) fn tod_set_room_photographs(&mut self, descs: Vec<ProbeDesc>) {
        self.tod.room_full = descs;
        let k = if self.tod.runtime.is_some() { self.tod.daylight } else { 1.0 };
        self.tod_relight_rooms(k);
    }

    /// Every room's light for the models at `daylight` (`room_light::relit`):
    /// cheap (27 floats a photograph), so it follows each snapshot.
    pub(super) fn tod_relight_rooms(&mut self, daylight: f32) {
        let lamps = (self.tod.room_lamps.len() == self.tod.room_full.len()).then_some(&self.tod.room_lamps);
        self.room_descs = self
            .tod
            .room_full
            .iter()
            .enumerate()
            .map(|(i, d)| ProbeDesc {
                room_light: d.room_light.map(|full| {
                    crate::renderer::room_light::relit(&full, lamps.and_then(|l| l[i].as_ref()), daylight)
                }),
                ..*d
            })
            .collect();
    }

    pub(super) fn tod_probe_source(&self, full: ProbeSource) -> ProbeSource {
        let lamps = self.tod.probe_lamps.clone().unwrap_or_else(|| Arc::new(|_| None));
        relit_probe_source(full, lamps, self.tod.probe_daylight.clone())
    }

    pub(super) fn tod_meter_probes(
        &mut self,
        descs: &[ProbeDesc],
        resolution: u32,
        full: &ProbeSource,
    ) -> crate::renderer::exposure::EyeAdaptation {
        let mut eye = crate::renderer::exposure::EyeAdaptation::sky_only(self.tod.baked_sky.unwrap_or(self.sky.irradiance));
        let lamps = self.tod.probe_lamps.clone();
        let mean = |f: &[u8]| crate::renderer::uniforms::probe_mean_radiance(f, resolution);
        self.tod.probe_means = descs
            .iter()
            .enumerate()
            .map(|(i, d)| match full(i) {
                Some(faces) => {
                    let lamp_faces = lamps.as_ref().and_then(|l| l(i));
                    eye.add_probe_layered(&faces, lamp_faces.as_deref(), resolution, d);
                    (mean(&faces), lamp_faces.as_deref().map_or(0.0, mean))
                }
                None => (0.0, 0.0),
            })
            .collect();
        self.probe_brightness = self.tod.probe_means.iter().map(|m| m.0).collect();
        self.tod.probe_resolution = resolution;
        if self.tod.runtime.is_some() {
            eye.set_max_exposure(NIGHT_MAX_EXPOSURE);
            eye.relight(self.sky.irradiance, self.tod.daylight);
        }
        eye
    }

    pub(super) fn tod_take_buildings(&mut self, buildings: Vec<(glam::Vec3, glam::Vec3, Vec<u8>)>) -> Vec<(glam::Vec3, glam::Vec3, Vec<u8>)> {
        let lamps = std::mem::take(&mut self.tod.building_lamps);
        let k = self.tod.probe_daylight.get();
        self.tod.buildings.clear();
        buildings
            .into_iter()
            .enumerate()
            .map(|(i, (lo, hi, faces))| {
                let lamp = lamps.get(i).filter(|l| l.len() == faces.len()).cloned().map(Arc::new);
                let relit = match (&lamp, k == 1.0) {
                    (_, true) => faces.clone(),
                    (l, false) => building_daylight_faces(&faces, l.as_deref().map_or(&[][..], |l| l.as_slice()), k),
                };
                self.tod.buildings.push((Arc::new(faces), lamp));
                (lo, hi, relit)
            })
            .collect()
    }

    pub(super) fn tod_probes_loaded(&mut self) {
        self.tod.probe_from = self.tod.runtime.is_some().then(|| {
            (self.tod.probe_daylight.get(), self.sky.sun.map_or([0.0, -1.0, 0.0], |l| l.direction))
        });
        if self.tod.runtime.is_some() {
            self.tod_probe_brightness();
        }
    }

    fn tod_probe_brightness(&mut self) {
        let k = self.tod.probe_daylight.get();
        self.probe_brightness = self.tod.probe_means.iter().map(|&(full, lamps)| lamps + (full - lamps).max(0.0) * k).collect();
    }

    fn relight_probes(&mut self, daylight: f32, dir: Option<[f32; 3]>) {
        self.tod.probe_daylight.set(daylight);
        self.tod.probe_from = dir.map(|d| (daylight, d));
        self.tod_probe_brightness();
        let res = self.tod.probe_resolution;
        let sky: Option<FixedFaces> = self.sky.reflection.clone().map(|r| -> FixedFaces { Box::new(move || Some(r.cube_faces(res))) });
        let buildings: Vec<FixedFaces> = self
            .tod
            .buildings
            .iter()
            .map(|(full, lamps)| -> FixedFaces {
                let (full, lamps) = (full.clone(), lamps.clone());
                Box::new(move || {
                    Some(match (&lamps, daylight == 1.0) {
                        (_, true) => full.to_vec(),
                        (l, false) => building_daylight_faces(&full, l.as_deref().map_or(&[][..], |l| l.as_slice()), daylight),
                    })
                })
            })
            .collect();
        if let Some(stream) = self.probe_stream.get_mut().as_mut() {
            stream.refresh(sky, buildings);
            log::info!("TIMEOFDAY probes relit at daylight x{daylight:.3e}");
        }
    }

    pub(super) fn tod_night_vision(&self, adapted: Option<f32>) -> f32 {
        match (self.tod.runtime.is_some() && self.levers.night_vision, adapted) {
            (true, Some(l)) => crate::renderer::time_of_day::night_vision(l, self.tod.candelas),
            _ => 0.0,
        }
    }
}
