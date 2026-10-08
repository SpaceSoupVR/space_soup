//! The time of day on the headset: what `XrRenderer` keeps for it, and how a
//! new moment reaches the sky, the lights, the shadows, the bake and the eye.
//! See `renderer::time_of_day` for the design.

use std::sync::{mpsc, Arc};

use super::XrRenderer;
use crate::renderer::probe_stream::{FixedFaces, ProbeDesc, ProbeSource};
use crate::renderer::time_of_day::{
    relight_probe_faces, relit_probe_source, ClockOverride, Prepared, ProbeDaylight, TimeOfDayRuntime, NIGHT_MAX_EXPOSURE,
    PROBE_RELIGHT_ANGLE_COS, PROBE_RELIGHT_RATIO,
};
use space_soup_sky::time_of_day::{AtlasLayers, TimeOfDayParams};
use space_soup_sky::{Panorama, SkyIrradiance, SkySun};

/// Everything the time of day holds between frames. Default: none of it, a
/// photographed sky lit exactly as before.
#[derive(Default)]
pub(super) struct TodState {
    pub runtime: Option<TimeOfDayRuntime>,
    /// The level's own time of day, when it authors one.
    pub scene_params: Option<TimeOfDayParams>,
    /// The photographed sky the scene names, to go back to.
    pub photographed: Option<(Panorama, f32, f32)>,
    /// What the level's bake was lit with: the photographed sky's ambient and
    /// sun. The layer weights are relative to them.
    pub baked_sky: Option<SkyIrradiance>,
    pub baked_sun: Option<SkySun>,
    /// The brush atlas's daylight layers, when the level has them.
    pub layers: Option<Arc<AtlasLayers>>,
    /// The shipped brush atlas, to put back.
    pub shipped_atlas: Option<(Vec<f32>, u32, u32)>,
    /// The brush atlas's bind group with the neutral sun mask. See
    /// `LoadedTexture::sunless_bind_group`.
    pub sunless: Option<(wgpu::BindGroup, wgpu::Texture)>,
    /// The sun stands off its baked direction: the brushes and the ground take
    /// the level's sun shadow from the static map.
    pub off_bake: bool,
    /// The brushes' sun classes, set aside while `off_bake` (each class's
    /// reader assumes the BAKED sun).
    pub sun_faces_baked: Option<crate::renderer::brush_pipeline::SunFaces>,
    /// cd/m^2 per engine unit, for the night vision.
    pub candelas: f32,
    /// The lever's clock, as last seen.
    pub lever: ClockOverride,
    pub last_tick: Option<std::time::Instant>,
    /// The ground map's light: the daylight share and the light's direction it
    /// was last built at, and a build in flight.
    pub ground_from: Option<(f32, [f32; 3])>,
    pub ground_build: Option<mpsc::Receiver<crate::renderer::ground_map::GroundMap>>,
    /// The latest daylight share (the ground's light now over the bake's).
    pub daylight: f32,
    /// The lamps-only probe photographs (`BAKE_DAYLIGHT=off bake probe`) and
    /// the buildings' outsides by the lamps alone, for the next probe set.
    pub probe_lamps: Option<ProbeSource>,
    pub building_lamps: Vec<Vec<u8>>,
    /// The daylight the probe source relights at (1: as photographed).
    pub probe_daylight: ProbeDaylight,
    /// Each probe's mean radiance as photographed and by the lamps alone,
    /// which make its brightness (`PROBE_NORMALISATION`) at any daylight.
    pub probe_means: Vec<(f32, f32)>,
    /// The buildings' outsides as photographed and by the lamps alone.
    pub buildings: Vec<(Arc<Vec<u8>>, Option<Arc<Vec<u8>>>)>,
    pub probe_resolution: u32,
    /// The daylight and the light's direction the probes were last lit at.
    pub probe_from: Option<(f32, [f32; 3])>,
}

/// A ground map is rebuilt (on a worker: ~0.5 s of one Quest core) when the
/// daylight has moved by this ratio, or the light by [`GROUND_ANGLE`].
const GROUND_RATIO: f32 = 1.5;
const GROUND_ANGLE_COS: f32 = 0.9986; // 3 degrees

impl XrRenderer {
    /// THE LEVEL'S TIME OF DAY, after its sky (`set_sky`): `Some` turns the
    /// photographed sky over to the time-of-day sky, `None` keeps the photograph
    /// -- unless the `time_of_day_hour` or `time_of_day_minutes` lever asks for
    /// one, which then takes the defaults. Blocks for the first moment's sky
    /// (~50 ms on a Quest core), so the first frame is already right.
    pub fn set_time_of_day(&mut self, params: Option<TimeOfDayParams>) {
        self.tod.scene_params = params;
        self.tod.runtime = None;
        self.refresh_time_of_day_mode();
    }

    /// The brush atlas's daylight layers (`space_soup_engine::daylight`), or
    /// `None`. Without them a moving sun still moves the direct light, the
    /// shadows and the sky, but the baked bounce keeps the bake's hour.
    pub fn set_daylight_layers(&mut self, layers: Option<AtlasLayers>) {
        self.tod.layers = layers.map(Arc::new);
        if self.tod.runtime.is_some() {
            self.tod.runtime = None;
            self.refresh_time_of_day_mode();
        }
    }

    /// The hour, while a time of day runs.
    pub fn time_of_day_hour(&self) -> Option<f32> {
        self.tod.runtime.as_ref().map(TimeOfDayRuntime::hour)
    }

    /// Start or stop the time of day to match the level and the levers.
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
        // Probes already in (a lever on a running level): relight them now.
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

    /// Back to the photographed sky, its sun, its baked shadows and its bake,
    /// exactly as loaded.
    fn stop_time_of_day(&mut self) {
        self.tod.runtime = None;
        if let Some((pano, rotation, intensity)) = self.tod.photographed.take() {
            let keep = (pano.clone(), rotation, intensity);
            self.sky = crate::renderer::sky::Sky::new(&self.wgpu_device, &self.wgpu_queue, &self.sky_pipeline.layout, &pano, rotation, intensity);
            self.tod.photographed = Some(keep);
        }
        if let (Some((atlas, w, h)), Some(lm)) = (self.tod.shipped_atlas.as_ref(), self.brush_lightmap.as_ref()) {
            crate::renderer::mesh::write_lightmap_mips(&self.wgpu_queue, &lm.texture, &crate::renderer::mesh::lightmap_mips_f16(atlas, *w, *h));
        }
        {
            let mut eye = self.eye.borrow_mut();
            eye.set_max_exposure(crate::renderer::exposure::MAX_EXPOSURE);
            eye.relight(self.sky.irradiance, 1.0);
        }
        self.relight_probes(1.0, None);
        self.set_sun_off_bake(false);
        self.ground_dirty = true;
        self.tod.ground_from = None;
        log::info!("TIMEOFDAY off: the photographed sky again");
    }

    /// The level's sun shadow from the bake (`false`) or from the static map
    /// (`true`): the brushes' lightmap group with or without the baked mask,
    /// their sun classes set aside or back, and the ground's map with its sun
    /// marked unbaked or as baked.
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
        // The ground: its baked sun marked absent (alpha 255 in its map), so
        // its reader takes the static map. See `rebuild_terrain_material`.
        if self.terrain_sky_occlusion.is_some() {
            self.terrain_sun_baked = !off && self.terrain_sky_occlusion.as_ref().is_some_and(|s| s.sun_baked_everywhere());
            self.rebuild_terrain_material();
        }
    }

    /// The brushes' lightmap group as the sun stands: see `brush_lightmap_bg`.
    pub(super) fn tod_brush_lightmap_bg(&self) -> Option<&wgpu::BindGroup> {
        self.tod.off_bake.then(|| self.tod.sunless.as_ref().map(|(bg, _)| bg)).flatten()
    }

    /// A new brush atlas arrived (`set_brush_lightmap`): keep it to put back,
    /// and remake the sunless group over it.
    pub(super) fn tod_brush_atlas_changed(&mut self, shipped: Option<(Vec<f32>, u32, u32)>) {
        self.tod.shipped_atlas = shipped;
        self.tod.sunless = None;
        if self.tod.off_bake {
            self.tod.sun_faces_baked = self.sun_faces.take();
            self.tod.sunless = self.brush_lightmap.as_ref().and_then(|lm| {
                lm.sunless_bind_group(&self.wgpu_device, &self.wgpu_queue, &self.brush_pipeline.lightmap_layout)
            });
        }
    }

    /// A new photographed sky arrived (`set_sky`): what the bake was lit with.
    pub(super) fn tod_sky_changed(&mut self, photographed: Option<(Panorama, f32, f32)>) {
        self.tod.photographed = photographed;
        self.tod.baked_sky = Some(self.sky.irradiance);
        self.tod.baked_sun = self.sky.sun;
        if self.tod.runtime.is_some() {
            // A new level: its own time of day (`set_time_of_day`) follows.
            self.tod.runtime = None;
            self.tod.scene_params = None;
            self.set_sun_off_bake(false);
            self.eye.borrow_mut().set_max_exposure(crate::renderer::exposure::MAX_EXPOSURE);
        }
    }

    /// EVERY FRAME, before anything reads the sky: run the clock, take the
    /// worker's newest moment, follow the levers, and keep the ground map's
    /// light in step.
    pub(super) fn update_time_of_day(&mut self) {
        let lever = ClockOverride { hour: self.levers.time_of_day_hour, day_minutes: self.levers.time_of_day_minutes };
        if lever != self.tod.lever {
            let was_forced = self.tod.lever.hour.is_some() || self.tod.lever.day_minutes.is_some();
            self.tod.lever = lever;
            // Lever cleared on a level with no time of day of its own: back
            // to the photograph. Set: start one (or jump the running clock).
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
        if let Some(p) = prepared {
            self.apply_time_of_day(p);
        } else {
            self.sky.set_clock(&self.wgpu_queue, seconds);
        }
        self.poll_ground_rebuild();
    }

    /// One moment, everywhere it is read.
    fn apply_time_of_day(&mut self, p: Prepared) {
        let Some(rt) = self.tod.runtime.as_ref() else { return };
        let (seconds, zenith) = (rt.seconds() as f32, rt.zenith_depth);
        self.sky.apply_snapshot(&self.wgpu_device, &self.wgpu_queue, &self.sky_pipeline.layout, &p.snapshot, seconds, zenith);
        self.tod.daylight = p.daylight;
        self.eye.borrow_mut().relight(self.sky.irradiance, p.daylight);
        if let (Some(chain), Some(lm)) = (p.atlas.as_ref(), self.brush_lightmap.as_ref()) {
            crate::renderer::mesh::write_lightmap_mips(&self.wgpu_queue, &lm.texture, chain);
        }
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
        // The ground map's light, when it has moved far enough.
        let dir = p.snapshot.light.map_or([0.0, -1.0, 0.0], |l| l.direction);
        // The probes' too, on their own step.
        let probes_stale = self.tod.probe_from.is_some_and(|(d, at)| {
            let ratio = (p.daylight.max(1e-9) / d.max(1e-9)).max(d.max(1e-9) / p.daylight.max(1e-9));
            ratio > PROBE_RELIGHT_RATIO || at[0] * dir[0] + at[1] * dir[1] + at[2] * dir[2] < PROBE_RELIGHT_ANGLE_COS
        });
        // One relight at a time: a 256 px level's round is ~17 prefilters on
        // the stream's worker, and asking again before it is through only
        // queues photographs of an hour already past.
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

    /// The ground map rebuilt off the render thread from copies of its inputs.
    fn start_ground_rebuild(&mut self, daylight: f32, dir: [f32; 3]) {
        let Some(h) = self.ground_heights.as_ref() else { return };
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
        let (tx, rx) = mpsc::channel();
        self.tod.ground_from = Some((daylight, dir));
        self.tod.ground_build = Some(rx);
        std::thread::Builder::new()
            .name("tod_ground_map".into())
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
                    },
                    crate::renderer::ground_map::GROUND_MAP_SIZE,
                );
                let _ = tx.send(map);
            })
            .ok();
    }

    fn poll_ground_rebuild(&mut self) {
        let Some(rx) = self.tod.ground_build.as_ref() else { return };
        match rx.try_recv() {
            Ok(map) => {
                self.tod.ground_build = None;
                let view = crate::renderer::ground_map::upload(&self.wgpu_device, &self.wgpu_queue, &map);
                let extent = map.max - map.min;
                self.ground_placement = Some(([map.min.x, map.min.y, 1.0 / extent.x, 1.0 / extent.y], map.top));
                self.uniform_buf.set_ground_map(view);
                self.rebind_scene_group();
                log::info!("TIMEOFDAY ground map relit");
            }
            Err(mpsc::TryRecvError::Disconnected) => self.tod.ground_build = None,
            Err(mpsc::TryRecvError::Empty) => {}
        }
    }

    /// THE LAMPS-ONLY PROBE PHOTOGRAPHS, before the probe set they belong to
    /// (`set_reflection_probes_with_depth`): what the probes, the buildings'
    /// outsides (`building_lamps`, in `set_building_outsides`' order) and the
    /// eye's meter are relit from as the daylight moves. `None` for a level
    /// without them: under the time of day its photographs are scaled whole.
    pub fn set_probe_lamps(&mut self, lamps: Option<ProbeSource>, building_lamps: Vec<Vec<u8>>) {
        self.tod.probe_lamps = lamps;
        self.tod.building_lamps = building_lamps;
    }

    /// The probe set's source as the stream reads it: relit at the probes'
    /// daylight, which is 1 -- the photograph untouched -- until a time of day
    /// moves it.
    pub(super) fn tod_probe_source(&self, full: ProbeSource) -> ProbeSource {
        let lamps = self.tod.probe_lamps.clone().unwrap_or_else(|| Arc::new(|_| None));
        relit_probe_source(full, lamps, self.tod.probe_daylight.clone())
    }

    /// THE EYE'S METER AND THE PROBES' BRIGHTNESS for a new probe set, in one
    /// pass over the photographs (`full`, as baked): each binned with its
    /// lamps-only layer when there is one, against the sky the BAKE was lit
    /// with, then relit to the hour when a time of day is running.
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

    /// The buildings' outsides for a new probe set, as the stream takes them:
    /// relit at the probes' daylight, the photographs kept to relight again.
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
                    (Some(l), false) => relight_probe_faces(&faces, l, k),
                    (None, false) => relight_probe_faces(&faces, &vec![0u8; faces.len()], k),
                };
                self.tod.buildings.push((Arc::new(faces), lamp));
                (lo, hi, relit)
            })
            .collect()
    }

    /// A new probe set is in: note what it was lit at, so the next moment
    /// relights it only once the light has moved.
    pub(super) fn tod_probes_loaded(&mut self) {
        self.tod.probe_from = self.tod.runtime.is_some().then(|| {
            (self.tod.probe_daylight.get(), self.sky.sun.map_or([0.0, -1.0, 0.0], |l| l.direction))
        });
        if self.tod.runtime.is_some() {
            self.tod_probe_brightness();
        }
    }

    /// Each probe's brightness at the probes' daylight: linear in it, so no
    /// pixels are read.
    fn tod_probe_brightness(&mut self) {
        let k = self.tod.probe_daylight.get();
        self.probe_brightness = self.tod.probe_means.iter().map(|&(full, lamps)| lamps + (full - lamps).max(0.0) * k).collect();
    }

    /// RELIGHT THE PROBES at `daylight`: the source reads it from now on, every
    /// resident photograph, the sky reflections see (as `self.sky` now holds
    /// it) and the buildings' outsides are made again on the probe stream's
    /// worker and written over their layers a few a frame, and the
    /// brightness the reflections are normalised by follows at once.
    /// `dir`: the light's direction now (`None`: no time of day).
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
                        (Some(l), false) => relight_probe_faces(&full, l, daylight),
                        (None, false) => relight_probe_faces(&full, &vec![0u8; full.len()], daylight),
                    })
                })
            })
            .collect();
        if let Some(stream) = self.probe_stream.get_mut().as_mut() {
            stream.refresh(sky, buildings);
            log::info!("TIMEOFDAY probes relit at daylight x{daylight:.3e}");
        }
    }

    /// How far toward rod vision the eye sees this frame: 0 for a photographed
    /// sky or with the lever off.
    pub(super) fn tod_night_vision(&self, adapted: Option<f32>) -> f32 {
        match (self.tod.runtime.is_some() && self.levers.night_vision, adapted) {
            (true, Some(l)) => crate::renderer::time_of_day::night_vision(l, self.tod.candelas),
            _ => 0.0,
        }
    }
}
