//! THE TIME OF DAY IN THE EDITOR'S VIEW (2026-10-21). The desktop view had
//! only ever drawn the photographed sky, so a level's day, its clouds and
//! their weather could be authored but never seen until the headset. This
//! runs the headset's own runtime (`TimeOfDayRuntime`, the same snapshot
//! worker) on the view's sky dome: the sky relit each snapshot and blended
//! in, the sun, moon and stars where they are this frame, and the volumetric
//! clouds built a slice a frame round the camera with their shadow on the
//! ground. What it does not do (yet): relight baked lightmaps or probes from
//! daylight layers -- the editor's scene light comes from `DayLight`.

use super::Renderer;
use crate::renderer::time_of_day::{ClockOverride, TimeOfDayRuntime};
use space_soup_sky::time_of_day::{Astronomy, TimeOfDayParams};
use space_soup_sky::{SkyIrradiance, SkySun};

pub(super) struct DesktopDay {
    runtime: TimeOfDayRuntime,
    /// The photographed sky's light, which every snapshot is relit from.
    baked: (SkyIrradiance, Option<SkySun>),
    last_apply: Option<std::time::Instant>,
    light: DayLight,
}

/// What the day lights the editor's scene with this frame.
#[derive(Clone, Copy, Debug)]
pub struct DayLight {
    /// Local solar hours.
    pub hour: f32,
    /// The one directional light: the sun, or the moon after it sets, as
    /// the clouds let it through on average.
    pub light: Option<SkySun>,
    pub light_is_moon: bool,
    /// The sky's light on the scene.
    pub irradiance: SkyIrradiance,
    /// The day's share of the photographed sky's light (1 at its hour).
    pub daylight: f32,
    /// Engine units per candela per square metre, for the eye's meter.
    pub candelas_per_engine: f32,
}

impl Renderer {
    /// Run the time of day on the view's sky (`None` stops it). The sky the
    /// view has (`set_sky_panorama`) is the photographed base it is relit
    /// from; with none, nothing runs and `None` comes back. After a stop the
    /// caller sets the photographed sky again.
    pub fn set_time_of_day(&mut self, params: Option<TimeOfDayParams>) -> Option<DayLight> {
        let Some(params) = params else {
            if self.day.take().is_some() {
                self.sky_dome = None;
                self.post.cloud_map = [[0.0; 4]; 3];
            }
            return None;
        };
        let sky = self.sky_dome.as_mut()?;
        crate::renderer::cloud_worker::enable();
        // Relit from the photograph as first set, not from a relit sky.
        let (baked_sky, baked_sun) = match &self.day {
            Some(d) => d.baked,
            None => (sky.irradiance, sky.sun),
        };
        let (runtime, first) = TimeOfDayRuntime::new(params, baked_sky, baked_sun, None);
        sky.apply_snapshot(&self.device, &self.queue, &self.sky_pipeline.layout, &first.snapshot, runtime.seconds() as f32, runtime.zenith_depth);
        let light = DayLight {
            hour: first.snapshot.hour,
            light: first.snapshot.light,
            light_is_moon: first.snapshot.light_is_moon,
            irradiance: first.snapshot.irradiance,
            daylight: first.daylight,
            candelas_per_engine: first.snapshot.candelas_per_engine,
        };
        self.day = Some(DesktopDay { runtime, baked: (baked_sky, baked_sun), last_apply: Some(std::time::Instant::now()), light });
        Some(light)
    }

    pub fn time_of_day_on(&self) -> bool {
        self.day.is_some()
    }

    /// One frame of the day: the clock moved on by `dt` seconds (or pinned
    /// by `clock`), the newest snapshot blended in, and the clouds' next
    /// slice seen from `eye` (world) with their shadow round it.
    pub fn tick_time_of_day(&mut self, dt: f32, clock: ClockOverride, eye: [f32; 3], cloud_shadows: bool) -> Option<DayLight> {
        let (Some(day), Some(sky)) = (self.day.as_mut(), self.sky_dome.as_mut()) else { return None };
        let rt = &mut day.runtime;
        if let Some(p) = rt.advance(dt, clock) {
            let now = std::time::Instant::now();
            let blend = day.last_apply.replace(now).map_or(0.0, |t| now.duration_since(t).as_secs_f32().clamp(0.25, 4.0));
            sky.apply_snapshot_blended(&self.device, &self.queue, &self.sky_pipeline.layout, &p.snapshot, rt.seconds() as f32, rt.zenith_depth, blend);
            day.light = DayLight {
                hour: p.snapshot.hour,
                light: p.snapshot.light,
                light_is_moon: p.snapshot.light_is_moon,
                irradiance: p.snapshot.irradiance,
                daylight: p.daylight,
                candelas_per_engine: p.snapshot.candelas_per_engine,
            };
        }
        let seconds = rt.seconds() as f32;
        let astro = Astronomy::at(&rt.params, rt.hour(), rt.params.days_at(rt.seconds()));
        sky.tick(&self.queue, seconds, dt, Some(&astro));
        sky.step_clouds(&self.device, &self.queue, eye, cloud_shadows, Some(self.uniform_buf.cloud_shadow_view()), false);
        self.post.cloud_map = sky.cloud_map(0.0, [0.0; 3]);
        day.light.hour = rt.hour();
        // The light as the blend has it, the clouds' average let through.
        day.light.irradiance = sky.irradiance;
        if let Some(l) = sky.sun_stepped() {
            day.light.light = Some(l);
        }
        Some(day.light)
    }
}
