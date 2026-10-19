//! THE GROUND'S PIPELINES AS A SET, so they can be built again for a level
//! whose ground is not what the last build assumed.
//!
//! Whether a level's ground is PAINTED (has a splat map) is a fact of the
//! level, and the ground's shaders are built for it
//! (`terrain_pipeline::GROUND_PAINTED`): a painted level's carry none of the
//! slope-driven layer weights -- 176 of the gentle reader's 2,719
//! instructions, an `acos` among them, that every pixel computed and threw
//! away while "painted" arrived in a uniform. The renderer exists before any
//! level does, so the first build goes by what the app expects
//! (`terrain_pipeline::expect_painted_ground`); `set_terrain_splat` then asks
//! [`XrRenderer::ground_for_level`], which builds the set again only when the
//! level's material says otherwise (a level change between a painted and an
//! unpainted level, or a wrong guess).
//!
//! The SAME PICTURE either way: the painted build is each shader with
//! `select(derived, authored, mat.use_splat > 0.5)` written as `authored`,
//! which is what the select returns when the material was given a splat map
//! (`terrain_pipeline::tests::the_painted_build_keeps_the_splats_weights_alone`,
//! `every_ground_twin_takes_the_painted_build`).
//!
//! Not rebuilt: the ground's `Scene`-role pipelines (`terrain_pipeline`, the
//! stereo family's), which no shipped frame draws with and whose material
//! layout every ground bind group was made from -- they keep the choice in
//! them, right for any level -- and the cave's own shaders, which are not the
//! ground's.

use super::XrRenderer;
use crate::renderer::terrain_pipeline::{TerrainPipeline, TerrainWeatherTwins};
use wgpu::{BindGroupLayout, Device, TextureFormat};

/// Every ground pipeline a shipped frame draws with, built from
/// `terrain_shader_for` as `GROUND_PAINTED` stands.
pub(super) struct GroundSet {
    pub pass: TerrainPipeline,
    pub pass_poolless: TerrainPipeline,
    pub dedup_passes: Option<[TerrainPipeline; 2]>,
    pub outdoor_passes: Option<[TerrainPipeline; 2]>,
    pub nowhere_passes: Option<[TerrainPipeline; 2]>,
    pub open_passes: Option<[TerrainPipeline; 2]>,
    /// The weather's open twin answering with the sky alone
    /// (`Levers::puddle_sky`).
    pub puddle_pass: Option<TerrainPipeline>,
    pub reader: TerrainPipeline,
    pub reader_twins: [TerrainPipeline; 3],
    pub gentle: Option<[TerrainPipeline; 4]>,
    pub steep: Option<[TerrainPipeline; 4]>,
}

impl GroundSet {
    pub(super) fn build(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        samples: u32,
        probe_pass_layout: &BindGroupLayout,
        fixups: &crate::renderer::probe_fixup::ProbeFixups,
    ) -> Self {
        let pass = TerrainPipeline::new_probe_pass(device, uniform_layout, fixups);
        let pass_poolless = TerrainPipeline::new_probe_pass_poolless(device, uniform_layout, fixups);
        let dedup_passes = crate::renderer::ground_twins::dedup_passes(device, uniform_layout, fixups, None);
        let weather_layout = crate::renderer::weather::bind_group_layout(device);
        // `[dry, weather]` of each of the pass's twins by where the ground lies.
        let pair = |build: &dyn Fn(Option<&BindGroupLayout>) -> Option<TerrainPipeline>| {
            build(None).zip(build(Some(&weather_layout))).map(|(plain, wet)| [plain, wet])
        };
        let outdoor_passes = pair(&|weather| TerrainPipeline::new_probe_pass_outdoors(device, uniform_layout, fixups, weather));
        if outdoor_passes.is_none() {
            log::warn!("terrain: the probe pass's outdoor twin no longer matches its shader; the ground draws by the full pass");
        }
        let nowhere_passes = pair(&|weather| TerrainPipeline::new_probe_pass_nowhere(device, uniform_layout, fixups, weather));
        if nowhere_passes.is_none() {
            log::warn!("terrain: the probe pass's nowhere twin no longer matches its shader; the ground draws by the full pass");
        }
        let open_passes = pair(&|weather| TerrainPipeline::new_probe_pass_open(device, uniform_layout, fixups, weather));
        if open_passes.is_none() {
            log::warn!("terrain: the probe pass's open-ground twin no longer matches its shader; that ground draws by the nowhere twin");
        }
        let puddle_pass = TerrainPipeline::new_probe_pass_puddle(device, uniform_layout, fixups, &weather_layout);
        if puddle_pass.is_none() {
            log::warn!("terrain: the probe pass's puddle twin no longer matches its shader; puddles draw by the open-ground twin");
        }
        if dedup_passes.is_none() {
            log::warn!("terrain: the probe pass's dedup edits no longer match; it draws as it was");
        }
        let reader = TerrainPipeline::new_probe_reader(device, format, uniform_layout, samples, probe_pass_layout);
        let reader_twins = TerrainPipeline::new_probe_reader_twins(device, format, uniform_layout, samples, probe_pass_layout);
        let gentle = crate::renderer::ground_twins::gentle_readers(device, format, uniform_layout, samples, probe_pass_layout, None);
        let steep = gentle
            .as_ref()
            .and_then(|_| crate::renderer::ground_twins::steep_readers(device, format, uniform_layout, samples, probe_pass_layout));
        if gentle.is_none() {
            log::warn!("terrain: the ground's steep tests are not where its gentle twins take them out; the full readers draw everywhere");
        }
        Self { pass, pass_poolless, dedup_passes, outdoor_passes, nowhere_passes, open_passes, puddle_pass, reader, reader_twins, gentle, steep }
    }
}

/// The weather's ground pipelines: the twins, the gentle readers' twins by
/// what the ground holds (empty when they did not build), the pass's dedup
/// twins, and the mixed and steep readers' wet-only twins. See
/// `XrRenderer::set_weather`.
pub(super) fn weather_ground(
    device: &Device,
    format: TextureFormat,
    uniform_layout: &BindGroupLayout,
    samples: u32,
    probe_pass_layout: &BindGroupLayout,
    fixups: &crate::renderer::probe_fixup::ProbeFixups,
    weather_layout: &BindGroupLayout,
) -> (TerrainWeatherTwins, Vec<[TerrainPipeline; 4]>, Option<[TerrainPipeline; 2]>, Option<[[TerrainPipeline; 4]; 2]>) {
    use crate::renderer::weather;
    let twins = TerrainPipeline::new_weather_twins(device, format, uniform_layout, samples, probe_pass_layout, fixups, weather_layout);
    // The gentle readers' twins, by what the ground holds. See `ground_twins`.
    let gentle: Vec<[TerrainPipeline; 4]> = weather::WeatherKinds::ALL
        .iter()
        .map_while(|kinds| {
            crate::renderer::ground_twins::gentle_readers(device, format, uniform_layout, samples, probe_pass_layout, Some((weather_layout, *kinds)))
        })
        .collect();
    if gentle.len() != weather::WeatherKinds::ALL.len() {
        log::warn!("WEATHER: the ground's gentle weather twins did not build; its full ones draw everywhere");
    }
    let gentle = if gentle.len() == weather::WeatherKinds::ALL.len() { gentle } else { Vec::new() };
    let dedup_passes = crate::renderer::ground_twins::dedup_passes(device, uniform_layout, fixups, Some(weather_layout));
    // The mixed and steep readers' wet-only twins, for chunks that hold no
    // standing water and no snow. See `ground_twins::wet_readers`.
    let wet = crate::renderer::ground_twins::wet_readers(device, format, uniform_layout, samples, probe_pass_layout, weather_layout);
    if wet.is_none() {
        log::warn!("WEATHER: the ground's wet-only mixed and steep twins did not build; its full weather reader draws them");
    }
    (twins, gentle, dedup_passes, wet)
}

impl XrRenderer {
    /// A level's ground material is in hand (`set_terrain_splat`): the
    /// ground's pipelines built again if they were built for the other kind
    /// of ground. Nothing, and no cost, when they already match -- which is
    /// every launch whose entry level the app guessed right.
    pub(super) fn ground_for_level(&mut self) {
        let painted = self.terrain_splat.is_some();
        if !crate::renderer::terrain_pipeline::ground_build_moves(painted) {
            return;
        }
        let started = std::time::Instant::now();
        let format = TextureFormat::Rgba8UnormSrgb;
        let samples = self.msaa.samples();
        let probe_pass_layout = crate::renderer::brush_pipeline::probe_pass::bind_group_layout(&self.wgpu_device);
        let set = GroundSet::build(&self.wgpu_device, format, &self.uniform_buf.layout, samples, &probe_pass_layout, &self.probe_fixups);
        self.terrain_probe_pass_pipeline = set.pass;
        self.terrain_probe_pass_poolless_pipeline = set.pass_poolless;
        self.terrain_dedup_passes = set.dedup_passes;
        self.terrain_outdoor_passes = set.outdoor_passes;
        self.terrain_nowhere_passes = set.nowhere_passes;
        self.terrain_open_passes = set.open_passes;
        self.terrain_puddle_pass = set.puddle_pass;
        self.terrain_probe_reader_pipeline = set.reader;
        self.terrain_reader_twins = set.reader_twins;
        self.terrain_gentle = set.gentle;
        self.terrain_steep = set.steep;
        // Measurement pipelines built for the other ground: dropped, so the
        // lever that asked for one builds it again.
        self.terrain_cut_pipeline = None;
        self.terrain_inlined = None;
        if self.weather.is_some() {
            let layout = crate::renderer::weather::bind_group_layout(&self.wgpu_device);
            let (twins, gentle, dedup_passes, wet) =
                weather_ground(&self.wgpu_device, format, &self.uniform_buf.layout, samples, &probe_pass_layout, &self.probe_fixups, &layout);
            if let Some(w) = self.weather.as_mut() {
                w.twins = twins;
                w.gentle = gentle;
                w.dedup_passes = dedup_passes;
                w.wet = wet;
            }
        }
        log::info!(
            "GROUND: pipelines rebuilt for a{} level in {:.0} ms",
            if painted { " painted" } else { "n unpainted" },
            started.elapsed().as_secs_f64() * 1000.0,
        );
    }
}
