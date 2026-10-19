use super::XrRenderer;
use crate::renderer::terrain_pipeline::{TerrainPipeline, TerrainWeatherTwins};
use wgpu::{BindGroupLayout, Device, TextureFormat};

pub(super) struct GroundSet {
    pub pass: TerrainPipeline,
    pub pass_poolless: TerrainPipeline,
    pub dedup_passes: Option<[TerrainPipeline; 2]>,
    pub outdoor_passes: Option<[TerrainPipeline; 2]>,
    pub nowhere_passes: Option<[TerrainPipeline; 2]>,
    pub open_passes: Option<[TerrainPipeline; 2]>,

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

    let wet = crate::renderer::ground_twins::wet_readers(device, format, uniform_layout, samples, probe_pass_layout, weather_layout);
    if wet.is_none() {
        log::warn!("WEATHER: the ground's wet-only mixed and steep twins did not build; its full weather reader draws them");
    }
    (twins, gentle, dedup_passes, wet)
}

impl XrRenderer {

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
