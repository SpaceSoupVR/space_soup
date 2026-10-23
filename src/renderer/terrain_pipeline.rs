use wgpu::*;

use super::cuboid::SolidVertex;
use super::material_wgsl::{wgsl_biplanar_block, wgsl_whiteout_block};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TerrainMaterialUniform {

    pub repeat: [f32; 4],

    pub slope_start_deg: f32,
    pub slope_end_deg: f32,

    pub height_start: f32,
    pub height_end: f32,

    pub macro_repeat: f32,
    pub macro_strength: f32,

    pub biplanar_start_deg: f32,

    pub use_splat: f32,

    pub normal_strength: f32,

    pub wet_line: f32,
    pub wet_band: f32,

    pub sea_line: f32,

    pub porosity: [f32; 4],

    pub film: [f32; 4],

    pub shape: [f32; 4],

    pub parallax: f32,

    pub mud_layer: f32,
    pub deform_metres: f32,
    pub deform_texel: f32,
    pub wet_dark: f32,
    pub relief_pad: [f32; 3],
}

impl Default for TerrainMaterialUniform {
    fn default() -> Self {
        Self {
            repeat: [8.0, 4.0, 10.0, 6.0],
            slope_start_deg: 22.0,
            slope_end_deg: 40.0,
            height_start: 1e9,
            height_end: 1e9,
            macro_repeat: 140.0,
            macro_strength: 0.35,
            biplanar_start_deg: 18.0,
            use_splat: 0.0,
            normal_strength: 1.0,
            wet_line: -1e9,
            wet_band: 1.0,
            sea_line: -1e9,

            porosity: WET_POROSITY,
            film: WET_FILM,
            shape: [0.0; 4],
            parallax: 0.0,
            mud_layer: -1.0,
            deform_metres: 0.1,
            deform_texel: 0.05,
            wet_dark: 0.5,
            relief_pad: [0.0; 3],
        }
    }
}

pub(crate) const SEA_LINE_WGSL: &str = "    // UNDER STILL WATER the ground is soaked: see `SEA_LINE_WGSL`.
    let sea_in = dot(w, mat.porosity) * clamp((mat.sea_line - in.tex_pos.y) * 20.0, 0.0, 1.0);
    let sea_back = 0.435 * albedo;
    albedo = albedo * ((1.0 - sea_in * 0.446) + (sea_in * 0.554) * (sea_back * sea_back + sea_back));
";

pub const WET_POROSITY: [f32; 4] = [0.85, 0.35, 0.7, 1.0];
pub const WET_FILM: [f32; 4] = [0.12, 0.75, 0.3, 0.9];

pub struct TerrainPipeline {

    pub pipeline: RenderPipeline,

    pub shaped: Option<RenderPipeline>,
    pub material_layout: BindGroupLayout,
}

const RELIEF_WGSL: &str = r#"
fn layers_height(uv: vec2<f32>, ddx: vec2<f32>, ddy: vec2<f32>, layer: i32) -> f32 {
    let r = max(mat.repeat[layer], 0.001);
    return textureSampleGrad(ao_tex, layer_samp, uv / r, layer, ddx / r, ddy / r).g;
}

fn dominant_layer(w: vec4<f32>) -> i32 {
    var best = 0;
    var bw = w.x;
    if (w.y > bw) { best = 1; bw = w.y; }
    if (w.z > bw) { best = 2; bw = w.z; }
    if (w.w > bw) { best = 3; }
    return best;
}

fn parallax_shift(uv: vec2<f32>, ddx: vec2<f32>, ddy: vec2<f32>, w: vec4<f32>, eye: vec3<f32>, relief: f32) -> vec2<f32> {
    let up = max(eye.y, 0.12);
    let slide = -eye.xz / up;
    let steps = mix(6.0, 14.0, 1.0 - up);
    let layer = dominant_layer(w);
    let layer_d = relief / steps;
    let step_uv = slide * layer_d;
    var cur_uv = uv;
    var cur_depth = 0.0;
    var surf = (1.0 - layers_height(cur_uv, ddx, ddy, layer)) * relief;
    var prev_uv = uv;
    var prev_depth = 0.0;
    var prev_surf = surf;
    for (var i = 0; i < 14; i++) {
        if (cur_depth >= surf || f32(i) >= steps) {
            break;
        }
        prev_uv = cur_uv;
        prev_depth = cur_depth;
        prev_surf = surf;
        cur_uv = cur_uv + step_uv;
        cur_depth = cur_depth + layer_d;
        surf = (1.0 - layers_height(cur_uv, ddx, ddy, layer)) * relief;
    }

    let after = surf - cur_depth;
    let before = prev_surf - prev_depth;
    let t = clamp(before / max(before - after, 1e-5), 0.0, 1.0);
    return mix(prev_uv, cur_uv, t) - uv;
}

const PARALLAX_NEAR_M: f32 = 18.0;
const PARALLAX_FAR_M: f32 = 30.0;

"#;

pub fn with_relief(src: &str) -> Option<String> {
    let call = "    let f = sample_frame(in.tex_pos, n, step_x, step_y, to_player_direction(shape.face), shape.shaped);\n";
    if src.matches(call).count() != 1 || src.matches("\nfn sample_frame(").count() != 1 {
        return None;
    }
    let marched = "    let relief_ddx = dpdx(in.tex_pos.xz);\n    let relief_ddy = dpdy(in.tex_pos.xz);\n    var relief_pos = in.tex_pos;\n    let relief = mat.parallax * (1.0 - smoothstep(PARALLAX_NEAR_M, PARALLAX_FAR_M, distance(in.world_pos, cam_pos())))\n        * select(1.0, 0.0, slope_deg >= mat.biplanar_start_deg);\n    if (relief > 0.0005) {\n        let shift = parallax_shift(in.tex_pos.xz, relief_ddx, relief_ddy, w, to_world_direction(normalize(cam_pos() - in.world_pos)), relief);\n        relief_pos = vec3<f32>(relief_pos.x + shift.x, relief_pos.y, relief_pos.z + shift.y);\n    }\n    let f = sample_frame(relief_pos, n, step_x, step_y, to_player_direction(shape.face), shape.shaped);\n";
    Some(src.replacen(call, marched, 1).replacen("\nfn sample_frame(", &format!("\n{RELIEF_WGSL}\nfn sample_frame("), 1))
}

pub static SHAPED_TWINS_BUILT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

pub fn twin_sources(source: String) -> (String, Option<String>) {
    match crate::renderer::ground_shape::unshaped(&source) {
        Some(plain) => (plain, Some(source)),
        None => (source, None),
    }
}

pub fn by_step(first: u32, len: u32, coarse_from: u32) -> impl Iterator<Item = (u32, u32, bool)> {
    let cut = coarse_from.clamp(first, first + len);
    [(first, cut - first, false), (cut, first + len - cut, true)].into_iter().filter(|p| p.1 > 0)
}

pub static OPAQUE_GROUND: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

pub static WORLD_GROUND: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

const POSED_VERTEX: &str = "@vertex fn vs_main(v: VIn) -> VOut {\n    var out: VOut;\n";
const WORLD_VERTEX: &str = "@vertex fn vs_main(w: VIn) -> VOut {\n    var out: VOut;\n    var v = w;\n    v.pos = to_player_space(w.pos);\n    v.norm = to_player_direction(w.norm);\n";

pub fn world_vertices(src: &str) -> Option<String> {
    (src.matches(POSED_VERTEX).count() == 1).then(|| src.replacen(POSED_VERTEX, WORLD_VERTEX, 1))
}

impl TerrainPipeline {

    pub fn at(&self, coarse: bool) -> &RenderPipeline {
        match (&self.shaped, coarse) {
            (Some(shaped), true) => shaped,
            _ => &self.pipeline,
        }
    }

    pub fn new(device: &Device, format: TextureFormat, uniform_layout: &BindGroupLayout) -> Self {
        let source = terrain_shader_built_mud(TerrainRole::Scene, GROUND_PAINTED.load(std::sync::atomic::Ordering::Relaxed), true);
        let source = with_relief(&source).unwrap_or(source);
        Self::build_from(device, format, uniform_layout, 1, crate::renderer::multiview::ViewMode::Mono, TerrainRole::Scene, None, source, "terrain_pipeline_desktop")
    }

    pub fn new_multisampled(

        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        samples: u32,
    ) -> Self {
        Self::new_with_view(device, format, uniform_layout, samples, crate::renderer::multiview::ViewMode::Mono)
    }

    pub fn new_multisampled_stereo(

        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        samples: u32,
    ) -> Self {
        Self::new_with_view(device, format, uniform_layout, samples, crate::renderer::multiview::ViewMode::Stereo)
    }

    fn new_with_view(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        samples: u32,
        view: crate::renderer::multiview::ViewMode,
    ) -> Self {
        Self::build(device, format, uniform_layout, samples, view, TerrainRole::Scene, None)
    }

    pub fn new_probe_reader(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        samples: u32,
        probe_layout: &BindGroupLayout,
    ) -> Self {
        Self::build(
            device,
            format,
            uniform_layout,
            samples,
            crate::renderer::multiview::ViewMode::Mono,
            TerrainRole::Read,
            Some(probe_layout),
        )
    }

    pub fn new_probe_reader_twins(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        samples: u32,
        probe_layout: &BindGroupLayout,
    ) -> [Self; 3] {
        [(true, false), (false, true), (true, true)].map(|(spotless, sun_baked)| {
            Self::new_probe_reader_twin(device, format, uniform_layout, samples, probe_layout, spotless, sun_baked)
        })
    }

    pub fn new_edited_probe_readers(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        samples: u32,
        probe_layout: &BindGroupLayout,
        edit: &str,
    ) -> Option<[Self; 4]> {
        let (edit, only) = edit.split_once('@').map_or((edit, None), |(e, c)| (e, Some(c)));
        let shipped = terrain_shader_for(TerrainRole::Read);
        let edited = crate::renderer::brush_pipeline::with_reader_edit(shipped.clone(), edit)?;
        let class = |name: &str| if only.map_or(true, |c| c == name) { edited.clone() } else { shipped.clone() };
        let baked = |s: String| crate::renderer::brush_pipeline::sun_reader_shader(s, crate::renderer::brush_pipeline::FaceSun::Baked);
        let spotless = crate::renderer::lights::without_spot_shadows;
        let build = |source: String, label: &str| {
            Self::build_from(
                device,
                format,
                uniform_layout,
                samples,
                crate::renderer::multiview::ViewMode::Mono,
                TerrainRole::Read,
                Some(probe_layout),
                source,
                label,
            )
        };
        Some([
            build(class("full"), "terrain_pipeline_read_edited"),
            build(spotless(class("full")), "terrain_pipeline_read_spotless_edited"),
            build(baked(class("baked")), "terrain_pipeline_read_baked_edited"),
            build(spotless(baked(class("baked"))), "terrain_pipeline_read_baked_spotless_edited"),
        ])
    }

    pub fn new_probe_reader_twin(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        samples: u32,
        probe_layout: &BindGroupLayout,
        spotless: bool,
        sun_baked: bool,
    ) -> Self {
        let mut source = terrain_shader_for(TerrainRole::Read);
        if sun_baked {
            source = crate::renderer::brush_pipeline::sun_reader_shader(source, crate::renderer::brush_pipeline::FaceSun::Baked);
        }
        if spotless {
            source = crate::renderer::lights::without_spot_shadows(source);
        }
        let label = match (spotless, sun_baked) {
            (false, false) => "terrain_pipeline_read",
            (true, false) => "terrain_pipeline_read_spotless",
            (false, true) => "terrain_pipeline_read_baked",
            (true, true) => "terrain_pipeline_read_baked_spotless",
        };
        Self::build_from(
            device,
            format,
            uniform_layout,
            samples,
            crate::renderer::multiview::ViewMode::Mono,
            TerrainRole::Read,
            Some(probe_layout),
            source,
            label,
        )
    }

    pub fn new_debug_reader(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        samples: u32,
        probe_layout: &BindGroupLayout,
        view: &str,
    ) -> Option<Self> {
        let source = ground_debug_shader(&terrain_shader_for(TerrainRole::Read), view)?;
        Some(Self::build_from(
            device,
            format,
            uniform_layout,
            samples,
            crate::renderer::multiview::ViewMode::Mono,
            TerrainRole::Read,
            Some(probe_layout),
            source,
            "terrain_pipeline_read_debug",
        ))
    }

    pub fn new_inlined(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        samples: u32,
        probe_layout: &BindGroupLayout,
        fixups: &crate::renderer::probe_fixup::ProbeFixups,
    ) -> [Self; 3] {
        let pass = with_inlined_layer_reads(terrain_shader_for(TerrainRole::ProbePass));
        let in_pass = |source: String, label: &str| {
            Self::build_from(
                device,
                crate::renderer::brush_pipeline::probe_pass::FORMAT,
                uniform_layout,
                1,
                crate::renderer::multiview::ViewMode::Mono,
                TerrainRole::ProbePass,
                Some(fixups.pass_layout()),
                source,
                label,
            )
        };
        [
            Self::build_from(
                device,
                format,
                uniform_layout,
                samples,
                crate::renderer::multiview::ViewMode::Mono,
                TerrainRole::Read,
                Some(probe_layout),
                with_inlined_layer_reads(terrain_shader_for(TerrainRole::Read)),
                "terrain_pipeline_read_inlined",
            ),
            in_pass(pass.clone(), "terrain_probe_pass_inlined"),
            in_pass(crate::renderer::lights::without_pool_maps(pass), "terrain_probe_pass_poolless_inlined"),
        ]
    }

    pub fn new_probe_pass(
        device: &Device,
        uniform_layout: &BindGroupLayout,
        fixups: &crate::renderer::probe_fixup::ProbeFixups,
    ) -> Self {
        Self::build(
            device,
            crate::renderer::brush_pipeline::probe_pass::FORMAT,
            uniform_layout,
            1,
            crate::renderer::multiview::ViewMode::Mono,
            TerrainRole::ProbePass,
            Some(fixups.pass_layout()),
        )
    }

    pub fn new_probe_pass_poolless(
        device: &Device,
        uniform_layout: &BindGroupLayout,
        fixups: &crate::renderer::probe_fixup::ProbeFixups,
    ) -> Self {
        Self::build_from(
            device,
            crate::renderer::brush_pipeline::probe_pass::FORMAT,
            uniform_layout,
            1,
            crate::renderer::multiview::ViewMode::Mono,
            TerrainRole::ProbePass,
            Some(fixups.pass_layout()),
            crate::renderer::lights::without_pool_maps(terrain_shader_for(TerrainRole::ProbePass)),
            "terrain_probe_pass_poolless",
        )
    }

    pub fn new_probe_pass_open(
        device: &Device,
        uniform_layout: &BindGroupLayout,
        fixups: &crate::renderer::probe_fixup::ProbeFixups,
        weather_layout: Option<&BindGroupLayout>,
    ) -> Option<Self> {
        let src = open_pass_shader()?;
        let (src, label) = match weather_layout {
            Some(_) => (crate::renderer::weather::with_weather(&src)?, "terrain_weather_probe_pass_open"),
            None => (src, "terrain_probe_pass_open"),
        };
        Some(Self::build_from_with(
            device,
            crate::renderer::brush_pipeline::probe_pass::FORMAT,
            uniform_layout,
            1,
            crate::renderer::multiview::ViewMode::Mono,
            TerrainRole::ProbePass,
            weather_layout,
            Some(fixups.pass_layout()),
            src,
            label,
        ))
    }

    pub fn new_probe_pass_puddle(
        device: &Device,
        uniform_layout: &BindGroupLayout,
        fixups: &crate::renderer::probe_fixup::ProbeFixups,
        weather_layout: &BindGroupLayout,
    ) -> Option<Self> {
        let src = crate::renderer::weather::with_weather(&puddle_pass_shader()?)?;
        Some(Self::build_from_with(
            device,
            crate::renderer::brush_pipeline::probe_pass::FORMAT,
            uniform_layout,
            1,
            crate::renderer::multiview::ViewMode::Mono,
            TerrainRole::ProbePass,
            Some(weather_layout),
            Some(fixups.pass_layout()),
            src,
            "terrain_weather_probe_pass_puddle",
        ))
    }

    pub fn new_probe_pass_nowhere(
        device: &Device,
        uniform_layout: &BindGroupLayout,
        fixups: &crate::renderer::probe_fixup::ProbeFixups,
        weather_layout: Option<&BindGroupLayout>,
    ) -> Option<Self> {
        let src = nowhere_pass_shader()?;
        let (src, label) = match weather_layout {
            Some(_) => (crate::renderer::weather::with_weather(&src)?, "terrain_weather_probe_pass_nowhere"),
            None => (src, "terrain_probe_pass_nowhere"),
        };
        Some(Self::build_from_with(
            device,
            crate::renderer::brush_pipeline::probe_pass::FORMAT,
            uniform_layout,
            1,
            crate::renderer::multiview::ViewMode::Mono,
            TerrainRole::ProbePass,
            weather_layout,
            Some(fixups.pass_layout()),
            src,
            label,
        ))
    }

    pub fn new_probe_pass_outdoors(
        device: &Device,
        uniform_layout: &BindGroupLayout,
        fixups: &crate::renderer::probe_fixup::ProbeFixups,
        weather_layout: Option<&BindGroupLayout>,
    ) -> Option<Self> {
        let src = outdoor_pass_shader()?;
        let (src, label) = match weather_layout {
            Some(_) => (crate::renderer::weather::with_weather(&src)?, "terrain_weather_probe_pass_outdoors"),
            None => (src, "terrain_probe_pass_outdoors"),
        };
        Some(Self::build_from_with(
            device,
            crate::renderer::brush_pipeline::probe_pass::FORMAT,
            uniform_layout,
            1,
            crate::renderer::multiview::ViewMode::Mono,
            TerrainRole::ProbePass,
            weather_layout,
            Some(fixups.pass_layout()),
            src,
            label,
        ))
    }

    pub fn new_probe_pass_with_cut(
        device: &Device,
        uniform_layout: &BindGroupLayout,
        fixups: &crate::renderer::probe_fixup::ProbeFixups,
        cut: &str,
    ) -> Option<Self> {
        let (label, edits) = PROBE_PASS_REGISTER_CUTS.iter().find(|(label, _)| *label == cut)?;
        let mut src = terrain_shader_for(TerrainRole::ProbePass);
        for (from, to) in edits.iter() {
            if !src.contains(from) {
                return None;
            }
            src = src.replacen(from, to, 1);
        }
        Some(Self::build_from(
            device,
            crate::renderer::brush_pipeline::probe_pass::FORMAT,
            uniform_layout,
            1,
            crate::renderer::multiview::ViewMode::Mono,
            TerrainRole::ProbePass,
            Some(fixups.pass_layout()),
            src,
            label,
        ))
    }

    pub fn log_probe_pass_register_cuts(
        device: &Device,
        uniform_layout: &BindGroupLayout,
        fixups: &crate::renderer::probe_fixup::ProbeFixups,
    ) {
        let base = terrain_shader_for(TerrainRole::ProbePass);
        for (label, edits) in PROBE_PASS_REGISTER_CUTS {
            let mut src = base.clone();
            if let Some((from, _)) = edits.iter().find(|(from, _)| !src.contains(from)) {
                log::warn!("register cut {label}: `{from}` is not in the shader");
                continue;
            }
            for (from, to) in edits.iter() {
                src = src.replacen(from, to, 1);
            }
            let _ = Self::build_from(
                device,
                crate::renderer::brush_pipeline::probe_pass::FORMAT,
                uniform_layout,
                1,
                crate::renderer::multiview::ViewMode::Mono,
                TerrainRole::ProbePass,
                Some(fixups.pass_layout()),
                src,
                label,
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_weather_twins(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        samples: u32,
        probe_layout: &BindGroupLayout,
        fixups: &crate::renderer::probe_fixup::ProbeFixups,
        weather_layout: &BindGroupLayout,
    ) -> TerrainWeatherTwins {
        let twin = |src: String| crate::renderer::weather::with_weather(&src).expect("the ground's weather twin no longer matches its shader");
        let read = terrain_shader_for(TerrainRole::Read);
        let baked = crate::renderer::brush_pipeline::sun_reader_shader(read.clone(), crate::renderer::brush_pipeline::FaceSun::Baked);
        let spotless = crate::renderer::lights::without_spot_shadows;
        let reader = |src: String, label: &str| {
            Self::build_from_with(
                device,
                format,
                uniform_layout,
                samples,
                crate::renderer::multiview::ViewMode::Mono,
                TerrainRole::Read,
                Some(weather_layout),
                Some(probe_layout),
                twin(src),
                label,
            )
        };
        let pass_src = terrain_shader_for(TerrainRole::ProbePass);
        let pass = |src: String, label: &str| {
            Self::build_from_with(
                device,
                crate::renderer::brush_pipeline::probe_pass::FORMAT,
                uniform_layout,
                1,
                crate::renderer::multiview::ViewMode::Mono,
                TerrainRole::ProbePass,
                Some(weather_layout),
                Some(fixups.pass_layout()),
                twin(src),
                label,
            )
        };
        TerrainWeatherTwins {
            readers: [
                reader(read.clone(), "terrain_weather_read"),
                reader(spotless(read), "terrain_weather_read_spotless"),
                reader(baked.clone(), "terrain_weather_read_baked"),
                reader(spotless(baked), "terrain_weather_read_baked_spotless"),
            ],
            pass: pass(pass_src.clone(), "terrain_weather_probe_pass"),
            pass_poolless: pass(crate::renderer::lights::without_pool_maps(pass_src), "terrain_weather_probe_pass_poolless"),
        }
    }

    fn build(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        samples: u32,
        view: crate::renderer::multiview::ViewMode,
        role: TerrainRole,
        group3: Option<&BindGroupLayout>,
    ) -> Self {
        let label = match role {
            TerrainRole::Scene => "terrain_pipeline",
            TerrainRole::Read => "terrain_pipeline_read",
            TerrainRole::ProbePass => "terrain_probe_pass",
        };
        Self::build_from(device, format, uniform_layout, samples, view, role, group3, terrain_shader_for(role), label)
    }

    #[allow(clippy::too_many_arguments)]
    fn build_from(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        samples: u32,
        view: crate::renderer::multiview::ViewMode,
        role: TerrainRole,
        group3: Option<&BindGroupLayout>,
        source: String,
        label: &str,
    ) -> Self {
        Self::build_from_with(device, format, uniform_layout, samples, view, role, None, group3, source, label)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn build_from_with(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        samples: u32,
        view: crate::renderer::multiview::ViewMode,
        role: TerrainRole,
        group2: Option<&BindGroupLayout>,
        group3: Option<&BindGroupLayout>,
        source: String,
        label: &str,
    ) -> Self {
        Self::build_faced(device, format, uniform_layout, samples, view, role, group2, group3, source, label, FrontFace::Ccw)
    }

    pub fn new_mirror(device: &Device, format: TextureFormat, uniform_layout: &BindGroupLayout) -> Self {
        let source = terrain_shader_for(TerrainRole::Scene);
        Self::build_faced(
            device,
            format,
            uniform_layout,
            1,
            crate::renderer::multiview::ViewMode::Mono,
            TerrainRole::Scene,
            None,
            None,
            source,
            "terrain_pipeline_mirror",
            FrontFace::Cw,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn build_faced(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        samples: u32,
        view: crate::renderer::multiview::ViewMode,
        role: TerrainRole,
        group2: Option<&BindGroupLayout>,
        group3: Option<&BindGroupLayout>,
        mut source: String,
        label: &str,
        front_face: FrontFace,
    ) -> Self {
        if WORLD_GROUND.load(std::sync::atomic::Ordering::Relaxed) {
            match world_vertices(&source) {
                Some(world) => source = world,

                None => log::error!("WORLD GROUND: the ground shader '{label}' does not open as expected; it reads its vertices as posed"),
            }
        }
        if role == TerrainRole::ProbePass && device.features().contains(wgpu::Features::SHADER_EARLY_DEPTH_TEST) {

            source = source.replacen("@fragment fn fs_main(", "@fragment @early_depth_test(force) fn fs_main(", 1);
        }
        let material_layout = material_bind_group_layout(device);

        let (plain, shaped) = twin_sources(source);
        let pipeline = Self::compile(device, format, uniform_layout, samples, view, role, group2, group3, plain, label, front_face, &material_layout);
        let shaped = shaped.map(|src| {
            SHAPED_TWINS_BUILT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Self::compile(device, format, uniform_layout, samples, view, role, group2, group3, src, &format!("{label}_shaped"), front_face, &material_layout)
        });
        Self { pipeline, shaped, material_layout }
    }

    #[allow(clippy::too_many_arguments)]
    fn compile(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        samples: u32,
        view: crate::renderer::multiview::ViewMode,
        role: TerrainRole,
        group2: Option<&BindGroupLayout>,
        group3: Option<&BindGroupLayout>,
        source: String,
        label: &str,
        front_face: FrontFace,
        material_layout: &BindGroupLayout,
    ) -> RenderPipeline {

        let shader = crate::renderer::shader_checks::audited_shader_module(device, ShaderModuleDescriptor {
            label: Some("terrain_shader"),
            source: ShaderSource::Wgsl(crate::renderer::shader_precision::for_device(device, view.shader(source)).into()),
        });
        let mut layouts = vec![Some(uniform_layout), Some(material_layout)];
        if group2.is_some() || group3.is_some() {
            layouts.push(group2);
        }
        if let Some(l) = group3 {
            layouts.push(Some(l));
        }
        let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("terrain_layout"),
            bind_group_layouts: &layouts,
            immediate_size: 0,
        });

        let probe_targets = crate::renderer::brush_pipeline::probe_pass::targets(false);

        let blend = if OPAQUE_GROUND.load(std::sync::atomic::Ordering::Relaxed) { None } else { Some(BlendState::ALPHA_BLENDING) };
        let scene_target = [Some(ColorTargetState { format, blend, write_mask: ColorWrites::ALL })];
        let targets: &[Option<ColorTargetState>] =
            if role == TerrainRole::ProbePass { &probe_targets } else { &scene_target };
        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(&layout),
            vertex: VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: PipelineCompilationOptions::default(),

                buffers: &[Some(SolidVertex::layout())],
            },
            fragment: Some(FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: PipelineCompilationOptions::default(),

                targets,
            }),
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                cull_mode: Some(Face::Back),
                front_face,
                polygon_mode: PolygonMode::Fill,
                ..Default::default()
            },
            depth_stencil: Some(DepthStencilState {
                format: TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(CompareFunction::Less),
                stencil: StencilState::default(),
                bias: DepthBiasState::default(),
            }),
            multisample: MultisampleState { count: samples, ..Default::default() },
            multiview_mask: view.mask(),
            cache: None,
        });
        pipeline
    }
}

pub struct TerrainWeatherTwins {

    pub readers: [TerrainPipeline; 4],
    pub pass: TerrainPipeline,
    pub pass_poolless: TerrainPipeline,
}

const ALL_OUTDOORS_OFF: &str = "const PROBE_ALL_OUTDOORS: bool = false;";
const ALL_OUTDOORS_ON: &str = "const PROBE_ALL_OUTDOORS: bool = true;";

const NOWHERE_OFF: &str = "const PROBE_NOWHERE: bool = false;";
const NOWHERE_ON: &str = "const PROBE_NOWHERE: bool = true;";

const OPEN_GROUND_OFF: &str = "const PROBE_OPEN_GROUND: bool = false;";
const OPEN_GROUND_ON: &str = "const PROBE_OPEN_GROUND: bool = true;";

pub(crate) fn open_pass_shader() -> Option<String> {
    let src = terrain_shader_for(TerrainRole::ProbePass);
    (src.matches(OPEN_GROUND_OFF).count() == 1).then(|| src.replacen(OPEN_GROUND_OFF, OPEN_GROUND_ON, 1))
}

const OPEN_COLOUR: &str = "        let open_col = probe_escape_colour(world_pos, open_dir, dir, roughness, open_lod);";
const OPEN_SKY_COLOUR: &str = "        let open_col = vec4<f32>(sky_reflection(open_dir, dir, open_lod), 1.0);";

pub(crate) fn puddle_pass_shader() -> Option<String> {
    let src = open_pass_shader()?;
    (src.matches(OPEN_COLOUR).count() == 1).then(|| src.replacen(OPEN_COLOUR, OPEN_SKY_COLOUR, 1))
}

pub(crate) fn nowhere_pass_shader() -> Option<String> {
    let src = terrain_shader_for(TerrainRole::ProbePass);
    (src.matches(NOWHERE_OFF).count() == 1).then(|| src.replacen(NOWHERE_OFF, NOWHERE_ON, 1))
}

pub(crate) fn outdoor_pass_shader() -> Option<String> {
    let src = terrain_shader_for(TerrainRole::ProbePass);
    if src.matches(ALL_OUTDOORS_OFF).count() != 1 {
        return None;
    }
    let src = src.replacen(ALL_OUTDOORS_OFF, ALL_OUTDOORS_ON, 1);

    let (from, to) = crate::renderer::brush_pipeline::DEDUP_OUTDOOR_SKY;
    Some(if src.matches(from).count() == 1 { src.replacen(from, to, 1) } else { src })
}

pub(crate) const PASS_ROUGH_SKIP: &str =
    "    if (clamp(rough, 0.04, 1.0) >= PROBE_LOBE_HEMISPHERICAL) {\n        return vec4<f32>(0.0);\n    }\n";

const PASS_ROUGH_SKIP_AT: &str = "    probe_fragment = in.clip;\n    return probe_env_for_pass(\n";
const PASS_ROUGH_SKIP_IN: &str = "    probe_fragment = in.clip;\n    if (clamp(rough, 0.04, 1.0) >= PROBE_LOBE_HEMISPHERICAL) {\n        return vec4<f32>(0.0);\n    }\n    return probe_env_for_pass(\n";

const PROBE_PASS_REGISTER_CUTS: &[(&str, &[(&str, &str)])] = &[
    ("terrain_cut_none", &[]),

    ("terrain_cut_rough_skip", &[(PASS_ROUGH_SKIP_AT, PASS_ROUGH_SKIP_IN)]),

    ("terrain_cut_all_outdoors", &[(ALL_OUTDOORS_OFF, ALL_OUTDOORS_ON)]),

    ("terrain_cut_nowhere", &[(NOWHERE_OFF, NOWHERE_ON)]),

    ("terrain_cut_open", &[(OPEN_GROUND_OFF, OPEN_GROUND_ON)]),

    (
        "terrain_cut_gate",
        &[(
            "rgb * seen / a\n            * capsule_glass_beam(p, camera.capsules[glow_i * 2 + 1].xyz);",
            "rgb * seen / a;",
        )],
    ),

    (
        "terrain_cut_relight",
        &[(crate::renderer::lights::SURFACE_RELIT_READ, ""), (crate::renderer::lights::SURFACE_RELIT_CALL, "")],
    ),

    ("terrain_cut_dedup_all", crate::renderer::brush_pipeline::DEDUP_EDITS),

    ("terrain_cut_dedup_sky", &[crate::renderer::brush_pipeline::DEDUP_OUTDOOR_SKY]),
    ("terrain_cut_dedup_portals", &[crate::renderer::brush_pipeline::DEDUP_PORTAL_SAMPLES]),
    ("terrain_cut_dedup_near_far", &[crate::renderer::brush_pipeline::DEDUP_NEAR_FAR]),

    (
        "terrain_cut_dedup_sky_near_far",
        &[crate::renderer::brush_pipeline::DEDUP_OUTDOOR_SKY, crate::renderer::brush_pipeline::DEDUP_NEAR_FAR],
    ),

    ("terrain_cut_escape_colour", &[("    if (h.escaped) {\n        col = probe_escape_colour(", "    if (false) {\n        col = probe_escape_colour(")]),
];

fn terrain_lights_block(options: crate::renderer::lights::LightsBlockOptions) -> String {
    crate::renderer::lights::wgsl_lights_block_with(
        0,
        1,
        crate::renderer::lights::LightsBlockOptions { cull_range_first: true, long_step_from_plane: true, open_fallback: true, ..options },
    )
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum TerrainRole {

    Scene,

    Read,

    ProbePass,
}

pub static GROUND_PAINTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Whether the level's ground is built with its deform map read (mud prints,
/// wet zones, puddles). Without a stamped map every deform term is zero, so
/// the ground is built without that code ([`deformless`]), which would
/// otherwise cost by its presence in every ground pixel.
pub static GROUND_MUDDY: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// The deform map as the ground and the water read it: per texel the height
/// and the wetness in their 0..255 steps (128 flat), then the height's change
/// across the texel each way, `h[x+1] - h[x-1]` and `h[z+1] - h[z-1]`,
/// wrapping as the sampler does. Bilinear filtering is linear, so the filtered
/// difference is the difference of the filtered heights one texel either side:
/// the slope the ground took from four more reads, in the one it already
/// makes. Half floats hold every one of those integers exactly.
pub const DEFORM_FORMAT: TextureFormat = TextureFormat::Rgba16Float;
pub const DEFORM_TEXEL_BYTES: u32 = 8;
pub const DEFORM_FLAT_TEXEL: [u8; 2] = [128, 0];

/// A small integer as a half float, exactly (|n| <= 2048).
fn half_of_int(n: i32) -> u16 {
    if n == 0 {
        return 0;
    }
    let sign = if n < 0 { 0x8000 } else { 0 };
    let m = n.unsigned_abs();
    let e = 31 - m.leading_zeros();
    sign | (((e + 15) as u16) << 10) | (((m << (10 - e)) & 0x3ff) as u16)
}

/// Rows `from..to` of the deform map's GPU texels from its `[height, wet]`
/// bytes, `width` x `height` row-major.
pub fn deform_rows(bytes: &[u8], width: u32, height: u32, from: u32, to: u32) -> Vec<u16> {
    let (w, h) = (width as usize, height as usize);
    let at = |x: usize, z: usize| bytes[2 * (z * w + x)] as i32;
    let mut out = Vec::with_capacity(4 * w * (to - from) as usize);
    for z in from as usize..to as usize {
        for x in 0..w {
            let i = 2 * (z * w + x);
            let dx = at((x + 1) % w, z) - at((x + w - 1) % w, z);
            let dz = at(x, (z + 1) % h) - at(x, (z + h - 1) % h);
            out.extend([half_of_int(bytes[i] as i32), half_of_int(bytes[i + 1] as i32), half_of_int(dx), half_of_int(dz)]);
        }
    }
    out
}

/// The rows to send after `from..to` changed: their neighbours either way
/// carry slopes that read them, the first and last rows each other's.
pub fn deform_rows_touched(from: u32, to: u32, height: u32) -> Vec<(u32, u32)> {
    let mut runs = vec![(from.saturating_sub(1), (to + 1).min(height))];
    if from == 0 && height > 1 {
        runs.push((height - 1, height));
    }
    if to >= height && height > 1 {
        runs.push((0, 1));
    }
    runs
}

pub(crate) const DEFORM_VERTEX: &str = "    let deform = textureSampleLevel(deform_tex, layer_samp, v.uv2, 0.0).r * DEFORM_UNIT;\n    pos.y = pos.y + (deform - DEFORM_FLAT) * 2.0 * mat.deform_metres * mud_weight_v(v.uv2);\n";
pub(crate) const DEFORM_FRAGMENT_FROM: &str = "    let mudw = mud_weight_f(w);\n";
pub(crate) const DEFORM_FRAGMENT_TO: &str = "    rough_map = mix(rough_map, 0.08, pool);\n";

/// The ground without its deform map: a flat map moves no vertex, and sinks,
/// wets, tilts and pools nothing.
pub fn deformless(src: &str) -> Option<String> {
    if src.matches(DEFORM_VERTEX).count() != 1 || src.matches(DEFORM_FRAGMENT_FROM).count() != 1 || src.matches(DEFORM_FRAGMENT_TO).count() != 1 {
        return None;
    }
    let mut s = src.replacen(DEFORM_VERTEX, "", 1);
    let from = s.find(DEFORM_FRAGMENT_FROM)?;
    let to = s.find(DEFORM_FRAGMENT_TO)? + DEFORM_FRAGMENT_TO.len();
    s.replace_range(from..to, "");
    Some(s)
}

static GROUND_BUILD_PINNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn expect_painted_ground(painted: bool) {
    if !GROUND_BUILD_PINNED.load(std::sync::atomic::Ordering::Relaxed) {
        GROUND_PAINTED.store(painted, std::sync::atomic::Ordering::Relaxed);
    }
}

pub fn pin_ground_build(painted: bool) {
    GROUND_PAINTED.store(painted, std::sync::atomic::Ordering::Relaxed);
    GROUND_BUILD_PINNED.store(true, std::sync::atomic::Ordering::Relaxed);
}

pub fn ground_build_wanted(built_painted: bool, level_painted: bool, pinned: bool) -> bool {
    if pinned { built_painted } else { level_painted }
}

pub fn ground_build_moves(level_painted: bool, level_muddy: bool) -> bool {
    use std::sync::atomic::Ordering::Relaxed;
    let built = GROUND_PAINTED.load(Relaxed);
    let want = ground_build_wanted(built, level_painted, GROUND_BUILD_PINNED.load(Relaxed));
    GROUND_PAINTED.store(want, Relaxed);
    let was_muddy = GROUND_MUDDY.swap(level_muddy, Relaxed);
    want != built || was_muddy != level_muddy
}

pub(crate) const PAINTED_FROM: &str = "    return select(derived, authored, mat.use_splat > 0.5);\n";
pub(crate) const PAINTED_TO: &str = "    return authored;\n";

pub(crate) fn painted_shader(src: &str) -> Option<String> {
    (src.matches(PAINTED_FROM).count() == 1).then(|| src.replacen(PAINTED_FROM, PAINTED_TO, 1))
}

const GROUND_ANSWER: &str = "    return vec4<f32>(tonemap(in.col.rgb * lit), 1.0);\n";

pub const GROUND_DEBUG_VIEWS: &[(&str, &str)] = &[
    ("albedo", "albedo"),
    ("plane", "select(select(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(0.0, 1.0, 0.0), f.detail_axis == 1u), vec3<f32>(1.0, 0.0, 0.0), f.detail_axis == 0u) * (2.0 * f.blend - 1.0 + 0.25)"),
    (
        "stretch",
        "select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(dbg_face[f.detail_axis]), dbg_face[f.detail_axis] > 0.25)",
    ),
    ("facet", "vec3<f32>(abs(degrees(acos(clamp(dbg_face.y, 0.0, 1.0))) - slope_deg) / 45.0)"),
    (
        "lod",
        "vec3<f32>(0.5 * log2(max(max(dot(f.detail_ddx, f.detail_ddx), dot(f.detail_ddy, f.detail_ddy)), 1e-12) * 512.0 * 512.0) / 10.0)",
    ),
    ("rough", "vec3<f32>(rough)"),
    ("weights", "w.rgb"),
    ("sky", "vec3<f32>(sky_vis)"),
    ("sun", "vec3<f32>(receiver_sun_mask)"),
    ("normal", "to_world_direction(shaded_n) * 0.5 + 0.5"),
    ("shape", "to_world_direction(n) * 0.5 + 0.5"),
    ("layer0", "layer_colour_at(0, f)"),
    ("layer1", "layer_colour_at(1, f)"),
    ("layer2", "layer_colour_at(2, f)"),
    ("ao", "vec3<f32>(ao_map)"),
];

pub fn ground_debug_shader(src: &str, view: &str) -> Option<String> {
    if let Some(cut) = view.strip_prefix("cut:") {
        let (_, edits) = crate::renderer::ground_cuts::GROUND_CUTS.iter().find(|(name, _)| *name == cut)?;
        return crate::renderer::ground_cuts::with_cut(src, edits);
    }
    // `scene:<cut>`: one of the brushes' scene cuts
    // (`brush_pipeline::SCENE_REGISTER_CUTS`), where its lines are the
    // ground's too -- the light loops both readers share.
    if let Some(cut) = view.strip_prefix("scene:") {
        let (_, edits) = crate::renderer::brush_pipeline::SCENE_REGISTER_CUTS.iter().find(|(name, _)| *name == cut)?;
        return crate::renderer::ground_cuts::with_cut(src, edits);
    }
    if let Some(edits) = view.strip_prefix("edit:") {
        let edits: Vec<(&str, &str)> = edits.split(";;").map(|e| e.split_once("=>")).collect::<Option<_>>()?;
        return crate::renderer::ground_cuts::with_cut(src, &edits);
    }
    let (_, colour) = GROUND_DEBUG_VIEWS.iter().find(|(name, _)| *name == view)?;
    (src.matches(GROUND_ANSWER).count() == 1).then(|| {
        src.replacen(
            GROUND_ANSWER,
            &format!(
                "    let dbg_face = abs(normalize(cross(dpdx(in.tex_pos), dpdy(in.tex_pos))));\n    return vec4<f32>(clamp({colour}, vec3<f32>(0.0), vec3<f32>(1.0)) + 0.0 * lit, 1.0);\n"
            ),
            1,
        )
    })
}

pub fn terrain_shader_for(role: TerrainRole) -> String {
    terrain_shader_built(role, GROUND_PAINTED.load(std::sync::atomic::Ordering::Relaxed))
}

pub fn terrain_shader_built(role: TerrainRole, painted: bool) -> String {
    terrain_shader_built_mud(role, painted, GROUND_MUDDY.load(std::sync::atomic::Ordering::Relaxed))
}

pub fn terrain_shader_built_mud(role: TerrainRole, painted: bool, muddy: bool) -> String {
    let src = terrain_shader();
    let src = match muddy {
        true => src,
        false => deformless(&src).expect("the ground's deform reads are not where `deformless` takes them out"),
    };
    let src = match painted {
        true => painted_shader(&src).unwrap_or(src),
        false => src,
    };
    if role == TerrainRole::Scene {
        return src;
    }
    let plain_lights = terrain_lights_block(crate::renderer::lights::LightsBlockOptions::default());
    assert!(src.contains(&plain_lights), "the terrain shader no longer embeds the lights block as generated");
    let shade = "    let lit = shade_material_env(\n";
    assert!(src.contains(shade), "the terrain shader no longer shades through `shade_material_env`");
    match role {
        TerrainRole::Scene => unreachable!(),
        TerrainRole::Read => {
            let read_lights = terrain_lights_block(crate::renderer::lights::LightsBlockOptions {
                probe_from_pass: true,
                ..Default::default()
            });
            let src = src.replacen(&plain_lights, &read_lights, 1).replacen(
                shade,
                &format!(
                    "    // From the half-resolution probe pass, as the brushes read it. See\n    // `brush_pipeline::probe_pass`.\n    let probe_pass_tolerance = max(4.0 * fwidth(in.clip.z), 1e-6);\n    probe_env_given = probe_pass_upsample(in.clip.xy, in.clip.z, probe_pass_tolerance, 0.0);\n{shade}"
                ),
                1,
            );

            format!(
                "{src}{}{}",
                crate::renderer::brush_pipeline::probe_pass::READER_WGSL,
                crate::renderer::brush_pipeline::probe_pass::FACE_CODE_WGSL,
            )
        }
        TerrainRole::ProbePass => {
            let pass_lights = terrain_lights_block(crate::renderer::lights::LightsBlockOptions {
                defer_secondary: true,
                ..Default::default()
            });
            let src = src.replacen(&plain_lights, &pass_lights, 1);
            let start = src.find(shade).expect("checked above");
            let end = start + src[start..].find("\n}\n").expect("the fragment stage ends");

            format!(
                "{}    probe_fragment = in.clip;\n    return probe_env_for_pass(\n        in.world_pos, shaded_n, rough, clamp(ao_map, 0.0, 1.0), sky_vis, vec3<f32>(0.0), in.world_pos, n,\n    );{}",
                &src[..start],
                &src[end..],
            )
        }
    }
}

pub fn material_bind_group_layout(device: &Device) -> BindGroupLayout {
    device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("terrain_material_layout"),
        entries: &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::D2Array,
                    multisampled: false,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::VERTEX_FRAGMENT,
                ty: BindingType::Sampler(SamplerBindingType::Filtering),
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 10,
                visibility: ShaderStages::VERTEX_FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 3,
                visibility: ShaderStages::VERTEX_FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },

            BindGroupLayoutEntry {
                binding: 5,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::D2Array,
                    multisampled: false,
                },
                count: None,
            },

            BindGroupLayoutEntry {
                binding: 6,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::D2Array,
                    multisampled: false,
                },
                count: None,
            },

            BindGroupLayoutEntry {
                binding: 7,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::D2Array,
                    multisampled: false,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 8,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::D2Array,
                    multisampled: false,
                },
                count: None,
            },

            BindGroupLayoutEntry {
                binding: 4,
                visibility: ShaderStages::VERTEX_FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },

            BindGroupLayoutEntry {
                binding: 9,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: false },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
        ],
    })
}

fn terrain_shader() -> String {
    format!(
        r#"
@group(1) @binding(0) var layer_tex: texture_2d_array<f32>;
@group(1) @binding(1) var layer_samp: sampler;
@group(1) @binding(2) var macro_tex: texture_2d<f32>;

struct Material {{
    repeat: vec4<f32>,
    slope_start_deg: f32,
    slope_end_deg: f32,
    height_start: f32,
    height_end: f32,
    macro_repeat: f32,
    macro_strength: f32,
    biplanar_start_deg: f32,
    use_splat: f32,
    normal_strength: f32,
    wet_line: f32,
    wet_band: f32,
    sea_line: f32,
    porosity: vec4<f32>,
    film: vec4<f32>,
    shape: vec4<f32>,
    parallax: f32,
    mud_layer: f32,
    deform_metres: f32,
    deform_texel: f32,
    wet_dark: f32,
}}
@group(1) @binding(3) var<uniform> mat: Material;
@group(1) @binding(4) var splat_tex: texture_2d<f32>;
@group(1) @binding(10) var deform_tex: texture_2d<f32>;

const DEFORM_FLAT: f32 = 128.0 / 255.0;
const DEFORM_UNIT: f32 = 1.0 / 255.0;
const DEFORM_POOL_FROM: f32 = 0.35;
const DEFORM_POOL_TO: f32 = 0.7;
const DEFORM_NORMAL: f32 = 1.0;

fn mud_weight_v(uv: vec2<f32>) -> f32 {{
    if (mat.mud_layer < -0.5) {{
        return 0.0;
    }}
    if (mat.use_splat > 0.5) {{
        let s = textureSampleLevel(splat_tex, layer_samp, uv, 0.0);
        let total = max(s.x + s.y + s.z + s.w, 0.001);
        let w = select(select(select(s.x, s.y, mat.mud_layer > 0.5), s.z, mat.mud_layer > 1.5), s.w, mat.mud_layer > 2.5);
        return clamp(w / total, 0.0, 1.0);
    }}
    return select(0.0, 1.0, mat.mud_layer < 0.5);
}}

fn mud_weight_f(w: vec4<f32>) -> f32 {{
    if (mat.mud_layer < -0.5) {{
        return 0.0;
    }}
    return select(select(select(w.x, w.y, mat.mud_layer > 0.5), w.z, mat.mud_layer > 1.5), w.w, mat.mud_layer > 2.5);
}}
@group(1) @binding(5) var normal_tex: texture_2d_array<f32>;
@group(1) @binding(7) var rough_tex: texture_2d_array<f32>;
@group(1) @binding(8) var ao_tex: texture_2d_array<f32>;

@group(1) @binding(6) var sky_occ_tex: texture_2d_array<f32>;

{lights_block}
{biplanar_block}
{whiteout_block}
{shape_block}

struct VIn  {{ @location(0) pos: vec3<f32>, @location(1) norm: vec3<f32>, @location(2) col: vec4<f32>, @location(3) uv2: vec2<f32> }}
struct VOut {{
    @builtin(position) @invariant clip: vec4<f32>,
    @location(0) col: vec4<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) world_pos: vec3<f32>,

    @location(4) tex_pos: vec3<f32>,

    @location(3) uv: vec2<f32>,

    @location(5) @interpolate(flat) corner: vec2<f32>,
}}

@vertex fn vs_main(v: VIn) -> VOut {{
    var out: VOut;
    var pos = v.pos;
    let deform = textureSampleLevel(deform_tex, layer_samp, v.uv2, 0.0).r * DEFORM_UNIT;
    pos.y = pos.y + (deform - DEFORM_FLAT) * 2.0 * mat.deform_metres * mud_weight_v(v.uv2);
    out.clip      = cam_view_proj() * vec4<f32>(pos, 1.0);
    out.col       = v.col;
    out.normal    = v.norm;
    out.world_pos = pos;
    out.tex_pos   = to_world_space(pos);
    out.uv        = v.uv2;
    out.corner    = v.uv2;
    return out;
}}

fn sample_planar(layer: i32, world: vec3<f32>, repeat: f32) -> vec3<f32> {{
    let uv = world.xz / max(repeat, 0.001);
    return textureSample(layer_tex, layer_samp, uv, layer).rgb;
}}

fn sample_biplanar(layer: i32, world: vec3<f32>, n: vec3<f32>, repeat: f32) -> vec3<f32> {{
    let b = biplanar_axes(world, n, repeat);
    let c_major = textureSample(layer_tex, layer_samp, b.uv_major, layer).rgb;
    let c_minor = textureSample(layer_tex, layer_samp, b.uv_minor, layer).rgb;
    return mix(c_minor, c_major, b.w);
}}

fn layer_normal(layer: i32, world: vec3<f32>, n: vec3<f32>, repeat: f32) -> vec3<f32> {{
    let uv = world.xz / max(repeat, 0.001);
    let packed = textureSample(normal_tex, layer_samp, uv, layer).rgb;

    var tn = packed * 2.0 - 1.0;
    tn = vec3<f32>(tn.xy * mat.normal_strength, tn.z);

    return normalize(whiteout(1u, world_planar_bend(tn), n));
}}

fn world_planar_bend(tn: vec3<f32>) -> vec3<f32> {{
    let p = to_player_direction(vec3<f32>(tn.x, 0.0, -tn.y));
    return vec3<f32>(p.x, -p.z, tn.z);
}}

fn layer_colour(layer: i32, world: vec3<f32>, n: vec3<f32>, slope_deg: f32) -> vec3<f32> {{
    let repeat = mat.repeat[layer];
    if (slope_deg < mat.biplanar_start_deg) {{
        return sample_planar(layer, world, repeat);
    }}
    return sample_biplanar(layer, world, n, repeat);
}}

struct SampleFrame {{
    minor_uv: vec2<f32>,
    minor_ddx: vec2<f32>,
    minor_ddy: vec2<f32>,

    blend: f32,

    second: f32,

    detail_uv: vec2<f32>,
    detail_ddx: vec2<f32>,
    detail_ddy: vec2<f32>,
    detail_axis: u32,

    macro_value: f32,

    fold: f32,
    minor_axis: u32,
}}

const CLIFF_WARP_M: f32 = 2.5;

const CLIFF_BAND_WARP_M: f32 = 3.0;
const CLIFF_BAND_SCALE_M: f32 = 7.0;

const GROUND_MACRO_MAP: bool = false;

const PLANE_TIE: f32 = 0.57735;

const PLANE_ALONE: f32 = 0.999;

const FACE_DIFFERS: f32 = 0.9703;
const FACE_BAND: f32 = 41.3;

const MACRO_NEUTRAL: f32 = 0.5019608;

fn sample_frame(world: vec3<f32>, n: vec3<f32>, step_x: vec3<f32>, step_y: vec3<f32>, shape_face: vec3<f32>, shaped: bool) -> SampleFrame {{
    var f: SampleFrame;
    let planar_uv  = world.xz;

    let world_ddx = dpdx(world);
    let world_ddy = dpdy(world);
    let planar_ddx = world_ddx.xz;
    let planar_ddy = world_ddy.xz;

    let across = cross(step_x, step_y);
    let along = dot(across, n);
    let across_len = inverseSqrt(max(dot(across, across), 1e-30));
    let face = select(across * (sign(along) * across_len), shape_face, shaped);

    let steep = min(n.y, face.y) <= cos(radians(mat.biplanar_start_deg));

    let agree = clamp((select(abs(along) * across_len, abs(dot(shape_face, n)), shaped) - FACE_DIFFERS) * FACE_BAND, 0.0001, 1.0);
    let pn = to_world_direction(normalize(mix(face, n, agree)));

    let b = biplanar_axes(world, pn, 1.0);
    let a = abs(pn);
    let ma = max(a.x, max(a.y, a.z));
    let me = a.x + a.y + a.z - ma - min(a.x, min(a.y, a.z));

    let lead = ma - PLANE_TIE;
    let share = (lead + 0.0001) / (lead + max(me - PLANE_TIE, 0.0) + 0.0001);
    let major_ddx = select(select(world_ddx.xy, world_ddx.xz, b.axis_major == 1u), world_ddx.zy, b.axis_major == 0u);
    let major_ddy = select(select(world_ddy.xy, world_ddy.xz, b.axis_major == 1u), world_ddy.zy, b.axis_major == 0u);
    f.minor_uv  = b.uv_minor;
    f.minor_ddx = select(select(world_ddx.xy, world_ddx.xz, b.axis_minor == 1u), world_ddx.zy, b.axis_minor == 0u);
    f.minor_ddy = select(select(world_ddy.xy, world_ddy.xz, b.axis_minor == 1u), world_ddy.zy, b.axis_minor == 0u);
    f.blend     = select(1.0, share, steep);
    f.detail_uv   = select(planar_uv, b.uv_major, steep);
    f.detail_ddx  = select(planar_ddx, major_ddx, steep);
    f.detail_ddy  = select(planar_ddy, major_ddy, steep);
    f.detail_axis = select(1u, b.axis_major, steep);
    f.minor_axis  = b.axis_minor;
    f.fold        = select(1.0, agree, steep);

    f.macro_value = MACRO_NEUTRAL;
    if (GROUND_MACRO_MAP) {{
        let mr = max(mat.macro_repeat, 0.001);
        f.macro_value = textureSampleGrad(macro_tex, layer_samp, f.detail_uv / mr, f.detail_ddx / mr, f.detail_ddy / mr).r;
    }}

    f.second = select(0.0, (0.3 + 0.3 * f.macro_value) * clamp(4.0 * f.blend - 2.0, 0.0, 1.0), f.detail_axis != 1u);
    var warp = select(vec2<f32>(0.0), vec2<f32>(f.macro_value - 0.5, 0.4 - 0.8 * f.macro_value) * CLIFF_WARP_M, steep);

    if (steep) {{
        let ms = 1.0 / CLIFF_BAND_SCALE_M;
        var m2 = MACRO_NEUTRAL;
        if (GROUND_MACRO_MAP) {{
            m2 = textureSampleGrad(macro_tex, layer_samp, f.detail_uv * ms + vec2<f32>(0.31, 0.57), f.detail_ddx * ms, f.detail_ddy * ms).r;
        }}
        warp = warp + vec2<f32>(0.35, 1.0) * (m2 - 0.5) * CLIFF_BAND_WARP_M;
    }}

    f.minor_uv  = f.minor_uv + select(warp, vec2<f32>(0.0), f.minor_axis == 1u);
    f.detail_uv = f.detail_uv + select(warp, vec2<f32>(0.0), f.detail_axis == 1u);
    return f;
}}

fn layer_colour_at(layer: i32, f: SampleFrame) -> vec3<f32> {{
    let r = max(mat.repeat[layer], 0.001);

    var c = textureSampleGrad(layer_tex, layer_samp, f.detail_uv / r, layer, f.detail_ddx / r, f.detail_ddy / r).rgb;

    if (f.second > 0.0) {{
        let rot = mat2x2<f32>(0.8, 0.6, -0.6, 0.8);
        let s2 = 1.0 / (r * 2.3);
        c = mix(
            c,
            textureSampleGrad(
                layer_tex, layer_samp, rot * f.detail_uv * s2 + vec2<f32>(0.37, 0.11), layer,
                rot * f.detail_ddx * s2, rot * f.detail_ddy * s2,
            ).rgb,
            f.second,
        );
    }}

    if (f.blend < PLANE_ALONE) {{
        c = mix(
            textureSampleGrad(layer_tex, layer_samp, f.minor_uv / r, layer, f.minor_ddx / r, f.minor_ddy / r).rgb,
            c,
            f.blend,
        );
    }}
    return c;
}}

fn layer_rough_at(layer: i32, f: SampleFrame) -> f32 {{
    let r = max(mat.repeat[layer], 0.001);
    var v = textureSampleGrad(
        rough_tex, layer_samp, f.detail_uv / r, layer, f.detail_ddx / r, f.detail_ddy / r,
    ).r;
    if (f.blend < PLANE_ALONE) {{
        v = mix(textureSampleGrad(rough_tex, layer_samp, f.minor_uv / r, layer, f.minor_ddx / r, f.minor_ddy / r).r, v, f.blend);
    }}
    return v;
}}

fn layer_ao_at(layer: i32, f: SampleFrame) -> f32 {{
    let r = max(mat.repeat[layer], 0.001);
    var v = textureSampleGrad(
        ao_tex, layer_samp, f.detail_uv / r, layer, f.detail_ddx / r, f.detail_ddy / r,
    ).r;
    if (f.blend < PLANE_ALONE) {{
        v = mix(textureSampleGrad(ao_tex, layer_samp, f.minor_uv / r, layer, f.minor_ddx / r, f.minor_ddy / r).r, v, f.blend);
    }}
    return v;
}}

fn fold_bumps(axis: u32, tn: vec3<f32>, n: vec3<f32>, keep: f32) -> vec3<f32> {{
    let z = abs(tn.z);
    if (axis == 1u) {{
        let g = mix(z, 1.0, keep);
        return vec3<f32>(tn.x + g * n.x, z * n.y, g * n.z - tn.y);
    }}
    return n * z + select(vec3<f32>(tn.x, -tn.y, 0.0), vec3<f32>(0.0, -tn.y, tn.x), axis == 0u);
}}

fn layer_normal_at(layer: i32, n: vec3<f32>, f: SampleFrame) -> vec3<f32> {{
    let r = max(mat.repeat[layer], 0.001);
    let packed = textureSampleGrad(
        normal_tex, layer_samp, f.detail_uv / r, layer, f.detail_ddx / r, f.detail_ddy / r,
    ).rgb;

    var tn = packed * 2.0 - 1.0;
    tn = vec3<f32>(tn.xy * mat.normal_strength, tn.z);

    if (f.detail_axis == 1u && f.blend >= PLANE_ALONE) {{
        return fold_bumps(1u, world_planar_bend(tn), n, f.fold);
    }}

    if (f.second > 0.0) {{
        let rot = mat2x2<f32>(0.8, 0.6, -0.6, 0.8);
        let s2 = 1.0 / (r * 2.3);
        let p2 = textureSampleGrad(
            normal_tex, layer_samp, rot * f.detail_uv * s2 + vec2<f32>(0.37, 0.11), layer,
            rot * f.detail_ddx * s2, rot * f.detail_ddy * s2,
        ).rgb;
        var t2 = p2 * 2.0 - 1.0;
        t2 = vec3<f32>(transpose(rot) * (t2.xy * mat.normal_strength), t2.z);
        tn = vec3<f32>(mix(tn.xy, t2.xy, f.second) * (1.0 + 0.667 * f.second), mix(tn.z, t2.z, f.second));
    }}

    let nw = to_world_direction(n);
    var bent = fold_bumps(f.detail_axis, tn, nw, f.fold);

    if (f.blend < PLANE_ALONE) {{
        let pm = textureSampleGrad(
            normal_tex, layer_samp, f.minor_uv / r, layer, f.minor_ddx / r, f.minor_ddy / r,
        ).rgb;
        var tm = pm * 2.0 - 1.0;
        tm = vec3<f32>(tm.xy * mat.normal_strength, tm.z);
        bent = mix(fold_bumps(f.minor_axis, tm, nw, f.fold), bent, f.blend);
    }}
    return to_player_direction(bent);
}}

const WEIGHT_EPS: f32 = 0.004;

const TERRAIN_DETAIL_FADE: f32 = 8.0;

fn layer_weights(uv: vec2<f32>, world_y: f32, slope_deg: f32) -> vec4<f32> {{

    let authored_raw = textureSample(splat_tex, layer_samp, uv);
    let total = authored_raw.r + authored_raw.g + authored_raw.b + authored_raw.a;
    let authored = authored_raw / max(total, 0.001);

    let rock_w = smoothstep(mat.slope_start_deg, mat.slope_end_deg, slope_deg);
    var derived = vec4<f32>(1.0 - rock_w, rock_w, 0.0, 0.0);
    if (mat.height_end > mat.height_start) {{
        let high_w = smoothstep(mat.height_start, mat.height_end, world_y);
        derived = vec4<f32>(derived.x * (1.0 - high_w), derived.y * (1.0 - high_w), high_w, 0.0);
    }}

    return select(derived, authored, mat.use_splat > 0.5);
}}

@fragment fn fs_main(in: VOut) -> @location(0) vec4<f32> {{

    let step_x = dpdx(in.world_pos);
    let step_y = dpdy(in.world_pos);
    pixel_footprint = sqrt(length(step_x) * length(step_y));

    let shape = ground_shape_at(in.uv, in.corner, in.col.a);
    let n = normalize(select(in.normal, to_player_direction(shape.n), shape.shaped));
    set_pixel_long_step(n, normalize(cam_pos() - in.world_pos));

    let slope_deg = degrees(acos(clamp(n.y, -1.0, 1.0)));

    let w = layer_weights(in.uv, in.tex_pos.y, slope_deg);

    let f = sample_frame(in.tex_pos, n, step_x, step_y, to_player_direction(shape.face), shape.shaped);

    let detail_far = camera.post_params.z;
    let detail = select(1.0, 1.0 - smoothstep(detail_far, detail_far + TERRAIN_DETAIL_FADE, distance(in.world_pos, cam_pos())), detail_far > 0.0);

{layer_reads}
    shaded_n = normalize(select(n, shaded_n, length(shaded_n) > 0.0001));


    let mudw = mud_weight_f(w);
    let deform = textureSampleLevel(deform_tex, layer_samp, in.uv, 0.0) * DEFORM_UNIT;
    let sunk = max(DEFORM_FLAT - deform.r, 0.0) / DEFORM_FLAT * mudw;
    let wet_zone = deform.g * mudw;
    if (mudw > 0.001) {{
        let slope = deform.ba * (2.0 * mat.deform_metres / max(2.0 * mat.deform_texel, 1e-4)) * mudw * DEFORM_NORMAL;
        let bent = to_player_direction(normalize(vec3<f32>(-slope.x, 1.0, -slope.y)));
        shaded_n = normalize(shaded_n + bent - to_player_direction(vec3<f32>(0.0, 1.0, 0.0)));
    }}
    let wet = clamp(max(wet_zone, sunk), 0.0, 1.0);
    let pool = smoothstep(DEFORM_POOL_FROM, DEFORM_POOL_TO, sunk);
    albedo = albedo * (1.0 - mat.wet_dark * wet) * (1.0 - 0.35 * pool);
    rough_map = rough_map * (1.0 - 0.6 * wet);
    rough_map = mix(rough_map, 0.08, pool);
    let rough = specular_aa_roughness(clamp(rough_map, 0.0, 1.0), dpdx(shaded_n), dpdy(shaded_n));

    let m = f.macro_value;
    albedo = albedo * (1.0 + (m - 0.5) * 2.0 * mat.macro_strength);
{sea_line}

    let ground_map = textureSample(sky_occ_tex, layer_samp, in.uv, 0);
    let sky_vis = ground_map.r;

    let ground_sun_d = (ground_map.g - 0.5) * (2.0 * {sun_range:?});
    let ground_sun_w = max(max(ground_map.b * {sun_range:?}, 0.5 * fwidth(ground_sun_d)), 0.02);
    receiver_sun_mask = select(-1.0, smoothstep(-ground_sun_w, ground_sun_w, ground_sun_d), ground_map.a < 0.5);

    let st_layers = textureNumLayers(sky_occ_tex) - 1u;
    var st_0 = vec4<f32>(1.0);
    var st_1 = vec4<f32>(1.0);
    var st_2 = vec4<f32>(1.0);
    var st_3 = vec4<f32>(1.0);
    if (st_layers > 0u) {{
        st_0 = textureSample(sky_occ_tex, layer_samp, in.uv, 1);
    }}
    if (st_layers > 1u) {{
        st_1 = textureSample(sky_occ_tex, layer_samp, in.uv, 2);
    }}
    if (st_layers > 2u) {{
        st_2 = textureSample(sky_occ_tex, layer_samp, in.uv, 3);
    }}
    if (st_layers > 3u) {{
        st_3 = textureSample(sky_occ_tex, layer_samp, in.uv, 4);
    }}
    set_stationary_masks(st_0, st_1, st_2, st_3, {stationary_range:?});

    let lit = shade_material_env(
        in.world_pos,
        shaded_n,
        rough,
        clamp(ao_map, 0.0, 1.0),
        sky_vis,
        vec3<f32>(0.0),
        vec4<f32>(0.5, 0.5, 0.5, 0.0),
        albedo,

        in.world_pos,

        n,
    );
    return vec4<f32>(tonemap(in.col.rgb * lit), 1.0);
}}
"#,
        lights_block = terrain_lights_block(crate::renderer::lights::LightsBlockOptions::default()),
        sea_line = SEA_LINE_WGSL,
        sun_range = super::brush_pipeline::SUN_MASK_DISTANCE_TEXELS,
        stationary_range = super::brush_pipeline::STATIONARY_MASK_DISTANCE_TEXELS,
        biplanar_block = wgsl_biplanar_block(),
        whiteout_block = wgsl_whiteout_block(),
        shape_block = super::ground_shape::WGSL,
        layer_reads = LAYER_READS,
    )
}

const LAYER_READS: &str = "    var albedo = vec3<f32>(0.0);
    var shaded_n = vec3<f32>(0.0);
    var rough_map = 0.0;
    var ao_map = 0.0;
    var layers_left = select(0u, 1u, w.x > WEIGHT_EPS) | select(0u, 2u, w.y > WEIGHT_EPS)
        | select(0u, 4u, w.z > WEIGHT_EPS) | select(0u, 8u, w.w > WEIGHT_EPS);
    loop {
        if (layers_left == 0u) {
            break;
        }
        let layer = i32(countTrailingZeros(layers_left));
        layers_left = layers_left & (layers_left - 1u);
        let lw = select(select(select(w.x, w.y, layer == 1), w.z, layer == 2), w.w, layer == 3);
        albedo = albedo + layer_colour_at(layer, f) * lw;
        if (detail > 0.0) {
            shaded_n = shaded_n + layer_normal_at(layer, n, f) * lw;
        }
        rough_map = rough_map + layer_rough_at(layer, f) * lw;
        ao_map = ao_map + layer_ao_at(layer, f) * lw;
    }
    if (detail > 0.0) {
        shaded_n = mix(n, shaded_n, detail);
    }
";

const LAYER_READS_INLINED: &str = "    var albedo = vec3<f32>(0.0);
    if (w.x > WEIGHT_EPS) { albedo = albedo + layer_colour_at(0, f) * w.x; }
    if (w.y > WEIGHT_EPS) { albedo = albedo + layer_colour_at(1, f) * w.y; }
    if (w.z > WEIGHT_EPS) { albedo = albedo + layer_colour_at(2, f) * w.z; }
    if (w.w > WEIGHT_EPS) { albedo = albedo + layer_colour_at(3, f) * w.w; }
    var shaded_n = vec3<f32>(0.0);
    if (detail > 0.0) {
        if (w.x > WEIGHT_EPS) { shaded_n = shaded_n + layer_normal_at(0, n, f) * w.x; }
        if (w.y > WEIGHT_EPS) { shaded_n = shaded_n + layer_normal_at(1, n, f) * w.y; }
        if (w.z > WEIGHT_EPS) { shaded_n = shaded_n + layer_normal_at(2, n, f) * w.z; }
        if (w.w > WEIGHT_EPS) { shaded_n = shaded_n + layer_normal_at(3, n, f) * w.w; }
        shaded_n = mix(n, shaded_n, detail);
    }
    var rough_map = 0.0;
    var ao_map = 0.0;
    if (w.x > WEIGHT_EPS) { rough_map = rough_map + layer_rough_at(0, f) * w.x; ao_map = ao_map + layer_ao_at(0, f) * w.x; }
    if (w.y > WEIGHT_EPS) { rough_map = rough_map + layer_rough_at(1, f) * w.y; ao_map = ao_map + layer_ao_at(1, f) * w.y; }
    if (w.z > WEIGHT_EPS) { rough_map = rough_map + layer_rough_at(2, f) * w.z; ao_map = ao_map + layer_ao_at(2, f) * w.z; }
    if (w.w > WEIGHT_EPS) { rough_map = rough_map + layer_rough_at(3, f) * w.w; ao_map = ao_map + layer_ao_at(3, f) * w.w; }
";

pub fn with_inlined_layer_reads(src: String) -> String {
    assert_eq!(src.matches(LAYER_READS).count(), 1, "the ground shader's layer reads moved");
    src.replacen(LAYER_READS, LAYER_READS_INLINED, 1)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::renderer::cuboid::SolidVertex;
    use crate::renderer::lights::LightsUniform;
    use wgpu::util::DeviceExt;

    pub fn headless_gpu() -> Option<(Device, Queue)> {
        let instance = Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&RequestAdapterOptions {
            apply_limit_buckets: false,
            power_preference: PowerPreference::default(),
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok()?;
        pollster::block_on(adapter.request_device(&DeviceDescriptor {
            required_features: Features::empty(),
            required_limits: crate::renderer::uniforms::scene_limits(Limits::default()),
            ..Default::default()
        }))
        .ok()
    }

    fn vertex(pos: [f32; 3], normal: [f32; 3]) -> SolidVertex {
        SolidVertex {
            position: pos,
            normal,
            color: [1.0, 1.0, 1.0, 1.0],
            uv2: [0.0, 0.0],
            reflectivity: 0.0,
        }
    }

    #[derive(Copy, Clone)]
    pub enum Palette {
        Test,
        Fallback,

        Ramp,
    }

    pub fn render_quad_with_normal(normal: [f32; 3]) -> Option<[u8; 4]> {
        render_quad(normal, Palette::Test, None, None)
    }

    pub fn render_quad_with_splat(normal: [f32; 3], splat: &TerrainImage) -> Option<[u8; 4]> {
        render_quad(normal, Palette::Test, Some(splat), None)
    }

    pub fn render_quad_with_fallback_material(normal: [f32; 3]) -> Option<[u8; 4]> {
        render_quad(normal, Palette::Fallback, None, None)
    }

    fn render_quad(
        normal: [f32; 3],
        palette: Palette,
        splat: Option<&TerrainImage>,
        sky_occlusion: Option<&TerrainImage>,
    ) -> Option<[u8; 4]> {
        render_quad_full(normal, palette, splat, sky_occlusion, &[None, None, None, None], false)
    }

    pub fn render_quad_full(
        normal: [f32; 3],
        palette: Palette,
        splat: Option<&TerrainImage>,
        sky_occlusion: Option<&TerrainImage>,
        normals: &[Option<TerrainImage>],
        lit: bool,
    ) -> Option<[u8; 4]> {
        render_quad_lit_by(normal, palette, splat, sky_occlusion, normals, lit.then_some(None))
    }

    pub fn render_quad_lit_by(
        normal: [f32; 3],
        palette: Palette,
        splat: Option<&TerrainImage>,
        sky_occlusion: Option<&TerrainImage>,
        normals: &[Option<TerrainImage>],
        light: Option<Option<u8>>,
    ) -> Option<[u8; 4]> {
        render_quad_light_at(normal, palette, splat, sky_occlusion, normals, light, glam::Vec3::new(3.0, 2.0, 0.0))
    }

    pub fn render_quad_light_at(
        normal: [f32; 3],
        palette: Palette,
        splat: Option<&TerrainImage>,
        sky_occlusion: Option<&TerrainImage>,
        normals: &[Option<TerrainImage>],
        light: Option<Option<u8>>,
        light_pos: glam::Vec3,
    ) -> Option<[u8; 4]> {
        render_quad_turned(normal, palette, splat, sky_occlusion, normals, light, light_pos, 0.0)
    }

    pub fn render_quad_turned(
        normal: [f32; 3],
        palette: Palette,
        splat: Option<&TerrainImage>,
        sky_occlusion: Option<&TerrainImage>,
        normals: &[Option<TerrainImage>],
        light: Option<Option<u8>>,
        light_pos: glam::Vec3,
        yaw: f32,
    ) -> Option<[u8; 4]> {

        let read = glam::Vec3::new(0.125, -0.125, 0.5);
        let yaw_inv = glam::Quat::from_rotation_y(-yaw);
        let to_player = |p: glam::Vec3| read + yaw_inv * (p - read);
        let light_pos = to_player(light_pos);

        const WORLD_SHIFT: glam::Vec3 = glam::Vec3::new(2.375, 0.625, 4.0);
        let offset = read + WORLD_SHIFT - glam::Quat::from_rotation_y(yaw) * read;
        let normal = (yaw_inv * glam::Vec3::from(normal)).to_array();

        let plane_n = glam::Vec3::from(normal).try_normalize().unwrap_or(glam::Vec3::Z);
        let plane_t = plane_n.any_orthonormal_vector();
        let plane_b = plane_n.cross(plane_t);
        let on_plane = |x: f32, y: f32| (read + plane_t * (x - read.x) + plane_b * (y - read.y)).to_array();
        let quad_view = glam::Mat4::from_cols(
            glam::Vec4::new(plane_t.x, plane_b.x, 0.0, 0.0),
            glam::Vec4::new(plane_t.y, plane_b.y, 0.0, 0.0),
            glam::Vec4::new(plane_t.z, plane_b.z, 0.0, 0.0),
            glam::Vec4::new(read.x - plane_t.dot(read), read.y - plane_b.dot(read), 0.5, 1.0),
        );
        let lit = light.is_some();
        let (device, queue) = headless_gpu()?;
        let format = TextureFormat::Rgba8Unorm;

        let lights = LightsUniform::new(&device);
        let (_shadows, uniforms) =
            crate::renderer::uniforms::test_support::scene_uniforms(&device, &lights);

        uniforms.upload_scene(
            &queue,
            quad_view,

            to_player(glam::Vec3::new(0.0, 0.0, 5.0)),
            &crate::renderer::uniforms::ShadowUpload::disabled(),
            &crate::renderer::uniforms::SkyUpload::none(),
            &crate::renderer::uniforms::PostUpload::default(),
            &crate::renderer::uniforms::PlayerUpload { yaw, offset, ..Default::default() },
        );
        if lit {

            lights.upload(&queue, &[crate::renderer::lights::Light {
                mask_channel: light.flatten(),
                shadow_near: None,
                source_radius: 0.0,
                in_level_bake: true,
                position: light_pos,
                direction: glam::Vec3::new(0.0, -1.0, 0.0),
                kind: crate::renderer::lights::LightKind::Point,
                color: crate::renderer::Color3(255, 255, 255, 255),

                intensity: 3.0,
                range: 50.0,
                cone_angle_deg: 180.0,
                inner_cone_angle_deg: 0.0,
            }]);
        } else {
            lights.upload(&queue, &[]);
        }

        let pipeline = TerrainPipeline::new(&device, format, &uniforms.layout);

        let solid = |rgb: [u8; 4]| TerrainImage { width: 1, height: 1, rgba: rgb.to_vec() };
        let test_layers = [
            solid([255, 0, 0, 255]),
            solid([0, 0, 255, 255]),
            solid([0, 255, 0, 255]),
            solid([255, 255, 255, 255]),
        ];
        let neutral = solid([128, 128, 128, 255]);

        let material = match palette {
            Palette::Test => TerrainMaterial::new(
                &device, &queue, &pipeline.material_layout,
                &test_layers, &neutral, splat, sky_occlusion, normals,
                &[], &[], TerrainMaterialUniform::default(),
            ),
            Palette::Fallback => {
                TerrainMaterial::fallback(&device, &queue, &pipeline.material_layout)
            }
            Palette::Ramp => {
                let ramp = TerrainImage {
                    width: 64,
                    height: 64,
                    rgba: (0..64 * 64).flat_map(|i| [(i % 64) as u8 * 4, 0, 0, 255]).collect(),
                };

                let layers = [ramp, solid_image([0, 0, 255], 64, 64), solid_image([0, 255, 0], 64, 64), solid_image([255, 255, 255], 64, 64)];
                TerrainMaterial::new(
                    &device, &queue, &pipeline.material_layout,
                    &layers, &neutral, splat, sky_occlusion, normals,
                    &[], &[], TerrainMaterialUniform::default(),
                )
            }
        };

        let verts = [
            vertex(on_plane(-1.0, -1.0), normal),
            vertex(on_plane(3.0, -1.0), normal),
            vertex(on_plane(-1.0, 3.0), normal),
        ];
        let vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("test_vb"),
            contents: bytemuck::cast_slice(&verts),
            usage: BufferUsages::VERTEX,
        });
        let ib = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("test_ib"),
            contents: bytemuck::cast_slice(&[0u32, 1, 2]),
            usage: BufferUsages::INDEX,
        });

        const SIZE: u32 = 8;
        let target = device.create_texture(&TextureDescriptor {
            label: Some("test_target"),
            size: Extent3d { width: SIZE, height: SIZE, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1,
            dimension: TextureDimension::D2,
            format,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let depth = device.create_texture(&TextureDescriptor {
            label: Some("test_depth"),
            size: Extent3d { width: SIZE, height: SIZE, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Depth32Float,
            usage: TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let target_view = target.create_view(&Default::default());
        let depth_view = depth.create_view(&Default::default());

        let readback = device.create_buffer(&BufferDescriptor {
            label: Some("test_readback"),
            size: (256 * SIZE) as u64,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("test_pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &target_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: Operations { load: LoadOp::Clear(Color::BLACK), store: StoreOp::Store },
                })],
                depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(Operations { load: LoadOp::Clear(1.0), store: StoreOp::Store }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                multiview_mask: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&pipeline.pipeline);
            pass.set_bind_group(0, &uniforms.bind_group, &[]);
            pass.set_bind_group(1, &material.bind_group, &[]);
            pass.set_vertex_buffer(0, vb.slice(..));
            pass.set_index_buffer(ib.slice(..), IndexFormat::Uint32);
            pass.draw_indexed(0..3, 0, 0..1);
        }
        encoder.copy_texture_to_buffer(
            TexelCopyTextureInfo {
                texture: &target, mip_level: 0, origin: Origin3d::ZERO, aspect: TextureAspect::All,
            },
            TexelCopyBufferInfo {
                buffer: &readback,
                layout: TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(256), rows_per_image: Some(SIZE) },
            },
            Extent3d { width: SIZE, height: SIZE, depth_or_array_layers: 1 },
        );
        queue.submit(Some(encoder.finish()));

        let slice = readback.slice(..);
        slice.map_async(MapMode::Read, |_| {});
        device.poll(PollType::Wait { submission_index: None, timeout: None }).ok();
        let data = slice.get_mapped_range().unwrap();
        let centre = (SIZE / 2) as usize * 256 + (SIZE / 2) as usize * 4;
        Some([data[centre], data[centre + 1], data[centre + 2], data[centre + 3]])
    }

    #[test]
    fn the_outdoor_twin_is_the_pass_with_its_switch_turned() {
        use wgpu::naga;
        let pass = terrain_shader_for(TerrainRole::ProbePass);
        let twin = outdoor_pass_shader().expect("the outdoor switch moved");
        let (sky_twice, sky_once) = crate::renderer::brush_pipeline::DEDUP_OUTDOOR_SKY;
        assert_eq!(twin.replacen(ALL_OUTDOORS_ON, ALL_OUTDOORS_OFF, 1).replacen(sky_once, sky_twice, 1), pass);
        assert!(twin.contains(sky_once), "the outdoor sky's two reads are no longer where the dedup edit expects them");
        assert!(pass.contains("const PROBE_FACE_ALWAYS_GIVEN: bool = false;"));
        assert!(!pass.contains("probe_face_given = ") && !pass.contains("probe_volume_pos = "));
        let at = |s: &str, m: &str| s.find(m).unwrap_or_else(|| panic!("`{m}` is not in the shader"));

        let env = &twin[at(&twin, "fn probe_environment(")..at(&twin, "fn probe_trace(")];
        assert!(at(env, "    if (PROBE_ALL_OUTDOORS) {") < at(env, "    let hit = probe_trace(world_pos, d, trace_room, roughness);"));
        assert!(at(env, "    if (PROBE_ALL_OUTDOORS) {") < at(env, "probe_choose(select_world, volume_world)"));
        assert!(at(env, "    if (PROBE_ALL_OUTDOORS) {") < at(env, "probe_portal_holding(volume_world)"));
        for (label, src) in [
            ("twin", twin.clone()),
            ("weather twin", crate::renderer::weather::with_weather(&twin).expect("the weather's edits moved")),
        ] {
            let module = naga::front::wgsl::parse_str(&src).unwrap_or_else(|e| panic!("{label}: {}", e.emit_to_string(&src)));
            naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
                .validate(&module)
                .unwrap_or_else(|e| panic!("{label}: {e:?}"));
        }
        let size = |s: &str| -> usize {
            crate::renderer::shader_inlining::inlined_sizes(s, "fs_main").iter().map(|f| f.own * f.copies).sum()
        };
        let (full, out) = (size(&pass), size(&twin));
        println!("the ground's pass: {full} proxy units; its outdoor twin: {out}");
        if std::env::var("TWIN_REPORT").is_ok() {
            println!("{}", crate::renderer::shader_inlining::report("outdoor twin", &twin, "fs_main", 30));
        }

        assert!(out * 5 < full * 3, "the outdoor twin is {out} of {full}: the rooms' code is still in it");
        let Some((device, _queue)) = headless_gpu() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let lights = LightsUniform::new(&device);
        let (_shadows, uniforms) = crate::renderer::uniforms::test_support::scene_uniforms(&device, &lights);
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let fixups = crate::renderer::probe_fixup::ProbeFixups::new(&device, &uniforms.layout, 1024);
        let weather_layout = crate::renderer::weather::bind_group_layout(&device);
        assert!(TerrainPipeline::new_probe_pass_outdoors(&device, &uniforms.layout, &fixups, None).is_some());
        assert!(TerrainPipeline::new_probe_pass_outdoors(&device, &uniforms.layout, &fixups, Some(&weather_layout)).is_some());
        let err = pollster::block_on(scope.pop());
        assert!(err.is_none(), "the outdoor twins failed to build: {err:?}");
    }

    #[test]
    fn the_painted_build_keeps_the_splats_weights_alone() {
        use wgpu::naga;
        assert!(!GROUND_PAINTED.load(std::sync::atomic::Ordering::Relaxed));
        for role in [TerrainRole::Scene, TerrainRole::Read, TerrainRole::ProbePass] {
            let src = terrain_shader_for(role);
            let painted = painted_shader(&src).unwrap_or_else(|| panic!("{role:?}: the weights' choice moved"));
            assert_eq!(painted.replacen(PAINTED_TO, PAINTED_FROM, 1), src);
            let module = naga::front::wgsl::parse_str(&painted).unwrap_or_else(|e| panic!("{role:?}: {}", e.emit_to_string(&painted)));
            naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
                .validate(&module)
                .unwrap_or_else(|e| panic!("{role:?}: {e:?}"));
            let size = |s: &str| -> usize {
                crate::renderer::shader_inlining::inlined_sizes(s, "fs_main").iter().map(|f| f.own * f.copies).sum()
            };
            assert!(size(&painted) < size(&src), "{role:?}: nothing left the shader");
        }

        assert!(PAINTED_FROM.contains("select(derived, authored, mat.use_splat > 0.5)"));
    }

    #[test]
    fn the_ground_is_built_for_the_level_unless_pinned() {
        for built in [false, true] {
            for level in [false, true] {
                assert_eq!(ground_build_wanted(built, level, false), level, "built {built}, level {level}");
                assert_eq!(ground_build_wanted(built, level, true), built, "pinned at {built}, level {level}");
            }
        }
    }

    #[test]
    fn every_ground_twin_takes_the_painted_build() {
        use crate::renderer::brush_pipeline::{sun_reader_shader, FaceSun};
        use crate::renderer::ground_twins::{dedup_pass_shader, gentle_shader, steep_shader};
        use crate::renderer::lights::{without_pool_maps, without_spot_shadows};
        use crate::renderer::weather::{with_weather, with_weather_kinds, WeatherKinds};
        use wgpu::naga;
        let read = terrain_shader_built(TerrainRole::Read, true);
        let pass = terrain_shader_built(TerrainRole::ProbePass, true);
        for (role, src) in [(TerrainRole::Read, &read), (TerrainRole::ProbePass, &pass)] {
            assert_eq!(src.matches(PAINTED_TO).count(), 1, "{role:?}");
            assert!(!src.contains(PAINTED_FROM), "{role:?}: the choice is still in the painted build");
            assert_eq!(src.replacen(PAINTED_TO, PAINTED_FROM, 1), terrain_shader_built(role, false), "{role:?}");
        }
        let turned = |off: &str, on: &str| {
            assert_eq!(pass.matches(off).count(), 1, "`{off}` is not in the painted pass once");
            pass.replacen(off, on, 1)
        };
        let gentle = gentle_shader(&read).expect("gentle");
        let steep = steep_shader(&read).expect("steep");
        let dedup = dedup_pass_shader(&pass).expect("dedup");
        let nowhere = turned(NOWHERE_OFF, NOWHERE_ON);
        let outdoors = turned(ALL_OUTDOORS_OFF, ALL_OUTDOORS_ON);
        let open = turned(OPEN_GROUND_OFF, OPEN_GROUND_ON);
        let mut sources: Vec<(String, String)> = vec![
            ("read baked spotless".into(), without_spot_shadows(sun_reader_shader(read.clone(), FaceSun::Baked))),
            ("gentle baked spotless".into(), without_spot_shadows(sun_reader_shader(gentle.clone(), FaceSun::Baked))),
            ("steep baked spotless".into(), without_spot_shadows(sun_reader_shader(steep.clone(), FaceSun::Baked))),
            ("dedup poolless".into(), without_pool_maps(dedup.clone())),
            ("nowhere".into(), nowhere.clone()),
            ("outdoors".into(), outdoors.clone()),
            ("open".into(), open.clone()),
        ];
        for (label, src) in [("read", &read), ("pass", &pass), ("dedup", &dedup), ("nowhere", &nowhere), ("outdoors", &outdoors), ("open", &open)] {
            sources.push((format!("weather {label}"), with_weather(src).unwrap_or_else(|| panic!("{label}: the weather's edits moved"))));
        }
        for kinds in WeatherKinds::ALL {
            let twin = with_weather_kinds(&gentle, kinds).unwrap_or_else(|| panic!("gentle {kinds:?}: the weather's edits moved"));
            sources.push((format!("weather gentle {kinds:?}"), twin));
        }
        for (label, src) in &sources {
            let module = naga::front::wgsl::parse_str(src).unwrap_or_else(|e| panic!("{label}: {}", e.emit_to_string(src)));
            naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
                .validate(&module)
                .unwrap_or_else(|e| panic!("{label}: {e:?}"));
        }
    }

    #[test]
    fn the_grounds_scene_shaders_write_alpha_one() {
        use crate::renderer::weather::{with_weather, with_weather_kinds, WeatherKinds};
        const ANSWER: &str = "return vec4<f32>(tonemap(in.col.rgb * lit), 1.0);";
        let mut seen = 0;
        for role in [TerrainRole::Scene, TerrainRole::Read] {
            let plain = terrain_shader_for(role);
            let mut sources = vec![plain.clone()];
            sources.extend(crate::renderer::ground_twins::steep_shader(&plain));
            for src in sources.clone() {
                sources.extend(with_weather(&src));
                sources.extend(WeatherKinds::ALL.iter().filter_map(|&k| with_weather_kinds(&src, k)));
            }
            for src in sources {
                let main = &src[src.find("fn fs_main(").expect("the ground's fragment shader")..];
                let main = &main[..main.find("\n}\n").expect("its end")];
                let returns: Vec<&str> = main.lines().map(str::trim).filter(|l| l.starts_with("return ")).collect();
                assert_eq!(returns, [ANSWER], "a ground reader answers with something a blend would not keep");
                seen += 1;
            }
        }
        assert!(seen >= 8, "{seen} readers looked at");
    }

    #[test]
    fn every_ground_shader_takes_its_vertices_in_the_world() {
        use crate::renderer::weather::{with_weather, with_weather_kinds, WeatherKinds};
        let mut seen = 0;
        for role in [TerrainRole::Scene, TerrainRole::Read, TerrainRole::ProbePass] {
            let plain = terrain_shader_for(role);
            let mut sources = vec![plain.clone()];
            sources.extend(crate::renderer::ground_twins::steep_shader(&plain));
            if role == TerrainRole::ProbePass {
                sources.extend(nowhere_pass_shader());
                sources.extend(open_pass_shader());
                sources.extend(puddle_pass_shader());
            }
            for src in sources.clone() {
                sources.extend(with_weather(&src));
                sources.extend(WeatherKinds::ALL.iter().filter_map(|&k| with_weather_kinds(&src, k)));
            }
            for src in sources {
                let world = world_vertices(&src).expect("a ground shader that does not open as the ground's does");
                assert!(world.contains("v.pos = to_player_space(w.pos);") && world.contains("v.norm = to_player_direction(w.norm);"));
                assert_eq!(world.matches("fn vs_main(").count(), 1);

                let body = &world[world.find("v.norm = to_player_direction(w.norm);").unwrap() + 40..];
                let body = &body[..body.find("\n}\n").unwrap()];
                assert!(!body.contains("w."), "the vertex shader reads the world's vertex past its posing");
                seen += 1;
            }
        }
        assert!(seen >= 12, "{seen} shaders looked at");
        let validate = |label: &str, src: &str| {
            use wgpu::naga;
            let module = naga::front::wgsl::parse_str(src).unwrap_or_else(|e| panic!("{label}: {}", e.emit_to_string(src)));
            naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
                .validate(&module)
                .unwrap_or_else(|e| panic!("{label}: {e:?}"));
        };
        validate("the ground reader from world vertices", &world_vertices(&terrain_shader_for(TerrainRole::Read)).unwrap());
        validate("the ground's reflection pass from world vertices", &world_vertices(&terrain_shader_for(TerrainRole::ProbePass)).unwrap());
        let solid = world_vertices(&crate::renderer::pipeline::solid_shader_src()).expect("the solid pipeline's shader opens as the ground's");
        validate("the solid pipeline from world vertices", &solid);
    }

    #[test]
    fn the_puddle_twin_is_the_open_twin_answering_with_the_sky() {
        use wgpu::naga;
        let open = open_pass_shader().expect("the open-ground switch moved");
        let puddle = puddle_pass_shader().expect("the open ground's answer moved");
        assert_eq!(puddle.replacen(OPEN_SKY_COLOUR, OPEN_COLOUR, 1), open);
        assert!(open.contains("    return sky_reflection(d, dir, lod);"), "the outdoors no longer ends in the sky");
        let wet = crate::renderer::weather::with_weather(&puddle).expect("the weather's edits");
        for src in [&puddle, &wet] {
            let module = naga::front::wgsl::parse_str(src).unwrap_or_else(|e| panic!("{}", e.emit_to_string(src)));
            naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
                .validate(&module)
                .unwrap_or_else(|e| panic!("{e:?}"));
        }
    }

    #[test]
    fn the_open_twin_is_the_pass_with_its_switch_turned() {
        use wgpu::naga;
        let pass = terrain_shader_for(TerrainRole::ProbePass);
        let twin = open_pass_shader().expect("the open-ground switch moved");
        assert_eq!(twin.replacen(OPEN_GROUND_ON, OPEN_GROUND_OFF, 1), pass);
        let at = |s: &str, m: &str| s.find(m).unwrap_or_else(|| panic!("`{m}` is not in the shader"));
        let env = &twin[at(&twin, "fn probe_environment(")..at(&twin, "fn probe_outdoor_untraced(")];
        let branch = at(env, "    if (PROBE_OPEN_GROUND) {");
        for later in [
            "    if (PROBE_NOWHERE) {",
            "    if (PROBE_ALL_OUTDOORS) {",
            "probe_portal_holding(volume_world)",
            "probe_choose(select_world, volume_world)",
            "    let hit = probe_trace(world_pos, d, trace_room, roughness);",
        ] {
            assert!(branch < at(env, later), "`{later}` comes before the open ground's answer");
        }
        let body = &env[branch..at(env, "    if (PROBE_NOWHERE) {")];

        assert!(body.contains("if (roughness > PROBE_TRACE_MAX_ROUGHNESS) {"));
        assert!(body.contains("let open_dir = normalize(world_dir);"));
        assert!(body.contains("let open_lod = clamp(roughness * PROBE_ROUGHNESS_MIPS, PROBE_MIN_LOD, PROBE_MAX_LOD);"));
        assert!(body.contains("probe_escape_colour(world_pos, open_dir, dir, roughness, open_lod)"));
        assert!(env.contains("    let d = normalize(world_dir);"));
        assert!(env.contains("    let probe_lod = clamp(roughness * PROBE_ROUGHNESS_MIPS, PROBE_MIN_LOD, PROBE_MAX_LOD);"));

        let trace = &pass[at(&pass, "fn probe_trace_skipping(")..at(&pass, "fn ground_trace_until(")];
        let holding = at(trace, "        let p = probe_portal_holding(world_pos);\n        if (p < 0) {\n            if (camera.portal_params.y < -0.5) {");
        let fallback = &trace[holding..holding + at(&trace[holding..], "            return hit;\n        }\n")];
        for line in ["hit.pos = world_pos;", "hit.found = true;", "hit.escaped = true;"] {
            assert!(fallback.contains(line), "`{line}` is not in the open ground's hit");
        }
        assert!(at(trace, "if (roughness > PROBE_TRACE_MAX_ROUGHNESS || camera.portal_params.y > 0.5) {") < holding);
        assert!(pass.contains("        col = probe_escape_colour(h.pos, d, sky_dir, roughness, lod);"));

        let brush = crate::renderer::brush_pipeline::brush_shader_src();
        assert!(brush.contains("let p = probe_portal_holding(world_pos);"));
        assert!(!brush.contains("camera.portal_params.y < -0.5"));
        for (label, src) in [
            ("twin", twin.clone()),
            ("weather twin", crate::renderer::weather::with_weather(&twin).expect("the weather's edits moved")),
        ] {
            let module = naga::front::wgsl::parse_str(&src).unwrap_or_else(|e| panic!("{label}: {}", e.emit_to_string(&src)));
            naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
                .validate(&module)
                .unwrap_or_else(|e| panic!("{label}: {e:?}"));
        }
        let size = |s: &str| -> usize {
            crate::renderer::shader_inlining::inlined_sizes(s, "fs_main").iter().map(|f| f.own * f.copies).sum()
        };
        let (full, out) = (size(&pass), size(&twin));
        println!("the ground's pass: {full} proxy units; its open-ground twin: {out}");
        assert!(out * 5 < full * 3, "the open-ground twin is {out} of {full}: the rooms' machinery is still in it");
        let Some((device, _queue)) = headless_gpu() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let lights = LightsUniform::new(&device);
        let (_shadows, uniforms) = crate::renderer::uniforms::test_support::scene_uniforms(&device, &lights);
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let fixups = crate::renderer::probe_fixup::ProbeFixups::new(&device, &uniforms.layout, 1024);
        let weather_layout = crate::renderer::weather::bind_group_layout(&device);
        assert!(TerrainPipeline::new_probe_pass_open(&device, &uniforms.layout, &fixups, None).is_some());
        assert!(TerrainPipeline::new_probe_pass_open(&device, &uniforms.layout, &fixups, Some(&weather_layout)).is_some());
        let err = pollster::block_on(scope.pop());
        assert!(err.is_none(), "the open-ground twins failed to build: {err:?}");
    }

    #[test]
    fn the_nowhere_twin_is_the_pass_with_its_switch_turned() {
        use wgpu::naga;
        let pass = terrain_shader_for(TerrainRole::ProbePass);
        let twin = nowhere_pass_shader().expect("the nowhere switch moved");
        assert_eq!(twin.replacen(NOWHERE_ON, NOWHERE_OFF, 1), pass);
        let at = |s: &str, m: &str| s.find(m).unwrap_or_else(|| panic!("`{m}` is not in the shader"));
        let env = &twin[at(&twin, "fn probe_environment(")..at(&twin, "fn probe_outdoor_untraced(")];
        let branch = at(env, "    if (PROBE_NOWHERE) {");
        for later in [
            "    if (PROBE_ALL_OUTDOORS) {",
            "probe_portal_holding(volume_world)",
            "probe_choose(select_world, volume_world)",
            "    let hit = probe_trace(world_pos, d, trace_room, roughness);",
            "    return probe_through_portals(own, own_room, select_world, world_pos, d, probe_lod);",
        ] {
            assert!(branch < at(env, later), "`{later}` comes before the nowhere answer");
        }

        assert!(at(env, "    probe_brightness = 0.0;") < branch);
        let body = &env[branch..branch + at(&env[branch..], "    }\n")];
        assert!(body.contains("probe_eye_distance = distance(cam_pos(), frag_pos);"));
        assert!(body.contains("let nothing = vec4<f32>(0.0);") && body.contains("return nothing;"));
        assert!(!body.contains("probe_reach"));
        for (label, src) in [
            ("twin", twin.clone()),
            ("weather twin", crate::renderer::weather::with_weather(&twin).expect("the weather's edits moved")),
        ] {
            let module = naga::front::wgsl::parse_str(&src).unwrap_or_else(|e| panic!("{label}: {}", e.emit_to_string(&src)));
            naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
                .validate(&module)
                .unwrap_or_else(|e| panic!("{label}: {e:?}"));
        }
        let size = |s: &str| -> usize {
            crate::renderer::shader_inlining::inlined_sizes(s, "fs_main").iter().map(|f| f.own * f.copies).sum()
        };
        let (full, out) = (size(&pass), size(&twin));
        println!("the ground's pass: {full} proxy units; its nowhere twin: {out}");

        assert!(out * 5 < full * 2, "the nowhere twin is {out} of {full}: the lookup is still in it");
        if std::env::var("TWIN_REPORT").is_ok() {
            println!("{}", crate::renderer::shader_inlining::report("nowhere twin", &twin, "fs_main", 20));
        }
        let Some((device, _queue)) = headless_gpu() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let lights = LightsUniform::new(&device);
        let (_shadows, uniforms) = crate::renderer::uniforms::test_support::scene_uniforms(&device, &lights);
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let fixups = crate::renderer::probe_fixup::ProbeFixups::new(&device, &uniforms.layout, 1024);
        let weather_layout = crate::renderer::weather::bind_group_layout(&device);
        assert!(TerrainPipeline::new_probe_pass_nowhere(&device, &uniforms.layout, &fixups, None).is_some());
        assert!(TerrainPipeline::new_probe_pass_nowhere(&device, &uniforms.layout, &fixups, Some(&weather_layout)).is_some());
        let err = pollster::block_on(scope.pop());
        assert!(err.is_none(), "the nowhere twins failed to build: {err:?}");
    }

    #[test]
    fn the_rough_skip_is_a_cut_and_not_in_the_ground_pass() {
        let pass = terrain_shader_for(TerrainRole::ProbePass);
        assert!(!pass.contains(PASS_ROUGH_SKIP));
        assert_eq!(PASS_ROUGH_SKIP_IN, PASS_ROUGH_SKIP_AT.replacen("    return", &format!("{PASS_ROUGH_SKIP}    return"), 1));
        let (_, edits) = PROBE_PASS_REGISTER_CUTS.iter().find(|(l, _)| *l == "terrain_cut_rough_skip").unwrap();
        let wet = crate::renderer::weather::with_weather(&pass).expect("the ground's weather twin no longer matches its shader");
        assert!(wet.contains("    var rough = specular_aa_roughness("));
        let at = |s: &str, m: &str| s.find(m).unwrap_or_else(|| panic!("`{m}` is not in the shader"));
        for src in [&pass, &wet] {
            assert_eq!(src.matches(edits[0].0).count(), 1);
            let cut = src.replacen(edits[0].0, edits[0].1, 1);
            assert_eq!(cut.matches(PASS_ROUGH_SKIP).count(), 1);
            assert!(at(&cut, "    probe_fragment = in.clip;\n") < at(&cut, PASS_ROUGH_SKIP));
            assert!(at(&cut, PASS_ROUGH_SKIP) < at(&cut, "    return probe_env_for_pass(\n        in.world_pos"));
        }

        let read = terrain_shader_for(TerrainRole::Read);
        for line in [
            "    let lobe_is_hemispherical = smoothstep(0.25, PROBE_LOBE_HEMISPHERICAL, r);",
            "    let environment = mix(sharp, baseline, lobe_is_hemispherical);",
            "    let image_weight = (1.0 - lobe_is_hemispherical) * clamp(probe.a, 0.0, 1.0);",
            "    let r = clamp(roughness, 0.04, 1.0);",
        ] {
            assert!(read.contains(line), "the scene shader no longer has `{line}`");
        }
        let smoothstep = |lo: f32, hi: f32, x: f32| {
            let t = ((x - lo) / (hi - lo)).clamp(0.0, 1.0);
            t * t * (3.0 - 2.0 * t)
        };
        for r in [0.75f32, 0.7500001, 0.8, 0.93, 1.0] {
            assert_eq!(smoothstep(0.25, 0.75, r), 1.0, "{r}");
        }
        assert!(smoothstep(0.25, 0.75, 0.7499) < 1.0);
    }

    #[test]

    #[test]
    fn the_probe_pass_terrain_shaders_validate_and_build() {
        use wgpu::naga;

        let poolless = crate::renderer::lights::without_pool_maps(terrain_shader_for(TerrainRole::ProbePass));
        let read = terrain_shader_for(TerrainRole::Read);
        let baked = crate::renderer::brush_pipeline::sun_reader_shader(read.clone(), crate::renderer::brush_pipeline::FaceSun::Baked);
        let spotless = crate::renderer::lights::without_spot_shadows;
        for (role, src) in [
            ("read", read.clone()),
            ("pass", terrain_shader_for(TerrainRole::ProbePass)),
            ("poolless pass", poolless),
            ("spotless read", spotless(read.clone())),
            ("baked read", baked.clone()),
            ("baked spotless read", spotless(baked)),
            ("inlined read", with_inlined_layer_reads(read)),
            ("inlined pass", with_inlined_layer_reads(terrain_shader_for(TerrainRole::ProbePass))),
        ] {
            let module = naga::front::wgsl::parse_str(&src)
                .unwrap_or_else(|e| panic!("{role}: {}", e.emit_to_string(&src)));
            naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
                .validate(&module)
                .unwrap_or_else(|e| panic!("{role}: {e:?}"));
        }
        assert!(terrain_shader_for(TerrainRole::Read).contains("const PROBE_ENV_FROM_PASS: bool = true;"));
        let pass = terrain_shader_for(TerrainRole::ProbePass);
        assert!(pass.contains("const PROBE_SECONDARY_DEFERRED: bool = true;") && pass.contains("return probe_env_for_pass("));
        let Some((device, _queue)) = headless_gpu() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let lights = LightsUniform::new(&device);
        let (_shadows, uniforms) = crate::renderer::uniforms::test_support::scene_uniforms(&device, &lights);
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let probe_layout = crate::renderer::brush_pipeline::probe_pass::bind_group_layout(&device);
        let _read = TerrainPipeline::new_probe_reader(&device, TextureFormat::Rgba8UnormSrgb, &uniforms.layout, 4, &probe_layout);
        let fixups = crate::renderer::probe_fixup::ProbeFixups::new(&device, &uniforms.layout, 1024);
        let _pass = TerrainPipeline::new_probe_pass(&device, &uniforms.layout, &fixups);
        let _poolless = TerrainPipeline::new_probe_pass_poolless(&device, &uniforms.layout, &fixups);
        let _twins =
            TerrainPipeline::new_probe_reader_twins(&device, TextureFormat::Rgba8UnormSrgb, &uniforms.layout, 4, &probe_layout);
        let _inlined =
            TerrainPipeline::new_inlined(&device, TextureFormat::Rgba8UnormSrgb, &uniforms.layout, 4, &probe_layout, &fixups);
        let err = pollster::block_on(scope.pop());
        assert!(err.is_none(), "the terrain probe pipelines failed to build: {err:?}");
    }

    #[test]
    fn the_ground_readers_twins_and_inlined_reads_change_only_their_switches() {
        let read = terrain_shader_for(TerrainRole::Read);
        let baked = crate::renderer::brush_pipeline::sun_reader_shader(read.clone(), crate::renderer::brush_pipeline::FaceSun::Baked);
        assert_eq!(baked.replacen("const SUN_MASK_EVERYWHERE: bool = true;", "const SUN_MASK_EVERYWHERE: bool = false;", 1), read);
        let spotless = crate::renderer::lights::without_spot_shadows(read.clone());
        assert_ne!(spotless, read, "the spotless twin switched nothing off");
        let inlined = with_inlined_layer_reads(read.clone());
        assert_eq!(inlined.replacen(LAYER_READS_INLINED, LAYER_READS, 1), read);

        for call in ["layer_colour_at(", "layer_normal_at(", "layer_rough_at(", "layer_ao_at("] {
            let body = |src: &str| src.split("@fragment").nth(1).map_or(0, |entry| entry.matches(call).count());
            assert_eq!((body(&read), body(&inlined)), (1, 4), "{call}");
        }
    }

    #[test]
    fn a_ground_map_is_baked_everywhere_only_when_every_texel_is() {
        let map = |alphas: &[u8]| TerrainImage {
            width: alphas.len() as u32,
            height: 1,
            rgba: alphas.iter().flat_map(|&a| [200, 128, 0, a]).collect(),
        };
        assert!(map(&[0, 0, 0, 0]).sun_baked_everywhere());
        assert!(map(&[0, 127]).sun_baked_everywhere(), "127 reads as 0.498, under the shader's half");
        assert!(!map(&[0, 128]).sun_baked_everywhere(), "128 reads as 0.502: not baked there");
        assert!(!map(&[255]).sun_baked_everywhere());
        assert!(!FULL_SKY.sun_baked_everywhere(), "no map is no bake");
        assert!(!TerrainImage { width: 0, height: 0, rgba: Vec::new() }.sun_baked_everywhere());
    }

    #[test]
    fn every_terrain_register_cut_applies_and_validates() {
        use wgpu::naga;
        let base = terrain_shader_for(TerrainRole::ProbePass);
        for (label, edits) in PROBE_PASS_REGISTER_CUTS {
            let mut src = base.clone();
            for (from, to) in edits.iter() {
                assert!(src.contains(from), "{label}: `{from}` is not in the ground's probe pass");

                assert_eq!(src.matches(from).count(), 1, "{label}: `{from}` is in the ground's probe pass more than once");
                src = src.replacen(from, to, 1);
            }
            let module = naga::front::wgsl::parse_str(&src)
                .unwrap_or_else(|e| panic!("{label}: {}", e.emit_to_string(&src)));
            naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
                .validate(&module)
                .unwrap_or_else(|e| panic!("{label}: {e:?}"));
        }
    }

    #[test]
    fn weightless_layers_are_skipped_rather_than_sampled() {
        let src = super::terrain_shader();
        assert!(
            src.contains("textureSampleGrad"),
            "explicit gradients are what make a per-fragment branch legal at all",
        );
        for w in ["w.x > WEIGHT_EPS", "w.y > WEIGHT_EPS", "w.z > WEIGHT_EPS", "w.w > WEIGHT_EPS"] {
            assert!(src.contains(w), "missing the skip for {w}");
        }
    }

    #[test]
    fn the_gradients_are_taken_outside_every_branch() {

        let src = super::terrain_shader();
        let frame = src
            .split("fn sample_frame")
            .nth(1)
            .and_then(|s| s.split("\nfn ").next())
            .expect("sample_frame present");
        let entry = src
            .split("@fragment")
            .nth(1)
            .expect("fragment entry present");
        let allowed = |tok: &str| frame.matches(tok).count() + entry.matches(tok).count();
        assert_eq!(
            src.matches("dpdx(").count(),
            allowed("dpdx("),
            "a derivative escaped both sample_frame and the fragment entry",
        );
        assert_eq!(src.matches("dpdy(").count(), allowed("dpdy("));

        let footprint = "let step_x = dpdx(in.world_pos);\n    let step_y = dpdy(in.world_pos);";
        let extra = usize::from(entry.contains(footprint));
        assert!(
            entry.matches("dpdx(").count() <= 1 + extra && entry.matches("dpdy(").count() <= 1 + extra,
            "more than the two named derivative pairs in the fragment entry; \
             sampling gradients belong in sample_frame",
        );
    }

    #[test]
    fn baked_sky_occlusion_darkens_the_terrain() {

        let sealed = TerrainImage { width: 1, height: 1, rgba: vec![0, 0, 0, 255] };
        let Some(open) = render_quad([0.0, 1.0, 0.0], Palette::Test, None, None) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let dark = render_quad([0.0, 1.0, 0.0], Palette::Test, None, Some(&sealed)).unwrap();
        assert!(
            dark[0] < open[0],
            "zero sky visibility must darken the ground: {dark:?} vs open {open:?}",
        );
    }

    #[test]
    fn unbaked_terrain_shades_exactly_as_it_did_before() {

        let full = TerrainImage { width: 1, height: 1, rgba: vec![255, 255, 255, 255] };
        let Some(absent) = render_quad([0.0, 1.0, 0.0], Palette::Test, None, None) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let explicit = render_quad([0.0, 1.0, 0.0], Palette::Test, None, Some(&full)).unwrap();
        assert_eq!(
            absent, explicit,
            "no occlusion map must shade identically to a fully-open one: \
             {absent:?} vs {explicit:?}",
        );
    }

    #[test]
    fn a_mip_chain_runs_all_the_way_down_to_one_texel() {

        assert_eq!(mip_levels_for(1024, 1024), 11);
        assert_eq!(mip_levels_for(1, 1), 1);
        assert_eq!(mip_levels_for(256, 64), 9, "levels follow the LONGER side");

        let levels = mip_levels_for(8, 8);
        let chain = mip_chain(&[128u8; 8 * 8 * 4], 8, 8, levels);
        assert_eq!(chain.len() as u32, levels);
        assert_eq!((chain[0].1, chain[0].2), (8, 8));
        assert_eq!(chain.last().unwrap().1, 1, "chain must reach 1x1");
        for (data, w, h) in &chain {
            assert_eq!(data.len(), (w * h * 4) as usize, "level {w}x{h} is the wrong size");
        }
    }

    #[test]
    fn mips_are_averaged_in_linear_space_not_srgb() {

        let mut px = vec![0u8; 2 * 2 * 4];
        for i in 0..4 {
            let v = if i % 2 == 0 { 0u8 } else { 255u8 };
            for c in 0..3 {
                px[i * 4 + c] = v;
            }
            px[i * 4 + 3] = 255;
        }
        let chain = mip_chain(&px, 2, 2, 2);
        let r = chain[1].0[0];
        assert!(
            (185..=191).contains(&r),
            "black+white must average to sRGB ~188 (linear 0.5), got {r} -- \
             {} suggests the average was taken on the encoded bytes",
            if r < 140 { "which" } else { "this" },
        );
        assert_eq!(chain[1].0[3], 255, "alpha is already linear and must not be transferred");
    }

    #[test]
    fn a_flat_colour_survives_every_mip_level() {

        let chain = mip_chain(&[200u8; 16 * 16 * 4], 16, 16, mip_levels_for(16, 16));
        for (data, w, h) in &chain {
            for (i, b) in data.iter().enumerate() {
                assert!(
                    b.abs_diff(200) <= 1,
                    "level {w}x{h} byte {i} drifted to {b} from 200",
                );
            }
        }
    }

    fn flat_ground_shades_from_the_ground_layer() {
        let Some(px) = render_quad_with_normal([0.0, 1.0, 0.0]) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };

        assert!(px[0] > px[2], "flat ground should take the ground layer (red), got {px:?}");
    }

    #[test]
    fn a_cliff_shades_from_the_rock_layer() {

        let Some(px) = render_quad_with_normal([1.0, 0.0, 0.0]) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        assert!(px[2] > px[0], "a cliff should take the rock layer (blue), got {px:?}");
    }

    #[test]
    fn the_slope_blend_is_gradual_rather_than_a_hard_switch() {

        let a = 31.0_f32.to_radians();
        let Some(px) = render_quad_with_normal([a.sin(), a.cos(), 0.0]) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        assert!(px[0] > 8, "ground layer vanished mid-blend: {px:?}");
        assert!(px[2] > 8, "rock layer absent mid-blend: {px:?}");
    }
}

pub struct TerrainMaterial {
    pub bind_group: BindGroup,
    pub uniform: Buffer,

    pub splat_texture: Texture,

    pub splat_size: [u32; 2],
}

pub fn mip_levels_for(w: u32, h: u32) -> u32 {
    32 - w.max(h).max(1).leading_zeros()
}

pub const DEFAULT_TERRAIN_ROUGHNESS: u8 = 200;

const TERRAIN_NORMAL_VARIANCE_CLAMP: f32 = 0.18;

pub fn roughness_chain_with_normal_variance(
    rough: &[u8],
    normal: &[u8],
    w: u32,
    h: u32,
    levels: u32,
) -> Vec<(Vec<u8>, u32, u32)> {
    let mut chain = mip_chain_linear(rough, w, h, levels);

    let mut vecs: Vec<[f32; 3]> = (0..(w * h) as usize)
        .map(|i| {
            [
                normal[i * 4] as f32 / 255.0 * 2.0 - 1.0,
                normal[i * 4 + 1] as f32 / 255.0 * 2.0 - 1.0,
                normal[i * 4 + 2] as f32 / 255.0 * 2.0 - 1.0,
            ]
        })
        .collect();
    let (mut cw, mut ch) = (w, h);

    for level in 1..chain.len() {
        let (nw, nh) = ((cw / 2).max(1), (ch / 2).max(1));
        let mut next = vec![[0.0f32; 3]; (nw * nh) as usize];
        for y in 0..nh {
            for x in 0..nw {
                let mut acc = [0.0f32; 3];
                for (dx, dy) in [(0u32, 0u32), (1, 0), (0, 1), (1, 1)] {
                    let sx = (x * 2 + dx).min(cw - 1);
                    let sy = (y * 2 + dy).min(ch - 1);
                    let v = vecs[(sy * cw + sx) as usize];
                    acc[0] += v[0];
                    acc[1] += v[1];
                    acc[2] += v[2];
                }
                next[(y * nw + x) as usize] = [acc[0] / 4.0, acc[1] / 4.0, acc[2] / 4.0];
            }
        }
        vecs = next;
        cw = nw;
        ch = nh;

        let (data, lw, lh) = &mut chain[level];
        if *lw != cw || *lh != ch {

            break;
        }
        for i in 0..(cw * ch) as usize {
            let v = vecs[i];
            let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            let sigma2 = if len > 1e-4 { (1.0 - len) / len } else { 1.0 };
            let kernel = (2.0 * sigma2).min(TERRAIN_NORMAL_VARIANCE_CLAMP);
            let base = data[i * 4] as f32 / 255.0;
            let filtered = (base * base + kernel).clamp(0.0, 1.0).sqrt();
            let byte = (filtered * 255.0).round().clamp(0.0, 255.0) as u8;
            data[i * 4] = byte;
            data[i * 4 + 1] = byte;
            data[i * 4 + 2] = byte;
        }
    }
    chain
}

pub fn mip_chain_linear(rgba: &[u8], w: u32, h: u32, levels: u32) -> Vec<(Vec<u8>, u32, u32)> {
    let mut out = vec![(rgba.to_vec(), w, h)];
    for _ in 1..levels {
        let (src, sw, sh) = out.last().unwrap();
        let (dw, dh) = ((sw / 2).max(1), (sh / 2).max(1));
        let mut dst = vec![0u8; (dw * dh * 4) as usize];
        for y in 0..dh {
            for x in 0..dw {
                for c in 0..4 {
                    let fetch = |sx: u32, sy: u32| -> f32 {
                        let i = ((sy.min(sh - 1) * sw + sx.min(sw - 1)) * 4 + c) as usize;
                        src[i] as f32 / 255.0
                    };
                    let (x0, y0) = (x * 2, y * 2);
                    let avg = (fetch(x0, y0) + fetch(x0 + 1, y0) + fetch(x0, y0 + 1)
                        + fetch(x0 + 1, y0 + 1))
                        * 0.25;
                    dst[((y * dw + x) * 4 + c) as usize] =
                        (avg.clamp(0.0, 1.0) * 255.0).round() as u8;
                }
            }
        }
        out.push((dst, dw, dh));
    }
    out
}

pub fn mip_chain(rgba: &[u8], w: u32, h: u32, levels: u32) -> Vec<(Vec<u8>, u32, u32)> {
    fn to_linear(c: u8) -> f32 {
        let s = c as f32 / 255.0;
        if s <= 0.040_45 { s / 12.92 } else { ((s + 0.055) / 1.055).powf(2.4) }
    }
    fn to_srgb(l: f32) -> u8 {
        let s = if l <= 0.003_130_8 { l * 12.92 } else { 1.055 * l.powf(1.0 / 2.4) - 0.055 };
        (s.clamp(0.0, 1.0) * 255.0).round() as u8
    }

    let mut out = vec![(rgba.to_vec(), w, h)];
    for _ in 1..levels {
        let (src, sw, sh) = out.last().unwrap();
        let (dw, dh) = ((sw / 2).max(1), (sh / 2).max(1));
        let mut dst = vec![0u8; (dw * dh * 4) as usize];
        for y in 0..dh {
            for x in 0..dw {
                for c in 0..4 {

                    let fetch = |sx: u32, sy: u32| -> f32 {
                        let i = ((sy.min(sh - 1) * sw + sx.min(sw - 1)) * 4 + c) as usize;
                        if c == 3 { src[i] as f32 / 255.0 } else { to_linear(src[i]) }
                    };
                    let (x0, y0) = (x * 2, y * 2);
                    let avg = (fetch(x0, y0) + fetch(x0 + 1, y0) + fetch(x0, y0 + 1)
                        + fetch(x0 + 1, y0 + 1))
                        * 0.25;
                    dst[((y * dw + x) * 4 + c) as usize] = if c == 3 {
                        (avg.clamp(0.0, 1.0) * 255.0).round() as u8
                    } else {
                        to_srgb(avg)
                    };
                }
            }
        }
        out.push((dst, dw, dh));
    }
    out
}

impl TerrainMaterial {

    pub fn new(
        device: &Device,
        queue: &Queue,
        layout: &BindGroupLayout,
        layers: &[TerrainImage],
        macro_image: &TerrainImage,

        splat: Option<&TerrainImage>,
        sky_occlusion: Option<&TerrainImage>,

        normals: &[Option<TerrainImage>],

        rough: &[Option<TerrainImage>],
        ao: &[Option<TerrainImage>],
        settings: TerrainMaterialUniform,
    ) -> Self {
        Self::new_full(device, queue, layout, layers, macro_image, splat, sky_occlusion, normals, rough, ao, settings, None, None)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_with_shape(
        device: &Device,
        queue: &Queue,
        layout: &BindGroupLayout,
        layers: &[TerrainImage],
        macro_image: &TerrainImage,
        splat: Option<&TerrainImage>,
        sky_occlusion: Option<&TerrainImage>,
        normals: &[Option<TerrainImage>],
        rough: &[Option<TerrainImage>],
        ao: &[Option<TerrainImage>],
        settings: TerrainMaterialUniform,
        shape: Option<&super::ground_shape::GroundShape>,
    ) -> Self {
        Self::new_full(device, queue, layout, layers, macro_image, splat, sky_occlusion, normals, rough, ao, settings, shape, None)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_deformed(
        device: &Device,
        queue: &Queue,
        layout: &BindGroupLayout,
        layers: &[TerrainImage],
        macro_image: &TerrainImage,
        splat: Option<&TerrainImage>,
        sky_occlusion: Option<&TerrainImage>,
        normals: &[Option<TerrainImage>],
        rough: &[Option<TerrainImage>],
        ao: &[Option<TerrainImage>],
        settings: TerrainMaterialUniform,
        deform: Option<&TextureView>,
    ) -> Self {
        Self::new_full(device, queue, layout, layers, macro_image, splat, sky_occlusion, normals, rough, ao, settings, None, deform)
    }

    pub fn flat_deform_view(device: &Device, queue: &Queue) -> TextureView {
        let tex = device.create_texture(&TextureDescriptor {
            label: Some("terrain_deform_flat"),
            size: Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: DEFORM_FORMAT,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            TexelCopyTextureInfo { texture: &tex, mip_level: 0, origin: Origin3d::ZERO, aspect: TextureAspect::All },
            bytemuck::cast_slice(&deform_rows(&[crate::renderer::terrain_pipeline::DEFORM_FLAT_TEXEL[0], 0], 1, 1, 0, 1)),
            TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(DEFORM_TEXEL_BYTES), rows_per_image: Some(1) },
            Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
        );
        tex.create_view(&TextureViewDescriptor::default())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_full(
        device: &Device,
        queue: &Queue,
        layout: &BindGroupLayout,
        layers: &[TerrainImage],
        macro_image: &TerrainImage,
        splat: Option<&TerrainImage>,
        sky_occlusion: Option<&TerrainImage>,
        normals: &[Option<TerrainImage>],
        rough: &[Option<TerrainImage>],
        ao: &[Option<TerrainImage>],
        settings: TerrainMaterialUniform,
        shape: Option<&super::ground_shape::GroundShape>,
        deform: Option<&TextureView>,
    ) -> Self {
        let arrays = TerrainLayerArrays::from_images(device, queue, layers, normals, rough, ao);
        Self::with_arrays(device, queue, layout, &arrays, macro_image, splat, sky_occlusion, settings, shape, deform)
    }

    /// A material over layer arrays already on the GPU ([`TerrainLayerArrays`]):
    /// what a renderer rebuilds when the splat, the settings, the ground's
    /// shape or the mud map change, without uploading the layers again.
    #[allow(clippy::too_many_arguments)]
    pub fn with_arrays(
        device: &Device,
        queue: &Queue,
        layout: &BindGroupLayout,
        arrays: &TerrainLayerArrays,
        macro_image: &TerrainImage,
        splat: Option<&TerrainImage>,
        sky_occlusion: Option<&TerrainImage>,
        settings: TerrainMaterialUniform,
        shape: Option<&super::ground_shape::GroundShape>,
        deform: Option<&TextureView>,
    ) -> Self {
        let macro_tex = device.create_texture(&TextureDescriptor {
            label: Some("terrain_macro"),
            size: Extent3d { width: macro_image.width, height: macro_image.height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,

            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            TexelCopyTextureInfo {
                texture: &macro_tex,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            &macro_image.rgba,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * macro_image.width),
                rows_per_image: Some(macro_image.height),
            },
            Extent3d { width: macro_image.width, height: macro_image.height, depth_or_array_layers: 1 },
        );

        let sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("terrain_sampler"),
            address_mode_u: AddressMode::Repeat,
            address_mode_v: AddressMode::Repeat,
            address_mode_w: AddressMode::Repeat,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: MipmapFilterMode::Linear,

            anisotropy_clamp: 8,
            ..Default::default()
        });

        let splat_image = splat.unwrap_or(&NO_SPLAT);
        let splat_tex = device.create_texture(&TextureDescriptor {
            label: Some("terrain_splat"),
            size: Extent3d {
                width: splat_image.width,
                height: splat_image.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            TexelCopyTextureInfo {
                texture: &splat_tex,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            &splat_image.rgba,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * splat_image.width),
                rows_per_image: Some(splat_image.height),
            },
            Extent3d {
                width: splat_image.width,
                height: splat_image.height,
                depth_or_array_layers: 1,
            },
        );

        let occ_image = sky_occlusion.unwrap_or(&FULL_SKY);
        let occ_layers = (occ_image.rgba.len() / (4 * occ_image.width as usize * occ_image.height as usize).max(1)).max(1) as u32;
        let sky_occ_tex = device.create_texture(&TextureDescriptor {
            label: Some("terrain_sky_occlusion"),
            size: Extent3d {
                width: occ_image.width,
                height: occ_image.height,
                depth_or_array_layers: occ_layers,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,

            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            TexelCopyTextureInfo {
                texture: &sky_occ_tex,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            &occ_image.rgba[..(4 * occ_image.width * occ_image.height * occ_layers) as usize],
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * occ_image.width),
                rows_per_image: Some(occ_image.height),
            },
            Extent3d {
                width: occ_image.width,
                height: occ_image.height,
                depth_or_array_layers: occ_layers,
            },
        );

        let mut settings = settings;
        settings.use_splat = if splat.is_some() { 1.0 } else { 0.0 };

        let placeholder = super::ground_shape::GroundShape { size: [1, 1], texels: vec![[0.0, 1.0, 0.0, 0.0]], ..Default::default() };
        let shape_image = shape.filter(|s| s.size[0] >= 2 && s.size[1] >= 2).unwrap_or(&placeholder);
        settings.shape = match shape {
            Some(s) if s.size[0] >= 2 && s.size[1] >= 2 => [s.uv_cells[0], s.uv_cells[1], 1.0 / s.cell[0], 1.0 / s.cell[1]],
            _ => [0.0; 4],
        };
        let shape_extent = Extent3d { width: shape_image.size[0], height: shape_image.size[1], depth_or_array_layers: 1 };
        let shape_tex = device.create_texture(&TextureDescriptor {
            label: Some("terrain_ground_shape"),
            size: shape_extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba32Float,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            TexelCopyTextureInfo { texture: &shape_tex, mip_level: 0, origin: Origin3d::ZERO, aspect: TextureAspect::All },
            shape_image.bytes(),
            TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(16 * shape_image.size[0]), rows_per_image: Some(shape_image.size[1]) },
            shape_extent,
        );

        let uniform = device.create_buffer(&BufferDescriptor {
            label: Some("terrain_material_uniform"),
            size: std::mem::size_of::<TerrainMaterialUniform>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform, 0, bytemuck::bytes_of(&settings));

        let flat_deform;
        let deform_view = match deform {
            Some(v) => v,
            None => {
                flat_deform = Self::flat_deform_view(device, queue);
                &flat_deform
            }
        };
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("terrain_material"),
            layout,
            entries: &[
                BindGroupEntry { binding: 10, resource: BindingResource::TextureView(deform_view) },
                BindGroupEntry { binding: 0, resource: BindingResource::TextureView(&arrays.albedo) },
                BindGroupEntry { binding: 1, resource: BindingResource::Sampler(&sampler) },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(
                        &macro_tex.create_view(&TextureViewDescriptor::default()),
                    ),
                },
                BindGroupEntry { binding: 3, resource: uniform.as_entire_binding() },
                BindGroupEntry {
                    binding: 4,
                    resource: BindingResource::TextureView(
                        &splat_tex.create_view(&TextureViewDescriptor::default()),
                    ),
                },
                BindGroupEntry { binding: 5, resource: BindingResource::TextureView(&arrays.normals) },
                BindGroupEntry {
                    binding: 6,
                    resource: BindingResource::TextureView(&sky_occ_tex.create_view(&TextureViewDescriptor {
                        dimension: Some(TextureViewDimension::D2Array),
                        ..Default::default()
                    })),
                },
                BindGroupEntry { binding: 7, resource: BindingResource::TextureView(&arrays.rough) },
                BindGroupEntry { binding: 8, resource: BindingResource::TextureView(&arrays.ao) },
                BindGroupEntry { binding: 9, resource: BindingResource::TextureView(&shape_tex.create_view(&TextureViewDescriptor::default())) },
            ],
        });

        let splat_size = [splat_image.width, splat_image.height];
        Self { bind_group, uniform, splat_texture: splat_tex, splat_size }
    }

    pub fn update_splat(&self, queue: &Queue, image: &TerrainImage) -> bool {
        if [image.width, image.height] != self.splat_size {
            return false;
        }
        queue.write_texture(
            TexelCopyTextureInfo {
                texture: &self.splat_texture,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            &image.rgba,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * image.width),
                rows_per_image: Some(image.height),
            },
            Extent3d { width: image.width, height: image.height, depth_or_array_layers: 1 },
        );
        true
    }

    pub fn fallback(device: &Device, queue: &Queue, layout: &BindGroupLayout) -> Self {
        Self::fallback_with_splat(device, queue, layout, None, None)
    }

    pub fn fallback_with_splat(
        device: &Device,
        queue: &Queue,
        layout: &BindGroupLayout,
        splat: Option<&TerrainImage>,
        sky_occlusion: Option<&TerrainImage>,
    ) -> Self {
        Self::from_layers(
            device, queue, layout, &[None, None, None, None], &[None, None, None, None], splat,
            sky_occlusion,
        )
    }

    pub fn from_layers(
        device: &Device,
        queue: &Queue,
        layout: &BindGroupLayout,
        layers: &[Option<TerrainImage>],
        normals: &[Option<TerrainImage>],
        splat: Option<&TerrainImage>,
        sky_occlusion: Option<&TerrainImage>,
    ) -> Self {
        Self::from_layers_with(
            device,
            queue,
            layout,
            layers,
            normals,

            &[],
            &[],
            splat,
            sky_occlusion,
            TerrainMaterialUniform::default(),
        )
    }

    pub fn from_layers_with(
        device: &Device,
        queue: &Queue,
        layout: &BindGroupLayout,
        layers: &[Option<TerrainImage>],
        normals: &[Option<TerrainImage>],
        rough: &[Option<TerrainImage>],
        ao: &[Option<TerrainImage>],
        splat: Option<&TerrainImage>,
        sky_occlusion: Option<&TerrainImage>,
        settings: TerrainMaterialUniform,
    ) -> Self {
        Self::from_layers_full(device, queue, layout, layers, normals, rough, ao, splat, sky_occlusion, settings, None, None)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_layers_with_shape(
        device: &Device,
        queue: &Queue,
        layout: &BindGroupLayout,
        layers: &[Option<TerrainImage>],
        normals: &[Option<TerrainImage>],
        rough: &[Option<TerrainImage>],
        ao: &[Option<TerrainImage>],
        splat: Option<&TerrainImage>,
        sky_occlusion: Option<&TerrainImage>,
        settings: TerrainMaterialUniform,
        shape: Option<&super::ground_shape::GroundShape>,
    ) -> Self {
        Self::from_layers_full(device, queue, layout, layers, normals, rough, ao, splat, sky_occlusion, settings, shape, None)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_layers_deformed(
        device: &Device,
        queue: &Queue,
        layout: &BindGroupLayout,
        layers: &[Option<TerrainImage>],
        normals: &[Option<TerrainImage>],
        rough: &[Option<TerrainImage>],
        ao: &[Option<TerrainImage>],
        splat: Option<&TerrainImage>,
        sky_occlusion: Option<&TerrainImage>,
        settings: TerrainMaterialUniform,
        deform: Option<&TextureView>,
    ) -> Self {
        Self::from_layers_full(device, queue, layout, layers, normals, rough, ao, splat, sky_occlusion, settings, None, deform)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_layers_full(
        device: &Device,
        queue: &Queue,
        layout: &BindGroupLayout,
        layers: &[Option<TerrainImage>],
        normals: &[Option<TerrainImage>],
        rough: &[Option<TerrainImage>],
        ao: &[Option<TerrainImage>],
        splat: Option<&TerrainImage>,
        sky_occlusion: Option<&TerrainImage>,
        settings: TerrainMaterialUniform,
        shape: Option<&super::ground_shape::GroundShape>,
        deform: Option<&TextureView>,
    ) -> Self {
        let (w, h) = layers
            .iter()
            .flatten()
            .map(|l| (l.width, l.height))
            .next()
            .unwrap_or((1, 1));

        let _ = (w, h);
        let filled = filled_layers(layers);

        Self::new_full(
            device,
            queue,
            layout,
            &filled,
            &solid_image([128, 128, 128], 1, 1),
            splat,
            sky_occlusion,
            normals,
            rough,
            ao,
            settings,
            shape,
            deform,
        )
    }

}

#[derive(Clone)]
pub struct TerrainImage {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

impl TerrainImage {

    pub fn with_stationary_masks(&self, masks: &[TerrainImage]) -> Option<TerrainImage> {
        if masks.iter().any(|m| m.width != self.width || m.height != self.height || m.rgba.len() != self.rgba.len()) {
            return None;
        }
        let mut rgba = self.rgba.clone();
        for m in masks {
            rgba.extend_from_slice(&m.rgba);
        }
        Some(TerrainImage { width: self.width, height: self.height, rgba })
    }

    pub fn sun_baked_everywhere(&self) -> bool {
        !self.rgba.is_empty() && self.rgba.chunks_exact(4).all(|texel| texel[3] < 128)
    }
}

#[cfg(test)]
mod material_tests {
    use super::tests::*;

    #[test]
    fn a_stationary_lamp_on_the_ground_reads_its_mask() {
        let map = super::TerrainImage { width: 1, height: 1, rgba: vec![255, 128, 0, 255] };
        let mask = |d: u8| super::TerrainImage { width: 1, height: 1, rgba: vec![d, 0, 255, 255] };
        let layered = |d: u8| map.with_stationary_masks(&[mask(d)]).unwrap();

        let up = [0.0, 1.0, 0.0];
        let render = |sky: &super::TerrainImage, light: Option<Option<u8>>| {
            render_quad_lit_by(up, Palette::Test, None, Some(sky), &[None, None, None, None], light)
        };
        let Some(dark) = render(&map, None) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let lamp = render(&map, Some(None)).unwrap();
        let shadowed = render(&layered(0), Some(Some(0))).unwrap();
        let lit = render(&layered(255), Some(Some(0))).unwrap();
        assert!(lamp[0] > dark[0] + 10, "the test lamp does not light the ground: {lamp:?} vs {dark:?}");
        assert_eq!(shadowed, dark, "a lamp its mask hides still lit the ground");
        assert_eq!(lit, lamp, "a lamp its mask shows is not the unmasked lamp");

        let other_channel = render(&layered(0), Some(Some(1))).unwrap();
        assert_eq!(other_channel, lamp, "a lamp on another channel read this one's shadow");
    }
    use super::*;

    #[test]
    fn the_fallback_material_selects_layers_the_same_way() {
        let Some(flat) = render_quad_with_fallback_material([0.0, 1.0, 0.0]) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let cliff = render_quad_with_fallback_material([1.0, 0.05, 0.0]).unwrap();

        let greenness = |p: [u8; 4]| p[1] as i32 - p[2] as i32;
        assert!(
            greenness(flat) > greenness(cliff) + 4 && greenness(flat) > 4,
            "flat ground should read greener than a cliff: flat {flat:?} cliff {cliff:?}",
        );
        assert!(flat[3] == 255 && cliff[3] == 255, "terrain must be opaque");
    }

    #[test]
    fn a_short_layer_list_pads_to_a_full_array() {
        let Some((device, queue)) = headless_gpu() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let layout = material_bind_group_layout(&device);
        let one = TerrainImage { width: 1, height: 1, rgba: vec![10, 20, 30, 255] };
        let err_scope_1 = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let _m = TerrainMaterial::new(
            &device, &queue, &layout,
            std::slice::from_ref(&one),
            &one,
            None,
            None,
            &[None, None, None, None],
            &[],
            &[],
            TerrainMaterialUniform::default(),
        );
        assert!(
            pollster::block_on(err_scope_1.pop()).is_none(),
            "a single-layer terrain material must still bind validly",
        );
    }
}

static NO_SPLAT: std::sync::LazyLock<TerrainImage> = std::sync::LazyLock::new(|| TerrainImage {
    width: 1,
    height: 1,
    rgba: vec![255, 0, 0, 0],
});

static FULL_SKY: std::sync::LazyLock<TerrainImage> = std::sync::LazyLock::new(|| TerrainImage {
    width: 1,
    height: 1,
    rgba: vec![255, 255, 255, 255],
});

#[cfg(test)]
mod authored_splat_tests {
    use super::tests::*;
    use super::*;

    fn splat(rgba: [u8; 4]) -> TerrainImage {
        TerrainImage { width: 1, height: 1, rgba: rgba.to_vec() }
    }

    const FLAT: [f32; 3] = [0.0, 1.0, 0.0];
    const CLIFF: [f32; 3] = [1.0, 0.05, 0.0];

    #[test]
    fn authored_weights_override_the_slope_blend() {
        let Some(unauthored) = render_quad_with_normal(FLAT) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        assert!(unauthored[0] > unauthored[2], "flat ground should be red without a map: {unauthored:?}");

        let painted = render_quad_with_splat(FLAT, &splat([0, 255, 0, 0])).unwrap();
        assert!(
            painted[2] > painted[0],
            "a map demanding layer 1 must beat the slope rule on flat ground: {painted:?}",
        );
    }

    #[test]
    fn authored_weights_override_a_cliff_too() {
        let Some(unauthored) = render_quad_with_normal(CLIFF) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        assert!(unauthored[2] > unauthored[0], "a cliff should be blue without a map: {unauthored:?}");

        let painted = render_quad_with_splat(CLIFF, &splat([255, 0, 0, 0])).unwrap();
        assert!(
            painted[0] > painted[2],
            "a map demanding layer 0 must beat the slope rule on a cliff: {painted:?}",
        );
    }

    #[test]
    fn a_mixed_texel_blends_rather_than_picking_a_winner() {
        let Some(mixed) = render_quad_with_splat(FLAT, &splat([128, 0, 128, 0])) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        assert!(mixed[0] > 30, "layer 0 (red) should contribute: {mixed:?}");
        assert!(mixed[1] > 30, "layer 2 (green) should contribute: {mixed:?}");
        assert!(mixed[2] < mixed[0], "layer 1 (blue) was not painted: {mixed:?}");
    }

    #[test]
    fn unnormalised_weights_are_normalised_not_amplified() {

        let Some(normal_sum) = render_quad_with_splat(FLAT, &splat([128, 0, 128, 0])) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let double_sum = render_quad_with_splat(FLAT, &splat([255, 0, 255, 0])).unwrap();

        for channel in 0..3 {
            let delta = normal_sum[channel].abs_diff(double_sum[channel]);
            assert!(
                delta <= 4,
                "weights summing to 510 must shade like weights summing to 255 \
                 (channel {channel}: {normal_sum:?} vs {double_sum:?})",
            );
        }
    }

    #[test]
    fn an_empty_texel_does_not_produce_a_divide_by_zero() {
        let Some(px) = render_quad_with_splat(FLAT, &splat([0, 0, 0, 0])) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        assert_eq!(px[3], 255, "terrain must stay opaque: {px:?}");
        for channel in 0..3 {
            assert!(px[channel] < 250, "an empty texel must not blow out: {px:?}");
        }
    }
}

impl TerrainImage {

    pub fn load(path: &std::path::Path) -> Option<Self> {
        let decoded = match ::image::open(path) {
            Ok(d) => d.to_rgba8(),
            Err(e) => {
                log::warn!("terrain texture {}: {e}", path.display());
                return None;
            }
        };
        Some(Self {
            width: decoded.width(),
            height: decoded.height(),
            rgba: decoded.into_raw(),
        })
    }
}

/// The four layers' colours at one size -- the largest layer's, so a set
/// mixing a 1K grass with a 2K mud keeps the mud's detail -- a missing one
/// its fallback colour: what the layer arrays are built from
/// (`TerrainMaterial::from_layers_full`).
pub fn filled_layers(layers: &[Option<TerrainImage>]) -> Vec<TerrainImage> {
    let (w, h) = layers.iter().flatten().map(|l| (l.width, l.height)).max_by_key(|(w, h)| w * h).unwrap_or((1, 1));
    FALLBACK_LAYER_COLOURS
        .iter()
        .enumerate()
        .map(|(i, rgb)| match layers.get(i).and_then(|l| l.as_ref()) {
            Some(img) if img.width == w && img.height == h => img.clone_image(),
            Some(img) => resample(img, w, h),
            None => solid_image(*rgb, w, h),
        })
        .collect()
}

/// The four arrays of a ground's layers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerMap {
    Albedo,
    Normal,
    Rough,
    Ao,
}

impl LayerMap {
    pub const ALL: [Self; 4] = [Self::Albedo, Self::Normal, Self::Rough, Self::Ao];

    pub fn name(self) -> &'static str {
        match self {
            Self::Albedo => "albedo",
            Self::Normal => "normal",
            Self::Rough => "rough",
            Self::Ao => "ao",
        }
    }

    /// Colour is stored in sRGB; everything else is data.
    pub fn srgb(self) -> bool {
        self == Self::Albedo
    }

    fn label(self) -> &'static str {
        match self {
            Self::Albedo => "terrain_layers",
            Self::Normal => "terrain_normals",
            Self::Rough => "terrain_rough",
            Self::Ao => "terrain_ao",
        }
    }
}

/// Every mip of one layer of one map, as the arrays hold it. `layers` is
/// [`filled_layers`]; the rest as loaded (`load_terrain_normals` and the
/// others), fitted to the layers' size or filled with their defaults. The one
/// filtering both the RGBA upload and the ASTC encoder (`bin/terrain_astc`)
/// use, so a compressed set is that upload, encoded.
pub fn layer_mips(map: LayerMap, slot: usize, layers: &[TerrainImage], normals: &[Option<TerrainImage>], rough: &[Option<TerrainImage>], ao: &[Option<TerrainImage>]) -> Vec<(Vec<u8>, u32, u32)> {
    let (w, h) = (layers[0].width, layers[0].height);
    let levels = mip_levels_for(w, h);
    let fit = |src: &[Option<TerrainImage>], fallback: [u8; 3]| -> TerrainImage {
        match src.get(slot).and_then(|x| x.as_ref()) {
            Some(img) if img.width == w && img.height == h => img.clone_image(),
            Some(img) => resample(img, w, h),
            None => solid_image(fallback, w, h),
        }
    };
    match map {
        LayerMap::Albedo => mip_chain(&layers.get(slot).unwrap_or_else(|| layers.last().unwrap()).rgba, w, h, levels),
        LayerMap::Normal => mip_chain_linear(&fit(normals, FLAT_NORMAL).rgba, w, h, levels),
        LayerMap::Rough => roughness_chain_with_normal_variance(&fit(rough, [DEFAULT_TERRAIN_ROUGHNESS; 3]).rgba, &fit(normals, FLAT_NORMAL).rgba, w, h, levels),
        LayerMap::Ao => mip_chain_linear(&fit(ao, [255; 3]).rgba, w, h, levels),
    }
}

/// THE GROUND'S LAYER ARRAYS on the GPU -- colour, normal, roughness, and
/// occlusion with height -- four layers each with every mip. Built once per
/// texture set and shared by every material rebuild
/// ([`TerrainMaterial::with_arrays`]): rebuilding them with the material put
/// four 2K arrays' uploads, mip work and copies on every splat, settings and
/// mud-map change, and a 2K set's load past the Quest 3's memory (2026-10-20).
/// From the layer images ([`Self::from_images`]), or from a set encoded ahead
/// as ASTC ([`Self::from_astc`], `bin/terrain_astc`): 8 bits a texel against
/// 32, and no decoding on the headset.
pub struct TerrainLayerArrays {
    pub albedo: TextureView,
    pub normals: TextureView,
    pub rough: TextureView,
    pub ao: TextureView,
    pub size: [u32; 2],
    pub compressed: bool,
}

/// The file an encoded set keeps one map in: `<set>/astc/<map>.ssastc`.
pub fn astc_path(set: &std::path::Path, map: LayerMap) -> std::path::PathBuf {
    set.join("astc").join(format!("{}.ssastc", map.name()))
}

const ASTC_MAGIC: &[u8; 8] = b"SSASTC1\0";

/// One map of an encoded set: a D2 array's every layer and mip, as ASTC
/// blocks of `block` texels (16 bytes each).
#[derive(Debug, Clone, PartialEq)]
pub struct AstcArray {
    pub block: [u8; 2],
    pub srgb: bool,
    pub width: u32,
    pub height: u32,
    pub levels: u32,
    /// `[layer][level]`, each level's blocks row by row.
    pub data: Vec<Vec<Vec<u8>>>,
}

impl AstcArray {
    /// The bytes one level of one layer holds.
    pub fn level_bytes(block: [u8; 2], width: u32, height: u32, level: u32) -> usize {
        let (w, h) = ((width >> level).max(1), (height >> level).max(1));
        (w.div_ceil(block[0] as u32) * h.div_ceil(block[1] as u32) * 16) as usize
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(ASTC_MAGIC);
        out.extend_from_slice(&[self.block[0], self.block[1], self.srgb as u8, 0]);
        for v in [self.width, self.height, self.levels, self.data.len() as u32] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        for layer in &self.data {
            for level in layer {
                out.extend_from_slice(&(level.len() as u32).to_le_bytes());
                out.extend_from_slice(level);
            }
        }
        out
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        let head = bytes.get(..28).ok_or("too short for its header")?;
        if &head[..8] != ASTC_MAGIC {
            return Err("not an SSASTC1 file".into());
        }
        let block = [head[8], head[9]];
        if !matches!(block, [4, 4] | [5, 5] | [6, 6] | [8, 8]) {
            return Err(format!("block {block:?} is not one this reads"));
        }
        let word = |i: usize| u32::from_le_bytes(head[12 + 4 * i..16 + 4 * i].try_into().unwrap());
        let (width, height, levels, layers) = (word(0), word(1), word(2), word(3));
        if width == 0 || height == 0 || levels == 0 || levels > mip_levels_for(width, height) || layers == 0 || layers > 16 {
            return Err(format!("{width}x{height}, {levels} levels, {layers} layers is not an array"));
        }
        let mut at = 28;
        let mut data = Vec::with_capacity(layers as usize);
        for _ in 0..layers {
            let mut chain = Vec::with_capacity(levels as usize);
            for level in 0..levels {
                let len = u32::from_le_bytes(bytes.get(at..at + 4).ok_or("cut short")?.try_into().unwrap()) as usize;
                if len != Self::level_bytes(block, width, height, level) {
                    return Err(format!("level {level} is {len} bytes, not {}", Self::level_bytes(block, width, height, level)));
                }
                chain.push(bytes.get(at + 4..at + 4 + len).ok_or("cut short")?.to_vec());
                at += 4 + len;
            }
            data.push(chain);
        }
        Ok(Self { block, srgb: head[10] != 0, width, height, levels, data })
    }

    pub fn format(&self) -> Option<TextureFormat> {
        let block = match self.block {
            [4, 4] => AstcBlock::B4x4,
            [5, 5] => AstcBlock::B5x5,
            [6, 6] => AstcBlock::B6x6,
            [8, 8] => AstcBlock::B8x8,
            _ => return None,
        };
        Some(TextureFormat::Astc { block, channel: if self.srgb { AstcChannel::UnormSrgb } else { AstcChannel::Unorm } })
    }
}

/// The block every encoder here writes: 4x4, the finest ASTC has, so
/// normals keep their detail.
pub const ASTC_BLOCK: [u8; 2] = [4, 4];

/// One level of one layer encoded by ARM's `astcenc` in [`ASTC_BLOCK`]s:
/// colour as sRGB, the rest linear. Dev tooling for the encoders
/// (`bin/terrain_astc`, `bin/brush_astc`); nothing on the headset runs it.
/// `scratch` is a folder for its image in and blocks out.
pub fn astcenc_level(astcenc: &std::path::Path, quality: &str, srgb: bool, rgba: &[u8], w: u32, h: u32, scratch: &std::path::Path) -> Result<Vec<u8>, String> {
    // astcenc's .astc files start with a 16-byte header before the blocks.
    const ASTC_HEADER: usize = 16;
    let png = scratch.join("level.png");
    let out = scratch.join("level.astc");
    ::image::save_buffer(&png, rgba, w, h, ::image::ExtendedColorType::Rgba8).map_err(|e| format!("write {}: {e}", png.display()))?;
    let status = std::process::Command::new(astcenc)
        .arg(if srgb { "-cs" } else { "-cl" })
        .arg(&png)
        .arg(&out)
        .arg(format!("{}x{}", ASTC_BLOCK[0], ASTC_BLOCK[1]))
        .arg(format!("-{quality}"))
        .arg("-silent")
        .status()
        .map_err(|e| format!("run {}: {e}", astcenc.display()))?;
    if !status.success() {
        return Err(format!("astcenc failed on a {w}x{h} level ({status})"));
    }
    let bytes = std::fs::read(&out).map_err(|e| format!("read {}: {e}", out.display()))?;
    let blocks = bytes.get(ASTC_HEADER..).ok_or("astcenc wrote no blocks")?.to_vec();
    let want = AstcArray::level_bytes(ASTC_BLOCK, w, h, 0);
    if blocks.len() != want {
        return Err(format!("a {w}x{h} level came out {} bytes, not {want}", blocks.len()));
    }
    Ok(blocks)
}

fn array_view(texture: &Texture) -> TextureView {
    texture.create_view(&TextureViewDescriptor { dimension: Some(TextureViewDimension::D2Array), ..Default::default() })
}

impl TerrainLayerArrays {
    /// Upload the layer images, every mip of each made here ([`layer_mips`]).
    pub fn from_images(device: &Device, queue: &Queue, layers: &[TerrainImage], normals: &[Option<TerrainImage>], rough: &[Option<TerrainImage>], ao: &[Option<TerrainImage>]) -> Self {
        assert!(!layers.is_empty(), "terrain material needs at least one layer");
        let (w, h) = (layers[0].width, layers[0].height);
        assert!(layers.iter().all(|l| l.width == w && l.height == h), "all terrain layers must share one size to live in a D2Array");
        let mip_level_count = mip_levels_for(w, h);
        let make = |map: LayerMap| {
            let texture = device.create_texture(&TextureDescriptor {
                label: Some(map.label()),
                size: Extent3d { width: w, height: h, depth_or_array_layers: 4 },
                mip_level_count,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: if map.srgb() { TextureFormat::Rgba8UnormSrgb } else { TextureFormat::Rgba8Unorm },
                usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
                view_formats: &[],
            });
            for slot in 0..4 {
                for (level, (data, lw, lh)) in layer_mips(map, slot, layers, normals, rough, ao).iter().enumerate() {
                    queue.write_texture(
                        TexelCopyTextureInfo { texture: &texture, mip_level: level as u32, origin: Origin3d { x: 0, y: 0, z: slot as u32 }, aspect: TextureAspect::All },
                        data,
                        TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(4 * lw), rows_per_image: Some(*lh) },
                        Extent3d { width: *lw, height: *lh, depth_or_array_layers: 1 },
                    );
                }
            }
            array_view(&texture)
        };
        Self { albedo: make(LayerMap::Albedo), normals: make(LayerMap::Normal), rough: make(LayerMap::Rough), ao: make(LayerMap::Ao), size: [w, h], compressed: false }
    }

    /// Upload a set encoded ahead (`<set>/astc/*.ssastc`, `bin/terrain_astc`).
    /// None when the device has no ASTC, or a map is missing, unreadable or
    /// unlike the others: the caller then uploads the images.
    pub fn from_astc(device: &Device, queue: &Queue, set: &std::path::Path) -> Option<Self> {
        if !device.features().contains(Features::TEXTURE_COMPRESSION_ASTC) {
            return None;
        }
        let mut views = Vec::with_capacity(4);
        let mut size = None;
        for map in LayerMap::ALL {
            let path = astc_path(set, map);
            let bytes = std::fs::read(&path).ok()?;
            let a = AstcArray::from_bytes(&bytes).map_err(|e| log::warn!("terrain: {}: {e}", path.display())).ok()?;
            if a.srgb != map.srgb() || a.data.len() != 4 || a.levels != mip_levels_for(a.width, a.height) || size.is_some_and(|s| s != [a.width, a.height]) {
                log::warn!("terrain: {} does not fit the set", path.display());
                return None;
            }
            size = Some([a.width, a.height]);
            let format = a.format()?;
            let texture = device.create_texture(&TextureDescriptor {
                label: Some(map.label()),
                size: Extent3d { width: a.width, height: a.height, depth_or_array_layers: 4 },
                mip_level_count: a.levels,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format,
                usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let (bw, bh) = (a.block[0] as u32, a.block[1] as u32);
            for (slot, chain) in a.data.iter().enumerate() {
                for (level, data) in chain.iter().enumerate() {
                    let (w, h) = ((a.width >> level).max(1), (a.height >> level).max(1));
                    let (cols, rows) = (w.div_ceil(bw), h.div_ceil(bh));
                    queue.write_texture(
                        TexelCopyTextureInfo { texture: &texture, mip_level: level as u32, origin: Origin3d { x: 0, y: 0, z: slot as u32 }, aspect: TextureAspect::All },
                        data,
                        TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(cols * 16), rows_per_image: Some(rows) },
                        Extent3d { width: cols * bw, height: rows * bh, depth_or_array_layers: 1 },
                    );
                }
            }
            views.push(array_view(&texture));
        }
        let size = size?;
        let mut views = views.into_iter();
        Some(Self { albedo: views.next()?, normals: views.next()?, rough: views.next()?, ao: views.next()?, size, compressed: true })
    }
}

/// A layer's colour at no more than `side` texels a side, filtered as its
/// mips are: what the CPU keeps of a layer (the ground map reads it) once the
/// arrays are on the GPU.
pub fn small_copy(img: &TerrainImage, side: u32) -> TerrainImage {
    if img.width <= side && img.height <= side {
        return img.clone_image();
    }
    let chain = mip_chain(&img.rgba, img.width, img.height, mip_levels_for(img.width, img.height));
    let (rgba, width, height) = chain.into_iter().find(|(_, w, h)| *w <= side && *h <= side).expect("a 1x1 mip at the end");
    TerrainImage { width, height, rgba }
}

pub const TERRAIN_LAYER_FILES: [&str; 4] = ["ground.jpg", "rock.jpg", "high.jpg", "sediment.jpg"];

pub const TERRAIN_NORMAL_FILES: [&str; 4] =
    ["ground_n.jpg", "rock_n.jpg", "high_n.jpg", "sediment_n.jpg"];

pub const TERRAIN_ROUGH_FILES: [&str; 4] =
    ["ground_r.jpg", "rock_r.jpg", "high_r.jpg", "sediment_r.jpg"];

pub const TERRAIN_AO_FILES: [&str; 4] =
    ["ground_ao.jpg", "rock_ao.jpg", "high_ao.jpg", "sediment_ao.jpg"];

pub fn load_terrain_layers(dir: &std::path::Path) -> Vec<Option<TerrainImage>> {
    TERRAIN_LAYER_FILES
        .iter()
        .map(|name| TerrainImage::load(&dir.join(name)))
        .collect()
}

pub fn load_terrain_rough(dir: &std::path::Path) -> Vec<Option<TerrainImage>> {
    TERRAIN_ROUGH_FILES
        .iter()
        .enumerate()
        .map(|(layer, name)| {
            let mut img = TerrainImage::load(&dir.join(name));
            if let (Some(img), Some((lo, hi))) = (img.as_mut(), terrain_roughness_range(dir, layer)) {
                for px in img.rgba.chunks_exact_mut(4) {
                    for c in &mut px[..3] {
                        *c = ((lo + (*c as f32 / 255.0) * (hi - lo)).clamp(0.0, 1.0) * 255.0).round() as u8;
                    }
                }
                log::info!("terrain: layer {layer}'s roughness read as {lo:.2}..{hi:.2}");
            }
            img
        })
        .collect()
}

pub fn terrain_roughness_range(dir: &std::path::Path, layer: usize) -> Option<(f32, f32)> {
    let read = |path: std::path::PathBuf| -> Option<serde_json::Value> { serde_json::from_str(&std::fs::read_to_string(path).ok()?).ok() };
    let role = TERRAIN_LAYER_FILES.get(layer)?.strip_suffix(".jpg")?;
    let sources = read(dir.join("sources.json"))?;
    let id = sources.get(role)?.get("assetId")?.as_str()?;
    let overrides = read(dir.parent()?.parent()?.join("material_overrides.json"))?;
    let r = overrides.get(id)?.get("roughness")?;
    let lo = r.get("min")?.as_f64()? as f32;
    let hi = r.get("max")?.as_f64()? as f32;
    (lo.is_finite() && hi.is_finite()).then_some((lo.clamp(0.0, 1.0), hi.clamp(0.0, 1.0)))
}

pub fn load_terrain_ao(dir: &std::path::Path) -> Vec<Option<TerrainImage>> {
    TERRAIN_AO_FILES
        .iter()
        .zip(TERRAIN_DISPLACEMENT_FILES.iter())
        .map(|(ao_name, d_name)| {
            let ao = TerrainImage::load(&dir.join(ao_name));
            let disp = TerrainImage::load(&dir.join(d_name));
            with_height_in_green(ao, disp)
        })
        .collect()
}

pub const TERRAIN_DISPLACEMENT_FILES: [&str; 4] =
    ["ground_d.jpg", "rock_d.jpg", "high_d.jpg", "sediment_d.jpg"];

pub fn with_height_in_green(ao: Option<TerrainImage>, disp: Option<TerrainImage>) -> Option<TerrainImage> {
    let mut out = match (ao, &disp) {
        (Some(ao), _) => ao,
        (None, Some(d)) => solid_image([255, 255, 255], d.width, d.height),
        (None, None) => return None,
    };
    let height = match disp {
        Some(d) if d.width == out.width && d.height == out.height => d,
        Some(d) => resample(&d, out.width, out.height),
        None => solid_image([255, 255, 255], out.width, out.height),
    };
    for (px, h) in out.rgba.chunks_exact_mut(4).zip(height.rgba.chunks_exact(4)) {
        px[1] = h[0];
    }
    Some(out)
}

pub fn load_terrain_settings(dir: &std::path::Path) -> TerrainMaterialUniform {
    let mut out = TerrainMaterialUniform::default();
    let path = dir.join(TERRAIN_SETTINGS_FILE);
    let text = match std::fs::read_to_string(&path) {
        Ok(t) => t,
        Err(_) => return out,
    };
    let raw: serde_json::Value = match serde_json::from_str(&text) {
        Ok(v) => v,
        Err(e) => {
            log::warn!("{}: {e} -- using default terrain settings", path.display());
            return out;
        }
    };
    if let Some(values) = raw.get("repeat").and_then(|r| r.as_array()) {
        for (i, slot) in out.repeat.iter_mut().enumerate() {
            if let Some(v) = values.get(i).and_then(|v| v.as_f64()) {

                *slot = (v as f32).max(MIN_REPEAT);
            }
        }
    }
    if let Some(v) = raw.get("normal_strength").and_then(|v| v.as_f64()) {
        out.normal_strength = (v as f32).clamp(0.0, 4.0);
    }
    if let Some(v) = raw.get("parallax").and_then(|v| v.as_f64()) {
        out.parallax = (v as f32).clamp(0.0, 0.5);
    }

    for (key, into) in [("porosity", &mut out.porosity), ("film", &mut out.film)] {
        if let Some(values) = raw.get(key).and_then(|r| r.as_array()) {
            for (i, slot) in into.iter_mut().enumerate() {
                if let Some(v) = values.get(i).and_then(|v| v.as_f64()) {
                    *slot = (v as f32).clamp(0.0, 1.0);
                }
            }
        }
    }
    if let Some(v) = raw.get("mud_layer").and_then(|v| v.as_i64()) {
        out.mud_layer = if (0..4).contains(&v) { v as f32 } else { -1.0 };
    }
    if let Some(v) = raw.get("deform").and_then(|v| v.as_f64()) {
        out.deform_metres = (v as f32).clamp(0.0, 0.5);
    }
    if let Some(v) = raw.get("wet_dark").and_then(|v| v.as_f64()) {
        out.wet_dark = (v as f32).clamp(0.0, 1.0);
    }
    out
}

pub const TERRAIN_SETTINGS_FILE: &str = "settings.json";

pub const MIN_REPEAT: f32 = 0.05;

pub fn load_terrain_normals(dir: &std::path::Path) -> Vec<Option<TerrainImage>> {
    TERRAIN_NORMAL_FILES
        .iter()
        .map(|name| TerrainImage::load(&dir.join(name)))
        .collect()
}

pub const FALLBACK_LAYER_COLOURS: [[u8; 3]; 4] = [
    [86, 112, 62],
    [104, 100, 94],
    [132, 128, 120],
    [92, 84, 70],
];

pub(crate) fn solid_image(rgb: [u8; 3], width: u32, height: u32) -> TerrainImage {
    let mut rgba = Vec::with_capacity((width * height * 4) as usize);
    for _ in 0..(width * height) {
        rgba.extend_from_slice(&[rgb[0], rgb[1], rgb[2], 255]);
    }
    TerrainImage { width, height, rgba }
}

pub(crate) fn resample(src: &TerrainImage, width: u32, height: u32) -> TerrainImage {
    let mut rgba = Vec::with_capacity((width * height * 4) as usize);
    for y in 0..height {
        let sy = (y as u64 * src.height as u64 / height.max(1) as u64) as u32;
        for x in 0..width {
            let sx = (x as u64 * src.width as u64 / width.max(1) as u64) as u32;
            let at = ((sy.min(src.height - 1) * src.width + sx.min(src.width - 1)) * 4) as usize;
            rgba.extend_from_slice(&src.rgba[at..at + 4]);
        }
    }
    TerrainImage { width, height, rgba }
}

#[cfg(test)]
mod layer_loading_tests {
    use super::tests::headless_gpu;
    use super::*;

    fn img(rgb: [u8; 3], w: u32, h: u32) -> TerrainImage {
        solid_image(rgb, w, h)
    }

    #[test]
    fn a_missing_layer_falls_back_without_taking_the_others_with_it() {
        let Some((device, queue)) = headless_gpu() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let layout = material_bind_group_layout(&device);

        let err_scope_2 = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let _m = TerrainMaterial::from_layers(
            &device,
            &queue,
            &layout,
            &[Some(img([10, 20, 30], 64, 64)), None, Some(img([1, 2, 3], 64, 64)), None],
            &[None, None, None, None],
            None,
            None,
        );
        assert!(
            pollster::block_on(err_scope_2.pop()).is_none(),
            "a partial layer set must still bind validly",
        );
    }

    #[test]
    fn mixed_sizes_are_normalised_rather_than_failing_validation() {
        let Some((device, queue)) = headless_gpu() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let layout = material_bind_group_layout(&device);

        let err_scope_3 = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let _m = TerrainMaterial::from_layers(
            &device,
            &queue,
            &layout,
            &[
                Some(img([10, 20, 30], 64, 64)),
                Some(img([40, 50, 60], 16, 16)),
                None,
                Some(img([70, 80, 90], 128, 32)),
            ],
            &[None, None, None, None],
            None,
            None,
        );
        assert!(
            pollster::block_on(err_scope_3.pop()).is_none(),
            "mixed layer sizes must be normalised, not rejected",
        );
    }

    #[test]
    fn no_layers_at_all_is_the_old_flat_palette() {
        let Some((device, queue)) = headless_gpu() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let layout = material_bind_group_layout(&device);
        let err_scope_4 = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let _m = TerrainMaterial::from_layers(
            &device, &queue, &layout, &[None, None, None, None], &[None, None, None, None], None, None,
        );
        assert!(pollster::block_on(err_scope_4.pop()).is_none());
    }

    #[test]
    fn a_missing_file_reports_none_rather_than_failing() {
        assert!(TerrainImage::load(std::path::Path::new("/nonexistent/ground.jpg")).is_none());
    }

    #[test]
    fn the_layer_file_names_match_the_shader_slots() {
        assert_eq!(TERRAIN_LAYER_FILES.len(), FALLBACK_LAYER_COLOURS.len());
        assert_eq!(TERRAIN_LAYER_FILES[0], "ground.jpg");
        assert_eq!(TERRAIN_LAYER_FILES[1], "rock.jpg");
    }

    fn settings_dir(label: &str, body: Option<&str>) -> std::path::PathBuf {
        let dir = std::env::temp_dir()
            .join(format!("terrain_settings_{label}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        if let Some(text) = body {
            std::fs::write(dir.join(TERRAIN_SETTINGS_FILE), text).unwrap();
        }
        dir
    }

    #[test]
    fn absent_settings_are_the_built_in_defaults() {
        let dir = settings_dir("absent", None);
        let loaded = load_terrain_settings(&dir);
        assert_eq!(loaded.repeat, TerrainMaterialUniform::default().repeat);
        assert_eq!(loaded.normal_strength, 1.0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn repeat_comes_from_the_file() {
        let dir = settings_dir("repeat", Some(r#"{"repeat": [2.0, 3.5, 12.0, 6.0]}"#));
        assert_eq!(load_terrain_settings(&dir).repeat, [2.0, 3.5, 12.0, 6.0]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_unmentioned_key_keeps_its_default() {
        let dir = settings_dir("partial", Some(r#"{"repeat": [1.0, 1.0, 1.0, 1.0]}"#));
        let loaded = load_terrain_settings(&dir);
        assert_eq!(loaded.repeat, [1.0; 4]);
        assert_eq!(loaded.normal_strength, 1.0);
        assert_eq!(loaded.slope_start_deg, TerrainMaterialUniform::default().slope_start_deg);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_short_repeat_array_only_sets_what_it_names() {
        let dir = settings_dir("short", Some(r#"{"repeat": [2.0]}"#));
        let loaded = load_terrain_settings(&dir);
        let default = TerrainMaterialUniform::default().repeat;
        assert_eq!(loaded.repeat, [2.0, default[1], default[2], default[3]]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_zero_tile_size_is_clamped_not_taken() {
        let dir = settings_dir("zero", Some(r#"{"repeat": [0.0, -4.0, 10.0, 6.0]}"#));
        let loaded = load_terrain_settings(&dir);
        assert_eq!(loaded.repeat[0], MIN_REPEAT);
        assert_eq!(loaded.repeat[1], MIN_REPEAT);
        assert_eq!(loaded.repeat[2], 10.0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn malformed_settings_fall_back_rather_than_failing() {
        let dir = settings_dir("broken", Some("{ this is not json"));
        assert_eq!(
            load_terrain_settings(&dir).repeat,
            TerrainMaterialUniform::default().repeat,
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn resampling_preserves_the_requested_size() {
        let out = resample(&img([9, 9, 9], 7, 3), 16, 16);
        assert_eq!((out.width, out.height), (16, 16));
        assert_eq!(out.rgba.len(), 16 * 16 * 4);
    }
}

pub const FLAT_NORMAL: [u8; 3] = [128, 128, 255];

impl TerrainImage {
    fn clone_image(&self) -> TerrainImage {
        TerrainImage { width: self.width, height: self.height, rgba: self.rgba.clone() }
    }
}

#[cfg(test)]
mod normal_map_tests {
    use super::tests::{render_quad_full, Palette};
    use super::*;

    const FLAT: [f32; 3] = [0.0, 1.0, 0.0];

    fn normal_map(rgb: [u8; 3]) -> TerrainImage {
        solid_image(rgb, 1, 1)
    }

    fn only_layer0(map: TerrainImage) -> [Option<TerrainImage>; 4] {
        [Some(map), None, None, None]
    }

    fn brightness(px: [u8; 4]) -> u32 {
        px[0] as u32 + px[1] as u32 + px[2] as u32
    }

    #[test]
    fn a_tilted_normal_changes_the_lighting() {
        let Some(flat) = render_quad_full(FLAT, Palette::Test, None, None, &only_layer0(normal_map(FLAT_NORMAL)), true)
        else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };

        let toward = render_quad_full(FLAT, Palette::Test, None, None, &only_layer0(normal_map([230, 128, 160])), true).unwrap();

        let away = render_quad_full(FLAT, Palette::Test, None, None, &only_layer0(normal_map([25, 128, 160])), true).unwrap();

        assert_ne!(brightness(toward), brightness(flat), "a tilted normal must change shading: {toward:?} vs {flat:?}");
        assert!(
            brightness(toward) > brightness(away),
            "tilting toward the light must be brighter than tilting away: {toward:?} vs {away:?}",
        );
    }

    #[test]
    fn green_tilts_the_ground_toward_the_top_of_its_picture() {
        let north = glam::Vec3::new(0.4, 2.0, -3.0);
        let at = |green: u8| {
            super::tests::render_quad_light_at(
                FLAT, Palette::Test, None, None, &only_layer0(normal_map([128, green, 180])), Some(None), north,
            )
        };
        let Some(faces_top) = at(230) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let faces_bottom = at(26).unwrap();
        assert!(
            brightness(faces_top) > brightness(faces_bottom) + 6,
            "a texel facing the top of the picture (-z) must be lit more by a lamp toward -z: \
             {faces_top:?} vs {faces_bottom:?}",
        );
    }

    #[test]
    fn a_normal_maps_bumps_do_not_turn_with_the_player() {
        let lamp = glam::Vec3::new(0.4, 2.0, -3.0);
        let at = |rgb: [u8; 3], yaw: f32| {
            super::tests::render_quad_turned(
                FLAT, Palette::Test, None, None, &only_layer0(normal_map(rgb)), Some(None), lamp, yaw,
            )
        };
        let tilted = [210, 60, 170];
        let Some(still) = at(tilted, 0.0) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };

        let opposite = at([46, 196, 170], 0.0).unwrap();
        assert!(
            brightness(still).abs_diff(brightness(opposite)) > 6,
            "the test's lamp must tell the tilts apart: {still:?} vs {opposite:?}",
        );
        use std::f32::consts::{FRAC_PI_2, FRAC_PI_4, PI};
        for yaw in [FRAC_PI_2, PI, -FRAC_PI_4, 1.0] {
            let turned = at(tilted, yaw).unwrap();
            let off = still.iter().zip(&turned).map(|(a, b)| a.abs_diff(*b)).max().unwrap();
            assert!(off <= 1, "turned {yaw} radians: {turned:?}, unturned {still:?}");
        }
    }

    #[test]
    fn steep_ground_is_projected_along_the_worlds_axes_at_any_turn() {
        let lamp = glam::Vec3::new(0.4, 2.0, -3.0);
        let tilt = 20f32.to_radians();
        let slope = [tilt.sin(), tilt.cos(), 0.0];
        let at = |yaw: f32| {
            super::tests::render_quad_turned(
                slope, Palette::Ramp, None, None, &[None, None, None, None], Some(None), lamp, yaw,
            )
        };
        let Some(still) = at(0.0) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        assert!(still[0] > 10, "the ramp's red shows: {still:?}");
        use std::f32::consts::{FRAC_PI_2, FRAC_PI_4, PI};
        for yaw in [FRAC_PI_2, PI, -FRAC_PI_4, 1.0] {
            let turned = at(yaw).unwrap();
            let off = still.iter().zip(&turned).map(|(a, b)| a.abs_diff(*b)).max().unwrap();
            assert!(off <= 1, "turned {yaw} radians: {turned:?}, unturned {still:?}");
        }
    }

    #[test]
    fn a_flat_normal_map_is_a_true_no_op() {
        let Some(without) = render_quad_full(FLAT, Palette::Test, None, None, &[None, None, None, None], true)
        else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let with_flat = render_quad_full(FLAT, Palette::Test, None, None, &only_layer0(normal_map(FLAT_NORMAL)), true).unwrap();
        assert_eq!(with_flat, without, "a flat normal map must shade identically to none");
    }

    #[test]
    fn zero_strength_disables_the_maps() {
        let Some((device, queue)) = super::tests::headless_gpu() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let layout = material_bind_group_layout(&device);
        let mut settings = TerrainMaterialUniform::default();
        settings.normal_strength = 0.0;

        let err_scope_5 = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let _m = TerrainMaterial::new(
            &device, &queue, &layout,
            &[solid_image([200, 200, 200], 4, 4)],
            &solid_image([128, 128, 128], 1, 1),
            None,
            None,
            &only_layer0(solid_image([230, 128, 160], 4, 4)),
            &[],
            &[],
            settings,
        );
        assert!(pollster::block_on(err_scope_5.pop()).is_none());
    }

    #[test]
    fn a_normal_map_of_a_different_size_is_resampled() {
        let Some((device, queue)) = super::tests::headless_gpu() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let layout = material_bind_group_layout(&device);
        let err_scope_6 = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let _m = TerrainMaterial::from_layers(
            &device, &queue, &layout,
            &[Some(solid_image([10, 20, 30], 64, 64)), None, None, None],
            &[Some(solid_image(FLAT_NORMAL, 16, 16)), None, None, None],
            None,
            None,
        );
        assert!(
            pollster::block_on(err_scope_6.pop()).is_none(),
            "a mismatched normal map must be resampled, not rejected",
        );
    }

    #[test]
    fn the_flat_normal_texel_unpacks_to_straight_out() {

        let unpack = |v: u8| (v as f32 / 255.0) * 2.0 - 1.0;
        assert!(unpack(FLAT_NORMAL[0]).abs() < 0.01);
        assert!(unpack(FLAT_NORMAL[1]).abs() < 0.01);
        assert!((unpack(FLAT_NORMAL[2]) - 1.0).abs() < 0.01);
    }
}

#[cfg(test)]
pub fn terrain_shader_src() -> String {
    terrain_shader()
}

#[cfg(test)]
mod terrain_material_parity_tests {

    #[test]
    fn terrain_uses_the_material_path_with_albedo_passed_in() {
        let src = super::terrain_shader_src();
        assert!(
            src.contains("let lit = shade_material_env("),
            "terrain no longer shades through the material path",
        );
        assert!(
            !src.contains("shade_with_sky(in.world_pos"),
            "terrain went back to the sky-only path",
        );
        assert!(
            src.contains("tonemap(in.col.rgb * lit)"),
            "the albedo is being multiplied over the lit result again; it must \
             go INTO the shading so highlights are not tinted by it",
        );
        assert!(
            !src.contains("tonemap(albedo * in.col.rgb * lit)"),
            "the old multiply-over form is back",
        );
    }

    #[test]
    fn the_normal_variance_lands_in_roughness() {
        let src = super::terrain_shader_src();

        assert!(
            src.contains("layer_rough_at(layer, f)"),
            "terrain no longer samples a roughness map",
        );
        assert!(
            src.contains("layer_ao_at(layer, f)"),
            "terrain no longer samples an occlusion map",
        );
        assert!(
            !src.contains("min(2.0 * sigma2"),
            "the runtime variance term is back alongside the baked mips; that \
             counts the same measurement twice",
        );

        assert!(
            src.contains("specular_aa_roughness(clamp(rough_map, 0.0, 1.0), dpdx(shaded_n), dpdy(shaded_n))"),
            "the geometric specular-AA term is gone; the bake alone cannot see \
             curvature or the anisotropy gap",
        );
    }
}

#[cfg(test)]
mod terrain_mip_tests {

    /// Every map of every layer is a full mip chain -- a single level makes
    /// distant ground point-sample full-resolution normals and shimmer -- each
    /// filtered as what it is: colour sRGB-correct, normals and occlusion
    /// linear, roughness widened by the normals' spread. The one chain both the
    /// RGBA upload and the ASTC encoder take (`layer_mips`).
    #[test]
    fn every_layer_map_is_a_full_chain_filtered_as_what_it_holds() {
        use super::*;
        let (w, h) = (16, 16);
        let noisy = |seed: u32| TerrainImage { width: w, height: h, rgba: (0..w * h * 4).map(|i| ((i * 37 + seed * 101) % 251) as u8).collect() };
        let layers = filled_layers(&[Some(noisy(1)), None, None, None]);
        let (normals, rough, ao) = (vec![Some(noisy(2))], vec![Some(noisy(3))], vec![Some(noisy(4))]);
        let levels = mip_levels_for(w, h) as usize;
        for map in LayerMap::ALL {
            let chain = layer_mips(map, 0, &layers, &normals, &rough, &ao);
            assert_eq!(chain.len(), levels, "{map:?} lost its mip chain");
        }
        assert_eq!(layer_mips(LayerMap::Albedo, 0, &layers, &normals, &rough, &ao), mip_chain(&layers[0].rgba, w, h, levels as u32));
        assert_eq!(layer_mips(LayerMap::Normal, 0, &layers, &normals, &rough, &ao), mip_chain_linear(&normals[0].as_ref().unwrap().rgba, w, h, levels as u32));
        assert_eq!(layer_mips(LayerMap::Ao, 0, &layers, &normals, &rough, &ao), mip_chain_linear(&ao[0].as_ref().unwrap().rgba, w, h, levels as u32));
        assert_eq!(
            layer_mips(LayerMap::Rough, 0, &layers, &normals, &rough, &ao),
            roughness_chain_with_normal_variance(&rough[0].as_ref().unwrap().rgba, &normals[0].as_ref().unwrap().rgba, w, h, levels as u32),
        );
        assert_ne!(mip_chain(&layers[0].rgba, w, h, 2)[1], mip_chain_linear(&layers[0].rgba, w, h, 2)[1], "the colour chain must not be the linear one");
        // A layer with no maps takes the defaults: a flat normal, the default
        // roughness, no occlusion.
        let flat = layer_mips(LayerMap::Normal, 2, &layers, &normals, &rough, &ao);
        assert_eq!(&flat[0].0[..3], &FLAT_NORMAL);
        assert_eq!(layer_mips(LayerMap::Ao, 2, &layers, &normals, &rough, &ao)[0].0[0], 255);
        assert!(LayerMap::ALL.iter().all(|m| m.srgb() == (*m == LayerMap::Albedo)), "only colour is sRGB; a measurement array must be linear");
    }

    /// A set mixing sizes is built at its LARGEST layer's size, wherever that
    /// layer sits: a 2K mud in slot 2 beside 1K grass keeps its 2K detail.
    #[test]
    fn mixed_layer_sizes_are_built_at_the_largest() {
        use super::*;
        let img = |side: u32| TerrainImage { width: side, height: side, rgba: vec![128; (side * side * 4) as usize] };
        let layers = filled_layers(&[Some(img(8)), Some(img(8)), Some(img(32)), None]);
        assert!(layers.iter().all(|l| (l.width, l.height) == (32, 32)), "{:?}", layers.iter().map(|l| l.width).collect::<Vec<_>>());
    }

    #[test]
    fn a_layers_roughness_takes_its_materials_override() {
        let game = std::env::temp_dir().join(format!("terrain_rough_{}", std::process::id()));
        let dir = game.join("textures").join("terrain");
        let _ = std::fs::remove_dir_all(&game);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("sources.json"), r#"{"ground": {"assetId": "Grass005"}, "rock": {"assetId": "Rock063"}}"#).unwrap();
        std::fs::write(game.join("material_overrides.json"), r#"{"Rock063": {"roughness": {"min": 0.55, "max": 1.0}}}"#).unwrap();
        assert_eq!(super::terrain_roughness_range(&dir, 1), Some((0.55, 1.0)), "the rock layer is Rock063");
        assert_eq!(super::terrain_roughness_range(&dir, 0), None, "grass has no override");
        assert_eq!(super::terrain_roughness_range(&dir, 2), None, "a layer with no source");

        let grey = image::RgbImage::from_pixel(4, 4, image::Rgb([102, 102, 102]));
        grey.save(dir.join("rock_r.jpg")).unwrap();
        grey.save(dir.join("ground_r.jpg")).unwrap();
        let maps = super::load_terrain_rough(&dir);
        let read = |m: &Option<super::TerrainImage>| m.as_ref().unwrap().rgba[0] as f32 / 255.0;
        assert!((read(&maps[1]) - 0.73).abs() < 0.02, "rock read as {}", read(&maps[1]));
        assert!((read(&maps[0]) - 0.4).abs() < 0.02, "grass read as {}", read(&maps[0]));
        let _ = std::fs::remove_dir_all(&game);
    }

    #[test]
    fn the_shipped_cliffs_read_as_dry_rock() {
        let dir = crate::test_game_dir().join("textures/terrain");
        let Some(Some(rock)) = super::load_terrain_rough(&dir).into_iter().nth(1) else {
            eprintln!("skipping: no terrain rock map");
            return;
        };
        let mut v: Vec<u8> = rock.rgba.chunks_exact(4).map(|p| p[0]).collect();
        v.sort_unstable();
        let median = v[v.len() / 2] as f32 / 255.0;
        assert!(median > 0.7, "the cliffs' rock reads at a median roughness of {median}: wet stone");
    }

    #[test]
    fn every_layer_has_a_roughness_and_occlusion_file_name() {
        assert_eq!(super::TERRAIN_ROUGH_FILES.len(), super::TERRAIN_LAYER_FILES.len());
        assert_eq!(super::TERRAIN_AO_FILES.len(), super::TERRAIN_LAYER_FILES.len());
        for (i, colour) in super::TERRAIN_LAYER_FILES.iter().enumerate() {
            let stem = colour.trim_end_matches(".jpg");
            assert_eq!(super::TERRAIN_ROUGH_FILES[i], format!("{stem}_r.jpg"));
            assert_eq!(super::TERRAIN_AO_FILES[i], format!("{stem}_ao.jpg"));
        }
    }

    #[test]
    fn the_linear_chain_preserves_a_flat_value() {
        let px = vec![128u8; 8 * 8 * 4];
        let chain = super::mip_chain_linear(&px, 8, 8, super::mip_levels_for(8, 8));
        assert!(chain.len() > 1, "no chain was built");
        for (data, w, h) in &chain {
            assert_eq!(data.len(), (w * h * 4) as usize);
            for b in data {
                assert!(
                    (*b as i32 - 128).abs() <= 1,
                    "a flat 128 became {b}; the linear chain is bending values",
                );
            }
        }
    }
}

#[cfg(test)]
mod steep_shapes_frame {
    use glam::Vec3;

    const CELL: f32 = 0.43;
    const TIE: f32 = 0.57735;
    const STEEP_COS: f32 = 0.95106;

    const BUMP_Z: f32 = 0.63;

    struct Mesh {
        p: Vec<Vec3>,
        n: Vec<Vec3>,
        tris: Vec<[usize; 3]>,
    }

    fn mesh(size: usize, h: impl Fn(usize, usize) -> f32) -> Mesh {
        let at = |i: usize, j: usize| i * size + j;
        let p: Vec<Vec3> = (0..size * size).map(|k| Vec3::new((k % size) as f32 * CELL, h(k / size, k % size), (k / size) as f32 * CELL)).collect();
        let mut tris = Vec::new();
        for i in 0..size - 1 {
            for j in 0..size - 1 {
                tris.push([at(i, j), at(i + 1, j), at(i, j + 1)]);
                tris.push([at(i, j + 1), at(i + 1, j), at(i + 1, j + 1)]);
            }
        }
        let mut n = vec![Vec3::ZERO; p.len()];
        for t in &tris {
            let c = (p[t[1]] - p[t[0]]).cross(p[t[2]] - p[t[0]]);
            let c = if c.y < 0.0 { -c } else { c };
            for &v in t {
                n[v] += c;
            }
        }
        Mesh { p, n: n.into_iter().map(|v| v.normalize()).collect(), tris }
    }

    fn shapes() -> Vec<(&'static str, Mesh)> {
        let s = 12;
        vec![
            ("wall", mesh(s, |_, j| if j >= 6 { 13.0 } else { 0.0 })),
            ("face85", mesh(s, |_, j| (j.clamp(4, 8) - 4) as f32 * CELL * 85f32.to_radians().tan())),
            ("ridge", mesh(s, |_, j| (6.0 - (j as f32 - 6.0).abs() * CELL * 70f32.to_radians().tan()).max(0.0))),
            ("foot", mesh(s, |_, j| (j.max(6) - 6) as f32 * CELL * 80f32.to_radians().tan())),
            ("stair", mesh(s, |_, j| (j / 2) as f32 * 0.8)),
        ]
    }

    fn planes_new(n: Vec3, face: Vec3) -> ([(usize, f32); 2], f32) {
        let steep = n.y.min(face.y) <= STEEP_COS;
        let agree = ((n.dot(face).abs() - 0.9703) * 41.3).clamp(0.0001, 1.0);
        let pn = face.lerp(n, agree).normalize();
        let (major, minor) = axes(pn);
        let a = pn.abs();
        let (ma, me) = (a[major], a[minor]);
        let lead = ma - TIE;
        let share = (lead + 0.0001) / (lead + (me - TIE).max(0.0) + 0.0001);
        if !steep {
            return ([(1, 1.0), (minor, 0.0)], 1.0);
        }
        ([(major, share), (minor, 1.0 - share)], agree)
    }

    fn planes_old(n: Vec3) -> ([(usize, f32); 2], [(usize, f32); 2]) {
        let (major, minor) = axes(n);
        if n.y > STEEP_COS {
            return ([(1, 1.0), (minor, 0.0)], [(1, 1.0), (minor, 0.0)]);
        }
        let a = n.abs();
        let w = a[major] / (a[major] + a[minor]);
        ([(major, w), (minor, 1.0 - w)], [(major, 1.0), (minor, 0.0)])
    }

    fn axes(n: Vec3) -> (usize, usize) {
        let a = n.abs();
        let major = if a.y >= a.x && a.y >= a.z { 1 } else if a.x >= a.z { 0 } else { 2 };
        let minor = (0..3).filter(|&k| k != major).max_by(|&i, &j| a[i].total_cmp(&a[j])).unwrap();
        (major, minor)
    }

    fn fold(axis: usize, n: Vec3, keep: f32) -> Vec3 {
        if axis == 1 {
            let g = BUMP_Z + (1.0 - BUMP_Z) * keep;
            return Vec3::new(g * n.x, BUMP_Z * n.y, g * n.z);
        }
        n * BUMP_Z
    }
    fn whiteout(axis: usize, n: Vec3) -> Vec3 {
        let mut v = n;
        v[axis] *= BUMP_Z;
        v
    }

    fn samples(m: &Mesh) -> Vec<(usize, Vec3, Vec3, Vec3)> {
        let mut out = Vec::new();
        for (t, tri) in m.tris.iter().enumerate() {
            let [a, b, c] = tri.map(|v| m.p[v]);
            let face = (b - a).cross(c - a).normalize();
            let face = if face.y < 0.0 { -face } else { face };
            for (u, v) in [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.5, 0.0), (0.0, 0.5), (0.5, 0.5), (0.33, 0.33), (0.1, 0.1), (0.8, 0.1), (0.1, 0.8)] {
                let w = 1.0 - u - v;
                let pos = a * w + b * u + c * v;
                let n = (m.n[tri[0]] * w + m.n[tri[1]] * u + m.n[tri[2]] * v).normalize();
                out.push((t, pos, n, face));
            }
        }
        out
    }

    fn worst_stretch(planes: &[(usize, f32)], face: Vec3) -> f32 {
        planes.iter().filter(|(_, s)| *s >= 0.1).map(|&(k, _)| 1.0 / face.abs()[k].max(0.01)).fold(1.0, f32::max)
    }

    #[test]
    fn no_plane_is_read_stretched_on_any_steep_shape() {
        for (name, m) in shapes() {
            let (mut new, mut old) = (1.0f32, 1.0f32);
            for (_, _, n, face) in samples(&m) {
                new = new.max(worst_stretch(&planes_new(n, face).0, face));
                old = old.max(worst_stretch(&planes_old(n).0, face));
            }
            assert!(new <= 2.8, "{name}: a plane is read stretched {new:.1} times");
            assert!(old > 3.0, "{name}: the old reading stretched only {old:.1} times -- the shape no longer tests anything");
        }
    }

    #[test]
    fn the_mean_shading_normal_never_jumps() {
        let shade_new = |n: Vec3, face: Vec3| {
            let (planes, keep) = planes_new(n, face);
            planes.iter().map(|&(k, s)| fold(k, n, keep) * s).sum::<Vec3>().normalize()
        };
        let shade_old = |n: Vec3| {
            let (_, normal) = planes_old(n);
            whiteout(normal[0].0, n).normalize()
        };
        for (name, m) in shapes() {
            let (mut new, mut old) = (0.0f32, 0.0f32);

            let pts = samples(&m);
            for (i, &(ti, pi, ni, fi)) in pts.iter().enumerate() {
                for &(tj, pj, nj, fj) in &pts[i + 1..] {
                    if ti == tj || pi.distance(pj) > 1e-4 {
                        continue;
                    }
                    new = new.max(shade_new(ni, fi).angle_between(shade_new(nj, fj)).to_degrees());
                    old = old.max(shade_old(ni).angle_between(shade_old(nj)).to_degrees());
                }
            }

            let k = |r: usize, c: usize| r * 12 + c;
            let steps = 4000;
            let along = |t: f32| {
                let (j, f) = ((t as usize).min(10), t - (t as usize).min(10) as f32);
                m.n[k(5, j)].lerp(m.n[k(5, j + 1)], f).normalize()
            };
            for step in 0..steps {
                let (t0, t1) = (step as f32 * 11.0 / steps as f32, (step + 1) as f32 * 11.0 / steps as f32);
                old = old.max(shade_old(along(t0)).angle_between(shade_old(along(t1))).to_degrees());
            }
            assert!(new < 1.0, "{name}: the shading normal jumps {new:.1} degrees across a line");

            if name != "face85" && name != "stair" {
                assert!(old > 4.0, "{name}: the old reading jumped only {old:.1} degrees -- the shape no longer tests anything");
            }
        }
    }

    #[test]
    fn the_frame_takes_no_derivative_of_a_chosen_coordinate() {
        let src = super::terrain_shader();
        let frame = src.split("fn sample_frame").nth(1).and_then(|s| s.split("\nfn ").next()).expect("sample_frame");
        for (i, _) in frame.match_indices("dpdx(").chain(frame.match_indices("dpdy(")) {
            assert!(frame[i + 5..].starts_with("world)"), "a derivative of something other than the world position: {}", &frame[i..i + 40]);
        }
    }

    #[test]
    fn a_steep_faced_triangle_is_never_gentle() {
        for (name, m) in shapes() {
            for tri in &m.tris {
                let pos = tri.map(|v| m.p[v].to_array());
                let face_y = crate::renderer::ground_twins::face_height(pos);
                let normals = tri.map(|v| m.n[v].to_array());
                if face_y <= STEEP_COS {
                    assert!(!crate::renderer::ground_twins::triangle_is_gentle(normals, face_y, 18.0), "{name}: a steep face drawn gentle");
                }
            }
        }
    }
}

#[cfg(test)]
mod relief_tests {
    use super::*;

    #[test]
    fn only_the_desktop_ground_marches_the_relief() {
        for role in [TerrainRole::Scene, TerrainRole::Read, TerrainRole::ProbePass] {
            let src = terrain_shader_for(role);
            assert!(!src.contains("parallax_shift"), "{role:?} carries the relief march");
            let (plain, shaped) = twin_sources(src);
            assert!(!plain.contains("parallax_shift"));
            assert!(shaped.map_or(true, |s| !s.contains("parallax_shift")));
        }
        let desktop = with_relief(&terrain_shader_for(TerrainRole::Scene)).expect("the scene reader takes the relief");
        assert_eq!(desktop.matches("fn parallax_shift(").count(), 1);
        assert_eq!(desktop.matches("parallax_shift(in.tex_pos.xz").count(), 1);
        let (plain, _) = twin_sources(desktop);
        wgpu::naga::front::wgsl::parse_str(&plain).expect("the desktop ground parses");
    }

    #[test]
    fn the_material_uniform_is_the_wgsl_structs_size() {
        assert_eq!(std::mem::size_of::<TerrainMaterialUniform>() % 16, 0);
    }
}

#[cfg(test)]
mod mud_build_tests {
    use super::*;
    use wgpu::naga;

    fn half_to_f32(h: u16) -> f32 {
        let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
        let e = ((h >> 10) & 0x1f) as i32;
        let m = (h & 0x3ff) as f32;
        match e {
            0 => sign * m * 2f32.powi(-24),
            _ => sign * (1.0 + m / 1024.0) * 2f32.powi(e - 15),
        }
    }

    fn validate(label: &str, src: &str) {
        let module = naga::front::wgsl::parse_str(src).unwrap_or_else(|e| panic!("{label}: {}", e.emit_to_string(src)));
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
            .validate(&module)
            .unwrap_or_else(|e| panic!("{label}: {e:?}"));
    }

    fn total(s: &str) -> usize {
        crate::renderer::shader_inlining::inlined_sizes(s, "fs_main").iter().map(|f| f.own * f.copies).sum()
    }

    #[test]
    fn the_deform_map_holds_every_step_and_slope_exactly() {
        for n in -2048..=2048 {
            assert_eq!(half_to_f32(half_of_int(n)), n as f32, "{n}");
        }
        // 3 x 2: heights 10 20 40 / 70 110 160, wet 1..6.
        let bytes: Vec<u8> = [10u8, 20, 40, 70, 110, 160].iter().enumerate().flat_map(|(i, h)| [*h, i as u8 + 1]).collect();
        let rows = deform_rows(&bytes, 3, 2, 0, 2);
        let texel = |x: usize, z: usize| -> [f32; 4] { std::array::from_fn(|c| half_to_f32(rows[4 * (z * 3 + x) + c])) };
        assert_eq!(texel(0, 0), [10.0, 1.0, (20 - 40) as f32, (70 - 70) as f32], "x wraps to the last column, z to the last row");
        assert_eq!(texel(1, 1), [110.0, 5.0, (160 - 70) as f32, (20 - 20) as f32]);
        assert_eq!(deform_rows_touched(0, 1, 5), vec![(0, 2), (4, 5)]);
        assert_eq!(deform_rows_touched(2, 3, 5), vec![(1, 4)]);
        assert_eq!(deform_rows_touched(3, 5, 5), vec![(2, 5), (0, 1)]);
    }

    #[test]
    fn every_ground_source_builds_with_and_without_its_deform_map() {
        let deform_reads = |s: &str| s.matches("textureSampleLevel(deform_tex").count();
        for painted in [false, true] {
            for role in [TerrainRole::Scene, TerrainRole::Read, TerrainRole::ProbePass] {
                let muddy = terrain_shader_built_mud(role, painted, true);
                let plain = terrain_shader_built_mud(role, painted, false);
                assert_eq!(deform_reads(&muddy), 2, "{role:?}: one read for the vertex, one for the pixel");
                assert_eq!(deform_reads(&plain), 0, "{role:?}");
                validate(&format!("{role:?} muddy"), &muddy);
                validate(&format!("{role:?} deformless"), &plain);
                assert!(crate::renderer::weather::with_weather(&muddy).is_some(), "{role:?}: the weather's edits miss the muddy ground");
            }
        }
    }

    #[test]
    fn the_muddy_ground_stays_under_the_readers_caps() {
        use crate::renderer::ground_twins::{gentle_shader, torch_weather_twin};
        use crate::renderer::weather::{with_weather_kinds, WeatherKinds};
        let baked = |src: String| crate::renderer::brush_pipeline::sun_reader_shader(src, crate::renderer::brush_pipeline::FaceSun::Baked);
        let gentle = baked(gentle_shader(&terrain_shader_built_mud(TerrainRole::Read, true, true)).unwrap());
        let puddles = torch_weather_twin(with_weather_kinds(&gentle, WeatherKinds::ALL[1]).unwrap());
        validate("muddy torch puddle reader", &puddles);
        let pass = terrain_shader_built_mud(TerrainRole::ProbePass, false, true);
        let nowhere = pass.replacen(NOWHERE_OFF, NOWHERE_ON, 1);
        let (full, out) = (total(&pass), total(&nowhere));
        eprintln!("muddy: torch puddle reader {}, probe pass {full}, its nowhere twin {out}", total(&puddles));
        assert!(total(&puddles) <= 2600, "the muddy torch puddle reader is {} by the proxy", total(&puddles));
        // The nowhere twin's job is to drop the probe lookup; the mud is in the
        // pass and the twin alike, so the twin must drop the same lookup it drops
        // from the deformless pass (whose share the nowhere test bounds).
        let plain = terrain_shader_built_mud(TerrainRole::ProbePass, false, false);
        let plain_nowhere = plain.replacen(NOWHERE_OFF, NOWHERE_ON, 1);
        assert_eq!(full - out, total(&plain) - total(&plain_nowhere), "the muddy nowhere twin keeps part of the lookup");
    }

    /// The slope stored beside each height, filtered once, against the four
    /// filtered heights a texel either side it replaces, on a map of random
    /// heights (the worst case: every texel a step).
    #[test]
    fn one_slope_read_is_the_four_height_reads() {
        let Some((device, queue)) = super::tests::headless_gpu() else {
            eprintln!("no GPU: skipped");
            return;
        };
        let (w, h) = (37u32, 23u32);
        let mut seed = 12345u32;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            (seed >> 24) as u8
        };
        let bytes: Vec<u8> = (0..w * h * 2).map(|_| next()).collect();
        let texture = |format: TextureFormat, data: &[u8], bpr: u32| {
            let t = device.create_texture(&TextureDescriptor {
                label: None,
                size: Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format,
                usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
                view_formats: &[],
            });
            queue.write_texture(
                TexelCopyTextureInfo { texture: &t, mip_level: 0, origin: Origin3d::ZERO, aspect: TextureAspect::All },
                data,
                TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(bpr), rows_per_image: Some(h) },
                Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            );
            t.create_view(&TextureViewDescriptor::default())
        };
        let old = texture(TextureFormat::Rg8Unorm, &bytes, w * 2);
        let rows = deform_rows(&bytes, w, h, 0, h);
        let new = texture(DEFORM_FORMAT, bytemuck::cast_slice(&rows), w * DEFORM_TEXEL_BYTES);
        let sampler = device.create_sampler(&SamplerDescriptor {
            address_mode_u: AddressMode::Repeat,
            address_mode_v: AddressMode::Repeat,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            ..Default::default()
        });
        const N: u32 = 4096;
        let out = device.create_buffer(&BufferDescriptor { label: None, size: N as u64 * 32, usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC, mapped_at_creation: false });
        let read = device.create_buffer(&BufferDescriptor { label: None, size: N as u64 * 32, usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST, mapped_at_creation: false });
        let src = format!(
            "
@group(0) @binding(0) var old_map: texture_2d<f32>;
@group(0) @binding(1) var new_map: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var<storage, read_write> out: array<vec4<f32>>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
    let i = id.x;
    if (i >= {N}u) {{ return; }}
    let uv = vec2<f32>(fract(f32(i) * 0.6180339887), fract(f32(i) * 0.7548776662 + 0.1));
    let e = 1.0 / vec2<f32>(textureDimensions(old_map));
    let hx = textureSampleLevel(old_map, samp, uv + vec2<f32>(e.x, 0.0), 0.0).r - textureSampleLevel(old_map, samp, uv - vec2<f32>(e.x, 0.0), 0.0).r;
    let hz = textureSampleLevel(old_map, samp, uv + vec2<f32>(0.0, e.y), 0.0).r - textureSampleLevel(old_map, samp, uv - vec2<f32>(0.0, e.y), 0.0).r;
    let o = textureSampleLevel(old_map, samp, uv, 0.0);
    let n = textureSampleLevel(new_map, samp, uv, 0.0) * {unit:?};
    out[2u * i] = vec4<f32>(hx, hz, n.b, n.a);
    out[2u * i + 1u] = vec4<f32>(uv, o.r - n.r, o.g - n.g);
}}
",
            unit = 1.0f32 / 255.0
        );
        let module = device.create_shader_module(ShaderModuleDescriptor { label: None, source: ShaderSource::Wgsl(src.into()) });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let group = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry { binding: 0, resource: BindingResource::TextureView(&old) },
                BindGroupEntry { binding: 1, resource: BindingResource::TextureView(&new) },
                BindGroupEntry { binding: 2, resource: BindingResource::Sampler(&sampler) },
                BindGroupEntry { binding: 3, resource: out.as_entire_binding() },
            ],
        });
        let mut enc = device.create_command_encoder(&CommandEncoderDescriptor::default());
        {
            let mut pass = enc.begin_compute_pass(&ComputePassDescriptor::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(N / 64, 1, 1);
        }
        enc.copy_buffer_to_buffer(&out, 0, &read, 0, N as u64 * 32);
        queue.submit([enc.finish()]);
        let slice = read.slice(..);
        slice.map_async(MapMode::Read, |_| {});
        device.poll(PollType::Wait { submission_index: None, timeout: None }).ok();
        let data = slice.get_mapped_range().unwrap();
        let rows: &[[f32; 4]] = bytemuck::cast_slice(&data);
        // The exact answer: bilinear heights, wrapping, a texel either side.
        let height = |x: i64, z: i64| bytes[2 * (z.rem_euclid(h as i64) as usize * w as usize + x.rem_euclid(w as i64) as usize)] as f64 / 255.0;
        let bilinear = |u: f64, v: f64| {
            let (x, z) = (u * w as f64 - 0.5, v * h as f64 - 0.5);
            let (x0, z0) = (x.floor(), z.floor());
            let (fx, fz) = (x - x0, z - z0);
            let (x0, z0) = (x0 as i64, z0 as i64);
            (height(x0, z0) * (1.0 - fx) + height(x0 + 1, z0) * fx) * (1.0 - fz) + (height(x0, z0 + 1) * (1.0 - fx) + height(x0 + 1, z0 + 1) * fx) * fz
        };
        let (ex, ez) = (1.0 / w as f64, 1.0 / h as f64);
        let (mut four, mut one, mut level) = (0f64, 0f64, 0f64);
        for pair in rows.chunks_exact(2) {
            let (got, at) = (pair[0], pair[1]);
            let (u, v) = (at[0] as f64, at[1] as f64);
            let exact = [bilinear(u + ex, v) - bilinear(u - ex, v), bilinear(u, v + ez) - bilinear(u, v - ez)];
            four = four.max((got[0] as f64 - exact[0]).abs()).max((got[1] as f64 - exact[1]).abs());
            one = one.max((got[2] as f64 - exact[0]).abs()).max((got[3] as f64 - exact[1]).abs());
            level = level.max((at[2] as f64).abs()).max((at[3] as f64).abs());
        }
        eprintln!(
            "worst slope error against exact bilinear, in 8-bit steps: four reads {:.4}, one read {:.4}; height/wet against the 8-bit map {:.4}",
            four * 255.0,
            one * 255.0,
            level * 255.0
        );
        // One read of the stored slope must be no further from the exact slope
        // than the four reads it replaces (the filter's weight precision
        // bounds both), and the height and wetness must read as they did.
        assert!(one <= four * 1.05 + 1e-5, "one read {one} vs four {four}");
        assert!(level * 255.0 < 0.1, "height/wet moved {level}");
    }
}

#[cfg(test)]
mod astc_layer_tests {
    use super::*;

    fn array(srgb: bool, side: u32) -> AstcArray {
        let levels = mip_levels_for(side, side);
        let data = (0..4u8).map(|layer| (0..levels).map(|l| vec![layer; AstcArray::level_bytes([4, 4], side, side, l)]).collect()).collect();
        AstcArray { block: [4, 4], srgb, width: side, height: side, levels, data }
    }

    #[test]
    fn an_encoded_array_reads_back_as_written_and_a_wrong_one_is_refused() {
        let a = array(true, 16);
        assert_eq!(AstcArray::from_bytes(&a.to_bytes()), Ok(a.clone()));
        // 16x16 in 4x4 blocks: 16 blocks, then 4, then 1 and 1 (a 2x2 and a
        // 1x1 level still take a whole block).
        assert_eq!((0..5).map(|l| AstcArray::level_bytes([4, 4], 16, 16, l) / 16).collect::<Vec<_>>(), [16, 4, 1, 1, 1]);
        let mut bytes = a.to_bytes();
        bytes.truncate(bytes.len() - 3);
        assert!(AstcArray::from_bytes(&bytes).is_err(), "cut short");
        assert!(AstcArray::from_bytes(b"not one").is_err());
        let mut odd = a.clone();
        odd.data[1][0].pop();
        assert!(AstcArray::from_bytes(&odd.to_bytes()).is_err(), "a level of the wrong size");
    }

    /// The upload is the risk: a wrong row pitch or copy size for a block
    /// format is a validation error, and on the headset those are silent.
    #[test]
    fn an_encoded_set_uploads_with_no_validation_error_where_the_gpu_reads_astc() {
        let instance = Instance::default();
        let Ok(adapter) = pollster::block_on(instance.request_adapter(&RequestAdapterOptions::default())) else {
            eprintln!("no GPU: skipped");
            return;
        };
        let set = std::env::temp_dir().join(format!("ss_astc_set_{}", std::process::id()));
        std::fs::create_dir_all(set.join("astc")).unwrap();
        for map in LayerMap::ALL {
            std::fs::write(astc_path(&set, map), array(map.srgb(), 64).to_bytes()).unwrap();
        }
        let (plain, plain_queue) = pollster::block_on(adapter.request_device(&DeviceDescriptor::default())).unwrap();
        assert!(TerrainLayerArrays::from_astc(&plain, &plain_queue, &set).is_none(), "no ASTC asked of the device, none used");
        if !adapter.features().contains(Features::TEXTURE_COMPRESSION_ASTC) {
            eprintln!("this GPU reads no ASTC: the upload is not tried");
            let _ = std::fs::remove_dir_all(&set);
            return;
        }
        let (device, queue) = pollster::block_on(adapter.request_device(&DeviceDescriptor { required_features: Features::TEXTURE_COMPRESSION_ASTC, ..Default::default() })).unwrap();
        let scope = device.push_error_scope(ErrorFilter::Validation);
        let arrays = TerrainLayerArrays::from_astc(&device, &queue, &set).expect("the set loads");
        queue.submit(None);
        let error = pollster::block_on(scope.pop());
        assert!(error.is_none(), "{error:?}");
        assert!(arrays.compressed && arrays.size == [64, 64]);
        // A map unlike the others is refused, not half used.
        std::fs::write(astc_path(&set, LayerMap::Rough), array(false, 32).to_bytes()).unwrap();
        assert!(TerrainLayerArrays::from_astc(&device, &queue, &set).is_none());
        let _ = std::fs::remove_dir_all(&set);
    }
}
