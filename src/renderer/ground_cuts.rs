use crate::renderer::terrain_pipeline::{terrain_shader_for, TerrainPipeline, TerrainRole};
use wgpu::{BindGroupLayout, Device, TextureFormat};

pub(crate) const GROUND_CUTS: &[(&str, &[(&str, &str)])] = &[
    ("none", &[]),

    ("purkinje", &[("    exposed = mix(exposed, rods, camera.proxy_params.y * smoothstep(0.05, 1.0, local_exposure));\n", "")]),

    (
        "gentle",
        &[
            (crate::renderer::ground_twins::STEEP_TEST, "    let steep = false;\n"),
            (crate::renderer::ground_twins::FOLD_KEEP, crate::renderer::ground_twins::FOLD_KEPT),
        ],
    ),

    (
        "steep",
        &[
            (crate::renderer::ground_twins::STEEP_TEST, "    let steep = true;\n"),
        ],
    ),

    ("dedup_all", crate::renderer::brush_pipeline::DEDUP_EDITS),

    ("band", &[("    if (steep) {\n        let ms = 1.0 / CLIFF_BAND_SCALE_M;", "    if (false) {\n        let ms = 1.0 / CLIFF_BAND_SCALE_M;")]),

    (
        "warp",
        &[(
            "    f.minor_uv  = f.minor_uv + select(warp, vec2<f32>(0.0), f.minor_axis == 1u);\n    f.detail_uv = f.detail_uv + select(warp, vec2<f32>(0.0), f.detail_axis == 1u);\n",
            "",
        )],
    ),

    (
        "colour2",
        &[(
            "    if (f.second > 0.0) {\n        let rot = mat2x2<f32>(0.8, 0.6, -0.6, 0.8);\n        let s2 = 1.0 / (r * 2.3);\n        c = mix(\n",
            "    if (false) {\n        let rot = mat2x2<f32>(0.8, 0.6, -0.6, 0.8);\n        let s2 = 1.0 / (r * 2.3);\n        c = mix(\n",
        )],
    ),

    ("normal2", &[("        tn = vec3<f32>(mix(tn.xy, t2.xy, f.second) * (1.0 + 0.667 * f.second), mix(tn.z, t2.z, f.second));\n", "")]),

    ("minor_normal", &[("    if (f.blend < PLANE_ALONE) {\n        let pm = textureSampleGrad(", "    if (false) {\n        let pm = textureSampleGrad(")]),

    ("side_normal", &[("    if (f.detail_axis == 1u && f.blend >= PLANE_ALONE) {", "    if (true) {")]),

    ("wet", &[(crate::renderer::terrain_pipeline::SEA_LINE_WGSL, "")]),

    ("wx_snow", &[("const WX_SNOW: bool = true;", "const WX_SNOW: bool = false;")]),
    ("wx_water", &[("const WX_WATER: bool = true;", "const WX_WATER: bool = false;")]),
    ("wx_dry", &[("const WX_SNOW: bool = true;", "const WX_SNOW: bool = false;"), ("const WX_WATER: bool = true;", "const WX_WATER: bool = false;")]),
    (
        "gentle_snowless",
        &[
            (crate::renderer::ground_twins::STEEP_TEST, "    let steep = false;\n"),
            (crate::renderer::ground_twins::FOLD_KEEP, crate::renderer::ground_twins::FOLD_KEPT),
            ("const WX_SNOW: bool = true;", "const WX_SNOW: bool = false;"),
        ],
    ),
    (
        "gentle_waterless",
        &[
            (crate::renderer::ground_twins::STEEP_TEST, "    let steep = false;\n"),
            (crate::renderer::ground_twins::FOLD_KEEP, crate::renderer::ground_twins::FOLD_KEPT),
            ("const WX_WATER: bool = true;", "const WX_WATER: bool = false;"),
        ],
    ),
    ("wx_rings", &[("        if (at.area.x > 0.0 && at.area.y < 0.5 && s.water > 0.0) {", "        if (false) {")]),
    ("wx_noise", &[("        fine = fine + select(0.65, 0.35, k == 1u) * wx_noise(p.xz * select(3.1, 9.7, k == 1u));", "        fine = 0.5;")]),
];

pub(crate) fn ground_cut_bases() -> Vec<(&'static str, TerrainRole, bool, String)> {
    let read = terrain_shader_for(TerrainRole::Read);
    let baked_spotless = crate::renderer::lights::without_spot_shadows(crate::renderer::brush_pipeline::sun_reader_shader(
        read.clone(),
        crate::renderer::brush_pipeline::FaceSun::Baked,
    ));
    let weather = crate::renderer::weather::with_weather(&baked_spotless).expect("the ground's weather twin no longer matches");
    vec![
        ("pass", TerrainRole::ProbePass, false, terrain_shader_for(TerrainRole::ProbePass)),
        ("read", TerrainRole::Read, false, read),
        ("bs", TerrainRole::Read, false, baked_spotless),
        ("wx_bs", TerrainRole::Read, true, weather),
    ]
}

pub(crate) fn with_cut(src: &str, edits: &[(&str, &str)]) -> Option<String> {
    let mut s = src.to_string();
    for (from, to) in edits {
        if s.matches(from).count() != 1 {
            return None;
        }
        s = s.replacen(from, to, 1);
    }
    Some(s)
}

#[allow(clippy::too_many_arguments, reason = "GPU setup passes each resource explicitly; bundling them is a separate refactor")]
pub fn log_ground_cuts(
    device: &Device,
    format: TextureFormat,
    uniform_layout: &BindGroupLayout,
    samples: u32,
    probe_layout: &BindGroupLayout,
    fixups: &crate::renderer::probe_fixup::ProbeFixups,
) {
    let weather_layout = crate::renderer::weather::bind_group_layout(device);
    for (base, role, weathered, src) in ground_cut_bases() {
        for (cut, edits) in GROUND_CUTS {
            let Some(source) = with_cut(&src, edits) else {
                continue;
            };
            let label = format!("gcut_{base}_{cut}");
            let pass = role == TerrainRole::ProbePass;
            let _ = TerrainPipeline::build_from_with(
                device,
                if pass { crate::renderer::brush_pipeline::probe_pass::FORMAT } else { format },
                uniform_layout,
                if pass { 1 } else { samples },
                crate::renderer::multiview::ViewMode::Mono,
                role,
                weathered.then_some(&weather_layout),
                Some(if pass { fixups.pass_layout() } else { probe_layout }),
                source,
                &label,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_ground_cut_matches_its_shaders() {
        use wgpu::naga;
        for (base, _, weathered, src) in ground_cut_bases() {
            for (cut, edits) in GROUND_CUTS {
                let applies = with_cut(&src, edits);
                let expected = if *cut == "wet" { !weathered } else { !(cut.starts_with("wx_") || cut.ends_with("less")) || weathered };
                assert_eq!(applies.is_some(), expected, "cut {cut} on {base}");
                if let Some(s) = applies {
                    let module = naga::front::wgsl::parse_str(&s).unwrap_or_else(|e| panic!("{base}/{cut}: {}", e.emit_to_string(&s)));
                    naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
                        .validate(&module)
                        .unwrap_or_else(|e| panic!("{base}/{cut}: {}", e.emit_to_string(&s)));
                }
            }
        }
    }

    #[test]
    #[ignore = "measurement, not an assertion: prints shader sizes"]
    fn ground_cut_sizes() {
        for (base, _, _, src) in ground_cut_bases() {
            let total = |s: &str| crate::renderer::shader_inlining::inlined_sizes(s, "fs_main").iter().map(|f| f.own * f.copies).sum::<usize>();
            let whole = total(&src);
            println!("{base}: {whole}");
            for (cut, edits) in GROUND_CUTS.iter().skip(1) {
                if let Some(s) = with_cut(&src, edits) {
                    println!("  {cut:14} -{}", whole as i64 - total(&s) as i64);
                }
            }
        }
    }
}
