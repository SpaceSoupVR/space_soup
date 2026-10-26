use crate::renderer::terrain_pipeline::{terrain_shader_for, TerrainPipeline, TerrainRole};
use wgpu::{BindGroupLayout, Device, TextureFormat};

pub(crate) const GROUND_CUTS: &[(&str, &[(&str, &str)])] = &[
    ("none", &[]),

    ("purkinje", &[("    exposed = mix(exposed, rods, camera.proxy_params.y);\n", "")]),

    (
        "gentle",
        &[
            ("    f.biplanar  = select(0.0, 1.0, slope_deg >= mat.biplanar_start_deg);\n", "    f.biplanar  = 0.0;\n"),
            ("    let steep = slope_deg >= mat.biplanar_start_deg;\n", "    let steep = false;\n"),
        ],
    ),

    (
        "steep",
        &[
            ("    f.biplanar  = select(0.0, 1.0, slope_deg >= mat.biplanar_start_deg);\n", "    f.biplanar  = 1.0;\n"),
            ("    let steep = slope_deg >= mat.biplanar_start_deg;\n", "    let steep = true;\n"),
        ],
    ),

    ("dedup_all", crate::renderer::brush_pipeline::DEDUP_EDITS),

    ("band", &[("    if (steep) {\n        let ms = 1.0 / CLIFF_BAND_SCALE_M;", "    if (false) {\n        let ms = 1.0 / CLIFF_BAND_SCALE_M;")]),

    (
        "warp",
        &[("    f.major_uv  = f.major_uv + warp;\n    f.minor_uv  = f.minor_uv + warp;\n    f.detail_uv = f.detail_uv + warp;\n", "")],
    ),

    (
        "colour2",
        &[(
            "    let c_major = mix(\n        textureSampleGrad(layer_tex, layer_samp, f.major_uv / r, layer, f.major_ddx / r, f.major_ddy / r).rgb,\n",
            "    let c_major = textureSampleGrad(layer_tex, layer_samp, f.major_uv / r, layer, f.major_ddx / r, f.major_ddy / r).rgb;\n    let c_unused = mix(\n        vec3<f32>(0.0),\n",
        )],
    ),

    ("normal2", &[("        tn = vec3<f32>(mix(tn.xy, t2.xy, k) * 1.3, mix(tn.z, t2.z, k));\n", "")]),

    ("minor_normal", &[("        if (w_major < 0.999 && f.biplanar > 0.5) {", "        if (false) {")]),

    ("side_normal", &[("    if (f.detail_axis != 1u) {", "    if (false) {")]),

    ("wet", &[("    albedo = albedo * (1.0 - 0.45 * (1.0 - smoothstep(mat.wet_line, mat.wet_line + mat.wet_band, in.tex_pos.y)));\n", "")]),

    ("wx_snow", &[("const WX_SNOW: bool = true;", "const WX_SNOW: bool = false;")]),
    ("wx_water", &[("const WX_WATER: bool = true;", "const WX_WATER: bool = false;")]),
    ("wx_dry", &[("const WX_SNOW: bool = true;", "const WX_SNOW: bool = false;"), ("const WX_WATER: bool = true;", "const WX_WATER: bool = false;")]),
    (
        "gentle_snowless",
        &[
            ("    f.biplanar  = select(0.0, 1.0, slope_deg >= mat.biplanar_start_deg);\n", "    f.biplanar  = 0.0;\n"),
            ("    let steep = slope_deg >= mat.biplanar_start_deg;\n", "    let steep = false;\n"),
            ("const WX_SNOW: bool = true;", "const WX_SNOW: bool = false;"),
        ],
    ),
    (
        "gentle_waterless",
        &[
            ("    f.biplanar  = select(0.0, 1.0, slope_deg >= mat.biplanar_start_deg);\n", "    f.biplanar  = 0.0;\n"),
            ("    let steep = slope_deg >= mat.biplanar_start_deg;\n", "    let steep = false;\n"),
            ("const WX_WATER: bool = true;", "const WX_WATER: bool = false;"),
        ],
    ),
    ("wx_rings", &[("        if (at.area.x > 0.0 && at.area.y < 0.5 && s.water > 0.0) {", "        if (false) {")]),
    ("wx_noise", &[("        fine = 0.65 * wx_noise(p.xz * 3.1) + 0.35 * wx_noise(p.xz * 9.7);", "        fine = 0.5;")]),
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
                let expected = !(cut.starts_with("wx_") || cut.ends_with("less")) || weathered;
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
    #[ignore = "diagnostic that prints sizes or timings or writes files; run by hand with --ignored"]
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
