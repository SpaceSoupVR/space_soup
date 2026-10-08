//! MEASUREMENT ONLY: the ground's shaders with one part cut out each, built
//! while the pipeline statistics are on so the PIPESTATS log lists what each
//! part costs in instructions and registers -- the scene readers (the full
//! one and the outdoor one, baked and spotless), their weather twin, and the
//! probe pass. Nothing draws with them.
//!
//! 2026-10-08: the readers went from 19 registers to 20 (62% -> 50%
//! occupancy) and the full one to 3,398 instructions, past the instruction
//! cache (wall #20), over a night of cliff texturing, weather and time of
//! day; the weather twins sat at 3,732-4,247. One deploy with these says
//! which additions moved them.

use crate::renderer::terrain_pipeline::{terrain_shader_for, TerrainPipeline, TerrainRole};
use wgpu::{BindGroupLayout, Device, TextureFormat};

/// Each cut: a label and the text edits of the generated WGSL that take one
/// part out. A cut whose text no longer matches exactly once is skipped with
/// a warning (and fails `every_ground_cut_matches_its_shaders`).
pub(crate) const GROUND_CUTS: &[(&str, &[(&str, &str)])] = &[
    ("none", &[]),
    // The eye at night (`tonemap::wgsl_tonemap_block`): its mix, by day 0.
    ("purkinje", &[("    exposed = mix(exposed, rods, camera.proxy_params.y);\n", "")]),
    // No steep ground: every steep-face read (the cliff's side planes, its
    // warp and bands, the second scale, the minor plane) dead.
    (
        "gentle",
        &[
            ("    f.biplanar  = select(0.0, 1.0, slope_deg >= mat.biplanar_start_deg);\n", "    f.biplanar  = 0.0;\n"),
            ("    let steep = slope_deg >= mat.biplanar_start_deg;\n", "    let steep = false;\n"),
        ],
    ),
    // Only steep ground: the planar reads dead (`ground_twins`' other side).
    (
        "steep",
        &[
            ("    f.biplanar  = select(0.0, 1.0, slope_deg >= mat.biplanar_start_deg);\n", "    f.biplanar  = 1.0;\n"),
            ("    let steep = slope_deg >= mat.biplanar_start_deg;\n", "    let steep = true;\n"),
        ],
    ),
    // The brushes' repeated code written once (`brush_pipeline::DEDUP_EDITS`).
    ("dedup_all", crate::renderer::brush_pipeline::DEDUP_EDITS),
    // The cliff's bands: the macro map read a second time.
    ("band", &[("    if (steep) {\n        let ms = 1.0 / CLIFF_BAND_SCALE_M;", "    if (false) {\n        let ms = 1.0 / CLIFF_BAND_SCALE_M;")]),
    // The cliff's uv warp altogether (the bands with it).
    (
        "warp",
        &[("    f.major_uv  = f.major_uv + warp;\n    f.minor_uv  = f.minor_uv + warp;\n    f.detail_uv = f.detail_uv + warp;\n", "")],
    ),
    // The colour's second scale on the face the cliff looks along.
    (
        "colour2",
        &[(
            "    let c_major = mix(\n        textureSampleGrad(layer_tex, layer_samp, f.major_uv / r, layer, f.major_ddx / r, f.major_ddy / r).rgb,\n",
            "    let c_major = textureSampleGrad(layer_tex, layer_samp, f.major_uv / r, layer, f.major_ddx / r, f.major_ddy / r).rgb;\n    let c_unused = mix(\n        vec3<f32>(0.0),\n",
        )],
    ),
    // The normal's second scale on a cliff.
    ("normal2", &[("        tn = vec3<f32>(mix(tn.xy, t2.xy, k) * 1.3, mix(tn.z, t2.z, k));\n", "")]),
    // The minor plane's normal blended in near a 45-degree turn.
    ("minor_normal", &[("        if (w_major < 0.999 && f.biplanar > 0.5) {", "        if (false) {")]),
    // A side plane's normal at all: the top plane's everywhere.
    ("side_normal", &[("    if (f.detail_axis != 1u) {", "    if (false) {")]),
    // Wet sand.
    ("wet", &[("    albedo = albedo * (1.0 - 0.45 * (1.0 - smoothstep(mat.wet_line, mat.wet_line + mat.wet_band, in.tex_pos.y)));\n", "")]),
    // THE WEATHER'S PARTS (its twins only): the snow's, the rain's rings,
    // and the noise that breaks both edges.
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

/// The shaders the cuts are measured on, by label: the readers as the frames
/// choose them (`XrRenderer::terrain_reader`), the outdoor one's weather twin
/// and the probe pass. The probe pass first.
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

/// `src` with cut `edits`, or `None` when one does not match exactly once.
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

/// MEASUREMENT ONLY: every base with every cut that applies to it, one
/// pipeline each, labelled `gcut_<base>_<cut>`, built for the PIPESTATS log.
#[allow(clippy::too_many_arguments)]
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

    /// Every cut applies to the readers (the weather's to its twin), and each
    /// cut shader still validates.
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

    /// WHERE THE GROUND READERS' SIZE GOES, by the proxy: each cut's saving.
    /// `cargo test --lib ground_cut_sizes -- --ignored --nocapture`.
    #[test]
    #[ignore]
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
