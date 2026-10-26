use crate::renderer::ground_cuts::{ground_cut_bases, with_cut};
use crate::renderer::terrain_pipeline::{TerrainPipeline, TerrainRole};
use wgpu::{BindGroupLayout, Device, TextureFormat};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Picture {

    Same,

    Near,

    Probe,
}

pub(crate) struct Opt {
    pub label: &'static str,
    pub picture: Picture,
    pub edit: fn(&str) -> Option<String>,
}

fn every(src: &str, from: &str, to: &str) -> Option<String> {
    src.contains(from).then(|| src.replace(from, to))
}

fn once(src: &str, from: &str, to: &str) -> Option<String> {
    (src.matches(from).count() == 1).then(|| src.replacen(from, to, 1))
}

fn yaw_probe(src: &str) -> Option<String> {
    every(
        src,
        "    let s = sin(yaw);\n    let c = cos(yaw);\n",
        "    let s = camera.sky_params.z;\n    let c = camera.sky_params.w;\n",
    )
}

const SKY_AS_SHIPPED: &str = "    var e = vec3<f32>(0.0);
    e = e + camera.sky_sh[0].rgb * 0.282095 * 1.0;
    e = e + camera.sky_sh[1].rgb * (0.488603 * y) * 0.6666667;
    e = e + camera.sky_sh[2].rgb * (0.488603 * z) * 0.6666667;
    e = e + camera.sky_sh[3].rgb * (0.488603 * x) * 0.6666667;
    e = e + camera.sky_sh[4].rgb * (1.092548 * x * y) * 0.25;
    e = e + camera.sky_sh[5].rgb * (1.092548 * y * z) * 0.25;
    e = e + camera.sky_sh[6].rgb * (0.315392 * (3.0 * z * z - 1.0)) * 0.25;
    e = e + camera.sky_sh[7].rgb * (1.092548 * x * z) * 0.25;
    e = e + camera.sky_sh[8].rgb * (0.546274 * (x * x - y * y)) * 0.25;
";

pub(crate) const SKY_FOLDED: &str = "    var e = camera.sky_sh[0].rgb * 0.282095;
    e = e + (camera.sky_sh[1].rgb * 0.3257353) * y;
    e = e + (camera.sky_sh[2].rgb * 0.3257353) * z;
    e = e + (camera.sky_sh[3].rgb * 0.3257353) * x;
    e = e + (camera.sky_sh[4].rgb * 0.273137) * (x * y);
    e = e + (camera.sky_sh[5].rgb * 0.273137) * (y * z);
    e = e + (camera.sky_sh[6].rgb * 0.078848) * (3.0 * z * z - 1.0);
    e = e + (camera.sky_sh[7].rgb * 0.273137) * (x * z);
    e = e + (camera.sky_sh[8].rgb * 0.1365685) * (x * x - y * y);
";

fn sky_folded(src: &str) -> Option<String> {
    once(src, SKY_AS_SHIPPED, SKY_FOLDED)
}

fn dims_probe(src: &str) -> Option<String> {
    let mut s = every(src, "textureDimensions(tex)", "vec2<u32>(2048u, 2048u)")?;
    s = s.replace("f32(textureDimensions(sun_dynamic_shadow_tex).x)", "2048.0");
    s = s.replace("    let dims = textureDimensions(probe_pass_tex);\n", "    let dims = vec2<u32>(1024u, 1104u);\n");
    Some(s)
}

fn layer_rcp(src: &str) -> Option<String> {
    let mut s = every(src, " / r, ", " * ir, ")?;
    s = s.replace(" / r,\n", " * ir,\n").replace(" / r)", " * ir)");
    every(&s, "    let r = max(mat.repeat[layer], 0.001);\n", "    let r = max(mat.repeat[layer], 0.001);\n    let ir = 1.0 / r;\n")
}

fn layer_inv_probe(src: &str) -> Option<String> {
    let mut s = every(src, " / r, ", " * ir, ")?;
    s = s.replace(" / r,\n", " * ir,\n").replace(" / r)", " * ir)");
    every(&s, "    let r = max(mat.repeat[layer], 0.001);\n", "    let r = max(mat.repeat[layer], 0.001);\n    let ir = mat.repeat[layer];\n")
}

fn lamp_probe(src: &str) -> Option<String> {
    let mut s = every(src, "        let d_over_r = dist / max(l.params.x, 0.0001);\n", "        let d_over_r = dist * l.params.x;\n")?;
    s = every(&s, "    let radiance = l.color_intensity.rgb * l.color_intensity.a;\n", "    let radiance = l.color_intensity.rgb;\n")?;
    s = s.replace(
        "        let d2_over_r2 = dist_sq / max(range * range, 1e-8);\n",
        "        let d2_over_r2 = dist_sq * lights.lights[i].position.w;\n",
    );
    s = s.replace(
        "        let radiance = lights.lights[i].color_intensity.rgb * lights.lights[i].color_intensity.a;\n",
        "        let radiance = lights.lights[i].color_intensity.rgb;\n",
    );
    s = s.replace(
        "(dot(-l_dir, lights.lights[i].direction.xyz) - cos_outer) / (lights.lights[i].direction.w - cos_outer)",
        "(dot(-l_dir, lights.lights[i].direction.xyz) - cos_outer) * lights.lights[i].direction.w",
    );
    s = s.replace(
        "dist_sq >= lights.lights[i].params.x * lights.lights[i].params.x",
        "dist_sq >= lights.lights[i].params.x",
    );
    Some(s)
}

const TENT_AS_SHIPPED: &str = "    for (var t = 0; t < 9; t = t + 1) {
        let x = pcf_tent_tap_axis(f32(t % 3) - 1.0, f.x);
        let y = pcf_tent_tap_axis(f32(t / 3) - 1.0, f.y);
        sum += x.y * y.y * pcf_tent_tap(tex, tile, grid, (corner + vec2<f32>(x.x, y.x)) * tile_texel, lo, hi, c.z);
    }
";

pub(crate) const TENT_BY_ROW: &str = "    for (var ty = 0; ty < 3; ty = ty + 1) {
        let y = pcf_tent_tap_axis(f32(ty) - 1.0, f.y);
        for (var tx = 0; tx < 3; tx = tx + 1) {
            let x = pcf_tent_tap_axis(f32(tx) - 1.0, f.x);
            sum += x.y * y.y * pcf_tent_tap(tex, tile, grid, (corner + vec2<f32>(x.x, y.x)) * tile_texel, lo, hi, c.z);
        }
    }
";

fn tent_by_row(src: &str) -> Option<String> {
    once(src, TENT_AS_SHIPPED, TENT_BY_ROW)
}

pub(crate) const ACOS_FIT: &str = "    let slope_x = abs(clamp(n.y, -1.0, 1.0));
    let slope_a = sqrt(1.0 - slope_x) * (1.5707288 + slope_x * (-0.2121144 + slope_x * (0.0742610 - 0.0187293 * slope_x)));
    let slope_deg = degrees(select(3.14159265 - slope_a, slope_a, n.y >= 0.0));
";

fn acos_fit(src: &str) -> Option<String> {
    once(src, "    let slope_deg = degrees(acos(clamp(n.y, -1.0, 1.0)));\n", ACOS_FIT)
}

fn splat_probe(src: &str) -> Option<String> {
    once(src, "    return select(derived, authored, mat.use_splat > 0.5);\n", "    return authored;\n")
}

fn expand_once(src: &str) -> Option<String> {
    once(
        src,
        "    return vec4<f32>(probe_pass_expand(pre.rgb / max(pre.a, 1e-4)), pre.a);\n",
        "    let pre_a = max(pre.a, 1e-4);\n    return vec4<f32>(pre.rgb / max(pre_a - dot(pre.rgb, vec3<f32>(0.2126, 0.7152, 0.0722)), 1e-3 * pre_a), pre.a);\n",
    )
}

const FACES_AS_SHIPPED: &str = "    let face_dx = abs(codes_i / vec4<i32>(32) - vec4<i32>(code_i / 32));
    let face_dy = abs(codes_i % vec4<i32>(32) - vec4<i32>(code_i % 32));
";

pub(crate) const FACES_BY_SHIFT: &str = "    let face_dx = abs((codes_i >> vec4<u32>(5u)) - vec4<i32>(code_i >> 5u));
    let face_dy = abs((codes_i & vec4<i32>(31)) - vec4<i32>(code_i & 31));
";

fn face_shift(src: &str) -> Option<String> {
    once(src, FACES_AS_SHIPPED, FACES_BY_SHIFT)
}

fn axis_select(src: &str) -> Option<String> {
    let s = every(src, "let a = (axis + 1) % 3;", "let a = select(axis + 1, axis - 2, axis >= 2);")?;
    every(&s, "let b = (axis + 2) % 3;", "let b = select(axis + 2, axis - 1, axis >= 1);")
}

fn frame_was(src: &str) -> Option<String> {
    let mut s = src.to_string();
    for (from, to) in [
        (
            "struct SampleFrame {\n",
            "struct SampleFrame {\n    planar_uv: vec2<f32>,\n    planar_ddx: vec2<f32>,\n    planar_ddy: vec2<f32>,\n    major_uv: vec2<f32>,\n    major_ddx: vec2<f32>,\n    major_ddy: vec2<f32>,\n",
        ),
        (
            "    let planar_ddx = world_ddx.xz;\n    let planar_ddy = world_ddy.xz;\n",
            "    f.planar_uv  = planar_uv;\n    f.planar_ddx = world_ddx.xz;\n    f.planar_ddy = world_ddy.xz;\n    let planar_ddx = f.planar_ddx;\n    let planar_ddy = f.planar_ddy;\n",
        ),
        (
            "    let major_ddx = select(select(world_ddx.xy, world_ddx.xz, b.axis_major == 1u), world_ddx.zy, b.axis_major == 0u);\n    let major_ddy = select(select(world_ddy.xy, world_ddy.xz, b.axis_major == 1u), world_ddy.zy, b.axis_major == 0u);\n",
            "    f.major_ddx = select(select(world_ddx.xy, world_ddx.xz, b.axis_major == 1u), world_ddx.zy, b.axis_major == 0u);\n    f.major_ddy = select(select(world_ddy.xy, world_ddy.xz, b.axis_major == 1u), world_ddy.zy, b.axis_major == 0u);\n    let major_ddx = f.major_ddx;\n    let major_ddy = f.major_ddy;\n",
        ),
        (
            "    f.detail_uv = f.detail_uv + select(warp, vec2<f32>(0.0), f.detail_axis == 1u);\n",
            "    f.detail_uv = f.detail_uv + select(warp, vec2<f32>(0.0), f.detail_axis == 1u);\n    f.major_uv  = f.detail_uv;\n",
        ),
        (
            "    var c = textureSampleGrad(layer_tex, layer_samp, f.detail_uv / r, layer, f.detail_ddx / r, f.detail_ddy / r).rgb;\n",
            "    var c = vec3<f32>(0.0);\n    if (f.detail_axis == 1u) {\n        c = textureSampleGrad(layer_tex, layer_samp, f.planar_uv / r, layer, f.planar_ddx / r, f.planar_ddy / r).rgb;\n    } else {\n        c = textureSampleGrad(layer_tex, layer_samp, f.major_uv / r, layer, f.major_ddx / r, f.major_ddy / r).rgb;\n    }\n",
        ),
    ] {
        s = once(&s, from, to)?;
    }
    Some(s)
}

pub(crate) const OPTS: &[Opt] = &[
    Opt { label: "none", picture: Picture::Same, edit: |s| Some(s.to_string()) },
    Opt { label: "yaw_probe", picture: Picture::Probe, edit: yaw_probe },
    Opt { label: "sky_folded", picture: Picture::Same, edit: sky_folded },
    Opt { label: "dims_probe", picture: Picture::Probe, edit: dims_probe },
    Opt { label: "layer_rcp", picture: Picture::Same, edit: layer_rcp },
    Opt { label: "layer_inv_probe", picture: Picture::Probe, edit: layer_inv_probe },
    Opt { label: "lamp_probe", picture: Picture::Probe, edit: lamp_probe },
    Opt { label: "tent_by_row", picture: Picture::Same, edit: tent_by_row },
    Opt { label: "acos_fit", picture: Picture::Same, edit: acos_fit },
    Opt { label: "splat_probe", picture: Picture::Probe, edit: splat_probe },
    Opt { label: "expand_once", picture: Picture::Same, edit: expand_once },
    Opt { label: "face_shift", picture: Picture::Same, edit: face_shift },
    Opt { label: "axis_select", picture: Picture::Same, edit: axis_select },
    Opt { label: "frame_was", picture: Picture::Near, edit: frame_was },
];

#[cfg(test)]
pub(crate) const OPT_ALL: &str = "opt_all";

pub(crate) fn edit_named(src: &str, name: &str) -> Option<String> {
    let label = name.strip_prefix("opt_")?;
    let mut out = src.to_string();
    let mut known = false;
    for opt in OPTS.iter().filter(|o| o.picture != Picture::Probe && o.label != "none") {
        if (label == "all" && opt.picture == Picture::Same) || label == opt.label {
            known = true;
            if let Some(edited) = (opt.edit)(&out) {
                out = edited;
            }
        }
    }
    known.then_some(out)
}

pub(crate) fn ground_bases() -> Vec<(&'static str, TerrainRole, bool, String)> {
    let gentle = crate::renderer::ground_cuts::GROUND_CUTS.iter().find(|(label, _)| *label == "gentle").map(|(_, edits)| *edits);
    let steep = crate::renderer::ground_cuts::GROUND_CUTS.iter().find(|(label, _)| *label == "steep").map(|(_, edits)| *edits);
    let mut out = Vec::new();
    for (base, role, weathered, src) in ground_cut_bases() {
        match base {
            "pass" => out.push(("gpass", role, weathered, src)),
            "bs" => {
                if let Some(g) = gentle.and_then(|edits| with_cut(&src, edits)) {
                    out.push(("ggentle", role, weathered, g));
                }
                if let Some(g) = steep.and_then(|edits| with_cut(&src, edits)) {
                    out.push(("gsteep", role, weathered, g));
                }
                out.push(("gread", role, weathered, src));
            }
            "wx_bs" => {
                if let Some(g) = gentle.and_then(|edits| with_cut(&src, edits)) {
                    out.push(("gwx", role, weathered, g));
                }
            }
            _ => {}
        }
    }
    out
}

pub fn log_ground_opts(
    device: &Device,
    format: TextureFormat,
    uniform_layout: &BindGroupLayout,
    samples: u32,
    probe_layout: &BindGroupLayout,
    fixups: &crate::renderer::probe_fixup::ProbeFixups,
) {
    let weather_layout = crate::renderer::weather::bind_group_layout(device);
    for (base, role, weathered, src) in ground_bases() {
        for opt in OPTS {
            let Some(source) = (opt.edit)(&src) else {
                continue;
            };
            let label = format!("opt_{base}_{}", opt.label);
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

    fn validates(label: &str, src: &str) {
        let src = crate::renderer::multiview::ViewMode::Mono.shader(src.to_string());
        let module = wgpu::naga::front::wgsl::parse_str(&src).unwrap_or_else(|e| panic!("{label}: {}", e.emit_to_string(&src)));
        wgpu::naga::valid::Validator::new(wgpu::naga::valid::ValidationFlags::all(), wgpu::naga::valid::Capabilities::all())
            .validate(&module)
            .unwrap_or_else(|e| panic!("{label}: {e:?}"));
    }

    fn all_bases() -> Vec<(String, String)> {
        let mut bases: Vec<(String, String)> = ground_bases().into_iter().map(|(b, _, _, s)| (b.to_string(), s)).collect();
        bases.extend(crate::renderer::brush_pipeline::opt_bases().into_iter().map(|(b, _, s)| (b.to_string(), s)));
        bases
    }

    #[test]
    fn every_candidate_applies_and_validates() {
        let bases = all_bases();
        for opt in OPTS {
            let mut applied = 0;
            for (base, src) in &bases {
                let Some(edited) = (opt.edit)(src) else {
                    continue;
                };
                applied += 1;
                assert_eq!(edited == *src, opt.label == "none", "{} on {base}", opt.label);
                validates(&format!("{} on {base}", opt.label), &edited);
            }
            assert!(applied > 0, "{} applies to no base", opt.label);
            println!("{:16} applies to {applied} of {}", opt.label, bases.len());
        }
    }

    #[test]
    fn the_acos_fit_is_within_its_bound() {
        let mut worst = 0.0f64;
        let mut x = -1.0f64;
        while x <= 1.0 {
            let a = x.abs();
            let f = (1.0 - a).sqrt() * (1.5707288 + a * (-0.2121144 + a * (0.0742610 - 0.0187293 * a)));
            let fit = if x >= 0.0 { f } else { std::f64::consts::PI - f };
            worst = worst.max((fit - x.acos()).abs());
            x += 1e-5;
        }
        assert!(worst < 6.8e-5, "{worst}");
    }

    #[test]
    fn the_face_codes_split_the_same_by_shift_and_mask() {
        for code in 1..=1024i32 {
            for lane in 1..=1024i32 {
                let (a, b) = (lane - 1, code - 1);
                assert_eq!((a / 32 - b / 32, a % 32 - b % 32), ((a >> 5) - (b >> 5), (a & 31) - (b & 31)));
            }
        }
    }

    #[test]
    fn the_reader_edit_names_apply() {
        for (base, src) in all_bases() {
            let all = edit_named(&src, OPT_ALL).unwrap();
            assert_ne!(all, src, "{base}");
            validates(&format!("opt_all on {base}"), &all);
            assert!(edit_named(&src, "opt_sky_folded").is_some_and(|s| s != src), "{base}");
        }
        let (_, src) = &all_bases()[0];
        assert!(edit_named(src, "opt_yaw_probe").is_none());

        let ground = &ground_bases().into_iter().find(|b| b.0 == "gread").unwrap().3;
        assert!(edit_named(ground, "opt_frame_was").is_some_and(|s| s != *ground));
        assert!(!edit_named(ground, OPT_ALL).unwrap().contains("planar_uv: vec2<f32>"));
        assert!(edit_named(src, "opt_no_such").is_none());
        assert!(edit_named(src, "sky_walked").is_none());
    }

    #[test]
    fn the_other_axes_are_the_remainders() {
        for axis in 0..=3i32 {
            assert_eq!((axis + 1) % 3, if axis >= 2 { axis - 2 } else { axis + 1 });
            assert_eq!((axis + 2) % 3, if axis >= 1 { axis - 1 } else { axis + 2 });
        }
    }

    #[test]
    fn the_folded_sky_is_the_shipped_sum() {
        let shipped = |sh: &[[f64; 3]; 9], n: [f64; 3]| {
            let (x, y, z) = (n[0], n[1], n[2]);
            let basis = [
                0.282095 * 1.0,
                (0.488603 * y) * 0.6666667,
                (0.488603 * z) * 0.6666667,
                (0.488603 * x) * 0.6666667,
                (1.092548 * x * y) * 0.25,
                (1.092548 * y * z) * 0.25,
                (0.315392 * (3.0 * z * z - 1.0)) * 0.25,
                (1.092548 * x * z) * 0.25,
                (0.546274 * (x * x - y * y)) * 0.25,
            ];
            (0..3).map(|c| (0..9).map(|k| sh[k][c] * basis[k]).sum::<f64>()).collect::<Vec<_>>()
        };
        let folded = |sh: &[[f64; 3]; 9], n: [f64; 3]| {
            let (x, y, z) = (n[0], n[1], n[2]);
            let basis = [
                0.282095,
                0.3257353 * y,
                0.3257353 * z,
                0.3257353 * x,
                0.273137 * (x * y),
                0.273137 * (y * z),
                0.078848 * (3.0 * z * z - 1.0),
                0.273137 * (x * z),
                0.1365685 * (x * x - y * y),
            ];
            (0..3).map(|c| (0..9).map(|k| sh[k][c] * basis[k]).sum::<f64>()).collect::<Vec<_>>()
        };
        let mut seed = 7u64;
        let mut rnd = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f64) / (1u64 << 31) as f64
        };
        let mut worst = 0.0f64;
        for _ in 0..2000 {
            let mut sh = [[0.0; 3]; 9];
            for k in &mut sh {
                for c in k.iter_mut() {
                    *c = rnd() * 4.0 - 1.0;
                }
            }
            let v = [rnd() * 2.0 - 1.0, rnd() * 2.0 - 1.0, rnd() * 2.0 - 1.0];
            let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-6);
            let n = [v[0] / len, v[1] / len, v[2] / len];
            let (a, b) = (shipped(&sh, n), folded(&sh, n));
            for c in 0..3 {
                worst = worst.max((a[c] - b[c]).abs());
            }
        }

        assert!(worst < 2e-6, "{worst}");
    }

    #[test]
    #[ignore = "measurement, not an assertion: prints shader sizes"]
    fn opt_sizes() {
        for (base, src) in all_bases() {
            let size = |s: &str| {
                let s = crate::renderer::multiview::ViewMode::Mono.shader(s.to_string());
                let total: usize = crate::renderer::shader_inlining::inlined_sizes(&s, "fs_main").iter().map(|f| f.own * f.copies).sum();
                let (_, special) = crate::renderer::shader_audit::specials(&s, "fs_main");
                (total, special.values().sum::<usize>())
            };
            let (base_size, base_special) = size(&src);
            println!("{base}: {base_size} proxy units, {base_special} special");
            for opt in OPTS.iter().skip(1) {
                if let Some(edited) = (opt.edit)(&src) {
                    let (s, sp) = size(&edited);
                    println!("  {:16} {:+5} proxy  {:+4} special", opt.label, s as i64 - base_size as i64, sp as i64 - base_special as i64);
                }
            }
        }
    }
}
