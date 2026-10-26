pub const STEREO_VIEWS: u32 = 2;

pub const STEREO_VIEW_MASK: Option<core::num::NonZeroU32> =
    core::num::NonZeroU32::new((1u32 << STEREO_VIEWS) - 1);

const VIEW_PARAM: &str = "@builtin(view_index) view_index_in: u32";

pub fn as_multiview(src: &str) -> String {
    let mut out = String::with_capacity(src.len() + 256);

    if next_entry_point(src).is_some() && !src.contains("var<private> view_slot") {
        out.push_str("var<private> view_slot: i32 = 0;\n");
    }
    let mut rest = src;
    loop {
        let Some((stage_at, _stage)) = next_entry_point(rest) else {
            out.push_str(rest);
            return out;
        };

        out.push_str(&rest[..stage_at]);
        rest = &rest[stage_at..];

        let Some(open) = rest.find('(') else {
            out.push_str(rest);
            return out;
        };
        let Some(close) = matching_paren(rest, open) else {
            out.push_str(rest);
            return out;
        };
        let params = &rest[open + 1..close];
        out.push_str(&rest[..=open]);
        out.push_str(params);
        if params.trim().is_empty() {
            out.push_str(VIEW_PARAM);
        } else {
            out.push_str(", ");
            out.push_str(VIEW_PARAM);
        }
        rest = &rest[close..];

        let Some(brace) = rest.find('{') else {
            out.push_str(rest);
            return out;
        };
        out.push_str(&rest[..=brace]);

        out.push_str("\n    view_slot = i32(view_index_in);");
        rest = &rest[brace + 1..];
    }
}

fn next_entry_point(src: &str) -> Option<(usize, &'static str)> {
    let v = src.find("@vertex").map(|i| (i, "@vertex"));
    let f = src.find("@fragment").map(|i| (i, "@fragment"));
    match (v, f) {
        (Some(a), Some(b)) => Some(if a.0 < b.0 { a } else { b }),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    }
}

fn matching_paren(src: &str, open: usize) -> Option<usize> {
    let bytes = src.as_bytes();
    let mut depth = 0usize;
    for (i, b) in bytes.iter().enumerate().skip(open) {
        match b {
            b'(' => depth += 1,
            b')' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
    }
    None
}


#[cfg(test)]
pub fn multiview_validation_error(src: &str) -> Option<String> {
    multiview_validation_error_of(src, true)
}

#[cfg(test)]
pub fn multiview_validation_error_of(src: &str, transform: bool) -> Option<String> {
    use wgpu::naga;
    let transformed = if transform { as_multiview(src) } else { src.to_string() };
    let module = match naga::front::wgsl::parse_str(&transformed) {
        Ok(m) => m,
        Err(e) => return Some(format!("parse: {}", e.emit_to_string(&transformed))),
    };
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),

        naga::valid::Capabilities::all(),
    );
    match validator.validate(&module) {
        Ok(_) => None,
        Err(e) => Some(format!("validate: {e:?}")),
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum ViewMode {

    Mono,

    Stereo,
}

impl ViewMode {

    pub fn mask(self) -> Option<core::num::NonZeroU32> {
        match self {
            Self::Mono => None,
            Self::Stereo => STEREO_VIEW_MASK,
        }
    }

    pub fn shader(self, src: String) -> String {
        match self {
            Self::Mono => src,
            Self::Stereo => as_multiview(&src),
        }
    }

    pub fn is_stereo(self) -> bool {
        self == Self::Stereo
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_shader_transformed_for_multiview_is_never_built_without_the_mask() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/renderer");
        let mut offenders = Vec::new();
        for entry in std::fs::read_dir(&dir).expect("renderer source directory") {
            let path = entry.expect("dir entry").path();
            if path.extension().and_then(|e| e.to_str()) != Some("rs") {
                continue;
            }
            let src = std::fs::read_to_string(&path).expect("read source");
            if path.file_name().and_then(|n| n.to_str()) == Some("multiview.rs") {
                continue;
            }
            let shapes = src.matches("view.shader(").count();
            let masks = src.matches("view.mask()").count();
            if shapes > 0 && masks == 0 {
                offenders.push(format!(
                    "{}: transforms a shader for multiview ({shapes}x) and never sets a view mask",
                    path.file_name().unwrap().to_string_lossy(),
                ));
            }
        }
        assert!(
            offenders.is_empty(),
            "a pipeline carries one half of multiview:\n  {}",
            offenders.join("\n  "),
        );
    }

    #[test]
    fn every_scene_pipeline_has_a_stereo_constructor() {
        use crate::renderer::{
            brush_pipeline::{BrushPipeline, BrushSealPipeline},
            layered_mesh_pipeline::LayeredMeshPipeline,
            mesh_pipeline::{MeshPipeline, SkinnedMeshPipeline},
            pipeline::{SolidPipeline, WirePipeline},
            sky::SkyPipeline,
            terrain_pipeline::TerrainPipeline,
            water_pipeline::WaterPipeline,
        };

        let _: fn(&wgpu::Device, wgpu::TextureFormat, &wgpu::BindGroupLayout, u32) -> SolidPipeline =
            SolidPipeline::new_multisampled_stereo;
        let _: fn(&wgpu::Device, wgpu::TextureFormat, &wgpu::BindGroupLayout, u32) -> WirePipeline =
            WirePipeline::new_multisampled_stereo;
        let _: fn(&wgpu::Device, wgpu::TextureFormat, &wgpu::BindGroupLayout, u32) -> MeshPipeline =
            MeshPipeline::new_multisampled_stereo;
        let _: fn(
            &wgpu::Device,
            wgpu::TextureFormat,
            &wgpu::BindGroupLayout,
            u32,
        ) -> SkinnedMeshPipeline = SkinnedMeshPipeline::new_multisampled_stereo;
        let _: fn(&wgpu::Device, wgpu::TextureFormat, &wgpu::BindGroupLayout, u32) -> TerrainPipeline =
            TerrainPipeline::new_multisampled_stereo;
        let _: fn(&wgpu::Device, wgpu::TextureFormat, &wgpu::BindGroupLayout, u32) -> SkyPipeline =
            SkyPipeline::new_stereo;
        let _: fn(&wgpu::Device, wgpu::TextureFormat, &wgpu::BindGroupLayout, u32) -> WaterPipeline =
            WaterPipeline::new_stereo;
        let _: fn(
            &wgpu::Device,
            wgpu::TextureFormat,
            &wgpu::BindGroupLayout,
            &wgpu::BindGroupLayout,
            u32,
        ) -> LayeredMeshPipeline = LayeredMeshPipeline::new_multisampled_stereo;
        let _: fn(&wgpu::Device, wgpu::TextureFormat, &wgpu::BindGroupLayout, u32) -> BrushPipeline =
            BrushPipeline::new_multisampled_stereo;
        let _: fn(&wgpu::Device, wgpu::TextureFormat, &wgpu::BindGroupLayout, u32) -> BrushPipeline =
            BrushPipeline::new_multisampled_opaque_stereo;
        let _: fn(&wgpu::Device, wgpu::TextureFormat, &wgpu::BindGroupLayout, u32) -> BrushPipeline =
            BrushPipeline::new_multisampled_sources_stereo;
        let _: fn(
            &wgpu::Device,
            wgpu::TextureFormat,
            &wgpu::BindGroupLayout,
            u32,
        ) -> BrushSealPipeline = BrushSealPipeline::new_stereo;
        let _: fn(
            &wgpu::Device,
            wgpu::TextureFormat,
            &wgpu::BindGroupLayout,
            &wgpu::BindGroupLayout,
            &wgpu::BindGroupLayout,
            u32,
        ) -> crate::renderer::effects::EffectsPipeline = crate::renderer::effects::EffectsPipeline::new_multisampled_stereo;
    }

    #[test]
    fn every_scene_shader_survives_the_multiview_transform() {
        use crate::renderer::{
            brush_pipeline, effects, glare, layered_mesh_pipeline, mesh_pipeline, particle, pipeline, sky,
            terrain_pipeline, water_pipeline,
        };
        let shaders: Vec<(&str, String)> = vec![
            ("solid", pipeline::solid_shader_src()),
            ("solid_ssr", pipeline::solid_ssr_shader_src()),
            ("wire", pipeline::wire_shader_src()),
            ("brush", brush_pipeline::brush_shader_src()),
            ("brush_seal", brush_pipeline::brush_seal_shader_src()),
            ("mesh", mesh_pipeline::mesh_shader_src()),
            ("skinned_mesh", mesh_pipeline::skinned_mesh_shader_src()),
            ("layered_mesh", layered_mesh_pipeline::layered_mesh_shader_src()),
            ("terrain", terrain_pipeline::terrain_shader_src()),
            ("water", water_pipeline::water_shader_src()),
            ("sky", sky::sky_shader_src()),
            ("particle", particle::particle_shader()),
            ("glare", glare::glare_shader()),
            ("effects", effects::effects_shader()),
        ];
        let mut broken = Vec::new();
        for (name, src) in &shaders {

            if let Some(e) = multiview_validation_error_of(src, false) {
                broken.push(format!("{name} (BEFORE the transform): {e}"));
                continue;
            }
            if let Some(e) = multiview_validation_error(src) {
                broken.push(format!("{name}: {e}"));
            }

            let one_camera = *name != "solid_ssr" && src.match_indices("view_proj: mat4x4<f32>").any(|(at, _)| {

                !src[..at].chars().next_back().is_some_and(|c| c.is_ascii_alphanumeric() || c == '_')
            });
            if one_camera {
                broken.push(format!("{name}: its camera is one matrix, not one a view (`view_proj[view_slot]`)"));
            }
        }
        assert!(
            broken.is_empty(),
            "{} of {} scene shaders do not validate as multiview:\n\n{}",
            broken.len(),
            shaders.len(),
            broken.join("\n\n"),
        );
    }

    #[test]
    fn a_view_mode_cannot_carry_one_half_of_multiview() {
        assert_eq!(ViewMode::Mono.mask(), None);
        assert_eq!(ViewMode::Mono.shader("@vertex\nfn v() {}".into()), "@vertex\nfn v() {}");
        assert_eq!(ViewMode::Stereo.mask(), STEREO_VIEW_MASK);
        assert!(
            ViewMode::Stereo.shader("@vertex\nfn v() {}".into()).contains("view_index_in: u32"),
            "the stereo mode carries the mask but not the transform, so both \
             eyes would be drawn with the left eye's camera",
        );
    }

    #[test]
    fn the_stereo_mask_names_both_layers_and_not_the_second_one() {

        assert_eq!(STEREO_VIEW_MASK.map(core::num::NonZeroU32::get), Some(0b11));
        assert_eq!(STEREO_VIEWS, 2);
    }

    #[test]
    fn an_entry_point_gains_the_builtin_and_records_it() {
        let src = "@vertex\nfn vs_main(v: VIn) -> VOut {\n    var out: VOut;\n    return out;\n}\n";
        let got = as_multiview(src);
        assert!(
            got.contains("fn vs_main(v: VIn, @builtin(view_index) view_index_in: u32) -> VOut"),
            "parameter not appended:\n{got}",
        );
        assert!(
            got.contains("-> VOut {\n    view_slot = i32(view_index_in);"),
            "the view was not recorded at the top of the body:\n{got}",
        );
    }

    #[test]
    fn both_stages_are_rewritten() {

        let src = "@vertex\nfn vs(v: VIn) -> VOut { return v; }\n\
                   @fragment\nfn fs(in: VOut) -> @location(0) vec4<f32> { return c; }\n";
        let got = as_multiview(src);
        assert_eq!(got.matches("view_slot = i32(view_index_in);").count(), 2, "{got}");
    }

    #[test]
    fn an_entry_point_with_no_parameters_gets_no_stray_comma() {
        let src = "@fragment\nfn fs() -> @location(0) vec4<f32> { return c; }\n";
        let got = as_multiview(src);
        assert!(got.contains("fn fs(@builtin(view_index) view_index_in: u32)"), "{got}");
        assert!(!got.contains("(, "), "stray comma in:\n{got}");
    }

    #[test]
    fn ordinary_functions_are_left_alone() {

        let src = "fn helper(x: f32) -> f32 { return x; }\n\
                   @vertex\nfn vs(v: VIn) -> VOut { return v; }\n";
        let got = as_multiview(src);
        assert!(got.contains("fn helper(x: f32) -> f32 { return x; }"), "{got}");
        assert_eq!(got.matches("view_index_in").count(), 2, "helper was touched:\n{got}");
    }

    #[test]
    fn a_shader_with_no_entry_points_is_unchanged() {
        let src = "fn a(x: f32) -> f32 { return x; }\nconst K: f32 = 1.0;\n";
        assert_eq!(as_multiview(src), src);
    }

    #[test]
    fn a_parameter_list_spanning_lines_is_still_closed_correctly() {
        let src = "@vertex\nfn vs(\n    v: VIn,\n    i: Inst,\n) -> VOut {\n    return o;\n}\n";
        let got = as_multiview(src);
        assert!(got.contains("i: Inst,\n, @builtin(view_index)") || got.contains(", @builtin(view_index) view_index_in: u32\n) -> VOut") || got.contains("@builtin(view_index) view_index_in: u32"), "{got}");
        assert_eq!(got.matches("view_slot = i32(view_index_in);").count(), 1, "{got}");
    }
}
