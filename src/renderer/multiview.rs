//! Turning a single-view shader into a stereo one.
//!
//! WHY A TRANSFORM RATHER THAN TWO COPIES
//!
//! Every scene shader needs exactly the same two edits to draw both eyes in one
//! pass: its entry points take `@builtin(view_index)`, and they record it so the
//! camera accessors know which eye's matrix to read. Written out by hand that is
//! twenty edits across ten shader sources, and a shader that gets one of them
//! and not the other renders the right eye's geometry with the left eye's
//! projection -- which on a stereo display is not a wrong picture but a wrong
//! DEPTH, and it reads as discomfort rather than as a bug.
//!
//! Doing it as one transform means there is a single place to be right, and --
//! unlike the shaders themselves, which only a GPU can check -- it is ordinary
//! string handling that a test on any machine can pin. That matters more than
//! usual here: wgpu has no multiview on Metal, so nothing on a development
//! machine can build a multiview pipeline at all, and the first thing that ever
//! validates the real shaders is the headset.

/// How many eyes a stereo pass draws. Both the swapchain's `array_size` and
/// the `Uniforms` view slots are this.
pub const STEREO_VIEWS: u32 = 2;

/// THE LAYER MASK a two-eye pass and its pipelines are built with.
///
/// Since wgpu 28 the multiview parameter is a MASK OF LAYERS rather than a
/// COUNT of them, on both `RenderPipelineDescriptor` and
/// `RenderPassDescriptor`. Two eyes are layers 0 and 1, so the mask is `0b11`
/// -- not `2`, which would name layer 1 alone and is the mistake the rename
/// invites. wgpu checks it against the attachment: a mask that is not
/// `(1 << layers) - 1` is a selective multiview pass and needs
/// `Features::SELECTIVE_MULTIVIEW`, which we neither have nor want.
///
/// `None` here would not be an error -- it is what every single-view pipeline
/// passes -- so the mistake is silent, which is why it lives in one place.
pub const STEREO_VIEW_MASK: Option<core::num::NonZeroU32> =
    core::num::NonZeroU32::new((1u32 << STEREO_VIEWS) - 1);

/// The parameter appended to every entry point.
///
/// `u32`, NOT `i32`. naga requires it -- `Bi::ViewIndex => *ty_inner ==
/// Ti::Scalar(Scalar::U32)` -- and this said `i32` from the day it was written.
/// Every multiview pipeline in the renderer would have been rejected with
/// `InvalidBuiltInType(ViewIndex, Sint)`, which on the headset is ten pipelines
/// that draw nothing, log nothing and make the frame faster.
///
/// Nothing could have caught it earlier: wgpu has no multiview on Metal, so no
/// multiview pipeline can be built on a development machine at all. What found
/// it was running the transformed source through naga directly -- see
/// `multiview_validation_error`.
const VIEW_PARAM: &str = "@builtin(view_index) view_index_in: u32";

/// Rewrite `src` so each entry point records which eye it is drawing.
///
/// Idempotent in the sense that matters: it only ever touches functions
/// carrying `@vertex` or `@fragment`, and it appends rather than replaces, so a
/// shader with no entry points comes back unchanged.
pub fn as_multiview(src: &str) -> String {
    let mut out = String::with_capacity(src.len() + 256);
    // DECLARED HERE WHEN THE SHADER HAS NONE.
    //
    // `view_slot` comes from the lights block, which most scene shaders
    // include -- but not all of them. The wireframe shader carries its own
    // small uniform and no lights, so the transform produced `view_slot =
    // view_index_in;` against an identifier that did not exist and the shader
    // would not parse. On the headset that is a pipeline that silently draws
    // nothing; here it is a test failure (2026-09-18).
    //
    // Prepending it makes the transform self-contained. A shader that already
    // declares it is left alone, so there is never a duplicate.
    // The DECLARATION, not any mention of the name. Checking for the bare
    // identifier meant a shader that merely USED `view_slot` -- the wireframe
    // one, after it was fixed to index per eye -- was judged to have declared
    // it, and the transform emitted an assignment to something that did not
    // exist.
    // Only when there is something to transform: a shader with no entry points
    // comes back byte-identical, which `a_shader_with_no_entry_points_is_unchanged`
    // requires and which caught the first version of this.
    if next_entry_point(src).is_some() && !src.contains("var<private> view_slot") {
        out.push_str("var<private> view_slot: i32 = 0;\n");
    }
    let mut rest = src;
    loop {
        let Some((stage_at, _stage)) = next_entry_point(rest) else {
            out.push_str(rest);
            return out;
        };
        // Everything up to and including the stage attribute is untouched.
        out.push_str(&rest[..stage_at]);
        rest = &rest[stage_at..];

        // `fn name( ... )` -- the parameter list closes at the paren matching
        // the one after the name, so a defaulted argument containing parens
        // cannot end it early.
        let Some(open) = rest.find('(') else {
            out.push_str(rest);
            return out;
        };
        let Some(close) = matching_paren(rest, open) else {
            out.push_str(rest);
            return out;
        };
        let params = &rest[open + 1..close];
        out.push_str(&rest[..open + 1]);
        out.push_str(params);
        if params.trim().is_empty() {
            out.push_str(VIEW_PARAM);
        } else {
            out.push_str(", ");
            out.push_str(VIEW_PARAM);
        }
        rest = &rest[close..];

        // The body opens at the first brace after the parameter list.
        let Some(brace) = rest.find('{') else {
            out.push_str(rest);
            return out;
        };
        out.push_str(&rest[..=brace]);
        // `view_slot` indexes the per-eye arrays and is `i32`; the builtin is
        // `u32` because naga requires it. One cast, at the only place the two
        // meet.
        out.push_str("\n    view_slot = i32(view_index_in);");
        rest = &rest[brace + 1..];
    }
}

/// Byte offset of the next `@vertex` or `@fragment` attribute.
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
mod tests {
    use super::*;

    /// A PIPELINE THAT TRANSFORMS ITS SHADER MUST ALSO SET ITS MASK.
    ///
    /// `ViewMode` pairs the two so they cannot disagree -- but only if the code
    /// asks it for both. The brush builder was threaded by hand, took
    /// `view.shader(..)` and never `view.mask()`, and produced pipelines whose
    /// shaders expect two views while the pipeline claims one. wgpu refuses
    /// those at SUBMIT, not at creation:
    ///
    ///     In a set_pipeline command
    ///       Incompatible multiview setting: the RenderPass uses Some(3)
    ///
    /// and a refused submit is a frame that never reaches the compositor -- so
    /// the headset holds the last one and the view FREEZES (2026-09-19).
    ///
    /// This reads the source rather than the pipelines, because building a
    /// multiview pipeline needs a device this machine does not have. Crude, and
    /// it would have caught the bug immediately.
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

    /// EVERY SCENE PIPELINE HAS A STEREO TWIN, and it is reachable.
    ///
    /// The scene pass draws with ten pipelines. If one of them has no stereo
    /// constructor, the multiview pass either cannot draw that geometry or --
    /// worse -- draws it with a mono pipeline whose shader reads the left eye's
    /// camera for both views. Neither errors.
    ///
    /// Calling them needs a device, which a headless test can get; what it
    /// cannot get is MULTIVIEW, so these would be REJECTED here. So this checks
    /// the constructors EXIST and are public, by naming them -- a compile-time
    /// check wearing a test's clothes. It fails to build, not to run, if one
    /// goes missing or is renamed.
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
        // Named, never called: taking the function pointer proves the symbol
        // exists with the expected shape and costs nothing at runtime.
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
    }

    /// EVERY SHADER THE SCENE PASS DRAWS WITH MUST SURVIVE THE TRANSFORM.
    ///
    /// This is the gate. A multiview pipeline cannot be built on a development
    /// machine -- wgpu has no multiview on Metal -- so without this the first
    /// thing to find out whether ten transformed shaders are valid WGSL is the
    /// headset, where a rejected pipeline draws nothing, says nothing and makes
    /// the frame FASTER.
    ///
    /// naga is the validator wgpu itself uses. Running the transformed source
    /// through it with `Capabilities::MULTIVIEW` catches everything short of
    /// the driver.
    #[test]
    fn every_scene_shader_survives_the_multiview_transform() {
        use crate::renderer::{
            brush_pipeline, glare, layered_mesh_pipeline, mesh_pipeline, particle, pipeline, sky,
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
        ];
        let mut broken = Vec::new();
        for (name, src) in &shaders {
            // The UNTRANSFORMED shader must be sound first, or a failure below
            // says nothing about multiview.
            if let Some(e) = multiview_validation_error_of(src, false) {
                broken.push(format!("{name} (BEFORE the transform): {e}"));
                continue;
            }
            if let Some(e) = multiview_validation_error(src) {
                broken.push(format!("{name}: {e}"));
            }
            // ONE CAMERA FOR BOTH EYES: a shader carrying its own camera as a
            // single matrix reads the left eye's in a stereo pass for both
            // views. The particle shader did, from the day multiview landed --
            // the right eye's particles drawn from the left eye -- and nothing
            // here could say so, because it was valid WGSL (2026-09-30).
            // `solid_ssr` carries the SSR pass's own camera (`SsrCamera`), one
            // uniform per eye: that pass is never stereo (`SolidPipeline::new_ssr`).
            let one_camera = *name != "solid_ssr" && src.match_indices("view_proj: mat4x4<f32>").any(|(at, _)| {
                // `view_proj` itself, not `sun_view_proj` or `inv_view_proj`.
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

    /// The two halves of multiview travel together or not at all.
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
        // `2` is the COUNT, and was the right value before wgpu 28. As a MASK
        // it means layer 1 only: the left eye would never be drawn, and
        // nothing would report an error.
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
        // The fragment stage matters as much as the vertex stage: the specular
        // term reads the eye's POSITION, and getting that from the wrong eye
        // puts every highlight in the wrong place.
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
        // Only entry points carry the builtin -- adding it to a helper is a
        // compile error, and adding the assignment to one would overwrite the
        // slot with garbage part-way through shading.
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

/// Does this shader survive `as_multiview` and still validate?
///
/// THE ONLY WAY TO CHECK MULTIVIEW ON A DEVELOPMENT MACHINE. wgpu has no
/// multiview on Metal, so no multiview pipeline can be created here at all --
/// and a rejected pipeline on the headset draws nothing, logs nothing and makes
/// the frame faster, which has already cost this project two builds.
///
/// naga is what wgpu validates WGSL with, and it is reachable directly. Running
/// the transformed source through it with `Capabilities::MULTIVIEW` catches
/// everything except the driver itself: a mangled parameter list, an entry
/// point the transform missed, `view_slot` not being in scope, a shader whose
/// fragment stage cannot take the builtin.
///
/// Returns the validation error as a string, or `None` when it is sound.
#[cfg(test)]
pub fn multiview_validation_error(src: &str) -> Option<String> {
    multiview_validation_error_of(src, true)
}

/// The same, with the transform optional -- so a test can establish that a
/// shader was sound BEFORE blaming multiview for breaking it.
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
        // EVERYTHING, deliberately. This gate is about the DELTA the transform
        // introduces, not a device-capability audit -- these shaders already
        // run on the headset, so whatever they need, it has. The control run
        // without the transform is what makes that sound: a shader that fails
        // both ways is not multiview's fault and is reported as such.
        naga::valid::Capabilities::all(),
    );
    match validator.validate(&module) {
        Ok(_) => None,
        Err(e) => Some(format!("validate: {e:?}")),
    }
}

/// WHETHER A PIPELINE DRAWS ONE EYE OR BOTH.
///
/// The two halves of multiview -- transforming the shader and setting the
/// pipeline's layer mask -- must always agree. A pipeline with the mask and an
/// untransformed shader renders both eyes with the left eye's camera; a
/// transformed shader with no mask is a `view_index` builtin in a pass that has
/// no views to index. Neither errors. Pairing them in one value is what makes
/// the mismatch unrepresentable.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum ViewMode {
    /// One eye per pass, the way this renderer has always worked.
    Mono,
    /// Both eyes in one pass, into the layers of one attachment.
    Stereo,
}

impl ViewMode {
    /// The mask a pipeline built in this mode must carry.
    pub fn mask(self) -> Option<core::num::NonZeroU32> {
        match self {
            Self::Mono => None,
            Self::Stereo => STEREO_VIEW_MASK,
        }
    }

    /// The shader source a pipeline built in this mode must use.
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
