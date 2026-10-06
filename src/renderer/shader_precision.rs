//! HALF-PRECISION ARITHMETIC in the audited hot shaders, where the device has it.
//!
//! Adreno runs `f16` arithmetic at twice the rate of `f32` and packs two
//! halves into one register, and register pressure is occupancy. What suits
//! it is colour and factor maths: light falloff, cone and diffuse terms,
//! mixing, tonemapping. What does not: positions and distances (world-space
//! precision), depth, and high-exponent `pow`s (a Blinn-Phong exponent of 320
//! on `dot(n, h)` quantised to half precision bands a marble highlight).
//!
//! THE MECHANISM, safe by default. The lights block declares
//!
//! ```wgsl
//! alias hf = f32; alias hf2 = vec2<f32>; alias hf3 = vec3<f32>; alias hf4 = vec4<f32>;
//! ```
//!
//! and maths written in those types is exactly `f32` maths -- the same
//! instructions, the same pixels -- everywhere, unless [`for_device`] rewrites
//! the aliases to `f16` and prepends `enable f16;`. It does that only for the
//! modules passed through it (the audited brush and terrain shaders), only when
//! the device has `Features::SHADER_F16` (on the Quest: `shaderFloat16`, see
//! `VkContext::shader_f16`), and only while [`HALF_PRECISION`] is on. Uniform
//! and storage buffers stay `f32` whatever happens: Adreno has no 16-bit
//! uniform access, which is why the SpaceSoupVR naga declares 16-bit storage
//! capabilities only for buffers that hold a 16-bit type.
//!
//! Every conversion to `hf` changes pixels only on an `f16` device, so each
//! is checked with `quest_app`'s offline renders at `f16` (Metal has it too)
//! against the `f32` ones, and on the headset.

/// The switch. Off: every shader is `f32`, as before.
///
/// OFF SINCE B1 (2026-10-06). With the lamp loops' colours at `hf`, `f16` put
/// the full scene reader at 19 registers against `f32`'s 21 (62% occupancy
/// against 50%) and bought nothing: one build, the whole build at each
/// precision (`debug.spacesoup.nof16`), two passes, seven views -- the same
/// within 0.1 ms everywhere but torch_doorway_out, where `f16` was 0.45 ms
/// SLOWER in both passes. The loops written in `hf` cost nothing at `f32`;
/// the restructure that came with them (each lamp weighted before its
/// shadow, the lean tent) is what put the baked and ground readers at 19,
/// at either precision.
pub const HALF_PRECISION: bool = false;

/// What the lights block declares: the half-precision aliases, at `f32`.
pub const F32_ALIASES: &str = "alias hf = f32;\nalias hf2 = vec2<f32>;\nalias hf3 = vec3<f32>;\nalias hf4 = vec4<f32>;\n";

const F16_ALIASES: &str = "alias hf = f16;\nalias hf2 = vec2<f16>;\nalias hf3 = vec3<f16>;\nalias hf4 = vec4<f16>;\n";

/// `src` as `device` should compile it: at `f16` -- the aliases rewritten and
/// `enable f16;` first -- when the device has `SHADER_F16`, [`HALF_PRECISION`]
/// is on and the source declares the aliases; otherwise unchanged. Apply it to
/// the FINISHED source (after any multiview transform): `enable` must be the
/// module's first line.
pub fn for_device(device: &wgpu::Device, src: String) -> String {
    with_half_precision(src, HALF_PRECISION && device.features().contains(wgpu::Features::SHADER_F16))
}

/// [`for_device`] with the decision made: `src` at `f16` when `f16` is set.
pub fn with_half_precision(src: String, f16: bool) -> String {
    if f16 && src.contains(F32_ALIASES) {
        format!("enable f16;\n{}", src.replacen(F32_ALIASES, F16_ALIASES, 1))
    } else {
        src
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32_is_the_default_and_f16_rewrites_only_the_aliases() {
        let src = format!("// head\n{F32_ALIASES}fn f(x: hf) -> hf {{ return x; }}\n");
        assert_eq!(with_half_precision(src.clone(), false), src);
        let h = with_half_precision(src.clone(), true);
        assert!(h.starts_with("enable f16;\n"), "`enable` must come first: {h}");
        assert!(h.contains("alias hf = f16;") && !h.contains("alias hf = f32;"));
        // A module without the aliases is left alone either way.
        assert_eq!(with_half_precision("fn g() {}".into(), true), "fn g() {}");
    }

    /// Both forms of the lights block parse: at f32, and rewritten to f16.
    #[test]
    fn the_lights_block_compiles_at_both_precisions() {
        use wgpu::naga;
        let block = crate::renderer::lights::wgsl_lights_block(0, 1);
        assert!(block.contains(F32_ALIASES), "the lights block declares the aliases");
        for f16 in [false, true] {
            let src = with_half_precision(block.clone(), f16);
            let module = naga::front::wgsl::parse_str(&src)
                .unwrap_or_else(|e| panic!("f16={f16}: {}", e.emit_to_string(&src)));
            naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
                .validate(&module)
                .unwrap_or_else(|e| panic!("f16={f16}: {e:?}"));
        }
    }
}
