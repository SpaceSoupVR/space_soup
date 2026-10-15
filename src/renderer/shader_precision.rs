pub const HALF_PRECISION: bool = false;

pub const F32_ALIASES: &str = "alias hf = f32;\nalias hf2 = vec2<f32>;\nalias hf3 = vec3<f32>;\nalias hf4 = vec4<f32>;\n";

const F16_ALIASES: &str = "alias hf = f16;\nalias hf2 = vec2<f16>;\nalias hf3 = vec3<f16>;\nalias hf4 = vec4<f16>;\n";

pub fn for_device(device: &wgpu::Device, src: String) -> String {
    with_half_precision(src, HALF_PRECISION && device.features().contains(wgpu::Features::SHADER_F16))
}

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

        assert_eq!(with_half_precision("fn g() {}".into(), true), "fn g() {}");
    }

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
