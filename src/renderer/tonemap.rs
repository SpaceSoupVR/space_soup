use glam::Vec3;
use serde::{Deserialize, Serialize};

pub const ACES_INPUT: [[f32; 3]; 3] = [
    [0.59719, 0.35458, 0.04823],
    [0.07600, 0.90834, 0.01566],
    [0.02840, 0.13383, 0.83777],
];

pub const ACES_OUTPUT: [[f32; 3]; 3] = [
    [1.60475, -0.53108, -0.07367],
    [-0.10208, 1.10813, -0.00605],
    [-0.00327, -0.07276, 1.07602],
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ToneMapping {

    #[default]
    Aces,

    None,
}

fn mul3(m: &[[f32; 3]; 3], v: Vec3) -> Vec3 {
    Vec3::new(
        m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z,
        m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z,
        m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z,
    )
}

fn rrt_and_odt_fit(v: Vec3) -> Vec3 {
    let a = v * (v + 0.024_578_6) - Vec3::splat(0.000_090_537);
    let b = v * (0.983_729 * v + Vec3::splat(0.432_951)) + Vec3::splat(0.238_081);
    a / b
}

pub fn aces_fitted(color: Vec3) -> Vec3 {
    let c = mul3(&ACES_INPUT, color);
    let c = rrt_and_odt_fit(c);
    mul3(&ACES_OUTPUT, c).clamp(Vec3::ZERO, Vec3::ONE)
}

pub const SCOTOPIC_WEIGHTS: [f32; 3] = [-0.0714, 0.6447, 0.4267];

pub const NIGHT_TINT: [f32; 3] = [1.04, 0.96, 1.26];

/// THE RODS SEE THE NIGHT DIM, not only grey and blue (user, 2026-10-23:
/// the night and its clouds far too bright; "yes for darkening the night like
/// what rod vision sees"). Scotopic vision reads a moonlit scene as dark,
/// desaturated, blue-shifted and low in contrast (Jensen et al. 2000, "A
/// Physically-Based Night Sky Model"; Thompson, Shirley & Ferwerda 2002), and
/// the eye's partial adaptation (`NIGHT_ADAPTATION_STRENGTH`) lifted ours to
/// a dusk. The rods' share of a pixel is drawn at this much of its light:
/// about a stop darker at the deepest night (share `NIGHT_VISION_MAX`),
/// two thirds of one under the moon. Folded into the tint, so it costs the
/// shaders nothing; lever `night_vision` off draws the cones' night.
pub const ROD_SIGNAL: f32 = 0.4;

/// `NIGHT_TINT` at the rods' signal (`ROD_SIGNAL`).
pub fn rod_tint() -> [f32; 3] {
    NIGHT_TINT.map(|t| t * ROD_SIGNAL)
}

pub fn night_vision(color: Vec3, night: f32) -> Vec3 {
    let rods = color.dot(Vec3::from(SCOTOPIC_WEIGHTS)).max(0.0) * Vec3::from(rod_tint());
    color + (rods - color) * night.clamp(0.0, 1.0)
}

pub fn night_vision_for(luminance_cd: f32) -> f32 {
    let (hi, lo) = (3.0f32.log10(), 0.001f32.log10());
    let t = ((hi - luminance_cd.max(1e-9).log10()) / (hi - lo)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t) * NIGHT_VISION_MAX
}

pub const NIGHT_VISION_MAX: f32 = 0.85;

pub fn tonemap(color: Vec3, exposure: f32, mode: ToneMapping) -> Vec3 {
    let exposed = color * exposure.max(0.0);
    match mode {
        ToneMapping::Aces => aces_fitted(exposed),
        ToneMapping::None => exposed.clamp(Vec3::ZERO, Vec3::ONE),
    }
}

pub fn own_light_scale(exposure: f32, drive: f32, bulb_level: f32, adapted: f32) -> f32 {
    let exposed = exposure * drive;
    if bulb_level > 0.0 && exposed > bulb_level && adapted > 0.0 {
        (bulb_level / exposed).powf(adapted.min(1.0))
    } else {
        1.0
    }
}

pub const ADAPTS_FROM_DEGREES: f32 = 1.5;

pub const ADAPTS_NOT_BELOW_DEGREES: f32 = 0.25;

pub fn bulb_adaptation(bulb_radius: f32, distance: f32, in_view: f32) -> f32 {
    if !(distance > 0.0) || !(bulb_radius > 0.0) {
        return 0.0;
    }
    let degrees = (bulb_radius / distance).atan().to_degrees();
    let t = ((degrees / ADAPTS_NOT_BELOW_DEGREES).ln() / (ADAPTS_FROM_DEGREES / ADAPTS_NOT_BELOW_DEGREES).ln()).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t) * in_view.clamp(0.0, 1.0)
}

pub fn wgsl_tonemap_block() -> String {
    format!(
        "{}{}",
        wgsl_aces_block(),
        format!(
            r"
// The room-local adaptation of what this pixel shows (`room_local_exposure`):
// 1 but where a brush or model fragment sets it from its vertices.
var<private> local_exposure: f32 = 1.0;

fn tonemap(color: vec3<f32>) -> vec3<f32> {{
    var exposed = color * (max(camera.post_params.x, 0.0) * local_exposure);

    // A room drawn at its own adaptation is seen by cones: no rods' tint.
    let rods = max(dot(exposed, vec3<f32>({w0:?}, {w1:?}, {w2:?})), 0.0) * vec3<f32>({t0:?}, {t1:?}, {t2:?});
    exposed = mix(exposed, rods, camera.proxy_params.y * smoothstep(0.05, 1.0, local_exposure));
    if (camera.post_params.y > 0.5) {{
        return clamp(exposed, vec3<f32>(0.0), vec3<f32>(1.0));
    }}
    return aces_fitted(exposed);
}}
",
            w0 = SCOTOPIC_WEIGHTS[0], w1 = SCOTOPIC_WEIGHTS[1], w2 = SCOTOPIC_WEIGHTS[2],
            t0 = rod_tint()[0], t1 = rod_tint()[1], t2 = rod_tint()[2],
        )
    )
}

pub fn wgsl_aces_block() -> String {
    let i = ACES_INPUT;
    let o = ACES_OUTPUT;
    format!(
        r"
fn aces_rrt_odt_fit(v: vec3<f32>) -> vec3<f32> {{
    let a = v * (v + 0.0245786) - vec3<f32>(0.000090537);
    let b = v * (0.983729 * v + vec3<f32>(0.4329510)) + vec3<f32>(0.238081);
    return a / b;
}}

fn aces_fitted(color: vec3<f32>) -> vec3<f32> {{
    var c = vec3<f32>(
        dot(color, vec3<f32>({i00}, {i01}, {i02})),
        dot(color, vec3<f32>({i10}, {i11}, {i12})),
        dot(color, vec3<f32>({i20}, {i21}, {i22})),
    );
    c = aces_rrt_odt_fit(c);
    c = vec3<f32>(
        dot(c, vec3<f32>({o00}, {o01}, {o02})),
        dot(c, vec3<f32>({o10}, {o11}, {o12})),
        dot(c, vec3<f32>({o20}, {o21}, {o22})),
    );
    return clamp(c, vec3<f32>(0.0), vec3<f32>(1.0));
}}
",
        i00 = i[0][0], i01 = i[0][1], i02 = i[0][2],
        i10 = i[1][0], i11 = i[1][1], i12 = i[1][2],
        i20 = i[2][0], i21 = i[2][1], i22 = i[2][2],
        o00 = o[0][0], o01 = o[0][1], o02 = o[0][2],
        o10 = o[1][0], o11 = o[1][1], o12 = o[1][2],
        o20 = o[2][0], o21 = o[2][1], o22 = o[2][2],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const REFERENCE: &[([f32; 3], [f32; 3])] = &[
        ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
        ([0.18, 0.18, 0.18], [0.105591, 0.105591, 0.105590]),
        ([1.0, 1.0, 1.0], [0.619115, 0.619115, 0.619109]),
        ([4.0, 4.0, 4.0], [0.909014, 0.909014, 0.909005]),
        ([20.0, 20.0, 20.0], [0.995285, 0.995285, 0.995275]),
        ([1.0, 0.96, 0.86], [0.618086, 0.605797, 0.573001]),
    ];

    #[test]
    fn matches_the_editor_curve() {
        for (input, want) in REFERENCE {
            let got = aces_fitted(Vec3::from_array(*input));
            for c in 0..3 {
                assert!(
                    (got[c] - want[c]).abs() < 1e-4,
                    "at {input:?} channel {c}: got {got:?}, want {want:?}",
                );
            }
        }
    }

    #[test]
    fn eighteen_percent_grey_lands_where_film_puts_it() {

        let g = aces_fitted(Vec3::splat(0.18)).x;
        assert!((g - 0.1056).abs() < 1e-3, "got {g}");
    }

    #[test]
    fn highlights_compress_instead_of_clipping() {

        let a = aces_fitted(Vec3::splat(2.0)).x;
        let b = aces_fitted(Vec3::splat(8.0)).x;
        let c = aces_fitted(Vec3::splat(40.0)).x;
        assert!(a < b && b < c, "{a} {b} {c} must stay distinguishable");
        assert!(c <= 1.0);

        assert_eq!(
            tonemap(Vec3::splat(2.0), 1.0, ToneMapping::None),
            tonemap(Vec3::splat(40.0), 1.0, ToneMapping::None),
        );
    }

    #[test]
    fn an_over_bright_warm_light_stays_warm() {

        let warm = Vec3::new(4.0, 3.84, 3.44);
        let clipped = tonemap(warm, 1.0, ToneMapping::None);
        assert_eq!(clipped, Vec3::ONE, "clipping throws the colour away");

        let mapped = aces_fitted(warm);
        assert!(mapped.x > mapped.y && mapped.y > mapped.z, "{mapped:?}");
        assert!(mapped.x - mapped.z > 0.01, "warmth survived: {mapped:?}");
    }

    #[test]
    fn never_leaves_the_displayable_range() {
        for v in [-5.0_f32, 0.0, 0.5, 1.0, 1e3, 1e6] {
            let c = aces_fitted(Vec3::splat(v));
            for ch in 0..3 {
                assert!((0.0..=1.0).contains(&c[ch]), "{v} -> {c:?}");
                assert!(c[ch].is_finite(), "{v} -> {c:?}");
            }
        }
    }

    #[test]
    fn the_curve_is_monotonic() {

        let mut last = -1.0;
        let mut x = 0.0_f32;
        while x < 30.0 {
            let v = aces_fitted(Vec3::splat(x)).x;
            assert!(v >= last - 1e-6, "dipped at {x}: {v} after {last}");
            last = v;
            x += 0.05;
        }
    }

    #[test]
    fn exposure_is_applied_before_the_curve() {

        let c = Vec3::splat(1.0);
        assert_eq!(tonemap(c, 2.0, ToneMapping::Aces), aces_fitted(c * 2.0));
        assert!(tonemap(c, 2.0, ToneMapping::Aces).x <= 1.0);
    }

    #[test]
    fn a_negative_exposure_cannot_invert_the_image() {
        let c = tonemap(Vec3::splat(1.0), -3.0, ToneMapping::Aces);
        for ch in 0..3 {
            assert!((0.0..=1.0).contains(&c[ch]), "{c:?}");
        }
    }

    #[test]
    fn none_is_exactly_the_old_clamp() {

        for v in [0.0_f32, 0.5, 1.0, 9.0] {
            let got = tonemap(Vec3::splat(v), 1.0, ToneMapping::None);
            assert_eq!(got, Vec3::splat(v.clamp(0.0, 1.0)));
        }
    }

    #[test]
    fn the_shader_is_generated_from_these_very_constants() {

        let wgsl = wgsl_tonemap_block();
        for row in ACES_INPUT.iter().chain(ACES_OUTPUT.iter()) {
            for v in row {
                assert!(
                    wgsl.contains(&format!("{v}")),
                    "constant {v} never reached the shader",
                );
            }
        }
        assert!(wgsl.contains("camera.post_params.x"), "exposure not wired");
    }

    #[test]
    fn a_fixtures_own_light_keeps_its_ratios_with_the_bulb_at_its_level() {
        let (exposure, drive, level) = (5.33, 3600.0, 16.0);
        let k = own_light_scale(exposure, drive, level, 1.0);
        assert!((exposure * drive * k - level).abs() < 1e-3, "{k}");
        let shown = |share: f32| aces_fitted(Vec3::splat(exposure * drive * share * k)).x;

        let [bulb, near, behind, rim] = [1.0, 0.25, 0.111, 0.032].map(shown);
        assert!(bulb > 0.98, "the bulb is white: {bulb}");
        assert!(bulb - near > 0.05 && near > behind && behind > rim, "{bulb} {near} {behind} {rim}");
        assert!(rim < 0.5, "the rim is shaded: {rim}");

        let raw = |share: f32| aces_fitted(Vec3::splat(exposure * drive * share)).x;
        assert!(raw(0.032) > 0.99, "{}", raw(0.032));
    }

    #[test]
    fn a_dim_bulb_or_no_level_leaves_a_fixtures_light_alone() {
        assert_eq!(own_light_scale(1.0, 4.0, 16.0, 1.0), 1.0);
        assert_eq!(own_light_scale(5.33, 3600.0, 0.0, 1.0), 1.0);
        assert_eq!(own_light_scale(5.33, 3600.0, 16.0, 0.0), 1.0);
        assert_eq!(own_light_scale(f32::NAN, 3600.0, 16.0, 1.0), 1.0);
        assert_eq!(own_light_scale(5.33, 0.0, 16.0, 1.0), 1.0);

        let half = own_light_scale(5.33, 3600.0, 16.0, 0.5);
        assert!((half - own_light_scale(5.33, 3600.0, 16.0, 1.0).sqrt()).abs() < 1e-6, "{half}");
    }

    #[test]
    fn the_eye_adapts_to_a_bulb_it_sees_and_can_look_at() {
        assert_eq!(bulb_adaptation(0.05, 1.0, 1.0), 1.0);
        assert_eq!(bulb_adaptation(0.05, 15.0, 1.0), 0.0);
        assert_eq!(bulb_adaptation(0.05, 1.0, 0.0), 0.0);
        let at_five = bulb_adaptation(0.05, 5.0, 1.0);
        assert!(at_five > 0.2 && at_five < 0.8, "{at_five}");
        assert!((bulb_adaptation(0.05, 1.0, 0.3) - 0.3).abs() < 1e-6);
        assert!(bulb_adaptation(0.05, 2.0, 1.0) > bulb_adaptation(0.05, 3.0, 1.0));
        assert_eq!(bulb_adaptation(0.05, 0.0, 1.0), 0.0);
        assert_eq!(bulb_adaptation(0.05, f32::NAN, 1.0), 0.0);
    }

    #[test]
    fn night_vision_is_nothing_by_day_keeps_white_and_greys_colour_at_night() {
        let c = Vec3::new(0.8, 0.3, 0.1);
        assert_eq!(night_vision(c, 0.0), c);
        let white = night_vision(Vec3::ONE, 1.0);
        let lum = |v: Vec3| v.dot(Vec3::new(0.2126, 0.7152, 0.0722));
        assert!((lum(white) - ROD_SIGNAL).abs() < 0.02, "the rods see white at their signal: {white}");
        assert!(white.z > white.x * 1.15, "and bluer: {white}");

        let red = lum(night_vision(Vec3::new(1.0, 0.0, 0.0), 1.0));
        let blue = lum(night_vision(Vec3::new(0.0, 0.0, 1.0), 1.0));
        assert!(red < 0.01 && blue > 0.3 * ROD_SIGNAL, "{red} {blue}");
        assert_eq!(night_vision_for(100.0), 0.0);
        assert!((night_vision_for(1e-4) - NIGHT_VISION_MAX).abs() < 1e-6);
        let moonlit = night_vision_for(0.03);
        assert!(moonlit > 0.4 && moonlit < NIGHT_VISION_MAX, "{moonlit}");
        let wgsl = wgsl_tonemap_block();
        for v in SCOTOPIC_WEIGHTS.iter().chain(rod_tint().iter()) {
            assert!(wgsl.contains(&format!("{v:?}")), "{v} never reached the shader");
        }
        assert!(wgsl.contains("camera.proxy_params.y"));
    }

    #[test]
    fn aces_is_the_default() {

        assert_eq!(ToneMapping::default(), ToneMapping::Aces);
    }
}
