//! Turning unbounded light into something a display can show.
//!
//! Shading produces radiance, which has no upper limit -- a lamp two metres
//! away lands around 5, a filament far higher, the sky around 0.3. A display
//! takes 0..1. Until this module existed the renderer simply clamped, and
//! clamping destroys three things at once:
//!
//! 1. **Highlight detail.** A lit wall at 1.0 and a lamp at 12.0 are both
//!    `#FFFFFF`. Every gradient above 1.0 becomes one flat colour, which is
//!    why a spot light's rim was invisible: the rolloff lived in the clipped
//!    range.
//! 2. **Hue.** A warm bulb is (1.0, 0.96, 0.86). Brighten it and red clips
//!    first, then green, then blue -- so it does not fade to white, it lurches
//!    through yellow and then white. Bright areas come out the wrong colour.
//! 3. **The point of HDR.** An HDRI sky carries real intensity precisely so a
//!    scene can hold a 1000:1 range. Clamping throws that away and makes the
//!    format pointless.
//!
//! A tone mapper maps [0, inf) to [0, 1] along a curve with a shoulder, so
//! highlights compress instead of clipping.
//!
//! WHY ACES, AND WHY THIS PARTICULAR FIT
//!
//! ACES is what Unreal, Unity and Godot all reach for by default, so it is the
//! look people expect. More to the point, it is what BABYLON implements, and
//! the editor is Babylon -- the whole value of this project's preview is that
//! it shows what the headset will show. Babylon uses Stephen Hill's `ACESFitted`
//! (an input matrix, a rational RRT/ODT fit, an output matrix) rather than the
//! cheaper Narkowicz curve, so this uses the same one and the same constants.
//!
//! WHY IT IS APPLIED IN THE FORWARD PASS AND NOT AS A POST PROCESS
//!
//! Every desktop engine tone maps in a full-screen pass over an HDR buffer.
//! That needs an Rgba16Float render target and one more pass, and on a tile
//! GPU the pass is the expensive unit -- at Quest 3's per-eye resolution the
//! extra target alone is tens of megabytes before MSAA, and the resolve costs
//! bandwidth in the budget that matters. Godot's mobile renderer tone maps
//! inline in the forward pass for exactly this reason, and so does this.
//!
//! The cost of that choice is honest: no bloom, because bloom needs the HDR
//! buffer this deliberately does not allocate.

use glam::Vec3;
use serde::{Deserialize, Serialize};

/// ACEScg input matrix, ROW-major (each row is a row of the matrix).
///
/// Row-major here and emitted as explicit dot products below, because WGSL's
/// `mat3x3` constructor takes COLUMNS -- transcribing a published row-major
/// matrix straight into it silently transposes the colour transform, which
/// looks like a plausible grade rather than like a bug.
pub const ACES_INPUT: [[f32; 3]; 3] = [
    [0.59719, 0.35458, 0.04823],
    [0.07600, 0.90834, 0.01566],
    [0.02840, 0.13383, 0.83777],
];

/// ACEScg output matrix, ROW-major. See [`ACES_INPUT`].
pub const ACES_OUTPUT: [[f32; 3]; 3] = [
    [1.60475, -0.53108, -0.07367],
    [-0.10208, 1.10813, -0.00605],
    [-0.00327, -0.07276, 1.07602],
];

/// How a scene's radiance is mapped to the display.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ToneMapping {
    /// Filmic shoulder. The default, and what the editor shows.
    #[default]
    Aces,
    /// Hard clamp -- what everything did before this module existed.
    ///
    /// Kept because it is the only way to see raw radiance, which matters when
    /// judging whether a light is genuinely over-bright or merely being rolled
    /// off, and because removing a scene's only escape hatch to a look it was
    /// authored against would be rude.
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
    let b = v * (0.983_729 * v + Vec3::splat(0.432_951_0)) + Vec3::splat(0.238_081);
    a / b
}

/// Stephen Hill's ACES fit, identical to Babylon's `ACESFitted`.
pub fn aces_fitted(color: Vec3) -> Vec3 {
    let c = mul3(&ACES_INPUT, color);
    let c = rrt_and_odt_fit(c);
    mul3(&ACES_OUTPUT, c).clamp(Vec3::ZERO, Vec3::ONE)
}

/// Exposure then tone curve, in that order -- the order Babylon uses.
///
/// Exposure has to come first: it is a camera setting, and scaling AFTER the
/// curve would just brighten an already-compressed image instead of choosing
/// which part of the range to look at.
pub fn tonemap(color: Vec3, exposure: f32, mode: ToneMapping) -> Vec3 {
    let exposed = color * exposure.max(0.0);
    match mode {
        ToneMapping::Aces => aces_fitted(exposed),
        ToneMapping::None => exposed.clamp(Vec3::ZERO, Vec3::ONE),
    }
}

/// The same curve as WGSL, generated from the same constants.
///
/// Emitted rather than hand-written so the shader and [`aces_fitted`] cannot
/// drift: there is one set of numbers in this file and both readers use it.
/// `camera.post_params` carries x = exposure, y = mode (0 = ACES, 1 = none).
pub fn wgsl_tonemap_block() -> String {
    format!(
        "{}{}",
        wgsl_aces_block(),
        r#"
// The last thing every lit fragment does. Output stays LINEAR: the swapchain is
// Rgba8UnormSrgb, so the hardware does the sRGB encode on write, and doing it
// here as well would gamma-correct twice and wash the whole image out.
fn tonemap(color: vec3<f32>) -> vec3<f32> {
    let exposed = color * max(camera.post_params.x, 0.0);
    if (camera.post_params.y > 0.5) {
        return clamp(exposed, vec3<f32>(0.0), vec3<f32>(1.0));
    }
    return aces_fitted(exposed);
}
"#
    )
}

/// [`aces_fitted`] alone, as WGSL, for a shader that tone maps light it has
/// already exposed -- the glare's veil (`glare`), which must meet the display
/// through the same curve as the scene it lies over.
pub fn wgsl_aces_block() -> String {
    let i = ACES_INPUT;
    let o = ACES_OUTPUT;
    format!(
        r#"
fn aces_rrt_odt_fit(v: vec3<f32>) -> vec3<f32> {{
    let a = v * (v + 0.0245786) - vec3<f32>(0.000090537);
    let b = v * (0.983729 * v + vec3<f32>(0.4329510)) + vec3<f32>(0.238081);
    return a / b;
}}

// Dot products rather than a mat3x3: WGSL's constructor takes COLUMNS, and a
// published row-major matrix dropped into it transposes the colour transform
// into something that looks like a deliberate grade instead of a bug.
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
"#,
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

    /// Values from an independent transcription of Babylon's GLSL, computed
    /// outside this crate. Checking the curve against itself would only prove
    /// it is self-consistent; these say it is the SAME curve the editor draws,
    /// which is the entire reason for picking this fit.
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
        // The anchor everyone grades against. If this drifts, every scene in
        // the project is being judged against a different midpoint.
        let g = aces_fitted(Vec3::splat(0.18)).x;
        assert!((g - 0.1056).abs() < 1e-3, "got {g}");
    }

    #[test]
    fn highlights_compress_instead_of_clipping() {
        // The whole point. Under a clamp these are all the same white and every
        // gradient between them is destroyed -- which is why a spot light's rim
        // was invisible, the rolloff living entirely above 1.0.
        let a = aces_fitted(Vec3::splat(2.0)).x;
        let b = aces_fitted(Vec3::splat(8.0)).x;
        let c = aces_fitted(Vec3::splat(40.0)).x;
        assert!(a < b && b < c, "{a} {b} {c} must stay distinguishable");
        assert!(c <= 1.0);
        // A clamp would have flattened all three.
        assert_eq!(
            tonemap(Vec3::splat(2.0), 1.0, ToneMapping::None),
            tonemap(Vec3::splat(40.0), 1.0, ToneMapping::None),
        );
    }

    #[test]
    fn an_over_bright_warm_light_stays_warm() {
        // Clipping does not fade a warm bulb to white, it lurches it through
        // yellow: red pins at 1.0 while green and blue are still climbing.
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
    fn is_monotonic() {
        // A curve that dipped would make a brighter light render darker, which
        // reads as a lighting bug anywhere it shows up.
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
        // After the curve it would only brighten an already-compressed image
        // rather than choosing which part of the range to look at -- and it
        // would be able to push the result past 1.0 again.
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
        // So a scene authored against the previous look has somewhere to stand.
        for v in [0.0_f32, 0.5, 1.0, 9.0] {
            let got = tonemap(Vec3::splat(v), 1.0, ToneMapping::None);
            assert_eq!(got, Vec3::splat(v.clamp(0.0, 1.0)));
        }
    }

    #[test]
    fn the_shader_is_generated_from_these_very_constants() {
        // Not a style check. Two hand-written copies of a colour matrix drift,
        // and a transposed one looks like a deliberate grade rather than a bug.
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
    fn aces_is_the_default() {
        // A contemporary renderer tone maps; opting in would leave every scene
        // clipping by default, which is the thing this module exists to fix.
        assert_eq!(ToneMapping::default(), ToneMapping::Aces);
    }
}

