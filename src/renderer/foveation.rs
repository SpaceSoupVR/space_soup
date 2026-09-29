//! FIXED FOVEATED RENDERING: shade the edges of each eye's image more coarsely,
//! where the headset's lenses blur it anyway.
//!
//! The GPU does it (`VK_EXT_fragment_density_map`, through the SpaceSoupVR
//! wgpu fork: `wgpu_hal::vulkan::Device::enable_foveation`): a small map over
//! the eye image says, per 32-pixel block, how many pixels one fragment may
//! cover -- one, two side by side, or a 2x2 square. Nothing in a shader
//! changes; the scene pass simply runs fewer fragments at the edges, and the
//! driver writes the full-resolution image at the end of the pass.
//!
//! Each eye's map is centred where that eye looks straight ahead, which is not
//! the middle of its image: a headset's per-eye field of view is asymmetric
//! (wider toward the temple). See [`centre_of_view`].

use serde::Deserialize;

/// Pixels a density-map texel covers, a side. The GPU picks the texel size as
/// the power of two nearest the map-to-image ratio, so a map of
/// `ceil(extent / 32)` texels gets exactly this.
pub const DENSITY_TEXEL: u32 = 32;

/// A density-map byte for every pixel shaded.
const FULL: u8 = 255;

/// Every other pixel: 127/255 asks for a fragment 2.008 pixels wide. 128
/// would ask for 1.992, which a driver keeping at least the requested density
/// rounds DOWN to one -- a map of 128s changes nothing.
const HALF: u8 = 127;

/// One pixel in four: 63/255 is 4.05 pixels.
const QUARTER: u8 = 63;

/// The level the renderer ships at; the lever `foveation` overrides it.
///
/// LOW: on the headset (synced GPU, `docs/bench/2026-09-29_0200_ffr_*`) it
/// saved as much as Medium -- outdoors_front 15.37 -> 13.16 ms,
/// outdoors_walls 15.22 -> 13.26, the brick hall 13.03 -> 11.73, hall_back
/// 11.45 -> 10.59 -- while keeping every pixel out to three quarters of the
/// way to each edge. High coarsened the ground a third of the way down from
/// the centre of view, visibly.
pub const SHIPPED: FoveationLevel = FoveationLevel::Low;

/// How much of each eye's image is shaded coarsely. `Off` shades every pixel.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FoveationLevel {
    #[default]
    Off,
    Low,
    Medium,
    High,
}

impl FoveationLevel {
    pub fn label(self) -> &'static str {
        match self {
            FoveationLevel::Off => "off",
            FoveationLevel::Low => "low",
            FoveationLevel::Medium => "medium",
            FoveationLevel::High => "high",
        }
    }

    /// The rings, innermost first: out to this radius, this density across
    /// and down ([`FULL`] every pixel, [`HALF`] every other, [`QUARTER`] one
    /// in four). The radius is measured from the eye's centre of view in
    /// half-image units, so 1 is the middle of an edge; past the last ring,
    /// the last density holds.
    fn rings(self) -> &'static [(f32, [u8; 2])] {
        match self {
            FoveationLevel::Off => &[(f32::MAX, [FULL, FULL])],
            FoveationLevel::Low => &[(0.75, [FULL, FULL]), (1.05, [HALF, FULL]), (f32::MAX, [HALF, HALF])],
            FoveationLevel::Medium => &[(0.6, [FULL, FULL]), (0.85, [HALF, FULL]), (f32::MAX, [HALF, HALF])],
            FoveationLevel::High => {
                &[(0.45, [FULL, FULL]), (0.7, [HALF, FULL]), (0.95, [HALF, HALF]), (f32::MAX, [QUARTER, HALF])]
            }
        }
    }
}

/// Where an eye looks straight ahead in its own image, 0..1 across from the
/// left and down from the top, from its field of view (OpenXR's angles, in
/// radians: left and down negative).
pub fn centre_of_view(angle_left: f32, angle_right: f32, angle_up: f32, angle_down: f32) -> (f32, f32) {
    let (l, r) = (angle_left.tan(), angle_right.tan());
    let (u, d) = (angle_up.tan(), angle_down.tan());
    (-l / (r - l), u / (u - d))
}

/// A density map for an image of `width` x `height` pixels at `level`,
/// centred on `centre` (see [`centre_of_view`]): `ceil(width / 32)` x
/// `ceil(height / 32)` texels, two bytes each, row-major.
pub fn density_pattern(width: u32, height: u32, centre: (f32, f32), level: FoveationLevel) -> (u32, u32, Vec<u8>) {
    let (w, h) = (width.div_ceil(DENSITY_TEXEL).max(1), height.div_ceil(DENSITY_TEXEL).max(1));
    let rings = level.rings();
    let mut texels = Vec::with_capacity((w * h * 2) as usize);
    for j in 0..h {
        for i in 0..w {
            // The texel's middle, in 0..1 of the image.
            let x = ((i as f32 + 0.5) * DENSITY_TEXEL as f32 / width as f32).min(1.0);
            let y = ((j as f32 + 0.5) * DENSITY_TEXEL as f32 / height as f32).min(1.0);
            let r = ((x - centre.0) / 0.5).hypot((y - centre.1) / 0.5);
            let density = rings.iter().find(|(reach, _)| r < *reach).map_or(rings[rings.len() - 1].1, |(_, d)| *d);
            texels.extend_from_slice(&density);
        }
    }
    (w, h, texels)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_symmetric_view_looks_through_the_middle() {
        let a = 0.8f32;
        let (x, y) = centre_of_view(-a, a, a, -a);
        assert!((x - 0.5).abs() < 1e-6 && (y - 0.5).abs() < 1e-6);
    }

    /// Wider toward the temple: the left eye sees further left than right, so
    /// straight ahead lies right of its image's middle.
    #[test]
    fn a_wider_left_puts_the_centre_right_of_the_middle() {
        let (x, _) = centre_of_view(-0.9, 0.7, 0.8, -0.8);
        assert!(x > 0.5, "{x}");
        let (_, y) = centre_of_view(-0.8, 0.8, 0.9, -0.7);
        assert!(y > 0.5, "wider up puts straight ahead below the middle: {y}");
    }

    #[test]
    fn the_map_covers_the_image_at_one_texel_per_32_pixels() {
        let (w, h, texels) = density_pattern(1176, 1232, (0.5, 0.5), FoveationLevel::Medium);
        assert_eq!((w, h), (37, 39));
        assert_eq!(texels.len(), 37 * 39 * 2);
    }

    #[test]
    fn off_is_full_density_everywhere() {
        let (_, _, texels) = density_pattern(640, 480, (0.3, 0.6), FoveationLevel::Off);
        assert!(texels.iter().all(|&b| b == 255));
    }

    /// The centre of view keeps every pixel at every level; the corners are
    /// coarse at every level but off; and each level is at least as coarse as
    /// the one below it, texel for texel.
    #[test]
    fn higher_levels_shade_less_and_the_centre_is_always_full() {
        let centre = (0.55, 0.48);
        let texel_at = |t: &[u8], w: u32, h: u32, x: f32, y: f32| {
            let (i, j) = (((x * w as f32) as u32).min(w - 1), ((y * h as f32) as u32).min(h - 1));
            let k = ((j * w + i) * 2) as usize;
            [t[k], t[k + 1]]
        };
        let levels = [FoveationLevel::Off, FoveationLevel::Low, FoveationLevel::Medium, FoveationLevel::High];
        let maps: Vec<_> = levels.iter().map(|&l| density_pattern(1176, 1232, centre, l)).collect();
        for (l, (w, h, t)) in levels.iter().zip(&maps) {
            assert_eq!(texel_at(t, *w, *h, centre.0, centre.1), [255, 255], "{l:?} coarsened the centre of view");
            if *l != FoveationLevel::Off {
                assert_ne!(texel_at(t, *w, *h, 0.01, 0.01), [255, 255], "{l:?} left a corner at full density");
            }
        }
        for pair in maps.windows(2) {
            let (lower, higher) = (&pair[0].2, &pair[1].2);
            let fragments = |t: &[u8]| t.chunks(2).map(|d| d[0] as f32 * d[1] as f32).sum::<f32>();
            assert!(fragments(higher) < fragments(lower), "a higher level shaded no less");
            assert!(lower.iter().zip(higher.iter()).all(|(a, b)| b <= a), "a higher level was finer somewhere");
        }
    }
}
