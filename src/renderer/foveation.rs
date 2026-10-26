use serde::Deserialize;

pub const DENSITY_TEXEL: u32 = 32;

const FULL: u8 = 255;

const HALF: u8 = 127;

const QUARTER: u8 = 63;

pub const SHIPPED: FoveationLevel = FoveationLevel::Low;

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

pub fn centre_of_view(angle_left: f32, angle_right: f32, angle_up: f32, angle_down: f32) -> (f32, f32) {
    let (l, r) = (angle_left.tan(), angle_right.tan());
    let (u, d) = (angle_up.tan(), angle_down.tan());
    (-l / (r - l), u / (u - d))
}

pub fn density_pattern(width: u32, height: u32, centre: (f32, f32), level: FoveationLevel) -> (u32, u32, Vec<u8>) {
    density_pattern_scaled(width, height, centre, level, 1.0)
}

/// The map for an eye drawn into the top-left `scale` of its target
/// (`dynamic_resolution`): the same rings over the drawn rectangle, and the
/// least density outside it, where nothing is shaded.
pub fn density_pattern_scaled(width: u32, height: u32, centre: (f32, f32), level: FoveationLevel, scale: f32) -> (u32, u32, Vec<u8>) {
    let s = scale.clamp(0.05, 1.0);
    let (w, h) = (width.div_ceil(DENSITY_TEXEL).max(1), height.div_ceil(DENSITY_TEXEL).max(1));
    let rings = level.rings();
    let mut texels = Vec::with_capacity((w * h * 2) as usize);
    for j in 0..h {
        for i in 0..w {

            let x = ((i as f32 + 0.5) * DENSITY_TEXEL as f32 / width as f32).min(1.0);
            let y = ((j as f32 + 0.5) * DENSITY_TEXEL as f32 / height as f32).min(1.0);
            let r = ((x - centre.0 * s) / (0.5 * s)).hypot((y - centre.1 * s) / (0.5 * s));
            let outside = s < 1.0 && (x > s || y > s);
            let least = rings[rings.len() - 1].1;
            let density = if outside { least } else { rings.iter().find(|(reach, _)| r < *reach).map_or(least, |(_, d)| *d) };
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
    fn a_scaled_eye_keeps_its_rings_inside_its_rectangle() {
        let (w, h, full) = density_pattern(1024, 1024, (0.5, 0.5), FoveationLevel::High);
        let (_, _, half) = density_pattern_scaled(1024, 1024, (0.5, 0.5), FoveationLevel::High, 0.5);
        let at = |m: &[u8], i: u32, j: u32| m[((j * w + i) * 2) as usize];
        // The middle of the full map is the middle of the half map's rectangle.
        assert_eq!(at(&full, w / 2, h / 2), at(&half, w / 4, h / 4));
        assert_eq!(at(&half, w / 4, h / 4), 255);
        // Outside the drawn rectangle, the least density.
        assert!(at(&half, w - 1, h - 1) < 255);
        assert_eq!(density_pattern_scaled(1024, 1024, (0.5, 0.5), FoveationLevel::High, 1.0).2, full);
    }

    #[test]
    fn off_is_full_density_everywhere() {
        let (_, _, texels) = density_pattern(640, 480, (0.3, 0.6), FoveationLevel::Off);
        assert!(texels.iter().all(|&b| b == 255));
    }

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
            let fragments = |t: &[u8]| t.chunks(2).map(|d| f32::from(d[0]) * f32::from(d[1])).sum::<f32>();
            assert!(fragments(higher) < fragments(lower), "a higher level shaded no less");
            assert!(lower.iter().zip(higher.iter()).all(|(a, b)| b <= a), "a higher level was finer somewhere");
        }
    }
}
