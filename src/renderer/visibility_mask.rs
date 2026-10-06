//! THE LENSES' HIDDEN AREA (`XR_KHR_visibility_mask`): the part of each eye's
//! image the headset never shows. The runtime gives it as triangles in the
//! view's tangent plane -- x right, y up, on the plane z = -1 -- and drawn into
//! depth before a pass it would let the GPU skip those pixels (plan A2,
//! `docs/frame-budget-plan-2026-10-06.md`). This module measures what that is
//! worth: the share of an eye's pixels the mask covers, and the share of the
//! fragments the scene pass actually shades there, since foveation already
//! shades the edges coarsely.
//!
//! Quest 3's runtime v209.91 lists the extension and gives no mask -- zero
//! triangles hidden and visible, both eyes (deploy101-102, 2026-10-06);
//! presumably its per-eye field of view is fitted to what the lenses show.
//! quest_app logs the measurement at startup (`VISMASK`), so a runtime that
//! starts giving one is seen.

use crate::renderer::foveation::{centre_of_view, density_pattern, FoveationLevel, DENSITY_TEXEL};

/// What a mask covers of one eye's image, 0..1.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MaskShare {
    /// Of the image's pixels.
    pub pixels: f32,
    /// Of the fragments a pass shades at the foveation level: below `pixels`
    /// wherever the mask lies in a coarsely shaded ring.
    pub fragments: f32,
}

/// The share of a `width` x `height` eye image, drawn through `fov`
/// (OpenXR's angles in radians, left, right, up, down; left and down
/// negative), that the triangles `indices` make of `vertices` cover -- the
/// runtime's mask, `[x, y]` on the z = -1 plane. Pixels are counted at their
/// centres, each once however many triangles reach it; fragments weigh each
/// pixel by `level`'s density map, centred where the eye looks straight
/// ahead, as `XrRenderer::apply_foveation` centres it.
pub fn share(vertices: &[[f32; 2]], indices: &[u32], fov: [f32; 4], width: u32, height: u32, level: FoveationLevel) -> MaskShare {
    let (w, h) = (width.max(1), height.max(1));
    let [left, right, up, down] = fov.map(f32::tan);
    // A vertex in pixels, x from the left edge, y down from the top.
    let to_pixel = |[x, y]: [f32; 2]| [(x - left) / (right - left) * w as f32, (up - y) / (up - down) * h as f32];
    let mut covered = vec![false; (w * h) as usize];
    for triangle in indices.chunks_exact(3) {
        let corner = |i: u32| vertices.get(i as usize).copied().map(to_pixel);
        let (Some(a), Some(b), Some(c)) = (corner(triangle[0]), corner(triangle[1]), corner(triangle[2])) else {
            continue;
        };
        let area = edge(a, b, c);
        if area == 0.0 {
            continue;
        }
        let x0 = a[0].min(b[0]).min(c[0]).floor().clamp(0.0, w as f32) as u32;
        let x1 = a[0].max(b[0]).max(c[0]).ceil().clamp(0.0, w as f32) as u32;
        let y0 = a[1].min(b[1]).min(c[1]).floor().clamp(0.0, h as f32) as u32;
        let y1 = a[1].max(b[1]).max(c[1]).ceil().clamp(0.0, h as f32) as u32;
        for y in y0..y1 {
            for x in x0..x1 {
                let p = [x as f32 + 0.5, y as f32 + 0.5];
                let sides = [edge(b, c, p), edge(c, a, p), edge(a, b, p)];
                if sides.iter().all(|&s| s * area >= 0.0) {
                    covered[(y * w + x) as usize] = true;
                }
            }
        }
    }
    let (map_w, _, density) = density_pattern(w, h, centre_of_view(fov[0], fov[1], fov[2], fov[3]), level);
    let (mut pixels, mut fragments, mut every_fragment) = (0u64, 0.0f64, 0.0f64);
    for y in 0..h {
        for x in 0..w {
            let t = (((y / DENSITY_TEXEL) * map_w + x / DENSITY_TEXEL) * 2) as usize;
            let weight = (density[t] as f64 / 255.0) * (density[t + 1] as f64 / 255.0);
            every_fragment += weight;
            if covered[(y * w + x) as usize] {
                pixels += 1;
                fragments += weight;
            }
        }
    }
    MaskShare { pixels: pixels as f32 / (w * h) as f32, fragments: (fragments / every_fragment.max(f64::MIN_POSITIVE)) as f32 }
}

/// Twice the signed area of `a`, `b`, `p`: which side of `a`-`b` `p` is on.
fn edge(a: [f32; 2], b: [f32; 2], p: [f32; 2]) -> f32 {
    (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A symmetric 90-degree view: tangents -1..1 both ways.
    const FOV: [f32; 4] = [-std::f32::consts::FRAC_PI_4, std::f32::consts::FRAC_PI_4, std::f32::consts::FRAC_PI_4, -std::f32::consts::FRAC_PI_4];

    /// Two triangles over the tangent rectangle from `x0` to `x1` across and
    /// `y0` to `y1` up, in either winding.
    fn rect(x0: f32, x1: f32, y0: f32, y1: f32, flip: bool) -> (Vec<[f32; 2]>, Vec<u32>) {
        let vertices = vec![[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
        let indices = if flip { vec![0, 2, 1, 0, 3, 2] } else { vec![0, 1, 2, 0, 2, 3] };
        (vertices, indices)
    }

    #[test]
    fn a_mask_over_the_whole_view_covers_every_pixel_and_fragment() {
        let (v, i) = rect(-1.0, 1.0, -1.0, 1.0, false);
        assert_eq!(share(&v, &i, FOV, 64, 64, FoveationLevel::Low), MaskShare { pixels: 1.0, fragments: 1.0 });
    }

    #[test]
    fn half_the_view_is_half_the_pixels_in_either_winding_and_shared_edges_count_once() {
        for flip in [false, true] {
            let (v, i) = rect(-1.0, 0.0, -1.0, 1.0, flip);
            let s = share(&v, &i, FOV, 64, 64, FoveationLevel::Off);
            assert_eq!(s.pixels, 0.5, "flip {flip}");
            assert_eq!(s.fragments, 0.5, "with every pixel shaded, fragments are pixels");
        }
    }

    #[test]
    fn a_mask_outside_the_view_covers_nothing() {
        let (v, i) = rect(1.5, 3.0, -1.0, 1.0, false);
        assert_eq!(share(&v, &i, FOV, 64, 64, FoveationLevel::Low).pixels, 0.0);
    }

    #[test]
    fn a_mask_in_the_coarse_corners_saves_fewer_fragments_than_pixels() {
        // The four corners, a density texel (32 pixels) each, past the Low
        // level's full-density disc and its half-density ring.
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        for (x0, x1, y0, y1) in [(-1.0, -0.75, -1.0, -0.75), (0.75, 1.0, -1.0, -0.75), (-1.0, -0.75, 0.75, 1.0), (0.75, 1.0, 0.75, 1.0)] {
            let (v, i) = rect(x0, x1, y0, y1, false);
            let base = vertices.len() as u32;
            vertices.extend(v);
            indices.extend(i.into_iter().map(|k| k + base));
        }
        let s = share(&vertices, &indices, FOV, 256, 256, FoveationLevel::Low);
        assert_eq!(s.pixels, 4.0 * 32.0 * 32.0 / (256.0 * 256.0), "{s:?}");
        assert!(s.fragments < s.pixels * 0.6, "the corners shade every other pixel each way: {s:?}");
    }

    #[test]
    fn an_index_past_the_vertices_is_skipped_not_a_panic() {
        let (v, _) = rect(-1.0, 1.0, -1.0, 1.0, false);
        assert_eq!(share(&v, &[0, 1, 9], FOV, 16, 16, FoveationLevel::Off).pixels, 0.0);
    }
}
