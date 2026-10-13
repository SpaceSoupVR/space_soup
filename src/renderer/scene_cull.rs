//! Which models the scene pass draws for an eye (`Levers::scene_cull`).
//!
//! The scene pass drew every model in the level for both eyes wherever it
//! stood -- behind the player, in another room, out of doors -- and the GPU
//! clipped it after shading its vertices: a per-draw capture of the hall
//! (docs/shader-optimization-2026-10-12.md, 2026-10-12) had 33 model draws an
//! eye, 116,000 indices, for 4,600 pixels between them, and from the field
//! the same 33 for none at all. The motion pass has skipped what an eye's
//! frustum cannot hold since round 3 (`space_warp`); this is the same test for
//! the pass that shades. Lossless: only what the GPU would clip whole.
//!
//! Its own module, because the XR renderer compiles for the headset alone and
//! this is arithmetic a development machine can test.

use glam::{Vec3, Vec4};

/// How far past its bound a model is still drawn: its thin parts are widened
/// to a pixel on screen (`mesh::thin_parts`), which at any distance a wire is
/// seen from is millimetres.
pub const MARGIN: f32 = 0.1;

/// Whether any of the pass's eyes can see into a model's bounding sphere
/// (`shadow::mesh_caster_bound`), `planes` being each eye's frustum
/// (`shadow::frustum_planes`) in the frame the bound is in.
pub fn sees(planes: &[[Vec4; 6]], bound: Vec4) -> bool {
    let bound = bound + Vec4::new(0.0, 0.0, 0.0, MARGIN);
    planes.iter().any(|p| crate::renderer::shadow::sphere_in_frustum(p, bound))
}

/// The same for a box (a cave's, `cave_twins::placed_box`).
pub fn sees_box(planes: &[[Vec4; 6]], (lo, hi): (Vec3, Vec3)) -> bool {
    let m = Vec3::splat(MARGIN);
    planes.iter().any(|p| crate::renderer::shadow::aabb_in_frustum(p, lo - m, hi + m))
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Mat4;

    /// An eye at `eye` looking down -z with a 90 degree view.
    fn eye_at(eye: Vec3) -> [Vec4; 6] {
        let proj = Mat4::perspective_rh(std::f32::consts::FRAC_PI_2, 1.0, 0.05, 500.0);
        crate::renderer::shadow::frustum_planes(proj * Mat4::look_to_rh(eye, Vec3::NEG_Z, Vec3::Y))
    }

    #[test]
    fn a_model_ahead_is_drawn_and_one_behind_or_far_aside_is_not() {
        let eyes = [eye_at(Vec3::new(-0.03, 1.6, 0.0)), eye_at(Vec3::new(0.03, 1.6, 0.0))];
        assert!(sees(&eyes, Vec4::new(0.0, 1.6, -4.0, 0.5)));
        assert!(!sees(&eyes, Vec4::new(0.0, 1.6, 4.0, 0.5)));
        assert!(!sees(&eyes, Vec4::new(20.0, 1.6, -4.0, 0.5)));
        assert!(!sees(&eyes, Vec4::new(0.0, 30.0, -4.0, 0.5)));
        // Past the far plane.
        assert!(!sees(&eyes, Vec4::new(0.0, 1.6, -600.0, 0.5)));
    }

    /// A model whose sphere only touches the view's edge is drawn: at 4 m
    /// ahead the 90 degree view's edge is 4 m aside, and a sphere centred
    /// 4.6 m aside reaches it with a radius of 0.5 only through the margin's
    /// side of the diagonal (0.6 * cos 45 = 0.42 < 0.5).
    #[test]
    fn a_model_touching_the_edge_is_drawn() {
        let eyes = [eye_at(Vec3::new(0.0, 1.6, 0.0))];
        assert!(sees(&eyes, Vec4::new(4.6, 1.6, -4.0, 0.5)));
        assert!(!sees(&eyes, Vec4::new(5.0, 1.6, -4.0, 0.5)));
        // Within the margin alone: 0.75 * cos 45 = 0.53, under 0.5 + 0.1.
        assert!(sees(&eyes, Vec4::new(4.75, 1.6, -4.0, 0.5)));
    }

    /// Drawn when either eye sees it: a pass of both eyes must not lose what
    /// one of them holds.
    #[test]
    fn one_eye_seeing_it_is_enough() {
        let left = eye_at(Vec3::new(-3.0, 1.6, 0.0));
        let right = eye_at(Vec3::new(3.0, 1.6, 0.0));
        let model = Vec4::new(4.2, 1.6, -1.0, 0.1);
        assert!(!sees(&[left], model));
        assert!(sees(&[right], model));
        assert!(sees(&[left, right], model));
        assert!(!sees(&[], model));
    }

    #[test]
    fn a_cave_is_drawn_from_inside_and_towards_and_not_with_its_back_to_it() {
        let cave = (Vec3::new(-5.0, 0.0, -30.0), Vec3::new(5.0, 6.0, -10.0));
        // Looking at it from outside, and standing in it.
        assert!(sees_box(&[eye_at(Vec3::new(0.0, 1.6, 0.0))], cave));
        assert!(sees_box(&[eye_at(Vec3::new(0.0, 1.6, -20.0))], cave));
        // Past it, looking on: it is behind.
        assert!(!sees_box(&[eye_at(Vec3::new(0.0, 1.6, -40.0))], cave));
    }
}
