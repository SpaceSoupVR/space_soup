use glam::{Vec3, Vec4};

pub const MARGIN: f32 = 0.1;

pub fn sees(planes: &[[Vec4; 6]], bound: Vec4) -> bool {
    let bound = bound + Vec4::new(0.0, 0.0, 0.0, MARGIN);
    planes.iter().any(|p| crate::renderer::shadow::sphere_in_frustum(p, bound))
}

pub fn sees_box(planes: &[[Vec4; 6]], (lo, hi): (Vec3, Vec3)) -> bool {
    let m = Vec3::splat(MARGIN);
    planes.iter().any(|p| crate::renderer::shadow::aabb_in_frustum(p, lo - m, hi + m))
}

pub fn distance_squared_to_box(p: Vec3, lo: Vec3, hi: Vec3) -> f32 {
    (p.clamp(lo, hi) - p).length_squared()
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Mat4;

    fn eye_at(eye: Vec3) -> [Vec4; 6] {
        let proj = Mat4::perspective_rh(std::f32::consts::FRAC_PI_2, 1.0, 0.05, 500.0);
        crate::renderer::shadow::frustum_planes(proj * Mat4::look_to_rh(eye, Vec3::NEG_Z, Vec3::Y))
    }

    #[test]
    fn a_box_is_as_far_as_its_nearest_point() {
        let (lo, hi) = (Vec3::new(1.0, 0.0, -1.0), Vec3::new(3.0, 2.0, 1.0));
        assert_eq!(distance_squared_to_box(Vec3::new(2.0, 1.0, 0.0), lo, hi), 0.0);
        assert_eq!(distance_squared_to_box(Vec3::new(0.0, 1.0, 0.0), lo, hi), 1.0);
        assert_eq!(distance_squared_to_box(Vec3::new(5.0, 4.0, 0.0), lo, hi), 8.0);

        assert!(distance_squared_to_box(Vec3::ZERO, Vec3::splat(-1.0), Vec3::ONE) < distance_squared_to_box(Vec3::ZERO, lo, hi));
    }

    #[test]
    fn a_model_ahead_is_drawn_and_one_behind_or_far_aside_is_not() {
        let eyes = [eye_at(Vec3::new(-0.03, 1.6, 0.0)), eye_at(Vec3::new(0.03, 1.6, 0.0))];
        assert!(sees(&eyes, Vec4::new(0.0, 1.6, -4.0, 0.5)));
        assert!(!sees(&eyes, Vec4::new(0.0, 1.6, 4.0, 0.5)));
        assert!(!sees(&eyes, Vec4::new(20.0, 1.6, -4.0, 0.5)));
        assert!(!sees(&eyes, Vec4::new(0.0, 30.0, -4.0, 0.5)));

        assert!(!sees(&eyes, Vec4::new(0.0, 1.6, -600.0, 0.5)));
    }

    #[test]
    fn a_model_touching_the_edge_is_drawn() {
        let eyes = [eye_at(Vec3::new(0.0, 1.6, 0.0))];
        assert!(sees(&eyes, Vec4::new(4.6, 1.6, -4.0, 0.5)));
        assert!(!sees(&eyes, Vec4::new(5.0, 1.6, -4.0, 0.5)));

        assert!(sees(&eyes, Vec4::new(4.75, 1.6, -4.0, 0.5)));
    }

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

        assert!(sees_box(&[eye_at(Vec3::new(0.0, 1.6, 0.0))], cave));
        assert!(sees_box(&[eye_at(Vec3::new(0.0, 1.6, -20.0))], cave));

        assert!(!sees_box(&[eye_at(Vec3::new(0.0, 1.6, -40.0))], cave));
    }
}
