use glam::{Mat3, Mat4, Vec3, Vec4};

use super::MirrorVertex;

pub fn build_mirror_quad(half_width: f32, half_height: f32) -> (Vec<MirrorVertex>, Vec<u32>) {
    let verts = vec![
        MirrorVertex {
            position: [-half_width, -half_height, 0.0],
        },
        MirrorVertex {
            position: [half_width, -half_height, 0.0],
        },
        MirrorVertex {
            position: [half_width, half_height, 0.0],
        },
        MirrorVertex {
            position: [-half_width, half_height, 0.0],
        },
    ];
    let idx = vec![0, 1, 2, 0, 2, 3];
    (verts, idx)
}

pub fn reflection_matrix(plane_point: Vec3, plane_normal: Vec3) -> Mat4 {
    let n = plane_normal.normalize();
    let linear = Mat3::from_cols(
        Vec3::X - 2.0 * n.x * n,
        Vec3::Y - 2.0 * n.y * n,
        Vec3::Z - 2.0 * n.z * n,
    );
    let translation = 2.0 * plane_point.dot(n) * n;
    Mat4::from_cols(
        linear.x_axis.extend(0.0),
        linear.y_axis.extend(0.0),
        linear.z_axis.extend(0.0),
        translation.extend(1.0),
    )
}

pub fn world_plane_equation(plane_point: Vec3, plane_normal: Vec3) -> Vec4 {
    let n = plane_normal.normalize();
    n.extend(-plane_point.dot(n))
}

pub fn plane_to_eye_space(eye_to_world: Mat4, plane_world: Vec4) -> Vec4 {
    eye_to_world.transpose() * plane_world
}

pub fn oblique_near_clip(proj: Mat4, clip_plane: Vec4) -> Mat4 {
    let m = proj.to_cols_array();
    let sign = |v: f32| if v > 0.0 { 1.0 } else if v < 0.0 { -1.0 } else { 0.0 };

    let q = Vec4::new(
        (sign(clip_plane.x) + m[8]) / m[0],
        (sign(clip_plane.y) + m[9]) / m[5],
        -1.0,
        (1.0 + m[10]) / m[14],
    );

    let c = clip_plane * (2.0 / clip_plane.dot(q));

    let mut m2 = m;
    m2[2] = c.x;
    m2[6] = c.y;
    m2[10] = c.z + 1.0;
    m2[14] = c.w;
    Mat4::from_cols_array(&m2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Quat;

    #[test]
    fn reflection_matrix_reflects_a_point_across_the_plane() {
        let m = reflection_matrix(Vec3::new(2.0, 0.0, 0.0), Vec3::X);
        let p = m.transform_point3(Vec3::new(0.0, 1.0, 3.0));
        assert!((p.x - 4.0).abs() < 1e-5, "expected x=4.0, got {p:?}");
        assert!((p.y - 1.0).abs() < 1e-5, "y/z should pass through unchanged, got {p:?}");
        assert!((p.z - 3.0).abs() < 1e-5, "y/z should pass through unchanged, got {p:?}");
    }

    #[test]
    fn reflection_matrix_is_its_own_inverse() {
        let m = reflection_matrix(Vec3::new(1.0, 2.0, 3.0), Vec3::new(0.3, 0.7, 0.1));
        let p = Vec3::new(5.0, -2.0, 0.5);
        let reflected_twice = m.transform_point3(m.transform_point3(p));
        assert!(
            p.distance(reflected_twice) < 1e-4,
            "reflecting twice should return the original point: {p:?} vs {reflected_twice:?}"
        );
    }

    #[test]
    fn reflection_matrix_leaves_plane_points_fixed() {
        let plane_point = Vec3::new(0.0, 1.5, -3.0);
        let m = reflection_matrix(plane_point, Vec3::Y);
        let reflected = m.transform_point3(plane_point);
        assert!(
            plane_point.distance(reflected) < 1e-5,
            "a point on the plane should map to itself, got {reflected:?}"
        );
    }

    fn test_projection(near: f32, far: f32) -> Mat4 {
        let fov_y = 60_f32.to_radians();
        let aspect = 1.0;
        let h_half = (fov_y * 0.5).tan();
        let w_half = h_half * aspect;
        let w = 2.0 * w_half;
        let h = 2.0 * h_half;
        Mat4::from_cols_array(&[
            2.0 / w,
            0.0,
            0.0,
            0.0,
            0.0,
            2.0 / h,
            0.0,
            0.0,
            0.0,
            0.0,
            -(far + near) / (far - near),
            -1.0,
            0.0,
            0.0,
            -(2.0 * far * near) / (far - near),
            0.0,
        ])
    }

    fn ndc_z(proj: Mat4, eye_space_point: Vec3) -> f32 {
        let clip = proj * eye_space_point.extend(1.0);
        clip.z / clip.w
    }

    #[test]
    fn oblique_near_clip_preserves_the_boundary_at_the_original_near_plane() {
        let near = 0.1;
        let proj = test_projection(near, 100.0);

        let p_near = Vec3::new(0.0, 0.0, -near);
        let ndc_before = ndc_z(proj, p_near);

        let clip_plane = Vec4::new(0.0, 0.0, -1.0, -near);
        assert!(clip_plane.w < 0.0, "camera must be on the negative side");

        let clipped = oblique_near_clip(proj, clip_plane);
        let ndc_after = ndc_z(clipped, p_near);

        assert!(
            (ndc_before - ndc_after).abs() < 1e-4,
            "a plane coinciding with the existing near plane shouldn't move \
             its boundary: before={ndc_before}, after={ndc_after}"
        );
    }

    #[test]
    fn oblique_near_clip_moves_the_boundary_to_an_arbitrary_plane() {
        let near = 0.1;
        let proj = test_projection(near, 100.0);
        let boundary_ndc_z = ndc_z(proj, Vec3::new(0.0, 0.0, -near));

        let plane_distance = 5.0;
        let clip_plane = Vec4::new(0.0, 0.0, -1.0, -plane_distance);
        assert!(clip_plane.w < 0.0);

        let clipped = oblique_near_clip(proj, clip_plane);

        let p_on_plane = Vec3::new(1.5, -0.7, -plane_distance);
        let ndc_on_plane = ndc_z(clipped, p_on_plane);
        assert!(
            (ndc_on_plane - boundary_ndc_z).abs() < 1e-3,
            "a point on the new clip plane should map to the near-plane \
             boundary ({boundary_ndc_z}), got {ndc_on_plane}"
        );

        let p_before_plane = Vec3::new(0.0, 0.0, -1.0);
        let ndc_before_plane = ndc_z(clipped, p_before_plane);
        assert!(
            ndc_before_plane < boundary_ndc_z,
            "a point nearer than the new clip plane should fall outside the \
             visible range (boundary={boundary_ndc_z}), got {ndc_before_plane}"
        );
    }

    #[test]
    fn world_plane_equation_matches_reflection_plane_convention() {
        let point = Vec3::new(2.0, 0.0, 0.0);
        let normal = Vec3::X;
        let plane = world_plane_equation(point, normal);
        let value = plane.x * point.x + plane.y * point.y + plane.z * point.z + plane.w;
        assert!(value.abs() < 1e-5, "point on the plane should satisfy its own equation, got {value}");
        let ahead = point + normal * 3.0;
        let ahead_value = plane.x * ahead.x + plane.y * ahead.y + plane.z * ahead.z + plane.w;
        assert!(ahead_value > 0.0, "point on the normal side should be positive, got {ahead_value}");
    }

    #[test]
    fn plane_to_eye_space_matches_direct_point_transform() {
        let rot = Quat::from_rotation_y(35_f32.to_radians()) * Quat::from_rotation_x(10_f32.to_radians());
        let pos = Vec3::new(3.0, 1.5, -2.0);
        let camera_world = Mat4::from_rotation_translation(rot, pos);
        let view = camera_world.inverse();

        let plane_point = Vec3::new(-1.0, 2.0, 0.5);
        let plane_normal = Vec3::new(0.2, 0.9, 0.1);
        let plane_world = world_plane_equation(plane_point, plane_normal);

        let plane_eye = plane_to_eye_space(camera_world, plane_world);

        let p_world = plane_point + Vec3::new(4.0, -3.0, 7.0).reject_from(plane_normal.normalize());
        let p_eye = view.transform_point3(p_world);
        let value = plane_eye.x * p_eye.x + plane_eye.y * p_eye.y + plane_eye.z * p_eye.z + plane_eye.w;
        assert!(
            value.abs() < 1e-3,
            "a point on the world plane should satisfy the eye-space plane equation after \
             transforming by the same view matrix, got {value}"
        );
    }
}
