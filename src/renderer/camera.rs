use glam::{Mat4, Quat, Vec3};

pub struct Camera {
    pub position: Vec3,
    pub rotation: Quat,
    pub fov_y: f32,
    pub aspect: f32,
    pub near: f32,
    pub far: f32,
}

impl Camera {
    pub fn new(aspect: f32) -> Self {
        Self {
            position: Vec3::new(0.0, 1.6, 5.0),
            rotation: Quat::IDENTITY,
            fov_y: 60_f32.to_radians(),
            aspect,
            near: 0.01,
            far: 1000.0,
        }
    }

    pub fn view(&self) -> Mat4 {
        Mat4::from_rotation_translation(self.rotation, self.position).inverse()
    }

    /// Where the camera is looking, in world space.
    ///
    /// -Z is forward, matching the right-handed convention `view` and
    /// `perspective_rh` already use. Taking +Z here would aim the sun's shadow
    /// box behind the viewer, which shows up as shadows that only appear when
    /// you turn around.
    pub fn forward(&self) -> Vec3 {
        self.rotation * Vec3::NEG_Z
    }

    pub fn projection(&self) -> Mat4 {
        Mat4::perspective_rh(self.fov_y, self.aspect, self.near, self.far)
    }

    #[cfg(target_os = "android")]
    pub fn xr_view(pose: openxr::Posef) -> Mat4 {
        let pos = Vec3::new(pose.position.x, pose.position.y, pose.position.z);
        let rot = Quat::from_xyzw(
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        );
        Mat4::from_rotation_translation(rot, pos).inverse()
    }

    #[cfg(target_os = "android")]
    pub fn xr_projection(fov: openxr::Fovf, near: f32, far: f32) -> Mat4 {
        let l = fov.angle_left.tan();
        let r = fov.angle_right.tan();
        let u = fov.angle_up.tan();
        let d = fov.angle_down.tan();
        let w = r - l;
        let h = u - d;
        Mat4::from_cols_array(&[
            2.0 / w,
            0.0,
            0.0,
            0.0,
            0.0,
            2.0 / h,
            0.0,
            0.0,
            (r + l) / w,
            (u + d) / h,
            -(far + near) / (far - near),
            -1.0,
            0.0,
            0.0,
            -(2.0 * far * near) / (far - near),
            0.0,
        ])
    }

    pub fn gl_to_wgpu_ndc(proj: Mat4) -> Mat4 {
        let remap = Mat4::from_cols_array(&[
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 0.5, 0.0,
            0.0, 0.0, 0.5, 1.0,
        ]);
        remap * proj
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gl_to_wgpu_ndc_maps_gl_near_and_far_bounds_to_wgpu_range() {
        let gl_proj = Mat4::from_cols_array(&[
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, -3.0, -1.0,
            0.0, 0.0, -4.0, 0.0,
        ]);
        let ndc_z = |proj: Mat4, z_eye: f32| {
            let clip = proj * Vec3::new(0.0, 0.0, z_eye).extend(1.0);
            clip.z / clip.w
        };
        assert!((ndc_z(gl_proj, -1.0) - (-1.0)).abs() < 1e-5, "sanity: gl near should be -1");
        assert!((ndc_z(gl_proj, -2.0) - 1.0).abs() < 1e-5, "sanity: gl far should be 1");

        let wgpu_proj = Camera::gl_to_wgpu_ndc(gl_proj);
        assert!(
            (ndc_z(wgpu_proj, -1.0) - 0.0).abs() < 1e-5,
            "near should map to wgpu's 0, got {}",
            ndc_z(wgpu_proj, -1.0)
        );
        assert!(
            (ndc_z(wgpu_proj, -2.0) - 1.0).abs() < 1e-5,
            "far should map to wgpu's 1, got {}",
            ndc_z(wgpu_proj, -2.0)
        );
    }
}

