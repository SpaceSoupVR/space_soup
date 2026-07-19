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
        // Deliberately OpenGL-convention NDC z ([-1, 1], not wgpu's native
        // [0, 1]) — see `gl_to_wgpu_ndc`'s doc comment for why, and apply
        // that before this ever reaches the GPU.
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

    /// `xr_projection` (and, downstream of it, `mirror::oblique_near_clip`,
    /// which needs this exact convention as input — Lengyel's algorithm is
    /// derived for it) build an OpenGL-style projection: NDC z in [-1, 1].
    /// wgpu's rasterizer instead clips to [0, 1] like Direct3D/Metal/Vulkan,
    /// silently discarding any fragment whose z_ndc < 0. For the main
    /// camera this is invisible — the resulting dead zone is only the
    /// couple centimeters between `near` and where z_ndc crosses 0. But for
    /// the mirror's oblique-clipped projection, the near plane sits at the
    /// mirror surface itself, so that same crossing lands roughly *twice*
    /// the viewer-to-mirror distance into the reflection — which, because
    /// the reflected camera sits symmetrically behind the glass, is almost
    /// exactly where the viewer's own body stands regardless of how far or
    /// close they are from the mirror. That silently clipped away most of
    /// a player's own reflection (only whatever sliver of their body
    /// happened to fall on the far side of that crossing survived), while
    /// scene geometry farther from the mirror than that crossing rendered
    /// fine — exactly the "only a sliver, and only right up against the
    /// glass" symptom this fixes. A GL-to-wgpu NDC remap (z' = (z+w)/2)
    /// applied to the final matrix, after any oblique-clipping, fixes this
    /// without needing a second, wgpu-native derivation of Lengyel's
    /// algorithm — apply it last, never before `oblique_near_clip`, which
    /// still needs the original [-1, 1]-convention matrix as input.
    pub fn gl_to_wgpu_ndc(proj: Mat4) -> Mat4 {
        let remap = Mat4::from_cols_array(&[
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 0.5, 0.0, //
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
        // A minimal OpenGL-style perspective row: z_ndc = -1 at z_eye=-1
        // (near), z_ndc = +1 at z_eye=-2 (far) — same shape xr_projection
        // produces, just simplified to round numbers for a readable test.
        let gl_proj = Mat4::from_cols_array(&[
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, -3.0, -1.0, //
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
