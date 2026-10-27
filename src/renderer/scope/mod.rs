//! Magnified optics: a second view of the world from the objective lens,
//! composited into the ocular.
//!
//! A scope is the same kind of object as the planar mirror: render the world
//! from another viewpoint into a texture, then show that texture inside a
//! masked region. Two facts about real optics shape it and both save work.
//! The image forms at the objective, so the camera sits at the front lens on
//! the optical axis. A parallax-free optic is collimated, so its image does not
//! depend on where the eye is: one render serves both eyes, and the eye only
//! decides the eye-box vignette. The composite therefore samples by ray angle
//! scaled by magnification, which is what makes the glass behave like glass.

use glam::{Mat4, Vec3};

mod lens;
mod stage;
mod target;

pub use lens::{LensParams, LensPipeline};
pub use stage::{EyeSample, ScopeStage};
pub use target::{MipBlit, ScopeTarget};

/// Smallest scope target worth rendering; below it the picture is mush.
pub const MIN_SCOPE_TARGET: u32 = 128;
/// Near plane of the scope camera: past the optic's own objective bell.
pub const SCOPE_NEAR_M: f32 = 0.06;
/// Below this power the image through the glass is the world itself, so the
/// lens draws only its reticle and shadow over the eye's own view.
pub const WINDOW_MAGNIFICATION: f32 = 1.02;

/// The reticle the lens draws, centred on the optical axis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScopeReticle {
    /// Index into the lens shader's styles (`ReticleStyle::shader_index`).
    pub style: u32,
    /// Linear colour times brightness; black draws an etched reticle.
    pub emission: [f32; 3],
    /// Converts world angle (mrad) into the reticle's own unit.
    pub scale: f32,
}

/// How much a scope may cost: the render target's cap and its supersampling.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScopeQuality {
    pub max_size: u32,
    pub supersample: f32,
}

/// One optical path to draw this frame, in render space.
///
/// Built by the client from the held object's pose and its `OpticDef`, so the
/// renderer stays ignorant of scenes and gameplay.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScopeRender {
    /// Front lens: the scope camera sits here.
    pub objective: Vec3,
    /// Rear lens: the image is masked to it.
    pub ocular: Vec3,
    pub ocular_radius_m: f32,
    pub magnification: f32,
    /// True (world) field of view in radians: apparent field over magnification.
    pub true_fov_y_rad: f32,
    /// Per eye: 1 is the full picture, 0 blacked out (the engine's eye box).
    pub occupancy: [f32; 2],
    /// Whether misalignment blacks the glass out (a scope) or only hides the
    /// reticle (a reflex window).
    pub shadow_darkens: bool,
    pub edge_feather_px: f32,
    pub reticle: ScopeReticle,
    pub quality: ScopeQuality,
}

/// An optic's frame: x right, y up, z down the optical axis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScopeBasis {
    pub right: Vec3,
    pub up: Vec3,
    pub forward: Vec3,
}

impl ScopeBasis {
    /// The frame of an optic looking along `axis`, kept level with `world_up`.
    pub fn along(axis: Vec3, world_up: Vec3) -> Self {
        let forward = axis.normalize_or_zero();
        if forward == Vec3::ZERO {
            return Self { right: Vec3::X, up: Vec3::Y, forward: Vec3::NEG_Z };
        }
        let up_hint = if forward.dot(world_up).abs() > 0.999 { Vec3::Z } else { world_up };
        let right = forward.cross(up_hint).normalize_or_zero();
        Self { right, up: right.cross(forward).normalize_or_zero(), forward }
    }

    /// Where a ray leaving the ocular lands in the scope render, or `None`
    /// outside the optic's field: the hard edge of the sight picture.
    ///
    /// Collimated light carries the image in angle: a ray leaving the eyepiece
    /// at angle a off-axis shows the world at a / magnification.
    pub fn uv_for_ray(&self, ray: Vec3, magnification: f32, true_fov_y_rad: f32) -> Option<[f32; 2]> {
        let d = ray.normalize_or_zero();
        let z = d.dot(self.forward);
        if z <= 1e-5 {
            return None;
        }
        let m = magnification.max(1e-3);
        let half = (true_fov_y_rad.max(1e-3) * 0.5).tan();
        let u = ((d.dot(self.right) / z).atan() / m).tan() / half;
        let v = ((d.dot(self.up) / z).atan() / m).tan() / half;
        let inside = (-1.0..=1.0).contains(&u) && (-1.0..=1.0).contains(&v);
        inside.then_some([u * 0.5 + 0.5, 0.5 - v * 0.5])
    }

    /// The direction an eye has moved off the axis, in the lens plane; the
    /// scope shadow grows from the other side. Zero on the axis.
    pub fn offset_direction(&self, ocular: Vec3, eye: Vec3) -> [f32; 2] {
        let to_eye = eye - ocular;
        let lateral = glam::Vec2::new(to_eye.dot(self.right), to_eye.dot(self.up));
        lateral.try_normalize().map_or([0.0; 2], |v| v.to_array())
    }
}

impl ScopeRender {
    /// Optical axis, ocular toward objective.
    pub fn axis(&self) -> Vec3 {
        (self.objective - self.ocular).normalize_or_zero()
    }

    pub fn basis(&self) -> ScopeBasis {
        ScopeBasis::along(self.axis(), Vec3::Y)
    }

    /// Whether the glass shows the world unmagnified, so no world view is rendered.
    pub fn is_window(&self) -> bool {
        self.magnification < WINDOW_MAGNIFICATION
    }

    /// The scope camera, in wgpu clip space, with the eye's own depth range.
    pub fn view_proj(&self, far: f32) -> Mat4 {
        let basis = self.basis();
        let view = Mat4::look_to_rh(self.objective, basis.forward, basis.up);
        let proj = Mat4::perspective_rh_gl(self.true_fov_y_rad.max(1e-3), 1.0, SCOPE_NEAR_M, far);
        super::camera::Camera::gl_to_wgpu_ndc(proj) * view
    }

    /// Diameter in screen pixels that the ocular covers for an eye `eye_to_ocular_m` behind it.
    pub fn ocular_screen_px(&self, eye_to_ocular_m: f32, viewport_height_px: u32, vertical_fov_rad: f32) -> f32 {
        if eye_to_ocular_m <= 1e-4 || vertical_fov_rad <= 1e-4 {
            return 0.0;
        }
        let subtended = 2.0 * (self.ocular_radius_m.max(0.0) / eye_to_ocular_m).atan();
        subtended / vertical_fov_rad * viewport_height_px as f32
    }

    /// Side of the square render target, from how big the lens is on screen.
    ///
    /// A fixed size is both wasteful and the cause of shimmer: compositing a
    /// large target onto a small disc minifies it several times. Rounded up to
    /// a power of two so the mip chain is exact, and capped by the quality.
    pub fn target_size(&self, eye_to_ocular_m: f32, viewport_height_px: u32, vertical_fov_rad: f32) -> u32 {
        let disc = self.ocular_screen_px(eye_to_ocular_m, viewport_height_px, vertical_fov_rad);
        let wanted = (disc * self.quality.supersample.max(1.0)).ceil().max(1.0);
        let pow2 = (wanted as u32).next_power_of_two();
        pow2.clamp(MIN_SCOPE_TARGET, self.quality.max_size.max(MIN_SCOPE_TARGET))
    }

    /// The lens quad in render space: a square over the ocular, facing back
    /// along the axis, which the shader masks to a disc. Nudged 1.5 mm toward
    /// the eye so it wins the depth test against the eyepiece glass it covers.
    pub fn lens_quad(&self) -> [[f32; 3]; 6] {
        let basis = self.basis();
        let r = self.ocular_radius_m.max(1e-4);
        let c = self.ocular - basis.forward * 0.0015;
        let p = |sx: f32, sy: f32| (c + basis.right * (sx * r) + basis.up * (sy * r)).to_array();
        [p(-1.0, -1.0), p(1.0, -1.0), p(1.0, 1.0), p(-1.0, -1.0), p(1.0, 1.0), p(-1.0, 1.0)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FOV: f32 = 26.0_f32 * std::f32::consts::PI / 180.0;
    // Quest-3-ish: ~1000 px tall per eye over a ~96 degree vertical field.
    const VP_H: u32 = 1000;
    const EYE_FOV: f32 = 96.0_f32 * std::f32::consts::PI / 180.0;
    const RELIEF: f32 = 0.095;

    pub(super) fn scope(magnification: f32) -> ScopeRender {
        ScopeRender {
            objective: Vec3::new(0.0, 0.0, -0.2),
            ocular: Vec3::ZERO,
            ocular_radius_m: 0.017,
            magnification,
            true_fov_y_rad: FOV / magnification,
            occupancy: [1.0; 2],
            shadow_darkens: true,
            edge_feather_px: 2.0,
            reticle: ScopeReticle { style: 3, emission: [0.0; 3], scale: 1.0 },
            quality: ScopeQuality { max_size: 1024, supersample: 1.0 },
        }
    }

    fn forward_z() -> ScopeBasis {
        ScopeBasis::along(Vec3::NEG_Z, Vec3::Y)
    }

    #[test]
    fn the_basis_is_orthonormal_and_follows_the_axis() {
        let b = forward_z();
        assert!((b.forward - Vec3::NEG_Z).length() < 1e-5);
        assert!(b.right.dot(b.up).abs() < 1e-5 && b.right.dot(b.forward).abs() < 1e-5);
        assert!((b.up - Vec3::Y).length() < 1e-5, "a level optic keeps world up: {:?}", b.up);
    }

    #[test]
    fn a_vertical_optic_still_gets_a_usable_basis() {
        let b = ScopeBasis::along(Vec3::Y, Vec3::Y);
        assert!(b.right.is_finite() && b.right.length() > 0.9 && b.right.dot(b.forward).abs() < 1e-4);
    }

    #[test]
    fn the_axis_ray_samples_the_centre_at_every_power() {
        for m in [1.0, 4.0, 6.0, 25.0] {
            let uv = forward_z().uv_for_ray(Vec3::NEG_Z, m, FOV).unwrap();
            assert!((uv[0] - 0.5).abs() < 1e-4 && (uv[1] - 0.5).abs() < 1e-4, "{m}x: {uv:?}");
        }
    }

    #[test]
    fn more_power_pulls_the_same_ray_toward_the_centre() {
        let ray = Vec3::new(0.05, 0.0, -1.0);
        let at = |m: f32| forward_z().uv_for_ray(ray, m, FOV).unwrap()[0] - 0.5;
        assert!(at(6.0) > 0.0 && at(6.0) < at(1.0) * 0.25, "1x {} 6x {}", at(1.0), at(6.0));
    }

    #[test]
    fn rays_outside_the_field_or_behind_have_no_sample() {
        let far_off = Vec3::new(40f32.to_radians().tan(), 0.0, -1.0);
        assert!(forward_z().uv_for_ray(far_off, 1.0, FOV).is_none());
        assert!(forward_z().uv_for_ray(Vec3::Z, 1.0, FOV).is_none());
    }

    #[test]
    fn the_eye_offset_points_the_way_the_eye_moved() {
        let b = forward_z();
        assert_eq!(b.offset_direction(Vec3::ZERO, Vec3::new(0.0, 0.0, 0.09)), [0.0; 2]);
        let moved = b.offset_direction(Vec3::ZERO, Vec3::new(0.003, 0.0, 0.09));
        assert!((moved[0] - 1.0).abs() < 1e-5, "{moved:?}");
    }

    #[test]
    fn the_frustum_narrows_with_power_and_looks_down_the_axis() {
        let p = Vec3::new(0.5, 0.0, -10.0);
        let wide = scope(1.0).view_proj(1800.0).project_point3(p);
        let narrow = scope(6.0).view_proj(1800.0).project_point3(p);
        assert!(narrow.x.abs() > wide.x.abs() * 3.0, "{} vs {}", wide.x, narrow.x);
        let centre = scope(6.0).view_proj(1800.0).project_point3(Vec3::new(0.0, 0.0, -50.0));
        assert!(centre.x.abs() < 1e-4 && centre.y.abs() < 1e-4 && (0.0..1.0).contains(&centre.z), "{centre:?}");
    }

    #[test]
    fn a_real_ocular_covers_a_couple_hundred_pixels() {
        let d = scope(4.0).ocular_screen_px(RELIEF, VP_H, EYE_FOV);
        assert!((150.0..280.0).contains(&d), "{d:.0}px");
    }

    #[test]
    fn the_target_follows_the_disc_and_the_quality_caps_it() {
        let s = scope(4.0);
        let size = s.target_size(RELIEF, VP_H, EYE_FOV);
        assert!(size.is_power_of_two() && size < 768 && size as f32 >= s.ocular_screen_px(RELIEF, VP_H, EYE_FOV));
        assert!(s.target_size(0.30, VP_H, EYE_FOV) <= size, "a smaller disc must not cost more");
        let mut fine = s;
        fine.quality = ScopeQuality { max_size: 512, supersample: 2.0 };
        assert!(fine.target_size(RELIEF, VP_H, EYE_FOV) > size);
        assert_eq!(fine.target_size(0.01, VP_H, EYE_FOV), 512);
        assert_eq!(s.target_size(0.0, 0, 0.0), MIN_SCOPE_TARGET);
    }

    #[test]
    fn unmagnified_glass_is_a_window() {
        assert!(scope(1.0).is_window());
        assert!(!scope(1.5).is_window());
    }

    #[test]
    fn the_lens_quad_sits_on_the_ocular_facing_the_eye() {
        let s = scope(4.0);
        let quad = s.lens_quad();
        let centre = quad.iter().map(|p| Vec3::from(*p)).sum::<Vec3>() / 6.0;
        assert!((centre - Vec3::new(0.0, 0.0, 0.0015)).length() < 1e-3, "{centre:?}");
        let corner = Vec3::from(quad[0]) - centre;
        assert!(corner.dot(s.axis()).abs() < 1e-6, "the quad lies in the lens plane");
    }
}
