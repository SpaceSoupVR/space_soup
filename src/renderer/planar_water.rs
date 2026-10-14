//! THE PLANAR WATER MIRROR (`Levers::planar_water`, 2026-10-14).
//!
//! Outdoor water mirrors the ground map -- a top-down picture of the land --
//! marched along the reflected ray. Seen from the ground, a steep bank behind
//! a pond is a few texels of that picture wide, so a calm pond mirrored it as
//! a smeared streak however still the wind (offline and headset alike). A
//! small calm body is a plane, so here the scene is drawn again mirrored in
//! it, each eye, and the water reads what it mirrors from that image (the
//! water's mirror twins, `water_pipeline::mirrored_wgsl`).
//!
//! THE SCALE RULE: at most ONE body a frame -- of the bodies authored as
//! mirrors (`WaterDef::mirror`), the one covering the most of the screen, by
//! the screen rectangle of its box ([`screen_rect`]), held with hysteresis
//! ([`PlanarChoice`]) so two ponds of a size do not trade the mirror frame to
//! frame. None while no flagged body covers [`ON_SHARE`] of the view, and none
//! with an eye in the water. The others mirror the ground map as before.
//!
//! THE CONTENT BUDGET: the mirrored pass is drawn at [`SCALE`] of the eye's
//! resolution, scissored to the body's rectangle ([`scissor`]), its ground
//! culled by the mirrored frustum, and models only within [`MODEL_REACH`] of
//! the eye.
//!
//! Everything here is in the PLAYER frame, as the renderer's cameras are:
//! the water's still surface is the plane y = its world height minus the
//! player's offset.

use glam::{Mat4, Vec3, Vec4};

/// The share of the view a flagged body must cover for its mirror to start.
pub const ON_SHARE: f32 = 0.03;
/// ... and below which the mirror stops: the gap is the hysteresis.
pub const OFF_SHARE: f32 = 0.02;
/// How much more of the view another flagged body must cover to take the
/// mirror from the one that holds it.
pub const SWITCH_RATIO: f32 = 1.5;
/// The mirror image's resolution, per axis, as a share of the eye's.
pub const SCALE: f32 = 0.5;
/// Metres from the eye past which no model is drawn in the mirror. The bank,
/// the rocks and the sky are; a crate across the field is not.
pub const MODEL_REACH: f32 = 25.0;
/// The mirror image is drawn at this share of the frame's exposure and
/// untoned: the clamp then cuts at four times the eye's white, which keeps a
/// bright sky in the 8-bit image and leaves the water's own tone curve the
/// only one applied.
pub const HEADROOM: f32 = 0.25;
/// The share of the image added round the body's rectangle so a wave's tilt
/// still reads inside what was drawn.
pub const MARGIN: f32 = 0.06;
/// Metres below the still surface the mirrored view starts: the bank where
/// it meets the water is mirrored a little past the waterline, so a ripple
/// or the half-resolution image's edge reading just under it finds the bank
/// and not the sky beyond the cut -- at 2 cm a sky-blue line ran along the
/// far shore (offline, 2026-10-14). What it costs: in water shallower than
/// this, the surface mirrors its own bed, which is what shows through it.
pub const CLIP_BELOW: f32 = 0.15;

/// The screen rectangle -- NDC, `[x0, y0, x1, y1]` within -1..1 -- of a box
/// as seen through `clip` (the box's frame to clip space), or `None` when
/// none of it is in front of the eye or on the screen. The box's edges are
/// cut where they pass behind the eye, so a box the eye stands beside still
/// has the rectangle of the part in front.
pub fn screen_rect(clip: Mat4, lo: Vec3, hi: Vec3) -> Option<[f32; 4]> {
    const NEAR_W: f32 = 1e-3;
    let corner = |i: usize| {
        let p = Vec3::new(
            if i & 1 == 0 { lo.x } else { hi.x },
            if i & 2 == 0 { lo.y } else { hi.y },
            if i & 4 == 0 { lo.z } else { hi.z },
        );
        clip * p.extend(1.0)
    };
    let c: [Vec4; 8] = std::array::from_fn(corner);
    let mut r = [f32::INFINITY, f32::INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY];
    let mut any = false;
    let mut take = |v: Vec4| {
        let (x, y) = (v.x / v.w, v.y / v.w);
        r = [r[0].min(x), r[1].min(y), r[2].max(x), r[3].max(y)];
        any = true;
    };
    for v in &c {
        if v.w > NEAR_W {
            take(*v);
        }
    }
    // Where an edge passes behind the eye, the point it does so.
    for a in 0..8usize {
        for bit in [1usize, 2, 4] {
            let b = a | bit;
            if b == a {
                continue;
            }
            let (va, vb) = (c[a], c[b]);
            if (va.w > NEAR_W) != (vb.w > NEAR_W) {
                let t = (NEAR_W - va.w) / (vb.w - va.w);
                take(va + (vb - va) * t);
            }
        }
    }
    if !any {
        return None;
    }
    let r = [r[0].max(-1.0), r[1].max(-1.0), r[2].min(1.0), r[3].min(1.0)];
    (r[0] < r[2] && r[1] < r[3]).then_some(r)
}

/// Metres above and below the still surface a mirror body's box is taken
/// to reach for its rectangle: a calm body's waves are a few centimetres.
pub const SURFACE_SLACK: f32 = 0.1;

/// The box a mirror body's rectangle is found from: its bounds across, its
/// still surface give or take [`SURFACE_SLACK`] up and down -- not the
/// bounds' two metres of wave allowance, which stood above the eye and made
/// every pond fill the screen.
pub fn surface_box((lo, hi): (Vec3, Vec3), surface: f32) -> (Vec3, Vec3) {
    (Vec3::new(lo.x, surface - SURFACE_SLACK, lo.z), Vec3::new(hi.x, surface + SURFACE_SLACK, hi.z))
}

/// The share of the screen a rectangle from [`screen_rect`] covers.
pub fn share(rect: [f32; 4]) -> f32 {
    (rect[2] - rect[0]) * (rect[3] - rect[1]) / 4.0
}

/// The rectangle grown by [`MARGIN`] and cut to the screen, in pixels of a
/// `width` x `height` image: `(x, y, w, h)` from the top left, for
/// `set_scissor_rect`.
pub fn scissor(rect: [f32; 4], width: u32, height: u32) -> (u32, u32, u32, u32) {
    let m = 2.0 * MARGIN;
    let x0 = ((rect[0] - m).max(-1.0) * 0.5 + 0.5) * width as f32;
    let x1 = ((rect[2] + m).min(1.0) * 0.5 + 0.5) * width as f32;
    // NDC y is up; pixels count down.
    let y0 = (0.5 - 0.5 * (rect[3] + m).min(1.0)) * height as f32;
    let y1 = (0.5 - 0.5 * (rect[1] - m).max(-1.0)) * height as f32;
    let x = (x0.floor().max(0.0) as u32).min(width);
    let y = (y0.floor().max(0.0) as u32).min(height);
    let w = (x1.ceil() as u32).min(width).saturating_sub(x);
    let h = (y1.ceil() as u32).min(height).saturating_sub(y);
    (x, y, w, h)
}

/// WHICH BODY IS THE FRAME'S MIRROR, held from frame to frame.
#[derive(Default, Debug, Clone, Copy)]
pub struct PlanarChoice {
    pub body: Option<usize>,
}

impl PlanarChoice {
    /// This frame's mirror from each body's share of the view (`None`: not a
    /// mirror, or not seen): the one holding it keeps it while it covers
    /// [`OFF_SHARE`] and no other covers [`SWITCH_RATIO`] times as much; a
    /// new one needs [`ON_SHARE`].
    pub fn choose(&mut self, shares: &[Option<f32>]) -> Option<usize> {
        let best = shares
            .iter()
            .enumerate()
            .filter_map(|(i, s)| s.map(|s| (i, s)))
            .max_by(|a, b| a.1.total_cmp(&b.1));
        let held = self.body.and_then(|i| shares.get(i).copied().flatten().map(|s| (i, s)));
        self.body = match (held, best) {
            (Some((i, s)), Some((j, t))) if s >= OFF_SHARE => {
                if j != i && t >= ON_SHARE && t > SWITCH_RATIO * s { Some(j) } else { Some(i) }
            }
            (_, Some((j, t))) if t >= ON_SHARE => Some(j),
            _ => None,
        };
        self.body
    }
}

/// THE MIRRORED CAMERA for an eye: its view-projection (wgpu clip space) and
/// its eye, mirrored in the still surface at player-frame height `surface`.
/// `view` is the eye's view, `proj` its GL-style projection; the near plane
/// is made the water's (an oblique near plane), so nothing under the
/// surface -- the bed, the bank below the waterline -- is mirrored above it.
/// Every winding is reversed: draw with the mirror pipelines.
pub fn mirror_camera(view: Mat4, proj: Mat4, surface: f32) -> (Mat4, Vec3) {
    use crate::renderer::mirror;
    let plane_point = Vec3::new(0.0, surface, 0.0);
    let mirror_view = view * mirror::reflection_matrix(plane_point, Vec3::Y);
    let world_plane = mirror::world_plane_equation(plane_point - Vec3::Y * CLIP_BELOW, Vec3::Y);
    let eye_plane = mirror::plane_to_eye_space(mirror_view.inverse(), world_plane);
    let mirror_proj = mirror::oblique_near_clip(proj, eye_plane);
    let eye = mirror_view.inverse().transform_point3(Vec3::ZERO);
    (gl_to_wgpu(mirror_proj) * mirror_view, eye)
}

/// `Camera::gl_to_wgpu_ndc`, here so the module needs no headset.
fn gl_to_wgpu(proj: Mat4) -> Mat4 {
    Mat4::from_cols_array(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 1.0]) * proj
}

/// The mirror twins' uniform lane (`WaterUniform::mirror`) for this frame:
/// authored, this frame's mirror, one over the exposure the image was drawn
/// at, and whether it is flipped left to right.
pub fn uniform_lane(authored: bool, active: bool, exposure: f32, flipped: bool) -> [f32; 4] {
    [
        if authored { 1.0 } else { 0.0 },
        if active { 1.0 } else { 0.0 },
        1.0 / (exposure * HEADROOM).max(1e-6),
        if flipped { 1.0 } else { 0.0 },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn eye() -> (Mat4, Mat4) {
        let view = Mat4::look_at_rh(Vec3::new(0.0, 1.6, 0.0), Vec3::new(0.0, 0.0, -10.0), Vec3::Y);
        let proj = Mat4::perspective_rh_gl(90f32.to_radians(), 1.0, 0.03, 1000.0);
        (view, proj)
    }

    #[test]
    fn a_pond_ahead_has_a_rect_and_one_behind_has_none() {
        let (view, proj) = eye();
        let clip = gl_to_wgpu(proj) * view;
        let ahead = screen_rect(clip, Vec3::new(-3.0, -0.4, -12.0), Vec3::new(3.0, -0.3, -6.0)).unwrap();
        assert!(ahead[1] < 0.0 && share(ahead) > 0.01 && share(ahead) < 0.5, "{ahead:?}");
        assert!(screen_rect(clip, Vec3::new(-3.0, -0.4, 6.0), Vec3::new(3.0, -0.3, 12.0)).is_none());
    }

    #[test]
    fn standing_at_the_edge_the_pond_fills_the_lower_screen() {
        let (view, proj) = eye();
        let clip = gl_to_wgpu(proj) * view;
        // The eye over the near edge: corners behind it are cut, not lost.
        let r = screen_rect(clip, Vec3::new(-10.0, -0.4, -20.0), Vec3::new(10.0, -0.3, 2.0)).unwrap();
        assert!(r[0] <= -0.99 && r[2] >= 0.99 && r[1] <= -0.99, "{r:?}");
    }

    #[test]
    fn the_mirror_holds_against_flicker_and_passes_to_a_much_larger_pond() {
        let mut c = PlanarChoice::default();
        assert_eq!(c.choose(&[Some(0.01), None]), None, "too small to start");
        assert_eq!(c.choose(&[Some(0.04), None]), Some(0));
        assert_eq!(c.choose(&[Some(0.025), None]), Some(0), "held between off and on");
        assert_eq!(c.choose(&[Some(0.03), Some(0.04)]), Some(0), "a little larger does not take it");
        assert_eq!(c.choose(&[Some(0.03), Some(0.05)]), Some(1), "half as large again does");
        assert_eq!(c.choose(&[Some(0.03), None]), Some(0), "gone from sight: the other takes it");
        assert_eq!(c.choose(&[Some(0.015), None]), None, "below the off share it stops");
    }

    #[test]
    fn the_mirrored_camera_sees_the_bank_where_the_water_mirrors_it() {
        let (view, proj) = eye();
        let surface = -0.35;
        let (mvp, meye) = mirror_camera(view, proj, surface);
        assert!((meye - Vec3::new(0.0, 2.0 * surface - 1.6, 0.0)).length() < 1e-4);
        // A rock top above the far bank, and where the still water mirrors it.
        let rock = Vec3::new(1.0, 2.0, -20.0);
        let on_mirror = mvp * rock.extend(1.0);
        let image = Vec3::new(rock.x, 2.0 * surface - rock.y, rock.z);
        let on_screen = gl_to_wgpu(proj) * view * image.extend(1.0);
        let a = on_mirror.truncate().truncate() / on_mirror.w;
        let b = on_screen.truncate().truncate() / on_screen.w;
        assert!((a - b).length() < 1e-4, "{a} vs {b}");
        // The bed under the water is clipped away by the near plane.
        let bed = mvp * Vec3::new(0.0, surface - 0.5, -5.0).extend(1.0);
        assert!(bed.z < 0.0 || bed.z > bed.w, "bed depth {} of {}", bed.z, bed.w);
    }

    #[test]
    fn the_scissor_is_the_rect_and_a_margin_in_pixels() {
        assert_eq!(scissor([-1.0, -1.0, 1.0, 1.0], 800, 600), (0, 0, 800, 600));
        let (x, y, w, h) = scissor([-0.5, -1.0, 0.5, -0.2], 1000, 1000);
        assert_eq!((x, w), (190, 620));
        assert_eq!((y, y + h), (540, 1000));
    }
}
