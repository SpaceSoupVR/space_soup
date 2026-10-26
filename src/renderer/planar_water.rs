use glam::{Mat4, Vec3, Vec4};

pub const ON_SHARE: f32 = 0.03;

pub const OFF_SHARE: f32 = 0.02;

pub const SWITCH_RATIO: f32 = 1.5;

pub const SCALE: f32 = 0.5;

pub const MODEL_REACH: f32 = 25.0;

pub const HEADROOM: f32 = 0.25;

pub const MARGIN: f32 = 0.06;

pub const CLIP_BELOW: f32 = 0.15;

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

pub const SURFACE_SLACK: f32 = 0.1;

pub fn surface_box((lo, hi): (Vec3, Vec3), surface: f32) -> (Vec3, Vec3) {
    (Vec3::new(lo.x, surface - SURFACE_SLACK, lo.z), Vec3::new(hi.x, surface + SURFACE_SLACK, hi.z))
}

pub fn share(rect: [f32; 4]) -> f32 {
    (rect[2] - rect[0]) * (rect[3] - rect[1]) / 4.0
}

pub fn scissor(rect: [f32; 4], width: u32, height: u32) -> (u32, u32, u32, u32) {
    let m = 2.0 * MARGIN;
    let x0 = ((rect[0] - m).max(-1.0) * 0.5 + 0.5) * width as f32;
    let x1 = ((rect[2] + m).min(1.0) * 0.5 + 0.5) * width as f32;

    let y0 = (0.5 - 0.5 * (rect[3] + m).min(1.0)) * height as f32;
    let y1 = (0.5 - 0.5 * (rect[1] - m).max(-1.0)) * height as f32;
    let x = (x0.floor().max(0.0) as u32).min(width);
    let y = (y0.floor().max(0.0) as u32).min(height);
    let w = (x1.ceil() as u32).min(width).saturating_sub(x);
    let h = (y1.ceil() as u32).min(height).saturating_sub(y);
    (x, y, w, h)
}

#[derive(Default, Debug, Clone, Copy)]
pub struct PlanarChoice {
    pub body: Option<usize>,
}

impl PlanarChoice {

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

fn gl_to_wgpu(proj: Mat4) -> Mat4 {
    Mat4::from_cols_array(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 1.0]) * proj
}

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

        let rock = Vec3::new(1.0, 2.0, -20.0);
        let on_mirror = mvp * rock.extend(1.0);
        let image = Vec3::new(rock.x, 2.0 * surface - rock.y, rock.z);
        let on_screen = gl_to_wgpu(proj) * view * image.extend(1.0);
        let a = on_mirror.truncate().truncate() / on_mirror.w;
        let b = on_screen.truncate().truncate() / on_screen.w;
        assert!((a - b).length() < 1e-4, "{a} vs {b}");

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
