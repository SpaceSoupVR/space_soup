use glam::{Vec3, Vec4};

use crate::renderer::effect_asset::REFERENCE_FOCAL_PIXELS;

pub const REFERENCE_TAN: f32 = 773.0 / REFERENCE_FOCAL_PIXELS;

pub const STANDING_EYE: f32 = 1.6;

pub const POINT_PIXELS: f32 = 1.5 * 1.5;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Eye {
    pub at: Vec3,
    pub forward: Vec3,
    pub right: Vec3,
    pub up: Vec3,
    pub tan: [f32; 4],
}

impl Eye {

    pub fn looking(at: Vec3, toward: Vec3) -> Self {
        let forward = (toward - at).try_normalize().unwrap_or(Vec3::NEG_Z);
        let right = forward.cross(Vec3::Y).try_normalize().unwrap_or(Vec3::X);
        Self { at, forward, right, up: right.cross(forward), tan: [REFERENCE_TAN; 4] }
    }

    pub fn from_planes(at: Vec3, planes: &[Vec4; 6]) -> Option<Self> {
        let forward = planes[4].truncate().try_normalize()?;
        let flat = |k: usize| {
            let n = planes[k].truncate();
            (n, n - forward * n.dot(forward))
        };
        let ((l, lf), (r, rf), (b, bf), (t, tf)) = (flat(0), flat(1), flat(2), flat(3));
        let right = (lf - rf).try_normalize()?;
        let up = (bf - tf).try_normalize()?;
        let tan = |n: Vec3, side: f32| n.dot(forward) / side.max(1e-6);
        let tan = [tan(l, l.dot(right)), tan(r, -r.dot(right)), tan(b, b.dot(up)), tan(t, -t.dot(up))];
        tan.iter().all(|t| t.is_finite() && *t > 0.0).then_some(Self { at, forward, right, up, tan })
    }

    pub fn view_pixels(&self) -> f32 {
        (self.tan[0] + self.tan[1]) * (self.tan[2] + self.tan[3]) * REFERENCE_FOCAL_PIXELS * REFERENCE_FOCAL_PIXELS
    }

    pub fn pixels(&self, lo: Vec3, hi: Vec3) -> f32 {
        let a = self.at;
        if a.cmpge(lo).all() && a.cmple(hi).all() {
            return self.view_pixels();
        }
        let mut area = 0.0;
        for k in 0..3 {
            for (plane, facing) in [(lo[k], a[k] < lo[k]), (hi[k], a[k] > hi[k])] {
                if !facing {
                    continue;
                }
                let (i, j) = ((k + 1) % 3, (k + 2) % 3);
                let corner = |u: f32, v: f32| {
                    let mut p = Vec3::ZERO;
                    p[k] = plane;
                    p[i] = u;
                    p[j] = v;
                    let q = p - a;
                    Vec3::new(q.dot(self.right), q.dot(self.up), q.dot(self.forward))
                };
                let quad = [corner(lo[i], lo[j]), corner(hi[i], lo[j]), corner(hi[i], hi[j]), corner(lo[i], hi[j])];
                area += self.projected(&quad);
            }
        }
        (area * REFERENCE_FOCAL_PIXELS * REFERENCE_FOCAL_PIXELS).min(self.view_pixels())
    }

    fn projected(&self, poly: &[Vec3]) -> f32 {
        const NEAR: f32 = 0.01;
        let [l, r, b, t] = self.tan;
        let planes: [&dyn Fn(Vec3) -> f32; 5] =
            [&|q| q.z - NEAR, &|q| q.x + l * q.z, &|q| r * q.z - q.x, &|q| q.y + b * q.z, &|q| t * q.z - q.y];
        let mut pts: Vec<Vec3> = poly.to_vec();
        for f in planes {
            if pts.is_empty() {
                return 0.0;
            }
            let mut out = Vec::with_capacity(pts.len() + 2);
            for n in 0..pts.len() {
                let (p, q) = (pts[n], pts[(n + 1) % pts.len()]);
                let (fp, fq) = (f(p), f(q));
                if fp >= 0.0 {
                    out.push(p);
                }
                if (fp >= 0.0) != (fq >= 0.0) {
                    out.push(p + (q - p) * (fp / (fp - fq)));
                }
            }
            pts = out;
        }
        if pts.len() < 3 {
            return 0.0;
        }
        let flat: Vec<(f32, f32)> = pts.iter().map(|q| (q.x / q.z, q.y / q.z)).collect();
        let twice: f32 = (0..flat.len()).map(|n| {
            let ((x0, y0), (x1, y1)) = (flat[n], flat[(n + 1) % flat.len()]);
            x0 * y1 - x1 * y0
        }).sum();
        0.5 * twice.abs()
    }
}

impl Eye {

    pub fn holds(&self, p: Vec3, margin: f32) -> bool {
        let q = p - self.at;
        let (x, y, z) = (q.dot(self.right), q.dot(self.up), q.dot(self.forward));
        let [l, r, b, t] = self.tan;
        let side = |d: f32, tan: f32| d >= -margin * (1.0 + tan * tan).sqrt();
        z >= -margin && side(x + l * z, l) && side(r * z - x, r) && side(y + b * z, b) && side(t * z - y, t)
    }

    pub fn through(&self, lo: Vec3, hi: Vec3) -> Option<Self> {
        let mut w = [f32::INFINITY, f32::INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY];
        for i in 0..8 {
            let c = Vec3::new(if i & 1 == 0 { lo.x } else { hi.x }, if i & 2 == 0 { lo.y } else { hi.y }, if i & 4 == 0 { lo.z } else { hi.z });
            let q = c - self.at;
            let z = q.dot(self.forward);
            if z <= 0.01 {
                return Some(*self);
            }
            let (x, y) = (q.dot(self.right) / z, q.dot(self.up) / z);
            w = [w[0].min(x), w[1].min(y), w[2].max(x), w[3].max(y)];
        }
        let [l, r, b, t] = self.tan;
        let w = [w[0].max(-l), w[1].max(-b), w[2].min(r), w[3].min(t)];
        (w[0] < w[2] && w[1] < w[3]).then_some(Self { tan: [-w[0], w[2], -w[1], w[3]], ..*self })
    }
}

pub fn sight(
    views: &[[Vec4; 6]],
    head: Vec3,
    lo: Vec3,
    hi: Vec3,
    at: Vec3,
    rooms: &[crate::renderer::portal_cull::CullRoom],
    portals: &[crate::renderer::uniforms::ProbePortal],
) -> Vec<Eye> {
    use crate::renderer::portal_cull::{CullRoom, MAX_DOORWAYS};
    let eyes: Vec<Eye> = views.iter().filter_map(|p| Eye::from_planes(head, p)).filter(|e| e.pixels(lo, hi) > 0.0).collect();
    if eyes.is_empty() {
        return eyes;
    }
    let room_of = |p: Vec3| {
        rooms
            .iter()
            .filter(|r| (r.min.cmplt(p) & p.cmplt(r.max)).all())
            .min_by(|a, b| (a.max - a.min).element_product().total_cmp(&(b.max - b.min).element_product()))
    };
    let Some(start) = room_of(head).filter(|r| r.closed) else { return eyes };
    let goal: Option<&CullRoom> = room_of(at);
    if goal.is_some_and(|g| g.id == start.id) {
        return eyes;
    }

    let mut out = Vec::new();
    let mut frontier: Vec<(u32, Option<usize>, Eye)> = eyes.iter().map(|e| (start.id, None, *e)).collect();
    for _ in 0..MAX_DOORWAYS {
        let mut next = Vec::new();
        for (room, entered, eye) in &frontier {
            for (i, p) in portals.iter().enumerate() {
                if (p.low != *room && p.high != *room) || Some(i) == *entered {
                    continue;
                }
                let Some(narrow) = eye.through(p.min, p.max) else { continue };
                let other = if p.low == *room { p.high } else { p.low };
                let closed = rooms.iter().any(|r| r.id == other && r.closed);
                let arrived = match goal {
                    Some(g) => g.id == other,

                    None => !closed,
                };
                if arrived {
                    out.push(narrow);
                } else if closed {
                    next.push((other, Some(i), narrow));
                }
            }
        }
        if next.is_empty() {
            break;
        }
        frontier = next;
    }
    out.retain(|e| e.pixels(lo, hi) > 0.0);
    out
}

pub fn layers(facing: f32, flat: f32, lo: Vec3, hi: Vec3, v: Vec3) -> f32 {
    let h = 0.5 * (hi - lo).max(Vec3::splat(1e-3));
    let v = v.try_normalize().unwrap_or(Vec3::NEG_Z);
    let across = 4.0 * (h.y * h.z * v.x.abs() + h.x * h.z * v.y.abs() + h.x * h.y * v.z.abs());
    (facing.max(0.0) + flat.max(0.0) * v.y.abs()) / across.max(1e-6)
}

pub fn seen_fill(eyes: &[Eye], lo: Vec3, hi: Vec3, facing: f32, flat: f32, points: f32, volumes: f32) -> f32 {
    let mid = 0.5 * (lo + hi);
    eyes.iter()
        .map(|e| {
            let px = e.pixels(lo, hi);
            if !(px > 0.0) {
                return 0.0;
            }
            let v = (mid - e.at).try_normalize().unwrap_or(e.forward);
            px * (layers(facing, flat, lo, hi, v) + volumes.max(0.0)) + points.max(0.0) * POINT_PIXELS
        })
        .sum::<f32>()
        / 1e6
}

pub fn stated_view(lateral: f32, below: f32, above: f32, distance: f32) -> (Vec3, Vec3, [Eye; 2]) {
    stated_view_of([lateral, lateral], below, above, distance)
}

pub fn stated_view_of(half: [f32; 2], below: f32, above: f32, distance: f32) -> (Vec3, Vec3, [Eye; 2]) {
    let (lo, hi) = (Vec3::new(-half[0], -below, -half[1]), Vec3::new(half[0], above, half[1]));
    let eye = Eye::looking(Vec3::new(0.0, STANDING_EYE, half[1] + distance.max(0.05)), 0.5 * (lo + hi));
    (lo, hi, [eye, eye])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::renderer::camera::Camera;
    use glam::{Mat4, Quat};

    #[test]
    fn a_small_box_far_off_is_its_particles_and_a_big_one_is_clipped_to_the_view() {
        let area = 0.5;
        let eye = Eye::looking(Vec3::new(0.0, 0.0, 10.0), Vec3::ZERO);
        let (lo, hi) = (Vec3::splat(-0.25), Vec3::splat(0.25));
        let fill = seen_fill(&[eye], lo, hi, area, 0.0, 0.0, 0.0);

        let first = area * (REFERENCE_FOCAL_PIXELS / 10.0).powi(2) / 1e6;
        assert!((fill / first - 1.0).abs() < 0.08, "{fill} against {first}");

        let (lo, hi) = (Vec3::new(-10.0, -0.3, -10.0), Vec3::new(10.0, 0.3, 10.0));
        let eye = Eye::looking(Vec3::new(0.0, 1.6, 12.0), Vec3::ZERO);
        let px = eye.pixels(lo, hi);
        assert!(px < 0.5 * eye.view_pixels(), "a thin layer below the eye covers the lower part of the view: {px}");
        let inside = Eye::looking(Vec3::new(0.0, 0.0, 0.0), Vec3::X);
        assert_eq!(inside.pixels(lo, hi), inside.view_pixels());
    }

    #[test]
    fn the_eye_from_a_views_planes_is_the_view() {
        let pose = Quat::from_rotation_y(0.7) * Quat::from_rotation_x(-0.2);
        let at = Vec3::new(3.0, 1.6, -2.0);
        let fov = (-0.9f32, 0.8f32, 0.75f32, -0.85f32);
        let (l, r, u, d) = (fov.0.tan(), fov.1.tan(), fov.2.tan(), fov.3.tan());
        let near = 0.03;
        let far = 1000.0;
        let gl = Mat4::from_cols(
            Vec4::new(2.0 / (r - l), 0.0, 0.0, 0.0),
            Vec4::new(0.0, 2.0 / (u - d), 0.0, 0.0),
            Vec4::new((r + l) / (r - l), (u + d) / (u - d), -(far + near) / (far - near), -1.0),
            Vec4::new(0.0, 0.0, -2.0 * far * near / (far - near), 0.0),
        );
        let view = Mat4::from_rotation_translation(pose, at).inverse();
        let planes = crate::renderer::shadow::frustum_planes(Camera::gl_to_wgpu_ndc(gl) * view);
        let e = Eye::from_planes(at, &planes).expect("a perspective view");
        assert!((e.forward - pose * Vec3::NEG_Z).length() < 1e-3, "{:?}", e.forward);
        assert!((e.right - pose * Vec3::X).length() < 1e-3, "{:?}", e.right);
        assert!((e.up - pose * Vec3::Y).length() < 1e-3, "{:?}", e.up);
        for (got, want) in e.tan.iter().zip([-l, r, -d, u]) {
            assert!((got - want).abs() < 1e-3, "{:?} against {:?}", e.tan, [-l, r, -d, u]);
        }
    }

    #[test]
    fn flat_sheets_are_seen_as_much_as_the_eye_looks_down_on_them() {
        let (lo, hi) = (Vec3::new(-5.0, -0.1, -5.0), Vec3::new(5.0, 0.1, 5.0));
        let top = layers(0.0, 100.0, lo, hi, Vec3::NEG_Y);
        assert!((top - 1.0).abs() < 1e-4, "{top}");
        assert!(layers(0.0, 100.0, lo, hi, Vec3::X) < 1e-6);
    }
}
