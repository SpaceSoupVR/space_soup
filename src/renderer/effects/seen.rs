//! WHAT AN EFFECT FILLS WHERE IT IS SEEN: the pixels its bounds cover in a
//! view, times the layers of particles over each of them.
//!
//! Counting every particle as if it stood at one reference distance from the
//! eye (the first estimate) made a mist seven metres wide, seen from its bank,
//! about ten times dearer than it is: the far sheets were priced as near ones,
//! and the whole was priced past the edges of the view. What a frame pays is
//! (pixels the effect covers) x (layers over each) -- and no effect covers more
//! than the view. So the cost is the effect's box clipped to the view and
//! projected ([`Eye::pixels`]), times its LAYERS ([`layers`]): its particles'
//! drawn area over the box's own, both seen along the line of sight. That is
//! the same number at every scale: a pond's mist or a puddle's, one torch's
//! smoke or a hall of them.
//!
//! WHO USES IT. The frame's governor (`stack::BurstSet::frame_with`,
//! `effects::governor_wants`) weighs each placed effect by it from the eyes
//! of this frame. The validator (`effect_asset::EffectAsset::check`) states a
//! placed effect's fill from ONE stated view -- standing `reference` metres
//! off its side, eyes 1.6 m over its base, looking at its middle
//! ([`stated_view`]) -- which the editor shows and the budgets refuse by.
//!
//! WHAT IT COSTS: per effect in view and per eye, at most three faces
//! clipped against five planes. A few hundred multiplies.

use glam::{Vec3, Vec4};

use crate::renderer::effect_asset::REFERENCE_FOCAL_PIXELS;

/// The reference view's half-width as a tangent: 1546 pixels over 104
/// degrees at 604 pixels of focal length (`effect_asset`), square.
pub const REFERENCE_TAN: f32 = 773.0 / REFERENCE_FOCAL_PIXELS;
/// Where a standing player's eyes are over the ground a placed effect stands
/// on, metres.
pub const STANDING_EYE: f32 = 1.6;
/// What one point-sized particle covers, pixels (the 1.5 pixel floor).
pub const POINT_PIXELS: f32 = 1.5 * 1.5;

/// One eye's view: where it is, its axes, and how far the view reaches each
/// way as a tangent (left, right, down, up; each positive).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Eye {
    pub at: Vec3,
    pub forward: Vec3,
    pub right: Vec3,
    pub up: Vec3,
    pub tan: [f32; 4],
}

impl Eye {
    /// An eye at `at` looking at `toward`, with the reference view.
    pub fn looking(at: Vec3, toward: Vec3) -> Self {
        let forward = (toward - at).try_normalize().unwrap_or(Vec3::NEG_Z);
        let right = forward.cross(Vec3::Y).try_normalize().unwrap_or(Vec3::X);
        Self { at, forward, right, up: right.cross(forward), tan: [REFERENCE_TAN; 4] }
    }

    /// The eye a view's six frustum planes (`shadow::frustum_planes`: left,
    /// right, bottom, top, near, far, interior positive) describe, at `at`.
    /// A side plane's inward normal is its edge's sideways axis plus the
    /// forward axis times that side's tangent. None for planes that are not
    /// a perspective view's.
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

    /// The whole view, pixels at the reference focal length.
    pub fn view_pixels(&self) -> f32 {
        (self.tan[0] + self.tan[1]) * (self.tan[2] + self.tan[3]) * REFERENCE_FOCAL_PIXELS * REFERENCE_FOCAL_PIXELS
    }

    /// THE PIXELS A BOX COVERS in this view: each face turned toward the eye
    /// clipped to the view and projected (they do not overlap, a box being
    /// convex); all of the view from inside it.
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

    /// A polygon in eye space (x right, y up, z forward) clipped to the view
    /// and projected: its area in tangent units.
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
    /// Whether a point is within the view, or within `margin` metres of it.
    pub fn holds(&self, p: Vec3, margin: f32) -> bool {
        let q = p - self.at;
        let (x, y, z) = (q.dot(self.right), q.dot(self.up), q.dot(self.forward));
        let [l, r, b, t] = self.tan;
        let side = |d: f32, tan: f32| d >= -margin * (1.0 + tan * tan).sqrt();
        z >= -margin && side(x + l * z, l) && side(r * z - x, r) && side(y + b * z, b) && side(t * z - y, t)
    }

    /// The view narrowed to where it sees a box (a doorway's opening): the
    /// box's outline, its tangents, within this view's. None when it is out
    /// of sight; the whole view when it reaches behind the eye (a doorway
    /// the eye stands in passes everything on, as `portal_cull` does).
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

/// THE EYES A BOX IS SEEN THROUGH this frame: each view (`views`, frustum
/// planes, WORLD frame) that holds it -- and from inside a CLOSED room, only
/// through the doorways that lead to the box's room (`goal`, the room of the
/// point it stands at), each view NARROWED to the doorways' openings one
/// after another, as `portal_cull` narrows the terrain's. Empty when it is
/// out of sight. An effect in another room is then simulated and weighed
/// only for what its doorway shows (`effects::emitter_sight`), not whole the
/// moment its door opens.
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
    // The walk: (room, doorway entered by, the eye narrowed so far).
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
                    // Outdoors: any room that is not closed opens onto it.
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

/// THE LAYERS over each pixel of a box seen along `v`: its particles' drawn
/// area (`facing` -- turned to the eye, all of it seen; `flat` -- lying
/// level, seen as much as `v` looks down on it) over the box's own area
/// across `v`. Particles spread through the box, so this is the mean.
pub fn layers(facing: f32, flat: f32, lo: Vec3, hi: Vec3, v: Vec3) -> f32 {
    let h = 0.5 * (hi - lo).max(Vec3::splat(1e-3));
    let v = v.try_normalize().unwrap_or(Vec3::NEG_Z);
    let across = 4.0 * (h.y * h.z * v.x.abs() + h.x * h.z * v.y.abs() + h.x * h.y * v.z.abs());
    (facing.max(0.0) + flat.max(0.0) * v.y.abs()) / across.max(1e-6)
}

/// What a box of particles fills in these eyes, megapixel-layers: each eye's
/// covered pixels times the layers along its line to the box's middle,
/// `points` the 1.5 pixel floor each where the box is in view at all, and
/// `volumes` fog slabs (`sprite_fog`) one layer each over all it covers.
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

/// THE STATED VIEW a placed effect's fill is given for: standing `distance`
/// metres off the side of its box (`lateral` out, `below`/`above` its origin,
/// its up the world's), eyes [`STANDING_EYE`] over its origin, looking at the
/// box's middle -- both eyes, one view each.
pub fn stated_view(lateral: f32, below: f32, above: f32, distance: f32) -> (Vec3, Vec3, [Eye; 2]) {
    stated_view_of([lateral, lateral], below, above, distance)
}

/// [`stated_view`] of a box `half` wide across x and z (a placed effect
/// stretched to its object's footprint), stood off its z side.
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

    /// A small box far off is its particles' own pixels -- the first
    /// estimate, which was right there; a big one near is clipped to the
    /// view, which it was not.
    #[test]
    fn a_small_box_far_off_is_its_particles_and_a_big_one_is_clipped_to_the_view() {
        let area = 0.5;
        let eye = Eye::looking(Vec3::new(0.0, 0.0, 10.0), Vec3::ZERO);
        let (lo, hi) = (Vec3::splat(-0.25), Vec3::splat(0.25));
        let fill = seen_fill(&[eye], lo, hi, area, 0.0, 0.0, 0.0);
        // Each particle at 10 m: its area x (focal / distance)^2.
        let first = area * (REFERENCE_FOCAL_PIXELS / 10.0).powi(2) / 1e6;
        assert!((fill / first - 1.0).abs() < 0.08, "{fill} against {first}");
        // A box twenty metres across, from two metres off its side: never
        // more than the view times its layers.
        let (lo, hi) = (Vec3::new(-10.0, -0.3, -10.0), Vec3::new(10.0, 0.3, 10.0));
        let eye = Eye::looking(Vec3::new(0.0, 1.6, 12.0), Vec3::ZERO);
        let px = eye.pixels(lo, hi);
        assert!(px < 0.5 * eye.view_pixels(), "a thin layer below the eye covers the lower part of the view: {px}");
        let inside = Eye::looking(Vec3::new(0.0, 0.0, 0.0), Vec3::X);
        assert_eq!(inside.pixels(lo, hi), inside.view_pixels());
    }

    /// The eye recovered from a view's planes is the view's: its axes, its
    /// tangents (an asymmetric headset view included), and so its pixels.
    #[test]
    fn the_eye_from_a_views_planes_is_the_view() {
        let pose = Quat::from_rotation_y(0.7) * Quat::from_rotation_x(-0.2);
        let at = Vec3::new(3.0, 1.6, -2.0);
        let fov = (-0.9f32, 0.8f32, 0.75f32, -0.85f32); // left, right, up, down (radians)
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

    /// Flat sheets seen from straight above are all of their area; from
    /// level with them, none.
    #[test]
    fn flat_sheets_are_seen_as_much_as_the_eye_looks_down_on_them() {
        let (lo, hi) = (Vec3::new(-5.0, -0.1, -5.0), Vec3::new(5.0, 0.1, 5.0));
        let top = layers(0.0, 100.0, lo, hi, Vec3::NEG_Y);
        assert!((top - 1.0).abs() < 1e-4, "{top}");
        assert!(layers(0.0, 100.0, lo, hi, Vec3::X) < 1e-6);
    }
}
