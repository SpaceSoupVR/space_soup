use bytemuck::{Pod, Zeroable};
use glam::{Quat, Vec3};
use wgpu::*;

pub const LAMP_RADIUS: f32 = 0.05;

pub const VEIL_SHARE: f32 = 0.06;

const CIE_CUBE: f32 = 10.0;
const CIE_AGE: f32 = 1.0 + 0.4 * 0.4 * 0.4 * 0.4;
const CIE_SQUARE: f32 = 5.0 * CIE_AGE;
const CIE_LINEAR: f32 = 0.1 * 0.5 * CIE_AGE;

/// `cie_veil`'s 1/theta, 1/theta^2 and 1/theta^3 terms, for the sky
/// shader's sun and moon halos (`sky::sky_shader`).
pub const CIE_TERMS: [f32; 3] = [CIE_LINEAR, CIE_SQUARE, CIE_CUBE];

const VEIL_FLOOR: f32 = 0.02;

pub const HALO_CAP: f32 = 0.04;

pub const MAX_GLARE_DEGREES: f32 = 25.0;

const MIN_CORE_DEGREES: f32 = 0.5;

const WALL_MARGIN: f32 = LAMP_RADIUS;

#[derive(Clone, Debug, PartialEq)]
pub struct GlareTable {

    pub rows: usize,

    pub cols: usize,

    pub share: Vec<f32>,

    pub centre: Vec<Vec3>,

    pub split: Option<GlareTableSplit>,

    pub air: Option<GlareAir>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GlareAir {

    pub min: Vec3,
    pub max: Vec3,

    pub dims: [usize; 3],

    pub bulb: Vec3,

    pub seen: Vec<f32>,

    pub reach: f32,
}

impl GlareAir {

    pub fn seen_from(&self, local: Vec3) -> f32 {
        let [nx, ny, nz] = self.dims;
        if nx < 2 || ny < 2 || nz < 2 || self.seen.len() != nx * ny * nz || !local.is_finite() {
            return 1.0;
        }
        let last = Vec3::new((nx - 1) as f32, (ny - 1) as f32, (nz - 1) as f32);
        let g = ((local - self.min) / (self.max - self.min).max(Vec3::splat(1e-6)) * last).clamp(Vec3::ZERO, last);
        let (lo, f) = (g.floor(), g - g.floor());
        let (x0, y0, z0) = (lo.x as usize, lo.y as usize, lo.z as usize);
        let (x1, y1, z1) = ((x0 + 1).min(nx - 1), (y0 + 1).min(ny - 1), (z0 + 1).min(nz - 1));
        let at = |x: usize, y: usize, z: usize| self.seen[(z * ny + y) * nx + x];
        let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
        let plane = |z: usize| {
            lerp(lerp(at(x0, y0, z), at(x1, y0, z), f.x), lerp(at(x0, y1, z), at(x1, y1, z), f.x), f.y)
        };
        lerp(plane(z0), plane(z1), f.z).clamp(0.0, 1.0)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct GlareTableSplit {

    pub bulb: Vec<f32>,

    pub bulb_centre: Vec<Vec3>,

    pub lit_centre: Vec<Vec3>,

    pub lit_spread: Vec<f32>,

    pub bulb_fine: Option<GlareBulbRows>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GlareBulbRows {
    pub rows: usize,
    pub share: Vec<f32>,
    pub centre: Vec<Vec3>,

    pub whole: f32,
}

const MIN_BULB_SHOWING: f32 = 0.05;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GlareLobe {
    pub share: f32,
    pub centre: Vec3,
    pub radius: f32,
}

impl GlareTable {

    pub fn direction(&self, row: usize, col: usize) -> Vec3 {
        let theta = (row as f32 + 0.5) / self.rows.max(1) as f32 * std::f32::consts::PI;
        let phi = (col as f32 + 0.5) / self.cols.max(1) as f32 * std::f32::consts::TAU;
        Vec3::new(
            theta.sin() * phi.cos(),
            theta.cos(),
            theta.sin() * phi.sin(),
        )
    }

    pub fn sample(&self, local: Vec3) -> (f32, Vec3) {
        let Some(around) = self.around(local) else {
            return (0.0, Vec3::ZERO);
        };
        let (mut share, mut centre) = (0.0f32, Vec3::ZERO);
        for (i, w) in around {
            let w = w * self.share[i].max(0.0);
            share += w;
            centre += self.centre[i] * w;
        }
        (
            share,
            if share > 1e-9 {
                centre / share
            } else {
                Vec3::ZERO
            },
        )
    }

    pub fn sample_split(&self, local: Vec3) -> Option<[GlareLobe; 2]> {
        let split = self.split.as_ref()?;
        let n = self.rows * self.cols;
        if split.bulb.len() != n || split.bulb_centre.len() != n || split.lit_centre.len() != n || split.lit_spread.len() != n {
            return None;
        }
        let around = self.around(local)?;
        let (mut bulb, mut bulb_at) = (0.0f32, Vec3::ZERO);
        let (mut lit, mut lit_at, mut spread) = (0.0f32, Vec3::ZERO, 0.0f32);
        for (i, w) in around {
            let whole = self.share[i].max(0.0);
            let b = split.bulb[i].clamp(0.0, whole);
            bulb += w * b;
            bulb_at += split.bulb_centre[i] * (w * b);
            lit += w * (whole - b);
            lit_at += split.lit_centre[i] * (w * (whole - b));
            spread += split.lit_spread[i] * (w * (whole - b));
        }

        let mut bulb_radius = LAMP_RADIUS;
        if let Some(fine) = split.bulb_fine.as_ref().filter(|f| {
            f.rows > 0 && f.share.len() == f.rows * self.cols && f.centre.len() == f.share.len()
        }) {
            (bulb, bulb_at) = (0.0, Vec3::ZERO);
            for (i, w) in around_in(fine.rows, self.cols, local)? {
                let b = w * fine.share[i].max(0.0);
                bulb += b;
                bulb_at += fine.centre[i] * b;
            }
            if fine.whole > 0.0 {
                bulb_radius = LAMP_RADIUS * (bulb / fine.whole).clamp(MIN_BULB_SHOWING, 1.0).sqrt();
            }
        }
        let middle = |sum: Vec3, weight: f32| if weight > 1e-9 { sum / weight } else { Vec3::ZERO };
        Some([
            GlareLobe { share: bulb, centre: middle(bulb_at, bulb), radius: bulb_radius },
            GlareLobe {
                share: lit,
                centre: middle(lit_at, lit),
                radius: (std::f32::consts::SQRT_2 * if lit > 1e-9 { spread / lit } else { 0.0 }).max(LAMP_RADIUS),
            },
        ])
    }

    pub fn glow_middle(&self) -> Vec3 {
        let (mut sum, mut weight) = (Vec3::ZERO, 0.0f32);
        for (share, centre) in self.share.iter().zip(&self.centre) {
            sum += *centre * share.max(0.0);
            weight += share.max(0.0);
        }
        if weight > 1e-9 {
            sum / weight
        } else {
            Vec3::ZERO
        }
    }

    fn around(&self, local: Vec3) -> Option<[(usize, f32); 4]> {
        let n = self.rows * self.cols;
        if n == 0 || self.share.len() != n || self.centre.len() != n {
            return None;
        }
        around_in(self.rows, self.cols, local)
    }
}

fn around_in(rows: usize, cols: usize, local: Vec3) -> Option<[(usize, f32); 4]> {
    let d = local.try_normalize()?;
    if rows == 0 || cols == 0 {
        return None;
    }
    let theta = d.y.clamp(-1.0, 1.0).acos();
    let phi = d.z.atan2(d.x).rem_euclid(std::f32::consts::TAU);
    let y = (theta / std::f32::consts::PI * rows as f32 - 0.5).clamp(0.0, (rows - 1) as f32);
    let x = phi / std::f32::consts::TAU * cols as f32 - 0.5;
    let (r0, fy) = (y.floor() as usize, y.fract());
    let r1 = (r0 + 1).min(rows - 1);
    let x0 = x.floor();
    let fx = x - x0;
    let c0 = (x0 as i64).rem_euclid(cols as i64) as usize;
    let c1 = (c0 + 1) % cols;
    let at = |r: usize, c: usize| r * cols + c;
    Some([
        (at(r0, c0), (1.0 - fx) * (1.0 - fy)),
        (at(r0, c1), fx * (1.0 - fy)),
        (at(r1, c0), (1.0 - fx) * fy),
        (at(r1, c1), fx * fy),
    ])
}

fn table_direction(table: &GlareTable, rotation: Quat, origin: Vec3, eye: Vec3) -> Vec3 {
    rotation.inverse() * (eye - (origin + rotation * table.glow_middle()))
}

#[derive(Clone, Debug, PartialEq)]
pub struct GlareSource {

    pub position: Vec3,

    pub radiance: Vec3,

    pub sides: [f32; 6],

    pub rotation: Quat,

    pub cone: Option<(Vec3, f32, f32)>,

    pub centres: Option<[Vec3; 6]>,

    pub table: Option<(std::sync::Arc<GlareTable>, Vec3)>,

    pub halo_only: bool,

    pub mirror: Option<Vec3>,
}

pub fn visible_centre(s: &GlareSource, eye: Vec3) -> Vec3 {
    if let Some((table, origin)) = &s.table {
        let (share, centre) = table.sample(table_direction(table, s.rotation, *origin, eye));
        return if share > 0.0 {
            *origin + s.rotation * centre
        } else {
            s.position
        };
    }
    let Some(centres) = s.centres else { return s.position };
    let Some(to_eye) = (eye - s.position).try_normalize() else { return s.position };
    let local = s.rotation.inverse() * to_eye;
    let (mut sum, mut weight) = (Vec3::ZERO, 0.0f32);
    for a in 0..3 {
        let c = local[a];
        let k = if c >= 0.0 { 2 * a } else { 2 * a + 1 };
        let w = c * c * s.sides[k].max(0.0);
        sum += centres[k] * w;
        weight += w;
    }
    if weight > 1e-6 {
        sum / weight
    } else {
        s.position
    }
}

pub fn visible_share(s: &GlareSource, eye: Vec3) -> f32 {
    if let Some((table, origin)) = &s.table {
        return table.sample(table_direction(table, s.rotation, *origin, eye)).0;
    }
    let Some(to_eye) = (eye - s.position).try_normalize() else { return 0.0 };
    let local = s.rotation.inverse() * to_eye;
    let mut share = 0.0;
    for a in 0..3 {
        let c = local[a];
        share += c * c * s.sides[if c >= 0.0 { 2 * a } else { 2 * a + 1 }];
    }
    if let Some((axis, cos_outer, cos_inner)) = s.cone {
        let c = axis.normalize_or_zero().dot(to_eye);
        share *= smoothstep(cos_outer, cos_inner.max(cos_outer + 1e-3), c);
    }
    share.max(0.0)
}

pub fn glare_lobes(s: &GlareSource, eye: Vec3) -> Vec<GlareLobe> {
    if let Some((table, origin)) = &s.table {
        if let Some(lobes) = table.sample_split(table_direction(table, s.rotation, *origin, eye)) {
            return lobes
                .into_iter()
                .filter(|l| l.share > 0.0)
                .map(|l| GlareLobe { centre: *origin + s.rotation * l.centre, ..l })
                .collect();
        }
    }
    let share = visible_share(s, eye);
    if share > 0.0 {
        vec![GlareLobe { share, centre: visible_centre(s, eye), radius: LAMP_RADIUS }]
    } else {
        Vec::new()
    }
}

pub fn bulb_in_view(s: &GlareSource, eye: Vec3) -> f32 {
    if let Some((table, origin)) = &s.table {
        let local = table_direction(table, s.rotation, *origin, eye);
        let fine = table.split.as_ref().and_then(|split| split.bulb_fine.as_ref()).filter(|f| f.whole > 0.0);
        if let (Some(fine), Some(lobes)) = (fine, table.sample_split(local)) {
            return (lobes[0].share / fine.whole).clamp(0.0, 1.0);
        }
        let most = table.share.iter().copied().fold(0.0, f32::max);
        return if most > 0.0 { (table.sample(local).0 / most).clamp(0.0, 1.0) } else { 0.0 };
    }
    let most = s.sides.iter().copied().fold(0.0, f32::max);
    if most > 0.0 {
        (visible_share(s, eye) / most).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GlareVertex {
    pub position: [f32; 3],

    pub colour: [f32; 3],

    pub uv: [f32; 2],

    pub shape: [f32; 4],

    pub test: [f32; 4],

    pub reach: [f32; 4],
}

const AIR_CELLS: usize = 16;

impl GlareVertex {
    pub const ATTRIBS: [VertexAttribute; 6] = vertex_attr_array![
        0 => Float32x3, 1 => Float32x3, 2 => Float32x2, 3 => Float32x4, 4 => Float32x4, 5 => Float32x4
    ];

    pub fn layout() -> VertexBufferLayout<'static> {
        VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as BufferAddress,
            step_mode: VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

const EYE_CLEARANCE: f32 = 0.02;

pub fn capsule_visibility(bulb: Vec3, eye: Vec3, capsules: &[(Vec3, Vec3, f32)]) -> f32 {
    let mut open = 1.0;
    for &(a, b, radius) in capsules {
        let (_, eye_gap) = closest_on_segment(eye, a, b);
        if eye_gap < radius + EYE_CLEARANCE {
            continue;
        }
        let (t, gap) = segment_gap(eye, bulb, a, b);

        let soft = (LAMP_RADIUS * t).max(1e-4);
        open *= smoothstep(radius - soft, radius + soft, gap);
    }
    open
}

pub fn disc_visibility(disc: Vec3, radius: f32, from: Vec3, capsules: &[(Vec3, Vec3, f32)]) -> f32 {
    let mut open = 1.0;
    for &(a, b, r) in capsules {
        if closest_on_segment(disc, a, b).1 < r + radius {
            continue;
        }
        let (t, gap) = segment_gap(from, disc, a, b);
        let soft = (radius * t).max(1e-4);
        open *= smoothstep(r - soft, r + soft, gap);
    }
    open
}

fn closest_on_segment(p: Vec3, a: Vec3, b: Vec3) -> (f32, f32) {
    let ab = b - a;
    let t = if ab.length_squared() > 0.0 { ((p - a).dot(ab) / ab.length_squared()).clamp(0.0, 1.0) } else { 0.0 };
    (t, (a + ab * t - p).length())
}

fn segment_gap(p: Vec3, q: Vec3, a: Vec3, b: Vec3) -> (f32, f32) {
    let (d1, d2, r) = (q - p, b - a, p - a);
    let (aa, ee, f) = (d1.length_squared(), d2.length_squared(), d2.dot(r));
    let (s, t) = if aa <= 1e-12 {
        (0.0, if ee > 1e-12 { (f / ee).clamp(0.0, 1.0) } else { 0.0 })
    } else {
        let c = d1.dot(r);
        if ee <= 1e-12 {
            ((-c / aa).clamp(0.0, 1.0), 0.0)
        } else {
            let bb = d1.dot(d2);
            let denom = aa * ee - bb * bb;
            let mut s = if denom > 1e-12 { ((bb * f - c * ee) / denom).clamp(0.0, 1.0) } else { 0.0 };
            let mut t = (bb * s + f) / ee;
            if t < 0.0 {
                t = 0.0;
                s = (-c / aa).clamp(0.0, 1.0);
            } else if t > 1.0 {
                t = 1.0;
                s = ((bb - c) / aa).clamp(0.0, 1.0);
            }
            (s, t)
        }
    };
    (s, ((p + d1 * s) - (a + d2 * t)).length())
}

pub fn build_glare(
    sources: &[GlareSource],
    eyes: [Vec3; 2],
    right: Vec3,
    up: Vec3,
    exposure: f32,
    strength: f32,
    test_walls: bool,
    capsules: &[(Vec3, Vec3, f32)],
    adapted: &[bool],
) -> (Vec<GlareVertex>, Vec<u32>, u32) {
    let mut verts = Vec::new();
    let mut idx = Vec::new();
    let mut cores = Vec::new();
    let eye = 0.5 * (eyes[0] + eyes[1]);

    let mut quad = |into: &mut Vec<u32>,
                    q: &GlareQuad,
                    centre: Vec3,
                    half: f32,
                    degrees: f32,
                    test: [f32; 4],
                    reach: Vec3,
                    cells: usize,
                    air: &dyn Fn(Vec3) -> f32| {
        let base = verts.len() as u32;
        let n = cells + 1;
        for j in 0..n {
            for i in 0..n {
                let (du, dv) = (2.0 * i as f32 / cells as f32 - 1.0, 2.0 * j as f32 / cells as f32 - 1.0);
                let position = centre + right * (du * half) + up * (dv * half);
                verts.push(GlareVertex {
                    position: position.to_array(),
                    colour: q.colour.to_array(),
                    uv: [du, dv],
                    shape: [q.a, q.core2, degrees * degrees, q.edge],
                    test,
                    reach: [reach.x, reach.y, reach.z, air(position)],
                });
            }
        }
        let n = n as u32;
        for j in 0..cells as u32 {
            for i in 0..cells as u32 {
                let a = base + j * n + i;

                let dark = [a, a + 1, a + 1 + n, a + n].iter().all(|&c| verts[c as usize].reach[3] <= 0.0);
                let (lo, hi) = (verts[a as usize].uv, verts[(a + 1 + n) as usize].uv);
                let (nu, nv) = (0.0f32.clamp(lo[0], hi[0]), 0.0f32.clamp(lo[1], hi[1]));
                if dark || nu * nu + nv * nv >= 1.0 {
                    continue;
                }
                into.extend_from_slice(&[a, a + 1, a + 1 + n, a, a + 1 + n, a + n]);
            }
        }
    };

    let forward = up.cross(right).normalize_or_zero();

    let parts = sources.iter().enumerate().flat_map(|(k, s)| glare_lobes(s, eye).into_iter().map(move |l| (k, s, l)));
    for (k, s, lobe) in parts {
        let at = lobe.centre;

        let seen = s.mirror.unwrap_or(at);
        let shielded =
            0.5 * (capsule_visibility(seen, eyes[0], capsules) + capsule_visibility(seen, eyes[1], capsules));
        let Some(q) = glare_quad(s, &lobe, eye, exposure, strength * shielded) else { continue };

        let to_eye = eye - at;
        let d = to_eye.length().max(1e-6);
        let toward = to_eye / d;

        let from_seen = (eye - seen).length();
        let tested = seen + toward * WALL_MARGIN.min(0.5 * from_seen);
        let from_tested = (eye - tested).length();
        let half = q.half * from_tested / d;

        let halo_only = s.halo_only || s.mirror.is_some();
        let flag = if halo_only { 3.0 } else { 1.0 };
        let test = [tested.x, tested.y, tested.z, if test_walls { flag } else { -flag }];

        let reach = right * lobe.radius * if s.mirror.is_some() { from_tested / d } else { 1.0 };

        let lit_air = s.table.as_ref().and_then(|(t, origin)| t.air.as_ref().map(|a| (a, *origin)));
        let cells = if lit_air.is_some() { AIR_CELLS } else { 1 };
        let air = |point: Vec3| -> f32 {
            let Some((a, origin)) = lit_air else { return 1.0 };
            let Some(ray) = (point - eye).try_normalize() else { return 1.0 };
            let bulb = origin + s.rotation * a.bulb;
            let nearest = eye + ray * (bulb - eye).dot(ray).max(0.0);
            a.seen_from(s.rotation.inverse() * (nearest - origin))
        };
        quad(&mut idx, &q, tested, half, q.degrees, test, reach, cells, &air);
        if q.core_degrees > 0.0 && !halo_only {

            let behind = lit_air.filter(|(a, _)| a.reach > 0.0 && adapted.get(k).copied().unwrap_or(false));
            let (core_centre, core_test) = match behind {
                Some((a, origin)) => {
                    let bulb = origin + s.rotation * a.bulb;
                    let along = (-toward).dot(forward).max(0.5);
                    let past = ((bulb - at).dot(forward) + a.reach + WALL_MARGIN).max(0.0) / along;
                    (at - toward * past, [test[0], test[1], test[2], if test_walls { 2.0 } else { -1.0 }])
                }
                None => (at + toward * WALL_MARGIN.max(lobe.radius).min(0.5 * d), test),
            };
            let from_core = (eye - core_centre).length();
            let core_half = from_core * q.core_degrees.to_radians().tan();
            quad(&mut cores, &q, core_centre, core_half, q.core_degrees, core_test, reach, cells, &air);
        }
    }
    let halos = idx.len() as u32;
    idx.extend(cores);
    (verts, idx, halos)
}

pub fn cie_veil(theta2: f32) -> f32 {
    let inv = theta2.max(1e-6).sqrt().recip();
    inv * (CIE_LINEAR + inv * (CIE_SQUARE + inv * CIE_CUBE))
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GlareQuad {
    pub half: f32,
    pub colour: Vec3,

    pub a: f32,

    pub core2: f32,

    pub degrees: f32,

    pub edge: f32,

    pub peak: f32,

    pub core_degrees: f32,
}

pub fn glare_quad(s: &GlareSource, lobe: &GlareLobe, eye: Vec3, exposure: f32, strength: f32) -> Option<GlareQuad> {
    let d = (eye - lobe.centre).length();
    let luminance = s.radiance.dot(Vec3::new(0.2126, 0.7152, 0.0722));
    if !(d > 0.0) || luminance <= 0.0 || lobe.share <= 0.0 || strength <= 0.0 {
        return None;
    }

    let e = luminance * lobe.share / (d * d).max(LAMP_RADIUS * LAMP_RADIUS);
    let a = exposure * strength * VEIL_SHARE * std::f32::consts::PI * e;

    let core = (lobe.radius / d).atan().to_degrees().max(MIN_CORE_DEGREES);
    let core2 = core * core;
    let veil = |theta: f32| a * cie_veil(theta * theta + core2);
    if veil(0.0) < VEIL_FLOOR {
        return None;
    }

    let degrees = if veil(MAX_GLARE_DEGREES) >= VEIL_FLOOR {
        MAX_GLARE_DEGREES
    } else {
        let (mut lo, mut hi) = (0.0, MAX_GLARE_DEGREES);
        for _ in 0..20 {
            let mid = 0.5 * (lo + hi);
            if veil(mid) >= VEIL_FLOOR {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        hi.max(2.0 * core).min(MAX_GLARE_DEGREES)
    };
    let edge = veil(degrees);
    let colour = s.radiance / s.radiance.max_element().max(1e-6);
    let peak = veil(0.0) - edge;

    let core_degrees = if peak <= HALO_CAP {
        0.0
    } else {
        let (mut lo, mut hi) = (0.0, degrees);
        for _ in 0..20 {
            let mid = 0.5 * (lo + hi);
            if veil(mid) - edge > HALO_CAP {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        hi
    };
    Some(GlareQuad { half: d * degrees.to_radians().tan(), colour, a, core2, degrees, edge, peak, core_degrees })
}

pub struct GlarePipeline {
    pub pipeline: RenderPipeline,
    pub core: RenderPipeline,
}

impl GlarePipeline {
    pub fn new_multisampled(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        probe_layout: &BindGroupLayout,
        samples: u32,
    ) -> Self {
        Self::new_with_view(device, format, uniform_layout, probe_layout, samples, crate::renderer::multiview::ViewMode::Mono)
    }

    pub fn new_multisampled_stereo(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        probe_layout: &BindGroupLayout,
        samples: u32,
    ) -> Self {
        Self::new_with_view(device, format, uniform_layout, probe_layout, samples, crate::renderer::multiview::ViewMode::Stereo)
    }

    fn new_with_view(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        probe_layout: &BindGroupLayout,
        samples: u32,
        view: crate::renderer::multiview::ViewMode,
    ) -> Self {
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("glare_shader"),
            source: ShaderSource::Wgsl(view.shader(glare_shader()).into()),
        });
        let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("glare_layout"),
            bind_group_layouts: &[Some(uniform_layout), Some(probe_layout)],
            immediate_size: 0,
        });

        let screen = BlendComponent { src_factor: BlendFactor::OneMinusDst, dst_factor: BlendFactor::One, operation: BlendOperation::Add };
        let build = |label: &str, fragment: &str, depth_compare: CompareFunction| {
            device.create_render_pipeline(&RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                vertex: VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    compilation_options: PipelineCompilationOptions::default(),
                    buffers: &[Some(GlareVertex::layout())],
                },
                fragment: Some(FragmentState {
                    module: &shader,
                    entry_point: Some(fragment),
                    compilation_options: PipelineCompilationOptions::default(),
                    targets: &[Some(ColorTargetState {
                        format,
                        blend: Some(BlendState { color: screen, alpha: BlendComponent::OVER }),
                        write_mask: ColorWrites::COLOR,
                    })],
                }),
                primitive: PrimitiveState {
                    topology: PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    front_face: FrontFace::Ccw,
                    polygon_mode: PolygonMode::Fill,
                    ..Default::default()
                },
                depth_stencil: Some(DepthStencilState {
                    format: TextureFormat::Depth32Float,
                    depth_write_enabled: Some(false),
                    depth_compare: Some(depth_compare),
                    stencil: StencilState::default(),
                    bias: DepthBiasState::default(),
                }),
                multisample: MultisampleState { count: samples, ..Default::default() },
                multiview_mask: view.mask(),
                cache: None,
            })
        };
        Self {
            pipeline: build("glare_pipeline", "fs_main", CompareFunction::Always),
            core: build("glare_core_pipeline", "fs_core", CompareFunction::LessEqual),
        }
    }
}

pub fn glare_shader() -> String {
    format!(
        "{}{}{}",
        crate::renderer::tonemap::wgsl_aces_block(),
        format!(
            "const CIE_CUBE: f32 = {CIE_CUBE:?};\nconst CIE_SQUARE: f32 = {CIE_SQUARE:?};\nconst CIE_LINEAR: f32 = {CIE_LINEAR:?};\nconst HALO_CAP: f32 = {HALO_CAP:?};\n"
        ),
        r#"
var<private> view_slot: i32 = 0;
struct Camera { view_proj: array<mat4x4<f32>, 2> }
@group(0) @binding(0) var<uniform> camera: Camera;
@group(1) @binding(1) var probe_pass_depth: texture_depth_2d_array;
struct VIn {
    @location(0) pos: vec3<f32>,
    @location(1) colour: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) shape: vec4<f32>,
    @location(4) test: vec4<f32>,
    @location(5) reach: vec4<f32>,
}
struct VOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) colour: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) @interpolate(flat) shape: vec4<f32>,
    @location(3) air: f32,
    @location(4) @interpolate(flat) cap: f32,
}

fn glare_bulb_visible(centre: vec3<f32>, reach: vec3<f32>, test: f32) -> f32 {
    let c = camera.view_proj[view_slot] * vec4<f32>(centre, 1.0);
    let e = camera.view_proj[view_slot] * vec4<f32>(centre + reach, 1.0);
    if (test < 0.0 || c.w <= 0.0 || e.w <= 0.0) {
        return 1.0;
    }
    let ndc = c.xyz / c.w;
    let radius = length(e.xy / e.w - ndc.xy);
    let size = vec2<f32>(textureDimensions(probe_pass_depth));
    let hi = vec2<i32>(size) - vec2<i32>(1);
    var open = 0.0;
    for (var i = 0; i < 7; i = i + 1) {
        var offset = vec2<f32>(0.0);
        if (i > 0) {
            let a = f32(i) * 1.0471976;
            offset = vec2<f32>(cos(a), sin(a)) * radius * 0.66;
        }
        let p = ndc.xy + offset;
        let texel = clamp(vec2<i32>(vec2<f32>(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5) * size), vec2<i32>(0), hi);
        let wall = textureLoad(probe_pass_depth, texel, view_slot, 0);
        open = open + select(0.0, 1.0, wall >= ndc.z);
    }
    return open / 7.0;
}

fn glare_before_wall(clip: vec4<f32>, lamp: vec3<f32>) -> f32 {
    let front = camera.view_proj[view_slot] * vec4<f32>(lamp, 1.0);
    if (clip.w <= 0.0 || front.w <= 0.0) {
        return clip.z;
    }
    let ndc = clip.xyz / clip.w;
    let size = vec2<f32>(textureDimensions(probe_pass_depth));
    let texel = clamp(vec2<i32>(vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5) * size), vec2<i32>(0), vec2<i32>(size) - vec2<i32>(1));
    let wall = textureLoad(probe_pass_depth, texel, view_slot, 0);
    let lamp_z = front.z / front.w;
    if (wall > lamp_z && wall < ndc.z) {
        return mix(wall, lamp_z, 0.25) * clip.w;
    }
    return clip.z;
}
@vertex fn vs_main(v: VIn) -> VOut {
    var out: VOut;
    out.clip = camera.view_proj[view_slot] * vec4<f32>(v.pos, 1.0);
    if (v.test.w > 1.5 && v.test.w < 2.5) {
        out.clip.z = glare_before_wall(out.clip, v.test.xyz);
    }
    out.colour = v.colour;
    out.uv = v.uv;

    let open = glare_bulb_visible(v.test.xyz, v.reach.xyz, v.test.w);
    out.shape = vec4<f32>(v.shape.x * open, v.shape.y, v.shape.z, v.shape.w * open);

    if (open <= 0.0) {
        out.clip = vec4<f32>(2.0, 2.0, 2.0, 1.0);
    }
    out.air = v.reach.w;

    out.cap = select(HALO_CAP, 1.0e30, abs(v.test.w) > 2.5);
    return out;
}

fn glare_veil(in: VOut) -> f32 {
    let theta2 = dot(in.uv, in.uv) * in.shape.z + in.shape.y;
    let inv = inverseSqrt(max(theta2, 1e-6));
    return max(in.shape.x * inv * (CIE_LINEAR + inv * (CIE_SQUARE + inv * CIE_CUBE)) - in.shape.w, 0.0) * in.air;
}

@fragment fn fs_main(in: VOut) -> @location(0) vec4<f32> {
    return vec4<f32>(aces_fitted(in.colour * min(glare_veil(in), in.cap)), 0.0);
}

@fragment fn fs_core(in: VOut) -> @location(0) vec4<f32> {
    let veil = glare_veil(in);
    let whole = aces_fitted(in.colour * veil);
    let halo = aces_fitted(in.colour * min(veil, HALO_CAP));
    let one = vec3<f32>(1.0);
    return vec4<f32>(max(one - (one - whole) / (one - halo), vec3<f32>(0.0)), 0.0);
}
"#
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sconce(sides: [f32; 6]) -> GlareSource {
        GlareSource {
            position: Vec3::new(0.0, 2.0, 0.0),
            radiance: Vec3::new(1.0, 0.9, 0.8) * 3.0,
            sides,
            rotation: Quat::IDENTITY,
            cone: None,
            centres: None,
            table: None,
            halo_only: false,
            mirror: None,
        }
    }

    #[test]
    fn a_veil_grows_from_where_the_light_shows() {
        let bulb = Vec3::new(0.0, 2.0, 0.0);
        let mouth = Vec3::new(0.0, 1.85, 0.0);
        let mut centres = [bulb; 6];
        centres[3] = mouth;
        let s = GlareSource { centres: Some(centres), ..sconce([0.0, 0.0, 0.0, 0.5, 0.0, 0.0]) };
        assert!((visible_centre(&s, Vec3::new(0.0, 0.0, 0.0)) - mouth).length() < 1e-5);
        assert!((visible_centre(&s, Vec3::new(3.0, 2.0, 0.0)) - bulb).length() < 1e-5, "a dark side pulls nothing");
        let mut both = centres;
        both[0] = Vec3::new(0.1, 2.0, 0.0);
        let s2 = GlareSource { centres: Some(both), ..sconce([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]) };
        let c = visible_centre(&s2, bulb + Vec3::new(1.0, -1.0, 0.0));
        assert!((c - (both[0] + mouth) * 0.5).length() < 1e-5, "{c}");

        assert_eq!(visible_centre(&sconce([1.0; 6]), Vec3::ZERO), bulb);
    }

    fn mouth_down_table() -> GlareTable {
        let (rows, cols) = (18, 36);
        let mut t = GlareTable {
            rows,
            cols,
            share: vec![0.0; rows * cols],
            centre: vec![Vec3::ZERO; rows * cols],
            split: None,
            air: None,
        };
        for row in 0..rows {
            for col in 0..cols {
                if t.direction(row, col).y < -(30.0f32).to_radians().cos() {
                    t.share[row * cols + col] = 1.0;
                    t.centre[row * cols + col] = Vec3::new(0.0, -0.1, 0.0);
                }
            }
        }
        t
    }

    #[test]
    fn a_glare_table_shows_only_what_the_fixture_shows() {
        let origin = Vec3::new(0.0, 2.0, 0.0);
        let s = GlareSource {
            table: Some((std::sync::Arc::new(mouth_down_table()), origin)),
            ..sconce([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        };
        let off_axis = |deg: f32| {
            origin + Vec3::new(deg.to_radians().sin(), -deg.to_radians().cos(), 0.0) * 3.0
        };
        assert!(visible_share(&s, off_axis(0.0)) > 0.99, "straight below");
        assert_eq!(
            visible_share(&s, off_axis(60.0)),
            0.0,
            "60 degrees off, behind the shade"
        );
        let blended = GlareSource {
            table: None,
            ..s.clone()
        };
        assert!(
            visible_share(&blended, off_axis(60.0)) > 0.2,
            "the six sides' blend leaked there"
        );
        assert!((visible_centre(&s, off_axis(10.0)) - Vec3::new(0.0, 1.9, 0.0)).length() < 1e-5);

        let over = GlareSource {
            rotation: Quat::from_rotation_x(std::f32::consts::PI),
            ..s.clone()
        };
        assert!(visible_share(&over, origin + Vec3::Y * 3.0) > 0.99);
        assert!(
            (visible_centre(&over, origin + Vec3::Y * 3.0) - Vec3::new(0.0, 2.1, 0.0)).length()
                < 1e-5
        );

        assert_eq!(visible_centre(&s, origin + Vec3::Y * 3.0), s.position);
        assert!(glare_lobes(&s, origin + Vec3::Y * 3.0).is_empty());
    }

    fn quad(s: &GlareSource, eye: Vec3, exposure: f32) -> Option<GlareQuad> {
        let lobes = glare_lobes(s, eye);
        assert!(lobes.len() <= 1, "{lobes:?}");
        glare_quad(s, lobes.first()?, eye, exposure, 1.0)
    }

    #[test]
    fn a_table_is_read_from_where_its_fixture_glows() {
        let (rows, cols) = (18, 36);
        let glow = Vec3::new(0.0, -0.5, 0.0);
        let mut t = GlareTable { rows, cols, share: vec![0.0; rows * cols], centre: vec![glow; rows * cols], split: None, air: None };
        for row in 0..rows {
            for col in 0..cols {
                if t.direction(row, col).y < -(60.0f32).to_radians().cos() {
                    t.share[row * cols + col] = 1.0;
                }
            }
        }
        assert!((t.glow_middle() - glow).length() < 1e-5);
        let origin = Vec3::new(0.0, 3.0, 0.0);
        let s = GlareSource { position: origin + glow, table: Some((std::sync::Arc::new(t), origin)), ..sconce([0.0; 6]) };
        let beside = origin + glow + Vec3::new(1.0, -0.3, 0.0);
        assert!(s.table.as_ref().unwrap().0.sample(beside - origin).0 > 0.99, "from the box's middle it showed");
        assert_eq!(visible_share(&s, beside), 0.0, "from the bulb the shade hides it");
        assert!(glare_lobes(&s, beside).is_empty());
        assert!(visible_share(&s, origin + glow - Vec3::Y * 2.0) > 0.99, "straight below");
    }

    #[test]
    fn a_veil_shows_only_in_the_air_its_bulb_lights() {
        let (rows, cols, n) = (18, 36, 5);
        let air = GlareAir {
            min: Vec3::splat(-0.5),
            max: Vec3::splat(0.5),
            dims: [n; 3],
            bulb: Vec3::ZERO,

            seen: (0..n * n * n).map(|i| if (i / n) % n < 2 { 1.0 } else { 0.0 }).collect(),
            reach: 0.0,
        };
        assert_eq!(air.seen_from(Vec3::new(0.3, -0.5, -0.2)), 1.0);
        assert_eq!(air.seen_from(Vec3::new(0.0, 0.2, 9.0)), 0.0, "past the edge, the edge's");
        assert!((air.seen_from(Vec3::new(0.1, -0.125, 0.0)) - 0.5).abs() < 1e-5, "between points, between them");
        let origin = Vec3::new(0.0, 2.0, 0.0);
        let table = GlareTable { rows, cols, share: vec![1.0; rows * cols], centre: vec![Vec3::ZERO; rows * cols], split: None, air: Some(air) };
        let s = GlareSource {
            position: origin,
            radiance: Vec3::splat(100.0),
            table: Some((std::sync::Arc::new(table.clone()), origin)),
            ..sconce([0.0; 6])
        };
        let eye = Vec3::new(0.0, 1.0, 2.0);
        let build = |s: &GlareSource| build_glare(std::slice::from_ref(s), [eye; 2], Vec3::X, Vec3::Y, 1.0, 1.0, false, &[], &[]);
        let (verts, idx, halos) = build(&s);
        let side = AIR_CELLS + 1;

        assert!(halos > 0 && (halos as usize) < 6 * AIR_CELLS * AIR_CELLS && halos % 6 == 0, "{halos}");
        assert!(idx.iter().all(|&i| (i as usize) < verts.len()) && verts.len() >= side * side);

        let row = |j: usize| (0..side).map(|i| verts[j * side + i].reach[3]).collect::<Vec<f32>>();
        assert!(row(0).iter().all(|&a| a > 0.99), "below the bulb the air is lit: {:?}", row(0));
        assert!(row(side - 1).iter().all(|&a| a < 0.01), "above it, dark: {:?}", row(side - 1));

        let plain = GlareSource { table: Some((std::sync::Arc::new(GlareTable { air: None, ..table }), origin)), ..s.clone() };
        let (verts, _, halos) = build(&plain);
        assert_eq!(halos, 6);
        assert!(verts.iter().all(|v| v.reach[3] == 1.0));
    }

    #[test]
    fn a_cell_that_shows_nothing_is_not_drawn() {
        let (rows, cols, n) = (18, 36, 5);
        let air = GlareAir {
            min: Vec3::splat(-0.5),
            max: Vec3::splat(0.5),
            dims: [n; 3],
            bulb: Vec3::ZERO,

            seen: (0..n * n * n).map(|i| if (i / n) % n < 2 { 1.0 } else { 0.0 }).collect(),
            reach: 0.0,
        };
        let origin = Vec3::new(0.0, 2.0, 0.0);
        let table = GlareTable { rows, cols, share: vec![1.0; rows * cols], centre: vec![Vec3::ZERO; rows * cols], split: None, air: Some(air) };
        let s = GlareSource {
            position: origin,
            radiance: Vec3::splat(100.0),
            table: Some((std::sync::Arc::new(table.clone()), origin)),
            ..sconce([0.0; 6])
        };
        let eye = Vec3::new(0.0, 1.0, 2.0);
        let (verts, idx, halos) = build_glare(std::slice::from_ref(&s), [eye; 2], Vec3::X, Vec3::Y, 1.0, 1.0, false, &[], &[]);
        let side = AIR_CELLS + 1;

        assert_eq!(verts.len(), 2 * side * side, "a halo and a core");
        let drawn: std::collections::HashSet<usize> = idx.chunks(6).map(|c| c[0] as usize).collect();
        assert!(idx[..halos as usize].chunks(6).all(|c| (c[0] as usize) < side * side), "halo cells first");
        let (mut left_out, mut kept) = ([0; 2], [0; 2]);
        for quad in 0..2 {
            for j in 0..AIR_CELLS {
                for i in 0..AIR_CELLS {
                    let a = quad * side * side + j * side + i;
                    if drawn.contains(&a) {
                        kept[quad] += 1;
                        continue;
                    }
                    left_out[quad] += 1;
                    let (c00, c10, c11, c01) = (&verts[a], &verts[a + 1], &verts[a + 1 + side], &verts[a + side]);
                    let shape = c00.shape;
                    for (si, ti) in (0..9).flat_map(|si| (0..9).map(move |ti| (si, ti))) {
                        let (s, t) = (si as f32 / 8.0, ti as f32 / 8.0);
                        let air = if s >= t {
                            c00.reach[3] + s * (c10.reach[3] - c00.reach[3]) + t * (c11.reach[3] - c10.reach[3])
                        } else {
                            c00.reach[3] + t * (c01.reach[3] - c00.reach[3]) + s * (c11.reach[3] - c01.reach[3])
                        };
                        let u = c00.uv[0] + s * (c10.uv[0] - c00.uv[0]);
                        let v = c00.uv[1] + t * (c01.uv[1] - c00.uv[1]);
                        let veil = (shape[0] * cie_veil((u * u + v * v) * shape[2] + shape[1]) - shape[3]).max(0.0) * air;
                        let most = if quad == 0 { 0.0 } else { HALO_CAP };
                        assert!(veil <= most + 1e-6, "quad {quad} cell {i},{j} left out shows {veil} at {s},{t}");
                    }
                }
            }
        }
        assert!(left_out.iter().all(|&n| n > 0) && kept.iter().all(|&n| n > 0), "left out {left_out:?}, kept {kept:?}");

        let plain = GlareSource { table: Some((std::sync::Arc::new(GlareTable { air: None, ..table }), origin)), ..s };
        let (_, idx, halos) = build_glare(std::slice::from_ref(&plain), [eye; 2], Vec3::X, Vec3::Y, 1.0, 1.0, false, &[], &[]);
        assert_eq!((halos, idx.len()), (6, 12));
    }

    #[test]
    fn a_fixture_the_eye_is_adapted_to_takes_its_core_only_behind_it() {
        let (rows, cols, n) = (18, 36, 5);
        let reach = 0.4;
        let bulb_at = Vec3::new(0.0, -0.1, 0.0);
        let air = GlareAir { min: Vec3::splat(-0.5), max: Vec3::splat(0.5), dims: [n; 3], bulb: bulb_at, seen: vec![1.0; n * n * n], reach };
        let origin = Vec3::new(0.0, 2.0, 0.0);
        let table = GlareTable { rows, cols, share: vec![1.0; rows * cols], centre: vec![bulb_at; rows * cols], split: None, air: Some(air.clone()) };
        let s = GlareSource { position: origin, radiance: Vec3::splat(100.0), table: Some((std::sync::Arc::new(table.clone()), origin)), ..sconce([0.0; 6]) };

        let eye = Vec3::new(0.0, 1.0, 2.0);
        let bulb = origin + bulb_at;
        let ahead = (bulb - eye).normalize();
        let right = Vec3::X;
        let up = right.cross(ahead).normalize();
        let build = |s: &GlareSource, adapted: &[bool], walls: bool| {
            build_glare(std::slice::from_ref(s), [eye; 2], right, up, 3.6, 1.0, walls, &[], adapted)
        };
        let depth = |p: [f32; 3]| (Vec3::from(p) - eye).dot(ahead);
        let (verts, idx, halos) = build(&s, &[true], true);
        let (halo, core) = idx.split_at(halos as usize);
        assert!(!core.is_empty(), "bright enough for a core");
        for &i in core {
            let v = &verts[i as usize];
            assert!(depth(v.position) >= depth(bulb.to_array()) + reach - 1e-4, "behind the fixture: {}", depth(v.position));
            assert_eq!(v.test[3], 2.0, "brought forward of a nearer wall");
        }
        for &i in halo {
            let v = &verts[i as usize];
            assert!(depth(v.position) < depth(bulb.to_array()) && v.test[3] == 1.0, "the halo in front of the light");
        }

        let (plain_verts, plain_idx, plain_halos) = build(&s, &[false], true);
        assert_eq!((plain_idx.len(), plain_halos), (idx.len(), halos));
        for (&i, &j) in core.iter().zip(&plain_idx[plain_halos as usize..]) {
            let (v, p) = (&verts[i as usize], &plain_verts[j as usize]);
            let along = |q: [f32; 3]| (Vec3::from(q) - eye).normalize();
            assert!((along(v.position) - along(p.position)).length() < 1e-4, "the same view");
            assert!(depth(p.position) < depth(bulb.to_array()) && p.test[3] == 1.0, "in front of the light");
        }

        let (verts, idx, halos) = build(&s, &[true], false);
        assert!(idx[halos as usize..].iter().all(|&i| verts[i as usize].test[3] == -1.0));

        let unmeasured = GlareTable { air: Some(GlareAir { reach: 0.0, ..air }), ..table };
        let s = GlareSource { table: Some((std::sync::Arc::new(unmeasured), origin)), ..s.clone() };
        let (verts, idx, halos) = build(&s, &[true], true);
        assert!(idx[halos as usize..].iter().all(|&i| depth(verts[i as usize].position) < depth(bulb.to_array())));
    }

    #[test]
    fn a_halo_alone_has_no_core_and_no_cap() {
        let torch = GlareSource { halo_only: true, ..sconce([1.0; 6]) };
        let (verts, idx, halos) = build_glare(&[torch.clone()], [Vec3::ZERO; 2], Vec3::X, Vec3::Z, 3.6, 1.0, true, &[], &[]);
        assert_eq!(halos as usize, idx.len(), "no core");
        assert!(halos > 0 && verts.iter().all(|v| v.test[3] == 3.0));
        let (verts, _, _) = build_glare(&[torch], [Vec3::ZERO; 2], Vec3::X, Vec3::Z, 3.6, 1.0, false, &[], &[]);
        assert!(verts.iter().all(|v| v.test[3] == -3.0), "no probe pass, no wall test");

        let (_, idx, halos) = build_glare(&[sconce([1.0; 6])], [Vec3::ZERO; 2], Vec3::X, Vec3::Z, 3.6, 1.0, true, &[], &[]);
        assert!((halos as usize) < idx.len());
    }

    #[test]
    fn a_bulbs_finer_rows_cut_its_veil_off_where_the_shade_does() {
        let mut t = mouth_down_table();
        let n = t.rows * t.cols;
        let fine_rows = 180;

        let showing = |r: usize| match r as f32 + 0.5 {
            d if d > 154.0 => 1.0,
            d if d > 152.0 => 1.0 / 3.0,
            _ => 0.0,
        };
        let fine = GlareBulbRows {
            rows: fine_rows,
            share: (0..fine_rows * t.cols).map(|i| showing(i / t.cols)).collect(),
            centre: (0..fine_rows * t.cols)
                .map(|i| if showing(i / t.cols) < 1.0 { Vec3::new(0.0, -0.14, 0.0) } else { Vec3::new(0.0, -0.1, 0.0) })
                .collect(),
            whole: 1.0,
        };
        let split = GlareTableSplit {
            bulb: t.share.clone(),
            bulb_centre: vec![Vec3::new(0.0, -0.1, 0.0); n],
            lit_centre: vec![Vec3::ZERO; n],
            lit_spread: vec![0.0; n],
            bulb_fine: None,
        };
        let from_below = |deg: f32| Vec3::new(deg.to_radians().sin(), -deg.to_radians().cos(), 0.0);
        t.split = Some(split.clone());
        let coarse = t.sample_split(from_below(30.0)).unwrap()[0].share;
        assert!(coarse > 0.4, "the table's rows read between: {coarse}");
        t.split = Some(GlareTableSplit { bulb_fine: Some(fine), ..split });
        assert_eq!(t.sample_split(from_below(30.0)).unwrap()[0].share, 0.0, "the shade hides it");
        let seen = t.sample_split(from_below(20.0)).unwrap()[0];
        assert!((seen.share - 1.0).abs() < 1e-5 && (seen.centre - Vec3::new(0.0, -0.1, 0.0)).length() < 1e-5, "{seen:?}");
        assert_eq!(seen.radius, LAMP_RADIUS, "all of it shows");

        let sliver = t.sample_split(from_below(27.0)).unwrap()[0];
        assert!((sliver.share - 1.0 / 3.0).abs() < 1e-5, "{sliver:?}");
        assert!((sliver.centre - Vec3::new(0.0, -0.14, 0.0)).length() < 1e-5, "{sliver:?}");
        assert!((sliver.radius - LAMP_RADIUS * (1.0f32 / 3.0).sqrt()).abs() < 1e-5, "{sliver:?}");

        let origin = Vec3::new(0.0, 3.0, 0.0);
        let s = GlareSource { position: origin, table: Some((std::sync::Arc::new(t.clone()), origin)), ..sconce([0.0; 6]) };
        let middle = origin + t.glow_middle();
        assert!((bulb_in_view(&s, middle + from_below(20.0) * 3.0) - 1.0).abs() < 1e-5);
        assert_eq!(bulb_in_view(&s, middle + from_below(30.0) * 3.0), 0.0);
        assert!((bulb_in_view(&s, middle + from_below(27.0) * 3.0) - 1.0 / 3.0).abs() < 1e-5);
    }

    #[test]
    fn a_split_table_glares_from_its_bulb_and_from_its_lit_inside() {
        let mut t = mouth_down_table();
        let n = t.rows * t.cols;
        let (bulb_at, lit_at) = (Vec3::new(0.0, -0.1, 0.02), Vec3::new(0.0, -0.05, -0.04));
        t.split = Some(GlareTableSplit {
            bulb: t.share.iter().map(|s| 0.75 * s).collect(),
            bulb_centre: vec![bulb_at; n],
            lit_centre: vec![lit_at; n],
            lit_spread: vec![0.1; n],
            bulb_fine: None,
        });
        let origin = Vec3::new(0.0, 2.0, 0.0);
        let turn = Quat::from_rotation_y(0.5);
        let s = GlareSource { rotation: turn, table: Some((std::sync::Arc::new(t), origin)), ..sconce([0.0; 6]) };
        let below = origin - Vec3::Y * 3.0;
        let lobes = glare_lobes(&s, below);
        assert_eq!(lobes.len(), 2, "{lobes:?}");
        let (bulb, lit) = (lobes[0], lobes[1]);
        assert!((bulb.share - 0.75).abs() < 1e-4 && (lit.share - 0.25).abs() < 1e-4, "{lobes:?}");
        assert!((bulb.centre - (origin + turn * bulb_at)).length() < 1e-5, "{lobes:?}");
        assert!((lit.centre - (origin + turn * lit_at)).length() < 1e-5, "{lobes:?}");
        assert_eq!(bulb.radius, LAMP_RADIUS);
        assert!((lit.radius - 0.1 * std::f32::consts::SQRT_2).abs() < 1e-5, "{lit:?}");

        let (b, l) = (glare_quad(&s, &bulb, below, 3.6, 1.0).unwrap(), glare_quad(&s, &lit, below, 3.6, 1.0).unwrap());
        assert!(l.core2 > b.core2 && l.peak < b.peak, "{b:?} {l:?}");
        let (v, i, halos) = build_glare(&[s.clone()], [below; 2], Vec3::X, Vec3::Z, 3.6, 1.0, true, &[], &[]);
        assert_eq!(halos, 12, "a halo for each part");

        let reach = |k: usize| Vec3::from_slice(&v[i[6 * k] as usize].reach[..3]);
        assert!((reach(0) - Vec3::X * LAMP_RADIUS).length() < 1e-5, "{}", reach(0));
        assert!((reach(1) - Vec3::X * lit.radius).length() < 1e-5, "{}", reach(1));

        assert!(glare_lobes(&s, origin + Vec3::Y * 3.0).is_empty());
    }

    #[test]
    fn a_glare_table_reads_between_its_entries_and_wraps_round() {
        let (rows, cols) = (4, 8);
        let mut t = GlareTable {
            rows,
            cols,
            share: vec![0.0; rows * cols],
            centre: vec![Vec3::ZERO; rows * cols],
            split: None,
            air: None,
        };
        for row in 0..rows {
            t.share[row * cols] = 1.0;
            t.share[row * cols + cols - 1] = 3.0;
            t.centre[row * cols] = Vec3::X;
            t.centre[row * cols + cols - 1] = Vec3::Z;
        }

        let at = |row: usize, col: usize| t.sample(t.direction(row, col));
        assert!((at(1, 0).0 - 1.0).abs() < 1e-4 && (at(1, cols - 1).0 - 3.0).abs() < 1e-4);

        let between = t.sample(Vec3::new(1.0, 0.1, 0.0));
        assert!((between.0 - 2.0).abs() < 1e-3, "{between:?}");

        assert!(
            (between.1 - (Vec3::X + 3.0 * Vec3::Z) / 4.0).length() < 1e-3,
            "{between:?}"
        );
    }

    #[test]
    fn the_lamp_radius_is_the_lightings() {
        let wgsl = include_str!("lights.rs");
        assert!(wgsl.contains(&format!("const LAMP_RADIUS: f32 = {LAMP_RADIUS:?};")), "lights block's LAMP_RADIUS moved");
    }

    #[test]
    fn a_bulb_glares_only_from_the_sides_it_shows_from() {
        let below_only = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let s = sconce(below_only);
        assert!(visible_share(&s, Vec3::new(0.0, 0.0, 0.0)) > 0.99, "straight below");
        assert!(visible_share(&s, Vec3::new(0.0, 4.0, 0.0)) < 0.01, "straight above");
        let slant = visible_share(&s, Vec3::new(1.0, 1.0, 0.0));
        assert!(slant > 0.4 && slant < 0.6, "45 degrees below: half the axis weight: {slant}");
        let upside = GlareSource { rotation: Quat::from_rotation_x(std::f32::consts::PI), ..s };
        assert!(visible_share(&upside, Vec3::new(0.0, 4.0, 0.0)) > 0.99, "turned over, it shows upward");
    }

    #[test]
    fn a_spots_bulb_shows_from_inside_its_beam() {
        let cos = |deg: f32| deg.to_radians().cos();
        let s = GlareSource { cone: Some((Vec3::NEG_Y, cos(30.0), cos(20.0))), ..sconce([1.0; 6]) };
        assert!(visible_share(&s, Vec3::new(0.0, 0.0, 0.0)) > 0.99, "on the axis");
        assert_eq!(visible_share(&s, Vec3::new(3.0, 2.0, 0.0)), 0.0, "beside the beam");
    }

    #[test]
    fn the_veil_follows_the_light_at_the_eye() {
        let s = sconce([1.0; 6]);
        let near = quad(&s, Vec3::new(0.0, 0.0, 0.0), 3.6).unwrap();
        let far = quad(&s, Vec3::new(0.0, -2.0, 0.0), 3.6).unwrap();

        assert!((far.a / near.a - 0.25).abs() < 1e-4, "{} vs {}", far.a, near.a);
        assert!((far.core2 / near.core2 - 0.25).abs() < 0.01);
        let brighter = quad(&s, Vec3::new(0.0, 0.0, 0.0), 7.2).unwrap();
        assert!((brighter.a / near.a - 2.0).abs() < 1e-4);

        let (dim, less_dim) = (quad(&s, Vec3::ZERO, 0.5).unwrap(), quad(&s, Vec3::ZERO, 1.0).unwrap());
        assert!(dim.degrees < less_dim.degrees && less_dim.degrees < MAX_GLARE_DEGREES, "{dim:?} {less_dim:?}");
        assert!(quad(&s, Vec3::new(0.0, -200.0, 0.0), 0.5).is_none(), "a faint lamp far off");
        assert!(quad(&sconce([0.0; 6]), Vec3::ZERO, 3.6).is_none(), "a bulb that shows from nowhere");
    }

    #[test]
    fn the_quad_reaches_to_the_faintest_veil_drawn() {
        let s = sconce([1.0; 6]);
        let q = quad(&s, Vec3::ZERO, 3.6).unwrap();
        assert!(q.degrees <= MAX_GLARE_DEGREES && q.degrees > 2.0, "{q:?}");
        assert!((q.half - 2.0 * q.degrees.to_radians().tan()).abs() < 1e-4);

        assert!((q.edge - VEIL_FLOOR).abs() < 0.01 * VEIL_FLOOR || q.degrees == MAX_GLARE_DEGREES, "{q:?}");
        assert!((q.a * cie_veil(q.degrees * q.degrees + q.core2) - q.edge).abs() < 1e-6);
        assert!((q.peak - (q.a * cie_veil(q.core2) - q.edge)).abs() < 1e-3 * q.peak);

        let blinding = quad(&s, Vec3::ZERO, 3600.0).unwrap();
        assert_eq!(blinding.degrees, MAX_GLARE_DEGREES);
        let (v, i, halos) = build_glare(&[s.clone()], [Vec3::ZERO; 2], Vec3::X, Vec3::Z, 3.6, 1.0, true, &[], &[]);
        assert_eq!((v.len(), i.len(), halos), (8, 12, 6), "a halo, then its core");

        let centre = Vec3::from_slice(&v[0].test[..3]);
        assert!((centre - Vec3::new(0.0, 2.0 - WALL_MARGIN, 0.0)).length() < 1e-5, "{centre}");
        let half = (Vec3::from(v[0].position) - centre).x.abs();
        assert!((half - (2.0 - WALL_MARGIN) * q.degrees.to_radians().tan()).abs() < 1e-4);
        assert!((Vec3::from_slice(&v[0].reach[..3]) - Vec3::X * LAMP_RADIUS).length() < 1e-5 && v[0].test[3] > 0.0);
        let (untested, _, _) = build_glare(&[s.clone()], [Vec3::ZERO; 2], Vec3::X, Vec3::Z, 3.6, 1.0, false, &[], &[]);
        assert!(untested.iter().all(|v| v.test[3] < 0.0), "no probe pass, no wall test");
    }

    #[test]
    fn a_veils_core_reaches_to_the_halo_cap_from_in_front_of_what_glows() {
        let s = sconce([1.0; 6]);
        let q = quad(&s, Vec3::ZERO, 3.6).unwrap();
        let after_edge = |q: &GlareQuad, deg: f32| q.a * cie_veil(deg * deg + q.core2) - q.edge;
        assert!(q.core_degrees > 0.0 && q.core_degrees < q.degrees, "{q:?}");
        assert!((after_edge(&q, q.core_degrees) - HALO_CAP).abs() < 1e-4, "{}", after_edge(&q, q.core_degrees));
        let (v, i, halos) = build_glare(&[s.clone()], [Vec3::ZERO; 2], Vec3::X, Vec3::Z, 3.6, 1.0, true, &[], &[]);
        let corners = |at: usize| -> Vec<Vec3> { (0..4).map(|k| Vec3::from(v[i[at] as usize + k].position)).collect() };
        let middle = |c: &[Vec3]| c.iter().copied().sum::<Vec3>() / 4.0;
        let (halo, core) = (corners(0), corners(halos as usize));
        let tested = Vec3::from_slice(&v[i[0] as usize].test[..3]);
        assert_eq!(v[i[halos as usize] as usize].test[..3], v[i[0] as usize].test[..3], "one point tested for walls");

        assert!((middle(&core) - tested).length() < 1e-5 && (middle(&halo) - tested).length() < 1e-5);
        let core_half = (core[0] - middle(&core)).x.abs();
        assert!((core_half - (2.0 - WALL_MARGIN) * q.core_degrees.to_radians().tan()).abs() < 1e-5);
        let (core_v, halo_v) = (&v[i[halos as usize] as usize], &v[i[0] as usize]);
        assert!(core_v.reach == halo_v.reach && core_v.test == halo_v.test, "the same bulb for the walls");
        assert!((Vec3::from_slice(&core_v.reach[..3]).length() - LAMP_RADIUS).abs() < 1e-5);

        let bright = q.a * cie_veil(q.core2);
        let dim = quad(&s, Vec3::ZERO, 3.6 * (VEIL_FLOOR + 0.5 * HALO_CAP) / bright).unwrap();
        assert!(dim.peak > 0.0 && dim.peak <= HALO_CAP && dim.core_degrees == 0.0, "{dim:?}");
        let exposure = 3.6 * (VEIL_FLOOR + 0.5 * HALO_CAP) / bright;
        let (_, i, halos) = build_glare(&[s.clone()], [Vec3::ZERO; 2], Vec3::X, Vec3::Z, exposure, 1.0, true, &[], &[]);
        assert_eq!(halos as usize, i.len(), "no core");

        let mut t = mouth_down_table();
        let n = t.rows * t.cols;
        let lit_at = Vec3::new(0.0, -0.05, -0.04);
        t.split = Some(GlareTableSplit {
            bulb: t.share.iter().map(|s| 0.75 * s).collect(),
            bulb_centre: vec![Vec3::new(0.0, -0.1, 0.02); n],
            lit_centre: vec![lit_at; n],
            lit_spread: vec![0.1; n],
            bulb_fine: None,
        });
        let origin = Vec3::new(0.0, 2.0, 0.0);
        let s = GlareSource { table: Some((std::sync::Arc::new(t), origin)), ..sconce([0.0; 6]) };
        let below = origin - Vec3::Y * 3.0;
        let lit = glare_lobes(&s, below)[1];
        let (v, i, halos) = build_glare(&[s.clone()], [below; 2], Vec3::X, Vec3::Z, 36.0, 1.0, true, &[], &[]);
        assert_eq!(i.len() as u32, 2 * halos, "both parts have a core: {} indices, {halos} halo", i.len());
        let lit_core: Vec<Vec3> = (0..4).map(|k| Vec3::from(v[i[halos as usize + 6] as usize + k].position)).collect();
        let toward = (below - lit.centre).normalize();
        assert!((middle(&lit_core) - (lit.centre + toward * lit.radius)).length() < 1e-5, "{lit:?}");
    }

    #[test]
    fn an_image_in_a_mirror_is_tested_at_the_mirror() {
        let eye = Vec3::new(0.0, 1.6, 0.0);
        let glass = Vec3::new(0.0, 1.0, -1.0);
        let image = Vec3::new(glass.x, -glass.y, glass.z);

        let on_floor = eye + (image - eye) * (eye.y / (eye.y - image.y));
        let s = GlareSource { position: image, mirror: Some(on_floor), ..sconce([1.0; 6]) };
        let (verts, idx, halos) = build_glare(&[s.clone()], [eye; 2], Vec3::X, Vec3::Y, 3.6, 1.0, true, &[], &[]);
        assert_eq!(halos as usize, idx.len(), "no core: it would stand under the floor");
        assert!(!verts.is_empty() && verts.iter().all(|v| v.test[3] == 3.0));
        let tested = Vec3::new(verts[0].test[0], verts[0].test[1], verts[0].test[2]);
        assert!(tested.y > 0.0 && tested.y < 0.06, "just above the floor: {tested}");
        let sight = (image - eye).normalize();
        assert!(((tested - eye).normalize() - sight).length() < 1e-4, "on the line of sight to the image");

        let direct = GlareSource { mirror: None, halo_only: true, ..s.clone() };
        let (plain, _, _) = build_glare(&[direct], [eye; 2], Vec3::X, Vec3::Y, 3.6, 1.0, true, &[], &[]);
        let span = |v: &[GlareVertex]| {
            let (a, b) = (Vec3::from(v[0].position), Vec3::from(v[3].position));
            (a - eye).normalize().angle_between((b - eye).normalize())
        };
        assert!((span(&verts) - span(&plain)).abs() < 1e-3, "{} vs {}", span(&verts), span(&plain));

        let hand_at = |c: Vec3| (c - Vec3::X * 0.08, c + Vec3::X * 0.08, 0.045);
        let veil = |caps: &[(Vec3, Vec3, f32)]| build_glare(&[s.clone()], [eye; 2], Vec3::X, Vec3::Y, 3.6, 1.0, true, caps, &[]).0.first().map_or(0.0, |v| v.shape[0]);
        assert!(veil(&[hand_at(eye + (on_floor - eye) * 0.5)]) < 1e-3 * veil(&[]), "a hand over the floor point");
        assert_eq!(veil(&[hand_at(image + (on_floor - image) * 0.5)]), veil(&[]), "under the floor");
    }

    #[test]
    fn a_hand_raised_against_the_light_shields_the_eye() {
        let bulb = Vec3::new(0.0, 2.0, -3.0);
        let eye = Vec3::new(0.0, 1.6, 0.0);
        let to_bulb = (bulb - eye).normalize();
        let across = to_bulb.cross(Vec3::Y).normalize();

        let hand = |off: f32| {
            let c = eye + to_bulb * 0.3 + across * off;
            (c - Vec3::Y * 0.08, c + Vec3::Y * 0.08, 0.045)
        };
        assert!(capsule_visibility(bulb, eye, &[hand(0.0)]) < 1e-3, "over the bulb");
        assert!(capsule_visibility(bulb, eye, &[hand(0.2)]) > 0.999, "beside it");
        let edge = capsule_visibility(bulb, eye, &[hand(0.045)]);
        assert!(edge > 0.2 && edge < 0.8, "its edge across the bulb: {edge}");
        let head = (eye - Vec3::Y * 0.05, eye + Vec3::Y * 0.05, 0.1);
        assert_eq!(capsule_visibility(bulb, eye, &[head]), 1.0, "the eye's own head");

        let s = GlareSource { position: bulb, ..sconce([1.0; 6]) };
        let eyes = [eye - across * 0.032, eye + across * 0.032];
        let peak = |caps: &[(Vec3, Vec3, f32)]| build_glare(&[s.clone()], eyes, across, Vec3::Y, 3.6, 1.0, false, caps, &[]).0[0].shape[0];

        let one_eye = (eyes[0] + to_bulb * 0.3, eyes[0] + to_bulb * 0.3 + Vec3::Y * 0.01, 0.02);
        assert!((peak(&[one_eye]) / peak(&[]) - 0.5).abs() < 0.02, "{} vs {}", peak(&[one_eye]), peak(&[]));
    }

    #[test]
    fn a_hand_in_front_of_a_glass_hides_it_and_the_hand_holding_it_does_not() {
        let glass = Vec3::new(0.0, 1.2, 0.0);
        let mirror = Vec3::new(0.3, 1.0, -2.0);
        let to_mirror = (mirror - glass).normalize();
        let across = to_mirror.cross(Vec3::Y).normalize();
        let radius = 0.0168;

        let raised = |off: f32| {
            let c = glass + to_mirror * 0.25 + across * off;
            (c - Vec3::Y * 0.05, c + Vec3::Y * 0.05, 0.04)
        };
        assert!(disc_visibility(glass, radius, mirror, &[raised(0.0)]) < 1e-3, "over the line");
        assert!(disc_visibility(glass, radius, mirror, &[raised(0.2)]) > 0.999, "beside it");
        let edge = disc_visibility(glass, radius, mirror, &[raised(0.04)]);
        assert!(edge > 0.2 && edge < 0.8, "its edge across the glass: {edge}");
        let holding = (glass - to_mirror * 0.12, glass + to_mirror * 0.01, 0.04);
        assert_eq!(disc_visibility(glass, radius, mirror, &[holding]), 1.0, "the hand round the torch");
        assert_eq!(disc_visibility(glass, radius, mirror, &[holding, raised(0.0)]), 0.0, "both");
    }

    #[test]
    fn a_veil_is_cut_by_what_stands_in_front_and_goes_when_a_wall_hides_its_bulb() {
        let Some((device, queue)) = crate::renderer::terrain_pipeline::tests::headless_gpu() else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        const W: u32 = 64;
        let format = TextureFormat::Rgba8Unorm;
        let uniform_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: None,
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::VERTEX,
                ty: BindingType::Buffer { ty: BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            }],
        });
        let probe_layout = crate::renderer::brush_pipeline::probe_pass::bind_group_layout(&device);
        let glare = GlarePipeline::new_multisampled(&device, format, &uniform_layout, &probe_layout, 4);

        let one: [f32; 16] = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.5, 1.0];
        let view_proj = [one, one];
        let camera = wgpu::util::DeviceExt::create_buffer_init(
            &device,
            &wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&view_proj), usage: BufferUsages::UNIFORM },
        );
        let bind = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &uniform_layout,
            entries: &[BindGroupEntry { binding: 0, resource: camera.as_entire_binding() }],
        });

        let wall = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: None,
            layout: None,
            vertex: VertexState {
                module: &device.create_shader_module(ShaderModuleDescriptor {
                    label: None,
                    source: ShaderSource::Wgsl(
                        "struct W { x: f32, z: f32 }
                        @group(0) @binding(0) var<uniform> w: W;
                        @vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
                            let c = array<vec2<f32>, 6>(vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(1.0, 1.0), vec2(-1.0, -1.0), vec2(1.0, 1.0), vec2(-1.0, 1.0));
                            let x = mix(-1.0, w.x, c[i].x * 0.5 + 0.5);
                            return vec4<f32>(x, c[i].y, w.z, 1.0);
                        }"
                        .into(),
                    ),
                }),
                entry_point: Some("vs"),
                compilation_options: PipelineCompilationOptions::default(),
                buffers: &[],
            },
            fragment: None,
            primitive: PrimitiveState::default(),
            depth_stencil: Some(DepthStencilState {
                format: TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(CompareFunction::Always),
                stencil: StencilState::default(),
                bias: DepthBiasState::default(),
            }),
            multisample: MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        const CORE2: f32 = 0.25;
        const REACH2: f32 = 100.0;
        let edge = |a: f32| a * cie_veil(REACH2 + CORE2);

        let render_with = |scene_depth: f32, wall_x: f32, wall_z: f32, a: f32, under: f64, flag: f32| -> Vec<u8> {
            let quad = [(-1.0f32, -1.0f32), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)].map(|(u, v)| GlareVertex {
                position: [u, v, if flag == 2.0 { 1.0 } else { 0.0 }],
                colour: [1.0, 1.0, 1.0],
                uv: [u, v],
                shape: [a, CORE2, REACH2, edge(a)],

                test: [0.0, 0.0, 0.0, flag],
                reach: [0.5, 0.0, 0.0, 1.0],
            });
            let vb = wgpu::util::DeviceExt::create_buffer_init(
                &device,
                &wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&quad), usage: BufferUsages::VERTEX },
            );
            let ib = wgpu::util::DeviceExt::create_buffer_init(
                &device,
                &wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&[0u32, 1, 2, 0, 2, 3]), usage: BufferUsages::INDEX },
            );
            let probe = crate::renderer::brush_pipeline::probe_pass::Target::new(&device, &probe_layout, W, W, 1);
            let wall_x_buf = wgpu::util::DeviceExt::create_buffer_init(
                &device,
                &wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&[wall_x, wall_z, 0.0, 0.0]), usage: BufferUsages::UNIFORM },
            );
            let wall_bind = device.create_bind_group(&BindGroupDescriptor {
                label: None,
                layout: &wall.get_bind_group_layout(0),
                entries: &[BindGroupEntry { binding: 0, resource: wall_x_buf.as_entire_binding() }],
            });
            let target = |fmt: TextureFormat, samples: u32, usage: TextureUsages| {
                device.create_texture(&TextureDescriptor {
                    label: None,
                    size: Extent3d { width: W, height: W, depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: samples,
                    dimension: TextureDimension::D2,
                    format: fmt,
                    usage,
                    view_formats: &[],
                })
            };
            let msaa = target(format, 4, TextureUsages::RENDER_ATTACHMENT);
            let resolved = target(format, 1, TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC);
            let depth = target(TextureFormat::Depth32Float, 4, TextureUsages::RENDER_ATTACHMENT);
            let (msaa_v, resolved_v, depth_v) = (
                msaa.create_view(&Default::default()),
                resolved.create_view(&Default::default()),
                depth.create_view(&Default::default()),
            );
            let mut enc = device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_render_pass(&RenderPassDescriptor {
                    label: None,
                    color_attachments: &[],
                    depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                        view: &probe.depth_view,
                        depth_ops: Some(Operations { load: LoadOp::Clear(1.0), store: StoreOp::Store }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                if wall_x > -1.0 {
                    pass.set_pipeline(&wall);
                    pass.set_bind_group(0, &wall_bind, &[]);
                    pass.draw(0..6, 0..1);
                }
            }
            {
                let mut pass = enc.begin_render_pass(&RenderPassDescriptor {
                    label: None,
                    color_attachments: &[Some(RenderPassColorAttachment {
                        view: &msaa_v,
                        depth_slice: None,
                        resolve_target: Some(&resolved_v),
                        ops: Operations { load: LoadOp::Clear(Color { r: under, g: under, b: under, a: 1.0 }), store: StoreOp::Discard },
                    })],
                    depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                        view: &depth_v,
                        depth_ops: Some(Operations { load: LoadOp::Clear(scene_depth), store: StoreOp::Discard }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                for pipeline in [&glare.core, &glare.pipeline] {
                    pass.set_pipeline(pipeline);
                    pass.set_bind_group(0, &bind, &[]);
                    pass.set_bind_group(1, &probe.bind_group, &[]);
                    pass.set_vertex_buffer(0, vb.slice(..));
                    pass.set_index_buffer(ib.slice(..), IndexFormat::Uint32);
                    pass.draw_indexed(0..6, 0, 0..1);
                }
            }
            let read = device.create_buffer(&BufferDescriptor {
                label: None,
                size: (W * W * 4) as u64,
                usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            enc.copy_texture_to_buffer(
                TexelCopyTextureInfo { texture: &resolved, mip_level: 0, origin: Origin3d::ZERO, aspect: TextureAspect::All },
                TexelCopyBufferInfo {
                    buffer: &read,
                    layout: TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(W * 4), rows_per_image: Some(W) },
                },
                Extent3d { width: W, height: W, depth_or_array_layers: 1 },
            );
            queue.submit([enc.finish()]);
            read.slice(..).map_async(MapMode::Read, |_| {});
            let _ = device.poll(PollType::Wait { submission_index: None, timeout: None });
            let data = read.slice(..).get_mapped_range().expect("mapped").to_vec();
            data
        };
        let render = |scene_depth: f32, wall_x: f32, a: f32, under: f64| render_with(scene_depth, wall_x, 0.25, a, under, 1.0);
        let px = |img: &[u8], x: u32, y: u32| img[((y * W + x) * 4) as usize];

        let veil_at = |a: f32, share: f32, k: u32| {
            let u = ((W / 2 + k) as f32 + 0.5) / W as f32 * 2.0 - 1.0;
            let v = (W / 2) as f32 + 0.5;
            let v = v / W as f32 * 2.0 - 1.0;
            share * (a * cie_veil((u * u + v * v) * REACH2 + CORE2) - edge(a)).max(0.0)
        };
        let shown = |veil: f32| 255.0 * crate::renderer::tonemap::aces_fitted(Vec3::splat(veil)).x;
        let expected = |a: f32, share: f32, k: u32| shown(veil_at(a, share, k));
        let open = render(1.0, -1.0, 1.0, 0.0);
        let (centre, mid, rim) = (px(&open, W / 2, W / 2), px(&open, W / 2 + W / 8, W / 2), px(&open, W - 1, W / 2));
        assert!(centre > 240, "the core is white: {centre}");
        assert!(mid > 20 && mid < centre);
        assert_eq!(rim, 0, "nothing at the quad's edge");

        assert!(veil_at(1.0, 1.0, W / 2 - 2) < HALO_CAP && veil_at(1.0, 1.0, W / 8) > 4.0 * HALO_CAP);
        for k in 0..W / 2 {
            let (got, want) = (px(&open, W / 2 + k, W / 2) as f32, expected(1.0, 1.0, k));
            assert!((got - want).abs() <= 2.0, "{k} pixels out: {got} vs {want}");
        }

        let nearer = render(0.25, -1.0, 1.0, 0.0);
        assert!(shown(HALO_CAP) < 3.0, "the cap is a haze: {}", shown(HALO_CAP));
        for k in 0..W / 2 {
            let (got, want) = (px(&nearer, W / 2 + k, W / 2) as f32, shown(veil_at(1.0, 1.0, k).min(HALO_CAP)));
            assert!((got - want).abs() <= 1.0, "over the nearer surface {k} pixels out: {got} vs {want}");
        }

        let lit = 0.8f32;
        let over = render(1.0, -1.0, 1.0, lit as f64);
        for k in [0, W / 16, W / 8, W / 4] {
            let veil = crate::renderer::tonemap::aces_fitted(Vec3::splat(veil_at(1.0, 1.0, k))).x;
            let (got, want) = (px(&over, W / 2 + k, W / 2) as f32, 255.0 * (1.0 - (1.0 - lit) * (1.0 - veil)));
            assert!((got - want).abs() <= 3.0, "over the lit inside {k} pixels out: {got} vs {want}");
        }
        assert!(px(&over, W / 2 + W / 8, W / 2) < 250, "not burnt white where the veil alone shows two thirds of white");

        let fixture = render_with(0.6, -1.0, 0.25, 1.0, 0.0, 2.0);
        let room = render_with(1.0, -1.0, 0.25, 1.0, 0.0, 2.0);
        let ceiling = render_with(0.65, 1.0, 0.65, 1.0, 0.0, 2.0);
        let fixture_under_ceiling = render_with(0.6, 1.0, 0.65, 1.0, 0.0, 2.0);

        let torch = render_with(0.25, -1.0, 0.25, 1.0, 0.0, 3.0);
        for k in 0..W / 2 {
            let (got, want) = (px(&torch, W / 2 + k, W / 2) as f32, expected(1.0, 1.0, k));
            assert!((got - want).abs() <= 2.0, "a halo alone {k} pixels out: {got} vs {want}");
        }
        for k in 0..W / 2 {
            let want = expected(1.0, 1.0, k);
            for (img, what) in [(&room, "the room behind"), (&ceiling, "a ceiling nearer than the core")] {
                let got = px(img, W / 2 + k, W / 2) as f32;
                assert!((got - want).abs() <= 2.0, "over {what} {k} pixels out: {got} vs {want}");
            }
            let haze = shown(veil_at(1.0, 1.0, k).min(HALO_CAP));
            for (img, what) in [(&fixture, "the fixture"), (&fixture_under_ceiling, "the fixture under a ceiling")] {
                let got = px(img, W / 2 + k, W / 2) as f32;
                assert!((got - haze).abs() <= 1.0, "over {what} {k} pixels out: {got} vs {haze}");
            }
        }
        let walled = render(1.0, 1.0, 1.0, 0.0);
        assert_eq!(px(&walled, W / 2, W / 2), 0, "a wall in front of the bulb takes its veil");

        let dim = 0.01;
        let half = px(&render(1.0, -0.1, dim, 0.0), W / 2, W / 2) as f32;
        let whole = px(&render(1.0, -1.0, dim, 0.0), W / 2, W / 2) as f32;
        assert!((whole - expected(dim, 1.0, 0)).abs() <= 2.0, "{whole} vs {}", expected(dim, 1.0, 0));
        assert!(
            (half - expected(dim, 4.0 / 7.0, 0)).abs() <= 2.0,
            "part of the bulb, part of the veil: {half} vs {}",
            expected(dim, 4.0 / 7.0, 0)
        );
        assert!(whole - half > 20.0);
    }
}
