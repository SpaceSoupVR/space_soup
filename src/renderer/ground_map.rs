use glam::{Vec2, Vec3};
use wgpu::{Device, Queue, TextureView};

use crate::renderer::sky::{SkyIrradiance, SkySun};
use crate::renderer::terrain_pipeline::{TerrainImage, TerrainMaterialUniform, FALLBACK_LAYER_COLOURS};
use spacesoup_graphics::ground;

pub const GROUND_MAP_SIZE: u32 = 1024;

const BURIED_SKY_VIS: f32 = 0.05;

const BURIED_FILL_TEXELS: usize = 8;

const F0: f32 = 0.04;

pub struct HeightGrid {
    pub min: Vec2,
    pub max: Vec2,
    pub width: u32,
    pub height: u32,
    pub heights: Vec<f32>,
}

impl HeightGrid {

    pub fn sample(min: Vec2, max: Vec2, width: u32, height: u32, height_at: impl Fn(f32, f32) -> f32) -> Self {
        let (w, h) = (width.max(2), height.max(2));
        let heights = (0..h)
            .flat_map(|j| {
                let z = min.y + (max.y - min.y) * j as f32 / (h - 1) as f32;
                (0..w).map(move |i| (min.x + (max.x - min.x) * i as f32 / (w - 1) as f32, z))
            })
            .map(|(x, z)| height_at(x, z))
            .collect();
        Self { min, max, width: w, height: h, heights }
    }

    pub fn at(&self, x: f32, z: f32) -> f32 {
        let fx = ((x - self.min.x) / (self.max.x - self.min.x)).clamp(0.0, 1.0) * (self.width - 1) as f32;
        let fz = ((z - self.min.y) / (self.max.y - self.min.y)).clamp(0.0, 1.0) * (self.height - 1) as f32;
        let (i0, j0) = (fx.floor() as u32, fz.floor() as u32);
        let (i1, j1) = ((i0 + 1).min(self.width - 1), (j0 + 1).min(self.height - 1));
        let (tx, tz) = (fx - i0 as f32, fz - j0 as f32);
        let s = |i: u32, j: u32| self.heights[(j * self.width + i) as usize];
        let a = s(i0, j0) + (s(i1, j0) - s(i0, j0)) * tx;
        let b = s(i0, j1) + (s(i1, j1) - s(i0, j1)) * tx;
        a + (b - a) * tz
    }

    pub fn normal(&self, x: f32, z: f32) -> Vec3 {
        let dx = (self.max.x - self.min.x) / (self.width - 1) as f32;
        let dz = (self.max.y - self.min.y) / (self.height - 1) as f32;
        let sx = (self.at(x + dx, z) - self.at(x - dx, z)) / (2.0 * dx);
        let sz = (self.at(x, z + dz) - self.at(x, z - dz)) / (2.0 * dz);
        Vec3::new(-sx, 1.0, -sz).normalize()
    }
}

pub struct GroundInputs<'a> {
    pub heights: &'a HeightGrid,

    pub sky: &'a SkyIrradiance,

    pub sun: Option<&'a SkySun>,

    pub sky_occlusion: Option<&'a TerrainImage>,
    pub layers: &'a [Option<TerrainImage>],
    pub splat: Option<&'a TerrainImage>,
    pub settings: &'a TerrainMaterialUniform,
    pub sea_line: f32,
}

pub const RIM_FADE_METRES: f32 = 4.0;

pub struct GroundMap {
    pub width: u32,
    pub height: u32,

    pub texels: Vec<[f32; 4]>,
    pub min: Vec2,
    pub max: Vec2,

    pub top: f32,
}

fn srgb_to_linear(c: u8) -> f32 {
    let c = c as f32 / 255.0;
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

fn layer_mean(layer: Option<&TerrainImage>, fallback: [u8; 3]) -> Vec3 {
    let Some(img) = layer.filter(|i| i.width > 0 && i.height > 0 && i.rgba.len() >= 4) else {
        return Vec3::new(srgb_to_linear(fallback[0]), srgb_to_linear(fallback[1]), srgb_to_linear(fallback[2]));
    };
    let mut sum = Vec3::ZERO;
    let n = img.rgba.len() / 4;
    for px in img.rgba.chunks_exact(4) {
        sum += Vec3::new(srgb_to_linear(px[0]), srgb_to_linear(px[1]), srgb_to_linear(px[2]));
    }
    sum / n.max(1) as f32
}

struct LayerDetail {
    size: u32,
    texels: Vec<Vec3>,
}

impl LayerDetail {
    fn of(img: &TerrainImage, repeat: f32, texel: f32) -> Option<Self> {
        if img.width == 0 || img.height == 0 || img.rgba.len() < (img.width * img.height * 4) as usize || repeat <= 0.0 || texel <= 0.0 {
            return None;
        }
        let size = ((repeat / texel).round() as u32).clamp(1, img.width.min(img.height));
        let mut texels = Vec::with_capacity((size * size) as usize);
        for j in 0..size {
            let (y0, y1) = (j * img.height / size, ((j + 1) * img.height / size).max(j * img.height / size + 1));
            for i in 0..size {
                let (x0, x1) = (i * img.width / size, ((i + 1) * img.width / size).max(i * img.width / size + 1));
                let mut sum = Vec3::ZERO;
                for y in y0..y1 {
                    for x in x0..x1 {
                        let k = ((y * img.width + x) * 4) as usize;
                        sum += Vec3::new(srgb_to_linear(img.rgba[k]), srgb_to_linear(img.rgba[k + 1]), srgb_to_linear(img.rgba[k + 2]));
                    }
                }
                texels.push(sum / ((y1 - y0) * (x1 - x0)) as f32);
            }
        }
        Some(Self { size, texels })
    }

    fn at(&self, uv: Vec2) -> Vec3 {
        let n = self.size as f32;
        let f = (uv - uv.floor()) * n - Vec2::splat(0.5);
        let (fx, fy) = (f.x.rem_euclid(n), f.y.rem_euclid(n));
        let (x0, y0) = (fx.floor() as u32 % self.size, fy.floor() as u32 % self.size);
        let (x1, y1) = ((x0 + 1) % self.size, (y0 + 1) % self.size);
        let (tx, ty) = (fx - fx.floor(), fy - fy.floor());
        let p = |x: u32, y: u32| self.texels[(y * self.size + x) as usize];
        let a = p(x0, y0).lerp(p(x1, y0), tx);
        let b = p(x0, y1).lerp(p(x1, y1), tx);
        a.lerp(b, ty)
    }

    fn colour(&self, world: Vec3, n: Vec3, repeat: f32, steep: bool) -> Vec3 {
        let r = repeat.max(1e-3);
        if !steep {
            return self.at(Vec2::new(world.x, world.z) / r);
        }
        let a = n.abs();
        let w = Vec3::new(a.x.powi(4), a.y.powi(4), a.z.powi(4));
        let w = w / (w.x + w.y + w.z).max(1e-6);
        self.at(Vec2::new(world.z, world.y) / r) * w.x + self.at(Vec2::new(world.x, world.z) / r) * w.y + self.at(Vec2::new(world.x, world.y) / r) * w.z
    }
}

fn sample_rgba(img: &TerrainImage, uv: Vec2) -> [f32; 4] {
    let fx = (uv.x.clamp(0.0, 1.0) * img.width as f32 - 0.5).max(0.0);
    let fy = (uv.y.clamp(0.0, 1.0) * img.height as f32 - 0.5).max(0.0);
    let (x0, y0) = ((fx.floor() as u32).min(img.width - 1), (fy.floor() as u32).min(img.height - 1));
    let (x1, y1) = ((x0 + 1).min(img.width - 1), (y0 + 1).min(img.height - 1));
    let (tx, ty) = (fx - x0 as f32, fy - y0 as f32);
    let px = |x: u32, y: u32, c: usize| img.rgba[((y * img.width + x) * 4) as usize + c] as f32 / 255.0;
    let mut out = [0.0; 4];
    for (c, o) in out.iter_mut().enumerate() {
        let a = px(x0, y0, c) + (px(x1, y0, c) - px(x0, y0, c)) * tx;
        let b = px(x0, y1, c) + (px(x1, y1, c) - px(x0, y1, c)) * tx;
        *o = a + (b - a) * ty;
    }
    out
}

fn sample_open(img: &TerrainImage, uv: Vec2) -> Option<[f32; 4]> {
    let fx = (uv.x.clamp(0.0, 1.0) * img.width as f32 - 0.5).max(0.0);
    let fy = (uv.y.clamp(0.0, 1.0) * img.height as f32 - 0.5).max(0.0);
    let (x0, y0) = ((fx.floor() as u32).min(img.width - 1), (fy.floor() as u32).min(img.height - 1));
    let (x1, y1) = ((x0 + 1).min(img.width - 1), (y0 + 1).min(img.height - 1));
    let (tx, ty) = (fx - x0 as f32, fy - y0 as f32);
    let mut out = [0.0f32; 4];
    let mut total = 0.0;
    for (x, y, w) in [(x0, y0, (1.0 - tx) * (1.0 - ty)), (x1, y0, tx * (1.0 - ty)), (x0, y1, (1.0 - tx) * ty), (x1, y1, tx * ty)] {
        let k = ((y * img.width + x) * 4) as usize;
        if (img.rgba[k] as f32 / 255.0) < BURIED_SKY_VIS || w <= 0.0 {
            continue;
        }
        for c in 0..4 {
            out[c] += img.rgba[k + c] as f32 / 255.0 * w;
        }
        total += w;
    }
    (total > 0.0).then(|| out.map(|v| v / total))
}

fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn layer_weights(settings: &TerrainMaterialUniform, splat: Option<&TerrainImage>, uv: Vec2, world_y: f32, slope_deg: f32) -> [f32; 4] {
    if let Some(s) = splat {
        let w = sample_rgba(s, uv);
        let total = (w[0] + w[1] + w[2] + w[3]).max(0.001);
        return w.map(|v| v / total);
    }
    let rock = smoothstep(settings.slope_start_deg, settings.slope_end_deg, slope_deg);
    let mut w = [1.0 - rock, rock, 0.0, 0.0];
    if settings.height_end > settings.height_start {
        let high = smoothstep(settings.height_start, settings.height_end, world_y);
        w = [w[0] * (1.0 - high), w[1] * (1.0 - high), high, 0.0];
    }
    w
}

fn sky_and_sun(map: Option<&TerrainImage>, uv: Vec2) -> (f32, f32, bool) {
    let Some(map) = map.filter(|m| m.width > 0 && m.height > 0) else { return (1.0, 1.0, false) };
    let Some(t) = sample_open(map, uv) else { return (0.0, 0.0, true) };
    let sky_vis = t[0];
    if t[3] >= 0.5 {
        return (sky_vis, 1.0, false);
    }
    let range = crate::renderer::brush_pipeline::SUN_MASK_DISTANCE_TEXELS;
    let d = (t[1] - 0.5) * (2.0 * range);
    let w = (t[2] * range).max(0.02);
    (sky_vis, smoothstep(-w, w, d), false)
}

pub fn build(inputs: &GroundInputs, size: u32) -> GroundMap {
    let size = size.max(1);
    let g = inputs.heights;
    let extent = g.max - g.min;
    let means: Vec<Vec3> = (0..4)
        .map(|i| layer_mean(inputs.layers.get(i).and_then(|l| l.as_ref()), FALLBACK_LAYER_COLOURS[i]))
        .collect();

    let texel = (extent.x / size as f32).max(extent.y / size as f32);
    let details: Vec<Option<LayerDetail>> = (0..4)
        .map(|i| inputs.layers.get(i).and_then(|l| l.as_ref()).and_then(|img| LayerDetail::of(img, inputs.settings.repeat[i], texel)))
        .collect();
    let steep_cos = inputs.settings.biplanar_start_deg.to_radians().cos();
    let sun = inputs.sun.map(|s| (Vec3::from(s.direction).normalize_or_zero(), Vec3::from(s.light_rgb)));
    let texel = (extent.x / size as f32).max(extent.y / size as f32);
    let open_light = {
        let mut open = Vec3::from(inputs.sky.evaluate([0.0, 1.0, 0.0]));
        if let Some((to_sun, rgb)) = sun {
            open += rgb * to_sun.y.max(0.0);
        }
        open
    };
    let mut texels = Vec::with_capacity((size * size) as usize);
    let mut buried = Vec::with_capacity((size * size) as usize);
    let mut rims: Vec<ground::Rim> = Vec::new();
    let mut top = f32::MIN;
    for j in 0..size {
        for i in 0..size {
            let uv = Vec2::new((i as f32 + 0.5) / size as f32, (j as f32 + 0.5) / size as f32);
            let (x, z) = (g.min.x + extent.x * uv.x, g.min.y + extent.y * uv.y);
            let y = g.at(x, z);
            let n = g.normal(x, z);
            let slope_deg = n.y.clamp(-1.0, 1.0).acos().to_degrees();
            let w = layer_weights(inputs.settings, inputs.splat, uv, y, slope_deg);
            let world = Vec3::new(x, y, z);
            let albedo: Vec3 = (0..4)
                .filter(|&k| w[k] > 1e-3)
                .map(|k| match &details[k] {
                    Some(d) => d.colour(world, n, inputs.settings.repeat[k], n.y < steep_cos) * w[k],
                    None => means[k] * w[k],
                })
                .sum();

            let s = inputs.settings;
            let soaked = (0..4).map(|k| w[k] * s.porosity[k]).sum::<f32>() * ((s.sea_line - y) * 20.0).clamp(0.0, 1.0);
            let albedo = albedo + (albedo.map(crate::renderer::weather::wet_albedo) - albedo) * soaked;
            let (sky_vis, sun_vis, under_a_floor) = sky_and_sun(inputs.sky_occlusion, uv);
            buried.push(under_a_floor);
            let e = inputs.sky.evaluate([n.x, n.y, n.z]);
            let mut light = Vec3::from(e) * sky_vis;
            if let Some((to_sun, rgb)) = sun {
                light += rgb * n.dot(to_sun).max(0.0) * sun_vis;
            }
            let rgb = albedo * light * (1.0 - F0);
            let from_edge = ground::from_edge([uv.x, uv.y], [extent.x, extent.y]);
            let rim = ground::rim_weight(from_edge, texel, inputs.sea_line, y);
            if rim > 0.0 {
                let level = layer_weights(inputs.settings, inputs.splat, uv, y, 0.0);
                let mean: Vec3 = (0..4).map(|k| means[k] * level[k]).sum();
                let soaked = (0..4).map(|k| level[k] * s.porosity[k]).sum::<f32>() * ((s.sea_line - y) * 20.0).clamp(0.0, 1.0);
                let mean = mean + (mean.map(crate::renderer::weather::wet_albedo) - mean) * soaked;
                rims.push(ground::Rim { texel: texels.len(), weight: rim, level: mean * open_light * (1.0 - F0) });
            }
            top = top.max(y);
            texels.push([rgb.x, rgb.y, rgb.z, y]);
        }
    }
    fill_under_buildings(&mut texels, &mut buried, size as usize, size as usize);
    ground::one_shelf(&mut texels, &rims);
    GroundMap { width: size, height: size, texels, min: g.min, max: g.max, top }
}

fn fill_under_buildings(texels: &mut [[f32; 4]], buried: &mut [bool], w: usize, h: usize) {
    for _ in 0..BURIED_FILL_TEXELS {
        let mut filled = Vec::new();
        for j in 0..h {
            for i in 0..w {
                if !buried[j * w + i] {
                    continue;
                }
                let mut sum = [0.0f32; 3];
                let mut n = 0.0;
                for (di, dj) in [(-1i64, 0i64), (1, 0), (0, -1), (0, 1)] {
                    let (x, y) = (i as i64 + di, j as i64 + dj);
                    if x < 0 || y < 0 || x >= w as i64 || y >= h as i64 {
                        continue;
                    }
                    let k = y as usize * w + x as usize;
                    if !buried[k] {
                        for c in 0..3 {
                            sum[c] += texels[k][c];
                        }
                        n += 1.0;
                    }
                }
                if n > 0.0 {
                    filled.push((j * w + i, [sum[0] / n, sum[1] / n, sum[2] / n]));
                }
            }
        }
        if filled.is_empty() {
            break;
        }
        for (k, rgb) in filled {
            texels[k][0] = rgb[0];
            texels[k][1] = rgb[1];
            texels[k][2] = rgb[2];
            buried[k] = false;
        }
    }
}

fn mip_levels(size: u32) -> u32 {
    32 - size.max(1).leading_zeros()
}

pub const GROUND_TRACE_FINEST_LEVEL: u32 = 4;

pub const GROUND_TRACE_READINGS: u32 = 5;

pub const GROUND_TRACE_START_LEVEL: u32 = 6;

pub const GROUND_TRACE_MAX_STEPS: u32 = 64;

const GROUND_MAX_MARGIN: f32 = 0.02;

pub(crate) fn f16_to_f32(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = ((h >> 10) & 0x1f) as i32;
    let mant = (h & 0x3ff) as f32;
    match exp {
        0 => sign * mant * 2f32.powi(-24),
        0x1f => sign * f32::INFINITY,
        _ => sign * (1.0 + mant / 1024.0) * 2f32.powi(exp - 15),
    }
}

fn as_half(v: f32) -> f32 {
    f16_to_f32(crate::renderer::sky::f32_to_f16(v))
}

fn half_at_least(v: f32) -> f32 {
    let h = crate::renderer::sky::f32_to_f16(v);
    if f16_to_f32(h) >= v {
        return f16_to_f32(h);
    }

    f16_to_f32(if h & 0x8000 == 0 { h + 1 } else { h - 1 })
}

pub struct GroundLevel {
    pub width: u32,
    pub height: u32,
    pub texels: Vec<[f32; 4]>,
}

pub fn levels(map: &GroundMap) -> Vec<GroundLevel> {
    let count = mip_levels(map.width.max(map.height));
    let base = GroundLevel { width: map.width, height: map.height, texels: map.texels.iter().map(|t| t.map(as_half)).collect() };
    let mut out = vec![base];
    for level in 1..count {
        let next = {
            let prev = &out[out.len() - 1];
            let base = &out[0];
            let (w, h) = (prev.width, prev.height);
            let (nw, nh) = ((w / 2).max(1), (h / 2).max(1));
            let mut next = Vec::with_capacity((nw * nh) as usize);
            for y in 0..nh {
                for x in 0..nw {
                    let mut rgb = [0.0f32; 3];
                    let mut highest = f32::MIN;
                    for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                        let (sx, sy) = ((2 * x + dx).min(w - 1), (2 * y + dy).min(h - 1));
                        let t = prev.texels[(sy * w + sx) as usize];
                        for c in 0..3 {
                            rgb[c] += t[c] * 0.25;
                        }
                        highest = highest.max(t[3]);
                    }
                    if level == 1 {

                        highest = f32::MIN;
                        let (bw, bh) = (base.width as i64, base.height as i64);
                        for sy in 2 * y as i64 - 1..=2 * y as i64 + 2 {
                            for sx in 2 * x as i64 - 1..=2 * x as i64 + 2 {
                                let k = (sy.clamp(0, bh - 1) * bw + sx.clamp(0, bw - 1)) as usize;
                                highest = highest.max(base.texels[k][3]);
                            }
                        }
                        highest = half_at_least(highest + GROUND_MAX_MARGIN);
                    }
                    next.push([as_half(rgb[0]), as_half(rgb[1]), as_half(rgb[2]), highest]);
                }
            }
            GroundLevel { width: nw, height: nh, texels: next }
        };
        out.push(next);
    }
    out
}

fn height_at(base: &GroundLevel, q: Vec2) -> f32 {
    let (w, h) = (base.width as f32, base.height as f32);
    let fx = (q.x - 0.5).clamp(0.0, w - 1.0);
    let fy = (q.y - 0.5).clamp(0.0, h - 1.0);
    let (x0, y0) = (fx.floor() as u32, fy.floor() as u32);
    let (x1, y1) = ((x0 + 1).min(base.width - 1), (y0 + 1).min(base.height - 1));
    let (tx, ty) = (fx - x0 as f32, fy - y0 as f32);
    let s = |x: u32, y: u32| base.texels[(y * base.width + x) as usize][3];
    let a = s(x0, y0) + (s(x1, y0) - s(x0, y0)) * tx;
    let b = s(x0, y1) + (s(x1, y1) - s(x0, y1)) * tx;
    a + (b - a) * ty
}

#[derive(Clone, Copy, Debug)]
pub struct GroundTrace {
    pub t: Option<f32>,
    pub steps: u32,
    pub reads: u32,
}

pub fn trace(levels: &[GroundLevel], min: Vec2, max: Vec2, top: f32, e: Vec3, d: Vec3) -> GroundTrace {
    trace_with(levels, min, max, top, e, d, TraceShape::SHIPPED)
}

#[derive(Clone, Copy, Debug)]
pub struct TraceShape {
    pub finest: u32,
    pub start: u32,
    pub readings: u32,
    pub max_steps: u32,
}

impl TraceShape {

    pub const SHIPPED: Self = Self {
        finest: GROUND_TRACE_FINEST_LEVEL,
        start: GROUND_TRACE_START_LEVEL,
        readings: GROUND_TRACE_READINGS,
        max_steps: GROUND_TRACE_MAX_STEPS,
    };
}

pub fn trace_with(levels: &[GroundLevel], min: Vec2, max: Vec2, top: f32, e: Vec3, d: Vec3, shape: TraceShape) -> GroundTrace {
    let miss = GroundTrace { t: None, steps: 0, reads: 0 };
    let Some(base) = levels.first() else { return miss };
    let size = base.width as f32;
    let scale = size / (max - min);
    let q0 = (Vec2::new(e.x, e.z) - min) * scale;
    let raw = Vec2::new(d.x, d.z) * scale;

    let dq = Vec2::new(if raw.x.abs() < 1e-6 { 1e-6 } else { raw.x }, if raw.y.abs() < 1e-6 { 1e-6 } else { raw.y });
    let inv = Vec2::ONE / dq;

    let ta = -q0 * inv;
    let tb = (Vec2::splat(size) - q0) * inv;
    let mut t_in = ta.min(tb).max_element().max(0.0);
    let mut t_out = ta.max(tb).min_element();
    if d.y > 0.0 {
        t_out = t_out.min((top - e.y) / d.y);
    } else if d.y < 0.0 {
        t_in = t_in.max((top - e.y) / d.y);
    } else if e.y > top {
        return miss;
    }
    let top_level = levels.len() as i32 - 1;
    let finest = (shape.finest as i32).min(top_level);
    let mut level = (shape.start as i32).clamp(finest, top_level);
    let ahead = Vec2::new(if dq.x > 0.0 { 1.0 } else { 0.0 }, if dq.y > 0.0 { 1.0 } else { 0.0 });

    let nudge = Vec2::new(dq.x.signum(), dq.y.signum()) * 1e-3;
    let gaps = shape.readings.max(2) - 1;
    let mut t = t_in;
    let mut steps = 0;
    let mut reads = 0;

    let mut risen = d.y <= 0.0;
    while steps < shape.max_steps && t < t_out {
        steps += 1;
        reads += 1;
        let lv = &levels[level as usize];
        let cell = (1u32 << level) as f32;
        let last = Vec2::new(lv.width as f32 - 1.0, lv.height as f32 - 1.0);
        let c = ((q0 + dq * t + nudge) / cell).floor().clamp(Vec2::ZERO, last);
        let t_exit = (((c + ahead) * cell - q0) * inv).min_element().min(t_out);
        let highest = lv.texels[(c.y as u32 * lv.width + c.x as u32) as usize][3];
        let y_in = e.y + d.y * t;
        let y_out = e.y + d.y * t_exit;
        risen = risen || y_in > highest;
        if y_in.min(y_out) <= highest {

            let t_top = if y_in > highest { t + (y_in - highest) / -d.y } else { t };
            if level > finest {
                level -= 1;
                t = t_top;
                continue;
            }

            let at = |t: f32| e.y + d.y * t - height_at(base, q0 + dq * t);
            let span = (t_exit - t_top) / gaps as f32;
            reads += gaps + 1;
            let mut f_prev = at(t_top);
            if f_prev <= 0.0 && risen {
                return GroundTrace { t: Some(t_top), steps, reads };
            }
            risen = risen || f_prev > 0.0;
            for k in 1..=gaps {
                let tk = t_top + span * k as f32;
                let f = at(tk);
                if f <= 0.0 && risen {
                    return GroundTrace { t: Some(tk - span + span * f_prev / (f_prev - f)), steps, reads };
                }
                risen = risen || f > 0.0;
                f_prev = f;
            }
        }

        let next = ((q0 + dq * t_exit + nudge) / cell).floor();
        if (next * 0.5).floor() != (c * 0.5).floor() {
            level = (level + 1).min(top_level);
        }
        t = t_exit;
    }
    GroundTrace { t: None, steps, reads }
}

pub fn upload(device: &Device, queue: &Queue, map: &GroundMap) -> TextureView {
    let chain = levels(map);
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("ground_map"),
        size: wgpu::Extent3d { width: map.width, height: map.height, depth_or_array_layers: 1 },
        mip_level_count: chain.len() as u32,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    for (mip, level) in chain.iter().enumerate() {
        let bytes: Vec<u16> = level.texels.iter().flat_map(|t| t.map(crate::renderer::sky::f32_to_f16)).collect();
        queue.write_texture(
            wgpu::TexelCopyTextureInfo { texture: &tex, mip_level: mip as u32, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            bytemuck::cast_slice(&bytes),
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(level.width * 8), rows_per_image: Some(level.height) },
            wgpu::Extent3d { width: level.width, height: level.height, depth_or_array_layers: 1 },
        );
    }
    tex.create_view(&wgpu::TextureViewDescriptor::default())
}

pub fn none(device: &Device, queue: &Queue) -> TextureView {
    upload(
        device,
        queue,
        &GroundMap {
            width: 1,
            height: 1,
            texels: vec![[0.0, 0.0, 0.0, 0.0]],
            min: Vec2::ZERO,
            max: Vec2::ONE,
            top: f32::MIN,
        },
    )
}

#[cfg(test)]
pub(crate) mod fixtures {
    use super::*;

    pub fn hills(x: f32, z: f32) -> f32 {
        let rise = smoothstep(25.0, 50.0, (x * x + z * z).sqrt());
        rise * (5.0 + 2.5 * (x * 0.13).sin() * (z * 0.11).cos() + (x * 0.37 + z * 0.23).sin())
    }

    pub fn height_map(size: u32, height_at: impl Fn(f32, f32) -> f32) -> GroundMap {
        let (min, max) = (Vec2::splat(-55.0), Vec2::splat(55.0));
        let g = HeightGrid::sample(min, max, 257, 257, height_at);
        let mut texels = Vec::with_capacity((size * size) as usize);
        let mut top = f32::MIN;
        for j in 0..size {
            for i in 0..size {
                let uv = Vec2::new(i as f32 + 0.5, j as f32 + 0.5) / size as f32;
                let y = g.at(min.x + 110.0 * uv.x, min.y + 110.0 * uv.y);
                top = top.max(y);
                texels.push([0.0, 0.0, 0.0, y]);
            }
        }
        GroundMap { width: size, height: size, texels, min, max, top }
    }

    pub fn rays(n: usize, seed: u64) -> Vec<(Vec3, Vec3)> {
        let mut s = seed.max(1);
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 40) as f32 / (1u64 << 24) as f32
        };
        (0..n)
            .map(|_| {
                let e = Vec3::new(24.0 * next() - 12.0, 3.4 * next(), 24.0 * next() - 12.0);
                let (az, el) = (std::f32::consts::TAU * next(), (-35.0 + 55.0 * next()).to_radians());
                (e, Vec3::new(az.cos() * el.cos(), el.sin(), az.sin() * el.cos()))
            })
            .collect()
    }

    pub struct Marched {
        pub t: Option<f32>,

        pub graze: f32,
    }

    pub fn march(base: &GroundLevel, min: Vec2, max: Vec2, top: f32, e: Vec3, d: Vec3) -> Marched {
        let scale = base.width as f32 / (max - min);
        let at = |t: f32| {
            let p = e + d * t;
            p.y - height_at(base, (Vec2::new(p.x, p.z) - min) * scale)
        };
        let inside = |t: f32| {
            let p = e + d * t;
            p.x >= min.x && p.x <= max.x && p.z >= min.y && p.z <= max.y && !(p.y > top && d.y >= 0.0)
        };
        let step = 0.01;
        let mut prev = at(0.0);
        let mut graze = prev.abs();
        if prev <= 0.0 {
            return Marched { t: Some(0.0), graze };
        }
        let mut t = step;
        while inside(t) {
            let f = at(t);
            if f <= 0.0 {
                let hit = t - step + step * prev / (prev - f);

                let deepest = (1..=100).map(|k| at(t + k as f32 * step)).fold(f, f32::min);
                return Marched { t: Some(hit), graze: -deepest };
            }
            graze = graze.min(f);
            prev = f;
            t += step;
        }
        Marched { t: None, graze }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat(y: f32) -> HeightGrid {
        HeightGrid::sample(Vec2::new(-10.0, -10.0), Vec2::new(10.0, 10.0), 21, 21, |_, _| y)
    }

    fn sky() -> SkyIrradiance {
        let mut s = SkyIrradiance::default();

        s.sh[0] = [3.5449077; 3];
        s
    }

    #[test]
    fn the_transcribed_rules_are_the_terrain_shaders() {
        let src = crate::renderer::terrain_pipeline::terrain_shader_src();
        assert!(src.contains("let rock_w = smoothstep(mat.slope_start_deg, mat.slope_end_deg, slope_deg);"));
        assert!(src.contains("let high_w = smoothstep(mat.height_start, mat.height_end, world_y);"));
        assert!(src.contains("let ground_sun_d = (ground_map.g - 0.5) * (2.0 * "));
        assert!(src.contains("receiver_sun_mask = select(-1.0, smoothstep(-ground_sun_w, ground_sun_w, ground_sun_d), ground_map.a < 0.5);"));
        assert!(src.contains("let sky_vis = ground_map.r;"));
    }

    #[test]
    fn the_height_is_the_grids_and_the_top_is_its_highest_point() {
        let g = HeightGrid::sample(Vec2::new(0.0, 0.0), Vec2::new(10.0, 10.0), 11, 11, |x, z| 0.1 * x + 0.2 * z);
        assert!((g.at(5.0, 5.0) - 1.5).abs() < 1e-5);
        let s = TerrainMaterialUniform::default();
        let map = build(
            &GroundInputs { heights: &g, sky: &sky(), sun: None, sky_occlusion: None, layers: &[], splat: None, settings: &s, sea_line: -1e9 },
            8,
        );
        let t = map.texels[(3 * 8 + 5) as usize];
        let (x, z) = (10.0 * 5.5 / 8.0, 10.0 * 3.5 / 8.0);
        assert!((t[3] - (0.1 * x + 0.2 * z)).abs() < 1e-4, "height {} at ({x}, {z})", t[3]);
        assert!((map.top - map.texels.iter().map(|t| t[3]).fold(f32::MIN, f32::max)).abs() < 1e-6);
        let n = g.normal(5.0, 5.0);
        assert!((n - Vec3::new(-0.1, 1.0, -0.2).normalize()).length() < 1e-4, "{n:?}");
    }

    #[test]
    fn ground_in_the_buildings_shadow_is_darker_by_exactly_the_sun() {
        let g = flat(0.0);
        let s = TerrainMaterialUniform::default();
        let sun = SkySun { direction: [0.0, 1.0, 0.0], light_rgb: [4.0, 4.0, 4.0], texels: 1, energy_fraction: 0.5 };
        let grass = TerrainImage { width: 1, height: 1, rgba: vec![128, 128, 128, 255] };
        let layers = [Some(grass)];

        let occ = TerrainImage {
            width: 2,
            height: 1,
            rgba: vec![255, 255, 0, 0, 255, 0, 0, 0],
        };
        let map = build(
            &GroundInputs { heights: &g, sky: &sky(), sun: Some(&sun), sky_occlusion: Some(&occ), layers: &layers, splat: None, settings: &s, sea_line: -1e9 },
            8,
        );
        let albedo = srgb_to_linear(128);
        let e = sky().evaluate([0.0, 1.0, 0.0])[0];
        let lit = map.texels[0][0];
        let shaded = map.texels[7][0];
        assert!((lit - albedo * (e + 4.0) * (1.0 - F0)).abs() < 1e-3, "sunlit ground {lit}");
        assert!((shaded - albedo * e * (1.0 - F0)).abs() < 1e-3, "shadowed ground {shaded}");
    }

    #[test]
    fn ground_under_a_building_takes_the_colour_around_it() {
        let g = flat(0.0);
        let s = TerrainMaterialUniform::default();
        let grass = TerrainImage { width: 1, height: 1, rgba: vec![128, 128, 128, 255] };
        let layers = [Some(grass)];

        let occ = TerrainImage {
            width: 4,
            height: 1,
            rgba: vec![255, 255, 0, 0, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        };
        let map = build(
            &GroundInputs { heights: &g, sky: &sky(), sun: None, sky_occlusion: Some(&occ), layers: &layers, splat: None, settings: &s, sea_line: -1e9 },
            8,
        );
        let open = map.texels[0][0];
        assert!(open > 0.1);

        for t in &map.texels[..8] {
            assert!((t[0] - open).abs() < 0.02 * open, "{:?}", &map.texels[..8]);
        }
    }

    #[test]
    fn a_bound_rounded_to_half_float_stays_a_bound() {
        for v in [1.9f32, -1.8, 0.3, 8.8, 0.0, 7.123_456, -0.000_1] {
            let h = half_at_least(v);
            assert!(h >= v, "{v} -> {h}");
            assert_eq!(as_half(h), h, "{v} -> {h} is not a half float");
            assert!(h - v <= v.abs() / 512.0 + 1e-4, "{v} -> {h} is more than one step up");
        }
    }

    #[test]
    fn every_level_bounds_the_ground_under_it() {

        let map = fixtures::height_map(64, |x, z| 3.0 * (x * 0.9).sin() * (z * 0.7).cos() + 0.5 * (x * 2.1 + z).sin());
        let chain = levels(&map);
        assert_eq!(chain.len(), 7);
        let base = &chain[0];
        for (l, lv) in chain.iter().enumerate().skip(1) {
            let cell = (1u32 << l) as f32;
            for y in 0..lv.height {
                for x in 0..lv.width {
                    let bound = lv.texels[(y * lv.width + x) as usize][3];
                    let mut highest = f32::MIN;
                    for sy in 0..=8 {
                        for sx in 0..=8 {
                            let q = (Vec2::new(x as f32, y as f32) + Vec2::new(sx as f32, sy as f32) / 8.0) * cell;
                            highest = highest.max(height_at(base, q));
                        }
                    }
                    assert!(highest <= bound, "level {l} texel ({x}, {y}): ground {highest} over bound {bound}");

                    let span = 1i64 << l;
                    let mut reachable = f32::MIN;
                    for sy in y as i64 * span - 1..=(y as i64 + 1) * span {
                        for sx in x as i64 * span - 1..=(x as i64 + 1) * span {
                            let k = (sy.clamp(0, 63) * 64 + sx.clamp(0, 63)) as usize;
                            reachable = reachable.max(base.texels[k][3]);
                        }
                    }
                    let slack = bound - reachable;
                    assert!(slack >= GROUND_MAX_MARGIN && slack < GROUND_MAX_MARGIN + 0.01, "level {l} texel ({x}, {y}): bound {bound} over reachable {reachable}");
                }
            }
        }
    }

    #[test]
    fn a_ray_from_under_the_maps_ground_meets_only_what_it_rises_to() {
        let map = fixtures::height_map(1024, fixtures::hills);
        let chain = levels(&map);
        let size = map.max - map.min;
        let mut checked = 0;
        for i in 0..400 {
            let x = map.min.x + size.x * (0.1 + 0.8 * ((i * 37 % 400) as f32 / 400.0));
            let z = map.min.y + size.y * (0.1 + 0.8 * ((i * 91 % 400) as f32 / 400.0));
            let under = height_at(&chain[0], (Vec2::new(x, z) - map.min) * (chain[0].width as f32 / size)) - 0.05;
            let e = Vec3::new(x, under, z);

            let up = trace(&chain, map.min, map.max, map.top, e, Vec3::Y);
            assert!(up.t.is_none(), "from under the ground at {e}, straight up met {:?}", up.t);

            let d = Vec3::new(0.7, 0.05, 0.7).normalize();
            if let Some(t) = trace(&chain, map.min, map.max, map.top, e, d).t {
                assert!(t > 0.05, "from under the ground at {e}, met it where it started ({t})");
                checked += 1;
            }
        }
        assert!(checked > 0, "no level ray met a hill");
    }

    #[test]
    fn a_ray_heading_down_from_under_the_ground_meets_it_where_it_starts() {
        let map = fixtures::height_map(1024, fixtures::hills);
        let chain = levels(&map);
        let size = map.max - map.min;
        for i in 0..400 {
            let x = map.min.x + size.x * (0.1 + 0.8 * ((i * 37 % 400) as f32 / 400.0));
            let z = map.min.y + size.y * (0.1 + 0.8 * ((i * 91 % 400) as f32 / 400.0));
            let under = height_at(&chain[0], (Vec2::new(x, z) - map.min) * (chain[0].width as f32 / size)) - 0.05;
            let e = Vec3::new(x, under, z);
            for d in [Vec3::new(0.7, -0.05, 0.7).normalize(), Vec3::new(-0.3, -0.6, 0.2).normalize(), Vec3::new(1.0, 0.0, 0.0)] {
                let r = trace(&chain, map.min, map.max, map.top, e, d);
                assert!(r.t.is_some_and(|t| t < 1e-3), "from under the ground at {e} along {d}: {:?}", r.t);
                assert!(r.steps <= 4, "from under the ground at {e} along {d}: {} steps", r.steps);
            }
        }
    }

    #[test]
    fn the_trace_meets_the_ground_where_a_fine_march_does() {
        let map = fixtures::height_map(1024, fixtures::hills);
        let chain = levels(&map);
        let mut grazing = 0;
        let mut steps = Vec::new();
        let rays = fixtures::rays(400, 7);
        for (i, &(e, d)) in rays.iter().enumerate() {
            let traced = trace(&chain, map.min, map.max, map.top, e, d);
            let fine = fixtures::march(&chain[0], map.min, map.max, map.top, e, d);
            steps.push(traced.steps);
            match (traced.t, fine.t) {
                (Some(a), Some(b)) => {
                    assert!((a - b).abs() <= 0.02 + 0.002 * b, "ray {i} from {e} along {d}: traced {a}, marched {b}");
                }
                (None, None) => {}
                (a, b) => {
                    assert!(fine.graze < 0.03, "ray {i} from {e} along {d}: traced {a:?}, marched {b:?}, clearance {}", fine.graze);
                    grazing += 1;
                }
            }
        }
        assert!(grazing <= rays.len() / 100, "{grazing} grazing disagreements");
        let most = *steps.iter().max().unwrap();
        let mean = steps.iter().sum::<u32>() as f32 / steps.len() as f32;
        eprintln!("ground trace over test_room-like hills: mean {mean:.1} steps, most {most}");
        assert!(most < GROUND_TRACE_MAX_STEPS, "a ray used the whole budget ({most} steps)");
    }

    #[test]
    fn a_ridge_between_32_and_64_metres_is_met() {
        let map = fixtures::height_map(1024, |x, _| if x > 40.0 { ((x - 40.0) * 0.8).min(4.0) } else { 0.0 });
        let chain = levels(&map);
        let e = Vec3::new(0.0, 1.5, 0.0);
        let d = Vec3::new(1.0, 0.02, 0.0).normalize();
        let t = trace(&chain, map.min, map.max, map.top, e, d).t.expect("the ray passed through the ridge");
        let p = e + d * t;
        assert!(p.x > 40.0 && p.x < 44.0, "met the ground at {p}");
        assert!((p.y - map_height(&chain[0], &map, p)).abs() < 0.02, "at {p}, not on the ground");
    }

    fn map_height(base: &GroundLevel, map: &GroundMap, p: Vec3) -> f32 {
        height_at(base, (Vec2::new(p.x, p.z) - map.min) * (base.width as f32 / (map.max - map.min)))
    }

    #[test]
    fn a_cliff_is_rock_and_the_plain_is_grass() {
        let g = HeightGrid::sample(Vec2::new(0.0, 0.0), Vec2::new(10.0, 10.0), 11, 11, |x, _| if x > 5.0 { (x - 5.0) * 3.0 } else { 0.0 });
        let s = TerrainMaterialUniform::default();
        let layers = [
            Some(TerrainImage { width: 1, height: 1, rgba: vec![0, 255, 0, 255] }),
            Some(TerrainImage { width: 1, height: 1, rgba: vec![255, 0, 0, 255] }),
        ];
        let map = build(
            &GroundInputs { heights: &g, sky: &sky(), sun: None, sky_occlusion: None, layers: &layers, splat: None, settings: &s, sea_line: -1e9 },
            10,
        );
        let plain = map.texels[5 * 10 + 1];
        let cliff = map.texels[5 * 10 + 8];
        assert!(plain[1] > plain[0], "the plain is the grass layer: {plain:?}");
        assert!(cliff[0] > cliff[1], "the cliff is the rock layer: {cliff:?}");
    }

    #[test]
    fn the_sea_floor_at_the_maps_edge_is_one_colour_along_it() {

        let g = HeightGrid::sample(Vec2::new(-10.0, -10.0), Vec2::new(10.0, 10.0), 81, 81, |x, _| -10.0 + 0.6 * (x * 1.7).sin());
        let sun = SkySun { direction: [0.6, 0.8, 0.0], light_rgb: [4.0, 4.0, 4.0], texels: 1, energy_fraction: 0.5 };

        let checker: Vec<u8> = (0..16).flat_map(|i| if (i % 4 + i / 4) % 2 == 0 { [40, 40, 40, 255] } else { [220, 220, 220, 255] }).collect();
        let layers = [Some(TerrainImage { width: 4, height: 4, rgba: checker }), Some(TerrainImage { width: 1, height: 1, rgba: vec![60, 50, 30, 255] })];

        let splat = TerrainImage {
            width: 4,
            height: 4,
            rgba: (0..16).flat_map(|i| if i == 1 { [0u8, 255, 0, 0] } else { [255u8, 0, 0, 0] }).collect(),
        };

        let occ_w = 16;
        let occ = TerrainImage {
            width: occ_w,
            height: 1,
            rgba: (0..occ_w).flat_map(|i| if i % 2 == 0 { [255u8, 255, 0, 0] } else { [255, 0, 0, 0] }).collect(),
        };
        let size = 64u32;
        let built = |sea_line: f32| {
            let mut s = TerrainMaterialUniform::default();
            s.repeat = [1.5; 4];
            s.sea_line = sea_line;
            build(
                &GroundInputs { heights: &g, sky: &sky(), sun: Some(&sun), sky_occlusion: Some(&occ), layers: &layers, splat: Some(&splat), settings: &s, sea_line },
                size,
            )
        };
        let spread = |map: &GroundMap, row: u32| {
            let r: Vec<f32> = (0..size).map(|i| map.texels[(row * size + i) as usize][0]).collect();
            let (lo, hi) = r.iter().fold((f32::MAX, f32::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)));
            (hi - lo) / hi.max(1e-6)
        };
        let sea = built(-1.4);
        let dry = built(-1e9);

        assert!(spread(&sea, 0) < 1e-4, "the near edge's row varies by {:.3} of its brightest", spread(&sea, 0));
        assert!(spread(&sea, size - 1) < 1e-4, "the far edge's row varies by {:.3}", spread(&sea, size - 1));

        let side: Vec<f32> = (0..size).map(|j| sea.texels[(j * size) as usize][0]).collect();
        let (lo, hi) = side.iter().fold((f32::MAX, f32::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)));
        assert!(hi - lo < 1e-4 * hi, "a side edge varies by {:.3}", (hi - lo) / hi);

        let far = sea.texels[(size * size - 1) as usize][0];
        assert!((side[0] - sea.texels[0][0]).abs() < 1e-6 && (far - side[0]).abs() < 1e-4 * hi, "the edges differ: {} {far}", side[0]);

        let mid = size / 2;
        let inner_spread = {
            let r: Vec<f32> = (size / 4..size - size / 4).map(|i| sea.texels[(mid * size + i) as usize][0]).collect();
            let (lo, hi) = r.iter().fold((f32::MAX, f32::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)));
            (hi - lo) / hi
        };
        assert!(inner_spread > 0.3, "the floor's own shadows inside the rim: {inner_spread:.3}");

        assert!(spread(&dry, 0) > 0.3, "a dry rim is left as it is: {:.3}", spread(&dry, 0));
    }

    #[test]
    fn a_painted_splat_is_the_ground_maps_colour() {
        let g = HeightGrid::sample(Vec2::new(0.0, 0.0), Vec2::new(10.0, 10.0), 11, 11, |_, _| 0.0);
        let s = TerrainMaterialUniform::default();
        assert_eq!(s.use_splat, 0.0, "the renderer's settings carry no splat flag");
        let layers = [
            Some(TerrainImage { width: 1, height: 1, rgba: vec![0, 255, 0, 255] }),
            Some(TerrainImage { width: 1, height: 1, rgba: vec![255, 0, 0, 255] }),
            Some(TerrainImage { width: 1, height: 1, rgba: vec![0, 0, 255, 255] }),
            Some(TerrainImage { width: 1, height: 1, rgba: vec![255, 255, 0, 255] }),
        ];

        let splat = TerrainImage { width: 2, height: 2, rgba: [0u8, 0, 0, 255].repeat(4) };
        let map = build(
            &GroundInputs { heights: &g, sky: &sky(), sun: None, sky_occlusion: None, layers: &layers, splat: Some(&splat), settings: &s, sea_line: -1e9 },
            10,
        );
        let t = map.texels[5 * 10 + 5];
        assert!(t[0] > 0.0 && t[1] > 0.0 && t[2] < 1e-6, "the plain is the painted sediment, not grass by slope: {t:?}");
        assert!((t[0] - t[1]).abs() < 1e-3 * t[0], "sediment's red and green, equally: {t:?}");
    }
}

#[cfg(test)]
mod step_census {
    use super::*;

    #[test]
    #[ignore]
    fn shapes() {
        let map = fixtures::height_map(1024, fixtures::hills);
        let chain = levels(&map);
        let rays = fixtures::rays(4000, 3);
        let fine: Vec<fixtures::Marched> = rays.iter().map(|&(e, d)| fixtures::march(&chain[0], map.min, map.max, map.top, e, d)).collect();
        for (finest, readings) in [(2, 3), (3, 3), (3, 5), (4, 5), (4, 9), (5, 9), (5, 17)] {
            for start in [5, 6, 7] {
                let shape = TraceShape { finest, start, readings, max_steps: 256 };
                let (mut sum, mut most, mut bad, mut graze, mut near, mut reads) = (0u32, 0u32, 0u32, 0u32, 0u32, 0u32);
                let mut hist = [0u32; 5];
                for (&(e, d), f) in rays.iter().zip(&fine) {
                    let r = trace_with(&chain, map.min, map.max, map.top, e, d, shape);
                    sum += r.steps;
                    reads += r.reads;
                    most = most.max(r.steps);
                    hist[match r.steps { 0..=8 => 0, 9..=16 => 1, 17..=32 => 2, 33..=64 => 3, _ => 4 }] += 1;
                    match (r.t, f.t) {
                        (Some(a), Some(b)) if (a - b).abs() <= 0.02 + 0.002 * b => near += 1,
                        (None, None) => {}
                        _ if f.graze < 0.03 => graze += 1,
                        _ => bad += 1,
                    }
                }
                eprintln!("finest {finest} readings {readings:>2} start {start}: mean {:>5.2} most {most:>3} steps<=8/16/32/64/more {:?}  reads {:>5.2}  bad {bad} grazing {graze} near {near}", sum as f32 / rays.len() as f32, hist, reads as f32 / rays.len() as f32);
            }
        }
    }

    #[test]
    #[ignore]
    fn steps_by_elevation() {
        let map = fixtures::height_map(1024, fixtures::hills);
        let chain = levels(&map);
        let mut buckets = std::collections::BTreeMap::<i32, (u32, u32, u32, u32)>::new();
        for (e, d) in fixtures::rays(20000, 3) {
            let r = trace(&chain, map.min, map.max, map.top, e, d);
            let el = (d.y.asin().to_degrees() / 5.0).floor() as i32 * 5;
            let b = buckets.entry(el).or_default();
            b.0 += 1;
            b.1 += r.steps;
            b.2 = b.2.max(r.steps);
            b.3 += r.t.is_some() as u32;
        }
        for (el, (n, s, m, h)) in buckets {
            eprintln!("elev {el:>4}..{:>3}: rays {n:>5}  mean steps {:>5.1}  max {m:>3}  hit {:>3.0}%", el + 5, s as f32 / n as f32, 100.0 * h as f32 / n as f32);
        }
    }

    #[test]
    fn the_sea_floor_at_the_maps_edge_is_one_colour_along_it() {
        let g = HeightGrid::sample(Vec2::new(-10.0, -10.0), Vec2::new(10.0, 10.0), 81, 81, |x, _| -10.0 + 0.6 * (x * 1.7).sin());
        let sun = SkySun { direction: [0.6, 0.8, 0.0], light_rgb: [4.0, 4.0, 4.0], texels: 1, energy_fraction: 0.5 };
        let checker: Vec<u8> = (0..16).flat_map(|i| if (i % 4 + i / 4) % 2 == 0 { [40, 40, 40, 255] } else { [220, 220, 220, 255] }).collect();
        let layers = [Some(TerrainImage { width: 4, height: 4, rgba: checker }), Some(TerrainImage { width: 1, height: 1, rgba: vec![60, 50, 30, 255] })];
        let splat = TerrainImage {
            width: 4,
            height: 4,
            rgba: (0..16).flat_map(|i| if i == 1 { [0u8, 255, 0, 0] } else { [255u8, 0, 0, 0] }).collect(),
        };
        let occ_w = 16;
        let occ = TerrainImage {
            width: occ_w,
            height: 1,
            rgba: (0..occ_w).flat_map(|i| if i % 2 == 0 { [255u8, 255, 0, 0] } else { [255, 0, 0, 0] }).collect(),
        };
        let size = 64u32;
        let mut sky = SkyIrradiance::default();
        sky.sh[0] = [3.5449077; 3];
        let built = |sea_line: f32| {
            let mut s = TerrainMaterialUniform::default();
            s.repeat = [1.5; 4];
            build(
                &GroundInputs { heights: &g, sky: &sky, sun: Some(&sun), sky_occlusion: Some(&occ), layers: &layers, splat: Some(&splat), settings: &s, sea_line },
                size,
            )
        };
        let spread = |map: &GroundMap, row: u32| {
            let r: Vec<f32> = (0..size).map(|i| map.texels[(row * size + i) as usize][0]).collect();
            let (lo, hi) = r.iter().fold((f32::MAX, f32::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)));
            (hi - lo) / hi.max(1e-6)
        };
        let sea = built(-1.4);
        let dry = built(-1e9);
        assert!(spread(&sea, 0) < 1e-4, "the near edge's row varies by {:.3} of its brightest", spread(&sea, 0));
        assert!(spread(&sea, size - 1) < 1e-4, "the far edge's row varies by {:.3}", spread(&sea, size - 1));
        let side: Vec<f32> = (0..size).map(|j| sea.texels[(j * size) as usize][0]).collect();
        let (lo, hi) = side.iter().fold((f32::MAX, f32::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)));
        assert!(hi - lo < 1e-4 * hi, "a side edge varies by {:.3}", (hi - lo) / hi);
        let far = sea.texels[(size * size - 1) as usize][0];
        assert!((side[0] - sea.texels[0][0]).abs() < 1e-6 && (far - side[0]).abs() < 1e-4 * hi, "the edges differ: {} {far}", side[0]);
        let mid = size / 2;
        let inner_spread = {
            let r: Vec<f32> = (size / 4..size - size / 4).map(|i| sea.texels[(mid * size + i) as usize][0]).collect();
            let (lo, hi) = r.iter().fold((f32::MAX, f32::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)));
            (hi - lo) / hi
        };
        assert!(inner_spread > 0.3, "the floor's own shadows inside the rim: {inner_spread:.3}");
        assert!(spread(&dry, 0) > 0.3, "a dry rim is left as it is: {:.3}", spread(&dry, 0));
    }
}

#[cfg(test)]
mod under_ground_census {
    use super::*;

    #[test]
    #[ignore]
    fn steps_from_under_the_ground() {
        let map = fixtures::height_map(1024, fixtures::hills);
        let chain = levels(&map);
        for under in [0.0f32, 0.03] {
            let mut buckets = std::collections::BTreeMap::<i32, (u32, u32, u32, u32)>::new();
            for (e, d) in fixtures::rays(20000, 5) {
                let q = (Vec2::new(e.x, e.z) - map.min) / (map.max - map.min) * Vec2::new(chain[0].width as f32, chain[0].height as f32);
                let e = Vec3::new(e.x, height_at(&chain[0], q) - under, e.z);
                let r = trace(&chain, map.min, map.max, map.top, e, d);
                let el = (d.y.asin().to_degrees() / 5.0).floor() as i32 * 5;
                let b = buckets.entry(el).or_default();
                b.0 += 1;
                b.1 += r.steps;
                b.2 = b.2.max(r.steps);
                b.3 += r.t.is_some() as u32;
            }
            for (el, (n, s, m, h)) in buckets {
                eprintln!("under {under}: elev {el:>4}..{:>3}: rays {n:>5}  mean steps {:>5.1}  max {m:>3}  hit {:>3.0}%", el + 5, s as f32 / n as f32, 100.0 * h as f32 / n as f32);
            }
        }
    }
}
