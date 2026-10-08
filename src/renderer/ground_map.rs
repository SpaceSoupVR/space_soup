//! THE GROUND AS A REFLECTION SEES IT: one top-down picture of the terrain --
//! what it looks like lit, and how high it is -- for a reflected ray that has
//! left the rooms to land on.
//!
//! # Why the photographs could not do this
//!
//! A reflection that leaves the building meets either the ground or the sky.
//! The level's one outdoor photograph is taken from above the roofs, on one
//! side of the building, so the ground beside every other wall is hidden from
//! it behind the building itself -- and that ground is exactly what the base of
//! a polished wall, or a ceiling seen through a doorway, reflects. The trace
//! instead ran the ray into the outdoor volume's BOX, whose floor lies metres
//! under the terrain, and read the photograph toward that point: sunlit ground
//! or roof. The shaded wall facing the lake glowed at its foot as if lit from
//! underneath, and the ceiling mirrored the ground outside the front door as a
//! blocky patchwork (headset, 2026-09-28).
//!
//! # What this is instead
//!
//! The terrain seen from straight above: RGB is the light it sends back, as the
//! terrain shader lights it -- its layers' albedo, the sky through its baked sky
//! visibility, the sun through its baked shadow -- and A is its height. A ray is
//! marched against A until it passes below the ground and read from RGB there
//! (`outdoor_radiance` in the lights block). Built at load from what the
//! renderer already holds, so it cannot drift from the ground it describes, and
//! it sees everything: nothing hides one patch of ground from a top-down view.
//!
//! The ground is rough enough that the light it returns barely depends on where
//! it is seen from, which is what lets one picture serve every viewpoint.
//! What it leaves out, deliberately: the layers' normal-map detail and texture
//! grain (averaged into each layer's mean colour), the ground's own specular,
//! and moving shadows. Buildings are not in it -- a reflection that should meet
//! another building's wall outdoors passes behind it.

use glam::{Vec2, Vec3};
use wgpu::{Device, Queue, TextureView};

use crate::renderer::sky::{SkyIrradiance, SkySun};
use crate::renderer::terrain_pipeline::{TerrainImage, TerrainMaterialUniform, FALLBACK_LAYER_COLOURS};

/// Texels a side. 110 m of test_room's terrain at 1024 is 11 cm a texel --
/// finer than any baked shadow edge it carries -- for 8 MB of half floats.
pub const GROUND_MAP_SIZE: u32 = 1024;

/// Ground seeing less of the sky than this lies UNDER a building's floor --
/// sealed in, where nothing sees it. Outdoor ground never does: at the foot of
/// a wall half the sky is still overhead (test_room reads 0.54 there). See
/// [`fill_under_buildings`].
const BURIED_SKY_VIS: f32 = 0.05;

/// How far, in texels, the ground around a building is carried in under it.
const BURIED_FILL_TEXELS: usize = 8;

/// A dielectric's normal-incidence reflectance: the share of light the ground's
/// diffuse term gives up to its specular, as `shade_material_lamps` takes it.
const F0: f32 = 0.04;

/// The terrain's heights on a regular grid over its footprint, in world
/// metres: sample `(i, j)` stands at `min + (i, j) * (max - min) / (size - 1)`.
pub struct HeightGrid {
    pub min: Vec2,
    pub max: Vec2,
    pub width: u32,
    pub height: u32,
    pub heights: Vec<f32>,
}

impl HeightGrid {
    /// `height_at` sampled on a `width` x `height` grid spanning `min..max`.
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

    /// Height at world `(x, z)`, bilinear, clamped to the grid.
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

    /// The ground's upward normal at `(x, z)`, by central differences over one
    /// grid step.
    pub fn normal(&self, x: f32, z: f32) -> Vec3 {
        let dx = (self.max.x - self.min.x) / (self.width - 1) as f32;
        let dz = (self.max.y - self.min.y) / (self.height - 1) as f32;
        let sx = (self.at(x + dx, z) - self.at(x - dx, z)) / (2.0 * dx);
        let sz = (self.at(x, z + dz) - self.at(x, z - dz)) / (2.0 * dz);
        Vec3::new(-sx, 1.0, -sz).normalize()
    }
}

/// Everything the ground's light is made of, as the renderer holds it.
pub struct GroundInputs<'a> {
    pub heights: &'a HeightGrid,
    /// The sky's ambient, WITHOUT its sun. See `sky::Sky::irradiance`.
    pub sky: &'a SkyIrradiance,
    /// The sky's sun, in world space. See `sky::Sky::sun`.
    pub sun: Option<&'a SkySun>,
    /// The ground's baked map over the footprint: sky visibility in red, the
    /// sun's shadow as a distance field in green and blue. See the terrain
    /// shader's `ground_map`.
    pub sky_occlusion: Option<&'a TerrainImage>,
    pub layers: &'a [Option<TerrainImage>],
    pub splat: Option<&'a TerrainImage>,
    pub settings: &'a TerrainMaterialUniform,
}

/// The picture: `texels` row-major, x fastest, row 0 at `min.y` (world z).
pub struct GroundMap {
    pub width: u32,
    pub height: u32,
    /// RGB the light the ground sends back, A its world height.
    pub texels: Vec<[f32; 4]>,
    pub min: Vec2,
    pub max: Vec2,
    /// The highest the ground rises: above it, a rising ray can only meet sky.
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

/// A layer's mean colour, linear: what its texture averages to from far off.
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

/// Bilinear RGBA of an 8-bit image at `uv` in 0..1, each channel 0..1.
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

/// Bilinear like [`sample_rgba`], but over OPEN ground only: a tap under a
/// building's floor (red below [`BURIED_SKY_VIS`]) is left out and the rest
/// renormalised, so the ground at a wall's foot reads the ground beside it and
/// not a blend with the sealed dark under the floor. `None` when every tap is
/// buried.
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

/// The terrain shader's `layer_weights`, transcribed: the painted splat map
/// when there is one, else rock by slope and high ground by height.
///
/// WHEN THERE IS ONE, as `TerrainMaterial::new` decides the shader's
/// `use_splat` -- not by the settings' own `use_splat`, which only the
/// material's copy has set: the renderer's stays 0. Read from it, the map
/// coloured test_room's sand beach as grass, and so did every reflection of
/// it and the water's view of the bed past the map's edge (2026-10-07).
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

/// How much of the sky and of the sun reach the ground at `uv`, and whether
/// it lies buried under a floor: the terrain shader's decode of its baked map,
/// read over open ground only ([`sample_open`]). `(1, 1, false)` where there is
/// no map -- the headset then shades that ground from the live shadow map,
/// which knows no more about a building's shadow than this does.
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

/// The ground map for these inputs, `size` texels a side.
pub fn build(inputs: &GroundInputs, size: u32) -> GroundMap {
    let size = size.max(1);
    let g = inputs.heights;
    let extent = g.max - g.min;
    let means: Vec<Vec3> = (0..4)
        .map(|i| layer_mean(inputs.layers.get(i).and_then(|l| l.as_ref()), FALLBACK_LAYER_COLOURS[i]))
        .collect();
    let sun = inputs.sun.map(|s| (Vec3::from(s.direction).normalize_or_zero(), Vec3::from(s.light_rgb)));
    let mut texels = Vec::with_capacity((size * size) as usize);
    let mut buried = Vec::with_capacity((size * size) as usize);
    let mut top = f32::MIN;
    for j in 0..size {
        for i in 0..size {
            let uv = Vec2::new((i as f32 + 0.5) / size as f32, (j as f32 + 0.5) / size as f32);
            let (x, z) = (g.min.x + extent.x * uv.x, g.min.y + extent.y * uv.y);
            let y = g.at(x, z);
            let n = g.normal(x, z);
            let slope_deg = n.y.clamp(-1.0, 1.0).acos().to_degrees();
            let w = layer_weights(inputs.settings, inputs.splat, uv, y, slope_deg);
            let albedo = means[0] * w[0] + means[1] * w[1] + means[2] * w[2] + means[3] * w[3];
            // Wet sand, as the terrain shader darkens it.
            let s = inputs.settings;
            let albedo = albedo * (1.0 - 0.45 * (1.0 - smoothstep(s.wet_line, s.wet_line + s.wet_band, y)));
            let (sky_vis, sun_vis, under_a_floor) = sky_and_sun(inputs.sky_occlusion, uv);
            buried.push(under_a_floor);
            let e = inputs.sky.evaluate([n.x, n.y, n.z]);
            let mut light = Vec3::from(e) * sky_vis;
            if let Some((to_sun, rgb)) = sun {
                light += rgb * n.dot(to_sun).max(0.0) * sun_vis;
            }
            let rgb = albedo * light * (1.0 - F0);
            top = top.max(y);
            texels.push([rgb.x, rgb.y, rgb.z, y]);
        }
    }
    fill_under_buildings(&mut texels, &mut buried, size as usize, size as usize);
    GroundMap { width: size, height: size, texels, min: g.min, max: g.max, top }
}

/// THE GROUND UNDER A BUILDING TAKES THE COLOUR OF THE GROUND AROUND IT.
///
/// Buried under a floor, it is black -- no sky, no sun -- and nothing ever sees
/// it; but a texel's filter and every coarser mip reach across the wall line,
/// and the black bled onto the sunlit grass at the building's foot: the ceiling
/// mirrored the ground outside the front door with a dark band along the door
/// (offline, 2026-09-28). Filled from outside inward, a texel ring at a time,
/// as a lightmap's gutter is. The height is left as it is.
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

/// Mip levels for a square of `size` texels.
fn mip_levels(size: u32) -> u32 {
    32 - size.max(1).leading_zeros()
}

/// The trace's finest cells: level 4, sixteen texels a side -- 1.7 m on
/// test_room. It reads the ground itself across one of these at
/// [`GROUND_TRACE_READINGS`] points, 43 cm apart: one step of the height grid
/// the map is built from (257 samples over 110 m), so between two readings the
/// ground is a single smooth patch and the crossing is interpolated between
/// them. Finer cells cost more steps for the same answers: over test_room-
/// shaped hills (`fixtures::hills`), cells of 43 cm read three times took 8.7
/// steps a ray (53 at worst) and 11.9 texture reads; these take 6.5 (36) and
/// 11.0, and meet the ground at the same points (step_census, 2026-09-29).
pub const GROUND_TRACE_FINEST_LEVEL: u32 = 4;

/// Readings of the ground across a finest cell, its two sides included.
pub const GROUND_TRACE_READINGS: u32 = 5;

/// The level a trace starts from: 64 texels a side, 7 m on test_room. Rays
/// off a building's walls take the fewest steps from here -- coarse enough to
/// skip open field at once, fine enough not to spend their first steps coming
/// down (levels 5 and 7 cost 7 and 1 percent more).
pub const GROUND_TRACE_START_LEVEL: u32 = 6;

/// The most cells a trace visits. The worst of 20,000 rays off test_room's
/// walls takes 50; one that ran out would be read as passing over.
pub const GROUND_TRACE_MAX_STEPS: u32 = 64;

/// Metres over each cell's highest ground its level stores, so the GPU's
/// filtering and half-float rounding cannot put a reading above it.
const GROUND_MAX_MARGIN: f32 = 0.02;

/// A half float's value back as a single: the exact value the GPU reads.
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

/// `v` as the half float the upload writes (`sky::f32_to_f16` truncates).
fn as_half(v: f32) -> f32 {
    f16_to_f32(crate::renderer::sky::f32_to_f16(v))
}

/// The smallest half float not below `v`: a bound stays a bound once stored.
fn half_at_least(v: f32) -> f32 {
    let h = crate::renderer::sky::f32_to_f16(v);
    if f16_to_f32(h) >= v {
        return f16_to_f32(h);
    }
    // Truncation lost the fraction: one step away from zero for a positive
    // value, one step toward it for a negative one -- up, either way.
    f16_to_f32(if h & 0x8000 == 0 { h + 1 } else { h - 1 })
}

/// One level of the map as it is uploaded, in half-float values.
pub struct GroundLevel {
    pub width: u32,
    pub height: u32,
    pub texels: Vec<[f32; 4]>,
}

/// EVERY LEVEL OF THE MAP, as the GPU holds it.
///
/// RGB is box-filtered down the chain, so a distant patch of ground is read
/// averaged rather than aliased. A is the height at level 0, and above it the
/// HIGHEST ground under each texel -- the bound that lets a trace skip a whole
/// cell the ray passes over (see [`trace`]). Taken over every level-0 texel a
/// bilinear read inside the cell can reach -- one beyond each side -- plus
/// [`GROUND_MAX_MARGIN`], and rounded up: a bound that rounding put below the
/// ground would let a ray through a hill.
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
                        // From level 0 itself, one texel wider on every side;
                        // each coarser level is then the highest of its four.
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

/// The ground's height at `q`, in level-0 texels from the map's corner:
/// bilinear between texel centres, clamped at the edges -- what the shader's
/// `textureSampleLevel(ground_map, probe_samp, uv, 0.0).a` returns.
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

/// Where a trace ended: the distance to the ground along the ray, `None` for
/// sky; how many cells it visited on the way, and how many texture reads
/// that took -- a cell's highest ground each, and the readings of the ground
/// itself in the finest ones.
#[derive(Clone, Copy, Debug)]
pub struct GroundTrace {
    pub t: Option<f32>,
    pub steps: u32,
    pub reads: u32,
}

/// THE GROUND A RAY MEETS, from `e` along unit `d` over `levels` placed at
/// `min..max`: the shader's `ground_trace`, step for step.
///
/// A walk through the cells the ray crosses, coarse where it passes high over
/// the ground and fine where it comes close. A cell whose highest ground stays
/// below the ray on both sides is passed over whole, and the walk climbs a
/// level once it leaves its parent; one the ray may touch is entered a level
/// finer, from where the ray has come down to its highest ground. At the
/// finest level the ground itself is read evenly from there to the cell's far
/// side, and the crossing interpolated between the first reading below the
/// ray and the one before it.
///
/// It replaced a march in doubling steps -- 12.5 cm out to 256 m -- that
/// tested the ground at 32 m and next at 64 m, past test_room's edge, so the
/// hills between were never tested: a reflection's horizon was the height of
/// the ground 32 m out, it stepped wherever that reading changed hands, and it
/// crawled as the head moved (headset, 2026-09-29).
pub fn trace(levels: &[GroundLevel], min: Vec2, max: Vec2, top: f32, e: Vec3, d: Vec3) -> GroundTrace {
    trace_with(levels, min, max, top, e, d, TraceShape::SHIPPED)
}

/// How a trace walks: its finest level, the level it starts from, how many
/// readings of the ground it takes across a finest cell, and its budget.
#[derive(Clone, Copy, Debug)]
pub struct TraceShape {
    pub finest: u32,
    pub start: u32,
    pub readings: u32,
    pub max_steps: u32,
}

impl TraceShape {
    /// The shader's, from the `GROUND_TRACE_*` constants.
    pub const SHIPPED: Self = Self {
        finest: GROUND_TRACE_FINEST_LEVEL,
        start: GROUND_TRACE_START_LEVEL,
        readings: GROUND_TRACE_READINGS,
        max_steps: GROUND_TRACE_MAX_STEPS,
    };
}

/// [`trace`] walked as `shape` says.
pub fn trace_with(levels: &[GroundLevel], min: Vec2, max: Vec2, top: f32, e: Vec3, d: Vec3, shape: TraceShape) -> GroundTrace {
    let miss = GroundTrace { t: None, steps: 0, reads: 0 };
    let Some(base) = levels.first() else { return miss };
    let size = base.width as f32;
    let scale = size / (max - min);
    let q0 = (Vec2::new(e.x, e.z) - min) * scale;
    let raw = Vec2::new(d.x, d.z) * scale;
    // An axis the ray barely moves along never bounds a cell it is in.
    let dq = Vec2::new(if raw.x.abs() < 1e-6 { 1e-6 } else { raw.x }, if raw.y.abs() < 1e-6 { 1e-6 } else { raw.y });
    let inv = Vec2::ONE / dq;
    // On the map, and below its highest ground.
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
    // A thousandth of a texel along the ray: a point on a border is in the
    // cell ahead.
    let nudge = Vec2::new(dq.x.signum(), dq.y.signum()) * 1e-3;
    let gaps = shape.readings.max(2) - 1;
    let mut t = t_in;
    let mut steps = 0;
    let mut reads = 0;
    // A ray that starts under the map's ground meets nothing until it has
    // risen above it (`ground_trace_until`).
    let mut risen = false;
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
            // Nothing in this cell before the ray is down to its highest ground.
            let t_top = if y_in > highest { t + (y_in - highest) / -d.y } else { t };
            if level > finest {
                level -= 1;
                t = t_top;
                continue;
            }
            // The ground itself, read evenly from there to the cell's far side;
            // the crossing between the first reading below it and the one before.
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
        // Over this cell: on to the next, a level coarser once it is in
        // another parent.
        let next = ((q0 + dq * t_exit + nudge) / cell).floor();
        if (next * 0.5).floor() != (c * 0.5).floor() {
            level = (level + 1).min(top_level);
        }
        t = t_exit;
    }
    GroundTrace { t: None, steps, reads }
}

/// The map on the GPU: [`levels`] as half floats.
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

/// What is bound when a level has no ground: one texel, never read -- the
/// shader is told there is no ground by a top of -1e30 (see
/// `ProbeUpload::ground`).
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

/// Ground, rays and a brute-force reference, for the trace's tests here and
/// its GPU test in the lights block.
#[cfg(test)]
pub(crate) mod fixtures {
    use super::*;

    /// test_room's ground in shape: flat for 25 m around the buildings, then
    /// hills rising to 5-8 m by 50 m out.
    pub fn hills(x: f32, z: f32) -> f32 {
        let rise = smoothstep(25.0, 50.0, (x * x + z * z).sqrt());
        rise * (5.0 + 2.5 * (x * 0.13).sin() * (z * 0.11).cos() + (x * 0.37 + z * 0.23).sin())
    }

    /// A map `size` texels a side over test_room's 110 m, heights only.
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

    /// `n` rays off a building's walls: from 0-3.4 m up, anywhere within 12 m
    /// of the middle, every way round, from 35 degrees down to 20 up.
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
        /// How close the ray came to the ground without passing below it, or
        /// how little it passed below before coming back out: what separates
        /// a real disagreement from a ray that only grazes.
        pub graze: f32,
    }

    /// The ground along the ray read every centimetre, to where it leaves
    /// the map or rises above its top; the crossing interpolated.
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
                // How deep the ray goes over the next metre.
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
        // A uniform sky of radiance 1: band 0 only.
        s.sh[0] = [3.5449077; 3];
        s
    }

    /// The terrain shader's rules, pinned to its text: this transcribes them,
    /// and the reflection of the ground must be the ground the headset draws.
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
            &GroundInputs { heights: &g, sky: &sky(), sun: None, sky_occlusion: None, layers: &[], splat: None, settings: &s },
            8,
        );
        let t = map.texels[(3 * 8 + 5) as usize];
        let (x, z) = (10.0 * 5.5 / 8.0, 10.0 * 3.5 / 8.0);
        assert!((t[3] - (0.1 * x + 0.2 * z)).abs() < 1e-4, "height {} at ({x}, {z})", t[3]);
        assert!((map.top - map.texels.iter().map(|t| t[3]).fold(f32::MIN, f32::max)).abs() < 1e-6);
        let n = g.normal(5.0, 5.0);
        assert!((n - Vec3::new(-0.1, 1.0, -0.2).normalize()).length() < 1e-4, "{n:?}");
    }

    /// Open flat ground under a uniform sky and an overhead sun returns its
    /// albedo times what arrives -- and ground the baked map shadows from the
    /// sun returns only the sky's share.
    #[test]
    fn ground_in_the_buildings_shadow_is_darker_by_exactly_the_sun() {
        let g = flat(0.0);
        let s = TerrainMaterialUniform::default();
        let sun = SkySun { direction: [0.0, 1.0, 0.0], light_rgb: [4.0, 4.0, 4.0], texels: 1, energy_fraction: 0.5 };
        let grass = TerrainImage { width: 1, height: 1, rgba: vec![128, 128, 128, 255] };
        let layers = [Some(grass)];
        // Left half fully sky-visible and sunlit (sun distance +range), right
        // half fully sky-visible and in shadow (-range).
        let occ = TerrainImage {
            width: 2,
            height: 1,
            rgba: vec![255, 255, 0, 0, 255, 0, 0, 0],
        };
        let map = build(
            &GroundInputs { heights: &g, sky: &sky(), sun: Some(&sun), sky_occlusion: Some(&occ), layers: &layers, splat: None, settings: &s },
            8,
        );
        let albedo = srgb_to_linear(128);
        let e = sky().evaluate([0.0, 1.0, 0.0])[0];
        let lit = map.texels[0][0];
        let shaded = map.texels[7][0];
        assert!((lit - albedo * (e + 4.0) * (1.0 - F0)).abs() < 1e-3, "sunlit ground {lit}");
        assert!((shaded - albedo * e * (1.0 - F0)).abs() < 1e-3, "shadowed ground {shaded}");
    }

    /// The black ground under a floor must not bleed onto the grass outside:
    /// the texels beside the wall line take the outside's colour.
    #[test]
    fn ground_under_a_building_takes_the_colour_around_it() {
        let g = flat(0.0);
        let s = TerrainMaterialUniform::default();
        let grass = TerrainImage { width: 1, height: 1, rgba: vec![128, 128, 128, 255] };
        let layers = [Some(grass)];
        // Left half open sky and sun; right half sealed under a floor.
        let occ = TerrainImage {
            width: 4,
            height: 1,
            rgba: vec![255, 255, 0, 0, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        };
        let map = build(
            &GroundInputs { heights: &g, sky: &sky(), sun: None, sky_occlusion: Some(&occ), layers: &layers, splat: None, settings: &s },
            8,
        );
        let open = map.texels[0][0];
        assert!(open > 0.1);
        // Up to the wall line and on under the floor, the open ground's colour:
        // no dark band where the two meet.
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

    /// Wherever a bilinear read of the ground can land inside a coarse texel,
    /// the ground is no higher than that texel says -- and not much lower,
    /// or the trace would enter cells it has no need to.
    #[test]
    fn every_level_bounds_the_ground_under_it() {
        // Bumpier than any real terrain, so a read between texels matters.
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
                    // No looser than the level-0 texels a read can reach --
                    // one beyond each side -- plus the margin.
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

    /// A ray from a hollow finer than the map's texels -- a puddle, under the
    /// map's ground there -- meets nothing until it has risen above it: it
    /// went straight to the ground where it started, and a mirror puddle
    /// showed the grass (headset, 2026-10-08). Rising away it meets nothing;
    /// rising over a hill it still meets the hill.
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
            // Straight up: nothing above but the sky.
            let up = trace(&chain, map.min, map.max, map.top, e, Vec3::Y);
            assert!(up.t.is_none(), "from under the ground at {e}, straight up met {:?}", up.t);
            // Nearly level: whatever it meets is past the point it rose clear.
            let d = Vec3::new(0.7, 0.05, 0.7).normalize();
            if let Some(t) = trace(&chain, map.min, map.max, map.top, e, d).t {
                assert!(t > 0.05, "from under the ground at {e}, met it where it started ({t})");
                checked += 1;
            }
        }
        assert!(checked > 0, "no level ray met a hill");
    }

    /// The trace against brute force: the same ground read every centimetre
    /// along the ray. They must meet the ground at the same point, and differ
    /// on hit or miss only for a ray that grazes it within a few centimetres.
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

    /// The case that broke on the headset: flat ground to 40 m and a ridge
    /// beyond it, and a ray from eye height rising just too slowly to clear
    /// it. The old march read the ground at 32 m and next past the edge.
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

    /// Steep ground takes the rock layer, as the shader blends it by slope.
    #[test]
    fn a_cliff_is_rock_and_the_plain_is_grass() {
        let g = HeightGrid::sample(Vec2::new(0.0, 0.0), Vec2::new(10.0, 10.0), 11, 11, |x, _| if x > 5.0 { (x - 5.0) * 3.0 } else { 0.0 });
        let s = TerrainMaterialUniform::default();
        let layers = [
            Some(TerrainImage { width: 1, height: 1, rgba: vec![0, 255, 0, 255] }),
            Some(TerrainImage { width: 1, height: 1, rgba: vec![255, 0, 0, 255] }),
        ];
        let map = build(
            &GroundInputs { heights: &g, sky: &sky(), sun: None, sky_occlusion: None, layers: &layers, splat: None, settings: &s },
            10,
        );
        let plain = map.texels[5 * 10 + 1];
        let cliff = map.texels[5 * 10 + 8];
        assert!(plain[1] > plain[0], "the plain is the grass layer: {plain:?}");
        assert!(cliff[0] > cliff[1], "the cliff is the rock layer: {cliff:?}");
    }

    /// A painted splat map decides the layers, as it does in the terrain
    /// shader -- with the settings as the renderer holds them, whose own
    /// `use_splat` is never set.
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
        // All sediment: the flat plain painted as a beach.
        let splat = TerrainImage { width: 2, height: 2, rgba: [0u8, 0, 0, 255].repeat(4) };
        let map = build(
            &GroundInputs { heights: &g, sky: &sky(), sun: None, sky_occlusion: None, layers: &layers, splat: Some(&splat), settings: &s },
            10,
        );
        let t = map.texels[5 * 10 + 5];
        assert!(t[0] > 0.0 && t[1] > 0.0 && t[2] < 1e-6, "the plain is the painted sediment, not grass by slope: {t:?}");
        assert!((t[0] - t[1]).abs() < 1e-3 * t[0], "sediment's red and green, equally: {t:?}");
    }
}

/// THE MEASUREMENTS BEHIND `GROUND_TRACE_*`, run by hand:
/// `cargo test --release --lib step_census -- --ignored --nocapture`.
#[cfg(test)]
mod step_census {
    use super::*;

    /// Steps, texture reads and agreement with brute force, for several
    /// finest levels, readings and starting levels.
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

    /// Where the shipped walk's steps go, by the ray's elevation: skims just
    /// above the hills cost the most.
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
}
