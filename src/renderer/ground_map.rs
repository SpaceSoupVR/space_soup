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
fn layer_weights(settings: &TerrainMaterialUniform, splat: Option<&TerrainImage>, uv: Vec2, world_y: f32, slope_deg: f32) -> [f32; 4] {
    if settings.use_splat > 0.5 {
        if let Some(s) = splat {
            let w = sample_rgba(s, uv);
            let total = (w[0] + w[1] + w[2] + w[3]).max(0.001);
            return w.map(|v| v / total);
        }
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

/// The map on the GPU: half floats with a full box-filtered mip chain, so a
/// distant patch of ground is read averaged rather than aliased. The height
/// in A is only ever read at level 0.
pub fn upload(device: &Device, queue: &Queue, map: &GroundMap) -> TextureView {
    let levels = mip_levels(map.width.max(map.height));
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("ground_map"),
        size: wgpu::Extent3d { width: map.width, height: map.height, depth_or_array_layers: 1 },
        mip_level_count: levels,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let mut level: Vec<[f32; 4]> = map.texels.clone();
    let (mut w, mut h) = (map.width, map.height);
    for mip in 0..levels {
        let bytes: Vec<u16> = level.iter().flat_map(|t| t.map(crate::renderer::sky::f32_to_f16)).collect();
        queue.write_texture(
            wgpu::TexelCopyTextureInfo { texture: &tex, mip_level: mip, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            bytemuck::cast_slice(&bytes),
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w * 8), rows_per_image: Some(h) },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        if w == 1 && h == 1 {
            break;
        }
        let (nw, nh) = ((w / 2).max(1), (h / 2).max(1));
        let mut next = Vec::with_capacity((nw * nh) as usize);
        for y in 0..nh {
            for x in 0..nw {
                let mut acc = [0.0f32; 4];
                for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    let (sx, sy) = ((2 * x + dx).min(w - 1), (2 * y + dy).min(h - 1));
                    let t = level[(sy * w + sx) as usize];
                    for c in 0..4 {
                        acc[c] += t[c] * 0.25;
                    }
                }
                next.push(acc);
            }
        }
        level = next;
        w = nw;
        h = nh;
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
}
