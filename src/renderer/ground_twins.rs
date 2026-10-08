//! THE GROUND'S SLOPE TWINS: the ground's scene readers without the steep
//! ground's code, drawn for the triangles no steep pixel can come from.
//!
//! WHY. The cliff texturing of 2026-10-08 -- the side planes' warp and bands,
//! the colour's and the normal's second scale, the minor plane's normal near a
//! 45-degree turn -- took every ground reader from 18-20 registers to 27
//! (occupancy 62% -> 37%) and the full one to 3,665 instructions, past the
//! Quest's instruction cache (wall #20), in every view of the ground, flat or
//! not: outdoors_field drew in 21.7 ms against 13.2 on 10-06. All of it runs
//! only where `slope_deg >= mat.biplanar_start_deg`. PIPESTATS of the readers
//! with that test made false (`ground_cuts` `gentle`): 18 registers, 62%,
//! 2,719 instructions for the outdoor one (build of 2026-10-08 10:44).
//!
//! THE SAME PICTURE. A pixel's slope is that of its interpolated normal, and
//! the normalised blend of a triangle's vertex normals with positive weights
//! lies in the cone they span: no steeper than its steepest vertex. So a
//! triangle whose every vertex is gentler than the threshold, by a margin for
//! the centre of an MSAA edge pixel just outside it, draws no steep pixel, and
//! the twin -- the reader with that test false -- draws it as the reader does.
//!
//! HOW. Each terrain chunk's triangles are reordered once, its gentle ones
//! first ([`SlopeSplit`]); the chunk keeps its range, so every other pass and
//! test (shadows, culling, the weather's chunks) is unchanged, and the scene
//! pass draws each chunk in two ranges, the gentle one with the twin.

use crate::renderer::cuboid::SolidVertex;
use crate::renderer::shadow::CasterChunk;
use crate::renderer::terrain_pipeline::{terrain_shader_for, TerrainPipeline, TerrainRole};
use wgpu::{BindGroupLayout, Device, TextureFormat};

/// The ground shader's two steep tests, as `terrain_shader`'s `sample_frame`
/// writes them, and the twin's.
pub(crate) const GENTLE_EDITS: &[(&str, &str)] = &[
    ("    f.biplanar  = select(0.0, 1.0, slope_deg >= mat.biplanar_start_deg);\n", "    f.biplanar  = 0.0;\n"),
    ("    let steep = slope_deg >= mat.biplanar_start_deg;\n", "    let steep = false;\n"),
];

/// Degrees under the threshold every vertex of a twin's triangle must be: room
/// for the centre of an MSAA edge pixel, just outside the triangle, where its
/// normal is extrapolated.
pub const GENTLE_MARGIN_DEG: f32 = 5.0;

/// `src` (any ground shader, any twin) without its steep code, or `None` when
/// its steep tests are not there once each.
pub fn gentle_shader(src: &str) -> Option<String> {
    crate::renderer::ground_cuts::with_cut(src, GENTLE_EDITS)
}

/// Whether a triangle with these vertex normals can draw no steep pixel under
/// a `threshold_deg` slope (`TerrainMaterialUniform::biplanar_start_deg`).
pub fn triangle_is_gentle(normals: [[f32; 3]; 3], threshold_deg: f32) -> bool {
    let limit = threshold_deg - GENTLE_MARGIN_DEG;
    if limit <= 0.0 {
        return false;
    }
    let cos_limit = limit.to_radians().cos();
    normals.iter().all(|n| {
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        len > 0.0 && n[1] / len > cos_limit
    })
}

/// The ground shader's two steep tests made true: the twin for triangles
/// every pixel of which is steep (`ground_cuts` `steep`: 24 registers, 50%,
/// against the reader's 27, 37%).
pub(crate) const STEEP_EDITS: &[(&str, &str)] = &[
    ("    f.biplanar  = select(0.0, 1.0, slope_deg >= mat.biplanar_start_deg);\n", "    f.biplanar  = 1.0;\n"),
    ("    let steep = slope_deg >= mat.biplanar_start_deg;\n", "    let steep = true;\n"),
];

/// `src` with only the steep ground's code, or `None` when its steep tests
/// are not there once each.
pub fn steep_shader(src: &str) -> Option<String> {
    crate::renderer::ground_cuts::with_cut(src, STEEP_EDITS)
}

/// Whether a triangle with these vertex normals can draw no gentle pixel.
/// Not its gentlest vertex's slope: two steep normals leaning opposite ways
/// blend to a level one. Every normal of the triangle (a blend of its
/// vertices') lies within the cap round their mean as wide as the farthest of
/// them, so its slope is at least the mean's less that width.
pub fn triangle_is_steep(normals: [[f32; 3]; 3], threshold_deg: f32) -> bool {
    let unit = |n: [f32; 3]| {
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        (len > 0.0).then(|| [n[0] / len, n[1] / len, n[2] / len])
    };
    let (Some(a), Some(b), Some(c)) = (unit(normals[0]), unit(normals[1]), unit(normals[2])) else {
        return false;
    };
    let Some(m) = unit([a[0] + b[0] + c[0], a[1] + b[1] + c[1], a[2] + b[2] + c[2]]) else {
        return false;
    };
    let angle = |u: [f32; 3]| (u[0] * m[0] + u[1] * m[1] + u[2] * m[2]).clamp(-1.0, 1.0).acos().to_degrees();
    let spread = angle(a).max(angle(b)).max(angle(c));
    let slope = m[1].clamp(-1.0, 1.0).acos().to_degrees();
    spread < 60.0 && slope - spread >= threshold_deg + GENTLE_MARGIN_DEG
}

/// THE TERRAIN'S INDICES WITH EACH CHUNK'S GENTLE TRIANGLES FIRST, and how
/// many of each chunk's indices they are. Built once per terrain and
/// threshold; `key` tells a new terrain from the one it was built for.
pub struct SlopeSplit {
    key: (usize, usize, usize, usize, u32, u64),
    /// The terrain's index buffer, reordered within each chunk.
    pub indices: Vec<u32>,
    /// Per chunk, as handed: how many of its indices, from its first, are
    /// gentle triangles'.
    pub gentle: Vec<u32>,
    /// Gentle indices of all.
    pub gentle_total: u64,
    /// Per chunk: how many of its indices, up to its last, are steep
    /// triangles' (`triangle_is_steep`); the rest, between, are mixed.
    pub steep: Vec<u32>,
    pub steep_total: u64,
}

impl SlopeSplit {
    fn key_of(verts: &[SolidVertex], idx: &[u32], chunks: &[CasterChunk], threshold_deg: f32) -> (usize, usize, usize, usize, u32, u64) {
        // A sample of the data as well as where it lies: a terrain loaded
        // again can land at the same address.
        let mut h: u64 = 1469598103934665603;
        let mut mix = |v: u64| {
            h ^= v;
            h = h.wrapping_mul(1099511628211);
        };
        for c in chunks {
            mix(c.first_index as u64);
            mix(c.index_count as u64);
        }
        let step = (verts.len() / 61).max(1);
        for v in verts.iter().step_by(step) {
            mix(v.position[1].to_bits() as u64);
            mix(v.normal[1].to_bits() as u64);
        }
        let step = (idx.len() / 61).max(1);
        for i in idx.iter().step_by(step) {
            mix(*i as u64);
        }
        (verts.as_ptr() as usize, verts.len(), idx.as_ptr() as usize, idx.len(), threshold_deg.to_bits(), h)
    }

    /// Whether this split is the one for this terrain.
    pub fn is_for(&self, verts: &[SolidVertex], idx: &[u32], chunks: &[CasterChunk], threshold_deg: f32) -> bool {
        self.key == Self::key_of(verts, idx, chunks, threshold_deg)
    }

    /// `idx` with each chunk's gentle triangles first. No chunks: the whole
    /// terrain as one.
    pub fn new(verts: &[SolidVertex], idx: &[u32], chunks: &[CasterChunk], threshold_deg: f32) -> Self {
        let whole = [CasterChunk { first_index: 0, index_count: idx.len() as u32, min: glam::Vec3::ZERO, max: glam::Vec3::ZERO }];
        let ranges: &[CasterChunk] = if chunks.is_empty() { &whole } else { chunks };
        let mut indices = idx.to_vec();
        let mut gentle = Vec::with_capacity(ranges.len());
        let mut gentle_total = 0u64;
        let mut steep = Vec::with_capacity(ranges.len());
        let mut steep_total = 0u64;
        let normal = |i: u32| verts.get(i as usize).map_or([0.0; 3], |v| v.normal);
        for c in ranges {
            let (start, end) = (c.first_index as usize, (c.first_index + c.index_count) as usize);
            if end > idx.len() || (end - start) % 3 != 0 {
                gentle.push(0);
                steep.push(0);
                continue;
            }
            let tris = &idx[start..end];
            let (mut soft, mut mixed, mut hard) = (Vec::with_capacity(tris.len()), Vec::new(), Vec::new());
            for t in tris.chunks_exact(3) {
                let n = [normal(t[0]), normal(t[1]), normal(t[2])];
                if triangle_is_gentle(n, threshold_deg) {
                    soft.extend_from_slice(t);
                } else if triangle_is_steep(n, threshold_deg) {
                    hard.extend_from_slice(t);
                } else {
                    mixed.extend_from_slice(t);
                }
            }
            gentle.push(soft.len() as u32);
            gentle_total += soft.len() as u64;
            steep.push(hard.len() as u32);
            steep_total += hard.len() as u64;
            let (a, b) = (start + soft.len(), start + soft.len() + mixed.len());
            indices[start..a].copy_from_slice(&soft);
            indices[a..b].copy_from_slice(&mixed);
            indices[b..end].copy_from_slice(&hard);
        }
        Self { key: Self::key_of(verts, idx, chunks, threshold_deg), indices, gentle, gentle_total, steep, steep_total }
    }
}

/// The ground's gentle scene readers, in the weather twins' order `[full,
/// spotless, baked, baked_spotless]` (`XrRenderer::terrain_twin`), from the
/// shaders the shipped ones are built from; `weathered` the weather's twin of
/// each, with its group 2 and its kinds (`weather::with_weather_kinds`).
/// `None` when the ground's steep tests are not there to take out.
#[allow(clippy::too_many_arguments)]
pub fn gentle_readers(
    device: &Device,
    format: TextureFormat,
    uniform_layout: &BindGroupLayout,
    samples: u32,
    probe_layout: &BindGroupLayout,
    weather: Option<(&BindGroupLayout, crate::renderer::weather::WeatherKinds)>,
) -> Option<[TerrainPipeline; 4]> {
    let read = gentle_shader(&terrain_shader_for(TerrainRole::Read))?;
    let baked = crate::renderer::brush_pipeline::sun_reader_shader(read.clone(), crate::renderer::brush_pipeline::FaceSun::Baked);
    let spotless = crate::renderer::lights::without_spot_shadows;
    let sources = [read.clone(), spotless(read), baked.clone(), spotless(baked)];
    let kind = weather.map_or("", |(_, k)| k.label());
    let labels = ["", "_spotless", "_baked", "_baked_spotless"];
    let mut built = Vec::with_capacity(4);
    for (src, suffix) in sources.into_iter().zip(labels) {
        let (src, label) = match weather {
            Some((_, kinds)) => (crate::renderer::weather::with_weather_kinds(&src, kinds)?, format!("terrain_weather{kind}_gentle_read{suffix}")),
            None => (src, format!("terrain_gentle_read{suffix}")),
        };
        built.push(TerrainPipeline::build_from_with(
            device,
            format,
            uniform_layout,
            samples,
            crate::renderer::multiview::ViewMode::Mono,
            TerrainRole::Read,
            weather.map(|(l, _)| l),
            Some(probe_layout),
            src,
            &label,
        ));
    }
    built.try_into().ok()
}

/// The ground's steep-only scene readers, `[full, spotless, baked,
/// baked_spotless]`, dry: for the triangles no gentle pixel comes from.
pub fn steep_readers(
    device: &Device,
    format: TextureFormat,
    uniform_layout: &BindGroupLayout,
    samples: u32,
    probe_layout: &BindGroupLayout,
) -> Option<[TerrainPipeline; 4]> {
    let read = steep_shader(&terrain_shader_for(TerrainRole::Read))?;
    let baked = crate::renderer::brush_pipeline::sun_reader_shader(read.clone(), crate::renderer::brush_pipeline::FaceSun::Baked);
    let spotless = crate::renderer::lights::without_spot_shadows;
    let sources = [read.clone(), spotless(read), baked.clone(), spotless(baked)];
    let built: Vec<TerrainPipeline> = sources
        .into_iter()
        .zip(["", "_spotless", "_baked", "_baked_spotless"])
        .map(|(src, suffix)| {
            TerrainPipeline::build_from_with(
                device,
                format,
                uniform_layout,
                samples,
                crate::renderer::multiview::ViewMode::Mono,
                TerrainRole::Read,
                None,
                Some(probe_layout),
                src,
                &format!("terrain_steep_read{suffix}"),
            )
        })
        .collect();
    built.try_into().ok()
}

/// THE GROUND'S PROBE PASS WITH ITS REPEATED CODE WRITTEN ONCE
/// (`brush_pipeline::DEDUP_EDITS`: the same calls in the same order, the same
/// picture to the byte), as `[pass, poolless]`, with the weather's twin of
/// each when `weather` is its group 2. PIPESTATS 2026-10-08 10:59: the shipped
/// pass 9,225 instructions, 26 registers, 37% occupancy; this 8,558, 23, 50%.
/// `None` when the edits no longer match.
pub fn dedup_passes(
    device: &Device,
    uniform_layout: &BindGroupLayout,
    fixups: &crate::renderer::probe_fixup::ProbeFixups,
    weather: Option<&BindGroupLayout>,
) -> Option<[TerrainPipeline; 2]> {
    let pass = dedup_pass_shader(&terrain_shader_for(TerrainRole::ProbePass))?;
    let poolless = crate::renderer::lights::without_pool_maps(pass.clone());
    let prefix = if weather.is_some() { "terrain_weather_probe_pass_dedup" } else { "terrain_probe_pass_dedup" };
    let mut built = Vec::with_capacity(2);
    for (src, suffix) in [(pass, ""), (poolless, "_poolless")] {
        let src = match weather {
            Some(_) => crate::renderer::weather::with_weather(&src)?,
            None => src,
        };
        built.push(TerrainPipeline::build_from_with(
            device,
            crate::renderer::brush_pipeline::probe_pass::FORMAT,
            uniform_layout,
            1,
            crate::renderer::multiview::ViewMode::Mono,
            TerrainRole::ProbePass,
            weather,
            Some(fixups.pass_layout()),
            src,
            &format!("{prefix}{suffix}"),
        ));
    }
    built.try_into().ok()
}

/// The ground's probe pass `src` with [`crate::renderer::brush_pipeline::DEDUP_EDITS`].
pub fn dedup_pass_shader(src: &str) -> Option<String> {
    crate::renderer::ground_cuts::with_cut(src, crate::renderer::brush_pipeline::DEDUP_EDITS)
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_ground_probe_pass_takes_its_dedup_and_validates() {
        use wgpu::naga;
        let pass = dedup_pass_shader(&terrain_shader_for(TerrainRole::ProbePass)).expect("the dedup edits moved");
        let poolless = crate::renderer::lights::without_pool_maps(pass.clone());
        for s in [pass.clone(), poolless.clone(), crate::renderer::weather::with_weather(&pass).unwrap(), crate::renderer::weather::with_weather(&poolless).unwrap()] {
            let module = naga::front::wgsl::parse_str(&s).unwrap_or_else(|e| panic!("{}", e.emit_to_string(&s)));
            naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all()).validate(&module).unwrap();
        }
    }

    use super::*;

    fn vertex(n: [f32; 3]) -> SolidVertex {
        SolidVertex { position: [0.0; 3], normal: n, color: [1.0; 4], uv2: [0.0; 2], reflectivity: 0.0 }
    }

    #[test]
    fn every_ground_reader_takes_its_gentle_twin_and_validates() {
        use wgpu::naga;
        let read = terrain_shader_for(TerrainRole::Read);
        let baked = crate::renderer::brush_pipeline::sun_reader_shader(read.clone(), crate::renderer::brush_pipeline::FaceSun::Baked);
        for src in [read.clone(), crate::renderer::lights::without_spot_shadows(baked)] {
            let g = gentle_shader(&src).expect("the ground's steep tests moved");
            for s in [g.clone(), crate::renderer::weather::with_weather(&g).unwrap()] {
                let module = naga::front::wgsl::parse_str(&s).unwrap_or_else(|e| panic!("{}", e.emit_to_string(&s)));
                naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
                    .validate(&module)
                    .unwrap();
            }
        }
    }

    #[test]
    fn a_triangle_is_gentle_only_when_every_vertex_is_under_the_margin() {
        let flat = [0.0, 1.0, 0.0];
        let tilt = |deg: f32| [deg.to_radians().sin(), deg.to_radians().cos(), 0.0];
        assert!(triangle_is_gentle([flat, flat, flat], 18.0));
        assert!(triangle_is_gentle([flat, tilt(12.9), flat], 18.0));
        assert!(!triangle_is_gentle([flat, tilt(13.1), flat], 18.0));
        assert!(!triangle_is_gentle([flat, flat, [0.0, 0.0, 0.0]], 18.0));
        assert!(!triangle_is_gentle([flat, flat, flat], 4.0));
    }

    #[test]
    fn a_triangle_is_steep_only_when_no_blend_of_its_normals_is_gentle() {
        let tilt = |deg: f32, az: f32| {
            let (s, c) = (deg.to_radians().sin(), deg.to_radians().cos());
            [s * az.to_radians().cos(), c, s * az.to_radians().sin()]
        };
        assert!(triangle_is_steep([tilt(60.0, 0.0), tilt(62.0, 5.0), tilt(58.0, -5.0)], 18.0));
        // Leaning opposite ways: their middle is level.
        assert!(!triangle_is_steep([tilt(40.0, 0.0), tilt(40.0, 180.0), tilt(40.0, 0.0)], 18.0));
        // Within the margin over the threshold: mixed.
        assert!(!triangle_is_steep([tilt(22.0, 0.0), tilt(22.0, 0.0), tilt(22.0, 0.0)], 18.0));
        assert!(triangle_is_steep([tilt(24.0, 0.0), tilt(24.0, 0.0), tilt(24.0, 0.0)], 18.0));
    }

    #[test]
    fn the_split_keeps_every_chunk_its_own_triangles_gentle_first() {
        let flat = [0.0, 1.0, 0.0];
        let steep = [0.7, 0.7, 0.0];
        let verts = vec![vertex(flat), vertex(flat), vertex(flat), vertex(steep), vertex(flat), vertex(flat)];
        // Chunk 0: steep, flat. Chunk 1: flat, steep, flat.
        let idx = vec![3, 4, 5, 0, 1, 2, 0, 1, 2, 3, 4, 5, 1, 2, 0];
        let chunk = |first, count| CasterChunk { first_index: first, index_count: count, min: glam::Vec3::ZERO, max: glam::Vec3::ZERO };
        let chunks = [chunk(0, 6), chunk(6, 9)];
        let s = SlopeSplit::new(&verts, &idx, &chunks, 18.0);
        assert_eq!(s.gentle, vec![3, 6]);
        assert_eq!(s.steep, vec![0, 0]);
        assert_eq!(s.indices, vec![0, 1, 2, 3, 4, 5, 0, 1, 2, 1, 2, 0, 3, 4, 5]);
        assert_eq!(s.gentle_total, 9);
        assert!(s.is_for(&verts, &idx, &chunks, 18.0));
        assert!(!s.is_for(&verts, &idx, &chunks, 20.0));
    }
}
