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
//! THE SAME PICTURE. A pixel is steep where its interpolated normal is, or
//! where the triangle it is drawn on is (`terrain_pipeline`'s `sample_frame`:
//! the interpolated normal reads level along a knife ridge's crest). The
//! normalised blend of a triangle's vertex normals with positive weights lies
//! in the cone they span: no steeper than its steepest vertex. So a triangle
//! whose every vertex AND whose own plane is gentler than the threshold, by a
//! margin for the centre of an MSAA edge pixel just outside it, draws no steep
//! pixel, and the twin -- the reader with that test false -- draws it as the
//! reader does.
//!
//! HOW. Each terrain chunk's triangles are reordered once, its gentle ones
//! first ([`SlopeSplit`]); the chunk keeps its range, so every other pass and
//! test (shadows, culling, the weather's chunks) is unchanged, and the scene
//! pass draws each chunk in two ranges, the gentle one with the twin.

use crate::renderer::cuboid::SolidVertex;
use crate::renderer::shadow::CasterChunk;
use crate::renderer::terrain_pipeline::{terrain_shader_for, TerrainPipeline, TerrainRole};
use wgpu::{BindGroupLayout, Device, TextureFormat};

/// The ground shader's steep test, as `terrain_shader`'s `sample_frame`
/// writes it.
pub(crate) const STEEP_TEST: &str = "    let steep = min(n.y, face.y) <= cos(radians(mat.biplanar_start_deg));\n";

/// The steep test and the twin's.
/// The top plane's fold where the frame says how (`fold_bumps`): on gentle
/// ground always the fold it always had, written so, so that the gentle
/// twins carry none of the steep frame's arithmetic.
pub(crate) const FOLD_KEEP: &str = "        let g = mix(z, 1.0, keep);\n";
pub(crate) const FOLD_KEPT: &str = "        let g = 1.0;\n";
pub(crate) const GENTLE_EDITS: &[(&str, &str)] = &[(STEEP_TEST, "    let steep = false;\n"), (FOLD_KEEP, FOLD_KEPT)];

/// Degrees under the threshold every vertex of a twin's triangle must be: room
/// for the centre of an MSAA edge pixel, just outside the triangle, where its
/// normal is extrapolated.
pub const GENTLE_MARGIN_DEG: f32 = 5.0;

/// `src` (any ground shader, any twin) without its steep code, or `None` when
/// its steep tests are not there once each.
pub fn gentle_shader(src: &str) -> Option<String> {
    crate::renderer::ground_cuts::with_cut(src, GENTLE_EDITS)
}

/// The height (the slope's cosine) of the plane a triangle with these corners
/// lies in, whichever way it is wound; 1 for a triangle with no area, which
/// draws no pixel.
pub fn face_height(positions: [[f32; 3]; 3]) -> f32 {
    use glam::Vec3;
    let [a, b, c] = positions.map(Vec3::from);
    (b - a).cross(c - a).try_normalize().map_or(1.0, |n| n.y.abs())
}

/// Whether a triangle with these vertex normals, lying in a plane of height
/// `face_y` ([`face_height`]), can draw no steep pixel under a `threshold_deg`
/// slope (`TerrainMaterialUniform::biplanar_start_deg`): a pixel is steep
/// where its interpolated normal is or where its triangle is.
pub fn triangle_is_gentle(normals: [[f32; 3]; 3], face_y: f32, threshold_deg: f32) -> bool {
    let limit = threshold_deg - GENTLE_MARGIN_DEG;
    if limit <= 0.0 {
        return false;
    }
    let cos_limit = limit.to_radians().cos();
    face_y > cos_limit
        && normals.iter().all(|n| {
            let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            len > 0.0 && n[1] / len > cos_limit
        })
}

/// The ground shader's two steep tests made true: the twin for triangles
/// every pixel of which is steep (`ground_cuts` `steep`: 24 registers, 50%,
/// against the reader's 27, 37%).
pub(crate) const STEEP_EDITS: &[(&str, &str)] = &[(STEEP_TEST, "    let steep = true;\n")];

/// `src` with only the steep ground's code, or `None` when its steep tests
/// are not there once each.
pub fn steep_shader(src: &str) -> Option<String> {
    crate::renderer::ground_cuts::with_cut(src, STEEP_EDITS)
}

/// Whether a triangle with these vertex normals, lying in a plane of height
/// `face_y` ([`face_height`]), can draw no gentle pixel: its own plane is
/// steep, which makes every pixel of it steep, or no blend of its normals is
/// gentle.
/// Not its gentlest vertex's slope: two steep normals leaning opposite ways
/// blend to a level one. A pixel's normal is a blend of the vertices' with
/// positive weights -- a point of the spherical triangle they span -- so the
/// triangle is steep when the most level point of that triangle is: the
/// straight-up direction if it lies inside, else the most level point of an
/// edge (where the edge's great circle passes nearest straight up, when that
/// is between its ends; else an end).
///
/// (Until 2026-10-13 the bound was the cap round the normals' mean as wide as
/// the farthest of them: its slope less that width. Right, and far from
/// tight on a face that turns in plan -- normals strung along one latitude
/// span a wide cap and a thin triangle. The exact bound moves 1.4% of
/// test_room's ground from the reader for both kinds of pixel to the steep
/// twin: 156,873 of its 386,040 indices (40.6%) against 39.2%.)
pub fn triangle_is_steep(normals: [[f32; 3]; 3], face_y: f32, threshold_deg: f32) -> bool {
    let limit = threshold_deg + GENTLE_MARGIN_DEG;
    limit < 90.0 && (face_y <= limit.to_radians().cos() || steepest_bound_y(normals).is_some_and(|max_y| max_y <= limit.to_radians().cos()))
}

/// The greatest `y` (the cosine of the least slope) any normalised blend of
/// these normals with positive weights can have, or `None` where that cannot
/// be bounded below 1: a zero normal, normals spread so wide their blends pass
/// near zero, or straight up among the blends.
fn steepest_bound_y(normals: [[f32; 3]; 3]) -> Option<f32> {
    use glam::Vec3;
    let unit = |n: [f32; 3]| Vec3::from(n).try_normalize();
    let (a, b, c) = (unit(normals[0])?, unit(normals[1])?, unit(normals[2])?);
    let mean = (a + b + c).try_normalize()?;
    // Within 60 degrees of their mean: a proper triangle on the sphere, far
    // from the blends that cancel.
    if a.dot(mean).min(b.dot(mean)).min(c.dot(mean)) <= 0.5 {
        return None;
    }
    // Straight up strictly inside the triangle: some pixel is level. (On an
    // edge, or in a triangle too thin to have an inside, the edges below
    // find it: their great circle passes through straight up.)
    let up = Vec3::Y;
    let sides = [up.dot(a.cross(b)), up.dot(b.cross(c)), up.dot(c.cross(a))];
    const EDGE: f32 = 1e-6;
    let (pos, neg) = (sides.iter().all(|s| *s > EDGE), sides.iter().all(|s| *s < -EDGE));
    if (pos || neg) && mean.y > 0.0 {
        return None;
    }
    let edge = |p: Vec3, q: Vec3| {
        let ends = p.y.max(q.y);
        let Some(m) = p.cross(q).try_normalize() else { return ends };
        // The great circle's point nearest straight up, and whether it lies
        // between the ends.
        let g = up - m * up.dot(m);
        let len = g.length();
        if len < 1e-6 {
            return ends;
        }
        let g = g / len;
        // A little past each end counts as between: the bound only grows.
        if p.cross(g).dot(m) >= -1e-4 && g.cross(q).dot(m) >= -1e-4 {
            ends.max(len)
        } else {
            ends
        }
    };
    Some(edge(a, b).max(edge(b, c)).max(edge(c, a)))
}

/// The order the scene pass draws a chunk's three slope ranges' readers in
/// when it walks the chunks reader by reader -- 0 the reader for triangles
/// with both kinds of pixel, 1 the gentle twin, 2 the steep twin. What stands
/// up first under `Levers::ground_steep_first`: a cliff hides the slopes
/// behind it, level ground hides next to nothing.
pub fn reader_order(steep_first: bool) -> [usize; 3] {
    if steep_first { [2, 0, 1] } else { [1, 0, 2] }
}

/// How far a sampled normal's height may move and the ground still be the
/// ground the split was built for: far over what a pose's arithmetic leaves
/// (a few parts in ten million), far under what another ground's differs by.
pub const HEIGHT_TOLERANCE: f32 = 1e-4;

/// THE TERRAIN'S INDICES WITH EACH CHUNK'S GENTLE TRIANGLES FIRST, and how
/// many of each chunk's indices they are. Built once per terrain and
/// threshold; `key` tells a new terrain from the one it was built for.
pub struct SlopeSplit {
    key: (usize, usize, usize, usize, u32, u64),
    /// The sampled normals' heights the split was built from (`heights_of`).
    heights: Vec<f32>,
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
    /// Per chunk: how many indices each of its nine BUCKETS holds, in the
    /// order they lie in the chunk's range -- the slope class major (gentle,
    /// mixed, steep), the triangle's GROUND CLASS minor (0, 1, 2: what the
    /// reflection pass may draw it with, `outdoor_pass::LevelGround`). So the
    /// scene pass's three slope ranges are as they were, and the reflection
    /// pass draws a ground class as at most three runs a chunk
    /// ([`Self::class_ranges`]). All in class 0 without a classifier.
    pub classes: Vec<[u32; 9]>,
    /// Indices of all, by ground class.
    pub class_totals: [u64; 3],
}

impl SlopeSplit {
    fn key_of(verts: &[SolidVertex], idx: &[u32], chunks: &[CasterChunk], threshold_deg: f32, class_key: u64) -> (usize, usize, usize, usize, u32, u64) {
        // A sample of the data as well as where it lies: a terrain loaded
        // again can land at the same address. Of what a POSE leaves alone --
        // the indices here, and the normals' heights in `heights_of` -- and
        // not the positions: the app hands the ground posed into the player's
        // frame, so a sampled height changed with every step up or down a
        // slope, and the split (128,680 triangles sorted) was built again on
        // each such frame.
        let mut h: u64 = 1469598103934665603 ^ class_key;
        let mut mix = |v: u64| {
            h ^= v;
            h = h.wrapping_mul(1099511628211);
        };
        // A sample of the chunks, as of the indices: with chunks of a fixed
        // size (and the coarser lists' runs among them) there are tens of
        // thousands on a large level, and this is asked every frame.
        mix(chunks.len() as u64);
        for c in chunks.iter().step_by((chunks.len() / 997).max(1)) {
            mix(c.first_index as u64);
            mix(c.index_count as u64);
        }
        let step = (idx.len() / 997).max(1);
        for i in idx.iter().step_by(step) {
            mix(*i as u64);
        }
        (verts.as_ptr() as usize, verts.len(), idx.as_ptr() as usize, idx.len(), threshold_deg.to_bits(), h)
    }

    /// A sample of the normals' heights, which a turn about the vertical
    /// keeps -- to its last bits only: the app's turn leaves each a few ulps
    /// different from one yaw to the next. COMPARED WITHIN A TOLERANCE
    /// ([`HEIGHT_TOLERANCE`]), not hashed: rounded into a hash (to a part in
    /// 8,192, until 2026-10-13) some sample of a thousand always lay on a
    /// rounding's edge, the key changed on about every other frame the rig
    /// turned, and the whole split was built again each time -- 33 to 42 ms
    /// of the render thread on the Quest, 400 times in 14 windows of a
    /// walking bench view. That was the lag as played.
    fn heights_of(verts: &[SolidVertex]) -> Vec<f32> {
        let step = (verts.len() / 997).max(1);
        verts.iter().step_by(step).map(|v| v.normal[1]).collect()
    }

    /// Whether this split is the one for this terrain and these ground
    /// classes (`class_key`: 0 for none).
    pub fn is_for(&self, verts: &[SolidVertex], idx: &[u32], chunks: &[CasterChunk], threshold_deg: f32, class_key: u64) -> bool {
        if self.key != Self::key_of(verts, idx, chunks, threshold_deg, class_key) {
            return false;
        }
        let step = (verts.len() / 997).max(1);
        let mut now = verts.iter().step_by(step).map(|v| v.normal[1]);
        self.heights.len() == now.len() && self.heights.iter().all(|was| now.next().is_some_and(|is| (is - was).abs() <= HEIGHT_TOLERANCE))
    }

    /// `idx` with each chunk's gentle triangles first. No chunks: the whole
    /// terrain as one.
    pub fn new(verts: &[SolidVertex], idx: &[u32], chunks: &[CasterChunk], threshold_deg: f32) -> Self {
        Self::classed(verts, idx, chunks, threshold_deg, 0, &|_| 0)
    }

    /// [`Self::new`], and within each slope class the triangles ordered by
    /// their GROUND CLASS: `class_of` a triangle's three positions (as
    /// `verts` holds them), 0, 1 or 2 -- anything else counts as 0.
    /// `class_key` names the classifier, for [`Self::is_for`].
    pub fn classed(
        verts: &[SolidVertex],
        idx: &[u32],
        chunks: &[CasterChunk],
        threshold_deg: f32,
        class_key: u64,
        class_of: &dyn Fn([[f32; 3]; 3]) -> usize,
    ) -> Self {
        let whole = [CasterChunk { first_index: 0, index_count: idx.len() as u32, min: glam::Vec3::ZERO, max: glam::Vec3::ZERO }];
        let ranges: &[CasterChunk] = if chunks.is_empty() { &whole } else { chunks };
        let mut indices = idx.to_vec();
        let mut gentle = Vec::with_capacity(ranges.len());
        let mut gentle_total = 0u64;
        let mut steep = Vec::with_capacity(ranges.len());
        let mut steep_total = 0u64;
        let mut classes = Vec::with_capacity(ranges.len());
        let mut class_totals = [0u64; 3];
        let normal = |i: u32| verts.get(i as usize).map_or([0.0; 3], |v| v.normal);
        let position = |i: u32| verts.get(i as usize).map_or([0.0; 3], |v| v.position);
        for c in ranges {
            let (start, end) = (c.first_index as usize, (c.first_index + c.index_count) as usize);
            if end > idx.len() || (end - start) % 3 != 0 {
                gentle.push(0);
                steep.push(0);
                // Unsorted: all of it the full pass's.
                classes.push([c.index_count, 0, 0, 0, 0, 0, 0, 0, 0]);
                continue;
            }
            let tris = &idx[start..end];
            // Nine buckets: slope class * 3 + ground class.
            let mut buckets: [Vec<u32>; 9] = Default::default();
            for t in tris.chunks_exact(3) {
                let n = [normal(t[0]), normal(t[1]), normal(t[2])];
                let face_y = face_height([position(t[0]), position(t[1]), position(t[2])]);
                let slope = if triangle_is_gentle(n, face_y, threshold_deg) {
                    0
                } else if triangle_is_steep(n, face_y, threshold_deg) {
                    2
                } else {
                    1
                };
                let class = match class_of([position(t[0]), position(t[1]), position(t[2])]) {
                    k @ 0..=2 => k,
                    _ => 0,
                };
                buckets[slope * 3 + class].extend_from_slice(t);
            }
            let count: [u32; 9] = std::array::from_fn(|b| buckets[b].len() as u32);
            let (soft, hard) = (count[0] + count[1] + count[2], count[6] + count[7] + count[8]);
            gentle.push(soft);
            gentle_total += soft as u64;
            steep.push(hard);
            steep_total += hard as u64;
            for b in 0..9 {
                class_totals[b % 3] += count[b] as u64;
            }
            classes.push(count);
            let mut at = start;
            for bucket in &buckets {
                indices[at..at + bucket.len()].copy_from_slice(bucket);
                at += bucket.len();
            }
        }
        Self {
            key: Self::key_of(verts, idx, chunks, threshold_deg, class_key),
            heights: Self::heights_of(verts),
            indices,
            gentle,
            gentle_total,
            steep,
            steep_total,
            classes,
            class_totals,
        }
    }

    /// THE SPLIT FOR THIS GROUND, MADE IF `kept` IS NOT IT: the one call
    /// both the level's load and the frame make, so the load's split is the
    /// frame's -- a frame that finds it made does nothing but compare a
    /// thousand samples ([`Self::is_for`]). Returns the milliseconds a build
    /// took, `None` when `kept` stood.
    ///
    /// It was made on the first FRAME that drew the ground, on the render
    /// thread: a hitch of 35 ms on test_room, 152 ms at 625 samples a side,
    /// 9.9 s at 2,049 (`docs/scale-audit-2026-10-13.md`, F12).
    pub fn ensure(
        kept: &mut Option<SlopeSplit>,
        verts: &[SolidVertex],
        idx: &[u32],
        chunks: &[CasterChunk],
        threshold_deg: f32,
        class_key: u64,
        class_of: &dyn Fn([[f32; 3]; 3]) -> usize,
    ) -> Option<f64> {
        if kept.as_ref().is_some_and(|s| s.is_for(verts, idx, chunks, threshold_deg, class_key)) {
            return None;
        }
        let started = std::time::Instant::now();
        *kept = Some(Self::classed(verts, idx, chunks, threshold_deg, class_key, class_of));
        Some(started.elapsed().as_secs_f64() * 1000.0)
    }

    /// Whether any triangle has a ground class of its own (1 or 2).
    pub fn has_classes(&self) -> bool {
        self.class_totals[1] + self.class_totals[2] > 0
    }

    /// The runs of chunk `k`'s range -- `(offset from its first index, index
    /// count)`, in order, neighbours merged -- that hold the triangles of the
    /// ground classes `pick`ed, and how many: at most three for one class
    /// (one a slope class), four for two (the first and the last, whose
    /// buckets meet across the slope classes). The three classes' runs tile
    /// the chunk's range exactly.
    pub fn class_ranges(&self, k: usize, pick: [bool; 3]) -> ([(u32, u32); 4], usize) {
        let mut runs = [(0u32, 0u32); 4];
        let mut n = 0usize;
        let Some(count) = self.classes.get(k) else { return (runs, 0) };
        let mut at = 0u32;
        for (b, &len) in count.iter().enumerate() {
            if len > 0 && pick[b % 3] {
                if n > 0 && runs[n - 1].0 + runs[n - 1].1 == at {
                    runs[n - 1].1 += len;
                } else if n < runs.len() {
                    runs[n] = (at, len);
                    n += 1;
                }
            }
            at += len;
        }
        (runs, n)
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

/// THE WET-ONLY TWINS OF THE GROUND'S MIXED AND STEEP SCENE READERS, `[mixed,
/// steep]`, each `[full, spotless, baked, baked_spotless]`: the weather's
/// twin with no standing water's code and no snow's
/// (`weather::WeatherKinds::WetOnly`), for the triangles that are not gentle
/// in a chunk that holds neither -- a beach's washed chunks with the cliff
/// over them, a rained-on slope that cannot pool.
///
/// WHY. Such a chunk's gentle triangles had a wet-only twin and the rest drew
/// the weather's full reader, snow and puddles and all: 3,820 instructions,
/// past the instruction cache (3,387), where the dry steep twin they drew
/// before the shore's wetness is 2,831. Measured 2026-10-13 on the Quest:
/// cave_mouth +3.8 ms, beach_south +1.8, shore +0.9 from that alone. These
/// are about 3,270 by the host proxy (2,183 x the same ratio), under the edge;
/// to be confirmed on the headset. `Levers::wet_steep_twins`.
///
/// `None` when the ground's steep tests are not there to take out, or the
/// weather's edits no longer match a reader.
pub fn wet_readers(
    device: &Device,
    format: TextureFormat,
    uniform_layout: &BindGroupLayout,
    samples: u32,
    probe_layout: &BindGroupLayout,
    weather_layout: &BindGroupLayout,
) -> Option<[[TerrainPipeline; 4]; 2]> {
    let spotless = crate::renderer::lights::without_spot_shadows;
    let mut both = Vec::with_capacity(2);
    for (read, name) in [(terrain_shader_for(TerrainRole::Read), "read"), (steep_shader(&terrain_shader_for(TerrainRole::Read))?, "steep_read")] {
        let baked = crate::renderer::brush_pipeline::sun_reader_shader(read.clone(), crate::renderer::brush_pipeline::FaceSun::Baked);
        let sources = [read.clone(), spotless(read), baked.clone(), spotless(baked)];
        let mut built = Vec::with_capacity(4);
        for (src, suffix) in sources.into_iter().zip(["", "_spotless", "_baked", "_baked_spotless"]) {
            let src = crate::renderer::weather::with_weather_kinds(&src, crate::renderer::weather::WeatherKinds::WetOnly)?;
            built.push(TerrainPipeline::build_from_with(
                device,
                format,
                uniform_layout,
                samples,
                crate::renderer::multiview::ViewMode::Mono,
                TerrainRole::Read,
                Some(weather_layout),
                Some(probe_layout),
                src,
                &format!("terrain_weather_wet_{name}{suffix}"),
            ));
        }
        both.push(<[TerrainPipeline; 4]>::try_from(built).ok()?);
    }
    both.try_into().ok()
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

    /// THE SPLIT MADE AT A LEVEL'S LOAD IS THE ONE ITS FRAMES FIND: `ensure`
    /// builds it once -- sorting a ground larger than test_room's, 320,000
    /// triangles -- and every later call for the same ground, as each frame
    /// makes, builds nothing. Another probe table (its key), another
    /// threshold or another ground builds it again, once.
    #[test]
    fn the_split_made_at_load_is_the_one_the_frames_find() {
        let n = 401usize;
        let mut verts = Vec::with_capacity(n * n);
        for z in 0..n {
            for x in 0..n {
                let tilt = ((x as f32 * 0.21).sin() * (z as f32 * 0.17).cos()).abs();
                let mut v = vertex([tilt, 1.0 - tilt * 0.9, 0.2]);
                v.position = [x as f32 * 0.43, tilt * 4.0, z as f32 * 0.43];
                verts.push(v);
            }
        }
        let mut idx = Vec::with_capacity((n - 1) * (n - 1) * 6);
        for z in 0..n - 1 {
            for x in 0..n - 1 {
                let a = (z * n + x) as u32;
                idx.extend_from_slice(&[a, a + n as u32, a + 1, a + 1, a + n as u32, a + n as u32 + 1]);
            }
        }
        let chunks: Vec<CasterChunk> = idx
            .chunks(idx.len() / 64)
            .scan(0u32, |at, c| {
                let first = *at;
                *at += c.len() as u32;
                Some(CasterChunk { first_index: first, index_count: c.len() as u32, min: glam::Vec3::ZERO, max: glam::Vec3::ZERO })
            })
            .collect();
        let class_of = |c: [[f32; 3]; 3]| if c[0][0] < 60.0 { 1 } else { 2 };
        let mut kept = None;
        // The load.
        assert!(SlopeSplit::ensure(&mut kept, &verts, &idx, &chunks, 35.0, 77, &class_of).is_some());
        let made = kept.as_ref().unwrap().indices.clone();
        assert!(kept.as_ref().unwrap().has_classes());
        // Its frames: the chunks' boxes follow the rig, nothing else changes.
        for frame in 0..30 {
            let moved: Vec<CasterChunk> = chunks.iter().map(|c| CasterChunk { min: glam::Vec3::splat(frame as f32), ..*c }).collect();
            assert!(SlopeSplit::ensure(&mut kept, &verts, &idx, &moved, 35.0, 77, &class_of).is_none(), "built again on frame {frame}");
        }
        assert_eq!(kept.as_ref().unwrap().indices, made);
        // Another probe table; another threshold; another ground.
        assert!(SlopeSplit::ensure(&mut kept, &verts, &idx, &chunks, 35.0, 78, &class_of).is_some());
        assert!(SlopeSplit::ensure(&mut kept, &verts, &idx, &chunks, 35.0, 78, &class_of).is_none());
        assert!(SlopeSplit::ensure(&mut kept, &verts, &idx, &chunks, 30.0, 78, &class_of).is_some());
        let other: Vec<SolidVertex> = verts.iter().map(|v| { let mut o = *v; o.normal = [0.0, 1.0, 0.0]; o }).collect();
        assert!(SlopeSplit::ensure(&mut kept, &other, &idx, &chunks, 30.0, 78, &class_of).is_some());
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
        assert!(triangle_is_gentle([flat, flat, flat], 1.0, 18.0));
        assert!(triangle_is_gentle([flat, tilt(12.9), flat], 1.0, 18.0));
        assert!(!triangle_is_gentle([flat, tilt(13.1), flat], 1.0, 18.0));
        assert!(!triangle_is_gentle([flat, flat, [0.0, 0.0, 0.0]], 1.0, 18.0));
        assert!(!triangle_is_gentle([flat, flat, flat], 1.0, 4.0));
        // ITS OWN PLANE COUNTS: a knife ridge's crest carries level normals
        // on faces that stand at seventy degrees.
        assert!(triangle_is_gentle([flat, flat, flat], 12.9f32.to_radians().cos(), 18.0));
        assert!(!triangle_is_gentle([flat, flat, flat], 13.1f32.to_radians().cos(), 18.0));
        assert!(!triangle_is_gentle([flat, flat, flat], 70.0f32.to_radians().cos(), 18.0));
    }

    /// A triangle's plane from its corners, whichever way it is wound; a
    /// triangle with no area is level (it draws nothing).
    #[test]
    fn a_triangles_own_height_is_its_planes() {
        let level = [[0.0, 2.0, 0.0], [1.0, 2.0, 0.0], [0.0, 2.0, 1.0]];
        assert!((face_height(level) - 1.0).abs() < 1e-6);
        let wall = [[0.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 1.0]];
        assert!(face_height(wall) < 1e-6);
        let ramp = [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        assert!((face_height(ramp) - 45.0f32.to_radians().cos()).abs() < 1e-6);
        assert!((face_height([ramp[0], ramp[2], ramp[1]]) - face_height(ramp)).abs() < 1e-6);
        assert_eq!(face_height([[1.0; 3]; 3]), 1.0);
    }

    /// THE BOUND HOLDS AND IS TIGHT: over random triangles of normals, no
    /// blend with positive weights is more level than the bound says, and
    /// some blend comes within a degree of it. And the case the cap bound
    /// left out: a cliff turning in plan, its normals along one latitude.
    #[test]
    fn the_steep_bound_is_the_most_level_blend() {
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut rnd = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f32 / (1u64 << 53) as f32
        };
        let dir = |slope: f32, turn: f32| {
            let (s, t) = (slope.to_radians(), turn.to_radians());
            [s.sin() * t.cos(), s.cos(), s.sin() * t.sin()]
        };
        let (mut bounded, mut steep, mut cap_missed) = (0, 0, 0);
        for _ in 0..4000 {
            let (slope, turn) = (rnd() * 85.0, rnd() * 360.0);
            let n: [[f32; 3]; 3] = std::array::from_fn(|_| {
                let len = 0.5 + rnd();
                dir((slope + (rnd() - 0.5) * 50.0).clamp(0.0, 89.0), turn + (rnd() - 0.5) * 90.0).map(|v| v * len)
            });
            let Some(max_y) = steepest_bound_y(n) else { continue };
            bounded += 1;
            let mut most_level = -1.0f32;
            for i in 0..=40 {
                for j in 0..=(40 - i) {
                    let w = [i as f32, j as f32, (40 - i - j) as f32];
                    let v: glam::Vec3 = (0..3).map(|k| glam::Vec3::from(n[k]) * w[k]).sum();
                    most_level = most_level.max(v.normalize().y);
                }
            }
            assert!(most_level <= max_y + 1e-4, "a blend is more level ({most_level}) than the bound ({max_y}): {n:?}");
            assert!(most_level.acos().to_degrees() - max_y.acos().to_degrees() < 1.5, "the bound is loose: {n:?}");
            if triangle_is_steep(n, 1.0, 18.0) {
                steep += 1;
                assert!(most_level.acos().to_degrees() >= 23.0 - 0.01);
                // What the cap round the mean said of the same triangle.
                let u = n.map(|v| glam::Vec3::from(v).normalize());
                let m = (u[0] + u[1] + u[2]).normalize();
                let spread = u.iter().map(|v| v.dot(m).clamp(-1.0, 1.0).acos().to_degrees()).fold(0.0f32, f32::max);
                if m.y.acos().to_degrees() - spread < 23.0 {
                    cap_missed += 1;
                }
            }
        }
        assert!(bounded > 2000 && steep > 500, "{bounded} bounded, {steep} steep");
        assert!(cap_missed > 50, "the cap's bound left out {cap_missed}");
        // A face 60 degrees steep turning 70 degrees in plan across a triangle.
        let turning = [dir(60.0, 0.0), dir(60.0, 35.0), dir(60.0, 70.0)];
        assert!(triangle_is_steep(turning, 1.0, 18.0));
        // Straight up among the blends: never.
        assert!(!triangle_is_steep([dir(40.0, 0.0), dir(40.0, 120.0), dir(40.0, 240.0)], 1.0, 18.0));
        assert!(steepest_bound_y([dir(30.0, 0.0), dir(30.0, 100.0), dir(30.0, 200.0)]).is_none());
    }

    /// Either order draws each of the three readers once.
    #[test]
    fn the_readers_are_each_drawn_once_in_either_order() {
        for steep_first in [false, true] {
            let mut order = reader_order(steep_first);
            assert_eq!(order[0] == 2, steep_first);
            order.sort_unstable();
            assert_eq!(order, [0, 1, 2]);
        }
    }

    #[test]
    fn a_triangle_is_steep_only_when_no_blend_of_its_normals_is_gentle() {
        let tilt = |deg: f32, az: f32| {
            let (s, c) = (deg.to_radians().sin(), deg.to_radians().cos());
            [s * az.to_radians().cos(), c, s * az.to_radians().sin()]
        };
        assert!(triangle_is_steep([tilt(60.0, 0.0), tilt(62.0, 5.0), tilt(58.0, -5.0)], 1.0, 18.0));
        // Leaning opposite ways: their middle is level.
        assert!(!triangle_is_steep([tilt(40.0, 0.0), tilt(40.0, 180.0), tilt(40.0, 0.0)], 1.0, 18.0));
        // Within the margin over the threshold: mixed.
        assert!(!triangle_is_steep([tilt(22.0, 0.0), tilt(22.0, 0.0), tilt(22.0, 0.0)], 1.0, 18.0));
        assert!(triangle_is_steep([tilt(24.0, 0.0), tilt(24.0, 0.0), tilt(24.0, 0.0)], 1.0, 18.0));
        // A steep plane makes every pixel steep, whatever its normals say:
        // the faces under a ridge's level crest.
        let level = [0.0, 1.0, 0.0];
        assert!(triangle_is_steep([level, level, level], 24.0f32.to_radians().cos(), 18.0));
        assert!(!triangle_is_steep([level, level, level], 22.0f32.to_radians().cos(), 18.0));
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
        assert!(s.is_for(&verts, &idx, &chunks, 18.0, 0));
        assert!(!s.is_for(&verts, &idx, &chunks, 20.0, 0));
        assert!(!s.is_for(&verts, &idx, &chunks, 18.0, 7), "another classifier's split");
        // No classifier: every triangle the full pass's, one run a chunk.
        assert!(!s.has_classes());
        assert_eq!(s.class_ranges(1, [true, false, false]), ([(0, 9), (0, 0), (0, 0), (0, 0)], 1));
        assert_eq!(s.class_ranges(1, [false, true, true]).1, 0);
        // The pose is not the terrain: the same ground, posed a step up.
        let moved: Vec<SolidVertex> = verts.iter().map(|v| SolidVertex { position: [v.position[0], v.position[1] - 0.3, v.position[2]], ..*v }).collect();
        let again = SlopeSplit::new(&moved, &idx, &chunks, 18.0);
        assert_eq!(again.indices, s.indices);
        // ... and, where the app posed it into the same buffer, the same split:
        // nothing to build again.
        let mut posed = verts.clone();
        let split = SlopeSplit::new(&posed, &idx, &chunks, 18.0);
        for v in posed.iter_mut() {
            v.position[1] -= 0.3;
            v.normal[1] += 1e-7;
        }
        assert!(split.is_for(&posed, &idx, &chunks, 18.0, 0));
        // ... at EVERY turn of the rig: a thousand yaws, each leaving the
        // normals' heights their last bits different, and none a new ground.
        let world = posed.clone();
        for k in 0..1000 {
            let turn = glam::Quat::from_rotation_y(k as f32 * 0.0123);
            for (p, w) in posed.iter_mut().zip(&world) {
                p.normal = (turn * glam::Vec3::from(w.normal)).to_array();
                p.position = (turn * glam::Vec3::from(w.position)).to_array();
            }
            assert!(split.is_for(&posed, &idx, &chunks, 18.0, 0), "turn {k} built the split again");
        }
        posed[0].normal = [0.7, 0.7, 0.0];
        assert!(!split.is_for(&posed, &idx, &chunks, 18.0, 0), "the ground itself changed");
    }

    /// With ground classes: each chunk keeps its own triangles, its slope
    /// ranges are what they were without the classes, every run of a class
    /// holds triangles of that class alone, and the three classes' runs tile
    /// the chunk.
    #[test]
    fn the_ground_classes_sort_within_the_slope_classes_and_tile_each_chunk() {
        let mut seed = 11u64;
        let mut rnd = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f32) / (1u64 << 31) as f32
        };
        let tilt = |deg: f32| [deg.to_radians().sin(), deg.to_radians().cos(), 0.0];
        // One vertex a triangle corner, its x the triangle's class times ten.
        let mut verts = Vec::new();
        let mut idx = Vec::new();
        let tris = 300usize;
        for t in 0..tris {
            let slope = [3.0, 20.0, 70.0][(rnd() * 3.0) as usize % 3];
            let class = (rnd() * 3.0) as usize % 3;
            for _ in 0..3 {
                let mut v = vertex(tilt(slope));
                v.position = [class as f32 * 10.0 + rnd(), t as f32, 0.0];
                idx.push(verts.len() as u32);
                verts.push(v);
            }
        }
        let chunk = |first, count| CasterChunk { first_index: first, index_count: count, min: glam::Vec3::ZERO, max: glam::Vec3::ZERO };
        let chunks = [chunk(0, 300), chunk(300, 450), chunk(750, 150)];
        let class_of = |p: [[f32; 3]; 3]| (p[0][0] / 10.0) as usize;
        let s = SlopeSplit::classed(&verts, &idx, &chunks, 18.0, 5, &class_of);
        let plain = SlopeSplit::new(&verts, &idx, &chunks, 18.0);
        assert_eq!((&s.gentle, &s.steep), (&plain.gentle, &plain.steep));
        assert!(s.has_classes());
        assert_eq!(s.class_totals.iter().sum::<u64>(), idx.len() as u64);
        let tri_key = |t: &[u32]| verts[t[0] as usize].position[1] as u32;
        for (k, c) in chunks.iter().enumerate() {
            let (a, b) = (c.first_index as usize, (c.first_index + c.index_count) as usize);
            let sorted = |v: &[u32]| {
                let mut keys: Vec<u32> = v.chunks_exact(3).map(tri_key).collect();
                keys.sort_unstable();
                keys
            };
            assert_eq!(sorted(&s.indices[a..b]), sorted(&idx[a..b]), "chunk {k} lost or gained a triangle");
            // The slope ranges hold the triangles they held.
            let (g, st) = (s.gentle[k] as usize, s.steep[k] as usize);
            assert_eq!(sorted(&s.indices[a..a + g]), sorted(&plain.indices[a..a + g]));
            assert_eq!(sorted(&s.indices[b - st..b]), sorted(&plain.indices[b - st..b]));
            let mut covered = vec![0u8; b - a];
            for class in 0..3 {
                let (runs, n) = s.class_ranges(k, std::array::from_fn(|i| i == class));
                assert!(n <= 3, "one class in {n} runs");
                for &(o, len) in &runs[..n] {
                    assert!(len > 0 && len % 3 == 0);
                    for t in s.indices[a + o as usize..a + (o + len) as usize].chunks_exact(3) {
                        assert_eq!(class_of([verts[t[0] as usize].position; 3]), class, "chunk {k}: a triangle in another class's run");
                    }
                    covered[o as usize..(o + len) as usize].iter_mut().for_each(|x| *x += 1);
                }
                // In order, and not touching: neighbours were merged.
                for w in runs[..n].windows(2) {
                    assert!(w[0].0 + w[0].1 < w[1].0);
                }
            }
            assert!(covered.iter().all(|&x| x == 1), "chunk {k}: the classes' runs do not tile it");
            // Two classes together, and all three: fewer runs, the same indices.
            let (all, n) = s.class_ranges(k, [true; 3]);
            assert_eq!((n, all[0]), (1, (0, c.index_count)));
            let (two, n2) = s.class_ranges(k, [true, false, true]);
            let (one, n1) = s.class_ranges(k, [false, true, false]);
            let total: u32 = two[..n2].iter().chain(&one[..n1]).map(|r| r.1).sum();
            assert_eq!(total, c.index_count);
        }
        assert_eq!(s.class_ranges(9, [true; 3]).1, 0, "no such chunk");
    }
}
