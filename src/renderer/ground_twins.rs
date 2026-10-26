use crate::renderer::cuboid::SolidVertex;
use crate::renderer::shadow::CasterChunk;
use crate::renderer::terrain_pipeline::{terrain_shader_for, TerrainPipeline, TerrainRole};
use wgpu::{BindGroupLayout, Device, TextureFormat};

pub(crate) const STEEP_TEST: &str = "    let steep = min(n.y, face.y) <= cos(radians(mat.biplanar_start_deg));\n";

pub(crate) const FOLD_KEEP: &str = "        let g = mix(z, 1.0, keep);\n";
pub(crate) const FOLD_KEPT: &str = "        let g = 1.0;\n";
pub(crate) const GENTLE_EDITS: &[(&str, &str)] = &[(STEEP_TEST, "    let steep = false;\n"), (FOLD_KEEP, FOLD_KEPT)];

pub const GENTLE_MARGIN_DEG: f32 = 5.0;

pub fn gentle_shader(src: &str) -> Option<String> {
    crate::renderer::ground_cuts::with_cut(src, GENTLE_EDITS)
}

pub fn face_height(positions: [[f32; 3]; 3]) -> f32 {
    use glam::Vec3;
    let [a, b, c] = positions.map(Vec3::from);
    (b - a).cross(c - a).try_normalize().map_or(1.0, |n| n.y.abs())
}

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

pub(crate) const STEEP_EDITS: &[(&str, &str)] = &[(STEEP_TEST, "    let steep = true;\n")];

pub fn steep_shader(src: &str) -> Option<String> {
    crate::renderer::ground_cuts::with_cut(src, STEEP_EDITS)
}

pub fn triangle_is_steep(normals: [[f32; 3]; 3], face_y: f32, threshold_deg: f32) -> bool {
    let limit = threshold_deg + GENTLE_MARGIN_DEG;
    limit < 90.0 && (face_y <= limit.to_radians().cos() || steepest_bound_y(normals).is_some_and(|max_y| max_y <= limit.to_radians().cos()))
}

fn steepest_bound_y(normals: [[f32; 3]; 3]) -> Option<f32> {
    use glam::Vec3;
    let unit = |n: [f32; 3]| Vec3::from(n).try_normalize();
    let (a, b, c) = (unit(normals[0])?, unit(normals[1])?, unit(normals[2])?);
    let mean = (a + b + c).try_normalize()?;

    if a.dot(mean).min(b.dot(mean)).min(c.dot(mean)) <= 0.5 {
        return None;
    }

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

        let g = up - m * up.dot(m);
        let len = g.length();
        if len < 1e-6 {
            return ends;
        }
        let g = g / len;

        if p.cross(g).dot(m) >= -1e-4 && g.cross(q).dot(m) >= -1e-4 {
            ends.max(len)
        } else {
            ends
        }
    };
    Some(edge(a, b).max(edge(b, c)).max(edge(c, a)))
}

pub fn reader_order(steep_first: bool) -> [usize; 3] {
    if steep_first { [2, 0, 1] } else { [1, 0, 2] }
}

pub const HEIGHT_TOLERANCE: f32 = 1e-4;

pub struct SlopeSplit {
    key: (usize, usize, usize, usize, u32, u64),

    heights: Vec<f32>,

    pub indices: Vec<u32>,

    pub gentle: Vec<u32>,

    pub gentle_total: u64,

    pub steep: Vec<u32>,
    pub steep_total: u64,

    pub classes: Vec<[u32; 9]>,

    pub class_totals: [u64; 3],

    pub piece_of: Vec<(u32, u8)>,

    pub piece_runs: Vec<[u32; 4]>,

    pub piece_block: Vec<u32>,
}

impl SlopeSplit {
    fn key_of(verts: &[SolidVertex], idx: &[u32], chunks: &[CasterChunk], threshold_deg: f32, class_key: u64) -> (usize, usize, usize, usize, u32, u64) {

        let mut h: u64 = 1469598103934665603 ^ class_key;
        let mut mix = |v: u64| {
            h ^= v;
            h = h.wrapping_mul(1099511628211);
        };

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

    fn heights_of(verts: &[SolidVertex]) -> Vec<f32> {
        let step = (verts.len() / 997).max(1);
        verts.iter().step_by(step).map(|v| v.normal[1]).collect()
    }

    pub fn is_for(&self, verts: &[SolidVertex], idx: &[u32], chunks: &[CasterChunk], threshold_deg: f32, class_key: u64) -> bool {
        if self.key != Self::key_of(verts, idx, chunks, threshold_deg, class_key) {
            return false;
        }
        let step = (verts.len() / 997).max(1);
        let mut now = verts.iter().step_by(step).map(|v| v.normal[1]);
        self.heights.len() == now.len() && self.heights.iter().all(|was| now.next().is_some_and(|is| (is - was).abs() <= HEIGHT_TOLERANCE))
    }

    pub fn new(verts: &[SolidVertex], idx: &[u32], chunks: &[CasterChunk], threshold_deg: f32) -> Self {
        Self::classed(verts, idx, chunks, threshold_deg, 0, &|_| 0)
    }

    #[allow(clippy::too_many_arguments)]

    #[allow(clippy::too_many_arguments)]
    pub fn classed_on(
        verts: &[SolidVertex],
        idx: &[u32],
        chunks: &[CasterChunk],
        threshold_deg: f32,
        class_key: u64,
        class_of: &dyn Fn([[f32; 3]; 3]) -> usize,
        lattice: Option<[u32; 2]>,
        blocks: Option<&[u32]>,
    ) -> Self {
        let cover = lattice.and_then(|cells| CellSlopes::build(cells, verts, idx, threshold_deg));
        let mut split = Self::classed_with(verts, idx, chunks, threshold_deg, class_key ^ lattice_key(lattice) ^ blocks_key(blocks), class_of, cover.as_ref());
        if let Some(blocks) = blocks.filter(|b| b.len() == chunks.len()) {
            split.lay_blocks(chunks, blocks);
        }
        split
    }

    fn lay_blocks(&mut self, chunks: &[CasterChunk], blocks: &[u32]) {
        let real = chunks.len();
        self.piece_of = vec![(0, 0); real];
        self.piece_block = vec![u32::MAX; real];

        let mut laid: Vec<(usize, [u32; 4])> = Vec::new();
        let mut k = 0usize;
        while k < real {
            let b = blocks[k];
            if b == crate::renderer::ground_lod::SLIVER_RUN {

                let count = chunks[k].index_count;
                self.gentle_total -= self.gentle[k] as u64;
                self.steep_total -= self.steep[k] as u64;
                for (c, n) in self.classes[k].iter().enumerate() {
                    self.class_totals[c % 3] -= *n as u64;
                }
                self.class_totals[0] += count as u64;
                (self.gentle[k], self.steep[k]) = (0, 0);
                self.classes[k] = [0, 0, 0, count, 0, 0, 0, 0, 0];
                k += 1;
                continue;
            }
            let mut e = k + 1;
            while b != u32::MAX && e < real && blocks[e] == b && chunks[e].first_index == chunks[e - 1].first_index + chunks[e - 1].index_count {
                e += 1;
            }
            let sorted = |u: usize| self.classes[u].iter().sum::<u32>() == chunks[u].index_count && self.gentle[u] + self.steep[u] <= chunks[u].index_count;
            if b == u32::MAX || e - k < 2 || !(k..e).all(sorted) {
                k = e;
                continue;
            }
            let start = chunks[k].first_index as usize;
            let end = (chunks[e - 1].first_index + chunks[e - 1].index_count) as usize;
            let src = self.indices[start..end].to_vec();
            let mut at = start;
            laid.clear();
            for bucket in 0..9 {
                for u in k..e {
                    let len = self.classes[u][bucket] as usize;
                    if len == 0 {
                        continue;
                    }
                    let off = chunks[u].first_index as usize - start + self.classes[u][..bucket].iter().sum::<u32>() as usize;
                    self.indices[at..at + len].copy_from_slice(&src[off..off + len]);
                    laid.push((u, [at as u32, len as u32, 0, bucket as u32]));
                    at += len;
                }
            }

            laid.sort_by_key(|(u, p)| (*u, p[3]));
            for u in k..e {
                self.piece_of[u] = (self.piece_runs.len() as u32, 0);
                self.piece_block[u] = start as u32;
                for (_, p) in laid.iter().filter(|(v, _)| *v == u) {
                    self.piece_runs.push(*p);
                    self.piece_of[u].1 += 1;
                }
            }
            k = e;
        }

        let n = self.piece_runs.len();
        self.gentle.resize(real + n, 0);
        self.steep.resize(real + n, 0);
        self.classes.resize(real + n, [0; 9]);
        for u in 0..real {
            let (at, len) = self.piece_of[u];
            for p in at as usize..at as usize + len as usize {
                let id = real + p;
                let [_, count, _, bucket] = self.piece_runs[p];
                let bucket = bucket as usize;
                self.classes[id][bucket] = count;
                self.gentle[id] = if bucket < 3 { count } else { 0 };
                self.steep[id] = if bucket >= 6 { count } else { 0 };
                self.piece_runs[p][2] = id as u32;
            }
        }
    }

    pub fn pieces(&self) -> Option<crate::renderer::ground_lod::Pieces> {
        (!self.piece_runs.is_empty()).then(|| crate::renderer::ground_lod::Pieces { of: self.piece_of.clone(), runs: self.piece_runs.clone(), block: self.piece_block.clone() })
    }

    pub fn classed(
        verts: &[SolidVertex],
        idx: &[u32],
        chunks: &[CasterChunk],
        threshold_deg: f32,
        class_key: u64,
        class_of: &dyn Fn([[f32; 3]; 3]) -> usize,
    ) -> Self {
        Self::classed_with(verts, idx, chunks, threshold_deg, class_key, class_of, None)
    }

    fn classed_with(
        verts: &[SolidVertex],
        idx: &[u32],
        chunks: &[CasterChunk],
        threshold_deg: f32,
        class_key: u64,
        class_of: &dyn Fn([[f32; 3]; 3]) -> usize,
        cover: Option<&CellSlopes>,
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

                classes.push([c.index_count, 0, 0, 0, 0, 0, 0, 0, 0]);
                continue;
            }
            let tris = &idx[start..end];

            let mut buckets: [Vec<u32>; 9] = Default::default();
            for t in tris.chunks_exact(3) {
                let n = [normal(t[0]), normal(t[1]), normal(t[2])];
                let face_y = face_height([position(t[0]), position(t[1]), position(t[2])]);
                let slope = match cover.and_then(|c| c.class_of_coarse([t[0], t[1], t[2]])) {
                    Some(k) => k,
                    None => slope_class(n, face_y, threshold_deg),
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
            piece_of: Vec::new(),
            piece_runs: Vec::new(),
            piece_block: Vec::new(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn ensure(
        kept: &mut Option<SlopeSplit>,
        verts: &[SolidVertex],
        idx: &[u32],
        chunks: &[CasterChunk],
        threshold_deg: f32,
        class_key: u64,
        class_of: &dyn Fn([[f32; 3]; 3]) -> usize,
        lattice: Option<[u32; 2]>,
        blocks: Option<&[u32]>,
    ) -> Option<f64> {
        if kept.as_ref().is_some_and(|s| s.is_for(verts, idx, chunks, threshold_deg, class_key ^ lattice_key(lattice) ^ blocks_key(blocks))) {
            return None;
        }
        let started = std::time::Instant::now();
        *kept = Some(Self::classed_on(verts, idx, chunks, threshold_deg, class_key, class_of, lattice, blocks));
        Some(started.elapsed().as_secs_f64() * 1000.0)
    }

    pub fn has_classes(&self) -> bool {
        self.class_totals[1] + self.class_totals[2] > 0
    }

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

fn slope_class(n: [[f32; 3]; 3], face_y: f32, threshold_deg: f32) -> usize {
    if triangle_is_gentle(n, face_y, threshold_deg) {
        0
    } else if triangle_is_steep(n, face_y, threshold_deg) {
        2
    } else {
        1
    }
}

fn blocks_key(blocks: Option<&[u32]>) -> u64 {
    let Some(blocks) = blocks.filter(|b| b.iter().any(|x| *x != u32::MAX)) else { return 0 };
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for (i, b) in blocks.iter().enumerate() {
        h ^= (*b as u64) ^ ((i as u64) << 32);
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    h | 1
}

fn lattice_key(lattice: Option<[u32; 2]>) -> u64 {
    match lattice {
        Some([x, z]) if x > 0 && z > 0 => 0x9e37_79b9_7f4a_7c15u64.wrapping_mul(1 + x as u64 + ((z as u64) << 32)),
        _ => 0,
    }
}

pub struct CellSlopes {
    cells: [u32; 2],

    not_gentle: Vec<u32>,
    not_steep: Vec<u32>,
}

impl CellSlopes {

    pub fn build(cells: [u32; 2], verts: &[SolidVertex], idx: &[u32], threshold_deg: f32) -> Option<Self> {
        let [cx, cz] = cells;
        let stride = cx + 1;
        let samples = (stride * (cz + 1)) as usize;
        if cx == 0 || cz == 0 || verts.len() < samples {
            return None;
        }

        let mut kinds = vec![0u8; (cx * cz) as usize];
        let class = |t: [u32; 3]| {
            let n = t.map(|v| verts[v as usize].normal);
            slope_class(n, face_height(t.map(|v| verts[v as usize].position)), threshold_deg)
        };
        for t in idx.chunks_exact(3) {
            let t = [t[0], t[1], t[2]];
            if let Some((x0, z0, 1, 1)) = Self::span(stride, samples, t) {
                kinds[(z0 * cx + x0) as usize] |= 1 << class(t);
            }
        }
        for z in 0..cz {
            for x in 0..cx {
                let k = &mut kinds[(z * cx + x) as usize];
                if *k == 0 {
                    let (a, b, c, d) = (z * stride + x, z * stride + x + 1, (z + 1) * stride + x, (z + 1) * stride + x + 1);
                    *k = (1 << class([a, d, b])) | (1 << class([a, c, d]));
                }
            }
        }
        let table = |hit: &dyn Fn(u8) -> bool| {
            let mut sat = vec![0u32; ((cx + 1) * (cz + 1)) as usize];
            for z in 0..cz {
                let mut row = 0u32;
                for x in 0..cx {
                    row += hit(kinds[(z * cx + x) as usize]) as u32;
                    sat[((z + 1) * (cx + 1) + x + 1) as usize] = sat[(z * (cx + 1) + x + 1) as usize] + row;
                }
            }
            sat
        };
        Some(Self { cells, not_gentle: table(&|k| k & 0b110 != 0), not_steep: table(&|k| k & 0b011 != 0) })
    }

    fn span(stride: u32, samples: usize, t: [u32; 3]) -> Option<(u32, u32, u32, u32)> {
        if t.iter().any(|&v| v as usize >= samples) {
            return None;
        }
        let at = t.map(|v| (v % stride, v / stride));
        let x0 = at.iter().map(|a| a.0).min()?;
        let z0 = at.iter().map(|a| a.1).min()?;
        let x1 = at.iter().map(|a| a.0).max()?;
        let z1 = at.iter().map(|a| a.1).max()?;
        Some((x0, z0, x1 - x0, z1 - z0))
    }

    pub fn class_of_coarse(&self, t: [u32; 3]) -> Option<usize> {
        let [cx, cz] = self.cells;
        let stride = cx + 1;
        let (x0, z0, w, d) = Self::span(stride, (stride * (cz + 1)) as usize, t)?;
        if w <= 1 && d <= 1 {
            return None;
        }
        let (a, b) = (x0.saturating_sub(1), z0.saturating_sub(1));
        let (c, e) = ((x0 + w + 1).min(cx), (z0 + d + 1).min(cz));
        let count = |sat: &[u32]| {
            let at = |x: u32, z: u32| sat[(z * (cx + 1) + x) as usize];
            at(c, e) + at(a, b) - at(a, e) - at(c, b)
        };
        Some(if count(&self.not_gentle) == 0 {
            0
        } else if count(&self.not_steep) == 0 {
            2
        } else {
            1
        })
    }
}

pub static TORCH_WEATHER_SKY_WALK: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

pub(crate) fn torch_weather_twin(src: String) -> String {
    if !TORCH_WEATHER_SKY_WALK.load(std::sync::atomic::Ordering::Relaxed) {
        return src;
    }
    crate::renderer::brush_pipeline::with_reader_edit(src.clone(), "sky_walked").unwrap_or_else(|| {
        log::warn!("TORCH_WEATHER_SKY_WALK: the sky's calls are no longer where `sky_walked` finds them; the twin is built as it was");
        src
    })
}

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
            Some((_, kinds)) => {
                let twin = crate::renderer::weather::with_weather_kinds(&src, kinds)?;

                let twin = if suffix.ends_with("spotless") { twin } else { torch_weather_twin(twin) };
                (twin, format!("terrain_weather{kind}_gentle_read{suffix}"))
            }
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

        assert!(SlopeSplit::ensure(&mut kept, &verts, &idx, &chunks, 35.0, 77, &class_of, None, None).is_some());
        let made = kept.as_ref().unwrap().indices.clone();
        assert!(kept.as_ref().unwrap().has_classes());

        for frame in 0..30 {
            let moved: Vec<CasterChunk> = chunks.iter().map(|c| CasterChunk { min: glam::Vec3::splat(frame as f32), ..*c }).collect();
            assert!(SlopeSplit::ensure(&mut kept, &verts, &idx, &moved, 35.0, 77, &class_of, None, None).is_none(), "built again on frame {frame}");
        }
        assert_eq!(kept.as_ref().unwrap().indices, made);

        assert!(SlopeSplit::ensure(&mut kept, &verts, &idx, &chunks, 35.0, 78, &class_of, None, None).is_some());
        assert!(SlopeSplit::ensure(&mut kept, &verts, &idx, &chunks, 35.0, 78, &class_of, None, None).is_none());
        assert!(SlopeSplit::ensure(&mut kept, &verts, &idx, &chunks, 30.0, 78, &class_of, None, None).is_some());
        let other: Vec<SolidVertex> = verts.iter().map(|v| { let mut o = *v; o.normal = [0.0, 1.0, 0.0]; o }).collect();
        assert!(SlopeSplit::ensure(&mut kept, &other, &idx, &chunks, 30.0, 78, &class_of, None, None).is_some());
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

        assert!(triangle_is_gentle([flat, flat, flat], 12.9f32.to_radians().cos(), 18.0));
        assert!(!triangle_is_gentle([flat, flat, flat], 13.1f32.to_radians().cos(), 18.0));
        assert!(!triangle_is_gentle([flat, flat, flat], 70.0f32.to_radians().cos(), 18.0));
    }

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

        let turning = [dir(60.0, 0.0), dir(60.0, 35.0), dir(60.0, 70.0)];
        assert!(triangle_is_steep(turning, 1.0, 18.0));

        assert!(!triangle_is_steep([dir(40.0, 0.0), dir(40.0, 120.0), dir(40.0, 240.0)], 1.0, 18.0));
        assert!(steepest_bound_y([dir(30.0, 0.0), dir(30.0, 100.0), dir(30.0, 200.0)]).is_none());
    }

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

        assert!(!triangle_is_steep([tilt(40.0, 0.0), tilt(40.0, 180.0), tilt(40.0, 0.0)], 1.0, 18.0));

        assert!(!triangle_is_steep([tilt(22.0, 0.0), tilt(22.0, 0.0), tilt(22.0, 0.0)], 1.0, 18.0));
        assert!(triangle_is_steep([tilt(24.0, 0.0), tilt(24.0, 0.0), tilt(24.0, 0.0)], 1.0, 18.0));

        let level = [0.0, 1.0, 0.0];
        assert!(triangle_is_steep([level, level, level], 24.0f32.to_radians().cos(), 18.0));
        assert!(!triangle_is_steep([level, level, level], 22.0f32.to_radians().cos(), 18.0));
    }

    #[test]
    fn the_split_keeps_every_chunk_its_own_triangles_gentle_first() {
        let flat = [0.0, 1.0, 0.0];
        let steep = [0.7, 0.7, 0.0];
        let verts = vec![vertex(flat), vertex(flat), vertex(flat), vertex(steep), vertex(flat), vertex(flat)];

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

        assert!(!s.has_classes());
        assert_eq!(s.class_ranges(1, [true, false, false]), ([(0, 9), (0, 0), (0, 0), (0, 0)], 1));
        assert_eq!(s.class_ranges(1, [false, true, true]).1, 0);

        let moved: Vec<SolidVertex> = verts.iter().map(|v| SolidVertex { position: [v.position[0], v.position[1] - 0.3, v.position[2]], ..*v }).collect();
        let again = SlopeSplit::new(&moved, &idx, &chunks, 18.0);
        assert_eq!(again.indices, s.indices);

        let mut posed = verts.clone();
        let split = SlopeSplit::new(&posed, &idx, &chunks, 18.0);
        for v in posed.iter_mut() {
            v.position[1] -= 0.3;
            v.normal[1] += 1e-7;
        }
        assert!(split.is_for(&posed, &idx, &chunks, 18.0, 0));

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

    #[test]
    fn a_blocks_chunks_are_laid_bucket_by_bucket_and_drawn_as_their_pieces() {
        use crate::renderer::ground_lod::{merged, tests::ground, GridInput, GroundLod};
        let cells = 160;
        let (positions, triangles) = ground(cells, &|x, z| 6.0 * (x * 0.031).sin() * (z * 0.027).cos() + 2.0 * (x * 0.35).sin() * (z * 0.4).sin(), &|_, _| false);
        let mut normals = vec![glam::Vec3::ZERO; positions.len()];
        for t in triangles.chunks_exact(3) {
            let f = (positions[t[1] as usize] - positions[t[0] as usize]).cross(positions[t[2] as usize] - positions[t[0] as usize]);
            t.iter().for_each(|v| normals[*v as usize] += f);
        }
        let verts: Vec<SolidVertex> = positions
            .iter()
            .zip(&normals)
            .map(|(p, n)| SolidVertex { position: p.to_array(), normal: n.normalize().to_array(), color: [1.0; 4], uv2: [0.0; 2], reflectivity: 0.0 })
            .collect();
        let mut lod = GroundLod::build(&GridInput { cells: [cells, cells], positions: &positions, triangles: &triangles, open: None, chunk_cells: 32, lists: false });
        let chunks = lod.all_units();
        let blocks = lod.blocks();
        let threshold = 18.0;
        let plain = SlopeSplit::classed_on(&verts, &lod.indices, &chunks, threshold, 0, &|_| 0, Some(lod.lattice), None);
        let split = SlopeSplit::classed_on(&verts, &lod.indices, &chunks, threshold, 0, &|_| 0, Some(lod.lattice), Some(&blocks));
        assert!(plain.piece_runs.is_empty());
        assert!(split.steep_total > 0 && split.gentle_total > 0, "the hills have both kinds at {threshold} degrees");
        let tri_set = |idx: &[u32]| {
            let mut t: Vec<[u32; 3]> = idx.chunks_exact(3).map(|t| { let mut t = [t[0], t[1], t[2]]; t.sort(); t }).collect();
            t.sort();
            t
        };
        assert_eq!(tri_set(&split.indices), tri_set(&plain.indices), "the same triangles");
        let real = chunks.len();
        let mut laid = 0;
        for (u, c) in chunks.iter().enumerate() {
            let (at, n) = split.piece_of[u];
            let own = &plain.indices[c.first_index as usize..(c.first_index + c.index_count) as usize];
            if n == 0 {
                assert_eq!(&split.indices[c.first_index as usize..(c.first_index + c.index_count) as usize], own);
                continue;
            }
            laid += 1;
            let mut held = Vec::new();
            assert!(n <= 9, "a chunk is at most nine pieces");
            for [first, count, id, bucket] in &split.piece_runs[at as usize..(at + n as u32) as usize] {
                let (id, slope) = (*id as usize, *bucket as usize / 3);
                assert_eq!(split.classes[id][*bucket as usize], *count);
                assert!(id >= real);
                let piece = &split.indices[*first as usize..(*first + *count) as usize];
                held.extend_from_slice(piece);

                assert_eq!(split.classes[id].iter().sum::<u32>(), *count);
                assert_eq!(split.classes[id][slope * 3..slope * 3 + 3].iter().sum::<u32>(), *count);
                assert_eq!(split.gentle[id], if slope == 0 { *count } else { 0 });
                assert_eq!(split.steep[id], if slope == 2 { *count } else { 0 });
                for t in piece.chunks_exact(3) {
                    let n = [verts[t[0] as usize].normal, verts[t[1] as usize].normal, verts[t[2] as usize].normal];
                    let f = face_height([verts[t[0] as usize].position, verts[t[1] as usize].position, verts[t[2] as usize].position]);
                    assert_eq!(slope_class(n, f, threshold), slope);
                }
            }
            assert_eq!(tri_set(&held), tri_set(own), "chunk {u}'s pieces are its triangles");
        }
        assert!(laid >= 16, "{laid} chunks laid in blocks");

        let scene_draws = |split: &SlopeSplit, lod: &mut GroundLod| {
            lod.set_pieces(split.pieces());
            let f = lod.frame(glam::Vec3::new(0.0, 30.0, 0.0), &[], false, glam::Vec3::ZERO, 0.0, 0);
            assert!(f.chunks.windows(2).all(|w| w[0].first_index < w[1].first_index), "by first index");
            let mut all: Vec<u32> = f.chunks.iter().flat_map(|c| split.indices[c.first_index as usize..(c.first_index + c.index_count) as usize].iter().copied()).collect();
            assert_eq!(tri_set(&all), tri_set(&plain.indices), "the frame draws every triangle once");
            all.clear();
            (0..3)
                .map(|which| {
                    merged(f.chunks.iter().zip(&f.units).map(|(c, u)| {
                        let (g, st) = (split.gentle[*u as usize].min(c.index_count), split.steep[*u as usize]);
                        let end = c.index_count - st.min(c.index_count - g);
                        match which {
                            0 => (c.first_index, g),
                            1 => (c.first_index + g, end - g),
                            _ => (c.first_index + end, c.index_count - end),
                        }
                    }))
                    .len()
                })
                .sum::<usize>()
        };
        let (before, after) = (scene_draws(&plain, &mut lod), scene_draws(&split, &mut lod));
        eprintln!("scene draws for {} chunks: {before} chunk by chunk, {after} block by block", chunks.len());
        assert!(after * 2 < before, "{after} against {before}");

        lod.set_pieces(split.pieces());
        let f = lod.frame(glam::Vec3::new(0.0, 30.0, 0.0), &[], false, glam::Vec3::ZERO, 0.0, 0);
        let drawn = |runs: &[(u32, u32)]| {
            tri_set(&runs.iter().flat_map(|(first, n)| split.indices[*first as usize..(*first + *n) as usize].iter().copied()).collect::<Vec<u32>>())
        };
        let all = f.wholes.runs(&f.chunks, |_| true);
        assert_eq!(all.len(), 1, "every chunk chosen: one range, {all:?}");
        assert_eq!(drawn(&all), tri_set(&plain.indices), "every triangle once");

        let (mut p, mut m) = (0usize, 0usize);
        let mut full: Option<(std::ops::Range<usize>, Vec<u32>)> = None;
        for g in &f.wholes.groups {
            if g[3] as usize - m == 16 {
                full = Some((p..g[2] as usize, f.wholes.members[m..g[3] as usize].to_vec()));
            }
            (p, m) = (g[2] as usize, g[3] as usize);
        }
        let (pieces, members) = full.expect("a block of 16 chunks");

        let own = |k: usize| -> Vec<[u32; 3]> {
            drawn(&pieces.clone().filter(|q| f.chunks[*q].min == f.chunks[k].min).map(|q| (f.chunks[q].first_index, f.chunks[q].index_count)).collect::<Vec<_>>())
        };

        let one = members[5] as usize;
        let seen_one = |k: usize| pieces.contains(&k) && f.chunks[k].min == f.chunks[one].min;
        let runs = f.wholes.runs(&f.chunks, seen_one);
        assert_eq!(drawn(&runs), own(one), "one unit seen: its triangles");

        let skip = f.chunks[members[3] as usize].min;
        let runs = f.wholes.runs(&f.chunks, |k| pieces.contains(&k) && f.chunks[k].min != skip);
        assert_eq!(runs.len(), 1, "15 of 16 seen: the block's range");

        let half = |k: usize| pieces.contains(&k) && members.iter().position(|first| f.chunks[*first as usize].min == f.chunks[k].min).is_some_and(|j| j % 2 == 0);
        let (new, old) = (f.wholes.runs(&f.chunks, half).len(), merged(f.chunks.iter().enumerate().filter(|(k, _)| half(*k)).map(|(_, c)| (c.first_index, c.index_count))).len());
        assert!(new <= old, "{new} against {old}");

        lod.set_pieces(plain.pieces());
        let f = lod.frame(glam::Vec3::new(0.0, 30.0, 0.0), &[], false, glam::Vec3::ZERO, 0.0, 0);
        assert_eq!(f.wholes.runs(&f.chunks, |_| true), merged(f.chunks.iter().map(|c| (c.first_index, c.index_count))));
    }

    #[test]
    fn the_ground_classes_sort_within_the_slope_classes_and_tile_each_chunk() {
        let mut seed = 11u64;
        let mut rnd = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f32) / (1u64 << 31) as f32
        };
        let tilt = |deg: f32| [deg.to_radians().sin(), deg.to_radians().cos(), 0.0];

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

                for w in runs[..n].windows(2) {
                    assert!(w[0].0 + w[0].1 < w[1].0);
                }
            }
            assert!(covered.iter().all(|&x| x == 1), "chunk {k}: the classes' runs do not tile it");

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
