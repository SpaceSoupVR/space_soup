use crate::renderer::brush_pipeline::{BrushVertex, SunFaces};
use crate::renderer::shadow::CasterChunk;

pub const WORLD_BRUSHES_BY_DEFAULT: bool = true;

const SAMPLES: usize = 61;

fn mix(h: u64, v: u32) -> u64 {
    (h ^ v as u64).wrapping_mul(0x0000_0100_0000_01B3)
}

fn samples(len: usize) -> impl Iterator<Item = usize> {
    let n = SAMPLES.min(len);
    (0..n).map(move |k| if n > 1 { k * (len - 1) / (n - 1) } else { 0 })
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BrushKey {
    vertices_at: usize,
    vertices: usize,
    indices_at: usize,
    indices: usize,

    pose: [u32; 4],
    sampled: u64,

    sun_mask: u64,
}

pub fn brush_key(vertices: &[BrushVertex], indices: &[u32], pose: (glam::Vec3, f32), sun_faces: Option<&SunFaces>) -> BrushKey {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for k in samples(vertices.len()) {
        let v = &vertices[k];
        for f in v.position.iter().chain(v.normal.iter()).chain(v.uv2_rect.iter()) {
            h = mix(h, f.to_bits());
        }
        h = mix(h, v.material);
    }
    for k in samples(indices.len()) {
        h = mix(h, indices[k]);
    }
    BrushKey {
        vertices_at: vertices.as_ptr() as usize,
        vertices: vertices.len(),
        indices_at: indices.as_ptr() as usize,
        indices: indices.len(),
        pose: [pose.0.x.to_bits(), pose.0.y.to_bits(), pose.0.z.to_bits(), pose.1.to_bits()],
        sampled: h,
        sun_mask: sun_faces.map_or(0, SunFaces::serial),
    }
}

pub struct Prepared {
    pub indices: Vec<u32>,
    pub sun_ends: [u32; 2],
    pub runs: Vec<CasterChunk>,
}

pub fn prepare(vertices: &[BrushVertex], indices: &[u32], sun_faces: Option<&SunFaces>) -> Prepared {
    let (indices, sun_ends) = match sun_faces {
        Some(faces) => faces.partition(vertices, indices),
        None => (indices.to_vec(), [0, 0]),
    };
    let runs = crate::renderer::space_warp::index_runs(&indices, |k| glam::Vec3::from(vertices[k as usize].position));
    Prepared { indices, sun_ends, runs }
}

pub struct StaticBrushes {
    pub vertices: wgpu::Buffer,
    pub indices: wgpu::Buffer,
    pub index_count: u32,

    pub sun_ends: [u32; 2],

    pub runs: Vec<CasterChunk>,
    key: Option<BrushKey>,
}

impl StaticBrushes {

    pub fn update(
        kept: &mut Option<StaticBrushes>,
        device: &wgpu::Device,
        vertices: &[BrushVertex],
        indices: &[u32],
        pose: (glam::Vec3, f32),
        sun_faces: Option<&SunFaces>,
    ) -> u64 {
        use wgpu::util::DeviceExt;
        let key = brush_key(vertices, indices, pose, sun_faces);
        if kept.as_ref().is_some_and(|k| k.key == Some(key)) {
            return 0;
        }
        let made = prepare(vertices, indices, sun_faces);
        let bytes = std::mem::size_of_val(vertices) as u64 + std::mem::size_of_val(made.indices.as_slice()) as u64;
        *kept = Some(StaticBrushes {
            vertices: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("brush_vb"),
                contents: bytemuck::cast_slice(vertices),
                usage: wgpu::BufferUsages::VERTEX,
            }),
            indices: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("brush_ib"),
                contents: bytemuck::cast_slice(&made.indices),
                usage: wgpu::BufferUsages::INDEX,
            }),
            index_count: made.indices.len() as u32,
            sun_ends: made.sun_ends,
            runs: made.runs,
            key: Some(key),
        });
        bytes
    }

    pub fn forget(kept: &mut Option<StaticBrushes>) {
        if let Some(k) = kept {
            k.key = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wall(n: usize, lift: f32) -> (Vec<BrushVertex>, Vec<u32>) {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        for q in 0..n {
            let base = vertices.len() as u32;
            let u = q as f32 / n as f32;
            for (dx, dy) in [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)] {
                vertices.push(BrushVertex {
                    position: [q as f32 + dx, lift + dy, 0.0],
                    normal: [0.0, 0.0, 1.0],
                    uv2_rect: [u, 0.0, u + 0.5 / n as f32, 0.5],
                    ..bytemuck::Zeroable::zeroed()
                });
            }
            indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        }
        (vertices, indices)
    }

    #[test]
    fn the_same_brushes_are_not_made_again_and_other_ones_are() {
        let (v, i) = wall(300, 0.0);
        let key = brush_key(&v, &i, (glam::Vec3::ZERO, 0.0), None);
        assert_eq!(key, brush_key(&v, &i, (glam::Vec3::ZERO, 0.0), None), "the same slices, the same rig");

        assert_ne!(key, brush_key(&v, &i, (glam::Vec3::new(0.5, 0.0, 0.0), 0.0), None), "the rig moved");
        assert_ne!(key, brush_key(&v, &i, (glam::Vec3::ZERO, 0.1), None), "the rig turned");

        let (mut moved, _) = wall(300, 0.0);
        let at = v.as_ptr();
        let mut same_place = v;
        for (a, b) in same_place.iter_mut().zip(moved.iter_mut()) {
            a.position[1] = b.position[1] + 2.0;
        }
        assert_eq!(same_place.as_ptr(), at);
        assert_ne!(key, brush_key(&same_place, &i, (glam::Vec3::ZERO, 0.0), None), "other vertices where the old ones were");

        assert_ne!(key, brush_key(&same_place[..same_place.len() - 4], &i[..i.len() - 6], (glam::Vec3::ZERO, 0.0), None));
    }

    fn mask(width: u32, height: u32) -> SunFaces {
        let mut rgba = vec![0u8; (width * height * 4) as usize];
        for y in 0..height {
            for x in 0..width {
                let t = &mut rgba[((y * width + x) * 4) as usize..][..4];
                t[0] = if x < width / 2 { 0 } else { 255 };
                t[1] = 255;
            }
        }
        SunFaces::from_mask(&rgba, width, height).unwrap()
    }

    #[test]
    fn another_sun_mask_makes_them_again() {
        let (v, i) = wall(40, 0.0);
        let (a, b) = (mask(64, 64), mask(64, 64));
        let key = brush_key(&v, &i, (glam::Vec3::ZERO, 0.0), Some(&a));
        assert_eq!(key, brush_key(&v, &i, (glam::Vec3::ZERO, 0.0), Some(&a)));
        assert_ne!(key, brush_key(&v, &i, (glam::Vec3::ZERO, 0.0), Some(&b)), "another mask, even one that reads the same");
        assert_ne!(key, brush_key(&v, &i, (glam::Vec3::ZERO, 0.0), None), "no mask");
    }

    #[test]
    fn what_is_kept_is_what_the_frame_made_every_frame() {
        let (v, i) = wall(2048, 0.0);
        let faces = mask(2048, 64);
        let made = prepare(&v, &i, Some(&faces));
        let (split, ends) = faces.partition(&v, &i);
        assert_eq!((made.indices.as_slice(), made.sun_ends), (split.as_slice(), ends));
        assert!(ends[0] > 0 && ends[1] > ends[0] || ends[1] as usize == i.len(), "both classes on this wall: {ends:?}");
        let runs = crate::renderer::space_warp::index_runs(&split, |k| glam::Vec3::from(v[k as usize].position));
        assert_eq!(made.runs.len(), runs.len());
        for (a, b) in made.runs.iter().zip(&runs) {
            assert_eq!((a.first_index, a.index_count, a.min, a.max), (b.first_index, b.index_count, b.min, b.max));
        }

        let mut all: Vec<[u32; 3]> = made.indices.chunks_exact(3).map(|t| [t[0], t[1], t[2]]).collect();
        let mut was: Vec<[u32; 3]> = i.chunks_exact(3).map(|t| [t[0], t[1], t[2]]).collect();
        all.sort_unstable();
        was.sort_unstable();
        assert_eq!(all, was);

        let plain = prepare(&v, &i, None);
        assert_eq!((plain.indices.as_slice(), plain.sun_ends), (i.as_slice(), [0, 0]));
    }

    #[test]
    fn a_large_levels_brushes_are_sent_once_in_the_world_and_every_moving_frame_posed() {
        let Some((device, _queue)) = crate::renderer::terrain_pipeline::tests::headless_gpu() else {
            eprintln!("skipping: no GPU");
            return;
        };
        let (world, indices) = wall(10_000, 0.0);
        let faces = mask(2048, 64);
        let mut kept = None;
        let first = StaticBrushes::update(&mut kept, &device, &world, &indices, (glam::Vec3::ZERO, 0.0), Some(&faces));
        assert_eq!(first, (world.len() * std::mem::size_of::<BrushVertex>() + indices.len() * 4) as u64);
        let mut sent_world = 0u64;
        for _frame in 0..50 {

            sent_world += StaticBrushes::update(&mut kept, &device, &world, &indices, (glam::Vec3::ZERO, 0.0), Some(&faces));
        }
        assert_eq!(sent_world, 0, "world brushes re-sent on a frame that changed nothing");
        assert_eq!(kept.as_ref().unwrap().index_count as usize, indices.len());

        let mut posed = world.clone();
        let mut kept = None;
        let mut sent_posed = 0u64;
        for frame in 0..50 {
            let offset = glam::Vec3::new(frame as f32 * 0.02, 0.0, 0.0);
            for (p, w) in posed.iter_mut().zip(&world) {
                p.position[0] = w.position[0] - offset.x;
            }
            sent_posed += StaticBrushes::update(&mut kept, &device, &posed, &indices, (offset, 0.0), Some(&faces));
        }
        assert_eq!(sent_posed, 50 * first, "posed brushes follow the rig: sent on every frame it moved");

        StaticBrushes::forget(&mut kept);
        assert_eq!(StaticBrushes::update(&mut kept, &device, &posed, &indices, (glam::Vec3::new(0.98, 0.0, 0.0), 0.0), Some(&faces)), first);
    }
}
