//! THE GROUND'S VERTICES AND INDICES KEPT ON THE GPU (`Levers::ground_buffers`).
//!
//! The frame drew the ground from a vertex and an index buffer it made NEW
//! every frame: the whole ground appended to the frame's cuboids, every index
//! rebased past them, and both sent to fresh GPU buffers -- 5.6 MB of vertices
//! and 1.7 MB of indices a frame on test_room, standing still, growing with
//! the level's area (`docs/scale-audit-2026-10-13.md`, F1).
//!
//! Here the two buffers live across frames. The ground comes FIRST in each, so
//! its indices are the app's own, unrebased, and the frame's cuboids -- a few
//! hundred vertices -- follow it, their indices moved past the ground's
//! vertices. A frame writes:
//!
//! - the cuboids, always (they are rebuilt each frame);
//! - the ground's VERTICES only when the ones handed in are not the ones last
//!   written: another slice, another length, the rig somewhere else (the app
//!   re-poses them when it moves), or other contents at the places sampled;
//! - the ground's INDICES only when they are not the ones last written (the
//!   slope split redone, another level).
//!
//! The bytes on the GPU are the bytes the frame used to send, at other
//! offsets; every draw takes the same triangles in the same order. The same
//! picture.

use crate::renderer::cuboid::SolidVertex;

/// Vertices and indices sampled into a key: enough to tell another level's
/// ground, or the same one re-posed, from the one on the GPU when the slice
/// and its length happen to be the same.
const SAMPLES: usize = 61;

/// What the ground's vertices on the GPU were written from.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VertexKey {
    at: usize,
    len: usize,
    /// The rig the app posed them for, bit for bit.
    pose: [u32; 4],
    sampled: u64,
}

/// What the ground's indices on the GPU were written from.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IndexKey {
    at: usize,
    len: usize,
    sampled: u64,
}

fn mix(h: u64, v: u32) -> u64 {
    (h ^ v as u64).wrapping_mul(0x0000_0100_0000_01B3)
}

/// A slice's samples, evenly through it, first and last included.
fn samples(len: usize) -> impl Iterator<Item = usize> {
    let n = SAMPLES.min(len);
    (0..n).map(move |k| if n > 1 { k * (len - 1) / (n - 1) } else { 0 })
}

pub fn vertex_key(vertices: &[SolidVertex], offset: glam::Vec3, yaw: f32) -> VertexKey {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for k in samples(vertices.len()) {
        let v = &vertices[k];
        for f in v.position.iter().chain(v.normal.iter()) {
            h = mix(h, f.to_bits());
        }
    }
    VertexKey {
        at: vertices.as_ptr() as usize,
        len: vertices.len(),
        pose: [offset.x.to_bits(), offset.y.to_bits(), offset.z.to_bits(), yaw.to_bits()],
        sampled: h,
    }
}

pub fn index_key(indices: &[u32]) -> IndexKey {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for k in samples(indices.len()) {
        h = mix(h, indices[k]);
    }
    IndexKey { at: indices.as_ptr() as usize, len: indices.len(), sampled: h }
}

/// The frame's solid ranges with the ground first in the buffers: every
/// cuboid range moved past the ground's `ground_indices`, in its own order,
/// and the ground's range -- the whole of its indices, from 0 -- LAST, where
/// the frame has always listed it. Returns the ground's range.
pub fn ranges_after_ground<T>(ranges: &mut Vec<(Option<T>, u32, u32, f32)>, ground_indices: u32) -> Option<(u32, u32)> {
    for r in ranges.iter_mut() {
        r.1 += ground_indices;
    }
    (ground_indices > 0).then(|| {
        ranges.push((None, 0, ground_indices, 0.0));
        (0, ground_indices)
    })
}

/// The cuboids' indices moved past the ground's vertices.
pub fn rebased(indices: &[u32], ground_vertices: u32) -> Vec<u32> {
    indices.iter().map(|i| i + ground_vertices).collect()
}

/// Room for this many of a thing and what a frame may add to it, so the
/// buffers are not remade as the cuboids come and go.
fn capacity(ground: usize, others: usize, least: usize) -> usize {
    ground + (others * 2).max(least)
}

/// What one frame sent to the GPU, bytes.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Written {
    pub ground_vertices: u64,
    pub ground_indices: u64,
    pub others: u64,
    /// The buffers were made anew this frame.
    pub remade: bool,
}

impl Written {
    pub fn bytes(&self) -> u64 {
        self.ground_vertices + self.ground_indices + self.others
    }
}

/// The two buffers and what their ground was written from.
pub struct StaticSolid {
    pub vertices: wgpu::Buffer,
    pub indices: wgpu::Buffer,
    vertex_room: usize,
    index_room: usize,
    vertex_key: Option<VertexKey>,
    index_key: Option<IndexKey>,
}

const VERTEX_BYTES: u64 = std::mem::size_of::<SolidVertex>() as u64;

impl StaticSolid {
    fn buffers(device: &wgpu::Device, vertex_room: usize, index_room: usize) -> (wgpu::Buffer, wgpu::Buffer) {
        (
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("solid_vb"),
                size: vertex_room as u64 * VERTEX_BYTES,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("solid_ib"),
                size: index_room as u64 * 4,
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
        )
    }

    /// This frame's solid geometry onto the GPU: the ground (`None`: a level
    /// without one) as the app posed it for the rig at `pose`, then the
    /// frame's cuboids, whose `cuboid_indices` count from their own first
    /// vertex. `kept`: `None` on the first frame.
    pub fn update(
        kept: &mut Option<StaticSolid>,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        ground: Option<(&[SolidVertex], &[u32])>,
        pose: (glam::Vec3, f32),
        cuboid_vertices: &[SolidVertex],
        cuboid_indices: &[u32],
    ) -> Written {
        let (gv, gi) = ground.unwrap_or((&[], &[]));
        let mut written = Written::default();
        let (need_v, need_i) = (gv.len() + cuboid_vertices.len(), gi.len() + cuboid_indices.len());
        // Made anew when the ground is another size (its part is then another
        // length, and what follows it somewhere else) or the cuboids outgrow
        // their room.
        let fits = kept.as_ref().is_some_and(|k| {
            k.vertex_key.map_or(0, |v| v.len) == gv.len()
                && k.index_key.map_or(0, |i| i.len) == gi.len()
                && need_v <= k.vertex_room
                && need_i <= k.index_room
        });
        if !fits {
            let vertex_room = capacity(gv.len(), cuboid_vertices.len(), 4096);
            let index_room = capacity(gi.len(), cuboid_indices.len(), 8192);
            let (vertices, indices) = Self::buffers(device, vertex_room, index_room);
            *kept = Some(StaticSolid { vertices, indices, vertex_room, index_room, vertex_key: None, index_key: None });
            written.remade = true;
        }
        let k = kept.as_mut().expect("made above");
        let vkey = vertex_key(gv, pose.0, pose.1);
        if k.vertex_key != Some(vkey) {
            if !gv.is_empty() {
                queue.write_buffer(&k.vertices, 0, bytemuck::cast_slice(gv));
                written.ground_vertices = gv.len() as u64 * VERTEX_BYTES;
            }
            k.vertex_key = Some(vkey);
        }
        let ikey = index_key(gi);
        if k.index_key != Some(ikey) {
            if !gi.is_empty() {
                queue.write_buffer(&k.indices, 0, bytemuck::cast_slice(gi));
                written.ground_indices = gi.len() as u64 * 4;
            }
            k.index_key = Some(ikey);
        }
        if !cuboid_vertices.is_empty() {
            queue.write_buffer(&k.vertices, gv.len() as u64 * VERTEX_BYTES, bytemuck::cast_slice(cuboid_vertices));
            let moved = rebased(cuboid_indices, gv.len() as u32);
            queue.write_buffer(&k.indices, gi.len() as u64 * 4, bytemuck::cast_slice(&moved));
            written.others = cuboid_vertices.len() as u64 * VERTEX_BYTES + moved.len() as u64 * 4;
        }
        written
    }

    /// Forget what the ground was written from: the next frame writes it
    /// again whatever it is handed. For whoever changes the ground's data in
    /// place.
    pub fn forget(kept: &mut Option<StaticSolid>) {
        if let Some(k) = kept {
            k.vertex_key = None;
            k.index_key = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    fn ground(n: usize, lift: f32) -> Vec<SolidVertex> {
        (0..n)
            .map(|k| {
                let mut v: SolidVertex = bytemuck::Zeroable::zeroed();
                v.position = [k as f32, lift, (k * 7 % 13) as f32];
                v.normal = [0.0, 1.0, 0.0];
                v
            })
            .collect()
    }

    #[test]
    fn the_same_ground_for_the_same_rig_is_not_written_again() {
        let g = ground(5000, 0.0);
        let a = vertex_key(&g, Vec3::new(1.0, 0.0, 2.0), 0.5);
        assert_eq!(a, vertex_key(&g, Vec3::new(1.0, 0.0, 2.0), 0.5));
        // The rig a millimetre on, or turned: the app has posed it again.
        assert_ne!(a, vertex_key(&g, Vec3::new(1.001, 0.0, 2.0), 0.5));
        assert_ne!(a, vertex_key(&g, Vec3::new(1.0, 0.0, 2.0), 0.5001));
        // Another length, and another slice of the same length and contents.
        assert_ne!(a, vertex_key(&g[..4999], Vec3::new(1.0, 0.0, 2.0), 0.5));
        let copy = g.clone();
        assert_ne!(a, vertex_key(&copy, Vec3::new(1.0, 0.0, 2.0), 0.5));
    }

    /// A level loaded into the allocation the last one left: the same slice
    /// and length, the rig where it was. Its contents tell it apart.
    #[test]
    fn other_ground_in_the_same_slice_is_written() {
        let mut g = ground(5000, 0.0);
        let before = vertex_key(&g, Vec3::ZERO, 0.0);
        for v in g.iter_mut() {
            v.position[1] += 3.0;
        }
        assert_ne!(before, vertex_key(&g, Vec3::ZERO, 0.0));
        let mut i: Vec<u32> = (0..9000).collect();
        let before = index_key(&i);
        assert_eq!(before, index_key(&i));
        i.reverse();
        assert_ne!(before, index_key(&i));
        // Empty is a key too, and the same one every time.
        assert_eq!(index_key(&[]), index_key(&[]));
        assert_eq!(vertex_key(&[], Vec3::ZERO, 0.0).len, 0);
    }

    #[test]
    fn the_samples_span_the_slice_and_stay_in_it() {
        for len in [0usize, 1, 2, 60, 61, 62, 5000, 107_584] {
            let s: Vec<usize> = samples(len).collect();
            assert_eq!(s.len(), SAMPLES.min(len));
            assert!(s.iter().all(|&k| k < len.max(1)));
            if len > 1 {
                assert_eq!((s[0], *s.last().unwrap()), (0, len - 1));
                assert!(s.windows(2).all(|w| w[0] < w[1]) || len < SAMPLES);
            }
        }
    }

    /// With the ground first, the cuboids' ranges move past its indices and
    /// their indices past its vertices; the ground's range is the whole of
    /// its own, listed last as it always was.
    #[test]
    fn the_cuboids_follow_the_ground_and_every_range_names_the_same_triangles() {
        // What the frame built before: cuboids first, the ground appended.
        let cuboid_vertices = 8u32;
        let cuboid_indices: Vec<u32> = vec![0, 1, 2, 2, 3, 0, 4, 5, 6, 6, 7, 4];
        let ground_indices: Vec<u32> = vec![0, 1, 2, 1, 3, 2];
        let ground_vertices = 4u32;
        let mut old_idx = cuboid_indices.clone();
        old_idx.extend(ground_indices.iter().map(|i| i + cuboid_vertices));
        let old_ranges = vec![(Some("a"), 0u32, 6u32, 1.0f32), (Some("b"), 6, 6, 2.0), (None, 12, 6, 0.0)];
        // Now: the ground first.
        let mut new_idx = ground_indices.clone();
        new_idx.extend(rebased(&cuboid_indices, ground_vertices));
        let mut ranges = vec![(Some("a"), 0u32, 6u32, 1.0f32), (Some("b"), 6, 6, 2.0)];
        let terrain = ranges_after_ground(&mut ranges, ground_indices.len() as u32);
        assert_eq!(terrain, Some((0, 6)));
        assert_eq!(ranges.len(), old_ranges.len());
        // A vertex's place before and now.
        let place = |old: u32| if old < cuboid_vertices { old + ground_vertices } else { old - cuboid_vertices };
        for (new, old) in ranges.iter().zip(&old_ranges) {
            assert_eq!((new.0, new.2, new.3), (old.0, old.2, old.3), "the same entry in the same order");
            let was = &old_idx[old.1 as usize..(old.1 + old.2) as usize];
            let is = &new_idx[new.1 as usize..(new.1 + new.2) as usize];
            assert_eq!(was.iter().map(|&i| place(i)).collect::<Vec<_>>(), is, "the same vertices in the same order");
        }
        // No ground: nothing moves and nothing is listed for it.
        let mut alone = vec![(Some("a"), 0u32, 6u32, 1.0f32)];
        assert_eq!(ranges_after_ground(&mut alone, 0), None);
        assert_eq!(alone, vec![(Some("a"), 0, 6, 1.0)]);
        assert_eq!(rebased(&cuboid_indices, 0), cuboid_indices);
    }

    #[test]
    fn the_buffers_have_room_for_the_ground_and_twice_the_cuboids() {
        assert_eq!(capacity(1000, 0, 4096), 5096);
        assert_eq!(capacity(1000, 3000, 4096), 7000);
        assert_eq!(capacity(0, 0, 8192), 8192);
        assert_eq!(Written { ground_vertices: 5, ground_indices: 7, others: 11, remade: false }.bytes(), 23);
    }
}
