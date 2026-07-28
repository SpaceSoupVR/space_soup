use super::Color3;
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum CuboidStyle {
    Solid,
    Wireframe,
    SolidAndWire,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Cuboid {
    pub position: Vec3,
    pub half_size: Vec3,
    pub rotation: glam::Quat,
    pub color: Color3,
    pub wire_color: Color3,
    pub style: CuboidStyle,
    pub id: u64,
    /// Stable key (the scene object's own string id, not the ephemeral `id`
    /// above which is re-allocated every frame by callers that reconstruct
    /// `Cuboid`s from scratch each frame) used to look up this object's baked
    /// lightmap texture in `Renderer::cuboid_lightmaps`. `None` = use the
    /// pipeline's default white fallback.
    #[serde(default)]
    pub lightmap_key: Option<String>,
    /// 0.0 = not reflective, 1.0 = fully mirror-blended -- baked into
    /// `SolidVertex` and consumed by the screen-space-reflection pass (see
    /// `ssr.rs`), not by anything in this file directly.
    #[serde(default)]
    pub reflectivity: f32,
}

impl Cuboid {
    pub fn solid(position: Vec3, half_size: Vec3, color: Color3) -> Self {
        Self {
            position,
            half_size,
            rotation: glam::Quat::IDENTITY,
            color,
            wire_color: Color3(0, 0, 0, 255),
            style: CuboidStyle::Solid,
            id: new_id(),
            lightmap_key: None,
            reflectivity: 0.0,
        }
    }

    pub fn wireframe(position: Vec3, half_size: Vec3, color: Color3) -> Self {
        Self {
            position,
            half_size,
            rotation: glam::Quat::IDENTITY,
            color: Color3(0, 0, 0, 0),
            wire_color: color,
            style: CuboidStyle::Wireframe,
            id: new_id(),
            lightmap_key: None,
            reflectivity: 0.0,
        }
    }

    pub fn solid_and_wire(position: Vec3, half_size: Vec3, fill: Color3, wire: Color3) -> Self {
        Self {
            position,
            half_size,
            rotation: glam::Quat::IDENTITY,
            color: fill,
            wire_color: wire,
            style: CuboidStyle::SolidAndWire,
            id: new_id(),
            lightmap_key: None,
            reflectivity: 0.0,
        }
    }

    pub fn with_lightmap_key(mut self, key: impl Into<String>) -> Self {
        self.lightmap_key = Some(key.into());
        self
    }

    pub fn with_reflectivity(mut self, reflectivity: f32) -> Self {
        self.reflectivity = reflectivity.clamp(0.0, 1.0);
        self
    }

    pub fn model_matrix(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(self.half_size * 2.0, self.rotation, self.position)
    }

    pub fn aabb(&self) -> (Vec3, Vec3) {
        let min = self.position - self.half_size;
        let max = self.position + self.half_size;
        (min, max)
    }

    pub fn ray_intersect(&self, ray_origin: Vec3, ray_dir: Vec3) -> Option<f32> {
        let (min, max) = self.aabb();
        let inv = Vec3::new(1.0 / ray_dir.x, 1.0 / ray_dir.y, 1.0 / ray_dir.z);

        let t1 = (min - ray_origin) * inv;
        let t2 = (max - ray_origin) * inv;

        let tmin = t1.min(t2);
        let tmax = t1.max(t2);

        let t_enter = tmin.x.max(tmin.y).max(tmin.z);
        let t_exit = tmax.x.min(tmax.y).min(tmax.z);

        if t_enter <= t_exit && t_exit > 0.0 {
            Some(t_enter.max(0.0))
        } else {
            None
        }
    }

    pub fn snapshot(&self) -> CuboidSnapshot {
        CuboidSnapshot {
            position: self.position,
            half_size: self.half_size,
            rotation: self.rotation,
            color: self.color,
            wire_color: self.wire_color,
            style: self.style,
            reflectivity: self.reflectivity,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CuboidSnapshot {
    pub position: Vec3,
    pub half_size: Vec3,
    pub rotation: glam::Quat,
    pub color: Color3,
    pub wire_color: Color3,
    pub style: CuboidStyle,
    pub reflectivity: f32,
}

fn new_id() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(1);
    COUNTER.fetch_add(1, Ordering::Relaxed)
}

impl Serialize for Color3 {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        (self.0, self.1, self.2, self.3).serialize(s)
    }
}

impl<'de> Deserialize<'de> for Color3 {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let (r, g, b, a) = <(u8, u8, u8, u8)>::deserialize(d)?;
        Ok(Color3(r, g, b, a))
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct SolidVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 4],
    pub uv2: [f32; 2],
    pub reflectivity: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct WireVertex {
    pub position: [f32; 3],
    pub color: [f32; 4],
}

impl SolidVertex {
    pub const ATTRIBS: [wgpu::VertexAttribute; 5] = wgpu::vertex_attr_array![
        0 => Float32x3, 1 => Float32x3, 2 => Float32x4, 3 => Float32x2, 4 => Float32
    ];

    pub fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

impl WireVertex {
    pub const ATTRIBS: [wgpu::VertexAttribute; 2] =
        wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x4];

    pub fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

const CORNERS: [[f32; 3]; 8] = [
    [-0.5, -0.5, -0.5],
    [0.5, -0.5, -0.5],
    [0.5, 0.5, -0.5],
    [-0.5, 0.5, -0.5],
    [-0.5, -0.5, 0.5],
    [0.5, -0.5, 0.5],
    [0.5, 0.5, 0.5],
    [-0.5, 0.5, 0.5],
];

const FACES: [([usize; 4], [f32; 3]); 6] = [
    ([0, 1, 2, 3], [0.0, 0.0, -1.0]),
    ([5, 4, 7, 6], [0.0, 0.0, 1.0]),
    ([4, 0, 3, 7], [-1.0, 0.0, 0.0]),
    ([1, 5, 6, 2], [1.0, 0.0, 0.0]),
    ([3, 2, 6, 7], [0.0, 1.0, 0.0]),
    ([4, 5, 1, 0], [0.0, -1.0, 0.0]),
];

const EDGES: [[usize; 2]; 12] = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 0],
    [4, 5],
    [5, 6],
    [6, 7],
    [7, 4],
    [0, 4],
    [1, 5],
    [2, 6],
    [3, 7],
];

/// Shared box-unwrap convention for the lightmap atlas: a 3-column x 2-row grid
/// (one cell per cuboid face), with each face's local (s,t) derived purely from
/// its two non-normal local axes (u_axis/v_axis = the axes after the normal axis,
/// cyclically) normalized from the unit-cube's -0.5..0.5 extent to 0..1. This is
/// re-derived independently (not transmitted) by `lightmap_bake.py` (server) and
/// `objectRenderer.js` (frontend box UV2) from position+normal alone, so all three
/// must stay in lockstep with this exact formula.
fn face_uv2(face_idx: usize, local_pos: [f32; 3], normal: [f32; 3]) -> [f32; 2] {
    let n_axis = if normal[0].abs() > 0.5 {
        0
    } else if normal[1].abs() > 0.5 {
        1
    } else {
        2
    };
    let u_axis = (n_axis + 1) % 3;
    let v_axis = (n_axis + 2) % 3;
    let s = local_pos[u_axis] + 0.5;
    let t = local_pos[v_axis] + 0.5;
    let col = (face_idx % 3) as f32;
    let row = (face_idx / 3) as f32;
    [(col + s) / 3.0, (row + t) / 2.0]
}

pub fn build_solid_mesh_one(c: &Cuboid) -> Option<(Vec<SolidVertex>, Vec<u32>)> {
    if matches!(c.style, CuboidStyle::Wireframe) {
        return None;
    }

    let mut verts: Vec<SolidVertex> = Vec::with_capacity(24);
    let mut indices: Vec<u32> = Vec::with_capacity(36);

    let model = c.model_matrix();
    let color = c.color.to_linear();

    for (face_idx, (corners, normal)) in FACES.iter().enumerate() {
        let face_base = verts.len() as u32;
        let world_normal = c.rotation * Vec3::from(*normal);

        for &ci in corners {
            let local = CORNERS[ci];
            let world = model.transform_point3(Vec3::from(local));
            let uv2 = face_uv2(face_idx, local, *normal);
            verts.push(SolidVertex {
                position: world.into(),
                normal: world_normal.into(),
                color,
                uv2,
                reflectivity: c.reflectivity,
            });
        }
        indices.extend_from_slice(&[
            face_base,
            face_base + 1,
            face_base + 2,
            face_base,
            face_base + 2,
            face_base + 3,
        ]);
    }
    Some((verts, indices))
}

pub fn build_wire_mesh_one(c: &Cuboid) -> Option<(Vec<WireVertex>, Vec<u32>)> {
    if matches!(c.style, CuboidStyle::Solid) {
        return None;
    }

    let mut verts: Vec<WireVertex> = Vec::with_capacity(8);
    let mut indices: Vec<u32> = Vec::with_capacity(24);

    let model = c.model_matrix();
    let color = c.wire_color.to_linear();

    for edge in &EDGES {
        let base = verts.len() as u32;
        for &ci in edge {
            let world = model.transform_point3(Vec3::from(CORNERS[ci]));
            verts.push(WireVertex {
                position: world.into(),
                color,
            });
        }
        indices.push(base);
        indices.push(base + 1);
    }
    Some((verts, indices))
}

/// Same combined vertex/index buffer `build_solid_mesh` always produced, plus
/// a per-cuboid `(lightmap_key, index_start, count, reflectivity)` range into
/// that shared index buffer -- lets a caller bind a different lightmap
/// texture per `draw_indexed` range without needing a separate buffer per
/// cuboid, and (via `reflectivity`) pick out which ranges need a second,
/// reflective-only redraw pass (see `ssr.rs`).
pub fn build_solid_mesh_with_ranges(
    cuboids: &[Cuboid],
) -> (Vec<SolidVertex>, Vec<u32>, Vec<(Option<String>, u32, u32, f32)>) {
    let mut verts: Vec<SolidVertex> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    let mut ranges: Vec<(Option<String>, u32, u32, f32)> = Vec::new();
    for c in cuboids {
        if let Some((v, i)) = build_solid_mesh_one(c) {
            let base = verts.len() as u32;
            let index_start = indices.len() as u32;
            let count = i.len() as u32;
            verts.extend(v);
            indices.extend(i.into_iter().map(|x| x + base));
            ranges.push((c.lightmap_key.clone(), index_start, count, c.reflectivity));
        }
    }
    (verts, indices, ranges)
}

pub fn build_wire_mesh(cuboids: &[Cuboid]) -> (Vec<WireVertex>, Vec<u32>) {
    let mut verts: Vec<WireVertex> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    for c in cuboids {
        if let Some((v, i)) = build_wire_mesh_one(c) {
            let base = verts.len() as u32;
            verts.extend(v);
            indices.extend(i.into_iter().map(|x| x + base));
        }
    }
    (verts, indices)
}

