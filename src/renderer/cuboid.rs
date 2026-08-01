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

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum CuboidShape {
    Box,
    Cylinder,
}

impl Default for CuboidShape {
    fn default() -> Self {
        Self::Box
    }
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
    #[serde(default)]
    pub lightmap_key: Option<String>,
    #[serde(default)]
    pub reflectivity: f32,
    #[serde(default)]
    pub shape: CuboidShape,
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
            shape: CuboidShape::Box,
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
            shape: CuboidShape::Box,
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
            shape: CuboidShape::Box,
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
            shape: self.shape,
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
    pub shape: CuboidShape,
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
    if matches!(c.shape, CuboidShape::Cylinder) {
        return Some(build_cylinder_solid_mesh(c));
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

fn build_cylinder_solid_mesh(c: &Cuboid) -> (Vec<SolidVertex>, Vec<u32>) {
    const SEGMENTS: usize = 24;

    let mut verts: Vec<SolidVertex> = Vec::with_capacity(SEGMENTS * 4 + 2);
    let mut indices: Vec<u32> = Vec::with_capacity(SEGMENTS * 12);

    let model = c.model_matrix();
    let color = c.color.to_linear();
    let uv2 = [0.5, 0.5];
    let angle_at = |i: usize| (i as f32 / SEGMENTS as f32) * std::f32::consts::TAU;

    let side_base = verts.len() as u32;
    for i in 0..=SEGMENTS {
        let (sin, cos) = angle_at(i % SEGMENTS).sin_cos();
        let radial = Vec3::new(cos * 0.5, 0.0, sin * 0.5);
        let world_normal = (c.rotation * radial).normalize();
        for y in [0.5, -0.5] {
            let world = model.transform_point3(radial + Vec3::new(0.0, y, 0.0));
            verts.push(SolidVertex {
                position: world.into(),
                normal: world_normal.into(),
                color,
                uv2,
                reflectivity: c.reflectivity,
            });
        }
    }
    for i in 0..SEGMENTS as u32 {
        let top0 = side_base + i * 2;
        let bot0 = top0 + 1;
        let top1 = side_base + (i + 1) * 2;
        let bot1 = top1 + 1;
        indices.extend_from_slice(&[top0, bot0, bot1, top0, bot1, top1]);
    }

    for (y, normal_sign, flip) in [(0.5, 1.0, false), (-0.5, -1.0, true)] {
        let cap_base = verts.len() as u32;
        let world_normal = c.rotation * Vec3::new(0.0, normal_sign, 0.0);
        verts.push(SolidVertex {
            position: model.transform_point3(Vec3::new(0.0, y, 0.0)).into(),
            normal: world_normal.into(),
            color,
            uv2,
            reflectivity: c.reflectivity,
        });
        for i in 0..SEGMENTS {
            let (sin, cos) = angle_at(i).sin_cos();
            let local = Vec3::new(cos * 0.5, y, sin * 0.5);
            verts.push(SolidVertex {
                position: model.transform_point3(local).into(),
                normal: world_normal.into(),
                color,
                uv2,
                reflectivity: c.reflectivity,
            });
        }
        for i in 0..SEGMENTS as u32 {
            let a = cap_base + 1 + i;
            let b = cap_base + 1 + (i + 1) % SEGMENTS as u32;
            if flip {
                indices.extend_from_slice(&[cap_base, b, a]);
            } else {
                indices.extend_from_slice(&[cap_base, a, b]);
            }
        }
    }

    (verts, indices)
}

pub fn build_wire_mesh_one(c: &Cuboid) -> Option<(Vec<WireVertex>, Vec<u32>)> {
    if matches!(c.style, CuboidStyle::Solid) || matches!(c.shape, CuboidShape::Cylinder) {
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

#[cfg(test)]
mod cylinder_test {
    use super::*;

    fn cylinder(color: Color3) -> Cuboid {
        let mut c = Cuboid::solid(Vec3::ZERO, Vec3::new(1.0, 0.5, 1.0), color);
        c.shape = CuboidShape::Cylinder;
        c
    }

    #[test]
    fn vertex_and_index_counts_match_the_side_plus_two_cap_layout() {
        const SEGMENTS: usize = 24;
        let (verts, indices) = build_cylinder_solid_mesh(&cylinder(Color3(255, 255, 255, 255)));

        let side_verts = (SEGMENTS + 1) * 2;
        let side_indices = SEGMENTS * 6;
        let cap_verts = SEGMENTS + 1;
        let cap_indices = SEGMENTS * 3;

        assert_eq!(verts.len(), side_verts + cap_verts * 2);
        assert_eq!(indices.len(), side_indices + cap_indices * 2);
        assert!(indices.iter().all(|&i| (i as usize) < verts.len()));
    }

    #[test]
    fn side_wall_winds_outward_matching_the_box_faces_convention() {
        let (verts, indices) = build_cylinder_solid_mesh(&cylinder(Color3(255, 255, 255, 255)));
        let v0 = Vec3::from(verts[indices[0] as usize].position);
        let v1 = Vec3::from(verts[indices[1] as usize].position);
        let v2 = Vec3::from(verts[indices[2] as usize].position);
        let outward = Vec3::from(verts[indices[0] as usize].normal);

        let winding_normal = (v1 - v0).cross(v2 - v0);
        assert!(
            winding_normal.dot(outward) < 0.0,
            "expected cross(edge1, edge2) to point opposite the outward normal, got {winding_normal:?} vs outward {outward:?}"
        );
    }

    #[test]
    fn wireframe_style_and_cylinder_shape_both_suppress_the_wire_mesh() {
        let mut c = cylinder(Color3(255, 255, 255, 255));
        assert!(build_wire_mesh_one(&c).is_none(), "cylinders have no wireframe path");
        c.style = CuboidStyle::Wireframe;
        assert!(build_wire_mesh_one(&c).is_none());
    }
}

