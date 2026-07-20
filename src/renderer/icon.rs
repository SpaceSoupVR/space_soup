use std::sync::Arc;

use glam::{Quat, Vec3};
use wgpu::util::DeviceExt;
use wgpu::{BindGroupLayout, Device, Queue};

use super::mesh::{create_texture_from_rgba, GltfMesh, MeshPrimitive};
use super::panel::quad_geometry;

/// Editor-only gizmo marker drawn over non-visual objects (lights, sound
/// sources) so they're pickable/visible in the viewport despite having no
/// mesh or solid cuboid of their own.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IconKind {
    Light,
    Sound,
}

const ICON_SIZE_M: f32 = 0.25;
const TEX_SIZE: u32 = 32;

pub struct IconAssets {
    light: GltfMesh,
    sound: GltfMesh,
}

impl IconAssets {
    pub fn new(device: &Device, queue: &Queue, texture_layout: &BindGroupLayout) -> Self {
        Self {
            light: build_icon_mesh(device, queue, texture_layout, &draw_light_bulb()),
            sound: build_icon_mesh(device, queue, texture_layout, &draw_speaker()),
        }
    }

    /// Cheap clone — the GPU vertex/index/texture handles are shared; the
    /// caller sets `.position`/`.rotation` on the returned value per instance.
    pub fn mesh_for(&self, kind: IconKind) -> GltfMesh {
        match kind {
            IconKind::Light => self.light.clone(),
            IconKind::Sound => self.sound.clone(),
        }
    }
}

/// Faces the icon toward the camera by matching its orientation exactly —
/// simple and stable for a small marker, no gimbal issues to worry about.
pub fn billboard_rotation(camera_rotation: Quat) -> Quat {
    camera_rotation
}

fn build_icon_mesh(
    device: &Device,
    queue: &Queue,
    texture_layout: &BindGroupLayout,
    rgba: &[u8],
) -> GltfMesh {
    let (vertices, indices) = quad_geometry(ICON_SIZE_M, ICON_SIZE_M);

    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("icon_vb"),
        contents: bytemuck::cast_slice(&vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("icon_ib"),
        contents: bytemuck::cast_slice(&indices),
        usage: wgpu::BufferUsages::INDEX,
    });
    let texture = Arc::new(create_texture_from_rgba(
        device,
        queue,
        texture_layout,
        rgba,
        TEX_SIZE,
        TEX_SIZE,
    ));

    GltfMesh {
        primitives: vec![MeshPrimitive {
            vertices,
            indices,
            texture,
            vertex_buffer,
            index_buffer,
            double_sided: false,
        }],
        skin: None,
        position: Vec3::ZERO,
        rotation: Quat::IDENTITY,
        scale: Vec3::ONE,
        bounding_radius: ICON_SIZE_M,
    }
}

fn blank_rgba() -> Vec<u8> {
    vec![0u8; (TEX_SIZE * TEX_SIZE * 4) as usize]
}

fn set_px(buf: &mut [u8], x: i32, y: i32, color: [u8; 4]) {
    if x < 0 || y < 0 || x >= TEX_SIZE as i32 || y >= TEX_SIZE as i32 {
        return;
    }
    let i = ((y as u32 * TEX_SIZE + x as u32) * 4) as usize;
    buf[i..i + 4].copy_from_slice(&color);
}

fn draw_light_bulb() -> Vec<u8> {
    let mut buf = blank_rgba();
    let glass = [255, 240, 180, 255];
    let base = [200, 200, 210, 255];
    let center = (16.0f32, 12.0f32);
    let radius = 8.0f32;

    for y in 0..TEX_SIZE as i32 {
        for x in 0..TEX_SIZE as i32 {
            let dx = x as f32 - center.0;
            let dy = y as f32 - center.1;
            if dx * dx + dy * dy <= radius * radius {
                set_px(&mut buf, x, y, glass);
            }
        }
    }
    for y in 20..26 {
        for x in 12..20 {
            set_px(&mut buf, x, y, base);
        }
    }
    buf
}

fn draw_speaker() -> Vec<u8> {
    let mut buf = blank_rgba();
    let body = [235, 235, 240, 255];
    let center = (12.0f32, 16.0f32);

    for y in 11..21 {
        for x in 4..12 {
            set_px(&mut buf, x, y, body);
        }
    }
    for x in 12..22 {
        let t = (x as f32 - 12.0) / 10.0;
        let half_height = 4.0 + t * 6.0;
        let y_lo = (16.0 - half_height).round() as i32;
        let y_hi = (16.0 + half_height).round() as i32;
        for y in y_lo..y_hi {
            set_px(&mut buf, x, y, body);
        }
    }
    for &radius in &[25.0f32, 29.0f32] {
        for y in 0..TEX_SIZE as i32 {
            for x in 0..TEX_SIZE as i32 {
                if x < 14 {
                    continue;
                }
                let dx = x as f32 - center.0;
                let dy = y as f32 - center.1;
                let dist = (dx * dx + dy * dy).sqrt();
                if dist <= radius && dist >= radius - 2.0 {
                    set_px(&mut buf, x, y, body);
                }
            }
        }
    }
    buf
}
