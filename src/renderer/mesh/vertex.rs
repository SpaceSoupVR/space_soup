use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

use super::texture::LoadedTexture;
use crate::renderer::layered_mesh_pipeline::LayeredVertex;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct MeshVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
    pub uv2: [f32; 2],

    pub emissive: u32,

    pub uv2_rect: [u16; 4],
}

impl MeshVertex {
    pub const ATTRIBS: [wgpu::VertexAttribute; 6] = wgpu::vertex_attr_array![
        0 => Float32x3, 1 => Float32x3, 2 => Float32x2, 3 => Float32x2, 4 => Uint32,

        6 => Unorm16x4
    ];

    pub const WHOLE_ATLAS: [u16; 4] = [0, 0, u16::MAX, u16::MAX];

    pub fn uv2_rect_of(uv2: [[f32; 2]; 3]) -> [u16; 4] {
        let s = f32::from(u16::MAX);
        let lo = |k: usize| (uv2.iter().map(|c| c[k]).fold(f32::MAX, f32::min).clamp(0.0, 1.0) * s).floor() as u16;
        let hi = |k: usize| (uv2.iter().map(|c| c[k]).fold(f32::MIN, f32::max).clamp(0.0, 1.0) * s).ceil() as u16;
        [lo(0), lo(1), hi(0), hi(1)]
    }

    pub const MAX_EMISSIVE_STRENGTH: f32 = 64.0;

    pub fn pack_emissive(rgb: [f32; 3]) -> u32 {
        let enc = |v: f32| -> u32 {
            let l = v.clamp(0.0, 1.0);
            let s = if l <= 0.003_130_8 { l * 12.92 } else { 1.055 * l.powf(1.0 / 2.4) - 0.055 };
            (s * 255.0).round() as u32
        };
        let peak = rgb[0].max(rgb[1]).max(rgb[2]);
        if !(peak > 0.0) {
            return Self::NO_EMISSIVE;
        }

        let scale = peak.max(1.0).min(Self::MAX_EMISSIVE_STRENGTH);
        let inv = 1.0 / scale;
        let a = ((scale / Self::MAX_EMISSIVE_STRENGTH) * 255.0).round() as u32;
        enc(rgb[0] * inv) | (enc(rgb[1] * inv) << 8) | (enc(rgb[2] * inv) << 16) | (a << 24)
    }

    pub const NO_EMISSIVE: u32 = 255 << 24;

    pub fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct SkinnedMeshVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
    pub joint_ids: [u32; 4],
    pub joint_weights: [f32; 4],
}

impl SkinnedMeshVertex {
    pub const ATTRIBS: [wgpu::VertexAttribute; 5] = wgpu::vertex_attr_array![
        0 => Float32x3,
        1 => Float32x3,
        2 => Float32x2,
        3 => Uint32x4,
        4 => Float32x4,
    ];

    pub fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }

    pub fn dominant_joint(&self) -> usize {
        let mut best = 0;
        for i in 1..4 {
            if self.joint_weights[i] > self.joint_weights[best] {
                best = i;
            }
        }
        self.joint_ids[best] as usize
    }
}

#[derive(Clone)]
pub struct LayeredPrimitive {
    pub vertices: Vec<LayeredVertex>,
    pub vertex_buffer: wgpu::Buffer,
}

#[derive(Clone)]
pub struct MeshPrimitive {
    pub vertices: Vec<MeshVertex>,
    pub indices: Vec<u32>,
    pub texture: Arc<LoadedTexture>,
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,

    pub layered: Option<LayeredPrimitive>,

    pub casts_shadow: bool,

    pub blended: bool,

    pub thin: Option<super::thin_parts::ThinParts>,
}

#[cfg(test)]
mod emissive_tests {
    use super::MeshVertex;

    fn unpack(packed: u32) -> [f32; 3] {

        let dec = |b: u32| {
            let c = b as f32 / 255.0;
            if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
        };
        let scale = ((packed >> 24) & 255) as f32 / 255.0 * MeshVertex::MAX_EMISSIVE_STRENGTH;
        [
            dec(packed & 255) * scale,
            dec((packed >> 8) & 255) * scale,
            dec((packed >> 16) & 255) * scale,
        ]
    }

    #[test]
    fn an_emissive_brighter_than_white_survives_the_round_trip() {

        let back = unpack(MeshVertex::pack_emissive([25.0, 25.0, 25.0]));
        for c in back {
            assert!(
                (c - 25.0).abs() < 25.0 * 0.02,
                "a strength of 25 came back as {c}, not within 2% of 25",
            );
        }
    }

    #[test]
    fn an_ordinary_emissive_is_unchanged() {

        for v in [[1.0f32, 1.0, 1.0], [0.5, 0.25, 0.0], [0.0, 0.0, 0.8]] {
            let back = unpack(MeshVertex::pack_emissive(v));
            for (got, want) in back.iter().zip(v.iter()) {
                assert!((got - want).abs() < 0.02, "{v:?} came back as {back:?}");
            }
        }
    }

    #[test]
    fn the_colour_of_a_bright_emissive_is_preserved_not_just_its_level() {

        let back = unpack(MeshVertex::pack_emissive([20.0, 10.0, 5.0]));
        assert!((back[0] / back[1] - 2.0).abs() < 0.1, "{back:?}");
        assert!((back[1] / back[2] - 2.0).abs() < 0.1, "{back:?}");
    }

    #[test]
    fn a_material_that_does_not_glow_stays_black() {
        assert_eq!(unpack(MeshVertex::NO_EMISSIVE), [0.0, 0.0, 0.0]);
        assert_eq!(unpack(MeshVertex::pack_emissive([0.0, 0.0, 0.0])), [0.0, 0.0, 0.0]);
    }
}
