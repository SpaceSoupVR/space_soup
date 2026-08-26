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
}

impl MeshVertex {
    pub const ATTRIBS: [wgpu::VertexAttribute; 4] =
        wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x2, 3 => Float32x2];

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

/// The same triangles, as vertices the layered-mesh pipeline can draw.
///
/// Alongside the textured form rather than instead of it. The index buffer is
/// shared -- it is the same mesh -- and keeping the ordinary vertices means a
/// build without the layered path, or a pass that has no material array to hand,
/// still has something to draw.
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
    /// Present when the file asked for layered shading AND carried the weights
    /// to do it with. Both, because a mesh that asks and does not supply would
    /// otherwise render as a single flat layer with no clue why.
    pub layered: Option<LayeredPrimitive>,
}
