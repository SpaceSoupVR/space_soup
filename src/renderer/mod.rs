pub mod brush_pipeline;
pub mod camera;
pub mod cuboid;
mod desktop_renderer;
pub mod icon;
pub mod lights;
pub mod mesh;
pub mod mesh_pipeline;
pub mod mirror;
pub mod panel;
pub mod particle;
pub mod pipeline;
pub mod ssr;
pub mod terrain_pipeline;
pub mod uniforms;

#[cfg(target_os = "android")]
pub mod xr_renderer;

pub use camera::Camera;
pub use cuboid::{Cuboid, CuboidShape, CuboidStyle};
pub use desktop_renderer::Renderer;
pub use icon::{billboard_rotation, IconAssets, IconKind};
pub use lights::{Light, LightKind};
pub use mesh::GltfMesh;
pub use mirror::MirrorSurface;
pub use panel::WorldPanel;
pub use particle::{Beam, Particle, ParticlePipeline, ParticleVertex};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Color3(pub u8, pub u8, pub u8, pub u8);

impl Default for Color3 {
    fn default() -> Self {
        Color3(255, 255, 255, 255)
    }
}

impl Color3 {
    pub fn to_linear(&self) -> [f32; 4] {
        let c = |v: u8| {
            let f = v as f32 / 255.0;
            if f <= 0.04045 {
                f / 12.92
            } else {
                ((f + 0.055) / 1.055).powf(2.4)
            }
        };
        [c(self.0), c(self.1), c(self.2), self.3 as f32 / 255.0]
    }
}

pub struct MeshInstance<'a> {
    pub mesh: &'a GltfMesh,
    pub model: &'a mesh_pipeline::ModelUniform,
    pub lightmap_key: Option<&'a str>,
}

