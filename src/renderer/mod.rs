pub mod brush_pipeline;
pub mod camera;
pub mod character_cards;
pub mod cuboid;
mod desktop_renderer;
pub mod icon;
pub mod layer_settings;
pub mod layered_mesh_pipeline;
pub mod lights;
pub mod material_wgsl;
pub mod mesh;
pub mod mesh_pipeline;
pub mod mirror;
pub mod performance_level;
pub mod dynamic_resolution;
pub mod panel;
pub mod glare;
pub mod effects;
pub mod particle;
pub mod pipeline;
pub mod particle_pipeline;
pub mod profiler;
pub mod shadow;
pub mod space_warp;
pub mod tonemap;
pub mod exposure;
pub mod probe_stream;
pub mod room_light;
pub mod sky;
pub mod time_of_day;
pub mod ssr;
pub mod terrain_pipeline;
pub mod water_pipeline;
pub mod underwater;
pub mod weather;
pub mod ground_cuts;
pub mod ground_twins;
pub mod uniforms;
pub mod probe_prefilter;
pub mod multiview;
pub mod pass_timers;
pub mod pipeline_cache_file;
pub mod bench;
pub mod levers;
pub mod perf_ab;
pub mod scene_pass_plan;
pub mod shader_checks;
pub mod portal_cull;
pub mod doors;
pub mod shader_precision;
#[cfg(test)]
pub(crate) mod shader_inlining;
pub mod skin_compute;
pub mod water_waves;
pub mod probe_blur;
pub mod probe_fixup;
pub mod pool_cards;
pub mod foveation;
pub mod visibility_mask;
pub mod ground_map;
pub mod proxy_cards;
pub mod proxy_field;

#[cfg(target_os = "android")]
pub mod xr_renderer;

pub const RENDER_SCALE: f32 = 0.7;

pub use camera::Camera;
pub use cuboid::{Cuboid, CuboidShape, CuboidStyle};
pub use desktop_renderer::{DesktopGround, Renderer};
pub use icon::{billboard_rotation, IconAssets, IconKind};
pub use lights::{Light, LightKind};
pub use mesh::{load_clips_for, GltfClip, GltfMesh, MeshLightmapUv};
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

    pub emissive_drive: f32,

    pub own_light: Option<Light>,

    pub tile_caster: bool,
}
