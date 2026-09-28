pub mod brush_pipeline;
pub mod camera;
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
pub mod panel;
pub mod particle;
pub mod pipeline;
pub mod profiler;
pub mod shadow;
pub mod tonemap;
pub mod exposure;
pub mod probe_stream;
pub mod sky;
pub mod ssr;
pub mod terrain_pipeline;
pub mod water_pipeline;
pub mod uniforms;
pub mod probe_prefilter;
pub mod multiview;
pub mod pass_timers;
pub mod bench;
pub mod levers;
pub mod perf_ab;
pub mod scene_pass_plan;

#[cfg(target_os = "android")]
pub mod xr_renderer;

/// HOW MUCH OF THE RUNTIME'S RECOMMENDED EYE BUFFER IS ACTUALLY RENDERED.
///
/// Lives here rather than in `xr_renderer` because that module is Android-only
/// and this number is not just a resolution: anything calibrated against a
/// PIXEL'S WORLD FOOTPRINT has to scale with it. `lights::PROBE_BOX_MARGIN` is
/// the first such thing and will not be the last.
///
/// The frame is fill bound, measured across three values -- see
/// `memory/quest3-frame-is-fill-bound-not-pass-bound.md`. 0.7 renders 49% of
/// the pixels and took the frame from 40.6 ms to ~21 ms with no visible loss,
/// confirmed on the headset with corrective lenses (2026-09-19).
/// HOW MUCH OF THE RUNTIME'S RECOMMENDED EYE BUFFER IS ACTUALLY RENDERED.
///
/// Lives here rather than in `xr_renderer` because that module is Android-only
/// and this number is not just a resolution: anything calibrated against a
/// PIXEL'S WORLD FOOTPRINT has to scale with it.
///
/// The frame is fill bound, measured across three values -- see
/// `memory/quest3-frame-is-fill-bound-not-pass-bound.md`. 0.7 renders 49% of
/// the pixels and took the frame from 40.6 ms to ~21 ms with no visible loss,
/// confirmed on the headset with corrective lenses (2026-09-19).
///
/// A 1.0 diagnostic build was prepared to test whether the fine-detail shimmer
/// was plain minification aliasing. It never reached the headset, and it is no
/// longer the cheapest way to find out: the aliasing hypothesis is now
/// supported by Valve's VR guidance directly, and the actual fix (roughness
/// mips carrying normal-map variance) is implemented. Judging THAT at the
/// shipping resolution answers the question that matters.
///
/// Meta's own guidance is that raising this is a better use of GPU time than
/// MSAA above 4x -- worth remembering when the 2x-MSAA A/B happens.
pub const RENDER_SCALE: f32 = 0.7;
// THE 1.0 DIAGNOSTIC RAN, AND IT SPLIT THE TWO SYMPTOMS (headset, 2026-09-21).
//
//   * TERRAIN SHIMMER: mainly FIXED at 1.0, grass steady even up close. So the
//     fine-detail shimmer IS sampling density, and the work is buying density
//     back cheaply -- the MSAA-down/resolution-up trade Meta's own guidance
//     recommends, FFR, and the SSR composite restructure.
//   * CEILING/WALL SEAMS: UNCHANGED at 1.0. So render scale was never their
//     cause, and the first fix for them -- scaling PROBE_BOX_MARGIN by this
//     constant -- was built on a premise the headset has now disproven. The
//     per-face probe selection that replaced it removed some seams and stands
//     on its own evidence; the rest are still unexplained, and the 408 real
//     CSG T-junctions whose repair is OFF are the standing suspect, being the
//     one candidate whose artifact is resolution-independent.
//
// Two artifacts that arrived together turned out to be independent. Back to
// the shipping value; the shimmer returns with it, and that is now a known
// cost with a known plan rather than a mystery.
pub use camera::Camera;
pub use cuboid::{Cuboid, CuboidShape, CuboidStyle};
pub use desktop_renderer::Renderer;
pub use icon::{billboard_rotation, IconAssets, IconKind};
pub use lights::{Light, LightKind};
pub use mesh::{GltfMesh, MeshLightmapUv};
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
    /// How hard this object's emissive materials are driven, this frame.
    ///
    /// 0.0 for everything that is not a lit fixture, which is almost everything.
    /// A lamp passes its light's `emissive_drive`, so its bulb brightens, dims
    /// and goes out with the beam and nothing has to keep the two in step.
    pub emissive_drive: f32,
}

