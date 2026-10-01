//! Ground shading: splat materials, biplanar on slopes, macro variation.
//!
//! Terrain was appended to the cuboid solid pass, and the commit that did it
//! argued against giving it a pipeline of its own -- "a second place for
//! shading to drift". That was the right call while terrain was flat vertex
//! colour and is the wrong one now: terrain needs a material array and a
//! sampler that no cuboid will ever use, and forcing that into the solid
//! pipeline means every cuboid in the scene carries a binding it does not want.
//!
//! The drift concern is answered directly rather than ignored: this shares
//! `wgsl_lights_block` with the solid pipeline, so lighting, shadowing and
//! attenuation are literally the same code. What differs is only how albedo is
//! obtained, which is the thing that genuinely differs.
//!
//! Three techniques, chosen for a tile-based mobile GPU rendering stereo:
//!
//!  - SPLAT: up to four tiling materials in a texture array, blended by slope
//!    and height. This is what makes ground read as ground rather than as a
//!    coloured mesh.
//!
//!  - BIPLANAR rather than triplanar. A heightfield has no natural UVs, and
//!    planar UVs stretch badly on exactly the cliffs you sculpt for cover.
//!    Triplanar fixes that with three samples per material; biplanar drops the
//!    axis contributing least and uses two, for a difference nobody can see.
//!    Flat ground skips it entirely and takes a single planar sample, and most
//!    of a map is flat, so most pixels pay the cheap path.
//!
//!  - MACRO VARIATION: one low-frequency sample multiplied over the result.
//!    A single tiling texture over 500 metres repeats visibly from any ridge.
//!    Stochastic/hex tiling solves that properly at three samples plus a
//!    histogram transform, which is not affordable here -- and stochastic
//!    blending that is not temporally stable shimmers under head motion, which
//!    is far more noticeable in stereo than on a monitor. One extra sample
//!    removes the repetition you actually notice.

use wgpu::*;

use super::cuboid::SolidVertex;
use super::material_wgsl::{wgsl_biplanar_block, wgsl_whiteout_block};

/// Per-scene terrain material settings, matching the WGSL uniform.
///
/// `repeat` values are in metres per tile: a 4m stone tile and a 12m grass tile
/// read very differently, and having one global scale is what makes every
/// terrain in an engine look like the same terrain.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TerrainMaterialUniform {
    /// Metres per tile for each of the four splat layers.
    pub repeat: [f32; 4],
    /// Slope in degrees at which layer 1 (rock) fully replaces layer 0 (ground).
    pub slope_start_deg: f32,
    pub slope_end_deg: f32,
    /// World Y band over which layer 2 (high ground) fades in.
    pub height_start: f32,
    pub height_end: f32,
    /// Metres per tile of the macro variation texture, and how strongly it
    /// modulates. Zero strength disables it without a shader variant.
    pub macro_repeat: f32,
    pub macro_strength: f32,
    /// Degrees past which biplanar sampling kicks in. Below this the shader
    /// takes one planar sample.
    pub biplanar_start_deg: f32,
    /// Non-zero when the bound splat map carries authored weights.
    ///
    /// A flag rather than a sentinel resolution or an all-zero texel: the
    /// shader must know whether to trust the map BEFORE it reads it, and a
    /// "weights that happen to look unauthored" rule would make a legitimately
    /// black-painted texel indistinguishable from no map at all.
    pub use_splat: f32,

    /// How strongly the normal maps perturb the surface. 0 disables them.
    ///
    /// No companion flag, unlike `use_splat`, because a flat normal map is a
    /// genuine no-op: the fallback texel (128, 128, 255) unpacks to (0, 0, 1),
    /// which is "unchanged" in tangent space. An unauthored normal set costs
    /// its samples and changes nothing, so the shader never has to be told
    /// whether to trust it.
    pub normal_strength: f32,
    pub _pad: [f32; 3],
}

impl Default for TerrainMaterialUniform {
    fn default() -> Self {
        Self {
            repeat: [8.0, 4.0, 10.0, 6.0],
            slope_start_deg: 22.0,
            slope_end_deg: 40.0,
            height_start: 1e9, // off by default: most scenes have no height band
            height_end: 1e9,
            macro_repeat: 140.0,
            macro_strength: 0.35,
            biplanar_start_deg: 18.0,
            use_splat: 0.0,
            normal_strength: 1.0,
            _pad: [0.0; 3],
        }
    }
}

pub struct TerrainPipeline {
    pub pipeline: RenderPipeline,
    pub material_layout: BindGroupLayout,
}

impl TerrainPipeline {
    pub fn new(device: &Device, format: TextureFormat, uniform_layout: &BindGroupLayout) -> Self {
        Self::new_multisampled(device, format, uniform_layout, 1)
    }

    /// See `pipeline::SolidPipeline::new_multisampled` -- a pipeline's sample
    /// count must match the pass it runs in, so a 4x eye pass needs its own.
    pub fn new_multisampled(

        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        samples: u32,
    ) -> Self {
        Self::new_with_view(device, format, uniform_layout, samples, crate::renderer::multiview::ViewMode::Mono)
    }

    /// The same, drawing BOTH EYES in one pass. See `multiview::ViewMode`.
    pub fn new_multisampled_stereo(

        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        samples: u32,
    ) -> Self {
        Self::new_with_view(device, format, uniform_layout, samples, crate::renderer::multiview::ViewMode::Stereo)
    }

    fn new_with_view(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        samples: u32,
        view: crate::renderer::multiview::ViewMode,
    ) -> Self {
        Self::build(device, format, uniform_layout, samples, view, TerrainRole::Scene, None)
    }

    /// THE GROUND IN THE SCENE PASS, reading its probe reflection from the
    /// half-resolution probe pass (group 3, `probe_pass::bind_group_layout`)
    /// instead of tracing it per pixel -- as the brushes do. Single eye.
    pub fn new_probe_reader(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        samples: u32,
        probe_layout: &BindGroupLayout,
    ) -> Self {
        Self::build(
            device,
            format,
            uniform_layout,
            samples,
            crate::renderer::multiview::ViewMode::Mono,
            TerrainRole::Read,
            Some(probe_layout),
        )
    }

    /// THE GROUND IN THE HALF-RESOLUTION PROBE PASS: its probe reflection and
    /// nothing else, with the secondary lookups left to `fixups`, exactly as the
    /// brushes' pass does. See `brush_pipeline::probe_pass` and `probe_fixup`.
    pub fn new_probe_pass(
        device: &Device,
        uniform_layout: &BindGroupLayout,
        fixups: &crate::renderer::probe_fixup::ProbeFixups,
    ) -> Self {
        Self::build(
            device,
            crate::renderer::brush_pipeline::probe_pass::FORMAT,
            uniform_layout,
            1,
            crate::renderer::multiview::ViewMode::Mono,
            TerrainRole::ProbePass,
            Some(fixups.pass_layout()),
        )
    }

    fn build(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        samples: u32,
        view: crate::renderer::multiview::ViewMode,
        role: TerrainRole,
        group3: Option<&BindGroupLayout>,
    ) -> Self {
        let mut source = terrain_shader_for(role);
        if role == TerrainRole::ProbePass && device.features().contains(wgpu::Features::SHADER_EARLY_DEPTH_TEST) {
            // It writes the fix-up list: see `BrushPipeline::new_probe_pass_deferred`.
            source = source.replacen("@fragment fn fs_main(", "@fragment @early_depth_test(force) fn fs_main(", 1);
        }
        // Audited: see `shader_checks`.
        let shader = crate::renderer::shader_checks::audited_shader_module(device, ShaderModuleDescriptor {
            label: Some("terrain_shader"),
            source: ShaderSource::Wgsl(crate::renderer::shader_precision::for_device(device, view.shader(source)).into()),
        });
        let material_layout = material_bind_group_layout(device);
        let mut layouts = vec![Some(uniform_layout), Some(&material_layout)];
        if let Some(l) = group3 {
            layouts.push(None);
            layouts.push(Some(l));
        }
        let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("terrain_layout"),
            bind_group_layouts: &layouts,
            immediate_size: 0,
        });

        // In the probe pass, both of its targets; the reach is left as the
        // brushes wrote it. See `probe_pass::targets`.
        let probe_targets = crate::renderer::brush_pipeline::probe_pass::targets(false);
        let scene_target = [Some(ColorTargetState { format, blend: Some(BlendState::ALPHA_BLENDING), write_mask: ColorWrites::ALL })];
        let targets: &[Option<ColorTargetState>] =
            if role == TerrainRole::ProbePass { &probe_targets } else { &scene_target };
        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some(match role {
                TerrainRole::Scene => "terrain_pipeline",
                TerrainRole::Read => "terrain_pipeline_read",
                TerrainRole::ProbePass => "terrain_probe_pass",
            }),
            layout: Some(&layout),
            vertex: VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: PipelineCompilationOptions::default(),
                // Same vertex format as the solid pass, so the geometry path is
                // untouched: terrain still arrives as SolidVertex.
                buffers: &[Some(SolidVertex::layout())],
            },
            fragment: Some(FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: PipelineCompilationOptions::default(),
                // The probe pass stores a reflection, not a colour to blend.
                targets,
            }),
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                cull_mode: Some(Face::Back),
                front_face: FrontFace::Ccw,
                polygon_mode: PolygonMode::Fill,
                ..Default::default()
            },
            depth_stencil: Some(DepthStencilState {
                format: TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(CompareFunction::Less),
                stencil: StencilState::default(),
                bias: DepthBiasState::default(),
            }),
            multisample: MultisampleState { count: samples, ..Default::default() },
            multiview_mask: view.mask(),
            cache: None,
        });

        Self { pipeline, material_layout }
    }
}

/// The lights block as the ground takes it: with `options`, testing each
/// lamp's range before its baked mask -- out of range is what the building's
/// lamps are from nearly all the ground. See `CULL_RANGE_FIRST`.
fn terrain_lights_block(options: crate::renderer::lights::LightsBlockOptions) -> String {
    crate::renderer::lights::wgsl_lights_block_with(
        0,
        1,
        crate::renderer::lights::LightsBlockOptions { cull_range_first: true, ..options },
    )
}

/// WHAT A TERRAIN SHADER IS FOR. See `TerrainPipeline::new_probe_reader` and
/// `new_probe_pass`.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum TerrainRole {
    /// Shades the ground, tracing its probe reflection per pixel. What
    /// shipped, and still what a stereo pass and full-resolution reflections
    /// draw with.
    Scene,
    /// Shades the ground, reading its probe reflection from the probe pass.
    Read,
    /// Writes the ground's probe reflection into the probe pass, deferring its
    /// secondary lookups to `probe_fixup`.
    ProbePass,
}

/// The terrain shader for `role`: `terrain_shader`, with its reflection read
/// from the probe pass or its fragment stage cut down to the reflection alone.
pub fn terrain_shader_for(role: TerrainRole) -> String {
    let src = terrain_shader();
    if role == TerrainRole::Scene {
        return src;
    }
    let plain_lights = terrain_lights_block(crate::renderer::lights::LightsBlockOptions::default());
    assert!(src.contains(&plain_lights), "the terrain shader no longer embeds the lights block as generated");
    let shade = "    let lit = shade_material_env(\n";
    assert!(src.contains(shade), "the terrain shader no longer shades through `shade_material_env`");
    match role {
        TerrainRole::Scene => unreachable!(),
        TerrainRole::Read => {
            let read_lights = terrain_lights_block(crate::renderer::lights::LightsBlockOptions {
                probe_from_pass: true,
                ..Default::default()
            });
            let src = src.replacen(&plain_lights, &read_lights, 1).replacen(
                shade,
                &format!(
                    "    // From the half-resolution probe pass, as the brushes read it. See\n    // `brush_pipeline::probe_pass`.\n    let probe_pass_tolerance = max(4.0 * fwidth(in.clip.z), 1e-6);\n    probe_env_given = probe_pass_upsample(in.clip.xy, in.clip.z, probe_pass_tolerance, 0.0);\n{shade}"
                ),
                1,
            );
            // The ground writes no face code (`probe_pass::targets(false)`), so
            // it reads with none: 0, which every texel matches.
            format!(
                "{src}{}{}",
                crate::renderer::brush_pipeline::probe_pass::READER_WGSL,
                crate::renderer::brush_pipeline::probe_pass::FACE_CODE_WGSL,
            )
        }
        TerrainRole::ProbePass => {
            let pass_lights = terrain_lights_block(crate::renderer::lights::LightsBlockOptions {
                defer_secondary: true,
                ..Default::default()
            });
            let src = src.replacen(&plain_lights, &pass_lights, 1);
            let start = src.find(shade).expect("checked above");
            let end = start + src[start..].find("\n}\n").expect("the fragment stage ends");
            // The reflection for the pass to store, from the same inputs the
            // shading below hands `shade_material_env`: no baked bounce, the
            // ground's own position to choose from, its shape for Fresnel.
            format!(
                "{}    probe_fragment = in.clip;\n    return probe_env_for_pass(\n        in.world_pos, shaded_n, rough, clamp(ao_map, 0.0, 1.0), sky_vis, vec3<f32>(0.0), in.world_pos, n,\n    );{}",
                &src[..start],
                &src[end..],
            )
        }
    }
}

pub fn material_bind_group_layout(device: &Device) -> BindGroupLayout {
    device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("terrain_material_layout"),
        entries: &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::D2Array,
                    multisampled: false,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Sampler(SamplerBindingType::Filtering),
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 3,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // Per-layer normal maps. Always bound, like the weights: an
            // optional binding would mean two layouts and therefore two
            // pipelines, and a flat placeholder costs one texel per layer and
            // changes nothing.
            BindGroupLayoutEntry {
                binding: 5,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::D2Array,
                    multisampled: false,
                },
                count: None,
            },
            // Baked sky visibility over the footprint. Always bound; an unbaked
            // terrain gets a 1x1 WHITE texel, which is "sees the whole sky" and
            // reproduces the old shading exactly. White is the neutral here
            // because this value is MULTIPLIED -- the opposite of the brush
            // lightmap's additive RGB, whose neutral is black.
            BindGroupLayoutEntry {
                binding: 6,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::D2Array,
                    multisampled: false,
                },
                count: None,
            },
            // Per-layer roughness and ambient occlusion, giving terrain the
            // same material set brush surfaces have. Always bound: a layer
            // with no map gets a solid neutral, so this is one pipeline rather
            // than two, for the same reason the normal array is.
            BindGroupLayoutEntry {
                binding: 7,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::D2Array,
                    multisampled: false,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 8,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::D2Array,
                    multisampled: false,
                },
                count: None,
            },
            // Authored blend weights over the terrain footprint. Always bound,
            // even when unauthored: an optional binding would mean two bind
            // group layouts and therefore two pipelines, and a 1x1 placeholder
            // costs one texel.
            BindGroupLayoutEntry {
                binding: 4,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
        ],
    })
}

fn terrain_shader() -> String {
    format!(
        r#"
// Group 0 -- the camera, the lights and both shadow maps -- is declared by
// `wgsl_lights_block` below, so there is one description of that layout rather
// than one per shader.

@group(1) @binding(0) var layer_tex: texture_2d_array<f32>;
@group(1) @binding(1) var layer_samp: sampler;
@group(1) @binding(2) var macro_tex: texture_2d<f32>;

struct Material {{
    repeat: vec4<f32>,
    slope_start_deg: f32,
    slope_end_deg: f32,
    height_start: f32,
    height_end: f32,
    macro_repeat: f32,
    macro_strength: f32,
    biplanar_start_deg: f32,
    use_splat: f32,
    normal_strength: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
}}
@group(1) @binding(3) var<uniform> mat: Material;
@group(1) @binding(4) var splat_tex: texture_2d<f32>;
@group(1) @binding(5) var normal_tex: texture_2d_array<f32>;
@group(1) @binding(7) var rough_tex: texture_2d_array<f32>;
@group(1) @binding(8) var ao_tex: texture_2d_array<f32>;
// Baked sky visibility over the terrain footprint, in R. 1 = open sky, 0 =
// sealed. Sampled by the same normalised footprint uv the splat map uses, so it
// needs no new vertex attribute.
//
// WHY TERRAIN NEEDS ITS OWN
//
// Brushes carry sky visibility in their lightmap's alpha, but terrain has no
// lightmap and no chart layout to put one in -- it is a heightfield sampled by
// world position. Without this the ground took the full open-sky term
// everywhere, so the floor INSIDE a sealed room lit exactly as brightly as the
// field outside it, and no amount of shadow-map work could change that: the
// term being wrong was the ambient, not the direct light.
// Layer 0 the ground's own map; after it the stationary lamps' masks, two
// lamps a layer. See `TerrainMaterial::new`.
@group(1) @binding(6) var sky_occ_tex: texture_2d_array<f32>;

{lights_block}
{biplanar_block}
{whiteout_block}

struct VIn  {{ @location(0) pos: vec3<f32>, @location(1) norm: vec3<f32>, @location(2) col: vec4<f32>, @location(3) uv2: vec2<f32> }}
struct VOut {{
    @builtin(position) clip: vec4<f32>,
    @location(0) col: vec4<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) world_pos: vec3<f32>,
    // The same point before the player-frame transform. Texture projection and
    // the height blend are properties of the GROUND, not of where the player is
    // standing, so they read this and never `world_pos`.
    @location(4) tex_pos: vec3<f32>,
    // Normalised position over the terrain footprint, carried in the slot the
    // cuboid path uses for lightmap coordinates. Terrain is lit dynamically and
    // has no lightmap, so that slot was sitting at (0,0) doing nothing.
    @location(3) uv: vec2<f32>,
}}

@vertex fn vs_main(v: VIn) -> VOut {{
    var out: VOut;
    out.clip      = cam_view_proj() * vec4<f32>(v.pos, 1.0);
    out.col       = v.col;
    out.normal    = v.norm;
    out.world_pos = v.pos;
    out.tex_pos   = to_world_space(v.pos);
    out.uv        = v.uv2;
    return out;
}}

// One splat layer, sampled planar from above. Ground is mostly flat, so this is
// the path most pixels take.
fn sample_planar(layer: i32, world: vec3<f32>, repeat: f32) -> vec3<f32> {{
    let uv = world.xz / max(repeat, 0.001);
    return textureSample(layer_tex, layer_samp, uv, layer).rgb;
}}

// Biplanar: the two strongest axes, weighted by the normal. Two samples rather
// than triplanar's three -- the third axis contributes least by construction,
// and dropping it is invisible while being a third cheaper.
//
// The axis choice itself lives in `material_wgsl`, because the cave shader needs
// exactly the same projection and a second copy of it would be a second thing to
// get wrong. What stays here is the sampling, which differs.
fn sample_biplanar(layer: i32, world: vec3<f32>, n: vec3<f32>, repeat: f32) -> vec3<f32> {{
    let b = biplanar_axes(world, n, repeat);
    let c_major = textureSample(layer_tex, layer_samp, b.uv_major, layer).rgb;
    let c_minor = textureSample(layer_tex, layer_samp, b.uv_minor, layer).rgb;
    return mix(c_minor, c_major, b.w);
}}

// Surface normal for one layer, sampled top-down and folded into the geometric
// normal by the "whiteout" blend.
//
// PLANAR ONLY, deliberately, while colour goes biplanar on steep ground. Each
// extra projection is another texture fetch per layer per pixel, and the four
// layers are already sampled unconditionally -- going biplanar here would take
// terrain from twelve fetches to sixteen on a machine that is not fast. The
// cost is that a near-vertical face gets a stretched normal; that is far less
// objectionable than stretched COLOUR, which is why colour keeps its second
// projection and this does not.
//
// Whiteout rather than simply replacing the normal: the map describes bumps
// relative to the surface, so it has to perturb the geometry rather than
// overwrite it. Replacing would make every slope light as though it were flat
// ground, which is exactly the artefact normal maps exist to avoid.
fn layer_normal(layer: i32, world: vec3<f32>, n: vec3<f32>, repeat: f32) -> vec3<f32> {{
    let uv = world.xz / max(repeat, 0.001);
    let packed = textureSample(normal_tex, layer_samp, uv, layer).rgb;

    // OpenGL convention: green is +Y in tangent space. ambientCG ships both
    // NormalGL and NormalDX; the installer takes GL, and a DX map loaded here
    // would light every bump from the opposite side.
    var tn = packed * 2.0 - 1.0;
    tn = vec3<f32>(tn.xy * mat.normal_strength, tn.z);

    // Axis 1 is the y projection, which is the only one this samples. For flat
    // ground (n = 0,1,0) and a flat texel (tn = 0,0,1) it returns the geometric
    // normal exactly, which is what makes an unauthored normal set a no-op.
    return normalize(whiteout(1u, tn, n));
}}

fn layer_colour(layer: i32, world: vec3<f32>, n: vec3<f32>, slope_deg: f32) -> vec3<f32> {{
    let repeat = mat.repeat[layer];
    if (slope_deg < mat.biplanar_start_deg) {{
        return sample_planar(layer, world, repeat);
    }}
    return sample_biplanar(layer, world, n, repeat);
}}

// Everything a layer needs to sample itself, computed ONCE per fragment.
//
// The point is the derivatives. `textureSample` picks its own mip level from
// implicit derivatives, which is why it may only be called in uniform control
// flow -- and that requirement is what forced all four layers to be sampled
// whether or not they contributed. `textureSampleGrad` takes the gradients as
// arguments and carries no such restriction, so a layer with no weight can be
// skipped entirely.
//
// The gradients are taken here, before any branch, and every projection scales
// as 1/repeat -- so a per-layer repeat is a divide, never another derivative.
struct SampleFrame {{
    planar_uv: vec2<f32>,
    planar_ddx: vec2<f32>,
    planar_ddy: vec2<f32>,
    major_uv: vec2<f32>,
    major_ddx: vec2<f32>,
    major_ddy: vec2<f32>,
    minor_uv: vec2<f32>,
    minor_ddx: vec2<f32>,
    minor_ddy: vec2<f32>,
    blend: f32,
    biplanar: f32,
}}

fn sample_frame(world: vec3<f32>, n: vec3<f32>, slope_deg: f32) -> SampleFrame {{
    var f: SampleFrame;
    f.planar_uv  = world.xz;
    f.planar_ddx = dpdx(world.xz);
    f.planar_ddy = dpdy(world.xz);

    // At unit repeat, so each layer divides rather than recomputing.
    let b = biplanar_axes(world, n, 1.0);
    f.major_uv  = b.uv_major;
    f.major_ddx = dpdx(b.uv_major);
    f.major_ddy = dpdy(b.uv_major);
    f.minor_uv  = b.uv_minor;
    f.minor_ddx = dpdx(b.uv_minor);
    f.minor_ddy = dpdy(b.uv_minor);
    f.blend     = b.w;
    f.biplanar  = select(0.0, 1.0, slope_deg >= mat.biplanar_start_deg);
    return f;
}}

fn layer_colour_at(layer: i32, f: SampleFrame) -> vec3<f32> {{
    let r = max(mat.repeat[layer], 0.001);
    if (f.biplanar < 0.5) {{
        return textureSampleGrad(
            layer_tex, layer_samp, f.planar_uv / r, layer,
            f.planar_ddx / r, f.planar_ddy / r,
        ).rgb;
    }}
    let c_major = textureSampleGrad(
        layer_tex, layer_samp, f.major_uv / r, layer, f.major_ddx / r, f.major_ddy / r,
    ).rgb;
    let c_minor = textureSampleGrad(
        layer_tex, layer_samp, f.minor_uv / r, layer, f.minor_ddx / r, f.minor_ddy / r,
    ).rgb;
    return mix(c_minor, c_major, f.blend);
}}

// Roughness and occlusion for one layer, sampled with the same frame and
// gradients the normal uses so all four maps agree about which mip they are on.
//
// Planar only: both are low-frequency compared with colour, and a biplanar pair
// for each would double the fetches on a fill-bound frame to move a value that
// barely changes across the blend.
fn layer_rough_at(layer: i32, f: SampleFrame) -> f32 {{
    let r = max(mat.repeat[layer], 0.001);
    return textureSampleGrad(
        rough_tex, layer_samp, f.planar_uv / r, layer, f.planar_ddx / r, f.planar_ddy / r,
    ).r;
}}

fn layer_ao_at(layer: i32, f: SampleFrame) -> f32 {{
    let r = max(mat.repeat[layer], 0.001);
    return textureSampleGrad(
        ao_tex, layer_samp, f.planar_uv / r, layer, f.planar_ddx / r, f.planar_ddy / r,
    ).r;
}}

fn layer_normal_at(layer: i32, n: vec3<f32>, f: SampleFrame) -> vec3<f32> {{
    let r = max(mat.repeat[layer], 0.001);
    let packed = textureSampleGrad(
        normal_tex, layer_samp, f.planar_uv / r, layer, f.planar_ddx / r, f.planar_ddy / r,
    ).rgb;

    // Identical to `layer_normal` above, which stays for the planar path used
    // elsewhere: OpenGL green-up convention, strength applied to the tangent
    // axes only, folded into the geometry by the whiteout blend rather than
    // replacing it.
    //
    // RETURNED UNNORMALISED, AND THAT IS THE WHOLE CHANGE.
    //
    // Mipping a normal map averages its normals, and where they disagreed the
    // average is SHORT. That shortness is a measurement of how much detail the
    // mip threw away -- and `normalize()` discards exactly that measurement
    // while restoring full strength, so a distant bumpy surface keeps shading
    // as though every bump were still resolvable. In a headset, where head
    // tracking moves the sampling point every frame, that is shimmer.
    //
    // The caller renormalises for the shading normal and reads the length as
    // variance. Nothing is lost by handing both back.
    var tn = packed * 2.0 - 1.0;
    tn = vec3<f32>(tn.xy * mat.normal_strength, tn.z);
    return whiteout(1u, tn, n);
}}

// Below this a layer changes the result by less than one 8-bit step, so
// sampling it buys nothing but bandwidth. Not renormalised afterwards:
// rescaling the surviving weights would move the shading further than the
// omission does.
const WEIGHT_EPS: f32 = 0.004;
// Metres over which the terrain's normal maps fade out past the detail
// distance, so no line marks where they stop.
const TERRAIN_DETAIL_FADE: f32 = 8.0;

// Blend weights for the four layers, authored or derived.
//
// One vec4 either way, so the fragment stage below has a single code path. The
// procedural branch reproduces the old mix chain exactly -- a weighted sum and
// a chain of mixes are the same arithmetic -- so turning authoring on and off
// is a change of WEIGHTS and never a change of shading model.
fn layer_weights(uv: vec2<f32>, world_y: f32, slope_deg: f32) -> vec4<f32> {{
    // Authored. Normalised by its own sum rather than trusted: the editor keeps
    // the four bytes summing to 255, but a hand-made or half-written file has
    // no such guarantee and unnormalised weights would blow out or black out
    // the ground rather than looking slightly wrong.
    let authored_raw = textureSample(splat_tex, layer_samp, uv);
    let total = authored_raw.r + authored_raw.g + authored_raw.b + authored_raw.a;
    let authored = authored_raw / max(total, 0.001);

    // Derived from slope, with the optional height band on top.
    let rock_w = smoothstep(mat.slope_start_deg, mat.slope_end_deg, slope_deg);
    var derived = vec4<f32>(1.0 - rock_w, rock_w, 0.0, 0.0);
    if (mat.height_end > mat.height_start) {{
        let high_w = smoothstep(mat.height_start, mat.height_end, world_y);
        derived = vec4<f32>(derived.x * (1.0 - high_w), derived.y * (1.0 - high_w), high_w, 0.0);
    }}

    // select, not an if: both sides are already computed and branching here
    // would put textureSample under non-uniform control flow.
    return select(derived, authored, mat.use_splat > 0.5);
}}

@fragment fn fs_main(in: VOut) -> @location(0) vec4<f32> {{
    // See `pixel_footprint` in the lights block: taken here, in uniform
    // control flow, so the light loop can keep a spot's edge a pixel wide.
    //
    // The GEOMETRIC MEAN of the two screen axes, not their sum: seen at a
    // grazing angle one axis stretches to metres while the other stays a
    // pixel, and the sum let that stretch widen a spot's pool across the floor.
    pixel_footprint = sqrt(length(dpdx(in.world_pos)) * length(dpdy(in.world_pos)));
    let n = normalize(in.normal);

    // Slope straight from the normal: no derivative, no extra sampling.
    let slope_deg = degrees(acos(clamp(n.y, -1.0, 1.0)));

    let w = layer_weights(in.uv, in.tex_pos.y, slope_deg);

    // Only the layers that contribute. Terrain is fill-bound, not geometry
    // bound -- quartering the triangle count moved the frame time by nothing,
    // while removing the ground entirely gave back 13 ms of a 13.9 ms budget --
    // and most fragments are one or two layers, not four.
    let f = sample_frame(in.tex_pos, n, slope_deg);
    var albedo = vec3<f32>(0.0);
    if (w.x > WEIGHT_EPS) {{ albedo = albedo + layer_colour_at(0, f) * w.x; }}
    if (w.y > WEIGHT_EPS) {{ albedo = albedo + layer_colour_at(1, f) * w.y; }}
    if (w.z > WEIGHT_EPS) {{ albedo = albedo + layer_colour_at(2, f) * w.z; }}
    if (w.w > WEIGHT_EPS) {{ albedo = albedo + layer_colour_at(3, f) * w.w; }}

    // Blend the layers' normals by the same weights, then renormalise. Summing
    // unit vectors shortens the result wherever they disagree, and a shortened
    // normal darkens the surface -- so the renormalise is load-bearing, not
    // tidiness.
    //
    // FADED OUT WITH DISTANCE when `post_params.z` says where: past it a
    // layer's normal map only shakes a pixel's normal around an average the
    // vertex normal already is, and what it would have added -- a wider
    // highlight -- is baked into the roughness mips read below. So the far
    // field skips the samples and the blend, and loses nothing it could show.
    let detail_far = camera.post_params.z;
    let detail = select(1.0, 1.0 - smoothstep(detail_far, detail_far + TERRAIN_DETAIL_FADE, distance(in.world_pos, cam_pos())), detail_far > 0.0);
    var shaded_n = vec3<f32>(0.0);
    if (detail > 0.0) {{
        if (w.x > WEIGHT_EPS) {{ shaded_n = shaded_n + layer_normal_at(0, n, f) * w.x; }}
        if (w.y > WEIGHT_EPS) {{ shaded_n = shaded_n + layer_normal_at(1, n, f) * w.y; }}
        if (w.z > WEIGHT_EPS) {{ shaded_n = shaded_n + layer_normal_at(2, n, f) * w.z; }}
        if (w.w > WEIGHT_EPS) {{ shaded_n = shaded_n + layer_normal_at(3, n, f) * w.w; }}
        shaded_n = mix(n, shaded_n, detail);
    }}

    // HOW MUCH DETAIL WAS AVERAGED AWAY, read before renormalising.
    //
    // Two things shortened this vector and both mean the same thing: the mip
    // chain averaging disagreeing normals inside one layer, and the weighted
    // blend of layers that disagree with each other. Either way a short result
    // says "the surface under this pixel is rougher than any single normal
    // here admits", and a wider highlight is the honest response.
    //
    // sigma2 = (1 - len)/len is the Toksvig variance; the factor scales the
    // specular exponent. Clamped at both ends: len can exceed 1 slightly where
    // normal_strength pushes past unit length, and a degenerate near-zero
    // blend must not divide by ~0.
    // ROUGHNESS AND OCCLUSION FROM THE MAPS, blended by the same weights the
    // colour and normal use so a splat boundary moves all four together.
    //
    // The normal's lost variation is ALREADY IN these mips -- baked in by
    // `roughness_chain_with_normal_variance` at load, per level, exactly as
    // brush materials get it. Adding a runtime term for it here as well would
    // count the same variance twice and over-roughen distant ground.
    var rough_map = 0.0;
    var ao_map = 0.0;
    if (w.x > WEIGHT_EPS) {{ rough_map = rough_map + layer_rough_at(0, f) * w.x; ao_map = ao_map + layer_ao_at(0, f) * w.x; }}
    if (w.y > WEIGHT_EPS) {{ rough_map = rough_map + layer_rough_at(1, f) * w.y; ao_map = ao_map + layer_ao_at(1, f) * w.y; }}
    if (w.z > WEIGHT_EPS) {{ rough_map = rough_map + layer_rough_at(2, f) * w.z; ao_map = ao_map + layer_ao_at(2, f) * w.z; }}
    if (w.w > WEIGHT_EPS) {{ rough_map = rough_map + layer_rough_at(3, f) * w.w; ao_map = ao_map + layer_ao_at(3, f) * w.w; }}

    shaded_n = normalize(select(n, shaded_n, length(shaded_n) > 0.0001));

    // What the MIPS could not see: the geometric normal's own variation across
    // this pixel, and the grazing-angle case where anisotropic filtering
    // fetches a sharper roughness mip than the isotropic level. Taken here, at
    // the top level of the entry point, because a derivative is only legal in
    // uniform control flow.
    let rough = specular_aa_roughness(clamp(rough_map, 0.0, 1.0), dpdx(shaded_n), dpdy(shaded_n));

    // Macro variation: one low-frequency sample, centred on 1 so it darkens and
    // lightens rather than only darkening.
    let m = textureSample(macro_tex, layer_samp, in.tex_pos.xz / max(mat.macro_repeat, 0.001)).r;
    albedo = albedo * (1.0 + (m - 0.5) * 2.0 * mat.macro_strength);

    // Vertex colour survives as a tint, so the editor can still mark up ground
    // per-vertex without a second pipeline.
    // The ambient term is scaled by baked sky visibility BEFORE the lights are
    // added, which is what `shade_with_sky` does and `shade` cannot.
    let ground_map = textureSample(sky_occ_tex, layer_samp, in.uv, 0);
    let sky_vis = ground_map.r;
    // THE SKY SUN'S SHADOW, baked beside the sky visibility when alpha is 0:
    // a signed distance to the edge in texels (green) and the sun's penumbra
    // (blue), rebuilt exactly as the brushes rebuild theirs. The live static
    // map drew the walls' shadows on the grass with blocky edges (headset,
    // 2026-09-25). An older map is greyscale with alpha 1 and leaves the
    // ground on that live map, as before.
    let ground_sun_d = (ground_map.g - 0.5) * (2.0 * {sun_range:?});
    let ground_sun_w = max(max(ground_map.b * {sun_range:?}, 0.5 * fwidth(ground_sun_d)), 0.02);
    receiver_sun_mask = select(-1.0, smoothstep(-ground_sun_w, ground_sun_w, ground_sun_d), ground_map.a < 0.5);
    // THE STATIONARY LAMPS' SHADOWS ON THE GROUND, in the layers after the
    // ground map, in the brushes' format and channels. Without them every
    // lamp lit the grass straight through the walls -- the hallway's sconces
    // lit the ground outside the hallway (headset, 2026-09-24). A map with no
    // mask layers reads fully lit, as the ground always was.
    let st_layers = textureNumLayers(sky_occ_tex) - 1u;
    var st_0 = vec4<f32>(1.0);
    var st_1 = vec4<f32>(1.0);
    var st_2 = vec4<f32>(1.0);
    var st_3 = vec4<f32>(1.0);
    if (st_layers > 0u) {{
        st_0 = textureSample(sky_occ_tex, layer_samp, in.uv, 1);
    }}
    if (st_layers > 1u) {{
        st_1 = textureSample(sky_occ_tex, layer_samp, in.uv, 2);
    }}
    if (st_layers > 2u) {{
        st_2 = textureSample(sky_occ_tex, layer_samp, in.uv, 3);
    }}
    if (st_layers > 3u) {{
        st_3 = textureSample(sky_occ_tex, layer_samp, in.uv, 4);
    }}
    set_stationary_masks(st_0, st_1, st_2, st_3, {stationary_range:?});
    // THE SAME SHADING PATH THE BRUSHES USE, and the albedo goes IN rather
    // than being multiplied over the result.
    //
    // Terrain used to call `shade_with_sky` and multiply its albedo over the
    // answer. That is the exact bug this renderer already fixed for brushes:
    // a dielectric's highlight is not tinted by its diffuse colour, and
    // multiplying a 0.3-albedo ground over the lit result made every highlight
    // on it about three times too dim. Ground was a second-class material with
    // no Fresnel, no energy conservation, no probe reflection and no specular
    // occlusion, and it looked it next to marble.
    //
    // WHAT IS STILL MISSING, deliberately, until terrain ships the maps
    // brushes do: `ao` is 1.0 and the baked bounce is zero, because terrain has
    // neither an AO map nor a directional bounce bake -- it carries only a
    // scalar sky-occlusion, which is passed as `sky_vis` exactly as before.
    // Roughness is a constant widened by the normal's own variance rather than
    // a map. Each of those is a texture array away from parity.
    let lit = shade_material_env(
        in.world_pos,
        shaded_n,
        rough,
        clamp(ao_map, 0.0, 1.0),
        sky_vis,
        vec3<f32>(0.0),
        vec4<f32>(0.5, 0.5, 0.5, 0.0),
        albedo,
        // No face to stand on: ground is one continuous surface, so the
        // fragment's own position is the honest point to choose a probe from.
        in.world_pos,
        // The INTERPOLATED surface normal, before the layer normal maps
        // perturbed it -- the ground's own shape, which is what Fresnel wants.
        n,
    );
    return vec4<f32>(tonemap(in.col.rgb * lit), 1.0);
}}
"#,
        lights_block = terrain_lights_block(crate::renderer::lights::LightsBlockOptions::default()),
        sun_range = super::brush_pipeline::SUN_MASK_DISTANCE_TEXELS,
        stationary_range = super::brush_pipeline::STATIONARY_MASK_DISTANCE_TEXELS,
        biplanar_block = wgsl_biplanar_block(),
        whiteout_block = wgsl_whiteout_block(),
    )
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::renderer::cuboid::SolidVertex;
    use crate::renderer::lights::LightsUniform;
    use wgpu::util::DeviceExt;

    /// The pipeline is only validated when it is CREATED, and correctness only
    /// when something is drawn. A test that builds a pipeline proves the WGSL
    /// parses; it says nothing about whether the shading responds to slope.
    /// So this renders and reads the pixels back.
    pub fn headless_gpu() -> Option<(Device, Queue)> {
        let instance = Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&RequestAdapterOptions {
            apply_limit_buckets: false,
            power_preference: PowerPreference::default(),
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok()?;
        pollster::block_on(adapter.request_device(&DeviceDescriptor {
            required_features: Features::empty(),
            required_limits: crate::renderer::uniforms::scene_limits(Limits::default()),
            ..Default::default()
        }))
        .ok()
    }

    fn vertex(pos: [f32; 3], normal: [f32; 3]) -> SolidVertex {
        SolidVertex {
            position: pos,
            normal,
            color: [1.0, 1.0, 1.0, 1.0],
            uv2: [0.0, 0.0],
            reflectivity: 0.0,
        }
    }

    /// Renders one full-screen quad with the given normal and returns its
    /// centre pixel.
    /// Which layer colours the material carries. The test palette is four
    /// primaries so a readback names the winning layer unambiguously; the
    /// fallback palette is the shipping one, whose colours are deliberately
    /// close together and cannot.
    #[derive(Copy, Clone)]
    pub enum Palette {
        Test,
        Fallback,
    }

    pub fn render_quad_with_normal(normal: [f32; 3]) -> Option<[u8; 4]> {
        render_quad(normal, Palette::Test, None, None)
    }

    /// Same geometry, with authored weights bound.
    pub fn render_quad_with_splat(normal: [f32; 3], splat: &TerrainImage) -> Option<[u8; 4]> {
        render_quad(normal, Palette::Test, Some(splat), None)
    }

    /// Same geometry, but bound through the shipping `TerrainMaterial::fallback`
    /// rather than a bind group assembled here. Worth its own path because the
    /// hand-built test material proves the SHADER and proves nothing about the
    /// code the renderer will actually call -- layer padding, the D2Array view
    /// dimension and the non-sRGB macro format all live in `TerrainMaterial`
    /// and are exactly where a binding mistake would hide.
    pub fn render_quad_with_fallback_material(normal: [f32; 3]) -> Option<[u8; 4]> {
        render_quad(normal, Palette::Fallback, None, None)
    }

    fn render_quad(
        normal: [f32; 3],
        palette: Palette,
        splat: Option<&TerrainImage>,
        sky_occlusion: Option<&TerrainImage>,
    ) -> Option<[u8; 4]> {
        render_quad_full(normal, palette, splat, sky_occlusion, &[None, None, None, None], false)
    }

    /// Full harness: optional per-layer normal maps and an optional point light.
    ///
    /// The light matters. With an empty light list `shade` returns the ambient
    /// constant regardless of the surface normal, so a normal-map test on the
    /// default harness would pass whether the perturbation worked or not.
    pub fn render_quad_full(
        normal: [f32; 3],
        palette: Palette,
        splat: Option<&TerrainImage>,
        sky_occlusion: Option<&TerrainImage>,
        normals: &[Option<TerrainImage>],
        lit: bool,
    ) -> Option<[u8; 4]> {
        render_quad_lit_by(normal, palette, splat, sky_occlusion, normals, lit.then_some(None))
    }

    /// `render_quad_full`'s light, as a STATIONARY lamp reading mask channel
    /// `channel` when `Some(Some(channel))`.
    pub fn render_quad_lit_by(
        normal: [f32; 3],
        palette: Palette,
        splat: Option<&TerrainImage>,
        sky_occlusion: Option<&TerrainImage>,
        normals: &[Option<TerrainImage>],
        light: Option<Option<u8>>,
    ) -> Option<[u8; 4]> {
        render_quad_light_at(normal, palette, splat, sky_occlusion, normals, light, glam::Vec3::new(3.0, 2.0, 0.0))
    }

    /// `render_quad_lit_by` with the lamp at `light_pos`. The default sits
    /// level with the quad in z, so it cannot tell a tilt toward +z from one
    /// toward -z -- which is the axis green lives on in the top-down projection.
    pub fn render_quad_light_at(
        normal: [f32; 3],
        palette: Palette,
        splat: Option<&TerrainImage>,
        sky_occlusion: Option<&TerrainImage>,
        normals: &[Option<TerrainImage>],
        light: Option<Option<u8>>,
        light_pos: glam::Vec3,
    ) -> Option<[u8; 4]> {
        let lit = light.is_some();
        let (device, queue) = headless_gpu()?;
        let format = TextureFormat::Rgba8Unorm;

        let lights = LightsUniform::new(&device);
        let (_shadows, uniforms) =
            crate::renderer::uniforms::test_support::scene_uniforms(&device, &lights);

        // Both uniform buffers are created UNINITIALISED. Without writing them
        // view_proj is garbage -- the triangle lands somewhere arbitrary and the
        // readback is just clear colour -- and the light count is undefined.
        // Identity view_proj means the vertex positions ARE clip coordinates,
        // and an empty light list leaves the shader's AMBIENT term, which is all
        // this test wants: it is asserting the splat blend, not the light rig.
        uniforms.upload(
            &queue,
            glam::Mat4::IDENTITY,
            // The eye. Only specular reads it, and these tests assert the splat
            // blend -- so it goes straight out in front of the quad, where a
            // highlight lands symmetrically and cannot be mistaken for one of
            // the layers winning.
            glam::Vec3::new(0.0, 0.0, 5.0),
            &crate::renderer::uniforms::ShadowUpload::disabled(),
        );
        if lit {
            // Placed off to one side so a tilt in the surface normal changes
            // how much light the fragment receives. Directly overhead would
            // make an x-tilt symmetric and hide exactly what is being tested.
            lights.upload(&queue, &[crate::renderer::lights::Light {
                mask_channel: light.flatten(),
                position: light_pos,
                direction: glam::Vec3::new(0.0, -1.0, 0.0),
                kind: crate::renderer::lights::LightKind::Point,
                color: crate::renderer::Color3(255, 255, 255, 255),
                // Deliberately dim. The test palette's layer 0 is pure red, so
                // ambient alone already puts that channel at 153 of 255; a
                // bright light clips it and the lighting difference this test
                // exists to measure vanishes into saturation. Both tilts
                // rendered [255, 0, 0] at intensity 40.
                intensity: 3.0,
                range: 50.0,
                cone_angle_deg: 180.0,
                inner_cone_angle_deg: 0.0,
            }]);
        } else {
            lights.upload(&queue, &[]);
        }

        let pipeline = TerrainPipeline::new(&device, format, &uniforms.layout);

        // Built through TerrainMaterial, not a hand-assembled bind group.
        // The hand-built version drifted the moment the layout gained a
        // binding, and worse, it meant every render test proved things about
        // test code rather than about the code the renderer calls.
        let solid = |rgb: [u8; 4]| TerrainImage { width: 1, height: 1, rgba: rgb.to_vec() };
        let test_layers = [
            solid([255, 0, 0, 255]),   // 0 ground -> red
            solid([0, 0, 255, 255]),   // 1 rock   -> blue
            solid([0, 255, 0, 255]),   // 2 high   -> green
            solid([255, 255, 255, 255]), // 3 spare -> white
        ];
        let neutral = solid([128, 128, 128, 255]);

        let material = match palette {
            Palette::Test => TerrainMaterial::new(
                &device, &queue, &pipeline.material_layout,
                &test_layers, &neutral, splat, sky_occlusion, normals,
                &[], &[], TerrainMaterialUniform::default(),
            ),
            Palette::Fallback => {
                TerrainMaterial::fallback(&device, &queue, &pipeline.material_layout)
            }
        };

        // A quad filling clip space, carrying the normal under test. The vertex
        // stage multiplies by view_proj, which defaults to identity here, so
        // positions ARE clip coordinates.
        let verts = [
            vertex([-1.0, -1.0, 0.5], normal),
            vertex([3.0, -1.0, 0.5], normal),
            vertex([-1.0, 3.0, 0.5], normal),
        ];
        let vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("test_vb"),
            contents: bytemuck::cast_slice(&verts),
            usage: BufferUsages::VERTEX,
        });
        let ib = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("test_ib"),
            contents: bytemuck::cast_slice(&[0u32, 1, 2]),
            usage: BufferUsages::INDEX,
        });

        const SIZE: u32 = 8;
        let target = device.create_texture(&TextureDescriptor {
            label: Some("test_target"),
            size: Extent3d { width: SIZE, height: SIZE, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1,
            dimension: TextureDimension::D2,
            format,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let depth = device.create_texture(&TextureDescriptor {
            label: Some("test_depth"),
            size: Extent3d { width: SIZE, height: SIZE, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Depth32Float,
            usage: TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let target_view = target.create_view(&Default::default());
        let depth_view = depth.create_view(&Default::default());

        let readback = device.create_buffer(&BufferDescriptor {
            label: Some("test_readback"),
            size: (256 * SIZE) as u64,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("test_pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &target_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: Operations { load: LoadOp::Clear(Color::BLACK), store: StoreOp::Store },
                })],
                depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(Operations { load: LoadOp::Clear(1.0), store: StoreOp::Store }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                multiview_mask: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&pipeline.pipeline);
            pass.set_bind_group(0, &uniforms.bind_group, &[]);
            pass.set_bind_group(1, &material.bind_group, &[]);
            pass.set_vertex_buffer(0, vb.slice(..));
            pass.set_index_buffer(ib.slice(..), IndexFormat::Uint32);
            pass.draw_indexed(0..3, 0, 0..1);
        }
        encoder.copy_texture_to_buffer(
            TexelCopyTextureInfo {
                texture: &target, mip_level: 0, origin: Origin3d::ZERO, aspect: TextureAspect::All,
            },
            TexelCopyBufferInfo {
                buffer: &readback,
                layout: TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(256), rows_per_image: Some(SIZE) },
            },
            Extent3d { width: SIZE, height: SIZE, depth_or_array_layers: 1 },
        );
        queue.submit(Some(encoder.finish()));

        let slice = readback.slice(..);
        slice.map_async(MapMode::Read, |_| {});
        device.poll(PollType::Wait { submission_index: None, timeout: None }).ok();
        let data = slice.get_mapped_range().unwrap();
        let centre = (SIZE / 2) as usize * 256 + (SIZE / 2) as usize * 4;
        Some([data[centre], data[centre + 1], data[centre + 2], data[centre + 3]])
    }

    #[test]
    /// The optimisation itself, pinned in the source.
    ///
    /// Its effect is a frame time, which no unit test can see, and its absence
    /// is invisible: reverting to unconditional sampling renders exactly the
    /// same picture, only slower. So the guard is that the branches and the
    /// explicit-gradient sampling are still there.
    ///
    /// Measured cause: removing the ground took the frame from 24 ms to 11 ms
    /// of a 13.9 ms budget, while quartering the triangle count changed nothing
    /// -- terrain is fill-bound, and four layers sampled whether or not they
    /// contribute is where that fill goes.
    /// The ground's two probe-pass shaders -- the one reading the pass and the
    /// one writing it -- are valid WGSL and build on a real device with the
    /// layouts they are drawn with. See `TerrainRole`.
    #[test]
    fn the_probe_pass_terrain_shaders_validate_and_build() {
        use wgpu::naga;
        for role in [TerrainRole::Read, TerrainRole::ProbePass] {
            let src = terrain_shader_for(role);
            let module = naga::front::wgsl::parse_str(&src)
                .unwrap_or_else(|e| panic!("{role:?}: {}", e.emit_to_string(&src)));
            naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
                .validate(&module)
                .unwrap_or_else(|e| panic!("{role:?}: {e:?}"));
        }
        assert!(terrain_shader_for(TerrainRole::Read).contains("const PROBE_ENV_FROM_PASS: bool = true;"));
        let pass = terrain_shader_for(TerrainRole::ProbePass);
        assert!(pass.contains("const PROBE_SECONDARY_DEFERRED: bool = true;") && pass.contains("return probe_env_for_pass("));
        let Some((device, _queue)) = headless_gpu() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let lights = LightsUniform::new(&device);
        let (_shadows, uniforms) = crate::renderer::uniforms::test_support::scene_uniforms(&device, &lights);
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let probe_layout = crate::renderer::brush_pipeline::probe_pass::bind_group_layout(&device);
        let _read = TerrainPipeline::new_probe_reader(&device, TextureFormat::Rgba8UnormSrgb, &uniforms.layout, 4, &probe_layout);
        let fixups = crate::renderer::probe_fixup::ProbeFixups::new(&device, &uniforms.layout, 1024);
        let _pass = TerrainPipeline::new_probe_pass(&device, &uniforms.layout, &fixups);
        let err = pollster::block_on(scope.pop());
        assert!(err.is_none(), "the terrain probe pipelines failed to build: {err:?}");
    }

    #[test]
    fn weightless_layers_are_skipped_rather_than_sampled() {
        let src = super::terrain_shader();
        assert!(
            src.contains("textureSampleGrad"),
            "explicit gradients are what make a per-fragment branch legal at all",
        );
        for w in ["w.x > WEIGHT_EPS", "w.y > WEIGHT_EPS", "w.z > WEIGHT_EPS", "w.w > WEIGHT_EPS"] {
            assert!(src.contains(w), "missing the skip for {w}");
        }
    }

    #[test]
    fn the_gradients_are_taken_outside_every_branch() {
        // The reason this is legal. `textureSample` chooses its mip from
        // implicit derivatives and is only valid in uniform control flow --
        // which is exactly what forced all four layers to be sampled. Taking a
        // derivative INSIDE a branch reintroduces that problem while looking
        // like it had been solved, so every dpdx/dpdy must live in the one
        // function that runs before any branching.
        //
        // ONE DELIBERATE EXCEPTION, at the top level of `fs_main`: the
        // specular-antialiasing term takes `dpdx(shaded_n)`. That is not
        // inside a branch -- the layer blends have reconverged by then -- and
        // naga's own uniformity analysis accepts it, which
        // `every_scene_shader_survives_the_multiview_transform` exercises on
        // this very shader. The brush path does the same thing for the same
        // reason. The rule being enforced is "no derivative INSIDE a branch",
        // and counting them per function is how that is approximated cheaply.
        let src = super::terrain_shader();
        let frame = src
            .split("fn sample_frame")
            .nth(1)
            .and_then(|s| s.split("\nfn ").next())
            .expect("sample_frame present");
        let entry = src
            .split("@fragment")
            .nth(1)
            .expect("fragment entry present");
        let allowed = |tok: &str| frame.matches(tok).count() + entry.matches(tok).count();
        assert_eq!(
            src.matches("dpdx(").count(),
            allowed("dpdx("),
            "a derivative escaped both sample_frame and the fragment entry",
        );
        assert_eq!(src.matches("dpdy(").count(), allowed("dpdy("));
        // And the entry's own use must be exactly the specular-AA one, not a
        // second sampling gradient that has quietly moved out of the frame.
        // TWO named pairs are allowed at the top of the entry, both before any
        // branch: the specular-AA one above, and the pixel footprint the light
        // loop uses to keep a spot's edge a pixel wide (`pixel_footprint`).
        // Anything beyond those is a sampling gradient that escaped the frame.
        let footprint = "pixel_footprint = sqrt(length(dpdx(in.world_pos)) * length(dpdy(in.world_pos)));";
        let extra = usize::from(entry.contains(footprint));
        assert!(
            entry.matches("dpdx(").count() <= 1 + extra && entry.matches("dpdy(").count() <= 1 + extra,
            "more than the two named derivative pairs in the fragment entry; \
             sampling gradients belong in sample_frame",
        );
    }

    #[test]
    fn baked_sky_occlusion_darkens_the_terrain() {
        // The ground inside a sealed room used to light exactly as brightly as
        // the field outside it, because `shade` had no occlusion term at all --
        // only the BRUSH shader did. No amount of shadow-map work could fix
        // that: the term that was wrong was the ambient, not the direct light.
        //
        // This test also compiles the terrain pipeline, which is the only thing
        // that validates its WGSL: a bad field accessor in that shader builds
        // clean and dies at pipeline creation on the device.
        let sealed = TerrainImage { width: 1, height: 1, rgba: vec![0, 0, 0, 255] };
        let Some(open) = render_quad([0.0, 1.0, 0.0], Palette::Test, None, None) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let dark = render_quad([0.0, 1.0, 0.0], Palette::Test, None, Some(&sealed)).unwrap();
        assert!(
            dark[0] < open[0],
            "zero sky visibility must darken the ground: {dark:?} vs open {open:?}",
        );
    }

    #[test]
    fn unbaked_terrain_shades_exactly_as_it_did_before() {
        // The neutral is WHITE here, not black: this value is MULTIPLIED into
        // the ambient term, the opposite of the brush lightmap's additive RGB.
        // Binding the wrong neutral would put every unbaked terrain in the game
        // into permanent night, which reads as a broken shader.
        let full = TerrainImage { width: 1, height: 1, rgba: vec![255, 255, 255, 255] };
        let Some(absent) = render_quad([0.0, 1.0, 0.0], Palette::Test, None, None) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let explicit = render_quad([0.0, 1.0, 0.0], Palette::Test, None, Some(&full)).unwrap();
        assert_eq!(
            absent, explicit,
            "no occlusion map must shade identically to a fully-open one: \
             {absent:?} vs {explicit:?}",
        );
    }

    #[test]
    fn a_mip_chain_runs_all_the_way_down_to_one_texel() {
        // Stopping early leaves the hardware clamped to the smallest level
        // present, which brings the shimmer back at exactly the distances mips
        // exist to fix.
        assert_eq!(mip_levels_for(1024, 1024), 11);
        assert_eq!(mip_levels_for(1, 1), 1);
        assert_eq!(mip_levels_for(256, 64), 9, "levels follow the LONGER side");

        let levels = mip_levels_for(8, 8);
        let chain = mip_chain(&[128u8; 8 * 8 * 4], 8, 8, levels);
        assert_eq!(chain.len() as u32, levels);
        assert_eq!((chain[0].1, chain[0].2), (8, 8));
        assert_eq!(chain.last().unwrap().1, 1, "chain must reach 1x1");
        for (data, w, h) in &chain {
            assert_eq!(data.len(), (w * h * 4) as usize, "level {w}x{h} is the wrong size");
        }
    }

    #[test]
    fn mips_are_averaged_in_linear_space_not_srgb() {
        // Averaging sRGB BYTES directly is the classic "distant ground goes
        // muddy" bug: the encoding is not linear in intensity, so the mean of
        // two encoded values is darker than the encoding of their mean. Black
        // and white must average to mid GREY -- sRGB 188, not sRGB 128.
        let mut px = vec![0u8; 2 * 2 * 4];
        for i in 0..4 {
            let v = if i % 2 == 0 { 0u8 } else { 255u8 };
            for c in 0..3 {
                px[i * 4 + c] = v;
            }
            px[i * 4 + 3] = 255;
        }
        let chain = mip_chain(&px, 2, 2, 2);
        let r = chain[1].0[0];
        assert!(
            (185..=191).contains(&r),
            "black+white must average to sRGB ~188 (linear 0.5), got {r} -- \
             {} suggests the average was taken on the encoded bytes",
            if r < 140 { "which" } else { "this" },
        );
        assert_eq!(chain[1].0[3], 255, "alpha is already linear and must not be transferred");
    }

    #[test]
    fn a_flat_colour_survives_every_mip_level() {
        // A round-trip check on the transfer functions: a uniform image cannot
        // change under a box filter, so any drift here is the sRGB conversion
        // being lossy in a direction that would tint every distant surface.
        let chain = mip_chain(&[200u8; 16 * 16 * 4], 16, 16, mip_levels_for(16, 16));
        for (data, w, h) in &chain {
            for (i, b) in data.iter().enumerate() {
                assert!(
                    b.abs_diff(200) <= 1,
                    "level {w}x{h} byte {i} drifted to {b} from 200",
                );
            }
        }
    }

    fn flat_ground_shades_from_the_ground_layer() {
        let Some(px) = render_quad_with_normal([0.0, 1.0, 0.0]) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        // Layer 0 is red. Lighting scales it, so assert the CHANNEL BALANCE
        // rather than an exact value -- an absolute assert would be a test of
        // the light rig rather than of the splat blend.
        assert!(px[0] > px[2], "flat ground should take the ground layer (red), got {px:?}");
    }

    #[test]
    fn a_cliff_shades_from_the_rock_layer() {
        // Normal pointing sideways: 90 degrees of slope, well past slope_end.
        let Some(px) = render_quad_with_normal([1.0, 0.0, 0.0]) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        assert!(px[2] > px[0], "a cliff should take the rock layer (blue), got {px:?}");
    }

    #[test]
    fn the_slope_blend_is_gradual_rather_than_a_hard_switch() {
        // Halfway through the 22..40 degree band, both layers should contribute
        // -- a hard switch reads as a visible seam right across a hillside.
        let a = 31.0_f32.to_radians();
        let Some(px) = render_quad_with_normal([a.sin(), a.cos(), 0.0]) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        assert!(px[0] > 8, "ground layer vanished mid-blend: {px:?}");
        assert!(px[2] > 8, "rock layer absent mid-blend: {px:?}");
    }
}

/// The textures and settings one terrain draws with.
///
/// Built separately from the pipeline because the pipeline is per-device and
/// this is per-scene: two levels want different ground, and rebuilding a
/// pipeline to change a texture would be absurd.
pub struct TerrainMaterial {
    pub bind_group: BindGroup,
    pub uniform: Buffer,
}

/// Mip levels for a texture of this size: down to 1x1, as the spec requires.
///
/// A chain that stops early leaves the smallest levels missing, and the
/// hardware clamps to the last one present -- which brings the aliasing back at
/// exactly the distances mips existed to fix.
pub fn mip_levels_for(w: u32, h: u32) -> u32 {
    32 - w.max(h).max(1).leading_zeros()
}

/// A box-filtered mip chain for one RGBA8 image, smallest-last.
///
/// Box filtering on the CPU rather than a blit pipeline: this runs once at
/// load, and a render pass per level per layer would need its own pipeline,
/// bind groups and a non-sRGB view of an sRGB texture. The quality difference
/// at these sizes is not visible on a headset; the difference in moving parts
/// is considerable.
///
/// Averaged in LINEAR space, not sRGB. Averaging sRGB bytes directly darkens
/// every mip -- the classic "distant ground goes muddy" artefact -- because the
/// encoding is not linear in intensity and the mean of two encoded values is
/// not the encoding of their mean.
/// The same box-filtered chain for data that is NOT colour.
///
/// Normal, roughness and occlusion maps are measurements, not colours. Putting
/// them through the sRGB transfer bends every value toward the dark end -- for
/// a normal map that tilts every bump, which this renderer has shipped once
/// already and does not intend to again.
/// Roughness for a layer that has no map, as an 8-bit value.
///
/// Matte, because dirt, gravel and grass are. A layer that disagrees ships a
/// roughness map and stops using this.
pub const DEFAULT_TERRAIN_ROUGHNESS: u8 = 200;

/// The most extra roughness-squared a normal's lost variation may add.
const TERRAIN_NORMAL_VARIANCE_CLAMP: f32 = 0.18;

/// A roughness mip chain that carries the variation the NORMAL map lost when it
/// was mipped -- the same treatment brush materials get.
///
/// Mipping a normal map averages its normals, and shading is not linear in the
/// normal: the lighting of an averaged normal is not the average of the
/// lighting. The LENGTH of the averaged (unnormalised) normal records how much
/// they disagreed, and that converts into variance, and variance is roughness.
///
/// `sigma2 = (1 - |Na|) / |Na|`, combined as `r' = sqrt(r^2 + min(2*sigma2, k))`.
/// ROUGHNESS VALUES DO NOT ADD -- they combine in variance space, which is why
/// this squares before summing and takes the root after.
///
/// Level 0 is untouched: nothing has been averaged there, so nothing was lost,
/// and ground seen underfoot keeps exactly the roughness its map specifies.
pub fn roughness_chain_with_normal_variance(
    rough: &[u8],
    normal: &[u8],
    w: u32,
    h: u32,
    levels: u32,
) -> Vec<(Vec<u8>, u32, u32)> {
    let mut chain = mip_chain_linear(rough, w, h, levels);

    // The normal halved repeatedly in FLOAT, kept UNNORMALISED -- the
    // shortening is the whole measurement, and doing it in f32 rather than
    // re-reading 8-bit mips avoids quantising twice.
    let mut vecs: Vec<[f32; 3]> = (0..(w * h) as usize)
        .map(|i| {
            [
                normal[i * 4] as f32 / 255.0 * 2.0 - 1.0,
                normal[i * 4 + 1] as f32 / 255.0 * 2.0 - 1.0,
                normal[i * 4 + 2] as f32 / 255.0 * 2.0 - 1.0,
            ]
        })
        .collect();
    let (mut cw, mut ch) = (w, h);

    for level in 1..chain.len() {
        let (nw, nh) = ((cw / 2).max(1), (ch / 2).max(1));
        let mut next = vec![[0.0f32; 3]; (nw * nh) as usize];
        for y in 0..nh {
            for x in 0..nw {
                let mut acc = [0.0f32; 3];
                for (dx, dy) in [(0u32, 0u32), (1, 0), (0, 1), (1, 1)] {
                    let sx = (x * 2 + dx).min(cw - 1);
                    let sy = (y * 2 + dy).min(ch - 1);
                    let v = vecs[(sy * cw + sx) as usize];
                    acc[0] += v[0];
                    acc[1] += v[1];
                    acc[2] += v[2];
                }
                next[(y * nw + x) as usize] = [acc[0] / 4.0, acc[1] / 4.0, acc[2] / 4.0];
            }
        }
        vecs = next;
        cw = nw;
        ch = nh;

        let (data, lw, lh) = &mut chain[level];
        if *lw != cw || *lh != ch {
            // The two chains fell out of step; stop rather than write variance
            // into the wrong texels.
            break;
        }
        for i in 0..(cw * ch) as usize {
            let v = vecs[i];
            let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            let sigma2 = if len > 1e-4 { (1.0 - len) / len } else { 1.0 };
            let kernel = (2.0 * sigma2).min(TERRAIN_NORMAL_VARIANCE_CLAMP);
            let base = data[i * 4] as f32 / 255.0;
            let filtered = (base * base + kernel).clamp(0.0, 1.0).sqrt();
            let byte = (filtered * 255.0).round().clamp(0.0, 255.0) as u8;
            data[i * 4] = byte;
            data[i * 4 + 1] = byte;
            data[i * 4 + 2] = byte;
        }
    }
    chain
}

pub fn mip_chain_linear(rgba: &[u8], w: u32, h: u32, levels: u32) -> Vec<(Vec<u8>, u32, u32)> {
    let mut out = vec![(rgba.to_vec(), w, h)];
    for _ in 1..levels {
        let (src, sw, sh) = out.last().unwrap();
        let (dw, dh) = ((sw / 2).max(1), (sh / 2).max(1));
        let mut dst = vec![0u8; (dw * dh * 4) as usize];
        for y in 0..dh {
            for x in 0..dw {
                for c in 0..4 {
                    let fetch = |sx: u32, sy: u32| -> f32 {
                        let i = ((sy.min(sh - 1) * sw + sx.min(sw - 1)) * 4 + c) as usize;
                        src[i] as f32 / 255.0
                    };
                    let (x0, y0) = (x * 2, y * 2);
                    let avg = (fetch(x0, y0) + fetch(x0 + 1, y0) + fetch(x0, y0 + 1)
                        + fetch(x0 + 1, y0 + 1))
                        * 0.25;
                    dst[((y * dw + x) * 4 + c) as usize] =
                        (avg.clamp(0.0, 1.0) * 255.0).round() as u8;
                }
            }
        }
        out.push((dst, dw, dh));
    }
    out
}

pub fn mip_chain(rgba: &[u8], w: u32, h: u32, levels: u32) -> Vec<(Vec<u8>, u32, u32)> {
    fn to_linear(c: u8) -> f32 {
        let s = c as f32 / 255.0;
        if s <= 0.040_45 { s / 12.92 } else { ((s + 0.055) / 1.055).powf(2.4) }
    }
    fn to_srgb(l: f32) -> u8 {
        let s = if l <= 0.003_130_8 { l * 12.92 } else { 1.055 * l.powf(1.0 / 2.4) - 0.055 };
        (s.clamp(0.0, 1.0) * 255.0).round() as u8
    }

    let mut out = vec![(rgba.to_vec(), w, h)];
    for _ in 1..levels {
        let (src, sw, sh) = out.last().unwrap();
        let (dw, dh) = ((sw / 2).max(1), (sh / 2).max(1));
        let mut dst = vec![0u8; (dw * dh * 4) as usize];
        for y in 0..dh {
            for x in 0..dw {
                for c in 0..4 {
                    // Alpha is linear already and must NOT go through the sRGB
                    // transfer; only the colour channels do.
                    let fetch = |sx: u32, sy: u32| -> f32 {
                        let i = ((sy.min(sh - 1) * sw + sx.min(sw - 1)) * 4 + c) as usize;
                        if c == 3 { src[i] as f32 / 255.0 } else { to_linear(src[i]) }
                    };
                    let (x0, y0) = (x * 2, y * 2);
                    let avg = (fetch(x0, y0) + fetch(x0 + 1, y0) + fetch(x0, y0 + 1)
                        + fetch(x0 + 1, y0 + 1))
                        * 0.25;
                    dst[((y * dw + x) * 4 + c) as usize] = if c == 3 {
                        (avg.clamp(0.0, 1.0) * 255.0).round() as u8
                    } else {
                        to_srgb(avg)
                    };
                }
            }
        }
        out.push((dst, dw, dh));
    }
    out
}

impl TerrainMaterial {
    /// Build from four RGBA8 layer images plus a macro image.
    ///
    /// Every layer must be the same size -- they go into one D2Array, which is
    /// what lets the shader index a layer by splat weight without a branch per
    /// layer. `layers` shorter than 4 is padded by repeating the last one, so a
    /// scene that only authors ground and rock still binds a complete array
    /// rather than failing validation.
    pub fn new(
        device: &Device,
        queue: &Queue,
        layout: &BindGroupLayout,
        layers: &[TerrainImage],
        macro_image: &TerrainImage,
        // Authored blend weights, or None to leave the shader on its slope- and
        // height-driven blend. None still binds a texture -- see the layout --
        // and clears `use_splat` so nothing reads it.
        splat: Option<&TerrainImage>,
        sky_occlusion: Option<&TerrainImage>,
        // Per-layer normal maps, in the same slot order as `layers`. Shorter or
        // sparser than four is fine; the gaps become flat, which the shader
        // treats as no perturbation at all.
        normals: &[Option<TerrainImage>],
        // Per-layer roughness and occlusion, same slot order. Gaps take the
        // neutral value for each -- `DEFAULT_TERRAIN_ROUGHNESS` and white --
        // so a project that has not authored them behaves as it always did.
        rough: &[Option<TerrainImage>],
        ao: &[Option<TerrainImage>],
        settings: TerrainMaterialUniform,
    ) -> Self {
        assert!(!layers.is_empty(), "terrain material needs at least one layer");
        let (w, h) = (layers[0].width, layers[0].height);
        assert!(
            layers.iter().all(|l| l.width == w && l.height == h),
            "all terrain layers must share one size to live in a D2Array",
        );

        // MIPMAPPED, and this is not a nicety.
        //
        // Ground is the one surface in the level that is always viewed at a
        // grazing angle running to the horizon, so a screen pixel a few metres
        // out covers many texels. With a single mip level the hardware picks
        // one of them, and which one changes as the head moves: the grass
        // crawls and sparkles. The sampler already asked for
        // `mipmap_filter: Linear` -- it just had nothing to filter between,
        // which is a silent no-op rather than an error.
        //
        // It is worth being precise about why this surfaced now: the layer
        // tiling used to be 8 m, and at that scale the ground was blurry enough
        // to hide the aliasing. Setting it to the material's authored 2 m
        // quadrupled the spatial frequency and made a pre-existing bug visible.
        // The tiling is right; the missing mips were always wrong.
        let mip_level_count = mip_levels_for(w, h);
        let array = device.create_texture(&TextureDescriptor {
            label: Some("terrain_layers"),
            size: Extent3d { width: w, height: h, depth_or_array_layers: 4 },
            mip_level_count,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8UnormSrgb,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });
        for slot in 0..4 {
            let src = layers.get(slot).unwrap_or_else(|| layers.last().unwrap());
            let chain = mip_chain(&src.rgba, w, h, mip_level_count);
            for (level, (data, lw, lh)) in chain.iter().enumerate() {
                queue.write_texture(
                    TexelCopyTextureInfo {
                        texture: &array,
                        mip_level: level as u32,
                        origin: Origin3d { x: 0, y: 0, z: slot as u32 },
                        aspect: TextureAspect::All,
                    },
                    data,
                    TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(4 * lw),
                        rows_per_image: Some(*lh),
                    },
                    Extent3d { width: *lw, height: *lh, depth_or_array_layers: 1 },
                );
            }
        }

        let macro_tex = device.create_texture(&TextureDescriptor {
            label: Some("terrain_macro"),
            size: Extent3d { width: macro_image.width, height: macro_image.height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            // NOT sRGB: this is a modulation factor, not a colour. Decoding it
            // through sRGB would bend the neutral point away from 0.5 and tint
            // the whole terrain.
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            TexelCopyTextureInfo {
                texture: &macro_tex,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            &macro_image.rgba,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * macro_image.width),
                rows_per_image: Some(macro_image.height),
            },
            Extent3d { width: macro_image.width, height: macro_image.height, depth_or_array_layers: 1 },
        );

        // Normal maps live in their own array, sized to the colour layers so
        // both index by the same slot. NOT sRGB: these encode a direction, and
        // decoding them through a colour curve bends every bump.
        // MIPPED, and that was the bug.
        //
        // The colour array above has had a mip chain all along; this one was
        // created with `mip_level_count: 1` while the shared sampler asks for
        // `mipmap_filter: Linear` and 8x anisotropy -- both silent no-ops
        // against a single level. So every distant pixel POINT-SAMPLED a
        // full-resolution normal map, the shading normal changed randomly from
        // pixel to pixel, and head tracking moved that around every frame.
        // That is the terrain shimmer, and it is why brush surfaces -- whose
        // normals are mipped -- never had it (headset, 2026-09-22).
        //
        // It also silently defeated the Toksvig term added just before this:
        // that measures how short the AVERAGED normal is, and with no mip
        // chain nothing is ever averaged, so the variance it read was always
        // zero. The method was right and had nothing to measure.
        let normal_array = device.create_texture(&TextureDescriptor {
            label: Some("terrain_normals"),
            size: Extent3d { width: w, height: h, depth_or_array_layers: 4 },
            mip_level_count,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });
        for slot in 0..4 {
            let flat = solid_image(FLAT_NORMAL, w, h);
            let src = match normals.get(slot).and_then(|n| n.as_ref()) {
                Some(img) if img.width == w && img.height == h => img.clone_image(),
                Some(img) => resample(img, w, h),
                None => flat,
            };
            // LINEAR, not sRGB: a normal is a direction.
            let chain = mip_chain_linear(&src.rgba, w, h, mip_level_count);
            for (level, (data, lw, lh)) in chain.iter().enumerate() {
                queue.write_texture(
                    TexelCopyTextureInfo {
                        texture: &normal_array,
                        mip_level: level as u32,
                        origin: Origin3d { x: 0, y: 0, z: slot as u32 },
                        aspect: TextureAspect::All,
                    },
                    data,
                    TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(4 * lw),
                        rows_per_image: Some(*lh),
                    },
                    Extent3d { width: *lw, height: *lh, depth_or_array_layers: 1 },
                );
            }
        }

        // ROUGHNESS AND OCCLUSION, mipped and linear like the normal array.
        //
        // Both are measurements rather than colours, so neither goes through
        // the sRGB transfer, and both need a mip chain for the same reason the
        // normals did: a distant pixel that point-samples a full-resolution
        // map gets a different answer every frame as the head moves.
        //
        // The roughness chain also folds in the NORMAL'S lost variation, which
        // is the same treatment brush materials get -- mipping a normal map
        // averages its normals, and the shortness of that average measures how
        // much detail the level threw away. Putting it back as roughness is
        // what keeps a highlight from popping in and out at distance.
        let mut rough_arr = Vec::with_capacity(4);
        let mut ao_arr = Vec::with_capacity(4);
        for slot in 0..4 {
            let pick = |src: &[Option<TerrainImage>], fallback: u8| -> TerrainImage {
                match src.get(slot).and_then(|x| x.as_ref()) {
                    Some(img) if img.width == w && img.height == h => img.clone_image(),
                    Some(img) => resample(img, w, h),
                    None => solid_image([fallback; 3], w, h),
                }
            };
            rough_arr.push(pick(rough, DEFAULT_TERRAIN_ROUGHNESS));
            ao_arr.push(pick(ao, 255));
        }

        let data_array = |label: &str| {
            device.create_texture(&TextureDescriptor {
                label: Some(label),
                size: Extent3d { width: w, height: h, depth_or_array_layers: 4 },
                mip_level_count,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Rgba8Unorm,
                usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
                view_formats: &[],
            })
        };
        let rough_array = data_array("terrain_rough");
        let ao_array = data_array("terrain_ao");

        for slot in 0..4 {
            let normal_src = match normals.get(slot).and_then(|n| n.as_ref()) {
                Some(img) if img.width == w && img.height == h => img.clone_image(),
                Some(img) => resample(img, w, h),
                None => solid_image(FLAT_NORMAL, w, h),
            };
            let chains = [
                (
                    &rough_array,
                    roughness_chain_with_normal_variance(
                        &rough_arr[slot].rgba,
                        &normal_src.rgba,
                        w,
                        h,
                        mip_level_count,
                    ),
                ),
                (&ao_array, mip_chain_linear(&ao_arr[slot].rgba, w, h, mip_level_count)),
            ];
            for (tex, chain) in chains {
                for (level, (data, lw, lh)) in chain.iter().enumerate() {
                    queue.write_texture(
                        TexelCopyTextureInfo {
                            texture: tex,
                            mip_level: level as u32,
                            origin: Origin3d { x: 0, y: 0, z: slot as u32 },
                            aspect: TextureAspect::All,
                        },
                        data,
                        TexelCopyBufferLayout {
                            offset: 0,
                            bytes_per_row: Some(4 * lw),
                            rows_per_image: Some(*lh),
                        },
                        Extent3d { width: *lw, height: *lh, depth_or_array_layers: 1 },
                    );
                }
            }
        }

        let sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("terrain_sampler"),
            address_mode_u: AddressMode::Repeat,
            address_mode_v: AddressMode::Repeat,
            address_mode_w: AddressMode::Repeat,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: MipmapFilterMode::Linear,
            // Ground is seen almost entirely at grazing angles, which is the
            // exact case trilinear filtering handles worst: it picks a mip for
            // the SHORT axis of the footprint and so over-blurs along the long
            // one. 8x is the usual sweet spot and is cheap on a tile GPU
            // compared with the fill it saves by not needing a sharper mip.
            anisotropy_clamp: 8,
            ..Default::default()
        });

        // RGBA8 unorm, NOT sRGB: these are blend weights, and sRGB-decoding
        // them would bend a 50/50 blend away from the middle -- the same reason
        // the macro texture is linear.
        let splat_image = splat.unwrap_or(&NO_SPLAT);
        let splat_tex = device.create_texture(&TextureDescriptor {
            label: Some("terrain_splat"),
            size: Extent3d {
                width: splat_image.width,
                height: splat_image.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            TexelCopyTextureInfo {
                texture: &splat_tex,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            &splat_image.rgba,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * splat_image.width),
                rows_per_image: Some(splat_image.height),
            },
            Extent3d {
                width: splat_image.width,
                height: splat_image.height,
                depth_or_array_layers: 1,
            },
        );

        // Sky visibility over the footprint. WHITE when unbaked: this value is
        // multiplied into the ambient term, so 1.0 is the neutral that
        // reproduces the shading terrain had before the map existed.
        //
        // AN ARRAY: layer 0 is the ground's own map, and any further layers
        // the image's bytes carry -- whole images of the same size, back to
        // back -- are the stationary lamps' masks (`with_stationary_masks`).
        let occ_image = sky_occlusion.unwrap_or(&FULL_SKY);
        let occ_layers = (occ_image.rgba.len() / (4 * occ_image.width as usize * occ_image.height as usize).max(1)).max(1) as u32;
        let sky_occ_tex = device.create_texture(&TextureDescriptor {
            label: Some("terrain_sky_occlusion"),
            size: Extent3d {
                width: occ_image.width,
                height: occ_image.height,
                depth_or_array_layers: occ_layers,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            // NOT sRGB: a visibility fraction, not a colour. Decoding it would
            // bend every value and darken the ground non-linearly.
            format: TextureFormat::Rgba8Unorm,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            TexelCopyTextureInfo {
                texture: &sky_occ_tex,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            &occ_image.rgba[..(4 * occ_image.width * occ_image.height * occ_layers) as usize],
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * occ_image.width),
                rows_per_image: Some(occ_image.height),
            },
            Extent3d {
                width: occ_image.width,
                height: occ_image.height,
                depth_or_array_layers: occ_layers,
            },
        );

        // The flag is derived here rather than left to the caller: a material
        // built with a map that the shader then ignores, or without one that it
        // then reads, are both silent and both look like a shader bug.
        let mut settings = settings;
        settings.use_splat = if splat.is_some() { 1.0 } else { 0.0 };

        let uniform = device.create_buffer(&BufferDescriptor {
            label: Some("terrain_material_uniform"),
            size: std::mem::size_of::<TerrainMaterialUniform>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform, 0, bytemuck::bytes_of(&settings));

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("terrain_material"),
            layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(
                        &array.create_view(&TextureViewDescriptor {
                            dimension: Some(TextureViewDimension::D2Array),
                            ..Default::default()
                        }),
                    ),
                },
                BindGroupEntry { binding: 1, resource: BindingResource::Sampler(&sampler) },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(
                        &macro_tex.create_view(&TextureViewDescriptor::default()),
                    ),
                },
                BindGroupEntry { binding: 3, resource: uniform.as_entire_binding() },
                BindGroupEntry {
                    binding: 4,
                    resource: BindingResource::TextureView(
                        &splat_tex.create_view(&TextureViewDescriptor::default()),
                    ),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: BindingResource::TextureView(
                        &normal_array.create_view(&TextureViewDescriptor {
                            dimension: Some(TextureViewDimension::D2Array),
                            ..Default::default()
                        }),
                    ),
                },
                BindGroupEntry {
                    binding: 6,
                    resource: BindingResource::TextureView(&sky_occ_tex.create_view(&TextureViewDescriptor {
                        dimension: Some(TextureViewDimension::D2Array),
                        ..Default::default()
                    })),
                },
                BindGroupEntry {
                    binding: 7,
                    resource: BindingResource::TextureView(
                        &rough_array.create_view(&TextureViewDescriptor {
                            dimension: Some(TextureViewDimension::D2Array),
                            ..Default::default()
                        }),
                    ),
                },
                BindGroupEntry {
                    binding: 8,
                    resource: BindingResource::TextureView(
                        &ao_array.create_view(&TextureViewDescriptor {
                            dimension: Some(TextureViewDimension::D2Array),
                            ..Default::default()
                        }),
                    ),
                },
            ],
        });

        Self { bind_group, uniform }
    }

    /// A material with no authored textures: flat colours per layer.
    ///
    /// This exists so terrain can go through the real pipeline from day one. A
    /// renderer path that only works once an artist has produced four tiling
    /// textures stays unwired and therefore unverified for as long as that
    /// takes, and the wiring bugs all surface later, at once, blamed on the art.
    /// The colours are the ones the previous flat-shaded terrain used, so
    /// turning this on changes shading but not palette.
    pub fn fallback(device: &Device, queue: &Queue, layout: &BindGroupLayout) -> Self {
        Self::fallback_with_splat(device, queue, layout, None, None)
    }

    /// The fallback palette, with authored blend weights applied to it.
    ///
    /// This is the state a level reaches first: someone has painted WHERE each
    /// material goes long before an artist has produced what each material
    /// looks like. Showing painted regions in flat colour is the honest render
    /// of that, and it matches what the editor previews.
    pub fn fallback_with_splat(
        device: &Device,
        queue: &Queue,
        layout: &BindGroupLayout,
        splat: Option<&TerrainImage>,
        sky_occlusion: Option<&TerrainImage>,
    ) -> Self {
        Self::from_layers(
            device, queue, layout, &[None, None, None, None], &[None, None, None, None], splat,
            sky_occlusion,
        )
    }

    /// Build from whatever layer textures loaded, filling the gaps.
    ///
    /// Per LAYER rather than all-or-nothing: a project part-way through
    /// authoring its materials should see the layers it has rather than lose
    /// all four because one file is missing.
    ///
    /// Gaps become solid images at the SAME SIZE as the loaded layers, not 1x1.
    /// The four share one D2Array, so a mismatched layer is a validation
    /// failure rather than a smaller texture -- and that failure would appear
    /// only on the project that happens to be missing one file.
    pub fn from_layers(
        device: &Device,
        queue: &Queue,
        layout: &BindGroupLayout,
        layers: &[Option<TerrainImage>],
        normals: &[Option<TerrainImage>],
        splat: Option<&TerrainImage>,
        sky_occlusion: Option<&TerrainImage>,
    ) -> Self {
        Self::from_layers_with(
            device,
            queue,
            layout,
            layers,
            normals,
            // This convenience form predates roughness and occlusion maps and
            // keeps its old shape: both fall back to their neutral, so callers
            // that have not been updated behave exactly as they did.
            &[],
            &[],
            splat,
            sky_occlusion,
            TerrainMaterialUniform::default(),
        )
    }

    /// The same, with the project's authored material settings.
    ///
    /// Split from `from_layers` rather than replacing it because the defaults
    /// are the right answer for every caller that has no project on disk to
    /// read from -- the fallback material and the pipeline's own tests.
    pub fn from_layers_with(
        device: &Device,
        queue: &Queue,
        layout: &BindGroupLayout,
        layers: &[Option<TerrainImage>],
        normals: &[Option<TerrainImage>],
        rough: &[Option<TerrainImage>],
        ao: &[Option<TerrainImage>],
        splat: Option<&TerrainImage>,
        sky_occlusion: Option<&TerrainImage>,
        settings: TerrainMaterialUniform,
    ) -> Self {
        let (w, h) = layers
            .iter()
            .flatten()
            .map(|l| (l.width, l.height))
            .next()
            .unwrap_or((1, 1));

        let filled: Vec<TerrainImage> = FALLBACK_LAYER_COLOURS
            .iter()
            .enumerate()
            .map(|(i, rgb)| match layers.get(i).and_then(|l| l.as_ref()) {
                // Already the right size by construction when it is the one the
                // size came from; resampled otherwise so a set of mixed-size
                // files still binds.
                Some(img) if img.width == w && img.height == h => TerrainImage {
                    width: img.width,
                    height: img.height,
                    rgba: img.rgba.clone(),
                },
                Some(img) => resample(img, w, h),
                None => solid_image(*rgb, w, h),
            })
            .collect();

        Self::new(
            device,
            queue,
            layout,
            &filled,
            &solid_image([128, 128, 128], 1, 1),
            splat,
            sky_occlusion,
            normals,
            rough,
            ao,
            settings,
        )
    }

}

/// A decoded RGBA8 image destined for a texture array.
///
/// Named for terrain because that is what first needed it; brush materials load
/// through the same type rather than through a copy of it. A second decode-and-
/// resample path is exactly the shape of duplication this codebase has been
/// bitten by before -- one copy gets fixed and the other keeps the bug.
pub struct TerrainImage {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

impl TerrainImage {
    /// The ground's own map with the stationary lamps' masks after it, as the
    /// layers of one array -- see `TerrainMaterial::new`. `None` when a mask
    /// is not the map's size (a bake from before they shared its grid): the
    /// lamps then light the ground unshadowed, as they always had.
    pub fn with_stationary_masks(&self, masks: &[TerrainImage]) -> Option<TerrainImage> {
        if masks.iter().any(|m| m.width != self.width || m.height != self.height || m.rgba.len() != self.rgba.len()) {
            return None;
        }
        let mut rgba = self.rgba.clone();
        for m in masks {
            rgba.extend_from_slice(&m.rgba);
        }
        Some(TerrainImage { width: self.width, height: self.height, rgba })
    }
}

#[cfg(test)]
mod material_tests {
    use super::tests::*;

    /// A STATIONARY LAMP ON THE GROUND TAKES ITS SHADOW FROM THE MASK LAYER
    /// after the ground map: channel 0 fully shadowed leaves the ground as dark
    /// as with no lamp at all, fully lit as bright as an unmasked lamp. Before
    /// the masks, the ground took every lamp through the walls.
    #[test]
    fn a_stationary_lamp_on_the_ground_reads_its_mask() {
        let map = super::TerrainImage { width: 1, height: 1, rgba: vec![255, 128, 0, 255] };
        let mask = |d: u8| super::TerrainImage { width: 1, height: 1, rgba: vec![d, 0, 255, 255] };
        let layered = |d: u8| map.with_stationary_masks(&[mask(d)]).unwrap();
        // Flat ground, facing the harness's lamp up and to one side.
        let up = [0.0, 1.0, 0.0];
        let render = |sky: &super::TerrainImage, light: Option<Option<u8>>| {
            render_quad_lit_by(up, Palette::Test, None, Some(sky), &[None, None, None, None], light)
        };
        let Some(dark) = render(&map, None) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let lamp = render(&map, Some(None)).unwrap();
        let shadowed = render(&layered(0), Some(Some(0))).unwrap();
        let lit = render(&layered(255), Some(Some(0))).unwrap();
        assert!(lamp[0] > dark[0] + 10, "the test lamp does not light the ground: {lamp:?} vs {dark:?}");
        assert_eq!(shadowed, dark, "a lamp its mask hides still lit the ground");
        assert_eq!(lit, lamp, "a lamp its mask shows is not the unmasked lamp");
        // And a map with no mask layers is the ground as it always was.
        let other_channel = render(&layered(0), Some(Some(1))).unwrap();
        assert_eq!(other_channel, lamp, "a lamp on another channel read this one's shadow");
    }
    use super::*;

    /// The fallback material must produce the SAME layer selection as the
    /// hand-built test material: flat ground takes layer 0, a cliff takes layer
    /// 1. Colours differ (the fallback ships real terrain colours, the test
    /// material ships primaries), so this asserts on which layer won, via the
    /// channel that separates them, rather than on exact bytes.
    #[test]
    fn the_fallback_material_selects_layers_the_same_way() {
        let Some(flat) = render_quad_with_fallback_material([0.0, 1.0, 0.0]) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let cliff = render_quad_with_fallback_material([1.0, 0.05, 0.0]).unwrap();

        // Ground (86,112,62) is green-dominant; rock (104,100,94) is near-grey.
        //
        // The margin is smaller than it looks it should be because this quad
        // renders dark, and a tone curve compresses hardest at the bottom --
        // ACES puts 18% grey at 0.106. The ORDERING this test cares about is
        // untouched; only the absolute byte gap shrank, so the threshold is
        // stated against the range the pixels actually occupy rather than the
        // untonemapped one it was first calibrated in.
        let greenness = |p: [u8; 4]| p[1] as i32 - p[2] as i32;
        assert!(
            greenness(flat) > greenness(cliff) + 4 && greenness(flat) > 4,
            "flat ground should read greener than a cliff: flat {flat:?} cliff {cliff:?}",
        );
        assert!(flat[3] == 255 && cliff[3] == 255, "terrain must be opaque");
    }

    /// A material built with fewer than four layers still binds a complete
    /// D2Array. Without the padding this is a validation error at bind-group
    /// creation, which is the kind of thing that only shows up on the scene
    /// that happens to author two layers.
    #[test]
    fn a_short_layer_list_pads_to_a_full_array() {
        let Some((device, queue)) = headless_gpu() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let layout = material_bind_group_layout(&device);
        let one = TerrainImage { width: 1, height: 1, rgba: vec![10, 20, 30, 255] };
        let err_scope_1 = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let _m = TerrainMaterial::new(
            &device, &queue, &layout,
            std::slice::from_ref(&one),
            &one,
            None,
            None,
            &[None, None, None, None],
            &[],
            &[],
            TerrainMaterialUniform::default(),
        );
        assert!(
            pollster::block_on(err_scope_1.pop()).is_none(),
            "a single-layer terrain material must still bind validly",
        );
    }
}

/// The 1x1 stand-in bound when a scene authors no splat map.
///
/// Its contents are never read -- `use_splat` is 0 -- but something has to
/// satisfy the binding, and a shared constant is cheaper than each caller
/// inventing one.
static NO_SPLAT: std::sync::LazyLock<TerrainImage> = std::sync::LazyLock::new(|| TerrainImage {
    width: 1,
    height: 1,
    rgba: vec![255, 0, 0, 0],
});

/// The 1x1 stand-in bound when a scene has no baked terrain occlusion.
///
/// WHITE, and unlike `NO_SPLAT` its contents ARE read: sky visibility is
/// multiplied into the ambient term, so 1.0 means "sees the whole sky" and
/// reproduces exactly the shading terrain had before this map existed. Black
/// here would put every terrain in the game in permanent night.
static FULL_SKY: std::sync::LazyLock<TerrainImage> = std::sync::LazyLock::new(|| TerrainImage {
    width: 1,
    height: 1,
    rgba: vec![255, 255, 255, 255],
});

#[cfg(test)]
mod authored_splat_tests {
    use super::tests::*;
    use super::*;

    fn splat(rgba: [u8; 4]) -> TerrainImage {
        TerrainImage { width: 1, height: 1, rgba: rgba.to_vec() }
    }

    const FLAT: [f32; 3] = [0.0, 1.0, 0.0];
    const CLIFF: [f32; 3] = [1.0, 0.05, 0.0];

    /// The point of the whole feature: what an author painted wins over what
    /// the slope rule would have chosen. Flat ground takes layer 0 (red) by
    /// slope, so a map demanding layer 1 must come back blue.
    #[test]
    fn authored_weights_override_the_slope_blend() {
        let Some(unauthored) = render_quad_with_normal(FLAT) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        assert!(unauthored[0] > unauthored[2], "flat ground should be red without a map: {unauthored:?}");

        let painted = render_quad_with_splat(FLAT, &splat([0, 255, 0, 0])).unwrap();
        assert!(
            painted[2] > painted[0],
            "a map demanding layer 1 must beat the slope rule on flat ground: {painted:?}",
        );
    }

    /// And the other direction, so the test cannot pass by the map simply
    /// being ignored in one particular case.
    #[test]
    fn authored_weights_override_a_cliff_too() {
        let Some(unauthored) = render_quad_with_normal(CLIFF) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        assert!(unauthored[2] > unauthored[0], "a cliff should be blue without a map: {unauthored:?}");

        let painted = render_quad_with_splat(CLIFF, &splat([255, 0, 0, 0])).unwrap();
        assert!(
            painted[0] > painted[2],
            "a map demanding layer 0 must beat the slope rule on a cliff: {painted:?}",
        );
    }

    /// A blend, not a winner-takes-all. Half layer 0 (red) and half layer 2
    /// (green) must show both channels -- if the shader picked a dominant layer
    /// instead of summing weights, one of these would be zero.
    #[test]
    fn a_mixed_texel_blends_rather_than_picking_a_winner() {
        let Some(mixed) = render_quad_with_splat(FLAT, &splat([128, 0, 128, 0])) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        assert!(mixed[0] > 30, "layer 0 (red) should contribute: {mixed:?}");
        assert!(mixed[1] > 30, "layer 2 (green) should contribute: {mixed:?}");
        assert!(mixed[2] < mixed[0], "layer 1 (blue) was not painted: {mixed:?}");
    }

    /// Weights that do not sum to full are normalised rather than trusted. The
    /// editor keeps them summing to 255, but a hand-made or half-written file
    /// has no such guarantee, and unnormalised weights would blow out or black
    /// out the ground instead of merely looking slightly wrong.
    #[test]
    fn unnormalised_weights_are_normalised_not_amplified() {
        // Both of these mean "half layer 0, half layer 2" once normalised, but
        // one sums to 255 and the other to 510.
        let Some(normal_sum) = render_quad_with_splat(FLAT, &splat([128, 0, 128, 0])) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let double_sum = render_quad_with_splat(FLAT, &splat([255, 0, 255, 0])).unwrap();

        for channel in 0..3 {
            let delta = normal_sum[channel].abs_diff(double_sum[channel]);
            assert!(
                delta <= 4,
                "weights summing to 510 must shade like weights summing to 255 \
                 (channel {channel}: {normal_sum:?} vs {double_sum:?})",
            );
        }
    }

    /// An all-zero texel cannot divide by zero and black out the ground.
    #[test]
    fn an_empty_texel_does_not_produce_a_divide_by_zero() {
        let Some(px) = render_quad_with_splat(FLAT, &splat([0, 0, 0, 0])) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        assert_eq!(px[3], 255, "terrain must stay opaque: {px:?}");
        for channel in 0..3 {
            assert!(px[channel] < 250, "an empty texel must not blow out: {px:?}");
        }
    }
}

impl TerrainImage {
    /// Decode an image file into the RGBA8 a terrain layer needs.
    ///
    /// Returns `None` rather than failing the frame when the file is missing or
    /// unreadable: terrain that renders in flat fallback colours is diagnosable
    /// from the log, and a client that refuses to start because one texture is
    /// absent is not. That is the same call `terrain_render::load` makes about
    /// missing ground.
    pub fn load(path: &std::path::Path) -> Option<Self> {
        let decoded = match ::image::open(path) {
            Ok(d) => d.to_rgba8(),
            Err(e) => {
                log::warn!("terrain texture {}: {e}", path.display());
                return None;
            }
        };
        Some(Self {
            width: decoded.width(),
            height: decoded.height(),
            rgba: decoded.into_raw(),
        })
    }
}

/// The four layer textures a terrain material wants, by role.
///
/// Named by ROLE rather than by what they depict, matching the files on disk,
/// so replacing what "rock" looks like is a file swap and touches no code. The
/// order is the shader's fixed slot order and is not rearrangeable.
pub const TERRAIN_LAYER_FILES: [&str; 4] = ["ground.jpg", "rock.jpg", "high.jpg", "sediment.jpg"];

/// Normal maps for the same four layers, in the same order.
///
/// Suffixed rather than kept in a subdirectory so a layer's files sort together
/// and it is obvious at a glance which layers have normals and which do not.
pub const TERRAIN_NORMAL_FILES: [&str; 4] =
    ["ground_n.jpg", "rock_n.jpg", "high_n.jpg", "sediment_n.jpg"];

/// Roughness maps for the same four layers, in the same order.
///
/// Ground is not one gloss. Wet rock, dry gravel and grass differ, and a single
/// constant made every layer the same material under a different picture --
/// which is what a terrain looks like next to a brush surface that has a map.
pub const TERRAIN_ROUGH_FILES: [&str; 4] =
    ["ground_r.jpg", "rock_r.jpg", "high_r.jpg", "sediment_r.jpg"];

/// Ambient occlusion for the same four layers, in the same order.
pub const TERRAIN_AO_FILES: [&str; 4] =
    ["ground_ao.jpg", "rock_ao.jpg", "high_ao.jpg", "sediment_ao.jpg"];

/// Load the terrain layer set from a directory, falling back per layer.
///
/// Per LAYER, not all-or-nothing: a project part-way through authoring its
/// materials should see the layers it has, not lose all four because one is
/// missing. Each gap keeps that layer's flat colour, which is exactly what the
/// whole terrain looked like before any textures existed.
///
/// Sizes are normalised to the first successfully loaded layer, because they
/// share one D2Array and a mismatched layer would otherwise fail validation.
pub fn load_terrain_layers(dir: &std::path::Path) -> Vec<Option<TerrainImage>> {
    TERRAIN_LAYER_FILES
        .iter()
        .map(|name| TerrainImage::load(&dir.join(name)))
        .collect()
}

/// Load the four roughness maps, per layer, falling back to none.
///
/// A missing map is ordinary and not an error: that layer takes
/// `DEFAULT_TERRAIN_ROUGHNESS`, which is what every layer used before these
/// existed, so a project that has not authored them looks exactly as it did.
pub fn load_terrain_rough(dir: &std::path::Path) -> Vec<Option<TerrainImage>> {
    TERRAIN_ROUGH_FILES
        .iter()
        .map(|name| TerrainImage::load(&dir.join(name)))
        .collect()
}

/// Load the four occlusion maps, per layer, falling back to none.
///
/// A missing map means fully unoccluded -- white -- which is the neutral value
/// for something that MULTIPLIES, and is how terrain behaved before.
pub fn load_terrain_ao(dir: &std::path::Path) -> Vec<Option<TerrainImage>> {
    TERRAIN_AO_FILES
        .iter()
        .map(|name| TerrainImage::load(&dir.join(name)))
        .collect()
}

/// The project's terrain material settings, from `settings.json` beside the
/// layer textures.
///
/// WHY A FILE RATHER THAN A CONSTANT
///
/// `repeat` -- metres per tile, per layer -- is the control that decides whether
/// ground reads as gravel or as noise, and it is judged by eye against the art
/// that is actually installed. Baking it into the binary means the person who
/// can see the problem cannot fix it, and the person who can fix it has to
/// rebuild the engine to try a number.
///
/// Beside the textures, not in the scene: which four materials a project uses
/// and how big they tile is one art decision for the whole game, exactly like
/// the layer files themselves. Copying it into every scene would mean changing
/// "how big is our gravel" in twenty places.
///
/// ABSENT IS THE NORMAL CASE. Every project that predates this file has none,
/// and every key is independent, so a file naming only `repeat` leaves the rest
/// at their defaults rather than zeroing them. A malformed file logs and yields
/// the defaults: unreadable settings must not cost a project its terrain.
pub fn load_terrain_settings(dir: &std::path::Path) -> TerrainMaterialUniform {
    let mut out = TerrainMaterialUniform::default();
    let path = dir.join(TERRAIN_SETTINGS_FILE);
    let text = match std::fs::read_to_string(&path) {
        Ok(t) => t,
        Err(_) => return out,
    };
    let raw: serde_json::Value = match serde_json::from_str(&text) {
        Ok(v) => v,
        Err(e) => {
            log::warn!("{}: {e} -- using default terrain settings", path.display());
            return out;
        }
    };
    if let Some(values) = raw.get("repeat").and_then(|r| r.as_array()) {
        for (i, slot) in out.repeat.iter_mut().enumerate() {
            if let Some(v) = values.get(i).and_then(|v| v.as_f64()) {
                // Clamped, not rejected: a zero here divides by zero in the
                // shader, and one bad number should cost that layer its scale
                // rather than the project its ground.
                *slot = (v as f32).max(MIN_REPEAT);
            }
        }
    }
    if let Some(v) = raw.get("normal_strength").and_then(|v| v.as_f64()) {
        out.normal_strength = (v as f32).clamp(0.0, 4.0);
    }
    out
}

/// Per-project terrain material settings, beside the layer textures.
pub const TERRAIN_SETTINGS_FILE: &str = "settings.json";

/// Smallest tile size in metres. Below this a tile is smaller than a texel of
/// anything and the ground reads as noise; at zero the shader divides by zero.
pub const MIN_REPEAT: f32 = 0.05;

/// Load the layer normal maps, per layer, from the same directory.
///
/// Missing is the NORMAL case rather than an error: a project with colour and
/// no normals is exactly what shipped before this existed, and each gap becomes
/// a flat map that perturbs nothing. `TerrainImage::load` already logs what it
/// could not read, so a typo is visible without failing the frame.
pub fn load_terrain_normals(dir: &std::path::Path) -> Vec<Option<TerrainImage>> {
    TERRAIN_NORMAL_FILES
        .iter()
        .map(|name| TerrainImage::load(&dir.join(name)))
        .collect()
}

/// Flat colours a layer falls back to when it has no texture.
///
/// The palette the terrain used before any textures existed, so a project with
/// no material art renders exactly as it always did rather than as black.
pub const FALLBACK_LAYER_COLOURS: [[u8; 3]; 4] = [
    [86, 112, 62],   // 0 ground
    [104, 100, 94],  // 1 rock
    [132, 128, 120], // 2 high ground
    [92, 84, 70],    // 3 sediment
];

pub(crate) fn solid_image(rgb: [u8; 3], width: u32, height: u32) -> TerrainImage {
    let mut rgba = Vec::with_capacity((width * height * 4) as usize);
    for _ in 0..(width * height) {
        rgba.extend_from_slice(&[rgb[0], rgb[1], rgb[2], 255]);
    }
    TerrainImage { width, height, rgba }
}

/// Nearest-neighbour resample onto a different size.
///
/// Only ever runs on a mismatched layer set, which is an authoring mistake
/// rather than a shipping configuration -- so this exists to keep such a
/// project RUNNING and legible, not to look good. Anything better would be
/// effort spent on a case the SOURCES.md tells authors to avoid.
pub(crate) fn resample(src: &TerrainImage, width: u32, height: u32) -> TerrainImage {
    let mut rgba = Vec::with_capacity((width * height * 4) as usize);
    for y in 0..height {
        let sy = (y as u64 * src.height as u64 / height.max(1) as u64) as u32;
        for x in 0..width {
            let sx = (x as u64 * src.width as u64 / width.max(1) as u64) as u32;
            let at = ((sy.min(src.height - 1) * src.width + sx.min(src.width - 1)) * 4) as usize;
            rgba.extend_from_slice(&src.rgba[at..at + 4]);
        }
    }
    TerrainImage { width, height, rgba }
}

#[cfg(test)]
mod layer_loading_tests {
    use super::tests::headless_gpu;
    use super::*;

    fn img(rgb: [u8; 3], w: u32, h: u32) -> TerrainImage {
        solid_image(rgb, w, h)
    }

    /// A project part-way through authoring its materials must see the layers
    /// it HAS, not lose all four because one file is missing.
    #[test]
    fn a_missing_layer_falls_back_without_taking_the_others_with_it() {
        let Some((device, queue)) = headless_gpu() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let layout = material_bind_group_layout(&device);

        let err_scope_2 = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let _m = TerrainMaterial::from_layers(
            &device,
            &queue,
            &layout,
            &[Some(img([10, 20, 30], 64, 64)), None, Some(img([1, 2, 3], 64, 64)), None],
            &[None, None, None, None],
            None,
            None,
        );
        assert!(
            pollster::block_on(err_scope_2.pop()).is_none(),
            "a partial layer set must still bind validly",
        );
    }

    /// The four share one D2Array, so a mismatched layer is a validation
    /// failure rather than a smaller texture -- and it would only appear on the
    /// project that happens to have mixed sizes.
    #[test]
    fn mixed_sizes_are_normalised_rather_than_failing_validation() {
        let Some((device, queue)) = headless_gpu() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let layout = material_bind_group_layout(&device);

        let err_scope_3 = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let _m = TerrainMaterial::from_layers(
            &device,
            &queue,
            &layout,
            &[
                Some(img([10, 20, 30], 64, 64)),
                Some(img([40, 50, 60], 16, 16)),
                None,
                Some(img([70, 80, 90], 128, 32)),
            ],
            &[None, None, None, None],
            None,
            None,
        );
        assert!(
            pollster::block_on(err_scope_3.pop()).is_none(),
            "mixed layer sizes must be normalised, not rejected",
        );
    }

    /// With nothing loaded the material must be exactly what it was before
    /// textures existed, so a project with no art is unchanged.
    #[test]
    fn no_layers_at_all_is_the_old_flat_palette() {
        let Some((device, queue)) = headless_gpu() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let layout = material_bind_group_layout(&device);
        let err_scope_4 = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let _m = TerrainMaterial::from_layers(
            &device, &queue, &layout, &[None, None, None, None], &[None, None, None, None], None, None,
        );
        assert!(pollster::block_on(err_scope_4.pop()).is_none());
    }

    #[test]
    fn a_missing_file_reports_none_rather_than_failing() {
        assert!(TerrainImage::load(std::path::Path::new("/nonexistent/ground.jpg")).is_none());
    }

    /// Role-named files, in the shader's fixed slot order.
    #[test]
    fn the_layer_file_names_match_the_shader_slots() {
        assert_eq!(TERRAIN_LAYER_FILES.len(), FALLBACK_LAYER_COLOURS.len());
        assert_eq!(TERRAIN_LAYER_FILES[0], "ground.jpg");
        assert_eq!(TERRAIN_LAYER_FILES[1], "rock.jpg");
    }

    /// A scratch directory holding exactly the `settings.json` given.
    fn settings_dir(label: &str, body: Option<&str>) -> std::path::PathBuf {
        let dir = std::env::temp_dir()
            .join(format!("terrain_settings_{label}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        if let Some(text) = body {
            std::fs::write(dir.join(TERRAIN_SETTINGS_FILE), text).unwrap();
        }
        dir
    }

    /// No file is the state every project was in before this existed, and it
    /// must render exactly as it did then.
    #[test]
    fn absent_settings_are_the_built_in_defaults() {
        let dir = settings_dir("absent", None);
        let loaded = load_terrain_settings(&dir);
        assert_eq!(loaded.repeat, TerrainMaterialUniform::default().repeat);
        assert_eq!(loaded.normal_strength, 1.0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The point of the file: the editor's tile sizes reach the headset.
    #[test]
    fn repeat_comes_from_the_file() {
        let dir = settings_dir("repeat", Some(r#"{"repeat": [2.0, 3.5, 12.0, 6.0]}"#));
        assert_eq!(load_terrain_settings(&dir).repeat, [2.0, 3.5, 12.0, 6.0]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Every key independent. A file that sets only `repeat` must not zero the
    /// normal maps -- which is what a `serde` struct with `Default::default()`
    /// per FIELD would do only if every field carried its own default, and what
    /// a plain deserialize would get wrong silently.
    #[test]
    fn an_unmentioned_key_keeps_its_default() {
        let dir = settings_dir("partial", Some(r#"{"repeat": [1.0, 1.0, 1.0, 1.0]}"#));
        let loaded = load_terrain_settings(&dir);
        assert_eq!(loaded.repeat, [1.0; 4]);
        assert_eq!(loaded.normal_strength, 1.0);
        assert_eq!(loaded.slope_start_deg, TerrainMaterialUniform::default().slope_start_deg);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A short array is a hand-edit, not a reason to lose the other layers.
    #[test]
    fn a_short_repeat_array_only_sets_what_it_names() {
        let dir = settings_dir("short", Some(r#"{"repeat": [2.0]}"#));
        let loaded = load_terrain_settings(&dir);
        let default = TerrainMaterialUniform::default().repeat;
        assert_eq!(loaded.repeat, [2.0, default[1], default[2], default[3]]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Zero would divide by zero in the shader. Clamped rather than rejected:
    /// one bad number costs that layer its scale, not the project its ground.
    #[test]
    fn a_zero_tile_size_is_clamped_not_taken() {
        let dir = settings_dir("zero", Some(r#"{"repeat": [0.0, -4.0, 10.0, 6.0]}"#));
        let loaded = load_terrain_settings(&dir);
        assert_eq!(loaded.repeat[0], MIN_REPEAT);
        assert_eq!(loaded.repeat[1], MIN_REPEAT);
        assert_eq!(loaded.repeat[2], 10.0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Unreadable settings must not cost a project its terrain.
    #[test]
    fn malformed_settings_fall_back_rather_than_failing() {
        let dir = settings_dir("broken", Some("{ this is not json"));
        assert_eq!(
            load_terrain_settings(&dir).repeat,
            TerrainMaterialUniform::default().repeat,
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn resampling_preserves_the_requested_size() {
        let out = resample(&img([9, 9, 9], 7, 3), 16, 16);
        assert_eq!((out.width, out.height), (16, 16));
        assert_eq!(out.rgba.len(), 16 * 16 * 4);
    }
}

/// The texel a normal map uses for "no perturbation".
///
/// (128, 128, 255) unpacks to (0, 0, 1) in tangent space -- straight out of the
/// surface. That is what lets an unauthored normal set be a true no-op and is
/// why the material needs no "has normals" flag, unlike the splat map, where
/// an all-zero texel is a legitimate authored value.
pub const FLAT_NORMAL: [u8; 3] = [128, 128, 255];

impl TerrainImage {
    fn clone_image(&self) -> TerrainImage {
        TerrainImage { width: self.width, height: self.height, rgba: self.rgba.clone() }
    }
}

#[cfg(test)]
mod normal_map_tests {
    use super::tests::{render_quad_full, Palette};
    use super::*;

    const FLAT: [f32; 3] = [0.0, 1.0, 0.0];

    fn normal_map(rgb: [u8; 3]) -> TerrainImage {
        solid_image(rgb, 1, 1)
    }

    /// Only layer 0 has weight, so only its normal map matters.
    fn only_layer0(map: TerrainImage) -> [Option<TerrainImage>; 4] {
        [Some(map), None, None, None]
    }

    fn brightness(px: [u8; 4]) -> u32 {
        px[0] as u32 + px[1] as u32 + px[2] as u32
    }

    /// The whole point: a normal map must change how the surface lights.
    ///
    /// Needs a real light. With the ambient-only harness the other tests use,
    /// `shade` returns a constant whatever the normal is, so this would pass
    /// with the perturbation entirely disconnected.
    #[test]
    fn a_tilted_normal_changes_the_lighting() {
        let Some(flat) = render_quad_full(FLAT, Palette::Test, None, None, &only_layer0(normal_map(FLAT_NORMAL)), true)
        else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        // Tangent normal tilted hard toward +x, where the test light sits.
        let toward = render_quad_full(FLAT, Palette::Test, None, None, &only_layer0(normal_map([230, 128, 160])), true).unwrap();
        // ...and hard away from it.
        let away = render_quad_full(FLAT, Palette::Test, None, None, &only_layer0(normal_map([25, 128, 160])), true).unwrap();

        assert_ne!(brightness(toward), brightness(flat), "a tilted normal must change shading: {toward:?} vs {flat:?}");
        assert!(
            brightness(toward) > brightness(away),
            "tilting toward the light must be brighter than tilting away: {toward:?} vs {away:?}",
        );
    }

    /// GREEN FACES THE TOP OF THE PICTURE, which on the ground is -z.
    ///
    /// Ground is textured top-down at uv = world.xz, and wgpu reads v = 0 at
    /// the picture's top row, so the top of the picture lies toward -z. An
    /// OpenGL map's green above 128 says "faces the top of the picture"; a lamp
    /// out toward -z must therefore light that texel more than one facing the
    /// bottom. The red-only test above cannot see this: red means the same in
    /// both conventions. Until 2026-09-28 the whiteout added green along +v and
    /// lit every bump upside down in z.
    #[test]
    fn green_tilts_the_ground_toward_the_top_of_its_picture() {
        let north = glam::Vec3::new(0.4, 2.0, -3.0);
        let at = |green: u8| {
            super::tests::render_quad_light_at(
                FLAT, Palette::Test, None, None, &only_layer0(normal_map([128, green, 180])), Some(None), north,
            )
        };
        let Some(faces_top) = at(230) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let faces_bottom = at(26).unwrap();
        assert!(
            brightness(faces_top) > brightness(faces_bottom) + 6,
            "a texel facing the top of the picture (-z) must be lit more by a lamp toward -z: \
             {faces_top:?} vs {faces_bottom:?}",
        );
    }

    /// A flat normal map must be indistinguishable from none at all -- that is
    /// what lets an unauthored normal set cost nothing and need no flag.
    #[test]
    fn a_flat_normal_map_is_a_true_no_op() {
        let Some(without) = render_quad_full(FLAT, Palette::Test, None, None, &[None, None, None, None], true)
        else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let with_flat = render_quad_full(FLAT, Palette::Test, None, None, &only_layer0(normal_map(FLAT_NORMAL)), true).unwrap();
        assert_eq!(with_flat, without, "a flat normal map must shade identically to none");
    }

    /// Strength 0 disables perturbation without needing a second code path.
    #[test]
    fn zero_strength_disables_the_maps() {
        let Some((device, queue)) = super::tests::headless_gpu() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let layout = material_bind_group_layout(&device);
        let mut settings = TerrainMaterialUniform::default();
        settings.normal_strength = 0.0;

        let err_scope_5 = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let _m = TerrainMaterial::new(
            &device, &queue, &layout,
            &[solid_image([200, 200, 200], 4, 4)],
            &solid_image([128, 128, 128], 1, 1),
            None,
            None,
            &only_layer0(solid_image([230, 128, 160], 4, 4)),
            &[],
            &[],
            settings,
        );
        assert!(pollster::block_on(err_scope_5.pop()).is_none());
    }

    /// Sized to the colour layers, so a normal map at a different resolution
    /// still binds rather than failing validation on whichever project has one.
    #[test]
    fn a_normal_map_of_a_different_size_is_resampled() {
        let Some((device, queue)) = super::tests::headless_gpu() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let layout = material_bind_group_layout(&device);
        let err_scope_6 = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let _m = TerrainMaterial::from_layers(
            &device, &queue, &layout,
            &[Some(solid_image([10, 20, 30], 64, 64)), None, None, None],
            &[Some(solid_image(FLAT_NORMAL, 16, 16)), None, None, None],
            None,
            None,
        );
        assert!(
            pollster::block_on(err_scope_6.pop()).is_none(),
            "a mismatched normal map must be resampled, not rejected",
        );
    }

    #[test]
    fn the_flat_normal_texel_unpacks_to_straight_out() {
        // (128,128,255) / 255 * 2 - 1 ~= (0, 0, 1).
        let unpack = |v: u8| (v as f32 / 255.0) * 2.0 - 1.0;
        assert!(unpack(FLAT_NORMAL[0]).abs() < 0.01);
        assert!(unpack(FLAT_NORMAL[1]).abs() < 0.01);
        assert!((unpack(FLAT_NORMAL[2]) - 1.0).abs() < 0.01);
    }
}

// Scene shader sources, for `multiview::every_scene_shader_survives_the_multiview_transform`.
// Test-only: the gate has to see exactly the text each pipeline is built from,
// and nothing on a development machine can build a multiview pipeline to check.
#[cfg(test)]
pub fn terrain_shader_src() -> String {
    terrain_shader()
}

#[cfg(test)]
mod terrain_material_parity_tests {
    /// Terrain shades through the SAME path as brushes, with the albedo passed
    /// IN rather than multiplied over the answer.
    ///
    /// The old form -- `shade_with_sky(...)` then `albedo * col * lit` -- is the
    /// exact bug this renderer already fixed for brushes: a dielectric's
    /// highlight is not tinted by its diffuse colour, and multiplying a
    /// low-albedo ground over the lit result made every highlight on it far too
    /// dim. Ground had no Fresnel, no energy conservation, no probe reflection
    /// and no specular occlusion, and it read as a second-class material next
    /// to marble.
    #[test]
    fn terrain_uses_the_material_path_with_albedo_passed_in() {
        let src = super::terrain_shader_src();
        assert!(
            src.contains("let lit = shade_material_env("),
            "terrain no longer shades through the material path",
        );
        assert!(
            !src.contains("shade_with_sky(in.world_pos"),
            "terrain went back to the sky-only path",
        );
        assert!(
            src.contains("tonemap(in.col.rgb * lit)"),
            "the albedo is being multiplied over the lit result again; it must \
             go INTO the shading so highlights are not tinted by it",
        );
        assert!(
            !src.contains("tonemap(albedo * in.col.rgb * lit)"),
            "the old multiply-over form is back",
        );
    }

    /// The normal's lost variation goes into ROUGHNESS, the same place the
    /// brush bake puts it -- not into a bespoke specular-exponent tweak.
    #[test]
    fn the_normal_variance_lands_in_roughness() {
        let src = super::terrain_shader_src();
        // The normal's lost variation now lives in the BAKED roughness mips --
        // `roughness_chain_with_normal_variance` folds it in per level at load,
        // exactly as brush materials get it. It must NOT also be added at
        // runtime, or the same variance is counted twice and distant ground
        // goes flat.
        assert!(
            src.contains("layer_rough_at(0, f)"),
            "terrain no longer samples a roughness map",
        );
        assert!(
            src.contains("layer_ao_at(0, f)"),
            "terrain no longer samples an occlusion map",
        );
        assert!(
            !src.contains("min(2.0 * sigma2"),
            "the runtime variance term is back alongside the baked mips; that \
             counts the same measurement twice",
        );
        // What the bake cannot see -- geometric normal variation and the
        // grazing-angle anisotropy gap -- is still handled at runtime.
        assert!(
            src.contains("specular_aa_roughness(clamp(rough_map, 0.0, 1.0), dpdx(shaded_n), dpdy(shaded_n))"),
            "the geometric specular-AA term is gone; the bake alone cannot see \
             curvature or the anisotropy gap",
        );
    }
}

#[cfg(test)]
mod terrain_mip_tests {
    /// EVERY TERRAIN ARRAY THAT IS SAMPLED BY A SHRINKING FOOTPRINT NEEDS MIPS.
    ///
    /// The normal array shipped with `mip_level_count: 1` while the shared
    /// sampler asked for `mipmap_filter: Linear` and 8x anisotropy -- both
    /// silent no-ops against one level. Distant pixels point-sampled a
    /// full-resolution normal map, so the shading normal changed randomly
    /// pixel to pixel and head tracking moved it every frame. Brush normals
    /// were mipped all along, which is why only terrain shimmered.
    ///
    /// Asserted on the SOURCE because the texture is built on a device this
    /// test does not have. Crude, and it would have caught the bug.
    #[test]
    fn the_normal_array_is_created_with_a_mip_chain() {
        let src = include_str!("terrain_pipeline.rs");
        let i = src.find(r#"label: Some("terrain_normals")"#).expect("normal array is gone");
        let window = &src[i..i + 400];
        assert!(
            window.contains("mip_level_count,"),
            "terrain_normals is back to a single mip level; distant ground will \
             point-sample full-resolution normals and shimmer",
        );
        assert!(
            !window.contains("mip_level_count: 1"),
            "terrain_normals explicitly requests one mip level",
        );
    }

    /// Roughness and occlusion need mips for the same reason the normal does,
    /// and must be LINEAR for the same reason: they are measurements.
    #[test]
    fn the_data_arrays_are_mipped_and_linear() {
        let src = include_str!("terrain_pipeline.rs");
        let i = src.find("let data_array = |label: &str|").expect("data arrays are gone");
        let window = &src[i..i + 600];
        assert!(
            window.contains("mip_level_count,") && !window.contains("mip_level_count: 1"),
            "the roughness/occlusion arrays lost their mip chain",
        );
        assert!(
            window.contains("TextureFormat::Rgba8Unorm") && !window.contains("Rgba8UnormSrgb"),
            "a measurement array is being created as sRGB",
        );
        // The roughness chain must be the variance-baking one, not a plain
        // box filter -- that is what carries the normal's lost detail.
        assert!(
            src.contains("roughness_chain_with_normal_variance(")
                && src.contains("mip_chain_linear(&ao_arr[slot].rgba"),
            "roughness is no longer baked with the normal's variance, or AO is \
             no longer a plain linear chain",
        );
    }

    /// Every layer ships the full material set, so terrain is not a
    /// second-class surface next to a brush.
    #[test]
    fn every_layer_has_a_roughness_and_occlusion_file_name() {
        assert_eq!(super::TERRAIN_ROUGH_FILES.len(), super::TERRAIN_LAYER_FILES.len());
        assert_eq!(super::TERRAIN_AO_FILES.len(), super::TERRAIN_LAYER_FILES.len());
        for (i, colour) in super::TERRAIN_LAYER_FILES.iter().enumerate() {
            let stem = colour.trim_end_matches(".jpg");
            assert_eq!(super::TERRAIN_ROUGH_FILES[i], format!("{stem}_r.jpg"));
            assert_eq!(super::TERRAIN_AO_FILES[i], format!("{stem}_ao.jpg"));
        }
    }

    /// A normal map mipped through the sRGB curve tilts every bump toward the
    /// surface. The chain for it must be the linear one.
    #[test]
    fn the_normal_chain_is_built_linear() {
        let src = include_str!("terrain_pipeline.rs");
        let i = src.find(r#"label: Some("terrain_normals")"#).expect("normal array is gone");
        let window = &src[i..i + 1600];
        assert!(
            window.contains("mip_chain_linear(&src.rgba, w, h, mip_level_count)"),
            "the normal mip chain is not the linear one",
        );
    }

    /// The linear chain must not apply any transfer curve: a mid-grey in is a
    /// mid-grey out at every level.
    #[test]
    fn the_linear_chain_preserves_a_flat_value() {
        let px = vec![128u8; 8 * 8 * 4];
        let chain = super::mip_chain_linear(&px, 8, 8, super::mip_levels_for(8, 8));
        assert!(chain.len() > 1, "no chain was built");
        for (data, w, h) in &chain {
            assert_eq!(data.len(), (w * h * 4) as usize);
            for b in data {
                assert!(
                    (*b as i32 - 128).abs() <= 1,
                    "a flat 128 became {b}; the linear chain is bending values",
                );
            }
        }
    }
}
