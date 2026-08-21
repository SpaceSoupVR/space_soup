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
use super::lights::wgsl_lights_block;

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
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("terrain_shader"),
            source: ShaderSource::Wgsl(terrain_shader().into()),
        });
        let material_layout = material_bind_group_layout(device);
        let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("terrain_layout"),
            bind_group_layouts: &[uniform_layout, &material_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("terrain_pipeline"),
            layout: Some(&layout),
            vertex: VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: PipelineCompilationOptions::default(),
                // Same vertex format as the solid pass, so the geometry path is
                // untouched: terrain still arrives as SolidVertex.
                buffers: &[SolidVertex::layout()],
            },
            fragment: Some(FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: PipelineCompilationOptions::default(),
                targets: &[Some(ColorTargetState {
                    format,
                    blend: Some(BlendState::ALPHA_BLENDING),
                    write_mask: ColorWrites::ALL,
                })],
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
                depth_write_enabled: true,
                depth_compare: CompareFunction::Less,
                stencil: StencilState::default(),
                bias: DepthBiasState::default(),
            }),
            multisample: MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        Self { pipeline, material_layout }
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
struct Uniforms {{ view_proj: mat4x4<f32> }}
@group(0) @binding(0) var<uniform> u: Uniforms;

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

{lights_block}

struct VIn  {{ @location(0) pos: vec3<f32>, @location(1) norm: vec3<f32>, @location(2) col: vec4<f32>, @location(3) uv2: vec2<f32> }}
struct VOut {{
    @builtin(position) clip: vec4<f32>,
    @location(0) col: vec4<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) world_pos: vec3<f32>,
    // Normalised position over the terrain footprint, carried in the slot the
    // cuboid path uses for lightmap coordinates. Terrain is lit dynamically and
    // has no lightmap, so that slot was sitting at (0,0) doing nothing.
    @location(3) uv: vec2<f32>,
}}

@vertex fn vs_main(v: VIn) -> VOut {{
    var out: VOut;
    out.clip      = u.view_proj * vec4<f32>(v.pos, 1.0);
    out.col       = v.col;
    out.normal    = v.norm;
    out.world_pos = v.pos;
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
fn sample_biplanar(layer: i32, world: vec3<f32>, n: vec3<f32>, repeat: f32) -> vec3<f32> {{
    let r = max(repeat, 0.001);
    let a = abs(n);

    // Rank the axes so we can take the top two without branching per pixel.
    let ma = max(a.x, max(a.y, a.z));
    let mi = min(a.x, min(a.y, a.z));
    let me = a.x + a.y + a.z - ma - mi;

    var uv_major: vec2<f32>;
    var uv_minor: vec2<f32>;
    if (a.y == ma) {{
        uv_major = world.xz / r;
        uv_minor = select(world.xy / r, world.zy / r, a.x >= a.z);
    }} else if (a.x == ma) {{
        uv_major = world.zy / r;
        uv_minor = select(world.xy / r, world.xz / r, a.z < a.y);
    }} else {{
        uv_major = world.xy / r;
        uv_minor = select(world.zy / r, world.xz / r, a.x < a.y);
    }}

    let c_major = textureSample(layer_tex, layer_samp, uv_major, layer).rgb;
    let c_minor = textureSample(layer_tex, layer_samp, uv_minor, layer).rgb;
    let w = ma / max(ma + me, 0.001);
    return mix(c_minor, c_major, w);
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

    // Y-projection whiteout. For flat ground (n = 0,1,0) and a flat texel
    // (tn = 0,0,1) this returns the geometric normal exactly, which is what
    // makes an unauthored normal set a true no-op.
    let t = vec3<f32>(tn.xy + n.xz, abs(tn.z) * n.y);
    return normalize(t.xzy);
}}

fn layer_colour(layer: i32, world: vec3<f32>, n: vec3<f32>, slope_deg: f32) -> vec3<f32> {{
    let repeat = mat.repeat[layer];
    if (slope_deg < mat.biplanar_start_deg) {{
        return sample_planar(layer, world, repeat);
    }}
    return sample_biplanar(layer, world, n, repeat);
}}

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
    let n = normalize(in.normal);

    // Slope straight from the normal: no derivative, no extra sampling.
    let slope_deg = degrees(acos(clamp(n.y, -1.0, 1.0)));

    let w = layer_weights(in.uv, in.world_pos.y, slope_deg);

    // All four layers, unconditionally. A weight-zero layer still costs its
    // samples, which is the price of keeping every textureSample in uniform
    // control flow -- skipping them per fragment is exactly the non-uniform
    // branch WGSL forbids around sampling.
    var albedo = layer_colour(0, in.world_pos, n, slope_deg) * w.x;
    albedo = albedo + layer_colour(1, in.world_pos, n, slope_deg) * w.y;
    albedo = albedo + layer_colour(2, in.world_pos, n, slope_deg) * w.z;
    albedo = albedo + layer_colour(3, in.world_pos, n, slope_deg) * w.w;

    // Blend the layers' normals by the same weights, then renormalise. Summing
    // unit vectors shortens the result wherever they disagree, and a shortened
    // normal darkens the surface -- so the renormalise is load-bearing, not
    // tidiness.
    var shaded_n = layer_normal(0, in.world_pos, n, mat.repeat[0]) * w.x;
    shaded_n = shaded_n + layer_normal(1, in.world_pos, n, mat.repeat[1]) * w.y;
    shaded_n = shaded_n + layer_normal(2, in.world_pos, n, mat.repeat[2]) * w.z;
    shaded_n = shaded_n + layer_normal(3, in.world_pos, n, mat.repeat[3]) * w.w;
    shaded_n = normalize(select(n, shaded_n, length(shaded_n) > 0.0001));

    // Macro variation: one low-frequency sample, centred on 1 so it darkens and
    // lightens rather than only darkening.
    let m = textureSample(macro_tex, layer_samp, in.world_pos.xz / max(mat.macro_repeat, 0.001)).r;
    albedo = albedo * (1.0 + (m - 0.5) * 2.0 * mat.macro_strength);

    // Vertex colour survives as a tint, so the editor can still mark up ground
    // per-vertex without a second pipeline.
    let lit = shade(in.world_pos, shaded_n);
    return vec4<f32>(albedo * in.col.rgb * lit, 1.0);
}}
"#,
        lights_block = wgsl_lights_block(0, 1)
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::renderer::cuboid::SolidVertex;
    use crate::renderer::lights::LightsUniform;
    use crate::renderer::uniforms::UniformBuffer;
    use wgpu::util::DeviceExt;

    /// The pipeline is only validated when it is CREATED, and correctness only
    /// when something is drawn. A test that builds a pipeline proves the WGSL
    /// parses; it says nothing about whether the shading responds to slope.
    /// So this renders and reads the pixels back.
    pub fn headless_gpu() -> Option<(Device, Queue)> {
        let instance = Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&RequestAdapterOptions {
            power_preference: PowerPreference::default(),
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok()?;
        pollster::block_on(adapter.request_device(&DeviceDescriptor {
            required_features: Features::empty(),
            required_limits: Limits::default(),
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
        render_quad(normal, Palette::Test, None)
    }

    /// Same geometry, with authored weights bound.
    pub fn render_quad_with_splat(normal: [f32; 3], splat: &TerrainImage) -> Option<[u8; 4]> {
        render_quad(normal, Palette::Test, Some(splat))
    }

    /// Same geometry, but bound through the shipping `TerrainMaterial::fallback`
    /// rather than a bind group assembled here. Worth its own path because the
    /// hand-built test material proves the SHADER and proves nothing about the
    /// code the renderer will actually call -- layer padding, the D2Array view
    /// dimension and the non-sRGB macro format all live in `TerrainMaterial`
    /// and are exactly where a binding mistake would hide.
    pub fn render_quad_with_fallback_material(normal: [f32; 3]) -> Option<[u8; 4]> {
        render_quad(normal, Palette::Fallback, None)
    }

    fn render_quad(
        normal: [f32; 3],
        palette: Palette,
        splat: Option<&TerrainImage>,
    ) -> Option<[u8; 4]> {
        render_quad_full(normal, palette, splat, &[None, None, None, None], false)
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
        normals: &[Option<TerrainImage>],
        lit: bool,
    ) -> Option<[u8; 4]> {
        let (device, queue) = headless_gpu()?;
        let format = TextureFormat::Rgba8Unorm;

        let lights = LightsUniform::new(&device);
        let uniforms = UniformBuffer::new(&device, &lights);

        // Both uniform buffers are created UNINITIALISED. Without writing them
        // view_proj is garbage -- the triangle lands somewhere arbitrary and the
        // readback is just clear colour -- and the light count is undefined.
        // Identity view_proj means the vertex positions ARE clip coordinates,
        // and an empty light list leaves the shader's AMBIENT term, which is all
        // this test wants: it is asserting the splat blend, not the light rig.
        queue.write_buffer(
            &uniforms.buffer,
            0,
            bytemuck::bytes_of(&crate::renderer::uniforms::Uniforms {
                view_proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
            }),
        );
        if lit {
            // Placed off to one side so a tilt in the surface normal changes
            // how much light the fragment receives. Directly overhead would
            // make an x-tilt symmetric and hide exactly what is being tested.
            lights.upload(&queue, &[crate::renderer::lights::Light {
                position: glam::Vec3::new(3.0, 2.0, 0.0),
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
                &test_layers, &neutral, splat, normals, TerrainMaterialUniform::default(),
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
                    resolve_target: None,
                    ops: Operations { load: LoadOp::Clear(Color::BLACK), store: StoreOp::Store },
                })],
                depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(Operations { load: LoadOp::Clear(1.0), store: StoreOp::Store }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
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
        device.poll(PollType::Wait).ok();
        let data = slice.get_mapped_range();
        let centre = (SIZE / 2) as usize * 256 + (SIZE / 2) as usize * 4;
        Some([data[centre], data[centre + 1], data[centre + 2], data[centre + 3]])
    }

    #[test]
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
        // Per-layer normal maps, in the same slot order as `layers`. Shorter or
        // sparser than four is fine; the gaps become flat, which the shader
        // treats as no perturbation at all.
        normals: &[Option<TerrainImage>],
        settings: TerrainMaterialUniform,
    ) -> Self {
        assert!(!layers.is_empty(), "terrain material needs at least one layer");
        let (w, h) = (layers[0].width, layers[0].height);
        assert!(
            layers.iter().all(|l| l.width == w && l.height == h),
            "all terrain layers must share one size to live in a D2Array",
        );

        let array = device.create_texture(&TextureDescriptor {
            label: Some("terrain_layers"),
            size: Extent3d { width: w, height: h, depth_or_array_layers: 4 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8UnormSrgb,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });
        for slot in 0..4 {
            let src = layers.get(slot).unwrap_or_else(|| layers.last().unwrap());
            queue.write_texture(
                TexelCopyTextureInfo {
                    texture: &array,
                    mip_level: 0,
                    origin: Origin3d { x: 0, y: 0, z: slot as u32 },
                    aspect: TextureAspect::All,
                },
                &src.rgba,
                TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(4 * w),
                    rows_per_image: Some(h),
                },
                Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            );
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
        let normal_array = device.create_texture(&TextureDescriptor {
            label: Some("terrain_normals"),
            size: Extent3d { width: w, height: h, depth_or_array_layers: 4 },
            mip_level_count: 1,
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
            queue.write_texture(
                TexelCopyTextureInfo {
                    texture: &normal_array,
                    mip_level: 0,
                    origin: Origin3d { x: 0, y: 0, z: slot as u32 },
                    aspect: TextureAspect::All,
                },
                &src.rgba,
                TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(4 * w),
                    rows_per_image: Some(h),
                },
                Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            );
        }

        let sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("terrain_sampler"),
            address_mode_u: AddressMode::Repeat,
            address_mode_v: AddressMode::Repeat,
            address_mode_w: AddressMode::Repeat,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Linear,
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
        Self::fallback_with_splat(device, queue, layout, None)
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
    ) -> Self {
        Self::from_layers(
            device, queue, layout, &[None, None, None, None], &[None, None, None, None], splat,
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
            normals,
            TerrainMaterialUniform::default(),
        )
    }

}

/// A decoded RGBA8 image destined for the terrain material.
pub struct TerrainImage {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

#[cfg(test)]
mod material_tests {
    use super::tests::*;
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
        let greenness = |p: [u8; 4]| p[1] as i32 - p[2] as i32;
        assert!(
            greenness(flat) > greenness(cliff) + 8,
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
        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let _m = TerrainMaterial::new(
            &device, &queue, &layout,
            std::slice::from_ref(&one),
            &one,
            None,
            &[None, None, None, None],
            TerrainMaterialUniform::default(),
        );
        assert!(
            pollster::block_on(device.pop_error_scope()).is_none(),
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

fn solid_image(rgb: [u8; 3], width: u32, height: u32) -> TerrainImage {
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
fn resample(src: &TerrainImage, width: u32, height: u32) -> TerrainImage {
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

        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let _m = TerrainMaterial::from_layers(
            &device,
            &queue,
            &layout,
            &[Some(img([10, 20, 30], 64, 64)), None, Some(img([1, 2, 3], 64, 64)), None],
            &[None, None, None, None],
            None,
        );
        assert!(
            pollster::block_on(device.pop_error_scope()).is_none(),
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

        device.push_error_scope(wgpu::ErrorFilter::Validation);
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
        );
        assert!(
            pollster::block_on(device.pop_error_scope()).is_none(),
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
        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let _m = TerrainMaterial::from_layers(
            &device, &queue, &layout, &[None, None, None, None], &[None, None, None, None], None,
        );
        assert!(pollster::block_on(device.pop_error_scope()).is_none());
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
        let Some(flat) = render_quad_full(FLAT, Palette::Test, None, &only_layer0(normal_map(FLAT_NORMAL)), true)
        else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        // Tangent normal tilted hard toward +x, where the test light sits.
        let toward = render_quad_full(FLAT, Palette::Test, None, &only_layer0(normal_map([230, 128, 160])), true).unwrap();
        // ...and hard away from it.
        let away = render_quad_full(FLAT, Palette::Test, None, &only_layer0(normal_map([25, 128, 160])), true).unwrap();

        assert_ne!(brightness(toward), brightness(flat), "a tilted normal must change shading: {toward:?} vs {flat:?}");
        assert!(
            brightness(toward) > brightness(away),
            "tilting toward the light must be brighter than tilting away: {toward:?} vs {away:?}",
        );
    }

    /// A flat normal map must be indistinguishable from none at all -- that is
    /// what lets an unauthored normal set cost nothing and need no flag.
    #[test]
    fn a_flat_normal_map_is_a_true_no_op() {
        let Some(without) = render_quad_full(FLAT, Palette::Test, None, &[None, None, None, None], true)
        else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let with_flat = render_quad_full(FLAT, Palette::Test, None, &only_layer0(normal_map(FLAT_NORMAL)), true).unwrap();
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

        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let _m = TerrainMaterial::new(
            &device, &queue, &layout,
            &[solid_image([200, 200, 200], 4, 4)],
            &solid_image([128, 128, 128], 1, 1),
            None,
            &only_layer0(solid_image([230, 128, 160], 4, 4)),
            settings,
        );
        assert!(pollster::block_on(device.pop_error_scope()).is_none());
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
        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let _m = TerrainMaterial::from_layers(
            &device, &queue, &layout,
            &[Some(solid_image([10, 20, 30], 64, 64)), None, None, None],
            &[Some(solid_image(FLAT_NORMAL, 16, 16)), None, None, None],
            None,
        );
        assert!(
            pollster::block_on(device.pop_error_scope()).is_none(),
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
