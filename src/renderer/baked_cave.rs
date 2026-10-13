//! THE BAKED CAVE: a distance-field cave as the editor's bake ships it
//! (`scene_editor_web/frontend/src/lib/sdfBake.js`, docs/cave-voxel-rework-
//! proposal-2026-10-12.md), drawn with the terrain's tiled layers through
//! BAKED UVs and lit through its own lightmap.
//!
//! WHAT THE OLD CAVE SHADER PAID FOR, AND THIS DOES NOT
//!
//! `layered_mesh_pipeline` textures a mesh that has no uv: per pixel it ranks
//! four weights, picks two projection planes from the normal, reads each of
//! two layers on both planes for colour and for normals, reads the terrain's
//! splat and ground map for its copy of the ground, and takes nine taps of
//! the sun's map. 4,702 instructions on the headset against an instruction
//! cache that holds about 3,387 (PIPESTATS, 2026-10-08).
//!
//! The bake does all of that once:
//! - every triangle lies in a BOX-PROJECTED CHART, so its vertices carry the
//!   world's planar coordinate on the chart's axis (`uv0`, metres) and the
//!   axis itself. A layer is read once, at `uv0 / its metres a tile` -- the
//!   same texels the terrain reads on that plane, so rock and ground match;
//! - the untouched ground round a mouth is not the cave's to draw at all: it
//!   goes out as a terrain patch and the terrain draws it
//!   (`space_soup_engine::terrain`), so there is no copy of the ground here;
//! - the lamps, their bounce and the sky each texel sees are in the cave's
//!   LIGHTMAP (`uv1`, laid out by the bake and read by `tools/bake` instead of
//!   a generated layout), bound as any model's is;
//! - the sun the hill lets through is a baked vertex value.
//!
//! WHERE THE ROCK MEETS THE GROUND it is lit AS the ground: each vertex
//! carries how much of the terrain's own lighting it takes (one at the join,
//! none a blend band in) and where the terrain's ground map is read for it,
//! so the sky and the sun's shadow are the terrain's own values at the join
//! and nothing steps from one draw to the other. That read is behind a
//! branch: a pixel with none of it takes no fetch.
//!
//! IN BRICK CHUNKS. The index buffer is the chunks one after another, each
//! with its own box; a chunk no eye of the pass can see into is not drawn
//! (`visible_ranges`). Depth, shadow and SpaceWarp motion passes draw the
//! mesh's ordinary vertices, as they do for every model.
//!
//! EMISSIVE, FREE WHERE UNUSED: a cave with no glowing paint draws with the
//! twin that has no emissive term (`CaveShader::emissive`).

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3, Vec4};
use wgpu::*;

use super::lights::LightsBlockOptions;
use super::multiview::ViewMode;

/// The file's marker: `extras.space_soup.shading` on the mesh.
pub const SHADING: &str = "baked_cave";

/// A vertex of a baked cave.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable, PartialEq)]
pub struct BakedCaveVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    /// The WORLD's planar coordinate on the chart's axis, metres.
    pub uv0: [f32; 2],
    /// The lightmap atlas.
    pub uv1: [f32; 2],
    /// The terrain footprint's uv at the vertex: where the ground's maps are
    /// read for it.
    pub ground_uv: [f32; 2],
    /// Four layer weights, summing to 255.
    pub weights: [u8; 4],
    /// Sky seen, static sun seen, the tide band, soot.
    pub aux: [u8; 4],
    /// The chart's axis (0 x, 1 y, 2 z), emissive (0 to 63), the ground's
    /// share of the vertex's lighting (0 to 255), spare.
    pub misc: [u8; 4],
}

impl BakedCaveVertex {
    pub const ATTRIBS: [VertexAttribute; 8] = vertex_attr_array![
        0 => Float32x3, 1 => Float32x3, 2 => Float32x2, 3 => Float32x2, 4 => Float32x2, 5 => Unorm8x4, 6 => Unorm8x4, 7 => Uint8x4
    ];

    pub fn layout() -> VertexBufferLayout<'static> {
        VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as BufferAddress,
            step_mode: VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }

    /// From the file's attributes: `COLOR_0` is three weights and a packed
    /// byte (the top two bits the chart's axis, the low six the emissive);
    /// the fourth weight is what the three leave of 255. `COLOR_1` is `aux`;
    /// `ground` is `TEXCOORD_2` and the red of `COLOR_2`.
    #[allow(clippy::too_many_arguments)]
    pub fn from_file(
        position: [f32; 3],
        normal: [f32; 3],
        uv0: [f32; 2],
        uv1: [f32; 2],
        colour0: [u8; 4],
        colour1: [u8; 4],
        ground: ([f32; 2], u8),
    ) -> Self {
        let [a, b, c, packed] = colour0;
        let rest = 255u16.saturating_sub(a as u16 + b as u16 + c as u16) as u8;
        Self {
            position,
            normal,
            uv0,
            uv1,
            ground_uv: ground.0,
            weights: [a, b, c, rest],
            aux: colour1,
            misc: [(packed >> 6).min(2), packed & 63, ground.1, 0],
        }
    }
}

/// One brick chunk of a baked cave: a run of the index buffer and the box
/// round it, in the model's frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CaveChunk {
    pub first: u32,
    pub count: u32,
    pub min: [f32; 3],
    pub max: [f32; 3],
    /// How many of the run's LAST indices are the lip the rock runs under the
    /// ground's edge: drawn, and not part of the cave's surface.
    pub lip: u32,
}

/// What a baked cave's file says of itself (`extras.space_soup`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BakedCaveMeta {
    pub chunks: Vec<CaveChunk>,
    pub emissive: bool,
}

impl BakedCaveMeta {
    /// From a mesh's `extras` JSON, or `None` when it is not a baked cave.
    pub fn parse(extras: &str) -> Option<Self> {
        let v: serde_json::Value = serde_json::from_str(extras).ok()?;
        let ss = v.get("space_soup")?;
        if ss.get("shading").and_then(|s| s.as_str()) != Some(SHADING) {
            return None;
        }
        let cave = ss.get("cave");
        let chunks = cave
            .and_then(|c| c.get("chunks"))
            .and_then(|c| c.as_array())
            .map(|list| {
                list.iter()
                    .filter_map(|c| {
                        let c = c.as_array()?;
                        let f = |i: usize| c.get(i).and_then(|v| v.as_f64());
                        Some(CaveChunk {
                            first: f(0)? as u32,
                            count: f(1)? as u32,
                            min: [f(2)? as f32, f(3)? as f32, f(4)? as f32],
                            max: [f(5)? as f32, f(6)? as f32, f(7)? as f32],
                            lip: f(8).map_or(0, |l| l as u32).min(f(1)? as u32),
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();
        let emissive = cave.and_then(|c| c.get("emissive")).and_then(|e| e.as_bool()).unwrap_or(false);
        Some(Self { chunks, emissive })
    }
}

/// The same triangles, as vertices the baked-cave pipeline draws.
#[derive(Clone)]
pub struct BakedCavePrimitive {
    pub vertices: Vec<BakedCaveVertex>,
    pub vertex_buffer: Buffer,
    /// In index order, covering the index buffer. One chunk for a file that
    /// names none.
    pub chunks: Vec<CaveChunk>,
    pub emissive: bool,
    /// The box round every vertex, in the model's frame.
    pub bounds: [[f32; 3]; 2],
}

impl BakedCavePrimitive {
    /// `index_count` is the primitive's; a chunk table that does not cover
    /// exactly that is replaced by one chunk over all of it (a stale or
    /// hand-edited file draws whole rather than in part).
    pub fn new(device: &Device, vertices: Vec<BakedCaveVertex>, index_count: u32, meta: &BakedCaveMeta) -> Self {
        use wgpu::util::DeviceExt;
        let bounds = vertices.iter().fold([[f32::INFINITY; 3], [f32::NEG_INFINITY; 3]], |[lo, hi], v| {
            [
                [lo[0].min(v.position[0]), lo[1].min(v.position[1]), lo[2].min(v.position[2])],
                [hi[0].max(v.position[0]), hi[1].max(v.position[1]), hi[2].max(v.position[2])],
            ]
        });
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("baked_cave_vb"),
            contents: bytemuck::cast_slice(&vertices),
            usage: BufferUsages::VERTEX,
        });
        Self { chunks: covering_chunks(&meta.chunks, index_count, bounds), vertices, vertex_buffer, emissive: meta.emissive, bounds }
    }
}

/// `chunks` if they tile `0..index_count` in order; otherwise one chunk over
/// everything with `bounds`.
pub fn covering_chunks(chunks: &[CaveChunk], index_count: u32, bounds: [[f32; 3]; 2]) -> Vec<CaveChunk> {
    let mut next = 0u32;
    let tiles = !chunks.is_empty()
        && chunks.iter().all(|c| {
            let ok = c.first == next && c.count % 3 == 0;
            next = next.saturating_add(c.count);
            ok
        })
        && next == index_count;
    if tiles {
        chunks.to_vec()
    } else {
        vec![CaveChunk { first: 0, count: index_count, min: bounds[0], max: bounds[1], lip: 0 }]
    }
}

/// The runs of the index buffer to draw: the chunks some eye of `planes` can
/// see into, with neighbours in the buffer joined into one run. `None` draws
/// every chunk.
pub fn visible_ranges(chunks: &[CaveChunk], model: Mat4, planes: Option<&[[Vec4; 6]]>) -> Vec<std::ops::Range<u32>> {
    let mut out: Vec<std::ops::Range<u32>> = Vec::new();
    for c in chunks {
        if c.count == 0 {
            continue;
        }
        let seen = match planes {
            None => true,
            Some(p) => super::scene_cull::sees_box(p, super::cave_twins::placed_box(model, [c.min, c.max])),
        };
        if !seen {
            continue;
        }
        match out.last_mut() {
            Some(last) if last.end == c.first => last.end = c.first + c.count,
            _ => out.push(c.first..c.first + c.count),
        }
    }
    out
}

/// Which twin of the baked-cave shader.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CaveShader {
    /// The cave has glowing paint.
    pub emissive: bool,
    /// For frames in which no spot casts a shadow
    /// (`lights::without_spot_shadows`).
    pub spotless: bool,
}

const EMISSIVE_LINE: &str = "    colour = colour + albedo * (in.glow * EMISSIVE_GAIN);\n";

/// How bright full emissive paint is, against the rock's own colour.
pub const EMISSIVE_GAIN: f32 = 6.0;

/// The baked cave's shader for a twin.
pub fn baked_cave_shader(which: CaveShader) -> String {
    let mut src = full_shader();
    if !which.emissive {
        assert_eq!(src.matches(EMISSIVE_LINE).count(), 1, "the baked cave's emissive line moved");
        src = src.replacen(EMISSIVE_LINE, "", 1);
    }
    // The sun is the bake's on every pixel: the level's static map is never read.
    src = super::brush_pipeline::sun_reader_shader(src, super::brush_pipeline::FaceSun::Baked);
    if which.spotless {
        src = super::lights::without_spot_shadows(src);
    }
    src
}

/// The terrain material's uniform, as the shipped cave shader declares it:
/// taken from there so there is one description to keep in step with
/// `terrain_pipeline`.
fn material_struct() -> String {
    let old = super::layered_mesh_pipeline::layered_mesh_shader();
    let from = old.find("struct Material {").expect("the cave shader declares the terrain's Material");
    let to = old[from..].find("@group(1) @binding(3)").expect("...and binds it") + from;
    old[from..to].to_string()
}

fn full_shader() -> String {
    format!(
        r#"
// Group 0 -- the camera, the lights and the shadow maps -- is the lights block's.
// Group 1 is the terrain's material group, bound unchanged.
@group(1) @binding(0) var layer_tex: texture_2d_array<f32>;
@group(1) @binding(1) var layer_samp: sampler;
@group(1) @binding(2) var macro_tex: texture_2d<f32>;
{material}@group(1) @binding(3) var<uniform> mat: Material;
// The terrain's painted layer weights, read where the rock meets the ground.
@group(1) @binding(4) var splat_tex: texture_2d<f32>;
@group(1) @binding(5) var normal_tex: texture_2d_array<f32>;
// The terrain's ground map: layer 0 its sky and its baked sun, as the ground reads them.
@group(1) @binding(6) var sky_occ_tex: texture_2d_array<f32>;
@group(1) @binding(7) var rough_tex: texture_2d_array<f32>;
@group(1) @binding(8) var ao_tex: texture_2d_array<f32>;

struct ModelUniform {{ model: mat4x4<f32> }}
@group(2) @binding(0) var<uniform> model_u: ModelUniform;

// Group 3 is a model's lightmap group, as `mesh_pipeline` binds it.
@group(3) @binding(0) var lm_tex: texture_2d<f32>;
@group(3) @binding(1) var lm_samp: sampler;
@group(3) @binding(4) var lm_sun_samp: sampler;
@group(3) @binding(5) var lm_stationary: texture_2d_array<f32>;
const STATIONARY_MASK_DISTANCE_TEXELS: f32 = {stationary_range:?};
const GROUND_SUN_RANGE: f32 = {sun_range:?};
const EMISSIVE_GAIN: f32 = {emissive_gain:?};
const WEIGHT_EPS: f32 = 0.004;

{lights_block}
{whiteout_block}

struct VIn {{
    @location(0) pos: vec3<f32>,
    @location(1) norm: vec3<f32>,
    @location(2) uv0: vec2<f32>,
    @location(3) uv1: vec2<f32>,
    @location(4) ground_uv: vec2<f32>,
    @location(5) weights: vec4<f32>,
    @location(6) aux: vec4<f32>,
    @location(7) misc: vec4<u32>,
}}

struct VOut {{
    @builtin(position) clip: vec4<f32>,
    @location(0) normal: vec3<f32>,
    // In the player's frame, as every position the shaders get: lighting.
    @location(1) world_pos: vec3<f32>,
    @location(2) weights: vec4<f32>,
    // xy: the layers' coordinate, metres on the chart's plane. zw: the lightmap.
    @location(3) uv: vec4<f32>,
    @location(4) aux: vec4<f32>,
    // The chart's axis: one value a triangle.
    @location(5) @interpolate(flat) axis: u32,
    // The point's height in the WORLD (the sea's wash line), and its glow.
    @location(6) height: f32,
    @location(7) glow: f32,
    // xy: the terrain footprint's uv. z: the ground's share of the lighting.
    @location(8) ground: vec3<f32>,
}}

@vertex fn vs_main(v: VIn) -> VOut {{
    let world = model_u.model * vec4<f32>(v.pos, 1.0);
    var out: VOut;
    out.clip      = cam_view_proj() * world;
    out.normal    = (model_u.model * vec4<f32>(v.norm, 0.0)).xyz;
    out.world_pos = world.xyz;
    out.weights   = v.weights;
    out.uv        = vec4<f32>(v.uv0, v.uv1);
    out.aux       = v.aux;
    out.axis      = v.misc.x;
    out.height    = to_world_space(world.xyz).y;
    out.glow      = f32(v.misc.y) / 63.0;
    out.ground    = vec3<f32>(v.ground_uv, f32(v.misc.z) / 255.0);
    return out;
}}

@fragment fn fs_main(in: VOut, @builtin(front_facing) front: bool) -> @location(0) vec4<f32> {{
    // Renormalised: interpolation shortens a normal wherever its corners
    // disagree. FACING THE VIEWER: a cave is one sheet seen from inside and
    // out, drawn without culling; the far side of it arrives turned away.
    var n = normalize(in.normal);
    n = select(-n, n, front);
    // The rock's textures lie in the WORLD, and their bumps fold into the
    // world's normal; the finished normal turns back into the player's
    // frame, where the lamps are.
    let nw = to_world_direction(n);

    // The layers, each read ONCE on the chart's plane: its gradients taken
    // here, in uniform control flow, and scaled into the loop.
    let uv = in.uv.xy;
    let uv_dx = dpdx(uv);
    let uv_dy = dpdy(uv);
    let total = in.weights.x + in.weights.y + in.weights.z + in.weights.w;
    var w = select(vec4<f32>(0.0, 1.0, 0.0, 0.0), in.weights / max(total, 0.0001), total > 0.0001);
    // WHERE THE ROCK MEETS THE GROUND IT IS THE GROUND: the terrain's own
    // paint and its own sky and sun, read from the terrain's maps where the
    // ground reads them, taken whole at the join and less over the blend
    // band -- so the two draws are the same surface where they touch, and the
    // rock comes back to what its sculpt says over the band. Behind a branch
    // (explicit levels, so it may be): most of a cave has none of it and
    // takes neither fetch.
    let ground = clamp(in.ground.z, 0.0, 1.0);
    var ground_map = vec4<f32>(1.0, 1.0, 0.0, 1.0);
    if (ground > 0.0) {{
        let paint = textureSampleLevel(splat_tex, layer_samp, in.ground.xy, 0.0);
        let painted = paint / max(paint.r + paint.g + paint.b + paint.a, 0.001);
        w = mix(w, painted, ground * step(0.5, mat.use_splat));
        ground_map = textureSampleLevel(sky_occ_tex, layer_samp, in.ground.xy, 0, 0.0);
    }}
    var albedo = vec3<f32>(0.0);
    var bent = vec3<f32>(0.0);
    var rough_map = 0.0;
    var ao_map = 0.0;
    // Walked as a mask, lowest layer first, each weight picked by `select`:
    // the ground's own loop (`terrain_pipeline::LAYER_READS`).
    var layers_left = select(0u, 1u, w.x > WEIGHT_EPS) | select(0u, 2u, w.y > WEIGHT_EPS)
        | select(0u, 4u, w.z > WEIGHT_EPS) | select(0u, 8u, w.w > WEIGHT_EPS);
    loop {{
        if (layers_left == 0u) {{
            break;
        }}
        let layer = i32(countTrailingZeros(layers_left));
        layers_left = layers_left & (layers_left - 1u);
        let lw = select(select(select(w.x, w.y, layer == 1), w.z, layer == 2), w.w, layer == 3);
        let r = 1.0 / max(mat.repeat[layer], 0.001);
        let at = uv * r;
        let gx = uv_dx * r;
        let gy = uv_dy * r;
        albedo = albedo + textureSampleGrad(layer_tex, layer_samp, at, layer, gx, gy).rgb * lw;
        // OpenGL convention: green is +Y in tangent space.
        var tn = textureSampleGrad(normal_tex, layer_samp, at, layer, gx, gy).rgb * 2.0 - 1.0;
        tn = vec3<f32>(tn.xy * mat.normal_strength, tn.z);
        bent = bent + whiteout(in.axis, tn, nw) * lw;
        rough_map = rough_map + textureSampleGrad(rough_tex, layer_samp, at, layer, gx, gy).r * lw;
        ao_map = ao_map + textureSampleGrad(ao_tex, layer_samp, at, layer, gx, gy).r * lw;
    }}
    bent = normalize(select(nw, bent, dot(bent, bent) > 0.00000001));
    let shaded_n = to_player_direction(bent);
    // The ground's macro variation, on the chart's plane as the ground reads
    // it on a face's: without it the rock is a flat tone beside ground that
    // drifts a third lighter and darker.
    let mr = 1.0 / max(mat.macro_repeat, 0.001);
    let macro_value = textureSampleGrad(macro_tex, layer_samp, uv * mr, uv_dx * mr, uv_dy * mr).r;
    albedo = albedo * (1.0 + (macro_value - 0.5) * 2.0 * mat.macro_strength);

    // THE LIGHTMAP: rgb the baked lamps and their bounce, ADDED; alpha the
    // sky this texel sees. The vertices carry the editor's measure of the
    // same sky, and the LESSER of the two is taken: multiplied, a wall that
    // sees a fifth of the sky by both counts would be lit by a twenty-fifth.
    // An unbaked cave binds black and one, and is lit by what its vertices
    // measured alone.
    let baked = textureSample(lm_tex, lm_samp, in.uv.zw);
    // The baked lamps are in this atlas. See `receiver_skips_baked`.
    receiver_skips_baked = true;
    // The stationary lamps' shadows, as every model takes them.
    let st_layers = textureNumLayers(lm_stationary);
    let st_0 = textureSample(lm_stationary, lm_sun_samp, in.uv.zw, 0);
    var st_1 = vec4<f32>(1.0);
    var st_2 = vec4<f32>(1.0);
    var st_3 = vec4<f32>(1.0);
    if (st_layers > 1u) {{
        st_1 = textureSample(lm_stationary, lm_sun_samp, in.uv.zw, 1);
    }}
    if (st_layers > 2u) {{
        st_2 = textureSample(lm_stationary, lm_sun_samp, in.uv.zw, 2);
    }}
    if (st_layers > 3u) {{
        st_3 = textureSample(lm_stationary, lm_sun_samp, in.uv.zw, 3);
    }}
    set_stationary_masks_exact(st_0, st_1, st_2, st_3, STATIONARY_MASK_DISTANCE_TEXELS);

    var sky = min(clamp(in.aux.x, 0.0, 1.0), baked.a);
    // THE SUN THE HILL LETS THROUGH, baked: none where no sky is seen.
    var sun = smoothstep(0.01, 0.08, sky) * clamp(in.aux.y, 0.0, 1.0);
    // The ground's sky and its baked sun (`ground_map`, read above), by the
    // ground's share. An older map has no baked sun (alpha 1): the rock keeps
    // its own there.
    let ground_sun_d = (ground_map.g - 0.5) * (2.0 * GROUND_SUN_RANGE);
    let ground_sun_w = max(ground_map.b * GROUND_SUN_RANGE, 0.02);
    let ground_sun = select(sun, smoothstep(-ground_sun_w, ground_sun_w, ground_sun_d), ground_map.a < 0.5);
    sky = mix(sky, ground_map.r, ground);
    sun = mix(sun, ground_sun, ground);
    let own = 1.0 - ground;
    // DAMP ROCK deep in, where little sky reaches: darker, more so toward the
    // floor. Nothing in the open: a wall there sees half the sky.
    let deep = 1.0 - smoothstep(0.08, 0.4, sky);
    albedo = albedo * (1.0 - 0.33 * own * deep * (0.6 + 0.4 * clamp(nw.y, 0.0, 1.0)));
    // THE TIDE BAND at a sea mouth, and SOOT over a hearth.
    albedo = mix(albedo, albedo * vec3<f32>(0.30, 0.33, 0.27), 0.85 * own * clamp(in.aux.z, 0.0, 1.0));
    albedo = albedo * (1.0 - 0.8 * clamp(in.aux.w, 0.0, 1.0));
    // WET SAND where the sea's wash reaches: on the floors it runs over, and
    // on everything at the join, as the terrain darkens it.
    let washed = 1.0 - smoothstep(mat.wet_line, mat.wet_line + mat.wet_band, in.height);
    albedo = albedo * (1.0 - 0.45 * washed * mix(smoothstep(0.5, 0.8, nw.y), 1.0, ground));

    receiver_sun_mask = sun;

    // The ground's own shading path, so rock and the terrain it grows out of
    // take the same light the same way.
    var colour = shade_material_env(
        in.world_pos, shaded_n, clamp(rough_map, 0.0, 1.0), clamp(ao_map, 0.0, 1.0), sky,
        baked.rgb * own, vec4<f32>(0.5, 0.5, 0.5, 0.0), albedo, in.world_pos, n,
    );
    colour = colour + albedo * (in.glow * EMISSIVE_GAIN);
    return vec4<f32>(tonemap(colour), 1.0);
}}
"#,
        material = material_struct(),
        // No reflection of its own: built as a reader of the probe pass whose
        // answer it never sets, as the shipped cave shader is and for its
        // reason (`layered_mesh_pipeline`): rough rock shows nothing a trace
        // would add. A cave with surfaces that do gets a probe-reading twin
        // when it has them, measured (user rule, 2026-10-12).
        lights_block = super::lights::wgsl_lights_block_with(
            0, 1, LightsBlockOptions { probe_from_pass: true, ..Default::default() },
        ),
        whiteout_block = super::material_wgsl::wgsl_whiteout_block(),
        stationary_range = super::brush_pipeline::STATIONARY_MASK_DISTANCE_TEXELS,
        sun_range = super::brush_pipeline::SUN_MASK_DISTANCE_TEXELS,
        emissive_gain = EMISSIVE_GAIN,
    )
}

/// The cave's depth alone, for the reflection pass: see `cave_twins`.
fn occluder_shader() -> String {
    format!(
        r#"
struct ModelUniform {{ model: mat4x4<f32> }}
@group(2) @binding(0) var<uniform> model_u: ModelUniform;
{lights_block}
@vertex fn vs_main(@location(0) pos: vec3<f32>) -> @builtin(position) vec4<f32> {{
    return cam_view_proj() * (model_u.model * vec4<f32>(pos, 1.0));
}}
@fragment fn fs_depth() {{}}
"#,
        lights_block = super::lights::wgsl_lights_block(0, 1),
    )
}

/// One baked cave to draw.
pub struct BakedCaveDraw<'a> {
    pub model: &'a BindGroup,
    pub lightmap: &'a BindGroup,
    pub vertices: &'a Buffer,
    pub indices: &'a Buffer,
    pub index_count: u32,
    pub chunks: &'a [CaveChunk],
    pub emissive: bool,
    /// Where the model's frame is, player frame.
    pub placed: Mat4,
    /// Its whole box, placed.
    pub bounds: (Vec3, Vec3),
}

/// Every pipeline a baked cave is drawn with.
pub struct BakedCavePipelines {
    /// `[emissive][spotless]`, for the single-eye scene pass.
    scene: [[RenderPipeline; 2]; 2],
    /// `[emissive]`, both eyes in one pass. Built only when asked for.
    stereo: Option<[RenderPipeline; 2]>,
    /// `[emissive]`, for a mirrored world (its windings reversed).
    mirror: [RenderPipeline; 2],
    /// Depth alone, in the reflection pass.
    occluder: RenderPipeline,
    pub model_layout: BindGroupLayout,
}

impl BakedCavePipelines {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        material_layout: &BindGroupLayout,
        lightmap_layout: &BindGroupLayout,
        samples: u32,
        stereo: bool,
    ) -> Self {
        // As `layered_mesh_model_bgl`: identical to the mesh pipeline's model
        // layout, flags included, so a model's bind group fits.
        let model_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("baked_cave_model_bgl"),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
                ty: BindingType::Buffer { ty: BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            }],
        });
        let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("baked_cave_layout"),
            bind_group_layouts: &[Some(uniform_layout), Some(material_layout), Some(&model_layout), Some(lightmap_layout)],
            immediate_size: 0,
        });
        let build = |which: CaveShader, front_face: FrontFace, samples: u32, view: ViewMode, label: &str| {
            let shader = device.create_shader_module(ShaderModuleDescriptor {
                label: Some(label),
                source: ShaderSource::Wgsl(view.shader(baked_cave_shader(which)).into()),
            });
            device.create_render_pipeline(&RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                vertex: VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    compilation_options: PipelineCompilationOptions::default(),
                    buffers: &[Some(BakedCaveVertex::layout())],
                },
                fragment: Some(FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    compilation_options: PipelineCompilationOptions::default(),
                    // Opaque: the shader writes alpha 1.
                    targets: &[Some(ColorTargetState { format, blend: None, write_mask: ColorWrites::ALL })],
                }),
                primitive: PrimitiveState {
                    topology: PrimitiveTopology::TriangleList,
                    // No culling: one sheet, seen from inside and out.
                    cull_mode: None,
                    front_face,
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
            })
        };
        let twin = |emissive: bool, spotless: bool| {
            let label = match (emissive, spotless) {
                (false, false) => "baked_cave",
                (false, true) => "baked_cave_spotless",
                (true, false) => "baked_cave_emissive",
                (true, true) => "baked_cave_emissive_spotless",
            };
            build(CaveShader { emissive, spotless }, FrontFace::Ccw, samples, ViewMode::Mono, label)
        };
        let scene = [[twin(false, false), twin(false, true)], [twin(true, false), twin(true, true)]];
        let stereo = stereo.then(|| {
            [false, true].map(|emissive| {
                build(CaveShader { emissive, spotless: false }, FrontFace::Ccw, samples, ViewMode::Stereo, "baked_cave_stereo")
            })
        });
        let mirror = [false, true]
            .map(|emissive| build(CaveShader { emissive, spotless: false }, FrontFace::Cw, 1, ViewMode::Mono, "baked_cave_mirror"));
        let occluder = Self::occluder_pipeline(device, uniform_layout, &model_layout);
        Self { scene, stereo, mirror, occluder, model_layout }
    }

    fn occluder_pipeline(device: &Device, uniform_layout: &BindGroupLayout, model_layout: &BindGroupLayout) -> RenderPipeline {
        use crate::renderer::brush_pipeline::probe_pass::{FORMAT, REACH_FORMAT};
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("baked_cave_occluder"),
            source: ShaderSource::Wgsl(occluder_shader().into()),
        });
        let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("baked_cave_occluder_layout"),
            bind_group_layouts: &[Some(uniform_layout), None, Some(model_layout)],
            immediate_size: 0,
        });
        let nothing = |format| Some(ColorTargetState { format, blend: None, write_mask: ColorWrites::empty() });
        device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("baked_cave_occluder"),
            layout: Some(&layout),
            vertex: VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: PipelineCompilationOptions::default(),
                buffers: &[Some(BakedCaveVertex::layout())],
            },
            fragment: Some(FragmentState {
                module: &shader,
                entry_point: Some("fs_depth"),
                compilation_options: PipelineCompilationOptions::default(),
                targets: &[nothing(FORMAT), nothing(REACH_FORMAT)],
            }),
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                cull_mode: None,
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
            multisample: MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        })
    }

    /// The scene pass's pipeline for a cave.
    pub fn scene_pipeline(&self, emissive: bool, spotless: bool, stereo: bool) -> &RenderPipeline {
        match (&self.stereo, stereo) {
            (Some(s), true) => &s[emissive as usize],
            _ => &self.scene[emissive as usize][spotless as usize],
        }
    }

    /// Every cave of the scene pass: the chunks `planes` can see into (all of
    /// them for `None`). Returns the triangles drawn.
    #[allow(clippy::too_many_arguments)]
    pub fn draw_scene<'a>(
        &'a self,
        pass: &mut RenderPass<'a>,
        uniform: &'a BindGroup,
        material: &'a BindGroup,
        draws: &[BakedCaveDraw<'a>],
        planes: Option<&[[Vec4; 6]]>,
        spotless: bool,
        stereo: bool,
    ) -> u32 {
        let mut triangles = 0;
        for d in draws {
            if let Some(p) = planes {
                if !super::scene_cull::sees_box(p, d.bounds) {
                    continue;
                }
            }
            let ranges = visible_ranges(d.chunks, d.placed, planes);
            if ranges.is_empty() {
                continue;
            }
            pass.set_pipeline(self.scene_pipeline(d.emissive, spotless, stereo));
            pass.set_bind_group(0, uniform, &[]);
            pass.set_bind_group(1, material, &[]);
            pass.set_bind_group(2, d.model, &[]);
            pass.set_bind_group(3, d.lightmap, &[]);
            pass.set_vertex_buffer(0, d.vertices.slice(..));
            pass.set_index_buffer(d.indices.slice(..), IndexFormat::Uint32);
            for r in ranges {
                triangles += (r.end - r.start) / 3;
                pass.draw_indexed(r, 0, 0..1);
            }
        }
        triangles
    }

    /// Every cave in a mirrored world, whole.
    pub fn draw_mirror<'a>(
        &'a self,
        pass: &mut RenderPass<'a>,
        uniform: &'a BindGroup,
        material: &'a BindGroup,
        draws: impl Iterator<Item = &'a BakedCaveDraw<'a>>,
    ) {
        for d in draws {
            pass.set_pipeline(&self.mirror[d.emissive as usize]);
            pass.set_bind_group(0, uniform, &[]);
            pass.set_bind_group(1, material, &[]);
            pass.set_bind_group(2, d.model, &[]);
            pass.set_bind_group(3, d.lightmap, &[]);
            pass.set_vertex_buffer(0, d.vertices.slice(..));
            pass.set_index_buffer(d.indices.slice(..), IndexFormat::Uint32);
            pass.draw_indexed(0..d.index_count, 0, 0..1);
        }
    }

    /// The depth of every cave the eye at `eye` is inside, in the reflection
    /// pass (as `CaveTwins::occluder`).
    pub fn draw_occluders<'a>(&'a self, pass: &mut RenderPass<'a>, uniform: &'a BindGroup, draws: &[BakedCaveDraw<'a>], eye: Vec3) {
        let inside = |d: &BakedCaveDraw| eye.cmpge(d.bounds.0).all() && eye.cmple(d.bounds.1).all();
        if !draws.iter().any(inside) {
            return;
        }
        pass.set_pipeline(&self.occluder);
        pass.set_bind_group(0, uniform, &[]);
        for d in draws.iter().filter(|d| inside(d)) {
            pass.set_bind_group(2, d.model, &[]);
            pass.set_vertex_buffer(0, d.vertices.slice(..));
            pass.set_index_buffer(d.indices.slice(..), IndexFormat::Uint32);
            pass.draw_indexed(0..d.index_count, 0, 0..1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunk(first: u32, count: u32, x: f32) -> CaveChunk {
        CaveChunk { first, count, min: [x, 0.0, -1.0], max: [x + 1.0, 1.0, 1.0], lip: 0 }
    }

    #[test]
    fn a_file_vertex_unpacks_its_weights_axis_and_glow() {
        let v = BakedCaveVertex::from_file(
            [1.0; 3], [0.0, 1.0, 0.0], [2.0, 3.0], [0.25, 0.5], [100, 50, 5, (2 << 6) | 40], [255, 128, 3, 9], ([0.7, 0.2], 77),
        );
        assert_eq!(v.weights, [100, 50, 5, 100], "the fourth weight is what the three leave of 255");
        assert_eq!(v.misc, [2, 40, 77, 0]);
        assert_eq!(v.ground_uv, [0.7, 0.2]);
        assert_eq!(v.aux, [255, 128, 3, 9]);
        assert_eq!(std::mem::size_of::<BakedCaveVertex>(), 60);
        // Three weights over 255 (a file by another hand) leave nothing, not a wrapped byte.
assert_eq!(BakedCaveVertex::from_file([0.0; 3], [0.0; 3], [0.0; 2], [0.0; 2], [200, 100, 0, 0], [0; 4], ([0.0; 2], 0)).weights[3], 0);
    }

    #[test]
    fn the_file_says_what_it_is_and_names_its_chunks() {
        let meta = BakedCaveMeta::parse(
            r#"{"space_soup":{"shading":"baked_cave","lightmap":{"set":1,"size":[512,512]},
                "cave":{"version":3,"emissive":true,"chunks":[[0,6,-1,-2,-3,1,2,3],[6,3,0,0,0,4,5,6,3]]}}}"#,
        )
        .expect("a baked cave");
        assert!(meta.emissive);
        assert_eq!(meta.chunks.len(), 2);
        assert_eq!(meta.chunks[1], CaveChunk { first: 6, count: 3, min: [0.0, 0.0, 0.0], max: [4.0, 5.0, 6.0], lip: 3 });
        assert_eq!(meta.chunks[0].lip, 0, "a table from before the lip was counted: none");
        assert!(BakedCaveMeta::parse(r#"{"space_soup":{"shading":"layered"}}"#).is_none(), "the old cave is not this one");
        assert!(BakedCaveMeta::parse("{}").is_none());
        // No chunk table: still a baked cave, drawn whole.
        assert_eq!(BakedCaveMeta::parse(r#"{"space_soup":{"shading":"baked_cave"}}"#).unwrap().chunks.len(), 0);
    }

    #[test]
    fn a_chunk_table_that_does_not_cover_the_indices_is_replaced_by_one_chunk() {
        let b = [[0.0; 3], [9.0; 3]];
        let good = [chunk(0, 6, 0.0), chunk(6, 9, 2.0)];
        assert_eq!(covering_chunks(&good, 15, b), good.to_vec());
        for (bad, count) in [(vec![chunk(0, 6, 0.0), chunk(9, 6, 2.0)], 15), (good.to_vec(), 18), (vec![chunk(0, 5, 0.0)], 5), (vec![], 12)] {
            let one = covering_chunks(&bad, count, b);
            assert_eq!(one, vec![CaveChunk { first: 0, count, min: b[0], max: b[1], lip: 0 }]);
        }
    }

    #[test]
    fn only_the_chunks_an_eye_can_see_into_are_drawn_and_neighbours_join() {
        // An eye at the origin looking down -z: chunks at z -6..-4, spread along x.
        let proj = Mat4::perspective_rh(std::f32::consts::FRAC_PI_2, 1.0, 0.05, 500.0);
        let planes = [crate::renderer::shadow::frustum_planes(proj * Mat4::look_to_rh(Vec3::ZERO, Vec3::NEG_Z, Vec3::Y))];
        let at = |first: u32, x: f32| CaveChunk { first, count: 3, min: [x, -0.5, -6.0], max: [x + 1.0, 0.5, -4.0], lip: 0 };
        let chunks = [at(0, -40.0), at(3, -1.0), at(6, 0.5), at(9, 40.0), at(12, 2.0)];
        assert_eq!(visible_ranges(&chunks, Mat4::IDENTITY, Some(&planes)), vec![3..9, 12..15]);
        assert_eq!(visible_ranges(&chunks, Mat4::IDENTITY, None), vec![0..15]);
        // Placed elsewhere, the boxes move with the model.
        let moved = Mat4::from_translation(Vec3::new(40.0, 0.0, 0.0));
        assert_eq!(visible_ranges(&chunks, moved, Some(&planes)), vec![0..3]);
        // Behind the eye: nothing.
        assert!(visible_ranges(&chunks, Mat4::from_translation(Vec3::new(0.0, 0.0, 20.0)), Some(&planes)).is_empty());
    }

    #[test]
    fn every_twin_validates_and_differs_only_as_it_says() {
        let full = baked_cave_shader(CaveShader { emissive: true, spotless: false });
        let plain = baked_cave_shader(CaveShader { emissive: false, spotless: false });
        assert!(full.contains(EMISSIVE_LINE) && !plain.contains(EMISSIVE_LINE));
        assert_eq!(full.replacen(EMISSIVE_LINE, "", 1), plain, "a cave with no glow has the same shader less one line");
        for emissive in [false, true] {
            for spotless in [false, true] {
                for view in [ViewMode::Mono, ViewMode::Stereo] {
                    let src = view.shader(baked_cave_shader(CaveShader { emissive, spotless }));
                    let module = naga::front::wgsl::parse_str(&src)
                        .unwrap_or_else(|e| panic!("baked cave (emissive {emissive}, spotless {spotless}, {view:?}): {}", e.emit_to_string(&src)));
                    naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
                        .validate(&module)
                        .unwrap_or_else(|e| panic!("baked cave (emissive {emissive}, spotless {spotless}, {view:?}) does not validate: {e:?}"));
                }
            }
        }
        let occ = occluder_shader();
        let module = naga::front::wgsl::parse_str(&occ).unwrap_or_else(|e| panic!("{}", e.emit_to_string(&occ)));
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all()).validate(&module).unwrap();
    }

    #[test]
    fn the_shader_reads_each_layer_once_and_takes_no_shadow_tap_of_the_levels_sun() {
        let src = baked_cave_shader(CaveShader::default());
        let fs = &src[src.find("@fragment fn fs_main").unwrap()..];
        // Four reads in the layer loop (colour, normal, roughness, occlusion)
        // and the macro variation; the lightmap and the stationary masks; and
        // the ground map behind its branch. No splat, no second plane.
        assert_eq!(fs.matches("textureSampleGrad(").count(), 5);
        assert_eq!(fs.matches("textureSample(").count(), 5);
        assert_eq!(fs.matches("textureSampleLevel(").count(), 2);
        let branch = fs.find("if (ground > 0.0) {").expect("the ground's maps are read behind a branch");
        let end = branch + fs[branch..].find("\n    }").unwrap();
        for read in ["textureSampleLevel(sky_occ_tex", "textureSampleLevel(splat_tex"] {
            let at = fs.find(read).unwrap();
            assert!(at > branch && at < end, "{read} is inside the branch");
        }
        assert!(!fs.contains("biplanar_axes"));
        assert!(src.contains("const SUN_MASK_EVERYWHERE: bool = true;"), "the level's static sun map is compiled out");
        assert!(fs.contains("receiver_sun_mask = "));
        assert!(fs.contains("receiver_skips_baked = true;"), "its baked lamps are in its atlas");
        assert!(fs.trim_end().ends_with("return vec4<f32>(tonemap(colour), 1.0);\n}"), "opaque: alpha one");
    }

    // ---- on a GPU: the real pipeline, drawn and read back -------------------

    use crate::renderer::terrain_pipeline::{self, material_bind_group_layout, TerrainImage, TerrainMaterial, TerrainMaterialUniform};
    use crate::renderer::uniforms::test_support::{scene_uniforms, TEST_EYE};
    use crate::renderer::uniforms::ShadowUpload;

    /// Red across and green down, sixteen texels each way.
    fn ramp() -> TerrainImage {
        let mut rgba = Vec::with_capacity(16 * 16 * 4);
        for y in 0..16u8 {
            for x in 0..16u8 {
                rgba.extend_from_slice(&[x * 16, y * 16, 128, 255]);
            }
        }
        TerrainImage { width: 16, height: 16, rgba }
    }
    fn sized(rgba: [u8; 4]) -> TerrainImage {
        TerrainImage { width: 16, height: 16, rgba: rgba.repeat(16 * 16) }
    }

    struct Shot {
        weights: [u8; 4],
        uv0: [f32; 2],
        /// Where the surface is in the world: must NOT change what a layer reads.
        world: [f32; 3],
        aux: [u8; 4],
        glow: u8,
        emissive: bool,
        spotless: bool,
        /// The lightmap's one texel.
        lightmap: [u8; 4],
        mirror: bool,
        /// The ground's share of the lighting, and the ground map's one texel
        /// (sky, sun distance, penumbra, 0 = "has a baked sun").
        ground: u8,
        ground_map: [u8; 4],
        /// The terrain's paint there, when it has any (`use_splat`).
        paint: Option<[u8; 4]>,
        /// The macro map's one texel is mid grey less this much of its range.
        macro_strength: f32,
        macro_value: u8,
    }
    impl Default for Shot {
        fn default() -> Self {
            Self {
                weights: [0, 255, 0, 0],
                uv0: [0.0; 2],
                world: [0.0; 3],
                aux: [255, 255, 0, 0],
                glow: 0,
                emissive: false,
                spotless: false,
                lightmap: [0, 0, 0, 255],
                mirror: false,
                ground: 0,
                ground_map: [255, 255, 0, 0],
                paint: None,
                macro_strength: 0.0,
                macro_value: 128,
            }
        }
    }

    /// One triangle over the whole target through the real pipeline, with a
    /// model's own bind groups; its centre pixel, or `None` with no GPU.
    /// Panics on any validation error (anti-pattern #673: a layout that only
    /// nearly fits a shared bind group draws nothing on the headset).
    fn render(shot: Shot) -> Option<[u8; 4]> {
        use wgpu::util::DeviceExt;
        let (device, queue) = terrain_pipeline::tests::headless_gpu()?;
        let scope = device.push_error_scope(ErrorFilter::Validation);
        let format = TextureFormat::Rgba8Unorm;
        let lights = crate::renderer::lights::LightsUniform::new(&device);
        let (_shadows, uniforms) = scene_uniforms(&device, &lights);
        lights.upload(&queue, &[]);
        let material_layout = material_bind_group_layout(&device);
        // A model's bind groups come from the MESH pipeline: its uniform and
        // its lightmap are what a frame binds here.
        let mesh = crate::renderer::mesh_pipeline::MeshPipeline::new_multisampled(&device, format, &uniforms.layout, 1);
        let pipelines = BakedCavePipelines::new(&device, format, &uniforms.layout, &material_layout, &mesh.lightmap_layout, 1, false);
        let material = TerrainMaterial::new(
            &device,
            &queue,
            &material_layout,
            &[sized([255, 0, 0, 255]), ramp(), sized([0, 0, 255, 255]), sized([255, 255, 255, 255])],
            &TerrainImage { width: 1, height: 1, rgba: vec![shot.macro_value, shot.macro_value, shot.macro_value, 255] },
            shot.paint.map(|p| TerrainImage { width: 1, height: 1, rgba: p.to_vec() }).as_ref(),
            Some(&TerrainImage { width: 1, height: 1, rgba: shot.ground_map.to_vec() }),
            &[None, None, None, None],
            &[],
            &[],
            TerrainMaterialUniform { macro_strength: shot.macro_strength, macro_repeat: 1.0, repeat: [1.0; 4], ..Default::default() },
        );
        let lightmap = crate::renderer::mesh::create_lightmap_texture(&device, &queue, &mesh.lightmap_layout, &shot.lightmap, 1, 1, None);
        let world = Vec3::from(shot.world);
        let model = mesh.create_model_uniform(&device);
        model.upload(&queue, Mat4::from_translation(world));
        // The eye goes with the surface, so only WHERE it is changes.
        uniforms.upload(&queue, Mat4::from_translation(-world), TEST_EYE + world, &ShadowUpload::disabled());

        let v = |p: [f32; 3]| BakedCaveVertex {
            position: p,
            // Facing the eye (+z toward it in clip space is away; the shader
            // faces the normal to the viewer either way).
            normal: [0.0, 0.3, -0.954],
            uv0: shot.uv0,
            uv1: [0.5, 0.5],
            ground_uv: [0.5, 0.5],
            weights: shot.weights,
            aux: shot.aux,
            misc: [2, shot.glow, shot.ground, 0],
        };
        let verts = [v([-1.0, -1.0, 0.5]), v([3.0, -1.0, 0.5]), v([-1.0, 3.0, 0.5])];
        let vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&verts),
            usage: BufferUsages::VERTEX,
        });
        let ib = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&[0u32, 1, 2]),
            usage: BufferUsages::INDEX,
        });
        const SIZE: u32 = 8;
        let texture = |format, usage| {
            device.create_texture(&TextureDescriptor {
                label: None,
                size: Extent3d { width: SIZE, height: SIZE, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
        };
        let target = texture(format, TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC);
        let depth = texture(TextureFormat::Depth32Float, TextureUsages::RENDER_ATTACHMENT);
        let (target_view, depth_view) = (target.create_view(&Default::default()), depth.create_view(&Default::default()));
        let readback = device.create_buffer(&BufferDescriptor {
            label: None,
            size: (256 * SIZE) as u64,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let chunks = [CaveChunk { first: 0, count: 3, min: [-1.0, -1.0, 0.0], max: [3.0, 3.0, 1.0], lip: 0 }];
        let draws = [BakedCaveDraw {
            model: &model.bind_group,
            lightmap: &lightmap.bind_group,
            vertices: &vb,
            indices: &ib,
            index_count: 3,
            chunks: &chunks,
            emissive: shot.emissive,
            placed: Mat4::from_translation(world),
            bounds: (world + Vec3::new(-1.0, -1.0, 0.0), world + Vec3::new(3.0, 3.0, 1.0)),
        }];
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: None,
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
            if shot.mirror {
                pipelines.draw_mirror(&mut pass, &uniforms.bind_group, &material.bind_group, draws.iter());
            } else {
                let drawn = pipelines.draw_scene(&mut pass, &uniforms.bind_group, &material.bind_group, &draws, None, shot.spotless, false);
                assert_eq!(drawn, 1);
            }
        }
        encoder.copy_texture_to_buffer(
            TexelCopyTextureInfo { texture: &target, mip_level: 0, origin: Origin3d::ZERO, aspect: TextureAspect::All },
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
        if let Some(e) = pollster::block_on(scope.pop()) {
            panic!("the baked cave's pipelines do not validate against a model's bind groups: {e}");
        }
        let data = slice.get_mapped_range().unwrap();
        let at = (SIZE / 2) as usize * 256 + (SIZE / 2) as usize * 4;
        Some([data[at], data[at + 1], data[at + 2], data[at + 3]])
    }

    macro_rules! shot {
        ($s:expr) => {
            match render($s) {
                Some(p) => p,
                None => {
                    eprintln!("no GPU adapter: skipped");
                    return;
                }
            }
        };
    }

    #[test]
    fn each_weight_draws_its_own_layer_and_the_pixel_is_opaque() {
        let red = shot!(Shot { weights: [255, 0, 0, 0], ..Default::default() });
        let blue = shot!(Shot { weights: [0, 0, 255, 0], ..Default::default() });
        let white = shot!(Shot { weights: [0, 0, 0, 255], ..Default::default() });
        assert!(red[0] > 40 && red[0] > red[2] * 3, "layer 0 is red: {red:?}");
        assert!(blue[2] > 40 && blue[2] > blue[0] * 3, "layer 2 is blue: {blue:?}");
        assert!(white[0] > 40 && white[0].abs_diff(white[2]) < 12, "layer 3 is white: {white:?}");
        assert_eq!([red[3], blue[3], white[3]], [255; 3]);
        // No weights at all (a damaged file): layer 1, not black.
        let none = shot!(Shot { weights: [0; 4], ..Default::default() });
        assert!(none[0] as u32 + none[1] as u32 + none[2] as u32 > 30, "{none:?}");
    }

    #[test]
    fn a_layer_is_read_at_the_baked_uv_and_nowhere_else() {
        // Layer 1 is the ramp: red rises with u, green with v, a tile a metre.
        let at = |uv0: [f32; 2], world: [f32; 3]| render(Shot { uv0, world, ..Default::default() });
        let Some(a) = at([0.2, 0.2], [0.0; 3]) else { return };
        let b = at([0.8, 0.2], [0.0; 3]).unwrap();
        let c = at([0.2, 0.8], [0.0; 3]).unwrap();
        assert!(b[0] > a[0] + 30 && b[1].abs_diff(a[1]) < 12, "u moves the read across: {a:?} {b:?}");
        assert!(c[1] > a[1] + 30 && c[0].abs_diff(a[0]) < 12, "v moves it down: {a:?} {c:?}");
        // The same uv with the surface somewhere else in the world: the same
        // texel. The old cave shader projected from the world's position.
        let moved = at([0.2, 0.2], [0.37, 5.21, -3.3]).unwrap();
        for k in 0..3 {
            assert!(moved[k].abs_diff(a[k]) <= 3, "the world's position must not move the read: {a:?} {moved:?}");
        }
    }

    #[test]
    fn the_lightmap_adds_its_light_and_its_alpha_takes_the_sky_away() {
        let w = Shot { weights: [0, 0, 0, 255], ..Default::default() };
        let plain = shot!(Shot { ..w });
        let lit = shot!(Shot { lightmap: [200, 60, 20, 255], weights: [0, 0, 0, 255], ..Default::default() });
        assert!(lit[0] > plain[0] + 20, "baked light is added: {plain:?} {lit:?}");
        assert!(lit[0] as i32 - lit[2] as i32 > plain[0] as i32 - plain[2] as i32 + 15, "...in its colour: {plain:?} {lit:?}");
        let shut = shot!(Shot { lightmap: [0, 0, 0, 0], weights: [0, 0, 0, 255], ..Default::default() });
        assert!(shut[1] + 20 < plain[1], "a texel that sees no sky takes none of its light: {plain:?} {shut:?}");
        // And the vertex's own measure of the sky does the same.
        let buried = shot!(Shot { aux: [0, 255, 0, 0], weights: [0, 0, 0, 255], ..Default::default() });
        assert!(buried[1] + 20 < plain[1], "{plain:?} {buried:?}");
    }

    #[test]
    fn at_the_join_the_rock_takes_the_grounds_sky_and_its_own_a_band_in() {
        let w = [0, 0, 0, 255];
        // The rock's own measure says open sky; the ground's map says none.
        let shut = [0, 255, 0, 0];
        let own = shot!(Shot { weights: w, ground: 0, ground_map: shut, ..Default::default() });
        let joined = shot!(Shot { weights: w, ground: 255, ground_map: shut, ..Default::default() });
        let half = shot!(Shot { weights: w, ground: 128, ground_map: shut, ..Default::default() });
        assert!(joined[1] + 20 < own[1], "at the join the sky is the ground's: {own:?} {joined:?}");
        assert!(half[1] > joined[1] && half[1] < own[1], "and between, between: {joined:?} {half:?} {own:?}");
        // The other way: the rock's own says buried, the ground's map says open.
        let open = [255, 255, 0, 0];
        let buried = shot!(Shot { weights: w, aux: [0, 0, 0, 0], ground: 0, ground_map: open, ..Default::default() });
        let lifted = shot!(Shot { weights: w, aux: [0, 0, 0, 0], ground: 255, ground_map: open, ..Default::default() });
        assert!(lifted[1] > buried[1] + 20, "{buried:?} {lifted:?}");
        // Its baked light and its tide band are its own: gone at the join, where the ground has neither.
        let lamp = shot!(Shot { weights: w, lightmap: [200, 60, 20, 255], ground: 255, ..Default::default() });
        let plain = shot!(Shot { weights: w, ground: 255, ..Default::default() });
        let tide = shot!(Shot { weights: w, aux: [255, 255, 255, 0], ground: 255, ..Default::default() });
        for k in 0..3 {
            assert!(lamp[k].abs_diff(plain[k]) <= 2 && tide[k].abs_diff(plain[k]) <= 2, "{plain:?} {lamp:?} {tide:?}");
        }
    }

    #[test]
    fn at_the_join_the_rock_wears_the_grounds_paint() {
        // The sculpt says layer 0 (red); the terrain is painted layer 2 (blue) there.
        let blue = Some([0, 0, 255, 0]);
        let own = shot!(Shot { weights: [255, 0, 0, 0], ground: 0, paint: blue, ..Default::default() });
        let joined = shot!(Shot { weights: [255, 0, 0, 0], ground: 255, paint: blue, ..Default::default() });
        let half = shot!(Shot { weights: [255, 0, 0, 0], ground: 128, paint: blue, ..Default::default() });
        assert!(own[0] > own[2] * 3, "a band in, the sculpt's own layer: {own:?}");
        assert!(joined[2] > joined[0] * 3, "at the join, the terrain's paint: {joined:?}");
        assert!(half[0] > 10 && half[2] > 10, "between, both: {half:?}");
        // A terrain with no paint blends by slope: the rock keeps its own.
        let unpainted = shot!(Shot { weights: [255, 0, 0, 0], ground: 255, paint: None, ..Default::default() });
        assert!(unpainted[0] > unpainted[2] * 3, "{unpainted:?}");
    }

    #[test]
    fn the_grounds_macro_variation_reaches_the_rock() {
        let w = [0, 0, 0, 255];
        let flat = shot!(Shot { weights: w, macro_strength: 0.0, macro_value: 20, ..Default::default() });
        let dark = shot!(Shot { weights: w, macro_strength: 0.35, macro_value: 20, ..Default::default() });
        let light = shot!(Shot { weights: w, macro_strength: 0.35, macro_value: 235, ..Default::default() });
        assert!(dark[0] + 10 < flat[0] && light[0] > flat[0] + 10, "{dark:?} {flat:?} {light:?}");
    }

    #[test]
    fn soot_and_the_tide_band_darken_and_glow_brightens_only_in_its_twin() {
        let w = [0, 0, 0, 255];
        let plain = shot!(Shot { weights: w, ..Default::default() });
        let soot = shot!(Shot { weights: w, aux: [255, 255, 0, 255], ..Default::default() });
        let tide = shot!(Shot { weights: w, aux: [255, 255, 255, 0], ..Default::default() });
        assert!(soot[0] + 30 < plain[0], "soot: {plain:?} {soot:?}");
        assert!(tide[0] + 30 < plain[0], "the tide band: {plain:?} {tide:?}");
        // A dark texel so the glow has room: no sky, no baked light.
        let dark = Shot { weights: w, lightmap: [0, 0, 0, 0], ..Default::default() };
        let off = shot!(Shot { glow: 63, ..Shot { ..dark } });
        let dark = Shot { weights: w, lightmap: [0, 0, 0, 0], ..Default::default() };
        let on = shot!(Shot { glow: 63, emissive: true, ..dark });
        assert!(on[0] > off[0] + 40, "emissive paint glows in the emissive twin: {off:?} {on:?}");
    }

    #[test]
    fn the_spotless_and_mirror_twins_draw_the_same_rock() {
        let base = shot!(Shot { uv0: [0.3, 0.6], ..Default::default() });
        let spotless = shot!(Shot { uv0: [0.3, 0.6], spotless: true, ..Default::default() });
        let mirror = shot!(Shot { uv0: [0.3, 0.6], mirror: true, ..Default::default() });
        for k in 0..4 {
            assert!(base[k].abs_diff(spotless[k]) <= 2, "{base:?} {spotless:?}");
            // The mirror's front faces are the other winding: this triangle is
            // seen there from behind, its normal turned, so its light differs
            // a little. The same rock all the same.
            assert!(base[k].abs_diff(mirror[k]) <= 12, "{base:?} {mirror:?}");
        }
        assert!(mirror[1] > 10, "the mirror twin draws: {mirror:?}");
    }

    /// MEASUREMENT: the baked cave's size by the inlining proxy, beside the
    /// shipped cave shader's (`layered_mesh_pipeline::tests::cave_shader_size`,
    /// 2,646 expressions for 4,702 headset instructions, x1.78). Run with
    /// `--ignored --nocapture`.
    #[test]
    #[ignore]
    fn baked_cave_sizes() {
        for (name, which) in [
            ("baked_cave", CaveShader { emissive: false, spotless: false }),
            ("baked_cave_spotless", CaveShader { emissive: false, spotless: true }),
            ("baked_cave_emissive", CaveShader { emissive: true, spotless: false }),
        ] {
            let src = ViewMode::Mono.shader(baked_cave_shader(which));
            println!("{}", crate::renderer::shader_inlining::report(name, &src, "fs_main", 8));
        }
        let old = ViewMode::Mono.shader(crate::renderer::layered_mesh_pipeline::layered_mesh_shader());
        println!("{}", crate::renderer::shader_inlining::report("layered_mesh (shipped)", &old, "fs_main", 4));
    }
}
