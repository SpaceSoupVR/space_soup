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
//! - every face reads the layers on ONE OF THE WORLD'S THREE PLANES, chosen
//!   by the bake (`sdfBake.js::planeLabels`), at the world's own coordinate
//!   on it divided by a layer's metres a tile -- the same texels the terrain
//!   reads on that plane, so rock and ground match. Each vertex carries the
//!   SHARE of each plane (`planes`): one plane whole over most of the rock,
//!   and two blended in a band where a smooth run turns from one plane to
//!   the next, so the pattern never changes along a line there (2026-10-13:
//!   read on one plane a chart, it did, and showed at arm's length under a
//!   torch). A pixel reads the planes it has a share of and no other;
//! - and THE SHADER IS THREE TWINS, ONE A CLASS OF TRIANGLE (`CavePlanes`,
//!   `partition`): one plane at every corner (seven tenths of the rock) is
//!   drawn by the one-plane shader, its coordinate worked out a vertex; two
//!   or three of the rock's own planes by a twin that reads those; and only
//!   the triangles that take some of a cliff's reading by the one that can
//!   do everything. One shader for all of it held 25 registers on the
//!   headset against the one-plane shader's 16 -- 37% occupancy against 75%
//!   -- on every pixel of rock (PIPESTATS, 2026-10-13);
//! - where the rock meets a CLIFF the terrain reads its layers on two planes,
//!   warped and at two scales (`terrain_pipeline::sample_frame`), and the
//!   rock takes that same reading by the share its vertices carry
//!   (`cliff`), behind the branch the ground's maps are read in;
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
//! twin that has no emissive term (`CaveShader::emissive`). The paint has a
//! brightness and a COLOUR a vertex: the colour is opened in the vertex stage
//! of the emissive twin alone (`EMISSIVE_PARTS`), and a pixel pays one
//! multiply and one add for it.

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
    /// The share of each plane the layers are read on (along x, y, z), and
    /// how much of a cliff's reading the vertex takes where it meets one.
    pub planes: [u8; 4],
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
    /// share of the vertex's lighting (0 to 255), and the HUE of the
    /// emissive paint's colour (`glow_hue_saturation`).
    pub misc: [u8; 4],
    /// Where the rock meets a cliff: the terrain's own shares of the planes
    /// there (x, y, z); and the SATURATION of the emissive paint's colour
    /// (0: the rock glows in its own colour, as a file without one does).
    pub cliff: [u8; 4],
}

impl BakedCaveVertex {
    pub const ATTRIBS: [VertexAttribute; 9] = vertex_attr_array![
        0 => Float32x3, 1 => Float32x3, 2 => Unorm8x4, 3 => Float32x2, 4 => Float32x2, 5 => Unorm8x4, 6 => Unorm8x4, 7 => Uint8x4,
        8 => Unorm8x4
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
    /// `ground` is `TEXCOORD_2` and the red of `COLOR_2`. `planes` is the
    /// rest of `COLOR_2` (the shares of the planes along x, y, z) with
    /// `COLOR_3` (the terrain's shares at a cliff, and in alpha how much of
    /// the cliff's reading is taken); `None` for a file that has no
    /// `COLOR_3`, which reads each chart on its own axis and nothing else.
    /// `glow` is `COLOR_4`, which only a cave with emissive paint has: the
    /// paint's colour at full brightness; `None` glows in the rock's own.
    #[allow(clippy::too_many_arguments)]
    pub fn from_file(
        position: [f32; 3],
        normal: [f32; 3],
        uv1: [f32; 2],
        colour0: [u8; 4],
        colour1: [u8; 4],
        ground: ([f32; 2], u8),
        planes: Option<([u8; 3], [u8; 4])>,
        glow: Option<[u8; 4]>,
    ) -> Self {
        let (hue, saturation) = glow.map_or((0, 0), |c| glow_hue_saturation([c[0], c[1], c[2]]));
        let [a, b, c, packed] = colour0;
        let rest = 255u16.saturating_sub(a as u16 + b as u16 + c as u16) as u8;
        let axis = (packed >> 6).min(2);
        let (own, cliff) = match planes {
            Some((own, cliff)) if own != [0, 0, 0] => (own, cliff),
            _ => ([(axis == 0) as u8 * 255, (axis == 1) as u8 * 255, (axis == 2) as u8 * 255], [0; 4]),
        };
        Self {
            position,
            normal,
            planes: [own[0], own[1], own[2], cliff[3]],
            cliff: [cliff[0], cliff[1], cliff[2], saturation],
            uv1,
            ground_uv: ground.0,
            weights: [a, b, c, rest],
            aux: colour1,
            misc: [axis, packed & 63, ground.1, hue],
        }
    }
}

/// A colour byte as a picker has it (sRGB) to the light it stands for.
fn srgb_to_linear(byte: u8) -> f32 {
    let c = byte as f32 / 255.0;
    if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
}

/// THE COLOUR OF EMISSIVE PAINT, as the two spare bytes of a vertex carry
/// it: the hue and the saturation of the light it stands for (the file's
/// colour is a picker's, sRGB). How bright it is rides apart, in the file's
/// six bits, so the colour is taken at full brightness. White, grey and
/// black are saturation 0: the rock glows in its own colour.
pub fn glow_hue_saturation(rgb: [u8; 3]) -> (u8, u8) {
    let [r, g, b] = rgb.map(srgb_to_linear);
    let (max, min) = (r.max(g).max(b), r.min(g).min(b));
    let spread = max - min;
    if max <= 0.0 || spread <= 0.0 {
        return (0, 0);
    }
    let sixth = if max == r { ((g - b) / spread).rem_euclid(6.0) } else if max == g { (b - r) / spread + 2.0 } else { (r - g) / spread + 4.0 };
    // 255 is the hue 0 again: the shader takes the fraction.
    (((sixth / 6.0) * 255.0).round() as u8, ((spread / max) * 255.0).round() as u8)
}

/// What the emissive twin's vertex stage makes of those two bytes
/// (`EMISSIVE_PARTS`), on the CPU: the tint at full brightness. For the tests.
#[cfg(test)]
fn glow_tint(hue: u8, saturation: u8) -> [f32; 3] {
    let h = hue as f32 / 255.0;
    let s = saturation as f32 / 255.0;
    [0.0f32, 2.0 / 3.0, 1.0 / 3.0].map(|k| {
        let pure = (((h + k).fract() * 6.0 - 3.0).abs() - 1.0).clamp(0.0, 1.0);
        1.0 + (pure - 1.0) * s
    })
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
    /// Which twin draws the run (`partition`). A file's own table says
    /// `Cliff`: the twin that draws anything.
    pub planes: CavePlanes,
}

/// HOW MANY PLANES A TRIANGLE'S LAYERS ARE READ ON, which is which twin of
/// the shader draws it (`partition`, `CaveShader::planes`).
///
/// Reading every pixel through the shader that can blend planes and take a
/// cliff's reading cost all of them: on the headset it held 25 registers
/// against the 16 of the one-plane shader, two occupancy tiers down (75% to
/// 37%, PIPESTATS 2026-10-13), for work seven triangles in ten never do.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CavePlanes {
    /// One plane at every corner, the same one: the layers are read once, at
    /// a coordinate the vertices carry.
    One,
    /// The rock's own planes, two or three of them, by their shares.
    Own,
    /// Some of a cliff's reading too (or shares that do not add up to one):
    /// everything the shader can do.
    #[default]
    Cliff,
}

impl CavePlanes {
    pub const ALL: [CavePlanes; 3] = [CavePlanes::One, CavePlanes::Own, CavePlanes::Cliff];

    /// A triangle's class, from its corners. A plane is read where its share
    /// passes `WEIGHT_EPS` of the corner's total, as the shader's own test.
    pub fn of(corners: [&BakedCaveVertex; 3]) -> CavePlanes {
        let mut read = [false; 3];
        for v in corners {
            let total = v.planes[0] as u32 + v.planes[1] as u32 + v.planes[2] as u32;
            // A cliff's reading, or shares the shader would have to
            // normalise: the twin that does.
            if v.planes[3] > 0 || total != 255 {
                return CavePlanes::Cliff;
            }
            for k in 0..3 {
                read[k] |= v.planes[k] as f32 / 255.0 > WEIGHT_EPS;
            }
        }
        match read.iter().filter(|r| **r).count() {
            1 => CavePlanes::One,
            _ => CavePlanes::Own,
        }
    }
}

/// A share under this is not read (the shader's constant of the same name).
pub const WEIGHT_EPS: f32 = 0.004;

/// THE TRIANGLES BY CLASS: the same triangles, reordered so that each twin of
/// the shader draws runs of its own. Returns the index buffer -- every
/// chunk's one-plane triangles, then every chunk's blended ones, then the
/// rest, each in the file's order -- and the table of those runs, a chunk's
/// box on each of its (up to three). The lip stays the tail of a chunk's last
/// run and is drawn by the twin that draws anything.
pub fn partition(vertices: &[BakedCaveVertex], indices: &[u32], chunks: &[CaveChunk]) -> (Vec<u32>, Vec<CaveChunk>) {
    let mut out = Vec::with_capacity(indices.len());
    let mut table = Vec::new();
    let class_of = |c: &CaveChunk, at: usize| {
        let surface_end = (c.first + c.count - c.lip.min(c.count)) as usize;
        if at >= surface_end {
            return CavePlanes::Cliff;
        }
        let corner = |k: usize| vertices.get(indices[at + k] as usize);
        match (corner(0), corner(1), corner(2)) {
            (Some(a), Some(b), Some(d)) => CavePlanes::of([a, b, d]),
            _ => CavePlanes::Cliff,
        }
    };
    for planes in CavePlanes::ALL {
        for c in chunks {
            let first = out.len() as u32;
            let run = c.first as usize..(c.first + c.count) as usize;
            for at in run.step_by(3) {
                if at + 2 < indices.len() && class_of(c, at) == planes {
                    out.extend_from_slice(&indices[at..at + 3]);
                }
            }
            let count = out.len() as u32 - first;
            if count > 0 {
                let lip = if planes == CavePlanes::Cliff { c.lip.min(count) } else { 0 };
                table.push(CaveChunk { first, count, min: c.min, max: c.max, lip, planes });
            }
        }
    }
    (out, table)
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
                            planes: CavePlanes::Cliff,
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
    /// The runs of the index buffer, in its order and covering it: each of
    /// the file's chunks (one over everything for a file that names none) in
    /// up to three runs, one a twin of the shader (`partition`).
    pub chunks: Vec<CaveChunk>,
    pub emissive: bool,
    /// The box round every vertex, in the model's frame.
    pub bounds: [[f32; 3]; 2],
}

impl BakedCavePrimitive {
    /// `indices` are the primitive's, and are REORDERED here, the same
    /// triangles by the twin that draws each (`partition`): every pass that
    /// draws the mesh takes them as they leave. A chunk table that does not
    /// cover exactly them is replaced by one chunk over all of it (a stale or
    /// hand-edited file draws whole rather than in part).
    pub fn new(device: &Device, vertices: Vec<BakedCaveVertex>, indices: &mut Vec<u32>, meta: &BakedCaveMeta) -> Self {
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
        let (by_class, chunks) = partition(&vertices, indices, &covering_chunks(&meta.chunks, indices.len() as u32, bounds));
        if by_class.len() == indices.len() {
            *indices = by_class;
            Self { chunks, vertices, vertex_buffer, emissive: meta.emissive, bounds }
        } else {
            // Not whole triangles: drawn as it is, by the twin that draws anything.
            let whole = vec![CaveChunk { first: 0, count: indices.len() as u32, min: bounds[0], max: bounds[1], lip: 0, planes: CavePlanes::Cliff }];
            Self { chunks: whole, vertices, vertex_buffer, emissive: meta.emissive, bounds }
        }
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
        vec![CaveChunk { first: 0, count: index_count, min: bounds[0], max: bounds[1], lip: 0, planes: CavePlanes::Cliff }]
    }
}

/// The runs of the index buffer to draw: the chunks some eye of `planes` can
/// see into, with neighbours in the buffer joined into one run. `None` draws
/// every chunk. `of`: only the runs of that twin (`CaveChunk::planes`).
pub fn visible_ranges(
    chunks: &[CaveChunk],
    model: Mat4,
    planes: Option<&[[Vec4; 6]]>,
    of: Option<CavePlanes>,
) -> Vec<std::ops::Range<u32>> {
    let mut out: Vec<std::ops::Range<u32>> = Vec::new();
    for c in chunks {
        if c.count == 0 || of.is_some_and(|k| c.planes != k) {
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
    /// For the triangles of which class. `Cliff`, the default, draws any.
    pub planes: CavePlanes,
}

/// The emissive term: the rock's colour times the paint's light, which the
/// vertex stage has already opened and scaled. The same in every twin.
fn emissive_line(_planes: CavePlanes) -> String {
    "    colour = colour + albedo * in.glow_rgb;\n".to_string()
}

/// WHAT ONLY THE EMISSIVE TWIN CARRIES, each a whole piece of text that a
/// cave with no glow is compiled without: the function that opens the
/// paint's colour, its varying, the line of the vertex stage that fills it,
/// and (`emissive_line`) the line of the pixel's that adds it. The paint's
/// brightness still rides where it did (a spare component of a varying the
/// twins already have); the colour is its own three.
const GLOW_FN: &str = r#"// EMISSIVE PAINT (the emissive twin only): its brightness (`misc.y`, 0 to
// 63) times its colour -- a hue (`misc.w`) and a saturation -- and the gain,
// worked out a VERTEX. Saturation 0 is white: the rock glows in its own colour.
fn glow_light(misc: vec4<u32>, saturation: f32) -> vec3<f32> {
    let hue = f32(misc.w) / 255.0;
    let pure = clamp(abs(fract(vec3<f32>(hue) + vec3<f32>(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0) - 1.0, vec3<f32>(0.0), vec3<f32>(1.0));
    return mix(vec3<f32>(1.0), pure, saturation) * (f32(misc.y) / 63.0 * EMISSIVE_GAIN);
}

"#;
const GLOW_FIELD: &str = "    @location(9) glow_rgb: vec3<f32>,\n";
const GLOW_VERTEX: &str = "    out.glow_rgb  = glow_light(v.misc, v.cliff.w);\n";

/// A twin's vertex text with the emissive twin's varying and the line that
/// fills it put in: after the last varying, and before the stage returns.
fn with_glow(vertex: &str) -> String {
    const LAST_FIELD: &str = "    @location(8) ground: vec3<f32>,\n}";
    const RETURN: &str = "    return out;\n";
    assert_eq!(vertex.matches(LAST_FIELD).count(), 1, "the baked cave's last varying moved");
    assert_eq!(vertex.matches(RETURN).count(), 1, "the baked cave's vertex stage returns once");
    vertex
        .replacen(LAST_FIELD, &format!("    @location(8) ground: vec3<f32>,\n{GLOW_FIELD}}}"), 1)
        .replacen(RETURN, &format!("{GLOW_VERTEX}{RETURN}"), 1)
}

/// Every piece of text the emissive twin has and the plain one has not.
fn emissive_parts(planes: CavePlanes) -> [String; 4] {
    [GLOW_FN.to_string(), GLOW_FIELD.to_string(), GLOW_VERTEX.to_string(), emissive_line(planes)]
}

/// How bright full emissive paint is, against the rock's own colour.
pub const EMISSIVE_GAIN: f32 = 6.0;

/// The baked cave's shader for a twin.
pub fn baked_cave_shader(which: CaveShader) -> String {
    let mut src = full_shader(which.planes);
    if !which.emissive {
        for part in emissive_parts(which.planes) {
            assert_eq!(src.matches(&part).count(), 1, "the baked cave's emissive part moved: {part}");
            src = src.replacen(&part, "", 1);
        }
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

/// ONE PLANE (`CavePlanes::One`): the layers' coordinate is the world's on
/// the triangle's plane, worked out a vertex and interpolated, and each layer
/// is read once. This is the shader the cave shipped with on 2026-10-13 (16
/// registers, 15 without the spots: 75% occupancy), its coordinate now the
/// world's own rather than a chart's.
const ONE_VERTEX: &str = r#"struct VOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) normal: vec3<f32>,
    // In the player's frame, as every position the shaders get: lighting.
    @location(1) world_pos: vec3<f32>,
    @location(2) weights: vec4<f32>,
    // xy: the layers' coordinate, the world's on the triangle's one plane. zw: the lightmap.
    @location(3) uv: vec4<f32>,
    @location(4) aux: vec4<f32>,
    // That plane: one value a triangle.
    @location(5) @interpolate(flat) axis: u32,
    // The point's height in the WORLD (the sea's wash line), and its glow.
    @location(6) height: f32,
    @location(7) glow: f32,
    // xy: the terrain footprint's uv. z: the ground's share of the lighting.
    @location(8) ground: vec3<f32>,
}

@vertex fn vs_main(v: VIn) -> VOut {
    let world = model_u.model * vec4<f32>(v.pos, 1.0);
    let wpos = to_world_space(world.xyz);
    let axis = select(select(2u, 0u, v.planes.x >= v.planes.z), 1u, v.planes.y >= max(v.planes.x, v.planes.z));
    var out: VOut;
    out.clip      = cam_view_proj() * world;
    out.normal    = (model_u.model * vec4<f32>(v.norm, 0.0)).xyz;
    out.world_pos = world.xyz;
    out.weights   = v.weights;
    out.uv        = vec4<f32>(plane_uv(axis, wpos), v.uv1);
    out.aux       = v.aux;
    out.axis      = axis;
    out.height    = wpos.y;
    out.glow      = f32(v.misc.y) / 63.0;
    out.ground    = vec3<f32>(v.ground_uv, f32(v.misc.z) / 255.0);
    return out;
}
"#;
const ONE_LAYERS: &str = r#"    // The layers, each read ONCE, on the triangle's one plane: its gradients taken
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
    if (ground > 0.0) {
        let paint = textureSampleLevel(splat_tex, layer_samp, in.ground.xy, 0.0);
        let painted = paint / max(paint.r + paint.g + paint.b + paint.a, 0.001);
        w = mix(w, painted, ground * step(0.5, mat.use_splat));
        ground_map = textureSampleLevel(sky_occ_tex, layer_samp, in.ground.xy, 0, 0.0);
    }
    var albedo = vec3<f32>(0.0);
    var bent = vec3<f32>(0.0);
    var rough_map = 0.0;
    var ao_map = 0.0;
    // Walked as a mask, lowest layer first, each weight picked by `select`:
    // the ground's own loop (`terrain_pipeline::LAYER_READS`).
    var layers_left = select(0u, 1u, w.x > WEIGHT_EPS) | select(0u, 2u, w.y > WEIGHT_EPS)
        | select(0u, 4u, w.z > WEIGHT_EPS) | select(0u, 8u, w.w > WEIGHT_EPS);
    loop {
        if (layers_left == 0u) {
            break;
        }
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
    }
    bent = normalize(select(nw, bent, dot(bent, bent) > 0.00000001));
    let shaded_n = to_player_direction(bent);
    // The ground's macro variation, on that plane as the ground reads
    // it on a face's: without it the rock is a flat tone beside ground that
    // drifts a third lighter and darker.
    let mr = 1.0 / max(mat.macro_repeat, 0.001);
    let macro_value = textureSampleGrad(macro_tex, layer_samp, uv * mr, uv_dx * mr, uv_dy * mr).r;
    albedo = albedo * (1.0 + (macro_value - 0.5) * 2.0 * mat.macro_strength);
"#;

/// THE ROCK'S OWN PLANES (`CavePlanes::Own`): two or three by their shares,
/// read at the world's coordinates. The planes are written out one after
/// another, each with its swizzle of the one position and its gradients, so
/// nothing is copied to choose a plane and nothing indexed by a counter.
const OWN_VERTEX: &str = r#"struct VOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) normal: vec3<f32>,
    // In the player's frame, as every position the shaders get: lighting.
    @location(1) world_pos: vec3<f32>,
    @location(2) weights: vec4<f32>,
    // The lightmap.
    @location(3) uv: vec2<f32>,
    @location(4) aux: vec4<f32>,
    // xyz: the share of each plane the layers are read on (they add up to
    // one: `CavePlanes::of`). w: the glow.
    @location(5) planes: vec4<f32>,
    // The point in the WORLD: the layers' coordinates on every plane, and
    // its height (the sea's wash line).
    @location(6) wpos: vec3<f32>,
    // xy: the terrain footprint's uv. z: the ground's share of the lighting.
    @location(8) ground: vec3<f32>,
}

@vertex fn vs_main(v: VIn) -> VOut {
    let world = model_u.model * vec4<f32>(v.pos, 1.0);
    var out: VOut;
    out.clip      = cam_view_proj() * world;
    out.normal    = (model_u.model * vec4<f32>(v.norm, 0.0)).xyz;
    out.world_pos = world.xyz;
    out.weights   = v.weights;
    out.uv        = v.uv1;
    out.aux       = v.aux;
    out.planes    = vec4<f32>(v.planes.xyz, f32(v.misc.y) / 63.0);
    out.wpos      = to_world_space(world.xyz);
    out.ground    = vec3<f32>(v.ground_uv, f32(v.misc.z) / 255.0);
    return out;
}
"#;
const OWN_LAYERS_HEAD: &str = r#"    // THE LAYERS ARE READ ON THE WORLD'S PLANES, at the world's own
    // coordinates: its gradients taken here, in uniform control flow.
    let wp = in.wpos;
    let wdx = dpdx(wp);
    let wdy = dpdy(wp);
    let total = in.weights.x + in.weights.y + in.weights.z + in.weights.w;
    var w = select(vec4<f32>(0.0, 1.0, 0.0, 0.0), in.weights / max(total, 0.0001), total > 0.0001);
    // WHERE THE ROCK MEETS THE GROUND IT IS THE GROUND (see the one-plane
    // twin): the terrain's paint, its sky and its sun, behind a branch.
    let ground = clamp(in.ground.z, 0.0, 1.0);
    var ground_map = vec4<f32>(1.0, 1.0, 0.0, 1.0);
    if (ground > 0.0) {
        let paint = textureSampleLevel(splat_tex, layer_samp, in.ground.xy, 0.0);
        let painted = paint / max(paint.r + paint.g + paint.b + paint.a, 0.001);
        w = mix(w, painted, ground * step(0.5, mat.use_splat));
        ground_map = textureSampleLevel(sky_occ_tex, layer_samp, in.ground.xy, 0, 0.0);
    }
    let mr = 1.0 / max(mat.macro_repeat, 0.001);
    let layers = select(0u, 1u, w.x > WEIGHT_EPS) | select(0u, 2u, w.y > WEIGHT_EPS)
        | select(0u, 4u, w.z > WEIGHT_EPS) | select(0u, 8u, w.w > WEIGHT_EPS);
    var albedo = vec3<f32>(0.0);
    var bent = vec3<f32>(0.0);
    var rough_map = 0.0;
    var ao_map = 0.0;
    var macro_value = 0.0;
"#;
/// One plane of `Own`: `AXIS` its number, `SHARE` its component of the
/// shares, `SWZ` the swizzle that is its coordinate (`plane_uv`).
const OWN_LAYERS_PLANE: &str = r#"    if (in.planes.SHARE > WEIGHT_EPS) {
        // The ground's macro variation on this plane, by its share.
        macro_value = macro_value
            + textureSampleGrad(macro_tex, layer_samp, wp.SWZ * mr, wdx.SWZ * mr, wdy.SWZ * mr).r * in.planes.SHARE;
        // The layers, walked as a mask, lowest first: the ground's own loop.
        var layers_left = layers;
        loop {
            if (layers_left == 0u) {
                break;
            }
            let layer = i32(countTrailingZeros(layers_left));
            layers_left = layers_left & (layers_left - 1u);
            let lw = select(select(select(w.x, w.y, layer == 1), w.z, layer == 2), w.w, layer == 3) * in.planes.SHARE;
            let r = 1.0 / max(mat.repeat[layer], 0.001);
            let at = wp.SWZ * r;
            let gx = wdx.SWZ * r;
            let gy = wdy.SWZ * r;
            albedo = albedo + textureSampleGrad(layer_tex, layer_samp, at, layer, gx, gy).rgb * lw;
            // OpenGL convention: green is +Y in tangent space.
            var tn = textureSampleGrad(normal_tex, layer_samp, at, layer, gx, gy).rgb * 2.0 - 1.0;
            tn = vec3<f32>(tn.xy * mat.normal_strength, tn.z);
            bent = bent + whiteout(AXISu, tn, nw) * lw;
            rough_map = rough_map + textureSampleGrad(rough_tex, layer_samp, at, layer, gx, gy).r * lw;
            ao_map = ao_map + textureSampleGrad(ao_tex, layer_samp, at, layer, gx, gy).r * lw;
        }
    }
"#;
const OWN_LAYERS_TAIL: &str = r#"    bent = normalize(select(nw, bent, dot(bent, bent) > 0.00000001));
    let shaded_n = to_player_direction(bent);
    albedo = albedo * (1.0 + (macro_value - 0.5) * 2.0 * mat.macro_strength);
"#;

/// EVERYTHING (`CavePlanes::Cliff`): the rock's own planes by their shares
/// and a cliff's two by theirs, warped and at two scales. Draws any triangle
/// (the mirrored world and the two-eye pass draw every one with it).
const CLIFF_VERTEX: &str = r#"struct VOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) normal: vec3<f32>,
    // In the player's frame, as every position the shaders get: lighting.
    @location(1) world_pos: vec3<f32>,
    @location(2) weights: vec4<f32>,
    // The lightmap.
    @location(3) uv: vec2<f32>,
    @location(4) aux: vec4<f32>,
    // The share of each plane the layers are read on (x, y, z), and of a
    // cliff's reading.
    @location(5) planes: vec4<f32>,
    // The point in the WORLD: the layers' coordinates on every plane, and
    // its height (the sea's wash line).
    @location(6) wpos: vec3<f32>,
    // xyz: the terrain's shares of the planes at a cliff. w: the glow.
    @location(7) cliff: vec4<f32>,
    // xy: the terrain footprint's uv. z: the ground's share of the lighting.
    @location(8) ground: vec3<f32>,
}

@vertex fn vs_main(v: VIn) -> VOut {
    let world = model_u.model * vec4<f32>(v.pos, 1.0);
    var out: VOut;
    out.clip      = cam_view_proj() * world;
    out.normal    = (model_u.model * vec4<f32>(v.norm, 0.0)).xyz;
    out.world_pos = world.xyz;
    out.weights   = v.weights;
    out.uv        = v.uv1;
    out.aux       = v.aux;
    out.planes    = v.planes;
    out.wpos      = to_world_space(world.xyz);
    out.cliff     = vec4<f32>(v.cliff.xyz, f32(v.misc.y) / 63.0);
    out.ground    = vec3<f32>(v.ground_uv, f32(v.misc.z) / 255.0);
    return out;
}
"#;
const CLIFF_LAYERS: &str = r#"    // THE LAYERS ARE READ ON THE WORLD'S PLANES, at the world's own
    // coordinates: its gradients taken here, in uniform control flow, and
    // swizzled and scaled into the loops.
    let wp = in.wpos;
    let wdx = dpdx(wp);
    let wdy = dpdy(wp);
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
    // AND WHERE THAT GROUND IS A CLIFF, its pattern is the cliff's: the
    // terrain reads a cliff on its two strongest planes, both shifted by a
    // warp the macro map steers on the stronger, the stronger again at a
    // second scale (`terrain_pipeline::sample_frame`, `layer_colour_at`).
    // The vertices carry the terrain's two shares and how much of that
    // reading the rock takes; what is read here is what the terrain reads.
    let mr = 1.0 / max(mat.macro_repeat, 0.001);
    let cliff_total = in.cliff.x + in.cliff.y + in.cliff.z;
    let cliff = clamp(in.planes.w, 0.0, 1.0) * step(0.0001, cliff_total);
    var cliff_major = 3u;
    var cliff_macro = 0.5;
    var cliff_warp = vec2<f32>(0.0);
    if (ground > 0.0) {
        let paint = textureSampleLevel(splat_tex, layer_samp, in.ground.xy, 0.0);
        let painted = paint / max(paint.r + paint.g + paint.b + paint.a, 0.001);
        w = mix(w, painted, ground * step(0.5, mat.use_splat));
        ground_map = textureSampleLevel(sky_occ_tex, layer_samp, in.ground.xy, 0, 0.0);
        if (cliff > 0.0) {
            cliff_major = select(select(2u, 0u, in.cliff.x >= in.cliff.z), 1u, in.cliff.y >= max(in.cliff.x, in.cliff.z));
            let muv = plane_uv(cliff_major, wp);
            let mdx = plane_uv(cliff_major, wdx);
            let mdy = plane_uv(cliff_major, wdy);
            cliff_macro = textureSampleGrad(macro_tex, layer_samp, muv * mr, mdx * mr, mdy * mr).r;
            let ms = 1.0 / CLIFF_BAND_SCALE_M;
            let m2 = textureSampleGrad(macro_tex, layer_samp, muv * ms + vec2<f32>(0.31, 0.57), mdx * ms, mdy * ms).r;
            cliff_warp = vec2<f32>(cliff_macro - 0.5, 0.4 - 0.8 * cliff_macro) * CLIFF_WARP_M
                + vec2<f32>(0.35, 1.0) * (m2 - 0.5) * CLIFF_BAND_WARP_M;
        }
    }
    // What is read: the rock's own planes by their shares, less the cliff's;
    // and the cliff's two by theirs. Six entries at most, one for most pixels.
    let own_total = in.planes.x + in.planes.y + in.planes.z;
    let ours = select(vec3<f32>(0.0, 1.0, 0.0), in.planes.xyz / max(own_total, 0.0001), own_total > 0.0001) * (1.0 - cliff);
    // THE CLIFF'S SHARES AS THE TERRAIN NOW TAKES THEM (`terrain_pipeline`'s
    // `sample_frame`, 2026-10-13): a plane's share is how far the surface
    // faces it past the three-way tie, not its plain part of the normal. The
    // vertices carry the plain parts; as a direction they are the normal the
    // terrain chose its planes by (exactly, where the bake wrote all three;
    // with the two strongest alone, to within the third's part).
    let cliff_faces = max(in.cliff.xyz * inverseSqrt(max(dot(in.cliff.xyz, in.cliff.xyz), 0.00000001)) - vec3<f32>(0.57735), vec3<f32>(0.0));
    let theirs = cliff_faces / max(cliff_faces.x + cliff_faces.y + cliff_faces.z, 0.0001) * cliff;
    var entries_left = select(0u, 1u, ours.x > WEIGHT_EPS) | select(0u, 2u, ours.y > WEIGHT_EPS) | select(0u, 4u, ours.z > WEIGHT_EPS)
        | select(0u, 8u, theirs.x > WEIGHT_EPS) | select(0u, 16u, theirs.y > WEIGHT_EPS) | select(0u, 32u, theirs.z > WEIGHT_EPS);
    let layers = select(0u, 1u, w.x > WEIGHT_EPS) | select(0u, 2u, w.y > WEIGHT_EPS)
        | select(0u, 4u, w.z > WEIGHT_EPS) | select(0u, 8u, w.w > WEIGHT_EPS);
    var albedo = vec3<f32>(0.0);
    var bent = vec3<f32>(0.0);
    var rough_map = 0.0;
    var ao_map = 0.0;
    var macro_value = 0.0;
    loop {
        if (entries_left == 0u) {
            break;
        }
        let entry = countTrailingZeros(entries_left);
        entries_left = entries_left & (entries_left - 1u);
        let of_cliff = entry >= 3u;
        let axis = select(entry, entry - 3u, of_cliff);
        let shares = select(ours, theirs, of_cliff);
        let share = select(select(shares.x, shares.y, axis == 1u), shares.z, axis == 2u);
        var uv = plane_uv(axis, wp);
        let uv_dx = plane_uv(axis, wdx);
        let uv_dy = plane_uv(axis, wdy);
        // The ground's macro variation, on the plane as the ground reads it
        // there: without it the rock is a flat tone beside ground that
        // drifts a third lighter and darker. A cliff's is its stronger
        // plane's, read above; its second scale is that plane's alone.
        var tone = cliff_macro;
        var second = 0.0;
        if (of_cliff) {
            // The side planes' alone, and the second scale a side plane's by
            // how far it leads, as the terrain has them.
            uv = uv + select(cliff_warp, vec2<f32>(0.0), axis == 1u);
            second = select(
                0.0,
                (0.3 + 0.3 * cliff_macro) * clamp(4.0 * (share / max(theirs.x + theirs.y + theirs.z, 0.0001)) - 2.0, 0.0, 1.0),
                axis == cliff_major && axis != 1u,
            );
        } else {
            tone = textureSampleGrad(macro_tex, layer_samp, uv * mr, uv_dx * mr, uv_dy * mr).r;
        }
        macro_value = macro_value + tone * share;
        // The layers, walked as a mask, lowest first, each weight picked by
        // `select`: the ground's own loop (`terrain_pipeline::LAYER_READS`).
        var layers_left = layers;
        loop {
            if (layers_left == 0u) {
                break;
            }
            let layer = i32(countTrailingZeros(layers_left));
            layers_left = layers_left & (layers_left - 1u);
            let lw = select(select(select(w.x, w.y, layer == 1), w.z, layer == 2), w.w, layer == 3) * share;
            let r = 1.0 / max(mat.repeat[layer], 0.001);
            let at = uv * r;
            let gx = uv_dx * r;
            let gy = uv_dy * r;
            var c = textureSampleGrad(layer_tex, layer_samp, at, layer, gx, gy).rgb;
            // OpenGL convention: green is +Y in tangent space.
            var tn = textureSampleGrad(normal_tex, layer_samp, at, layer, gx, gy).rgb * 2.0 - 1.0;
            tn = vec3<f32>(tn.xy * mat.normal_strength, tn.z);
            if (second > 0.0) {
                // The cliff's second scale: 2.3 times larger, turned 37 degrees.
                let s2 = r / 2.3;
                let at2 = CLIFF_ROT * uv * s2 + vec2<f32>(0.37, 0.11);
                let g2x = CLIFF_ROT * uv_dx * s2;
                let g2y = CLIFF_ROT * uv_dy * s2;
                c = mix(c, textureSampleGrad(layer_tex, layer_samp, at2, layer, g2x, g2y).rgb, second);
                if (axis != 1u) {
                    var t2 = textureSampleGrad(normal_tex, layer_samp, at2, layer, g2x, g2y).rgb * 2.0 - 1.0;
                    t2 = vec3<f32>(transpose(CLIFF_ROT) * (t2.xy * mat.normal_strength), t2.z);
                    tn = vec3<f32>(mix(tn.xy, t2.xy, second) * (1.0 + 0.667 * second), mix(tn.z, t2.z, second));
                }
            }
            albedo = albedo + c * lw;
            bent = bent + whiteout(axis, tn, nw) * lw;
            rough_map = rough_map + textureSampleGrad(rough_tex, layer_samp, at, layer, gx, gy).r * lw;
            ao_map = ao_map + textureSampleGrad(ao_tex, layer_samp, at, layer, gx, gy).r * lw;
        }
    }
    bent = normalize(select(nw, bent, dot(bent, bent) > 0.00000001));
    let shaded_n = to_player_direction(bent);
    albedo = albedo * (1.0 + (macro_value - 0.5) * 2.0 * mat.macro_strength);
"#;

/// A twin's vertex stage and its reading of the layers.
fn planes_text(planes: CavePlanes) -> (String, String) {
    match planes {
        CavePlanes::One => (ONE_VERTEX.to_string(), ONE_LAYERS.to_string()),
        CavePlanes::Own => {
            let mut layers = OWN_LAYERS_HEAD.to_string();
            for (axis, share, swizzle) in [("0", "x", "zy"), ("1", "y", "xz"), ("2", "z", "xy")] {
                layers += &OWN_LAYERS_PLANE.replace("AXIS", axis).replace("SHARE", share).replace("SWZ", swizzle);
            }
            (OWN_VERTEX.to_string(), layers + OWN_LAYERS_TAIL)
        }
        CavePlanes::Cliff => (CLIFF_VERTEX.to_string(), CLIFF_LAYERS.to_string()),
    }
}

fn full_shader(planes: CavePlanes) -> String {
    let (vertex, layers) = planes_text(planes);
    let (lm_uv, height) = match planes {
        CavePlanes::One => ("in.uv.zw", "in.height"),
        CavePlanes::Own | CavePlanes::Cliff => ("in.uv", "wp.y"),
    };
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
const WEIGHT_EPS: f32 = {weight_eps:?};
// The terrain's cliff reading, as `terrain_pipeline` has it: how far the
// macro map shifts a cliff's planes, the band warp and its scale, and the
// turn of the second scale. Pinned to the terrain's text by a test.
const CLIFF_WARP_M: f32 = 2.5;
const CLIFF_BAND_WARP_M: f32 = 3.0;
const CLIFF_BAND_SCALE_M: f32 = 7.0;
const CLIFF_ROT: mat2x2<f32> = mat2x2<f32>(0.8, 0.6, -0.6, 0.8);

{lights_block}
{whiteout_block}

// A point's (or a gradient's) coordinate on the plane read along `axis`:
// 0 along x (z, y), 1 along y (x, z), 2 along z (x, y), as the terrain's
// `biplanar_uv`.
fn plane_uv(axis: u32, p: vec3<f32>) -> vec2<f32> {{
    return select(select(p.zy, p.xz, axis == 1u), p.xy, axis == 2u);
}}

struct VIn {{
    @location(0) pos: vec3<f32>,
    @location(1) norm: vec3<f32>,
    @location(2) planes: vec4<f32>,
    @location(3) uv1: vec2<f32>,
    @location(4) ground_uv: vec2<f32>,
    @location(5) weights: vec4<f32>,
    @location(6) aux: vec4<f32>,
    @location(7) misc: vec4<u32>,
    @location(8) cliff: vec4<f32>,
}}

{glow_fn}{vertex}
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

{layers}
    // THE LIGHTMAP: rgb the baked lamps and their bounce, ADDED; alpha the
    // sky this texel sees. The vertices carry the editor's measure of the
    // same sky, and THE MEAN of the two is taken. They are two measures of
    // one thing with errors of their own: traced on the shipped cave's front
    // collar (2026-10-13, 157 places, 384 rays each) the sky really seen was
    // 0.469; the vertices said 0.440, the lightmap 0.507, the lesser of the
    // two 0.428 and their mean 0.474 -- with half the scatter of either
    // (rms 0.035 against 0.066 and 0.077). (Multiplied, a wall that sees a
    // fifth of the sky by both counts would be lit by a twenty-fifth.)
    // An unbaked cave binds the neutral lightmap, one texel of black and
    // one, and is lit by what its vertices measured alone.
    let baked = textureSample(lm_tex, lm_samp, {lm_uv});
    // The baked lamps are in this atlas. See `receiver_skips_baked`.
    receiver_skips_baked = true;
    // The stationary lamps' shadows, as every model takes them.
    let st_layers = textureNumLayers(lm_stationary);
    let st_0 = textureSample(lm_stationary, lm_sun_samp, {lm_uv}, 0);
    var st_1 = vec4<f32>(1.0);
    var st_2 = vec4<f32>(1.0);
    var st_3 = vec4<f32>(1.0);
    if (st_layers > 1u) {{
        st_1 = textureSample(lm_stationary, lm_sun_samp, {lm_uv}, 1);
    }}
    if (st_layers > 2u) {{
        st_2 = textureSample(lm_stationary, lm_sun_samp, {lm_uv}, 2);
    }}
    if (st_layers > 3u) {{
        st_3 = textureSample(lm_stationary, lm_sun_samp, {lm_uv}, 3);
    }}
    set_stationary_masks_exact(st_0, st_1, st_2, st_3, STATIONARY_MASK_DISTANCE_TEXELS);

    let vertex_sky = clamp(in.aux.x, 0.0, 1.0);
    var sky = select(vertex_sky, 0.5 * (vertex_sky + baked.a), textureDimensions(lm_tex).x > 1u);
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
    let washed = 1.0 - smoothstep(mat.wet_line, mat.wet_line + mat.wet_band, {height});
    albedo = albedo * (1.0 - 0.45 * washed * mix(smoothstep(0.5, 0.8, nw.y), 1.0, ground));

    receiver_sun_mask = sun;

    // The ground's own shading path, so rock and the terrain it grows out of
    // take the same light the same way.
    var colour = shade_material_env(
        in.world_pos, shaded_n, clamp(rough_map, 0.0, 1.0), clamp(ao_map, 0.0, 1.0), sky,
        baked.rgb * own, vec4<f32>(0.5, 0.5, 0.5, 0.0), albedo, in.world_pos, n,
    );
{emissive}    return vec4<f32>(tonemap(colour), 1.0);
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
        weight_eps = WEIGHT_EPS,
        glow_fn = GLOW_FN,
        vertex = with_glow(&vertex),
        layers = layers,
        lm_uv = lm_uv,
        height = height,
        emissive = emissive_line(planes),
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

impl BakedCaveDraw<'_> {
    /// Whether `eye` (player frame) is inside the cave's box.
    pub fn holds(&self, eye: Vec3) -> bool {
        eye.cmpge(self.bounds.0).all() && eye.cmple(self.bounds.1).all()
    }
}

/// Every pipeline a baked cave is drawn with.
pub struct BakedCavePipelines {
    /// `[planes][emissive][spotless]`, for the single-eye scene pass: a twin
    /// a class of triangle (`CavePlanes`).
    scene: [[[RenderPipeline; 2]; 2]; 3],
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
        let twin = |planes: CavePlanes, emissive: bool, spotless: bool| {
            let label = format!(
                "baked_cave{}{}{}",
                match planes {
                    CavePlanes::One => "_one",
                    CavePlanes::Own => "_own",
                    CavePlanes::Cliff => "",
                },
                if emissive { "_emissive" } else { "" },
                if spotless { "_spotless" } else { "" },
            );
            build(CaveShader { emissive, spotless, planes }, FrontFace::Ccw, samples, ViewMode::Mono, &label)
        };
        let scene = CavePlanes::ALL
            .map(|planes| [false, true].map(|emissive| [false, true].map(|spotless| twin(planes, emissive, spotless))));
        // The two-eye pass and the mirrored world draw every triangle with
        // the twin that draws anything.
        let any = |emissive: bool| CaveShader { emissive, spotless: false, planes: CavePlanes::Cliff };
        let stereo = stereo
            .then(|| [false, true].map(|emissive| build(any(emissive), FrontFace::Ccw, samples, ViewMode::Stereo, "baked_cave_stereo")));
        let mirror = [false, true].map(|emissive| build(any(emissive), FrontFace::Cw, 1, ViewMode::Mono, "baked_cave_mirror"));
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

    /// The scene pass's pipeline for a cave's triangles of one class.
    pub fn scene_pipeline(&self, planes: CavePlanes, emissive: bool, spotless: bool) -> &RenderPipeline {
        &self.scene[planes as usize][emissive as usize][spotless as usize]
    }

    /// Every cave of the scene pass: the chunks `planes` can see into (all of
    /// them for `None`). Returns the triangles drawn. `only`: just the caves
    /// whose box holds that eye (`true`) or does not (`false`) -- the pass
    /// draws those it is inside before the ground (`Levers::cave_first`).
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
        only: Option<(Vec3, bool)>,
    ) -> u32 {
        let mut triangles = 0;
        for d in draws {
            if only.is_some_and(|(eye, inside)| d.holds(eye) != inside) {
                continue;
            }
            if let Some(p) = planes {
                if !super::scene_cull::sees_box(p, d.bounds) {
                    continue;
                }
            }
            // Both eyes in one pass: one pipeline, every run. Otherwise a
            // twin a class of triangle, each drawing its own runs.
            let both_eyes = self.stereo.as_ref().filter(|_| stereo).map(|s| &s[d.emissive as usize]);
            let passes: Vec<(&RenderPipeline, Vec<std::ops::Range<u32>>)> = match both_eyes {
                Some(pipeline) => vec![(pipeline, visible_ranges(d.chunks, d.placed, planes, None))],
                None => CavePlanes::ALL
                    .iter()
                    .map(|&k| (self.scene_pipeline(k, d.emissive, spotless), visible_ranges(d.chunks, d.placed, planes, Some(k))))
                    .collect(),
            };
            let mut bound = false;
            for (pipeline, ranges) in passes {
                if ranges.is_empty() {
                    continue;
                }
                pass.set_pipeline(pipeline);
                if !bound {
                    // One layout for every twin: bound once.
                    pass.set_bind_group(0, uniform, &[]);
                    pass.set_bind_group(1, material, &[]);
                    pass.set_bind_group(2, d.model, &[]);
                    pass.set_bind_group(3, d.lightmap, &[]);
                    pass.set_vertex_buffer(0, d.vertices.slice(..));
                    pass.set_index_buffer(d.indices.slice(..), IndexFormat::Uint32);
                    bound = true;
                }
                for r in ranges {
                    triangles += (r.end - r.start) / 3;
                    pass.draw_indexed(r, 0, 0..1);
                }
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
    /// pass (as `CaveTwins::occluder`). `planes`: only the chunks some eye of
    /// them can see into (`visible_ranges`); `None` draws each cave whole.
    pub fn draw_occluders<'a>(
        &'a self,
        pass: &mut RenderPass<'a>,
        uniform: &'a BindGroup,
        draws: &[BakedCaveDraw<'a>],
        eye: Vec3,
        planes: Option<&[[Vec4; 6]]>,
    ) {
        let inside = |d: &BakedCaveDraw| d.holds(eye);
        if !draws.iter().any(inside) {
            return;
        }
        pass.set_pipeline(&self.occluder);
        pass.set_bind_group(0, uniform, &[]);
        for d in draws.iter().filter(|d| inside(d)) {
            pass.set_bind_group(2, d.model, &[]);
            pass.set_vertex_buffer(0, d.vertices.slice(..));
            pass.set_index_buffer(d.indices.slice(..), IndexFormat::Uint32);
            for r in occluder_ranges(d.chunks, d.index_count, d.placed, planes) {
                pass.draw_indexed(r, 0, 0..1);
            }
        }
    }
}

/// The runs of a cave the depth-only passes draw: all of it for no `planes`,
/// else the chunks some eye of them can see into, whatever their twin. What
/// is left out lies wholly outside a frustum the GPU would have clipped it
/// against, so no pixel changes.
pub fn occluder_ranges(
    chunks: &[CaveChunk],
    index_count: u32,
    placed: Mat4,
    planes: Option<&[[Vec4; 6]]>,
) -> Vec<std::ops::Range<u32>> {
    match planes {
        None => vec![0..index_count],
        Some(_) => visible_ranges(chunks, placed, planes, None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunk(first: u32, count: u32, x: f32) -> CaveChunk {
        CaveChunk { first, count, min: [x, 0.0, -1.0], max: [x + 1.0, 1.0, 1.0], lip: 0, planes: CavePlanes::Cliff }
    }

    #[test]
    fn a_file_vertex_unpacks_its_weights_axis_and_glow() {
        let v = BakedCaveVertex::from_file(
            [1.0; 3], [0.0, 1.0, 0.0], [0.25, 0.5], [100, 50, 5, (2 << 6) | 40], [255, 128, 3, 9], ([0.7, 0.2], 77),
            Some(([200, 55, 0], [30, 0, 225, 90])),
            None,
        );
        assert_eq!(v.weights, [100, 50, 5, 100], "the fourth weight is what the three leave of 255");
        assert_eq!(v.misc, [2, 40, 77, 0]);
        assert_eq!(v.planes, [200, 55, 0, 90], "the planes' shares, and the cliff's");
        assert_eq!(v.cliff, [30, 0, 225, 0], "the terrain's shares at a cliff");
        // A file from before the planes: each chart on its own axis, no cliff.
        let old = BakedCaveVertex::from_file([0.0; 3], [0.0; 3], [0.0; 2], [0, 255, 0, 2 << 6], [0; 4], ([0.0; 2], 0), None, None);
        assert_eq!((old.planes, old.cliff), ([0, 0, 255, 0], [0; 4]));
        let old = BakedCaveVertex::from_file([0.0; 3], [0.0; 3], [0.0; 2], [0, 255, 0, 0], [0; 4], ([0.0; 2], 0), None, None);
        assert_eq!(old.planes, [255, 0, 0, 0]);
        assert_eq!(v.ground_uv, [0.7, 0.2]);
        assert_eq!(v.aux, [255, 128, 3, 9]);
        assert_eq!(std::mem::size_of::<BakedCaveVertex>(), 60);
        // Three weights over 255 (a file by another hand) leave nothing, not a wrapped byte.
        assert_eq!(BakedCaveVertex::from_file([0.0; 3], [0.0; 3], [0.0; 2], [200, 100, 0, 0], [0; 4], ([0.0; 2], 0), None, None).weights[3], 0);
    }

    #[test]
    fn a_glows_colour_rides_in_the_two_spare_bytes_and_opens_to_the_same_colour() {
        // The paint's colour (COLOR_4) beside everything else the vertex had: nothing else moves.
        let args = ([1.0; 3], [0.0, 1.0, 0.0], [0.25, 0.5], [100, 50, 5, (2 << 6) | 40], [255, 128, 3, 9], ([0.7, 0.2], 77), Some(([200, 55, 0], [30, 0, 225, 90])));
        let plain = BakedCaveVertex::from_file(args.0, args.1, args.2, args.3, args.4, args.5, args.6, None);
        let teal = BakedCaveVertex::from_file(args.0, args.1, args.2, args.3, args.4, args.5, args.6, Some([60, 220, 200, 255]));
        assert_eq!((plain.misc[3], plain.cliff[3]), (0, 0), "no colour: the rock's own");
        assert_ne!(teal.cliff[3], 0);
        assert_eq!(BakedCaveVertex { misc: [2, 40, 77, 0], cliff: [30, 0, 225, 0], ..teal }, plain, "the colour is in the two spare bytes and nowhere else");
        // White, grey and black have no colour of their own.
        for grey in [[255, 255, 255], [128, 128, 128], [0, 0, 0]] {
            assert_eq!(glow_hue_saturation(grey), (0, 0));
        }
        // What the shader opens is the colour that was painted, at full brightness, to a hundredth.
        for rgb in [[255u8, 0, 0], [0, 255, 0], [0, 0, 255], [255, 180, 90], [60, 220, 200], [180, 140, 255], [255, 255, 0], [255, 0, 254], [255, 200, 200]] {
            let (hue, saturation) = glow_hue_saturation(rgb);
            let got = glow_tint(hue, saturation);
            let linear = rgb.map(srgb_to_linear);
            let top = linear[0].max(linear[1]).max(linear[2]);
            for k in 0..3 {
                assert!((got[k] - linear[k] / top).abs() < 0.012, "{rgb:?}: opened {got:?}, painted {:?}", linear.map(|c| c / top));
            }
        }
        // And the shader's own text is that arithmetic: the CPU copy above is checked against it by name.
        assert!(GLOW_FN.contains("abs(fract(vec3<f32>(hue) + vec3<f32>(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0) - 1.0"));
        assert!(GLOW_FN.contains("mix(vec3<f32>(1.0), pure, saturation) * (f32(misc.y) / 63.0 * EMISSIVE_GAIN)"));
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
        assert_eq!(meta.chunks[1], CaveChunk { first: 6, count: 3, min: [0.0, 0.0, 0.0], max: [4.0, 5.0, 6.0], lip: 3, planes: CavePlanes::Cliff });
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
            assert_eq!(one, vec![CaveChunk { first: 0, count, min: b[0], max: b[1], lip: 0, planes: CavePlanes::Cliff }]);
        }
    }

    /// THE DEPTH-ONLY PASSES DRAW A LONG CAVE BY WHAT IS IN VIEW, not by its
    /// length: a tunnel of 400 chunks along x, an eye inside it looking down
    /// the tunnel one way. Drawn whole (as the occluder and motion passes did)
    /// it is 400 chunks whatever the eye sees; by its chunks, fewer than half,
    /// and every triangle with a corner inside the frustum is still drawn.
    #[test]
    fn a_long_caves_depth_is_drawn_by_the_chunks_in_view_and_none_in_view_is_lost() {
        let proj = Mat4::perspective_rh(std::f32::consts::FRAC_PI_2, 1.0, 0.05, 500.0);
        let view_proj = proj * Mat4::look_to_rh(Vec3::new(0.0, 0.0, 0.0), Vec3::X, Vec3::Y);
        let planes = [crate::renderer::shadow::frustum_planes(view_proj)];
        let placed = Mat4::from_translation(Vec3::new(0.0, -1.0, 0.0));
        // Chunk k: 2 m of tunnel from x = 2k - 400, one triangle in its box.
        let chunks: Vec<CaveChunk> = (0..400u32)
            .map(|k| {
                let x = 2.0 * k as f32 - 400.0;
                CaveChunk { first: 3 * k, count: 3, min: [x, 0.0, -1.5], max: [x + 2.0, 2.5, 1.5], lip: 0, planes: CavePlanes::ALL[k as usize % 3] }
            })
            .collect();
        let whole = occluder_ranges(&chunks, 1200, placed, None);
        assert_eq!(whole, vec![0..1200]);
        let seen = occluder_ranges(&chunks, 1200, placed, Some(&planes));
        let drawn: u32 = seen.iter().map(|r| r.end - r.start).sum();
        assert!(drawn < 1200 / 2 + 12, "{drawn} of 1200 indices");
        assert!(drawn >= 3 * 190, "{drawn}: the half ahead is in view");
        // No chunk with a point inside the frustum is left out.
        for c in &chunks {
            let centre = placed.transform_point3((Vec3::from(c.min) + Vec3::from(c.max)) * 0.5);
            let inside = planes[0].iter().all(|p| p.truncate().dot(centre) + p.w >= 0.0);
            if inside {
                assert!(seen.iter().any(|r| r.contains(&c.first)), "chunk at {} is in view and not drawn", c.first);
            }
        }
    }

    #[test]
    fn only_the_chunks_an_eye_can_see_into_are_drawn_and_neighbours_join() {
        // An eye at the origin looking down -z: chunks at z -6..-4, spread along x.
        let proj = Mat4::perspective_rh(std::f32::consts::FRAC_PI_2, 1.0, 0.05, 500.0);
        let planes = [crate::renderer::shadow::frustum_planes(proj * Mat4::look_to_rh(Vec3::ZERO, Vec3::NEG_Z, Vec3::Y))];
        let at = |first: u32, x: f32| CaveChunk { first, count: 3, min: [x, -0.5, -6.0], max: [x + 1.0, 0.5, -4.0], lip: 0, planes: CavePlanes::Cliff };
        let chunks = [at(0, -40.0), at(3, -1.0), at(6, 0.5), at(9, 40.0), at(12, 2.0)];
        assert_eq!(visible_ranges(&chunks, Mat4::IDENTITY, Some(&planes), None), vec![3..9, 12..15]);
        assert_eq!(visible_ranges(&chunks, Mat4::IDENTITY, None, None), vec![0..15]);
        // Placed elsewhere, the boxes move with the model.
        let moved = Mat4::from_translation(Vec3::new(40.0, 0.0, 0.0));
        assert_eq!(visible_ranges(&chunks, moved, Some(&planes), None), vec![0..3]);
        // Behind the eye: nothing.
        assert!(visible_ranges(&chunks, Mat4::from_translation(Vec3::new(0.0, 0.0, 20.0)), Some(&planes), None).is_empty());
    }

    #[test]
    fn every_twin_validates_and_differs_only_as_it_says() {
        for planes in CavePlanes::ALL {
            let full = baked_cave_shader(CaveShader { emissive: true, spotless: false, planes });
            let plain = baked_cave_shader(CaveShader { emissive: false, spotless: false, planes });
            // A cave with no glow has the same shader less the emissive twin's four pieces,
            // and no trace of them: not the varying, not the function, not the line.
            let mut less = full.clone();
            for part in emissive_parts(planes) {
                assert_eq!(full.matches(&part).count(), 1, "{planes:?}: {part}");
                assert!(!plain.contains(&part), "{planes:?}: the plain twin carries {part}");
                less = less.replacen(&part, "", 1);
            }
            assert_eq!(less, plain, "a cave with no glow has the same shader less the emissive pieces");
            assert!(!plain.contains("glow_rgb") && !plain.contains("glow_light") && !plain.contains("@location(9)"));
            assert!(full.contains("@location(9) glow_rgb: vec3<f32>,") && full.contains("out.glow_rgb  = glow_light(v.misc, v.cliff.w);"));
            for emissive in [false, true] {
                for spotless in [false, true] {
                    for view in [ViewMode::Mono, ViewMode::Stereo] {
                        let what = format!("baked cave ({planes:?}, emissive {emissive}, spotless {spotless}, {view:?})");
                        let src = view.shader(baked_cave_shader(CaveShader { emissive, spotless, planes }));
                        let module = naga::front::wgsl::parse_str(&src).unwrap_or_else(|e| panic!("{what}: {}", e.emit_to_string(&src)));
                        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
                            .validate(&module)
                            .unwrap_or_else(|e| panic!("{what} does not validate: {e:?}"));
                    }
                }
            }
        }
        let occ = occluder_shader();
        let module = naga::front::wgsl::parse_str(&occ).unwrap_or_else(|e| panic!("{}", e.emit_to_string(&occ)));
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all()).validate(&module).unwrap();
    }

    #[test]
    fn the_shader_reads_each_layer_once_a_plane_and_takes_no_shadow_tap_of_the_levels_sun() {
        let src = baked_cave_shader(CaveShader::default());
        let fs = &src[src.find("@fragment fn fs_main").unwrap()..];
        // In the layer loop colour, normal, roughness and occlusion, and a
        // cliff's second scale of colour and normal behind its own branch; the
        // macro variation a plane; a cliff's two macro reads behind the
        // ground's branch; the lightmap and the stationary masks; and the
        // ground's maps behind that branch. One loop body, whatever the planes.
        assert_eq!(fs.matches("textureSampleGrad(").count(), 9);
        let second = fs.find("if (second > 0.0) {").expect("the cliff's second scale is behind a branch");
        assert_eq!(fs[second..].matches("textureSampleGrad(").count(), 4, "two reads behind it, and roughness and occlusion after");
        assert_eq!(fs.matches("textureSample(").count(), 5);
        assert_eq!(fs.matches("textureSampleLevel(").count(), 2);
        let branch = fs.find("if (ground > 0.0) {").expect("the ground's maps are read behind a branch");
        let end = branch + fs[branch..].find("\n    }\n").unwrap();
        for read in ["textureSampleLevel(sky_occ_tex", "textureSampleLevel(splat_tex", "m2 = textureSampleGrad(macro_tex"] {
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
        /// The shares of the planes the layers are read on (x, y, z), and of a cliff's reading.
        planes: [u8; 4],
        /// The terrain's shares of the planes at a cliff.
        cliff: [u8; 3],
        /// Where the surface is in the world: what a layer reads there.
        world: [f32; 3],
        aux: [u8; 4],
        glow: u8,
        /// The emissive paint's colour (`COLOR_4`), or none: the rock's own.
        glow_colour: Option<[u8; 3]>,
        emissive: bool,
        spotless: bool,
        /// The lightmap's texel (a 2 x 2 lightmap of it: a baked one), or with
        /// `unbaked` the one neutral texel a cave with no bake binds.
        lightmap: [u8; 4],
        unbaked: bool,
        mirror: bool,
        /// Drawn by the twin that draws anything, whatever its class.
        any_twin: bool,
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
    fn copy_of(s: &Shot) -> Shot {
        Shot {
            weights: s.weights, planes: s.planes, cliff: s.cliff, world: s.world, aux: s.aux, glow: s.glow, glow_colour: s.glow_colour, emissive: s.emissive,
            spotless: s.spotless, lightmap: s.lightmap, unbaked: s.unbaked, mirror: s.mirror, any_twin: s.any_twin, ground: s.ground,
            ground_map: s.ground_map, paint: s.paint, macro_strength: s.macro_strength, macro_value: s.macro_value,
        }
    }
    impl Default for Shot {
        fn default() -> Self {
            Self {
                weights: [0, 255, 0, 0],
                planes: [0, 0, 255, 0],
                cliff: [0; 3],
                world: [0.0; 3],
                aux: [255, 255, 0, 0],
                glow: 0,
                glow_colour: None,
                emissive: false,
                spotless: false,
                lightmap: [0, 0, 0, 255],
                unbaked: false,
                mirror: false,
                any_twin: false,
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
        let lightmap = if shot.unbaked {
            crate::renderer::mesh::create_lightmap_texture(&device, &queue, &mesh.lightmap_layout, &[0, 0, 0, 255], 1, 1, None)
        } else {
            crate::renderer::mesh::create_lightmap_texture(&device, &queue, &mesh.lightmap_layout, &shot.lightmap.repeat(4), 2, 2, None)
        };
        let world = Vec3::from(shot.world);
        let model = mesh.create_model_uniform(&device);
        model.upload(&queue, Mat4::from_translation(world));
        // The eye goes with the surface, so only WHERE it is changes.
        uniforms.upload(&queue, Mat4::from_translation(-world), TEST_EYE + world, &ShadowUpload::disabled());

        let (hue, saturation) = shot.glow_colour.map_or((0, 0), glow_hue_saturation);
        let v = |p: [f32; 3]| BakedCaveVertex {
            position: p,
            // Facing the eye (+z toward it in clip space is away; the shader
            // faces the normal to the viewer either way).
            normal: [0.0, 0.3, -0.954],
            planes: shot.planes,
            cliff: [shot.cliff[0], shot.cliff[1], shot.cliff[2], saturation],
            uv1: [0.5, 0.5],
            ground_uv: [0.5, 0.5],
            weights: shot.weights,
            aux: shot.aux,
            misc: [2, shot.glow, shot.ground, hue],
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
        let planes = if shot.any_twin { CavePlanes::Cliff } else { CavePlanes::of([&verts[0], &verts[1], &verts[2]]) };
        let chunks = [CaveChunk { first: 0, count: 3, min: [-1.0, -1.0, 0.0], max: [3.0, 3.0, 1.0], lip: 0, planes }];
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
                let drawn = pipelines.draw_scene(&mut pass, &uniforms.bind_group, &material.bind_group, &draws, None, shot.spotless, false, None);
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
    fn a_layer_is_read_at_the_worlds_coordinate_on_its_plane() {
        // Layer 1 is the ramp: red rises with u, green with v, a tile a metre.
        let at = |planes: [u8; 4], world: [f32; 3]| render(Shot { planes, world, ..Default::default() });
        // Along z: u is the world's x, v its y, and its z moves nothing.
        let z = [0, 0, 255, 0];
        let Some(a) = at(z, [0.1, 0.1, 0.0]) else { return };
        let b = at(z, [0.7, 0.1, 0.0]).unwrap();
        let c = at(z, [0.1, -0.3, 0.0]).unwrap();
        let d = at(z, [0.1, 0.1, 0.4]).unwrap();
        assert!(b[0] > a[0] + 15 && b[1].abs_diff(a[1]) < 12, "x moves the read across: {a:?} {b:?}");
        assert!(a[1] > c[1] + 10 && c[0].abs_diff(a[0]) < 12, "y moves it down: {a:?} {c:?}");
        for k in 0..3 {
            assert!(d[k].abs_diff(a[k]) <= 3, "along the plane's own axis nothing moves: {a:?} {d:?}");
        }
        // Along x: u is the world's z, v its y. Along y: u its x, v its z.
        let x = [255, 0, 0, 0];
        let (xa, xb, xc) = (at(x, [0.0, 0.1, -0.4]).unwrap(), at(x, [0.0, 0.1, 0.2]).unwrap(), at(x, [0.4, 0.1, -0.4]).unwrap());
        assert!(xb[0] > xa[0] + 15 && xb[1].abs_diff(xa[1]) < 12, "{xa:?} {xb:?}");
        assert!((0..3).all(|k| xc[k].abs_diff(xa[k]) <= 3), "{xa:?} {xc:?}");
        let y = [0, 255, 0, 0];
        let (ya, yb, yc) = (at(y, [0.1, 0.0, -0.4]).unwrap(), at(y, [0.1, 0.0, 0.2]).unwrap(), at(y, [0.1, 0.4, -0.4]).unwrap());
        assert!(yb[1] > ya[1] + 10 && yb[0].abs_diff(ya[0]) < 12, "{ya:?} {yb:?}");
        assert!((0..3).all(|k| yc[k].abs_diff(ya[k]) <= 3), "{ya:?} {yc:?}");
    }

    #[test]
    fn two_planes_blend_by_their_shares_and_a_plane_with_no_share_is_not_read() {
        // Somewhere the x plane and the z plane read different texels of the ramp.
        let world = [0.1, 0.3, 0.3];
        let Some(x) = render(Shot { planes: [255, 0, 0, 0], world, ..Default::default() }) else { return };
        let z = render(Shot { planes: [0, 0, 255, 0], world, ..Default::default() }).unwrap();
        let half = render(Shot { planes: [128, 0, 127, 0], world, ..Default::default() }).unwrap();
        let mostly_z = render(Shot { planes: [26, 0, 229, 0], world, ..Default::default() }).unwrap();
        assert!(x[0].abs_diff(z[0]) > 25, "the two planes read apart here: {x:?} {z:?}");
        let (lo, hi) = (x[0].min(z[0]), x[0].max(z[0]));
        assert!(half[0] > lo + 6 && half[0] < hi - 6, "half and half lies between: {x:?} {half:?} {z:?}");
        assert!(mostly_z[0].abs_diff(z[0]) < half[0].abs_diff(z[0]), "a tenth of x is nearer z: {mostly_z:?} {half:?} {z:?}");
        // No shares at all (a damaged file): the top plane, not black.
        let none = render(Shot { planes: [0; 4], world, ..Default::default() }).unwrap();
        assert!(none[0] as u32 + none[1] as u32 > 30, "{none:?}");
    }

    #[test]
    fn at_a_cliff_the_rock_takes_the_cliffs_reading_by_its_share() {
        // The macro map's one texel steers the warp: a constant shift of the
        // read, so the ramp is read somewhere else; a flat layer is the same
        // colour either way.
        let ramp = |cliff_share: u8, ground: u8| {
            render(Shot { planes: [0, 0, 255, cliff_share], cliff: [0, 0, 255], ground, macro_value: 200, ..Default::default() })
        };
        let Some(own) = ramp(0, 255) else { return };
        let cliff = ramp(255, 255).unwrap();
        let half = ramp(128, 255).unwrap();
        let no_ground = ramp(255, 0).unwrap();
        assert!(cliff[0].abs_diff(own[0]) > 25 || cliff[1].abs_diff(own[1]) > 25, "the warp moves the cliff's read: {own:?} {cliff:?}");
        for k in 0..2 {
            let (lo, hi) = (own[k].min(cliff[k]), own[k].max(cliff[k]));
            assert!(half[k] + 4 >= lo && half[k] <= hi + 4, "half the cliff's share lies between: {own:?} {half:?} {cliff:?}");
        }
        assert!((0..3).all(|k| no_ground[k].abs_diff(own[k]) <= 3), "none of it away from the ground: {own:?} {no_ground:?}");
        // No terrain shares to read it by: none of it.
        let unshared = render(Shot { planes: [0, 0, 255, 255], cliff: [0; 3], ground: 255, macro_value: 200, ..Default::default() }).unwrap();
        assert!((0..3).all(|k| unshared[k].abs_diff(own[k]) <= 3), "{own:?} {unshared:?}");
        let flat = |cliff_share: u8| {
            render(Shot { weights: [255, 0, 0, 0], planes: [0, 0, 255, cliff_share], cliff: [77, 0, 178], ground: 255, macro_value: 200, ..Default::default() })
                .unwrap()
        };
        let (a, b) = (flat(0), flat(255));
        assert!((0..3).all(|k| a[k].abs_diff(b[k]) <= 3), "a flat layer is one colour however it is read: {a:?} {b:?}");
    }

    /// The cliff's constants are the terrain's own: the two shaders are read
    /// side by side where rock meets cliff, and a constant that drifted would
    /// show as a line there.
    #[test]
    fn the_cliffs_reading_is_the_terrains_text() {
        let cave = baked_cave_shader(CaveShader::default());
        let ground = crate::renderer::terrain_pipeline::terrain_shader_built(crate::renderer::terrain_pipeline::TerrainRole::Scene, true);
        for both in [
            "const CLIFF_WARP_M: f32 = 2.5;",
            "const CLIFF_BAND_WARP_M: f32 = 3.0;",
            "const CLIFF_BAND_SCALE_M: f32 = 7.0;",
            "mat2x2<f32>(0.8, 0.6, -0.6, 0.8)",
            "vec2<f32>(0.31, 0.57)",
            "vec2<f32>(0.37, 0.11)",
            "vec2<f32>(0.35, 1.0)",
            "0.4 - 0.8 * ",
            "0.3 + 0.3 * ",
            "2.3",
            "* (1.0 + 0.667 * ",
            " - 2.0, 0.0, 1.0)",
            "0.57735",
        ] {
            assert!(cave.contains(both), "the cave's cliff reading lost `{both}`");
            assert!(ground.contains(both), "the terrain's cliff reading no longer has `{both}`: bring the cave's along");
        }
    }

    #[test]
    fn the_lightmap_adds_its_light_and_its_alpha_takes_the_sky_away() {
        let w = Shot { weights: [0, 0, 0, 255], ..Default::default() };
        let plain = shot!(Shot { ..w });
        let lit = shot!(Shot { lightmap: [200, 60, 20, 255], weights: [0, 0, 0, 255], ..Default::default() });
        assert!(lit[0] > plain[0] + 20, "baked light is added: {plain:?} {lit:?}");
        assert!(lit[0] as i32 - lit[2] as i32 > plain[0] as i32 - plain[2] as i32 + 15, "...in its colour: {plain:?} {lit:?}");
        // The sky is the MEAN of the texel's measure and the vertex's: either
        // saying none takes half of it away, both take it all.
        let texel = shot!(Shot { lightmap: [0, 0, 0, 0], weights: [0, 0, 0, 255], ..Default::default() });
        let vertex = shot!(Shot { aux: [0, 255, 0, 0], weights: [0, 0, 0, 255], ..Default::default() });
        let both = shot!(Shot { lightmap: [0, 0, 0, 0], aux: [0, 255, 0, 0], weights: [0, 0, 0, 255], ..Default::default() });
        assert!(texel[1] + 8 < plain[1] && vertex[1] + 8 < plain[1], "either measure alone darkens: {plain:?} {texel:?} {vertex:?}");
        assert!(texel[1].abs_diff(vertex[1]) <= 3, "...and by the same: {texel:?} {vertex:?}");
        assert!(both[1] + 8 < texel[1] && both[1] + 20 < plain[1], "a place neither sees sky from takes none of its light: {plain:?} {texel:?} {both:?}");
        // An unbaked cave (the neutral lightmap, one texel) is lit by its vertices alone.
        let unbaked_open = shot!(Shot { unbaked: true, weights: [0, 0, 0, 255], ..Default::default() });
        let unbaked_shut = shot!(Shot { unbaked: true, aux: [0, 255, 0, 0], weights: [0, 0, 0, 255], ..Default::default() });
        assert!((0..3).all(|k| unbaked_open[k].abs_diff(plain[k]) <= 2), "{plain:?} {unbaked_open:?}");
        assert!((0..3).all(|k| unbaked_shut[k].abs_diff(both[k]) <= 2), "{both:?} {unbaked_shut:?}");
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
        let buried = shot!(Shot { weights: w, aux: [0, 0, 0, 0], lightmap: [0, 0, 0, 0], ground: 0, ground_map: open, ..Default::default() });
        let lifted = shot!(Shot { weights: w, aux: [0, 0, 0, 0], lightmap: [0, 0, 0, 0], ground: 255, ground_map: open, ..Default::default() });
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
    fn a_glow_takes_its_paints_colour_in_every_class_of_triangle_and_none_in_the_plain_twin() {
        // Grey rock (a layer whose albedo is even), in the dark, glowing half way.
        let w = [0, 0, 0, 255];
        let classes: [(&str, [u8; 4], [u8; 3]); 3] = [
            ("one plane", [0, 0, 255, 0], [0; 3]),
            ("its own planes", [60, 195, 0, 0], [0; 3]),
            ("a cliff's", [0, 55, 200, 128], [0, 100, 155]),
        ];
        for (what, planes, cliff) in classes {
            let dark = || Shot { weights: w, planes, cliff, world: [0.6, 0.3, 0.2], lightmap: [0, 0, 0, 0], glow: 24, ..Default::default() };
            let none = shot!(Shot { ..dark() });
            let white = shot!(Shot { emissive: true, ..dark() });
            let red = shot!(Shot { emissive: true, glow_colour: Some([255, 0, 0]), ..dark() });
            let blue = shot!(Shot { emissive: true, glow_colour: Some([0, 0, 255]), ..dark() });
            let also_white = shot!(Shot { emissive: true, glow_colour: Some([255, 255, 255]), ..dark() });
            // No colour and white are the same glow: the rock's own colour, as before the colour existed.
            assert_eq!(white, also_white, "{what}");
            assert!(white[0] > none[0] + 20 && white[2] > none[2] + 20, "{what}: {none:?} {white:?}");
            // Red paint lights the rock red and blue paint blue (the tone map lets a little of a
            // strong colour into the other channels, so these are comparisons, not equalities).
            assert!(red[0] > none[0] + 20 && red[0] > red[1] + 60 && red[0] > red[2] + 60, "{what}: red {red:?}, none {none:?}");
            assert!(blue[2] > none[2] + 20 && blue[2] > blue[0] + 60 && blue[2] > blue[1] + 60, "{what}: blue {blue:?}, none {none:?}");
            assert!(red[1] + 40 < white[1] && blue[0] + 40 < white[0], "{what}: red {red:?}, blue {blue:?}, white {white:?}");
            // And a painted colour does nothing at all in the twin a cave with no glow draws with.
            let plain_red = shot!(Shot { glow_colour: Some([255, 0, 0]), ..dark() });
            assert_eq!(plain_red, none, "{what}");
        }
    }

    #[test]
    fn the_spotless_and_mirror_twins_draw_the_same_rock() {
        let base = shot!(Shot { world: [0.3, 0.6, 0.0], ..Default::default() });
        let spotless = shot!(Shot { world: [0.3, 0.6, 0.0], spotless: true, ..Default::default() });
        let mirror = shot!(Shot { world: [0.3, 0.6, 0.0], mirror: true, ..Default::default() });
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
    fn corner(planes: [u8; 4]) -> BakedCaveVertex {
        BakedCaveVertex { planes, ..Zeroable::zeroed() }
    }

    #[test]
    fn a_triangle_goes_to_the_twin_that_reads_what_it_reads() {
        let (x, z) = (corner([255, 0, 0, 0]), corner([0, 0, 255, 0]));
        assert_eq!(CavePlanes::of([&x, &x, &x]), CavePlanes::One);
        assert_eq!(CavePlanes::of([&z, &z, &z]), CavePlanes::One);
        // A share the shader does not read (one part in 255 is under its
        // threshold) is not a second plane.
        assert_eq!(CavePlanes::of([&x, &corner([254, 1, 0, 0]), &x]), CavePlanes::One);
        // One plane each, and not the same one: two planes across the triangle.
        assert_eq!(CavePlanes::of([&x, &x, &z]), CavePlanes::Own);
        assert_eq!(CavePlanes::of([&x, &corner([128, 127, 0, 0]), &x]), CavePlanes::Own);
        assert_eq!(CavePlanes::of([&corner([85, 85, 85, 0]), &x, &z]), CavePlanes::Own);
        // Any of a cliff's reading, at any corner; and shares that do not add up.
        assert_eq!(CavePlanes::of([&x, &x, &corner([255, 0, 0, 1])]), CavePlanes::Cliff);
        assert_eq!(CavePlanes::of([&x, &corner([200, 0, 0, 0]), &x]), CavePlanes::Cliff);
        assert_eq!(CavePlanes::default(), CavePlanes::Cliff, "a file's own table, and a shader asked for by no class: anything");
    }

    #[test]
    fn the_triangles_are_reordered_by_twin_and_none_is_lost() {
        // 0 1 2: one plane. 3: half and half. 4: a cliff's.
        let vertices = [corner([255, 0, 0, 0]), corner([255, 0, 0, 0]), corner([255, 0, 0, 0]), corner([128, 127, 0, 0]), corner([255, 0, 0, 9])];
        // Chunk a: own, one, cliff, and a lip (one plane by its corners: the lip is the last twin's all the same).
        // Chunk b: one, one.
        let indices = [0, 1, 3, 0, 1, 2, 0, 1, 4, 2, 1, 0, 1, 2, 0, 2, 0, 1];
        let a = CaveChunk { first: 0, count: 12, min: [0.0; 3], max: [1.0; 3], lip: 3, planes: CavePlanes::Cliff };
        let b = CaveChunk { first: 12, count: 6, min: [2.0; 3], max: [3.0; 3], lip: 0, planes: CavePlanes::Cliff };
        let (out, table) = partition(&vertices, &indices, &[a, b]);
        assert_eq!(out, vec![0, 1, 2, 1, 2, 0, 2, 0, 1, 0, 1, 3, 0, 1, 4, 2, 1, 0]);
        let run = |first, count, of: CaveChunk, lip, planes| CaveChunk { first, count, min: of.min, max: of.max, lip, planes };
        assert_eq!(
            table,
            vec![
                run(0, 3, a, 0, CavePlanes::One),
                run(3, 6, b, 0, CavePlanes::One),
                run(9, 3, a, 0, CavePlanes::Own),
                run(12, 6, a, 3, CavePlanes::Cliff),
            ]
        );
        // The same triangles: nothing lost, nothing twice.
        let mut before: Vec<[u32; 3]> = indices.chunks(3).map(|t| [t[0], t[1], t[2]]).collect();
        let mut after: Vec<[u32; 3]> = out.chunks(3).map(|t| [t[0], t[1], t[2]]).collect();
        before.sort();
        after.sort();
        assert_eq!(before, after);
        // A twin's runs are its own, joined where they are neighbours.
        assert_eq!(visible_ranges(&table, Mat4::IDENTITY, None, Some(CavePlanes::One)), vec![0..9]);
        assert_eq!(visible_ranges(&table, Mat4::IDENTITY, None, Some(CavePlanes::Own)), vec![9..12]);
        assert_eq!(visible_ranges(&table, Mat4::IDENTITY, None, Some(CavePlanes::Cliff)), vec![12..18]);
        assert_eq!(visible_ranges(&table, Mat4::IDENTITY, None, None), vec![0..18]);
        // An index past the vertices: the twin that draws anything, not a panic.
        let (out, table) = partition(&vertices, &[0, 1, 99], &[CaveChunk { first: 0, count: 3, ..a }]);
        assert_eq!((out, table[0].planes), (vec![0, 1, 99], CavePlanes::Cliff));
    }

    #[test]
    fn the_one_plane_twin_reads_each_layer_once_and_the_blended_twin_once_a_plane() {
        let fs_of = |planes| {
            let src = baked_cave_shader(CaveShader { planes, ..Default::default() });
            src[src.find("@fragment fn fs_main").unwrap()..].to_string()
        };
        // ONE: colour, normal, roughness, occlusion in the layer loop, and the macro variation. No position's
        // gradients, no shares, nothing of a cliff: the shader measured at 16 registers.
        let one = fs_of(CavePlanes::One);
        assert_eq!(one.matches("textureSampleGrad(").count(), 5);
        for absent in ["in.planes", "in.cliff", "in.wpos", "dpdx(wp)", "cliff_warp", "entries_left"] {
            assert!(!one.contains(absent), "the one-plane twin has no {absent}");
        }
        assert!(one.contains("let uv = in.uv.xy;") && one.contains("whiteout(in.axis, tn, nw)"));
        // OWN: the same five reads, written out for each of the three planes behind its share; nothing of a cliff,
        // no plane chosen by a counter.
        let own = fs_of(CavePlanes::Own);
        assert_eq!(own.matches("textureSampleGrad(").count(), 15);
        for (share, swizzle, axis) in [("x", "zy", "0u"), ("y", "xz", "1u"), ("z", "xy", "2u")] {
            let block = own.find(&format!("if (in.planes.{share} > WEIGHT_EPS) {{")).expect("a plane is read behind its share");
            let end = block + own[block..].find("\n    }\n").unwrap();
            assert_eq!(own[block..end].matches("textureSampleGrad(").count(), 5);
            assert_eq!(own[block..end].matches(&format!("wp.{swizzle} * ")).count(), 2);
            assert!(own[block..end].contains(&format!("whiteout({axis}, tn, nw)")));
        }
        for absent in ["in.cliff", "cliff_warp", "entries_left", "plane_uv("] {
            assert!(!own.contains(absent), "the blended twin has no {absent}");
        }
        // Each twin's vertex stage carries what its pixels read.
        let vs_of = |planes| {
            let src = baked_cave_shader(CaveShader { planes, ..Default::default() });
            src[src.find("struct VOut").unwrap()..src.find("@fragment fn fs_main").unwrap()].to_string()
        };
        assert!(vs_of(CavePlanes::One).contains("out.uv        = vec4<f32>(plane_uv(axis, wpos), v.uv1);"));
        assert!(vs_of(CavePlanes::Own).contains("out.planes    = vec4<f32>(v.planes.xyz, f32(v.misc.y) / 63.0);"));
    }

    #[test]
    fn each_twin_draws_what_the_twin_that_draws_anything_draws() {
        let mixed = [100, 80, 50, 25];
        let cases = [
            ("one plane, along z", Shot { world: [0.3, 0.2, 0.0], ..Default::default() }),
            ("one plane, along x", Shot { planes: [255, 0, 0, 0], world: [5.0, 0.45, 0.6], weights: mixed, ..Default::default() }),
            ("one plane, along y", Shot { planes: [0, 255, 0, 0], world: [0.35, 2.0, 0.7], weights: mixed, macro_strength: 0.8, macro_value: 40, ..Default::default() }),
            ("one plane at the ground", Shot { world: [0.6, 0.3, 0.0], ground: 200, ground_map: [90, 200, 60, 0], paint: Some([0, 0, 255, 0]), ..Default::default() }),
            ("one plane, glowing", Shot { world: [0.6, 0.3, 0.0], glow: 40, emissive: true, ..Default::default() }),
            ("two planes", Shot { planes: [128, 0, 127, 0], world: [5.2, 0.45, 0.6], weights: mixed, macro_strength: 0.8, macro_value: 200, ..Default::default() }),
            ("three planes", Shot { planes: [100, 80, 75, 0], world: [0.35, 0.8, 0.7], ..Default::default() }),
            ("two planes at the ground", Shot { planes: [0, 200, 55, 0], world: [0.6, 0.3, 0.2], ground: 128, ground_map: [90, 200, 60, 0], ..Default::default() }),
            ("two planes, glowing, spotless", Shot { planes: [60, 195, 0, 0], world: [0.6, 0.3, 0.2], glow: 40, emissive: true, spotless: true, ..Default::default() }),
        ];
        for (what, case) in cases {
            let class = CavePlanes::of([&corner(case.planes), &corner(case.planes), &corner(case.planes)]);
            assert_ne!(class, CavePlanes::Cliff, "{what} is a twin's");
            let any = shot!(Shot { any_twin: true, ..copy_of(&case) });
            let own = shot!(case);
            assert!(any[0] > 8 || any[1] > 8 || any[2] > 8, "{what}: something is drawn ({any:?})");
            assert!((0..4).all(|k| any[k].abs_diff(own[k]) <= 2), "{what}: the {class:?} twin draws {own:?}, the twin that draws anything {any:?}");
        }
    }

    #[test]
    #[ignore]
    fn baked_cave_sizes() {
        for (name, which) in [
            ("baked_cave", CaveShader { emissive: false, spotless: false, ..Default::default() }),
            ("baked_cave_spotless", CaveShader { emissive: false, spotless: true, ..Default::default() }),
            ("baked_cave_emissive", CaveShader { emissive: true, spotless: false, ..Default::default() }),
            ("baked_cave_one", CaveShader { emissive: false, spotless: false, planes: CavePlanes::One }),
            ("baked_cave_one_spotless", CaveShader { emissive: false, spotless: true, planes: CavePlanes::One }),
            ("baked_cave_own", CaveShader { emissive: false, spotless: false, planes: CavePlanes::Own }),
            ("baked_cave_own_spotless", CaveShader { emissive: false, spotless: true, planes: CavePlanes::Own }),
            ("baked_cave_one_emissive", CaveShader { emissive: true, spotless: false, planes: CavePlanes::One }),
            ("baked_cave_own_emissive", CaveShader { emissive: true, spotless: false, planes: CavePlanes::Own }),
        ] {
            let src = ViewMode::Mono.shader(baked_cave_shader(which));
            println!("{}", crate::renderer::shader_inlining::report(name, &src, "fs_main", 8));
        }
        let old = ViewMode::Mono.shader(crate::renderer::layered_mesh_pipeline::layered_mesh_shader());
        println!("{}", crate::renderer::shader_inlining::report("layered_mesh (shipped)", &old, "fs_main", 4));
    }
}
