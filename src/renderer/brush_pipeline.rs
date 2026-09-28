//! Textured level geometry: walls, floors and stairs built from brushes.
//!
//! WHY BRUSHES DO NOT RIDE THE SOLID PIPELINE
//!
//! They did, and it was the right first step -- correct geometry, correct
//! shading, one flat colour per object. What it cannot express is the thing a
//! brush is FOR. A face carries a material id and a tile scale in metres, and a
//! wall that is concrete where the author said concrete is most of what makes
//! level geometry read as a place rather than as blocking volumes.
//!
//! ONE DRAW CALL, NOT ONE PER MATERIAL
//!
//! The material index is per VERTEX and every colour map lives in one texture
//! array, so a level of twenty materials is a single draw. Sorting into a draw
//! per material would be the obvious alternative and is the wrong trade on a
//! Quest: draw calls are the scarce thing, and level geometry is exactly the
//! case with many materials and no per-object state to change between them.
//!
//! TANGENTS COME FROM THE BRUSH, NOT FROM THE TRIANGLES
//!
//! Normal mapping needs a tangent frame, and the usual way to get one is to
//! derive it from the triangles' uvs -- which is fiddly, degenerate on thin
//! triangles, and an approximation of something this data already knows
//! exactly. A brush face carries the u and v axes its uvs were generated from.
//! Taking the tangent straight from those is both cheaper and exact.

use bytemuck::{Pod, Zeroable};
use wgpu::*;

use super::lights::wgsl_lights_block;
use super::terrain_pipeline::{resample, solid_image, TerrainImage};

/// A brush vertex: geometry, its material, and the frame to light it in.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct BrushVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    /// The face's u axis. `w` carries the bitangent's handedness.
    pub tangent: [f32; 4],
    /// Texture coordinate in TILES, not in 0..1 -- the face's scale is metres
    /// per tile, so a 4m wall at scale 2 arrives here spanning 0..2 and the
    /// sampler's repeat wrapping does the rest.
    pub uv: [f32; 2],
    /// Which layer of the material array to sample.
    pub material: u32,
    /// Multiplied over the sampled colour. Carries the object's authored colour,
    /// which is the whole appearance for a face whose material is missing or
    /// unassigned -- that case binds a white layer, so the tint is all there is.
    pub tint: [f32; 4],
    /// Lightmap coordinate, 0..1 over this brush object's own baked atlas.
    ///
    /// A SECOND uv set. `uv` above is in tiles and deliberately shared between
    /// faces so brickwork lines up across a corner; lighting needs one unshared
    /// patch per face, or two walls sample each other's shadows. The layout is
    /// decided by space_soup_engine::brush_lightmap, which the baker uses too.
    pub uv2: [f32; 2],
    /// THE CENTRE OF THE FACE THIS VERTEX BELONGS TO, in world space.
    ///
    /// The same value on all three vertices of a triangle, so it interpolates
    /// to itself and arrives in the fragment shader unchanged. It exists to
    /// CHOOSE THE REFLECTION PROBE, and nothing else reads it.
    ///
    /// Why a face needs a position that is not the fragment's own: a room's
    /// probe box IS its interior, so every wall, floor and ceiling fragment
    /// lies exactly ON the box surface, where a containment test is a coin
    /// toss. Two attempts to widen the test by a margin -- a constant, then a
    /// per-pixel `fwidth` -- both failed on the headset (2026-09-19), because
    /// the fragment position at a polygon edge is not reliably inside the
    /// polygon at all: MSAA shades at the pixel centre, which can sit outside,
    /// and the interpolated position is then EXTRAPOLATED past the edge.
    ///
    /// A face's centre is immune to all of it. It is deep inside the room by
    /// construction, it is identical for every fragment of the face, so the
    /// whole face agrees on one probe and no seam can appear between
    /// neighbouring pixels. It is also strictly cheaper than what it replaces.
    pub face_centre: [f32; 3],
    /// THE BOUNDS OF THIS FACE'S OWN LIGHTMAP FOOTPRINT: (min u, min v, max u,
    /// max v) in atlas uv, identical on every vertex of the face.
    ///
    /// The fragment clamps `uv2` into it before sampling. Same reason the probe
    /// parallax clamps into its box: with MSAA a pixel on a polygon edge is
    /// shaded at its centre, which can sit outside the polygon, and the
    /// interpolated uv2 is then EXTRAPOLATED past the face's patch in the
    /// atlas -- into the gutter, or past it into a neighbouring chart.
    ///
    /// THE GUTTER CANNOT FIX THIS, and that is why it is a clamp. The lightmap
    /// has no mips (`mip_level_count: 1`), so a distant pixel spans many atlas
    /// texels and the overshoot is many texels wide; the gutter is 2. Measured
    /// on the headset in the lighting-sources view: at every sample along a
    /// ceiling seam the BAKED channel dropped 22-61 while direct barely moved
    /// (2026-09-22).
    ///
    /// The face's own uv2 bounding box rather than the chart rect, because it
    /// is strictly inside the chart and is already in hand where the vertices
    /// are built -- no layout lookup, and correct even if charts are repacked.
    pub uv2_rect: [f32; 4],
    /// HALF THE FACE'S EXTENT ALONG ITS OWN TANGENT AND BITANGENT, from
    /// `face_centre`. Identical on every vertex of the face.
    ///
    /// The fragment clamps its interpolated position into this box before
    /// taking a view direction. Same reason `uv2` is clamped into `uv2_rect`
    /// and the probe is chosen by `face_centre`: with MSAA a pixel on a
    /// polygon edge is shaded at its CENTRE, which can sit outside the
    /// polygon, and every interpolated value arrives extrapolated past the
    /// edge.
    ///
    /// WHY THIS ONE MATTERS ON A MIRROR AND NOT ON A WALL. `view_dir` comes
    /// from the fragment position, and `cos_v` from `view_dir`. Fresnel is
    /// `f0 + (f_max - f0) * pow(1 - cos_v, 5)`, whose slope near grazing is
    /// about `5 * (f_max - f0)`. On Marble020 (roughness 0.048) `f_max` is
    /// 0.952, so a `cos_v` error of 0.05 moves Fresnel by ~0.23 -- and the
    /// probe is weighted by Fresnel. On a rough face `f_max` is 0.04 and the
    /// same error moves it by 0.01, which is why the seam appears on polished
    /// surfaces only. Measured on the headset: the probe channel is 0 across
    /// the ceiling and spikes to ~26% on the seam row (2026-09-22).
    ///
    /// AND WHY THE ERROR IS BIG IN WORLD UNITS. The overshoot is about one
    /// PIXEL, but at the grazing angles near a ceiling-wall junction one pixel
    /// covers a long stretch of surface, so a sub-pixel error in screen space
    /// is a large error in metres -- and `cos_v` is smallest exactly there,
    /// where `pow(1 - cos_v, 5)` is steepest.
    ///
    /// AN EXTENT AND NOT A BOUNDING BOX, so it needs no transform. A world
    /// AABB would have to be re-derived whenever the geometry is rotated into
    /// the player's frame, and a rotated AABB is not an AABB. Two half-widths
    /// in the face's own basis are invariant under that rotation, and the
    /// basis (`normal`, `tangent`) is already transformed for other reasons.
    pub face_half_extent: [f32; 2],
}

impl BrushVertex {
    pub const ATTRIBS: [VertexAttribute; 10] = vertex_attr_array![
        0 => Float32x3, 1 => Float32x3, 2 => Float32x4, 3 => Float32x2, 4 => Uint32, 5 => Float32x4,
        6 => Float32x2, 7 => Float32x3, 8 => Float32x4, 9 => Float32x2
    ];

    pub fn layout() -> VertexBufferLayout<'static> {
        VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as BufferAddress,
            step_mode: VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

/// The most materials one level may use at once.
///
/// A limit rather than a growing array because the array is one GPU allocation
/// sized at build time, and a level that quietly exceeded it would drop the
/// materials at the end -- silently, and only for whoever authored the
/// twenty-fifth. Exceeding it logs and clamps, so it is diagnosable.
pub const MAX_BRUSH_MATERIALS: usize = 24;

pub struct BrushPipeline {
    pub pipeline: RenderPipeline,
    pub material_layout: BindGroupLayout,
    /// Layout for group 2, a brush object's baked lighting.
    pub lightmap_layout: BindGroupLayout,
}

/// Roughness assumed for a material that ships no roughness map.
///
/// 0.55 in 0..1, so a plain surface has a soft, wide highlight rather than
/// none. Deliberately not 1.0 (see `BrushMaterials::new`): 1.0 zeroes the
/// specular term outright, turning "we don't know" into "guaranteed matte".
pub const DEFAULT_ROUGHNESS: u8 = 140;

/// The neutral lightmap for a brush that has not been baked.
///
/// BLACK with an OPAQUE alpha, and neither half is a detail. The RGB is ADDED
/// to the shaded result rather than multiplied into it, so zero is the value
/// that changes nothing -- the way white is for the mesh pipeline, which
/// multiplies. The alpha is baked sky visibility and IS multiplied, so its
/// neutral is 255: an unbaked brush means "no occlusion measured", which has to
/// read as full sky rather than as a surface sealed inside a box.
///
/// The two neutrals are opposite ends of the range in the same texel, so a
/// single "neutral colour" is wrong whichever one you pick.
pub fn default_brush_lightmap(
    device: &Device,
    queue: &Queue,
    layout: &BindGroupLayout,
) -> crate::renderer::mesh::LoadedTexture {
    crate::renderer::mesh::create_lightmap_texture(
        device, queue, layout, &[0u8, 0, 0, 255], 1, 1, None,
    )
}

/// Colour and normal maps for every material a scene's brushes use.
pub struct BrushMaterials {
    pub bind_group: BindGroup,
    /// Whether ANY of these materials is smooth enough to be worth reflecting.
    ///
    /// A gate on the whole reflective pass, not a per-pixel term -- the shader
    /// still weights each fragment by its own roughness and Fresnel. This only
    /// decides whether to run the pass at all, because doing so costs a
    /// full-resolution offscreen copy of the scene and a second draw over every
    /// brush face. A level built entirely from rock and grass should pay
    /// neither, and keeps the cheaper straight-to-swapchain path.
    pub reflective: bool,
    /// `material_uv_scales`, bound at 6.
    _uv_scales: wgpu::Buffer,
}

/// Roughness below which a material is worth reflecting the scene in.
///
/// Marble in this project measures 0.048 and is unmistakably a mirror; rock and
/// gravel sit near 1.0. Anything between is a judgement, and this is set well
/// clear of both so the decision is never marginal.
/// The full mip chain for one material layer, smallest level last.
///
/// WHY THE BRUSH TEXTURES NEED THIS AT ALL
///
/// They were uploaded with `mip_level_count: 1` while the sampler asked for
/// `mipmap_filter: Linear` -- a filter with one level to choose from, which is
/// a silent no-op. A wall tiles its material several times across itself, so
/// at a grazing angle a screen pixel covers dozens of texels and samples
/// exactly one of them. The result is shimmering vertical banding down every
/// surface seen edge-on, which reads as a reflection artefact and is not one:
/// it is ordinary minification aliasing, and it survived every change to SSR
/// because SSR was never involved.
///
/// COLOUR AVERAGES IN LINEAR SPACE
///
/// An sRGB texture's bytes are not proportional to light, so averaging four of
/// them directly darkens every level -- a checkerboard of black and white
/// averages to byte 128, which is 22% grey rather than 50%. Every mip would be
/// darker than the one above it and a wall would visibly dim with distance.
/// Normal, roughness and occlusion maps are NOT sRGB and must be averaged
/// exactly as they are stored.
/// The most extra roughness-squared the normal's lost variation may add.
///
/// Uncapped, a texel whose normals point in wildly different directions goes
/// fully rough and its highlight disappears entirely. The same constant the
/// runtime path uses, so the two agree about how far this is allowed to go.
const NORMAL_VARIANCE_CLAMP: f32 = 0.18;

/// Roughness mips that carry the variation the NORMAL MAP lost when IT was
/// mipped.
///
/// WHY THIS IS A TEXTURE-PIPELINE JOB AND NOT A SHADER ONE. Mipping a normal
/// map averages its normals, and shading is not linear in the normal -- the
/// lighting of an averaged normal is not the average of the lighting. So a
/// correctly-averaged normal mip still produces a highlight that pops in and
/// out as the sampling point moves, which in a headset is the "fine detail
/// vibrating at a distance" that head tracking drives every single frame.
///
/// Toksvig's observation is that the information is not actually lost: the
/// LENGTH of the averaged (unnormalised) normal records how much the normals
/// disagreed. |Na| = 1 means they all pointed the same way; shorter means they
/// spread. That converts straight into variance, and variance is roughness.
///
/// `sigma2 = (1 - |Na|) / |Na|`, combined as `r' = sqrt(r^2 + min(2*sigma2, k))`.
///
/// ROUGHNESS VALUES DO NOT ADD. They combine in VARIANCE space, which is why
/// this squares before summing and takes the root after -- adding the two
/// roughnesses directly is a different and wrong number.
///
/// Level 0 is left EXACTLY alone: nothing has been averaged yet, so there is
/// no lost variation to restore, and a close-up surface keeps precisely the
/// roughness its author painted.
///
/// Zero runtime cost, which is the whole point of doing it here -- and unlike
/// a per-pixel derivative it is correct PER MIP LEVEL, because each level gets
/// the variance of exactly the normals that level averaged. Valve shipped this
/// approach for VR specifically (Advanced VR Rendering, GDC 2015).
///
/// KNOWN GAP: anisotropic filtering can sample a HIGHER-resolution mip than
/// the isotropic level, one that has had less correction applied, so shimmer
/// can survive at grazing angles -- documented in the same Valve talk. The
/// runtime derivative term in the brush shader covers that case and geometric
/// (non-map) normal variation, which a texture bake cannot see at all.
fn roughness_chain_with_normal_variance(
    rough: &TerrainImage,
    normal: Option<&TerrainImage>,
) -> Vec<TerrainImage> {
    let mut rough_levels = mip_chain(rough, false);
    let Some(normal) = normal else {
        return rough_levels;
    };
    if normal.width != rough.width || normal.height != rough.height {
        // Different footprints cannot be matched texel for texel, and guessing
        // a correspondence would apply one surface's variance to another's
        // roughness. Leaving it alone is the honest failure.
        return rough_levels;
    }

    // The normal map halved repeatedly in FLOAT, keeping the averaged vectors
    // UNNORMALISED -- their shortening is the entire measurement. Doing this in
    // f32 rather than re-reading the 8-bit mips avoids quantising twice, which
    // is the precision problem the runtime form of Toksvig is known for.
    let mut w = normal.width;
    let mut h = normal.height;
    let mut vecs: Vec<[f32; 3]> = (0..(w * h) as usize)
        .map(|i| {
            let px = &normal.rgba[i * 4..i * 4 + 3];
            [
                px[0] as f32 / 255.0 * 2.0 - 1.0,
                px[1] as f32 / 255.0 * 2.0 - 1.0,
                px[2] as f32 / 255.0 * 2.0 - 1.0,
            ]
        })
        .collect();

    for level in 1..rough_levels.len() {
        let (nw, nh) = (w.max(2) / 2, h.max(2) / 2);
        let mut next = vec![[0.0f32; 3]; (nw * nh) as usize];
        for y in 0..nh {
            for x in 0..nw {
                let mut acc = [0.0f32; 3];
                for (dx, dy) in [(0u32, 0u32), (1, 0), (0, 1), (1, 1)] {
                    let sx = (x * 2 + dx).min(w - 1);
                    let sy = (y * 2 + dy).min(h - 1);
                    let v = vecs[(sy * w + sx) as usize];
                    acc[0] += v[0];
                    acc[1] += v[1];
                    acc[2] += v[2];
                }
                next[(y * nw + x) as usize] = [acc[0] / 4.0, acc[1] / 4.0, acc[2] / 4.0];
            }
        }
        vecs = next;
        w = nw;
        h = nh;

        let lvl = &mut rough_levels[level];
        if lvl.width != w || lvl.height != h {
            // The two chains fell out of step; stop rather than write variance
            // into the wrong texels.
            break;
        }
        for i in 0..(w * h) as usize {
            let v = vecs[i];
            let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            // A degenerate average (normals cancelling out) would divide by
            // ~zero; clamp to the fully-spread answer instead of exploding.
            let sigma2 = if len > 1e-4 { (1.0 - len) / len } else { 1.0 };
            let kernel = (2.0 * sigma2).min(NORMAL_VARIANCE_CLAMP);
            // Only the RED channel is read for roughness, but all three are
            // written so the map stays greyscale and remains readable by eye
            // in a debugger or an exported dump.
            let base = lvl.rgba[i * 4] as f32 / 255.0;
            let filtered = (base * base + kernel).clamp(0.0, 1.0).sqrt();
            let byte = (filtered * 255.0).round().clamp(0.0, 255.0) as u8;
            lvl.rgba[i * 4] = byte;
            lvl.rgba[i * 4 + 1] = byte;
            lvl.rgba[i * 4 + 2] = byte;
        }
    }
    rough_levels
}

fn mip_chain(img: &TerrainImage, srgb: bool) -> Vec<TerrainImage> {
    let to_linear = |b: u8| {
        let c = b as f32 / 255.0;
        if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
    };
    let to_srgb = |v: f32| {
        let c = if v <= 0.0031308 { v * 12.92 } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 };
        (c.clamp(0.0, 1.0) * 255.0).round() as u8
    };

    let mut levels = vec![TerrainImage { width: img.width, height: img.height, rgba: img.rgba.clone() }];
    loop {
        let prev = levels.last().unwrap();
        if prev.width <= 1 && prev.height <= 1 {
            break;
        }
        let (w, h) = (prev.width.max(2) / 2, prev.height.max(2) / 2);
        let mut rgba = vec![0u8; (w * h * 4) as usize];
        for y in 0..h {
            for x in 0..w {
                for c in 0..4 {
                    // ALPHA is never sRGB, even in an sRGB texture: the format
                    // decodes RGB and leaves alpha linear, so encoding it here
                    // would bend a channel nothing asked to be bent.
                    let linear = srgb && c < 3;
                    let mut sum = 0.0f32;
                    for (dx, dy) in [(0u32, 0u32), (1, 0), (0, 1), (1, 1)] {
                        let sx = (x * 2 + dx).min(prev.width - 1);
                        let sy = (y * 2 + dy).min(prev.height - 1);
                        let b = prev.rgba[((sy * prev.width + sx) * 4 + c as u32) as usize];
                        sum += if linear { to_linear(b) } else { b as f32 / 255.0 };
                    }
                    let avg = sum / 4.0;
                    rgba[((y * w + x) * 4 + c as u32) as usize] =
                        if linear { to_srgb(avg) } else { (avg * 255.0).round() as u8 };
                }
            }
        }
        levels.push(TerrainImage { width: w, height: h, rgba });
    }
    levels
}

/// Diagnostic false-colour for the brush shader. MUST be false in any build
/// that ships. See `brush_shader_with`.
pub const BRUSH_LIGHT_DEBUG: bool = false;

/// Paint every brush pixel by WHERE ITS LIGHT CAME FROM.
///
///   RED    runtime lights, after their shadow test
///   GREEN  the baked lightmap's bounce
///   BLUE   the reflection probe
///
/// A second diagnostic, for a second question the screenshots could not
/// settle: light appearing in squares that correspond to no lamp. Three
/// hypotheses were eliminated from source first -- probe cell boundaries (a
/// blend did not remove them), lightmap atlas bleed (no mips, and a gutter),
/// and shadow-caster culling (it uses the LIGHT's frustum, which is right) --
/// so the next step is to look rather than guess a fourth time.
///
/// Must be `false` in anything shipped; `the_source_diagnostic_is_off` asserts
/// it.
pub const BRUSH_SOURCE_DEBUG: bool = false;

/// Paint WHERE MSAA SHADES OUTSIDE THE POLYGON.
///
///   BLACK  the pixel centre lies inside the triangle -- ordinary interior
///   RED    the centre lies outside it, so every interpolated attribute
///          arriving at this fragment is EXTRAPOLATED past the face's edge
///
/// Red scales with how far outside the centre is, measured in PIXELS: full red
/// is one pixel or more.
///
/// # Why this view exists
///
/// Nine fixes have been aimed at the thin lines along the ceiling-wall
/// junctions, and each one clamped a different attribute that this mechanism
/// corrupts -- the probe box test, the probe parallax position, `uv2`, the
/// Fresnel normal. The mechanism itself is documented on
/// `BrushVertex::face_centre` and is not in doubt; what is in doubt is whether
/// it is what the user is still looking at. Patching one attribute per round
/// cannot answer that, because there is one attribute per thing the shader
/// interpolates.
///
/// This measures the mechanism directly instead of inferring it from a value.
/// It works by interpolating the SAME varying twice -- once at the pixel
/// centre, once at the centroid of the covered samples -- and painting their
/// difference. The two agree exactly wherever the centre is covered, so any
/// red pixel is a fragment MSAA shaded from outside its own polygon.
///
/// If the seam lines come back red, the cause is settled and the fix is
/// centroid interpolation (or a per-face constant) rather than a tenth clamp.
/// If the seam is black, every extrapolation hypothesis is dead at once.
///
/// Must be `false` in anything shipped; `the_edge_diagnostic_is_off` asserts it.
pub const BRUSH_EDGE_DEBUG: bool = false;

/// What the lighting-sources view paints when it is toggled on.
///
/// The sources view is the one debug view that is a RUNTIME toggle (left
/// stick) and that works in both SSR states, because the reflective passes
/// read the scene colour back rather than re-shading. A diagnostic that
/// needs to be looked at on the headset belongs INSIDE it, not in front of
/// it: an earlier build put one in front, which replaced both variants (so
/// the stick toggled nothing) and painted black wherever SSR skipped the
/// shading (2026-09-22). `a_debug_view_changes_only_the_shader_it_is_built_for`
/// failed on that build and was misread as an expected failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourcesView {
    /// Each source's SHARE of the total: red direct, green baked, blue probe.
    /// Reads which term dominates, and cannot say whether a line where one
    /// share rises is that term going up or another going down.
    Ratios,
    /// The probe term's factors: red Fresnel / f_max, green probe.a, blue
    /// spec_occ * probe_scale. Measured smooth across the front ceiling
    /// junction (2026-09-22), which exonerated all three.
    ProbeFactors,
    /// ABSOLUTE radiance, each channel through `x / (x + 0.05)` so a dim room
    /// and a lit one both land mid-scale: red baked, green probe, blue direct.
    ///
    /// The discriminator `Ratios` cannot be: along the seam the probe's SHARE
    /// rises, which is either the probe sample going UP or the baked term --
    /// the dominant one, so the denominator -- going DOWN. Here a red dip and
    /// a green spike are different pictures.
    Absolute,
}

/// See `SourcesView`. Only reached when the sources view is toggled on, so a
/// build with this set to anything still renders normally until it is.
pub const SOURCES_VIEW: SourcesView = SourcesView::Absolute;

/// Interpolate the TWO varyings that feed the two channels measured to move at
/// a seam -- `world_pos` and `uv2` -- at the centroid of the covered samples
/// rather than at the pixel centre.
///
/// # Why two and not all nine
///
/// Centroid interpolation was tried once before, on every varying the brush
/// has, and cost ~16 ms a frame on the headset -- more than the whole 13.9 ms
/// budget, so it was reverted and `the_brush_does_not_pay_for_centroid_
/// interpolation` was left behind to keep it out. That measurement is why this
/// is not simply turned on everywhere.
///
/// It does not have to be everywhere. The lighting-sources view measured which
/// channel actually moves across a seam: direct was flat at -0.3 while baked
/// moved -31.5 and probe +24.8. Baked is addressed by `uv2`; probe parallax is
/// addressed by `world_pos`. The rest do not need it:
///
///   `normal`, `tangent`  constant over a flat brush face, so centroid and
///                        centre interpolate to the same value -- it would buy
///                        nothing and cost the same as anything else
///   `material`, `tint`,  already `flat`, which is immune by construction and
///   `face_centre`,       cheaper than either smooth mode
///   `uv2_rect`
///   `uv`                 plane-consistent and tiling: extrapolating it walks
///                        further along the same surface, which the repeat
///                        sampler handles
///
/// # MEASURED, AND IT DOES NOT FIT. THE COST IS NOT PER-VARYING (2026-09-22)
///
/// The expectation above was that two varyings would cost a small fraction of
/// what nine did. It was wrong, and wrong in the way that matters. Shipped to
/// the headset with centroid on exactly `world_pos` and `uv2`:
///
///   PERF: cpu_avg=4.16ms | gpu_avg=33.38ms gpu_max=35.94ms | frame=38.30ms
///
/// against a scene that runs in the low twenties -- so roughly the same ~16 ms
/// that centroid on ALL NINE varyings cost. Cutting nine to two bought nothing.
///
/// THE PENALTY IS A THRESHOLD, NOT A SLOPE. Asking for centroid on any varying
/// at all appears to move the whole fragment shader onto a slower interpolation
/// path, so the cost is paid once and the count is irrelevant. Anyone reading
/// the old "~16 ms" and reasoning that a smaller slice would cost less is
/// making the mistake this note exists to stop: there is no slice small enough.
///
/// So this stays `false`, and the remaining route to the same correctness is to
/// make the baked and probe terms FACE-CONSTANT the way probe selection already
/// is via `face_centre` -- `flat` is immune to extrapolation by construction
/// and is cheaper than either smooth mode, so it pays nothing at all.
pub const BRUSH_CENTROID_VARYINGS: bool = false;

/// Which diagnostic the brushes are drawn with, chosen AT RUNTIME.
///
/// The build switches (`BRUSH_SOURCE_DEBUG`, `ssr::SSR_DEBUG`) need a rebuild
/// and a deploy per question, and an artifact that only appears from one
/// position in the headset cannot be pointed at from the host. So the debug
/// pipelines are built once beside the shipped ones and swapped at draw time,
/// and the player cycles them where the artifact is.
///
/// - `Sources`: red = direct light, green = baked bounce, blue = probe, by ratio.
/// - `Ssr`: blue = too rough to march, red = ray left the frame, green = out of
///   steps, magenta = faces the viewer (not marched), grey = a hit, shaded by
///   how much of it survived the fades.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DebugView {
    #[default]
    Off,
    Sources,
    Ssr,
}

impl DebugView {
    /// The next view in the cycle, wrapping back to `Off`.
    pub fn next(self) -> Self {
        match self {
            DebugView::Off => DebugView::Sources,
            DebugView::Sources => DebugView::Ssr,
            DebugView::Ssr => DebugView::Off,
        }
    }
}

pub const SSR_ROUGHNESS_THRESHOLD: f32 = 0.35;

/// Whether a roughness map is smooth enough to reflect.
///
/// Reads the MEAN, not the minimum: a mostly-matte surface with a few polished
/// specks is not a mirror, and taking the minimum would turn the reflective
/// pass on for a wall that shows nothing.
///
/// A material with NO roughness map is fully rough -- the same default the
/// shader uses when the map is missing -- so an unauthored material never
/// silently turns the pass on.
pub fn roughness_is_reflective(rough: Option<&crate::renderer::terrain_pipeline::TerrainImage>) -> bool {
    let Some(img) = rough else { return false };
    if img.rgba.is_empty() {
        return false;
    }
    // Red channel, which is where the shader reads roughness from.
    let sum: u64 = img.rgba.chunks_exact(4).map(|p| p[0] as u64).sum();
    let n = (img.rgba.len() / 4).max(1) as f64;
    let mean = (sum as f64 / n) / 255.0;
    (mean as f32) < SSR_ROUGHNESS_THRESHOLD
}

pub fn brush_material_bind_group_layout(device: &Device) -> BindGroupLayout {
    device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("brush_material_layout"),
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
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::D2Array,
                    multisampled: false,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Sampler(SamplerBindingType::Filtering),
                count: None,
            },
            // Roughness, then ambient occlusion. Separate arrays rather than
            // channels of one packed map: the material library stores them as
            // the separate greyscale files ambientCG ships, and packing them
            // here would mean a second representation to keep in step with the
            // editor, which samples the same two files.
            BindGroupLayoutEntry {
                binding: 3,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::D2Array,
                    multisampled: false,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 4,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::D2Array,
                    multisampled: false,
                },
                count: None,
            },
            // A SECOND SAMPLER, FOR ROUGHNESS ONLY. See `brush_rough_sampler`.
            BindGroupLayoutEntry {
                binding: 5,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Sampler(SamplerBindingType::Filtering),
                count: None,
            },
            // Each material's own proportions, for the vertex stage. See
            // `material_uv_scales`.
            BindGroupLayoutEntry {
                binding: 6,
                visibility: ShaderStages::VERTEX,
                ty: BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}

/// EVERY MATERIAL AT ITS OWN PROPORTIONS, one scale a layer for the brush
/// UVs: `u` as authored, `v` times the colour map's width over its height.
///
/// A face's UVs are in square tiles (`BrushFace::scale`), and a layer spans
/// one tile whatever its shape, so a 1024x512 map -- Rock063, Bricks075A --
/// drew every stone twice as tall as it is. With `v` scaled, the tile keeps
/// its authored width and is as tall as the map says: nothing stretches. The
/// texture array's own square layers do not matter to this; they change how
/// many texels a layer has, not what shape it depicts. Missing maps are
/// square.
pub fn material_uv_scales(colours: &[Option<TerrainImage>]) -> [[f32; 4]; MAX_BRUSH_MATERIALS] {
    let mut out = [[1.0, 1.0, 0.0, 0.0]; MAX_BRUSH_MATERIALS];
    for (slot, img) in out.iter_mut().zip(colours) {
        if let Some(img) = img.as_ref().filter(|i| i.width > 0 && i.height > 0) {
            slot[1] = img.width as f32 / img.height as f32;
        }
    }
    out
}

impl BrushMaterials {
    /// Build the material arrays.
    ///
    /// `colours` and `normals` are parallel and both are padded to
    /// MAX_BRUSH_MATERIALS: every layer of a texture array shares one
    /// allocation and must match in size, so a missing or differently-sized map
    /// cannot simply be skipped. A missing colour becomes white (the vertex
    /// tint then decides the look) and a missing normal becomes flat.
    pub fn new(
        device: &Device,
        queue: &Queue,
        layout: &BindGroupLayout,
        colours: &[Option<TerrainImage>],
        normals: &[Option<TerrainImage>],
        roughs: &[Option<TerrainImage>],
        aos: &[Option<TerrainImage>],
    ) -> Self {
        // Sized from the first real map rather than from a constant, so a
        // project of 2K materials is not downsampled to someone's guess.
        let (w, h) = colours
            .iter()
            .flatten()
            .map(|i| (i.width, i.height))
            .next()
            .unwrap_or((1, 1));

        let colour_tex = Self::array_texture(
            device,
            queue,
            "brush_colour_array",
            TextureFormat::Rgba8UnormSrgb,
            colours,
            [255, 255, 255],
            w,
            h,
            None,
        );
        // NOT sRGB. A normal map is a direction, not a colour, and decoding it
        // through the sRGB curve bends every normal toward the surface -- which
        // looks like weak lighting rather than like a format mistake.
        let normal_tex = Self::array_texture(
            device,
            queue,
            "brush_normal_array",
            TextureFormat::Rgba8Unorm,
            normals,
            [128, 128, 255],
            w,
            h,
            None,
        );

        // Not sRGB: these are measurements, not colours, and decoding them
        // through the sRGB curve would skew every value toward the dark end.
        //
        // The roughness default is NOT white, and that is a deliberate change.
        // White is 1.0, fully rough, and `spec_strength = SPEC_STRENGTH *
        // (1 - r)` makes that EXACTLY ZERO -- so a material with no roughness
        // map could not produce a highlight under any light, at any angle,
        // ever. Three of the five materials in the test level were in that
        // state, which is why "the specular isn't working" was true and not a
        // matter of looking from the wrong place.
        //
        // "No measurement" is not "provably a perfect diffuser". A mid value is
        // the honest default: most real surfaces are somewhat glossy, and a
        // material that genuinely is matte can say so by shipping a map.
        // THE ONE ARRAY THAT GETS THE NORMAL-VARIANCE BAKE. Its mips carry the
        // variation the normal map lost when it was mipped, as extra roughness.
        let rough_tex = Self::array_texture(
            device, queue, "brush_rough_array", TextureFormat::Rgba8Unorm,
            roughs, [DEFAULT_ROUGHNESS, DEFAULT_ROUGHNESS, DEFAULT_ROUGHNESS], w, h,
            Some(normals),
        );
        let ao_tex = Self::array_texture(
            device, queue, "brush_ao_array", TextureFormat::Rgba8Unorm,
            aos, [255, 255, 255], w, h, None,
        );

        let sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("brush_sampler"),
            // Repeat, because the uv is in tiles: a wall covering six tiles of
            // its material arrives with uv spanning 0..6 and clamping would
            // smear the last pixel across five of them.
            address_mode_u: AddressMode::Repeat,
            address_mode_v: AddressMode::Repeat,
            address_mode_w: AddressMode::Repeat,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: MipmapFilterMode::Linear,
            // Anisotropy is what keeps a mipped wall SHARP at a grazing angle
            // rather than merely un-aliased. Without it the sampler picks a mip
            // for the worst axis and blurs the other one to match, so a wall
            // stops shimmering and goes soft instead -- trading one grazing
            // artefact for another.
            anisotropy_clamp: 8,
            ..Default::default()
        });

        // ROUGHNESS IS SAMPLED TRILINEAR, NOT ANISOTROPIC, AND THAT IS THE POINT.
        //
        // The roughness mips carry the normal map's lost variation as extra
        // roughness (see `roughness_chain_with_normal_variance`), so each mip
        // level is only correct for the amount of averaging THAT level did.
        // Anisotropic filtering breaks exactly that: at a grazing angle it
        // fetches a HIGHER-resolution mip than the isotropic level, one that
        // has had less variance folded in, so the surface goes back to
        // shimmering precisely where a glancing view makes it worst. Valve
        // documents this failure in the same talk that recommends the roughness
        // mips (Advanced VR Rendering, GDC 2015).
        //
        // Trilinear takes the mip the footprint actually asks for. The price is
        // a slightly over-smoothed roughness at grazing angles -- the highlight
        // is a little broader than it should be -- which is the lesser evil
        // against a shimmering one, and it is invisible next to the sharpness
        // that COLOUR and NORMAL keep, since those still sample at 8x.
        let rough_sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("brush_rough_sampler"),
            address_mode_u: AddressMode::Repeat,
            address_mode_v: AddressMode::Repeat,
            address_mode_w: AddressMode::Repeat,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: MipmapFilterMode::Linear,
            anisotropy_clamp: 1,
            ..Default::default()
        });

        let uv_scales = wgpu::util::DeviceExt::create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("brush_material_uv_scales"),
                contents: bytemuck::cast_slice(&material_uv_scales(colours)),
                usage: wgpu::BufferUsages::UNIFORM,
            },
        );
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("brush_materials"),
            layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&colour_tex.create_view(
                        &TextureViewDescriptor {
                            dimension: Some(TextureViewDimension::D2Array),
                            ..Default::default()
                        },
                    )),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&normal_tex.create_view(
                        &TextureViewDescriptor {
                            dimension: Some(TextureViewDimension::D2Array),
                            ..Default::default()
                        },
                    )),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::Sampler(&sampler),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::TextureView(&rough_tex.create_view(
                        &TextureViewDescriptor {
                            dimension: Some(TextureViewDimension::D2Array),
                            ..Default::default()
                        },
                    )),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: BindingResource::TextureView(&ao_tex.create_view(
                        &TextureViewDescriptor {
                            dimension: Some(TextureViewDimension::D2Array),
                            ..Default::default()
                        },
                    )),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: BindingResource::Sampler(&rough_sampler),
                },
                BindGroupEntry { binding: 6, resource: uv_scales.as_entire_binding() },
            ],
        });

        // Decided once, here, rather than per frame: the maps do not change
        // between bakes and this walks every texel of every roughness map.
        let reflective = roughs.iter().any(|r| roughness_is_reflective(r.as_ref()));
        Self { bind_group, reflective, _uv_scales: uv_scales }
    }

    /// A material array with nothing in it, for a scene that has no brushes.
    ///
    /// Bound anyway rather than left absent: an optional binding would mean two
    /// pipeline layouts and two shaders that have to agree, which is a bigger
    /// thing to keep right than one white texture.
    pub fn fallback(device: &Device, queue: &Queue, layout: &BindGroupLayout) -> Self {
        Self::new(device, queue, layout, &[], &[], &[], &[])
    }

    #[allow(clippy::too_many_arguments)]
    fn array_texture(
        device: &Device,
        queue: &Queue,
        label: &str,
        format: TextureFormat,
        images: &[Option<TerrainImage>],
        fallback: [u8; 3],
        w: u32,
        h: u32,
        // THE NORMAL MAPS, when this texture is the ROUGHNESS array.
        //
        // Passing them turns on the normal-variance bake -- see
        // `roughness_chain_with_normal_variance`. Every other array passes
        // None and keeps plain box-filtered mips. It lives here rather than in
        // a separate function because the per-layer fill/resample logic below
        // has to be applied identically to both arrays or the two chains would
        // not line up texel for texel.
        normals_for_variance: Option<&[Option<TerrainImage>]>,
    ) -> Texture {
        let filled: Vec<TerrainImage> = (0..MAX_BRUSH_MATERIALS)
            .map(|i| match images.get(i).and_then(|x| x.as_ref()) {
                Some(img) if img.width == w && img.height == h => TerrainImage {
                    width: img.width,
                    height: img.height,
                    rgba: img.rgba.clone(),
                },
                Some(img) => resample(img, w, h),
                None => solid_image(fallback, w, h),
            })
            .collect();

        // One chain per layer, built on the CPU. These textures are uploaded
        // from memory anyway, so a GPU blit pass would cost more code and more
        // frame time than a handful of box filters at load.
        let chains: Vec<Vec<TerrainImage>> = match normals_for_variance {
            None => filled.iter().map(|img| mip_chain(img, format.is_srgb())).collect(),
            Some(normals) => {
                // The normals go through the SAME fill/resample path, so layer
                // i of this array and layer i of the normals describe the same
                // material at the same resolution.
                let normals_filled: Vec<Option<TerrainImage>> = (0..MAX_BRUSH_MATERIALS)
                    .map(|i| match normals.get(i).and_then(|x| x.as_ref()) {
                        Some(img) if img.width == w && img.height == h => Some(TerrainImage {
                            width: img.width,
                            height: img.height,
                            rgba: img.rgba.clone(),
                        }),
                        Some(img) => Some(resample(img, w, h)),
                        // No normal map means no lost variation to restore.
                        None => None,
                    })
                    .collect();
                filled
                    .iter()
                    .zip(normals_filled.iter())
                    .map(|(r, n)| roughness_chain_with_normal_variance(r, n.as_ref()))
                    .collect()
            }
        };
        let mip_level_count = chains.first().map(|c| c.len() as u32).unwrap_or(1);

        let tex = device.create_texture(&TextureDescriptor {
            label: Some(label),
            size: Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: MAX_BRUSH_MATERIALS as u32,
            },
            mip_level_count,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });

        for (layer, chain) in chains.iter().enumerate() {
            for (level, img) in chain.iter().enumerate() {
                queue.write_texture(
                    TexelCopyTextureInfo {
                        texture: &tex,
                        mip_level: level as u32,
                        origin: Origin3d { x: 0, y: 0, z: layer as u32 },
                        aspect: TextureAspect::All,
                    },
                    &img.rgba,
                    TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(4 * img.width),
                        rows_per_image: Some(img.height),
                    },
                    Extent3d { width: img.width, height: img.height, depth_or_array_layers: 1 },
                );
            }
        }
        tex
    }
}

impl BrushPipeline {
    pub fn new(device: &Device, format: TextureFormat, uniform_layout: &BindGroupLayout) -> Self {
        Self::new_with_front_face(device, format, uniform_layout, FrontFace::Ccw, 1, BRUSH_SOURCE_DEBUG, crate::renderer::multiview::ViewMode::Mono,
        )
    }

    /// The scene-pass pipeline with blending OFF, for the `perf_ab` phase that
    /// measures whether declaring blending on opaque walls costs fill.
    pub fn new_multisampled_opaque(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        samples: u32,
    ) -> Self {
        Self::new_blended(
            device, format, uniform_layout, FrontFace::Ccw, samples, BRUSH_SOURCE_DEBUG, None, crate::renderer::multiview::ViewMode::Mono,
        )
    }

    /// The same, drawing BOTH EYES in one pass. See `multiview::ViewMode`.
    pub fn new_multisampled_opaque_stereo(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        samples: u32,
    ) -> Self {
        Self::new_blended(
            device, format, uniform_layout, FrontFace::Ccw, samples, BRUSH_SOURCE_DEBUG, None,
            crate::renderer::multiview::ViewMode::Stereo,
        )
    }

    /// The same for the lighting-sources diagnostic.
    pub fn new_multisampled_sources_stereo(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        samples: u32,
    ) -> Self {
        Self::new_with_front_face(
            device, format, uniform_layout, FrontFace::Ccw, samples, true, crate::renderer::multiview::ViewMode::Stereo,
        )
    }

    /// The same for the blended scene-pass brush.
    pub fn new_multisampled_stereo(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        samples: u32,
    ) -> Self {
        Self::new_with_front_face(
            device, format, uniform_layout, FrontFace::Ccw, samples, BRUSH_SOURCE_DEBUG,
            crate::renderer::multiview::ViewMode::Stereo,
        )
    }

    /// The scene-pass pipeline drawing the LIGHTING SOURCES diagnostic -- red
    /// direct, green baked, blue probe, by ratio. See [`DebugView::Sources`].
    pub fn new_multisampled_sources(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        samples: u32,
    ) -> Self {
        Self::new_with_front_face(device, format, uniform_layout, FrontFace::Ccw, samples, true, crate::renderer::multiview::ViewMode::Mono,
        )
    }

    /// See `pipeline::SolidPipeline::new_multisampled` -- a pipeline's sample
    /// count must match the pass it runs in, so a 4x eye pass needs its own.
    pub fn new_multisampled(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        samples: u32,
    ) -> Self {
        Self::new_with_front_face(device, format, uniform_layout, FrontFace::Ccw, samples, BRUSH_SOURCE_DEBUG, crate::renderer::multiview::ViewMode::Mono,
        )
    }

    /// The mirror pass draws a reflected world, which reverses every winding.
    pub fn new_mirror(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
    ) -> Self {
        Self::new_with_front_face(device, format, uniform_layout, FrontFace::Cw, 1, BRUSH_SOURCE_DEBUG, crate::renderer::multiview::ViewMode::Mono,
        )
    }

    /// The reflective variant, drawn over the blitted scene in the eye pass.
    ///
    /// FOUR bind groups exactly -- scene uniform, materials, lightmap, scene
    /// colour+depth -- which is this hardware's limit. The cuboid form binds a
    /// fifth for its own SSR camera; a brush cannot afford that, so this shares
    /// the camera already in group 0.
    pub fn new_ssr(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        ssr_scene_layout: &BindGroupLayout,
        // Must match the scene target's depth sample count.
    ) -> Self {
        Self::new_ssr_variant(
            device, format, uniform_layout, ssr_scene_layout,
            BRUSH_SOURCE_DEBUG, crate::renderer::ssr::SSR_DEBUG,
        )
    }

    /// The reflective pipeline drawing a diagnostic: the lighting sources for
    /// [`DebugView::Sources`], the SSR false colour for [`DebugView::Ssr`].
    pub fn new_ssr_debug(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        ssr_scene_layout: &BindGroupLayout,
        view: DebugView,
    ) -> Self {
        Self::new_ssr_variant(
            device, format, uniform_layout, ssr_scene_layout,
            view == DebugView::Sources, view == DebugView::Ssr,
        )
    }

    /// THE REFLECTION TRACE pipeline: same geometry, same march, but it writes
    /// radiance and confidence into the reflection buffer for the resolve to
    /// filter. See `brush_shader_modes`.
    ///
    /// `format` is the reflection buffer's, not the swapchain's -- it carries
    /// HDR radiance and wants more than 8 bits a channel.
    pub fn new_ssr_trace(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        ssr_scene_layout: &BindGroupLayout,
    ) -> Self {
        Self::new_ssr_modes(
            device, format, uniform_layout, ssr_scene_layout, false, false, SsrPath::Trace,
        )
    }

    /// The pass that puts the filtered reflection on the surface. See
    /// [`SsrPath`].
    pub fn new_ssr_composite(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        ssr_scene_layout: &BindGroupLayout,
    ) -> Self {
        Self::new_ssr_modes(
            device, format, uniform_layout, ssr_scene_layout, false, false, SsrPath::Composite,
        )
    }

    fn new_ssr_variant(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        ssr_scene_layout: &BindGroupLayout,
        sources: bool,
        ssr_debug: bool,
    ) -> Self {
        Self::new_ssr_modes(
            device, format, uniform_layout, ssr_scene_layout, sources, ssr_debug, SsrPath::Inline,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new_ssr_modes(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        ssr_scene_layout: &BindGroupLayout,
        sources: bool,
        ssr_debug: bool,
        path: SsrPath,
    ) -> Self {
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("brush_ssr_shader"),
            source: ShaderSource::Wgsl(
                brush_shader_modes(true, sources, ssr_debug, path).into(),
            ),
        });
        let material_layout = brush_material_bind_group_layout(device);
        let lightmap_layout = super::pipeline::lightmap_bind_group_layout(device);
        let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("brush_ssr_layout"),
            bind_group_layouts: &[
                Some(uniform_layout), Some(&material_layout), Some(&lightmap_layout), Some(ssr_scene_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("brush_ssr_pipeline"),
            layout: Some(&layout),
            vertex: VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: PipelineCompilationOptions::default(),
                buffers: &[Some(BrushVertex::layout())],
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
                depth_write_enabled: Some(true),
                depth_compare: Some(CompareFunction::LessEqual),
                stencil: StencilState::default(),
                bias: DepthBiasState::default(),
            }),
            // The eye pass is single-sampled: it draws over a resolved blit.
            multisample: MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        Self { pipeline, material_layout, lightmap_layout }
    }

    fn new_with_front_face(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        front_face: FrontFace,
        samples: u32,
        sources: bool,
        view: crate::renderer::multiview::ViewMode,
    ) -> Self {
        Self::new_blended(
            device, format, uniform_layout, front_face, samples, sources,
            Some(BlendState::ALPHA_BLENDING), view,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new_blended(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        front_face: FrontFace,
        samples: u32,
        sources: bool,
        blend: Option<BlendState>,
        view: crate::renderer::multiview::ViewMode,
    ) -> Self {
        Self::new_variant(
            device, format, uniform_layout, front_face, samples, sources, blend, view, BrushProbe::Trace, None,
        )
    }

    /// THE HALF-RESOLUTION PROBE PASS: the level's brushes drawn again at half
    /// the eye's resolution, writing only their probe reflection. See
    /// `probe_pass`. `Stereo` draws both eyes into the two layers of one
    /// target, for the multiview scene pass.
    pub fn new_probe_pass(
        device: &Device,
        uniform_layout: &BindGroupLayout,
        view: crate::renderer::multiview::ViewMode,
    ) -> Self {
        Self::new_variant(device, probe_pass::FORMAT, uniform_layout, FrontFace::Ccw, 1, false, None, view, BrushProbe::Pass, None)
    }

    /// THE SINGLE-EYE PROBE PASS THAT SHIPS: as [`Self::new_probe_pass`], with
    /// the secondary lookups left to `fixups` (`probe_fixup`). Its fragments
    /// write the record list, so their depth test is forced EARLY where the
    /// device allows it: a shader with side effects is otherwise tested after
    /// it runs, and every hidden fragment would be shaded.
    pub fn new_probe_pass_deferred(
        device: &Device,
        uniform_layout: &BindGroupLayout,
        fixups: &crate::renderer::probe_fixup::ProbeFixups,
    ) -> Self {
        Self::new_variant(
            device,
            probe_pass::FORMAT,
            uniform_layout,
            FrontFace::Ccw,
            1,
            false,
            None,
            crate::renderer::multiview::ViewMode::Mono,
            BrushProbe::PassDeferred,
            Some(fixups.pass_layout()),
        )
    }

    /// The scene pass's opaque brush, reading its probe reflection from that
    /// pass through group 3 (`probe_pass::bind_group_layout`) instead of
    /// tracing it per pixel.
    pub fn new_multisampled_probe_reader(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        samples: u32,
        probe_layout: &BindGroupLayout,
        view: crate::renderer::multiview::ViewMode,
    ) -> Self {
        Self::new_variant(
            device,
            format,
            uniform_layout,
            FrontFace::Ccw,
            samples,
            BRUSH_SOURCE_DEBUG,
            None,
            view,
            BrushProbe::Read,
            Some(probe_layout),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new_variant(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        front_face: FrontFace,
        samples: u32,
        sources: bool,
        blend: Option<BlendState>,
        view: crate::renderer::multiview::ViewMode,
        probe: BrushProbe,
        probe_layout: Option<&BindGroupLayout>,
    ) -> Self {
        let mut source = brush_shader_probe(false, sources, crate::renderer::ssr::SSR_DEBUG, SsrPath::Inline, probe);
        if probe == BrushProbe::PassDeferred && device.features().contains(wgpu::Features::SHADER_EARLY_DEPTH_TEST) {
            assert!(source.contains(FS_MAIN), "the fragment entry point moved");
            source = source.replacen(FS_MAIN, "@fragment @early_depth_test(force) fn fs_main(", 1);
        }
        // Named by what it does, so a GPU profile or `PIPESTATS` line says
        // which of them it is.
        let label = match probe {
            BrushProbe::Trace => "brush_pipeline",
            BrushProbe::Pass => "brush_probe_pass",
            BrushProbe::PassDeferred => "brush_probe_pass_deferred",
            BrushProbe::Read => "brush_pipeline_read",
        };
        Self::from_source(device, format, uniform_layout, front_face, samples, blend, view, probe_layout, label, source)
    }

    /// MEASUREMENT ONLY: the probe pass's shader with one part of it cut out
    /// at a time (`PROBE_PASS_REGISTER_CUTS`), each built as a pipeline of its
    /// own that nothing draws with, so the driver reports each one's
    /// registers (`PIPESTATS`; see `shader_checks::PIPELINE_STATISTICS`). The
    /// pass's occupancy is set by its register PEAK, and where the count
    /// drops is where the peak was.
    pub fn log_probe_pass_register_cuts(device: &Device, uniform_layout: &BindGroupLayout) {
        let base = brush_shader_probe(false, false, crate::renderer::ssr::SSR_DEBUG, SsrPath::Inline, BrushProbe::Pass);
        for (label, edits) in PROBE_PASS_REGISTER_CUTS {
            let mut src = base.clone();
            let mut missing = None;
            for (from, to) in edits.iter() {
                if !src.contains(from) {
                    missing = Some(*from);
                    break;
                }
                src = src.replacen(from, to, 1);
            }
            match missing {
                Some(from) => log::warn!("register cut {label}: `{from}` is not in the shader"),
                None => {
                    let _ = Self::from_source(
                        device,
                        probe_pass::FORMAT,
                        uniform_layout,
                        FrontFace::Ccw,
                        1,
                        None,
                        crate::renderer::multiview::ViewMode::Mono,
                        None,
                        label,
                        src,
                    );
                }
            }
        }
    }

    /// MEASUREMENT ONLY: `log_probe_pass_register_cuts` for the scene pass's
    /// brush shader -- the one reading the probe pass -- with the same
    /// target, samples and group 3 as the one that ships.
    pub fn log_scene_register_cuts(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        samples: u32,
        probe_layout: &BindGroupLayout,
    ) {
        let base = brush_shader_probe(false, BRUSH_SOURCE_DEBUG, crate::renderer::ssr::SSR_DEBUG, SsrPath::Inline, BrushProbe::Read);
        for (label, edits) in SCENE_REGISTER_CUTS {
            let mut src = base.clone();
            let mut missing = None;
            for (from, to) in edits.iter() {
                if !src.contains(from) {
                    missing = Some(*from);
                    break;
                }
                src = src.replacen(from, to, 1);
            }
            match missing {
                Some(from) => log::warn!("register cut {label}: `{from}` is not in the shader"),
                None => {
                    let _ = Self::from_source(
                        device,
                        format,
                        uniform_layout,
                        FrontFace::Ccw,
                        samples,
                        None,
                        crate::renderer::multiview::ViewMode::Mono,
                        Some(probe_layout),
                        label,
                        src,
                    );
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn from_source(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        front_face: FrontFace,
        samples: u32,
        blend: Option<BlendState>,
        view: crate::renderer::multiview::ViewMode,
        probe_layout: Option<&BindGroupLayout>,
        label: &str,
        source: String,
    ) -> Self {
        // Audited: see `shader_checks`.
        let shader = crate::renderer::shader_checks::audited_shader_module(device, ShaderModuleDescriptor {
            label: Some("brush_shader"),
            source: ShaderSource::Wgsl(crate::renderer::shader_precision::for_device(device, view.shader(source)).into()),
        });
        let material_layout = brush_material_bind_group_layout(device);
        // Shared with the mesh and cuboid pipelines: a lightmap is a lightmap,
        // and three layouts that must stay identical is three chances to drift.
        let lightmap_layout = super::pipeline::lightmap_bind_group_layout(device);
        let mut layouts = vec![Some(uniform_layout), Some(&material_layout), Some(&lightmap_layout)];
        if let Some(l) = probe_layout {
            layouts.push(Some(l));
        }
        let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("brush_layout"),
            bind_group_layouts: &layouts,
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(&layout),
            vertex: VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: PipelineCompilationOptions::default(),
                buffers: &[Some(BrushVertex::layout())],
            },
            fragment: Some(FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: PipelineCompilationOptions::default(),
                targets: &[Some(ColorTargetState {
                    format,
                    blend,
                    write_mask: ColorWrites::ALL,
                })],
            }),
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                cull_mode: Some(Face::Back),
                front_face,
                polygon_mode: PolygonMode::Fill,
                ..Default::default()
            },
            depth_stencil: Some(DepthStencilState {
                format: TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                // LessEqual, not Less: after the depth prepass the nearest
                // brush's depth is already in the buffer, and its own shading
                // must pass on EQUAL. Without a prepass the two differ only
                // for coplanar brush faces. See `new_depth_prepass`.
                depth_compare: Some(CompareFunction::LessEqual),
                stencil: StencilState::default(),
                bias: DepthBiasState::default(),
            }),
            multisample: MultisampleState { count: samples, ..Default::default() },
            multiview_mask: view.mask(),
            cache: None,
        });
        Self { pipeline, material_layout, lightmap_layout }
    }

    /// THE DEPTH PREPASS: the level's brushes drawn first in the scene pass,
    /// writing only depth, so everything drawn after -- the brushes' own
    /// shading at LessEqual, the terrain, meshes -- is shaded only where it is
    /// the nearest surface.
    ///
    /// Why: the first headset benchmark (2026-09-27) counted 17-34 M
    /// fragments shaded a frame for 3.6 M pixels, and the hardware's
    /// low-resolution depth rejected almost none -- the terrain is drawn
    /// before the walls and floors that hide it, and one draw of the whole
    /// level has no order. The frame is fragment-bound, so every hidden
    /// fragment shaded is time.
    ///
    /// The brushes' own vertex stage with an `@invariant` position, so both
    /// draws put a brush at the same depth to the bit. Its fragment stage does
    /// nothing and writes no colour; only groups 0 and 1 (camera, material
    /// proportions) are bound.
    pub fn new_depth_prepass(
        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        samples: u32,
        view: crate::renderer::multiview::ViewMode,
    ) -> Self {
        // Audited: see `shader_checks`.
        let shader = crate::renderer::shader_checks::audited_shader_module(device, ShaderModuleDescriptor {
            label: Some("brush_depth_prepass"),
            source: ShaderSource::Wgsl(
                view.shader(brush_shader_probe(false, false, crate::renderer::ssr::SSR_DEBUG, SsrPath::Inline, BrushProbe::Trace))
                    .into(),
            ),
        });
        let material_layout = brush_material_bind_group_layout(device);
        let lightmap_layout = super::pipeline::lightmap_bind_group_layout(device);
        let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("brush_depth_prepass_layout"),
            bind_group_layouts: &[Some(uniform_layout), Some(&material_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("brush_depth_prepass"),
            layout: Some(&layout),
            vertex: VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: PipelineCompilationOptions::default(),
                buffers: &[Some(BrushVertex::layout())],
            },
            fragment: Some(FragmentState {
                module: &shader,
                entry_point: Some("fs_depth"),
                compilation_options: PipelineCompilationOptions::default(),
                targets: &[Some(ColorTargetState { format, blend: None, write_mask: ColorWrites::empty() })],
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
        Self { pipeline, material_layout, lightmap_layout }
    }
}

/// Whether the scene pass seals the cracks between brush faces.
///
/// CSG leaves T-junctions -- a face's corner lying partway along its
/// neighbour's edge -- and the rasteriser can round the two edges differently,
/// leaving single-pixel holes along the seam. Through them the eye sees
/// whatever is behind the wall, usually the lit exterior, so every room edge
/// wore a dotted WHITE line (headset, 2026-09-17; 408 T-junctions counted in
/// `test_room` on 2026-09-10).
///
/// Repairing the triangulation removes them, but that repair made room edges
/// sawtoothed on the headset for a reason never found (see
/// `quest_app::brush_render::REPAIR_T_JUNCTIONS`). This instead draws the
/// level's BACK faces, flat and dark, after its front faces. A brush solid is
/// closed, so behind every crack there is a back face of that same solid a
/// wall's thickness away; depth-tested normally, the back faces lose to every
/// front face and survive only where the front faces left a hole. No geometry
/// changes, so nothing about the triangles the repair disturbed is touched.
pub const SEAL_BRUSH_CRACKS: bool = true;

/// What a sealed crack shows: about the scene's clear colour, so a hole in a
/// dark interior disappears instead of glowing.
pub const CRACK_SEAL_COLOUR: [f32; 3] = [0.02, 0.02, 0.05];

/// The brush back faces, drawn flat. See [`SEAL_BRUSH_CRACKS`].
///
/// Cheap by construction: it draws after the front faces, so early depth
/// rejection discards almost every fragment before the one-line shader runs,
/// and it binds only the scene uniform for the camera.
pub struct BrushSealPipeline {
    pub pipeline: RenderPipeline,
}

impl BrushSealPipeline {
    pub fn new(

        device: &Device,
        format: TextureFormat,
        uniform_layout: &BindGroupLayout,
        samples: u32,
    ) -> Self {
        Self::new_with_view(device, format, uniform_layout, samples, crate::renderer::multiview::ViewMode::Mono)
    }

    /// The same, drawing BOTH EYES in one pass. See `multiview::ViewMode`.
    pub fn new_stereo(

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
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("brush_seal_shader"),
            source: ShaderSource::Wgsl(view.shader(brush_seal_shader()).into()),
        });
        let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("brush_seal_layout"),
            bind_group_layouts: &[Some(uniform_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("brush_seal_pipeline"),
            layout: Some(&layout),
            vertex: VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: PipelineCompilationOptions::default(),
                buffers: &[Some(BrushVertex::layout())],
            },
            fragment: Some(FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: PipelineCompilationOptions::default(),
                // No blending: a sealed crack is opaque.
                targets: &[Some(ColorTargetState { format, blend: None, write_mask: ColorWrites::ALL })],
            }),
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                // FRONT culled, so only the faces pointing away are drawn --
                // the same winding the brush pipeline culls the other way.
                cull_mode: Some(Face::Front),
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
        Self { pipeline }
    }
}

fn brush_seal_shader() -> String {
    let [r, g, b] = CRACK_SEAL_COLOUR;
    format!(
        r#"
{lights_block}

@vertex fn vs_main(@location(0) pos: vec3<f32>) -> @builtin(position) vec4<f32> {{
    return cam_view_proj() * vec4<f32>(pos, 1.0);
}}

@fragment fn fs_main() -> @location(0) vec4<f32> {{
    return vec4<f32>({r:?}, {g:?}, {b:?}, 1.0);
}}
"#,
        lights_block = wgsl_lights_block(0, 1),
    )
}

fn brush_shader() -> String {
    brush_shader_with(false)
}

/// The brush shader, optionally reflecting the scene it is drawn over.
///
/// `ssr` is true for the reflective variant. It shares
/// the SCENE uniform's camera rather than binding its own, because a brush
/// already spends three bind groups and this hardware allows four -- see
/// `wgsl_ssr_block_shared_camera`.
/// How far the baked sun mask's signed distance reaches either side of the
/// shadow's edge, in mask texels. Red 0..255 maps to -this..+this; blue to a
/// penumbra half-width of 0..this. The baker's
/// `brush_lightmap::SUN_MASK_DISTANCE_TEXELS` must equal it -- the bake's
/// `the_sun_mask_range_matches_the_renderer` holds the two together.
pub const SUN_MASK_DISTANCE_TEXELS: f32 = 4.0;

/// How far a STATIONARY lamp's mask distance reaches, in mask texels, and its
/// penumbra: each lamp's pair of bytes decodes as the sun mask's red and blue.
/// `space_soup_engine::brush_lightmap::STATIONARY_MASK_DISTANCE_TEXELS` must
/// equal it, and the baker's `the_stationary_mask_range_matches_the_renderer`
/// holds the two together.
pub const STATIONARY_MASK_DISTANCE_TEXELS: f32 = 4.0;

fn brush_shader_with(ssr: bool) -> String {
    brush_shader_variant(ssr, BRUSH_SOURCE_DEBUG, crate::renderer::ssr::SSR_DEBUG)
}

/// The brush shader with each diagnostic chosen by the caller.
///
/// The shipped shader is `brush_shader_with`, which reads the build switches;
/// this exists so a debug pipeline can be built BESIDE it and chosen at runtime
/// (see [`DebugView`]) without the shipped text changing by a byte.
/// REFLECTIONS AT HALF RESOLUTION.
///
/// The probe reflection -- choosing the room's photographs, tracing the ray
/// through rooms, doorways and what stands in them, reading the photographs
/// that saw the hit -- was the most expensive thing the brush shader did:
/// 18 ms of a 46 ms frame in the marble hall (headset A/B, 2026-09-27), more
/// than everything else it lights. It is also the part whose DETAIL is
/// bounded by something coarser than the screen: a 256-pixel probe texel spans
/// about five screen pixels at the shipped render scale, so a reflection
/// computed at every screen pixel resolves nothing the photographs hold.
///
/// So it has its own pass: the brushes drawn again at half the eye's
/// resolution (a quarter of the pixels) with a shader that writes only the
/// probe reflection -- `lights::probe_env_for_pass`, normalised and
/// premultiplied by its coverage -- and the scene pass's brush shader reads it
/// back (`BrushProbe::Read`), four texels around each pixel, keeping only those
/// on the same surface by depth so a reflection never bleeds across a
/// silhouette. Built with `PROBE_ENV_FROM_PASS`, that shader carries none of the
/// trace, which also frees the registers the trace held.
///
/// Mono scene passes only for now; a stereo pass keeps tracing per pixel.
/// `Levers::half_res_reflections` switches it off to measure it.
/// MEASUREMENT ONLY: the scene pass's brush shader (the one reading the probe
/// pass) with one part cut out at a time -- see
/// `BrushPipeline::log_scene_register_cuts`. Text edits of the generated WGSL;
/// nothing draws with them.
const SCENE_REGISTER_CUTS: &[(&str, &[(&str, &str)])] = &[
    ("scene_cut_none", &[]),
    ("scene_cut_lamp_loop", &[("    var todo = reaching;\n", "    var todo = 0u;\n")]),
    (
        "scene_cut_lamps",
        &[(
            "i < live_light_count(); i = i + 1u) {\n        let kind = lights.lights[i].params.z;\n        let to_lamp",
            "i < 0u; i = i + 1u) {\n        let kind = lights.lights[i].params.z;\n        let to_lamp",
        )],
    ),
    ("scene_cut_stationary", &[("    set_stationary_masks(st_0, st_1, st_2, st_3, STATIONARY_MASK_DISTANCE_TEXELS);\n", "")]),
    ("scene_cut_bounce", &[("    if (has_dir) {", "    if (false) {")]),
    (
        "scene_cut_probe_read",
        &[("    probe_env_given = probe_pass_upsample(in.clip.xy, in.clip.z, probe_pass_tolerance);", "    probe_env_given = vec4<f32>(0.0);")],
    ),
    (
        "scene_cut_lamp_spec",
        &[(
            "    if (ndotl > 0.0 && atten > 0.0) {\n        let h = normalize(l_dir + view_dir);\n        let spec = pow(max(dot(n, h), 0.0), shininess) * spec_strength;\n        out.specular",
            "    if (false) {\n        let h = normalize(l_dir + view_dir);\n        let spec = pow(max(dot(n, h), 0.0), shininess) * spec_strength;\n        out.specular",
        )],
    ),
    (
        "scene_cut_spot_shadow",
        &[(
            "        if (layer >= 0 && f32(layer) < camera.shadow_params.y) {\n            shadow = shadow * pcf_layer(",
            "        if (false) {\n            shadow = shadow * pcf_layer(",
        )],
    ),
    (
        "scene_cut_sun_shadow",
        &[("        if (l.params.z > 1.5) {\n            shadow = sun_visibility(l, world_pos);", "        if (false) {\n            shadow = sun_visibility(l, world_pos);")],
    ),
    (
        "scene_cut_spot_cone",
        &[(
            "            let cos_angle = dot(-l_dir, l.direction.xyz);\n            atten = atten * spot_cone(cos_angle, cos_outer, cos_inner, dist);\n        }\n    }\n\n    let ndotl = max(dot(n, l_dir), 0.0);\n    let radiance = l.color_intensity.rgb * l.color_intensity.a;\n    out.diffuse",
            "            let cos_angle = dot(-l_dir, l.direction.xyz);\n        }\n    }\n\n    let ndotl = max(dot(n, l_dir), 0.0);\n    let radiance = l.color_intensity.rgb * l.color_intensity.a;\n    out.diffuse",
        )],
    ),
    (
        "scene_cut_sky",
        &[
            ("    if (occ > 0.0) {\n        diffuse = sky_irradiance(n) * occ;\n    }", "    if (false) {\n        diffuse = sky_irradiance(n) * occ;\n    }"),
            (
                "    if (occ > 0.0) {\n        sky_reflection = environment_radiance(refl) * occ;\n    }",
                "    if (false) {\n        sky_reflection = environment_radiance(refl) * occ;\n    }",
            ),
        ],
    ),
    (
        "scene_cut_sun_mask",
        &[("    receiver_sun_mask = select(-1.0, smoothstep(-sun_w, sun_w, sun_d), sun_mask.g > 0.25);", "    receiver_sun_mask = -1.0;")],
    ),
    ("scene_cut_spec_aa", &[("    let rough_aa = specular_aa_roughness(rough, dpdx(n), dpdy(n));", "    let rough_aa = rough;")]),
];

/// MEASUREMENT ONLY: what `BrushPipeline::log_probe_pass_register_cuts` cuts
/// out of the probe pass's shader, one pipeline each -- text edits of the
/// generated WGSL. They change what the shader computes; nothing draws with them.
const PROBE_PASS_REGISTER_CUTS: &[(&str, &[(&str, &str)])] = &[
    ("cut_none", &[]),
    ("cut_edge_lookup", &[("    if (hit.edge_code >= 0) {\n", "    if (false) {\n")]),
    ("cut_rim_lookup", &[("    if (hit.rim >= 0.0) {\n", "    if (false) {\n")]),
    (
        "cut_both_lookups",
        &[("    if (hit.edge_code >= 0) {\n", "    if (false) {\n"), ("    if (hit.rim >= 0.0) {\n", "    if (false) {\n")],
    ),
    ("cut_edge_detect", &[("        if (camera.probe_proxies[i * 3 + 1].w < 0.5 && out.edge < 0) {", "        if (false) {")]),
    ("cut_rim_detect", &[("        if (hit.rim < 0.0 && t_exit * lobe > PROBE_RIM_MIN_SPREAD) {", "        if (false) {")]),
    ("cut_proxies", &[("    for (var i = probe_room_proxy(room); i >= 0; i = probe_proxy_next(i)) {", "    for (var i = -1; i >= 0; i = probe_proxy_next(i)) {")]),
    ("cut_proxy_surface", &[("    if (s0 < 0) {\n        return 3.4e38;\n    }", "    if (true) {\n        return 3.4e38;\n    }")]),
    ("cut_escape_colour", &[("    if (h.escaped) {\n        col = probe_escape_colour(", "    if (false) {\n        col = probe_escape_colour(")]),
    ("cut_untraced", &[("    return probe_through_portals(own, own_room, select_world, world_pos, d, probe_lod);", "    return own;")]),
    ("cut_one_hop", &[("const PROBE_TRACE_ROOMS: i32 = 3;", "const PROBE_TRACE_ROOMS: i32 = 1;")]),
    ("cut_trace", &[("    if (roughness > PROBE_TRACE_MAX_ROUGHNESS || camera.portal_params.y > 0.5) {", "    if (true) {")]),
];

pub mod probe_pass {
    use wgpu::{
        AddressMode, BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor,
        BindGroupLayoutEntry, BindingResource, BindingType, Device, Extent3d, FilterMode, Sampler,
        SamplerBindingType, SamplerDescriptor, ShaderStages, Texture, TextureDescriptor, TextureDimension,
        TextureFormat, TextureSampleType, TextureUsages, TextureView, TextureViewDimension,
    };

    /// Radiance and coverage: premultiplied RGB, coverage in A.
    pub const FORMAT: TextureFormat = TextureFormat::Rgba16Float;

    /// Group 3 of the reading brush shader: the pass's colour and its depth,
    /// a bilinear sampler for the colour and a point sampler to gather the
    /// depths. See `READER_WGSL`.
    pub fn bind_group_layout(device: &Device) -> BindGroupLayout {
        let texture = |binding: u32, sample_type: TextureSampleType| BindGroupLayoutEntry {
            binding,
            visibility: ShaderStages::FRAGMENT,
            // Arrays, one layer an eye: the reader picks its eye's layer with
            // `view_slot`, which is 0 in a single-eye pass -- so one shader
            // serves the per-eye and the two-eye scene pass alike.
            ty: BindingType::Texture { sample_type, view_dimension: TextureViewDimension::D2Array, multisampled: false },
            count: None,
        };
        let sampler = |binding: u32, kind: SamplerBindingType| BindGroupLayoutEntry {
            binding,
            visibility: ShaderStages::FRAGMENT,
            ty: BindingType::Sampler(kind),
            count: None,
        };
        device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("probe_pass_layout"),
            entries: &[
                texture(0, TextureSampleType::Float { filterable: true }),
                texture(1, TextureSampleType::Depth),
                sampler(2, SamplerBindingType::Filtering),
                sampler(3, SamplerBindingType::NonFiltering),
            ],
        })
    }

    /// A half-resolution probe pass's target -- one eye's, or both eyes' as
    /// the two layers of one for the multiview scene pass -- and the bind
    /// group the scene pass reads it through.
    pub struct Target {
        _color: Texture,
        /// What the pass renders into: layer 0, or every layer when stereo.
        pub color_view: TextureView,
        _depth: Texture,
        pub depth_view: TextureView,
        _samplers: [Sampler; 2],
        pub bind_group: BindGroup,
        pub width: u32,
        pub height: u32,
    }

    impl Target {
        /// For eyes `eye_width` x `eye_height`: half each way, rounded up, so
        /// the last column and row of pixels still have a texel. `layers` is 1
        /// for one eye, 2 for both in one multiview pass.
        pub fn new(device: &Device, layout: &BindGroupLayout, eye_width: u32, eye_height: u32, layers: u32) -> Self {
            let (width, height) = (eye_width.div_ceil(2).max(1), eye_height.div_ceil(2).max(1));
            let layers = layers.max(1);
            let make = |label: &str, format: TextureFormat, extra: TextureUsages| {
                device.create_texture(&TextureDescriptor {
                    label: Some(label),
                    size: Extent3d { width, height, depth_or_array_layers: layers },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: TextureDimension::D2,
                    format,
                    usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING | extra,
                    view_formats: &[],
                })
            };
            // Storage as well, for one eye: `probe_fixup` writes the texels
            // whose secondary lookups the pass deferred.
            let color = make(
                "probe_pass_color",
                FORMAT,
                if layers == 1 { TextureUsages::STORAGE_BINDING } else { TextureUsages::empty() },
            );
            let depth = make("probe_pass_depth", TextureFormat::Depth32Float, TextureUsages::empty());
            // Rendered into as a plain 2D view for one eye, as the array for
            // two (a multiview pass takes its view count from it); read as
            // the array either way.
            let attachment = |t: &Texture| {
                t.create_view(&wgpu::TextureViewDescriptor {
                    dimension: Some(if layers > 1 { TextureViewDimension::D2Array } else { TextureViewDimension::D2 }),
                    ..Default::default()
                })
            };
            let array = |t: &Texture| {
                t.create_view(&wgpu::TextureViewDescriptor { dimension: Some(TextureViewDimension::D2Array), ..Default::default() })
            };
            let (color_view, depth_view) = (attachment(&color), attachment(&depth));
            let (color_array, depth_array) = (array(&color), array(&depth));
            // Clamped at the edges, exactly as the four-texel path clamps.
            let sampler = |label: &str, filter: FilterMode| {
                device.create_sampler(&SamplerDescriptor {
                    label: Some(label),
                    address_mode_u: AddressMode::ClampToEdge,
                    address_mode_v: AddressMode::ClampToEdge,
                    mag_filter: filter,
                    min_filter: filter,
                    ..Default::default()
                })
            };
            let samplers = [sampler("probe_pass_linear", FilterMode::Linear), sampler("probe_pass_point", FilterMode::Nearest)];
            let bind_group = device.create_bind_group(&BindGroupDescriptor {
                label: Some("probe_pass_bg"),
                layout,
                entries: &[
                    BindGroupEntry { binding: 0, resource: BindingResource::TextureView(&color_array) },
                    BindGroupEntry { binding: 1, resource: BindingResource::TextureView(&depth_array) },
                    BindGroupEntry { binding: 2, resource: BindingResource::Sampler(&samplers[0]) },
                    BindGroupEntry { binding: 3, resource: BindingResource::Sampler(&samplers[1]) },
                ],
            });
            Self { _color: color, color_view, _depth: depth, depth_view, _samplers: samplers, bind_group, width, height }
        }
    }

    /// The pass's fragment body, after the brush shader's shared preamble
    /// (footprint, albedo, normal mapping): the probe reflection for this
    /// surface and nothing else, from the same inputs the scene pass hands
    /// `shade_material_env`.
    pub(super) const PASS_LIGHTING: &str = r#"
    let rough = textureSample(mat_rough, mat_rough_samp, in.uv, i32(in.material)).r;
    let rough_aa = specular_aa_roughness(rough, dpdx(n), dpdy(n));
    let ao = textureSample(mat_ao, mat_samp, in.uv, i32(in.material)).r;
    let lm_uv = clamp(in.uv2, in.uv2_rect.xy, in.uv2_rect.zw);
    let baked = textureSample(lm_tex, lm_samp, lm_uv);
    let face_t = normalize(in.tangent.xyz - n_geom * dot(n_geom, in.tangent.xyz));
    let face_b = cross(n_geom, face_t) * in.tangent.w;
    let face_d = in.world_pos - in.face_centre;
    let face_pos = in.face_centre
        + face_t * clamp(dot(face_d, face_t), -in.face_half_extent.x, in.face_half_extent.x)
        + face_b * clamp(dot(face_d, face_b), -in.face_half_extent.y, in.face_half_extent.y);
    probe_volume_pos = vec4<f32>(in.face_centre, 1.0);
    probe_face_given = in.probe_face;
    return probe_env_for_pass(face_pos, n, rough_aa, ao, baked.a, baked.rgb, face_pos, n_geom);"#;

    /// Group 3 and the read, appended to the scene pass's brush shader.
    pub(super) const READER_WGSL: &str = r#"
@group(3) @binding(0) var probe_pass_tex: texture_2d_array<f32>;
@group(3) @binding(1) var probe_pass_depth: texture_depth_2d_array;
@group(3) @binding(2) var probe_pass_linear: sampler;
@group(3) @binding(3) var probe_pass_point: sampler;

// THE HALF-RESOLUTION PROBE REFLECTION AT THIS PIXEL: the four pass texels
// around it, weighted bilinearly and kept only where their depth is this
// pixel's -- the same surface -- so a reflection never bleeds across a
// silhouette. `tolerance` is how far two depths may differ and still be one
// surface, from this pixel's own depth slope. With no neighbour on this surface
// (a sliver the half-resolution pass missed), the nearest in depth. The pass
// stores the reflection premultiplied by its coverage, which is what makes the
// weighted sum a correct filter; it is divided back out here.
//
// TWO READS WHERE IT CAN BE, EIGHT WHERE IT MUST. The four depths come in one
// gather. When all four are this pixel's surface -- everywhere but along an
// outline -- the weighted sum IS bilinear filtering, so the hardware's one
// filtered read replaces four loads. Only along an edge are the texels loaded
// and weighted one by one. Unrolled rather than looped over an index: a
// dynamically indexed local array is the Adreno cliff that spills to memory.
//
// This eye's layer is `view_slot`: 0 in a single-eye pass, the view index in a
// multiview one.
fn probe_pass_upsample(pixel: vec2<f32>, depth: f32, tolerance: f32) -> vec4<f32> {
    let dims = textureDimensions(probe_pass_tex);
    // A full-resolution pixel centre at `pixel` is at `pixel * 0.5` in the
    // half-resolution texel grid.
    let uv = pixel * 0.5 / vec2<f32>(dims);
    // A gather returns the footprint as (0,1) (1,1) (1,0) (0,0) from its
    // corner; `.wzxy` puts it in the order (0,0) (1,0) (0,1) (1,1).
    let gaps = abs(textureGather(probe_pass_depth, probe_pass_point, uv, view_slot) - vec4<f32>(depth)).wzxy;
    if (all(gaps <= vec4<f32>(tolerance))) {
        let pre = textureSampleLevel(probe_pass_tex, probe_pass_linear, uv, view_slot, 0.0);
        return vec4<f32>(pre.rgb / max(pre.a, 1e-4), pre.a);
    }
    let size = vec2<i32>(dims);
    let h = pixel * 0.5 - vec2<f32>(0.5);
    let base = vec2<i32>(floor(h));
    let f = h - floor(h);
    let top = size - vec2<i32>(1);
    let c00 = textureLoad(probe_pass_tex, clamp(base, vec2<i32>(0), top), view_slot, 0);
    let c10 = textureLoad(probe_pass_tex, clamp(base + vec2<i32>(1, 0), vec2<i32>(0), top), view_slot, 0);
    let c01 = textureLoad(probe_pass_tex, clamp(base + vec2<i32>(0, 1), vec2<i32>(0), top), view_slot, 0);
    let c11 = textureLoad(probe_pass_tex, clamp(base + vec2<i32>(1, 1), vec2<i32>(0), top), view_slot, 0);
    let bilinear = vec4<f32>((1.0 - f.x) * (1.0 - f.y), f.x * (1.0 - f.y), (1.0 - f.x) * f.y, f.x * f.y);
    let w = select(vec4<f32>(0.0), bilinear, gaps <= vec4<f32>(tolerance));
    let weight = w.x + w.y + w.z + w.w;
    // Ties go to the first in that order, as they always have.
    var nearest = c00;
    var nearest_gap = gaps.x;
    if (gaps.y < nearest_gap) { nearest_gap = gaps.y; nearest = c10; }
    if (gaps.z < nearest_gap) { nearest_gap = gaps.z; nearest = c01; }
    if (gaps.w < nearest_gap) { nearest_gap = gaps.w; nearest = c11; }
    let sum = c00 * w.x + c10 * w.y + c01 * w.z + c11 * w.w;
    let pre = select(nearest, sum / max(weight, 1e-6), weight > 1e-4);
    return vec4<f32>(pre.rgb / max(pre.a, 1e-4), pre.a);
}
"#;
}

fn brush_shader_variant(ssr: bool, sources: bool, ssr_debug: bool) -> String {
    brush_shader_modes(ssr, sources, ssr_debug, SsrPath::Inline)
}

/// WHICH OF THE THREE REFLECTIVE SHADERS THIS IS.
///
/// `Inline` marches and blends in one pass, which is what has always shipped.
/// The other two are the buffered path: `Trace` marches and writes radiance and
/// confidence, a resolve filters it across pixels, and `Composite` reads the
/// filtered result and blends. Splitting them is what makes the hit/miss cliff
/// removable at all -- a fragment can see its own ray and nothing else.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum SsrPath {
    Inline,
    Trace,
    Composite,
}

/// The brush shader, with the reflection TRACE as a fourth mode.
///
/// `trace` draws the same geometry the reflective pass draws, runs the same
/// march, and writes RADIANCE AND CONFIDENCE instead of a shaded colour -- see
/// `ssr::SsrOutput::Radiance`. It is a separate pipeline rather than a flag
/// because the two have different fragment outputs, and it exists so the
/// reflection can be filtered ACROSS pixels before anything looks at it, which
/// is the only place the hit/miss cliff can be removed.
fn brush_shader_modes(ssr: bool, sources: bool, ssr_debug: bool, path: SsrPath) -> String {
    brush_shader_probe(ssr, sources, ssr_debug, path, BrushProbe::Trace)
}

/// WHERE A BRUSH SHADER'S PROBE REFLECTION COMES FROM. See `probe_pass`.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum BrushProbe {
    /// Traced per pixel, in the shader that shades. What always shipped.
    Trace,
    /// The half-resolution probe pass itself: shades nothing, writes only the
    /// probe reflection. See `probe_pass`.
    Pass,
    /// The same pass, leaving each traced hit's secondary lookups to
    /// `probe_fixup` (group 3: the record list). What ships in the single-eye
    /// pass. See `lights::PROBE_SECONDARY_DEFERRED`.
    PassDeferred,
    /// The scene pass's brush, reading that pass instead of tracing.
    Read,
}

/// How the brush shader's fragment entry point begins -- where an attribute on
/// it goes. See `BrushPipeline::new_probe_pass_deferred`.
const FS_MAIN: &str = "@fragment fn fs_main(";

/// `brush_shader_modes`, with the probe reflection's source. See `BrushProbe`.
fn brush_shader_probe(ssr: bool, sources: bool, ssr_debug: bool, path: SsrPath, probe: BrushProbe) -> String {
    // Both forms of the probe pass share everything but the deferral.
    let pass_like = matches!(probe, BrushProbe::Pass | BrushProbe::PassDeferred);
    let trace = path == SsrPath::Trace;
    let composite = path == SsrPath::Composite;
    let ssr_block = match (ssr, path) {
        (true, SsrPath::Trace) => crate::renderer::ssr::wgsl_ssr_block_shared_camera_radiance(3),
        // No march at all: the composite only reads what the resolve wrote.
        (true, SsrPath::Composite) => crate::renderer::ssr::wgsl_ssr_composite_block(3),
        (true, SsrPath::Inline) => {
            crate::renderer::ssr::wgsl_ssr_block_shared_camera_debug(3, ssr_debug)
        }
        (false, _) => String::new(),
    };
    // The reflective variant returns the scene reflected in this surface;
    // the plain one returns the shaded colour unchanged.
    // DIAGNOSTIC BUILD SWITCH. Off in every shipped build.
    //
    // Red   = punctual light arriving before shadowing.
    // Green = the same after shadowing.
    // So: yellow means lit, red means shadowed away, black means no light ever
    // reached the shader. `test_room` renders black interiors while the
    // arithmetic says the floor under a lamp should be blown out, and this is
    // the only way to tell those three apart on a headset that cannot be
    // pixel-read from the host.
    if BRUSH_LIGHT_DEBUG {
        return format!(
            r#"
{lights_block}

struct VIn {{
    @location(0) pos: vec3<f32>,
    @location(1) norm: vec3<f32>,
    @location(2) tangent: vec4<f32>,
    @location(3) uv: vec2<f32>,
    @location(4) material: u32,
    @location(5) tint: vec4<f32>,
    @location(6) uv2: vec2<f32>,
}}
struct VOut {{
    @builtin(position) clip: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) world_pos: vec3<f32>,
}}
@vertex fn vs_main(v: VIn) -> VOut {{
    var out: VOut;
    out.clip = cam_view_proj() * vec4<f32>(v.pos, 1.0);
    out.normal = v.norm;
    out.world_pos = v.pos;
    return out;
}}
@fragment fn fs_main(in: VOut) -> @location(0) vec4<f32> {{
    let d = light_debug(in.world_pos, normalize(in.normal));
    return vec4<f32>(clamp(d.x, 0.0, 1.0), clamp(d.y, 0.0, 1.0), 0.0, 1.0);
}}
"#,
            lights_block = wgsl_lights_block(0, 1),
        );
    }
    // See `BRUSH_SOURCE_DEBUG`. Normalised so the three sources are compared
    // by RATIO -- a dim room and a lit one read the same, and what shows is
    // which term dominates rather than how bright the pixel is.
    // See `BRUSH_EDGE_DEBUG`. Only emitted for the diagnostic, so a shipping
    // build carries neither the extra varying nor the extra interpolation.
    let (edge_varying, edge_vs, edge_debug) = if BRUSH_EDGE_DEBUG {
        (
            // THE SAME VALUE AS `world_pos`, interpolated at the centroid of
            // the covered samples instead of at the pixel centre. The two are
            // identical wherever the centre is inside the polygon, so their
            // difference IS the extrapolation, with nothing else in it.
            "    @location(9) @interpolate(perspective, centroid) world_pos_centroid: vec3<f32>,\n",
            "    out.world_pos_centroid = v.pos;\n",
            r#"
    // How far the pixel centre sits outside its own polygon, in PIXELS rather
    // than metres -- so a near wall and a far one read the same and what shows
    // is the geometry of the sampling, not the distance to it.
    //
    // `fwidth(world_pos)` is the world-space span of one pixel, which is the
    // only honest unit to divide by here.
    let edge_delta = length(in.world_pos - in.world_pos_centroid);
    let edge_pixel = length(fwidth(in.world_pos));
    let edge_out = edge_delta / max(edge_pixel, 1e-6);
    return vec4<f32>(clamp(edge_out, 0.0, 1.0), 0.02, 0.02, 1.0);"#,
        )
    } else {
        ("", "", "")
    };
    let source_debug = if sources {
        r#"
    let dbg_sum = dbg_direct + dbg_baked + dbg_probe;
    let dbg_total = max(dbg_sum.r + dbg_sum.g + dbg_sum.b, 1e-6);
    return vec4<f32>(
        (dbg_direct.r + dbg_direct.g + dbg_direct.b) / dbg_total,
        (dbg_baked.r + dbg_baked.g + dbg_baked.b) / dbg_total,
        (dbg_probe.r + dbg_probe.g + dbg_probe.b) / dbg_total,
        1.0,
    );"#
    } else {
        ""
    };
    // REPLACES the normal tail rather than preceding it. Emitting a debug
    // `return` in front of the real one leaves the real one unreachable, and
    // naga rejects that -- caught by the pipeline-build tests, which is twice
    // now that they have caught exactly this.
    let ssr_apply = if ssr {
        r#"
    // SSR runs on the TONEMAPPED colour, because that is what the scene buffer
    // it samples already holds -- reflecting a linear value into a tonemapped
    // frame would make every reflection brighter than the thing reflected.
    //
    // Reflectivity is the material's own: smooth surfaces reflect and rough
    // ones do not, weighted by Fresnel so a floor reflects hard at a grazing
    // angle and barely at all underfoot. A fully rough face lands on zero and
    // `ssr_reflect` returns immediately without marching.
    // THE NEIGHBOURHOOD'S ROUGHNESS AND NORMAL DECIDE THE REFLECTION, not
    // this texel's.
    //
    // Whether a pixel reflects, and where its ray goes, came from its own
    // roughness texel and its own normal-map texel. On stone both vary texel
    // to texel -- across the SSR threshold, and in direction -- so neighbouring
    // pixels alternated between a screen-space hit, a miss and "too rough".
    // The SSR false-colour view of the stone room (headset, 2026-09-11) was a
    // per-pixel scatter of all three, and with no temporal accumulation to
    // average it away that is sparkle.
    //
    // A coarse mip of the roughness map IS the neighbourhood's roughness, so a
    // region takes one path. The normal keeps its detail only where the
    // surface is smooth enough for detail to show in a reflection: polished
    // marble keeps it, rough stone reflects along its face.
    let ssr_rough = textureSampleLevel(mat_rough, mat_rough_samp, in.uv, i32(in.material), 4.0).r;
    let ssr_n = normalize(mix(n_geom, n, 1.0 - smoothstep(0.1, 0.4, ssr_rough)));
    let ssr_view = normalize(cam_pos() - in.world_pos);
    let ssr_cos = clamp(dot(ssr_n, ssr_view), 0.0, 1.0);
    let ssr_fresnel = 0.04 + 0.96 * pow(1.0 - ssr_cos, 5.0);
    // CAPPED, and no longer for the reason it first was.
    //
    // The original justification was that the reflection was SHARP whatever the
    // surface roughness -- a single scene texel, so marble came back looking
    // like chrome. That is fixed: the reflection is now read from a mip chosen
    // by roughness. The cap stays as a deliberate conservatism while this is
    // unverified on the headset, not as a correction for a missing blur, and it
    // can be raised towards 1.0 -- a mirror-flat dielectric really does approach
    // total reflection at a grazing angle.
    let reflectivity = clamp((1.0 - ssr_rough) * ssr_fresnel, 0.0, MAX_SSR_REFLECTIVITY);
    // WHAT THIS SURFACE REFLECTS WHEN THE SCREEN CANNOT SAY.
    //
    // The shaded colour already contains a blurry environment reflection --
    // the sky's harmonics plus the room's own bounce, Fresnel weighted. That
    // is exactly the role a reflection probe plays, so it is handed in as the
    // fallback: where a ray leaves the frame or hits nothing, the reflection
    // softens back into the environment instead of disappearing.
    //
    // Previously the miss case returned the surface's own colour, so the edge
    // of the screen was a hard line the reflection stopped at, and it moved
    // with your head.
    //
    // READ FROM THE SCENE PASS, NOT SHADED AGAIN.
    //
    // This pass draws over a blit of the finished scene, and that scene pass
    // already shaded this exact surface: every light, its shadow taps, the
    // probe search and the lightmap. The fallback used to be `tonemap(c)` with
    // `c` lit a second time here, so every visible reflective pixel paid for
    // its lighting twice -- the eye pass measured 5 to 10 ms an eye in the
    // marble hall (headset, 2026-09-11), in a frame whose budget is 13.9.
    //
    // The texel the blit wrote at this pixel IS that colour, already
    // tonemapped and MSAA-resolved -- the same load the blit does, so the two
    // cannot disagree. It also keeps anything the scene pass drew in front of
    // the surface without writing depth, which a re-shade painted over.
    "#
    } else {
        r#"
    return vec4<f32>(tonemap(c), albedo.a * in.tint.a);"#
    };
    // THE LAST STATEMENT OF THE REFLECTIVE PATH, and the only thing the trace
    // changes.
    //
    // Everything above it -- the normal, the roughness mip, the Fresnel term,
    // `reflectivity` -- is the prep both modes need, and it is written ONCE.
    // An earlier attempt gave the trace its own branch covering the whole
    // block, which silently dropped the prep and left `ssr_n` undefined; naga
    // caught it, but only because a test builds this pipeline.
    let ssr_tail = if !ssr {
        ""
    } else if composite {
        // THE COMPOSITE. The reflection was traced and filtered already; all
        // that is left is to decide how much of this pixel is reflection, which
        // is its OWN reflectivity, and to hand the rest to the probe.
        //
        // `refl.a` is how much of the neighbourhood found anything, so the
        // handover into the fallback is now a ramp rather than the cliff the
        // headset showed as a comb along every silhouette.
        r#"
    let ssr_fallback = textureLoad(ssr_scene_color, vec2<i32>(in.clip.xy), 0).rgb;
    let refl = ssr_reflection_at(in.clip.xy);
    return vec4<f32>(
        mix(ssr_fallback, refl.rgb, clamp(reflectivity, 0.0, 1.0) * refl.a),
        albedo.a * in.tint.a,
    );"#
    } else if trace {
        // THE TRACE writes what the march found and how much to trust it, and
        // nothing else: no fallback is read and no surface colour is written.
        // `ssr_reflect` is already a `vec4` in this mode, so it IS the output.
        r#"
    return ssr_reflect(in.world_pos, ssr_n, vec3<f32>(0.0), reflectivity, ssr_rough);"#
    } else {
        r#"
    let ssr_fallback = textureLoad(ssr_scene_color, vec2<i32>(in.clip.xy), 0).rgb;
    return vec4<f32>(
        ssr_reflect(in.world_pos, ssr_n, ssr_fallback, reflectivity, ssr_rough),
        albedo.a * in.tint.a,
    );"#
    };
    let ssr_apply = format!("{ssr_apply}{ssr_tail}");
    // THE SHADING, unless this is the reflective pass drawing a shipped frame.
    //
    // That pass reads its surface colour back from the scene pass instead --
    // see `ssr_fallback`. The lighting-sources diagnostic still shades here,
    // because the sources it paints are recorded by `shade_material_env` as it
    // runs and exist nowhere else.
    let lighting = if ssr && !sources {
        ""
    } else {
        r#"
    let rough = textureSample(mat_rough, mat_rough_samp, in.uv, i32(in.material)).r;
    // The normal's per-pixel variation, put back as roughness -- see
    // `specular_aa_roughness`. The derivatives are taken HERE, at the top
    // level of the fragment entry, because a derivative is only legal in
    // uniform control flow; the terrain shader has a test pinning that rule.
    let rough_aa = specular_aa_roughness(rough, dpdx(n), dpdy(n));
    let ao = textureSample(mat_ao, mat_samp, in.uv, i32(in.material)).r;

    // RGB is baked direct+bounce and is ADDED; ALPHA is baked sky visibility
    // and is MULTIPLIED into the sky term. The two neutral values differ --
    // black adds nothing, 255 scales by one -- which is exactly what the
    // unbaked default texture carries.
    // CLAMPED INTO THIS FACE'S OWN PATCH. See `BrushVertex::uv2_rect`: an MSAA
    // edge pixel's interpolated uv2 is extrapolated past the patch, and with no
    // mips on the lightmap that overshoot is many atlas texels wide -- far past
    // the 2-texel gutter. Sampling outside returns a neighbour's bake or the
    // clear colour, which reads as a dotted line of darker pixels along every
    // room seam, brightest where the baked contribution matters most.
    let lm_uv = clamp(in.uv2, in.uv2_rect.xy, in.uv2_rect.zw);
    let baked = textureSample(lm_tex, lm_samp, lm_uv);
    // The baked bounce is handed to the shading function as well as added:
    // added because it is light arriving diffusely, and handed over because a
    // shiny surface reflects the room that light came from. See
    // `shade_material_env`.
    // The direction map shares this atlas's layout exactly, so the same uv2
    // addresses both. `shade_material_env` adds the bounce itself now, shaped
    // by that direction -- adding it here as well would double it.
    let bdir = textureSample(lm_dir_tex, lm_samp, lm_uv);
    // THE SKY SUN'S BAKED VISIBILITY, from the mask that rides with this
    // atlas at four times its density -- same charts, same uv2, same clamp.
    // Green marks a baked texel; the neutral mask an older bake leaves in its
    // place has none, and then the sun falls back to the level's static map.
    // Sampled here, in uniform control flow, like every texture above it.
    let sun_mask = textureSample(lm_sun_tex, lm_sun_samp, lm_uv);
    // A SIGNED DISTANCE TO THE SHADOW'S EDGE, not a coverage.
    //
    // Coverage filtered bilinearly still draws the texel grid: a diagonal
    // edge came out as a staircase of 3-6 cm steps across the doorway's sun
    // patch (headset, 2026-09-25). Distances interpolate into a straight line,
    // so the edge is rebuilt at 0 at any magnification. Its width is the
    // sun's own penumbra (from how far the shadow's caster is) or one
    // screen pixel, whichever is wider -- sharp where the caster is close,
    // soft where it is far, and never aliased. An older coverage mask reads
    // as distance 0 at half coverage, so it still works, only harder-edged.
    //
    // Green carries the rest (see `mesh::pack_sun_mask_texel`): under a half,
    // no bake; from a half up, baked, with the penumbra's half-width across
    // the upper half.
    let sun_d = (sun_mask.r - 0.5) * (2.0 * SUN_MASK_DISTANCE_TEXELS);
    let sun_pen = max(sun_mask.g - 0.5, 0.0) * (2.0 * SUN_MASK_DISTANCE_TEXELS);
    let sun_w = max(max(sun_pen, 0.5 * fwidth(sun_d)), 0.02);
    receiver_sun_mask = select(-1.0, smoothstep(-sun_w, sun_w, sun_d), sun_mask.g > 0.25);
    // The baked lamps are in this atlas. See `receiver_skips_baked`.
    receiver_skips_baked = true;
    // The albedo goes IN rather than being multiplied over the result: a
    // dielectric's highlight is not tinted by its diffuse colour, and marble's
    // 0.373 albedo was making every highlight nearly three times too dim.
    // THE FRAGMENT'S POSITION, PUT BACK ON ITS OWN FACE.
    //
    // MSAA shades a pixel at its centre, which at a polygon edge can lie
    // outside the polygon, and `world_pos` then arrives extrapolated past the
    // edge. Every consumer of it inherits that, but only one AMPLIFIES it:
    // `view_dir` feeds `cos_v`, and Fresnel's `pow(1 - cos_v, 5)` is steepest
    // exactly at the grazing angles found along a ceiling-wall junction. On
    // marble (`f_max` 0.952) that turns a sub-pixel position error into a
    // ~26% swing in the probe term, which is the seam.
    //
    // Clamping into the face's own extent is a no-op for every interior pixel
    // -- they are inside by construction -- and returns an edge pixel to the
    // nearest point that is actually ON its face. Dropping the normal
    // component puts it back on the face plane too, where it belongs.
    //
    // This is the same remedy as `face_centre` for probe selection and
    // `uv2_rect` for the lightmap, and like those it is `flat`, so unlike
    // centroid interpolation it costs nothing. (Centroid was measured at
    // ~11 ms a frame for this same correction; see `BRUSH_CENTROID_VARYINGS`.)
    let face_t = normalize(in.tangent.xyz - n_geom * dot(n_geom, in.tangent.xyz));
    let face_b = cross(n_geom, face_t) * in.tangent.w;
    let face_d = in.world_pos - in.face_centre;
    let face_pos = in.face_centre
        + face_t * clamp(dot(face_d, face_t), -in.face_half_extent.x, in.face_half_extent.x)
        + face_b * clamp(dot(face_d, face_b), -in.face_half_extent.y, in.face_half_extent.y);
    // THE PROBE IS CHOSEN FROM THIS PIXEL, not from the face's centre. A room
    // now has a photograph per stretch of it (`bake::probe::MAX_CELL_EDGE`),
    // and a 20 m wall choosing one of them for its whole length reflected the
    // front of the hall at the back. Chosen per pixel, the two nearest hand
    // over across `PROBE_BLEND_BAND`, which is continuous in position -- and
    // `face_pos` is clamped onto the face, so an MSAA edge pixel chooses what
    // its neighbours do, which is what the face centre was for.
    // WHICH ROOM is a question about the FACE, and is answered from its
    // centre; see `probe_volume_pos`. Per pixel, an MSAA sample on a ceiling
    // seen edge-on is shaded at the pixel centre, which can extrapolate 10 cm
    // past the far wall -- and the ceiling brush runs on over the wall top, so
    // clamping onto the face does not bring it back. That one row of samples
    // then took the OUTDOOR photograph (offline_frame, 2026-09-23).
    probe_volume_pos = vec4<f32>(in.face_centre, 1.0);
    let env_part = shade_material_env_part(
        face_pos, n, rough_aa, ao, baked.a, baked.rgb, bdir, albedo.rgb, face_pos,
        n_geom,
    );
    // THE MASKS AFTER THE ENVIRONMENT, just before the lamps that read them:
    // taken first, their eight visibilities rode through all of the
    // environment's work, and that was the shader's register peak. See
    // `MaterialEnvPart`.
    // THE STATIONARY LAMPS' SHADOWS, rebuilt as the sun's is: per lamp a
    // signed distance and the bulb's penumbra, two lamps a layer (red and
    // green, blue and alpha), the edge put back at zero at any magnification
    // and never narrower than a pixel. Away from a sharp edge the baker stores
    // a lamp's visibility as a distance this smoothstep gives back exactly, so
    // soft shadows and the faint bands of thin things come through the same
    // code. Layers a level does not have are not fetched; the neutral mask an
    // unbaked level carries reads fully lit. See `stationary_visibility`.
    let st_layers = textureNumLayers(lm_stationary);
    let st_0 = textureSample(lm_stationary, lm_sun_samp, lm_uv, 0);
    var st_1 = vec4<f32>(1.0);
    var st_2 = vec4<f32>(1.0);
    var st_3 = vec4<f32>(1.0);
    if (st_layers > 1u) {
        st_1 = textureSample(lm_stationary, lm_sun_samp, lm_uv, 1);
    }
    if (st_layers > 2u) {
        st_2 = textureSample(lm_stationary, lm_sun_samp, lm_uv, 2);
    }
    if (st_layers > 3u) {
        st_3 = textureSample(lm_stationary, lm_sun_samp, lm_uv, 3);
    }
    set_stationary_masks(st_0, st_1, st_2, st_3, STATIONARY_MASK_DISTANCE_TEXELS);
    let lit = shade_material_lamps(env_part, face_pos, n, baked.rgb, albedo.rgb);
    let c = in.tint.rgb * lit;"#
    };
    // THE HALF-RESOLUTION PROBE PASS. See `probe_pass`.
    let lighting: String = match probe {
        BrushProbe::Trace => lighting.to_string(),
        BrushProbe::Pass => probe_pass::PASS_LIGHTING.to_string(),
        // The same, telling a deferred lookup's record which texel it is.
        BrushProbe::PassDeferred => {
            let given = "    probe_face_given = in.probe_face;\n";
            assert!(probe_pass::PASS_LIGHTING.contains(given), "the probe pass no longer hands in its face");
            probe_pass::PASS_LIGHTING.replacen(given, &format!("{given}    probe_fragment = in.clip;\n"), 1)
        }
        BrushProbe::Read => {
            let marker = "    probe_volume_pos = vec4<f32>(in.face_centre, 1.0);\n    let env_part = shade_material_env_part(";
            assert!(
                lighting.contains(marker),
                "the brush lighting no longer sets the probe position just before it shades; the pass read goes there",
            );
            format!(
                "    let probe_pass_tolerance = max(4.0 * fwidth(in.clip.z), 1e-6);\n{}",
                lighting.replacen(
                    marker,
                    &format!("    probe_env_given = probe_pass_upsample(in.clip.xy, in.clip.z, probe_pass_tolerance);\n{marker}"),
                    1,
                )
            )
        }
    };
    let src = format!(
        r#"
// Group 0 -- the camera, the lights and both shadow maps -- is declared by
// `wgsl_lights_block` below, so there is one description of that layout rather
// than one per shader.

@group(2) @binding(0) var lm_tex: texture_2d<f32>;
@group(2) @binding(2) var lm_dir_tex: texture_2d<f32>;
@group(2) @binding(1) var lm_samp: sampler;
@group(2) @binding(3) var lm_sun_tex: texture_2d<f32>;
@group(2) @binding(4) var lm_sun_samp: sampler;
// How far the sun mask's signed distance reaches, in mask texels. See
// `SUN_MASK_DISTANCE_TEXELS`.
const SUN_MASK_DISTANCE_TEXELS: f32 = {sun_mask_range:?};
// The stationary lamps' shadow masks, and their distance range. See
// `STATIONARY_MASK_DISTANCE_TEXELS`.
@group(2) @binding(5) var lm_stationary: texture_2d_array<f32>;
const STATIONARY_MASK_DISTANCE_TEXELS: f32 = {stationary_range:?};

@group(1) @binding(0) var mat_color: texture_2d_array<f32>;
@group(1) @binding(1) var mat_normal: texture_2d_array<f32>;
@group(1) @binding(2) var mat_samp: sampler;
@group(1) @binding(3) var mat_rough: texture_2d_array<f32>;
@group(1) @binding(4) var mat_ao: texture_2d_array<f32>;
// Trilinear, so the roughness mip that carries the normal's variance is
// the one actually fetched. See `brush_rough_sampler`.
@group(1) @binding(5) var mat_rough_samp: sampler;
// Each material's proportions: `u` as authored, `v` times its colour map's
// width over height. See `material_uv_scales`.
@group(1) @binding(6) var<uniform> mat_uv_scale: array<vec4<f32>, {max_materials}>;

{lights_block}
{ssr_block}

struct VIn {{
    @location(0) pos: vec3<f32>,
    @location(1) norm: vec3<f32>,
    @location(2) tangent: vec4<f32>,
    @location(3) uv: vec2<f32>,
    @location(4) material: u32,
    @location(5) tint: vec4<f32>,
    @location(6) uv2: vec2<f32>,
    @location(7) face_centre: vec3<f32>,
    @location(8) uv2_rect: vec4<f32>,
    @location(9) face_half_extent: vec2<f32>,
}}
struct VOut {{
    // INVARIANT: computed identically in every pipeline built from this
    // vertex stage -- the depth prepass and the shading passes must put each
    // brush at the same depth to the bit, or LessEqual fails where they
    // differ in the last place and the surface flickers away.
    @builtin(position) @invariant clip: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) tangent: vec4<f32>,
    // NOT centroid. Centroid interpolation stopped MSAA edge pixels being
    // shaded past the polygon, but the scene pass went from ~24 ms to ~40 ms
    // for both eyes the moment it shipped (headset, 2026-09-17) -- and that
    // was centroid on THIS ONE varying, so the cheap version is the one that
    // was measured and rejected.
    //
    // TWO MARGIN FIXES WERE TRIED AND BOTH FAILED (headset, 2026-09-19): a
    // constant `PROBE_BOX_MARGIN` scaled by render scale, then a margin
    // widened per pixel by `fwidth(world_pos)`. The dotted line survived both
    // at the same spacing, so "the fragment falls slightly outside its box"
    // is NOT the mechanism, and both were reverted rather than left in as
    // complexity that fixed nothing.
    //
    // What the sources view DOES show is that the seam pixels take far more
    // PROBE than their neighbours -- sampled across the seam row, green
    // (baked) drops while blue (probe) spikes 3x. So the fault is in probe
    // SELECTION, and the fix is to stop selecting per fragment at all: see
    // the per-face probe index below.
    @location(2) {centroid}world_pos: vec3<f32>,
    @location(3) uv: vec2<f32>,
    // Flat: a material index is a choice, not a quantity, and interpolating it
    // across a triangle that spans two materials would sample layer 1.5.
    @location(4) @interpolate(flat) material: u32,
    @location(5) tint: vec4<f32>,
    @location(6) {centroid}uv2: vec2<f32>,
    // FLAT, and that is the entire fix. Every vertex of a face carries the
    // same centre, so interpolation would be a no-op in exact arithmetic --
    // but `flat` also makes it immune to the EXTRAPOLATION that happens when
    // MSAA shades a pixel centre lying outside the polygon. That extrapolation
    // is what broke probe selection at room seams, so a smoothly-interpolated
    // copy of this value would inherit the very bug it exists to avoid.
    @location(7) @interpolate(flat) face_centre: vec3<f32>,
    // FLAT, for the same reason the face centre is: the value is constant over
    // the face, and a smoothly-interpolated copy would be EXTRAPOLATED at an
    // edge pixel -- inheriting the very bug it exists to bound.
    @location(8) @interpolate(flat) uv2_rect: vec4<f32>,
    // FLAT, like the other two per-face bounds beside it -- a smoothly
    // interpolated copy would be extrapolated at the very pixels it exists to
    // correct. See `BrushVertex::face_half_extent`.
    @location(9) @interpolate(flat) face_half_extent: vec2<f32>,
{probe_face_varying}{edge_varying}}}

@vertex fn vs_main(v: VIn) -> VOut {{
    var out: VOut;
    out.clip      = cam_view_proj() * vec4<f32>(v.pos, 1.0);
    out.normal    = v.norm;
    out.tangent   = v.tangent;
    out.world_pos = v.pos;
    out.face_centre = v.face_centre;
    out.uv2_rect  = v.uv2_rect;
    out.face_half_extent = v.face_half_extent;
    // At the material's own proportions, so no map is stretched to its tile.
    out.uv        = v.uv * mat_uv_scale[min(v.material, {max_materials}u - 1u)].xy;
    out.uv2       = v.uv2;
    out.material  = v.material;
    out.tint      = v.tint;
{probe_face_vs}{edge_vs}    return out;
}}

// THE DEPTH PREPASS's fragment stage: nothing, with every colour write
// masked. It exists only because a pass with a colour attachment takes a
// pipeline with a matching colour target. See `BrushPipeline::new_depth_prepass`.
@fragment fn fs_depth() -> @location(0) vec4<f32> {{
    return vec4<f32>(0.0);
}}

@fragment fn fs_main(in: VOut) -> @location(0) vec4<f32> {{
    // See `pixel_footprint` in the lights block: taken here, in uniform
    // control flow, so the light loop can keep a spot's edge a pixel wide.
    //
    // The GEOMETRIC MEAN of the two screen axes, not their sum: seen at a
    // grazing angle one axis stretches to metres while the other stays a
    // pixel, and the sum let that stretch widen a spot's pool across the floor.
    pixel_footprint = sqrt(length(dpdx(in.world_pos)) * length(dpdy(in.world_pos)));
    let albedo = textureSample(mat_color, mat_samp, in.uv, i32(in.material));

    // Tangent frame from the brush face's own axes, re-orthogonalised against
    // the interpolated normal so the two cannot drift apart.
    let n_geom = normalize(in.normal);
    let t = normalize(in.tangent.xyz - n_geom * dot(n_geom, in.tangent.xyz));
    let b = cross(n_geom, t) * in.tangent.w;
    let tn = textureSample(mat_normal, mat_samp, in.uv, i32(in.material)).xyz * 2.0 - 1.0;
    let n = normalize(t * tn.x + b * tn.y + n_geom * tn.z);

    // ADDED, not multiplied.
    //
    // The mesh pipeline multiplies its lightmap in, and multiplying cannot ADD
    // light: a wall the lamp never reaches directly sits at the ambient floor
    // and no amount of bounce can lift it, which is most of a real interior.
    // Bounced light is light and belongs in the sum.
    //
    // Adding is only correct because a light is either baked or realtime and
    // never both -- see LightMode. `shade` sees only the realtime ones, the
    // texture carries only the baked ones, and the ambient term belongs solely
    // to `shade`, which varies it by the direction the surface faces.
    //
    // An unbaked brush binds a BLACK texture, so this collapses to exactly the
    // old behaviour. Black is the neutral value for a sum the way white is for
    // a product, and binding the wrong one is a full stop of extra brightness
    // on every surface.
    // The material's own roughness and occlusion, so a polished surface and a
    // matte one do not light identically. A material with no map gets white in
    // both arrays, which is fully rough and fully unoccluded -- exactly how
    // every brush looked before these existed.
{lighting}{tail}
}}
"#,
        lights_block = crate::renderer::lights::wgsl_lights_block_with(
            0,
            1,
            probe == BrushProbe::Read,
            pass_like,
            probe == BrushProbe::PassDeferred,
        ),
        ssr_block = ssr_block,
        sun_mask_range = SUN_MASK_DISTANCE_TEXELS,
        stationary_range = STATIONARY_MASK_DISTANCE_TEXELS,
        max_materials = MAX_BRUSH_MATERIALS,
        lighting = lighting,
        centroid = if BRUSH_CENTROID_VARYINGS {
            "@interpolate(perspective, centroid) "
        } else {
            ""
        },
        edge_varying = edge_varying,
        edge_vs = edge_vs,
        // THE PROBE PASS CHOOSES ITS ROOM PER FACE, in the vertex stage. See
        // `probe_face_room`: the box tests against every resident probe were
        // made per pixel for a result that is the same across the face.
        probe_face_varying = if pass_like {
            "    // FLAT: the face's room. See `probe_face_room`.\n    @location(10) @interpolate(flat) probe_face: vec4<f32>,\n"
        } else {
            ""
        },
        probe_face_vs = if pass_like {
            "    out.probe_face = probe_face_room(v.face_centre);\n"
        } else {
            ""
        },
        tail = if pass_like {
            String::new()
        } else if BRUSH_EDGE_DEBUG {
            edge_debug.to_string()
        } else if sources {
            match SOURCES_VIEW {
                SourcesView::Ratios => source_debug.to_string(),
                SourcesView::ProbeFactors => {
                    "\n    return vec4<f32>(dbg_probe_factors, 1.0);".to_string()
                }
                SourcesView::Absolute => r#"
    let abs_lum = vec3<f32>(
        dot(dbg_baked, vec3<f32>(0.2126, 0.7152, 0.0722)),
        dot(dbg_probe, vec3<f32>(0.2126, 0.7152, 0.0722)),
        dot(dbg_direct, vec3<f32>(0.2126, 0.7152, 0.0722)),
    );
    return vec4<f32>(abs_lum / (abs_lum + vec3<f32>(0.05)), 1.0);"#
                    .to_string(),
            }
        } else {
            ssr_apply
        }
    );
    match probe {
        BrushProbe::Read => format!("{src}{}", probe_pass::READER_WGSL),
        _ => src,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::renderer::lights::{Light, LightKind, LightsUniform};
    use crate::renderer::Color3;
    use crate::renderer::terrain_pipeline::tests::headless_gpu;
    use crate::renderer::uniforms::test_support::{scene_uniforms, TEST_EYE};
    use crate::renderer::uniforms::ShadowUpload;
    use wgpu::util::DeviceExt;

    /// A flat image of one colour, standing in for a material's colour map.
    fn flat(rgb: [u8; 3], size: u32) -> TerrainImage {
        TerrainImage {
            width: size,
            height: size,
            rgba: (0..size * size)
                .flat_map(|_| [rgb[0], rgb[1], rgb[2], 255])
                .collect(),
        }
    }

    /// A normal map that tilts every texel hard along +u.
    ///
    /// Encoded the way a normal map is: 0..255 mapping to -1..1, so 255 in red
    /// is a full tilt toward the face's u axis and 128 is no tilt at all.
    /// Tilted along the face's V axis, so the surface leans UP.
    ///
    /// `tilted_normal` leans along U, which for a +Z face is world +X -- and a
    /// surface tilted in X faces light arriving from +Y and from -Y exactly
    /// alike. A test of "which side did the bounce come from" needs a normal
    /// that can tell those apart, and this is it.
    fn tilted_normal_up(size: u32) -> TerrainImage {
        TerrainImage {
            width: size,
            height: size,
            rgba: (0..size * size).flat_map(|_| [128, 250, 140, 255]).collect(),
        }
    }

    fn tilted_normal(size: u32) -> TerrainImage {
        TerrainImage {
            width: size,
            height: size,
            rgba: (0..size * size).flat_map(|_| [250, 128, 140, 255]).collect(),
        }
    }

    /// Draw one triangle of a brush and read the centre pixel back.
    ///
    /// Renders rather than merely building the pipeline. Creating the pipeline
    /// proves the WGSL parses and nothing more -- it cannot tell whether the
    /// right array layer was sampled, whether the tangent frame is the right way
    /// round, or whether the normal map moves the shading at all.
    fn render_brush(
        material: u32,
        colours: &[Option<TerrainImage>],
        normals: &[Option<TerrainImage>],
        tint: [f32; 4],
        uv_scale: f32,
        lit: bool,
    ) -> Option<[u8; 4]> {
        render_brush_material(
            material, colours, normals, &[], &[], tint, uv_scale, lit, SIDE_LIGHT,
        )
    }

    /// A 2:1 MATERIAL KEEPS ITS PROPORTIONS (tracker 2.3): its tile is as wide
    /// as authored and half as tall, so up a face it repeats twice as often as
    /// a square one. Read at v = 0.4 of an authored tile, a square map is in
    /// its top half; a 2:1 map is already at 0.8 of its own height, in its
    /// bottom half. Before, every 1024x512 map -- the hallway rock, the
    /// bricks -- drew every stone twice as tall as it is.
    #[test]
    fn a_two_to_one_material_is_not_stretched_to_its_tile() {
        // Top half red, bottom half blue.
        let halves = |w: u32, h: u32| TerrainImage {
            width: w,
            height: h,
            rgba: (0..h)
                .flat_map(|y| (0..w).flat_map(move |_| if y < h / 2 { [255, 0, 0, 255] } else { [0, 0, 255, 255] }))
                .collect(),
        };
        let Some(square) = render_brush(0, &[Some(halves(4, 4))], &[None], [1.0; 4], 0.8, false) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let wide = render_brush(0, &[Some(halves(4, 2))], &[None], [1.0; 4], 0.8, false).unwrap();
        assert!(square[0] > square[2] + 20, "a square map at v = 0.4 is in its red top half: {square:?}");
        assert!(wide[2] > wide[0] + 20, "a 2:1 map at v = 0.4 of a tile is in its blue bottom half: {wide:?}");
    }

    #[test]
    fn a_materials_uv_scale_is_its_width_over_its_height() {
        let img = |w, h| Some(TerrainImage { width: w, height: h, rgba: vec![0; (w * h * 4) as usize] });
        let scales = material_uv_scales(&[img(1024, 512), img(1024, 1024), None, img(512, 1024)]);
        assert_eq!(scales[0][..2], [1.0, 2.0]);
        assert_eq!(scales[1][..2], [1.0, 1.0]);
        assert_eq!(scales[2][..2], [1.0, 1.0], "a missing map is square");
        assert_eq!(scales[3][..2], [1.0, 0.5]);
        assert_eq!(scales[MAX_BRUSH_MATERIALS - 1][..2], [1.0, 1.0], "past the list, square");
    }

    /// Off to one side, so a normal tilted along u faces it differently from a
    /// flat one. What the normal-map tests need.
    const SIDE_LIGHT: glam::Vec3 = glam::Vec3::new(4.0, 0.0, 2.0);

    /// Straight down the view axis, so the half-vector lands on the normal and
    /// the specular highlight is at its peak.
    ///
    /// What the ROUGHNESS tests need. With the side light both a mirror and a
    /// matte surface come out with no highlight at all -- the matte one because
    /// its strength is zero, the mirror because its lobe is a pinpoint the
    /// sample misses -- so the two are identical and the test proves nothing.
    const HEAD_ON_LIGHT: glam::Vec3 = glam::Vec3::new(0.0, 0.0, 4.0);

    /// The same, with roughness and ambient-occlusion arrays supplied.
    ///
    /// Empty slices fill with white -- fully rough, fully unoccluded -- which is
    /// what every surface looked like before either map existed, so the older
    /// tests measure exactly what they always did.
    /// The neutral lightmap: adds nothing, and reports full sky visibility.
    const NEUTRAL_LM: [u8; 4] = [0, 0, 0, 255];

    /// The material harness with the baked lightmap left neutral.
    ///
    /// Every material test predates the lightmap and must keep measuring what
    /// it always did, so they all funnel through here rather than each carrying
    /// a texel they do not care about.
    #[allow(clippy::too_many_arguments)]
    fn render_brush_material(
        material: u32,
        colours: &[Option<TerrainImage>],
        normals: &[Option<TerrainImage>],
        roughs: &[Option<TerrainImage>],
        aos: &[Option<TerrainImage>],
        tint: [f32; 4],
        uv_scale: f32,
        lit: bool,
        light_pos: glam::Vec3,
    ) -> Option<[u8; 4]> {
        render_brush_baked(
            material, colours, normals, roughs, aos, tint, uv_scale, lit, light_pos, NEUTRAL_LM,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn render_brush_baked(
        material: u32,
        colours: &[Option<TerrainImage>],
        normals: &[Option<TerrainImage>],
        roughs: &[Option<TerrainImage>],
        aos: &[Option<TerrainImage>],
        tint: [f32; 4],
        uv_scale: f32,
        lit: bool,
        light_pos: glam::Vec3,
        lightmap_texel: [u8; 4],
    ) -> Option<[u8; 4]> {
        render_brush_baked_dir(
            material, colours, normals, roughs, aos, tint, uv_scale, lit, light_pos,
            lightmap_texel,
            crate::renderer::mesh::NEUTRAL_BOUNCE_DIRECTION,
        )
    }

    /// `render_brush_baked`, with a baked bounce DIRECTION as well.
    ///
    /// Split rather than adding a parameter to the original because the neutral
    /// direction has to stay the default: every existing test asserts absolute
    /// pixel values, and they are only still meaningful if a test that says
    /// nothing about direction gets exactly the shading it always got.
    #[allow(clippy::too_many_arguments)]
    fn render_brush_baked_dir(
        material: u32,
        colours: &[Option<TerrainImage>],
        normals: &[Option<TerrainImage>],
        roughs: &[Option<TerrainImage>],
        aos: &[Option<TerrainImage>],
        tint: [f32; 4],
        uv_scale: f32,
        lit: bool,
        light_pos: glam::Vec3,
        lightmap_texel: [u8; 4],
        direction_texel: [u8; 4],
    ) -> Option<[u8; 4]> {
        render_brush_probed(
            material, colours, normals, roughs, aos, tint, uv_scale, lit, light_pos,
            lightmap_texel, direction_texel, None,
        )
    }

    /// HALF-FLOAT LIGHT SHADES EXACTLY AS THE 8-BIT FORM OF THE SAME LIGHT.
    ///
    /// Every bake is now stored as half floats; an older one as sRGB bytes.
    /// The same light in either must draw the same pixel, or re-baking a level
    /// would shift its brightness. And light past 1.0, which the bytes clipped,
    /// must now draw brighter than 1.0 does.
    #[test]
    fn half_float_and_srgb_lightmaps_of_the_same_light_draw_the_same() {
        let at = |lm: crate::renderer::mesh::LightmapLight| {
            render_brush_lit_by(
                0, &[], &[], &[], &[], [1.0; 4], 1.0, false, glam::Vec3::ZERO,
                lm, crate::renderer::mesh::NEUTRAL_BOUNCE_DIRECTION, None, None,
            )
        };
        // Byte 188 is sRGB for linear 0.5029.
        let Some(bytes) = at(crate::renderer::mesh::LightmapLight::Srgb8(&[188, 188, 188, 255])) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let v = 0.5029f32;
        let floats = at(crate::renderer::mesh::LightmapLight::Linear(&[v, v, v, 1.0])).unwrap();
        for c in 0..3 {
            assert!((bytes[c] as i32 - floats[c] as i32).abs() <= 1, "{bytes:?} vs {floats:?}");
        }
        let bright = at(crate::renderer::mesh::LightmapLight::Linear(&[3.0, 3.0, 3.0, 1.0])).unwrap();
        let one = at(crate::renderer::mesh::LightmapLight::Linear(&[1.0, 1.0, 1.0, 1.0])).unwrap();
        assert!(bright[1] > one[1], "light past 1.0 was clipped: {bright:?} vs {one:?}");
    }

    /// THE SKY SUN ON A BRUSH IS ITS BAKED MASK.
    ///
    /// A black lightmap and the sky's sun head-on: with the mask at 0 the wall
    /// is in the level's baked shadow and gets none of the sun; at 255 it gets
    /// all of it; at half, about half. And a mask the baker never wrote (green
    /// 0) must not read as shadow -- it falls back to the static map, which
    /// this rig leaves off, so the sun arrives in full.
    #[test]
    fn the_sky_sun_on_a_brush_is_scaled_by_its_baked_mask() {
        let at = |mask: [u8; 4]| {
            render_brush_full(
                0, &[], &[], &[], &[], [1.0; 4], 1.0, false, glam::Vec3::ZERO,
                NEUTRAL_LM, crate::renderer::mesh::NEUTRAL_BOUNCE_DIRECTION, None, Some(mask),
            )
        };
        let Some(dark) = at([0, 255, 0, 255]) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let lit = at([255, 255, 0, 255]).unwrap();
        let half = at([128, 255, 0, 255]).unwrap();
        let unbaked = at([0, 0, 0, 0]).unwrap();
        eprintln!("mask 0: {dark:?}  128: {half:?}  255: {lit:?}  unbaked: {unbaked:?}");
        assert!(lit[1] > dark[1] + 40, "the sun did not light an unshadowed wall: {dark:?} vs {lit:?}");
        assert!(half[1] > dark[1] && half[1] < lit[1], "half a sun is not between none and all");
        assert!(
            (unbaked[1] as i32 - lit[1] as i32).abs() <= 2,
            "an unbaked mask read as shadow: {unbaked:?} vs {lit:?}",
        );
    }

    /// The same, with a reflection probe covering the surface.
    ///
    /// `probe` is the cube's flat colour and the world box it covers. `None`
    /// binds the neutral cube, which is what every test that predates probes
    /// wants -- and what an unbaked level gets.
    #[allow(clippy::too_many_arguments)]
    fn render_brush_probed(
        material: u32,
        colours: &[Option<TerrainImage>],
        normals: &[Option<TerrainImage>],
        roughs: &[Option<TerrainImage>],
        aos: &[Option<TerrainImage>],
        tint: [f32; 4],
        uv_scale: f32,
        lit: bool,
        light_pos: glam::Vec3,
        lightmap_texel: [u8; 4],
        direction_texel: [u8; 4],
        probe: Option<([u8; 4], glam::Vec3, glam::Vec3)>,
    ) -> Option<[u8; 4]> {
        render_brush_full(
            material, colours, normals, roughs, aos, tint, uv_scale, lit, light_pos,
            lightmap_texel, direction_texel, probe, None,
        )
    }

    /// The whole harness. `sky_sun`, when set, lights the brush with the SKY's
    /// sun alone -- head-on, flagged as the sky sun exactly as the frame
    /// uploads it -- and binds that sun-mask texel beside the lightmap.
    #[allow(clippy::too_many_arguments)]
    fn render_brush_full(
        material: u32,
        colours: &[Option<TerrainImage>],
        normals: &[Option<TerrainImage>],
        roughs: &[Option<TerrainImage>],
        aos: &[Option<TerrainImage>],
        tint: [f32; 4],
        uv_scale: f32,
        lit: bool,
        light_pos: glam::Vec3,
        lightmap_texel: [u8; 4],
        direction_texel: [u8; 4],
        probe: Option<([u8; 4], glam::Vec3, glam::Vec3)>,
        sky_sun: Option<[u8; 4]>,
    ) -> Option<[u8; 4]> {
        render_brush_lit_by(
            material, colours, normals, roughs, aos, tint, uv_scale, lit, light_pos,
            crate::renderer::mesh::LightmapLight::Srgb8(&lightmap_texel), direction_texel, probe, sky_sun,
        )
    }

    /// `render_brush_full` with the lightmap in either storage.
    #[allow(clippy::too_many_arguments)]
    fn render_brush_lit_by(
        material: u32,
        colours: &[Option<TerrainImage>],
        normals: &[Option<TerrainImage>],
        roughs: &[Option<TerrainImage>],
        aos: &[Option<TerrainImage>],
        tint: [f32; 4],
        uv_scale: f32,
        lit: bool,
        light_pos: glam::Vec3,
        lightmap: crate::renderer::mesh::LightmapLight,
        direction_texel: [u8; 4],
        probe: Option<([u8; 4], glam::Vec3, glam::Vec3)>,
        sky_sun: Option<[u8; 4]>,
    ) -> Option<[u8; 4]> {
        let (device, queue) = headless_gpu()?;
        let format = TextureFormat::Rgba8Unorm;

        let lights = LightsUniform::new(&device);
        if sky_sun.is_some() {
            // Travelling -z onto a +z-facing surface: N.L = 1.
            lights.upload_frame(
                &queue,
                &[Light {
                    mask_channel: None,
                    position: glam::Vec3::ZERO,
                    direction: glam::Vec3::new(0.0, 0.0, -1.0),
                    kind: LightKind::Directional,
                    color: Color3(255, 255, 255, 255),
                    intensity: 0.5,
                    range: 0.0,
                    cone_angle_deg: 0.0,
                    inner_cone_angle_deg: 0.0,
                }],
                &[],
                true,
            );
        } else if lit {
            // Off to one side, so a normal tilted along u faces it differently
            // from a flat one. With an empty list `shade` returns the ambient
            // constant whatever the normal, and a normal-map test on that rig
            // would pass without the mapping doing anything.
            lights.upload(
                &queue,
                &[Light {
                    mask_channel: None,
                    position: light_pos,
                    direction: glam::Vec3::new(-1.0, 0.0, 0.0),
                    kind: LightKind::Point,
                    color: Color3(255, 255, 255, 255),
                    intensity: 4.0,
                    range: 20.0,
                    cone_angle_deg: 90.0,
                    inner_cone_angle_deg: 0.0,
                }],
            );
        } else {
            lights.upload(&queue, &[]);
        }
        // `_probe_keep` holds the cube view and sampler alive: the scene bind
        // group references them, and dropping them here would leave it
        // pointing at freed resources.
        let (_shadows, uniforms, _probe_keep) = match probe {
            Some((colour, lo, hi)) => {
                let (sh, u, v, samp) = crate::renderer::uniforms::test_support::scene_uniforms_with_probe(
                    &device, &queue, &lights, colour, lo, hi,
                );
                (sh, u, Some((v, samp)))
            }
            None => {
                let (sh, u) = scene_uniforms(&device, &lights);
                (sh, u, None)
            }
        };
        uniforms.upload(&queue, glam::Mat4::IDENTITY, TEST_EYE, &ShadowUpload::disabled());

        let pipeline = BrushPipeline::new(&device, format, &uniforms.layout);
        let materials =
            // These tests are about the COLOUR and NORMAL path, so both new
            // arrays are left empty: they fill with white, which is fully rough
            // and fully unoccluded -- the shading every one of them was written
            // against.
            BrushMaterials::new(
                &device, &queue, &pipeline.material_layout, colours, normals, roughs, aos,
            );

        // A triangle covering the viewport in clip space, facing +z, with the
        // face's u axis along +x -- the frame a wall brush actually produces.
        let v = |p: [f32; 3], uv: [f32; 2]| BrushVertex {
            position: p,
            normal: [0.0, 0.0, 1.0],
            tangent: [1.0, 0.0, 0.0, 1.0],
            uv,
            material,
            tint,
            // These tests are about the MATERIAL path, so every vertex samples
            // the same lightmap texel and the harness binds a black one --
            // additively neutral, so it changes none of their measurements.
            uv2: [0.5, 0.5],
            // The probe is chosen from here. These fixtures have no probes
            // bound, so any value behaves alike; the triangle's own centre is
            // the honest one.
            face_centre: [1.0 / 3.0, 1.0 / 3.0, 0.0],
            // The whole atlas: these fixtures bind a 1x1 lightmap, so any
            // clamp is a no-op and the full range is the honest value.
            uv2_rect: [0.0, 0.0, 1.0, 1.0],
            // NO CLAMP. These fixtures are about the material and the
            // pipeline, not about edge extrapolation, and infinite extent is
            // exactly the behaviour they measured before the clamp existed --
            // so every expectation in them still means what it meant.
            face_half_extent: [f32::INFINITY; 2],
        };
        let verts = [
            v([-1.0, -1.0, 0.0], [0.0, 0.0]),
            v([3.0, -1.0, 0.0], [2.0 * uv_scale, 0.0]),
            v([-1.0, 3.0, 0.0], [0.0, 2.0 * uv_scale]),
        ];

        let vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("brush_test_vb"),
            contents: bytemuck::cast_slice(&verts),
            usage: BufferUsages::VERTEX,
        });
        let ib = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("brush_test_ib"),
            contents: bytemuck::cast_slice(&[0u32, 1, 2]),
            usage: BufferUsages::INDEX,
        });

        const SIZE: u32 = 8;
        let target = device.create_texture(&TextureDescriptor {
            label: Some("brush_test_target"),
            size: Extent3d { width: SIZE, height: SIZE, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let depth = device.create_texture(&TextureDescriptor {
            label: Some("brush_test_depth"),
            size: Extent3d { width: SIZE, height: SIZE, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Depth32Float,
            usage: TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let target_view = target.create_view(&Default::default());
        let depth_view = depth.create_view(&Default::default());

        let readback = device.create_buffer(&BufferDescriptor {
            label: Some("brush_test_readback"),
            size: (256 * SIZE) as u64,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&Default::default());
        {
        let sun_texel = sky_sun.unwrap_or(crate::renderer::mesh::NEUTRAL_SUN_MASK);
        let lightmap = crate::renderer::mesh::create_lightmap_texture_with_sun(
            &device,
            &queue,
            &pipeline.lightmap_layout,
            lightmap,
            1,
            1,
            Some((&direction_texel, 1, 1)),
            Some((&sun_texel, 1, 1)),
        );

            let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("brush_test_pass"),
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
            pass.set_bind_group(1, &materials.bind_group, &[]);
            // Neutral by default -- black adds nothing and alpha 255 leaves the
            // sky term alone -- so these material tests measure the material
            // path and nothing else.
            pass.set_bind_group(2, &lightmap.bind_group, &[]);
            pass.set_vertex_buffer(0, vb.slice(..));
            pass.set_index_buffer(ib.slice(..), IndexFormat::Uint32);
            pass.draw_indexed(0..3, 0, 0..1);
        }
        encoder.copy_texture_to_buffer(
            TexelCopyTextureInfo {
                texture: &target,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            TexelCopyBufferInfo {
                buffer: &readback,
                layout: TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(256),
                    rows_per_image: Some(SIZE),
                },
            },
            Extent3d { width: SIZE, height: SIZE, depth_or_array_layers: 1 },
        );
        queue.submit(Some(encoder.finish()));

        let slice = readback.slice(..);
        slice.map_async(MapMode::Read, |_| {});
        device.poll(PollType::Wait { submission_index: None, timeout: None }).ok();
        let data = slice.get_mapped_range().unwrap();
        let c = (SIZE / 2) as usize * 256 + (SIZE / 2) as usize * 4;
        Some([data[c], data[c + 1], data[c + 2], data[c + 3]])
    }

    /// Three materials, distinguishable by which channel dominates.
    fn palette() -> Vec<Option<TerrainImage>> {
        vec![
            Some(flat([200, 20, 20], 4)),
            Some(flat([20, 200, 20], 4)),
            Some(flat([20, 20, 200], 4)),
        ]
    }

    #[test]
    fn a_face_samples_the_material_its_index_names() {
        // The whole reason the index is per vertex: one draw call, many
        // materials. If the array layer were ignored every wall in a level
        // would come out the same colour, which reads as a texture-loading
        // failure rather than as an indexing one.
        for (index, expected) in [(0u32, 0usize), (1, 1), (2, 2)] {
            let Some(px) = render_brush(index, &palette(), &[], [1.0; 4], 1.0, false) else {
                eprintln!("skipping: no GPU adapter available");
                return;
            };
            let brightest = (0..3).max_by_key(|i| px[*i]).unwrap();
            assert_eq!(
                brightest, expected,
                "material {index} should sample layer {expected}, got {px:?}"
            );
        }
    }

    #[test]
    fn the_tint_multiplies_the_sampled_colour() {
        // A face whose material is missing binds a white layer, so the tint is
        // the entire appearance -- which is what keeps an unassigned brush
        // looking like the colour its object was authored in.
        let Some(white) = render_brush(0, &[], &[], [1.0, 1.0, 1.0, 1.0], 1.0, false) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let red = render_brush(0, &[], &[], [1.0, 0.0, 0.0, 1.0], 1.0, false).unwrap();
        assert!(white[1] > 40, "an absent material is white, not black: {white:?}");
        assert!(red[0] > red[1], "the tint must reach the output: {red:?}");
        assert!(red[1] < 20, "and must actually remove the channels it zeroes");
    }

    #[test]
    fn a_missing_normal_map_leaves_the_face_flat() {
        // The fallback layer is 128,128,255 -- straight out of the surface. A
        // fallback of zeros would decode to (-1,-1,-1) and light every
        // untextured wall as though it faced away from everything.
        let Some(with_fallback) = render_brush(0, &palette(), &[], [1.0; 4], 1.0, true) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let flat_map = render_brush(
            0,
            &palette(),
            &[Some(flat([128, 128, 255], 4))],
            [1.0; 4],
            1.0,
            true,
        )
        .unwrap();
        assert_eq!(
            with_fallback, flat_map,
            "no normal map must shade identically to an explicitly flat one"
        );
    }

    #[test]
    fn roughness_changes_the_specular_highlight() {
        // THE POINT of the roughness map. Without it every surface in a level
        // shades identically -- polished concrete lights exactly like rough
        // brick -- and no work on the lighting can tell them apart.
        //
        // A smooth surface concentrates the same energy into a tighter,
        // brighter highlight, so head on it must out-shine a matte one.
        let Some(matte) = render_brush_material(
            0, &palette(), &[None], &[Some(flat([255, 255, 255], 4))], &[], [1.0; 4], 1.0, true,
            HEAD_ON_LIGHT,
        ) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let glossy = render_brush_material(
            0, &palette(), &[None], &[Some(flat([20, 20, 20], 4))], &[], [1.0; 4], 1.0, true,
            HEAD_ON_LIGHT,
        )
        .unwrap();
        assert_ne!(
            matte, glossy,
            "roughness is bound but never reaches the shading: {matte:?} vs {glossy:?}",
        );
        assert!(
            glossy[0] > matte[0],
            "a smooth surface should out-shine a matte one head on: {glossy:?} vs {matte:?}",
        );
    }

    #[test]
    fn a_missing_roughness_map_takes_the_documented_default() {
        // SUPERSEDES a test that asserted a missing map must shade identically
        // to an explicitly fully-rough one. That was written to protect levels
        // already built from being relit, which was the right instinct and the
        // wrong constant: roughness 1.0 zeroes the specular term outright, so
        // the guarantee it locked in was "a material without a map can never
        // shine". See `DEFAULT_ROUGHNESS`.
        //
        // The property still worth pinning is that the default is the DOCUMENTED
        // one, so the constant and the texture cannot drift apart.
        let d = DEFAULT_ROUGHNESS;
        let Some(absent) = render_brush_material(
            0, &palette(), &[None], &[], &[], [1.0; 4], 1.0, true, HEAD_ON_LIGHT,
        ) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let explicit_default = render_brush_material(
            0, &palette(), &[None], &[Some(flat([d, d, d], 4))], &[], [1.0; 4], 1.0, true,
            HEAD_ON_LIGHT,
        )
        .unwrap();
        assert_eq!(
            absent, explicit_default,
            "no roughness map must shade as DEFAULT_ROUGHNESS: {absent:?} vs {explicit_default:?}",
        );
    }

    #[test]
    fn ambient_occlusion_darkens_without_wiping_out_direct_light() {
        // AO says how much of the SKY a crevice can see. Applying it to direct
        // light as well would darken a surface a lamp is shining straight onto,
        // which is a different effect and a wrong one.
        let Some(open) = render_brush_material(
            0, &palette(), &[None], &[], &[Some(flat([255, 255, 255], 4))], [1.0; 4], 1.0, true,
            HEAD_ON_LIGHT,
        ) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let occluded = render_brush_material(
            0, &palette(), &[None], &[], &[Some(flat([0, 0, 0], 4))], [1.0; 4], 1.0, true,
            HEAD_ON_LIGHT,
        )
        .unwrap();
        assert!(
            occluded[0] < open[0],
            "ambient occlusion darkened nothing: {occluded:?} vs {open:?}",
        );
        assert!(
            occluded[0] > 0,
            "AO wiped out the direct light too, not just the ambient: {occluded:?}",
        );
    }

    #[test]
    fn baked_sky_visibility_darkens_the_ambient_term() {
        // The lightmap's ALPHA is how much sky the texel can actually see. With
        // it ignored, a wall sealed inside a room was lit by the sky as
        // brightly as open ground -- which is why interiors looked flat and
        // shadows looked weak: the shadow was cast, and a full-strength ambient
        // term filled it straight back in.
        //
        // Unlit on purpose. The sky term is the whole measurement, and a lamp
        // in the frame would swamp it.
        let Some(open) = render_brush_baked(
            0, &palette(), &[None], &[], &[], [1.0; 4], 1.0, false, SIDE_LIGHT, [0, 0, 0, 255],
        ) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let sealed = render_brush_baked(
            0, &palette(), &[None], &[], &[], [1.0; 4], 1.0, false, SIDE_LIGHT, [0, 0, 0, 0],
        )
        .unwrap();
        assert!(
            sealed[0] < open[0],
            "sky visibility darkened nothing: sealed {sealed:?} vs open {open:?}",
        );
    }

    #[test]
    fn baked_rgb_still_adds_with_sky_visibility_at_zero() {
        // The two channels of one texel have OPPOSITE neutrals, so the risk in
        // reading both is that one gets applied to the other: multiplying the
        // added bounce by visibility would make a fully-occluded texel unable
        // to show any baked light at all, and a sealed room lit only by a baked
        // lamp would come out black.
        let Some(dark) = render_brush_baked(
            0, &palette(), &[None], &[], &[], [1.0; 4], 1.0, false, SIDE_LIGHT, [0, 0, 0, 0],
        ) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let baked = render_brush_baked(
            0, &palette(), &[None], &[], &[], [1.0; 4], 1.0, false, SIDE_LIGHT, [200, 200, 200, 0],
        )
        .unwrap();
        assert!(
            baked[0] > dark[0],
            "baked light was scaled away by zero sky visibility: {baked:?} vs {dark:?}",
        );
    }

    #[test]
    #[ignore = "measurement, not an assertion: prints the specular response"]
    fn measure_specular_response() {
        let Some(_) = headless_gpu() else { return };
        for (label, rough) in [
            ("marble (0.048)", 12u8),
            ("default (0.55)", DEFAULT_ROUGHNESS),
            ("fully rough", 255u8),
        ] {
            let lit = render_brush_material(
                0, &palette(), &[None], &[Some(flat([rough, rough, rough], 4))], &[],
                [1.0; 4], 1.0, true, HEAD_ON_LIGHT,
            )
            .unwrap();
            let unlit = render_brush_material(
                0, &palette(), &[None], &[Some(flat([rough, rough, rough], 4))], &[],
                [1.0; 4], 1.0, false, HEAD_ON_LIGHT,
            )
            .unwrap();
            eprintln!("{label:>16}: lit {lit:?}  unlit {unlit:?}  delta {}", lit[0] as i32 - unlit[0] as i32);
        }
    }

    #[test]
    fn a_material_with_no_roughness_map_can_still_shine() {
        // The regression that made "the specular isn't working" literally true:
        // a missing roughness map defaulted to WHITE, which is fully rough, and
        // spec_strength = SPEC_STRENGTH * (1 - r) is then exactly zero. No
        // light, no angle and no material could produce a highlight.
        //
        // Head-on light, so the half-vector lands on the normal and the lobe is
        // at its peak -- with the side light a matte and a mirror surface both
        // read as nothing and the test proves nothing.
        let Some(matte) = render_brush_material(
            0, &palette(), &[None], &[Some(flat([255, 255, 255], 4))], &[], [1.0; 4], 1.0, true,
            HEAD_ON_LIGHT,
        ) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let absent = render_brush_material(
            0, &palette(), &[None], &[], &[], [1.0; 4], 1.0, true, HEAD_ON_LIGHT,
        )
        .unwrap();
        assert!(
            absent[0] > matte[0],
            "no roughness map must NOT mean a perfect diffuser: absent {absent:?} vs \
             explicitly-fully-rough {matte:?}",
        );
    }

    #[test]
    fn a_polished_surface_reflects_more_of_the_sky_than_a_rough_one() {
        // What makes polished stone read as polished. A punctual lamp can only
        // ever put a small bright spot on a mirror; what a real floor shows is
        // the room and sky around it. Marble020's roughness map averages 0.048
        // -- near mirror -- and with no environment term it looked like matte
        // stone however the highlight was tuned, because the thing it should
        // have been reflecting was never sampled.
        //
        // Unlit on purpose: this is the environment term, and a lamp in frame
        // would swamp it.
        // Grazing, for the same reason as the sealed test below: at normal
        // incidence Fresnel is 0.04 and this term is invisible.
        let Some(smooth) = render_brush_material(
            0, &palette(), &[Some(tilted_normal(4))], &[Some(flat([12, 12, 12], 4))], &[],
            [1.0; 4], 1.0, false, HEAD_ON_LIGHT,
        ) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let rough = render_brush_material(
            0, &palette(), &[Some(tilted_normal(4))], &[Some(flat([255, 255, 255], 4))], &[],
            [1.0; 4], 1.0, false, HEAD_ON_LIGHT,
        )
        .unwrap();
        assert!(
            smooth[0] > rough[0],
            "a near-mirror must reflect more environment than a diffuser: \
             {smooth:?} vs {rough:?}",
        );
    }

    #[test]
    fn a_sealed_room_does_not_reflect_a_sky_it_cannot_see() {
        // The occlusion the reflection has to respect. Without it every
        // polished surface indoors glows with outdoor light -- the same class
        // of bug as the unoccluded ambient that made interiors look flat, and
        // it would undo that fix for exactly the shiniest surfaces.
        //
        // Compared SEALED-SMOOTH against SEALED-ROUGH, not sealed against open.
        // Sealed-vs-open is dominated by the ambient term, which is occluded
        // either way, so that comparison passes whether or not the REFLECTION
        // respects occlusion and proves nothing about it. With the sky fully
        // hidden, polish must make no difference at all.
        // A HARD-TILTED normal, so the surface is seen at a grazing angle.
        // Fresnel at normal incidence is 0.04 -- the reflection is about 2% of
        // the frame there and disappears below a byte, so a head-on test cannot
        // see this term at all whether it is occluded or not. Grazing is where
        // Fresnel approaches 1, which is both where the effect matters visually
        // and the only place a test can observe it.
        let sealed = |rough: u8| {
            render_brush_baked(
                0, &palette(), &[Some(tilted_normal(4))],
                &[Some(flat([rough, rough, rough], 4))], &[],
                [1.0; 4], 1.0, false, HEAD_ON_LIGHT, [0, 0, 0, 0],
            )
        };
        let Some(smooth) = sealed(12) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let rough = sealed(255).unwrap();
        assert_eq!(
            smooth, rough,
            "with no sky visible, a polished surface reflected something a matte \
             one did not: {smooth:?} vs {rough:?}",
        );
    }

    #[test]
    fn a_polished_surface_in_a_sealed_room_reflects_the_room_itself() {
        // The complement of `a_sealed_room_does_not_reflect_a_sky_it_cannot_see`,
        // and the half that was missing.
        //
        // That test pins what a reflection must NOT do: invent outdoor light in
        // a room with no sky. Taken alone it is satisfied by a reflection term
        // that is dead indoors -- which is what shipped, and it made a marble
        // floor in a lit hall read as matte stone. A shiny floor in a dark room
        // does show something: it shows the room.
        //
        // Sky visibility is still ZERO here. The only thing available to
        // reflect is the baked bounce in RGB, so anything the polished surface
        // shows over the matte one came from the room's own light.
        //
        // Grazing angle for the same reason as the sealed test: at normal
        // incidence Fresnel is 0.04 and the whole term lands below a byte.
        let room_lit = |rough: u8| {
            render_brush_baked(
                0, &palette(), &[Some(tilted_normal(4))],
                &[Some(flat([rough, rough, rough], 4))], &[],
                [1.0; 4], 1.0, false, HEAD_ON_LIGHT,
                // Bright bounce, NO sky. The alpha is what the sealed test sets
                // to zero, and it stays zero.
                [180, 180, 180, 0],
            )
        };
        let Some(smooth) = room_lit(12) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let rough = room_lit(255).unwrap();

        // MEASURED IN GREEN, not red, and that is the whole point.
        //
        // The albedo here is red; the room's bounce is white. A dielectric's
        // reflection is NOT tinted by the surface it bounces off, so the room
        // arrives in every channel while the surface's own diffuse arrives only
        // in red. Green is therefore the part of the pixel that can only have
        // come from the room.
        //
        // This test used to compare RED, which conflated the two -- and once
        // the shading became energy-conserving, a polished surface gives up
        // diffuse to gain specular, so its red went DOWN while it reflected
        // more. It was measuring the surface's colour and calling it the
        // reflection.
        let room = smooth[1] as i32 - rough[1] as i32;
        assert!(
            room > 2,
            "a polished surface in a lit but sealed room reflected no more of \
             that room than a matte one: {smooth:?} vs {rough:?}",
        );
        // And it must not have got there by simply being brighter overall in a
        // way a matte surface would match.
        let total = |p: [u8; 4]| p[0] as i32 + p[1] as i32 + p[2] as i32;
        assert!(
            total(smooth) > total(rough),
            "the polished surface is not showing more light in total: \
             {smooth:?} vs {rough:?}",
        );
    }

    #[test]
    fn bounced_light_falls_on_the_side_of_a_surface_it_came_from() {
        // The point of the direction map. A flat irradiance lights a surface
        // facing the lit wall and one facing away by exactly the same amount,
        // which is why bounce made rooms brighter without making them read as
        // lit: normal maps went dead wherever no lamp reached and corners lost
        // their shape.
        //
        // Same bounce energy in both renders. The only difference is which way
        // the light is coming from, so anything that separates them is the
        // direction doing work.
        //
        // +Y and -Y, because the test surface faces +Z with a tilted normal:
        // light arriving from above and from below must not shade alike.
        let lit_from = |dir: [u8; 4]| {
            render_brush_baked_dir(
                // Tilted UP, not sideways: see `tilted_normal_up`. With the
                // sideways tilt this test used, the surface faced +Y and -Y
                // identically and the only thing separating the two renders was
                // a sub-byte glossy term -- so it was passing on a rounding
                // difference, and any change to the shading could silently
                // collapse it. One did.
                0, &palette(), &[Some(tilted_normal_up(4))],
                &[Some(flat([200, 200, 200], 4))], &[],
                [1.0; 4], 1.0, false, HEAD_ON_LIGHT,
                // Sky visibility zero: a sealed room, so the only light in the
                // frame is the bounce being steered.
                [170, 170, 170, 0],
                dir,
            )
        };
        // Encoded as v * 0.5 + 0.5, and fully directional in the alpha.
        let from_above = lit_from([128, 255, 128, 255]);
        let Some(above) = from_above else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let below = lit_from([128, 0, 128, 255]).unwrap();
        // A MARGIN, not mere inequality. `assert_ne!` on bytes passes on a
        // one-unit rounding difference, which is indistinguishable from noise
        // and is what this test was actually resting on.
        let gap = (above[0] as i32 - below[0] as i32).abs();
        assert!(
            gap > 4,
            "bounce arriving from above ({above:?}) and from below ({below:?}) \
             differ by only {gap}; the direction is barely reaching the surface",
        );
        assert!(
            above[0] > below[0],
            "a surface tilted UP should catch more of a bounce arriving from \
             above than from below: {above:?} vs {below:?}",
        );
    }

    #[test]
    fn a_bounce_with_no_direction_shades_exactly_as_it_always_did() {
        // The compatibility guarantee. Directionality zero is the neutral
        // value every unbaked and every pre-existing lightmap carries, and it
        // has to reproduce the flat behaviour EXACTLY -- not approximately, or
        // every absolute-value test in this file is measuring something new.
        let baked = [170u8, 170, 170, 0];
        let flat_render = render_brush_baked(
            0, &palette(), &[Some(tilted_normal(4))],
            &[Some(flat([200, 200, 200], 4))], &[],
            [1.0; 4], 1.0, false, HEAD_ON_LIGHT, baked,
        );
        let Some(flat_px) = flat_render else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        // A direction that is present but carries zero weight must change nothing.
        let weightless = render_brush_baked_dir(
            0, &palette(), &[Some(tilted_normal(4))],
            &[Some(flat([200, 200, 200], 4))], &[],
            [1.0; 4], 1.0, false, HEAD_ON_LIGHT, baked,
            [128, 255, 128, 0],
        ).unwrap();
        assert_eq!(
            flat_px, weightless,
            "a direction with zero directionality changed the shading",
        );
    }

    #[test]
    fn an_unlit_sealed_room_still_reflects_nothing() {
        // The guard on the test above: it must be the ROOM's light being
        // reflected and not a constant that appeared in the term. With the
        // bounce at zero and the sky at zero there is nothing to see, and
        // polish must once again make no difference.
        let dark = |rough: u8| {
            render_brush_baked(
                0, &palette(), &[Some(tilted_normal(4))],
                &[Some(flat([rough, rough, rough], 4))], &[],
                [1.0; 4], 1.0, false, HEAD_ON_LIGHT, [0, 0, 0, 0],
            )
        };
        let Some(smooth) = dark(12) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        assert_eq!(smooth, dark(255).unwrap());
    }

    #[test]
    fn a_highlight_is_not_tinted_by_the_surface_colour() {
        // A dielectric's specular is light that bounced STRAIGHT OFF the
        // surface without entering it, so it comes back the colour of the lamp,
        // not the colour of the material. A red marble floor has a white
        // highlight.
        //
        // The shader used to return one combined value that the caller
        // multiplied by albedo, which tinted the highlight too. On this
        // project's marble -- albedo 0.373 -- that made every highlight nearly
        // three times dimmer than it should be, and the surface read as having
        // no shine at all, which is exactly how it was reported from the
        // headset.
        //
        // Material 0 in `palette()` is saturated red with almost no green or
        // blue, so any green or blue in the result can ONLY have come from an
        // untinted highlight.
        let shine = |rough: u8| {
            render_brush_material(
                0, &palette(), &[None],
                &[Some(flat([rough, rough, rough], 4))], &[],
                [1.0; 4], 1.0, true, HEAD_ON_LIGHT,
            )
        };
        // Near-mirror, so there is a highlight to find at all.
        let Some(px) = shine(12) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        // The SAME surface, fully rough, so it has no highlight. Whatever green
        // the polished one has that this one does not can only be specular.
        //
        // Measured against a matte control rather than an absolute threshold:
        // the previous form asserted `px[1] > 8` and the value was 8, so it
        // passed or failed on a single byte of a term the test does not
        // actually care about. Any change to ambient anywhere tipped it.
        let matte = shine(255).unwrap();
        assert!(
            px[1] as i32 > matte[1] as i32 && px[2] as i32 > matte[2] as i32,
            "a polished red surface ({px:?}) carries no more green or blue than a \
             matte one ({matte:?}), so it has no untinted highlight at all",
        );
        // And the highlight is NEUTRAL: it adds the lamp's colour, not the
        // surface's. Green and blue must move together.
        let dg = px[1] as i32 - matte[1] as i32;
        let db = px[2] as i32 - matte[2] as i32;
        assert!(
            (dg - db).abs() <= 2,
            "the highlight added {dg} green and {db} blue -- it is carrying the \
             albedo's hue, so specular is still being multiplied by the diffuse \
             colour ({px:?} against {matte:?})",
        );
    }

    #[test]
    fn a_matte_surface_of_the_same_colour_has_no_such_highlight() {
        // The guard on the test above: green and blue must come from the
        // SPECULAR lobe, not from something that leaks on every surface. Fully
        // rough kills the lobe, so the same red material must come back red.
        let Some(matte) = render_brush_material(
            0, &palette(), &[None],
            &[Some(flat([255, 255, 255], 4))], &[],
            [1.0; 4], 1.0, true, HEAD_ON_LIGHT,
        ) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let shiny = render_brush_material(
            0, &palette(), &[None], &[Some(flat([12, 12, 12], 4))], &[],
            [1.0; 4], 1.0, true, HEAD_ON_LIGHT,
        ).unwrap();
        assert!(
            shiny[1] > matte[1],
            "polish added no untinted light: matte {matte:?} vs shiny {shiny:?}",
        );
    }

    #[test]
    fn a_near_mirror_still_produces_a_visible_highlight() {
        // Marble020's roughness map averages 0.048. Before the lamp was given
        // an area, that put the exponent on its clamp and the highlight was
        // narrower than a pixel -- computed correctly, and invisible.
        //
        // Compared against a FULLY rough surface rather than an absolute value,
        // so this measures the lobe rather than the brightness of the test rig.
        let Some(mirror) = render_brush_material(
            0, &palette(), &[None], &[Some(flat([12, 12, 12], 4))], &[], [1.0; 4], 1.0, true,
            HEAD_ON_LIGHT,
        ) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let rough = render_brush_material(
            0, &palette(), &[None], &[Some(flat([255, 255, 255], 4))], &[], [1.0; 4], 1.0, true,
            HEAD_ON_LIGHT,
        )
        .unwrap();
        assert!(
            mirror[0] > rough[0],
            "a near-mirror must be brighter at the mirror angle than a diffuser: \
             {mirror:?} vs {rough:?}",
        );
    }

    #[test]
    fn a_normal_map_changes_the_shading() {
        // Lit from one side, a surface tilted toward the light must come out
        // brighter than a flat one. Without this the normal map could be bound,
        // sampled, and multiplied by a tangent frame that discards it, and
        // every other test here would still pass.
        let Some(flat_px) = render_brush(
            0,
            &palette(),
            &[Some(flat([128, 128, 255], 4))],
            [1.0; 4],
            1.0,
            true,
        ) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let tilted =
            render_brush(0, &palette(), &[Some(tilted_normal(4))], [1.0; 4], 1.0, true).unwrap();
        assert_ne!(
            flat_px, tilted,
            "the tangent frame is discarding the normal map: {flat_px:?} vs {tilted:?}"
        );
        assert!(
            tilted[0] > flat_px[0],
            "tilted toward the light should be brighter: {flat_px:?} vs {tilted:?}"
        );
    }

    #[test]
    fn a_material_beyond_the_array_does_not_read_out_of_bounds() {
        // A level that outgrew the array must draw something rather than
        // sampling whatever is past the end. The loader clamps; this pins that
        // the last layer is a real, bound layer rather than undefined.
        let Some(px) = render_brush(
            (MAX_BRUSH_MATERIALS - 1) as u32,
            &palette(),
            &[],
            [1.0; 4],
            1.0,
            false,
        ) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        assert!(px[3] > 0, "the last layer must be bound and drawable: {px:?}");
    }

    /// Four columns: black, red, black, blue. Needs spatial variation, because
    /// a one-colour map samples identically whether the address mode repeats or
    /// clamps -- which is how a vacuous version of the test below passed.
    fn striped() -> TerrainImage {
        let cols = [[0u8, 0, 0], [255, 0, 0], [0, 0, 0], [0, 0, 255]];
        let mut rgba = Vec::with_capacity(4 * 4 * 4);
        for _row in 0..4 {
            for c in cols {
                rgba.extend_from_slice(&[c[0], c[1], c[2], 255]);
            }
        }
        TerrainImage { width: 4, height: 4, rgba }
    }

    #[test]
    fn uv_beyond_one_repeats_rather_than_smearing() {
        // The uv is in TILES: a 6m wall of a material tiling every metre spans
        // uv 0..6. Clamping would stretch the last column across five of them,
        // which looks like a broken texture rather than a wrong address mode.
        //
        // Asserted as a PROPERTY rather than by predicting where a pixel lands.
        // Working out the exact uv of the centre pixel means re-deriving the
        // rasteriser's sample position, and a first attempt at that produced a
        // test that failed against a correct sampler. What is true without any
        // of that arithmetic: past uv 1 a repeating sampler keeps moving through
        // the columns, so different scales give different colours, while a
        // clamping one is pinned to the edge column and gives one colour for
        // every scale.
        let scales = [4.0f32, 5.0, 6.0, 7.0];
        let mut seen = std::collections::HashSet::new();
        for s in scales {
            let Some(px) = render_brush(0, &[Some(striped())], &[], [1.0; 4], s, false) else {
                eprintln!("skipping: no GPU adapter available");
                return;
            };
            seen.insert([px[0], px[1], px[2]]);
        }
        assert!(
            seen.len() > 1,
            "every scale past one tile sampled the same colour, which is what \
             clamping does: {seen:?}"
        );
    }

    /// The probe, all the way through to a pixel.
    ///
    /// Everything upstream of this is covered by pure tests -- box projection,
    /// face orientation, the capture. None of that proves the cube is bound,
    /// that the shader reads the right binding, or that what it reads reaches
    /// the surface. Building the pipeline proves only that the WGSL parses.
    mod probe_render {
        use super::*;
        // The probe, all the way through to a pixel.
        //
        // Everything upstream of this is covered by pure tests -- box projection,
        // face orientation, the capture. None of that proves the cube is bound,
        // that the shader reads the right binding, or that what it reads reaches
        // the surface. Building the pipeline proves only that the WGSL parses.
        use super::tests::*;
        use super::*;

        /// A sealed room: no sky reaches the surface, so anything it reflects can
        /// only have come from the probe.
        const SEALED: [u8; 4] = [0, 0, 0, 0];

        fn polished_wall(probe: Option<([u8; 4], glam::Vec3, glam::Vec3)>) -> Option<[u8; 4]> {
            render_brush_probed(
                0,
                &palette(),
                &[Some(tilted_normal(4))],
                // Smooth, so Fresnel gives the environment term something to work
                // with. A fully rough surface reflects almost nothing by design and
                // would make this test unable to fail.
                &[Some(flat([12, 12, 12], 4))],
                &[],
                [1.0; 4],
                1.0,
                false,
                HEAD_ON_LIGHT,
                SEALED,
                crate::renderer::mesh::NEUTRAL_BOUNCE_DIRECTION,
                probe,
            )
        }

        /// The whole point: a probe puts the room onto a polished surface that has
        /// no sky and no bounce to show.
        #[test]
        fn a_probe_lights_a_polished_surface_that_has_nothing_else_to_reflect() {
            let big = (glam::Vec3::splat(-50.0), glam::Vec3::splat(50.0));
            let Some(without) = polished_wall(None) else {
                eprintln!("skipping: no GPU adapter available");
                return;
            };
            // A strongly GREEN probe, because the surface's own albedo is red: the
            // channel the probe arrives in is one the surface cannot produce.
            let with = polished_wall(Some(([0, 255, 0, 255], big.0, big.1))).unwrap();
            assert!(
                with[1] as i32 - without[1] as i32 > 4,
                "a probe covering the surface changed it from {without:?} to {with:?}; \
                 the cube is not reaching the shader",
            );
        }

        /// A probe that does not contain the surface must not light it.
        ///
        /// This is what stops one room's reflection leaking into another, and it is
        /// the half that a "does the cube reach the shader" test cannot check --
        /// binding a cube and always sampling it would pass that one.
        #[test]
        fn a_probe_elsewhere_in_the_level_does_not_reach_this_surface() {
            let far = (glam::Vec3::new(500.0, 500.0, 500.0), glam::Vec3::new(600.0, 600.0, 600.0));
            let Some(without) = polished_wall(None) else {
                eprintln!("skipping: no GPU adapter available");
                return;
            };
            let with = polished_wall(Some(([0, 255, 0, 255], far.0, far.1))).unwrap();
            assert_eq!(
                without, with,
                "a probe whose box is 500m away still lit this surface: {without:?} \
                 vs {with:?}",
            );
        }

        /// A rough surface must reflect the probe far less than a polished one.
        ///
        /// Without this, "the probe reaches the pixel" would be satisfied by a
        /// probe added flat to every surface regardless of material -- which would
        /// make every wall in the level glow with its room's colour.
        #[test]
        fn roughness_still_governs_how_much_of_the_probe_shows() {
            let big = (glam::Vec3::splat(-50.0), glam::Vec3::splat(50.0));
            let probe = Some(([0u8, 255, 0, 255], big.0, big.1));
            let rough = render_brush_probed(
                0, &palette(), &[Some(tilted_normal(4))],
                &[Some(flat([255, 255, 255], 4))], &[],
                [1.0; 4], 1.0, false, HEAD_ON_LIGHT, SEALED, crate::renderer::mesh::NEUTRAL_BOUNCE_DIRECTION, probe,
            );
            let Some(rough) = rough else {
                eprintln!("skipping: no GPU adapter available");
                return;
            };
            let smooth = polished_wall(probe).unwrap();
            assert!(
                smooth[1] > rough[1],
                "a polished surface ({smooth:?}) reflected no more of the probe than \
                 a matte one ({rough:?})",
            );
        }

        /// A MATTE surface must take nothing at all from the probe.
        ///
        /// The test above only asks that it take *less* than a polished one,
        /// which a probe scaled by Fresnel satisfies while still painting the
        /// room's fixtures onto every rough wall in the level. That is the
        /// artefact: a lamp the probe photographed from its capture point,
        /// arriving crisply on brick that cannot see the lamp at all -- the
        /// "bright rectangles that come and go" as the resident probe changes.
        ///
        /// A probe carries no visibility information, so at the roughness where
        /// the specular lobe is the whole hemisphere its answer is strictly
        /// worse than the lightmap's, which was measured at this very texel.
        /// The blend hands the rough end back to the bake, and this pins that:
        /// the probe must make NO difference here.
        ///
        /// Bright bounce and no sky, so the surface is genuinely lit and the
        /// comparison is between two lit pixels rather than two black ones.
        #[test]
        fn a_matte_surface_takes_nothing_from_the_probe() {
            let big = (glam::Vec3::splat(-50.0), glam::Vec3::splat(50.0));
            let lit_matte = |probe| {
                render_brush_probed(
                    0, &palette(), &[Some(tilted_normal(4))],
                    &[Some(flat([255, 255, 255], 4))], &[],
                    [1.0; 4], 1.0, false, HEAD_ON_LIGHT,
                    // Bright bounce, no sky: `env` is what this surface should
                    // be reflecting, and the probe is what it should not.
                    [180, 180, 180, 0],
                    crate::renderer::mesh::NEUTRAL_BOUNCE_DIRECTION, probe,
                )
            };
            let Some(without) = lit_matte(None) else {
                eprintln!("skipping: no GPU adapter available");
                return;
            };
            // Green again: the albedo is red, so green can only have come from
            // the probe.
            let with = lit_matte(Some(([0, 255, 0, 255], big.0, big.1))).unwrap();
            assert_eq!(
                without, with,
                "a matte surface changed from {without:?} to {with:?} when a probe \
                 covered it; the probe's fixtures are reaching rough geometry",
            );
        }
    }
}

#[cfg(test)]
mod ssr_pipeline_tests {
    use super::*;
    use crate::renderer::terrain_pipeline::tests::headless_gpu;
    use crate::renderer::lights::LightsUniform;
    use crate::renderer::uniforms::test_support::scene_uniforms;

    /// The reflective brush pipeline must actually BUILD.
    ///
    /// This is the only thing that checks it. WGSL is validated by naga at
    /// pipeline creation, not by `cargo build`, and the bind group count is
    /// checked at the same moment -- a brush spends three groups already, and
    /// the cuboid form of SSR binds two more, which would put this at five and
    /// over the limit this hardware guarantees. Nothing before this point would
    /// have said so.
    #[test]
    fn the_reflective_brush_pipeline_builds_within_the_bind_group_limit() {
        let Some((device, queue)) = headless_gpu() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let _ = &queue;
        let lights = LightsUniform::new(&device);
        let (_shadows, uniforms) = scene_uniforms(&device, &lights);
        // EVERY REFLECTIVE PIPELINE THE RENDERER BUILDS, at every sample count
        // the scene pass runs at.
        //
        // It used to build only `new_ssr`, and only at the sample count the
        // test chose. When the depth binding became single-sampled the test was
        // updated and the RENDERER was not: it went on passing `samples > 1`,
        // so all four of these failed to build on the headset, reflections
        // stopped appearing, and the false-colour view came back as the
        // ordinary picture. The parameter that allowed the mismatch is gone
        // now; this builds the same set the renderer does so that a layout and
        // a shader drifting apart fails here instead.
        for scene_samples in [1u32, 4] {
            let ssr = crate::renderer::ssr::SsrPipelines::new_with_depth_samples(
                &device,
                TextureFormat::Rgba8UnormSrgb,
                scene_samples,
            );
            let err_scope_1 = device.push_error_scope(wgpu::ErrorFilter::Validation);
            let _pipeline = BrushPipeline::new_ssr(
                &device,
                TextureFormat::Rgba8UnormSrgb,
                &uniforms.layout,
                ssr.scene_texture_layout(),
            );
            for view in [DebugView::Sources, DebugView::Ssr] {
                let _p = BrushPipeline::new_ssr_debug(
                    &device,
                    TextureFormat::Rgba8UnormSrgb,
                    &uniforms.layout,
                    ssr.scene_texture_layout(),
                    view,
                );
            }
            // THE TRACE PIPELINE, against the reflection buffer's own format.
            //
            // This is the shape of failure that has cost this project the most:
            // a rejected pipeline draws nothing, logs nothing on the device and
            // makes the frame FASTER, which reads as a win. The trace shader
            // has a different RETURN TYPE from every other variant -- `vec4`
            // radiance-and-confidence rather than a blended `vec3` -- so if the
            // substitution that produces it is wrong, naga rejects it here
            // rather than on the headset.
            let _trace = BrushPipeline::new_ssr_trace(
                &device,
                crate::renderer::ssr::REFLECTION_FORMAT,
                &uniforms.layout,
                ssr.scene_texture_layout(),
            );
            // And the pass that reads what it wrote. Same layout, different
            // texture in binding 1 -- see `wgsl_ssr_composite_block`.
            let _composite = BrushPipeline::new_ssr_composite(
                &device,
                TextureFormat::Rgba8UnormSrgb,
                &uniforms.layout,
                ssr.scene_texture_layout(),
            );
            let _solid = crate::renderer::pipeline::SolidPipeline::new_ssr(
                &device,
                TextureFormat::Rgba8UnormSrgb,
                &uniforms.layout,
                ssr.camera_layout(),
                ssr.scene_texture_layout(),
            );
            let err = pollster::block_on(err_scope_1.pop());
            assert!(
                err.is_none(),
                "reflective brush pipeline failed to build \
                 (scene_samples={scene_samples}): {err:?}",
            );
        }
    }

    /// The runtime debug views: the shipped shader is untouched, and each
    /// variant carries the diagnostic it is named for.
    #[test]
    fn a_debug_view_changes_only_the_shader_it_is_built_for() {
        assert!(!BRUSH_SOURCE_DEBUG && !crate::renderer::ssr::SSR_DEBUG, "a build switch is on");
        for ssr in [false, true] {
            assert_eq!(brush_shader_variant(ssr, false, false), brush_shader_with(ssr));
        }
        let sources = brush_shader_variant(false, true, false);
        // WHICH sources picture depends on `SOURCES_VIEW`; that it paints the
        // one selected, and that the shaders above are untouched by it, is
        // the invariant.
        let marker = match SOURCES_VIEW {
            SourcesView::Ratios => "dbg_total",
            SourcesView::ProbeFactors => "dbg_probe_factors",
            SourcesView::Absolute => "abs_lum",
        };
        assert!(
            sources.contains(marker),
            "the sources view does not paint the {SOURCES_VIEW:?} picture it is set to",
        );
        let ssr_debug = brush_shader_variant(true, false, true);
        assert!(ssr_debug.contains("MAGENTA: faces the viewer"), "the SSR view does not paint the SSR paths");
        assert!(!brush_shader_with(true).contains("MAGENTA"), "the shipped shader paints SSR paths");
        assert_eq!(DebugView::Off.next().next().next(), DebugView::Off, "the cycle does not return to Off");
    }

    /// The two-eye probe pass and the reader that samples its layers pass
    /// naga once made multiview -- the only check a development machine can
    /// make, since none can build a multiview pipeline. A failure on the
    /// headset only switches half-resolution reflections off in stereo; see
    /// `StereoProbePass` in the XR renderer.
    #[test]
    fn the_stereo_probe_pass_shaders_validate_as_multiview() {
        for (probe, sources) in [(BrushProbe::Pass, false), (BrushProbe::Read, BRUSH_SOURCE_DEBUG)] {
            let src = brush_shader_probe(false, sources, crate::renderer::ssr::SSR_DEBUG, SsrPath::Inline, probe);
            assert_eq!(crate::renderer::multiview::multiview_validation_error_of(&src, false), None, "{probe:?} before the transform");
            assert_eq!(crate::renderer::multiview::multiview_validation_error(&src), None, "{probe:?} as multiview");
            if probe == BrushProbe::Read {
                assert!(crate::renderer::multiview::as_multiview(&src).contains("view_slot = i32(view_index_in);"));
            }
        }
    }

    /// THE UPSAMPLE IS THE DEPTH-AWARE BILINEAR FILTER, on a real device, at
    /// every pixel: one filtered read inside a surface, texel by texel along
    /// an edge -- across a vertical AND a horizontal step in depth, so a
    /// gather read in the wrong order or a swapped axis weights the wrong
    /// texels and fails. Checked against the four-texel sum on the CPU.
    #[test]
    fn the_probe_upsample_is_the_depth_aware_bilinear_filter() {
        let Some((device, queue)) = headless_gpu() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        const W: u32 = 64;
        const H: u32 = 32;
        const TOLERANCE: f32 = 0.0015;
        // The half-resolution pass's picture, as a function of its texel: a
        // colour ramp, a coverage that varies, and two steps in depth -- one
        // down the middle, one across -- over a gentle slope inside each.
        const PATTERN: &str = r#"
fn pass_colour(t: vec2<f32>) -> vec4<f32> {
    let a = 0.25 + 0.5 * fract(t.x * 0.37 + t.y * 0.11);
    return vec4<f32>(vec3<f32>(t.x / 32.0, t.y / 16.0, 0.5) * a, a);
}
fn pass_depth(t: vec2<f32>) -> f32 {
    return 0.3 + 0.4 * step(16.0, t.x) + 0.15 * step(8.0, t.y) + 0.001 * t.y;
}
// A full-resolution pixel's own depth: its surface by its OWN position, the
// slope interpolated between the half-resolution rows.
fn pixel_depth(p: vec2<f32>) -> f32 {
    return 0.3 + 0.4 * step(32.0, p.x) + 0.15 * step(16.0, p.y) + 0.001 * (p.y * 0.5 - 0.5);
}
@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let xy = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u)) * 2.0 - 1.0;
    return vec4<f32>(xy, 0.0, 1.0);
}
struct PassOut {
    @location(0) colour: vec4<f32>,
    @builtin(frag_depth) depth: f32,
}
@fragment
fn fill(@builtin(position) pos: vec4<f32>) -> PassOut {
    let t = floor(pos.xy);
    return PassOut(pass_colour(t), pass_depth(t));
}
"#;
        let layout = probe_pass::bind_group_layout(&device);
        let target = probe_pass::Target::new(&device, &layout, W, H, 1);
        let fill_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("upsample_fill"),
            source: wgpu::ShaderSource::Wgsl(PATTERN.into()),
        });
        let fill = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("upsample_fill"),
            layout: None,
            vertex: wgpu::VertexState { module: &fill_module, entry_point: Some("vs"), buffers: &[], compilation_options: Default::default() },
            fragment: Some(wgpu::FragmentState {
                module: &fill_module,
                entry_point: Some("fill"),
                targets: &[Some(probe_pass::FORMAT.into())],
                compilation_options: Default::default(),
            }),
            primitive: Default::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Always),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });
        let read_src = format!(
            "{PATTERN}var<private> view_slot: i32 = 0;\n{}\n@fragment\nfn read(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {{\n    return probe_pass_upsample(pos.xy, pixel_depth(pos.xy), {TOLERANCE:?});\n}}\n",
            probe_pass::READER_WGSL
        );
        let read_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("upsample_read"),
            source: wgpu::ShaderSource::Wgsl(read_src.into()),
        });
        let empty = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries: &[] });
        let empty_bg = device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &empty, entries: &[] });
        let read_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("upsample_read"),
            bind_group_layouts: &[Some(&empty), Some(&empty), Some(&empty), Some(&layout)],
            immediate_size: 0,
        });
        let read = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("upsample_read"),
            layout: Some(&read_layout),
            vertex: wgpu::VertexState { module: &read_module, entry_point: Some("vs"), buffers: &[], compilation_options: Default::default() },
            fragment: Some(wgpu::FragmentState {
                module: &read_module,
                entry_point: Some("read"),
                targets: &[Some(TextureFormat::Rgba32Float.into())],
                compilation_options: Default::default(),
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });
        let out = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("upsample_out"),
            size: wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let out_view = out.create_view(&Default::default());
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("upsample_fill"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target.color_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &target.depth_view,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            pass.set_pipeline(&fill);
            pass.draw(0..3, 0..1);
        }
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("upsample_read"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &out_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                })],
                ..Default::default()
            });
            pass.set_pipeline(&read);
            for g in 0..3 {
                pass.set_bind_group(g, &empty_bg, &[]);
            }
            pass.set_bind_group(3, &target.bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        let row = W * 16;
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("upsample_readback"),
            size: u64::from(row * H),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo { texture: &out, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(H) },
            },
            wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
        );
        queue.submit(Some(encoder.finish()));
        readback.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        let data = readback.slice(..).get_mapped_range().unwrap();
        let got: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&data).to_vec();
        drop(data);

        // The four-texel filter, on the CPU, from the same pattern.
        let step = |edge: f32, x: f32| if x >= edge { 1.0 } else { 0.0 };
        let colour = |t: [f32; 2]| {
            let a = 0.25 + 0.5 * (t[0] * 0.37 + t[1] * 0.11).fract();
            [t[0] / 32.0 * a, t[1] / 16.0 * a, 0.5 * a, a]
        };
        let texel_depth = |t: [f32; 2]| 0.3 + 0.4 * step(16.0, t[0]) + 0.15 * step(8.0, t[1]) + 0.001 * t[1];
        let (hw, hh) = (W / 2, H / 2);
        let (mut worst, mut edges) = (0.0f32, 0usize);
        for y in 0..H {
            for x in 0..W {
                let p = [x as f32 + 0.5, y as f32 + 0.5];
                let depth = 0.3 + 0.4 * step(32.0, p[0]) + 0.15 * step(16.0, p[1]) + 0.001 * (p[1] * 0.5 - 0.5);
                let h = [p[0] * 0.5 - 0.5, p[1] * 0.5 - 0.5];
                let base = [h[0].floor(), h[1].floor()];
                let f = [h[0] - base[0], h[1] - base[1]];
                let (mut sum, mut weight) = ([0.0f32; 4], 0.0f32);
                let (mut nearest, mut nearest_gap) = ([0.0f32; 4], f32::MAX);
                for (ox, oy) in [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)] {
                    let t = [(base[0] + ox).clamp(0.0, (hw - 1) as f32), (base[1] + oy).clamp(0.0, (hh - 1) as f32)];
                    let c = colour(t);
                    let gap = (texel_depth(t) - depth).abs();
                    let bw = (if ox == 1.0 { f[0] } else { 1.0 - f[0] }) * (if oy == 1.0 { f[1] } else { 1.0 - f[1] });
                    let w = if gap <= TOLERANCE { bw } else { 0.0 };
                    for k in 0..4 {
                        sum[k] += c[k] * w;
                    }
                    weight += w;
                    if gap < nearest_gap {
                        nearest_gap = gap;
                        nearest = c;
                    }
                }
                edges += usize::from(weight < 0.999);
                let pre = if weight > 1e-4 { sum.map(|v| v / weight) } else { nearest };
                let want = [pre[0] / pre[3].max(1e-4), pre[1] / pre[3].max(1e-4), pre[2] / pre[3].max(1e-4), pre[3]];
                let i = ((y * W + x) * 4) as usize;
                for k in 0..4 {
                    let d = (got[i + k] - want[k]).abs();
                    worst = worst.max(d);
                    assert!(d < 4e-3, "pixel ({x}, {y}) channel {k}: {} against {}", got[i + k], want[k]);
                }
            }
        }
        // Both paths were exercised: most pixels one surface, a band along
        // each step on the per-texel path.
        assert!(edges > 60 && edges < (W * H / 4) as usize, "{edges} edge pixels");
        eprintln!("upsample: worst difference {worst:.5} over {} pixels, {edges} on an edge", W * H);
    }

    /// Every measurement cut of the probe pass still finds its text in the
    /// generated shader -- a cut that no longer applies would report the
    /// uncut shader under its name -- and the result is valid WGSL. See
    /// `BrushPipeline::log_probe_pass_register_cuts`.
    #[test]
    fn every_scene_register_cut_applies_and_validates() {
        use wgpu::naga;
        let base = brush_shader_probe(false, BRUSH_SOURCE_DEBUG, crate::renderer::ssr::SSR_DEBUG, SsrPath::Inline, BrushProbe::Read);
        for (label, edits) in SCENE_REGISTER_CUTS {
            let mut src = base.clone();
            for (from, to) in edits.iter() {
                assert!(src.contains(from), "{label}: `{from}` is not in the scene shader");
                src = src.replacen(from, to, 1);
            }
            let module = naga::front::wgsl::parse_str(&src)
                .unwrap_or_else(|e| panic!("{label}: {}", e.emit_to_string(&src)));
            naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
                .validate(&module)
                .unwrap_or_else(|e| panic!("{label}: {e:?}"));
        }
    }

    #[test]
    fn every_probe_pass_register_cut_applies_and_validates() {
        use wgpu::naga;
        let base = brush_shader_probe(false, false, crate::renderer::ssr::SSR_DEBUG, SsrPath::Inline, BrushProbe::Pass);
        for (label, edits) in PROBE_PASS_REGISTER_CUTS {
            let mut src = base.clone();
            for (from, to) in edits.iter() {
                assert!(src.contains(from), "{label}: `{from}` is not in the probe pass shader");
                src = src.replacen(from, to, 1);
            }
            let module = naga::front::wgsl::parse_str(&src)
                .unwrap_or_else(|e| panic!("{label}: {}", e.emit_to_string(&src)));
            naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
                .validate(&module)
                .unwrap_or_else(|e| panic!("{label}: {e:?}"));
        }
    }

    /// THE HALF-RESOLUTION PROBE PASS AND THE BRUSH THAT READS IT build on a
    /// real device -- WGSL is only validated at pipeline creation -- and the
    /// reader is built without the trace. See `probe_pass`.
    #[test]
    fn the_probe_pass_pipelines_build_on_a_real_device() {
        let reader_src = brush_shader_probe(false, false, false, SsrPath::Inline, BrushProbe::Read);
        assert!(reader_src.contains("const PROBE_ENV_FROM_PASS: bool = true;"), "the reader still traces per pixel");
        assert!(brush_shader_variant(false, false, false).contains("const PROBE_ENV_FROM_PASS: bool = false;"));
        let Some((device, _queue)) = headless_gpu() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let lights = LightsUniform::new(&device);
        let (_shadows, uniforms) = scene_uniforms(&device, &lights);
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let _pass = BrushPipeline::new_probe_pass(&device, &uniforms.layout, crate::renderer::multiview::ViewMode::Mono);
        let layout = probe_pass::bind_group_layout(&device);
        let _reader = BrushPipeline::new_multisampled_probe_reader(
            &device, TextureFormat::Rgba8UnormSrgb, &uniforms.layout, 4, &layout, crate::renderer::multiview::ViewMode::Mono,
        );
        let target = probe_pass::Target::new(&device, &layout, 1445, 1546, 1);
        // Both eyes' target builds too; its pipelines need multiview, which
        // no development machine has -- see the naga check below.
        let _stereo = probe_pass::Target::new(&device, &layout, 1445, 1546, 2);
        assert_eq!((target.width, target.height), (723, 773));
        // The pass that ships, which defers its secondary lookups, and the
        // compute pass that makes them, writing into that target.
        let fixups = crate::renderer::probe_fixup::ProbeFixups::new(&device, &uniforms.layout, target.width * target.height);
        let _deferred = BrushPipeline::new_probe_pass_deferred(&device, &uniforms.layout, &fixups);
        let _fixup_target = fixups.target_bind_group(&device, &target);
        let err = pollster::block_on(scope.pop());
        assert!(err.is_none(), "the probe pass pipelines failed to build: {err:?}");
        let deferred_src = brush_shader_probe(false, false, false, SsrPath::Inline, BrushProbe::PassDeferred);
        assert!(deferred_src.contains("const PROBE_SECONDARY_DEFERRED: bool = true;"));
        assert!(deferred_src.contains("probe_fragment = in.clip;") && deferred_src.contains("atomicAdd(&probe_fixups.count"));
        assert!(brush_shader_probe(false, false, false, SsrPath::Inline, BrushProbe::Pass)
            .contains("const PROBE_SECONDARY_DEFERRED: bool = false;"));
    }

    /// The debug pipelines must build on a real device, or pressing the button
    /// on a headset is a crash rather than a diagnostic.
    #[test]
    fn the_debug_view_pipelines_build_on_a_real_device() {
        let Some((device, _queue)) = headless_gpu() else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        let lights = LightsUniform::new(&device);
        let (_shadows, uniforms) = scene_uniforms(&device, &lights);
        let err_scope_2 = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let _sources = BrushPipeline::new_multisampled_sources(
            &device, TextureFormat::Rgba8UnormSrgb, &uniforms.layout, 1,
        );
        let err = pollster::block_on(err_scope_2.pop());
        assert!(err.is_none(), "the sources scene pipeline failed to build: {err:?}");
        for view in [DebugView::Sources, DebugView::Ssr] {
            let ssr = crate::renderer::ssr::SsrPipelines::new_with_depth_samples(
                &device, TextureFormat::Rgba8UnormSrgb, 1,
            );
            let err_scope_3 = device.push_error_scope(wgpu::ErrorFilter::Validation);
            let _p = BrushPipeline::new_ssr_debug(
                &device, TextureFormat::Rgba8UnormSrgb, &uniforms.layout,
                ssr.scene_texture_layout(), view,
            );
            let err = pollster::block_on(err_scope_3.pop());
            assert!(err.is_none(), "the {view:?} reflective pipeline failed to build: {err:?}");
        }
    }

    /// SSR decides by the NEIGHBOURHOOD, not the texel. See the note on
    /// `ssr_rough` in the shader: per-texel decisions were the stone room's
    /// sparkle.
    #[test]
    fn screen_space_reflection_follows_the_neighbourhood_not_the_texel() {
        let src = brush_shader_with(true);
        assert!(
            src.contains("let ssr_rough = textureSampleLevel(mat_rough, mat_rough_samp, in.uv, i32(in.material), 4.0).r;"),
            "SSR reads roughness per texel again; stone will sparkle",
        );
        assert!(src.contains("let reflectivity = clamp((1.0 - ssr_rough) * ssr_fresnel"));
        assert!(
            src.contains("ssr_reflect(in.world_pos, ssr_n, ssr_fallback, reflectivity, ssr_rough)"),
            "the march is steered by the texel's normal again",
        );
    }

    /// The reflective pass must not light a surface the scene pass already lit.
    /// See `ssr_fallback`: doing it twice was most of the eye pass's cost.
    #[test]
    fn the_reflective_pass_does_not_light_the_surface_a_second_time() {
        const LIGHTS: &str = "let lit = shade_material_lamps(";
        const READ_BACK: &str =
            "let ssr_fallback = textureLoad(ssr_scene_color, vec2<i32>(in.clip.xy), 0).rgb;";
        for _ms in [false] {
            let shipped = brush_shader_variant(true, false, false);
            assert!(!shipped.contains(LIGHTS), "the reflective pass shades the surface again");
            assert!(shipped.contains(READ_BACK), "the reflective pass no longer reads the scene's colour");
            let ssr_view = brush_shader_variant(true, false, true);
            assert!(!ssr_view.contains(LIGHTS), "the SSR view shades the surface again");
        }
        // The scene pass is where the shading happens, and the sources view
        // needs it because its colours are recorded while shading runs.
        assert!(brush_shader_variant(false, false, false).contains(LIGHTS), "the scene pass stopped lighting");
        assert!(
            brush_shader_variant(true, true, false).contains(LIGHTS),
            "the reflective sources view has nothing to paint",
        );
    }

    /// Four groups is the whole budget, so the count is worth stating.
    #[test]
    fn the_reflective_brush_shader_binds_exactly_four_groups() {
        let src = brush_shader_with(true);
        for g in 0..4 {
            assert!(
                src.contains(&format!("@group({g})")),
                "group {g} is unused -- the layout and the shader disagree",
            );
        }
        assert!(
            !src.contains("@group(4)"),
            "a fifth bind group would exceed the limit this hardware guarantees",
        );
    }

    /// The plain variant must stay plain.
    #[test]
    fn the_ordinary_brush_shader_has_no_reflection_in_it() {
        let src = brush_shader();
        assert!(!src.contains("ssr_reflect"), "the non-reflective brush shader marches rays");
        assert!(!src.contains("@group(3)"), "the non-reflective brush shader binds a scene texture");
    }
}

#[cfg(test)]
mod reflectivity_tests {
    use super::*;
    use crate::renderer::terrain_pipeline::TerrainImage;

    fn rough_map(byte: u8) -> TerrainImage {
        TerrainImage { width: 4, height: 4, rgba: (0..16).flat_map(|_| [byte, byte, byte, 255]).collect() }
    }

    #[test]
    fn polished_marble_is_worth_reflecting() {
        // Marble020 measures 0.048 -- byte 12 -- and is unmistakably a mirror.
        assert!(roughness_is_reflective(Some(&rough_map(12))));
    }

    #[test]
    fn rock_and_gravel_are_not() {
        // Near 1.0. Turning the reflective pass on for these would pay a
        // full-resolution scene copy and a second draw over every face to show
        // nothing.
        assert!(!roughness_is_reflective(Some(&rough_map(230))));
        assert!(!roughness_is_reflective(Some(&rough_map(255))));
    }

    #[test]
    fn a_material_with_no_roughness_map_is_treated_as_fully_rough() {
        // The same default the shader uses for a missing map. An unauthored
        // material must never silently switch the pass on.
        assert!(!roughness_is_reflective(None));
    }

    #[test]
    fn an_empty_map_does_not_divide_by_zero() {
        let empty = TerrainImage { width: 0, height: 0, rgba: Vec::new() };
        assert!(!roughness_is_reflective(Some(&empty)));
    }

    #[test]
    fn the_mean_decides_not_the_shiniest_speck() {
        // A mostly-matte wall with a few polished flecks is not a mirror.
        // Taking the minimum would turn the pass on for a surface that shows
        // nothing, which is the expensive direction to be wrong in.
        let mut mostly_rough = rough_map(240);
        mostly_rough.rgba[0] = 0;
        mostly_rough.rgba[4] = 0;
        assert!(!roughness_is_reflective(Some(&mostly_rough)));
    }
}

#[cfg(test)]
mod bounce_softening_tests {
    use super::*;

    /// The shaping factor, as the shader computes it.
    fn shaped(directionality: f32, n_dot_dir: f32) -> f32 {
        let d = directionality.clamp(0.0, 0.5);
        ((1.0 - d) + d * (n_dot_dir + 1.0)).max(0.0)
    }

    #[test]
    fn a_confident_direction_cannot_swing_the_bounce_a_hundredfold() {
        // The measured cause of the reported squares. The baked directionality
        // averages 0.79 on the test room, which UNCAPPED spans 0.21..1.79 and
        // reaches 0.02..1.98 at the extremes -- while neighbouring texels were
        // measured pointing in exactly opposite directions. One texel boundary
        // could therefore cross almost the whole range, which reads as a hard
        // bright-and-dark grid the size of a lightmap texel.
        let lo = shaped(1.0, -1.0);
        let hi = shaped(1.0, 1.0);
        assert!(hi / lo.max(1e-6) <= 3.1, "shaping still swings {:.1}x", hi / lo);
    }

    #[test]
    fn it_is_still_directional() {
        // Capping the trust must not flatten the term into the constant it
        // replaced -- the whole point is that a surface facing the bounce is
        // lit more than one facing away.
        assert!(shaped(0.79, 1.0) > shaped(0.79, -1.0) * 2.0);
    }

    #[test]
    fn no_directionality_is_still_exactly_flat() {
        // The compatibility guarantee for every lightmap baked before the
        // direction map existed.
        for cos in [-1.0f32, -0.4, 0.0, 0.6, 1.0] {
            assert_eq!(shaped(0.0, cos), 1.0);
        }
    }

    #[test]
    fn the_reflection_is_not_a_mirror() {
        // Marble reads as polished chrome when a sharp screen-space reflection
        // is shown at full Fresnel strength, because nothing blurs it by the
        // surface's roughness. Reported from the headset as "too reflective".
        let src = brush_shader_with(true);
        assert!(
            src.contains("MAX_SSR_REFLECTIVITY"),
            "the reflection strength is uncapped",
        );
    }
}

#[cfg(test)]
mod mip_chain_tests {
    //! The mip chain the brush materials never had.
    use super::*;

    fn solid(w: u32, h: u32, px: [u8; 4]) -> TerrainImage {
        TerrainImage { width: w, height: h, rgba: px.iter().copied().cycle().take((w * h * 4) as usize).collect() }
    }

    #[test]
    fn a_chain_runs_all_the_way_down_to_one_texel() {
        let c = mip_chain(&solid(1024, 1024, [10, 20, 30, 255]), true);
        assert_eq!(c.len(), 11, "1024 needs 11 levels, got {}", c.len());
        assert_eq!((c[0].width, c[0].height), (1024, 1024));
        assert_eq!((c.last().unwrap().width, c.last().unwrap().height), (1, 1));
        for lv in &c {
            assert_eq!(lv.rgba.len(), (lv.width * lv.height * 4) as usize, "level is not its own size");
        }
    }

    /// THE sRGB trap. A flat colour must survive every level unchanged.
    ///
    /// Averaging sRGB bytes directly is the easy mistake and it does not show
    /// up here -- a flat image averages to itself either way. It shows up in
    /// `a_checkerboard_keeps_its_brightness` below. This one guards the other
    /// direction: that the decode/encode round trip does not drift.
    #[test]
    fn a_flat_colour_survives_every_level() {
        for srgb in [true, false] {
            let c = mip_chain(&solid(64, 64, [180, 90, 45, 255]), srgb);
            for (i, lv) in c.iter().enumerate() {
                let px = &lv.rgba[0..4];
                assert!(
                    px.iter().zip([180u8, 90, 45, 255]).all(|(a, b)| (*a as i32 - b as i32).abs() <= 1),
                    "srgb={srgb} level {i} drifted to {px:?}",
                );
            }
        }
    }

    /// Averaging sRGB bytes directly darkens every level.
    ///
    /// Black and white in equal measure is 50% of the light. In sRGB bytes
    /// that is 188, not 128 -- and 128 is 22% grey. Get this wrong and every
    /// wall visibly dims as it recedes.
    #[test]
    fn a_checkerboard_keeps_its_brightness() {
        let (w, h) = (64u32, 64u32);
        let mut rgba = vec![0u8; (w * h * 4) as usize];
        for y in 0..h {
            for x in 0..w {
                let v = if (x + y) % 2 == 0 { 255 } else { 0 };
                let i = ((y * w + x) * 4) as usize;
                rgba[i..i + 4].copy_from_slice(&[v, v, v, 255]);
            }
        }
        let img = TerrainImage { width: w, height: h, rgba };
        let level1 = &mip_chain(&img, true)[1];
        let got = level1.rgba[0];
        assert!(
            (got as i32 - 188).abs() <= 3,
            "half-lit sRGB averaged to {got}; 128 means it was averaged as raw bytes \
             and every mip is too dark",
        );
    }

    /// Alpha is linear even inside an sRGB texture.
    #[test]
    fn alpha_is_never_gamma_encoded() {
        let (w, h) = (2u32, 2u32);
        let rgba = vec![
            0, 0, 0, 255, 0, 0, 0, 0,
            0, 0, 0, 255, 0, 0, 0, 0,
        ];
        let img = TerrainImage { width: w, height: h, rgba };
        let a = mip_chain(&img, true)[1].rgba[3];
        assert!((a as i32 - 128).abs() <= 1, "alpha averaged to {a}, not 128");
    }
}

/// The shading-source diagnostic must not ship.
#[cfg(test)]
mod source_debug_tests {
    use super::*;

    /// A build with this on paints every brush surface in flat primaries. That
    /// is obvious on a headset and invisible in a diff, so it is asserted
    /// rather than remembered -- the same guard `SSR_DEBUG` carries.
    #[test]
    fn the_source_diagnostic_is_off() {
        assert!(
            !BRUSH_SOURCE_DEBUG,
            "BRUSH_SOURCE_DEBUG is on: every brush renders as red/green/blue by \
             which term lit it, not as the scene. Set it back to false.",
        );
    }

    /// And when it is on, it reaches the shader -- otherwise flipping the
    /// switch produces a normal-looking build and wastes a headset session.
    #[test]
    fn the_switch_reaches_the_shader() {
        let src = brush_shader_with(true);
        let painted = src.contains("let dbg_sum = dbg_direct + dbg_baked + dbg_probe;");
        assert_eq!(
            painted, BRUSH_SOURCE_DEBUG,
            "BRUSH_SOURCE_DEBUG is {BRUSH_SOURCE_DEBUG} but the shader {} the \
             diagnostic",
            if painted { "has" } else { "lacks" },
        );
    }

    /// The three terms are actually recorded where they are computed, or the
    /// diagnostic paints a uniform colour and says nothing.
    #[test]
    fn every_source_is_recorded_at_its_own_term() {
        let lights = super::super::lights::wgsl_lights_block(0, 1);
        assert!(
            // Occlusion and the roughness blend included deliberately. The
            // diagnostic is answering "which term lit this pixel", so it has to
            // record what the probe CONTRIBUTED, not what it sampled -- a probe
            // whose contribution is entirely blended away should read black
            // here, and pinning the raw sample would have painted it blue.
            lights.contains(
                "dbg_probe = probe.rgb * clamp(probe.a, 0.0, 1.0) * spec_occ * probe_scale \
                 * (1.0 - lobe_is_hemispherical) * fresnel;",
            ),
            "the probe term is not recorded as the picture weights it",
        );
        // WEIGHTED LIKE THE RETURN LINE: diffuse terms by albedo and
        // (1 - Fresnel), the probe by Fresnel. Unweighted, the sources view
        // painted walls blue that the probe barely lit (headset, 2026-09-17).
        assert!(
            lights.contains("dbg_baked = bounce * albedo * (1.0 - fresnel);"),
            "the baked term is not recorded as the picture weights it",
        );
        assert!(
            lights.contains(
                "dbg_direct = dbg_direct + (c.diffuse * albedo * (1.0 - fresnel) + c.specular) * shadow;"
            ),
            "the direct term is not recorded, or is recorded before its shadow test",
        );
    }
}

/// The crack seal draws a brush's BACK faces and nothing it should not.
#[cfg(test)]
mod crack_seal_tests {
    use super::*;
    use crate::renderer::lights::LightsUniform;
    use crate::renderer::terrain_pipeline::tests::headless_gpu;
    use crate::renderer::uniforms::test_support::{scene_uniforms, TEST_EYE};
    use crate::renderer::uniforms::ShadowUpload;
    use wgpu::util::DeviceExt;

    /// One full-screen triangle through the seal pipeline, on white, with the
    /// depth buffer cleared to `depth_clear`; returns the centre pixel.
    fn render_seal(facing_away: bool, z: f32, depth_clear: f32) -> Option<[u8; 4]> {
        let (device, queue) = headless_gpu()?;
        let format = TextureFormat::Rgba8Unorm;
        let lights = LightsUniform::new(&device);
        lights.upload(&queue, &[]);
        let (_shadows, uniforms) = scene_uniforms(&device, &lights);
        // Identity: vertex positions ARE clip coordinates.
        uniforms.upload(&queue, glam::Mat4::IDENTITY, TEST_EYE, &ShadowUpload::disabled());
        let pipeline = BrushSealPipeline::new(&device, format, &uniforms.layout, 1);

        let v = |x: f32, y: f32| BrushVertex {
            position: [x, y, z],
            normal: [0.0, 0.0, 1.0],
            tangent: [1.0, 0.0, 0.0, 1.0],
            uv: [0.0, 0.0],
            material: 0,
            tint: [1.0; 4],
            uv2: [0.0, 0.0],
            face_centre: [1.0 / 3.0, 1.0 / 3.0, z],
            uv2_rect: [0.0, 0.0, 1.0, 1.0],
            // NO CLAMP. These fixtures are about the material and the
            // pipeline, not about edge extrapolation, and infinite extent is
            // exactly the behaviour they measured before the clamp existed --
            // so every expectation in them still means what it meant.
            face_half_extent: [f32::INFINITY; 2],
        };
        // Counter-clockwise on screen faces the viewer, as the brush pipeline
        // expects; reversing the order turns the same triangle away.
        let mut verts = vec![v(-1.0, -1.0), v(3.0, -1.0), v(-1.0, 3.0)];
        if facing_away {
            verts.swap(1, 2);
        }
        let vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("seal_test_vb"),
            contents: bytemuck::cast_slice(&verts),
            usage: BufferUsages::VERTEX,
        });
        let ib = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("seal_test_ib"),
            contents: bytemuck::cast_slice(&[0u32, 1, 2]),
            usage: BufferUsages::INDEX,
        });

        const SIZE: u32 = 8;
        let tex = |label, format, usage| {
            device.create_texture(&TextureDescriptor {
                label: Some(label),
                size: Extent3d { width: SIZE, height: SIZE, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
        };
        let target = tex("seal_test_target", format, TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC);
        let depth = tex("seal_test_depth", TextureFormat::Depth32Float, TextureUsages::RENDER_ATTACHMENT);
        let target_view = target.create_view(&Default::default());
        let depth_view = depth.create_view(&Default::default());
        let readback = device.create_buffer(&BufferDescriptor {
            label: Some("seal_test_readback"),
            size: (256 * SIZE) as u64,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("seal_test_pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &target_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: Operations { load: LoadOp::Clear(Color::WHITE), store: StoreOp::Store },
                })],
                depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(Operations { load: LoadOp::Clear(depth_clear), store: StoreOp::Store }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                multiview_mask: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&pipeline.pipeline);
            pass.set_bind_group(0, &uniforms.bind_group, &[]);
            pass.set_vertex_buffer(0, vb.slice(..));
            pass.set_index_buffer(ib.slice(..), IndexFormat::Uint32);
            pass.draw_indexed(0..3, 0, 0..1);
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
        let data = slice.get_mapped_range().unwrap();
        let c = (SIZE / 2) as usize * 256 + (SIZE / 2) as usize * 4;
        Some([data[c], data[c + 1], data[c + 2], data[c + 3]])
    }

    fn is_seal(px: [u8; 4]) -> bool {
        let want = |c: f32| (c * 255.0).round() as i32;
        let [r, g, b] = CRACK_SEAL_COLOUR;
        (px[0] as i32 - want(r)).abs() <= 2
            && (px[1] as i32 - want(g)).abs() <= 2
            && (px[2] as i32 - want(b)).abs() <= 2
    }

    /// A face pointing AWAY is painted the seal colour -- that is the whole
    /// mechanism: behind a crack there is only a back face.
    #[test]
    fn a_face_pointing_away_is_sealed() {
        let Some(px) = render_seal(true, 0.5, 1.0) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        assert!(is_seal(px), "a back face drew {px:?}, not the seal colour");
    }

    /// A face pointing TOWARDS the viewer is left alone. Drawing front faces
    /// dark would black out the level wherever the seal wins a depth tie.
    #[test]
    fn a_face_pointing_at_the_viewer_is_not_drawn() {
        let Some(px) = render_seal(false, 0.5, 1.0) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        assert_eq!(px, [255, 255, 255, 255], "the seal painted a front face: {px:?}");
    }

    /// A back face BEHIND something already drawn loses the depth test. The
    /// cleared depth stands in for the wall's front face: nearer than the back
    /// face, so the seal must not show -- which is why the seal only appears
    /// where the front faces left a hole.
    #[test]
    fn a_back_face_behind_the_wall_stays_hidden() {
        let Some(px) = render_seal(true, 0.6, 0.3) else {
            eprintln!("skipping: no GPU adapter available");
            return;
        };
        assert_eq!(px, [255, 255, 255, 255], "a back face behind the wall showed through: {px:?}");
    }
}

/// The brush's world position is NOT centroid-interpolated. See the note on
/// `world_pos` in the shader: it nearly doubled the scene pass on the headset.
#[cfg(test)]
mod centroid_tests {
    #[test]
    fn the_brush_does_not_pay_for_centroid_interpolation() {
        // The DIAGNOSTIC adds a centroid varying on purpose, and is not a
        // build that ships -- `the_edge_diagnostic_is_off` is what catches one
        // that escapes. Guarding here instead would let the diagnostic hide
        // the very cost this test exists to keep out.
        if super::BRUSH_EDGE_DEBUG {
            return;
        }
        let src = super::brush_shader_with(false);

        // A BUDGET, NOT A BAN. Centroid on every varying cost ~16 ms a frame
        // on the headset, which is more than the whole budget; centroid on the
        // two varyings that feed the two channels measured to move at a seam
        // is a different proposition, and `BRUSH_CENTROID_VARYINGS` says which
        // two and why. What must not come back is the version that was
        // measured -- so this counts them and names them rather than forbidding
        // the qualifier outright.
        let allowed: &[&str] = if super::BRUSH_CENTROID_VARYINGS {
            &["world_pos", "uv2"]
        } else {
            &[]
        };
        let carriers: Vec<&str> = src
            .lines()
            .filter(|l| l.contains("@interpolate(perspective, centroid)"))
            .filter_map(|l| l.rsplit_once(") ").map(|(_, rest)| rest))
            .filter_map(|rest| rest.split(':').next())
            .map(str::trim)
            .collect();
        for name in &carriers {
            assert!(
                allowed.contains(name),
                "centroid interpolation has spread to `{name}`. It is allowed on \
                 {allowed:?} and nowhere else: on every varying it measured ~16 ms \
                 a frame on the headset, which does not fit in 13.9 ms. Add one \
                 only with a headset frame time next to it.",
            );
        }
        assert_eq!(
            carriers.len(),
            allowed.len(),
            "expected centroid on exactly {allowed:?}, found {carriers:?}",
        );
    }

    /// The diagnostic replaces the brush's whole shading tail with false
    /// colour, so a build that ships with it on renders the level in red and
    /// black. Every value test in this file fails first, but they fail for
    /// reasons that read like broken lighting; this one names the cause.
    #[test]
    fn the_edge_diagnostic_is_off() {
        assert!(
            !super::BRUSH_EDGE_DEBUG,
            "BRUSH_EDGE_DEBUG is on: this build paints every brush by whether \
             MSAA shaded it from outside its polygon, and is a measurement, \
             not a build to ship",
        );
    }
}

#[cfg(test)]
mod trace_shader_tests {
    use super::*;

    fn fragment_of(trace: bool) -> String {
        let path = if trace { SsrPath::Trace } else { SsrPath::Inline };
        let src = brush_shader_modes(true, false, false, path);
        let i = src.find("fn fs_main").expect("no fragment entry point");
        src[i..].to_string()
    }

    /// THE TRACE AND THE SHIPPED PASS MUST PREPARE THE RAY IDENTICALLY.
    ///
    /// They are different pipelines running different shaders over the same
    /// geometry, and the whole design rests on them agreeing about where each
    /// ray goes. The prep -- the roughness mip, the blended normal, the Fresnel
    /// term, `reflectivity` -- decides that, so it is written once and shared.
    /// If it is ever duplicated, this fails.
    #[test]
    fn the_trace_and_the_shipped_pass_share_the_ray_prep() {
        let (shipped, trace) = (fragment_of(false), fragment_of(true));
        for line in [
            "let ssr_rough = textureSampleLevel(mat_rough, mat_rough_samp, in.uv, i32(in.material), 4.0).r;",
            "let ssr_n = normalize(mix(n_geom, n, 1.0 - smoothstep(0.1, 0.4, ssr_rough)));",
            "let ssr_fresnel = 0.04 + 0.96 * pow(1.0 - ssr_cos, 5.0);",
            "let reflectivity = clamp((1.0 - ssr_rough) * ssr_fresnel, 0.0, MAX_SSR_REFLECTIVITY);",
        ] {
            assert!(shipped.contains(line), "the shipped reflective pass lost: {line}");
            assert!(
                trace.contains(line),
                "the trace prepares its ray differently from the pass that \
                 composites it, so the two will disagree about where the ray \
                 went. Missing: {line}",
            );
        }
    }

    /// The composite applies the pixel's OWN reflectivity, and the trace does
    /// not -- or it would be squared and every reflection would be far too dark.
    #[test]
    fn reflectivity_is_applied_exactly_once() {
        let trace = fragment_of(true);
        assert!(
            trace.contains("return vec4<f32>(0.0);") || trace.contains("ssr_reflect("),
            "the trace no longer calls the march",
        );
        let block = crate::renderer::ssr::wgsl_ssr_block_shared_camera_radiance(3);
        assert!(
            block.contains("return vec4<f32>(hit_color, clamp(fade, 0.0, 1.0));"),
            "the trace folds something other than `fade` alone into confidence; \
             if that something is `weight`, the composite squares it",
        );
        let composite = {
            let src = brush_shader_modes(true, false, false, SsrPath::Composite);
            let i = src.find("fn fs_main").expect("no fragment entry point");
            src[i..].to_string()
        };
        assert!(
            composite.contains("clamp(reflectivity, 0.0, 1.0) * refl.a"),
            "the composite does not weight the reflection by this pixel's own \
             reflectivity:\n{composite}",
        );
        assert!(
            !composite.contains("ssr_scene_depth"),
            "the composite still declares the depth pyramid; it does not march \
             and binding 1 is the filtered reflection now",
        );
    }

    /// The trace writes RADIANCE AND CONFIDENCE and nothing else.
    #[test]
    fn the_trace_writes_confidence_and_never_reads_the_fallback() {
        let trace = fragment_of(true);
        assert!(
            trace.contains(
                "return ssr_reflect(in.world_pos, ssr_n, vec3<f32>(0.0), reflectivity, ssr_rough);"
            ),
            "the trace no longer returns the march's result directly:\n{trace}",
        );
        // The fallback is the SURFACE's environment term. Baking it into the
        // buffer would let the filter blur one surface's environment into its
        // neighbour's, which is the artefact this whole pass exists to remove.
        assert!(
            !trace.contains("ssr_fallback"),
            "the trace reads the environment fallback; it must write only what \
             the march found, and leave the fallback to the composite",
        );
        // And the shipped path must still do the blending it always did.
        let shipped = fragment_of(false);
        assert!(
            shipped.contains("let ssr_fallback = textureLoad(ssr_scene_color, vec2<i32>(in.clip.xy), 0).rgb;")
                && shipped.contains("albedo.a * in.tint.a"),
            "the shipped reflective pass changed while the trace was added",
        );
    }
}

// Scene shader sources, for `multiview::every_scene_shader_survives_the_multiview_transform`.
// Test-only: the gate has to see exactly the text each pipeline is built from,
// and nothing on a development machine can build a multiview pipeline to check.
#[cfg(test)]
pub fn brush_shader_src() -> String {
    brush_shader()
}
#[cfg(test)]
pub fn brush_seal_shader_src() -> String {
    brush_seal_shader()
}

#[cfg(test)]
mod normal_variance_roughness_tests {
    use super::{roughness_chain_with_normal_variance, NORMAL_VARIANCE_CLAMP};
    use crate::renderer::terrain_pipeline::TerrainImage;

    fn solid(w: u32, h: u32, px: [u8; 4]) -> TerrainImage {
        TerrainImage { width: w, height: h, rgba: px.iter().copied().cycle().take((w * h * 4) as usize).collect() }
    }

    /// A checkerboard of normals leaning hard in OPPOSITE directions. Averaging
    /// two of these cancels most of the vector, which is the whole signal.
    fn opposing(w: u32, h: u32) -> TerrainImage {
        let mut rgba = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h {
            for x in 0..w {
                // +x-leaning and -x-leaning tangent normals, alternating.
                let lean = if (x + y) % 2 == 0 { 255u8 } else { 0u8 };
                rgba.extend_from_slice(&[lean, 128, 128, 255]);
            }
        }
        TerrainImage { width: w, height: h, rgba }
    }

    #[test]
    fn a_flat_normal_map_changes_the_roughness_not_at_all() {
        // |Na| == 1 at every level, so sigma2 == 0 and there is nothing to add.
        // This is the guard that the bake cannot quietly roughen every material
        // in the game just by being switched on.
        let rough = solid(8, 8, [100, 100, 100, 255]);
        let flat = solid(8, 8, [128, 128, 255, 255]);
        let with = roughness_chain_with_normal_variance(&rough, Some(&flat));
        let without = roughness_chain_with_normal_variance(&rough, None);
        assert_eq!(with.len(), without.len());
        for (lvl, (a, b)) in with.iter().zip(without.iter()).enumerate() {
            // One code value of slack: the flat map decodes to 0.00392 in z
            // rather than exactly 1.0 because 128 is not exactly 0.5.
            for (i, (x, y)) in a.rgba.iter().zip(b.rgba.iter()).enumerate() {
                assert!(
                    (*x as i32 - *y as i32).abs() <= 2,
                    "level {lvl} byte {i}: flat normals moved roughness {} -> {}",
                    y, x,
                );
            }
        }
    }

    #[test]
    fn level_zero_is_never_touched() {
        // Nothing has been averaged at level 0, so no variation has been lost.
        // A surface seen up close must keep exactly the roughness authored for
        // it -- and "up close" is where the player inspects a material.
        let rough = solid(8, 8, [60, 60, 60, 255]);
        let with = roughness_chain_with_normal_variance(&rough, Some(&opposing(8, 8)));
        assert_eq!(with[0].rgba, rough.rgba, "level 0 was modified");
    }

    #[test]
    fn opposing_normals_roughen_the_coarser_mips() {
        let rough = solid(8, 8, [60, 60, 60, 255]);
        let with = roughness_chain_with_normal_variance(&rough, Some(&opposing(8, 8)));
        let without = roughness_chain_with_normal_variance(&rough, None);
        assert!(with.len() > 2);
        assert!(
            with[1].rgba[0] > without[1].rgba[0],
            "mip 1 did not get rougher ({} vs {}) even though its normals cancel",
            with[1].rgba[0],
            without[1].rgba[0],
        );
    }

    #[test]
    fn roughness_combines_in_variance_space_not_by_addition() {
        // THE ARITHMETIC THIS IS EASY TO GET WRONG. Roughness values do not
        // add; their SQUARES do. With base r and added variance k the answer is
        // sqrt(r^2 + k) -- which for r = 60/255 and a saturated kernel is very
        // different from r + something.
        let base_byte = 60u8;
        let rough = solid(4, 4, [base_byte, base_byte, base_byte, 255]);
        let with = roughness_chain_with_normal_variance(&rough, Some(&opposing(4, 4)));
        let r = base_byte as f32 / 255.0;
        // Fully opposed normals saturate the clamp.
        let expected = ((r * r + NORMAL_VARIANCE_CLAMP).min(1.0)).sqrt();
        let expected_byte = (expected * 255.0).round() as u8;
        let got = with[1].rgba[0];
        assert!(
            (got as i32 - expected_byte as i32).abs() <= 2,
            "variance-space combine wrong: got {got}, expected ~{expected_byte}",
        );
        // And emphatically NOT the naive sum, which would be a different number.
        let naive = ((r + NORMAL_VARIANCE_CLAMP).min(1.0) * 255.0).round() as u8;
        assert!(
            (got as i32 - naive as i32).abs() > 2,
            "the result matches naive addition ({naive}); the squares are not being used",
        );
    }

    #[test]
    fn a_normal_map_of_a_different_size_is_refused_rather_than_guessed() {
        // Matching texels between differently-sized maps would apply one
        // surface's variance to another's roughness. Doing nothing is the
        // honest failure, and silence here is preferable to a wrong bake.
        let rough = solid(8, 8, [90, 90, 90, 255]);
        let wrong_size = solid(4, 4, [255, 128, 128, 255]);
        let with = roughness_chain_with_normal_variance(&rough, Some(&wrong_size));
        let without = roughness_chain_with_normal_variance(&rough, None);
        for (a, b) in with.iter().zip(without.iter()) {
            assert_eq!(a.rgba, b.rgba, "a mismatched normal map still altered roughness");
        }
    }

    #[test]
    fn the_added_roughness_is_capped() {
        // Without the cap a texel whose normals fully cancel goes to 1.0 and
        // loses its highlight completely -- trading a shimmer for a dead
        // surface.
        assert!(NORMAL_VARIANCE_CLAMP < 0.25, "the cap is high enough to flatten highlights");
        let rough = solid(4, 4, [10, 10, 10, 255]);
        let with = roughness_chain_with_normal_variance(&rough, Some(&opposing(4, 4)));
        let r = 10.0f32 / 255.0;
        let ceiling = ((r * r + NORMAL_VARIANCE_CLAMP).sqrt() * 255.0).round() as u8;
        assert!(with[1].rgba[0] <= ceiling + 2, "added roughness exceeded the cap");
    }
}

#[cfg(test)]
mod roughness_sampler_tests {
    /// Every `@group(1) @binding(N)` the brush shader declares.
    fn material_bindings(src: &str) -> Vec<u32> {
        let mut out: Vec<u32> = src
            .match_indices("@group(1) @binding(")
            .filter_map(|(i, _)| {
                let rest = &src[i + "@group(1) @binding(".len()..];
                rest.split(')').next()?.trim().parse().ok()
            })
            .collect();
        out.sort_unstable();
        out.dedup();
        out
    }

    /// THE WHOLE POINT OF THE SECOND SAMPLER. Roughness must not be read
    /// through the anisotropic one: aniso fetches a higher-resolution mip at
    /// grazing angles, and a higher-resolution roughness mip has had LESS of
    /// the normal's variance folded into it -- so the surface shimmers again
    /// exactly where a glancing view makes it worst.
    #[test]
    fn roughness_is_read_through_the_trilinear_sampler() {
        let src = super::brush_shader_src();
        assert!(
            src.contains("@group(1) @binding(5) var mat_rough_samp: sampler;"),
            "the roughness sampler is not declared",
        );
        for bad in [
            "textureSample(mat_rough, mat_samp",
            "textureSampleLevel(mat_rough, mat_samp",
        ] {
            assert!(
                !src.contains(bad),
                "roughness is still being read through the anisotropic sampler: {bad}",
            );
        }
        // COLOUR AND NORMAL MUST NOT FOLLOW IT. They keep 8x anisotropy --
        // that is what stops a mipped wall going soft at a grazing angle, and
        // only the ROUGHNESS map has a reason to give it up.
        for keep in ["textureSample(mat_color, mat_samp", "textureSample(mat_normal, mat_samp"] {
            assert!(
                src.contains(keep),
                "{keep} lost the anisotropic sampler; walls will go soft at grazing angles",
            );
        }
    }

    /// The shader's group-1 bindings must be a contiguous run starting at 0.
    ///
    /// A gap means the layout and the shader have drifted, and that failure
    /// surfaces at BIND GROUP CREATION on the device -- not at compile, and
    /// not in any test that does not own a GPU. Cheap to assert from the text.
    #[test]
    fn the_material_bindings_have_no_holes() {
        let b = material_bindings(&super::brush_shader_src());
        assert!(!b.is_empty(), "the brush shader declares no group-1 bindings at all");
        let expected: Vec<u32> = (0..b.len() as u32).collect();
        assert_eq!(
            b, expected,
            "group(1) bindings are {b:?}; the layout in \
             `brush_material_bind_group_layout` must declare exactly these",
        );
    }
}

#[cfg(test)]
mod lightmap_clamp_tests {
    /// THE MEASUREMENT THIS EXISTS FOR (headset, 2026-09-22). In the
    /// lighting-sources view, every sample taken along a ceiling seam showed
    /// the BAKED channel down 22-61 against its neighbours while direct barely
    /// moved -- the lightmap read was failing at edge pixels, not the probe.
    ///
    /// An MSAA edge pixel is shaded at its centre, which can lie outside the
    /// polygon, so its interpolated uv2 is EXTRAPOLATED past the face's patch
    /// in the atlas. The gutter cannot bound that: the lightmap has no mips, so
    /// a distant pixel spans many atlas texels and the overshoot is many texels
    /// wide against a gutter of 2.
    #[test]
    fn both_lightmap_reads_use_the_clamped_uv() {
        let src = super::brush_shader_src();
        assert!(
            src.contains("let lm_uv = clamp(in.uv2, in.uv2_rect.xy, in.uv2_rect.zw);"),
            "the lightmap uv is no longer clamped into the face's own patch",
        );
        for read in ["textureSample(lm_tex, lm_samp, lm_uv)", "textureSample(lm_dir_tex, lm_samp, lm_uv)"] {
            assert!(src.contains(read), "a lightmap read bypasses the clamp: {read}");
        }
        assert!(
            !src.contains("textureSample(lm_tex, lm_samp, in.uv2)")
                && !src.contains("textureSample(lm_dir_tex, lm_samp, in.uv2)"),
            "a lightmap read still samples the raw interpolated uv2",
        );
    }

    /// The rectangle must be FLAT. Interpolating it smoothly would extrapolate
    /// it at exactly the edge pixels it exists to bound -- the bound would
    /// inherit the bug.
    #[test]
    fn the_patch_rectangle_is_flat_interpolated() {
        let src = super::brush_shader_src();
        assert!(
            src.contains("@location(8) @interpolate(flat) uv2_rect: vec4<f32>;")
                || src.contains("@location(8) @interpolate(flat) uv2_rect: vec4<f32>,"),
            "uv2_rect lost its flat interpolation",
        );
    }
}

#[cfg(test)]
mod face_clamp_tests {
    /// The clamp must reach `shade_material_env`, not merely be computed.
    ///
    /// A `face_pos` that is worked out and then not passed is the failure mode
    /// this file has already seen once -- a callback declared, forwarded and
    /// emitted by nothing compiles clean and leaves the feature unreachable.
    #[test]
    fn the_shading_position_is_the_clamped_one() {
        let src = super::brush_shader_with(false);
        assert!(
            src.contains("let face_pos = in.face_centre"),
            "the per-face position clamp is gone from the brush shader",
        );
        assert!(
            src.contains("shade_material_env_part(\n        face_pos,"),
            "the clamp is computed but the RAW `in.world_pos` is still what \
             gets shaded, so it corrects nothing",
        );
    }

    /// The ROOM is chosen from the face centre, the photograph within it per
    /// pixel. An extrapolated edge sample on a grazing ceiling otherwise lands
    /// past the far wall and takes the outdoor probe (offline_frame, 2026-09-23).
    #[test]
    fn the_room_is_chosen_from_the_face_centre() {
        let src = super::brush_shader_with(false);
        assert!(src.contains("probe_volume_pos = vec4<f32>(in.face_centre, 1.0);"));
        assert!(src.contains("bdir, albedo.rgb, face_pos,"));
    }

    /// `flat`, or it inherits the extrapolation it exists to bound -- the same
    /// mistake `face_centre` and `uv2_rect` each carry a note about.
    #[test]
    fn the_extent_is_flat_interpolated() {
        let src = super::brush_shader_with(false);
        assert!(
            src.contains("@location(9) @interpolate(flat) face_half_extent: vec2<f32>,"),
            "face_half_extent is being smoothly interpolated, which extrapolates \
             it at exactly the edge pixels it is meant to correct",
        );
    }
}
