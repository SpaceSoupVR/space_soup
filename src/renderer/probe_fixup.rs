//! THE PROBE PASS'S SECONDARY LOOKUPS, MADE AFTER IT, FOR THE FEW TEXELS THAT
//! NEED THEM.
//!
//! A traced reflection sometimes needs a second look: its lobe straddles a
//! doorway's rim (the other side of the rim), or a solid proxy's outline (what
//! lies past it). See `lights::probe_secondary`. In the probe pass those
//! lookups -- a second trace and more photographs -- held the shader at 26
//! registers a pixel where the rest needs 18, so the pass kept 37% of its
//! waves in flight instead of 62% (the driver's own numbers, `PIPESTATS`,
//! 2026-09-28). And they are rare: 0.0-0.7% of the pass's texels in five of
//! the six benchmark views, 9% down the hallway.
//!
//! So the probe pass that ships defers them (`PROBE_SECONDARY_DEFERRED`): it
//! writes every texel's primary reflection as usual, and for a texel that
//! needs more it appends a record ([`RECORD_WGSL`]) of everything the lookups
//! read. A compute pass then makes them, through the same
//! `probe_secondary`, and writes the finished texel over the primary one --
//! so the result is the one the pass made before, and only the texels that
//! need the lookups pay for them.
//!
//! A record can come from a fragment that a nearer one of the same pass later
//! covered, so the fix-up keeps only records whose depth is the depth the
//! pass left in the texel.

use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
    BindingResource, BindingType, Buffer, BufferBindingType, BufferDescriptor, BufferUsages, CommandEncoder,
    ComputePipeline, Device, ShaderModuleDescriptor, ShaderSource, ShaderStages, StorageTextureAccess,
    TextureSampleType, TextureViewDimension,
};

use super::brush_pipeline::probe_pass;

/// One deferred texel: everything `probe_secondary` reads, as the probe pass
/// had it. Shared by the pass that writes it (through the lights block, see
/// [`lights_block_wgsl`]) and the fix-up that reads it.
pub const RECORD_WGSL: &str = r#"
// ONE TEXEL'S DEFERRED SECONDARY LOOKUPS. See `probe_fixup`.
struct ProbeFixup {
    // x, y: the texel; z: the depth its fragment was at, as bits; w: 1 where
    // the hit is on a model's cards, whose colour the pass left to the fix-up
    // (`lights::probe_hit_carded`).
    texel: vec4<u32>,
    // Its reflection before the lookups.
    col: vec4<f32>,
    // xyz: where the ray left from; w: the room it was traced from.
    from_pos: vec4<f32>,
    // xyz: the ray's world direction; w: the surface's roughness.
    dir_world: vec4<f32>,
    // xyz: the direction as `probe_environment` was handed it; w: the probe mip.
    dir_given: vec4<f32>,
    // xyz: the hit's `origin`; w: `pixel_footprint`.
    origin: vec4<f32>,
    // The hit's `rim`, `rim_t`, `edge_cover` and `edge_t`.
    hit: vec4<f32>,
    // x: the hit's `rim_code`, y: its `edge_code`, as bits; z:
    // `probe_eye_distance`.
    codes: vec4<f32>,
}
"#;

/// The size of one [`RECORD_WGSL`] record.
pub const RECORD_BYTES: u64 = 128;

/// The lights block's part in this: for the probe pass that defers its
/// lookups, the record list (group 3) and the two functions that fill a record
/// -- `probe_fixup_begin` the moment the trace ends, with the ray and the hit,
/// so none of that is carried through the colour lookup after it, and
/// `probe_fixup_finish` with the colour. For every other shader, stubs, so
/// `probe_environment` reads the same either way. See
/// `lights::PROBE_SECONDARY_DEFERRED`.
pub fn lights_block_wgsl(deferred: bool) -> String {
    if !deferred {
        return r#"
fn probe_fixup_begin(h: ProbeHit, world_pos: vec3<f32>, d: vec3<f32>, dir: vec3<f32>, roughness: f32, probe_lod: f32, trace_room: f32, recolour: bool) -> i32 {
    return -1;
}
fn probe_fixup_finish(slot: i32, col: vec4<f32>) {
}
// No floor mirror but in the pass that defers: see `floor_mirror_blend_wgsl`.
fn probe_floor_mirror_pass(col: vec4<f32>, roughness: f32) -> vec4<f32> {
    return col;
}
"#
        .to_string();
    }
    format!(
        "{RECORD_WGSL}{}{}",
        floor_mirror_blend_wgsl(),
        r#"
struct ProbeFixups {
    count: atomic<u32>,
    items: array<ProbeFixup>,
}
@group(3) @binding(0) var<storage, read_write> probe_fixups: ProbeFixups;
// This eye's floor mirror and its depth. See `floor_mirror_blend_wgsl`.
@group(3) @binding(1) var probe_fixup_mirror: texture_2d<f32>;
@group(3) @binding(2) var probe_fixup_mirror_depth: texture_depth_2d;

// The mirrored characters over this texel's reflection.
fn probe_floor_mirror_pass(col: vec4<f32>, roughness: f32) -> vec4<f32> {
    return floor_mirror_blend(
        col, probe_fixup_mirror, probe_fixup_mirror_depth, vec2<i32>(probe_fragment.xy), probe_fragment.z,
        probe_eye_distance, roughness, probe_reach,
    );
}

// Opens this texel's record with everything but the reflection's colour: its
// slot, or -1 where the list is full -- see `ProbeFixups::new` for why that
// does not happen; the texel would keep its primary reflection. `recolour`:
// the hit is on a model's cards, coloured here (`lights::probe_hit_carded`).
fn probe_fixup_begin(h: ProbeHit, world_pos: vec3<f32>, d: vec3<f32>, dir: vec3<f32>, roughness: f32, probe_lod: f32, trace_room: f32, recolour: bool) -> i32 {
    let k = atomicAdd(&probe_fixups.count, 1u);
    var slot = -1;
    if (k < arrayLength(&probe_fixups.items)) {
        slot = i32(k);
        probe_fixups.items[k].texel = vec4<u32>(vec2<u32>(probe_fragment.xy), bitcast<u32>(probe_fragment.z), select(0u, 1u, recolour));
        probe_fixups.items[k].from_pos = vec4<f32>(world_pos, trace_room);
        probe_fixups.items[k].dir_world = vec4<f32>(d, roughness);
        probe_fixups.items[k].dir_given = vec4<f32>(dir, probe_lod);
        probe_fixups.items[k].origin = vec4<f32>(h.origin, pixel_footprint);
        probe_fixups.items[k].hit = vec4<f32>(h.rim, h.rim_t, h.edge_cover, h.edge_t);
        // w: the characters' part, for the fix-up to make again -- the
        // floor mirror (negative), or the capsules lit by this luminance.
        let lit = dot(probe_capsule_lit, vec3<f32>(0.2126, 0.7152, 0.0722));
        probe_fixups.items[k].codes = vec4<f32>(
            bitcast<f32>(h.rim_code), bitcast<f32>(h.edge_code), probe_eye_distance, select(lit, -1.0 - lit, probe_floor_mirror_here),
        );
    }
    return slot;
}
// Adds the reflection's colour before the lookups.
fn probe_fixup_finish(slot: i32, col: vec4<f32>) {
    if (slot >= 0) {
        probe_fixups.items[slot].col = col;
    }
}
"#
    )
}

/// THE CHARACTERS' FLOOR MIRROR over a texel's reflection
/// (`brush_pipeline::probe_pass::MIRROR_FORMAT`): in the probe pass that
/// defers, and in the fix-up that finishes a texel it deferred -- the record's
/// `codes.w` says the texel is on the mirror's plane -- so a texel at a
/// doorway's rim keeps the character standing over it.
///
/// `floor_mirror_blend` reads the mirror as blurred as the floor is rough:
/// the level where the lobe, at a body's typical height above the floor, is a
/// texel of the pass across. It hides a character behind what the trace met
/// (`reach`), from the character's reflected distance: the mirror's depth and
/// this texel's, turned back into distances along the eye's axis, scaled to
/// the ray by the texel's own distance from the eye. Where the character
/// covers most of the texel the reach becomes the character's, so SpaceWarp
/// moves the reflected image with the character rather than with what stands
/// behind it.
fn floor_mirror_blend_wgsl() -> String {
    format!(
        r#"
const FLOOR_MIRROR_EXPOSURE: f32 = {exposure:?};
const FLOOR_MIRROR_REACH: f32 = 0.6;
{linear}
fn floor_mirror_blend(
    col: vec4<f32>,
    mirror: texture_2d<f32>,
    mirror_depth: texture_depth_2d,
    texel: vec2<i32>,
    depth_here: f32,
    t_here: f32,
    roughness: f32,
    reach: f32,
) -> vec4<f32> {{
    let dims = vec2<f32>(textureDimensions(mirror));
    let blur = probe_lobe_tan(roughness) * FLOOR_MIRROR_REACH / max(pixel_footprint, 1e-4);
    let lod = clamp(log2(max(blur, 1.0)), 0.0, f32(textureNumLevels(mirror)) - 1.0);
    let m = textureSampleLevel(mirror, probe_samp, (vec2<f32>(texel) + 0.5) / dims, lod);
    if (m.a < 0.002) {{
        return col;
    }}
    var cover = m.a;
    let d_char = textureLoad(mirror_depth, texel, 0);
    if (d_char < 1.0) {{
        let t_char = floor_mirror_linear_depth(d_char) * t_here / max(floor_mirror_linear_depth(depth_here), 1e-4) - t_here;
        cover = cover * smoothstep(t_char - 0.25, t_char - 0.05, reach);
        if (cover > 0.5) {{
            probe_reach = t_char;
        }}
    }}
    return vec4<f32>(mix(col.rgb, m.rgb / (m.a * FLOOR_MIRROR_EXPOSURE), cover), max(col.a, cover));
}}
"#,
        exposure = probe_pass::MIRROR_EXPOSURE,
        linear = probe_pass::eye_linear_depth_wgsl(),
    )
}

/// How far a texel's depth may lie from the depth a record was made at and
/// still be the same fragment: well under the gap between two surfaces a
/// centimetre apart at 10 m (about 5e-6 here), and a few units in the last
/// place of a depth near 1.
const DEPTH_TOLERANCE: f32 = 1e-6;

/// The fix-up's compute shader.
fn compute_wgsl() -> String {
    compute_wgsl_with_entry(FIXUP_ENTRY)
}

/// MEASUREMENT ONLY: the fix-up as it was to 2026-10-06, each lookup made by
/// its own call -- `probe_subsample`'s two, `probe_secondary`'s three and its
/// far end's -- so a trace and a colouring compiled in at every one: 56,153
/// instructions at 45 registers (PIPESTATS, deploy113). Run by the
/// `fixup_inlined` cut, to compare.
fn compute_wgsl_inlined() -> String {
    compute_wgsl_with_entry(FIXUP_ENTRY_INLINED)
}

/// The fix-up's bindings and constants, then `entry`: its `fixup`.
fn compute_wgsl_with_entry(entry: &str) -> String {
    format!(
        r#"
{lights}
{record}
struct ProbeFixupList {{
    count: u32,
    items: array<ProbeFixup>,
}}
@group(1) @binding(0) var<storage, read> fixups: ProbeFixupList;
@group(2) @binding(0) var probe_out: texture_storage_2d<rgba16float, write>;
@group(2) @binding(1) var probe_depth_in: texture_depth_2d;
// The floor mirror, its depth and the pass's reach. See `floor_mirror_blend_wgsl`.
@group(2) @binding(2) var fixup_mirror: texture_2d<f32>;
@group(2) @binding(3) var fixup_mirror_depth: texture_depth_2d;
@group(2) @binding(4) var fixup_reach: texture_2d<f32>;
{blend}

const FIXUP_DEPTH_TOLERANCE: f32 = {tolerance:?};

{entry}
"#,
        // Every texel meeting a model on cards comes through here, so its
        // cards' trust is filtered here, and the lamps the bake never saw
        // light it here. See `PROBE_CARD_TESTS_FILTERED`, `PROBE_CARD_RELIT`.
        lights = super::lights::wgsl_lights_block_with(
            0,
            1,
            super::lights::LightsBlockOptions { card_tests_filtered: true, card_relit: true, ..Default::default() },
        ),
        record = RECORD_WGSL,
        tolerance = DEPTH_TOLERANCE,
        blend = floor_mirror_blend_wgsl(),
        entry = entry,
    )
}


/// THE FIX-UP, ONE LOOKUP A TURN OF ONE LOOP. A record asks for up to six:
/// four rays across a model's outline (or its one ray again, for a retest),
/// the other side of a doorway's rim, that opening's far end, and what lies
/// past a solid proxy's outline -- each a ray to trace or a point already
/// known, then coloured. Made through `probe_subsample` and `probe_secondary`,
/// the trace and the colouring were compiled in at each of their eight calls:
/// 56,153 instructions at 45 registers, for a pass whose every wave runs a
/// different few thousand of them (PIPESTATS, deploy113). Here each is
/// compiled in once, and every step takes the same arithmetic in the same
/// order as those two functions do, so the texel comes out the same.
const FIXUP_ENTRY: &str = r#"
// The steps after the subsample's four (0-3). See `FIXUP_ENTRY`.
const FIXUP_RIM: u32 = 4u;
const FIXUP_FAR: u32 = 5u;
const FIXUP_EDGE: u32 = 6u;
const FIXUP_DONE: u32 = 7u;

// A NEAR RIM'S FAR END, as `probe_through_far_end` works it out: whether it
// applies, whether this line of the lobe leaves out of the far end, the far
// part's share of the opening, how far along the ray it lies, and where its
// lookup starts -- the side just inside the wall when it leaves, else the
// point it is traced on from.
struct FixupFar {
    applies: bool,
    out: bool,
    share: f32,
    t: f32,
    start: vec3<f32>,
}

fn fixup_far_end(hit: ProbeHit, d: vec3<f32>, roughness: f32) -> FixupFar {
    var far: FixupFar;
    if ((hit.rim_code & PROBE_RIM_FAR) != 0 || roughness >= PROBE_FAR_END_MAX_ROUGHNESS) {
        return far;
    }
    let p = (hit.rim_code >> 3u) & 31;
    let axis = (hit.rim_code >> 1u) & 3;
    // Parallel to the wall, the ray never reaches its far face.
    if (abs(d[axis]) < 1e-4) {
        return far;
    }
    let plo = camera.probe_portals[p * 3].xyz;
    let phi = camera.probe_portals[p * 3 + 1].xyz;
    let wall_far = select(camera.probe_portals[p * 3 + 2].y, camera.probe_portals[p * 3 + 2].z, d[axis] > 0.0);
    let e = hit.origin + d * hit.rim_t;
    let t2 = max((wall_far - e[axis]) / d[axis], 0.0);
    let e2 = e + d * t2;
    let a = (axis + 1) % 3;
    let b = (axis + 2) % 3;
    let in_a = min(e2[a] - plo[a], phi[a] - e2[a]);
    let in_b = min(e2[b] - plo[b], phi[b] - e2[b]);
    let t = hit.rim_t + t2;
    let spread = max(t * probe_lobe_tan(roughness), probe_pixel_spread(t));
    if (abs(min(in_a, in_b)) >= spread) {
        return far;
    }
    let inside = smoothstep(-spread, spread, in_a) * smoothstep(-spread, spread, in_b);
    far.applies = true;
    far.share = clamp(inside / max(hit.rim, 1e-3), 0.0, 1.0);
    far.out = min(in_a, in_b) > 0.0;
    far.t = t;
    far.start = probe_rim_point_far(e2, p, axis, !far.out);
    return far;
}

// THE RECORD, read where each step needs it rather than held: whatever the
// loop holds lives across the trace it calls, and holding the whole record,
// the hit and the step's flags there spilled 132 bytes a thread to scratch
// memory (PIPESTATS, deploy114) -- the trace's own state then waited on it.
// The record is read-only, so a second read of it is a cached load.
fn fixup_hit(k: u32) -> ProbeHit {
    var hit: ProbeHit;
    hit.found = true;
    hit.origin = fixups.items[k].origin.xyz;
    hit.rim = fixups.items[k].hit.x;
    hit.rim_t = fixups.items[k].hit.y;
    hit.edge_cover = fixups.items[k].hit.z;
    hit.edge_t = fixups.items[k].hit.w;
    hit.rim_code = bitcast<i32>(fixups.items[k].codes.x);
    hit.edge_code = bitcast<i32>(fixups.items[k].codes.y);
    return hit;
}

// Across one of a model's own outlines (`PROBE_SUBSAMPLE`).
fn fixup_marked(k: u32) -> bool {
    return bitcast<i32>(fixups.items[k].codes.y) >= 0 && fixups.items[k].hit.z < 0.0;
}

// A hit on a model's cards, which the pass did not colour: its one ray again,
// as a texel marked `PROBE_RETEST` takes it. Across one of the model's own
// outlines the subsample's rays colour it instead. Every colour made here
// lights a lit surface it meets, as the pass's did (`probe_surface_relit`).
fn fixup_recolour(k: u32) -> bool {
    return fixups.items[k].texel.w != 0u && !fixup_marked(k);
}

// One ray again rather than four across the outline.
fn fixup_retest(k: u32) -> bool {
    return fixup_recolour(k) || fixups.items[k].hit.z < 0.5 * (PROBE_SUBSAMPLE + PROBE_RETEST);
}

// The step after the rim's: the proxy's outline, if the record crosses one.
fn fixup_after_rim(k: u32) -> u32 {
    let has_edge = bitcast<i32>(fixups.items[k].codes.y) >= 0 && probe_edge_cover(fixups.items[k].hit.z) < 0.99;
    return select(FIXUP_DONE, FIXUP_EDGE, has_edge);
}

// The step after the subsample's: the rim's, if the record has one.
fn fixup_after_cards(k: u32) -> u32 {
    let has_rim = fixups.items[k].hit.x >= 0.0;
    return select(fixup_after_rim(k), FIXUP_RIM, has_rim);
}

@compute @workgroup_size(64)
fn fixup(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= min(fixups.count, arrayLength(&fixups.items))) {
        return;
    }
    let k = id.x;
    // Recorded by a fragment a nearer one of the same pass then covered.
    let texel0 = vec2<i32>(fixups.items[k].texel.xy);
    if (abs(textureLoad(probe_depth_in, texel0, 0) - bitcast<f32>(fixups.items[k].texel.z)) > FIXUP_DEPTH_TOLERANCE) {
        return;
    }
    pixel_footprint = fixups.items[k].origin.w;
    probe_eye_distance = fixups.items[k].codes.z;
    let has_cards = fixup_marked(k) || fixup_recolour(k);
    // ALL THE LOOP HOLDS: the colour so far -- the subsample's sum while it
    // runs, then the reflection; across a rim, what met the wall beside the
    // opening, with `through` what passed into it -- the far end's share and
    // side, and which step is next.
    var col = fixups.items[k].col;
    var through = vec4<f32>(0.0);
    var far_share = 0.0;
    var far_out = false;
    var turn = select(fixup_after_cards(k), 0u, has_cards);
    if (has_cards && !fixup_retest(k)) {
        col = vec4<f32>(0.0);
    }
    loop {
        if (turn >= FIXUP_DONE) {
            break;
        }
        // THIS TURN'S LOOKUP: a ray traced from `start`, or the point `start`
        // already known -- the wall beside a rim, the far end's side, a
        // proxy's outline -- `t_before` along the reflected ray.
        let hit = fixup_hit(k);
        let d = fixups.items[k].dir_world.xyz;
        let roughness = fixups.items[k].dir_world.w;
        var start = fixups.items[k].from_pos.xyz;
        var ray = d;
        var room = fixups.items[k].from_pos.w;
        var skip = -1;
        var traced = true;
        var t_before = 0.0;
        var point_room = -1.0;
        var point_other = -1.0;
        if (turn < FIXUP_RIM) {
            // A TEXEL ACROSS ONE OF A MODEL'S OWN OUTLINES, traced again: four
            // rays in a rotated grid across what the texel's reflection covers,
            // each coloured as the pass colours a hit, averaged. See
            // `PROBE_SUBSAMPLE`. Spread over the edge's softening width --
            // `PROBE_EDGE_FOOTPRINTS` of the pixel each side, or the lobe where a
            // rough surface's is wider -- as every other reflected edge is, so
            // the scene's bilinear read of this half-resolution pass does not
            // snap the outline to its texels. A ray that finds nothing keeps the
            // texel's own colour. The grid is turned a quarter each ray, from the
            // turn's number rather than an array, which Adreno puts in scratch
            // memory when a loop indexes it. A texel marked `PROBE_RETEST` takes
            // its one ray again instead.
            if (!fixup_retest(k)) {
                let spread = 2.0 * max(probe_lobe_tan(roughness), PROBE_EDGE_FOOTPRINTS * pixel_footprint / max(probe_eye_distance, 0.05));
                let ga = normalize(cross(d, select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(d.y) > 0.9)));
                let gb = cross(d, ga);
                var o = vec2<f32>(0.125, 0.375);
                if ((turn & 1u) != 0u) {
                    o = vec2<f32>(-o.y, o.x);
                }
                if ((turn & 2u) != 0u) {
                    o = -o;
                }
                ray = normalize(d + (ga * o.x + gb * o.y) * spread);
            }
        } else if (turn == FIXUP_RIM) {
            // ACROSS A DOORWAY'S RIM: the wall beside the opening where the ray
            // went through it, a point already known; else traced again from
            // just inside the opening. See `probe_secondary`.
            start = probe_hit_rim_point(hit, d);
            room = -1.0;
            t_before = hit.rim_t;
            traced = !probe_hit_rim_went_through(hit);
            point_room = probe_hit_rim_room(hit);
        } else if (turn == FIXUP_FAR) {
            let far = fixup_far_end(hit, d, roughness);
            start = far.start;
            room = -1.0;
            t_before = far.t;
            traced = !far.out;
            point_room = probe_hit_rim_room(hit);
        } else {
            // ACROSS A SOLID PROXY'S OUTLINE: what lies past the proxy, traced
            // again as though it were not there, where the ray hit it; else the
            // proxy itself, at its outline.
            traced = probe_hit_edge_hit(hit);
            skip = probe_hit_edge(hit);
            if (!traced) {
                start = hit.origin + d * hit.edge_t;
                t_before = hit.edge_t;
                point_room = probe_hit_edge_room(hit);
                point_other = probe_proxy_model(skip);
            }
        }
        var h: ProbeHit;
        if (traced) {
            h = probe_trace_skipping(start, ray, room, roughness, skip);
            h.t = t_before + h.t;
        } else {
            h = probe_point_hit(start, point_room, t_before);
            h.other = point_other;
        }
        var c = vec4<f32>(0.0);
        if (h.found) {
            c = probe_traced_colour(h, ray, roughness, fixups.items[k].dir_given.xyz, fixups.items[k].dir_given.w);
        }
        // WHAT THE LOOKUP IS TO ITS STEP, as `probe_subsample`,
        // `probe_secondary` and `probe_through_far_end` make of it.
        if (turn < FIXUP_RIM) {
            let found = select(fixups.items[k].col, c, h.found);
            if (fixup_retest(k)) {
                col = found;
                turn = fixup_after_cards(k);
            } else {
                col = col + found;
                turn = turn + 1u;
                if (turn == FIXUP_RIM) {
                    col = 0.25 * col;
                    turn = fixup_after_cards(k);
                }
            }
        } else if (turn == FIXUP_RIM) {
            turn = fixup_after_rim(k);
            if (h.found) {
                let went_through = (bitcast<i32>(fixups.items[k].codes.x) & 1) != 0;
                through = select(c, col, went_through);
                col = select(col, c, went_through);
                // A NEAR rim's opening also has a far end. See
                // `probe_through_far_end`.
                let far = fixup_far_end(fixup_hit(k), fixups.items[k].dir_world.xyz, fixups.items[k].dir_world.w);
                if (far.applies) {
                    far_share = far.share;
                    far_out = far.out;
                    turn = FIXUP_FAR;
                } else {
                    col = mix(col, through, fixups.items[k].hit.x);
                }
            }
        } else if (turn == FIXUP_FAR) {
            if (h.found) {
                through = select(mix(through, c, far_share), mix(c, through, far_share), far_out);
            }
            col = mix(col, through, fixups.items[k].hit.x);
            turn = fixup_after_rim(k);
        } else {
            if (h.found) {
                let edge_cover = probe_edge_cover(fixups.items[k].hit.z);
                col = select(mix(col, c, edge_cover), mix(c, col, edge_cover), traced);
            }
            turn = FIXUP_DONE;
        }
    }
    // The characters over it, as the pass laid them: mirrored on the floor,
    // else their capsules (lit grey by the luminance the record kept).
    let f = fixups.items[k];
    let texel = vec2<i32>(f.texel.xy);
    let here = to_player_space(f.from_pos.xyz);
    if (f.codes.w < 0.0) {
        col = floor_mirror_blend(
            col, fixup_mirror, fixup_mirror_depth, texel, textureLoad(probe_depth_in, texel, 0), f.codes.z,
            f.dir_world.w, textureLoad(fixup_reach, texel, 0).r,
        );
    } else {
        col = capsule_reflection(
            here, normalize(f.dir_given.xyz), f.dir_world.w, vec3<f32>(f.codes.w), col,
        );
    }
    // As `probe_env_for_pass` finishes a traced reflection: compressed and
    // premultiplied by its coverage, the brightness normalisation out of it (a
    // traced hit leaves `probe_brightness` at 0, and its scale is exactly 1),
    // a carried glass's glow added as this point sees it past the glass's own
    // beam (see `capsule_glass_beam`).
    let a = clamp(col.a, 0.0, 1.0);
    textureStore(probe_out, texel, vec4<f32>(probe_pass_compress(col.rgb * 1.0 + capsule_glow) * a, a));
}
"#;

/// The fix-up before `FIXUP_ENTRY`: see `compute_wgsl_inlined`.
const FIXUP_ENTRY_INLINED: &str = r#"
// A TEXEL ACROSS ONE OF A MODEL'S OWN OUTLINES, traced again: four rays in a
// rotated grid across what the texel's reflection covers, each coloured as the
// pass colours a hit, averaged. See `PROBE_SUBSAMPLE`. Spread over the edge's
// softening width -- `PROBE_EDGE_FOOTPRINTS` of the pixel each side, or the
// lobe where a rough surface's is wider -- as every other reflected edge is,
// so the scene's bilinear read of this half-resolution pass does not snap the
// outline to its texels. A ray that finds nothing keeps the texel's own
// colour. The grid is turned a quarter each ray rather than read from an
// array, which Adreno puts in scratch memory when a loop indexes it.
//
// A texel marked `PROBE_RETEST` -- on a model's cards, away from its outlines
// -- takes its one ray again instead: its colour is what this pass's filtered
// card tests make of it.
fn probe_subsample(primary: vec4<f32>, world_pos: vec3<f32>, d: vec3<f32>, dir: vec3<f32>, roughness: f32, probe_lod: f32, trace_room: f32, retest: bool) -> vec4<f32> {
    if (retest) {
        let h = probe_trace(world_pos, d, trace_room, roughness);
        if (h.found) {
            return probe_traced_colour(h, d, roughness, dir, probe_lod);
        }
        return primary;
    }
    let spread = 2.0 * max(probe_lobe_tan(roughness), PROBE_EDGE_FOOTPRINTS * pixel_footprint / max(probe_eye_distance, 0.05));
    let a = normalize(cross(d, select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(d.y) > 0.9)));
    let b = cross(d, a);
    var o = vec2<f32>(0.125, 0.375);
    var sum = vec4<f32>(0.0);
    for (var k = 0; k < 4; k = k + 1) {
        let dk = normalize(d + (a * o.x + b * o.y) * spread);
        let h = probe_trace(world_pos, dk, trace_room, roughness);
        var c = primary;
        if (h.found) {
            c = probe_traced_colour(h, dk, roughness, dir, probe_lod);
        }
        sum = sum + c;
        o = vec2<f32>(-o.y, o.x);
    }
    return 0.25 * sum;
}

@compute @workgroup_size(64)
fn fixup(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= min(fixups.count, arrayLength(&fixups.items))) {
        return;
    }
    let f = fixups.items[id.x];
    let texel = vec2<i32>(f.texel.xy);
    // Recorded by a fragment a nearer one of the same pass then covered.
    if (abs(textureLoad(probe_depth_in, texel, 0) - bitcast<f32>(f.texel.z)) > FIXUP_DEPTH_TOLERANCE) {
        return;
    }
    pixel_footprint = f.origin.w;
    probe_eye_distance = f.codes.z;
    var hit: ProbeHit;
    hit.found = true;
    hit.origin = f.origin.xyz;
    hit.rim = f.hit.x;
    hit.rim_t = f.hit.y;
    hit.edge_cover = f.hit.z;
    hit.edge_t = f.hit.w;
    hit.rim_code = bitcast<i32>(f.codes.x);
    hit.edge_code = bitcast<i32>(f.codes.y);
    var primary = f.col;
    let marked = hit.edge_code >= 0 && hit.edge_cover < 0.0;
    // A hit on a model's cards, which the pass did not colour: its one ray
    // again, as a texel marked `PROBE_RETEST` takes it. Across one of the
    // model's own outlines the rays below colour it instead. Every colour
    // made here lights a lit surface it meets, as the pass's did
    // (`probe_surface_relit`).
    let recolour = f.texel.w != 0u && !marked;
    if (marked || recolour) {
        primary = probe_subsample(
            f.col, f.from_pos.xyz, f.dir_world.xyz, f.dir_given.xyz, f.dir_world.w, f.dir_given.w, f.from_pos.w,
            recolour || hit.edge_cover < 0.5 * (PROBE_SUBSAMPLE + PROBE_RETEST),
        );
    }
    var col = probe_secondary(
        hit, primary, f.from_pos.xyz, f.dir_world.xyz, f.dir_given.xyz, f.dir_world.w, f.dir_given.w, f.from_pos.w,
    );
    // The characters over it, as the pass laid them: mirrored on the floor,
    // else their capsules (lit grey by the luminance the record kept).
    let here = to_player_space(f.from_pos.xyz);
    if (f.codes.w < 0.0) {
        col = floor_mirror_blend(
            col, fixup_mirror, fixup_mirror_depth, texel, textureLoad(probe_depth_in, texel, 0), f.codes.z,
            f.dir_world.w, textureLoad(fixup_reach, texel, 0).r,
        );
    } else {
        col = capsule_reflection(
            here, normalize(f.dir_given.xyz), f.dir_world.w, vec3<f32>(f.codes.w), col,
        );
    }
    // As `probe_env_for_pass` finishes a traced reflection: compressed and
    // premultiplied by its coverage, the brightness normalisation out of it (a
    // traced hit leaves `probe_brightness` at 0, and its scale is exactly 1),
    // a carried glass's glow added as this point sees it past the glass's own
    // beam (see `capsule_glass_beam`).
    let a = clamp(col.a, 0.0, 1.0);
    textureStore(probe_out, texel, vec4<f32>(probe_pass_compress(col.rgb * 1.0 + capsule_glow) * a, a));
}
"#;

/// MEASUREMENT ONLY: the fix-up with one kind of its work cut out -- text
/// edits of [`compute_wgsl`], each found exactly once -- run in its place by
/// the `fixup_cut` lever, so what each kind of record costs is measured on the
/// headset without a build. Lossy: the cut lookups are simply not made.
pub const FIXUP_CUTS: &[(&str, &[(&str, &str)])] = &[
    // A doorway rim's lookups: the other side of the rim, and its far end.
    ("fixup_cut_rims", &[("    let has_rim = fixups.items[k].hit.x >= 0.0;\n", "    let has_rim = false;\n")]),
    // A solid proxy's outline: what lies past it.
    (
        "fixup_cut_edges",
        &[(
            "    let has_edge = bitcast<i32>(fixups.items[k].codes.y) >= 0 && probe_edge_cover(fixups.items[k].hit.z) < 0.99;\n",
            "    let has_edge = false;\n",
        )],
    ),
    // A model's cards: the retested ray, and the four across its outline.
    ("fixup_cut_cards", &[("    let has_cards = fixup_marked(k) || fixup_recolour(k);\n", "    let has_cards = false;\n")]),
    // The characters' capsules over every record.
    (
        "fixup_cut_characters",
        &[(
            "        col = capsule_reflection(\n            here, normalize(f.dir_given.xyz), f.dir_world.w, vec3<f32>(f.codes.w), col,\n        );\n",
            "",
        )],
    ),
    // Everything past the depth test, the code kept: what launching the
    // threads and reading the records costs.
    (
        "fixup_cut_all",
        &[(
            "    pixel_footprint = fixups.items[k].origin.w;\n",
            "    if (texel0.y < 65536) {\n        return;\n    }\n    pixel_footprint = fixups.items[k].origin.w;\n",
        )],
    ),
];

/// MEASUREMENT ONLY: the cut that runs the fix-up as it was before
/// `FIXUP_ENTRY` (see `compute_wgsl_inlined`), in its place.
pub const FIXUP_INLINED: &str = "fixup_inlined";

/// [`compute_wgsl`] with the cut `cut` of [`FIXUP_CUTS`], or the fix-up before
/// `FIXUP_ENTRY` for [`FIXUP_INLINED`]; `None` when it names none, or an edit
/// no longer finds its text exactly once.
fn compute_wgsl_with_cut(cut: &str) -> Option<String> {
    if cut == FIXUP_INLINED {
        return Some(compute_wgsl_inlined());
    }
    let (_, edits) = FIXUP_CUTS.iter().find(|(label, _)| *label == cut)?;
    let mut src = compute_wgsl();
    for (from, to) in edits.iter() {
        if src.matches(from).count() != 1 {
            return None;
        }
        src = src.replacen(from, to, 1);
    }
    Some(src)
}

/// The fix-up's compute pipeline from `src`.
fn fixup_pipeline(device: &Device, layout: &wgpu::PipelineLayout, label: &str, src: String) -> ComputePipeline {
    // Audited like the probe pass it finishes: see `shader_checks`.
    let module = super::shader_checks::audited_shader_module(device, ShaderModuleDescriptor {
        label: Some(label),
        source: ShaderSource::Wgsl(src.into()),
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        module: &module,
        entry_point: Some("fixup"),
        compilation_options: Default::default(),
        cache: None,
    })
}

/// The record list, its two bindings and the fix-up pipeline. One list serves
/// both eyes: each eye's probe pass clears it, fills it, and has its fix-up
/// read it before the next eye's pass.
pub struct ProbeFixups {
    buffer: Buffer,
    capacity: u32,
    pass_layout: BindGroupLayout,
    list_bind_group: BindGroup,
    target_layout: BindGroupLayout,
    layout: wgpu::PipelineLayout,
    pipeline: ComputePipeline,
    /// MEASUREMENT: the `fixup_cut` lever's pipeline, run in `pipeline`'s place.
    cut: Option<ComputePipeline>,
    /// THE FIX-UP'S SIZE, from the list's own count: a workgroup for every 64
    /// records the pass made, not for every slot the list has room for. A
    /// thread per slot launched some 2,800 workgroups an eye to find a few
    /// hundred records, and the scene pass waits for all of them. Written by
    /// one thread (`args_pipeline`), read by `dispatch_workgroups_indirect`.
    args: Buffer,
    args_pipeline: ComputePipeline,
    args_bind_group: BindGroup,
}

/// The fix-up's indirect dispatch arguments from the list's count, clamped to
/// its capacity.
const ARGS_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> list_count: array<u32>;
@group(0) @binding(1) var<storage, read_write> args: array<u32>;
const CAPACITY: u32 = CAPACITY_VALUE;
@compute @workgroup_size(1)
fn args_main() {
    let n = min(list_count[0], CAPACITY);
    args[0] = (n + 63u) / 64u;
    args[1] = 1u;
    args[2] = 1u;
}
"#;

impl ProbeFixups {
    /// For probe pass targets of `texels` texels. ROOM FOR HALF OF THEM,
    /// about 13 MB at the shipped render scale: the most any benchmark view
    /// needed was 9% (the hallway), and a texel that does not fit keeps its
    /// primary reflection -- a softened rim drawn hard -- so the list is sized
    /// for views well past any measured, not for the average.
    pub fn new(device: &Device, uniform_layout: &BindGroupLayout, texels: u32) -> Self {
        let capacity = (texels / 2).max(64);
        let buffer = device.create_buffer(&BufferDescriptor {
            label: Some("probe_fixups"),
            size: 16 + capacity as u64 * RECORD_BYTES,
            // COPY_SRC for the offline harness's census (`list_buffer`).
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let storage = |read_only: bool, visibility: ShaderStages| BindGroupLayoutEntry {
            binding: 0,
            visibility,
            ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only }, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        let texture = |binding: u32, sample_type: TextureSampleType, visibility: ShaderStages| BindGroupLayoutEntry {
            binding,
            visibility,
            ty: BindingType::Texture { sample_type, view_dimension: TextureViewDimension::D2, multisampled: false },
            count: None,
        };
        // The list, and this eye's floor mirror and its depth.
        let pass_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("probe_fixups_pass_layout"),
            entries: &[
                storage(false, ShaderStages::FRAGMENT),
                texture(1, TextureSampleType::Float { filterable: true }, ShaderStages::FRAGMENT),
                texture(2, TextureSampleType::Depth, ShaderStages::FRAGMENT),
            ],
        });
        let list_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("probe_fixups_list_layout"),
            entries: &[storage(true, ShaderStages::COMPUTE)],
        });
        let target_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("probe_fixups_target_layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: StorageTextureAccess::WriteOnly,
                        format: probe_pass::FORMAT,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::Depth,
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                texture(2, TextureSampleType::Float { filterable: true }, ShaderStages::COMPUTE),
                texture(3, TextureSampleType::Depth, ShaderStages::COMPUTE),
                texture(4, TextureSampleType::Float { filterable: true }, ShaderStages::COMPUTE),
            ],
        });
        let bind = |layout: &BindGroupLayout, label: &str| {
            device.create_bind_group(&BindGroupDescriptor {
                label: Some(label),
                layout,
                entries: &[BindGroupEntry { binding: 0, resource: buffer.as_entire_binding() }],
            })
        };
        let list_bind_group = bind(&list_layout, "probe_fixups_list");
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("probe_fixup_layout"),
            bind_group_layouts: &[Some(uniform_layout), Some(&list_layout), Some(&target_layout)],
            immediate_size: 0,
        });
        let pipeline = fixup_pipeline(device, &layout, "probe_fixup", compute_wgsl());
        let args = device.create_buffer(&BufferDescriptor {
            label: Some("probe_fixups_args"),
            size: 16,
            usage: BufferUsages::STORAGE | BufferUsages::INDIRECT,
            mapped_at_creation: false,
        });
        let args_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("probe_fixup_args"),
            source: ShaderSource::Wgsl(ARGS_WGSL.replace("CAPACITY_VALUE", &format!("{capacity}u")).into()),
        });
        let args_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("probe_fixup_args"),
            layout: None,
            module: &args_module,
            entry_point: Some("args_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let args_bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("probe_fixup_args"),
            layout: &args_pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Buffer(wgpu::BufferBinding { buffer: &buffer, offset: 0, size: std::num::NonZeroU64::new(16) }),
                },
                BindGroupEntry { binding: 1, resource: args.as_entire_binding() },
            ],
        });
        Self {
            buffer,
            capacity,
            pass_layout,
            list_bind_group,
            target_layout,
            layout,
            pipeline,
            cut: None,
            args,
            args_pipeline,
            args_bind_group,
        }
    }

    /// MEASUREMENT: runs the fix-up with the cut `cut` of [`FIXUP_CUTS`] from
    /// now on, or as shipped for `None`. False when `cut` names no cut that
    /// still applies; the fix-up then runs as shipped.
    pub fn set_cut(&mut self, device: &Device, cut: Option<&str>) -> bool {
        self.cut = None;
        let Some(cut) = cut else {
            return true;
        };
        let Some(src) = compute_wgsl_with_cut(cut) else {
            return false;
        };
        self.cut = Some(fixup_pipeline(device, &self.layout, cut, src));
        true
    }

    /// MEASUREMENT ONLY: each of [`FIXUP_CUTS`] built once as a pipeline that
    /// nothing runs, so the driver reports its registers (`PIPESTATS`; see
    /// `shader_checks::PIPELINE_STATISTICS`).
    pub fn log_register_cuts(&self, device: &Device) {
        for label in FIXUP_CUTS.iter().map(|(label, _)| *label).chain([FIXUP_INLINED]) {
            match compute_wgsl_with_cut(label) {
                Some(src) => drop(fixup_pipeline(device, &self.layout, label, src)),
                None => log::warn!("fix-up cut {label}: an edit no longer finds its text once"),
            }
        }
    }

    /// Group 3 of the deferring probe pass: the list it appends to, and the
    /// floor mirror it lays over its floor's texels.
    /// MEASUREMENT: the record list itself -- a `u32` count, padding to 16
    /// bytes, then `RECORD_BYTES` a record -- for a census of what the
    /// fix-up is asked to do (`offline_frame`'s `FIXUP_STATS`).
    pub fn list_buffer(&self) -> &Buffer {
        &self.buffer
    }

    pub fn pass_layout(&self) -> &BindGroupLayout {
        &self.pass_layout
    }

    /// Group 3 for one single-eye probe pass target: the list, and that
    /// target's floor mirror and its depth.
    pub fn pass_bind_group_for(&self, device: &Device, target: &probe_pass::Target) -> BindGroup {
        device.create_bind_group(&BindGroupDescriptor {
            label: Some("probe_fixups_pass"),
            layout: &self.pass_layout,
            entries: &[
                BindGroupEntry { binding: 0, resource: self.buffer.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: BindingResource::TextureView(&target.mirror_view) },
                BindGroupEntry { binding: 2, resource: BindingResource::TextureView(&target.mirror_depth_views[0]) },
            ],
        })
    }

    /// What the fix-up writes and checks for one single-eye probe pass target:
    /// its colour, the depth it was drawn at, and its floor mirror, the
    /// mirror's depth and the pass's reach.
    pub fn target_bind_group(&self, device: &Device, target: &probe_pass::Target) -> BindGroup {
        device.create_bind_group(&BindGroupDescriptor {
            label: Some("probe_fixups_target"),
            layout: &self.target_layout,
            entries: &[
                BindGroupEntry { binding: 0, resource: BindingResource::TextureView(&target.color_view) },
                BindGroupEntry { binding: 1, resource: BindingResource::TextureView(&target.depth_view) },
                BindGroupEntry { binding: 2, resource: BindingResource::TextureView(&target.mirror_view) },
                BindGroupEntry { binding: 3, resource: BindingResource::TextureView(&target.mirror_depth_views[0]) },
                BindGroupEntry { binding: 4, resource: BindingResource::TextureView(&target.reach_single) },
            ],
        })
    }

    /// Empties the list: before each probe pass that appends to it.
    pub fn clear(&self, encoder: &mut CommandEncoder) {
        encoder.clear_buffer(&self.buffer, 0, Some(16));
    }

    /// Makes the recorded lookups and writes their texels into `target`'s
    /// colour: after the probe pass, before anything reads it. A thread per
    /// slot of the list; those past its count return at once.
    /// `timestamp_writes`: the pass timer's `fix_l`/`fix_r` slot, so what the
    /// fix-up costs is measured rather than hidden between the passes that are.
    pub fn dispatch(
        &self,
        encoder: &mut CommandEncoder,
        uniforms: &BindGroup,
        target: &BindGroup,
        timestamp_writes: Option<wgpu::ComputePassTimestampWrites<'_>>,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("probe_fixup"), timestamp_writes });
        // As many workgroups as the pass made records: see `args`.
        pass.set_pipeline(&self.args_pipeline);
        pass.set_bind_group(0, &self.args_bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
        pass.set_pipeline(self.cut.as_ref().unwrap_or(&self.pipeline));
        pass.set_bind_group(0, uniforms, &[]);
        pass.set_bind_group(1, &self.list_bind_group, &[]);
        pass.set_bind_group(2, target, &[]);
        pass.dispatch_workgroups_indirect(&self.args, 0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The compute shader parses and validates, and a record is the size the
    /// buffer is laid out for.
    #[test]
    fn the_fixup_shader_validates_and_a_record_is_128_bytes() {
        use wgpu::naga;
        let src = compute_wgsl();
        let module = naga::front::wgsl::parse_str(&src).unwrap_or_else(|e| panic!("{}", e.emit_to_string(&src)));
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
            .validate(&module)
            .unwrap_or_else(|e| panic!("{e:?}"));
        let ty = module
            .types
            .iter()
            .find(|(_, t)| t.name.as_deref() == Some("ProbeFixup"))
            .expect("the record type")
            .1;
        let size = ty.inner.size(module.to_ctx());
        assert_eq!(u64::from(size), RECORD_BYTES);
    }

    /// Every measurement cut of the fix-up finds its text exactly once -- a
    /// cut that no longer applies would run the shipped fix-up under its name
    /// -- and the result is valid WGSL.
    #[test]
    fn every_fixup_cut_applies_and_validates() {
        use wgpu::naga;
        for label in FIXUP_CUTS.iter().map(|(label, _)| *label).chain([FIXUP_INLINED]) {
            let src = compute_wgsl_with_cut(label).unwrap_or_else(|| panic!("{label}: an edit does not find its text once"));
            let module = naga::front::wgsl::parse_str(&src).unwrap_or_else(|e| panic!("{label}: {}", e.emit_to_string(&src)));
            naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
                .validate(&module)
                .unwrap_or_else(|e| panic!("{label}: {e:?}"));
        }
    }
}
