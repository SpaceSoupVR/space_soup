use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
    BindingResource, BindingType, Buffer, BufferBindingType, BufferDescriptor, BufferUsages, CommandEncoder,
    ComputePipeline, Device, ShaderModuleDescriptor, ShaderSource, ShaderStages, StorageTextureAccess,
    TextureSampleType, TextureViewDimension,
};

use super::brush_pipeline::probe_pass;

pub const RECORD_WGSL: &str = r#"
struct ProbeFixup {

    texel: vec4<u32>,

    col: vec4<f32>,

    from_pos: vec4<f32>,

    dir_world: vec4<f32>,

    dir_given: vec4<f32>,

    origin: vec4<f32>,

    hit: vec4<f32>,

    codes: vec4<f32>,
}
"#;

pub const RECORD_BYTES: u64 = 128;

pub fn lights_block_wgsl(deferred: bool) -> String {
    if !deferred {
        return r#"
fn probe_fixup_begin(h: ProbeHit, world_pos: vec3<f32>, d: vec3<f32>, dir: vec3<f32>, roughness: f32, probe_lod: f32, trace_room: f32, recolour: bool) -> i32 {
    return -1;
}
fn probe_fixup_finish(slot: i32, col: vec4<f32>) {
}

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

@group(3) @binding(1) var probe_fixup_mirror: texture_2d<f32>;
@group(3) @binding(2) var probe_fixup_mirror_depth: texture_depth_2d;

fn probe_floor_mirror_pass(col: vec4<f32>, roughness: f32) -> vec4<f32> {
    return floor_mirror_blend(
        col, probe_fixup_mirror, probe_fixup_mirror_depth, vec2<i32>(probe_fragment.xy), probe_fragment.z,
        probe_eye_distance, roughness, probe_reach,
    );
}

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

        let lit = dot(probe_capsule_lit, vec3<f32>(0.2126, 0.7152, 0.0722));
        probe_fixups.items[k].codes = vec4<f32>(
            bitcast<f32>(h.rim_code), bitcast<f32>(h.edge_code), probe_eye_distance, select(lit, -1.0 - lit, probe_floor_mirror_here),
        );
    }
    return slot;
}

fn probe_fixup_finish(slot: i32, col: vec4<f32>) {
    if (slot >= 0) {
        probe_fixups.items[slot].col = col;
    }
}
"#
    )
}

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

const DEPTH_TOLERANCE: f32 = 1e-6;

fn compute_wgsl() -> String {
    compute_wgsl_with_entry(FIXUP_ENTRY)
}

fn compute_wgsl_inlined() -> String {
    compute_wgsl_with_entry(FIXUP_ENTRY_INLINED)
}

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

@group(2) @binding(2) var fixup_mirror: texture_2d<f32>;
@group(2) @binding(3) var fixup_mirror_depth: texture_depth_2d;
@group(2) @binding(4) var fixup_reach: texture_2d<f32>;
{blend}

const FIXUP_DEPTH_TOLERANCE: f32 = {tolerance:?};

{entry}
"#,

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

const FIXUP_ENTRY: &str = r#"
const FIXUP_RIM: u32 = 4u;
const FIXUP_FAR: u32 = 5u;
const FIXUP_EDGE: u32 = 6u;
const FIXUP_DONE: u32 = 7u;

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

fn fixup_marked(k: u32) -> bool {
    return bitcast<i32>(fixups.items[k].codes.y) >= 0 && fixups.items[k].hit.z < 0.0;
}

fn fixup_recolour(k: u32) -> bool {
    return fixups.items[k].texel.w != 0u && !fixup_marked(k);
}

fn fixup_retest(k: u32) -> bool {
    return fixup_recolour(k) || fixups.items[k].hit.z < 0.5 * (PROBE_SUBSAMPLE + PROBE_RETEST);
}

fn fixup_after_rim(k: u32) -> u32 {
    let has_edge = bitcast<i32>(fixups.items[k].codes.y) >= 0 && probe_edge_cover(fixups.items[k].hit.z) < 0.99;
    return select(FIXUP_DONE, FIXUP_EDGE, has_edge);
}

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

    let texel0 = vec2<i32>(fixups.items[k].texel.xy);
    if (abs(textureLoad(probe_depth_in, texel0, 0) - bitcast<f32>(fixups.items[k].texel.z)) > FIXUP_DEPTH_TOLERANCE) {
        return;
    }
    pixel_footprint = fixups.items[k].origin.w;
    probe_eye_distance = fixups.items[k].codes.z;
    let has_cards = fixup_marked(k) || fixup_recolour(k);

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

    let a = clamp(col.a, 0.0, 1.0);
    textureStore(probe_out, texel, vec4<f32>(probe_pass_compress(col.rgb * 1.0 + capsule_glow) * a, a));
}
"#;

const FIXUP_ENTRY_INLINED: &str = r#"
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

    let a = clamp(col.a, 0.0, 1.0);
    textureStore(probe_out, texel, vec4<f32>(probe_pass_compress(col.rgb * 1.0 + capsule_glow) * a, a));
}
"#;

pub const FIXUP_CUTS: &[(&str, &[(&str, &str)])] = &[

    ("fixup_cut_rims", &[("    let has_rim = fixups.items[k].hit.x >= 0.0;\n", "    let has_rim = false;\n")]),

    (
        "fixup_cut_edges",
        &[(
            "    let has_edge = bitcast<i32>(fixups.items[k].codes.y) >= 0 && probe_edge_cover(fixups.items[k].hit.z) < 0.99;\n",
            "    let has_edge = false;\n",
        )],
    ),

    ("fixup_cut_cards", &[("    let has_cards = fixup_marked(k) || fixup_recolour(k);\n", "    let has_cards = false;\n")]),

    (
        "fixup_cut_characters",
        &[(
            "        col = capsule_reflection(\n            here, normalize(f.dir_given.xyz), f.dir_world.w, vec3<f32>(f.codes.w), col,\n        );\n",
            "",
        )],
    ),

    (
        "fixup_cut_all",
        &[(
            "    pixel_footprint = fixups.items[k].origin.w;\n",
            "    if (texel0.y < 65536) {\n        return;\n    }\n    pixel_footprint = fixups.items[k].origin.w;\n",
        )],
    ),
];

pub const FIXUP_INLINED: &str = "fixup_inlined";

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

fn fixup_pipeline(device: &Device, layout: &wgpu::PipelineLayout, label: &str, src: String) -> ComputePipeline {

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

pub struct ProbeFixups {
    buffer: Buffer,
    capacity: u32,
    pass_layout: BindGroupLayout,
    list_bind_group: BindGroup,
    target_layout: BindGroupLayout,
    layout: wgpu::PipelineLayout,
    pipeline: ComputePipeline,

    cut: Option<ComputePipeline>,

    args: Buffer,
    args_pipeline: ComputePipeline,
    args_bind_group: BindGroup,
}

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

    pub fn new(device: &Device, uniform_layout: &BindGroupLayout, texels: u32) -> Self {
        let capacity = (texels / 2).max(64);
        let buffer = device.create_buffer(&BufferDescriptor {
            label: Some("probe_fixups"),
            size: 16 + capacity as u64 * RECORD_BYTES,

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

    pub fn log_register_cuts(&self, device: &Device) {
        for label in FIXUP_CUTS.iter().map(|(label, _)| *label).chain([FIXUP_INLINED]) {
            match compute_wgsl_with_cut(label) {
                Some(src) => drop(fixup_pipeline(device, &self.layout, label, src)),
                None => log::warn!("fix-up cut {label}: an edit no longer finds its text once"),
            }
        }
    }

    pub fn list_buffer(&self) -> &Buffer {
        &self.buffer
    }

    pub fn pass_layout(&self) -> &BindGroupLayout {
        &self.pass_layout
    }

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

    pub fn clear(&self, encoder: &mut CommandEncoder) {
        encoder.clear_buffer(&self.buffer, 0, Some(16));
    }

    pub fn dispatch(
        &self,
        encoder: &mut CommandEncoder,
        uniforms: &BindGroup,
        target: &BindGroup,
        timestamp_writes: Option<wgpu::ComputePassTimestampWrites<'_>>,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("probe_fixup"), timestamp_writes });

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
