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
    // x, y: the texel; z: the depth its fragment was at, as bits.
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
fn probe_fixup_begin(h: ProbeHit, world_pos: vec3<f32>, d: vec3<f32>, dir: vec3<f32>, roughness: f32, probe_lod: f32, trace_room: f32) -> i32 {
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
// does not happen; the texel would keep its primary reflection.
fn probe_fixup_begin(h: ProbeHit, world_pos: vec3<f32>, d: vec3<f32>, dir: vec3<f32>, roughness: f32, probe_lod: f32, trace_room: f32) -> i32 {
    let k = atomicAdd(&probe_fixups.count, 1u);
    var slot = -1;
    if (k < arrayLength(&probe_fixups.items)) {
        slot = i32(k);
        probe_fixups.items[k].texel = vec4<u32>(vec2<u32>(probe_fragment.xy), bitcast<u32>(probe_fragment.z), 0u);
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

// A TEXEL ACROSS ONE OF A MODEL'S OWN OUTLINES, traced again: four rays in a
// rotated grid across what the texel's reflection covers, each coloured as the
// pass colours a hit, averaged. See `PROBE_SUBSAMPLE`. Spread over the edge's
// softening width -- `PROBE_EDGE_FOOTPRINTS` of the pixel each side, or the
// lobe where a rough surface's is wider -- as every other reflected edge is,
// so the scene's bilinear read of this half-resolution pass does not snap the
// outline to its texels. A ray that finds nothing keeps the texel's own
// colour. The grid is turned a quarter each ray rather than read from an
// array, which Adreno puts in scratch memory when a loop indexes it.
fn probe_subsample(primary: vec4<f32>, world_pos: vec3<f32>, d: vec3<f32>, dir: vec3<f32>, roughness: f32, probe_lod: f32, trace_room: f32) -> vec4<f32> {{
    let spread = 2.0 * max(probe_lobe_tan(roughness), PROBE_EDGE_FOOTPRINTS * pixel_footprint / max(probe_eye_distance, 0.05));
    let a = normalize(cross(d, select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(d.y) > 0.9)));
    let b = cross(d, a);
    var o = vec2<f32>(0.125, 0.375);
    var sum = vec4<f32>(0.0);
    for (var k = 0; k < 4; k = k + 1) {{
        let dk = normalize(d + (a * o.x + b * o.y) * spread);
        let h = probe_trace(world_pos, dk, trace_room, roughness);
        var c = primary;
        if (h.found) {{
            c = probe_traced_colour(h, dk, roughness, dir, probe_lod);
        }}
        sum = sum + c;
        o = vec2<f32>(-o.y, o.x);
    }}
    return 0.25 * sum;
}}

@compute @workgroup_size(64)
fn fixup(@builtin(global_invocation_id) id: vec3<u32>) {{
    if (id.x >= min(fixups.count, arrayLength(&fixups.items))) {{
        return;
    }}
    let f = fixups.items[id.x];
    let texel = vec2<i32>(f.texel.xy);
    // Recorded by a fragment a nearer one of the same pass then covered.
    if (abs(textureLoad(probe_depth_in, texel, 0) - bitcast<f32>(f.texel.z)) > FIXUP_DEPTH_TOLERANCE) {{
        return;
    }}
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
    if (hit.edge_code >= 0 && hit.edge_cover < 0.0) {{
        primary = probe_subsample(f.col, f.from_pos.xyz, f.dir_world.xyz, f.dir_given.xyz, f.dir_world.w, f.dir_given.w, f.from_pos.w);
    }}
    var col = probe_secondary(
        hit, primary, f.from_pos.xyz, f.dir_world.xyz, f.dir_given.xyz, f.dir_world.w, f.dir_given.w, f.from_pos.w,
    );
    // The characters over it, as the pass laid them: mirrored on the floor,
    // else their capsules (lit grey by the luminance the record kept).
    if (f.codes.w < 0.0) {{
        col = floor_mirror_blend(
            col, fixup_mirror, fixup_mirror_depth, texel, textureLoad(probe_depth_in, texel, 0), f.codes.z,
            f.dir_world.w, textureLoad(fixup_reach, texel, 0).r,
        );
    }} else {{
        col = capsule_reflection(
            to_player_space(f.from_pos.xyz), normalize(f.dir_given.xyz), f.dir_world.w, vec3<f32>(f.codes.w), col,
        );
    }}
    // As `probe_env_for_pass` finishes a traced reflection: compressed and
    // premultiplied by its coverage, the brightness normalisation out of it (a
    // traced hit leaves `probe_brightness` at 0, and its scale is exactly 1).
    let a = clamp(col.a, 0.0, 1.0);
    textureStore(probe_out, texel, vec4<f32>(probe_pass_compress(col.rgb * 1.0) * a, a));
}}
"#,
        lights = super::lights::wgsl_lights_block(0, 1),
        record = RECORD_WGSL,
        tolerance = DEPTH_TOLERANCE,
        blend = floor_mirror_blend_wgsl(),
    )
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
    pipeline: ComputePipeline,
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
        // Audited like the probe pass it finishes: see `shader_checks`.
        let module = super::shader_checks::audited_shader_module(device, ShaderModuleDescriptor {
            label: Some("probe_fixup"),
            source: ShaderSource::Wgsl(compute_wgsl().into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("probe_fixup_layout"),
            bind_group_layouts: &[Some(uniform_layout), Some(&list_layout), Some(&target_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("probe_fixup"),
            layout: Some(&layout),
            module: &module,
            entry_point: Some("fixup"),
            compilation_options: Default::default(),
            cache: None,
        });
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
        Self { buffer, capacity, pass_layout, list_bind_group, target_layout, pipeline, args, args_pipeline, args_bind_group }
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
        pass.set_pipeline(&self.pipeline);
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
}
