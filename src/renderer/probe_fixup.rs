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
"#
        .to_string();
    }
    format!(
        "{RECORD_WGSL}{}",
        r#"
struct ProbeFixups {
    count: atomic<u32>,
    items: array<ProbeFixup>,
}
@group(3) @binding(0) var<storage, read_write> probe_fixups: ProbeFixups;

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
        probe_fixups.items[k].codes = vec4<f32>(bitcast<f32>(h.rim_code), bitcast<f32>(h.edge_code), probe_eye_distance, 0.0);
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
    let col = probe_secondary(
        hit, primary, f.from_pos.xyz, f.dir_world.xyz, f.dir_given.xyz, f.dir_world.w, f.dir_given.w, f.from_pos.w,
    );
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
    )
}

/// The record list, its two bindings and the fix-up pipeline. One list serves
/// both eyes: each eye's probe pass clears it, fills it, and has its fix-up
/// read it before the next eye's pass.
pub struct ProbeFixups {
    buffer: Buffer,
    capacity: u32,
    pass_layout: BindGroupLayout,
    pass_bind_group: BindGroup,
    list_bind_group: BindGroup,
    target_layout: BindGroupLayout,
    pipeline: ComputePipeline,
}

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
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let storage = |read_only: bool, visibility: ShaderStages| BindGroupLayoutEntry {
            binding: 0,
            visibility,
            ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only }, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        let pass_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("probe_fixups_pass_layout"),
            entries: &[storage(false, ShaderStages::FRAGMENT)],
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
            ],
        });
        let bind = |layout: &BindGroupLayout, label: &str| {
            device.create_bind_group(&BindGroupDescriptor {
                label: Some(label),
                layout,
                entries: &[BindGroupEntry { binding: 0, resource: buffer.as_entire_binding() }],
            })
        };
        let pass_bind_group = bind(&pass_layout, "probe_fixups_pass");
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
        Self { buffer, capacity, pass_layout, pass_bind_group, list_bind_group, target_layout, pipeline }
    }

    /// Group 3 of the deferring probe pass: the list it appends to.
    pub fn pass_layout(&self) -> &BindGroupLayout {
        &self.pass_layout
    }

    pub fn pass_bind_group(&self) -> &BindGroup {
        &self.pass_bind_group
    }

    /// What the fix-up writes and checks for one single-eye probe pass target:
    /// its colour, and the depth it was drawn at.
    pub fn target_bind_group(&self, device: &Device, target: &probe_pass::Target) -> BindGroup {
        device.create_bind_group(&BindGroupDescriptor {
            label: Some("probe_fixups_target"),
            layout: &self.target_layout,
            entries: &[
                BindGroupEntry { binding: 0, resource: BindingResource::TextureView(&target.color_view) },
                BindGroupEntry { binding: 1, resource: BindingResource::TextureView(&target.depth_view) },
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
    pub fn dispatch(&self, encoder: &mut CommandEncoder, uniforms: &BindGroup, target: &BindGroup) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("probe_fixup"), timestamp_writes: None });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, uniforms, &[]);
        pass.set_bind_group(1, &self.list_bind_group, &[]);
        pass.set_bind_group(2, target, &[]);
        pass.dispatch_workgroups(self.capacity.div_ceil(64), 1, 1);
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
