use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use wgpu::*;

use super::Color3;

pub const MAX_LIGHTS: usize = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LightKind {
    Point,
    Spot,

    Directional,
}

pub fn influence_score(light: &Light) -> f32 {
    if light.kind == LightKind::Directional {
        return f32::INFINITY;
    }
    let d2 = light.position.length_squared();

    light.intensity.max(0.0) / (1.0 + d2)
}

pub const SHADOW_SLOT_MARGIN: f32 = 1.5;

pub fn spot_shadow_slots(
    scores: &[(usize, f32)],
    incumbents: &[usize],
    max: usize,
    margin: f32,
) -> Vec<usize> {
    use std::cmp::Ordering;
    let better = |a: &(usize, f32), b: &(usize, f32)| {
        b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal).then(a.0.cmp(&b.0))
    };
    let score_of = |id: usize| scores.iter().find(|(j, _)| *j == id).map(|(_, s)| *s);

    let mut out: Vec<(usize, f32)> = incumbents
        .iter()
        .filter_map(|&id| score_of(id).map(|s| (id, s)))
        .collect();
    out.sort_by(better);
    out.dedup_by_key(|(id, _)| *id);
    out.truncate(max);

    let mut challengers: Vec<(usize, f32)> = scores
        .iter()
        .copied()
        .filter(|(id, _)| !out.iter().any(|(h, _)| h == id))
        .collect();
    challengers.sort_by(better);

    let mut rest = Vec::new();
    for c in challengers.drain(..) {
        if out.len() < max {
            out.push(c);
        } else {
            rest.push(c);
        }
    }

    for c in rest {
        let Some((pos, weakest)) = out
            .iter()
            .copied()
            .enumerate()
            .min_by(|a, b| a.1 .1.partial_cmp(&b.1 .1).unwrap_or(Ordering::Equal))
        else {
            break;
        };
        if c.1 > weakest.1 * margin {
            out[pos] = c;
        }
    }

    position_stable_slots(incumbents, out)
}

fn position_stable_slots(incumbents: &[usize], chosen: Vec<(usize, f32)>) -> Vec<usize> {
    let ids: Vec<usize> = chosen.into_iter().map(|(id, _)| id).collect();
    let mut slots: Vec<Option<usize>> = vec![None; ids.len()];

    for (old_slot, id) in incumbents.iter().enumerate() {
        if old_slot < slots.len() && ids.contains(id) && !slots.contains(&Some(*id)) {
            slots[old_slot] = Some(*id);
        }
    }

    let free: Vec<usize> =
        slots.iter().enumerate().filter(|(_, s)| s.is_none()).map(|(i, _)| i).collect();
    let mut free = free.into_iter();
    for id in ids.iter().copied() {
        if slots.contains(&Some(id)) {
            continue;
        }
        if let Some(i) = free.next() {
            slots[i] = Some(id);
        }
    }
    slots.into_iter().flatten().collect()
}

#[cfg(test)]
mod shadow_slot_stability_tests {
    use super::spot_shadow_slots;

    const MAX: usize = 4;
    const MARGIN: f32 = 1.5;

    #[test]
    fn the_same_lights_keep_the_same_slots_when_only_their_ranking_moves() {

        let first = spot_shadow_slots(
            &[(0, 10.0), (1, 9.0), (2, 8.0), (3, 7.0)],
            &[],
            MAX,
            MARGIN,
        );
        assert_eq!(first.len(), 4);

        let second = spot_shadow_slots(
            &[(0, 7.0), (1, 8.0), (2, 9.0), (3, 10.0)],
            &first,
            MAX,
            MARGIN,
        );
        assert_eq!(
            second, first,
            "the set is unchanged but the slots moved; every reorder rewrites \
             a shadow atlas tile for no reason",
        );
    }

    #[test]
    fn an_eviction_disturbs_one_slot_and_leaves_the_others_alone() {
        let before = spot_shadow_slots(
            &[(0, 10.0), (1, 9.0), (2, 8.0), (3, 7.0)],
            &[],
            MAX,
            MARGIN,
        );

        let after = spot_shadow_slots(
            &[(0, 10.0), (1, 9.0), (2, 8.0), (3, 7.0), (4, 100.0)],
            &before,
            MAX,
            MARGIN,
        );
        assert!(after.contains(&4), "a light far past the margin never got in");
        assert!(!after.contains(&3), "the weakest incumbent was not the one evicted");
        let moved: Vec<usize> = (0..MAX).filter(|&i| before[i] != after[i]).collect();
        assert_eq!(
            moved.len(),
            1,
            "one eviction should rewrite ONE tile; it changed slots {moved:?}",
        );
    }

    #[test]
    fn a_challenger_inside_the_margin_still_changes_nothing() {
        let before =
            spot_shadow_slots(&[(0, 10.0), (1, 9.0), (2, 8.0), (3, 7.0)], &[], MAX, MARGIN);
        let after = spot_shadow_slots(
            &[(0, 10.0), (1, 9.0), (2, 8.0), (3, 7.0), (4, 8.0)],
            &before,
            MAX,
            MARGIN,
        );
        assert_eq!(after, before, "a marginal challenger displaced an incumbent");
    }
}

pub const PROBE_PORTAL_FADE: f32 = 0.5;

pub const PROBE_PORTAL_SIDE_FADE: f32 = 0.5;

pub fn probe_box_margin() -> f32 {
    0.05 / crate::renderer::RENDER_SCALE.clamp(0.3, 1.0)
}

pub fn rank_for_budget(lights: &[Light], max: usize) -> Vec<Light> {
    rank_for_budget_indices(lights, max)
        .into_iter()
        .map(|i| lights[i])
        .collect()
}

pub fn sky_sun_light(
    sun: Option<&crate::renderer::sky::SkySun>,
    lights: &[Light],
    world_to_player: glam::Quat,
) -> Option<Light> {
    let sun = sun?;
    if lights.iter().any(|l| l.kind == LightKind::Directional) {
        return None;
    }
    let peak = sun.light_rgb.iter().copied().fold(0.0f32, f32::max);
    if peak <= 0.0 {
        return None;
    }

    let srgb = |v: f32| {
        let v = (v / peak).clamp(0.0, 1.0);
        let e = if v <= 0.003_130_8 { v * 12.92 } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 };
        (e * 255.0).round() as u8
    };
    Some(Light {
        mask_channel: None,
        shadow_near: None,
        source_radius: 0.0,
        in_level_bake: true,
        position: Vec3::ZERO,

        direction: world_to_player * -Vec3::from(sun.direction),
        kind: LightKind::Directional,
        color: Color3(srgb(sun.light_rgb[0]), srgb(sun.light_rgb[1]), srgb(sun.light_rgb[2]), 255),
        intensity: peak,
        range: 0.0,
        cone_angle_deg: 0.0,
        inner_cone_angle_deg: 0.0,
    })
}

pub fn dynamic_sun_matrix(world_dir: Vec3, head_world: Vec3) -> glam::Mat4 {
    use crate::renderer::shadow::{directional_light_matrix, SUN_DYNAMIC_DIM, SUN_DYNAMIC_RADIUS};
    let centre = head_world - Vec3::Y * 0.9;
    let m = directional_light_matrix(world_dir, centre, SUN_DYNAMIC_RADIUS);
    let half = SUN_DYNAMIC_DIM as f32 * 0.5;
    let o = m.project_point3(Vec3::ZERO);
    let snap = |v: f32| (v * half).round() / half - v;
    glam::Mat4::from_translation(Vec3::new(snap(o.x), snap(o.y), 0.0)) * m
}

pub struct StaticSunShadow {

    pub signature: ((usize, usize), (usize, usize), [u32; 3]),

    pub world_view_proj: glam::Mat4,
}

pub const STATIC_SUN_MARGIN: f32 = 0.5;

pub fn static_sun_matrix(
    world_dir: Vec3,
    brushes: Option<impl Iterator<Item = [f32; 3]>>,
    ground: Option<impl Iterator<Item = [f32; 3]>>,
    player_to_world: glam::Mat4,
) -> glam::Mat4 {
    let bounds = |points: &mut dyn Iterator<Item = [f32; 3]>| {
        let mut lo = Vec3::splat(f32::INFINITY);
        let mut hi = Vec3::splat(f32::NEG_INFINITY);
        for p in points {
            let w = player_to_world.transform_point3(Vec3::from(p));
            lo = lo.min(w);
            hi = hi.max(w);
        }
        (lo.x <= hi.x).then_some((lo, hi))
    };
    let found = brushes
        .and_then(|mut b| bounds(&mut b))
        .or_else(|| ground.and_then(|mut g| bounds(&mut g)));
    let (lo, hi) = found.unwrap_or((Vec3::splat(-10.0), Vec3::splat(10.0)));
    let centre = (lo + hi) * 0.5;
    let radius = (hi - lo).length() * 0.5 * (1.0 + STATIC_SUN_MARGIN) + 1.0;
    crate::renderer::shadow::directional_light_matrix(world_dir, centre, radius)
}

pub fn rank_for_budget_indices(lights: &[Light], max: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..lights.len()).collect();
    if lights.len() > max {
        idx.sort_by(|&a, &b| {
            influence_score(&lights[b])
                .partial_cmp(&influence_score(&lights[a]))
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });
        idx.truncate(max);
    }

    idx.sort_by_key(|&i| !lights[i].is_surface_light());
    idx
}

#[derive(Debug, Clone, Copy)]
pub struct Light {
    pub position: Vec3,

    pub direction: Vec3,
    pub kind: LightKind,
    pub color: Color3,
    pub intensity: f32,
    pub range: f32,
    pub cone_angle_deg: f32,

    pub inner_cone_angle_deg: f32,

    pub mask_channel: Option<u8>,

    pub shadow_near: Option<f32>,

    pub source_radius: f32,

    pub in_level_bake: bool,
}

impl Light {

    pub fn casts_shadow(&self) -> bool {
        self.shadow_near.is_none_or(|near| near < self.range)
    }

    pub fn is_surface_light(&self) -> bool {
        !self.casts_shadow()
            && match self.kind {
                LightKind::Point => true,
                LightKind::Spot => {
                    let (cos_outer, cos_inner) = self.cone_cosines();
                    cos_inner - cos_outer > SURFACE_LIGHT_MIN_BAND
                }
                LightKind::Directional => false,
            }
    }

    pub fn cone_cosines(&self) -> (f32, f32) {
        let cos_outer = (self.cone_angle_deg.to_radians() * 0.5).cos();

        let cos_inner = (self.inner_cone_angle_deg.to_radians() * 0.5)
            .cos()
            .max(cos_outer + 1e-4);
        (cos_outer, cos_inner)
    }
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GpuLight {
    position: [f32; 4],
    direction: [f32; 4],
    color_intensity: [f32; 4],

    params: [f32; 4],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LitSurface {
    pub normal: Vec3,
    pub offset: f32,
    pub albedo: Vec3,
    pub lens: Vec3,
    pub forward: Vec3,
    pub right: Vec3,
    pub tan_half: f32,
    pub near: f32,
    pub far: f32,
}

pub const MAX_LIT_SURFACES: usize = 6;

pub const LIT_SURFACE_VEC4S: usize = 5;

fn pack_lit_surfaces(surfaces: &[LitSurface], pool_row: Option<u32>) -> [[f32; 4]; LIT_SURFACE_VEC4S * MAX_LIT_SURFACES] {
    let mut out = [[0.0; 4]; LIT_SURFACE_VEC4S * MAX_LIT_SURFACES];
    let Some(row) = pool_row else {
        return out;
    };
    for (k, s) in surfaces.iter().filter(|s| s.tan_half > 0.0).take(MAX_LIT_SURFACES).enumerate() {
        let at = LIT_SURFACE_VEC4S * k;
        out[at] = [s.normal.x, s.normal.y, s.normal.z, s.offset];
        out[at + 1] = [s.lens.x, s.lens.y, s.lens.z, s.tan_half];
        out[at + 2] = [s.forward.x, s.forward.y, s.forward.z, row as f32];
        out[at + 3] = [s.right.x, s.right.y, s.right.z, s.near];
        out[at + 4] = [s.albedo.x, s.albedo.y, s.albedo.z, s.far];
    }
    out
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GpuLights {

    count: [u32; 4],

    surface_lights: [u32; 4],
    lights: [GpuLight; MAX_LIGHTS],

    surfaces: [[f32; 4]; LIT_SURFACE_VEC4S * MAX_LIT_SURFACES],
}

pub struct LightsUniform {
    buffer: Buffer,

    culling: std::cell::Cell<bool>,

    terminator_aa: std::cell::Cell<bool>,

    surface_lights_apart: std::cell::Cell<bool>,

    surfaces: std::cell::Cell<[Option<LitSurface>; MAX_LIT_SURFACES]>,

    pool_row: std::cell::Cell<Option<u32>>,

    tile_lamps: std::cell::Cell<[usize; super::shadow::MAX_CHARACTER_SHADOWS]>,

    pool_light: std::cell::RefCell<Vec<(Vec3, f32, f32)>>,
}

impl LightsUniform {
    pub fn new(device: &Device) -> Self {
        let buffer = device.create_buffer(&BufferDescriptor {
            label: Some("lights_uniform"),
            size: std::mem::size_of::<GpuLights>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            buffer,
            culling: std::cell::Cell::new(true),
            terminator_aa: std::cell::Cell::new(true),
            surface_lights_apart: std::cell::Cell::new(true),
            surfaces: std::cell::Cell::new([None; MAX_LIT_SURFACES]),
            pool_row: std::cell::Cell::new(None),
            tile_lamps: std::cell::Cell::new([usize::MAX; super::shadow::MAX_CHARACTER_SHADOWS]),
            pool_light: std::cell::RefCell::new(Vec::new()),
        }
    }

    pub fn set_pool_light(&self, samples: &[(Vec3, f32, f32)]) {
        let mut held = self.pool_light.borrow_mut();
        held.clear();
        held.extend_from_slice(samples);
    }

    pub fn pool_light(&self) -> Vec<(Vec3, f32, f32)> {
        self.pool_light.borrow().clone()
    }

    pub fn set_tile_lamps(&self, lamps: &[usize]) {
        let mut held = [usize::MAX; super::shadow::MAX_CHARACTER_SHADOWS];
        for (slot, &l) in held.iter_mut().zip(lamps) {
            *slot = l;
        }
        self.tile_lamps.set(held);
    }

    pub fn set_lit_surfaces(&self, surfaces: &[LitSurface]) {
        let mut held = [None; MAX_LIT_SURFACES];
        for (slot, s) in held.iter_mut().zip(surfaces.iter().filter(|s| s.tan_half > 0.0)) {
            *slot = Some(*s);
        }
        self.surfaces.set(held);
    }

    pub fn lit_surfaces(&self) -> Vec<LitSurface> {
        self.surfaces.get().iter().flatten().copied().collect()
    }

    pub fn set_pool_row(&self, row: Option<u32>) {
        self.pool_row.set(row);
    }

    pub fn reads_pool_maps(&self) -> bool {
        self.pool_row.get().is_some() && self.surfaces.get().iter().any(Option::is_some)
    }

    pub fn set_culling(&self, on: bool) {
        self.culling.set(on);
    }

    pub fn set_terminator_aa(&self, on: bool) {
        self.terminator_aa.set(on);
    }

    pub fn set_surface_lights_apart(&self, on: bool) {
        self.surface_lights_apart.set(on);
    }

    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    pub fn upload(&self, queue: &Queue, lights: &[Light]) {
        self.upload_with_shadow_layers(queue, lights, &[]);
    }

    pub fn upload_with_shadow_layers(
        &self,
        queue: &Queue,
        lights: &[Light],
        spot_layers: &[usize],
    ) {
        self.upload_frame(queue, lights, spot_layers, false);
    }

    pub fn upload_frame(
        &self,
        queue: &Queue,
        lights: &[Light],
        spot_layers: &[usize],
        sun_is_baked: bool,
    ) {
        self.upload_frame_split(queue, lights, lights.len(), spot_layers, sun_is_baked);
    }

    pub fn upload_frame_split(
        &self,
        queue: &Queue,
        lights: &[Light],
        live: usize,
        spot_layers: &[usize],
        sun_is_baked: bool,
    ) {
        let mut gpu = pack_lights(lights, live, spot_layers, sun_is_baked, self.culling.get());
        for (k, &l) in self.tile_lamps.get().iter().enumerate() {

            if let Some(slot) = gpu.lights.get_mut(l).filter(|s| s.params[3] == -1.0) {
                slot.params[3] = (super::shadow::MAX_SPOT_SHADOWS + k) as f32;
            }
        }
        gpu.count[3] = u32::from(!self.terminator_aa.get());
        if !self.surface_lights_apart.get() {
            gpu.surface_lights[0] = 0;
        }
        gpu.surfaces = pack_lit_surfaces(&self.lit_surfaces(), self.pool_row.get());
        queue.write_buffer(&self.buffer, 0, bytemuck::bytes_of(&gpu));
    }
}

pub(crate) const SURFACE_RELIT_READ: &str = "    let relit = probe_surface_relit(h, d, roughness, t);\n";
pub(crate) const SURFACE_RELIT_CALL: &str = "    if (!model) {\n        col = vec4<f32>(col.rgb + relit, col.a);\n    }\n";

pub const SURFACE_LIGHT: f32 = -2.0;

pub const SURFACE_LIGHT_MIN_BAND: f32 = 0.5;

fn pack_lights(lights: &[Light], live: usize, spot_layers: &[usize], sun_is_baked: bool, culling: bool) -> GpuLights {
    {
        let count = lights.len().min(MAX_LIGHTS);
        let surface = lights.iter().take(live.min(count)).take_while(|l| l.is_surface_light()).count();
        let mut gpu = GpuLights {
            count: [count as u32, live.min(count) as u32, u32::from(!culling), 0],
            surface_lights: [surface as u32, 0, 0, 0],
            lights: [GpuLight::zeroed(); MAX_LIGHTS],
            surfaces: [[0.0; 4]; LIT_SURFACE_VEC4S * MAX_LIT_SURFACES],
        };
        for (slot, l) in gpu.lights.iter_mut().zip(lights.iter().take(MAX_LIGHTS)) {
            let color = l.color.to_linear();
            let (cos_outer, cos_inner) = l.cone_cosines();

            let kind = match l.kind {
                LightKind::Point => 0.0,
                LightKind::Spot => 1.0,
                LightKind::Directional => 2.0,
            };
            let baked = sun_is_baked && l.kind == LightKind::Directional;

            let marker = match l.mask_channel {
                Some(c) => 2.0 + c as f32,
                None if baked => 1.0,
                None if !l.in_level_bake => -1.0,
                None => 0.0,
            };
            *slot = GpuLight {
                position: [l.position.x, l.position.y, l.position.z, marker],
                direction: [l.direction.x, l.direction.y, l.direction.z, cos_inner],
                color_intensity: [color[0], color[1], color[2], l.intensity],
                params: [
                    l.range,
                    cos_outer,
                    kind,
                    if l.casts_shadow() { -1.0 } else { SURFACE_LIGHT - l.source_radius.max(0.0) },
                ],
            };
        }
        for (layer, &light_index) in spot_layers.iter().enumerate() {
            if let Some(slot) = gpu.lights.get_mut(light_index) {
                slot.params[3] = layer as f32;
            }
        }
        gpu
    }
}

pub fn append_baked(live: &[Light], baked: &[Light], max: usize) -> Vec<Light> {
    let mut out: Vec<Light> = live.iter().copied().take(max).collect();
    let room = max.saturating_sub(out.len());
    out.extend(rank_for_budget(baked, room));
    out
}

pub fn wgsl_lights_block(group_index: u32, binding_index: u32) -> String {
    wgsl_lights_block_with(group_index, binding_index, LightsBlockOptions::default())
}

pub const SPOT_SHADOWS_ON: &str = "const SPOT_SHADOWS: bool = true;";
const SPOT_SHADOWS_OFF: &str = "const SPOT_SHADOWS: bool = false;";

pub const SURFACE_LIGHTS_ON: &str = "const SURFACE_LIGHTS: bool = true;";
const SURFACE_LIGHTS_OFF: &str = "const SURFACE_LIGHTS: bool = false;";

pub fn without_spot_shadows(src: String) -> String {
    assert!(src.contains(SPOT_SHADOWS_ON), "the shader has no spot-shadow switch");
    assert!(src.contains(SURFACE_LIGHTS_ON), "the shader has no surface-light switch");
    src.replacen(SPOT_SHADOWS_ON, SPOT_SHADOWS_OFF, 1).replacen(SURFACE_LIGHTS_ON, SURFACE_LIGHTS_OFF, 1)
}

pub fn without_pool_maps(src: String) -> String {
    assert!(
        src.contains(SURFACE_RELIT_READ) && src.contains(SURFACE_RELIT_CALL),
        "the shader reads no pool map"
    );
    src.replacen(SURFACE_RELIT_READ, "", 1).replacen(SURFACE_RELIT_CALL, "", 1)
}

#[derive(Copy, Clone, Default, Debug, PartialEq, Eq)]
pub struct LightsBlockOptions {

    pub probe_from_pass: bool,

    pub probe_face_always: bool,

    pub defer_secondary: bool,

    pub cull_range_first: bool,

    pub card_tests_filtered: bool,

    pub card_relit: bool,

    pub long_step_from_plane: bool,

    pub open_fallback: bool,
}

pub const PROBE_EDGE_FOOTPRINTS: f32 = 1.0;

pub fn wgsl_lights_block_with(group_index: u32, binding_index: u32, options: LightsBlockOptions) -> String {
    let LightsBlockOptions {
        probe_from_pass,
        probe_face_always,
        defer_secondary,
        cull_range_first,
        card_tests_filtered,
        card_relit,
        long_step_from_plane,
        open_fallback,
    } = options;

    let open_fallback_code = if open_fallback {
        "            if (camera.portal_params.y < -0.5) {\n                hit.pos = world_pos;\n                hit.found = true;\n                hit.escaped = true;\n                return hit;\n            }\n"
    } else {
        ""
    };

    let spot_long_step_fn = if long_step_from_plane {
        "var<private> pixel_long_step: f32 = 0.0;\nfn set_pixel_long_step(n_plane: vec3<f32>, view_dir: vec3<f32>) {\n    pixel_long_step = pixel_footprint * inverseSqrt(max(abs(dot(view_dir, n_plane)), SPOT_LONG_MIN_COS));\n}\nfn spot_long_step(vn: f32) -> f32 {\n    return pixel_long_step;\n}"
    } else {
        "fn spot_long_step(vn: f32) -> f32 {\n    return pixel_footprint * inverseSqrt(max(abs(vn), SPOT_LONG_MIN_COS));\n}"
    };

    let card_relit_call = if card_relit {
        "    if (PROBE_CARD_RELIT && sum.best > 0.0) {\n        colour += probe_card_relit(row, sum.card, lod, h, q);\n    }\n"
    } else {
        ""
    };

    let (surface_relit_read, surface_relit_call) =
        if defer_secondary || card_relit { (SURFACE_RELIT_READ, SURFACE_RELIT_CALL) } else { ("", "") };
    let lit_surface_vec4s = LIT_SURFACE_VEC4S * MAX_LIT_SURFACES;
    let shadow_tex = binding_index + 1;
    let shadow_samp = binding_index + 2;
    let spot_tex = binding_index + 3;
    let probe_tex = binding_index + 4;
    let probe_samp = binding_index + 5;
    let sun_dynamic_tex = binding_index + 6;
    let probe_depth_tex = binding_index + 7;
    let probe_depth_samp = binding_index + 8;
    let ground_tex = binding_index + 9;
    let proxy_field_tex = binding_index + 10;
    let proxy_card_tex = binding_index + 11;
    let probe_select_binding = binding_index + 12;
    let cloud_shadow_binding = binding_index + 13;
    let probe_select_rows = 3 * crate::renderer::uniforms::MAX_PROBES;
    let proxy_field_rows = crate::renderer::proxy_field::MAX_PROXY_FIELDS * 3;
    let proxy_card_rows = crate::renderer::uniforms::MAX_PROXIES / 4;
    let building_rows = crate::renderer::uniforms::MAX_BUILDINGS * 2;
    let capsule_rows = crate::renderer::uniforms::MAX_CAPSULES * 2;
    let capsule_group_rows = crate::renderer::uniforms::MAX_CAPSULE_GROUPS * 2;
    let capsules_per_group = crate::renderer::uniforms::CAPSULES_PER_GROUP;
    let character_card_sets = super::proxy_cards::CHARACTER_CARD_SETS;
    let character_card_rows = 2 * character_card_sets;
    let room_exposure_rooms = crate::renderer::uniforms::ROOM_EXPOSURE_ROOMS;
    let room_exposure_rows = 2 * room_exposure_rooms;
    let room_near = 1.5f32;
    let character_card_max_lod = (super::character_cards::CARD_MIPS - 1) as f32;
    let pool_maps_across = super::pool_cards::POOL_MAPS_ACROSS;
    let reflection_contrast = format!("{:?}", crate::renderer::space_warp::REFLECTION_CONTRAST_RATIO);
    let probe_edge_footprints = PROBE_EDGE_FOOTPRINTS;
    let floor_mirror_bias = super::brush_pipeline::probe_pass::FLOOR_MIRROR_BIAS;

    let capsule_glow_term = if probe_from_pass { "" } else { " + capsule_glow * spec_occ" };

    let probe_slots = crate::renderer::uniforms::MAX_PROBES * 3;
    let max_probes = crate::renderer::uniforms::MAX_PROBES;

    let probe_nearest_unrolled: String = (0..max_probes)
        .map(|k| {
            format!(
                "    if (count > {k}) {{
        let room{k} = camera.probe_boxes[{room_at}].w;
        let v{k} = camera.probe_boxes[{centre_at}].xyz - p;
        let dd{k} = dot(v{k}, v{k});
        let mine{k} = room{k} == a || room{k} == b;
        if (mine{k} && dd{k} < n.d0) {{
            n.s1 = n.s0;
            n.d1 = n.d0;
            n.s0 = {k};
            n.d0 = dd{k};
        }} else if (mine{k} && dd{k} < n.d1) {{
            n.s1 = {k};
            n.d1 = dd{k};
        }}
    }}
",
                room_at = 3 * k + 2,
                centre_at = 3 * k,
            )
        })
        .collect();

    let probe_nearest_const_unrolled: String = (0..max_probes)
        .map(|k| {
            format!(
                "    if (count > {k}) {{
        let room{k} = probe_select[{room_at}].w;
        let v{k} = probe_select[{centre_at}].xyz - p;
        let dd{k} = dot(v{k}, v{k});
        let mine{k} = room{k} == a || room{k} == b;
        if (mine{k} && dd{k} < n.d0) {{
            n.s1 = n.s0;
            n.d1 = n.d0;
            n.s0 = {k};
            n.d0 = dd{k};
        }} else if (mine{k} && dd{k} < n.d1) {{
            n.s1 = {k};
            n.d1 = dd{k};
        }}
    }}
",
                room_at = 3 * k + 2,
                centre_at = 3 * k,
            )
        })
        .collect();
    let portal_slots = crate::renderer::uniforms::MAX_PORTALS * 3;
    let proxy_slots = crate::renderer::uniforms::MAX_PROXIES * 3;
    let room_table_rows = crate::renderer::uniforms::ROOM_TABLE_ROWS;
    let shadow_tiles = super::shadow::SHADOW_MATRICES;
    let moving_maps = shadow_tiles + 1;
    let max_spot_shadows = super::shadow::MAX_SPOT_SHADOWS;
    let atlas_rows = super::shadow::SPOT_ATLAS_ROWS;
    let sun_atlas_tiles = super::shadow::SUN_ATLAS_TILES;
    let sun_near_tile = super::shadow::SUN_NEAR_TILE;
    let sun_near_zoom = super::shadow::SUN_NEAR_ZOOM;

    let atlas_cols = super::shadow::SPOT_ATLAS_COLS;

    let tonemap_block = super::tonemap::wgsl_tonemap_block();

    let probe_box_margin = probe_box_margin();
    let portal_fade = PROBE_PORTAL_FADE;
    let portal_side_fade = PROBE_PORTAL_SIDE_FADE;
    let precision_aliases = crate::renderer::shader_precision::F32_ALIASES;
    let spot_shadows_on = SPOT_SHADOWS_ON;
    let surface_lights_on = SURFACE_LIGHTS_ON;
    let probe_fixup_wgsl = crate::renderer::probe_fixup::lights_block_wgsl(defer_secondary);
    let ground_trace_finest = crate::renderer::ground_map::GROUND_TRACE_FINEST_LEVEL;
    let ground_trace_start = crate::renderer::ground_map::GROUND_TRACE_START_LEVEL;
    let ground_trace_readings = crate::renderer::ground_map::GROUND_TRACE_READINGS;
    let ground_trace_steps = crate::renderer::ground_map::GROUND_TRACE_MAX_STEPS;
    format!(
        r#"
{precision_aliases}

const HF_MAX: f32 = 60000.0;

{spot_shadows_on}

{surface_lights_on}
struct Camera {{
    view_proj: array<mat4x4<f32>, 2>,
    inv_view_proj: array<mat4x4<f32>, 2>,
    sun_view_proj: mat4x4<f32>,

    moving_view_proj: array<mat4x4<f32>, {moving_maps}>,
    camera_pos: array<vec4<f32>, 2>,

    shadow_params: vec4<f32>,

    sky_params: vec4<f32>,

    sky_sh: array<vec4<f32>, 9>,

    post_params: vec4<f32>,

    player_frame: vec4<f32>,

    probe_params: vec4<f32>,

    probe_boxes: array<vec4<f32>, {probe_slots}>,

    portal_params: vec4<f32>,

    probe_portals: array<vec4<f32>, {portal_slots}>,

    proxy_params: vec4<f32>,

    probe_proxies: array<vec4<f32>, {proxy_slots}>,

    probe_rooms: array<vec4<f32>, {room_table_rows}>,

    ground_params: vec4<f32>,

    proxy_fields: array<vec4<f32>, {proxy_field_rows}>,

    proxy_cards: array<vec4<f32>, {proxy_card_rows}>,

    building_boxes: array<vec4<f32>, {building_rows}>,

    capsules: array<vec4<f32>, {capsule_rows}>,
    capsule_groups: array<vec4<f32>, {capsule_group_rows}>,
    capsule_params: vec4<f32>,

    character_cards: array<vec4<f32>, {character_card_rows}>,

    room_exposure: array<vec4<f32>, {room_exposure_rows}>,

    cloud_map: array<vec4<f32>, 3>,
}}

fn to_world_direction(d: vec3<f32>) -> vec3<f32> {{
    let yaw = camera.player_frame.w;
    let s = sin(yaw);
    let c = cos(yaw);
    return vec3<f32>(c * d.x + s * d.z, d.y, -s * d.x + c * d.z);
}}

fn to_world_space(p: vec3<f32>) -> vec3<f32> {{
    let yaw = camera.player_frame.w;
    let s = sin(yaw);
    let c = cos(yaw);
    let r = vec3<f32>(c * p.x + s * p.z, p.y, -s * p.x + c * p.z);
    return r + camera.player_frame.xyz;
}}

fn to_player_direction(d: vec3<f32>) -> vec3<f32> {{
    let yaw = camera.player_frame.w;
    let s = sin(yaw);
    let c = cos(yaw);
    return vec3<f32>(c * d.x - s * d.z, d.y, s * d.x + c * d.z);
}}

fn to_player_space(w: vec3<f32>) -> vec3<f32> {{
    let yaw = camera.player_frame.w;
    let s = sin(yaw);
    let c = cos(yaw);
    let p = w - camera.player_frame.xyz;
    return vec3<f32>(c * p.x - s * p.z, p.y, s * p.x + c * p.z);
}}
@group({group_index}) @binding(0) var<uniform> camera: Camera;
{tonemap_block}

// LOCAL ADAPTATION by room: a closed room seen from outside it -- and its
// doorways -- is drawn at the eye's exposure times its row's factor
// (`uniforms::RoomExposure`), so a lit hall seen from the night stays
// readable while the night stays night. Rows come in pairs, (min, factor)
// and (max, band); a factor of 0 is an empty row. Asked per pixel, and only
// where a vertex was near a box (`room_near`).
fn room_local_exposure(p: vec3<f32>) -> f32 {{
    var e = 1.0;
    for (var i = 0u; i < {room_exposure_rooms}u; i = i + 1u) {{
        let lo = camera.room_exposure[2u * i];
        let hi = camera.room_exposure[2u * i + 1u];
        if (lo.w <= 0.0) {{
            break;
        }}
        let inside = min(min(min(p.x - lo.x, hi.x - p.x), min(p.y - lo.y, hi.y - p.y)), min(p.z - lo.z, hi.z - p.z));
        e = min(e, pow(lo.w, clamp((inside + hi.w) / max(hi.w, 1e-3), 0.0, 1.0)));
    }}
    return e;
}}

// 1 for a vertex within ROOM_NEAR of a room's box, else 0: carried to the
// fragment, which asks `room_local_exposure` only where it is above 0.
fn room_near(p: vec3<f32>) -> f32 {{
    for (var i = 0u; i < {room_exposure_rooms}u; i = i + 1u) {{
        let lo = camera.room_exposure[2u * i];
        let hi = camera.room_exposure[2u * i + 1u];
        if (lo.w <= 0.0) {{
            break;
        }}
        if (lo.w < 1.0 && all(p > lo.xyz - vec3<f32>({room_near:?})) && all(p < hi.xyz + vec3<f32>({room_near:?}))) {{
            return 1.0;
        }}
    }}
    return 0.0;
}}

struct Light {{
    position: vec4<f32>,
    direction: vec4<f32>,
    color_intensity: vec4<f32>,
    params: vec4<f32>,
}}
struct Lights {{
    count: vec4<u32>,

    surface_lights: vec4<u32>,
    lights: array<Light, {MAX_LIGHTS}>,

    surfaces: array<vec4<f32>, {lit_surface_vec4s}>,
}}
@group({group_index}) @binding({binding_index}) var<uniform> lights: Lights;
@group({group_index}) @binding({shadow_tex}) var sun_shadow_tex: texture_depth_2d;

@group({group_index}) @binding({sun_dynamic_tex}) var sun_dynamic_shadow_tex: texture_depth_2d;
@group({group_index}) @binding({shadow_samp}) var shadow_samp: sampler_comparison;

@group({group_index}) @binding({spot_tex}) var spot_shadow_tex: texture_depth_2d;
@group({group_index}) @binding({probe_tex}) var probe_cube: texture_cube_array<f32>;
@group({group_index}) @binding({probe_samp}) var probe_samp: sampler;

@group({group_index}) @binding({probe_depth_tex}) var probe_depth: texture_cube_array<f32>;
@group({group_index}) @binding({probe_depth_samp}) var probe_depth_samp: sampler;

@group({group_index}) @binding({ground_tex}) var ground_map: texture_2d<f32>;

@group({group_index}) @binding({proxy_field_tex}) var proxy_field: texture_3d<f32>;

@group({group_index}) @binding({proxy_card_tex}) var proxy_cards: texture_2d<f32>;

@group({group_index}) @binding({cloud_shadow_binding}) var cloud_shadow_map: texture_2d<f32>;

@group({group_index}) @binding({probe_select_binding}) var<uniform> probe_select: array<vec4<f32>, {probe_select_rows}>;

const AMBIENT: f32 = 0.6;

fn sky_uv(d: vec3<f32>) -> vec2<f32> {{
    let n = normalize(d);
    let u = (atan2(n.x, n.z) + 3.14159265) / 6.28318531;
    let v = acos(clamp(n.y, -1.0, 1.0)) / 3.14159265;
    return vec2<f32>(u, v);
}}

fn sky_irradiance(n: vec3<f32>) -> vec3<f32> {{
    let x = n.x; let y = n.y; let z = n.z;

    var e = vec3<f32>(0.0);
    e = e + camera.sky_sh[0].rgb * 0.282095 * 1.0;
    e = e + camera.sky_sh[1].rgb * (0.488603 * y) * 0.6666667;
    e = e + camera.sky_sh[2].rgb * (0.488603 * z) * 0.6666667;
    e = e + camera.sky_sh[3].rgb * (0.488603 * x) * 0.6666667;
    e = e + camera.sky_sh[4].rgb * (1.092548 * x * y) * 0.25;
    e = e + camera.sky_sh[5].rgb * (1.092548 * y * z) * 0.25;
    e = e + camera.sky_sh[6].rgb * (0.315392 * (3.0 * z * z - 1.0)) * 0.25;
    e = e + camera.sky_sh[7].rgb * (1.092548 * x * z) * 0.25;
    e = e + camera.sky_sh[8].rgb * (0.546274 * (x * x - y * y)) * 0.25;
    return max(e, vec3<f32>(0.0));
}}

fn sky_irradiance_up() -> vec3<f32> {{
    var e = vec3<f32>(0.0);
    e = e + camera.sky_sh[0].rgb * 0.282095 * 1.0;
    e = e + camera.sky_sh[1].rgb * (0.488603 * 1.0) * 0.6666667;
    e = e + camera.sky_sh[6].rgb * (0.315392 * (3.0 * 0.0 * 0.0 - 1.0)) * 0.25;
    e = e + camera.sky_sh[8].rgb * (0.546274 * (0.0 * 0.0 - 1.0 * 1.0)) * 0.25;
    return max(e, vec3<f32>(0.0));
}}
const SPEC_STRENGTH: f32 = 0.35;
const SHININESS: f32 = 32.0;

const LIGHT_SOURCE_RADIUS: f32 = 0.35;

const MAX_SHININESS: f32 = 320.0;

const INV_PI: f32 = 0.31830987;

var<private> dbg_probe: vec3<f32> = vec3<f32>(0.0);
var<private> dbg_direct: vec3<f32> = vec3<f32>(0.0);
var<private> dbg_baked: vec3<f32> = vec3<f32>(0.0);

var<private> dbg_probe_factors: vec3<f32> = vec3<f32>(0.0);

var<private> pixel_footprint: f32 = 0.0;

const SPOT_LONG_MIN_COS: f32 = 1e-3;
{spot_long_step_fn}

var<private> terminator_width: f32 = 0.0;

const LAMP_RADIUS: f32 = 0.05;

var<private> receiver_sun_mask: f32 = -1.0;

var<private> stationary_vis_a: vec4<f32> = vec4<f32>(1.0);
var<private> stationary_vis_b: vec4<f32> = vec4<f32>(1.0);

fn set_stationary_masks(st_0: vec4<f32>, st_1: vec4<f32>, st_2: vec4<f32>, st_3: vec4<f32>, range: f32) {{
    let st_da = (vec4<f32>(st_0.r, st_0.b, st_1.r, st_1.b) - vec4<f32>(0.5)) * (2.0 * range);
    let st_db = (vec4<f32>(st_2.r, st_2.b, st_3.r, st_3.b) - vec4<f32>(0.5)) * (2.0 * range);
    let st_pa = vec4<f32>(st_0.g, st_0.a, st_1.g, st_1.a) * range;
    let st_pb = vec4<f32>(st_2.g, st_2.a, st_3.g, st_3.a) * range;
    let st_wa = max(max(st_pa, 0.5 * fwidth(st_da)), vec4<f32>(0.02));
    let st_wb = max(max(st_pb, 0.5 * fwidth(st_db)), vec4<f32>(0.02));
    stationary_vis_a = smoothstep(-st_wa, st_wa, st_da);
    stationary_vis_b = smoothstep(-st_wb, st_wb, st_db);
}}

fn set_stationary_masks_exact(st_0: vec4<f32>, st_1: vec4<f32>, st_2: vec4<f32>, st_3: vec4<f32>, range: f32) {{
    let st_da = (vec4<f32>(st_0.r, st_0.b, st_1.r, st_1.b) - vec4<f32>(0.5)) * (2.0 * range);
    let st_db = (vec4<f32>(st_2.r, st_2.b, st_3.r, st_3.b) - vec4<f32>(0.5)) * (2.0 * range);
    let st_wa = max(vec4<f32>(st_0.g, st_0.a, st_1.g, st_1.a) * range, vec4<f32>(0.02));
    let st_wb = max(vec4<f32>(st_2.g, st_2.a, st_3.g, st_3.a) * range, vec4<f32>(0.02));
    stationary_vis_a = smoothstep(-st_wa, st_wa, st_da);
    stationary_vis_b = smoothstep(-st_wb, st_wb, st_db);
}}

fn stationary_visibility(l: Light) -> f32 {{
    return stationary_visibility_of(l.position.w);
}}

fn stationary_visibility_of(marker: f32) -> f32 {{
    if (marker < 1.5) {{
        return 1.0;
    }}
    let c = i32(marker - 1.5);
    let v = select(stationary_vis_b, stationary_vis_a, c < 4);
    return v[c & 3];
}}

const CAPSULES_PER_GROUP: i32 = {capsules_per_group};

const CAPSULE_AMBIENT_FULL: f32 = 0.15;
const CAPSULE_AMBIENT_REACH: f32 = 0.6;

const CAPSULE_SHAPE_BLUR: f32 = 0.15;

var<private> capsule_receiver: bool = true;

fn capsule_nearest_to_ray(a: vec3<f32>, b: vec3<f32>, o: vec3<f32>, d: vec3<f32>) -> vec3<f32> {{
    let ba = b - a;
    let oa = a - o;
    let baba = dot(ba, ba);
    let bad = dot(ba, d);
    let denom = baba - bad * bad;
    var s = 0.0;
    if (denom > 1e-8) {{
        s = (bad * dot(oa, d) - dot(oa, ba)) / denom;
    }}
    return a + ba * clamp(s, 0.0, 1.0);
}}

fn capsule_ambient(p: vec3<f32>, n: vec3<f32>) -> f32 {{
    let groups = i32(camera.capsule_params.x);
    if (groups <= 0 || !capsule_receiver) {{
        return 1.0;
    }}
    var vis = 1.0;
    for (var g = 0; g < groups; g = g + 1) {{
        let bound = camera.capsule_groups[g * 2];
        if (length(bound.xyz - p) > bound.w + CAPSULE_AMBIENT_REACH) {{
            continue;
        }}
        let count = i32(camera.capsule_groups[g * 2 + 1].w);
        for (var k = 0; k < count; k = k + 1) {{
            let i = g * CAPSULES_PER_GROUP + k;
            let ar = camera.capsules[i * 2];
            let ba = camera.capsules[i * 2 + 1].xyz - ar.xyz;
            let s = clamp(dot(p - ar.xyz, ba) / max(dot(ba, ba), 1e-8), 0.0, 1.0);
            let w = ar.xyz + ba * s - p;
            let d = max(length(w), ar.w);

            let fade = 1.0 - smoothstep(CAPSULE_AMBIENT_FULL, CAPSULE_AMBIENT_REACH, d - ar.w);
            let k2 = (ar.w / d) * (ar.w / d) * fade;
            vis = vis * (1.0 - k2 * max(dot(n, w / d), 0.0));
        }}
    }}
    return vis;
}}

fn capsule_room_exit(p: vec3<f32>, d: vec3<f32>) -> f32 {{
    var t_max = 1e3;
    if (probe_face_given.y > 0.5) {{
        let slot = probe_room_slot(probe_face_given.x);
        if (slot >= 0) {{
            let pw = to_world_space(p);
            let dw = to_world_direction(d);
            let inv = select(vec3<f32>(3.4e38), 1.0 / dw, abs(dw) > vec3<f32>(1e-6));
            let far = max((camera.probe_boxes[slot * 3 + 2].xyz - pw) * inv, (camera.probe_boxes[slot * 3 + 1].xyz - pw) * inv);
            t_max = max(min(min(far.x, far.y), far.z), 0.0);
        }}
    }}
    return t_max;
}}

const CAPSULE_REFLECT_MAX_ROUGHNESS: f32 = 0.7;

const CHARACTER_CARD_SETS: i32 = {character_card_sets};
const CHARACTER_CARD_MAX_LOD: f32 = {character_card_max_lod:?};

fn character_card_at(card: f32, uv: vec2<f32>, row: f32, res: f32, lod: f32, dims: vec2<f32>) -> vec4<f32> {{
    let s = 0.5 * exp2(ceil(lod));
    let at = vec2<f32>(card * res, row) + clamp(uv * res, vec2<f32>(s), vec2<f32>(res - s));
    return textureSampleLevel(proxy_cards, probe_samp, at / dims, lod);
}}

fn character_card_look(g: i32, p: vec3<f32>, d: vec3<f32>, t: f32, lobe: f32, eye: f32) -> vec4<f32> {{
    var own = -1;
    for (var k = 0; k < CHARACTER_CARD_SETS; k = k + 1) {{
        if (camera.character_cards[k * 2].w > 0.5 && i32(camera.character_cards[k * 2 + 1].w) == g) {{
            own = k;
        }}
    }}
    if (own < 0) {{
        return vec4<f32>(-1.0);
    }}
    let centre = camera.character_cards[own * 2];
    let half = camera.character_cards[own * 2 + 1].xyz;
    let w = to_world_direction(d);
    let uvw = clamp(to_world_direction(p + d * t - centre.xyz) / half * 0.5 + vec3<f32>(0.5), vec3<f32>(0.0), vec3<f32>(1.0));
    let dims = vec2<f32>(textureDimensions(proxy_cards));
    let res = dims.x / 6.0;
    let footprint = max(t * lobe, pixel_footprint * (1.0 + t / eye));
    let texel = 2.0 * max(max(half.x, half.y), half.z) / res;
    let lod = clamp(log2(max(2.0 * footprint / texel, 1.0)), 0.0, CHARACTER_CARD_MAX_LOD);
    let row = (centre.w - 1.0) * res;
    let k = w * w;
    // The three cards through one copy of the read, in the same order.
    var across_x = vec4<f32>(0.0);
    var across_y = vec4<f32>(0.0);
    var across_z = vec4<f32>(0.0);
    for (var a = 0u; a < 3u; a = a + 1u) {{
        let side = select(select(w.z, w.y, a == 1u), w.x, a == 0u);
        let uv = select(select(uvw.xy, uvw.zx, a == 1u), uvw.yz, a == 0u);
        let card = character_card_at(f32(2u * a) + select(1.0, 0.0, side < 0.0), uv, row, res, lod, dims);
        across_x = select(across_x, card, a == 0u);
        across_y = select(across_y, card, a == 1u);
        across_z = select(across_z, card, a == 2u);
    }}

    let cover = min(across_x.a, min(across_y.a, across_z.a));
    let seen = k.x * across_x + k.y * across_y + k.z * across_z;
    return vec4<f32>(seen.rgb / max(seen.a, 1e-4) * cover, cover);
}}

var<private> capsule_glow: vec3<f32> = vec3<f32>(0.0);

fn capsule_reflection(p: vec3<f32>, d: vec3<f32>, roughness: f32, lit: vec3<f32>, behind: vec4<f32>) -> vec4<f32> {{
    let groups = i32(camera.capsule_params.x);
    if (groups <= 0 || roughness > CAPSULE_REFLECT_MAX_ROUGHNESS) {{
        return behind;
    }}
    let lobe = probe_lobe_tan(roughness);
    let eye = max(distance(cam_pos(), p), 0.05);
    var t_max = -1.0;

    var cover = 0.0;
    var win_i = -1;
    var win_t = 0.0;

    var glow = 0.0;
    var glow_i = -1;
    var glow_t = 0.0;
    for (var g = 0; g < groups; g = g + 1) {{
        let bound = camera.capsule_groups[g * 2];
        let oc = bound.xyz - p;
        let along = dot(oc, d);
        let at = max(along, 0.0);
        let spread = at * lobe + pixel_footprint * (1.0 + at / eye) + CAPSULE_SHAPE_BLUR;
        if (along < -bound.w || length(oc - d * along) > bound.w + spread) {{
            continue;
        }}

        if (t_max < 0.0) {{
            t_max = capsule_room_exit(p, d);
        }}
        let count = i32(camera.capsule_groups[g * 2 + 1].w);
        for (var k = 0; k < count; k = k + 1) {{
            let i = g * CAPSULES_PER_GROUP + k;
            let ar = camera.capsules[i * 2];
            let bs = camera.capsules[i * 2 + 1];

            var t = -1.0;
            var off = 0.0;
            if (bs.w > 0.0) {{
                let n = normalize(bs.xyz - ar.xyz);
                let facing = dot(d, n);
                t = select(-1.0, dot(bs.xyz - p, n) / facing, facing < -1e-4);
                off = distance(p + d * t, bs.xyz);
            }} else {{
                let q = capsule_nearest_to_ray(ar.xyz, bs.xyz, p, d);
                t = dot(q - p, d);
                off = length(q - (p + d * t));
            }}
            if (t <= 0.0 || t >= t_max) {{
                continue;
            }}

            let footprint =
                max(max(t * lobe, pixel_footprint * (1.0 + t / eye)), select(0.0, CAPSULE_SHAPE_BLUR, bs.w == 0.0));
            if (bs.w > 0.0) {{

                let f2 = footprint * footprint;
                let reach = max(ar.w, footprint);
                let c = ar.w * ar.w / (reach * reach + 0.2 * f2) * (1.0 - smoothstep(reach - footprint, reach + footprint, off));
                if (c * bs.w > glow) {{
                    glow = c * bs.w;
                    glow_i = i;
                    glow_t = t;
                }}
                continue;
            }}

            let c = (1.0 - smoothstep(ar.w - footprint, ar.w + footprint, off)) * min(ar.w / footprint, 1.0);
            if (c > cover || (c == cover && t < win_t)) {{
                cover = c;
                win_i = i;
                win_t = t;
            }}
        }}
    }}
    if (cover <= 0.0 && glow <= 0.0) {{
        return behind;
    }}
    var colour = vec3<f32>(0.0);
    if (cover > 0.0) {{

        let g = win_i / CAPSULES_PER_GROUP;
        let s = camera.capsules[win_i * 2 + 1].w;
        colour = camera.capsule_groups[g * 2 + 1].rgb * select(1.0, -s, s < 0.0);
        let look = character_card_look(g, p, d, win_t, lobe, eye);
        if (look.a >= 0.0) {{
            cover = min(look.a, 1.0);
            colour = look.rgb / max(look.a, 1e-4);
        }}
    }}
    var a = max(behind.a, cover);
    var seen = 0.0;
    if (glow_i >= 0) {{

        seen = glow * select(1.0, 1.0 - cover, win_t < glow_t);

        a = max(a, glow / camera.capsules[glow_i * 2 + 1].w);
    }}

    let answer = vec4<f32>(mix(behind.rgb, colour * lit * INV_PI, cover), a);
    if (glow_i >= 0) {{
        capsule_glow = camera.capsule_groups[(glow_i / CAPSULES_PER_GROUP) * 2 + 1].rgb * seen / a
            * capsule_glass_beam(p, camera.capsules[glow_i * 2 + 1].xyz);
    }}
    return answer;
}}

fn capsule_glass_beam(p: vec3<f32>, glass: vec3<f32>) -> f32 {{
    for (var i: u32 = 0u; i < live_light_count(); i = i + 1u) {{
        let apart = lights.lights[i].position.xyz - glass;
        if (dot(apart, apart) >= GLASS_ON_ITS_BEAM * GLASS_ON_ITS_BEAM) {{
            continue;
        }}

        let to_p = p - lights.lights[i].position.xyz;
        let cos_p = dot(to_p, lights.lights[i].direction.xyz) * inverseSqrt(max(dot(to_p, to_p), 1e-12));
        let cos_outer = lights.lights[i].params.y;
        let s = clamp((cos_p - cos_outer) / max(lights.lights[i].direction.w - cos_outer, 1e-4), 0.0, 1.0);
        let into = select(1.0, s * s * (3.0 - 2.0 * s), lights.lights[i].params.z > 0.5);
        let layer = i32(lights.lights[i].params.w);
        if (SPOT_SHADOWS && layer >= 0 && f32(layer) < camera.shadow_params.y) {{
            return into * pcf_layer_tap(spot_shadow_tex, layer, p, spot_view_proj(layer));
        }}
        return into;
    }}
    return 1.0;
}}

const GLASS_ON_ITS_BEAM: f32 = 0.01;

fn light_culling() -> bool {{
    return lights.count.z == 0u;
}}

var<private> receiver_skips_baked: bool = false;
fn live_light_count() -> u32 {{
    return select(lights.count.x, lights.count.y, receiver_skips_baked);
}}

fn surface_light_count() -> u32 {{
    return select(0u, lights.surface_lights.x, SURFACE_LIGHTS);
}}

const SPOT_EDGE_MIN_PIXELS: f32 = 1.5;

const SPOT_EDGE_MAX_WIDEN: f32 = 3.0;

const SPOT_EDGE_AVERAGE: bool = true;

fn spot_cone(cos_angle: f32, cos_outer: f32, cos_inner: f32, dist: f32, across: f32) -> f32 {{
    let authored = max(cos_inner - cos_outer, 0.0001);
    let sin_a = sqrt(max(1.0 - cos_angle * cos_angle, 0.0));
    let footprint = sin_a * pixel_footprint / max(dist, 0.001);
    let at_least = SPOT_EDGE_MIN_PIXELS * footprint;
    let width = max(authored, min(at_least, authored * SPOT_EDGE_MAX_WIDEN));

    let widen = width - authored;

    let half = select(0.0, 0.5 * sqrt(max(across * across - footprint * footprint, 0.0)) / width, SPOT_EDGE_AVERAGE);
    let s = clamp(((cos_angle - cos_outer + 0.5 * widen) / width + half) / (1.0 + 2.0 * half), 0.0, 1.0);
    return s * s * (3.0 - 2.0 * s);
}}

fn spot_cone_across(spot_dir: vec3<f32>, l_dir: vec3<f32>, cos_angle: f32, dist: f32, n: vec3<f32>, view_dir: vec3<f32>) -> f32 {{

    let vn = dot(view_dir, n);
    let g_along = dot(view_dir, spot_dir) - vn * dot(n, spot_dir) + cos_angle * (dot(view_dir, l_dir) - vn * dot(n, l_dir));
    return abs(g_along) * inverseSqrt(max(1.0 - vn * vn, 1e-8)) * spot_long_step(vn) / max(dist, 0.001);
}}

var<private> probe_brightness: f32 = 0.0;

var<private> probe_reach: f32 = 0.0;

var<private> ground_trace_coarsen: i32 = 0;

const PROBE_LOBE_HEMISPHERICAL: f32 = 0.75;

const PROBE_REACH_SKY: f32 = 10000.0;

const PROBE_ENV_FROM_PASS: bool = {probe_from_pass};

const PROBE_FACE_ALWAYS_GIVEN: bool = {probe_face_always};

const PROBE_SECONDARY_DEFERRED: bool = {defer_secondary};

const PROBE_ALL_OUTDOORS: bool = false;

const PROBE_NOWHERE: bool = false;

const PROBE_OPEN_GROUND: bool = false;

const PROBE_CARD_TESTS_FILTERED: bool = {card_tests_filtered};

const PROBE_CARD_RELIT: bool = {card_relit};

const PROBE_CARD_RELIT_LIFT: f32 = 0.02;

const CULL_RANGE_FIRST: bool = {cull_range_first};

const SKY_SUN_NEVER_REACHES: bool = false;

const SUN_MASK_EVERYWHERE: bool = false;

var<private> probe_fragment: vec4<f32> = vec4<f32>(0.0);
{probe_fixup_wgsl}

var<private> probe_env_given: vec4<f32> = vec4<f32>(0.0);

const FLOOR_MIRROR_BIAS: f32 = {floor_mirror_bias:?};

const FLOOR_MIRROR_SLAB: f32 = 0.02;
fn on_floor_mirror(pos: vec3<f32>, geom_n: vec3<f32>) -> bool {{
    let lane = camera.capsule_params.w;
    return lane != 0.0 && geom_n.y > 0.95 && abs(pos.y - (lane - FLOOR_MIRROR_BIAS)) < FLOOR_MIRROR_SLAB;
}}

var<private> probe_floor_mirror_here: bool = false;
var<private> probe_capsule_lit: vec3<f32> = vec3<f32>(0.0);

var<private> probe_volume_pos: vec4<f32> = vec4<f32>(0.0);

const PROBE_ROUGHNESS_MIPS: f32 = 7.0;

const PROBE_MIN_LOD: f32 = 0.5;

const PROBE_MAX_LOD: f32 = 7.0;

const PROBE_BLEND_BAND: f32 = 2.0;

const PROBE_BOX_MARGIN: f32 = {probe_box_margin:?};

const PROBE_NORMALISATION: bool = true;

const PROBE_NORMALISATION_FLOOR: f32 = 0.0;

const MIN_BOUNCE_DIR_LENGTH: f32 = 0.05;

const MAX_BOUNCE_DIRECTIONALITY: f32 = 0.5;

var<private> view_slot: i32 = 0;

fn cam_view_proj() -> mat4x4<f32> {{ return camera.view_proj[view_slot]; }}
fn cam_inv_view_proj() -> mat4x4<f32> {{ return camera.inv_view_proj[view_slot]; }}
fn cam_pos() -> vec3<f32> {{ return camera.camera_pos[view_slot].xyz; }}

const SUN_MOVING_MAP: i32 = 0;
const CHARACTER_MAPS: i32 = 1 + {max_spot_shadows};
fn spot_view_proj(i: i32) -> mat4x4<f32> {{ return camera.moving_view_proj[1 + i]; }}

fn shadow_coords(world_pos: vec3<f32>, light_view_proj: mat4x4<f32>) -> vec4<f32> {{
    let lp = light_view_proj * vec4<f32>(world_pos, 1.0);
    let ndc = lp.xyz / lp.w;
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    let bias = 0.0015;

    let depth = ndc.z - bias;
    var valid = 1.0;
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || depth > 1.0 || ndc.z < 0.0) {{
        valid = 0.0;
    }}
    return vec4<f32>(uv.x, uv.y, depth, valid);
}}

fn pcf(tex: texture_depth_2d, world_pos: vec3<f32>, light_view_proj: mat4x4<f32>) -> f32 {{
    let c = shadow_coords(world_pos, light_view_proj);
    if (c.w < 0.5) {{ return 1.0; }}
    let texel = 1.0 / vec2<f32>(textureDimensions(tex));
    var sum = 0.0;
    for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {{
        for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {{
            let off = vec2<f32>(f32(dx), f32(dy)) * texel;
            sum = sum + textureSampleCompareLevel(tex, shadow_samp, c.xy + off, c.z);
        }}
    }}
    return sum / 9.0;
}}

fn sun_visibility(l: Light, world_pos: vec3<f32>) -> f32 {{
    var vis = sun_level_visibility(world_pos);
    if (vis > 0.0 && l.position.w > 0.5 && camera.shadow_params.z > 0.5) {{
        vis = vis * sun_moving_visibility(world_pos);
    }}
    return vis;
}}

fn sun_level_visibility(world_pos: vec3<f32>) -> f32 {{
    var vis = 1.0;
    if (SUN_MASK_EVERYWHERE || receiver_sun_mask >= 0.0) {{
        vis = receiver_sun_mask;
    }} else if (camera.shadow_params.x > 0.5) {{
        vis = pcf(sun_shadow_tex, world_pos, camera.sun_view_proj);
    }}
    return vis * cloud_sun_shadow(world_pos);
}}

// The light the clouds let through to this point (`Uniforms::cloud_map`):
// their shadows sweep over the ground with the wind.
fn cloud_sun_shadow(p: vec3<f32>) -> f32 {{
    if (camera.cloud_map[2].w <= 0.0) {{
        return 1.0;
    }}
    let m = camera.cloud_map[0];
    let uv = vec2<f32>(m.x * p.x + m.y * p.z, m.z * p.x + m.w * p.z) + camera.cloud_map[1].xy;
    return textureSampleLevel(cloud_shadow_map, probe_samp, uv, 0.0).r;
}}

// What a cloud overhead scatters down in place of the sun it hides: its
// base glows, so ground in a cloud's shadow is lit from above, brighter
// than by the clear sky alone and with its relief still reading
// (`Uniforms::cloud_map`).
fn cloud_fill(p: vec3<f32>, n: vec3<f32>) -> vec3<f32> {{
    // No test for the map being off: its fill is then zero.
    return camera.cloud_map[2].xyz * ((1.0 - cloud_sun_shadow(p)) * (0.5 + 0.5 * n.y));
}}

fn sun_moving_visibility(world_pos: vec3<f32>) -> f32 {{
    let c = shadow_coords(world_pos, camera.moving_view_proj[SUN_MOVING_MAP]);
    if (c.w < 0.5) {{ return 1.0; }}
    let near = (c.xy - vec2<f32>(0.5)) * SUN_NEAR_ZOOM + vec2<f32>(0.5);

    let edge = 2.0 * SUN_ATLAS_GRID.x / f32(textureDimensions(sun_dynamic_shadow_tex).x);
    let in_near = all(abs(near - vec2<f32>(0.5)) < vec2<f32>(0.5 - edge));
    return pcf_tile_at(
        sun_dynamic_shadow_tex,
        vec2<f32>(select(0.0, SUN_NEAR_TILE, in_near), 0.0),
        SUN_ATLAS_GRID,
        vec3<f32>(select(c.xy, near, in_near), c.z),
    );
}}

fn sun_moving_tile_at(c: vec3<f32>) -> vec4<f32> {{
    let near = (c.xy - vec2<f32>(0.5)) * SUN_NEAR_ZOOM + vec2<f32>(0.5);

    let edge = 2.0 * SUN_ATLAS_GRID.x / f32(textureDimensions(sun_dynamic_shadow_tex).x);
    let in_near = all(abs(near - vec2<f32>(0.5)) < vec2<f32>(0.5 - edge));
    return vec4<f32>(select(c.xy, near, in_near), c.z, select(0.0, SUN_NEAR_TILE, in_near));
}}

fn pcf_layer(tex: texture_depth_2d, layer: i32, world_pos: vec3<f32>, light_view_proj: mat4x4<f32>) -> f32 {{
    let c = shadow_coords(world_pos, light_view_proj);
    if (c.w < 0.5) {{ return 1.0; }}
    return pcf_layer_at(tex, layer, c.xyz);
}}

fn pcf_layer_at(tex: texture_depth_2d, layer: i32, c: vec3<f32>) -> f32 {{
    return pcf_tile_tent_lean_at(
        tex,
        vec2<f32>(f32(layer % {atlas_cols}), f32(layer / {atlas_cols})),
        vec2<f32>(f32({atlas_cols}), f32({atlas_rows})),
        c,
    );
}}

fn pcf_layer_tap(tex: texture_depth_2d, layer: i32, world_pos: vec3<f32>, light_view_proj: mat4x4<f32>) -> f32 {{
    let c = shadow_coords(world_pos, light_view_proj);
    if (c.w < 0.5) {{ return 1.0; }}
    let grid = vec2<f32>(f32({atlas_cols}), f32({atlas_rows}));
    let lo = grid / vec2<f32>(textureDimensions(tex)) * 0.5;
    let tile = vec2<f32>(f32(layer % {atlas_cols}), f32(layer / {atlas_cols}));
    return textureSampleCompareLevel(tex, shadow_samp, (clamp(c.xy, lo, vec2<f32>(1.0) - lo) + tile) / grid, c.z);
}}

struct PcfTentAxis {{
    off: vec3<f32>,
    w: vec3<f32>,
}}
fn pcf_tent_axis(f: f32) -> PcfTentAxis {{
    let w = vec3<f32>((1.5 - f) * (1.5 - f) * 0.08, 0.0, (1.5 + f) * (1.5 + f) * 0.08);
    let w_mid = 1.0 - w.x - w.z;

    let fp = max(f, 0.0);
    let right = vec3<f32>((2.0 - 2.0 * f) * 0.08, (4.0 + 2.0 * f - 2.0 * fp * fp) * 0.08, (0.5 + f) * (0.5 + f) * 0.08);
    let weights = vec3<f32>(w.x, w_mid, w.z);
    return PcfTentAxis(vec3<f32>(-2.5, -0.5, 1.5) + right / weights, weights);
}}

fn pcf_tile_tent_unrolled_at(tex: texture_depth_2d, tile: vec2<f32>, grid: vec2<f32>, c: vec3<f32>) -> f32 {{
    let tile_texel = grid / vec2<f32>(textureDimensions(tex));
    let lo = tile_texel * 0.5;
    let hi = vec2<f32>(1.0) - lo;
    let p = c.xy / tile_texel;
    let corner = floor(p + vec2<f32>(0.5));
    let ax = pcf_tent_axis(p.x - corner.x);
    let ay = pcf_tent_axis(p.y - corner.y);
    let u = (vec3<f32>(corner.x) + ax.off) * tile_texel.x;
    let v = (vec3<f32>(corner.y) + ay.off) * tile_texel.y;
    var sum = 0.0;
    sum += ax.w.x * ay.w.x * pcf_tent_tap(tex, tile, grid, vec2<f32>(u.x, v.x), lo, hi, c.z);
    sum += ax.w.y * ay.w.x * pcf_tent_tap(tex, tile, grid, vec2<f32>(u.y, v.x), lo, hi, c.z);
    sum += ax.w.z * ay.w.x * pcf_tent_tap(tex, tile, grid, vec2<f32>(u.z, v.x), lo, hi, c.z);
    sum += ax.w.x * ay.w.y * pcf_tent_tap(tex, tile, grid, vec2<f32>(u.x, v.y), lo, hi, c.z);
    sum += ax.w.y * ay.w.y * pcf_tent_tap(tex, tile, grid, vec2<f32>(u.y, v.y), lo, hi, c.z);
    sum += ax.w.z * ay.w.y * pcf_tent_tap(tex, tile, grid, vec2<f32>(u.z, v.y), lo, hi, c.z);
    sum += ax.w.x * ay.w.z * pcf_tent_tap(tex, tile, grid, vec2<f32>(u.x, v.z), lo, hi, c.z);
    sum += ax.w.y * ay.w.z * pcf_tent_tap(tex, tile, grid, vec2<f32>(u.y, v.z), lo, hi, c.z);
    sum += ax.w.z * ay.w.z * pcf_tent_tap(tex, tile, grid, vec2<f32>(u.z, v.z), lo, hi, c.z);
    return sum;
}}

fn pcf_tile_tent_at(tex: texture_depth_2d, tile: vec2<f32>, grid: vec2<f32>, c: vec3<f32>) -> f32 {{
    let tile_texel = grid / vec2<f32>(textureDimensions(tex));
    let lo = tile_texel * 0.5;
    let hi = vec2<f32>(1.0) - lo;
    let p = c.xy / tile_texel;
    let corner = floor(p + vec2<f32>(0.5));
    let ax = pcf_tent_axis(p.x - corner.x);
    let ay = pcf_tent_axis(p.y - corner.y);
    let u = (vec3<f32>(corner.x) + ax.off) * tile_texel.x;
    let v = (vec3<f32>(corner.y) + ay.off) * tile_texel.y;
    var sum = 0.0;
    for (var j = 0; j < 3; j = j + 1) {{
        let vj = select(select(v.z, v.y, j == 1), v.x, j == 0);
        let wj = select(select(ay.w.z, ay.w.y, j == 1), ay.w.x, j == 0);
        for (var i = 0; i < 3; i = i + 1) {{
            let ui = select(select(u.z, u.y, i == 1), u.x, i == 0);
            let wi = select(select(ax.w.z, ax.w.y, i == 1), ax.w.x, i == 0);
            sum += wi * wj * pcf_tent_tap(tex, tile, grid, vec2<f32>(ui, vj), lo, hi, c.z);
        }}
    }}
    return sum;
}}

fn pcf_tent_tap_axis(k: f32, f: f32) -> vec2<f32> {{
    let a = 1.5 + k * f;
    let s = 1.0 - 1.0 / a;
    return select(
        vec2<f32>(k * (1.5 + s * s), 0.08 * a * a),
        vec2<f32>(f / (4.0 + 2.0 * abs(f)), 0.16 * (4.0 - f * f)),
        k == 0.0,
    );
}}

fn pcf_tile_tent_lean_at(tex: texture_depth_2d, tile: vec2<f32>, grid: vec2<f32>, c: vec3<f32>) -> f32 {{
    let tile_texel = grid / vec2<f32>(textureDimensions(tex));
    let lo = tile_texel * 0.5;
    let hi = vec2<f32>(1.0) - lo;
    let p = c.xy / tile_texel;
    let corner = floor(p + vec2<f32>(0.5));
    let f = p - corner;
    var sum = 0.0;
    for (var t = 0; t < 9; t = t + 1) {{
        let x = pcf_tent_tap_axis(f32(t % 3) - 1.0, f.x);
        let y = pcf_tent_tap_axis(f32(t / 3) - 1.0, f.y);
        sum += x.y * y.y * pcf_tent_tap(tex, tile, grid, (corner + vec2<f32>(x.x, y.x)) * tile_texel, lo, hi, c.z);
    }}
    return sum;
}}

fn pcf_tile_tent_half_at(tex: texture_depth_2d, tile: vec2<f32>, grid: vec2<f32>, c: vec3<f32>) -> f32 {{
    let tile_texel = grid / vec2<f32>(textureDimensions(tex));
    let lo = tile_texel * 0.5;
    let hi = vec2<f32>(1.0) - lo;
    let p = c.xy / tile_texel;
    let corner = floor(p + vec2<f32>(0.5));
    let ax = pcf_tent_axis(p.x - corner.x);
    let ay = pcf_tent_axis(p.y - corner.y);
    let ox = hf3(ax.off);
    let oy = hf3(ay.off);
    let wx = hf3(ax.w);
    let wy = hf3(ay.w);
    var sum = 0.0;
    for (var j = 0; j < 3; j = j + 1) {{
        let vj = (corner.y + f32(select(select(oy.z, oy.y, j == 1), oy.x, j == 0))) * tile_texel.y;
        let wj = select(select(wy.z, wy.y, j == 1), wy.x, j == 0);
        for (var i = 0; i < 3; i = i + 1) {{
            let ui = (corner.x + f32(select(select(ox.z, ox.y, i == 1), ox.x, i == 0))) * tile_texel.x;
            let wi = select(select(wx.z, wx.y, i == 1), wx.x, i == 0);
            sum += f32(wi * wj) * pcf_tent_tap(tex, tile, grid, vec2<f32>(ui, vj), lo, hi, c.z);
        }}
    }}
    return sum;
}}

fn pcf_tent_tap(tex: texture_depth_2d, tile: vec2<f32>, grid: vec2<f32>, local: vec2<f32>, lo: vec2<f32>, hi: vec2<f32>, depth: f32) -> f32 {{
    return textureSampleCompareLevel(tex, shadow_samp, (clamp(local, lo, hi) + tile) / grid, depth);
}}

fn pcf_tile(tex: texture_depth_2d, tile: vec2<f32>, grid: vec2<f32>, world_pos: vec3<f32>, light_view_proj: mat4x4<f32>) -> f32 {{
    let c = shadow_coords(world_pos, light_view_proj);
    if (c.w < 0.5) {{ return 1.0; }}
    return pcf_tile_at(tex, tile, grid, c.xyz);
}}

fn pcf_tile_at(tex: texture_depth_2d, tile: vec2<f32>, grid: vec2<f32>, c: vec3<f32>) -> f32 {{

    let tile_texel = grid / vec2<f32>(textureDimensions(tex));

    let guard = tile_texel * 0.5;
    let lo = guard;
    let hi = vec2<f32>(1.0) - guard;

    var sum = 0.0;
    for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {{
        for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {{

            let off = vec2<f32>(f32(dx), f32(dy)) * tile_texel;
            let local = clamp(c.xy + off, lo, hi);
            let uv = (local + tile) / grid;
            sum = sum + textureSampleCompareLevel(tex, shadow_samp, uv, c.z);
        }}
    }}
    return sum / 9.0;
}}

const SUN_ATLAS_GRID: vec2<f32> = vec2<f32>(f32({sun_atlas_tiles}), 1.0);

const SUN_NEAR_TILE: f32 = f32({sun_near_tile});
const SUN_NEAR_ZOOM: f32 = {sun_near_zoom:?};

fn character_shadow(layer: i32, world_pos: vec3<f32>) -> f32 {{
    let k = character_shadow_tile(layer);
    if (k < 0) {{
        return 1.0;
    }}
    return pcf_tile(
        sun_dynamic_shadow_tex, vec2<f32>(f32(1 + k), 0.0), SUN_ATLAS_GRID, world_pos,
        spot_view_proj({max_spot_shadows} + k),
    );
}}

fn character_shadow_tile(layer: i32) -> i32 {{
    return select(-1, layer - {max_spot_shadows}, layer >= {max_spot_shadows});
}}

fn light_contribution(l: Light, world_pos: vec3<f32>, n: vec3<f32>, view_dir: vec3<f32>) -> vec3<f32> {{
    return light_contribution_rough(l, world_pos, n, view_dir, SHININESS, SPEC_STRENGTH);
}}

struct LightSplit {{
    diffuse: vec3<f32>,
    specular: vec3<f32>,
}}

fn terminator_aa(nl: f32) -> vec2<f32> {{
    let w = select(terminator_width, 0.0, lights.count.w != 0u);
    let facing = clamp((nl + w) / max(2.0 * w, 1e-6), 0.0, 1.0);
    return vec2<f32>(select(max(nl, 0.0), 0.5 * (nl + w) * facing, abs(nl) < w), facing);
}}

const TERMINATOR_MAX_WIDTH: f32 = 0.5;
fn terminator_width_of(normal_length: f32) -> f32 {{
    let sigma2 = max(1.0 - normal_length - 0.01, 0.0) / max(normal_length, 1e-3);
    return min(sqrt(1.5 * sigma2), TERMINATOR_MAX_WIDTH);
}}

fn lamp_highlight_share(view_dir: vec3<f32>, h: vec3<f32>, nl: f32, facing: f32, shininess: f32, spec_strength: f32) -> f32 {{
    let g = 1.0 - clamp(dot(view_dir, h), 0.0, 1.0);
    let g2 = g * g;
    let fresnel = 0.04 + 0.96 * (g2 * g2 * g);
    return min(facing, fresnel * (shininess + 8.0) * 0.125 * nl / max(spec_strength, 1e-4));
}}

fn light_contribution_split(
    l: Light,
    world_pos: vec3<f32>,
    n: vec3<f32>,
    view_dir: vec3<f32>,
    shininess: f32,
    spec_strength: f32,
) -> LightSplit {{
    var out: LightSplit;
    out.diffuse = vec3<f32>(0.0);
    out.specular = vec3<f32>(0.0);

    let kind = l.params.z;
    var l_dir: vec3<f32>;
    var atten: f32;
    if (kind > 1.5) {{
        l_dir = normalize(-l.direction.xyz);
        atten = 1.0;
    }} else {{
        let to_light = l.position.xyz - world_pos;
        let dist = length(to_light);
        l_dir = to_light / max(dist, 0.0001);
        let d_over_r = dist / max(l.params.x, 0.0001);

        let d2_over_r2 = d_over_r * d_over_r;
        let window = clamp(1.0 - d2_over_r2 * d2_over_r2, 0.0, 1.0);

        let source = max(-2.0 - l.params.w, 0.0);
        atten = (window * window) / max(dist * dist + source * source, LAMP_RADIUS * LAMP_RADIUS);
        if (kind > 0.5) {{
            let cos_outer = l.params.y;
            let cos_inner = l.direction.w;
            let cos_angle = dot(-l_dir, l.direction.xyz);
            atten = atten * spot_cone(cos_angle, cos_outer, cos_inner, dist, spot_cone_across(l.direction.xyz, l_dir, cos_angle, dist, n, view_dir));
        }}
    }}

    let aa = terminator_aa(dot(n, l_dir));
    let ndotl = aa.x;
    let radiance = l.color_intensity.rgb * l.color_intensity.a;
    out.diffuse = radiance * ndotl * atten;

    if (aa.y > 0.0 && atten > 0.0 && l.params.w > -1.5) {{
        let h = normalize(l_dir + view_dir);
        let spec = pow(max(dot(n, h), 0.0), shininess) * spec_strength;
        out.specular = radiance * spec * atten * lamp_highlight_share(view_dir, h, ndotl, aa.y, shininess, spec_strength);
    }}
    return out;
}}

fn surface_lights(world_pos: vec3<f32>, n: vec3<f32>) -> vec3<f32> {{
    var lit = vec3<f32>(0.0);

    var todo = (1u << surface_light_count()) - 1u;
    loop {{
        if (todo == 0u) {{
            break;
        }}
        let i = countTrailingZeros(todo);
        todo = todo & (todo - 1u);
        let to_light = lights.lights[i].position.xyz - world_pos;
        let dist_sq = dot(to_light, to_light);
        let range = lights.lights[i].params.x;

        if (dist_sq >= range * range) {{
            continue;
        }}
        let l_dir = to_light * inverseSqrt(max(dist_sq, 1e-8));
        let d2_over_r2 = dist_sq / max(range * range, 1e-8);
        let window = clamp(1.0 - d2_over_r2 * d2_over_r2, 0.0, 1.0);
        let source = max(-2.0 - lights.lights[i].params.w, 0.0);
        var atten = (window * window) / max(dist_sq + source * source, LAMP_RADIUS * LAMP_RADIUS);
        if (lights.lights[i].params.z > 0.5) {{
            let cos_outer = lights.lights[i].params.y;
            let s = clamp((dot(-l_dir, lights.lights[i].direction.xyz) - cos_outer) / (lights.lights[i].direction.w - cos_outer), 0.0, 1.0);
            atten = atten * s * s * (3.0 - 2.0 * s);
        }}
        let radiance = lights.lights[i].color_intensity.rgb * lights.lights[i].color_intensity.a;
        lit = lit + radiance * (terminator_aa(dot(n, l_dir)).x * atten);
    }}
    return lit;
}}

fn light_contribution_rough(
    l: Light,
    world_pos: vec3<f32>,
    n: vec3<f32>,
    view_dir: vec3<f32>,
    shininess: f32,
    spec_strength: f32,
) -> vec3<f32> {{

    let kind = l.params.z;

    var l_dir: vec3<f32>;
    var atten: f32;
    if (kind > 1.5) {{

        l_dir = normalize(-l.direction.xyz);
        atten = 1.0;
    }} else {{
        let to_light = l.position.xyz - world_pos;
        let dist = length(to_light);
        l_dir = to_light / max(dist, 0.0001);

        let d_over_r = dist / max(l.params.x, 0.0001);

        let d2_over_r2 = d_over_r * d_over_r;
        let window = clamp(1.0 - d2_over_r2 * d2_over_r2, 0.0, 1.0);

        let source = max(-2.0 - l.params.w, 0.0);
        atten = (window * window) / max(dist * dist + source * source, LAMP_RADIUS * LAMP_RADIUS);

        if (kind > 0.5) {{
            let cos_outer = l.params.y;
            let cos_inner = l.direction.w;
            let cos_angle = dot(-l_dir, l.direction.xyz);
            atten = atten * spot_cone(cos_angle, cos_outer, cos_inner, dist, spot_cone_across(l.direction.xyz, l_dir, cos_angle, dist, n, view_dir));
        }}
    }}

    let ndotl = max(dot(n, l_dir), 0.0);
    let radiance = l.color_intensity.rgb * l.color_intensity.a;
    var out = radiance * ndotl * atten;

    if (ndotl > 0.0 && atten > 0.0 && l.params.w > -1.5) {{
        let h = normalize(l_dir + view_dir);
        let spec = pow(max(dot(n, h), 0.0), shininess) * spec_strength;
        out = out + radiance * spec * atten;
    }}
    return out;
}}

fn shade_material(world_pos: vec3<f32>, n: vec3<f32>, roughness: f32, ao: f32, sky_vis: f32) -> vec3<f32> {{

    return shade_material_env(
        world_pos, n, roughness, ao, sky_vis, vec3<f32>(0.0), vec4<f32>(0.5, 0.5, 0.5, 0.0),

        vec3<f32>(1.0),

        world_pos,

        n,
    );
}}

fn light_debug(world_pos: vec3<f32>, n: vec3<f32>) -> vec2<f32> {{
    let view_dir = normalize(cam_pos() - world_pos);
    var pre = 0.0;
    var post = 0.0;
    for (var i: u32 = 0u; i < live_light_count(); i = i + 1u) {{
        let l = lights.lights[i];
        if (l.params.z > 1.5) {{ continue; }}
        let c = light_contribution_split(l, world_pos, n, view_dir, 32.0, 0.2);
        let lum = dot(c.diffuse, vec3<f32>(0.2126, 0.7152, 0.0722));
        var shadow = stationary_visibility(l);
        let layer = i32(l.params.w);
        if (SPOT_SHADOWS && layer >= 0 && f32(layer) < camera.shadow_params.y) {{
            shadow = shadow * pcf_layer(spot_shadow_tex, layer, world_pos, spot_view_proj(layer));
        }}
        pre = pre + lum;
        post = post + lum * shadow;
    }}
    return vec2<f32>(pre, post);
}}

const SPECULAR_AA_SIGMA: f32 = 0.5;

const SPECULAR_AA_CLAMP: f32 = 0.18;

fn specular_aa_roughness(roughness: f32, dndx: vec3<f32>, dndy: vec3<f32>) -> f32 {{
    let variance = SPECULAR_AA_SIGMA * (dot(dndx, dndx) + dot(dndy, dndy));
    let kernel = min(2.0 * variance, SPECULAR_AA_CLAMP);
    return sqrt(clamp(roughness * roughness + kernel, 0.0, 1.0));
}}

var<private> probe_face_given: vec4<f32> = vec4<f32>(0.0);

fn probe_face_room(face_pos: vec3<f32>) -> vec4<f32> {{
    let volume_world = to_world_space(face_pos);
    let count = i32(camera.probe_params.x);
    var best_volume = 1e30;
    var best_room = -1.0;
    var held = 0.0;
    for (var i = 0; i < count; i = i + 1) {{
        let lo = camera.probe_boxes[i * 3 + 1].xyz;
        let hi = camera.probe_boxes[i * 3 + 2].xyz;
        if (any(volume_world < lo - vec3<f32>(PROBE_BOX_MARGIN)) || any(volume_world > hi + vec3<f32>(PROBE_BOX_MARGIN))) {{
            continue;
        }}
        let d = hi - lo;
        let volume = d.x * d.y * d.z;
        if (volume < best_volume * 0.999) {{
            best_volume = volume;
            best_room = camera.probe_boxes[i * 3 + 2].w;
            held = 1.0;
        }}
    }}
    let doorway = select(0.0, 1.0, probe_portal_holding(volume_world) >= 0);
    return vec4<f32>(best_room, held, doorway, 1.0);
}}

struct ProbeChoice {{

    best: i32,
    second: i32,
    best_dist: f32,
    second_dist: f32,

    room: f32,
}}

fn probe_choose(select_world: vec3<f32>, volume_world: vec3<f32>) -> ProbeChoice {{
    let count = i32(camera.probe_params.x);

    var best = -1;
    var best_volume = 1e30;
    var best_dist = 1e30;
    var best_room = -1.0;

    var second = -1;
    var second_dist = 1e30;

    let face_given = PROBE_FACE_ALWAYS_GIVEN || probe_face_given.w > 0.5;
    var c: ProbeChoice;
    c.best = -1;
    c.second = -1;
    c.best_dist = 1e30;
    c.second_dist = 1e30;
    c.room = -1.0;
    if (face_given && probe_face_given.y > 0.5) {{
        c = probe_choose_in_room(probe_face_given.x, select_world);
    }}
    for (var i = 0; i < count && !face_given; i = i + 1) {{
        let lo = camera.probe_boxes[i * 3 + 1].xyz;
        let hi = camera.probe_boxes[i * 3 + 2].xyz;

        if (any(volume_world < lo - vec3<f32>(PROBE_BOX_MARGIN)) || any(volume_world > hi + vec3<f32>(PROBE_BOX_MARGIN))) {{
            continue;
        }}
        let d = hi - lo;
        let volume = d.x * d.y * d.z;
        let to_centre = camera.probe_boxes[i * 3].xyz - select_world;
        let dist = dot(to_centre, to_centre);

        let room = camera.probe_boxes[i * 3 + 2].w;
        let tighter = volume < best_volume * 0.999;
        let same_room_but_nearer = room == best_room && dist < best_dist;
        if (tighter) {{

            second = -1;
            second_dist = 1e30;
            best_volume = volume;
            best_dist = dist;
            best_room = room;
            best = i;
        }} else if (same_room_but_nearer) {{

            second = best;
            second_dist = best_dist;
            best_dist = dist;
            best = i;
        }} else if (room == best_room && dist < second_dist) {{
            second = i;
            second_dist = dist;
        }}
    }}

    if (!face_given) {{
        c.best = best;
        c.second = second;
        c.best_dist = best_dist;
        c.second_dist = second_dist;
        c.room = best_room;
    }}
    return c;
}}

fn probe_choose_in_room(room: f32, select_world: vec3<f32>) -> ProbeChoice {{
    var c: ProbeChoice;
    c.best = -1;
    c.second = -1;
    c.best_dist = 1e30;
    c.second_dist = 1e30;
    c.room = room;

    for (var i = probe_room_slot(room); i >= 0; i = probe_slot_next(i)) {{
        let to_centre = camera.probe_boxes[i * 3].xyz - select_world;
        let dist = dot(to_centre, to_centre);
        if (dist < c.best_dist) {{
            c.second = c.best;
            c.second_dist = c.best_dist;
            c.best_dist = dist;
            c.best = i;
        }} else if (dist < c.second_dist) {{
            c.second = i;
            c.second_dist = dist;
        }}
    }}
    return c;
}}

fn probe_through_far_end(hit: ProbeHit, d: vec3<f32>, through: vec4<f32>, roughness: f32, dir: vec3<f32>, probe_lod: f32) -> vec4<f32> {{
    let p = (hit.rim_code >> 3u) & 31;
    let axis = (hit.rim_code >> 1u) & 3;

    if (abs(d[axis]) < 1e-4) {{
        return through;
    }}
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
    if (abs(min(in_a, in_b)) >= spread) {{
        return through;
    }}
    let far = smoothstep(-spread, spread, in_a) * smoothstep(-spread, spread, in_b);
    let share = clamp(far / max(hit.rim, 1e-3), 0.0, 1.0);

    let out = min(in_a, in_b) > 0.0;
    var other = probe_point_hit(probe_rim_point_far(e2, p, axis, false), probe_hit_rim_room(hit), t);
    if (!out) {{
        other = probe_trace(probe_rim_point_far(e2, p, axis, true), d, -1.0, roughness);
        other.t = t + other.t;
    }}
    if (!other.found) {{
        return through;
    }}
    let y = probe_traced_colour(other, d, roughness, dir, probe_lod);
    return select(mix(through, y, share), mix(y, through, share), out);
}}

fn probe_secondary(
    hit: ProbeHit,
    primary: vec4<f32>,
    world_pos: vec3<f32>,
    d: vec3<f32>,
    dir: vec3<f32>,
    roughness: f32,
    probe_lod: f32,
    trace_room: f32,
) -> vec4<f32> {{
    var col = primary;

    if (hit.rim >= 0.0) {{

        let rim_pos = probe_hit_rim_point(hit, d);
        let went_through = probe_hit_rim_went_through(hit);
        var side = probe_point_hit(rim_pos, probe_hit_rim_room(hit), hit.rim_t);
        if (!went_through) {{
            side = probe_trace(rim_pos, d, -1.0, roughness);
            side.t = hit.rim_t + side.t;
        }}
        if (side.found) {{
            let x = probe_traced_colour(side, d, roughness, dir, probe_lod);

            var through = select(x, col, went_through);
            let beside = select(col, x, went_through);

            if ((hit.rim_code & PROBE_RIM_FAR) == 0 && roughness < PROBE_FAR_END_MAX_ROUGHNESS) {{
                through = probe_through_far_end(hit, d, through, roughness, dir, probe_lod);
            }}
            col = mix(beside, through, hit.rim);
        }}
    }}

    let edge_cover = probe_edge_cover(hit.edge_cover);
    if (hit.edge_code >= 0 && edge_cover < 0.99) {{

        let edge_hit = probe_hit_edge_hit(hit);
        var side = probe_point_hit(hit.origin + d * hit.edge_t, probe_hit_edge_room(hit), hit.edge_t);
        side.other = probe_proxy_model(probe_hit_edge(hit));
        if (edge_hit) {{
            side = probe_trace_skipping(world_pos, d, trace_room, roughness, probe_hit_edge(hit));
        }}
        if (side.found) {{
            let x = probe_traced_colour(side, d, roughness, dir, probe_lod);
            if (edge_hit) {{
                col = mix(x, col, edge_cover);
            }} else {{
                col = mix(col, x, edge_cover);
            }}
        }}
    }}
    return col;
}}

fn probe_environment(
    frag_pos: vec3<f32>,
    dir: vec3<f32>,
    roughness: f32,

    select_pos: vec3<f32>,
) -> vec4<f32> {{

    let world_pos = to_world_space(frag_pos);

    let select_world = to_world_space(select_pos);
    let volume_world = select(select_world, to_world_space(probe_volume_pos.xyz), probe_volume_pos.w > 0.5);
    let world_dir = to_world_direction(dir);
    probe_brightness = 0.0;

    if (PROBE_OPEN_GROUND) {{

        probe_eye_distance = distance(cam_pos(), frag_pos);
        if (roughness > PROBE_TRACE_MAX_ROUGHNESS) {{
            let unlit = vec4<f32>(0.0);
            return unlit;
        }}
        let open_dir = normalize(world_dir);
        let open_lod = clamp(roughness * PROBE_ROUGHNESS_MIPS, PROBE_MIN_LOD, PROBE_MAX_LOD);
        let open_col = probe_escape_colour(world_pos, open_dir, dir, roughness, open_lod);
        return open_col;
    }}
    if (PROBE_NOWHERE) {{

        probe_eye_distance = distance(cam_pos(), frag_pos);
        let nothing = vec4<f32>(0.0);
        return nothing;
    }}
    if (PROBE_ALL_OUTDOORS) {{

        let away = normalize(world_dir);
        let away_lod = clamp(roughness * PROBE_ROUGHNESS_MIPS, PROBE_MIN_LOD, PROBE_MAX_LOD);
        probe_eye_distance = distance(cam_pos(), frag_pos);

        if (roughness > PROBE_TRACE_MAX_ROUGHNESS) {{
            return probe_outdoor_untraced(probe_choose(select_world, volume_world), world_pos, away, away_lod);
        }}
        let col = probe_escape_colour(world_pos, away, dir, roughness, away_lod);
        probe_reach = 0.0 + probe_reach;
        return col;
    }}
    let face_given = PROBE_FACE_ALWAYS_GIVEN || probe_face_given.w > 0.5;

    var in_doorway = false;
    if (face_given) {{
        in_doorway = probe_face_given.z > 0.5;
    }} else {{
        in_doorway = probe_portal_holding(volume_world) >= 0;
    }}

    let d = normalize(world_dir);
    let probe_lod = clamp(roughness * PROBE_ROUGHNESS_MIPS, PROBE_MIN_LOD, PROBE_MAX_LOD);

    var trace_room = -1.0;
    if (face_given && probe_face_given.y > 0.5) {{
        trace_room = probe_face_given.x;
    }}
    if (!face_given || in_doorway) {{
        let early = probe_choose(select_world, volume_world);
        trace_room = early.room;
        if (in_doorway && (early.best < 0 || probe_seen_distance(early.best, d) < 0.0)) {{
            trace_room = -1.0;
        }}
    }}
    probe_eye_distance = distance(cam_pos(), frag_pos);

    let hit = probe_trace(world_pos, d, trace_room, roughness);
    if (hit.found) {{

        let recolour = PROBE_SECONDARY_DEFERRED && probe_hit_carded(hit);
        let secondary = hit.rim >= 0.0 || hit.edge_code >= 0 || recolour;
        var slot = -1;
        if (PROBE_SECONDARY_DEFERRED && secondary) {{
            slot = probe_fixup_begin(hit, world_pos, d, dir, roughness, probe_lod, trace_room, recolour);
        }}
        let col = probe_traced_colour(hit, d, roughness, dir, probe_lod);
        if (!secondary) {{
            return col;
        }}
        if (PROBE_SECONDARY_DEFERRED) {{
            probe_fixup_finish(slot, col);
            return col;
        }}
        return probe_secondary(hit, col, world_pos, d, dir, roughness, probe_lod, trace_room);
    }}
    let choice = probe_choose(select_world, volume_world);
    let best = choice.best;
    let second = choice.second;
    let best_dist = choice.best_dist;
    let second_dist = choice.second_dist;
    let best_room = choice.room;

    var own = vec4<f32>(0.0);
    var own_room = -1.0;
    if (best >= 0) {{
        own_room = best_room;

        var band = 1.0;
        if (second >= 0) {{
            let gap = sqrt(second_dist) - sqrt(best_dist);
            band = 0.5 + 0.5 * smoothstep(0.0, PROBE_BLEND_BAND, gap);
        }}

        let sample_dir = probe_parallax_direction(world_pos, d, best);
        let layer = i32(camera.probe_boxes[best * 3].w);
        let near = textureSampleLevel(probe_cube, probe_samp, sample_dir, layer, probe_lod);
        probe_brightness = camera.probe_boxes[best * 3 + 1].w;
        own = near;
        if (second >= 0) {{
            let far_dir = probe_parallax_direction(world_pos, d, second);
            let far = textureSampleLevel(
                probe_cube, probe_samp, far_dir, i32(camera.probe_boxes[second * 3].w), probe_lod
            );

            probe_brightness = mix(camera.probe_boxes[second * 3 + 1].w, camera.probe_boxes[best * 3 + 1].w, band);
            own = mix(far, near, band);
        }}
    }}
    return probe_through_portals(own, own_room, select_world, world_pos, d, probe_lod);
}}

fn probe_outdoor_untraced(pick: ProbeChoice, world_pos: vec3<f32>, away: vec3<f32>, away_lod: f32) -> vec4<f32> {{
    var mine = vec4<f32>(0.0);
    if (pick.best >= 0) {{
        var share = 1.0;
        if (pick.second >= 0) {{
            let apart = sqrt(pick.second_dist) - sqrt(pick.best_dist);
            share = 0.5 + 0.5 * smoothstep(0.0, PROBE_BLEND_BAND, apart);
        }}
        let near_dir = probe_parallax_direction(world_pos, away, pick.best);
        let near_layer = i32(camera.probe_boxes[pick.best * 3].w);
        let near_photo = textureSampleLevel(probe_cube, probe_samp, near_dir, near_layer, away_lod);
        probe_brightness = camera.probe_boxes[pick.best * 3 + 1].w;
        mine = near_photo;
        if (pick.second >= 0) {{
            let far_photo_dir = probe_parallax_direction(world_pos, away, pick.second);
            let far_photo = textureSampleLevel(
                probe_cube, probe_samp, far_photo_dir, i32(camera.probe_boxes[pick.second * 3].w), away_lod
            );

            probe_brightness = mix(camera.probe_boxes[pick.second * 3 + 1].w, camera.probe_boxes[pick.best * 3 + 1].w, share);
            mine = mix(far_photo, near_photo, share);
        }}
    }}
    return mine;
}}

const PROBE_TRACE_MAX_ROUGHNESS: f32 = 0.75;

const PROBE_TRACE_ROOMS: i32 = 3;

const PROBE_SUBSAMPLE: f32 = -1.0;

const PROBE_RETEST: f32 = -4.0;

fn probe_edge_cover(edge_cover: f32) -> f32 {{
    let c = abs(edge_cover);
    return select(c, c + PROBE_RETEST, c > 2.0);
}}

const PROBE_PROXY_SAMPLES: i32 = 4;

const PROBE_SEEN_TOLERANCE: f32 = 0.03;

struct ProbeHit {{
    pos: vec3<f32>,
    room: f32,
    other: f32,
    found: bool,

    escaped: bool,
    portal: i32,

    t: f32,

    origin: vec3<f32>,

    rim: f32,
    rim_t: f32,
    rim_code: i32,

    edge_cover: f32,
    edge_t: f32,
    edge_code: i32,
}}

fn probe_hit_rim_went_through(h: ProbeHit) -> bool {{
    return (h.rim_code & 1) != 0;
}}
fn probe_hit_rim_room(h: ProbeHit) -> f32 {{
    return f32(((h.rim_code >> 8u) & 31) - 1);
}}

fn probe_hit_rim_point(h: ProbeHit, d: vec3<f32>) -> vec3<f32> {{
    let e = h.origin + d * h.rim_t;
    let p = (h.rim_code >> 3u) & 31;
    let axis = (h.rim_code >> 1u) & 3;
    let into = (h.rim_code & 1) == 0;
    if ((h.rim_code & PROBE_RIM_FAR) != 0) {{
        return probe_rim_point_far(e, p, axis, into);
    }}
    return probe_rim_point(e, p, axis, into);
}}

const PROBE_RIM_FAR: i32 = 8192;

fn probe_rim_point_far(e: vec3<f32>, p: i32, axis: i32, into: bool) -> vec3<f32> {{
    let plo = camera.probe_portals[p * 3].xyz;
    let phi = camera.probe_portals[p * 3 + 1].xyz;
    var q = e;
    let a = (axis + 1) % 3;
    let b = (axis + 2) % 3;
    q[axis] = clamp(e[axis], plo[axis] + PROBE_RIM_STEP, phi[axis] - PROBE_RIM_STEP);
    if (into) {{
        q[a] = clamp(e[a], plo[a] + PROBE_RIM_STEP, phi[a] - PROBE_RIM_STEP);
        q[b] = clamp(e[b], plo[b] + PROBE_RIM_STEP, phi[b] - PROBE_RIM_STEP);
        return q;
    }}
    let in_a = min(e[a] - plo[a], phi[a] - e[a]);
    let in_b = min(e[b] - plo[b], phi[b] - e[b]);
    if (in_a <= in_b) {{
        q[a] = select(phi[a], plo[a], e[a] - plo[a] < phi[a] - e[a]);
    }} else {{
        q[b] = select(phi[b], plo[b], e[b] - plo[b] < phi[b] - e[b]);
    }}
    return q;
}}

fn probe_hit_edge(h: ProbeHit) -> i32 {{
    return select(-1, (h.edge_code >> 1u) & 31, h.edge_code >= 0);
}}
fn probe_hit_edge_hit(h: ProbeHit) -> bool {{
    return h.edge_code >= 0 && (h.edge_code & 1) != 0;
}}
fn probe_hit_edge_room(h: ProbeHit) -> f32 {{
    return f32((h.edge_code >> 6u) - 1);
}}

var<private> probe_eye_distance: f32 = 1.0;

struct ProbeProxyHit {{
    t: f32,
    index: i32,
    edge: i32,
    edge_cover: f32,
    edge_t: f32,
}}

const PROBE_LOBE_SPREAD: f32 = 1.3;
fn probe_lobe_tan(roughness: f32) -> f32 {{
    return PROBE_LOBE_SPREAD * roughness * roughness;
}}

const PROBE_RIM_MIN_SPREAD: f32 = 0.01;

const PROBE_FAR_END_MAX_ROUGHNESS: f32 = 0.45;

const PROBE_EDGE_FOOTPRINTS: f32 = {probe_edge_footprints:?};

fn probe_pixel_spread(t: f32) -> f32 {{
    return PROBE_EDGE_FOOTPRINTS * pixel_footprint * (1.0 + t / max(probe_eye_distance, 0.05));
}}

const PROBE_RIM_STEP: f32 = 0.02;

struct ProbeRim {{
    portal: i32,
    through: f32,
}}

fn probe_rim_at(e: vec3<f32>, room: f32, axis: i32, spread: f32) -> ProbeRim {{
    var out: ProbeRim;
    out.portal = -1;
    out.through = 0.0;

    for (var p = probe_room_portal(room); p >= 0; p = probe_portal_next(p, room)) {{
        if (i32(camera.probe_portals[p * 3].w) != axis) {{
            continue;
        }}
        let plo = camera.probe_portals[p * 3].xyz;
        let phi = camera.probe_portals[p * 3 + 1].xyz;
        if (e[axis] < plo[axis] - 1e-3 || e[axis] > phi[axis] + 1e-3) {{
            continue;
        }}

        let a = (axis + 1) % 3;
        let b = (axis + 2) % 3;
        let in_a = min(e[a] - plo[a], phi[a] - e[a]);
        let in_b = min(e[b] - plo[b], phi[b] - e[b]);
        let nearest = min(in_a, in_b);
        if (nearest < -spread || nearest > spread) {{
            continue;
        }}
        out.portal = p;
        out.through = smoothstep(-spread, spread, in_a) * smoothstep(-spread, spread, in_b);
        return out;
    }}
    return out;
}}

fn probe_rim_point(e: vec3<f32>, p: i32, axis: i32, into: bool) -> vec3<f32> {{
    let plo = camera.probe_portals[p * 3].xyz;
    let phi = camera.probe_portals[p * 3 + 1].xyz;
    var q = e;
    let a = (axis + 1) % 3;
    let b = (axis + 2) % 3;
    if (into) {{
        q[a] = clamp(e[a], plo[a] + PROBE_RIM_STEP, phi[a] - PROBE_RIM_STEP);
        q[b] = clamp(e[b], plo[b] + PROBE_RIM_STEP, phi[b] - PROBE_RIM_STEP);
        return q;
    }}
    let in_a = min(e[a] - plo[a], phi[a] - e[a]);
    let in_b = min(e[b] - plo[b], phi[b] - e[b]);
    if (in_a <= in_b) {{
        q[a] = select(phi[a] + PROBE_RIM_STEP, plo[a] - PROBE_RIM_STEP, e[a] - plo[a] < phi[a] - e[a]);
    }} else {{
        q[b] = select(phi[b] + PROBE_RIM_STEP, plo[b] - PROBE_RIM_STEP, e[b] - plo[b] < phi[b] - e[b]);
    }}
    return q;
}}

fn probe_hit_lod(roughness: f32, t: f32, t_probe: f32) -> f32 {{
    let r = roughness * sqrt(max(t, 0.0) / max(t_probe, 0.05));
    return clamp(r * PROBE_ROUGHNESS_MIPS, PROBE_MIN_LOD, PROBE_MAX_LOD);
}}

fn probe_seen_distance(slot: i32, v: vec3<f32>) -> f32 {{
    let t = textureSampleLevel(probe_depth, probe_depth_samp, v, i32(camera.probe_boxes[slot * 3].w), 0.0);
    return probe_seen_distance_of(t, v);
}}

fn probe_seen_distance_of(t: vec4<f32>, v: vec3<f32>) -> f32 {{
    if (dot(t.xyz, t.xyz) < 0.25) {{
        return select(-1.0, 3.4e38, t.w > 0.5);
    }}
    let dn = dot(normalize(v), t.xyz);
    return select(3.4e38, max(t.w / dn, 0.0), dn < -1e-4);
}}

fn probe_clearance(slot: i32, p: vec3<f32>) -> f32 {{
    let v = p - camera.probe_boxes[slot * 3].xyz;
    let t = textureSampleLevel(probe_depth, probe_depth_samp, v, i32(camera.probe_boxes[slot * 3].w), 0.0);
    return probe_clearance_of(t, v);
}}

fn probe_clearance_of(t: vec4<f32>, v: vec3<f32>) -> f32 {{
    let s = probe_seen_distance_of(t, v);
    return select(s - length(v), -3.4e38, s < 0.0);
}}

fn probe_room_slot(room: f32) -> i32 {{
    if (!(room >= 0.0 && room < 16.0)) {{
        return -1;
    }}
    let r = i32(room);
    return i32(camera.probe_rooms[r >> 2u][r & 3]);
}}

fn probe_slot_next(slot: i32) -> i32 {{
    return i32(camera.probe_rooms[4 + (slot >> 2u)][slot & 3]);
}}

struct ProbeNearest {{
    s0: i32,
    s1: i32,
    d0: f32,
    d1: f32,
}}
const PROBE_MAX_SLOTS: i32 = {max_probes};

fn probe_nearest_two_unrolled(p: vec3<f32>, a: f32, b: f32) -> ProbeNearest {{
    let count = i32(camera.probe_params.x);
    var n = ProbeNearest(-1, -1, 3.4e38, 3.4e38);
{probe_nearest_unrolled}    return n;
}}
fn probe_nearest_two(p: vec3<f32>, a: f32, b: f32) -> ProbeNearest {{
    let count = i32(camera.probe_params.x);
    var n = ProbeNearest(-1, -1, 3.4e38, 3.4e38);
    for (var i = 0; i < PROBE_MAX_SLOTS; i = i + 1) {{
        if (i >= count) {{
            break;
        }}
        let room = camera.probe_boxes[i * 3 + 2].w;
        let v = camera.probe_boxes[i * 3].xyz - p;
        let dd = dot(v, v);
        let mine = room == a || room == b;
        if (mine && dd < n.d0) {{
            n.s1 = n.s0;
            n.d1 = n.d0;
            n.s0 = i;
            n.d0 = dd;
        }} else if (mine && dd < n.d1) {{
            n.s1 = i;
            n.d1 = dd;
        }}
    }}
    return n;
}}

fn probe_nearest_two_const_unrolled(p: vec3<f32>, a: f32, b: f32) -> ProbeNearest {{
    let count = i32(camera.probe_params.x);
    var n = ProbeNearest(-1, -1, 3.4e38, 3.4e38);
{probe_nearest_const_unrolled}    return n;
}}

fn probe_nearest_two_const(p: vec3<f32>, a: f32, b: f32) -> ProbeNearest {{
    let count = i32(camera.probe_params.x);
    var n = ProbeNearest(-1, -1, 3.4e38, 3.4e38);
    for (var i = 0; i < PROBE_MAX_SLOTS; i = i + 1) {{
        if (i >= count) {{
            break;
        }}
        let room = probe_select[i * 3 + 2].w;
        let v = probe_select[i * 3].xyz - p;
        let dd = dot(v, v);
        let mine = room == a || room == b;
        if (mine && dd < n.d0) {{
            n.s1 = n.s0;
            n.d1 = n.d0;
            n.s0 = i;
            n.d0 = dd;
        }} else if (mine && dd < n.d1) {{
            n.s1 = i;
            n.d1 = dd;
        }}
    }}
    return n;
}}

fn probe_room_portal(room: f32) -> i32 {{
    if (!(room >= 0.0 && room < 16.0)) {{
        return -1;
    }}
    let r = i32(room);
    return i32(camera.probe_rooms[8 + (r >> 2u)][r & 3]);
}}
fn probe_portal_next(p: i32, room: f32) -> i32 {{
    let k = p * 2 + select(1, 0, camera.probe_portals[p * 3 + 1].w == room);
    return i32(camera.probe_rooms[12 + (k >> 2u)][k & 3]);
}}

fn probe_room_proxy(room: f32) -> i32 {{
    if (!(room >= 0.0 && room < 16.0)) {{
        return -1;
    }}
    let r = i32(room);
    return i32(camera.probe_rooms[16 + (r >> 2u)][r & 3]);
}}
fn probe_proxy_next(i: i32) -> i32 {{
    return i32(camera.probe_rooms[20 + (i >> 2u)][i & 3]);
}}

fn probe_proxy_model(i: i32) -> f32 {{
    var m = -1.0;
    if (i >= 0 && camera.probe_proxies[i * 3 + 1].w > 0.5) {{
        m = -2.0 - f32(i);
    }}
    return m;
}}

fn probe_quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {{
    let t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}}

fn probe_proxy_hit(o: vec3<f32>, d: vec3<f32>, room: f32, t0: f32, t1: f32, skip: i32, lobe: f32) -> ProbeProxyHit {{
    var out: ProbeProxyHit;
    out.t = 3.4e38;
    out.index = -1;
    out.edge = -1;
    out.edge_cover = 0.0;
    out.edge_t = 0.0;
    var best = 3.4e38;

    var field_i = -1;
    var field_near = 3.4e38;
    var field_far = 0.0;

    for (var i = probe_room_proxy(room); i >= 0; i = probe_proxy_next(i)) {{
        if (i == skip) {{
            continue;
        }}

        let q = camera.probe_proxies[i * 3 + 2];
        var lo = o - camera.probe_proxies[i * 3].xyz;
        var ld = d;
        if (any(q != vec4<f32>(0.0, 0.0, 0.0, 1.0))) {{
            let qi = vec4<f32>(-q.xyz, q.w);
            lo = probe_quat_rotate(qi, lo);
            ld = probe_quat_rotate(qi, ld);
        }}

        let half = camera.probe_proxies[i * 3 + 1].xyz;
        let inv = 1.0 / select(vec3<f32>(1e-9), ld, abs(ld) > vec3<f32>(1e-9));
        let a = (-half - lo) * inv;
        let b = (half - lo) * inv;
        let lows = min(a, b);
        let highs = max(a, b);
        let near = max(max(lows.x, lows.y), lows.z);
        let far = min(min(highs.x, highs.y), highs.z);

        if (camera.probe_proxies[i * 3 + 1].w < 0.5 && out.edge < 0) {{
            var ax_in = 2;
            if (lows.x >= lows.y && lows.x >= lows.z) {{
                ax_in = 0;
            }} else if (lows.y >= lows.z) {{
                ax_in = 1;
            }}
            var ax_out = 2;
            if (highs.x <= highs.y && highs.x <= highs.z) {{
                ax_out = 0;
            }} else if (highs.y <= highs.z) {{
                ax_out = 1;
            }}
            let da = abs(ld[ax_in]);
            let db = abs(ld[ax_out]);
            let across = select(da * db / max(sqrt(da * da + db * db), 1e-6), 1.0, ax_in == ax_out);
            let inside = (far - near) * across;
            let t_edge = max(0.5 * (near + far), t0);
            let footprint = max(t_edge * lobe, probe_pixel_spread(t_edge));
            if (abs(inside) < footprint && t_edge > t0 && t_edge < t1 && far > t0 + 1e-3) {{
                out.edge = i;
                out.edge_cover = smoothstep(-footprint, footprint, inside);
                out.edge_t = t_edge;
            }}
        }}

        if (near <= far && far > t0 + 1e-3 && near < min(t1, best)) {{
            let t_in = max(near, t0);
            if (camera.probe_proxies[i * 3 + 1].w < 0.5) {{

                best = t_in;
                out.index = i;
            }} else if (camera.probe_proxies[i * 3 + 1].w > 1.5) {{

                if (t_in < field_near) {{
                    field_i = i;
                    field_near = t_in;
                    field_far = min(far, t1);
                }}
            }} else {{

                let t_surface = probe_proxy_surface(o, d, t_in, min(far, t1), room, camera.probe_proxies[i * 3].xyz);
                if (t_surface < best) {{
                    best = t_surface;
                    out.index = i;
                }}
            }}
        }}
    }}

    if (field_i >= 0 && field_near < best) {{
        let walk = probe_proxy_field_at(o, d, field_i, field_near, min(field_far, best), lobe);
        let touched = walk.x < best;
        if (touched) {{
            best = walk.x;
            out.index = field_i;
        }}

        let t_at = select(walk.y, walk.x, touched);
        let extent = (2.0 / 3.0) * dot(camera.probe_proxies[field_i * 3 + 1].xyz, vec3<f32>(1.0));
        let spread = t_at * lobe;
        let presence = extent * extent / max(extent * extent + spread * spread, 1e-8);
        let cover = select(presence * (1.0 - smoothstep(0.0, 1.0, walk.z)), presence, touched);

        let carded = camera.proxy_cards[field_i >> 2u][field_i & 3] > 0.5;
        if (out.edge < 0 && touched && walk.z < 0.0) {{

            out.edge = field_i;
            out.edge_cover = PROBE_SUBSAMPLE * presence;
            out.edge_t = walk.x;
        }} else if (out.edge < 0 && touched && carded) {{
            out.edge = field_i;
            out.edge_cover = PROBE_RETEST - presence;
            out.edge_t = walk.x;
        }} else if (out.edge < 0 && cover < 0.99 && cover > 0.004 && t_at > t0) {{
            out.edge = field_i;
            out.edge_cover = cover;
            out.edge_t = t_at;
        }}
    }}
    out.t = best;
    return out;
}}

fn probe_proxy_field_at(o: vec3<f32>, d: vec3<f32>, i: i32, t_in: f32, t_out: f32, lobe: f32) -> vec3<f32> {{
    let q = camera.probe_proxies[i * 3 + 2];
    var lo = o - camera.probe_proxies[i * 3].xyz;
    var ld = d;
    if (any(q != vec4<f32>(0.0, 0.0, 0.0, 1.0))) {{
        let qi = vec4<f32>(-q.xyz, q.w);
        lo = probe_quat_rotate(qi, lo);
        ld = probe_quat_rotate(qi, ld);
    }}
    let box = camera.probe_proxies[i * 3 + 1];
    return probe_proxy_field(lo, ld, box.xyz, t_in, t_out, i32(box.w) - 2, lobe);
}}

const PROXY_FIELD_STEPS: i32 = 32;

fn probe_proxy_field(lo: vec3<f32>, ld: vec3<f32>, half: vec3<f32>, t_in: f32, t_out: f32, field: i32, lobe: f32) -> vec3<f32> {{
    let slot = camera.proxy_fields[field * 3];
    let size = camera.proxy_fields[field * 3 + 1];

    let scale = size.xyz * (0.5 / max(half, vec3<f32>(1e-4)));
    let fo = slot.xyz + 0.5 * size.xyz + lo * scale;
    let fd = ld * scale;

    let texel = 0.5 / vec3<f32>(textureDimensions(proxy_field));
    let lo_uvw = slot.xyz + texel;
    let hi_uvw = slot.xyz + size.xyz - texel;
    var hit = 3.4e38;
    var near_t = t_in;
    var near = 3.4e38;
    var passed = false;
    var t = t_in;
    for (var k = 0; k < PROXY_FIELD_STEPS && t <= t_out; k = k + 1) {{
        let dist = textureSampleLevel(proxy_field, probe_samp, clamp(fo + fd * t, lo_uvw, hi_uvw), 0.0).r * slot.w;
        if (dist <= size.w) {{

            var th = t + (dist - size.w);
            th = th + (textureSampleLevel(proxy_field, probe_samp, clamp(fo + fd * th, lo_uvw, hi_uvw), 0.0).r * slot.w - size.w);
            hit = clamp(th, t_in, t_out);
            break;
        }}

        let footprint = min(max(t * lobe, probe_pixel_spread(t)), 4.0 * size.w);
        let r = (dist - size.w) / max(footprint, 1e-5);
        if (r < near) {{
            near = r;
            near_t = t;
        }}
        passed = passed || (near < 1.0 && r > near + 1.0);
        t = t + dist;
    }}
    return vec3<f32>(hit, near_t, select(near, -1.0, passed && hit < 3.0e38));
}}

fn probe_proxy_surface(o: vec3<f32>, d: vec3<f32>, t_in: f32, t_out: f32, room: f32, centre: vec3<f32>) -> f32 {{

    let near = probe_nearest_two(centre, room, room);
    let s0 = near.s0;
    let s1 = near.s1;
    if (s0 < 0) {{
        return 3.4e38;
    }}
    var prev_t = t_in;
    var prev_c = 0.0;
    for (var k = 0; k <= PROBE_PROXY_SAMPLES; k = k + 1) {{
        let t = mix(t_in, t_out, f32(k) / f32(PROBE_PROXY_SAMPLES));
        let p = o + d * t;
        var c = probe_clearance(s0, p);
        if (s1 >= 0 && c <= 0.0) {{
            c = max(c, probe_clearance(s1, p));
        }}
        if (c <= 0.0) {{
            if (k == 0) {{
                return t_in;
            }}
            return prev_t + (t - prev_t) * clamp(prev_c / max(prev_c - c, 1e-5), 0.0, 1.0);
        }}
        prev_t = t;
        prev_c = c;
    }}
    return 3.4e38;
}}

fn probe_portal_at(e: vec3<f32>, room: f32, axis: i32) -> i32 {{

    for (var p = probe_room_portal(room); p >= 0; p = probe_portal_next(p, room)) {{
        if (i32(camera.probe_portals[p * 3].w) != axis) {{
            continue;
        }}
        let lo = camera.probe_portals[p * 3].xyz - vec3<f32>(1e-3);
        let hi = camera.probe_portals[p * 3 + 1].xyz + vec3<f32>(1e-3);
        if (all(e >= lo) && all(e <= hi)) {{
            return p;
        }}
    }}
    return -1;
}}

fn probe_portal_holding(p: vec3<f32>) -> i32 {{
    let n = i32(camera.portal_params.x);
    for (var i = 0; i < n; i = i + 1) {{
        let lo = camera.probe_portals[i * 3].xyz - vec3<f32>(0.02);
        let hi = camera.probe_portals[i * 3 + 1].xyz + vec3<f32>(0.02);
        if (all(p >= lo) && all(p <= hi)) {{
            return i;
        }}
    }}
    return -1;
}}

fn probe_trace(world_pos: vec3<f32>, d: vec3<f32>, room: f32, roughness: f32) -> ProbeHit {{
    return probe_trace_skipping(world_pos, d, room, roughness, -1);
}}

fn probe_trace_skipping(world_pos: vec3<f32>, d: vec3<f32>, room: f32, roughness: f32, skip: i32) -> ProbeHit {{
    var hit: ProbeHit;
    hit.pos = vec3<f32>(0.0);
    hit.room = -1.0;
    hit.other = -1.0;
    hit.found = false;
    hit.escaped = false;
    hit.portal = -1;
    hit.t = 0.0;
    hit.origin = world_pos;
    hit.rim = -1.0;
    hit.rim_t = 0.0;
    hit.rim_code = -1;
    hit.edge_cover = 0.0;
    hit.edge_t = 0.0;
    hit.edge_code = -1;
    let lobe = probe_lobe_tan(roughness);

    if (roughness > PROBE_TRACE_MAX_ROUGHNESS || camera.portal_params.y > 0.5) {{
        return hit;
    }}
    let moving = abs(d) > vec3<f32>(1e-6);
    let inv = select(vec3<f32>(3.4e38), 1.0 / d, moving);
    var cur = room;
    var t0 = 0.0;

    var in_room = cur >= 0.0;
    if (!in_room) {{

        let p = probe_portal_holding(world_pos);
        if (p < 0) {{
{open_fallback_code}            return hit;
        }}
        let axis = i32(camera.probe_portals[p * 3].w);
        let plo = camera.probe_portals[p * 3].xyz;
        let phi = camera.probe_portals[p * 3 + 1].xyz;
        let low = camera.probe_portals[p * 3 + 1].w;
        let high = camera.probe_portals[p * 3 + 2].x;
        hit.origin = clamp(world_pos, plo, phi);
        var side = select(vec3<f32>(3.4e38), max((phi - hit.origin) * inv, (plo - hit.origin) * inv), moving);
        side[axis] = 3.4e38;
        let t_side = min(min(side.x, side.y), side.z);
        let next = select(low, high, d[axis] > 0.0);
        let nslot = probe_room_slot(next);
        if (nslot < 0) {{
            return hit;
        }}
        let wall_far = select(camera.probe_portals[p * 3 + 2].y, camera.probe_portals[p * 3 + 2].z, d[axis] > 0.0);
        let escapes = probe_seen_distance(nslot, d) < 0.0;
        let face = select(camera.probe_boxes[nslot * 3 + 2].xyz, camera.probe_boxes[nslot * 3 + 1].xyz, d > vec3<f32>(0.0));
        let t_enter = select(max((face[axis] - hit.origin[axis]) * inv[axis], 0.0), max((wall_far - hit.origin[axis]) * inv[axis], 0.0), escapes);
        if (t_side < t_enter) {{
            hit.pos = hit.origin + d * t_side;
            hit.room = low;
            hit.other = high;
            hit.found = true;
            hit.t = t_side;
            return hit;
        }}
        if (escapes) {{
            hit.pos = hit.origin + d * t_enter;
            hit.room = select(high, low, next == high);
            hit.other = next;
            hit.portal = p;
            hit.escaped = true;
            hit.found = true;
            hit.t = t_enter;
            return hit;
        }}
        cur = next;
        t0 = t_enter;
    }}
    for (var hop = 0; hop < PROBE_TRACE_ROOMS; hop = hop + 1) {{
        let slot = probe_room_slot(cur);
        if (slot < 0) {{
            return hit;
        }}
        let lo = camera.probe_boxes[slot * 3 + 1].xyz;
        let hi = camera.probe_boxes[slot * 3 + 2].xyz;
        if (hop == 0 && in_room) {{
            hit.origin = clamp(world_pos, lo, hi);
        }}

        if (cur + 1.0 == camera.portal_params.z) {{
            hit.pos = hit.origin + d * t0;
            hit.room = cur;
            hit.found = true;
            hit.escaped = true;
            hit.t = t0;
            return hit;
        }}

        let far = select(vec3<f32>(3.4e38), max((hi - hit.origin) * inv, (lo - hit.origin) * inv), moving);
        var axis = 2;
        var t_exit = far.z;
        if (far.x <= far.y && far.x <= far.z) {{
            axis = 0;
            t_exit = far.x;
        }} else if (far.y <= far.z) {{
            axis = 1;
            t_exit = far.y;
        }}

        let proxy = probe_proxy_hit(hit.origin, d, cur, t0, t_exit, skip, lobe);
        let t_obj = proxy.t;
        if (hit.edge_code < 0 && proxy.edge >= 0) {{
            hit.edge_cover = proxy.edge_cover;
            hit.edge_t = proxy.edge_t;
            hit.edge_code = ((i32(cur) + 1) << 6u) | (proxy.edge << 1u) | select(0, 1, t_obj < t_exit && proxy.index == proxy.edge);
        }}
        if (t_obj < t_exit) {{
            hit.pos = hit.origin + d * t_obj;
            hit.room = cur;
            hit.other = probe_proxy_model(proxy.index);
            hit.found = true;
            hit.t = t_obj;
            return hit;
        }}
        let e = hit.origin + d * t_exit;
        let p = probe_portal_at(e, cur, axis);

        let spread = max(t_exit * lobe, probe_pixel_spread(t_exit));
        if (hit.rim < 0.0 && spread > PROBE_RIM_MIN_SPREAD) {{
            let rim = probe_rim_at(e, cur, axis, spread);
            if (rim.portal >= 0) {{
                hit.rim = rim.through;
                hit.rim_t = t_exit;
                hit.rim_code = ((i32(cur) + 1) << 8u) | (rim.portal << 3u) | (axis << 1u) | select(0, 1, p >= 0);
            }}
        }}
        if (p < 0) {{

            hit.pos = e;
            hit.room = cur;
            hit.found = true;
            hit.t = t_exit;
            return hit;
        }}
        let low = camera.probe_portals[p * 3 + 1].w;
        let high = camera.probe_portals[p * 3 + 2].x;
        let other = select(low, high, cur == low);

        let plo = camera.probe_portals[p * 3].xyz;
        let phi = camera.probe_portals[p * 3 + 1].xyz;
        var side = select(vec3<f32>(3.4e38), max((phi - e) * inv, (plo - e) * inv), moving);
        side[axis] = 3.4e38;
        let t_side = t_exit + min(min(side.x, side.y), side.z);
        let oslot = probe_room_slot(other);
        if (oslot < 0) {{

            return hit;
        }}

        let escapes = probe_seen_distance(oslot, d) < 0.0;
        let wall_far = select(camera.probe_portals[p * 3 + 2].y, camera.probe_portals[p * 3 + 2].z, d[axis] > 0.0);
        let face = select(camera.probe_boxes[oslot * 3 + 2].xyz, camera.probe_boxes[oslot * 3 + 1].xyz, d > vec3<f32>(0.0));
        let t_enter = select(
            max((face[axis] - hit.origin[axis]) * inv[axis], t_exit),
            max((wall_far - hit.origin[axis]) * inv[axis], t_exit),
            escapes
        );

        if (hit.rim < 0.0) {{
            let spread_far = max(t_enter * lobe, probe_pixel_spread(t_enter));
            let e2 = hit.origin + d * t_enter;
            let a = (axis + 1) % 3;
            let b = (axis + 2) % 3;
            let in_a = min(e2[a] - plo[a], phi[a] - e2[a]);
            let in_b = min(e2[b] - plo[b], phi[b] - e2[b]);
            if (spread_far > PROBE_RIM_MIN_SPREAD && abs(min(in_a, in_b)) < spread_far) {{
                hit.rim = smoothstep(-spread_far, spread_far, in_a) * smoothstep(-spread_far, spread_far, in_b);
                hit.rim_t = t_enter;
                hit.rim_code = PROBE_RIM_FAR | ((i32(cur) + 1) << 8u) | (p << 3u) | (axis << 1u) | select(0, 1, t_side >= t_enter);
            }}
        }}
        if (t_side < t_enter) {{
            hit.pos = hit.origin + d * t_side;
            hit.room = cur;
            hit.other = other;
            hit.found = true;
            hit.t = t_side;
            return hit;
        }}
        if (escapes) {{
            hit.pos = hit.origin + d * t_enter;
            hit.room = cur;
            hit.other = other;
            hit.portal = p;
            hit.escaped = true;
            hit.found = true;
            hit.t = t_enter;
            return hit;
        }}
        cur = other;

        t0 = t_exit;
    }}
    return hit;
}}

const GROUND_TRACE_FINEST_LEVEL: i32 = {ground_trace_finest};
const GROUND_TRACE_START_LEVEL: i32 = {ground_trace_start};
const GROUND_TRACE_READINGS: i32 = {ground_trace_readings};
const GROUND_TRACE_MAX_STEPS: i32 = {ground_trace_steps};

fn sky_reflection(d: vec3<f32>, dir: vec3<f32>, lod: f32) -> vec3<f32> {{

    let layer = i32(camera.sky_params.y) - 1;
    if (layer < 0) {{
        return environment_radiance(dir);
    }}
    return textureSampleLevel(probe_cube, probe_samp, d, layer, lod).rgb;
}}

fn ground_uv(p: vec3<f32>) -> vec2<f32> {{
    return (p.xz - camera.ground_params.xy) * camera.ground_params.zw;
}}

fn ground_trace(e: vec3<f32>, d: vec3<f32>) -> f32 {{
    return ground_trace_until(e, d, 3.4e38);
}}

fn ground_trace_until(e: vec3<f32>, d: vec3<f32>, t_max: f32) -> f32 {{
    let top = camera.sky_params.z;
    let size = f32(textureDimensions(ground_map, 0).x);
    let scale = camera.ground_params.zw * size;

    let q0 = (e.xz - camera.ground_params.xy) * scale;
    let raw = d.xz * scale;
    let dq = select(raw, vec2<f32>(1e-6), abs(raw) < vec2<f32>(1e-6));
    let inv = 1.0 / dq;

    let ta = -q0 * inv;
    let tb = (vec2<f32>(size) - q0) * inv;
    var t = max(max(min(ta.x, tb.x), min(ta.y, tb.y)), 0.0);
    var t_out = min(min(max(ta.x, tb.x), max(ta.y, tb.y)), t_max);
    if (d.y > 0.0) {{
        t_out = min(t_out, (top - e.y) / d.y);
    }} else if (d.y < 0.0) {{
        t = max(t, (top - e.y) / d.y);
    }} else if (e.y > top) {{
        t_out = -1.0;
    }}
    let top_level = i32(textureNumLevels(ground_map)) - 1;
    let finest = min(GROUND_TRACE_FINEST_LEVEL + ground_trace_coarsen, top_level);
    var level = clamp(GROUND_TRACE_START_LEVEL, finest, top_level);
    let ahead = select(vec2<f32>(0.0), vec2<f32>(1.0), dq > vec2<f32>(0.0));

    let nudge = sign(dq) * 1e-3;
    var hit = -1.0;

    var risen = d.y <= 0.0;
    for (var steps = 0; steps < GROUND_TRACE_MAX_STEPS && t < t_out; steps = steps + 1) {{
        let cell = f32(1u << u32(level));
        let last = floor(size / cell) - 1.0;
        let c = clamp(floor((q0 + dq * t + nudge) / cell), vec2<f32>(0.0), vec2<f32>(last));
        let exits = ((c + ahead) * cell - q0) * inv;
        let t_exit = min(min(exits.x, exits.y), t_out);
        let highest = textureLoad(ground_map, vec2<i32>(c), level).a;
        let y_in = e.y + d.y * t;
        risen = risen || y_in > highest;
        if (min(y_in, e.y + d.y * t_exit) <= highest) {{

            let t_top = select(t, t + (y_in - highest) / -d.y, y_in > highest);
            if (level > finest) {{
                level = level - 1;
                t = t_top;
                continue;
            }}

            let span = (t_exit - t_top) / f32(GROUND_TRACE_READINGS - 1);
            var f_prev = e.y + d.y * t_top - textureSampleLevel(ground_map, probe_samp, (q0 + dq * t_top) / size, 0.0).a;
            var crossing = select(-1.0, t_top, f_prev <= 0.0 && risen);
            risen = risen || f_prev > 0.0;
            for (var k = 1; k < GROUND_TRACE_READINGS; k = k + 1) {{
                let tk = t_top + span * f32(k);
                let f = e.y + d.y * tk - textureSampleLevel(ground_map, probe_samp, (q0 + dq * tk) / size, 0.0).a;
                if (crossing < 0.0 && f <= 0.0 && risen) {{
                    crossing = tk - span + span * f_prev / (f_prev - f);
                }}
                risen = risen || f > 0.0;
                f_prev = f;
            }}
            if (crossing >= 0.0) {{
                hit = crossing;
                break;
            }}
        }}

        let next = floor((q0 + dq * t_exit + nudge) / cell);
        if (any(floor(next * 0.5) != floor(c * 0.5))) {{
            level = min(level + 1, top_level);
        }}
        t = t_exit;
    }}
    return hit;
}}

fn outdoor_radiance(e: vec3<f32>, d: vec3<f32>, dir: vec3<f32>, roughness: f32, lod: f32) -> vec3<f32> {{
    let b = building_hit(e, d);

    probe_reach = PROBE_REACH_SKY;
    if (camera.sky_params.w > 0.5) {{
        let t = ground_trace_until(e, d, select(3.4e38, b.x, b.x >= 0.0));
        if (t >= 0.0) {{
            probe_reach = t;
            let texel = 1.0 / max(camera.ground_params.z * f32(textureDimensions(ground_map, 0).x), 1e-6);
            let spread = max(t * probe_lobe_tan(roughness), 1e-4);
            let ground_lod = clamp(log2(spread / texel), 0.0, 12.0);
            return textureSampleLevel(ground_map, probe_samp, ground_uv(e + d * t), ground_lod).rgb;
        }}
    }}
    if (b.x >= 0.0) {{
        let c = building_colour(e + d * b.x, i32(b.y), roughness, b.x);
        probe_reach = b.x;
        return c.rgb * c.a + sky_reflection(d, dir, lod) * (1.0 - c.a);
    }}
    return sky_reflection(d, dir, lod);
}}

fn building_hit(e: vec3<f32>, d: vec3<f32>) -> vec2<f32> {{
    let n = i32(camera.portal_params.w);
    let inv = 1.0 / select(d, vec3<f32>(1e-8), abs(d) < vec3<f32>(1e-8));
    var best = vec2<f32>(-1.0, 0.0);
    for (var i = 0; i < n; i = i + 1) {{
        let t0 = (camera.building_boxes[i * 2].xyz - e) * inv;
        let t1 = (camera.building_boxes[i * 2 + 1].xyz - e) * inv;
        let near = min(t0, t1);
        let far = max(t0, t1);
        let t_in = max(max(near.x, near.y), near.z);
        let t_out = min(min(far.x, far.y), far.z);
        if (t_in > 1e-3 && t_in <= t_out && (best.x < 0.0 || t_in < best.x)) {{
            best = vec2<f32>(t_in, f32(i));
        }}
    }}
    return best;
}}

fn building_colour(p: vec3<f32>, slot: i32, roughness: f32, t: f32) -> vec4<f32> {{
    let lo = camera.building_boxes[slot * 2];
    let hi = camera.building_boxes[slot * 2 + 1].xyz;
    let half = max((hi - lo.xyz) * 0.5, vec3<f32>(1e-3));
    let texel = 2.0 * max(max(half.x, half.y), half.z) / f32(textureDimensions(probe_cube).x);
    let spread = max(t * probe_lobe_tan(roughness), 1e-4);
    let lod = clamp(log2(spread / texel), 0.0, PROBE_MAX_LOD);
    return textureSampleLevel(probe_cube, probe_samp, (p - (lo.xyz + hi) * 0.5) / half, i32(lo.w), lod);
}}

fn probe_escape_colour(e: vec3<f32>, d: vec3<f32>, sky_dir: vec3<f32>, roughness: f32, lod: f32) -> vec4<f32> {{
    return vec4<f32>(outdoor_radiance(e, d, sky_dir, roughness, lod), 1.0);
}}

const PROBE_MODEL_ALBEDO: f32 = 0.2;

fn probe_model_colour(proxy: i32, n: vec3<f32>, slot: i32) -> vec4<f32> {{
    let lit = textureSampleLevel(probe_cube, probe_samp, n, i32(camera.probe_boxes[slot * 3].w), PROBE_MAX_LOD);
    let kind = camera.probe_proxies[proxy * 3 + 1].w;
    let field = camera.proxy_fields[max(i32(kind) - 2, 0) * 3 + 2].xyz;
    let albedo = select(vec3<f32>(PROBE_MODEL_ALBEDO), field, kind > 1.5);
    return vec4<f32>(albedo * lit.rgb, 1.0);
}}

fn probe_view_trust(d: vec3<f32>, c: vec3<f32>, h: vec3<f32>) -> f32 {{
    return smoothstep(0.5, 0.9, dot(d, normalize(h - c)));
}}

struct ProbeCardVote {{
    rgb: vec3<f32>,
    w: f32,
    sure: f32,

    card: vec3<f32>,
}}
fn probe_card_vote(row: f32, face: f32, uv: vec2<f32>, t: f32, depth: f32, extent: vec2<f32>, field_stop: f32, lod: f32, cu: vec3<f32>, cv: vec3<f32>, cz: vec3<f32>, ld: vec3<f32>) -> ProbeCardVote {{
    let dims = vec2<f32>(textureDimensions(proxy_cards));
    let res = dims.x / 6.0;
    let s = exp2(lod);
    let origin = vec2<f32>(face * res, row * res);
    let card = textureSampleLevel(proxy_cards, probe_samp, (origin + clamp(uv * res, vec2<f32>(0.5 * s), vec2<f32>(res - 0.5 * s))) / dims, lod);
    var trust: vec2<f32>;
    if (PROBE_CARD_TESTS_FILTERED) {{

        let p = clamp(uv * res, vec2<f32>(0.5), vec2<f32>(res - 0.5)) - 0.5;
        let i0 = floor(p);
        let f = p - i0;
        let lo = vec2<i32>(origin + vec2<f32>(0.0, res));
        let hi = lo + vec2<i32>(i32(res) - 1);
        let a = vec2<i32>(i0) + lo;
        let b = min(a + vec2<i32>(1), hi);

        let m = extent / res;
        let t00 = probe_card_trust(textureLoad(proxy_cards, a, 0), f * m, t, depth, field_stop, cu, cv, cz, ld);
        let t10 = probe_card_trust(textureLoad(proxy_cards, vec2<i32>(b.x, a.y), 0), (f - vec2<f32>(1.0, 0.0)) * m, t, depth, field_stop, cu, cv, cz, ld);
        let t01 = probe_card_trust(textureLoad(proxy_cards, vec2<i32>(a.x, b.y), 0), (f - vec2<f32>(0.0, 1.0)) * m, t, depth, field_stop, cu, cv, cz, ld);
        let t11 = probe_card_trust(textureLoad(proxy_cards, b, 0), (f - vec2<f32>(1.0)) * m, t, depth, field_stop, cu, cv, cz, ld);
        trust = mix(mix(t00, t10, f.x), mix(t01, t11, f.x), f.y);
    }} else {{
        let test = textureSampleLevel(proxy_cards, probe_samp, (origin + vec2<f32>(0.0, res) + clamp(uv * res, vec2<f32>(0.5), vec2<f32>(res - 0.5))) / dims, 0.0);
        trust = probe_card_trust(test, vec2<f32>(0.0), t, depth, field_stop, cu, cv, cz, ld);
    }}
    var vote: ProbeCardVote;

    vote.rgb = card.rgb / max(1.0 - dot(card.rgb, vec3<f32>(0.2126, 0.7152, 0.0722)), 1e-3);
    vote.w = trust.x;
    vote.sure = trust.y;
    vote.card = vec3<f32>(face, uv);
    return vote;
}}

fn probe_card_trust(test: vec4<f32>, off: vec2<f32>, t: f32, depth: f32, field_stop: f32, cu: vec3<f32>, cv: vec3<f32>, cz: vec3<f32>, ld: vec3<f32>) -> vec2<f32> {{

    let c = probe_card_hemisphere(test.xy);
    let seen = c.x * cu + c.y * cv + c.z * cz;
    let square = c.z;
    let slope = clamp(c.xy / max(c.z, 0.1), vec2<f32>(-3.5), vec2<f32>(3.5));
    let behind = (t - test.z) * depth - dot(slope, off);
    let in_front = (test.w - t) * depth;
    let back_tol = 0.001 / max(square, 0.1);
    let front_tol = 0.005 + field_stop / max(square, 0.3);
    let valid = (1.0 - smoothstep(-0.03, 0.0, dot(seen, ld)))
        * (1.0 - smoothstep(back_tol, 2.0 * back_tol, behind))
        * (1.0 - smoothstep(front_tol, 2.0 * front_tol, in_front));
    return vec2<f32>(square * square * valid, valid * smoothstep(0.1, 0.3, abs(square)));
}}

fn probe_card_hemisphere(o: vec2<f32>) -> vec3<f32> {{
    return normalize(vec3<f32>(o.x, o.y, max(1.0 - abs(o.x) - abs(o.y), 0.0)));
}}

struct ProbeCardSum {{
    colour: vec4<f32>,
    sure: f32,

    best: f32,
    card: vec3<f32>,
}}
fn probe_card_add(sum: ProbeCardSum, vote: ProbeCardVote) -> ProbeCardSum {{
    var out = sum;
    let k = vote.w / (1.0 + dot(vote.rgb, vec3<f32>(0.2126, 0.7152, 0.0722)));
    out.colour += vec4<f32>(vote.rgb * k, k);
    out.sure = max(out.sure, vote.sure);
    out.card = select(out.card, vote.card, vote.w > out.best);
    out.best = max(out.best, vote.w);
    return out;
}}

fn probe_axis(k: u32) -> vec3<f32> {{
    return vec3<f32>(f32(k == 0u), f32(k == 1u), f32(k == 2u));
}}

fn probe_surface_relit(h: vec3<f32>, d: vec3<f32>, roughness: f32, t: f32) -> vec3<f32> {{
    if (lights.surfaces[1].w <= 0.0) {{
        return vec3<f32>(0.0);
    }}
    let p = to_player_space(h);
    let k = lit_surface_at(p, to_player_direction(d));
    let at = max(k, 0) * LIT_SURFACE_VEC4S;
    let glass = lights.surfaces[at + 1];
    let axis = lights.surfaces[at + 2];
    let right = lights.surfaces[at + 3].xyz;
    let v = p - glass.xyz;
    let z = dot(v, axis.xyz);
    let xy = vec2<f32>(dot(v, right), dot(v, cross(right, axis.xyz))) / (max(z, 1e-4) * glass.w);
    let inside = k >= 0 && z > 0.0 && max(abs(xy.x), abs(xy.y)) < 1.0;

    let dims = vec2<f32>(textureDimensions(proxy_cards));
    let block = dims.x / f32(POOL_MAPS_ACROSS);
    let footprint = max(t * probe_lobe_tan(roughness), pixel_footprint * (1.0 + t / max(probe_eye_distance, 0.05)));
    let lod = clamp(log2(max(footprint * block / (2.0 * max(z, 1e-4) * glass.w), 1.0)), 0.0, POOL_MAP_MAX_LOD);
    let s = 0.5 * exp2(ceil(lod));
    let kk = max(k, 0);
    let corner = vec2<f32>(f32(kk % POOL_MAPS_ACROSS), f32(kk / POOL_MAPS_ACROSS)) * block + vec2<f32>(0.0, axis.w);
    let texel = corner + clamp((clamp(xy, vec2<f32>(-1.0), vec2<f32>(1.0)) * 0.5 + vec2<f32>(0.5)) * block, vec2<f32>(s), vec2<f32>(block - s));
    let lit = textureSampleLevel(proxy_cards, probe_samp, texel / dims, lod).rgb;
    return select(vec3<f32>(0.0), lit, inside);
}}

fn pool_map_light(k: i32, uv: vec2<f32>, texels: f32) -> vec3<f32> {{
    if (k >= MAX_LIT_SURFACES) {{
        return vec3<f32>(0.0);
    }}
    let at = k * LIT_SURFACE_VEC4S;
    let glass = lights.surfaces[at + 1];
    if (glass.w <= 0.0) {{
        return vec3<f32>(0.0);
    }}
    let plane = lights.surfaces[at];
    let axis = lights.surfaces[at + 2].xyz;
    let right = lights.surfaces[at + 3].xyz;
    let xy = (uv * 2.0 - vec2<f32>(1.0)) * glass.w;
    let dir = normalize(axis + right * xy.x + cross(right, axis) * xy.y);
    let facing = dot(plane.xyz, dir);
    if (facing > -1e-4) {{
        return vec3<f32>(0.0);
    }}
    let along = (plane.w - dot(plane.xyz, glass.xyz)) / facing;
    if (along <= 0.0) {{
        return vec3<f32>(0.0);
    }}
    let p = glass.xyz + dir * along;
    let n = plane.xyz;
    pixel_footprint = along * 2.0 * glass.w / texels;
    var lit = vec3<f32>(0.0);
    for (var i: u32 = 0u; i < live_light_count(); i = i + 1u) {{
        let l = lights.lights[i];

        if (l.position.w > -0.5 || l.params.w < -1.5) {{
            continue;
        }}
        let c = light_contribution_split(l, p, n, -dir, 1.0, 0.0);
        if (max(max(c.diffuse.r, c.diffuse.g), c.diffuse.b) <= 0.0) {{
            continue;
        }}
        var shadow = 1.0;
        let layer = i32(l.params.w);
        if (SPOT_SHADOWS && layer >= 0 && f32(layer) < camera.shadow_params.y) {{
            shadow = pcf_layer(spot_shadow_tex, layer, p + n * PROBE_CARD_RELIT_LIFT, spot_view_proj(layer));
        }}
        lit += c.diffuse * shadow;
    }}
    return lights.surfaces[at + 4].rgb * lit;
}}

fn lit_surface_at(p: vec3<f32>, dp: vec3<f32>) -> i32 {{
    var k = -1;
    var best = 0.0;
    for (var j = MAX_LIT_SURFACES - 1; j >= 0; j = j - 1) {{
        let at = j * LIT_SURFACE_VEC4S;
        let plane = lights.surfaces[at];
        let glass = lights.surfaces[at + 1];
        let z = dot(p - glass.xyz, lights.surfaces[at + 2].xyz);
        let near = lights.surfaces[at + 3].w;
        let far = lights.surfaces[at + 4].w;
        let bounded = far > 0.0;
        let square = select(1.0, -dot(dp, plane.xyz), bounded);
        let within = !bounded || (z >= near && z <= far);
        let take = glass.w > 0.0 && abs(dot(plane.xyz, p) - plane.w) < LIT_SURFACE_TOLERANCE && within && square > 0.0 && square >= best;
        k = select(k, j, take);
        best = select(best, square, take);
    }}
    return k;
}}
const MAX_LIT_SURFACES: i32 = {MAX_LIT_SURFACES};
const LIT_SURFACE_VEC4S: i32 = {LIT_SURFACE_VEC4S};
const POOL_MAPS_ACROSS: i32 = {pool_maps_across};

const POOL_MAP_MAX_LOD: f32 = {character_card_max_lod:?};

const LIT_SURFACE_TOLERANCE: f32 = 0.05;

fn probe_card_relit(row: f32, card: vec3<f32>, lod: f32, h: vec3<f32>, q: vec4<f32>) -> vec3<f32> {{
    var unseen = false;
    for (var i: u32 = 0u; i < live_light_count(); i = i + 1u) {{
        unseen = unseen || lights.lights[i].position.w < -0.5;
    }}
    if (!unseen) {{
        return vec3<f32>(0.0);
    }}
    let dims = vec2<f32>(textureDimensions(proxy_cards));
    let res = dims.x / 6.0;
    let s = exp2(lod);
    let origin = vec2<f32>(card.x * res, row * res);
    let uv = card.yz;

    let test = textureSampleLevel(proxy_cards, probe_samp, (origin + vec2<f32>(0.0, res) + clamp(uv * res, vec2<f32>(0.5), vec2<f32>(res - 0.5))) / dims, 0.0);
    let albedo = textureSampleLevel(proxy_cards, probe_samp, (origin + vec2<f32>(0.0, 2.0 * res) + clamp(uv * res, vec2<f32>(0.5 * s), vec2<f32>(res - 0.5 * s))) / dims, lod).rgb;

    let a = u32(card.x) / 2u;
    let o = probe_card_hemisphere(test.xy);
    var n = o.x * probe_axis((a + 1u) % 3u) + o.y * probe_axis((a + 2u) % 3u)
        + o.z * probe_axis(a) * select(1.0, -1.0, (u32(card.x) & 1u) == 1u);
    if (any(q != vec4<f32>(0.0, 0.0, 0.0, 1.0))) {{
        n = probe_quat_rotate(q, n);
    }}

    let p = to_player_space(h);
    let np = normalize(to_player_direction(n));
    var lit = vec3<f32>(0.0);
    for (var i: u32 = 0u; i < live_light_count(); i = i + 1u) {{
        let l = lights.lights[i];
        if (l.position.w > -0.5) {{
            continue;
        }}
        let c = light_contribution_split(l, p, np, np, 1.0, 0.0);
        if (max(max(c.diffuse.r, c.diffuse.g), c.diffuse.b) <= 0.0) {{
            continue;
        }}
        var shadow = 1.0;
        let layer = i32(l.params.w);
        if (SPOT_SHADOWS && layer >= 0 && f32(layer) < camera.shadow_params.y) {{
            shadow = pcf_layer(spot_shadow_tex, layer, p + np * PROBE_CARD_RELIT_LIFT, spot_view_proj(layer));
        }}
        lit += c.diffuse * shadow;
    }}

    return albedo * lit;
}}

fn probe_card_colour(proxy: i32, h: vec3<f32>, d: vec3<f32>, t_hit: f32, lobe: f32) -> vec4<f32> {{
    let row = camera.proxy_cards[proxy >> 2u][proxy & 3] - 1.0;
    if (row < 0.0) {{
        return vec4<f32>(0.0);
    }}
    let q = camera.probe_proxies[proxy * 3 + 2];
    var lo = h - camera.probe_proxies[proxy * 3].xyz;
    var ld = d;
    if (any(q != vec4<f32>(0.0, 0.0, 0.0, 1.0))) {{
        let qi = vec4<f32>(-q.xyz, q.w);
        lo = probe_quat_rotate(qi, lo);
        ld = probe_quat_rotate(qi, ld);
    }}
    let box = camera.probe_proxies[proxy * 3 + 1];
    let half = max(box.xyz, vec3<f32>(1e-4));

    let field_stop = select(0.0, camera.proxy_fields[max(i32(box.w) - 2, 0) * 3 + 1].w, box.w > 1.5);
    let res = f32(textureDimensions(proxy_cards).x) / 6.0;

    let texel_m = 2.0 * max(max(half.x, half.y), half.z) / res;
    let footprint = max(t_hit * lobe, pixel_footprint * (1.0 + t_hit / max(probe_eye_distance, 0.05)));
    let lod = clamp(log2(max(footprint / texel_m, 1.0)), 0.0, log2(res));

    let uvw = lo / (2.0 * half) + 0.5;
    let ax = vec3<f32>(1.0, 0.0, 0.0);
    let ay = vec3<f32>(0.0, 1.0, 0.0);
    let az = vec3<f32>(0.0, 0.0, 1.0);

    var sum: ProbeCardSum;
    sum.colour = vec4<f32>(0.0);
    sum.sure = 0.0;
    sum.best = 0.0;
    sum.card = vec3<f32>(0.0);
    sum = probe_card_add(sum, probe_card_vote(row, 0.0, uvw.yz, 0.5 - 0.5 * lo.x / half.x, 2.0 * half.x, 2.0 * half.yz, field_stop, lod, ay, az, ax, ld));
    sum = probe_card_add(sum, probe_card_vote(row, 1.0, uvw.yz, 0.5 + 0.5 * lo.x / half.x, 2.0 * half.x, 2.0 * half.yz, field_stop, lod, ay, az, -ax, ld));
    sum = probe_card_add(sum, probe_card_vote(row, 2.0, uvw.zx, 0.5 - 0.5 * lo.y / half.y, 2.0 * half.y, 2.0 * half.zx, field_stop, lod, az, ax, ay, ld));
    sum = probe_card_add(sum, probe_card_vote(row, 3.0, uvw.zx, 0.5 + 0.5 * lo.y / half.y, 2.0 * half.y, 2.0 * half.zx, field_stop, lod, az, ax, -ay, ld));
    sum = probe_card_add(sum, probe_card_vote(row, 4.0, uvw.xy, 0.5 - 0.5 * lo.z / half.z, 2.0 * half.z, 2.0 * half.xy, field_stop, lod, ax, ay, az, ld));
    sum = probe_card_add(sum, probe_card_vote(row, 5.0, uvw.xy, 0.5 + 0.5 * lo.z / half.z, 2.0 * half.z, 2.0 * half.xy, field_stop, lod, ax, ay, -az, ld));

    var colour = sum.colour.rgb / max(sum.colour.w, 1e-9);
{card_relit_call}    return vec4<f32>(colour, sum.sure);
}}

fn probe_mix_bright(a: vec4<f32>, b: vec4<f32>, k: f32) -> vec4<f32> {{
    let luma = vec3<f32>(0.2126, 0.7152, 0.0722);
    let wa = (1.0 - k) / (1.0 + dot(a.rgb, luma));
    let wb = k / (1.0 + dot(b.rgb, luma));
    return (a * wa + b * wb) / max(wa + wb, 1e-9);
}}

fn probe_hit_colour(h: vec3<f32>, room: f32, other: f32, roughness: f32, t: f32, d: vec3<f32>) -> vec4<f32> {{

    let near = probe_nearest_two(h, room, other);
    let s0 = near.s0;
    let s1 = near.s1;
    let d0 = near.d0;
    let d1 = near.d1;

    let model = other < -1.5;

    let has1 = s1 >= 0;
    let b0 = camera.probe_boxes[s0 * 3];
    let b1 = camera.probe_boxes[select(s0, s1, has1) * 3];
    let v0 = h - b0.xyz;
    let v1 = h - b1.xyz;

    let lod0 = probe_hit_lod(roughness, t, sqrt(d0));
    let lod1 = probe_hit_lod(roughness, t, sqrt(select(d0, d1, has1)));
    let depth0 = textureSampleLevel(probe_depth, probe_depth_samp, v0, i32(b0.w), 0.0);
    let depth1 = textureSampleLevel(probe_depth, probe_depth_samp, v1, i32(b1.w), 0.0);
    let col0 = textureSampleLevel(probe_cube, probe_samp, v0, i32(b0.w), lod0);
    let col1 = textureSampleLevel(probe_cube, probe_samp, v1, i32(b1.w), lod1);
{surface_relit_read}    let c0 = probe_clearance_of(depth0, v0);
    let tol0 = PROBE_SEEN_TOLERANCE + 0.01 * sqrt(d0);

    var seen = 1.0 - smoothstep(tol0, 2.0 * tol0, abs(c0));
    if (model) {{
        seen *= probe_view_trust(d, b0.xyz, h);
    }}
    var w0 = seen / (d0 + 1.0);
    var w1 = 0.0;
    var c1 = -3.4e38;
    if (has1) {{
        c1 = probe_clearance_of(depth1, v1);
        let tol1 = PROBE_SEEN_TOLERANCE + 0.01 * sqrt(d1);
        var seen1 = 1.0 - smoothstep(tol1, 2.0 * tol1, abs(c1));
        if (model) {{
            seen1 *= probe_view_trust(d, b1.xyz, h);
        }}
        w1 = seen1 / (d1 + 1.0);
        seen = max(seen, seen1);
    }}
    if (w0 + w1 < 1e-6) {{
        w0 = select(1.0, 0.0, has1 && abs(c1) < abs(c0));
        w1 = 1.0 - w0;
    }}
    var col = col0 * w0;
    if (w1 > 0.0) {{
        col += col1 * w1;
    }}
    col = col / (w0 + w1);

    if (model && seen < 1.0) {{
        col = mix(probe_model_colour(i32(-2.0 - other), -d, s0), col, seen);
    }}

    if (!model && seen < 1.0) {{
        let wide = textureSampleLevel(probe_cube, probe_samp, v0, i32(b0.w), max(lod0, PROBE_UNSEEN_LOD));
        col = mix(wide, col, seen);
    }}

    if (model && !PROBE_SECONDARY_DEFERRED) {{
        let card = probe_card_colour(i32(-2.0 - other), h, d, t, probe_lobe_tan(roughness));
        col = probe_mix_bright(col, vec4<f32>(card.rgb, 1.0), card.w);
    }}
{surface_relit_call}    return col;
}}

fn probe_hit_carded(h: ProbeHit) -> bool {{
    let m = max(i32(-2.0 - h.other), 0);
    return h.other < -1.5 && camera.proxy_cards[m >> 2u][m & 3] > 0.5;
}}

const PROBE_UNSEEN_LOD: f32 = 4.0;

fn probe_traced_colour(h: ProbeHit, d: vec3<f32>, roughness: f32, sky_dir: vec3<f32>, lod: f32) -> vec4<f32> {{
    var col: vec4<f32>;
    if (h.escaped) {{
        col = probe_escape_colour(h.pos, d, sky_dir, roughness, lod);

        probe_reach = h.t + probe_reach;
    }} else {{
        col = probe_hit_colour(h.pos, h.room, h.other, roughness, h.t, d);
        probe_reach = h.t;
    }}
    return col;
}}

fn probe_point_hit(pos: vec3<f32>, room: f32, t: f32) -> ProbeHit {{
    var h: ProbeHit;
    h.pos = pos;
    h.room = room;
    h.other = -1.0;
    h.found = true;
    h.escaped = false;
    h.portal = -1;
    h.t = t;
    h.rim = -1.0;
    h.rim_code = -1;
    h.edge_code = -1;
    return h;
}}

fn probe_parallax_direction(world_pos: vec3<f32>, d: vec3<f32>, slot: i32) -> vec3<f32> {{
    let centre = camera.probe_boxes[slot * 3].xyz;
    let lo = camera.probe_boxes[slot * 3 + 1].xyz;
    let hi = camera.probe_boxes[slot * 3 + 2].xyz;

    let parallax_pos = clamp(world_pos, lo, hi);

    let inv = select(vec3<f32>(3.4e38), 1.0 / d, abs(d) > vec3<f32>(1e-6));
    let t_hi = (hi - parallax_pos) * inv;
    let t_lo = (lo - parallax_pos) * inv;
    let furthest = max(t_hi, t_lo);
    let dist = min(min(furthest.x, furthest.y), furthest.z);

    var sample_dir = d;
    if (dist >= 0.0) {{

        probe_reach = dist;
        let to_hit = parallax_pos + d * dist - centre;
        if (dot(to_hit, to_hit) > 1e-12) {{
            sample_dir = normalize(to_hit);
        }}
    }}
    return sample_dir;
}}

const PORTAL_FADE: f32 = {portal_fade:?};
const PORTAL_SIDE_FADE: f32 = {portal_side_fade:?};

var<private> volume_sample_brightness: f32 = 0.0;

fn probe_volume_sample(
    room: f32, select_world: vec3<f32>, world_pos: vec3<f32>, d: vec3<f32>, lod: f32,
) -> vec4<f32> {{

    var pick = -1;
    var pick_dist = 1e30;
    for (var i = probe_room_slot(room); i >= 0; i = probe_slot_next(i)) {{
        let to_centre = camera.probe_boxes[i * 3].xyz - select_world;
        let dist = dot(to_centre, to_centre);
        if (dist < pick_dist) {{
            pick = i;
            pick_dist = dist;
        }}
    }}
    if (pick < 0) {{
        return vec4<f32>(-1.0);
    }}
    volume_sample_brightness = camera.probe_boxes[pick * 3 + 1].w;
    return textureSampleLevel(
        probe_cube, probe_samp, probe_parallax_direction(world_pos, d, pick),
        i32(camera.probe_boxes[pick * 3].w), lod,
    );
}}

fn probe_through_portals(
    own: vec4<f32>, own_room: f32, select_world: vec3<f32>, world_pos: vec3<f32>,
    d: vec3<f32>, lod: f32,
) -> vec4<f32> {{
    let n = i32(camera.portal_params.x);
    for (var p = 0; p < n; p = p + 1) {{
        let lo = camera.probe_portals[p * 3].xyz;
        let hi = camera.probe_portals[p * 3 + 1].xyz;
        let axis = i32(camera.probe_portals[p * 3].w);
        let low_room = camera.probe_portals[p * 3 + 1].w;
        let high_room = camera.probe_portals[p * 3 + 2].x;
        let along = vec3<f32>(f32(axis == 0), f32(axis == 1), f32(axis == 2));

        let outside = max(max(lo - select_world, select_world - hi), vec3<f32>(0.0)) * (vec3<f32>(1.0) - along);
        let side = length(outside);
        let t = dot(select_world, along);
        let t0 = dot(lo, along) - PORTAL_FADE;
        let t1 = dot(hi, along) + PORTAL_FADE;
        if (side >= PORTAL_SIDE_FADE || t <= t0 || t >= t1) {{
            continue;
        }}
        let reach = 1.0 - smoothstep(0.0, PORTAL_SIDE_FADE, side);
        let w_high = smoothstep(t0, t1, t);
        if (own_room == low_room || own_room == high_room) {{

            let other_room = select(low_room, high_room, own_room == low_room);
            let w_other = reach * select(1.0 - w_high, w_high, own_room == low_room);
            let other = probe_volume_sample(other_room, select_world, world_pos, d, lod);
            if (other.a < 0.0) {{
                return own;
            }}
            probe_brightness = mix(probe_brightness, volume_sample_brightness, w_other);
            return mix(own, other, w_other);
        }}

        let a = probe_volume_sample(low_room, select_world, world_pos, d, lod);
        let a_bright = volume_sample_brightness;
        let b = probe_volume_sample(high_room, select_world, world_pos, d, lod);
        let b_bright = volume_sample_brightness;
        if (a.a < 0.0 && b.a < 0.0) {{
            return own;
        }}
        var through = mix(a, b, w_high);
        var through_bright = mix(a_bright, b_bright, w_high);
        if (a.a < 0.0) {{
            through = b;
            through_bright = b_bright;
        }} else if (b.a < 0.0) {{
            through = a;
            through_bright = a_bright;
        }}
        probe_brightness = mix(probe_brightness, through_bright, reach);
        return mix(own, through, reach);
    }}
    return own;
}}

const GROUND_ALBEDO: f32 = 0.28;

fn environment_radiance(dir: vec3<f32>) -> vec3<f32> {{
    return environment_radiance_of(dir, sky_irradiance(dir));
}}

fn environment_radiance_of(dir: vec3<f32>, sky: vec3<f32>) -> vec3<f32> {{

    let ground = camera.probe_params.yzw * GROUND_ALBEDO;

    return mix(ground, sky, smoothstep(-0.15, 0.15, dir.y));
}}

fn probe_pass_compress(c: vec3<f32>) -> vec3<f32> {{
    return c / (1.0 + dot(c, vec3<f32>(0.2126, 0.7152, 0.0722)));
}}

fn probe_env_for_pass(
    world_pos: vec3<f32>,
    n: vec3<f32>,
    roughness: f32,
    ao: f32,
    sky_vis: f32,
    env: vec3<f32>,
    probe_select_pos: vec3<f32>,
    geom_n: vec3<f32>,
) -> vec4<f32> {{
    let view_dir = normalize(cam_pos() - world_pos);
    let r = clamp(roughness, 0.04, 1.0);
    let env_n = normalize(mix(n, geom_n, smoothstep(0.1, 0.4, r)));
    let refl = reflect(-view_dir, env_n);

    let occ = clamp(ao, 0.0, 1.0) * clamp(sky_vis, 0.0, 1.0);

    var sky_here = vec3<f32>(0.0);
    if (occ > 0.0) {{
        sky_here = sky_irradiance(n) * occ;
    }}
    let ambient_here = dot(env + sky_here, vec3<f32>(0.2126, 0.7152, 0.0722));

    probe_floor_mirror_here = on_floor_mirror(world_pos, geom_n);
    probe_capsule_lit = env;
    var probe = probe_environment(world_pos, refl, roughness, probe_select_pos);
    if (probe_floor_mirror_here) {{
        probe = probe_floor_mirror_pass(probe, roughness);
    }} else {{
        probe = capsule_reflection(world_pos, refl, roughness, env, probe);
    }}
    let probe_scale = select(
        1.0,
        clamp(ambient_here / max(probe_brightness, 1e-4), PROBE_NORMALISATION_FLOOR, 1.0),
        PROBE_NORMALISATION && probe_brightness > 0.0,
    );
    let a = clamp(probe.a, 0.0, 1.0);

    return vec4<f32>(probe_pass_compress(probe.rgb * probe_scale + capsule_glow) * a, a);
}}

struct MaterialEnvPart {{
    diffuse: vec3<f32>,
    specular: vec3<f32>,

    bounce: vec3<f32>,
    bounce_dir: vec3<f32>,
    directionality: f32,
    has_dir: bool,
    fresnel: f32,
    view_dir: vec3<f32>,
    r: f32,

    surface_spec: f32,
}}

var<private> reflected_image: f32 = 0.0;

const REFLECTION_SHARP_ROUGHNESS: f32 = 0.1;
const REFLECTION_BLURRED_ROUGHNESS: f32 = 0.3;

const REFLECTION_CONTRAST_RATIO: f32 = {reflection_contrast};
fn reflection_alpha(lit: vec3<f32>) -> f32 {{

    if (camera.post_params.w < 0.5) {{
        return 1.0;
    }}
    let total = dot(lit, vec3<f32>(0.2126, 0.7152, 0.0722));
    let share = clamp(reflected_image / max(total, 1e-6), 0.0, 1.0);
    let image = REFLECTION_CONTRAST_RATIO * share;
    let surface = 1.0 - share;
    return 1.0 - image * image / max(image * image + surface * surface, 1e-12);
}}

fn shade_material_env_part(
    world_pos: vec3<f32>,
    n: vec3<f32>,
    roughness: f32,
    ao: f32,
    sky_vis: f32,
    env_baked: vec3<f32>,

    env_dir: vec4<f32>,

    albedo: vec3<f32>,

    probe_select_pos: vec3<f32>,

    geom_n: vec3<f32>,
) -> MaterialEnvPart {{
    let view_dir = normalize(cam_pos() - world_pos);
    let r = clamp(roughness, 0.04, 1.0);

    let contact = capsule_ambient(world_pos, n);
    let env = env_baked * contact;
    let occ = clamp(ao, 0.0, 1.0) * clamp(sky_vis, 0.0, 1.0) * contact;

    var diffuse = vec3<f32>(0.0);
    if (occ > 0.0) {{
        diffuse = sky_irradiance(n) * occ;
    }}
    diffuse = diffuse + cloud_fill(world_pos, n) * occ;
    var specular = vec3<f32>(0.0);

    let env_n = normalize(mix(n, geom_n, smoothstep(0.1, 0.4, r)));
    let refl = reflect(-view_dir, env_n);

    let cos_v = clamp(dot(env_n, view_dir), 0.0, 1.0);

    let f0 = 0.04;
    let f_max = max(1.0 - r, f0);

    let grazing = 1.0 - cos_v;
    let grazing2 = grazing * grazing;
    let fresnel = f0 + (f_max - f0) * (grazing2 * grazing2 * grazing);

    var probe = probe_env_given;
    if (!PROBE_ENV_FROM_PASS) {{
        probe = probe_environment(world_pos, refl, roughness, probe_select_pos);
    }}

    if (!PROBE_ENV_FROM_PASS) {{
        probe = capsule_reflection(world_pos, refl, roughness, env, probe);
    }}

    var sky_reflection = vec3<f32>(0.0);
    if (occ > 0.0) {{
        sky_reflection = environment_radiance(refl) * occ;
    }}

    let a_o = clamp(ao, 0.0, 1.0);
    let spec_occ = clamp(
        pow(cos_v + a_o, exp2(-16.0 * r - 1.0)) - 1.0 + a_o,
        0.0,
        1.0,
    );

    let indirect_specular = env * INV_PI;

    let baseline = indirect_specular + sky_reflection;

    let ambient_here = dot(env + diffuse, vec3<f32>(0.2126, 0.7152, 0.0722));
    let probe_scale = select(
        1.0,
        clamp(ambient_here / max(probe_brightness, 1e-4), PROBE_NORMALISATION_FLOOR, 1.0),
        PROBE_NORMALISATION && probe_brightness > 0.0,
    );
    let sharp = mix(baseline, probe.rgb * spec_occ * probe_scale{capsule_glow_term}, clamp(probe.a, 0.0, 1.0));

    let lobe_is_hemispherical = smoothstep(0.25, PROBE_LOBE_HEMISPHERICAL, r);
    let environment = mix(sharp, baseline, lobe_is_hemispherical);

    dbg_probe = probe.rgb * clamp(probe.a, 0.0, 1.0) * spec_occ * probe_scale * (1.0 - lobe_is_hemispherical) * fresnel;
    dbg_probe_factors = vec3<f32>(
        fresnel / max(f_max, 1e-4),
        clamp(probe.a, 0.0, 1.0),
        clamp(spec_occ * probe_scale, 0.0, 1.0),
    );

    specular = specular + environment * fresnel;

    let image_weight = (1.0 - lobe_is_hemispherical) * clamp(probe.a, 0.0, 1.0);
    let surface_spec = dot(indirect_specular, vec3<f32>(0.2126, 0.7152, 0.0722)) * (1.0 - image_weight) * fresnel;

    let bd = env_dir.xyz * 2.0 - 1.0;
    let bd_len = length(bd);
    let directionality = clamp(env_dir.w, 0.0, MAX_BOUNCE_DIRECTIONALITY);
    var bounce = env;
    var bounce_dir = vec3<f32>(0.0);
    let has_dir = bd_len > MIN_BOUNCE_DIR_LENGTH;
    if (has_dir) {{

        bounce_dir = to_player_direction(bd / bd_len);

        let shaped = 1.0 + directionality * (dot(n, bounce_dir) - dot(geom_n, bounce_dir));
        bounce = env * max(shaped, 0.0);
    }}
    var p: MaterialEnvPart;
    p.diffuse = diffuse;
    p.specular = specular;
    p.bounce = bounce;
    p.bounce_dir = bounce_dir;
    p.directionality = directionality;
    p.has_dir = has_dir;
    p.fresnel = fresnel;
    p.view_dir = view_dir;
    p.r = r;
    p.surface_spec = surface_spec;
    return p;
}}

fn shade_material_lamps(p: MaterialEnvPart, world_pos: vec3<f32>, n: vec3<f32>, env: vec3<f32>, albedo: vec3<f32>) -> vec3<f32> {{
    let view_dir = p.view_dir;
    let r = p.r;

    let pixel_long = spot_long_step(dot(view_dir, n));
    var light_dist_sq = 1e18;
    var reaching = 0u;
    let culling = light_culling();

    for (var i: u32 = surface_light_count(); i < live_light_count(); i = i + 1u) {{
        let kind = lights.lights[i].params.z;
        let to_lamp = lights.lights[i].position.xyz - world_pos;
        let dist_sq = dot(to_lamp, to_lamp);
        if (kind <= 1.5) {{

            light_dist_sq = select(light_dist_sq, min(light_dist_sq, dist_sq), lights.lights[i].params.w > -1.5);
        }} else if (SKY_SUN_NEVER_REACHES) {{

            continue;
        }}
        if (culling) {{

            if (CULL_RANGE_FIRST && kind < 1.5 && dist_sq >= lights.lights[i].params.x * lights.lights[i].params.x) {{
                continue;
            }}

            if (stationary_visibility_of(lights.lights[i].position.w) <= 0.0) {{
                continue;
            }}
            if (kind < 1.5) {{
                let reach = lights.lights[i].params.x;
                if (!CULL_RANGE_FIRST && dist_sq >= reach * reach) {{
                    continue;
                }}

                if (kind > 0.5) {{
                    let cos_outer = lights.lights[i].params.y;
                    let authored = max(lights.lights[i].direction.w - cos_outer, 0.0001);

                    let inv_dist = inverseSqrt(max(dist_sq, 1e-6));
                    let at_most = SPOT_EDGE_MIN_PIXELS * pixel_footprint * inv_dist;
                    let widen = clamp(at_most - authored, 0.0, authored * (SPOT_EDGE_MAX_WIDEN - 1.0));
                    let zero_below = cos_outer - 0.5 * widen - select(0.0, 0.5 * pixel_long * inv_dist, SPOT_EDGE_AVERAGE) - 1e-4;
                    let along = -dot(to_lamp, lights.lights[i].direction.xyz);
                    let bound_sq = zero_below * zero_below * max(dist_sq, 1e-8);
                    let outside = select(
                        along < 0.0 && along * along >= bound_sq,
                        along <= 0.0 || along * along <= bound_sq,
                        zero_below >= 0.0,
                    );
                    if (outside) {{
                        continue;
                    }}
                }}
            }} else if (receiver_sun_mask == 0.0) {{

                continue;
            }}
        }}
        reaching = reaching | (1u << i);
    }}
    let light_dist = sqrt(light_dist_sq);
    let alpha = r * r;

    let widened = clamp(alpha + LIGHT_SOURCE_RADIUS / (2.0 * max(light_dist, 0.05)), alpha, 1.0);
    let shininess = clamp(2.0 / (widened * widened) - 2.0, 1.0, MAX_SHININESS);

    let spec_strength = SPEC_STRENGTH * (1.0 - r);

    var specular = p.specular;
    let fresnel = p.fresnel;
    if (p.has_dir) {{
        let bounce_dir = p.bounce_dir;
        let directionality = p.directionality;

        let spread = directionality * (3.0 - directionality * directionality)
            / max(1.0 - directionality * directionality, 1e-3);
        let lobe = shininess * 0.25;
        let combined = max(4.0 * lobe * spread / max(lobe + spread, 1e-4), 1.0);
        let h = normalize(bounce_dir + view_dir);
        let gloss = pow(max(dot(n, h), 0.0), combined) * spec_strength * directionality
            * (combined + 2.0) / (shininess + 2.0);
        specular = specular + env * gloss;
    }}
    let bounce = p.bounce;

    let kd = albedo * (1.0 - fresnel);

    dbg_baked = bounce * kd;

    var colour = hf3(min((p.diffuse + bounce) * kd + specular, vec3<f32>(HF_MAX)));
    var spec_luma = hf(clamp(dot(specular, vec3<f32>(0.2126, 0.7152, 0.0722)) - p.surface_spec, -HF_MAX, HF_MAX));
    let kd_h = hf3(kd);
    let shininess_h = hf(shininess);
    let spec_strength_h = hf(spec_strength);

    let sharp = hf(1.0 - smoothstep(REFLECTION_SHARP_ROUGHNESS, REFLECTION_BLURRED_ROUGHNESS, r));

    if (SURFACE_LIGHTS) {{
        let surface_lit = hf3(min(surface_lights(world_pos, n), vec3<f32>(HF_MAX))) * kd_h;
        colour = min(colour + surface_lit, hf3(hf(HF_MAX)));
        dbg_direct = dbg_direct + vec3<f32>(surface_lit);
    }}

    var todo = reaching;
    loop {{
        if (todo == 0u) {{
            break;
        }}
        let i = countTrailingZeros(todo);
        todo = todo & (todo - 1u);
        let l = lights.lights[i];
        let c = light_contribution_split(l, world_pos, n, view_dir, f32(shininess_h), f32(spec_strength_h));

        if (max(max(c.diffuse.r + c.specular.r, c.diffuse.g + c.specular.g), c.diffuse.b + c.specular.b) <= 0.0) {{
            continue;
        }}

        let unshadowed = hf3(min(c.diffuse, vec3<f32>(HF_MAX))) * kd_h + hf3(min(c.specular, vec3<f32>(HF_MAX)));
        let unshadowed_luma = hf(min(dot(c.specular, vec3<f32>(0.2126, 0.7152, 0.0722)), HF_MAX));

        var shadow = stationary_visibility_of(l.position.w);
        if (!SKY_SUN_NEVER_REACHES && l.params.z > 1.5) {{
            shadow = sun_level_visibility(world_pos);
        }}

        let layer = i32(l.params.w);
        var map = -1;
        var tile = 0.0;
        if (SPOT_SHADOWS && layer >= 0 && f32(layer) < camera.shadow_params.y) {{
            map = 1 + layer;
        }} else if (l.params.z < 1.5) {{
            let k = character_shadow_tile(layer);
            if (k >= 0) {{
                map = CHARACTER_MAPS + k;
                tile = f32(1 + k);
            }}
        }} else if (!SKY_SUN_NEVER_REACHES && shadow > 0.0 && l.position.w > 0.5 && camera.shadow_params.z > 0.5) {{
            map = SUN_MOVING_MAP;
        }}
        if (map >= 0) {{
            let c = shadow_coords(world_pos, camera.moving_view_proj[map]);
            if (c.w >= 0.5) {{
                if (SPOT_SHADOWS && map > SUN_MOVING_MAP && map < CHARACTER_MAPS) {{
                    shadow = shadow * pcf_layer_at(spot_shadow_tex, layer, c.xyz);
                }} else {{
                    var at = vec4<f32>(c.xyz, tile);
                    if (!SKY_SUN_NEVER_REACHES && map == SUN_MOVING_MAP) {{
                        at = sun_moving_tile_at(c.xyz);
                    }}
                    shadow = shadow * pcf_tile_at(sun_dynamic_shadow_tex, vec2<f32>(at.w, 0.0), SUN_ATLAS_GRID, at.xyz);
                }}
            }}
        }}

        let lit = unshadowed * hf(shadow);
        colour = min(colour + lit, hf3(hf(HF_MAX)));
        spec_luma = min(spec_luma + unshadowed_luma * hf(shadow), hf(HF_MAX));

        dbg_direct = dbg_direct + vec3<f32>(lit);
    }}
    reflected_image = f32(sharp * max(spec_luma, hf(0.0)));
    return vec3<f32>(colour);
}}

fn shade_material_env(
    world_pos: vec3<f32>,
    n: vec3<f32>,
    roughness: f32,
    ao: f32,
    sky_vis: f32,
    env: vec3<f32>,

    env_dir: vec4<f32>,

    albedo: vec3<f32>,

    probe_select_pos: vec3<f32>,

    geom_n: vec3<f32>,
) -> vec3<f32> {{
    let p = shade_material_env_part(
        world_pos, n, roughness, ao, sky_vis, env, env_dir, albedo, probe_select_pos, geom_n,
    );
    return shade_material_lamps(p, world_pos, n, env, albedo);
}}

fn shade(world_pos: vec3<f32>, n: vec3<f32>) -> vec3<f32> {{
    return shade_with_sky(world_pos, n, 1.0);
}}

fn shade_with_sky(world_pos: vec3<f32>, n: vec3<f32>, sky_vis: f32) -> vec3<f32> {{
    let view_dir = normalize(cam_pos() - world_pos);
    let flash_idx = u32(camera.shadow_params.z);

    var lit = hf3(min((sky_irradiance(n) + cloud_fill(world_pos, n)) * clamp(sky_vis, 0.0, 1.0) * capsule_ambient(world_pos, n), vec3<f32>(HF_MAX)));
    for (var i: u32 = 0u; i < live_light_count(); i = i + 1u) {{
        let l = lights.lights[i];
        let arriving = light_contribution(l, world_pos, n, view_dir);

        if (max(max(arriving.r, arriving.g), arriving.b) <= 0.0) {{
            continue;
        }}
        var c = hf3(min(arriving, vec3<f32>(HF_MAX)));

        if (l.params.z > 1.5) {{
            c = c * hf(sun_visibility(l, world_pos));
        }}
        c = c * hf(stationary_visibility(l));

        let layer = i32(l.params.w);
        if (SPOT_SHADOWS && layer >= 0 && f32(layer) < camera.shadow_params.y) {{
            c = c * hf(pcf_layer(spot_shadow_tex, layer, world_pos, spot_view_proj(layer)));
        }} else if (l.params.z < 1.5) {{

            c = c * hf(character_shadow(layer, world_pos));
        }}
        lit = min(lit + c, hf3(hf(HF_MAX)));
    }}
    return vec3<f32>(lit);
}}
"#
    )
}

#[cfg(test)]
mod shadow_slot_tests {
    use super::*;

    const MAX: usize = 4;

    #[test]
    fn re_ranking_every_frame_churns_and_hysteresis_does_not() {

        let scores_at = |t: f32| -> Vec<(usize, f32)> {
            (0..6usize)
                .map(|i| {
                    let base = 3.0 + 0.05 * i as f32;
                    let jitter = 0.08 * (t * 7.0 + i as f32 * 2.3).sin();
                    let d = base + jitter;
                    (i, 1.0 / (1.0 + d * d))
                })
                .collect()
        };
        let mut naive_changes = 0usize;
        let mut held_naive: Vec<usize> = Vec::new();
        let mut hyst_changes = 0usize;
        let mut held_hyst: Vec<usize> = Vec::new();
        for step in 0..=60 {
            let scores = scores_at(step as f32 / 10.0);
            let mut naive: Vec<(usize, f32)> = scores.clone();
            naive.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
            naive.truncate(MAX);
            let mut naive: Vec<usize> = naive.into_iter().map(|(i, _)| i).collect();
            naive.sort_unstable();
            if !held_naive.is_empty() && naive != held_naive {
                naive_changes += 1;
            }
            held_naive = naive;

            let mut h = spot_shadow_slots(&scores, &held_hyst, MAX, SHADOW_SLOT_MARGIN);
            h.sort_unstable();
            if !held_hyst.is_empty() && h != held_hyst {
                hyst_changes += 1;
            }
            held_hyst = h;
        }
        println!("slot set changed: naive {naive_changes}, hysteresis {hyst_changes}");
        assert!(
            naive_changes > 0,
            "the walk no longer reshuffles the naive set, so this test is not \
             reproducing the artefact it was written for",
        );
        assert!(
            hyst_changes * 4 < naive_changes,
            "hysteresis changed the set {hyst_changes} times against the naive \
             {naive_changes}: it is not damping the churn that makes shadows \
             pop. The lamps here differ by a few percent, so NOTHING should \
             clear a 1.5x margin and the set should be almost perfectly still",
        );
    }

    #[test]
    fn free_slots_are_filled_without_a_margin() {
        let got = spot_shadow_slots(&[(7, 0.1), (9, 0.9)], &[], MAX, SHADOW_SLOT_MARGIN);
        assert_eq!(got.len(), 2, "both lights fit and both must be lit: {got:?}");
        assert_eq!(got[0], 9, "the strongest should come first: {got:?}");
    }

    #[test]
    fn a_marginally_better_light_does_not_steal_a_slot() {
        let incumbents: Vec<usize> = vec![0, 1, 2, 3];
        let close = [(0, 1.0), (1, 0.9), (2, 0.8), (3, 0.5), (4, 0.6)];
        let got = spot_shadow_slots(&close, &incumbents, MAX, SHADOW_SLOT_MARGIN);
        assert!(!got.contains(&4), "light 4 stole a slot on a 1.2x edge: {got:?}");
        assert!(got.contains(&3), "the incumbent was evicted without cause: {got:?}");

        let clear = [(0, 1.0), (1, 0.9), (2, 0.8), (3, 0.5), (4, 0.95)];
        let got = spot_shadow_slots(&clear, &incumbents, MAX, SHADOW_SLOT_MARGIN);
        assert!(got.contains(&4), "a light nearly 2x better never got a slot: {got:?}");
        assert!(!got.contains(&3), "the weakest incumbent should have gone: {got:?}");
    }

    #[test]
    fn a_light_that_is_gone_does_not_keep_its_slot() {
        let got = spot_shadow_slots(&[(5, 0.4)], &[0, 1, 2, 3], MAX, SHADOW_SLOT_MARGIN);
        assert_eq!(got, vec![5], "a departed incumbent still holds a slot: {got:?}");
    }

    #[test]
    fn a_light_whose_shadow_begins_past_its_range_casts_none() {
        let spot = Light {
            position: Vec3::ZERO,
            direction: Vec3::NEG_Y,
            kind: LightKind::Spot,
            color: crate::renderer::Color3(255, 255, 255, 255),
            intensity: 1.0,
            range: 8.0,
            cone_angle_deg: 180.0,
            inner_cone_angle_deg: 0.0,
            mask_channel: None,
            shadow_near: None,
            source_radius: 0.0,
            in_level_bake: true,
        };
        assert!(spot.casts_shadow(), "a fixture's");
        assert!(Light { shadow_near: Some(0.02), ..spot }.casts_shadow(), "a flashlight's, from its glass");
        assert!(!Light { shadow_near: Some(8.0), ..spot }.casts_shadow());
        let surface = Light { shadow_near: Some(f32::INFINITY), ..spot };
        assert!(!surface.casts_shadow());

        let gpu = pack_lights(&[spot, surface], 2, &[], false, true);
        assert_eq!(gpu.lights[0].params[3], -1.0);
        assert_eq!(gpu.lights[1].params[3], SURFACE_LIGHT);
        assert!(SURFACE_LIGHT < -1.5, "below the shader's test");

        let wide = pack_lights(
            &[Light { source_radius: 0.6, ..surface }, Light { source_radius: 0.6, ..spot }],
            2,
            &[],
            false,
            true,
        );
        assert_eq!(wide.lights[0].params[3], SURFACE_LIGHT - 0.6);
        assert_eq!(wide.lights[1].params[3], -1.0);
        let src = wgsl_lights_block(0, 1);
        assert_eq!(SURFACE_LIGHT, -2.0, "the lights block reads the radius back as -2 - params.w");
        assert_eq!(
            src.matches("let source = max(-2.0 - l.params.w, 0.0);").count(),
            2,
            "both lamp functions read the patch's radius back",
        );
        assert!(src.contains("let source = max(-2.0 - lights.lights[i].params.w, 0.0);"), "and the lit surfaces' own loop");
    }
}

#[cfg(test)]
mod budget_tests {
    use super::*;
    use crate::renderer::Color3;

    fn light(kind: LightKind, pos: Vec3, intensity: f32) -> Light {
        Light {
            mask_channel: None,
            shadow_near: None,
            source_radius: 0.0,
            in_level_bake: true,
            position: pos,
            direction: Vec3::NEG_Z,
            kind,
            color: Color3(255, 255, 255, 255),
            intensity,
            range: 20.0,
            cone_angle_deg: 45.0,
            inner_cone_angle_deg: 0.0,
        }
    }
    fn point(x: f32, intensity: f32) -> Light {
        light(LightKind::Point, Vec3::new(x, 0.0, 0.0), intensity)
    }

    #[test]
    fn the_lit_surfaces_lights_lead_the_list_and_are_counted_there() {
        let lamp = point(1.0, 1.0);
        let bounce = Light {
            kind: LightKind::Spot,
            direction: Vec3::X,
            range: 6.0,
            cone_angle_deg: 200.0,
            inner_cone_angle_deg: 0.0,
            shadow_near: Some(f32::INFINITY),
            source_radius: 0.3,
            in_level_bake: false,
            ..point(2.0, 1.0)
        };
        assert!(bounce.is_surface_light());
        assert!(Light { kind: LightKind::Point, ..bounce }.is_surface_light());
        assert!(!lamp.is_surface_light(), "a lamp casts");
        let narrow = Light { cone_angle_deg: 40.0, inner_cone_angle_deg: 20.0, ..bounce };
        assert!(!narrow.casts_shadow() && !narrow.is_surface_light(), "a narrow edge keeps the lamp loop's averaging");
        let ls = [lamp, bounce, Light { intensity: 2.0, ..lamp }, Light { source_radius: 0.5, ..bounce }];
        assert_eq!(rank_for_budget_indices(&ls, MAX_LIGHTS), vec![1, 3, 0, 2]);

        assert_eq!(rank_for_budget_indices(&ls, 3), vec![1, 2, 0]);
        assert_eq!(rank_for_budget_indices(&ls, 2), vec![2, 0]);
        let ranked = rank_for_budget(&ls, MAX_LIGHTS);
        assert_eq!(pack_lights(&ranked, 4, &[], false, true).surface_lights[0], 2);
        assert_eq!(pack_lights(&[lamp, bounce], 2, &[], false, true).surface_lights[0], 0, "only a leading run");
        assert_eq!(pack_lights(&[bounce, lamp], 0, &[], false, true).surface_lights[0], 0, "never a baked tail");

        let src = wgsl_lights_block(0, 1);
        assert!(src.contains(SURFACE_LIGHTS_ON));
        assert!(without_spot_shadows(src).contains(SURFACE_LIGHTS_OFF));
    }

    #[test]
    fn a_scene_inside_the_budget_is_passed_through_untouched() {

        let ls = vec![point(50.0, 1.0), point(1.0, 9.0), point(10.0, 3.0)];
        let ranked = rank_for_budget(&ls, MAX_LIGHTS);
        let positions: Vec<f32> = ranked.iter().map(|l| l.position.x).collect();
        assert_eq!(positions, vec![50.0, 1.0, 10.0]);
    }

    #[test]
    fn the_lamp_in_your_face_survives_and_the_distant_one_does_not() {

        let mut ls: Vec<Light> = (0..MAX_LIGHTS).map(|i| point(80.0 + i as f32, 5.0)).collect();
        ls.push(point(2.0, 5.0));
        let ranked = rank_for_budget(&ls, MAX_LIGHTS);
        assert_eq!(ranked.len(), MAX_LIGHTS);
        assert!(
            ranked.iter().any(|l| l.position.x == 2.0),
            "the nearest light must not be the one dropped",
        );
    }

    #[test]
    fn the_sun_is_never_dropped() {

        let mut ls: Vec<Light> = (0..MAX_LIGHTS).map(|_| point(0.5, 100.0)).collect();
        ls.push(light(LightKind::Directional, Vec3::ZERO, 0.1));
        let ranked = rank_for_budget(&ls, MAX_LIGHTS);
        assert!(ranked.iter().any(|l| l.kind == LightKind::Directional));
    }

    #[test]
    fn brightness_counts_as_well_as_distance() {

        let ls = vec![point(5.0, 0.1), point(5.0, 50.0), point(5.0, 1.0)];
        let ranked = rank_for_budget(&ls, 1);
        assert_eq!(ranked[0].intensity, 50.0);
    }

    #[test]
    fn ranking_is_stable_for_equal_lights() {

        let ls: Vec<Light> = (0..MAX_LIGHTS + 4).map(|i| point(3.0, 2.0)).collect();
        let _ = ls;
        let ls: Vec<Light> = (0..MAX_LIGHTS + 4)
            .map(|i| light(LightKind::Point, Vec3::new(3.0, 0.0, i as f32 * 0.0), 2.0))
            .collect();
        let a = rank_for_budget(&ls, MAX_LIGHTS);
        let b = rank_for_budget(&ls, MAX_LIGHTS);
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(x.position, y.position);
        }
    }

    #[test]
    fn a_light_at_the_players_own_position_is_finite() {
        assert!(influence_score(&point(0.0, 5.0)).is_finite());
    }

    #[test]
    fn an_empty_scene_ranks_to_nothing() {
        assert!(rank_for_budget(&[], MAX_LIGHTS).is_empty());
    }
}

#[cfg(test)]
mod energy_tests {

    use super::*;

    fn split(roughness: f32, view_deg: f32) -> (f32, f32) {
        let r = roughness.clamp(0.04, 1.0);
        let cos_v = view_deg.to_radians().cos().clamp(0.0, 1.0);
        let f0 = 0.04f32;
        let f_max = (1.0 - r).max(f0);
        let fresnel = f0 + (f_max - f0) * (1.0 - cos_v).powi(5);

        (fresnel, 1.0 - fresnel)
    }

    #[test]
    fn a_white_surface_never_emits_more_than_arrives() {
        for r in [0.048f32, 0.2, 0.437, 0.596, 0.9, 1.0] {
            for deg in [0.0f32, 30.0, 45.0, 60.0, 70.0, 80.0, 85.0, 89.0] {
                let (spec, diff) = split(r, deg);
                let total = spec + diff;
                assert!(
                    total <= 1.0001,
                    "roughness {r} at {deg} degrees emits {total} of the light that arrives",
                );
            }
        }
    }

    #[test]
    fn looking_straight_at_a_surface_barely_changes_it() {
        for r in [0.048f32, 0.437, 0.9] {
            let (spec, diff) = split(r, 0.0);
            assert!((spec - 0.04).abs() < 1e-3, "head-on reflectance is {spec}, not 4%");
            assert!(diff > 0.95, "head-on diffuse dropped to {diff}");
        }
    }

    #[test]
    fn a_rough_surface_reflects_less_at_grazing_than_a_polished_one() {
        let (rough, _) = split(0.9, 85.0);
        let (polished, _) = split(0.048, 85.0);
        assert!(
            polished > rough * 2.0,
            "polished reflects {polished} and rough {rough} at 85 degrees; the \
             roughness cap is not separating them",
        );
        assert!(rough < 0.15, "a nearly-matte surface reflects {rough} at grazing");
    }

    #[test]
    fn the_shader_conserves_energy_and_caps_the_fresnel() {

        let src = wgsl_lights_block(0, 1);
        let code: String = src
            .lines()
            .filter(|l| !l.trim_start().starts_with("//"))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            code.contains("let f_max = max(1.0 - r, f0);"),
            "the Fresnel is no longer capped by roughness",
        );

        assert!(
            code.contains("let kd = albedo * (1.0 - fresnel);")
                && code.contains("var colour = hf3(min((p.diffuse + bounce) * kd + specular, vec3<f32>(HF_MAX)));")
                && code.contains("let kd_h = hf3(kd);")
                && code.contains(
                    "let unshadowed = hf3(min(c.diffuse, vec3<f32>(HF_MAX))) * kd_h + hf3(min(c.specular, vec3<f32>(HF_MAX)));"
                )
                && code.contains("let lit = unshadowed * hf(shadow);"),
            "the diffuse no longer gives up what the specular reflects",
        );
        assert!(
            !code.contains("fresnel * (1.0 - r)"),
            "roughness is being applied to the Fresnel twice again",
        );
    }
}

#[cfg(test)]
mod horizon_tests {

    use super::*;

    const GROUND_ALBEDO_F: f32 = 0.28;

    fn env(sky_up: f32, sky_at_dir: f32, dir_y: f32) -> f32 {
        let ground = sky_up * GROUND_ALBEDO_F;
        let t = ((dir_y + 0.15) / 0.30).clamp(0.0, 1.0);
        let smooth = t * t * (3.0 - 2.0 * t);
        ground * (1.0 - smooth) + sky_at_dir * smooth
    }

    #[test]
    fn looking_down_does_not_reflect_blackness() {

        let got = env(1.0, 0.0, -0.9);
        assert!(
            got > 0.2,
            "a downward reflection came back at {got} under a bright sky; that \
             is the dark band on the wall",
        );
    }

    #[test]
    fn looking_up_is_still_the_sky() {
        assert!((env(1.0, 1.0, 0.9) - 1.0).abs() < 1e-5);
        assert!((env(1.0, 0.4, 0.9) - 0.4).abs() < 1e-5, "the sky was contaminated by the ground");
    }

    #[test]
    fn the_horizon_is_soft() {
        let below = env(1.0, 0.0, -0.05);
        let above = env(1.0, 0.0, 0.05);
        assert!(below > 0.0 && above > 0.0);
        assert!(
            (below - above).abs() < 0.25,
            "the horizon steps from {below} to {above} across a tenth of a unit",
        );
    }

    #[test]
    fn a_dark_sky_has_a_dark_ground() {
        assert!(env(0.0, 0.0, -1.0) < 1e-6);
    }

    #[test]
    fn the_probe_is_sampled_with_an_explicit_level() {
        let src = wgsl_lights_block(0, 1);
        let code: String = src
            .lines()
            .filter(|l| !l.trim_start().starts_with("//"))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            code.contains(
                "textureSampleLevel(probe_cube, probe_samp, sample_dir, layer, probe_lod)",
            ) && code.contains("let layer = i32(camera.probe_boxes[best * 3].w);"),
            "the probe is not sampled with an explicit level",
        );
        assert!(
            !code.contains("textureSample(probe_cube"),
            "the probe is back to implicit derivatives inside a branchy search",
        );
    }

    #[test]
    fn the_shader_uses_the_horizon_blend_for_reflections() {
        let src = wgsl_lights_block(0, 1);
        let code: String = src
            .lines()
            .filter(|l| !l.trim_start().starts_with("//"))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            code.contains("sky_reflection = environment_radiance(refl) * occ;"),
            "the specular environment is back to reading the raw sky harmonics",
        );

        assert!(
            code.contains("diffuse = sky_irradiance(n) * occ;"),
            "the diffuse ambient is no longer the plain sky along the normal",
        );

        assert!(
            code.contains("mix(ground, sky, smoothstep(-0.15, 0.15, dir.y))"),
            "the horizon is no longer a soft blend; a step here draws a line \
             across every reflective surface at eye level",
        );
        assert!(
            code.contains(&format!("const GROUND_ALBEDO: f32 = {GROUND_ALBEDO_F};")),
            "the ground albedo the mirror above assumes is not the shader's",
        );
    }
}

#[cfg(test)]
mod sky_agreement_tests {

    use super::*;
    use space_soup_sky::SkyIrradiance;

    fn wgsl_sky_irradiance(sh: &[[f32; 3]; 9], n: [f32; 3]) -> [f32; 3] {
        let (x, y, z) = (n[0], n[1], n[2]);
        let b = [
            0.282095,
            0.488603 * y,
            0.488603 * z,
            0.488603 * x,
            1.092548 * x * y,
            1.092548 * y * z,
            0.315392 * (3.0 * z * z - 1.0),
            1.092548 * x * z,
            0.546274 * (x * x - y * y),
        ];
        let a = [1.0, 0.6666667, 0.6666667, 0.6666667, 0.25, 0.25, 0.25, 0.25, 0.25];
        let mut e = [0.0f32; 3];
        for i in 0..9 {
            for c in 0..3 {
                e[c] += sh[i][c] * b[i] * a[i];
            }
        }

        for c in 0..3 {
            e[c] = e[c].max(0.0);
        }
        e
    }

    fn asymmetric_sky() -> SkyIrradiance {
        let mut irr = SkyIrradiance::default();
        for (i, row) in irr.sh.iter_mut().enumerate() {
            let f = i as f32;
            *row = [0.8 - 0.06 * f, 0.45 + 0.04 * f, 0.15 + 0.05 * f];
        }
        irr
    }

    #[test]
    fn the_baker_and_the_shader_agree_in_every_direction() {
        let irr = asymmetric_sky();

        let dirs: [[f32; 3]; 8] = [
            [0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0], [0.0, 0.0, -1.0],
            [0.5773503, 0.5773503, 0.5773503], [-0.2672612, 0.5345225, -0.8017837],
        ];
        for d in dirs {
            let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            assert!((len - 1.0).abs() < 1e-5, "test direction {d:?} is not unit length");
            let baker = irr.evaluate(d);
            let shader = wgsl_sky_irradiance(&irr.sh, d);
            for c in 0..3 {
                assert!(
                    (baker[c] - shader[c]).abs() < 1e-4,
                    "direction {d:?} channel {c}: baker {} vs shader {}",
                    baker[c],
                    shader[c],
                );
            }
        }
    }

    #[test]
    fn the_transcription_matches_the_shader_text() {
        let src = wgsl_lights_block(0, 1);
        let code: String = src
            .lines()
            .filter(|l| !l.trim_start().starts_with("//"))
            .collect::<Vec<_>>()
            .join("\n");

        for term in [
            "e = e + camera.sky_sh[0].rgb * 0.282095 * 1.0;",
            "e = e + camera.sky_sh[1].rgb * (0.488603 * y) * 0.6666667;",
            "e = e + camera.sky_sh[2].rgb * (0.488603 * z) * 0.6666667;",
            "e = e + camera.sky_sh[3].rgb * (0.488603 * x) * 0.6666667;",
            "e = e + camera.sky_sh[4].rgb * (1.092548 * x * y) * 0.25;",
            "e = e + camera.sky_sh[5].rgb * (1.092548 * y * z) * 0.25;",
            "e = e + camera.sky_sh[6].rgb * (0.315392 * (3.0 * z * z - 1.0)) * 0.25;",
            "e = e + camera.sky_sh[7].rgb * (1.092548 * x * z) * 0.25;",
            "e = e + camera.sky_sh[8].rgb * (0.546274 * (x * x - y * y)) * 0.25;",
        ] {
            assert!(
                code.contains(term),
                "the shader's harmonic sum no longer contains `{term}`, so the \
                 transcription in this module is measuring something else",
            );
        }
        assert!(code.contains("return max(e, vec3<f32>(0.0));"), "the shader stopped clamping");
    }

    #[test]
    fn a_uniform_sky_lights_every_direction_to_its_own_value() {
        let irr = SkyIrradiance::flat(0.5);
        for d in [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.577, -0.577, 0.577]] {
            let e = irr.evaluate(d);
            for c in 0..3 {
                assert!(
                    (e[c] - 0.5).abs() < 1e-4,
                    "a flat 0.5 sky lit {d:?} to {} in channel {c}",
                    e[c],
                );
            }
        }
    }
}

#[cfg(test)]
mod probe_selection_tests {
    use super::*;

    fn shader() -> String {
        wgsl_lights_block(0, 1)
    }

    #[test]
    fn the_nearest_capture_point_wins_within_one_volume() {
        let code = shader();
        assert!(
            code.contains("let to_centre = camera.probe_boxes[i * 3].xyz - select_world;"),
            "probe selection no longer measures distance to the capture point; \
             every cell of a room would tie and the first would always win",
        );
        assert!(
            code.contains("same_room_but_nearer"),
            "the nearest-capture-point rule is gone",
        );
    }

    #[test]
    fn a_tighter_volume_still_wins_over_one_that_encloses_it() {
        let code = shader();
        assert!(
            code.contains("let tighter = volume < best_volume * 0.999;"),
            "nesting no longer decides between volumes of different size; a room \
             inside the outdoor volume could reflect the sky",
        );
    }

    #[test]
    fn projection_uses_the_box_the_bake_supplied() {
        let code = shader();
        assert!(code.contains("let lo = camera.probe_boxes[slot * 3 + 1].xyz;"));
        assert!(code.contains("let hi = camera.probe_boxes[slot * 3 + 2].xyz;"));
    }
}

#[cfg(test)]
mod probe_blend_tests {
    use super::*;

    fn shader() -> String {
        wgsl_lights_block(0, 1)
    }

    #[test]
    fn the_two_nearest_cells_are_blended() {
        let code = shader();
        assert!(code.contains("var second = -1;"), "the runner-up cell is gone");
        assert!(
            code.contains("band = 0.5 + 0.5 * smoothstep(0.0, PROBE_BLEND_BAND, gap);") && code.contains("own = mix(far, near, band);"),
            "probe selection is back to picking one cell, which puts the cell \
             boundaries back on screen as squares",
        );
        assert!(
            code.contains("let gap = sqrt(second_dist) - sqrt(best_dist);"),
            "the blend is no longer weighted by distance to the capture points",
        );
    }

    #[test]
    fn cells_of_different_volumes_are_not_blended() {
        let code = shader();
        assert!(
            code.contains("let room = camera.probe_boxes[i * 3 + 2].w;")
                && code.contains("let same_room_but_nearer = room == best_room && dist < best_dist;")
                && code.contains("}} else if (room == best_room && dist < second_dist) {".replace("}}", "}").as_str()),
            "the runner-up is taken without checking it belongs to the same \
             room; an interior wall could blend in the outdoor probe",
        );

        let tighter = code.find("if (tighter) {").expect("the tighter branch is gone");
        let reset = code[tighter..].find("second = -1;").expect("tighter no longer resets the runner-up");
        assert!(reset < 400, "the runner-up reset is not in the tighter branch");
    }

    #[test]
    fn a_single_probe_returns_without_a_second_sample() {
        let code = shader();
        assert!(
            code.contains("if (second >= 0) {") && code.contains("own = near;"),
            "with no runner-up the shader would index probe slot -1",
        );
    }

    #[test]
    fn every_path_goes_through_the_portals() {
        let code = shader();

        let env = &code[code.find("fn probe_environment(").unwrap()..code.find("fn probe_trace(").unwrap()];
        assert!(!env.contains("return vec4<f32>(0.0);"), "an early return skips the doorways again");
        assert!(env.contains("return probe_through_portals(own, own_room, select_world, world_pos, d, probe_lod);"));
    }

    #[test]
    fn the_outdoor_twins_rough_answer_is_the_full_pass_s() {
        let code = shader();
        let bare = |text: &str| -> String {
            text.lines()
                .filter(|l| !l.trim_start().starts_with("//"))
                .flat_map(|l| l.chars())
                .filter(|c| !c.is_whitespace())
                .collect()
        };
        let at = |hay: &str, needle: &str| hay.find(needle).unwrap_or_else(|| panic!("missing: {needle}"));
        let env = &code[at(&code, "fn probe_environment(")..at(&code, "fn probe_outdoor_untraced(")];
        let full = &env[at(env, "    var own = vec4<f32>(0.0);")..at(env, "    return probe_through_portals(own")];
        let full = bare(full).replace("varown_room=-1.0;", "").replace("own_room=best_room;", "");
        let twin = &code[at(&code, "fn probe_outdoor_untraced(")..];
        let twin = &twin[at(twin, "    var mine = vec4<f32>(0.0);")..at(twin, "    return mine;")];
        let mut twin = bare(twin);
        for (mine, theirs) in [
            ("pick.second_dist", "second_dist"),
            ("pick.best_dist", "best_dist"),
            ("pick.second", "second"),
            ("pick.best", "best"),
            ("far_photo_dir", "far_dir"),
            ("far_photo", "far"),
            ("near_photo", "near"),
            ("near_dir", "sample_dir"),
            ("near_layer", "layer"),
            ("away_lod", "probe_lod"),
            ("away", "d"),
            ("apart", "gap"),
            ("share", "band"),
            ("mine", "own"),
        ] {
            twin = twin.replace(mine, theirs);
        }
        assert_eq!(twin, full);

        assert!(code.contains("if (roughness > PROBE_TRACE_MAX_ROUGHNESS || camera.portal_params.y > 0.5) {"));
        let branch = &env[at(env, "    if (PROBE_ALL_OUTDOORS) {")..at(env, "    let face_given = ")];
        assert!(at(branch, "if (roughness > PROBE_TRACE_MAX_ROUGHNESS) {") < at(branch, "probe_escape_colour("));
        assert!(branch.contains("return probe_outdoor_untraced(probe_choose(select_world, volume_world), world_pos, away, away_lod);"));
    }

    #[test]
    fn the_portal_fades_come_from_the_constants() {
        let code = shader();
        assert!(code.contains(&format!("const PORTAL_FADE: f32 = {:?};", PROBE_PORTAL_FADE)));
        assert!(code.contains(&format!("const PORTAL_SIDE_FADE: f32 = {:?};", PROBE_PORTAL_SIDE_FADE)));
    }

    fn portal_high_weight(t: f32, lo: f32, hi: f32, side: f32) -> f32 {
        let smooth = |e0: f32, e1: f32, x: f32| {
            let u = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
            u * u * (3.0 - 2.0 * u)
        };
        let reach = 1.0 - smooth(0.0, PROBE_PORTAL_SIDE_FADE, side);
        reach * smooth(lo - PROBE_PORTAL_FADE, hi + PROBE_PORTAL_FADE, t)
    }

    #[test]
    fn a_doorway_hands_over_continuously() {
        let (lo, hi) = (2.5, 3.1);
        assert_eq!(portal_high_weight(lo - PROBE_PORTAL_FADE, lo, hi, 0.0), 0.0);
        assert_eq!(portal_high_weight(hi + PROBE_PORTAL_FADE, lo, hi, 0.0), 1.0);
        assert!((portal_high_weight((lo + hi) * 0.5, lo, hi, 0.0) - 0.5).abs() < 1e-6);
        let mut last = 0.0;
        for i in 0..=100 {
            let t = lo - PROBE_PORTAL_FADE + (hi - lo + 2.0 * PROBE_PORTAL_FADE) * i as f32 / 100.0;
            let w = portal_high_weight(t, lo, hi, 0.0);
            assert!(w >= last && w - last < 0.05, "the doorway weight jumps by {} at {t}", w - last);
            last = w;
        }

        assert_eq!(portal_high_weight((lo + hi) * 0.5, lo, hi, PROBE_PORTAL_SIDE_FADE), 0.0);
    }

    fn probe_near_weight(best_dist_sq: f32, second_dist_sq: f32, band: f32) -> f32 {
        let gap = second_dist_sq.sqrt() - best_dist_sq.sqrt();
        let t = (gap / band).clamp(0.0, 1.0);
        0.5 + 0.5 * (t * t * (3.0 - 2.0 * t))
    }

    #[test]
    fn a_fragment_clear_of_the_boundary_takes_one_probe_alone() {
        let w = probe_near_weight(1.0, 4.0, 0.6);
        assert!((w - 1.0).abs() < 1e-6, "the far probe still contributes {} here", 1.0 - w);
    }

    #[test]
    fn on_the_boundary_the_two_probes_share_equally() {
        assert!((probe_near_weight(2.25, 2.25, 0.6) - 0.5).abs() < 1e-6);

        let just_inside = probe_near_weight(1.0, 1.1f32.powi(2), 0.6);
        assert!(just_inside > 0.5 && just_inside < 0.75, "weight {just_inside} jumps near the boundary");
    }

    #[test]
    fn the_shader_blends_probes_only_within_the_band() {
        let code = wgsl_lights_block(0, 1);
        assert!(
            code.contains("let gap = sqrt(second_dist) - sqrt(best_dist);")
                && code.contains("band = 0.5 + 0.5 * smoothstep(0.0, PROBE_BLEND_BAND, gap);") && code.contains("own = mix(far, near, band);"),
            "the probe blend is no longer the boundary band; blending two photographs \
             everywhere shows every lamp in the room twice",
        );
        assert!(code.contains("const PROBE_BLEND_BAND: f32 = 2.0;"));
        assert!(!code.contains("1.0 / max(best_dist"), "the everywhere-blend is back");
    }

    #[test]
    fn the_far_probe_gets_its_own_parallax() {
        let code = wgsl_lights_block(0, 1);
        assert!(code.contains("let sample_dir = probe_parallax_direction(world_pos, d, best);"));

        assert!(code.contains("let far_dir = probe_parallax_direction(world_pos, d, second);"));

        assert!(code.contains("let b1 = camera.probe_boxes[select(s0, s1, has1) * 3];"));
        assert!(code.contains("let v1 = h - b1.xyz;"));
        assert!(code.contains("let col1 = textureSampleLevel(probe_cube, probe_samp, v1, i32(b1.w), lod1);"));
        assert!(
            code.contains("let lod1 = probe_hit_lod(roughness, t, sqrt(select(d0, d1, has1)));"),
            "the far photograph's blur is not its own",
        );
        assert!(
            code.contains("probe_cube, probe_samp, far_dir, i32(camera.probe_boxes[second * 3].w), probe_lod"),
            "the far photograph is read along the near one's direction again",
        );
    }
}

#[cfg(test)]
mod probe_sharpness_tests {
    use super::*;

    #[test]
    fn even_a_mirror_smooth_surface_reads_a_blurred_level() {
        let code = wgsl_lights_block(0, 1);
        assert!(
            code.contains(
                "clamp(roughness * PROBE_ROUGHNESS_MIPS, PROBE_MIN_LOD, PROBE_MAX_LOD)"
            ),
            "the probe LOD floor is gone; a polished surface would sample the \
             cube's sharpest level and mirror whatever the probe happened to see",
        );
        assert!(
            code.contains("const PROBE_MIN_LOD: f32 = 0.5;"),
            "the LOD floor moved; it is what keeps the probe soft and leaves \
             sharp reflections to the screen",
        );
    }

    #[test]
    fn roughness_still_has_room_to_vary() {
        let code = wgsl_lights_block(0, 1);
        assert!(code.contains("const PROBE_MAX_LOD: f32 = 7.0;"));

        let lod = |r: f32| (r * 7.0).clamp(0.5, 7.0);
        assert!(
            lod(0.6) > lod(0.05) + 1.0,
            "brick and marble now sample the same level; roughness has stopped \
             changing the reflection at all",
        );
    }
}

#[cfg(test)]
mod shadow_skip_tests {
    use super::*;

    #[test]
    fn a_light_that_contributes_nothing_is_not_shadow_tested() {
        let code = wgsl_lights_block(0, 1);

        let env = code.find("fn shade_material_lamps(").expect("shade_material_lamps is gone");
        let guard = code[env..]
            .find("if (max(max(c.diffuse.r + c.specular.r, c.diffuse.g + c.specular.g), c.diffuse.b + c.specular.b) <= 0.0) {")
            .expect("the material path shadow-tests lights that contribute nothing");
        let kernel = code[env..].find("shadow_coords(world_pos, camera.moving_view_proj[map])").expect("the shadow test is gone");
        assert!(guard < kernel, "the guard runs after the shadow kernel it should skip");

        let sky = code.find("fn shade_with_sky(").expect("shade_with_sky is gone");
        let guard = code[sky..]
            .find("if (max(max(arriving.r, arriving.g), arriving.b) <= 0.0) {")
            .expect("the sky path shadow-tests lights that contribute nothing");
        let kernel = code[sky..].find("pcf").expect("the sky path's shadow test is gone");
        assert!(guard < kernel, "the sky path's guard runs after its shadow kernel");
    }
}

#[cfg(test)]
mod probe_normalisation_tests {

    fn probe_scale(ambient_here: f32, probe_brightness: f32, floor: f32) -> f32 {
        if probe_brightness > 0.0 {
            (ambient_here / probe_brightness.max(1e-4)).clamp(floor, 1.0)
        } else {
            1.0
        }
    }

    #[test]
    fn a_dark_corner_reflects_darkly_and_a_lit_one_as_photographed() {

        assert!((probe_scale(0.005, 0.05, 0.0) - 0.1).abs() < 1e-6, "a corner at a tenth dims to a tenth");
        assert_eq!(probe_scale(0.2, 0.05, 0.0), 1.0, "brighter than the photograph is NOT brightened");
        assert_eq!(probe_scale(0.0, 0.05, 0.0), 0.0, "no light reflects nothing");
        assert_eq!(probe_scale(0.001, 0.0, 0.0), 1.0, "an unknown probe is left alone");
    }

    #[test]
    fn the_shader_normalises_the_probe_by_the_ambient_light_here() {
        let code = super::wgsl_lights_block(0, 1);
        assert!(code.contains("const PROBE_NORMALISATION: bool = true;"));
        assert!(code.contains("let ambient_here = dot(env + diffuse, vec3<f32>(0.2126, 0.7152, 0.0722));"));
        assert!(code.contains(
            "clamp(ambient_here / max(probe_brightness, 1e-4), PROBE_NORMALISATION_FLOOR, 1.0)"
        ));
        assert!(code.contains("PROBE_NORMALISATION && probe_brightness > 0.0"));

        assert!(
            code.contains(
                "let sharp = mix(baseline, probe.rgb * spec_occ * probe_scale + capsule_glow * spec_occ, clamp(probe.a, 0.0, 1.0));"
            ),
            "the scale is computed but not applied to the probe",
        );

        let from_pass = super::wgsl_lights_block_with(
            0,
            1,
            super::LightsBlockOptions { probe_from_pass: true, ..Default::default() },
        );
        assert!(from_pass.contains("let sharp = mix(baseline, probe.rgb * spec_occ * probe_scale, clamp(probe.a, 0.0, 1.0));"));
        assert!(from_pass.contains("probe_pass_compress(probe.rgb * probe_scale + capsule_glow) * a"));

        assert!(code.contains("probe_brightness = camera.probe_boxes[best * 3 + 1].w;"));
        assert!(code.contains("probe_brightness = 0.0;"));
    }
}

#[cfg(test)]
mod probe_box_margin_tests {

    fn inside(p: [f32; 3], lo: [f32; 3], hi: [f32; 3], margin: f32) -> bool {
        (0..3).all(|i| p[i] >= lo[i] - margin && p[i] <= hi[i] + margin)
    }

    #[test]
    fn a_seam_pixel_just_past_the_wall_keeps_its_room() {

        let (lo, hi) = ([-2.7, 0.0, -15.7], [2.7, 3.1, 3.7]);
        let seam = [2.71, 3.11, -5.0];
        assert!(!inside(seam, lo, hi, 0.0), "without a margin the seam falls out of its room");
        assert!(inside(seam, lo, hi, 0.05), "with the margin the seam keeps its room's probe");

        assert!(!inside([3.0, 1.0, -5.0], lo, hi, 0.05));
    }

    #[test]
    fn the_shader_tests_the_box_with_the_margin() {
        let code = super::wgsl_lights_block(0, 1);

        assert!(
            code.contains(&format!(
                "const PROBE_BOX_MARGIN: f32 = {:?};",
                super::probe_box_margin()
            )),
            "the shader's probe margin is not the one `probe_box_margin` computes",
        );

        assert!(
            super::probe_box_margin() >= 0.05,
            "the margin got NARROWER than its full-resolution value",
        );

        assert!(
            super::probe_box_margin() < 0.15,
            "the margin is now more than half the thinnest wall in test_room \
             (0.3 m); it will start claiming the neighbouring room's surfaces",
        );
        assert!(
            code.contains(
                "if (any(volume_world < lo - vec3<f32>(PROBE_BOX_MARGIN)) || any(volume_world > hi + vec3<f32>(PROBE_BOX_MARGIN)))"
            ),
            "the box test moved off `select_world`; if it goes back to the \
             fragment's own position the room-seam dots come straight back",
        );
    }

    #[test]
    fn the_probe_is_chosen_from_a_point_that_is_not_the_fragment() {

        let code = super::wgsl_lights_block(0, 1);
        let sel = code
            .find("if (any(volume_world < lo")
            .expect("the candidate loop no longer tests the selection point");
        let parallax = code
            .find("let t_hi = (hi - parallax_pos) * inv;")
            .expect("the parallax projection no longer uses the clamped position");

        assert!(
            code.contains("let parallax_pos = clamp(world_pos, lo, hi);"),
            "the parallax position is no longer clamped into the box; an edge \
             pixel extrapolated outside it falls back to an uncorrected \
             reflection direction while its neighbours do not",
        );
        assert!(
            code.contains("let to_hit = parallax_pos + d * dist - centre;"),
            "the parallax sample direction went back to the unclamped position",
        );

        assert!(
            code.contains("if (dist >= 0.0) {"),
            "the parallax rejects a zero exit distance again -- the seam is back",
        );
        assert!(
            sel < parallax,
            "selection must happen before the parallax it feeds",
        );
        assert!(
            code.contains("let select_world = to_world_space(select_pos);"),
            "the selection point stopped being taken into world space, so it \
             is being compared against a box baked in a different frame",
        );
    }
}

#[cfg(test)]
mod shader_occupancy_tests {

    fn body() -> String {
        super::wgsl_lights_block(0, 1)
            .lines()
            .map(|l| match l.find("//") {
                Some(i) => &l[..i],
                None => l,
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn the_lights_block_declares_no_local_arrays() {
        let b = body();

        for (i, line) in b.lines().enumerate() {
            let t = line.trim();

            let local_array_decl = t.starts_with("var<function>")
                || (t.starts_with("var ") && (t.contains(": array<") || t.contains("= array<")));
            assert!(
                !local_array_decl,
                "line {i} declares a LOCAL array, which spills to scratch memory \
                 on Adreno if it is ever indexed dynamically: {t}",
            );
        }
    }

    #[test]
    fn the_nearest_light_pass_does_not_copy_the_whole_light() {
        let b = body();
        let start = b.find("var light_dist_sq = 1e18;").expect("the nearest-light pass is gone");
        let end = b[start..].find("let light_dist = sqrt(light_dist_sq);").expect("the pass lost its terminator") + start;
        let loop_body = &b[start..end];
        assert!(
            !loop_body.contains("let li = lights.lights[i];"),
            "the nearest-light loop copies the whole Light struct again",
        );
        assert!(
            loop_body.contains("lights.lights[i].position.xyz"),
            "the nearest-light loop no longer reads the position field directly",
        );

        assert!(
            !loop_body.contains("distance(") && !loop_body.contains("length(") && !loop_body.contains("sqrt("),
            "the nearest-light loop takes a square root per lamp again",
        );
    }
}

#[cfg(test)]
mod fresnel_aliasing_tests {

    #[test]
    fn the_environment_normal_is_blended_toward_the_geometry_by_roughness() {
        let code = super::wgsl_lights_block(0, 1);
        assert!(
            code.contains("let env_n = normalize(mix(n, geom_n, smoothstep(0.1, 0.4, r)));"),
            "the environment normal is no longer blended toward the geometric one",
        );
        for use_env in ["reflect(-view_dir, env_n)", "clamp(dot(env_n, view_dir), 0.0, 1.0)"] {
            assert!(code.contains(use_env), "the environment term stopped using env_n: {use_env}");
        }
    }

    #[test]
    fn a_polished_surface_still_sees_its_own_normal_map() {

        let blend_at = |r: f32| {
            let t = ((r - 0.1) / 0.3).clamp(0.0, 1.0);
            t * t * (3.0 - 2.0 * t)
        };
        assert_eq!(blend_at(0.05), 0.0, "a polished surface lost its normal detail");
        assert!(blend_at(0.5) > 0.99, "a rough surface is still driven by its normal map");
        assert!(
            blend_at(0.25) > 0.0 && blend_at(0.25) < 1.0,
            "the blend is not gradual in between",
        );
    }

    #[test]
    fn fresnel_is_still_capped_by_roughness() {
        let code = super::wgsl_lights_block(0, 1);
        assert!(
            code.contains("let f_max = max(1.0 - r, f0);"),
            "the roughness cap on Fresnel is gone; rough stone will develop a \
             bright rim at grazing angles again",
        );
    }
}

#[cfg(test)]
mod sky_sun_tests {
    use super::*;
    use glam::{Mat4, Quat};

    fn sky_sun() -> crate::renderer::sky::SkySun {
        crate::renderer::sky::SkySun {
            direction: Vec3::new(0.377, 0.743, 0.553).normalize().to_array(),
            light_rgb: [1.339, 1.354, 1.235],
            texels: 9,
            energy_fraction: 0.485,
        }
    }

    #[test]
    fn the_sky_sun_puts_back_exactly_the_light_the_ambient_lost() {
        let l = sky_sun_light(Some(&sky_sun()), &[], Quat::IDENTITY).unwrap();
        let c = l.color.to_linear();
        for (i, want) in sky_sun().light_rgb.iter().enumerate() {
            let got = c[i] * l.intensity;
            assert!((got / want - 1.0).abs() < 0.01, "channel {i}: {got} vs {want}");
        }

        assert!(l.direction.y < -0.7);
    }

    #[test]
    fn an_authored_sun_replaces_the_sky_sun() {
        let mut authored = sky_sun_light(Some(&sky_sun()), &[], Quat::IDENTITY).unwrap();
        authored.intensity = 3.0;
        assert!(sky_sun_light(Some(&sky_sun()), &[authored], Quat::IDENTITY).is_none());
        assert!(sky_sun_light(None, &[], Quat::IDENTITY).is_none());
    }

    #[test]
    fn the_sky_sun_turns_with_the_player_frame() {
        let yaw = Quat::from_rotation_y(1.1);
        let a = sky_sun_light(Some(&sky_sun()), &[], yaw.inverse()).unwrap();
        let world = -Vec3::from(sky_sun().direction);
        assert!((yaw * a.direction - world).length() < 1e-5);
    }

    #[test]
    fn a_world_point_keeps_its_shadow_texel_as_the_player_moves() {
        let world_dir = -Vec3::from(sky_sun().direction);
        let level: Vec<Vec3> = vec![Vec3::new(-4.0, 0.0, -16.0), Vec3::new(4.0, 4.0, 4.0), Vec3::new(1.0, 2.0, -3.0)];
        let frame = |offset: Vec3, yaw: f32| Mat4::from_rotation_translation(Quat::from_rotation_y(yaw), offset);
        let drawn_at = frame(Vec3::new(2.0, 0.0, 1.0), 0.4);
        let in_player = |f: Mat4, p: Vec3| f.inverse().transform_point3(p);
        let world_vp = static_sun_matrix(
            world_dir,
            Some(level.iter().map(|&p| in_player(drawn_at, p).to_array())),
            None::<std::iter::Empty<[f32; 3]>>,
            drawn_at,
        );
        for later in [frame(Vec3::ZERO, 0.0), frame(Vec3::new(-7.0, 0.0, 9.0), -2.0)] {
            for &p in &level {
                let then = (world_vp * drawn_at).project_point3(in_player(drawn_at, p));
                let now = (world_vp * later).project_point3(in_player(later, p));
                assert!((then - now).length() < 1e-4, "{p}: {then} vs {now}");

                assert!(now.x.abs() < 1.0 && now.y.abs() < 1.0 && now.z > 0.0 && now.z < 1.0, "{now}");
            }
        }
    }

    #[test]
    fn brushes_and_terrain_read_a_baked_sun_mask_and_everything_shades_the_sun_live() {
        let brush = crate::renderer::brush_pipeline::brush_shader_src();
        assert!(brush.contains("receiver_sun_mask = "), "brushes must read their baked mask");

        let terrain = crate::renderer::terrain_pipeline::terrain_shader_src();
        assert!(
            terrain.contains("receiver_sun_mask = select(-1.0, smoothstep(-ground_sun_w, ground_sun_w, ground_sun_d), ground_map.a < 0.5);"),
            "terrain must read its baked sun mask, and only where one was baked",
        );

        assert!(!brush.contains("receiver_has_baked_sun") && !terrain.contains("receiver_has_baked_sun"));
    }
}

#[cfg(test)]
mod baked_gloss_tests {
    use super::*;

    fn combined(shininess: f32, directionality: f32) -> f32 {
        let spread = directionality * (3.0 - directionality * directionality)
            / (1.0 - directionality * directionality).max(1e-3);
        let lobe = shininess * 0.25;
        (4.0 * lobe * spread / (lobe + spread).max(1e-4)).max(1.0)
    }

    #[test]
    fn a_polished_surface_reflects_baked_bounce_as_broadly_as_it_arrives() {

        let code = wgsl_lights_block(0, 1);
        assert!(code.contains("const MAX_BOUNCE_DIRECTIONALITY: f32 = 0.5;"));
        let marble = 2.0 / (0.048f32 * 0.048) - 2.0;
        let c = combined(marble, 0.5);
        assert!(c < 10.0, "baked bounce gives marble an exponent of {c}: a highlight sharper than the light it came from");

        assert!(code.contains("let combined = max(4.0 * lobe * spread / max(lobe + spread, 1e-4), 1.0);"));
        assert!(code.contains("pow(max(dot(n, h), 0.0), combined)"));
    }

    #[test]
    fn a_rough_surface_keeps_its_own_lobe() {
        let brick = 4.0;
        let c = combined(brick, 0.5);
        assert!(c <= brick + 1e-3, "the spread sharpened a rough lobe to {c}");
    }

    fn terminator_aa(nl: f32, w: f32) -> (f32, f32) {
        let facing = ((nl + w) / (2.0 * w).max(1e-6)).clamp(0.0, 1.0);
        let lambert = if nl.abs() < w { 0.5 * (nl + w) * facing } else { nl.max(0.0) };
        (lambert, facing)
    }

    fn terminator_width_of(normal_length: f32) -> f32 {
        let sigma2 = (1.0 - normal_length - 0.01).max(0.0) / normal_length.max(1e-3);
        (1.5 * sigma2).sqrt().min(0.5)
    }

    #[test]
    fn a_terminator_is_the_clamp_averaged_over_the_pixel() {
        for nl in [-0.5f32, -1e-3, 0.0, 1e-3, 0.3] {
            let (lambert, facing) = terminator_aa(nl, 0.0);
            assert_eq!(lambert, nl.max(0.0));
            assert_eq!(facing, if nl > 0.0 { 1.0 } else { 0.0 });
        }
        for w in [0.05f32, 0.2, 0.5] {
            for nl in [-0.6f32, -0.2, -0.05, 0.0, 0.05, 0.2, 0.6] {
                let n = 20_000;
                let (mut sum, mut lit) = (0.0f64, 0usize);
                for i in 0..n {
                    let x = nl + w * (2.0 * (i as f32 + 0.5) / n as f32 - 1.0);
                    sum += x.max(0.0) as f64;
                    lit += (x > 0.0) as usize;
                }
                let (lambert, facing) = terminator_aa(nl, w);
                assert!((lambert as f64 - sum / n as f64).abs() < 1e-4, "nl {nl} w {w}: {lambert} vs {}", sum / n as f64);
                assert!((facing as f64 - lit as f64 / n as f64).abs() < 1e-3, "nl {nl} w {w}: {facing}");
            }
        }
    }

    #[test]
    fn only_a_normal_its_texels_disagree_on_spreads_the_terminator() {
        assert_eq!(terminator_width_of(1.0), 0.0);
        assert_eq!(terminator_width_of(0.992), 0.0, "8-bit rounding is not spread");
        let (a, b) = (terminator_width_of(0.97), terminator_width_of(0.9));
        assert!(0.0 < a && a < b && b < 0.5, "{a} {b}");
        assert_eq!(terminator_width_of(0.2), 0.5);
    }

    #[test]
    fn the_shader_spreads_the_terminator_as_its_twin_does() {
        let code = wgsl_lights_block(0, 1);
        for line in [
            "let facing = clamp((nl + w) / max(2.0 * w, 1e-6), 0.0, 1.0);",
            "return vec2<f32>(select(max(nl, 0.0), 0.5 * (nl + w) * facing, abs(nl) < w), facing);",
            "const TERMINATOR_MAX_WIDTH: f32 = 0.5;",
            "let sigma2 = max(1.0 - normal_length - 0.01, 0.0) / max(normal_length, 1e-3);",
            "return min(sqrt(1.5 * sigma2), TERMINATOR_MAX_WIDTH);",
            "let aa = terminator_aa(dot(n, l_dir));",
            "out.specular = radiance * spec * atten * lamp_highlight_share(view_dir, h, ndotl, aa.y, shininess, spec_strength);",
        ] {
            assert!(code.contains(line), "the lights block no longer has `{line}`");
        }
    }
}

#[cfg(test)]
mod baked_light_split_tests {
    use super::*;

    fn point(x: f32, i: f32) -> Light {
        Light {
            mask_channel: None,
            shadow_near: None,
            source_radius: 0.0,
            in_level_bake: true,
            position: Vec3::new(x, 1.0, 0.0),
            direction: Vec3::NEG_Y,
            kind: LightKind::Point,
            color: Color3(255, 255, 255, 255),
            intensity: i,
            range: 6.0,
            cone_angle_deg: 0.0,
            inner_cone_angle_deg: 0.0,
        }
    }

    #[test]
    fn live_lights_come_first_and_baked_fill_what_is_left() {
        let live = [point(0.0, 1.0), point(1.0, 1.0)];

        let baked: Vec<Light> = (0..MAX_LIGHTS + 2).map(|i| point(i as f32, 0.5)).collect();
        let out = append_baked(&live, &baked, MAX_LIGHTS);
        assert_eq!(out.len(), MAX_LIGHTS);
        assert_eq!(out[0].position, live[0].position);
        assert_eq!(out[1].position, live[1].position);

        let all_live: Vec<Light> = (0..MAX_LIGHTS).map(|i| point(i as f32, 1.0)).collect();
        assert_eq!(append_baked(&all_live, &baked, MAX_LIGHTS).len(), MAX_LIGHTS);
        assert_eq!(append_baked(&all_live, &baked, MAX_LIGHTS)[MAX_LIGHTS - 1].intensity, 1.0);
    }

    #[test]
    fn the_upload_says_where_the_baked_tail_starts() {
        let lights: Vec<Light> = (0..5).map(|i| point(i as f32, 1.0)).collect();
        let gpu = pack_lights(&lights, 2, &[], false, true);
        assert_eq!(gpu.count[0], 5);
        assert_eq!(gpu.count[1], 2);

        let gpu = pack_lights(&lights, 5, &[], false, true);
        assert_eq!(gpu.count[1], 5);
    }

    #[test]
    fn lightmapped_surfaces_skip_the_baked_tail_and_nothing_else_does() {
        let code = wgsl_lights_block(0, 1);
        assert!(!code.contains("i < lights.count.x;"), "a light loop walks the baked tail");

        assert_eq!(code.matches("i < live_light_count();").count(), 7);
        assert!(code.contains("reaching = reaching | (1u << i);") && code.contains("var todo = reaching;"));
        let brush = crate::renderer::brush_pipeline::brush_shader_src();
        assert!(brush.contains("receiver_skips_baked = true;"), "brushes carry baked lamps in their atlas");
        let terrain = crate::renderer::terrain_pipeline::terrain_shader_src();
        assert!(!terrain.contains("receiver_skips_baked = true;"), "the ground has no lamp atlas and must keep them");
        let skinned = crate::renderer::mesh_pipeline::skinned_mesh_shader_src();
        assert!(!skinned.contains("receiver_skips_baked = true;"), "a character has no atlas and must keep them");
        let mesh = crate::renderer::mesh_pipeline::mesh_shader_src();
        assert!(mesh.contains("receiver_skips_baked = true;"), "a lightmapped mesh carries baked lamps in its atlas");
    }
}

#[cfg(test)]
mod baked_shaping_tests {
    use super::*;

    #[test]
    fn a_flat_face_keeps_its_texel_and_a_tilted_normal_redistributes() {
        let code = wgsl_lights_block(0, 1);
        assert!(
            code.contains("let shaped = 1.0 + directionality * (dot(n, bounce_dir) - dot(geom_n, bounce_dir));"),
            "the baked-light shaping is no longer relative to the geometric normal; a flat face \
             facing its lamp would be lit 1 + d times its own texel again",
        );

        let shaped = |n: Vec3, geom: Vec3, dir: Vec3, d: f32| 1.0 + d * (n.dot(dir) - geom.dot(dir));
        let up = Vec3::Y;
        assert!((shaped(up, up, up, 0.5) - 1.0).abs() < 1e-6, "a flat floor under its lamp");
        assert!((shaped(up, up, Vec3::X, 0.5) - 1.0).abs() < 1e-6, "a flat floor lit from the side");
        let tilted = Vec3::new(0.3, 0.95, 0.0).normalize();
        assert!(shaped(tilted, up, Vec3::X, 0.5) > 1.0, "a bump facing the light takes more");
        assert!(shaped(-tilted + 2.0 * up * up.dot(tilted), up, Vec3::X, 0.5) < 1.0, "a bump facing away takes less");
    }
}

#[cfg(test)]
mod probe_trace_gpu_tests {
    use super::*;
    use crate::renderer::uniforms::{ProbePortal, ProbeProxy, ProbeUpload, Uniforms};
    use glam::{Quat, Vec3};
    use wgpu::util::DeviceExt;

    const HALL_LO: Vec3 = Vec3::new(-2.7, 0.0, -15.7);
    const HALL_HI: Vec3 = Vec3::new(2.7, 3.1, 3.7);
    const HALLWAY_LO: Vec3 = Vec3::new(3.0, 0.0, -4.2);
    const HALLWAY_HI: Vec3 = Vec3::new(10.0, 2.6, -1.8);

    struct Hit {
        pos: Vec3,
        found: bool,
        room: f32,
        other: f32,
        escaped: bool,

        rim: f32,
        rim_went_through: bool,

        edge: i32,
        edge_cover: f32,
        edge_hit: bool,
    }

    fn trace(rays: &[(Vec3, f32, Vec3, f32)]) -> Option<Vec<Hit>> {
        trace_seen_from(rays, 0.0, 1.0)
    }

    fn trace_seen_from(rays: &[(Vec3, f32, Vec3, f32)], footprint: f32, eye: f32) -> Option<Vec<Hit>> {
        let (device, queue) = crate::renderer::terrain_pipeline::tests::headless_gpu()?;
        let mut probes = ProbeUpload::default();
        probes.count = 4;
        probes.set(0, 0, Vec3::new(0.0, 1.55, -8.425), HALL_LO, HALL_HI);
        probes.set(1, 1, Vec3::new(0.0, 1.55, -3.575), HALL_LO, HALL_HI);
        probes.set(2, 2, Vec3::new(4.75, 1.3, -3.0), HALLWAY_LO, HALLWAY_HI);
        probes.set_volume(0, 0);
        probes.set_volume(1, 0);
        probes.set_volume(2, 1);

        probes.set(3, 3, Vec3::new(7.5, 7.1, -6.0), Vec3::new(-14.7, -6.0, -27.7), Vec3::new(29.7, 9.1, 15.7));
        probes.set_volume(3, 2);
        let door = ProbePortal {
            min: Vec3::new(2.5, 0.0, -3.8),
            max: Vec3::new(3.1, 2.2, -2.2),
            axis: 0,
            low: 0,
            high: 1,
            wall: Some((2.7, 3.0)),
        };
        let front = ProbePortal {
            min: Vec3::new(-0.8, 0.0, 3.69),
            max: Vec3::new(0.8, 2.2, 4.4),
            axis: 2,
            low: 0,
            high: 2,
            wall: Some((3.7, 4.0)),
        };
        probes.set_portals(&[door, front], Vec3::ZERO, &[0, 1, 2]);
        let pillar = ProbeProxy {
            centre: Vec3::new(0.0, 1.55, -7.0),
            half_size: Vec3::new(0.45, 1.55, 0.45),
            rotation: Quat::IDENTITY,
            volume: 0,
            solid: true,
            field: None,
            cards: None,
        };
        let lamp = ProbeProxy {
            centre: Vec3::new(0.0, 2.5, -12.0),
            half_size: Vec3::new(0.3, 0.1, 0.1),
            rotation: Quat::from_rotation_y(std::f32::consts::FRAC_PI_4),
            volume: 0,
            solid: true,
            field: None,
            cards: None,
        };
        probes.set_proxies(&[pillar, lamp], Vec3::ZERO, &[0, 1]);

        let (dense, room_tables) = probes.dense_rooms();
        let mut u: Uniforms = bytemuck::Zeroable::zeroed();
        u.probe_params = [probes.count as f32, 0.0, 0.0, 0.0];
        u.probe_boxes = dense.boxes;
        u.portal_params = [probes.portal_count as f32, 0.0, 0.0, 0.0];
        u.probe_portals = dense.portals;
        u.proxy_params = [probes.proxy_count as f32, 0.0, 0.0, 0.0];
        u.probe_proxies = dense.proxies;
        u.probe_rooms = room_tables;

        let code = format!(
            "{}\n{}",
            wgsl_lights_block(0, 1),
            r#"
@group(1) @binding(0) var<storage, read> rays: array<vec4<f32>>;
@group(1) @binding(1) var<storage, read_write> hits: array<vec4<f32>>;
@compute @workgroup_size(1)
fn trace_main(@builtin(global_invocation_id) id: vec3<u32>) {
    pixel_footprint = PIXEL_FOOTPRINT;
    probe_eye_distance = EYE_DISTANCE;
    let o = rays[id.x * 2u];
    let d = rays[id.x * 2u + 1u];
    let h = probe_trace(o.xyz, normalize(d.xyz), o.w, d.w);
    hits[id.x * 4u] = vec4<f32>(h.pos, select(0.0, 1.0, h.found));
    hits[id.x * 4u + 1u] = vec4<f32>(h.room, h.other, select(0.0, 1.0, h.escaped), f32(h.portal));
    hits[id.x * 4u + 2u] = vec4<f32>(h.rim, select(0.0, 1.0, probe_hit_rim_went_through(h)), h.t, h.rim_t);
    hits[id.x * 4u + 3u] = vec4<f32>(f32(probe_hit_edge(h)), h.edge_cover, select(0.0, 1.0, probe_hit_edge_hit(h)), h.edge_t);
}
"#
        );
        let code = code
            .replace("PIXEL_FOOTPRINT", &format!("{footprint:?}"))
            .replace("EYE_DISTANCE", &format!("{eye:?}"));
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("probe_trace_test"),
            source: wgpu::ShaderSource::Wgsl(code.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("probe_trace_test"),
            layout: None,
            module: &module,
            entry_point: Some("trace_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let camera = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("camera"),
            contents: bytemuck::bytes_of(&u),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let packed: Vec<[f32; 4]> = rays
            .iter()
            .flat_map(|(o, room, d, rough)| [[o.x, o.y, o.z, *room], [d.x, d.y, d.z, *rough]])
            .collect();
        let ray_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rays"),
            contents: bytemuck::cast_slice(&packed),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let size = (rays.len() * 4 * 16) as u64;
        let hit_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hits"),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let read = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("read"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let depth_tex = device.create_texture(&crate::renderer::uniforms::probe_depth_descriptor(1, 4));

        for layer in 0..4u32 {
            let texel: [u16; 4] = if layer < 3 { [0, 0, 0, 0x3C00] } else { [0; 4] };
            let data: Vec<u16> = (0..6).flat_map(|_| texel).collect();
            crate::renderer::uniforms::write_probe_depth_layer(&queue, &depth_tex, layer, 1, &data);
        }
        let depth_view = depth_tex.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::CubeArray),
            ..Default::default()
        });
        let (_, depth_samp) = crate::renderer::uniforms::default_probe_depth(&device);

        let (_, probe_samp) = crate::renderer::uniforms::default_probe_cube(&device);
        let fields = crate::renderer::proxy_field::none(&device);
        let g0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: camera.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::Sampler(&probe_samp) },
                wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::TextureView(&depth_view) },
                wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::Sampler(&depth_samp) },
                wgpu::BindGroupEntry { binding: 11, resource: wgpu::BindingResource::TextureView(&fields) },
            ],
        });
        let g1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(1),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: ray_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: hit_buf.as_entire_binding() },
            ],
        });
        let mut enc = device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &g0, &[]);
            pass.set_bind_group(1, &g1, &[]);
            pass.dispatch_workgroups(rays.len() as u32, 1, 1);
        }
        enc.copy_buffer_to_buffer(&hit_buf, 0, &read, 0, size);
        queue.submit([enc.finish()]);
        read.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        let data: Vec<[f32; 4]> = bytemuck::cast_slice(&read.slice(..).get_mapped_range().ok()?).to_vec();
        Some(
            data.chunks(4)
                .map(|c| Hit {
                    pos: Vec3::new(c[0][0], c[0][1], c[0][2]),
                    found: c[0][3] > 0.5,
                    room: c[1][0],
                    other: c[1][1],
                    escaped: c[1][2] > 0.5,
                    rim: c[2][0],
                    rim_went_through: c[2][1] > 0.5,
                    edge: c[3][0].round() as i32,
                    edge_cover: c[3][1],
                    edge_hit: c[3][2] > 0.5,
                })
                .collect(),
        )
    }

    fn toward(from: Vec3, to: Vec3) -> Vec3 {
        (to - from).normalize()
    }

    fn near(a: Vec3, b: Vec3) -> bool {
        (a - b).length() < 2e-3
    }

    #[test]
    fn a_reflection_lands_exactly_on_what_it_hits() {
        let floor_a = Vec3::new(-1.2, 0.0, -5.8);
        let side_face = Vec3::new(-0.45, 1.0, -7.0);
        let floor_b = Vec3::new(1.5, 0.0, -3.0);
        let hallway_end = Vec3::new(10.0, 1.2, -3.0);
        let floor_c = Vec3::new(0.5, 0.0, -4.0);
        let jamb = Vec3::new(2.9, 1.0, -2.2);
        let floor_d = Vec3::new(0.2, 0.0, -12.2);
        let floor_e = Vec3::new(-1.0, 0.0, -1.0);
        let floor_f = Vec3::new(0.2, 0.0, 2.0);
        let floor_g = Vec3::new(0.0, 0.0, 2.0);
        let front_jamb = Vec3::new(0.8, 1.0, 3.9);
        let Some(h) = trace(&[

            (floor_a, 0.0, toward(floor_a, side_face), 0.0),

            (floor_b, 0.0, toward(floor_b, hallway_end), 0.0),

            (floor_c, 0.0, toward(floor_c, jamb), 0.0),

            (floor_d, 0.0, Vec3::Y, 0.0),

            (floor_e, 0.0, Vec3::new(0.0, 1.0, -0.2).normalize(), 0.0),

            (floor_e, 0.0, Vec3::Y, 0.8),

            (jamb, -1.0, Vec3::new(-0.05, 0.0, -1.0).normalize(), 0.0),

            (jamb, -1.0, Vec3::new(1.0, 0.0, -0.3).normalize(), 0.0),

            (floor_f, 0.0, toward(floor_f, Vec3::new(0.0, 1.5, 6.0)), 0.0),

            (floor_g, 0.0, toward(floor_g, front_jamb), 0.0),
        ]) else {
            eprintln!("skipping: no GPU");
            return;
        };
        assert!(h[0].found && near(h[0].pos, side_face) && h[0].room == 0.0, "pillar side: {:?}", h[0].pos);
        assert!(h[1].found && near(h[1].pos, hallway_end) && h[1].room == 1.0, "hallway end: {:?} room {}", h[1].pos, h[1].room);
        assert!(h[2].found && near(h[2].pos, jamb) && h[2].room == 0.0 && h[2].other == 1.0, "jamb: {:?} {} {}", h[2].pos, h[2].room, h[2].other);
        assert!(h[3].found && (h[3].pos.y - 2.4).abs() < 2e-3, "rotated lamp box: {:?}", h[3].pos);
        let ceiling = floor_e + Vec3::new(0.0, 1.0, -0.2).normalize() * (3.1 / Vec3::new(0.0, 1.0, -0.2).normalize().y);
        assert!(h[4].found && near(h[4].pos, ceiling), "ceiling: {:?}", h[4].pos);
        assert!(!h[5].found, "a rough surface was traced");
        let d6 = Vec3::new(-0.05, 0.0, -1.0).normalize();
        let other_jamb = jamb + d6 * ((-3.8 - jamb.z) / d6.z);
        assert!(h[6].found && near(h[6].pos, other_jamb) && h[6].room == 0.0 && h[6].other == 1.0, "jamb to jamb: {:?}", h[6].pos);
        let d7 = Vec3::new(1.0, 0.0, -0.3).normalize();
        let hallway_wall = jamb + d7 * ((-4.2 - jamb.z) / d7.z);
        assert!(h[7].found && near(h[7].pos, hallway_wall) && h[7].room == 1.0, "jamb to hallway wall: {:?} room {}", h[7].pos, h[7].room);
        let d8 = toward(floor_f, Vec3::new(0.0, 1.5, 6.0));
        let out_face = floor_f + d8 * ((4.0 - floor_f.z) / d8.z);
        assert!(h[8].escaped && near(h[8].pos, out_face) && h[8].room == 0.0 && h[8].other == 2.0, "front door escape: {:?} esc {}", h[8].pos, h[8].escaped);
        assert!(h[9].found && !h[9].escaped && near(h[9].pos, front_jamb), "front door jamb: {:?} esc {}", h[9].pos, h[9].escaped);
    }

    #[test]
    fn a_rough_reflection_blends_across_a_doorways_rim() {
        let floor = Vec3::new(0.5, 0.0, -1.5);
        let beside = Vec3::new(2.7, 1.0, -2.1);
        let inside = Vec3::new(2.7, 1.0, -2.3);
        let Some(h) = trace(&[
            (floor, 0.0, toward(floor, beside), 0.43),
            (floor, 0.0, toward(floor, inside), 0.43),
            (floor, 0.0, toward(floor, beside), 0.048),
            (floor, 0.0, toward(floor, Vec3::new(2.7, 1.0, 1.0)), 0.43),
        ]) else {
            eprintln!("skipping: no GPU");
            return;
        };
        assert!(h[0].found && near(h[0].pos, beside), "the ray itself should still meet the wall: {:?}", h[0].pos);
        assert!(!h[0].rim_went_through && (0.2..0.6).contains(&h[0].rim), "beside the door: rim {} through {}", h[0].rim, h[0].rim_went_through);
        assert!(h[1].rim_went_through && (0.5..0.95).contains(&h[1].rim), "inside the door: rim {} through {}", h[1].rim, h[1].rim_went_through);
        assert!(h[1].room == 1.0, "inside the door the ray goes on into the hallway: room {}", h[1].room);
        assert!(h[2].rim < 0.0, "marble softened a doorway's rim: {}", h[2].rim);
        assert!(h[3].rim < 0.0, "a plain wall has no rim: {}", h[3].rim);
    }

    #[test]
    fn a_mirrors_doorway_rim_blends_across_a_pixel() {
        let floor = Vec3::new(-1.1, 0.0, 1.0);
        let rays: Vec<(Vec3, f32, Vec3, f32)> = (0..21)
            .map(|k| {
                let x = -0.90 + 0.01 * k as f32;
                (floor, 0.0, toward(floor, Vec3::new(x, 0.5, 3.7)), 0.048)
            })
            .collect();
        let Some(h) = trace_seen_from(&rays, 0.036, 8.4) else {
            eprintln!("skipping: no GPU");
            return;
        };
        let through: Vec<f32> = h
            .iter()
            .map(|h| match (h.rim >= 0.0, h.rim_went_through, h.escaped) {

                (true, _, _) => h.rim,

                (false, _, true) => 1.0,
                (false, _, false) => 0.0,
            })
            .collect();
        for (k, (t, h)) in through.iter().zip(&h).enumerate() {
            eprintln!(
                "x {:+.2}: through {:.2} (rim {:.2} went {} escaped {} room {} at {:?})",
                -0.90 + 0.01 * k as f32, t, h.rim, h.rim_went_through, h.escaped, h.room, h.pos
            );
        }
        for w in through.windows(2) {
            assert!(w[1] + 1e-3 >= w[0] && w[1] - w[0] < 0.5, "the share through jumps between neighbours: {through:?}");
        }
        assert!(through[0] < 0.05 && through[20] > 0.95, "{through:?}");
    }

    #[test]
    fn a_reflection_blends_across_the_pillars_outline() {
        let from = Vec3::new(-2.0, 1.0, -4.0);

        let outside = Vec3::new(-0.52, 1.0, -7.45);
        let inside = Vec3::new(-0.38, 1.0, -7.45);
        let Some(h) = trace(&[
            (from, 0.0, toward(from, outside), 0.2),
            (from, 0.0, toward(from, inside), 0.2),
            (from, 0.0, toward(from, Vec3::new(-2.0, 1.0, -15.7)), 0.2),
            (from, 0.0, toward(from, outside), 0.0),
        ]) else {
            eprintln!("skipping: no GPU");
            return;
        };
        assert!(h[0].edge >= 0 && !h[0].edge_hit && (0.01..0.5).contains(&h[0].edge_cover), "just outside: edge {} hit {} cover {}", h[0].edge, h[0].edge_hit, h[0].edge_cover);
        assert!(h[1].edge >= 0 && h[1].edge_hit && (0.5..0.99).contains(&h[1].edge_cover), "just inside: edge {} hit {} cover {}", h[1].edge, h[1].edge_hit, h[1].edge_cover);
        assert!(h[2].edge < 0, "a ray well clear of the pillar reported its outline: {}", h[2].edge);
        assert!(h[3].edge < 0, "a mirror with no footprint blended the outline: {} cover {}", h[3].edge, h[3].edge_cover);
    }
}

#[cfg(test)]
mod ground_trace_gpu_tests {
    use super::*;
    use crate::renderer::ground_map::{self, fixtures, GroundMap};
    use crate::renderer::uniforms::Uniforms;
    use glam::Vec3;
    use wgpu::util::DeviceExt;

    fn trace(map: &GroundMap, rays: &[(Vec3, Vec3)]) -> Option<Vec<f32>> {
        let (device, queue) = crate::renderer::terrain_pipeline::tests::headless_gpu()?;
        let view = ground_map::upload(&device, &queue, map);
        let (_, samp) = crate::renderer::uniforms::default_probe_cube(&device);
        let extent = map.max - map.min;
        let mut u: Uniforms = bytemuck::Zeroable::zeroed();
        u.ground_params = [map.min.x, map.min.y, 1.0 / extent.x, 1.0 / extent.y];
        u.sky_params = [1.0, 0.0, map.top, 1.0];
        let code = format!(
            "{}\n{}",
            wgsl_lights_block(0, 1),
            r#"
@group(1) @binding(0) var<storage, read> rays: array<vec4<f32>>;
@group(1) @binding(1) var<storage, read_write> hits: array<f32>;
@compute @workgroup_size(1)
fn trace_main(@builtin(global_invocation_id) id: vec3<u32>) {
    hits[id.x] = ground_trace(rays[id.x * 2u].xyz, rays[id.x * 2u + 1u].xyz);
}
"#
        );
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ground_trace_test"),
            source: wgpu::ShaderSource::Wgsl(code.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ground_trace_test"),
            layout: None,
            module: &module,
            entry_point: Some("trace_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let camera = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("camera"),
            contents: bytemuck::bytes_of(&u),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let packed: Vec<[f32; 4]> = rays.iter().flat_map(|(e, d)| [[e.x, e.y, e.z, 0.0], [d.x, d.y, d.z, 0.0]]).collect();
        let ray_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rays"),
            contents: bytemuck::cast_slice(&packed),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let size = (rays.len() * 4) as u64;
        let hit_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hits"),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let read = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("read"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let g0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: camera.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::Sampler(&samp) },
                wgpu::BindGroupEntry { binding: 10, resource: wgpu::BindingResource::TextureView(&view) },
            ],
        });
        let g1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(1),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: ray_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: hit_buf.as_entire_binding() },
            ],
        });
        let mut enc = device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &g0, &[]);
            pass.set_bind_group(1, &g1, &[]);
            pass.dispatch_workgroups(rays.len() as u32, 1, 1);
        }
        enc.copy_buffer_to_buffer(&hit_buf, 0, &read, 0, size);
        queue.submit([enc.finish()]);
        read.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        let data: Vec<f32> = bytemuck::cast_slice(&read.slice(..).get_mapped_range().ok()?).to_vec();
        Some(data)
    }

    #[test]
    fn the_shader_meets_the_ground_where_its_twin_does() {
        let map = fixtures::height_map(1024, fixtures::hills);
        let chain = ground_map::levels(&map);
        let rays = fixtures::rays(400, 11);
        let Some(gpu) = trace(&map, &rays) else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        let mut grazing = 0;
        let mut hits = 0;
        for (i, (&(e, d), &g)) in rays.iter().zip(&gpu).enumerate() {
            let twin = ground_map::trace(&chain, map.min, map.max, map.top, e, d).t;
            let g = (g >= 0.0).then_some(g);
            hits += g.is_some() as usize;
            match (g, twin) {
                (Some(a), Some(b)) if (a - b).abs() <= 0.01 + 0.001 * b => {}
                (None, None) => {}
                _ => {
                    let fine = fixtures::march(&chain[0], map.min, map.max, map.top, e, d);
                    assert!(fine.graze < 0.03, "ray {i} from {e} along {d}: GPU {g:?}, twin {twin:?}, clearance {}", fine.graze);
                    grazing += 1;
                }
            }
        }
        assert!(hits > rays.len() / 3 && hits < rays.len(), "{hits} of {} rays met the ground: not a test of both", rays.len());
        assert!(grazing <= rays.len() / 100, "{grazing} grazing disagreements");
    }
}

#[cfg(test)]
mod proxy_field_gpu_tests {
    use super::*;
    use crate::renderer::proxy_field::{self, ProxyField};
    use crate::renderer::uniforms::Uniforms;
    use glam::Vec3;
    use wgpu::util::DeviceExt;

    const HALF: f32 = 0.3;
    const RADIUS: f32 = 0.2;
    const SAMPLES: u32 = 24;

    fn sphere() -> ProxyField {
        let cell = 2.0 * HALF / SAMPLES as f32;
        let reach = 4.0 * cell;
        let mut distances = Vec::with_capacity((SAMPLES * SAMPLES * SAMPLES) as usize);
        for k in 0..SAMPLES {
            for j in 0..SAMPLES {
                for i in 0..SAMPLES {
                    let p = (Vec3::new(i as f32, j as f32, k as f32) + 0.5) * cell - HALF;
                    let d = (p.length() - RADIUS).abs();
                    distances.push(((d / reach).min(1.0) * 255.0).round() as u8);
                }
            }
        }
        ProxyField { dims: [SAMPLES; 3], max_distance: reach, distances, albedo: [0.2; 3] }
    }

    fn walk(rays: &[(Vec3, Vec3)]) -> Option<Vec<f32>> {
        walk_full(sphere(), rays, 0.0).map(|v| v.into_iter().map(|h| h[0]).collect())
    }

    fn walk_full(field: ProxyField, rays: &[(Vec3, Vec3)], pixel_footprint: f32) -> Option<Vec<[f32; 4]>> {
        let (device, queue) = crate::renderer::terrain_pipeline::tests::headless_gpu()?;
        let (atlas, slots) = proxy_field::atlas(&device, &queue, &[field])?;
        let (_, samp) = crate::renderer::uniforms::default_probe_cube(&device);
        let mut u: Uniforms = bytemuck::Zeroable::zeroed();
        u.proxy_fields[0] = slots[0];
        let code = format!(
            "{}\n{}",
            wgsl_lights_block(0, 1),
            r#"
@group(1) @binding(0) var<storage, read> rays: array<vec4<f32>>;
@group(1) @binding(1) var<storage, read_write> hits: array<vec4<f32>>;
@compute @workgroup_size(1)
fn walk_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let o = rays[id.x * 2u].xyz;
    let d = normalize(rays[id.x * 2u + 1u].xyz);
    let half = vec3<f32>(rays[id.x * 2u].w);
    pixel_footprint = rays[id.x * 2u + 1u].w;
    probe_eye_distance = 1.0;

    let inv = 1.0 / d;
    let ta = (-half - o) * inv;
    let tb = (half - o) * inv;
    let t_in = max(max(min(ta.x, tb.x), min(ta.y, tb.y)), min(ta.z, tb.z));
    let t_out = min(min(max(ta.x, tb.x), max(ta.y, tb.y)), max(ta.z, tb.z));

    hits[id.x] = vec4<f32>(probe_proxy_field(o, d, half, t_in, t_out, 0, 0.0), 0.0);
}
"#
        );
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(code.into()) });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &module,
            entry_point: Some("walk_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let camera = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::bytes_of(&u),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let packed: Vec<[f32; 4]> =
            rays.iter().flat_map(|(o, d)| [[o.x, o.y, o.z, HALF], [d.x, d.y, d.z, pixel_footprint]]).collect();
        let ray_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&packed),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let size = (rays.len() * 16) as u64;
        let hit_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let read = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let g0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: camera.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::Sampler(&samp) },
                wgpu::BindGroupEntry { binding: 11, resource: wgpu::BindingResource::TextureView(&atlas) },
            ],
        });
        let g1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(1),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: ray_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: hit_buf.as_entire_binding() },
            ],
        });
        let mut enc = device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &g0, &[]);
            pass.set_bind_group(1, &g1, &[]);
            pass.dispatch_workgroups(rays.len() as u32, 1, 1);
        }
        enc.copy_buffer_to_buffer(&hit_buf, 0, &read, 0, size);
        queue.submit([enc.finish()]);
        read.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        let data: Vec<[f32; 4]> = bytemuck::cast_slice(&read.slice(..).get_mapped_range().ok()?).to_vec();
        Some(data)
    }

    fn bowl() -> ProxyField {
        const FINE: u32 = 48;
        let cell = 2.0 * HALF / FINE as f32;
        let reach = 4.0 * cell;
        let mut distances = Vec::with_capacity((FINE * FINE * FINE) as usize);
        for k in 0..FINE {
            for j in 0..FINE {
                for i in 0..FINE {
                    let p = (Vec3::new(i as f32, j as f32, k as f32) + 0.5) * cell - HALF;
                    let d = if p.y <= 0.0 {
                        (p.length() - RADIUS).abs()
                    } else {
                        (p.x.hypot(p.z) - RADIUS).hypot(p.y)
                    };
                    distances.push(((d / reach).min(1.0) * 255.0).round() as u8);
                }
            }
        }
        ProxyField { dims: [FINE; 3], max_distance: reach, distances, albedo: [0.2; 3] }
    }

    #[test]
    fn the_walk_stops_on_the_model_and_passes_the_air_around_it() {
        let x = Vec3::X;
        let edge_y = 0.18f32;
        let Some(t) = walk(&[
            (Vec3::new(-1.0, 0.0, 0.0), x),
            (Vec3::new(-1.0, edge_y, 0.0), x),
            (Vec3::new(-1.0, 0.25, 0.0), x),
            (Vec3::new(0.0, -1.0, 0.0), Vec3::Y),
        ]) else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        let stop = 4.0 * 2.0 * HALF / SAMPLES as f32 / 8.0;
        let chord = 1.0 - (RADIUS * RADIUS - edge_y * edge_y).sqrt();
        assert!((t[0] - (1.0 - RADIUS)).abs() < 2.0 * stop, "straight at it: t {} vs {}", t[0], 1.0 - RADIUS);
        assert!((t[1] - chord).abs() < 3.0 * stop, "through its edge: t {} vs {}", t[1], chord);
        assert!(t[2] > 1e30, "five centimetres above it, a hit at {}", t[2]);
        assert!((t[3] - (1.0 - RADIUS)).abs() < 2.0 * stop, "from below: t {}", t[3]);
    }

    #[test]
    fn a_hit_past_the_models_own_rim_is_sampled_again() {
        let d = Vec3::new(1.0, -1.0, 0.0).normalize();
        let over_rim = Vec3::new(-RADIUS, 0.0, 0.0) + Vec3::new(1.0, 1.0, 0.0).normalize() * 0.012;
        let Some(h) = walk_full(
            bowl(),
            &[
                (over_rim - d, d),
                (Vec3::new(0.0, 0.9, 0.0), Vec3::NEG_Y),
                (Vec3::new(-0.9, -0.19, 0.0), Vec3::X),
                (Vec3::new(-0.9, 0.012, 0.0), Vec3::X),
            ],
            0.01,
        ) else {
            eprintln!("no GPU adapter; skipping");
            return;
        };

        assert!(h[0][0] > 1.1 && h[0][0] < 1.35, "over the rim, into the bowl: hit at {:?}", h[0]);
        assert_eq!(h[0][2], -1.0, "straddles the rim: {:?}", h[0]);
        assert!((h[1][0] - (0.9 + RADIUS)).abs() < 0.03 && h[1][2] >= 0.0, "straight down: {:?}", h[1]);
        assert!(h[2][0] < 1e30 && h[2][2] >= 0.0, "a slanting hit on its outside: {:?}", h[2]);
        assert!(h[3][0] > 1e30 && h[3][2] >= 0.0 && h[3][2] < 1.0, "skimming the rim, a near miss: {:?}", h[3]);
    }
}

#[cfg(test)]
mod proxy_card_gpu_tests {
    use super::*;
    use crate::renderer::proxy_cards::{self, ProxyCards};
    use crate::renderer::proxy_field::{self, ProxyField};
    use crate::renderer::uniforms::{ProbeProxy, ProbeUpload, Uniforms};
    use glam::{Quat, Vec3};
    use wgpu::util::DeviceExt;

    const RES: u32 = 32;
    const CENTRE: Vec3 = Vec3::new(1.0, 2.0, 3.0);
    const HALF: f32 = 0.3;

    thread_local! {

        static FOOTPRINT: std::cell::Cell<f32> = const { std::cell::Cell::new(0.0) };
        static HIT_T: std::cell::Cell<f32> = const { std::cell::Cell::new(0.0) };

        static FILTERED: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };

        static RELIT: std::cell::RefCell<Option<Vec<Light>>> = const { std::cell::RefCell::new(None) };
    }
    const RADIUS: f32 = 0.2;
    const SAMPLES: u32 = 24;

    fn field() -> ProxyField {
        let cell = 2.0 * HALF / SAMPLES as f32;
        let reach = 4.0 * cell;
        let mut distances = Vec::new();
        for k in 0..SAMPLES {
            for j in 0..SAMPLES {
                for i in 0..SAMPLES {
                    let p = (Vec3::new(i as f32, j as f32, k as f32) + 0.5) * cell - HALF;
                    distances.push((((p.length() - RADIUS).abs() / reach).min(1.0) * 255.0).round() as u8);
                }
            }
        }
        ProxyField { dims: [SAMPLES; 3], max_distance: reach, distances, albedo: [0.2; 3] }
    }

    fn cards() -> ProxyCards {
        let mut texels = Vec::new();
        let mut normals = Vec::new();
        for face in 0..6usize {
            let (a, sign) = (face / 2, if face % 2 == 0 { 1.0 } else { -1.0 });
            for y in 0..RES {
                for x in 0..RES {
                    let (u, v) = ((x as f32 + 0.5) / RES as f32, (y as f32 + 0.5) / RES as f32);
                    let r2 = ((2.0 * u - 1.0) * HALF).powi(2) + ((2.0 * v - 1.0) * HALF).powi(2);
                    if r2 < RADIUS * RADIUS {
                        let mut p = Vec3::ZERO;
                        p[a] = sign * (RADIUS * RADIUS - r2).sqrt();
                        p[(a + 1) % 3] = (2.0 * u - 1.0) * HALF;
                        p[(a + 2) % 3] = (2.0 * v - 1.0) * HALF;
                        texels.push([0.0, 0.0, 0.0, (HALF - (RADIUS * RADIUS - r2).sqrt()) / (2.0 * HALF)]);
                        normals.push((p / RADIUS).to_array());
                    } else {
                        texels.push([0.0, 0.0, 0.0, 2.0]);
                        normals.push([0.0; 3]);
                    }
                }
            }
        }
        for (i, t) in texels.iter_mut().enumerate() {
            t[0] = (i as u32 / (RES * RES) + 1) as f32;
        }
        ProxyCards { resolution: RES, texels, normals, albedo: Vec::new() }
    }

    fn colours(rotation: Quat, with_cards: bool, rays: &[(Vec3, Vec3)]) -> Option<Vec<[f32; 4]>> {
        colours_of(field(), cards(), rotation, with_cards, rays)
    }

    fn colours_of(field: ProxyField, cards: ProxyCards, rotation: Quat, with_cards: bool, rays: &[(Vec3, Vec3)]) -> Option<Vec<[f32; 4]>> {
        let (device, queue) = crate::renderer::terrain_pipeline::tests::headless_gpu()?;
        let (card_atlas, rows) = proxy_cards::atlas(&device, &queue, &[cards])?;
        assert_eq!(rows, vec![Some(0)]);
        let (_field_atlas, slots) = proxy_field::atlas(&device, &queue, &[field])?;
        let (_, samp) = crate::renderer::uniforms::default_probe_cube(&device);
        let mut probes = ProbeUpload::default();
        let half = Vec3::splat(HALF);
        let proxy = ProbeProxy { centre: CENTRE, half_size: half, rotation, volume: 0, solid: false, field: Some(0), cards: with_cards.then_some(0) };
        probes.set_proxies(&[proxy], Vec3::ZERO, &[0]);
        let mut u: Uniforms = bytemuck::Zeroable::zeroed();
        u.probe_proxies = probes.proxies;
        u.proxy_cards = probes.proxy_cards;
        u.proxy_fields[0] = slots[0];
        let relit = RELIT.with(|r| r.borrow().clone());
        let options =
            LightsBlockOptions { card_tests_filtered: FILTERED.with(|f| f.get()), card_relit: relit.is_some(), ..Default::default() };

        let lights_uniform = LightsUniform::new(&device);
        let shadow_map = crate::renderer::shadow::ShadowMap::with_dimension(&device, 8);
        if let Some(lamps) = &relit {
            lights_uniform.upload_frame(&queue, lamps, &[], false);
        }
        let code = format!(
            "{}\n{}",
            wgsl_lights_block_with(0, 1, options),
            r#"
@group(1) @binding(0) var<storage, read> rays: array<vec4<f32>>;
@group(1) @binding(1) var<storage, read_write> out: array<vec4<f32>>;
@compute @workgroup_size(1)
fn cards_main(@builtin(global_invocation_id) id: vec3<u32>) {
    pixel_footprint = rays[id.x * 2u].w;
    probe_eye_distance = 1.0;
    out[id.x] = probe_card_colour(0, rays[id.x * 2u].xyz, normalize(rays[id.x * 2u + 1u].xyz), rays[id.x * 2u + 1u].w, 0.0);
}
"#
        );
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(code.into()) });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &module,
            entry_point: Some("cards_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let camera = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::bytes_of(&u),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let packed: Vec<[f32; 4]> = rays.iter().flat_map(|(h, d)| [[h.x, h.y, h.z, FOOTPRINT.with(|f| f.get())], [d.x, d.y, d.z, HIT_T.with(|f| f.get())]]).collect();
        let ray_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&packed),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let size = (rays.len() * 16) as u64;
        let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let read = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut entries = vec![
            wgpu::BindGroupEntry { binding: 0, resource: camera.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::Sampler(&samp) },
            wgpu::BindGroupEntry { binding: 12, resource: wgpu::BindingResource::TextureView(&card_atlas) },
        ];
        if relit.is_some() {
            entries.push(wgpu::BindGroupEntry { binding: 1, resource: lights_uniform.buffer().as_entire_binding() });
            entries.push(wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(shadow_map.sampler()) });
            entries.push(wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(shadow_map.spot_depth_view()) });
        }
        let g0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
        let g1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(1),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: ray_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: out_buf.as_entire_binding() },
            ],
        });
        let mut enc = device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &g0, &[]);
            pass.set_bind_group(1, &g1, &[]);
            pass.dispatch_workgroups(rays.len() as u32, 1, 1);
        }
        enc.copy_buffer_to_buffer(&out_buf, 0, &read, 0, size);
        queue.submit([enc.finish()]);
        read.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        let data: Vec<[f32; 4]> = bytemuck::cast_slice(&read.slice(..).get_mapped_range().ok()?).to_vec();
        Some(data)
    }

    fn karis(a: f32, b: f32) -> f32 {
        let (wa, wb) = (1.0 / (1.0 + 0.2126 * a), 1.0 / (1.0 + 0.2126 * b));
        (a * wa + b * wb) / (wa + wb)
    }

    fn on_sphere(n: Vec3) -> Vec3 {
        CENTRE + n.normalize() * (RADIUS + 0.003)
    }

    fn step_cards() -> ProxyCards {
        let mut texels = Vec::new();
        let mut normals = Vec::new();
        for face in 0..6u32 {
            for _y in 0..RES {
                for x in 0..RES {
                    if face == 4 {
                        let depth = if x < RES / 2 { 0.3 } else { 0.6 };
                        texels.push([5.0, 0.0, 0.0, depth]);
                        normals.push([0.0, 0.0, 1.0]);
                    } else {
                        texels.push([0.0, 0.0, 0.0, 2.0]);
                        normals.push([0.0; 3]);
                    }
                }
            }
        }
        ProxyCards { resolution: RES, texels, normals, albedo: Vec::new() }
    }

    #[test]
    fn filtered_trust_slides_as_the_hit_does() {
        let texel = 2.0 * HALF / RES as f32;
        let z = HALF - 0.6 * 2.0 * HALF;
        let rays: Vec<(Vec3, Vec3)> = (0..30)
            .map(|k| (CENTRE + Vec3::new(1.5 * texel - k as f32 * 0.1 * texel, 0.0, z + 0.002), Vec3::new(-0.2, 0.1, -1.0).normalize()))
            .collect();
        let sure = |filtered: bool| {
            FILTERED.with(|f| f.set(filtered));
            let c = colours_of(field(), step_cards(), Quat::IDENTITY, true, &rays);
            FILTERED.with(|f| f.set(true));
            c.map(|c| c.iter().map(|v| v[3]).collect::<Vec<f32>>())
        };
        let Some(plain) = sure(false) else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        let filtered = sure(true).unwrap();
        let worst = |v: &[f32]| v.windows(2).map(|w| (w[1] - w[0]).abs()).fold(0.0f32, f32::max);
        let round = |v: &[f32]| v.iter().map(|x| (x * 100.0).round() / 100.0).collect::<Vec<_>>();
        eprintln!("CARD TRUST across the step: plain {:?}", round(&plain));
        eprintln!("CARD TRUST across the step: filtered {:?}", round(&filtered));
        eprintln!("CARD TRUST worst step: plain {:.3}, filtered {:.3}", worst(&plain), worst(&filtered));
        assert!(worst(&plain) > 0.5, "the plain test drops at once here, or this measures nothing: {:.3}", worst(&plain));
        assert!(worst(&filtered) < 0.15, "filtered, a tenth of a texel moves it a tenth: {:.3}", worst(&filtered));
        assert!(filtered[0] > 0.95 && *filtered.last().unwrap() < filtered[0], "whole a texel off the step, less past it: {:?}", round(&filtered));
    }

    #[test]
    fn the_card_the_surface_faces_colours_it_whatever_the_ray() {
        let Some(c) = colours(
            Quat::IDENTITY,
            true,
            &[
                (on_sphere(Vec3::X), -Vec3::X),
                (on_sphere(Vec3::X), Vec3::new(-0.3, 0.95, 0.0)),
                (on_sphere(Vec3::new(1.0, 1.0, 0.0)), Vec3::new(-1.0, -1.0, 0.0)),
                (on_sphere(-Vec3::Z), Vec3::Z),
            ],
        ) else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        assert!((c[0][0] - 1.0).abs() < 0.05 && c[0][3] == 1.0, "facing +x, from +x: the +x card (1): {:?}", c[0]);
        assert!((c[1][0] - 1.0).abs() < 0.05, "facing +x, from below: still the +x card (1), not the one looking up (4): {:?}", c[1]);
        assert!((c[2][0] - karis(1.0, 3.0)).abs() < 0.05, "facing +x+y: the +x (1) and +y (3) cards alike: {:?}", c[2]);
        assert!((c[3][0] - 6.0).abs() < 0.05, "facing -z: the -z card (6): {:?}", c[3]);
    }

    #[test]
    fn a_shells_two_faces_are_told_apart_by_the_normal() {
        let n = Vec3::new(1.0, 1.0, 0.0).normalize();
        let hit = CENTRE + Vec3::new(0.05, -0.05, 0.0) + n * 0.003;
        let Some(c) = colours_of(plate_field(n), plate_cards(), Quat::IDENTITY, true, &[(hit, Vec3::new(-0.9, 0.43, 0.0))]) else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        assert!((c[0][0] - karis(1.0, 3.0)).abs() < 0.05 && c[0][3] == 1.0, "the upper face: the +x (1) and +y (3) cards alike, nothing of the underside's (2, 4): {:?}", c[0]);
    }

    fn plate_field(n: Vec3) -> ProxyField {
        let cell = 2.0 * HALF / SAMPLES as f32;
        let reach = 4.0 * cell;
        let mut distances = Vec::new();
        for k in 0..SAMPLES {
            for j in 0..SAMPLES {
                for i in 0..SAMPLES {
                    let p = (Vec3::new(i as f32, j as f32, k as f32) + 0.5) * cell - HALF;
                    distances.push(((p.dot(n).abs() / reach).min(1.0) * 255.0).round() as u8);
                }
            }
        }
        ProxyField { dims: [SAMPLES; 3], max_distance: reach, distances, albedo: [0.2; 3] }
    }

    fn plate_cards() -> ProxyCards {
        let n = Vec3::new(1.0, 1.0, 0.0).normalize();
        let mut texels = Vec::new();
        let mut normals = Vec::new();
        for face in 0..6u32 {
            for y in 0..RES {
                for x in 0..RES {
                    let v = ((y as f32 + 0.5) / RES as f32 * 2.0 - 1.0) * HALF;
                    let u = ((x as f32 + 0.5) / RES as f32 * 2.0 - 1.0) * HALF;

                    let (t, facing) = match face {
                        0 => ((HALF + u) / (2.0 * HALF), n),
                        1 => ((HALF - u) / (2.0 * HALF), -n),
                        2 => ((HALF + v) / (2.0 * HALF), n),
                        3 => ((HALF - v) / (2.0 * HALF), -n),
                        _ => (2.0, Vec3::ZERO),
                    };
                    texels.push([(face + 1) as f32, 0.0, 0.0, t]);
                    normals.push(facing.to_array());
                }
            }
        }
        ProxyCards { resolution: RES, texels, normals, albedo: Vec::new() }
    }

    #[test]
    fn a_ray_grazing_a_shell_takes_the_face_it_passes() {
        let n = Vec3::new(1.0, 1.0, 0.0).normalize();
        let along = Vec3::new(-1.0, 1.0, 0.0).normalize();
        let field = (n + 0.1 * along).normalize();
        let d = (along - 0.03 * n).normalize();
        assert!(d.dot(n) < 0.0 && d.dot(field) > 0.0, "the plate faces the ray; the field says it faces away");
        let Some(c) = colours_of(plate_field(field), plate_cards(), Quat::IDENTITY, true, &[(CENTRE + n * 0.003, d)]) else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        assert!((c[0][0] - karis(1.0, 3.0)).abs() < 0.05, "the upper face: the +x (1) and +y (3) cards, nothing of the underside's (2, 4): {:?}", c[0]);
        assert!(c[0][3] > 0.9, "and the cards are sure of it: {:?}", c[0]);
    }

    #[test]
    fn a_card_that_saw_a_surface_facing_away_from_the_ray_is_not_used() {
        let y0 = 0.1;
        let cell = 2.0 * HALF / SAMPLES as f32;
        let reach = 4.0 * cell;
        let mut distances = Vec::new();
        for k in 0..SAMPLES {
            for j in 0..SAMPLES {
                for i in 0..SAMPLES {
                    let p = (Vec3::new(i as f32, j as f32, k as f32) + 0.5) * cell - HALF;
                    distances.push((((p.y - y0).abs() / reach).min(1.0) * 255.0).round() as u8);
                }
            }
        }
        let collar = ProxyField { dims: [SAMPLES; 3], max_distance: reach, distances, albedo: [0.2; 3] };
        let inside = Vec3::new(0.94, -0.34, 0.0).normalize();
        let d = Vec3::new(0.6, 0.8, 0.0);
        assert!(inside.dot(d) > 0.2, "the inside faces away from the ray");
        let mut texels = Vec::new();
        let mut normals = Vec::new();
        for face in 0..6u32 {
            for _ in 0..RES * RES {

                if face == 3 {
                    texels.push([100.0, 0.0, 0.0, (HALF + y0 - 0.005) / (2.0 * HALF)]);
                    normals.push(inside.to_array());
                } else {
                    texels.push([1.0, 0.0, 0.0, 2.0]);
                    normals.push([0.0; 3]);
                }
            }
        }
        let cards = ProxyCards { resolution: RES, texels, normals, albedo: Vec::new() };
        let hit = CENTRE + Vec3::new(0.0, y0 - 0.003, 0.0);
        let Some(c) = colours_of(collar, cards, Quat::IDENTITY, true, &[(hit, d)]) else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        assert!(c[0][3] < 0.02, "no card vouches for the collar: {:?}", c[0]);
    }

    #[test]
    fn a_surface_only_one_card_sees_at_a_slant_is_that_cards() {
        let inside = Vec3::new(-0.95, -0.31, 0.0).normalize();
        let d = Vec3::new(0.8, 0.6, 0.0);
        assert!(inside.dot(d) < -0.5, "the inside faces the ray");
        let mut texels = Vec::new();
        let mut normals = Vec::new();
        for face in 0..6u32 {
            for i in 0..RES * RES {
                if face == 3 {

                    let x = ((i / RES) as f32 + 0.5) / RES as f32 * 2.0 * HALF - HALF;
                    let y = -x * inside.x / inside.y;
                    texels.push([100.0, 0.0, 0.0, 0.5 + y / (2.0 * HALF)]);
                    normals.push(inside.to_array());
                } else {
                    texels.push([0.0, 0.0, 0.0, 2.0]);
                    normals.push([0.0; 3]);
                }
            }
        }
        let cards = ProxyCards { resolution: RES, texels, normals, albedo: Vec::new() };

        let hit = CENTRE + Vec3::new(0.0, -0.003, 0.0);
        let Some(c) = colours_of(field(), cards, Quat::IDENTITY, true, &[(hit, d)]) else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        assert!((c[0][0] - 100.0).abs() < 1.0, "the card below's colour: {:?}", c[0]);
        assert!(c[0][3] > 0.99, "trusted fully: {:?}", c[0]);
    }

    #[test]
    fn a_card_read_from_afar_is_tested_where_the_hit_is() {
        let away = Vec3::new(0.9, -0.44, 0.0).normalize();
        let toward = Vec3::new(-0.9, -0.44, 0.0).normalize();
        let d = Vec3::new(0.8, 0.6, 0.0);
        assert!(away.dot(d) > 0.3 && toward.dot(d) < -0.3);
        let mut texels = Vec::new();
        let mut normals = Vec::new();
        for face in 0..6u32 {
            for _y in 0..RES {
                for x in 0..RES {
                    if face == 3 {
                        texels.push([100.0, 0.0, 0.0, 0.5]);
                        normals.push(if x % 2 == 1 { away } else { toward }.to_array());
                    } else {
                        texels.push([0.0, 0.0, 0.0, 2.0]);
                        normals.push([0.0; 3]);
                    }
                }
            }
        }
        let cards = ProxyCards { resolution: RES, texels, normals, albedo: Vec::new() };

        let z = ((7.5 / RES as f32) - 0.5) * 2.0 * HALF;
        let hit = CENTRE + Vec3::new(0.0, -0.003, z);
        FOOTPRINT.with(|f| f.set(0.1));
        HIT_T.with(|f| f.set(1.0));
        let c = colours_of(field(), cards, Quat::IDENTITY, true, &[(hit, d)]);
        FOOTPRINT.with(|f| f.set(0.0));
        HIT_T.with(|f| f.set(0.0));
        let Some(c) = c else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        assert!(c[0][3] < 0.02, "the surface at the hit faces away: no card vouches: {:?}", c[0]);
    }

    #[test]
    fn a_hit_behind_what_a_card_saw_is_not_what_it_saw() {
        let inside = Vec3::new(0.6, -0.87, 0.0).normalize();
        let d = Vec3::new(0.8, 0.6, 0.0);
        assert!(inside.dot(d) < 0.0 && inside.dot(d) > -0.1, "the inside just faces the ray");
        let mut texels = Vec::new();
        let mut normals = Vec::new();
        for face in 0..6u32 {
            for _ in 0..RES * RES {
                if face == 3 {

                    texels.push([100.0, 0.0, 0.0, 0.5 - 0.5 * 0.01 / HALF]);
                    normals.push(inside.to_array());
                } else {
                    texels.push([0.0, 0.0, 0.0, 2.0]);
                    normals.push([0.0; 3]);
                }
            }
        }
        let cards = ProxyCards { resolution: RES, texels, normals, albedo: Vec::new() };
        let Some(c) = colours_of(field(), cards, Quat::IDENTITY, true, &[(CENTRE, d)]) else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        assert!(c[0][3] < 0.02, "the hit is behind the inside: no card vouches: {:?}", c[0]);
    }

    #[test]
    fn a_card_read_from_afar_vouches_only_for_the_depths_it_saw() {
        let wall = 0.05;
        let mut texels = Vec::new();
        let mut normals = Vec::new();
        for face in 0..6u32 {
            for _ in 0..RES * RES {
                match face {

                    0 => {
                        texels.push([1.0, 0.0, 0.0, 0.5 - 0.5 * wall / HALF]);
                        normals.push([1.0, 0.0, 0.0]);
                    }

                    3 => {
                        texels.push([100.0, 0.0, 0.0, 0.5 - 0.5 * 0.05 / HALF]);
                        normals.push([0.0, -1.0, 0.0]);
                    }
                    _ => {
                        texels.push([0.0, 0.0, 0.0, 2.0]);
                        normals.push([0.0; 3]);
                    }
                }
            }
        }
        let cards = ProxyCards { resolution: RES, texels, normals, albedo: Vec::new() };
        let hit = CENTRE + Vec3::new(wall + 0.003, 0.0, 0.0);

        FOOTPRINT.with(|f| f.set(0.1));
        HIT_T.with(|f| f.set(1.0));
        let c = colours_of(field(), cards, Quat::IDENTITY, true, &[(hit, Vec3::new(-0.6, 0.8, 0.0))]);
        FOOTPRINT.with(|f| f.set(0.0));
        HIT_T.with(|f| f.set(0.0));
        let Some(c) = c else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        assert!(c[0][0] < 2.0, "the wall's own card, nothing of the bright surface 5 cm off: {:?}", c[0]);
        assert!(c[0][3] > 0.9, "and sure of it: {:?}", c[0]);
    }

    #[test]
    fn a_far_reflection_reads_a_small_glint_as_its_share() {
        let mut glinting = cards();
        let mid = RES / 2;
        for y in mid - 1..=mid {
            for x in mid - 1..=mid {
                glinting.texels[(y * RES + x) as usize][0] = 100.0;
            }
        }
        let near = colours_of(field(), glinting.clone(), Quat::IDENTITY, true, &[(on_sphere(Vec3::X), -Vec3::X)]);
        FOOTPRINT.with(|f| f.set(0.6));
        HIT_T.with(|f| f.set(1.0));
        let far = colours_of(field(), glinting, Quat::IDENTITY, true, &[(on_sphere(Vec3::X), -Vec3::X)]);
        FOOTPRINT.with(|f| f.set(0.0));
        HIT_T.with(|f| f.set(0.0));
        let (Some(near), Some(far)) = (near, far) else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        assert!(near[0][0] > 50.0, "near, the glint itself: {:?}", near[0]);
        assert!(far[0][0] > 1.0 && far[0][0] < 10.0, "far, the card's average (glint 4 of ~560 texels): {:?}", far[0]);
    }

    #[test]
    fn a_turned_box_is_read_in_its_own_frame() {
        let q = Quat::from_rotation_y(std::f32::consts::FRAC_PI_2);
        let Some(c) = colours(q, true, &[(on_sphere(q * Vec3::X), q * -Vec3::X)]) else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        assert!((c[0][0] - 1.0).abs() < 0.05, "{:?}", c[0]);
    }

    #[test]
    fn a_proxy_without_cards_has_no_colour_from_them() {
        let Some(c) = colours(Quat::IDENTITY, false, &[(on_sphere(Vec3::X), -Vec3::X)]) else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        assert_eq!(c[0][3], 0.0, "{:?}", c[0]);
    }

    fn albedo_cards(a: f32) -> ProxyCards {
        let mut c = cards();
        c.albedo = c.normals.iter().map(|n| if *n == [0.0; 3] { [0.0; 3] } else { [a; 3] }).collect();
        c
    }

    fn runtime_lamp(at: Vec3, intensity: f32) -> Light {
        Light {
            position: at,
            direction: Vec3::NEG_Y,
            kind: LightKind::Point,
            color: crate::renderer::Color3(255, 255, 255, 255),
            intensity,
            range: 20.0,
            cone_angle_deg: 180.0,
            inner_cone_angle_deg: 0.0,
            mask_channel: None,
            shadow_near: None,
            source_radius: 0.0,
            in_level_bake: false,
        }
    }

    fn relit(lamps: Option<Vec<Light>>, cards: ProxyCards, rays: &[(Vec3, Vec3)]) -> Option<Vec<[f32; 4]>> {
        RELIT.with(|r| *r.borrow_mut() = lamps);
        let out = colours_of(field(), cards, Quat::IDENTITY, true, rays);
        RELIT.with(|r| *r.borrow_mut() = None);
        out
    }

    #[test]
    fn a_lamp_the_bake_never_saw_lights_a_models_reflection() {
        let rays = [(on_sphere(Vec3::X), -Vec3::X), (on_sphere(-Vec3::X), Vec3::X)];
        let Some(unlit) = relit(None, albedo_cards(0.5), &rays) else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        let lamp = runtime_lamp(CENTRE + Vec3::X, 2.0);
        let lit = relit(Some(vec![lamp]), albedo_cards(0.5), &rays).unwrap();
        let seen = relit(Some(vec![Light { in_level_bake: true, ..lamp }]), albedo_cards(0.5), &rays).unwrap();
        let bare = relit(Some(vec![lamp]), cards(), &rays).unwrap();
        let d = 1.0 - (RADIUS + 0.003);
        let want = 0.5 * 2.0 / (d * d);
        eprintln!("CARD RELIT: unlit {unlit:?}\n  lit {lit:?}\n  seen by the bake {seen:?}\n  no albedo {bare:?}\n  want +{want} green on +x");
        assert!(unlit[0][3] > 0.5, "the cards do not vouch for the +x hit: {:?}", unlit[0]);
        assert!(
            (lit[0][1] - unlit[0][1] - want).abs() < 0.03 * want,
            "the +x side's reflection took {} of light, not {want}",
            lit[0][1] - unlit[0][1],
        );
        assert!((lit[1][1] - unlit[1][1]).abs() < 1e-4, "the far side, facing away, was lit: {:?} vs {:?}", lit[1], unlit[1]);
        assert!((seen[0][1] - unlit[0][1]).abs() < 1e-4, "a lamp the bake saw lit the cards twice: {:?}", seen[0]);
        assert!((bare[0][1] - unlit[0][1]).abs() < 1e-4, "cards without albedo were lit: {:?}", bare[0]);
    }
}

#[cfg(test)]
mod highlight_cap_tests {
    use glam::Vec3;
    use wgpu::util::DeviceExt;
    use wgpu::{BindGroupDescriptor, BindGroupEntry, BufferDescriptor, BufferUsages, ShaderModuleDescriptor, ShaderSource};

    fn split(cases: &[(Vec3, Vec3, f32, f32)]) -> Option<Vec<(f32, f32)>> {
        let (device, queue) = crate::renderer::terrain_pipeline::tests::headless_gpu()?;
        let code = format!(
            "{}\n
@group(1) @binding(0) var<storage, read> q: array<vec4<f32>>;
@group(1) @binding(1) var<storage, read_write> out: array<vec4<f32>>;
@compute @workgroup_size(1)
fn split_main(@builtin(global_invocation_id) id: vec3<u32>) {{
    var l: Light;
    l.position = vec4<f32>(q[id.x * 2u].xyz, -1.0);
    l.direction = vec4<f32>(0.0, 0.0, -1.0, 1.0);
    l.color_intensity = vec4<f32>(1.0, 1.0, 1.0, 1.0);
    l.params = vec4<f32>(100.0, -1.0, 0.0, -1.0);
    let eye = q[id.x * 2u + 1u].xyz;
    let c = light_contribution_split(l, vec3<f32>(0.0), vec3<f32>(0.0, 0.0, 1.0), normalize(eye), q[id.x * 2u].w, q[id.x * 2u + 1u].w);
    out[id.x] = vec4<f32>(c.diffuse.r, c.specular.r, 0.0, 0.0);
}}
",
            super::wgsl_lights_block(0, 1)
        );
        let module = device.create_shader_module(ShaderModuleDescriptor { label: None, source: ShaderSource::Wgsl(code.into()) });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &module,
            entry_point: Some("split_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let lights_uniform = super::LightsUniform::new(&device);
        lights_uniform.upload_frame(&queue, &[], &[], false);
        let packed: Vec<[f32; 4]> =
            cases.iter().flat_map(|(lamp, eye, s, k)| [[lamp.x, lamp.y, lamp.z, *s], [eye.x, eye.y, eye.z, *k]]).collect();
        let q = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&packed),
            usage: BufferUsages::STORAGE,
        });
        let size = (cases.len() * 16) as u64;
        let out = device.create_buffer(&BufferDescriptor { label: None, size, usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC, mapped_at_creation: false });
        let read = device.create_buffer(&BufferDescriptor { label: None, size, usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST, mapped_at_creation: false });
        let g0 = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[BindGroupEntry { binding: 1, resource: lights_uniform.buffer().as_entire_binding() }],
        });
        let g1 = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(1),
            entries: &[BindGroupEntry { binding: 0, resource: q.as_entire_binding() }, BindGroupEntry { binding: 1, resource: out.as_entire_binding() }],
        });
        let mut enc = device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &g0, &[]);
            pass.set_bind_group(1, &g1, &[]);
            pass.dispatch_workgroups(cases.len() as u32, 1, 1);
        }
        enc.copy_buffer_to_buffer(&out, 0, &read, 0, size);
        queue.submit([enc.finish()]);
        read.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        let got: Vec<[f32; 4]> = bytemuck::cast_slice(&read.slice(..).get_mapped_range().unwrap()).to_vec();
        Some(got.iter().map(|c| (c[0], c[1])).collect())
    }

    fn lobe(r: f32, d: f32) -> (f32, f32) {
        const LIGHT_SOURCE_RADIUS: f32 = 0.35;
        let alpha = r * r;
        let widened = (alpha + LIGHT_SOURCE_RADIUS / (2.0 * d.max(0.05))).clamp(alpha, 1.0);
        ((2.0 / (widened * widened) - 2.0).clamp(1.0, 320.0), 0.35 * (1.0 - r))
    }

    #[test]
    fn a_highlight_is_never_more_than_physics_allows() {

        let lamp = Vec3::new(0.15, 0.0, 0.6);
        let mirror = Vec3::new(-0.15, 0.0, 0.6);
        let aside = Vec3::new(-0.6 * 40f32.to_radians().sin(), 0.0, 0.6 * 40f32.to_radians().cos());
        let (stone_s, stone_k) = lobe(0.6, 0.6);
        let (marble_s, marble_k) = lobe(0.048, 3.0);
        let (marble_near_s, marble_near_k) = lobe(0.048, 0.6);
        let cases = [
            (lamp, mirror, stone_s, stone_k),
            (lamp, aside, stone_s, stone_k),
            (lamp * 5.0, mirror * 5.0, marble_s, marble_k),
            (lamp, mirror, marble_near_s, marble_near_k),
        ];
        let Some(got) = split(&cases) else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        for (c, (d, s)) in cases.iter().zip(&got) {
            eprintln!("exponent {:6.1} strength {:.3}: diffuse {d:.4} highlight {s:.4} ({:.1}% of the diffuse light)", c.2, c.3, 100.0 * s / d);
        }

        let shipped = |k: usize| {
            let (l, e, s, st) = cases[k];
            let h = (l.normalize() + e.normalize()).normalize();
            let nl = l.normalize().z;
            got[k].0 / nl * h.z.powf(s) * st
        };

        let stone = got[0].1 / got[0].0;
        assert!(stone < 0.07, "rough stone's highlight at the torch: {stone} of its diffuse light");
        assert!(got[0].1 < 0.5 * shipped(0), "and well under what shipped: {} against {}", got[0].1, shipped(0));
        assert!(got[1].1 <= got[0].1, "and less off the mirror angle");

        assert!((got[2].1 - shipped(2)).abs() <= 1e-4 * shipped(2).max(1e-6), "marble across the room: {} against {}", got[2].1, shipped(2));
        assert!(got[3].1 <= shipped(3) * (1.0 + 1e-4) && got[3].1 > 0.2 * shipped(3), "marble beside a torch: {} against {}", got[3].1, shipped(3));
    }
}

#[cfg(test)]
mod spot_edge_gpu_tests {
    use glam::Vec3;
    use wgpu::util::DeviceExt;
    use wgpu::{BindGroupDescriptor, BindGroupEntry, BufferDescriptor, BufferUsages, ShaderModuleDescriptor, ShaderSource};

    const WALL: f32 = 0.85;
    const OUTER_DEG: f32 = 17.0;
    const INNER_DEG: f32 = 8.0;
    const ALONG: f32 = 0.2;
    const UP: f32 = 0.019;

    fn cone(at: &[(Vec3, Vec3, Vec3, f32)], average: bool, plane: Option<Vec3>) -> Option<Vec<f32>> {
        let (device, queue) = crate::renderer::terrain_pipeline::tests::headless_gpu()?;
        let options = super::LightsBlockOptions { long_step_from_plane: plane.is_some(), ..Default::default() };
        let block = super::wgsl_lights_block_with(0, 1, options);
        let set_plane = plane.map_or(String::new(), |p| format!("    set_pixel_long_step(vec3<f32>({:?}, {:?}, {:?}), view);\n", p.x, p.y, p.z));
        let code = format!(
            "{}\n
@group(1) @binding(0) var<storage, read> q: array<vec4<f32>>;
@group(1) @binding(1) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(1)
fn edge_main(@builtin(global_invocation_id) id: vec3<u32>) {{
    let p = q[id.x * 3u].xyz;
    let view = q[id.x * 3u + 1u].xyz;
    let n = q[id.x * 3u + 2u].xyz;
    pixel_footprint = q[id.x * 3u].w;
{}    let dist = length(p);
    let l_dir = -p / dist;
    let d = vec3<f32>(1.0, 0.0, 0.0);
    let cos_angle = dot(-l_dir, d);
    out[id.x] = spot_cone(cos_angle, {:?}, {:?}, dist, spot_cone_across(d, l_dir, cos_angle, dist, n, view));
}}
",
            if average {
                block
            } else {
                block.replacen("const SPOT_EDGE_AVERAGE: bool = true;", "const SPOT_EDGE_AVERAGE: bool = false;", 1)
            },
            set_plane,
            OUTER_DEG.to_radians().cos(),
            INNER_DEG.to_radians().cos(),
        );
        let module = device.create_shader_module(ShaderModuleDescriptor { label: None, source: ShaderSource::Wgsl(code.into()) });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &module,
            entry_point: Some("edge_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let packed: Vec<[f32; 4]> = at
            .iter()
            .flat_map(|(p, v, n, f)| [[p.x, p.y, p.z, *f], [v.x, v.y, v.z, 0.0], [n.x, n.y, n.z, 0.0]])
            .collect();
        let q = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&packed),
            usage: BufferUsages::STORAGE,
        });
        let size = (at.len() * 4) as u64;
        let out = device.create_buffer(&BufferDescriptor { label: None, size, usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC, mapped_at_creation: false });
        let read = device.create_buffer(&BufferDescriptor { label: None, size, usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST, mapped_at_creation: false });
        let g0 = device.create_bind_group(&BindGroupDescriptor { label: None, layout: &pipeline.get_bind_group_layout(0), entries: &[] });
        let g1 = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(1),
            entries: &[
                BindGroupEntry { binding: 0, resource: q.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: out.as_entire_binding() },
            ],
        });
        let mut enc = device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &g0, &[]);
            pass.set_bind_group(1, &g1, &[]);
            pass.dispatch_workgroups(at.len() as u32, 1, 1);
        }
        enc.copy_buffer_to_buffer(&out, 0, &read, 0, size);
        queue.submit([enc.finish()]);
        read.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        let got: Vec<f32> = bytemuck::cast_slice(&read.slice(..).get_mapped_range().unwrap()).to_vec();
        Some(got)
    }

    fn shimmer(profile: &[f32], step: f32, slide: f32) -> f32 {
        let (every, by) = ((step * 1000.0).round() as usize, (slide * 1000.0).round() as usize);
        let (mut sum, mut n) = (0.0f32, 0usize);
        for start in (0..profile.len()).step_by(every) {
            let v: Vec<f32> = (0..=every / by + 1).filter_map(|j| profile.get(start + j * by).copied()).collect();
            for w in v.windows(3) {
                sum += (w[2] - 2.0 * w[1] + w[0]).powi(2);
                n += 1;
            }
        }
        (sum / n.max(1) as f32).sqrt()
    }

    #[test]
    fn an_edge_on_pool_is_averaged_along_the_long_step_alone() {
        let footprint = (ALONG * UP).sqrt();

        let wall_n = Vec3::NEG_X;
        let grazing = Vec3::new(-1.5, 0.0, 15.8).normalize();
        assert!((footprint / grazing.dot(wall_n).abs().sqrt() - ALONG).abs() < 0.002, "the pixel's long step here");
        let n = 1201;
        let along: Vec<Vec3> = (0..n).map(|i| Vec3::new(WALL, 0.0, -0.6 + i as f32 * 0.001)).collect();
        let up: Vec<Vec3> = (0..n).map(|i| Vec3::new(WALL, -0.6 + i as f32 * 0.001, 0.0)).collect();
        let mut q = Vec::new();
        for p in along.iter().chain(&up) {
            q.push((*p, grazing, wall_n, footprint));
        }
        for p in &along {
            q.push((*p, -wall_n, wall_n, UP));
        }
        let (Some(now_all), Some(was_all)) = (cone(&q, true, None), cone(&q, false, None)) else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        let pick = |c: &[f32], from: usize| -> Vec<f32> { c[from..from + n].to_vec() };
        let (along_now, along_was) = (pick(&now_all, 0), pick(&was_all, 0));
        let (up_now, up_was) = (pick(&now_all, n), pick(&was_all, n));
        let (head_now, head_was) = (pick(&now_all, 2 * n), pick(&was_all, 2 * n));
        let slide = 0.16 * ALONG;
        let (now, was) = (shimmer(&along_now, ALONG, slide), shimmer(&along_was, ALONG, slide));

        let w = (ALONG * 1000.0) as usize;
        let sensor: Vec<f32> = (0..n)
            .map(|i| (i.saturating_sub(w / 2)..(i + w / 2).min(n)).map(|k| along_was[k]).sum::<f32>() / w as f32)
            .collect();
        let floor = shimmer(&sensor, ALONG, slide);
        let total = |v: &[f32]| v.iter().sum::<f32>() * 0.001;
        let largest = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max);
        eprintln!(
            "SPOT EDGE ON: along the wall shimmer {was:.4} -> {now:.4} (a sensor's pixels {floor:.4}); light along it {:.4} -> {:.4} m; up it largest change {:.2e}; head-on {:.2e}",
            total(&along_was),
            total(&along_now),
            largest(&up_now, &up_was),
            largest(&head_now, &head_was),
        );
        assert!(was > 0.02, "the test no longer reproduces the shimmer: {was}");
        assert!(
            now < was / 2.0 && now < 1.5 * floor,
            "the far pool still shimmers along the wall: {was} -> {now}, where a sensor's pixels give {floor}",
        );

        assert!(
            (total(&along_now) / total(&along_was) - 1.0).abs() < 0.05,
            "averaging moved the pool's light along the wall: {} -> {}",
            total(&along_was),
            total(&along_now),
        );
        assert!(largest(&up_now, &up_was) < 1e-5, "the pool changed up the wall, where the pixel is short");
        assert!(largest(&head_now, &head_was) < 1e-5, "a pool seen head-on changed");
    }

    #[test]
    fn a_bump_turned_edge_on_pulls_no_light_past_the_pool() {
        let wall_n = Vec3::NEG_X;

        let (square, footprint) = (Vec3::NEG_X, 0.02);
        let bump = Vec3::new(-0.0005, 1.0, 0.0).normalize();

        let past: Vec<Vec3> = (0..=35).map(|i| Vec3::new(WALL, 0.0, 0.265 + i as f32 * 0.001)).collect();
        let bumpy: Vec<_> = past.iter().map(|&p| (p, square, bump, footprint)).collect();

        let grazing = Vec3::new(-1.5, 0.0, 15.8).normalize();
        let flat: Vec<_> = (0..=300)
            .map(|i| Vec3::new(WALL, 0.0, 0.05 + i as f32 * 0.001))
            .flat_map(|p| [(p, square, wall_n, footprint), (p, grazing, wall_n, (ALONG * UP).sqrt())])
            .collect();
        let (Some(from_bump), Some(from_plane), Some(flat_was), Some(flat_now)) =
            (cone(&bumpy, true, None), cone(&bumpy, true, Some(wall_n)), cone(&flat, true, None), cone(&flat, true, Some(wall_n)))
        else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        let most = |v: &[f32]| v.iter().cloned().fold(0.0f32, f32::max);
        let largest = flat_was.iter().zip(&flat_now).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        eprintln!(
            "PAST THE POOL: a bump's step lights it up to {:.3}, the plane's {:.2e}; a flat pixel's cone moves {largest:.2e} (lit {:.3} at most)",
            most(&from_bump),
            most(&from_plane),
            most(&flat_was),
        );
        assert!(most(&from_bump) > 0.1, "the test no longer reproduces the specks: {}", most(&from_bump));
        assert!(most(&from_plane) < 1e-6, "a bump still pulls light past the pool: {}", most(&from_plane));
        assert!(most(&flat_was) > 0.9, "the flat row does not cross the pool's edge: {}", most(&flat_was));
        assert!(largest < 1e-6, "a flat pixel's cone changed by {largest}");
    }
}

#[cfg(test)]
mod surface_light_gpu_tests {
    use super::{Light, LightKind, LightsUniform};
    use glam::Vec3;
    use wgpu::util::DeviceExt;
    use wgpu::{BindGroupDescriptor, BindGroupEntry, BufferDescriptor, BufferUsages, ShaderModuleDescriptor, ShaderSource};

    fn both(lights: &[Light], at: &[(Vec3, Vec3)], average: bool) -> Option<Vec<(Vec3, Vec3)>> {
        let (device, queue) = crate::renderer::terrain_pipeline::tests::headless_gpu()?;
        let uniform = LightsUniform::new(&device);
        uniform.upload_frame(&queue, lights, &[], false);
        let code = format!(
            "{}\n
@group(1) @binding(0) var<storage, read> q: array<vec4<f32>>;
@group(1) @binding(1) var<storage, read_write> out: array<vec4<f32>>;
@compute @workgroup_size(1)
fn surface_main(@builtin(global_invocation_id) id: vec3<u32>) {{
    let p = q[id.x * 2u].xyz;
    let n = q[id.x * 2u + 1u].xyz;

    pixel_footprint = 0.005;
    terminator_width = 0.1;
    let view = normalize(vec3<f32>(0.0, 1.6, 0.0) - p);
    var lamps = vec3<f32>(0.0);
    for (var i: u32 = 0u; i < lights.surface_lights.x; i = i + 1u) {{
        lamps = lamps + light_contribution_split(lights.lights[i], p, n, view, 1.0, 0.0).diffuse;
    }}
    out[id.x * 2u] = vec4<f32>(surface_lights(p, n), f32(lights.surface_lights.x));
    out[id.x * 2u + 1u] = vec4<f32>(lamps, 0.0);
}}
",
            if average {
                super::wgsl_lights_block(0, 1)
            } else {
                super::wgsl_lights_block(0, 1).replacen("const SPOT_EDGE_AVERAGE: bool = true;", "const SPOT_EDGE_AVERAGE: bool = false;", 1)
            },
        );
        let module = device.create_shader_module(ShaderModuleDescriptor { label: None, source: ShaderSource::Wgsl(code.into()) });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &module,
            entry_point: Some("surface_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let packed: Vec<[f32; 4]> = at.iter().flat_map(|(p, n)| [[p.x, p.y, p.z, 0.0], [n.x, n.y, n.z, 0.0]]).collect();
        let q = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&packed),
            usage: BufferUsages::STORAGE,
        });
        let size = (at.len() * 32) as u64;
        let out = device.create_buffer(&BufferDescriptor { label: None, size, usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC, mapped_at_creation: false });
        let read = device.create_buffer(&BufferDescriptor { label: None, size, usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST, mapped_at_creation: false });
        let g0 = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[BindGroupEntry { binding: 1, resource: uniform.buffer().as_entire_binding() }],
        });
        let g1 = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(1),
            entries: &[
                BindGroupEntry { binding: 0, resource: q.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: out.as_entire_binding() },
            ],
        });
        let mut enc = device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &g0, &[]);
            pass.set_bind_group(1, &g1, &[]);
            pass.dispatch_workgroups(at.len() as u32, 1, 1);
        }
        enc.copy_buffer_to_buffer(&out, 0, &read, 0, size);
        queue.submit([enc.finish()]);
        read.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        let got: Vec<[f32; 4]> = bytemuck::cast_slice(&read.slice(..).get_mapped_range().unwrap()).to_vec();
        assert!(got.iter().step_by(2).all(|v| v[3] == 2.0), "the upload counted the two leading surfaces' lights");
        Some(got.chunks(2).map(|c| (Vec3::new(c[0][0], c[0][1], c[0][2]), Vec3::new(c[1][0], c[1][1], c[1][2]))).collect())
    }

    #[test]
    fn the_surface_loop_lights_as_the_lamp_loop_did() {
        let bounce = Light {
            position: Vec3::new(0.0, 1.0, -2.0),
            direction: Vec3::Z,
            kind: LightKind::Spot,
            color: crate::renderer::Color3(230, 200, 170, 255),
            intensity: 3.0,
            range: 6.0,
            cone_angle_deg: 2.0 * 105.0,
            inner_cone_angle_deg: 0.0,
            mask_channel: None,
            shadow_near: Some(f32::INFINITY),
            source_radius: 0.4,
            in_level_bake: false,
        };
        let corner = Light { position: Vec3::new(1.5, 0.5, -1.0), kind: LightKind::Point, source_radius: 0.7, intensity: 1.0, ..bounce };
        let lamp = Light { position: Vec3::new(0.0, 2.5, 0.0), shadow_near: None, source_radius: 0.0, in_level_bake: true, ..corner };
        assert!(bounce.is_surface_light() && corner.is_surface_light() && !lamp.is_surface_light());
        let normals = [Vec3::Y, Vec3::NEG_Y, Vec3::Z, Vec3::NEG_Z, Vec3::X, Vec3::new(0.3, 0.8, -0.5).normalize()];
        let mut at = Vec::new();
        for k in 0..400 {
            let t = k as f32 / 400.0;

            let r = 0.05 + 7.0 * t;
            let a = 37.0 * t;
            let p = bounce.position + Vec3::new(r * a.cos() * (0.3 + t), r * (2.0 * t - 1.0), r * a.sin());
            at.push((p, normals[k % normals.len()]));
        }
        let lights = [bounce, corner, lamp];
        let (Some(plain), Some(averaged)) = (both(&lights, &at, false), both(&lights, &at, true)) else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        let off = |a: Vec3, b: Vec3| (a.max_element() - b.max_element()).abs() / b.max_element().max(1e-3);
        let mut lit = 0;
        for ((p, n), (apart, lamps)) in at.iter().zip(&plain) {
            lit += usize::from(lamps.max_element() > 0.0);
            assert!(off(*apart, *lamps) < 1e-4, "at {p:?} facing {n:?}: apart {apart:?}, as lamps {lamps:?}");
        }
        assert!(lit > at.len() / 3, "the test lit too little to say anything: {lit}");

        let far = at.iter().position(|(p, _)| (*p - bounce.position).length() > 6.5 && (*p - corner.position).length() > 6.5).unwrap();
        assert_eq!(plain[far].0, Vec3::ZERO);
        let worst = |near: f32, far: f32| {
            at.iter()
                .zip(&averaged)
                .filter(|((p, _), _)| (near..far).contains(&(*p - bounce.position).length()))
                .map(|(_, (apart, lamps))| off(*apart, *lamps))
                .fold(0.0f32, f32::max)
        };
        eprintln!(
            "SURFACE LIGHTS: {lit} of {} points lit; the lamp maths' average moved them by at most {:.1}% within 0.3 m of the bounce's light, {:.2}% from 0.3 to 1 m, {:.3}% past 1 m",
            at.len(),
            100.0 * worst(0.0, 0.3),
            100.0 * worst(0.3, 1.0),
            100.0 * worst(1.0, 100.0),
        );
    }
}

#[cfg(test)]
mod capsule_ambient_tests {
    use glam::Vec3;
    use wgpu::util::DeviceExt;
    use wgpu::{BindGroupDescriptor, BindGroupEntry, BufferDescriptor, BufferUsages, ShaderModuleDescriptor, ShaderSource};

    use crate::renderer::uniforms::{CapsuleGroup, CapsuleUpload, Uniforms};

    const REACH: f32 = 0.6;

    fn body(arm_out: bool) -> (CapsuleUpload, (Vec3, Vec3, f32)) {
        let v = Vec3::new;
        let arm = if arm_out {
            (v(0.22, 1.42, 0.0), v(0.22, 1.42, -0.42), 0.045)
        } else {
            (v(0.22, 1.42, 0.0), v(0.24, 0.9, 0.0), 0.045)
        };
        let caps = CapsuleUpload::from_groups(&[CapsuleGroup {
            capsules: vec![
                (v(0.0, 1.6, 0.0), v(0.0, 1.68, 0.0), 0.1),
                (v(0.0, 1.4, 0.0), v(0.0, 0.95, 0.0), 0.15),
                (v(-0.22, 1.42, 0.0), v(-0.24, 0.9, 0.0), 0.045),
                arm,
                (v(-0.1, 0.92, 0.0), v(-0.1, 0.08, 0.0), 0.06),
                (v(0.1, 0.92, 0.0), v(0.1, 0.08, 0.0), 0.06),
            ],
            colour: [0.5; 3],
            surfaces: Vec::new(),
        }]);
        (caps, arm)
    }

    fn from_surface(p: Vec3, (a, b, r): (Vec3, Vec3, f32)) -> f32 {
        let ba = b - a;
        let s = ((p - a).dot(ba) / ba.length_squared().max(1e-8)).clamp(0.0, 1.0);
        (a + ba * s - p).length() - r
    }

    fn ambient(caps: &CapsuleUpload, at: &[(Vec3, Vec3)]) -> Option<Vec<f32>> {
        let (device, queue) = crate::renderer::terrain_pipeline::tests::headless_gpu()?;
        let mut u: Uniforms = bytemuck::Zeroable::zeroed();
        u.capsules = caps.capsules;
        u.capsule_groups = caps.groups;
        u.capsule_params = [caps.group_count as f32, -1.0, -1.0, 0.0];
        let code = format!(
            "{}\n{}",
            super::wgsl_lights_block(0, 1),
            r#"
@group(1) @binding(0) var<storage, read> pts: array<vec4<f32>>;
@group(1) @binding(1) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(1)
fn ambient_main(@builtin(global_invocation_id) id: vec3<u32>) {
    out[id.x] = capsule_ambient(pts[id.x * 2u].xyz, pts[id.x * 2u + 1u].xyz);
}
"#
        );
        let module = device.create_shader_module(ShaderModuleDescriptor { label: None, source: ShaderSource::Wgsl(code.into()) });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &module,
            entry_point: Some("ambient_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let packed: Vec<[f32; 4]> = at.iter().flat_map(|(p, n)| [[p.x, p.y, p.z, 0.0], [n.x, n.y, n.z, 0.0]]).collect();
        let camera = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::bytes_of(&u),
            usage: BufferUsages::UNIFORM,
        });
        let pts = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&packed),
            usage: BufferUsages::STORAGE,
        });
        let size = (at.len() * 4) as u64;
        let out = device.create_buffer(&BufferDescriptor {
            label: None,
            size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let read = device.create_buffer(&BufferDescriptor {
            label: None,
            size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let g0 = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[BindGroupEntry { binding: 0, resource: camera.as_entire_binding() }],
        });
        let g1 = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(1),
            entries: &[
                BindGroupEntry { binding: 0, resource: pts.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: out.as_entire_binding() },
            ],
        });
        let mut enc = device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &g0, &[]);
            pass.set_bind_group(1, &g1, &[]);
            pass.dispatch_workgroups(at.len() as u32, 1, 1);
        }
        enc.copy_buffer_to_buffer(&out, 0, &read, 0, size);
        queue.submit([enc.finish()]);
        read.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        let got: Vec<f32> = bytemuck::cast_slice(&read.slice(..).get_mapped_range().unwrap()).to_vec();
        Some(got)
    }

    fn wall_and_floor() -> (Vec<(Vec3, Vec3)>, Vec<(Vec3, Vec3)>, Vec<(Vec3, Vec3)>) {
        let across = (0..=400).map(|i| (Vec3::new(-2.0 + i as f32 * 0.01, 1.2, -0.5), Vec3::Z)).collect();
        let up = (0..=300).map(|i| (Vec3::new(0.05, i as f32 * 0.01, -0.5), Vec3::Z)).collect();
        let floor = (0..=80)
            .flat_map(|i| (0..=80).map(move |j| (Vec3::new(-2.0 + i as f32 * 0.05, 0.0, -2.0 + j as f32 * 0.05), Vec3::Y)))
            .collect();
        (across, up, floor)
    }

    #[test]
    fn a_characters_contact_darkening_fades_out_without_a_rim() {
        let (across, up, _) = wall_and_floor();
        for arm_out in [false, true] {
            let (caps, _) = body(arm_out);
            for line in [&across, &up] {
                let Some(vis) = ambient(&caps, line) else {
                    eprintln!("no GPU adapter; skipping");
                    return;
                };
                let (i, step) = vis
                    .windows(2)
                    .map(|w| (w[1] - w[0]).abs())
                    .enumerate()
                    .fold((0, 0.0f32), |best, (i, s)| if s > best.1 { (i, s) } else { best });
                assert!(step < 0.004, "arm out {arm_out}: a step of {step:.4} at {:?}", line[i].0);

                let darkest = vis.iter().cloned().fold(1.0f32, f32::min);
                assert!(darkest < 0.97, "arm out {arm_out}: nothing darkened ({darkest})");
            }
        }
    }

    #[test]
    fn raising_an_arm_changes_only_the_darkening_round_the_arm() {
        let (across, up, floor) = wall_and_floor();
        let all: Vec<(Vec3, Vec3)> = across.into_iter().chain(up).chain(floor).collect();
        let ((down, arm_down), (out, arm_out)) = (body(false), body(true));
        let (Some(a), Some(b)) = (ambient(&down, &all), ambient(&out, &all)) else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        let mut checked = 0;
        for (k, (p, _)) in all.iter().enumerate() {
            if from_surface(*p, arm_down) > REACH && from_surface(*p, arm_out) > REACH {
                checked += 1;
                assert!((a[k] - b[k]).abs() < 1e-5, "{p}: {} with the arm down, {} with it out", a[k], b[k]);
            }
        }
        assert!(checked > 1000, "{checked}");
    }

    #[test]
    fn a_hand_near_a_wall_and_feet_on_the_floor_still_darken_them() {
        let (caps, _) = body(true);
        let Some(vis) = ambient(&caps, &[(Vec3::new(0.22, 1.42, -0.5), Vec3::Z), (Vec3::new(0.1, 0.0, 0.0), Vec3::Y)]) else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        assert!(vis[0] < 0.75, "the wall under the hand: {}", vis[0]);
        assert!(vis[1] < 0.8, "the floor under a foot: {}", vis[1]);
    }
}

#[cfg(test)]
mod capsule_reflection_tests {
    use glam::{Quat, Vec3};
    use wgpu::util::DeviceExt;
    use wgpu::{
        BindGroupDescriptor, BindGroupEntry, BindingResource, BufferDescriptor, BufferUsages, ShaderModuleDescriptor,
        ShaderSource,
    };

    use crate::renderer::uniforms::{CapsuleGroup, CapsuleUpload, Uniforms};

    const GLASS: [f32; 3] = [1.0, 0.945, 0.894];
    const DRIVE: f32 = 6400.0;
    const ALBEDO: f32 = 0.037;
    const GLASS_RADIUS: f32 = 0.0168;
    const BODY_RADIUS: f32 = 0.0165;

    fn torch(at: Vec3, forward: Vec3) -> CapsuleGroup {
        let back = -forward.normalize();
        CapsuleGroup {
            capsules: vec![
                (at + back * (0.15 - BODY_RADIUS), at + back * BODY_RADIUS, BODY_RADIUS),
                (at + back * 0.01, at, GLASS_RADIUS),
            ],
            colour: GLASS,
            surfaces: vec![-ALBEDO, DRIVE],
        }
    }

    fn trunk(at: Vec3) -> CapsuleGroup {
        CapsuleGroup {
            capsules: vec![(at + Vec3::Y * 0.3, at - Vec3::Y * 0.3, 0.3)],
            colour: [0.5, 0.25, 0.1],
            surfaces: Vec::new(),
        }
    }

    struct Seen {
        reflection: [f32; 4],
        glow: [f32; 3],
    }

    fn reflect(groups: &[CapsuleGroup], rays: &[(Vec3, Vec3, f32)]) -> Option<Vec<Seen>> {
        let (device, queue) = crate::renderer::terrain_pipeline::tests::headless_gpu()?;
        let caps = CapsuleUpload::from_groups(groups);
        let mut u: Uniforms = bytemuck::Zeroable::zeroed();
        u.capsules = caps.capsules;
        u.capsule_groups = caps.groups;
        u.capsule_params = [caps.group_count as f32, -1.0, -1.0, 0.0];

        u.camera_pos = [[0.0, 0.0, -1.0e4, 0.0]; 2];
        let code = format!(
            "{}\n{}",
            super::wgsl_lights_block(0, 1),
            r#"
@group(1) @binding(0) var<storage, read> rays: array<vec4<f32>>;
@group(1) @binding(1) var<storage, read_write> out: array<vec4<f32>>;
@compute @workgroup_size(1)
fn reflect_main(@builtin(global_invocation_id) id: vec3<u32>) {
    pixel_footprint = rays[id.x * 2u].w;
    let r = capsule_reflection(rays[id.x * 2u].xyz, normalize(rays[id.x * 2u + 1u].xyz), 0.0, vec3<f32>(3.14159265), vec4<f32>(0.0));
    out[id.x * 2u] = r;
    out[id.x * 2u + 1u] = vec4<f32>(capsule_glow * r.a, 0.0);
}
"#
        );
        let module = device.create_shader_module(ShaderModuleDescriptor { label: None, source: ShaderSource::Wgsl(code.into()) });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &module,
            entry_point: Some("reflect_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let packed: Vec<[f32; 4]> =
            rays.iter().flat_map(|(p, d, f)| [[p.x, p.y, p.z, *f], [d.x, d.y, d.z, 0.0]]).collect();
        let camera = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::bytes_of(&u),
            usage: BufferUsages::UNIFORM,
        });
        let ray_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&packed),
            usage: BufferUsages::STORAGE,
        });
        let size = (rays.len() * 32) as u64;
        let out = device.create_buffer(&BufferDescriptor {
            label: None,
            size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let read = device.create_buffer(&BufferDescriptor {
            label: None,
            size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let (_, samp) = crate::renderer::uniforms::default_probe_cube(&device);
        let cards = crate::renderer::proxy_cards::none(&device);

        let lights_uniform = super::LightsUniform::new(&device);
        let shadow_map = crate::renderer::shadow::ShadowMap::with_dimension(&device, 64);
        let g0 = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry { binding: 0, resource: camera.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: lights_uniform.buffer().as_entire_binding() },
                BindGroupEntry { binding: 3, resource: BindingResource::Sampler(shadow_map.sampler()) },
                BindGroupEntry { binding: 4, resource: BindingResource::TextureView(shadow_map.spot_depth_view()) },
                BindGroupEntry { binding: 6, resource: BindingResource::Sampler(&samp) },
                BindGroupEntry { binding: 12, resource: BindingResource::TextureView(&cards) },
            ],
        });
        let g1 = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(1),
            entries: &[
                BindGroupEntry { binding: 0, resource: ray_buf.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: out.as_entire_binding() },
            ],
        });
        let mut enc = device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &g0, &[]);
            pass.set_bind_group(1, &g1, &[]);
            assert!(rays.len() <= 65535, "one workgroup a ray, along x");
            pass.dispatch_workgroups(rays.len() as u32, 1, 1);
        }
        enc.copy_buffer_to_buffer(&out, 0, &read, 0, size);
        queue.submit([enc.finish()]);
        read.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        let got: Vec<[f32; 4]> = bytemuck::cast_slice(&read.slice(..).get_mapped_range().unwrap()).to_vec();
        Some(got.chunks(2).map(|c| Seen { reflection: c[0], glow: [c[1][0], c[1][1], c[1][2]] }).collect())
    }

    fn near(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn a_torchs_glass_glows_in_front_of_it_and_its_body_hides_it_from_behind() {
        let groups = [torch(Vec3::new(0.0, 1.0, 0.0), Vec3::NEG_Z)];
        let f = 0.002;
        let rays = [
            (Vec3::new(0.0, 1.0, -2.0), Vec3::Z, f),
            (Vec3::new(0.008, 1.005, -2.0), Vec3::Z, f),
            (Vec3::new(0.05, 1.0, -2.0), Vec3::Z, f),
            (Vec3::new(0.0, 1.0, 2.0), Vec3::NEG_Z, f),
            (Vec3::new(2.0, 1.0, 0.07), Vec3::NEG_X, f),
        ];
        let Some(got) = reflect(&groups, &rays) else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        for (i, s) in got.iter().enumerate() {
            eprintln!("ray {i}: reflection {:?} glow {:?}", s.reflection, s.glow);
        }
        for (i, what) in [(0, "the glass's middle"), (1, "across the glass")] {
            for c in 0..3 {
                assert!(near(got[i].glow[c], GLASS[c] * DRIVE, 0.01 * DRIVE), "{what}: {:?}", got[i].glow);
            }
            assert!(got[i].reflection[3] > 0.99, "{what} covers the ray: {:?}", got[i].reflection);
        }
        assert!(got[2].glow[0] < 1e-3 && got[2].reflection[3] < 1e-3, "beside it: {:?}", got[2].reflection);
        for (i, what) in [(3, "behind it"), (4, "beside the body")] {
            assert_eq!(got[i].glow, [0.0; 3], "{what}: no glass in sight");
            assert!(got[i].reflection[3] > 0.99, "{what}: the body covers the ray");
            for c in 0..3 {
                assert!(near(got[i].reflection[c], GLASS[c] * ALBEDO, 1e-3), "{what}: {:?}", got[i].reflection);
            }
        }

        let mut glass = torch(Vec3::new(0.0, 1.0, 0.0), Vec3::NEG_Z);
        glass.capsules.remove(0);
        glass.surfaces.remove(0);
        let Some(back) = reflect(&[glass], &rays[3..4]) else { return };
        assert_eq!(back[0].glow, [0.0; 3], "the glass's back");
    }

    #[test]
    fn a_blurred_glass_keeps_the_light_it_gives_off() {
        let at = Vec3::new(0.0, 1.0, 0.0);
        let glass_only = CapsuleGroup {
            capsules: vec![(at + Vec3::Z * 0.01, at, GLASS_RADIUS)],
            colour: [1.0; 3],
            surfaces: vec![DRIVE],
        };
        for f in [0.002, 0.01, 0.03, 0.1] {

            let reach = GLASS_RADIUS.max(f) + 1.5 * f;
            let step = GLASS_RADIUS.min(f) / 4.0;
            let n = (reach / step).ceil() as i32;
            let rays: Vec<(Vec3, Vec3, f32)> = (-n..=n)
                .flat_map(|i| (-n..=n).map(move |j| (Vec3::new(i as f32 * step, 1.0 + j as f32 * step, -2.0), Vec3::Z, f)))
                .collect();
            let Some(got) = reflect(&[glass_only.clone()], &rays) else {
                eprintln!("no GPU adapter; skipping");
                return;
            };
            let light: f32 = got.iter().map(|s| s.glow[0] * step * step).sum();
            let want = DRIVE * std::f32::consts::PI * GLASS_RADIUS * GLASS_RADIUS;
            eprintln!("footprint {f}: {light} of {want}, peak {}", got.iter().map(|s| s.glow[0]).fold(0.0, f32::max));
            assert!((light / want - 1.0).abs() < 0.03, "footprint {f}: {light} of {want}");
        }
    }

    #[test]
    fn a_torch_in_front_of_a_body_shows_and_one_behind_it_is_hidden() {

        let across = [trunk(Vec3::new(0.0, 1.0, 0.4)), torch(Vec3::new(0.05, 1.0, 0.0), Vec3::X)];

        let behind = [trunk(Vec3::new(0.0, 1.0, -0.5)), torch(Vec3::new(0.0, 1.0, 0.0), Vec3::NEG_Z)];
        let ray = [(Vec3::new(-0.04, 1.0, -2.0), Vec3::Z, 0.002), (Vec3::new(0.0, 1.0, -2.0), Vec3::Z, 0.002)];
        let (Some(a), Some(b)) = (reflect(&across, &ray[..1]), reflect(&behind, &ray[1..])) else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        eprintln!("across the chest: {:?}; behind someone: {:?} glow {:?}", a[0].reflection, b[0].reflection, b[0].glow);
        for c in 0..3 {
            assert!(near(a[0].reflection[c], GLASS[c] * ALBEDO, 1e-3), "the torch in front: {:?}", a[0].reflection);
        }
        assert!(b[0].glow.iter().all(|g| *g < 1e-3), "the glass behind the trunk: {:?}", b[0].glow);
        assert!(near(b[0].reflection[0], 0.5, 0.01), "the trunk in front: {:?}", b[0].reflection);
    }

    fn seen_past_the_beam(
        groups: &[CapsuleGroup],
        rays: &[(Vec3, Vec3, f32)],
        lamps: &[super::Light],
        layered: bool,
        depth: f32,
    ) -> Option<Vec<[f32; 3]>> {
        let (device, queue) = crate::renderer::terrain_pipeline::tests::headless_gpu()?;
        let caps = CapsuleUpload::from_groups(groups);
        let mut u: Uniforms = bytemuck::Zeroable::zeroed();
        u.capsules = caps.capsules;
        u.capsule_groups = caps.groups;
        u.capsule_params = [caps.group_count as f32, -1.0, -1.0, 0.0];
        u.camera_pos = [[0.0, 0.0, -1.0e4, 0.0]; 2];
        let shadow_map = crate::renderer::shadow::ShadowMap::with_dimension(&device, 64);
        if layered {
            let m = crate::renderer::shadow::spot_shadow_matrices(&lamps[0], shadow_map.spot_tile_dim());
            u.spot_view_proj[0] = m.lookup.to_cols_array_2d();
            u.shadow_params[1] = 1.0;
        }
        let lights_uniform = super::LightsUniform::new(&device);
        lights_uniform.upload_frame(&queue, lamps, if layered { &[0] } else { &[] }, false);
        let code = format!(
            "{}\n{}",
            super::wgsl_lights_block(0, 1),
            r#"
@group(1) @binding(0) var<storage, read> rays: array<vec4<f32>>;
@group(1) @binding(1) var<storage, read_write> out: array<vec4<f32>>;
@compute @workgroup_size(1)
fn seen_main(@builtin(global_invocation_id) id: vec3<u32>) {
    pixel_footprint = rays[id.x * 2u].w;
    let r = capsule_reflection(rays[id.x * 2u].xyz, normalize(rays[id.x * 2u + 1u].xyz), 0.0, vec3<f32>(3.14159265), vec4<f32>(0.0));
    out[id.x] = vec4<f32>(capsule_glow * r.a, 0.0);
}
"#
        );
        let module = device.create_shader_module(ShaderModuleDescriptor { label: None, source: ShaderSource::Wgsl(code.into()) });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &module,
            entry_point: Some("seen_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let packed: Vec<[f32; 4]> =
            rays.iter().flat_map(|(p, d, f)| [[p.x, p.y, p.z, *f], [d.x, d.y, d.z, 0.0]]).collect();
        let camera = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::bytes_of(&u),
            usage: BufferUsages::UNIFORM,
        });
        let ray_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&packed),
            usage: BufferUsages::STORAGE,
        });
        let size = (rays.len() * 16) as u64;
        let out = device.create_buffer(&BufferDescriptor {
            label: None,
            size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let read = device.create_buffer(&BufferDescriptor {
            label: None,
            size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let (_, samp) = crate::renderer::uniforms::default_probe_cube(&device);
        let cards = crate::renderer::proxy_cards::none(&device);
        let g0 = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry { binding: 0, resource: camera.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: lights_uniform.buffer().as_entire_binding() },
                BindGroupEntry { binding: 3, resource: BindingResource::Sampler(shadow_map.sampler()) },
                BindGroupEntry { binding: 4, resource: BindingResource::TextureView(shadow_map.spot_depth_view()) },
                BindGroupEntry { binding: 6, resource: BindingResource::Sampler(&samp) },
                BindGroupEntry { binding: 12, resource: BindingResource::TextureView(&cards) },
            ],
        });
        let g1 = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(1),
            entries: &[
                BindGroupEntry { binding: 0, resource: ray_buf.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: out.as_entire_binding() },
            ],
        });
        let mut enc = device.create_command_encoder(&Default::default());
        drop(enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: shadow_map.spot_depth_view(),
                depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(depth), store: wgpu::StoreOp::Store }),
                stencil_ops: None,
            }),
            ..Default::default()
        }));
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &g0, &[]);
            pass.set_bind_group(1, &g1, &[]);
            pass.dispatch_workgroups(rays.len() as u32, 1, 1);
        }
        enc.copy_buffer_to_buffer(&out, 0, &read, 0, size);
        queue.submit([enc.finish()]);
        read.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        let got: Vec<[f32; 4]> = bytemuck::cast_slice(&read.slice(..).get_mapped_range().unwrap()).to_vec();
        Some(got.iter().map(|c| [c[0], c[1], c[2]]).collect())
    }

    fn beam(at: Vec3, forward: Vec3) -> super::Light {
        super::Light {
            position: at,
            direction: forward.normalize(),
            kind: super::LightKind::Spot,
            color: crate::renderer::Color3(255, 255, 255, 255),
            intensity: 40.0,
            range: 8.0,
            cone_angle_deg: 50.0,
            inner_cone_angle_deg: 16.0,
            mask_channel: None,
            shadow_near: Some(0.02),
            source_radius: 0.0,
            in_level_bake: false,
        }
    }

    #[test]
    fn a_glass_is_hidden_where_its_own_beam_is_shadowed() {
        let glass = Vec3::new(0.0, 1.0, 0.0);
        let groups = [torch(glass, Vec3::NEG_Z)];

        let rays = [
            (Vec3::new(0.0, 1.0, -2.0), Vec3::Z, 0.002),
            (Vec3::new(0.3, 1.0, -2.0), (glass - Vec3::new(0.3, 1.0, -2.0)).normalize(), 0.002),
            (Vec3::new(1.5, 1.0, -0.26), (glass - Vec3::new(1.5, 1.0, -0.26)).normalize(), 0.002),
        ];
        let lamp = beam(glass, Vec3::NEG_Z);

        let Some(whole) = seen_past_the_beam(&groups, &rays, &[], false, 0.0) else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        let shadowed = seen_past_the_beam(&groups, &rays, &[lamp], true, 0.0).unwrap();
        let lit = seen_past_the_beam(&groups, &rays, &[lamp], true, 1.0).unwrap();
        let elsewhere = seen_past_the_beam(&groups, &rays, &[beam(glass + Vec3::X * 0.05, Vec3::NEG_Z)], true, 0.0).unwrap();
        let unlayered = seen_past_the_beam(&groups, &rays, &[lamp], false, 0.0).unwrap();
        eprintln!(
            "GLASS PAST ITS BEAM: whole {whole:?}\n  shadowed {shadowed:?}\n  lit {lit:?}\n  another lamp {elsewhere:?}\n  no layer {unlayered:?}"
        );
        for (i, what) in [(0, "head on"), (1, "off the axis")] {

            let k = beam_toward(&lamp, rays[i].0);
            assert!(k > 0.99, "{what}: inside the hot spot, or all but: {k}");
            assert!(whole[i][0] > 1.0, "{what}: the capsules give the glass a glow: {:?}", whole[i]);
            assert!(shadowed[i].iter().all(|g| *g < 1e-4), "{what}: the shadowed point still sees the glass: {:?}", shadowed[i]);
            for c in 0..3 {
                assert!(near(lit[i][c], whole[i][c] * k, 1e-3 * whole[i][c]), "{what}: a lit point sees all of it: {:?}", lit[i]);
                assert_eq!(elsewhere[i][c], whole[i][c], "{what}: a lamp elsewhere is not its beam");
                assert!(near(unlayered[i][c], whole[i][c] * k, 1e-3 * whole[i][c]), "{what}: a beam with no shadow layer: {:?}", unlayered[i]);
            }
        }

        assert!(whole[2][0] > 1.0, "with no beam known the glass shows: {:?}", whole[2]);
        assert!(shadowed[2].iter().all(|g| *g == 0.0), "outside its beam the glass sends nothing: {:?}", shadowed[2]);
        assert!(unlayered[2].iter().all(|g| *g == 0.0), "outside its beam, shadow layer or not: {:?}", unlayered[2]);
        assert_eq!(elsewhere[2], whole[2], "a lamp elsewhere is not its beam");
    }

    fn beam_toward(lamp: &super::Light, p: Vec3) -> f32 {
        let cos_p = (p - lamp.position).normalize().dot(lamp.direction.normalize());
        let (outer, inner) = lamp.cone_cosines();
        let s = ((cos_p - outer) / (inner - outer).max(1e-4)).clamp(0.0, 1.0);
        s * s * (3.0 - 2.0 * s)
    }

    #[test]
    fn a_glass_shows_in_a_mirror_only_into_its_beam() {
        let glass = Vec3::new(0.0, 1.0, 0.0);
        let groups = [torch(glass, Vec3::NEG_Z)];
        let lamp = beam(glass, Vec3::NEG_Z);

        let angles = [0.0f32, 6.0, 14.0, 20.0, 26.0, 40.0, 80.0];
        let rays: Vec<(Vec3, Vec3, f32)> = angles
            .iter()
            .map(|a| {
                let p = glass + Quat::from_rotation_y(a.to_radians()) * Vec3::NEG_Z;
                (p, (glass - p).normalize(), 0.002)
            })
            .collect();
        let Some(whole) = seen_past_the_beam(&groups, &rays, &[], false, 1.0) else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        let seen = seen_past_the_beam(&groups, &rays, &[lamp], false, 1.0).unwrap();
        for (k, a) in angles.iter().enumerate() {
            let share = beam_toward(&lamp, rays[k].0);
            eprintln!("{a:>4} deg off the beam: share {share:.4}, glow {:.3} of {:.3}", seen[k][0], whole[k][0]);
            assert!(whole[k][0] > 1.0, "{a} deg: the capsules give the glass a glow: {:?}", whole[k]);
            assert!(near(seen[k][0], whole[k][0] * share, 1e-3 * whole[k][0]), "{a} deg: {:?} against {} x {share}", seen[k], whole[k][0]);
        }
        assert!(seen[0][0] > 0.999 * whole[0][0], "on the axis the glass shows whole");
        assert!(seen[2][0] > 0.05 * whole[2][0] && seen[2][0] < 0.95 * whole[2][0], "on the ramp, part of it");
        assert!(seen[4..].iter().all(|g| g[0] == 0.0), "past the spill (25 degrees), none: {:?}", &seen[4..]);
    }

    #[test]
    fn a_capsule_keeps_what_it_is_past_one_left_out() {
        let mut g = torch(Vec3::ZERO, Vec3::NEG_Z);
        g.capsules.insert(0, (Vec3::splat(f32::NAN), Vec3::ZERO, 0.1));
        g.surfaces.insert(0, -0.5);
        let up = CapsuleUpload::from_groups(&[g]);
        assert_eq!(up.groups[1][3], 2.0, "two capsules kept");
        assert_eq!((up.capsules[1][3], up.capsules[3][3]), (-ALBEDO, DRIVE));

        let t = CapsuleUpload::from_groups(&[trunk(Vec3::ZERO)]);
        assert_eq!(t.capsules[1][3], 0.0);
    }
}

#[cfg(test)]
mod surface_relit_tests {
    use glam::Vec3;
    use wgpu::util::DeviceExt;
    use wgpu::{
        BindGroupDescriptor, BindGroupEntry, BindingResource, BufferDescriptor, BufferUsages, ShaderModuleDescriptor,
        ShaderSource,
    };

    use super::{LitSurface, LightsUniform};
    use crate::renderer::uniforms::Uniforms;

    const ALBEDO: Vec3 = Vec3::new(0.5, 0.4, 0.3);

    fn beam(at: Vec3, forward: Vec3) -> super::Light {
        super::Light {
            position: at,
            direction: forward.normalize(),
            kind: super::LightKind::Spot,
            color: crate::renderer::Color3(255, 255, 255, 255),
            intensity: 40.0,
            range: 8.0,
            cone_angle_deg: 50.0,
            inner_cone_angle_deg: 16.0,
            mask_channel: None,
            shadow_near: Some(0.02),
            source_radius: 0.0,
            in_level_bake: false,
        }
    }

    fn wall() -> LitSurface {
        LitSurface {
            normal: Vec3::Z,
            offset: -3.0,
            albedo: ALBEDO,
            lens: Vec3::new(0.0, 1.0, 0.0),
            forward: Vec3::NEG_Z,
            right: Vec3::X,
            tan_half: 26.25f32.to_radians().tan(),
            near: 0.0,
            far: 0.0,
        }
    }

    fn relit_at(
        lamps: &[super::Light],
        layered: bool,
        depth: f32,
        surfaces: &[LitSurface],
        hits: &[(Vec3, Vec3)],
    ) -> Option<Vec<Vec3>> {
        let (device, queue) = crate::renderer::terrain_pipeline::tests::headless_gpu()?;
        let mut u: Uniforms = bytemuck::Zeroable::zeroed();
        let shadow_map = crate::renderer::shadow::ShadowMap::with_dimension(&device, 64);
        if layered {
            let m = crate::renderer::shadow::spot_shadow_matrices(&lamps[0], shadow_map.spot_tile_dim());
            u.spot_view_proj[0] = m.lookup.to_cols_array_2d();
            u.shadow_params[1] = 1.0;
        }
        let atlas = crate::renderer::proxy_cards::atlas_with_characters(&device, &queue, &[], 1);

        let entry = |binding: u32, ty: wgpu::BindingType| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty,
            count: None,
        };
        let uniform = wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None };
        let pool_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                entry(0, uniform),
                entry(1, uniform),
                entry(3, wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison)),
                entry(
                    4,
                    wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                ),
            ],
        });
        let pools = crate::renderer::pool_cards::PoolCards::new(&device, &pool_layout, atlas.resolution);
        let mips = crate::renderer::brush_pipeline::probe_pass::MirrorMips::new(&device);
        let lights_uniform = LightsUniform::new(&device);
        lights_uniform.set_lit_surfaces(surfaces);

        assert!(!lights_uniform.reads_pool_maps());
        lights_uniform.set_pool_row(Some(pools.first_row(atlas.pool_row)));
        assert_eq!(lights_uniform.reads_pool_maps(), surfaces.iter().any(|s| s.tan_half > 0.0));
        lights_uniform.upload_frame(&queue, lamps, if layered { &[0] } else { &[] }, false);
        let code = format!(
            "{}\n{}",
            super::wgsl_lights_block(0, 1),
            r#"
@group(1) @binding(0) var<storage, read> rays: array<vec4<f32>>;
@group(1) @binding(1) var<storage, read_write> out: array<vec4<f32>>;
@compute @workgroup_size(1)
fn relit_main(@builtin(global_invocation_id) id: vec3<u32>) {
    pixel_footprint = 0.0005;
    probe_eye_distance = 2.0;
    out[id.x] = vec4<f32>(probe_surface_relit(rays[id.x * 2u].xyz, rays[id.x * 2u + 1u].xyz, 0.05, 2.0), pixel_footprint);
}
"#
        );
        let module = device.create_shader_module(ShaderModuleDescriptor { label: None, source: ShaderSource::Wgsl(code.into()) });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &module,
            entry_point: Some("relit_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let packed: Vec<[f32; 4]> = hits.iter().flat_map(|(h, d)| [[h.x, h.y, h.z, 0.0], [d.x, d.y, d.z, 0.0]]).collect();
        let camera = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::bytes_of(&u),
            usage: BufferUsages::UNIFORM,
        });
        let ray_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&packed),
            usage: BufferUsages::STORAGE,
        });
        let size = (hits.len() * 16) as u64;
        let out = device.create_buffer(&BufferDescriptor {
            label: None,
            size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let read = device.create_buffer(&BufferDescriptor {
            label: None,
            size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let pool_group = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pool_layout,
            entries: &[
                BindGroupEntry { binding: 0, resource: camera.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: lights_uniform.buffer().as_entire_binding() },
                BindGroupEntry { binding: 3, resource: BindingResource::Sampler(shadow_map.sampler()) },
                BindGroupEntry { binding: 4, resource: BindingResource::TextureView(shadow_map.spot_depth_view()) },
            ],
        });
        let linear = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });

        let g0 = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry { binding: 0, resource: camera.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: lights_uniform.buffer().as_entire_binding() },
                BindGroupEntry { binding: 6, resource: BindingResource::Sampler(&linear) },
                BindGroupEntry { binding: 12, resource: BindingResource::TextureView(&atlas.view) },
            ],
        });
        let g1 = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(1),
            entries: &[
                BindGroupEntry { binding: 0, resource: ray_buf.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: out.as_entire_binding() },
            ],
        });
        let mut enc = device.create_command_encoder(&Default::default());
        drop(enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: shadow_map.spot_depth_view(),
                depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(depth), store: wgpu::StoreOp::Store }),
                stencil_ops: None,
            }),
            ..Default::default()
        }));
        if !lights_uniform.lit_surfaces().is_empty() {
            pools.record(&device, &mut enc, &pool_group, &mips, &atlas.texture, pools.first_row(atlas.pool_row), None);
        }
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &g0, &[]);
            pass.set_bind_group(1, &g1, &[]);
            pass.dispatch_workgroups(hits.len() as u32, 1, 1);
        }
        enc.copy_buffer_to_buffer(&out, 0, &read, 0, size);
        queue.submit([enc.finish()]);
        read.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        let _ = device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        let got: Vec<[f32; 4]> = bytemuck::cast_slice(&read.slice(..).get_mapped_range().unwrap()).to_vec();
        for g in &got {
            assert_eq!(g[3], 0.0005, "the reflecting pixel's footprint is left as it was");
        }
        Some(got.iter().map(|g| Vec3::new(g[0], g[1], g[2])).collect())
    }

    fn expected_scale() -> f32 {
        let window = (1.0 - (3.0f32 / 8.0).powi(4)).powi(2);
        40.0 / 9.0 * window
    }

    #[test]
    fn a_reflected_point_on_a_lit_wall_takes_the_beams_light_and_its_shadow() {
        let glass = Vec3::new(0.0, 1.0, 0.0);
        let lamp = beam(glass, Vec3::NEG_Z);

        let ray = Vec3::new(0.0, 0.5, -1.0).normalize();
        let hits = [

            (Vec3::new(0.0, 1.0, -3.0), ray),

            (Vec3::new(0.0, 1.0, -2.9), ray),

            (Vec3::new(1.9, 1.0, -3.0), ray),

            (Vec3::new(0.5, 1.0, -3.0), ray),

            (Vec3::new(1.463, 1.0, -3.0), ray),
        ];
        let Some(lit) = relit_at(&[lamp], true, 1.0, &[wall()], &hits) else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        let shadowed = relit_at(&[lamp], true, 0.0, &[wall()], &hits).unwrap();
        let no_surface = relit_at(&[lamp], true, 1.0, &[], &hits).unwrap();

        let narrow = relit_at(&[lamp], true, 1.0, &[LitSurface { tan_half: 0.1, ..wall() }], &hits).unwrap();
        let baked = relit_at(&[super::Light { in_level_bake: true, ..lamp }], true, 1.0, &[wall()], &hits).unwrap();

        let decoy = |k: f32| LitSurface { normal: Vec3::Y, offset: 50.0 + k, ..wall() };
        let fifth = relit_at(&[lamp], true, 1.0, &[decoy(0.0), decoy(1.0), decoy(2.0), decoy(3.0), wall()], &hits).unwrap();
        assert!((fifth[0] - lit[0]).abs().max_element() < 1e-3 * expected_scale(), "the fifth map, read where it was made: {:?}", fifth[0]);
        eprintln!(
            "SURFACE RELIT: lit {lit:?}\n  shadowed {shadowed:?}\n  no surface {no_surface:?}\n  narrow {narrow:?}\n  a lamp the bake saw {baked:?}"
        );

        let expected = ALBEDO * expected_scale();
        assert!((lit[0] - expected).abs().max_element() < 0.01 * expected.max_element(), "on the axis: {:?} vs {expected:?}", lit[0]);
        for (i, why) in [(1, "off the wall's plane"), (2, "outside the cone and the map"), (4, "outside the cone")] {
            assert!(lit[i].max_element() < 1e-3 * expected.max_element(), "{why}: {:?}", lit[i]);
        }
        assert!(lit[3].max_element() > 0.5 * expected.max_element(), "inside the cone: {:?}", lit[3]);
        assert!(narrow[3].max_element() < 1e-4, "past the map's reach: {:?}", narrow[3]);
        assert!((narrow[0] - lit[0]).abs().max_element() < 0.01 * expected.max_element(), "within its reach, the same");
        assert!(shadowed[0].max_element() < 1e-4, "a hand's shadow on the wall is in its reflection: {:?}", shadowed[0]);
        assert!(no_surface[0].max_element() < 1e-4, "no lit surface, no light: {:?}", no_surface[0]);
        assert!(baked[0].max_element() < 1e-4, "a lamp the bake saw is in the photographs already: {:?}", baked[0]);
    }

    #[test]
    fn a_lit_planes_far_reach_does_not_claim_the_wall_it_crosses() {
        let glass = Vec3::new(0.0, 1.0, 0.0);
        let lamp = beam(glass, Vec3::NEG_Z);

        let side = LitSurface { normal: Vec3::NEG_X, offset: -0.3, near: 0.2, far: 1.5, ..wall() };
        let far_wall = LitSurface { near: 2.0, far: 4.5, ..wall() };
        let from_floor = Vec3::new(0.0, 0.5, -1.0).normalize();
        let hits = [

            (Vec3::new(0.3, 1.0, -3.0), from_floor),

            (Vec3::new(0.0, 1.0, -3.0), from_floor),

            (Vec3::new(0.3, 1.0, -1.0), Vec3::new(1.0, 0.0, -0.2).normalize()),
        ];
        let Some(bounded) = relit_at(&[lamp], true, 1.0, &[side, far_wall], &hits) else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        let wall_alone = relit_at(&[lamp], true, 1.0, &[far_wall], &hits).unwrap();
        let unbounded = relit_at(&[lamp], true, 1.0, &[LitSurface { far: 0.0, ..side }, LitSurface { far: 0.0, ..far_wall }], &hits).unwrap();
        eprintln!("CROSSING: bounded {bounded:?}\n  wall alone {wall_alone:?}\n  unbounded {unbounded:?}");
        let full = wall_alone[0].max_element();
        assert!(full > 0.5 * ALBEDO.max_element() * expected_scale(), "the wall is lit there: {:?}", wall_alone[0]);

        assert!(unbounded[0].max_element() < 0.3 * full, "unbounded, the side claims the wall: {:?}", unbounded[0]);
        assert!((bounded[0] - wall_alone[0]).abs().max_element() < 0.01 * full, "the wall's point is the wall's: {:?}", bounded[0]);
        assert!((bounded[1] - wall_alone[1]).abs().max_element() < 0.01 * full, "the axis as before: {:?}", bounded[1]);
        assert!(bounded[2].max_element() > 0.05 * full, "the side keeps its own pool: {:?}", bounded[2]);
        assert!((bounded[2] - unbounded[2]).abs().max_element() < 0.01 * bounded[2].max_element(), "and reads it as before");
    }
}

#[cfg(test)]
mod tent_tests {
    use wgpu::util::DeviceExt;
    use wgpu::*;

    fn tents(f16: bool) -> Option<Vec<[f32; 4]>> {
        let instance = Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&RequestAdapterOptions::default())).ok()?;
        if f16 && !adapter.features().contains(Features::SHADER_F16) {
            return None;
        }
        let (device, queue) = pollster::block_on(adapter.request_device(&DeviceDescriptor {
            required_features: if f16 { Features::SHADER_F16 } else { Features::empty() },
            required_limits: crate::renderer::uniforms::scene_limits(Limits::default()),
            ..Default::default()
        }))
        .ok()?;
        let block = crate::renderer::shader_precision::with_half_precision(super::wgsl_lights_block(0, 1), f16);
        let samp_binding: u32 = {
            let at = block.find(" var shadow_samp:").expect("the lights block declares shadow_samp");
            let head = &block[..at];
            let open = head.rfind("@binding(").unwrap() + "@binding(".len();
            head[open..].split(')').next().unwrap().parse().unwrap()
        };
        let code = format!(
            "{block}\n{}",
            r#"
@group(1) @binding(0) var test_depth: texture_depth_2d;
@group(1) @binding(1) var<storage, read> pts: array<vec4<f32>>;
@group(1) @binding(2) var<storage, read_write> out: array<vec4<f32>>;
@compute @workgroup_size(64)
fn tent_main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= arrayLength(&pts)) { return; }
    let p = pts[id.x];
    let tile = vec2<f32>(p.w, 0.0);
    let grid = vec2<f32>(2.0, 1.0);
    out[id.x] = vec4<f32>(
        pcf_tile_tent_at(test_depth, tile, grid, p.xyz),
        pcf_tile_tent_unrolled_at(test_depth, tile, grid, p.xyz),
        pcf_tile_tent_lean_at(test_depth, tile, grid, p.xyz),
        pcf_tile_tent_half_at(test_depth, tile, grid, p.xyz),
    );
}
"#
        );
        let module = device.create_shader_module(ShaderModuleDescriptor { label: None, source: ShaderSource::Wgsl(code.into()) });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &module,
            entry_point: Some("tent_main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let (w, h) = (64u32, 32u32);
        let mut seed = 0x2545_f491_u32;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed as f32 / u32::MAX as f32
        };
        let texels: Vec<u16> = (0..w * h).map(|_| (next() * 65535.0) as u16).collect();
        let depth = device.create_texture_with_data(
            &queue,
            &TextureDescriptor {
                label: None,
                size: Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::Depth16Unorm,
                usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
                view_formats: &[],
            },
            util::TextureDataOrder::LayerMajor,
            bytemuck::cast_slice(&texels),
        );
        let sampler = device.create_sampler(&SamplerDescriptor {
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            compare: Some(CompareFunction::LessEqual),
            ..Default::default()
        });

        let pts: Vec<[f32; 4]> = (0..4096).map(|i| [next(), next(), next(), (i % 2) as f32]).collect();
        let pts_buf = device.create_buffer_init(&util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&pts),
            usage: BufferUsages::STORAGE,
        });
        let size = (pts.len() * 16) as u64;
        let out = device.create_buffer(&BufferDescriptor {
            label: None,
            size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let read = device.create_buffer(&BufferDescriptor {
            label: None,
            size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let view = depth.create_view(&Default::default());
        let g0 = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[BindGroupEntry { binding: samp_binding, resource: BindingResource::Sampler(&sampler) }],
        });
        let g1 = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(1),
            entries: &[
                BindGroupEntry { binding: 0, resource: BindingResource::TextureView(&view) },
                BindGroupEntry { binding: 1, resource: pts_buf.as_entire_binding() },
                BindGroupEntry { binding: 2, resource: out.as_entire_binding() },
            ],
        });
        let mut enc = device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &g0, &[]);
            pass.set_bind_group(1, &g1, &[]);
            pass.dispatch_workgroups((pts.len() as u32).div_ceil(64), 1, 1);
        }
        enc.copy_buffer_to_buffer(&out, 0, &read, 0, size);
        queue.submit([enc.finish()]);
        read.slice(..).map_async(MapMode::Read, |_| {});
        let _ = device.poll(PollType::Wait { submission_index: None, timeout: None });
        let got: Vec<[f32; 4]> = bytemuck::cast_slice(&read.slice(..).get_mapped_range().unwrap()).to_vec();
        Some(got)
    }

    fn spread(got: &[[f32; 4]], k: usize) -> (f32, f32) {
        let d: Vec<f32> = got.iter().map(|r| (r[k] - r[0]).abs()).collect();
        (d.iter().cloned().fold(0.0, f32::max), d.iter().sum::<f32>() / d.len() as f32)
    }

    #[test]
    fn every_tent_form_gives_the_same_shadow() {
        let Some(got) = tents(false) else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        let lit = got.iter().filter(|r| r[0] > 0.0 && r[0] < 1.0).count();
        assert!(lit > got.len() / 4, "the points must fall on partial shadow: {lit} of {}", got.len());
        for (k, name, tol) in [(1, "unrolled", 1e-5), (2, "lean", 2e-3), (3, "half (at f32)", 1e-5)] {
            let (max, mean) = spread(&got, k);
            eprintln!("{name}: max {max:.2e}, mean {mean:.2e}");
            assert!(max <= tol, "{name} differs from the shipped tent by {max}");
        }
    }

    #[test]
    fn the_half_precision_tent_holds_at_f16() {
        let Some(got) = tents(true) else {
            eprintln!("no f16 adapter; skipping");
            return;
        };
        let (max, mean) = spread(&got, 3);
        eprintln!("half at f16: max {max:.2e}, mean {mean:.2e}");
        assert!(max <= 6e-3 && mean <= 1e-3, "half: max {max}, mean {mean}");
        let (max, mean) = spread(&got, 2);
        eprintln!("lean at f16 (f32 maths): max {max:.2e}, mean {mean:.2e}");
        assert!(max <= 2e-3, "lean: max {max}");
    }
}
