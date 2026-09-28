use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use wgpu::*;

use super::Color3;

/// Matches the fixed-size `array<Light, MAX_LIGHTS>` declared in the mesh/solid
/// fragment shaders — keep these in sync.
pub const MAX_LIGHTS: usize = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LightKind {
    Point,
    Spot,
    /// Infinitely-distant parallel light (sun). `position`/`range` are ignored;
    /// the beam travels along `direction`, so there is no distance attenuation.
    Directional,
}

/// How much a light matters to the viewer, for choosing which ones fit.
///
/// PRECONDITION: lights are in the player's frame, where the viewer sits at the
/// origin. Everything handed to the renderer already is -- see
/// `to_space_soup_light` -- so "distance from the origin" is distance from the
/// player.
///
/// A directional light outranks everything: it is the sun, it has no position
/// to be far from, and dropping it changes every surface in the level at once.
///
/// This is a heuristic about the VIEWER, not about what is on screen. A lamp
/// behind the player lighting a wall they can see still scores low. That is a
/// real limitation and it is still a large improvement on the previous rule,
/// which was the order the lights happened to appear in the scene file.
pub fn influence_score(light: &Light) -> f32 {
    if light.kind == LightKind::Directional {
        return f32::INFINITY;
    }
    let d2 = light.position.length_squared();
    // Matches the shape of the shader's falloff closely enough to rank by, and
    // the `+ 1` keeps a light at the player's own position finite.
    light.intensity.max(0.0) / (1.0 + d2)
}

/// HOW MUCH MORE INFLUENTIAL A LIGHT MUST BE TO TAKE A SHADOW SLOT.
///
/// Influence goes as `1 / distance²`, so 1.5 is about 22% closer. Small enough
/// that walking decisively toward a lamp still hands it a slot, large enough
/// that the set does not flip while you stand between two of them.
pub const SHADOW_SLOT_MARGIN: f32 = 1.5;

/// Which spots keep the shadow slots this frame, given who held them last.
///
/// WHY THIS IS NOT JUST "THE BEST `max`". `influence_score` is measured from
/// the PLAYER, so ranking afresh every frame means the set changes as they
/// walk. With more spots in a room than `MAX_SPOT_SHADOWS`, shadows appear and
/// disappear with movement: on the headset (2026-09-18) one side of the
/// avatar's hand was unshadowed when it should not have been, a shadow arrived
/// as the hand approached a lamp, and the wall spotlight in the back corner
/// cast none at all.
///
/// Each frame's choice was individually correct -- it really was the best four
/// -- and the defect existed only ACROSS frames, which is why nothing errored
/// and no single screenshot showed it.
///
/// So an incumbent keeps its slot unless a challenger beats it by
/// `SHADOW_SLOT_MARGIN`. Free slots are still filled by the best available
/// immediately: hysteresis is about not CHURNING, not about being slow.
///
/// `scores` is `(stable id, influence)` for every spot. The id must mean the
/// same light next frame -- the index into the caller's own light list, not
/// into anything this module reorders.
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

    // Incumbents that still exist at all, best first. A light that has left the
    // scene simply frees its slot.
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

    // Free slots go to the best challengers outright -- no margin, because
    // there is nothing to displace.
    let mut rest = Vec::new();
    for c in challengers.drain(..) {
        if out.len() < max {
            out.push(c);
        } else {
            rest.push(c);
        }
    }

    // Only now does anything get evicted, and only by a clear margin.
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

    // NO FINAL SORT, AND THAT IS THE POINT.
    //
    // A sort here re-ranked the SURVIVORS by their current score every frame.
    // Membership could be perfectly stable and the ORDER would still churn as
    // the player moved -- and this vector's index IS the atlas tile the light
    // renders its shadow into. A light that keeps its slot but changes tile has
    // its shadow map re-rasterised somewhere else, and the shader's lookup
    // switches tiles with it.
    //
    // Measured on the headset (2026-09-19): in one 0.6 s window the set
    // [0,3,1,4] became [0,3,1,2], then [0,3,2,1], then [0,2,3,1] -- one real
    // eviction and THREE pure reorders. That is four shadow-atlas rewrites in
    // just over half a second for a room nobody was changing, and it is what
    // the fixture's shadow was doing when it looked like it was vibrating.
    //
    // The hysteresis above was built to keep the SET stable and does. Keeping
    // the SLOT stable is a separate promise and nothing was making it.
    // `position_stable_slots` below keeps each surviving incumbent at the index
    // it already held, so a tile is rewritten only when its occupant actually
    // changes.
    position_stable_slots(incumbents, out)
}

/// Put each surviving incumbent back in the slot index it already occupied.
///
/// `chosen` is the winning set in ranked order and says nothing about where
/// anything should live. This maps it onto the previous frame's layout: an
/// incumbent that survived keeps its index, and anything new fills the lowest
/// index that was vacated. The result is the same SET in an order that changes
/// only when the set does.
fn position_stable_slots(incumbents: &[usize], chosen: Vec<(usize, f32)>) -> Vec<usize> {
    let ids: Vec<usize> = chosen.into_iter().map(|(id, _)| id).collect();
    let mut slots: Vec<Option<usize>> = vec![None; ids.len()];

    // Survivors first, each back where it was. An incumbent whose old index is
    // past the end of the new, shorter set falls through to the newcomer pass
    // rather than being dropped.
    for (old_slot, id) in incumbents.iter().enumerate() {
        if old_slot < slots.len() && ids.contains(id) && !slots.contains(&Some(*id)) {
            slots[old_slot] = Some(*id);
        }
    }
    // Then everything the set gained, into whatever is left, best first.
    // The free list is collected BEFORE the fill so the borrow ends first.
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

    /// THE REGRESSION THIS EXISTS FOR.
    ///
    /// The selector used to sort its winners by score on the way out. The set
    /// stayed stable and the ORDER did not -- and the index is the shadow
    /// ATLAS TILE, so a light that kept its slot still had its shadow map
    /// rewritten into a different tile whenever the ranking shifted. On the
    /// headset that was four atlas rewrites in 0.6 s with nothing moving but
    /// the player's head, and it read as the fixture's shadow vibrating.
    #[test]
    fn the_same_lights_keep_the_same_slots_when_only_their_ranking_moves() {
        // Four lights, comfortably inside the budget, so membership is fixed.
        let first = spot_shadow_slots(
            &[(0, 10.0), (1, 9.0), (2, 8.0), (3, 7.0)],
            &[],
            MAX,
            MARGIN,
        );
        assert_eq!(first.len(), 4);

        // Now completely reverse their scores. Every light is still in, so
        // every slot must stay exactly where it was.
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

    /// A slot changes only when its occupant does, and only that slot.
    #[test]
    fn an_eviction_disturbs_one_slot_and_leaves_the_others_alone() {
        let before = spot_shadow_slots(
            &[(0, 10.0), (1, 9.0), (2, 8.0), (3, 7.0)],
            &[],
            MAX,
            MARGIN,
        );
        // A newcomer clearly past the margin displaces the weakest (id 3).
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

    /// Hysteresis still holds: a marginal challenger does not get in at all.
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

/// The lights that fit in `max`, most influential first.
///
/// The budget used to be filled in SCENE ORDER, so a level with nine lamps
/// dropped whichever was authored last -- possibly the one two metres from the
/// player's face, while a dim one across the map kept its slot. Nothing errored
/// and nothing said so; the lamp simply stopped working.
///
/// Ordering is stable within equal scores, so a scene that fits inside the
/// budget is passed through untouched and cannot be reshuffled frame to frame.
/// How far outside its probe box a fragment still counts as inside, in metres.
///
/// 5 cm at full resolution: surfaces sit exactly on their room's box, and with
/// MSAA a pixel on a polygon's edge is shaded at its CENTRE, which can land
/// just outside. A fragment that falls out takes a different room's probe --
/// the dotted light line along every seam.
///
/// BUT THE OVERSHOOT IS MEASURED IN PIXELS, NOT METRES. At `RENDER_SCALE` 0.7 a
/// pixel spans 1/0.7 = 1.43x more world, so 5 cm stopped covering it and the
/// seam came back along the ceiling/wall junction at the front of the marble
/// hall -- and nowhere the probe boxes happen to sit differently (headset,
/// 2026-09-19). Dividing by the scale keeps the margin the same size in pixels,
/// which is what it always was.
///
/// The only thing a too-WIDE margin can break is pulling a neighbouring room's
/// surfaces in, and at 0.7 this is 7.1 cm against a thinnest wall of 0.3 m.
/// How far a doorway's handover reaches past each face of the opening, in
/// metres. The blend is centred on the doorway, so a surface right at a room's
/// wall beside the opening already carries some of the next room; half a metre
/// in, it is back to its own photograph alone. Wider spreads a photograph taken
/// in the other room -- wrong for most directions from here -- further into
/// this one.
pub const PROBE_PORTAL_FADE: f32 = 0.5;

/// How far past a doorway's EDGES the handover fades out, in metres, so the
/// wall around a door frame does not show the opening's outline.
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

/// The sky's sun as a frame light, in the PLAYER's frame, or `None` when the
/// sky has no sun or the scene authored a directional light of its own.
///
/// ONE SUN: an authored directional replaces the sky's rather than joining it
/// -- the same rule the baker's `scene_sky_lighting` applies, so the two can
/// never disagree about how many suns a level has.
///
/// `SkySun::light_rgb` is irradiance / pi, which is what this renderer's
/// `color x intensity` already means for a lamp (see `SkyIrradiance::evaluate`),
/// so it goes in unscaled: the light that left the ambient comes back exactly.
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
    // Colour3 is sRGB bytes and `to_linear` decodes them, so encode the
    // linear ratio the same way rather than storing it as if it were sRGB.
    let srgb = |v: f32| {
        let v = (v / peak).clamp(0.0, 1.0);
        let e = if v <= 0.003_130_8 { v * 12.92 } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 };
        (e * 255.0).round() as u8
    };
    Some(Light {
        mask_channel: None,
        position: Vec3::ZERO,
        // The way the light TRAVELS -- away from the sun.
        direction: world_to_player * -Vec3::from(sun.direction),
        kind: LightKind::Directional,
        color: Color3(srgb(sun.light_rgb[0]), srgb(sun.light_rgb[1]), srgb(sun.light_rgb[2]), 255),
        intensity: peak,
        range: 0.0,
        cone_angle_deg: 0.0,
        inner_cone_angle_deg: 0.0,
    })
}

/// The moving-objects sun map's world-space matrix, for a player whose head is
/// at `head_world`.
///
/// Centred a little under the head, where a standing body's mass is, and
/// SNAPPED so that the world's origin lands on a texel corner. Without the
/// snap, every centimetre the player moves shifts the whole texel grid under
/// the shadows by a fraction of a texel, and each moving shadow's edge
/// shimmers as it is re-rasterised onto a different grid -- the classic
/// crawling cascade.
pub fn dynamic_sun_matrix(world_dir: Vec3, head_world: Vec3) -> glam::Mat4 {
    use crate::renderer::shadow::{directional_light_matrix, SUN_DYNAMIC_DIM, SUN_DYNAMIC_RADIUS};
    let centre = head_world - Vec3::Y * 0.9;
    let m = directional_light_matrix(world_dir, centre, SUN_DYNAMIC_RADIUS);
    let half = SUN_DYNAMIC_DIM as f32 * 0.5;
    let o = m.project_point3(Vec3::ZERO);
    let snap = |v: f32| (v * half).round() / half - v;
    glam::Mat4::from_translation(Vec3::new(snap(o.x), snap(o.y), 0.0)) * m
}

/// A sun shadow map drawn once, and what it was drawn from.
pub struct StaticSunShadow {
    /// Vertex and index counts of the brushes and the ground, and the sun's
    /// world direction as bits: when any of them changes the map is redrawn.
    /// A brush shot away changes the counts; nothing else moves a static map.
    pub signature: ((usize, usize), (usize, usize), [u32; 3]),
    /// World space -> the sun's clip space. Times `player_to_world` each frame.
    pub world_view_proj: glam::Mat4,
}

/// How far past the level's own bounds the static sun map reaches, as a
/// fraction of the level's half-diagonal. A wall's shadow falls OUTSIDE the
/// wall -- 4 m of hall throws a 3.5 m shadow at this sky's 48-degree sun -- and
/// the ground there is exactly what this map exists to shade.
pub const STATIC_SUN_MARGIN: f32 = 0.5;

/// The static sun map's world-space matrix, fitted to the level.
///
/// Fitted to the BRUSHES when there are any -- the level is what casts and
/// what the player is near -- and to the ground otherwise. Positions arrive in
/// the player's frame, as all geometry does, and `player_to_world` puts them
/// back where they stand still.
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

/// The same choice, as indices into `lights`.
///
/// Exists because a caller that needs to remember something about a light
/// ACROSS FRAMES needs a name for it that survives this reordering, and the
/// only stable one is its position in the list it came in on. The shadow slots
/// need exactly that -- see [`spot_shadow_slots`].
pub fn rank_for_budget_indices(lights: &[Light], max: usize) -> Vec<usize> {
    if lights.len() <= max {
        return (0..lights.len()).collect();
    }
    let mut idx: Vec<usize> = (0..lights.len()).collect();
    idx.sort_by(|&a, &b| {
        influence_score(&lights[b])
            .partial_cmp(&influence_score(&lights[a]))
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    idx.truncate(max);
    idx
}

/// A light to be drawn this frame, in whatever world space `cuboids`/`meshes`
/// are already in for this call — the renderer has no opinion on game logic,
/// it just shades with what it's handed.
#[derive(Debug, Clone, Copy)]
pub struct Light {
    pub position: Vec3,
    /// Aim direction for `Spot` lights; ignored for `Point`.
    pub direction: Vec3,
    pub kind: LightKind,
    pub color: Color3,
    pub intensity: f32,
    pub range: f32,
    pub cone_angle_deg: f32,
    /// Full angle of the bright core. Zero means the beam fades from the axis
    /// to the rim with no edge, which is what a spot did before this existed.
    pub inner_cone_angle_deg: f32,
    /// A STATIONARY lamp's channel of the level's baked shadow masks: its
    /// direct light is shaded here every frame, its shadows come from the
    /// bake. `None` for every other light. See `stationary_visibility` in the
    /// lights block and `space_soup_engine::stationary`.
    pub mask_channel: Option<u8>,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GpuLight {
    position: [f32; 4],
    direction: [f32; 4],
    color_intensity: [f32; 4],
    /// x = range, y = cos(outer half-angle), z = kind (0 = point, 1 = spot),
    /// w = which spot shadow layer this light casts into, or -1 for none.
    ///
    /// `direction.w` carries cos(inner half-angle) -- it was padding, and the
    /// inner angle is measured against that very direction.
    ///
    /// On the LIGHT rather than in the camera uniform, because it is a property
    /// of the light. The camera used to carry a single "flashlight index",
    /// which by construction could only ever name one shadow-casting spot.
    params: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GpuLights {
    /// x = active light count; y = how many of them, from the front, are LIVE
    /// (the rest are baked into the level's lightmaps and shaded only by
    /// surfaces without one -- see `receiver_skips_baked`); z = 1 turns the
    /// shader's light culling OFF, to measure it (see `light_culling` in the
    /// shader); w pads the field to the array's 16-byte stride.
    count: [u32; 4],
    lights: [GpuLight; MAX_LIGHTS],
}

/// Owns just the GPU buffer — the bind group itself lives alongside the
/// camera uniform in `uniforms::UniformBuffer` (one shared group, two
/// bindings), since wgpu's default `max_bind_groups` limit of 4 leaves no
/// room for lights as their own group once model/texture/joint groups are
/// already spoken for on the skinned mesh pipeline.
pub struct LightsUniform {
    buffer: Buffer,
    /// Whether the shader may skip a lamp that cannot reach a pixel before any
    /// of its maths. Lossless; off only to measure what it saves (the
    /// `light_culling` lever). A `Cell` so the frame can set it while its draw
    /// lists still borrow the renderer.
    culling: std::cell::Cell<bool>,
}

impl LightsUniform {
    pub fn new(device: &Device) -> Self {
        let buffer = device.create_buffer(&BufferDescriptor {
            label: Some("lights_uniform"),
            size: std::mem::size_of::<GpuLights>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self { buffer, culling: std::cell::Cell::new(true) }
    }

    /// See `culling`. Takes effect with the next upload.
    pub fn set_culling(&self, on: bool) {
        self.culling.set(on);
    }

    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    pub fn upload(&self, queue: &Queue, lights: &[Light]) {
        self.upload_with_shadow_layers(queue, lights, &[]);
    }

    /// The same, telling each light which spot shadow layer it casts into.
    ///
    /// `spot_layers` lists light indices in layer order: the light at
    /// `spot_layers[0]` uses layer 0, and so on. A light not in the list gets
    /// -1 and is shaded unshadowed -- the honest failure for a level with more
    /// lamps than the shadow budget, since it still lights the room.
    pub fn upload_with_shadow_layers(
        &self,
        queue: &Queue,
        lights: &[Light],
        spot_layers: &[usize],
    ) {
        self.upload_frame(queue, lights, spot_layers, false);
    }

    /// The same, saying whether the directional light is the SKY's sun, whose
    /// level shadow is baked (`receiver_sun_mask`) or drawn once, and whose
    /// shadow of moving things comes from its own small map. See `Sky::sun`
    /// and `sun_visibility` in the shader.
    pub fn upload_frame(
        &self,
        queue: &Queue,
        lights: &[Light],
        spot_layers: &[usize],
        sun_is_baked: bool,
    ) {
        self.upload_frame_split(queue, lights, lights.len(), spot_layers, sun_is_baked);
    }

    /// As [`Self::upload_frame`], with only the first `live` lights shaded on
    /// lightmapped surfaces; the rest are baked lamps kept for what has no
    /// lightmap. See `receiver_skips_baked` in the shader.
    pub fn upload_frame_split(
        &self,
        queue: &Queue,
        lights: &[Light],
        live: usize,
        spot_layers: &[usize],
        sun_is_baked: bool,
    ) {
        let gpu = pack_lights(lights, live, spot_layers, sun_is_baked, self.culling.get());
        queue.write_buffer(&self.buffer, 0, bytemuck::bytes_of(&gpu));
    }
}

/// The GPU's copy of a frame's lights. See `GpuLights::count`.
fn pack_lights(lights: &[Light], live: usize, spot_layers: &[usize], sun_is_baked: bool, culling: bool) -> GpuLights {
    {
        let count = lights.len().min(MAX_LIGHTS);
        let mut gpu = GpuLights {
            count: [count as u32, live.min(count) as u32, u32::from(!culling), 0],
            lights: [GpuLight::zeroed(); MAX_LIGHTS],
        };
        for (slot, l) in gpu.lights.iter_mut().zip(lights.iter().take(MAX_LIGHTS)) {
            let color = l.color.to_linear();
            let cos_outer = (l.cone_angle_deg.to_radians() * 0.5).cos();
            // Clamped above cos_outer so an inner angle authored wider than the
            // outer one cannot invert the gradient (or divide by ~zero) and
            // turn the beam inside out.
            let cos_inner = (l.inner_cone_angle_deg.to_radians() * 0.5)
                .cos()
                .max(cos_outer + 1e-4);
            // Kind tag packed into params.z — must match the branch constants
            // in `wgsl_lights_block`: 0 = point, 1 = spot, 2 = directional.
            let kind = match l.kind {
                LightKind::Point => 0.0,
                LightKind::Spot => 1.0,
                LightKind::Directional => 2.0,
            };
            let baked = sun_is_baked && l.kind == LightKind::Directional;
            // position.w = 1 marks the sky's sun -- see `sun_visibility`; 2 + c
            // a stationary lamp shadowed by mask channel c -- see
            // `stationary_visibility`.
            let marker = match l.mask_channel {
                Some(c) => 2.0 + c as f32,
                None if baked => 1.0,
                None => 0.0,
            };
            *slot = GpuLight {
                position: [l.position.x, l.position.y, l.position.z, marker],
                direction: [l.direction.x, l.direction.y, l.direction.z, cos_inner],
                color_intensity: [color[0], color[1], color[2], l.intensity],
                params: [l.range, cos_outer, kind, -1.0],
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

/// The frame's list: the ranked live lights, then as many baked lamps as fit,
/// nearest-influence first. Baked lamps only reach surfaces without a
/// lightmap, so they rank among themselves and never displace a live one.
pub fn append_baked(live: &[Light], baked: &[Light], max: usize) -> Vec<Light> {
    let mut out: Vec<Light> = live.iter().copied().take(max).collect();
    let room = max.saturating_sub(out.len());
    out.extend(rank_for_budget(baked, room));
    out
}

/// WGSL for the whole of group 0: the camera, the lights, and both shadow maps.
///
/// EVERY SHADING SHADER TAKES THIS BLOCK ENTIRE
///
/// It emits the camera uniform's struct and binding as well as the lights',
/// which the raytracing branch this was ported from did not -- there, each
/// shader declared its own copy of the camera struct and `shade` took the four
/// fields it needed as arguments, so a call read:
///
/// ```text
/// shade(world, n, view_dir, camera.sun_view_proj, camera.spot_view_proj,
///       camera.shadow_params)
/// ```
///
/// repeated at seven call sites, each of which had to be edited in step with
/// the struct. Since a shader that calls `shade` needs the camera uniform
/// anyway, emitting both together makes the call `shade(world, n)` again and
/// leaves exactly one declaration of the layout to keep in step with
/// `uniforms::Uniforms`.
///
/// Shaders that only need `view_proj` -- wire, mirror, the ui2d passes -- keep
/// their own two-field struct. That is legal and deliberate: a uniform struct
/// may be a PREFIX of the buffer bound to it, so they are unaffected by
/// anything added here.

pub fn wgsl_lights_block(group_index: u32, binding_index: u32) -> String {
    wgsl_lights_block_with(group_index, binding_index, false)
}

/// `wgsl_lights_block`, where `probe_from_pass` makes `shade_material_env`
/// take its probe reflection from `probe_env_given` -- the half-resolution
/// probe pass's answer, set by the brush shader -- instead of tracing it per
/// pixel. See `brush_pipeline::probe_pass`.
pub fn wgsl_lights_block_with(group_index: u32, binding_index: u32, probe_from_pass: bool) -> String {
    let shadow_tex = binding_index + 1;
    let shadow_samp = binding_index + 2;
    let spot_tex = binding_index + 3;
    let probe_tex = binding_index + 4;
    let probe_samp = binding_index + 5;
    let sun_dynamic_tex = binding_index + 6;
    let probe_depth_tex = binding_index + 7;
    let probe_depth_samp = binding_index + 8;
    // Three vec4 per probe -- centre, min, max -- so the WGSL array length is
    // three times the probe count. Derived rather than written twice: a shader
    // array shorter than the uniform reads garbage past its end.
    let probe_slots = crate::renderer::uniforms::MAX_PROBES * 3;
    let portal_slots = crate::renderer::uniforms::MAX_PORTALS * 3;
    let proxy_slots = crate::renderer::uniforms::MAX_PROXIES * 3;
    let max_spot_shadows = super::shadow::MAX_SPOT_SHADOWS;
    // Emitted from the Rust constant so the shader cannot disagree with the
    // atlas the pass actually renders into.
    let atlas_cols = super::shadow::SPOT_ATLAS_COLS;
    // Generated from the constants in `tonemap`, so the shader and the CPU
    // reference cannot drift apart.
    let tonemap_block = super::tonemap::wgsl_tonemap_block();
    // Widened as the eye buffer shrinks -- see `probe_box_margin`.
    let probe_box_margin = probe_box_margin();
    let portal_fade = PROBE_PORTAL_FADE;
    let portal_side_fade = PROBE_PORTAL_SIDE_FADE;
    format!(
        r#"
struct Camera {{
    view_proj: array<mat4x4<f32>, 2>,
    inv_view_proj: array<mat4x4<f32>, 2>,
    sun_view_proj: mat4x4<f32>,
    sun_dynamic_view_proj: mat4x4<f32>,
    spot_view_proj: array<mat4x4<f32>, {max_spot_shadows}>,
    camera_pos: array<vec4<f32>, 2>,
    // x = sun shadow on, y = spot shadow on, z = which light is the flashlight.
    shadow_params: vec4<f32>,
    // x = sky intensity.
    sky_params: vec4<f32>,
    // Nine RGB irradiance coefficients. Must match `uniforms::Uniforms`.
    sky_sh: array<vec4<f32>, 9>,
    // x = exposure, y = tone mapping mode (0 = ACES, 1 = none).
    post_params: vec4<f32>,
    // xyz = player world position, w = player yaw.
    player_frame: vec4<f32>,
    // x = how many reflection probes are live.
    // yzw = the sky's downward irradiance, for the reflected ground term.
    probe_params: vec4<f32>,
    // Three vec4 per probe -- centre, box min, box max -- FLATTENED.
    //
    // Flat rather than `array<array<vec4, 3>, N>` because a nested array in the
    // uniform address space brings stride rules that buy nothing here: the
    // bytes are identical either way, and indexing `i * 3 + k` by hand is
    // clearer than trusting two languages to agree about padding.
    probe_boxes: array<vec4<f32>, {probe_slots}>,
    // x = how many doorway portals are live. Must match `uniforms::Uniforms`.
    portal_params: vec4<f32>,
    // Three vec4 per portal -- [min, axis], [max, low volume], [high volume].
    probe_portals: array<vec4<f32>, {portal_slots}>,
    // x = how many reflection proxies are live. Must match `uniforms::Uniforms`.
    proxy_params: vec4<f32>,
    // Three vec4 per proxy -- [centre, volume], [half size], [rotation xyzw].
    probe_proxies: array<vec4<f32>, {proxy_slots}>,
}}

// Undo the player-frame transform the CPU applied to this vertex.
//
// Geometry arrives as `yaw_inv * (world - offset)`. Anything that must stay
// pinned to the world -- a ground texture, a height-based blend -- needs the
// original back, and only for that purpose: lighting stays in the player frame
// because the lights were uploaded there too.
// The same, for a DIRECTION: rotation only, no translation.
//
// A direction has no position, so applying the player's offset to one turns a
// unit vector into a point somewhere near them -- which is not wrong by a
// little.
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
@group({group_index}) @binding(0) var<uniform> camera: Camera;
{tonemap_block}

struct Light {{
    position: vec4<f32>,
    direction: vec4<f32>,
    color_intensity: vec4<f32>,
    params: vec4<f32>,
}}
struct Lights {{
    count: vec4<u32>,
    lights: array<Light, {MAX_LIGHTS}>,
}}
@group({group_index}) @binding({binding_index}) var<uniform> lights: Lights;
@group({group_index}) @binding({shadow_tex}) var sun_shadow_tex: texture_depth_2d;
// The sun's shadow of moving things only. See `shadow::SUN_DYNAMIC_DIM`.
@group({group_index}) @binding({sun_dynamic_tex}) var sun_dynamic_shadow_tex: texture_depth_2d;
@group({group_index}) @binding({shadow_samp}) var shadow_samp: sampler_comparison;
// An ARRAY, one layer per shadow-casting spot. A level has more than one lamp,
// and the single map this replaced meant the first spot in the scene silently
// claimed it while the rest lit without shadows.
@group({group_index}) @binding({spot_tex}) var spot_shadow_tex: texture_depth_2d;
@group({group_index}) @binding({probe_tex}) var probe_cube: texture_cube_array<f32>;
@group({group_index}) @binding({probe_samp}) var probe_samp: sampler;
// Each probe texel's distance in metres, 0 where none was baked. See
// `probe_trace`.
@group({group_index}) @binding({probe_depth_tex}) var probe_depth: texture_cube_array<f32>;
@group({group_index}) @binding({probe_depth_samp}) var probe_depth_samp: sampler;

const AMBIENT: f32 = 0.6;

// Where a world direction lands in the sky panorama. Pinned to the editor's
// mapping by measurement -- see sky.rs.
fn sky_uv(d: vec3<f32>) -> vec2<f32> {{
    let n = normalize(d);
    let u = (atan2(n.x, n.z) + 3.14159265) / 6.28318531;
    let v = acos(clamp(n.y, -1.0, 1.0)) / 3.14159265;
    return vec2<f32>(u, v);
}}

// Ambient from the sky, in the direction the surface faces.
//
// The same arithmetic as `SkyIrradiance::evaluate` in sky.rs, which is what
// lets the projection be tested without a GPU and the shader be trusted to
// agree with it. A scene with no sky uploads a constant-band SH that evaluates
// to exactly AMBIENT everywhere, so this replaces the old flat term without
// changing what a level without a sky looks like.
fn sky_irradiance(n: vec3<f32>) -> vec3<f32> {{
    let x = n.x; let y = n.y; let z = n.z;
    var b = array<f32, 9>(
        0.282095,
        0.488603 * y,
        0.488603 * z,
        0.488603 * x,
        1.092548 * x * y,
        1.092548 * y * z,
        0.315392 * (3.0 * z * z - 1.0),
        1.092548 * x * z,
        0.546274 * (x * x - y * y),
    );
    // The cosine lobe's coefficients, already divided by pi -- see
    // SkyIrradiance::evaluate in sky.rs, which this mirrors exactly.
    var a = array<f32, 9>(
        1.0,
        0.6666667, 0.6666667, 0.6666667,
        0.25, 0.25, 0.25, 0.25, 0.25,
    );
    var e = vec3<f32>(0.0);
    for (var i: i32 = 0; i < 9; i = i + 1) {{
        e = e + camera.sky_sh[i].rgb * b[i] * a[i];
    }}
    return max(e, vec3<f32>(0.0));
}}
const SPEC_STRENGTH: f32 = 0.35;
const SHININESS: f32 = 32.0;

// How big a lamp is, in metres, for the purposes of its highlight.
//
// WHY A LIGHT NEEDS A SIZE
//
// A punctual light has zero area, and the mirror image of a zero-area source is
// a point. Marble020's roughness map averages 0.048 -- near mirror -- which put
// the Blinn-Phong exponent on its 2048 clamp: a highlight a few pixels across,
// visible only from the exact mirror angle, which in practice means never. The
// specular was being computed correctly and was invisible.
//
// Real lamps have area, and that area is what makes polished stone show a broad
// sheen rather than a glint. Widening the roughness by the solid angle the
// source subtends is the standard approximation (a sphere light, as in Karis
// 2013): far away it changes nothing, close up it spreads the highlight the way
// a real fixture does.
const LIGHT_SOURCE_RADIUS: f32 = 0.35;

// The tightest lobe worth rendering.
//
// Beyond this the highlight is narrower than the pixels sampling it, so it
// aliases into a flicker instead of reading as shine -- and on a headset that
// flicker is the most noticeable thing in the frame. 2048 was chosen as "very
// smooth"; it is really "smaller than a pixel".
const MAX_SHININESS: f32 = 320.0;

/// 1/pi, for turning an irradiance into the radiance a Lambertian surround
/// would have to emit to produce it.
const INV_PI: f32 = 0.31830987;

// WHERE A PIXEL'S LIGHT CAME FROM, for the source diagnostic.
//
// Written by `shade_material_env` as it goes and read by the brush shader when
// the diagnostic is on. Module-scope private storage rather than extra return
// values, so the shipping path costs three stores that the compiler removes
// when nothing reads them, instead of changing a signature that six shaders
// share.
var<private> dbg_probe: vec3<f32> = vec3<f32>(0.0);
var<private> dbg_direct: vec3<f32> = vec3<f32>(0.0);
var<private> dbg_baked: vec3<f32> = vec3<f32>(0.0);
// The probe term's FACTORS, separately: Fresnel over its own maximum, the
// probe's blend weight, and occlusion times normalisation. Painted by
// `BRUSH_PROBE_FACTOR_DEBUG` to name which one spikes along a seam.
var<private> dbg_probe_factors: vec3<f32> = vec3<f32>(0.0);
// The world-space size of this fragment's pixel, set once at the top of a
// fragment shader that wants distance-aware filtering (0 = none, which leaves
// every shader that does not set it exactly as it was). Taken there because
// derivatives are only valid in uniform control flow, and the light loop is
// not that.
var<private> pixel_footprint: f32 = 0.0;
// A LAMP'S FALLOFF IS THE INVERSE SQUARE, clamped at the bulb.
//
// It was `window^2 / (d^2 + 1)`, which is Unreal's formula -- where distances
// are in CENTIMETRES and the +1 only matters inside a centimetre. In metres it
// halves a lamp's light at 1 m and cuts it to 0.42 at the 0.85 m a wall spot
// sits from its wall: measured against the path-traced reference, the engine
// drew that pool at 3.26 where physics puts 7.8 (2026-09-23). The window keeps
// the smooth cutoff at `range`; the clamp stands in for the bulb's own size,
// inside which a surface cannot get any closer to the light. 5 cm, as in the
// reference. The baker and the editor preview use the same curve.
const LAMP_RADIUS: f32 = 0.05;
// The sky sun's BAKED VISIBILITY at the surface being shaded, 0..1, or -1 when
// there is none. A brush sets it from its sun mask -- sixteen rays per 3-6 cm
// texel across the sun's disc, so the edge is the real penumbra rather than a
// staircase of lightmap texels -- and everything else leaves it at -1 and is
// shadowed by the level's static sun map instead. See `sun_visibility`.
var<private> receiver_sun_mask: f32 = -1.0;

// THE STATIONARY LAMPS' BAKED SHADOWS at this receiver: the visibility of the
// lamp owning mask channel c is `stationary_vis_a[c]` for c < 4, else
// `stationary_vis_b[c - 4]`. Set by receivers that carry the masks -- the
// brushes, from their atlas; 1, unshadowed, everywhere else. See
// `stationary_visibility`.
var<private> stationary_vis_a: vec4<f32> = vec4<f32>(1.0);
var<private> stationary_vis_b: vec4<f32> = vec4<f32>(1.0);

// How much of lamp `l` the LEVEL lets through to this receiver: its channel of
// the baked mask when it is a stationary lamp (`position.w` = 2 + channel),
// else 1 -- a live lamp is shadowed by the shadow map, if it has a slot.
fn stationary_visibility(l: Light) -> f32 {{
    return stationary_visibility_of(l.position.w);
}}
// The same, from the light's `position.w` alone: what the light loop reads
// before it loads the rest of the light.
fn stationary_visibility_of(marker: f32) -> f32 {{
    if (marker < 1.5) {{
        return 1.0;
    }}
    let c = i32(marker - 1.5);
    let v = select(stationary_vis_b, stationary_vis_a, c < 4);
    return v[c & 3];
}}
// Whether the light loop may skip a lamp that cannot reach this pixel before
// doing any of its maths. See `GpuLights::count`; off only to measure it.
fn light_culling() -> bool {{
    return lights.count.z == 0u;
}}
// Whether this fragment's surface carries the BAKED lights in its lightmap.
// The light list is ordered live first, baked after -- `lights.count.y` is
// where the baked tail starts -- and a lightmapped surface stops there: its
// atlas already holds those lamps' light, and shading them again counted
// every baked lamp twice. A skinned character or the ground has no such
// atlas and takes the whole list. Set by the pipelines that bind a lightmap.
var<private> receiver_skips_baked: bool = false;
fn live_light_count() -> u32 {{
    return select(lights.count.x, lights.count.y, receiver_skips_baked);
}}
// A spot's soft edge is never drawn narrower than this many pixels.
const SPOT_EDGE_MIN_PIXELS: f32 = 1.5;
// ...but never more than this many times its AUTHORED width.
//
// Uncapped, a floor seen at a grazing angle from across the room has a pixel
// footprint of a metre or more, and the first build of this widened a spot's
// pool toward the doorway by that much -- a pool whose size depended on where
// the player stood, which read as a softer blend and shimmered where the
// footprint jumped between the floor and the door threshold (2026-09-23). A
// cap keeps the anti-aliasing and refuses to invent light.
const SPOT_EDGE_MAX_WIDEN: f32 = 3.0;
// A spot's cone, with its soft edge kept at least `SPOT_EDGE_MIN_PIXELS` wide
// ON SCREEN.
//
// Up close the authored edge spans many pixels and this is exactly the old
// smoothstep. At distance, a pool seen from far away or edge-on shrinks to a
// few pixels and its authored edge to less than one, so each sub-pixel head
// movement flipped whole pixels between lit and unlit -- the "dancing" wall
// spot seen from the doorway measured about 4 native pixels wide (2026-09-22).
// Widening the edge to a pixel or two keeps the pool's energy and stops it
// snapping: the same idea as keeping a thin wire at least one pixel wide and
// fading it (Persson, "Wire Antialiasing"), applied to a lighting edge.
//
// The pixel's angular size seen FROM THE LIGHT is its world footprint over the
// distance, and a step of angle becomes `sin(angle) * step` in cosine space,
// which is the space the cone is measured in.
fn spot_cone(cos_angle: f32, cos_outer: f32, cos_inner: f32, dist: f32) -> f32 {{
    let authored = max(cos_inner - cos_outer, 0.0001);
    let sin_a = sqrt(max(1.0 - cos_angle * cos_angle, 0.0));
    let at_least = SPOT_EDGE_MIN_PIXELS * sin_a * pixel_footprint / max(dist, 0.001);
    let width = max(authored, min(at_least, authored * SPOT_EDGE_MAX_WIDEN));
    // How much the edge grew, split evenly either side of the authored band.
    // ZERO in a bake, which has no pixels -- and with `widen` at zero this is
    // exactly the baker's cone, (cos_angle - cos_outer) / authored, which
    // `renderer_and_baker_agree_on_the_formula` pins by text.
    let widen = width - authored;
    let t = clamp((cos_angle - cos_outer + 0.5 * widen) / width, 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}}
// The average radiance the chosen probe photographed, written by
// `probe_environment` for `shade_material_env` to normalise against. Zero means
// unknown, and an unknown probe is used as it is. See `PROBE_NORMALISATION`.
var<private> probe_brightness: f32 = 0.0;
// WHETHER THIS SHADER READS ITS PROBE REFLECTION FROM THE HALF-RESOLUTION
// PASS rather than tracing it per pixel. See `brush_pipeline::probe_pass`. A
// constant, so a shader that reads it carries none of the trace: the branch in
// `shade_material_env` folds away, and the trace's registers with it.
const PROBE_ENV_FROM_PASS: bool = {probe_from_pass};
// The half-resolution pass's answer for this pixel -- the probe radiance,
// already normalised, and its coverage -- set by the brush shader before it
// shades. Only read when `PROBE_ENV_FROM_PASS`.
var<private> probe_env_given: vec4<f32> = vec4<f32>(0.0);
// WHERE TO STAND WHEN ASKING WHICH ROOM, set by a caller that knows better than
// the pixel (`w` = 1). A brush sets its face centre: a face belongs to one room,
// and an MSAA edge sample extrapolated along a grazing surface can land well
// outside it. Which photograph WITHIN the room is still chosen per pixel, from
// `select_pos`, where a small error only moves a smooth blend weight.
var<private> probe_volume_pos: vec4<f32> = vec4<f32>(0.0);
/// Mip levels a fully rough surface reaches for in the probe cube.
///
/// The cube is 128 px a face, so it has eight levels and the top one is a
/// single texel -- the average of everything the probe saw, which is exactly
/// what a perfectly diffuse surface should reflect. Roughness scales onto that
/// range the same way it does for the screen-space path, so a hit and a miss
/// on the same surface are blurred alike and the boundary between them stops
/// being visible.
const PROBE_ROUGHNESS_MIPS: f32 = 7.0;
/// The sharpest a probe reflection is ever allowed to be, in mip levels.
///
/// A PROBE IS NOT A MIRROR, but polished marble nearly is, and the floor has
/// to leave room for it.
///
/// Level 0 of a probe face asserts detail the photograph may not hold -- a
/// crisp light fixture on a glossy wall with no source in view was once how
/// the probe gave itself away. So the floor stays above zero.
///
/// It was 2.0 on a 64 px face: a 16 px view of the room for every surface, so
/// a bright doorway or a lamp's floor pool reflected off marble as a soft blob
/// many times its size -- the light-and-dark wall patches the lighting-sources
/// view traced to the probe on the headset (2026-09-11). On the 128 px face
/// that `DEFAULT_PROBE_RESOLUTION` now bakes, 1.5 is a 45 px view: a third of
/// the old blur, still well above a mirror.
///
/// (The note that stood here said levels above 0 came back BLACK. That was
/// fixed; `every_mip_level_receives_its_data` reads each level back.)
///
/// 0.5 since 2026-09-25. 1.5 was tuned while the probes photographed every
/// interior wall as black, so there was nothing on them to blur and the cost
/// was invisible. With the probes showing the room, 1.5 dissolved the marble
/// floor's reflection of the lit doorway, the ceiling and the lamps into a
/// wash (headset, 2026-09-25: "did not show at all"). Marble's own lobe is
/// 0.048 x 7 = 0.34 levels; 0.5 is just softer than that, so a probe is still
/// never read as a perfect mirror.
const PROBE_MIN_LOD: f32 = 0.5;
/// The highest level the cube actually has. See `probe_mip_levels`.
const PROBE_MAX_LOD: f32 = 7.0;
/// How far from equidistant, in metres, two probe cells still cross-fade.
///
/// A probe's reflection of anything NOT on the room's walls -- a hanging lamp,
/// its bulb -- lands somewhere different from each capture point. Blending two
/// photographs therefore shows that thing twice: a strong copy and a faint one.
/// Lagarde ("Local Image-based Lighting With Parallax-corrected Cubemaps",
/// 2012) calls this ghosting, and confines blending to the transition between
/// influence volumes for exactly that reason.
///
/// The two nearest cells used to be weighted by inverse distance EVERYWHERE, so
/// the room held two copies of every lamp except directly on top of a capture
/// point -- reported 2026-09-10 as a reflection "in two places on the wall",
/// with or without screen-space reflections. Now a fragment takes the nearest
/// probe outright unless the two capture points are within this much of
/// equidistant, and on the boundary itself they meet 50/50, so the hard seam
/// the blend was added to remove does not come back.
///
/// 2.0 m of distance difference, about a metre either side of the plane along
/// the hall's axis and wider towards its walls. It was 0.6 while the far
/// photograph was read along the NEAR one's parallax direction, which put its
/// lamps in the wrong place and made any blend look doubled; read from its own
/// capture point (`probe_parallax_direction`) the two agree on everything on
/// the room's walls, and a 6 m cell (`bake::probe::MAX_CELL_EDGE`) needs the
/// wider band for the handover not to read as a line.
const PROBE_BLEND_BAND: f32 = 2.0;
/// How far outside its box a fragment still counts as inside, in metres.
///
/// See the selection loop: surfaces sit exactly on their room's probe box.
/// 5 cm covers a pixel's overshoot AT FULL RESOLUTION; it is widened as the eye
/// buffer shrinks, because what it really measures is a PIXEL. See
/// `probe_box_margin` on the Rust side.
const PROBE_BOX_MARGIN: f32 = {probe_box_margin:?};
/// Whether a probe's reflection is scaled to the ambient light at each pixel.
///
/// The idea Unreal applies to its reflection captures: the lightmap knows how
/// much light reaches a point and the capture does not, so the capture's
/// brightness is normalised by the lightmap's. Needs each probe's average
/// radiance, which the renderer computes when the probes load
/// (`uniforms::probe_mean_radiance`); a probe without one is left unscaled.
const PROBE_NORMALISATION: bool = true;
/// The darkest a normalised reflection is allowed to get, as a fraction.
///
/// Zero: a corner with no light at all reflects nothing, which is what an
/// unlit corner does.
const PROBE_NORMALISATION_FLOOR: f32 = 0.0;

/// Shortest encoded bounce direction still treated as a direction.
///
/// QUANTISATION, not epsilon. The neutral texel is byte 128, and 128/255 is not
/// exactly 0.5: a vector meant to be zero decodes about 0.0068 long, which a
/// guard of 1e-4 waves through and normalises into a confident diagonal.
///
/// Harmless today only because that same texel carries a directionality of
/// zero, which scales both terms using the direction to nothing -- correct by
/// luck rather than by design. A real direction has length 1, so anything well
/// above the rounding and well below 1 separates them.
const MIN_BOUNCE_DIR_LENGTH: f32 = 0.05;

/// How far a per-texel dominant bounce direction is worth trusting.
///
/// The baked value is an honest measurement -- the length of the mean incoming
/// direction over the total light -- and on this project's test room it averages
/// 0.79. Used at full strength that makes the shaping factor span 0.02 to 1.98,
/// very nearly a hundredfold swing, and a DOMINANT DIRECTION is not smooth
/// enough to carry it: neighbouring texels were measured flipping to exactly
/// opposite directions, so the factor could jump from one end of that range to
/// the other across a single texel boundary.
///
/// On screen that is not soft directional shading. It is a grid of hard bright
/// and dark squares the size of a lightmap texel -- 25 cm of wall -- and it was
/// reported from the headset as circular lights reflecting as squares.
///
/// Capping the trust rather than the direction keeps the shading directional
/// where the direction is coherent, which is most of the map, and stops one
/// texel's disagreement with its neighbour from becoming a visible edge. A
/// smoother representation than one dominant direction -- spherical harmonics
/// per texel -- would raise this ceiling, and is the real fix if it is ever
/// worth the extra map.
const MAX_BOUNCE_DIRECTIONALITY: f32 = 0.5;

// Projects a world position into a light's clip space and returns
// (uv.x, uv.y, biased depth, valid). `valid` is 0 outside the light's frustum,
// where the caller must treat the fragment as fully lit rather than shadowed --
// otherwise everything beyond the shadow map's reach goes black.
// WHICH EYE THIS INVOCATION IS DRAWING.
//
// A private, not a constant, because the two render modes set it differently:
// a single-view pass leaves it at 0, and a multiview pass assigns
// `@builtin(view_index)` at the top of its entry point. Reading it through
// these accessors rather than indexing `camera.view_proj` directly is what
// makes that a one-line change per shader instead of an edit at every use.
//
// `view_index` is readable in the FRAGMENT stage as well as the vertex stage,
// which is what makes this workable at all -- the specular term needs the eye's
// position, and without it every shader's vertex-to-fragment struct would have
// had to carry an extra varying.
var<private> view_slot: i32 = 0;

fn cam_view_proj() -> mat4x4<f32> {{ return camera.view_proj[view_slot]; }}
fn cam_inv_view_proj() -> mat4x4<f32> {{ return camera.inv_view_proj[view_slot]; }}
fn cam_pos() -> vec3<f32> {{ return camera.camera_pos[view_slot].xyz; }}

fn shadow_coords(world_pos: vec3<f32>, light_view_proj: mat4x4<f32>) -> vec4<f32> {{
    let lp = light_view_proj * vec4<f32>(world_pos, 1.0);
    let ndc = lp.xyz / lp.w;
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    var valid = 1.0;
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || ndc.z > 1.0 || ndc.z < 0.0) {{
        valid = 0.0;
    }}
    let bias = 0.0015;
    return vec4<f32>(uv.x, uv.y, ndc.z - bias, valid);
}}

// 3x3 hardware PCF: 1 is lit, 0 is shadowed.
//
// `textureSampleCompareLevel` and not `textureSampleCompare`: the latter takes
// implicit derivatives and so may not be called after the per-fragment early
// return above, which is exactly the WGSL uniform-control-flow rule.
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

// How much of a directional light reaches `world_pos`, 0..1.
//
// THE LEVEL'S SHADOW, then MOVING THINGS'. The level never moves, so its sun
// shadow is either baked into the surface's mask (`receiver_sun_mask`, brushes)
// or drawn once into the static map (everything else). What moves -- the
// player, a prop -- is in its own small map around the player, redrawn every
// frame, and only the sky's sun (`position.w`) uses it: an authored sun still
// has one head-following map holding everything, as it always did.
//
// The moving-objects lookup is skipped wherever the level already hides the
// sun, which indoors is nearly everywhere -- that is what makes a live sun on
// the brushes cost almost nothing on a floor the sun never touches.
fn sun_visibility(l: Light, world_pos: vec3<f32>) -> f32 {{
    var vis = 1.0;
    if (receiver_sun_mask >= 0.0) {{
        vis = receiver_sun_mask;
    }} else if (camera.shadow_params.x > 0.5) {{
        vis = pcf(sun_shadow_tex, world_pos, camera.sun_view_proj);
    }}
    if (vis > 0.0 && l.position.w > 0.5 && camera.shadow_params.z > 0.5) {{
        vis = vis * pcf(sun_dynamic_shadow_tex, world_pos, camera.sun_dynamic_view_proj);
    }}
    return vis;
}}

// One spot's depth, read out of its tile of the shared atlas.
//
// The atlas exists because a pass is the expensive unit on a tile GPU, not the
// triangles in it -- see `ShadowMap::spots`. The cost of that is here: every
// sample has to be mapped into its own tile AND CLAMPED to it. Without the
// clamp, the 3x3 kernel at a tile's edge reaches into the neighbouring tile and
// reads another light's depth, which shows up as a shadow cast by a lamp that
// is nowhere near -- far more confusing than a missing shadow.
fn pcf_layer(tex: texture_depth_2d, layer: i32, world_pos: vec3<f32>, light_view_proj: mat4x4<f32>) -> f32 {{
    let c = shadow_coords(world_pos, light_view_proj);
    if (c.w < 0.5) {{ return 1.0; }}
    let cols = f32({atlas_cols});
    let tile = vec2<f32>(f32(layer % {atlas_cols}), f32(layer / {atlas_cols}));
    let atlas_texel = 1.0 / vec2<f32>(textureDimensions(tex));
    // Half a texel in from each edge of this tile, in tile space. Sampling
    // exactly ON the boundary already blends the neighbour under linear
    // filtering.
    let guard = atlas_texel * cols * 0.5;
    let lo = guard;
    let hi = vec2<f32>(1.0) - guard;

    var sum = 0.0;
    for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {{
        for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {{
            // Offset in TILE space, then clamped there, so the kernel never
            // walks out of the tile however close to its edge the sample is.
            let off = vec2<f32>(f32(dx), f32(dy)) * atlas_texel * cols;
            let local = clamp(c.xy + off, lo, hi);
            let uv = (local + tile) / cols;
            sum = sum + textureSampleCompareLevel(tex, shadow_samp, uv, c.z);
        }}
    }}
    return sum / 9.0;
}}

fn light_contribution(l: Light, world_pos: vec3<f32>, n: vec3<f32>, view_dir: vec3<f32>) -> vec3<f32> {{
    return light_contribution_rough(l, world_pos, n, view_dir, SHININESS, SPEC_STRENGTH);
}}

// The same, with the highlight's tightness and strength supplied by the caller.
//
// Split out rather than adding parameters to `light_contribution`, because that
// one is called by the mesh, cuboid and terrain shaders too and none of them
// have a roughness map to pass. They keep the constants they always used.
struct LightSplit {{
    diffuse: vec3<f32>,
    specular: vec3<f32>,
}}

/// A light's contribution, kept in its two halves.
///
/// Separate because they are tinted by different things. The DIFFUSE half is
/// light the surface absorbed and re-emitted, so it takes the surface's colour;
/// the SPECULAR half is light that bounced straight off the surface without
/// entering it, and a dielectric reflects it unchanged -- a red marble floor
/// has a WHITE highlight.
///
/// Combining them before the albedo is applied, as this did, multiplies the
/// highlight by the diffuse colour: on this project's marble, an albedo of
/// 0.373 made every highlight nearly three times dimmer than it should be, and
/// the surface read as though it had no shine at all.
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
        let window = clamp(1.0 - pow(d_over_r, 4.0), 0.0, 1.0);
        atten = (window * window) / max(dist * dist, LAMP_RADIUS * LAMP_RADIUS);
        if (kind > 0.5) {{
            let cos_outer = l.params.y;
            let cos_inner = l.direction.w;
            let cos_angle = dot(-l_dir, l.direction.xyz);
            atten = atten * spot_cone(cos_angle, cos_outer, cos_inner, dist);
        }}
    }}

    let ndotl = max(dot(n, l_dir), 0.0);
    let radiance = l.color_intensity.rgb * l.color_intensity.a;
    out.diffuse = radiance * ndotl * atten;
    if (ndotl > 0.0) {{
        let h = normalize(l_dir + view_dir);
        let spec = pow(max(dot(n, h), 0.0), shininess) * spec_strength;
        out.specular = radiance * spec * atten;
    }}
    return out;
}}

fn light_contribution_rough(
    l: Light,
    world_pos: vec3<f32>,
    n: vec3<f32>,
    view_dir: vec3<f32>,
    shininess: f32,
    spec_strength: f32,
) -> vec3<f32> {{
    // params.z tags the kind: 0 point, 1 spot, 2 directional.
    let kind = l.params.z;

    var l_dir: vec3<f32>;
    var atten: f32;
    if (kind > 1.5) {{
        // The sun. Parallel rays travel ALONG `direction`, so the surface-to-
        // light vector is its negation, and there is no distance falloff --
        // applying one would make the sun dim with the scene's origin.
        l_dir = normalize(-l.direction.xyz);
        atten = 1.0;
    }} else {{
        let to_light = l.position.xyz - world_pos;
        let dist = length(to_light);
        l_dir = to_light / max(dist, 0.0001);

        let d_over_r = dist / max(l.params.x, 0.0001);
        let window = clamp(1.0 - pow(d_over_r, 4.0), 0.0, 1.0);
        atten = (window * window) / max(dist * dist, LAMP_RADIUS * LAMP_RADIUS);

        if (kind > 0.5) {{
            let cos_outer = l.params.y;
            let cos_inner = l.direction.w;
            let cos_angle = dot(-l_dir, l.direction.xyz);
            atten = atten * spot_cone(cos_angle, cos_outer, cos_inner, dist);
        }}
    }}

    let ndotl = max(dot(n, l_dir), 0.0);
    let radiance = l.color_intensity.rgb * l.color_intensity.a;
    var out = radiance * ndotl * atten;

    // Blinn-Phong specular, gated on ndotl so a surface facing away from the
    // light gets no highlight. Ungated, the half-vector still lines up on the
    // far side and rims every object with light coming from behind it.
    if (ndotl > 0.0) {{
        let h = normalize(l_dir + view_dir);
        let spec = pow(max(dot(n, h), 0.0), shininess) * spec_strength;
        out = out + radiance * spec * atten;
    }}
    return out;
}}

// Shading that knows what the surface is MADE OF.
//
// `roughness` 0 is a mirror-smooth surface and 1 is fully matte; `ao` is the
// material's own baked contact shadow, 1 meaning unoccluded. Both come from a
// material's maps, and without them every surface in the level shades
// identically -- polished concrete lights exactly like rough brick, which no
// amount of work on the lights can distinguish.
//
// AO scales the AMBIENT term only. It is a statement about how much of the sky
// a crevice can see, not about whether a lamp is pointed at it, and multiplying
// direct light by it would darken surfaces a light is shining straight onto.
//
// `sky_vis` is the baked fraction of the hemisphere that actually reaches the
// sky from this point, and it scales the sky term for the same reason AO does
// -- a wall inside a sealed room can see none of it. Without this the sky lit
// every interior surface as brightly as open ground, which is why rooms looked
// flat and shadows looked weak: the shadow was there, but a full-strength
// ambient term was filling it back in.
fn shade_material(world_pos: vec3<f32>, n: vec3<f32>, roughness: f32, ao: f32, sky_vis: f32) -> vec3<f32> {{
    // NEUTRAL, not absent: no environment light and no direction, which shades
    // exactly as this did before either existed.
    return shade_material_env(
        world_pos, n, roughness, ao, sky_vis, vec3<f32>(0.0), vec4<f32>(0.5, 0.5, 0.5, 0.0),
        // WHITE: callers of this form apply their own albedo to the whole
        // result afterwards, which is what they have always done.
        vec3<f32>(1.0),
        // No face to stand on: choose the probe from the fragment itself,
        // which is exactly what every caller of this form did before.
        world_pos,
        // No separate geometric normal here either.
        n,
    );
}}

/// `shade_material`, told what the LOCAL environment is putting out.
///
/// WHY THE SKY IS NOT ENOUGH
///
/// The reflection term below started out sampling only the sky, scaled by how
/// much sky the surface can see. That is right outdoors and completely wrong
/// indoors: sealed rooms have a sky visibility of zero, so a polished marble
/// floor in a lit hall reflected NOTHING and read as matte stone. But a shiny
/// floor in a dark room does show something -- it shows the room. Light a wall
/// and the floor picks it up.
///
/// `env` is the indirect irradiance already known at this point -- the baked
/// bounce -- and it stands in for the radiance of everything around the
/// surface. It is not directional: an irradiance value says how much light
/// arrives, not from where, so this cannot place a reflection of a specific
/// lamp in a specific spot. What it does do is make a smooth surface take on
/// the brightness and COLOUR of the room it is in, and respond when the lights
/// in that room change -- which is the difference between marble that reads as
/// polished and marble that reads as grey.
///
/// Divided by pi because `env` is irradiance and the reflection wants radiance:
/// a Lambertian surround with irradiance E has radiance E/pi, and dropping the
/// factor makes indoor reflections about three times too bright.
/// DIAGNOSTIC: the punctual light reaching a point, before and after shadowing.
///
/// x is the sum of every point/spot light's diffuse term with the shadow factor
/// left out; y is the same with it applied. Rendering the pair as red and green
/// separates the three ways a surface can end up black -- no light arrives
/// (black), light arrives and is shadowed away (red), light arrives and
/// survives (yellow) -- which no single-channel debug view can do.
///
/// Unused unless `BRUSH_LIGHT_DEBUG` is on. See `brush_shader_with`.
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
        if (layer >= 0 && f32(layer) < camera.shadow_params.y) {{
            shadow = shadow * pcf_layer(spot_shadow_tex, layer, world_pos, camera.spot_view_proj[layer]);
        }}
        pre = pre + lum;
        post = post + lum * shadow;
    }}
    return vec2<f32>(pre, post);
}}

// The baked reflection covering a point, or alpha 0 where none does.
//
// PARALLAX-CORRECTED. A cubemap has no position, so sampling it with the raw
// mirror direction says the reflected world is infinitely far away -- and in a
// room that makes the reflection slide across the walls as you walk, which is
// the single thing that makes a probe read as fake. Intersecting the reflected
// ray with the probe's BOX and sampling toward where it actually lands anchors
// the reflection to the wall. Mirrors `reflection_probe::box_project`.
//
// The SMALLEST containing box wins, so an alcove's probe beats the hall's.
// GEOMETRIC SPECULAR ANTIALIASING (Kaplanyan/Tokuyoshi).
//
// WHY MIPS AND ANISOTROPY ARE NOT ENOUGH. Both are already on here (8x), and
// they correctly average the normal MAP. But shading is not linear in the
// normal: the lighting of an averaged normal is not the average of the
// lighting. A mip level that is the right average of a bumpy surface still
// produces a highlight that pops in and out as the sampling point moves, so
// detail SHIMMERS at a distance and settles as you walk towards it -- which is
// exactly the symptom reported from the headset (2026-09-19).
//
// The fix is to put the normal's lost variation back as ROUGHNESS. A pixel
// covering many normals really is, at that scale, a rougher surface; widening
// the lobe to match is both the physically honest answer and a stable one,
// because a wide lobe changes slowly when the sample point moves.
//
// This treats the SPECULAR term only. A diffuse lobe is already wide enough
// that normal variance barely moves it, which is also why this is expected to
// help stone and marble more than it helps grass -- terrain shades through
// `shade_with_sky` and its shimmer, if it survives, is a different problem.
const SPECULAR_AA_SIGMA: f32 = 0.5;
// The most roughness^2 this may add. Uncapped, a silhouette pixel -- where the
// normal derivative is enormous because the quad straddles two surfaces --
// would be driven to fully rough and the highlight would vanish along every
// edge, trading a shimmer for a dark rim.
const SPECULAR_AA_CLAMP: f32 = 0.18;

fn specular_aa_roughness(roughness: f32, dndx: vec3<f32>, dndy: vec3<f32>) -> f32 {{
    let variance = SPECULAR_AA_SIGMA * (dot(dndx, dndx) + dot(dndy, dndy));
    let kernel = min(2.0 * variance, SPECULAR_AA_CLAMP);
    return sqrt(clamp(roughness * roughness + kernel, 0.0, 1.0));
}}

fn probe_environment(
    frag_pos: vec3<f32>,
    dir: vec3<f32>,
    roughness: f32,
    // WHERE TO STAND WHEN CHOOSING THE PROBE, which is not where the fragment
    // is. For a brush this is the centre of the face, flat-interpolated, so
    // every fragment of a wall picks the same probe and no seam can open
    // between neighbouring pixels. Callers with nothing better pass the
    // fragment position and get exactly the old behaviour.
    //
    // The PARALLAX below still uses `frag_pos`: where the reflected ray leaves
    // from is genuinely per-pixel, and using the face centre for it would flatten
    // the reflection across the whole face.
    select_pos: vec3<f32>,
) -> vec4<f32> {{
    // INTO WORLD SPACE FIRST.
    //
    // Geometry reaches this shader in the PLAYER's frame -- `yaw_inv * (world -
    // offset)` -- and lighting stays there because the lights are uploaded the
    // same way. A probe cannot: its box and its cubemap were baked in world
    // space, against walls that do not move.
    //
    // Comparing a player-frame position against a world-space box means the box
    // slides as the player walks, so the reflection drifts across the wall and
    // stays visible where the room should have ended. That is what it did.
    let world_pos = to_world_space(frag_pos);
    // Selection happens in world space too -- the box it is tested against was
    // baked there. Same transform, different point.
    let select_world = to_world_space(select_pos);
    let volume_world = select(select_world, to_world_space(probe_volume_pos.xyz), probe_volume_pos.w > 0.5);
    let world_dir = to_world_direction(dir);
    probe_brightness = 0.0;
    let count = i32(camera.probe_params.x);
    // NEAREST CAPTURE POINT, then smallest box.
    //
    // Every cell of one room now carries the ROOM's box rather than its own --
    // the box is what a reflected ray is projected onto, and a cell-sized box
    // projects rays onto geometry that is not there. So containment no longer
    // separates the cells of a room; the capture point does, and the nearest
    // photograph is the one whose parallax error is smallest.
    //
    // Volume still breaks ties, so a tight volume nested inside a larger one --
    // an alcove inside a hall, a room inside the outdoor volume -- keeps
    // winning over the thing that encloses it.
    var best = -1;
    var best_volume = 1e30;
    var best_dist = 1e30;
    var best_room = -1.0;
    // The RUNNER-UP, so the change from one cell to the next is a blend rather
    // than a switch. See the mix at the end of this function.
    var second = -1;
    var second_dist = 1e30;
    for (var i = 0; i < count; i = i + 1) {{
        let lo = camera.probe_boxes[i * 3 + 1].xyz;
        let hi = camera.probe_boxes[i * 3 + 2].xyz;
        // WITH A MARGIN. A room's probe box IS its interior, so every wall,
        // floor and ceiling fragment lies exactly on the box's surface. With
        // MSAA a pixel on a polygon's edge is shaded at its centre, which can
        // sit just past the edge -- outside the box -- and the fragment then
        // took a DIFFERENT probe's photograph (the one from outside). That was
        // the dotted light line along every room seam, blue in the sources
        // view, visible only from a distance where a pixel spans more world
        // (headset, 2026-09-17). The margin is far thinner than any wall, so
        // it cannot pull a neighbouring room's surfaces in.
        if (any(volume_world < lo - vec3<f32>(PROBE_BOX_MARGIN)) || any(volume_world > hi + vec3<f32>(PROBE_BOX_MARGIN))) {{
            continue;
        }}
        let d = hi - lo;
        let volume = d.x * d.y * d.z;
        let to_centre = camera.probe_boxes[i * 3].xyz - select_world;
        let dist = dot(to_centre, to_centre);
        // WHICH ROOM this slot photographs. Cells of one room share it; see
        // `ProbeUpload::set_volume`. Equal box SIZE used to stand in for this,
        // which let two different rooms of the same size blend together.
        let room = camera.probe_boxes[i * 3 + 2].w;
        let tighter = volume < best_volume * 0.999;
        let same_room_but_nearer = room == best_room && dist < best_dist;
        if (tighter) {{
            // A new, tighter room: the runner-up belonged to the old one, and
            // blending across rooms would put the enclosing volume's light --
            // the sky, outdoors -- on an interior wall.
            second = -1;
            second_dist = 1e30;
            best_volume = volume;
            best_dist = dist;
            best_room = room;
            best = i;
        }} else if (same_room_but_nearer) {{
            // The old winner becomes the runner-up -- same room, so blending
            // the two is a handover, not a leak.
            second = best;
            second_dist = best_dist;
            best_dist = dist;
            best = i;
        }} else if (room == best_room && dist < second_dist) {{
            second = i;
            second_dist = dist;
        }}
    }}

    let d = normalize(world_dir);
    let probe_lod = clamp(roughness * PROBE_ROUGHNESS_MIPS, PROBE_MIN_LOD, PROBE_MAX_LOD);
    // TRACED: see `probe_trace` -- exact against the rooms, their doorways and
    // what stands in them -- then coloured by the photographs that saw the hit,
    // each read from its own capture point. See `probe_hit_colour`.
    //
    // A traced hit is the surface actually there, so the brightness
    // normalisation -- a guard against one photograph standing in for a whole
    // room -- stays out of it: `probe_brightness` stays 0.
    //
    // textureSampleLEVEL throughout, never textureSample: the plain form needs
    // implicit derivatives, which WGSL only defines in uniform control flow,
    // and this function is anything but. On Adreno that is not a validation
    // error but undefined behaviour that hung the GPU.
    //
    // THE ARRAY LAYER, not the loop slot, is what a cube is read from: the
    // array holds every probe the level baked, the loop walks only the ones
    // resident near the player. See `ProbeUpload::boxes`.
    // A SURFACE IN A DOORWAY -- the threshold, a jamb, the lintel -- lies in
    // no room, and the only box that contains it is the outdoor volume's,
    // which surrounds the whole level. Traced from there, its reflection met
    // open sky: the marble strip where the hall meets the hallway reflected
    // nothing at all (headset, 2026-09-27 19:09). A face inside a doorway's
    // carve that no room claims is traced from the doorway, which is what
    // `probe_trace` does with a room of -1.
    var trace_room = best_room;
    if (probe_portal_holding(volume_world) >= 0 && (best < 0 || probe_seen_distance(best, d) < 0.0)) {{
        trace_room = -1.0;
    }}
    probe_eye_distance = distance(cam_pos(), frag_pos);
    let hit = probe_trace(world_pos, d, trace_room, roughness);
    if (hit.found) {{
        var col = probe_hit_colour(hit.pos, hit.room, hit.other, roughness, hit.t);
        if (hit.escaped) {{
            col = probe_escape_colour(hit.pos, d, hit.room, hit.other, hit.portal, dir, probe_lod);
        }}
        // ACROSS A DOORWAY'S RIM, the lobe's two parts: what the ray found,
        // and the other side of the rim, weighted by how much of the lobe
        // passes through the opening. See `probe_rim_at`.
        if (hit.rim >= 0.0) {{
            if (hit.rim_went_through) {{
                let wall = probe_hit_colour(hit.rim_pos, hit.rim_room, -1.0, roughness, hit.rim_t);
                col = mix(wall, col, hit.rim);
            }} else {{
                // Traced again from just inside the opening: whatever the
                // doorway shows there, the next room or outdoors.
                let alt = probe_trace(hit.rim_pos, d, -1.0, roughness);
                if (alt.found) {{
                    var beyond = probe_hit_colour(alt.pos, alt.room, alt.other, roughness, hit.rim_t + alt.t);
                    if (alt.escaped) {{
                        beyond = probe_escape_colour(alt.pos, d, alt.room, alt.other, alt.portal, dir, probe_lod);
                    }}
                    col = mix(col, beyond, hit.rim);
                }}
            }}
        }}
        // ACROSS A SOLID PROXY'S OUTLINE, the footprint's two parts: the
        // proxy, and what lies past it, by how much of the footprint the
        // proxy covers. See `probe_proxy_hit`.
        if (hit.edge >= 0) {{
            if (hit.edge_hit) {{
                let past = probe_trace_skipping(world_pos, d, trace_room, roughness, hit.edge);
                if (past.found) {{
                    var beyond = probe_hit_colour(past.pos, past.room, past.other, roughness, past.t);
                    if (past.escaped) {{
                        beyond = probe_escape_colour(past.pos, d, past.room, past.other, past.portal, dir, probe_lod);
                    }}
                    col = mix(beyond, col, hit.edge_cover);
                }}
            }} else {{
                let proxy_col = probe_hit_colour(hit.edge_pos, hit.edge_room, -1.0, roughness, hit.edge_t);
                col = mix(col, proxy_col, hit.edge_cover);
            }}
        }}
        return col;
    }}
    // NOT an early return when nothing contains this surface: a doorway's own
    // jambs and threshold sit in the wall, inside no room, and the portal pass
    // below is what gives them a photograph.
    var own = vec4<f32>(0.0);
    var own_room = -1.0;
    if (best >= 0) {{
        own_room = best_room;
        // HOW MUCH THE NEARER PHOTOGRAPH COUNTS against the runner-up: all of
        // it a band away from the boundary between them, half on it.
        //
        // Picking exactly one made every cell boundary a seam, straight lines
        // meeting at right angles -- light SQUARES on a floor or a wall that
        // came and went as residency changed. Blending the two everywhere put
        // every hanging lamp in the room twice. See `PROBE_BLEND_BAND`.
        // `best_dist` and `second_dist` are SQUARED distances.
        var band = 1.0;
        if (second >= 0) {{
            let gap = sqrt(second_dist) - sqrt(best_dist);
            band = 0.5 + 0.5 * smoothstep(0.0, PROBE_BLEND_BAND, gap);
        }}
        // NOT TRACED -- a rough surface, or a ray leaving what is resident --
        // so each photograph is projected onto the room's box, corrected from
        // ITS OWN capture point: which probe is "near" flips on the boundary,
        // and reusing the near one's direction for both made that flip a hard
        // line across the hall (offline_frame, 2026-09-23). See
        // `probe_parallax_direction`.
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
            // The brightness blends with the same weight as the photographs it describes.
            probe_brightness = mix(camera.probe_boxes[second * 3 + 1].w, camera.probe_boxes[best * 3 + 1].w, band);
            own = mix(far, near, band);
        }}
    }}
    return probe_through_portals(own, own_room, select_world, world_pos, d, probe_lod);
}}

// As far as the probe is used at all: `shade_material_env` fades it into the
// lightmap's hemisphere by 0.75. A cut-off below that is a hard edge across
// any material whose roughness map straddles it -- the hallway rock spans
// 0.33-0.59 -- and every pixel past it fell back to the box projection, which
// ignores what stands in the room: at 0.3 the probe behind the pillar showed
// the corner lamp's pool straight through the pillar on distant marble that
// `specular_aa_roughness` had widened to 0.43 (headset, 2026-09-26). Rough
// reflections are soft because the lobe is wide, which the trace now models
// itself (`probe_rim_at`, `probe_hit_lod`), not because they are untraced.
const PROBE_TRACE_MAX_ROUGHNESS: f32 = 0.75;
// How many rooms one reflection may cross: its own and two doorways on.
const PROBE_TRACE_ROOMS: i32 = 3;
// Samples a ray takes inside a model's box looking for the model. See
// `probe_proxy_surface`.
const PROBE_PROXY_SAMPLES: i32 = 4;
// How close to what a photograph saw a hit must lie for that photograph to
// count as having seen it, in metres plus 1% of the distance: half-float
// planes, and texels a few centimetres wide at the far end of a hall.
const PROBE_SEEN_TOLERANCE: f32 = 0.03;

// Where a traced reflection landed. `room`: the room the hit is in. `other`:
// the room across the doorway when the hit is ON the doorway -- a jamb, the
// lintel, the threshold -- whose photographs may see it too; else -1.
struct ProbeHit {{
    pos: vec3<f32>,
    room: f32,
    other: f32,
    found: bool,
    // The ray LEFT THE ROOMS through doorway `portal` at `pos`, into the
    // outdoor volume, which has no walls to hit. See `probe_escape_colour`.
    escaped: bool,
    portal: i32,
    // How far along the ray the hit is, for the blur. See `probe_hit_lod`.
    t: f32,
    // A DOORWAY'S RIM INSIDE THE LOBE, the first one the ray met: how much of
    // the lobe passes through the opening (0..1), or -1 for none. `rim_pos` is
    // on the far side of the rim from where this ray went -- just inside the
    // opening when it hit the wall, on the wall just outside when it went
    // through -- in room `rim_room`, `rim_t` along the ray. See
    // `probe_rim_at` and `probe_environment`.
    rim: f32,
    rim_went_through: bool,
    rim_pos: vec3<f32>,
    rim_room: f32,
    rim_t: f32,
    // A SOLID PROXY'S OUTLINE INSIDE THE FOOTPRINT, the first one the ray
    // passed: which proxy (-1 for none), how much of the footprint it covers
    // there (0..1), whether this ray hit it, and where its outline is along
    // the ray. See `probe_proxy_hit` and `probe_environment`.
    edge: i32,
    edge_cover: f32,
    edge_hit: bool,
    edge_pos: vec3<f32>,
    edge_room: f32,
    edge_t: f32,
}}

// How far the eye is from the surface being shaded, for the footprint of a
// reflection. Set by `probe_environment` before it traces.
var<private> probe_eye_distance: f32 = 1.0;

// What `probe_proxy_hit` found: the nearest entry (3.4e38 for none) and which
// proxy it was, and the first SOLID proxy whose outline the ray passes within
// its footprint of -- see `ProbeHit::edge`.
struct ProbeProxyHit {{
    t: f32,
    index: i32,
    edge: i32,
    edge_cover: f32,
    edge_t: f32,
}}

// THE REFLECTION LOBE'S WIDTH, as the tangent of its half-angle at half
// maximum: GGX's half-vector falls to half its peak at about 0.64 alpha, and
// the reflection turns twice as far. `alpha` is roughness squared. At the
// marble's 0.048 that is a third of a centimetre per metre -- a mirror; at
// the hallway rock's 0.43, a quarter of the distance -- a doorway five metres
// off smears across more than its own width.
const PROBE_LOBE_SPREAD: f32 = 1.3;
fn probe_lobe_tan(roughness: f32) -> f32 {{
    return PROBE_LOBE_SPREAD * roughness * roughness;
}}

// The narrowest lobe, in metres where it meets a wall, worth softening a
// doorway's rim for: a centimetre is under a pixel wherever it is seen.
const PROBE_RIM_MIN_SPREAD: f32 = 0.01;
// How far past a rim the other side is looked up from, so the lookup lands
// clearly on that side: inside the opening, or on the wall beside it.
const PROBE_RIM_STEP: f32 = 0.02;

struct ProbeRim {{
    portal: i32,
    through: f32,
}}

// A DOORWAY'S RIM WITHIN THE LOBE. A ray leaving room `room` through the wall
// at `e` (its `axis` face) carries a lobe `spread` metres across there, and
// where a doorway's rim passes within that, part of the lobe goes through the
// opening and part meets the wall -- the reflection of the doorway should be
// as soft as the surface is rough. One ray decides it all or nothing, which on
// the hallway's rock (roughness 0.43) drew the far doorway as a hard-edged
// strip down the floor (headset, 2026-09-27 01:48).
//
// Returns that doorway and how much of the lobe -- a disc on the wall --
// falls inside the opening, as the product of a smooth coverage across each
// of its two sides. `portal` -1 when no rim is that near: the one ray is then
// the whole answer.
fn probe_rim_at(e: vec3<f32>, room: f32, axis: i32, spread: f32) -> ProbeRim {{
    var out: ProbeRim;
    out.portal = -1;
    out.through = 0.0;
    let n = i32(camera.portal_params.x);
    for (var p = 0; p < n; p = p + 1) {{
        if (i32(camera.probe_portals[p * 3].w) != axis) {{
            continue;
        }}
        let low = camera.probe_portals[p * 3 + 1].w;
        let high = camera.probe_portals[p * 3 + 2].x;
        if (low != room && high != room) {{
            continue;
        }}
        let plo = camera.probe_portals[p * 3].xyz;
        let phi = camera.probe_portals[p * 3 + 1].xyz;
        if (e[axis] < plo[axis] - 1e-3 || e[axis] > phi[axis] + 1e-3) {{
            continue;
        }}
        // How far inside the opening, across each of its two sides.
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

// The point just across doorway `p`'s rim from `e`: inside the opening when
// `into` (the ray met the wall), else on the wall just outside it (the ray
// went through) -- past the NEAREST side, which is the rim the lobe spans.
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

// THE BLUR AT A TRACED HIT, as the probe at `from` must be read to show it.
//
// A reflection's lobe spreads with distance: `t` metres along the ray it
// covers a disc `t * probe_lobe_tan(roughness)` across, so a rough floor
// reflects the foot of a wall sharply and its top softly -- the contact
// hardening every glossy reflection has. A photograph taken `t_probe` from
// the hit sees that disc under a smaller or larger angle than the surface
// does, and its prefiltered levels are laid out by angle: level `n` is the
// lobe of roughness `n / PROBE_ROUGHNESS_MIPS`. So the level is the
// roughness whose lobe, from the photograph, covers the same disc:
// `r * sqrt(t / t_probe)`, since the lobe grows as roughness squared.
fn probe_hit_lod(roughness: f32, t: f32, t_probe: f32) -> f32 {{
    let r = roughness * sqrt(max(t, 0.0) / max(t_probe, 0.05));
    return clamp(r * PROBE_ROUGHNESS_MIPS, PROBE_MIN_LOD, PROBE_MAX_LOD);
}}

// How far from photograph `slot`'s capture point, along `v` (any length), the
// surface it saw in that direction lies. The depth texel is that surface's
// PLANE, so this is exact for any direction, not only the texel's centre.
// 3.4e38: sky, or a plane the direction never meets. -1: no depth baked.
fn probe_seen_distance(slot: i32, v: vec3<f32>) -> f32 {{
    let t = textureSampleLevel(probe_depth, probe_depth_samp, v, i32(camera.probe_boxes[slot * 3].w), 0.0);
    if (dot(t.xyz, t.xyz) < 0.25) {{
        return select(-1.0, 3.4e38, t.w > 0.5);
    }}
    let dn = dot(normalize(v), t.xyz);
    return select(3.4e38, max(t.w / dn, 0.0), dn < -1e-4);
}}

// How far IN FRONT of what photograph `slot` saw the point `p` is: positive
// where it saw past `p`, negative where `p` is hidden behind something, zero
// where `p` is the very surface it photographed. A photograph with no depth
// vouches for nothing.
fn probe_clearance(slot: i32, p: vec3<f32>) -> f32 {{
    let v = p - camera.probe_boxes[slot * 3].xyz;
    let s = probe_seen_distance(slot, v);
    return select(s - length(v), -3.4e38, s < 0.0);
}}

// A resident slot photographing `room`, or -1. Any one serves for the room's
// BOX: every cell of a room carries the room's box.
fn probe_room_slot(room: f32) -> i32 {{
    let count = i32(camera.probe_params.x);
    for (var i = 0; i < count; i = i + 1) {{
        if (camera.probe_boxes[i * 3 + 2].w == room) {{
            return i;
        }}
    }}
    return -1;
}}

// `v` rotated by the unit quaternion `q`.
fn probe_quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {{
    let t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}}

// The nearest point, between `t0` and `t1` along the ray, where it enters
// something standing in `room` -- a pillar, a lamp; 3.4e38 where nothing is.
// See `ProbeProxy`. A ray starting on or inside a proxy (the proxy's own
// surface reflecting) leaves it rather than hitting it.
fn probe_proxy_hit(o: vec3<f32>, d: vec3<f32>, room: f32, t0: f32, t1: f32, skip: i32, lobe: f32) -> ProbeProxyHit {{
    var out: ProbeProxyHit;
    out.t = 3.4e38;
    out.index = -1;
    out.edge = -1;
    out.edge_cover = 0.0;
    out.edge_t = 0.0;
    var best = 3.4e38;
    let n = i32(camera.proxy_params.x);
    for (var i = 0; i < n; i = i + 1) {{
        if (camera.probe_proxies[i * 3].w != room || i == skip) {{
            continue;
        }}
        let q = camera.probe_proxies[i * 3 + 2];
        let qi = vec4<f32>(-q.xyz, q.w);
        let lo = probe_quat_rotate(qi, o - camera.probe_proxies[i * 3].xyz);
        let ld = probe_quat_rotate(qi, d);
        let half = camera.probe_proxies[i * 3 + 1].xyz;
        let inv = 1.0 / select(vec3<f32>(1e-9), ld, abs(ld) > vec3<f32>(1e-9));
        let a = (-half - lo) * inv;
        let b = (half - lo) * inv;
        let lows = min(a, b);
        let highs = max(a, b);
        let near = max(max(lows.x, lows.y), lows.z);
        let far = min(min(highs.x, highs.y), highs.z);
        // HOW NEAR THE OUTLINE, for a solid proxy -- a brush piece, the pillar.
        // `far - near` is how long the ray spends inside the box, and turned
        // sideways across the edge where the entering and leaving faces meet
        // it is how far inside the outline the ray passes (negative: how far
        // outside). One ray per pixel makes that outline a hard step that
        // crawls as the head moves -- the column's edge rippling in the back
        // wall's reflection (headset, 2026-09-27) -- so within the footprint
        // (a pixel's width there, or the lobe's, whichever is wider) it is
        // blended instead. The first such outline along the ray is kept.
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
            let footprint = max(t_edge * lobe, pixel_footprint * (1.0 + t_edge / max(probe_eye_distance, 0.05)));
            if (abs(inside) < footprint && t_edge > t0 && t_edge < t1 && far > t0 + 1e-3) {{
                out.edge = i;
                out.edge_cover = smoothstep(-footprint, footprint, inside);
                out.edge_t = t_edge;
            }}
        }}
        // `far` past the origin: the ray is headed into the box, or starts in
        // it. A ray starting ON a proxy's surface and leaving it has `far` at
        // the origin, and is the proxy's own reflection, not a hit.
        if (near <= far && far > t0 + 1e-3 && near < min(t1, best)) {{
            let t_in = max(near, t0);
            if (camera.probe_proxies[i * 3 + 1].w < 0.5) {{
                // A brush piece IS its box. A ray starting INSIDE one is at it
                // already: an MSAA edge pixel of the ceiling where the pillar
                // meets it is shaded at a centre past the ceiling's edge, inside
                // the pillar, and skipping the pillar from there showed the far
                // end of the hall in a line along the junction (offline_frame,
                // 2026-09-27).
                best = t_in;
                out.index = i;
            }} else {{
                let t_surface = probe_proxy_surface(o, d, t_in, min(far, t1), room, camera.probe_proxies[i * 3].xyz);
                if (t_surface < best) {{
                    best = t_surface;
                    out.index = i;
                }}
            }}
        }}
    }}
    out.t = best;
    return out;
}}

// WHERE INSIDE A MODEL'S BOX the model itself is, between `t_in` and `t_out`;
// 3.4e38 where the ray only passes through air inside the bounds.
//
// A model's box is only its bounds: a hanging lamp's is 1.4 m of mostly air
// around a cord, and stopping at it drew a box of displaced ceiling around
// every lamp's reflection. The room's two photographs nearest the object have
// seen it from close by; a point is empty where EITHER saw past it, and the
// first point neither did is the object. Short and rare -- only rays that
// enter a model's box pay for it.
fn probe_proxy_surface(o: vec3<f32>, d: vec3<f32>, t_in: f32, t_out: f32, room: f32, centre: vec3<f32>) -> f32 {{
    var s0 = -1;
    var s1 = -1;
    var d0 = 3.4e38;
    var d1 = 3.4e38;
    let count = i32(camera.probe_params.x);
    for (var i = 0; i < count; i = i + 1) {{
        if (camera.probe_boxes[i * 3 + 2].w != room) {{
            continue;
        }}
        let v = camera.probe_boxes[i * 3].xyz - centre;
        let dd = dot(v, v);
        if (dd < d0) {{
            s1 = s0;
            d1 = d0;
            s0 = i;
            d0 = dd;
        }} else if (dd < d1) {{
            s1 = i;
            d1 = dd;
        }}
    }}
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

// The doorway the exit point `e` of `room` lies in, crossed along `axis`; -1
// where `e` is on a wall. The portal's box is the carve through the wall, so
// `e` -- on the room's face -- lies inside it exactly when it is in the opening.
fn probe_portal_at(e: vec3<f32>, room: f32, axis: i32) -> i32 {{
    let n = i32(camera.portal_params.x);
    for (var p = 0; p < n; p = p + 1) {{
        if (i32(camera.probe_portals[p * 3].w) != axis) {{
            continue;
        }}
        let low = camera.probe_portals[p * 3 + 1].w;
        let high = camera.probe_portals[p * 3 + 2].x;
        if (room != low && room != high) {{
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

// The doorway whose carve holds `p` (2 cm of slack for MSAA), or -1.
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

// WHERE A REFLECTED RAY REALLY HITS, found exactly and without a single
// texture read.
//
// A room's probe box IS its interior, so its walls, floor and ceiling are the
// box's faces: the ray leaves the box where it meets them. Where it leaves
// through a doorway -- the exit lies in a portal's carve -- it either strikes
// the opening's sides (the jambs, the lintel, the threshold) or crosses into
// the next room's box and carries on there. What stands INSIDE a room is its
// proxies; see `probe_proxy_hit`.
//
// This replaced a walk against the probes' own depth, which could only know
// what the photographs had seen: a pillar on the hall's axis showed both of
// them only its front or back, its sides were seen by nothing, and rays aimed
// at them went through it to the lamp beyond (headset, 2026-09-26). It also
// cost 12-30 cube reads per pixel. This is slab tests.
//
// Rough surfaces skip it: their lobe is too wide for one hit point to mean
// anything, and the box projection is the better average.
fn probe_trace(world_pos: vec3<f32>, d: vec3<f32>, room: f32, roughness: f32) -> ProbeHit {{
    return probe_trace_skipping(world_pos, d, room, roughness, -1);
}}

// `probe_trace`, as though proxy `skip` were not there: the far side of an
// outline the lobe straddles. See `ProbeHit::edge`.
fn probe_trace_skipping(world_pos: vec3<f32>, d: vec3<f32>, room: f32, roughness: f32, skip: i32) -> ProbeHit {{
    var hit: ProbeHit;
    hit.pos = vec3<f32>(0.0);
    hit.room = -1.0;
    hit.other = -1.0;
    hit.found = false;
    hit.escaped = false;
    hit.portal = -1;
    hit.t = 0.0;
    hit.rim = -1.0;
    hit.rim_went_through = false;
    hit.rim_pos = vec3<f32>(0.0);
    hit.rim_room = -1.0;
    hit.rim_t = 0.0;
    hit.edge = -1;
    hit.edge_cover = 0.0;
    hit.edge_hit = false;
    hit.edge_pos = vec3<f32>(0.0);
    hit.edge_room = -1.0;
    hit.edge_t = 0.0;
    let lobe = probe_lobe_tan(roughness);
    // `portal_params.y`: switched off by `perf_ab` measuring it.
    if (roughness > PROBE_TRACE_MAX_ROUGHNESS || camera.portal_params.y > 0.5) {{
        return hit;
    }}
    let moving = abs(d) > vec3<f32>(1e-6);
    let inv = select(vec3<f32>(3.4e38), 1.0 / d, moving);
    var cur = room;
    var t0 = 0.0;
    // THE RAY STARTS INSIDE ITS ROOM. A wall's own fragment lies on the box,
    // and with MSAA an edge pixel is shaded at its centre, which can lie just
    // past it. Traced from there, the hit came out a hair behind the ceiling,
    // where no photograph saw it, and took another probe's colour: a one-pixel
    // line along every junction (offline_frame, 2026-09-27). Held into the box
    // once, and every point below measured from the held origin.
    var o = world_pos;
    var in_room = cur >= 0.0;
    if (!in_room) {{
        // A SURFACE IN A DOORWAY -- a jamb, the threshold, the lintel -- stands
        // in no room: the carve through the wall holds it. Its reflection
        // starts in the opening, and either meets the opening's other sides or
        // leaves it into the room on whichever side it is heading for. Left to
        // the untraced blend, these were the frames of flat, mismatched
        // reflection around every doorway (headset, 2026-09-26).
        let p = probe_portal_holding(world_pos);
        if (p < 0) {{
            return hit;
        }}
        let axis = i32(camera.probe_portals[p * 3].w);
        let plo = camera.probe_portals[p * 3].xyz;
        let phi = camera.probe_portals[p * 3 + 1].xyz;
        let low = camera.probe_portals[p * 3 + 1].w;
        let high = camera.probe_portals[p * 3 + 2].x;
        o = clamp(world_pos, plo, phi);
        var side = select(vec3<f32>(3.4e38), max((phi - o) * inv, (plo - o) * inv), moving);
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
        let t_enter = select(max((face[axis] - o[axis]) * inv[axis], 0.0), max((wall_far - o[axis]) * inv[axis], 0.0), escapes);
        if (t_side < t_enter) {{
            hit.pos = o + d * t_side;
            hit.room = low;
            hit.other = high;
            hit.found = true;
            hit.t = t_side;
            return hit;
        }}
        if (escapes) {{
            hit.pos = o + d * t_enter;
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
            o = clamp(world_pos, lo, hi);
        }}
        let start = clamp(o + d * t0, lo, hi);
        let far = select(vec3<f32>(3.4e38), max((hi - start) * inv, (lo - start) * inv), moving);
        var axis = 2;
        var t_exit = far.z;
        if (far.x <= far.y && far.x <= far.z) {{
            axis = 0;
            t_exit = far.x;
        }} else if (far.y <= far.z) {{
            axis = 1;
            t_exit = far.y;
        }}
        t_exit = t0 + t_exit;
        let proxy = probe_proxy_hit(o, d, cur, t0, t_exit, skip, lobe);
        let t_obj = proxy.t;
        if (hit.edge < 0 && proxy.edge >= 0) {{
            hit.edge = proxy.edge;
            hit.edge_cover = proxy.edge_cover;
            hit.edge_hit = t_obj < t_exit && proxy.index == proxy.edge;
            hit.edge_pos = o + d * proxy.edge_t;
            hit.edge_room = cur;
            hit.edge_t = proxy.edge_t;
        }}
        if (t_obj < t_exit) {{
            hit.pos = o + d * t_obj;
            hit.room = cur;
            hit.found = true;
            hit.t = t_obj;
            return hit;
        }}
        let e = o + d * t_exit;
        let p = probe_portal_at(e, cur, axis);
        // The first doorway rim within the lobe, whichever side of it this
        // ray lands on. See `probe_rim_at`.
        if (hit.rim < 0.0 && t_exit * lobe > PROBE_RIM_MIN_SPREAD) {{
            let rim = probe_rim_at(e, cur, axis, t_exit * lobe);
            if (rim.portal >= 0) {{
                hit.rim = rim.through;
                hit.rim_went_through = p >= 0;
                hit.rim_pos = probe_rim_point(e, rim.portal, axis, p < 0);
                hit.rim_room = cur;
                hit.rim_t = t_exit;
            }}
        }}
        if (p < 0) {{
            // The room's own wall, floor or ceiling.
            hit.pos = e;
            hit.room = cur;
            hit.found = true;
            hit.t = t_exit;
            return hit;
        }}
        let low = camera.probe_portals[p * 3 + 1].w;
        let high = camera.probe_portals[p * 3 + 2].x;
        let other = select(low, high, cur == low);
        // Through the opening. Its sides, along the two axes across it, are
        // the jambs, the lintel and the threshold.
        let plo = camera.probe_portals[p * 3].xyz;
        let phi = camera.probe_portals[p * 3 + 1].xyz;
        var side = select(vec3<f32>(3.4e38), max((phi - e) * inv, (plo - e) * inv), moving);
        side[axis] = 3.4e38;
        let t_side = t_exit + min(min(side.x, side.y), side.z);
        let oslot = probe_room_slot(other);
        if (oslot < 0) {{
            // A doorway onto nothing resident: the untraced path answers.
            return hit;
        }}
        // OUT OF THE ROOMS. A photograph with no depth is the outdoor volume,
        // whose box stands in for the sky dome: nothing to walk into. The
        // jambs end at the WALL's far face -- see `ProbePortal::wall` -- and
        // past it the ray has escaped. Traced into that box instead, every
        // reflection of the front door came out a black patch: the outdoor
        // photograph leaves its sky to the renderer, which occludes it by the
        // indoor surface's sky visibility (headset, 2026-09-27).
        let escapes = probe_seen_distance(oslot, d) < 0.0;
        let wall_far = select(camera.probe_portals[p * 3 + 2].y, camera.probe_portals[p * 3 + 2].z, d[axis] > 0.0);
        let face = select(camera.probe_boxes[oslot * 3 + 2].xyz, camera.probe_boxes[oslot * 3 + 1].xyz, d > vec3<f32>(0.0));
        let t_enter = select(
            max((face[axis] - o[axis]) * inv[axis], t_exit),
            max((wall_far - o[axis]) * inv[axis], t_exit),
            escapes
        );
        if (t_side < t_enter) {{
            hit.pos = o + d * t_side;
            hit.room = cur;
            hit.other = other;
            hit.found = true;
            hit.t = t_side;
            return hit;
        }}
        if (escapes) {{
            hit.pos = o + d * t_enter;
            hit.room = cur;
            hit.other = other;
            hit.portal = p;
            hit.escaped = true;
            hit.found = true;
            hit.t = t_enter;
            return hit;
        }}
        cur = other;
        t0 = t_enter;
    }}
    return hit;
}}

// How far past a doorway what a reflection sees out there is taken to be when
// nothing nearer is found: the sky, and ground past the photograph's depth.
// Sets only the parallax between the photograph's capture point and the ray;
// the sky is at infinity either way.
const PROBE_ESCAPE_DISTANCE: f32 = 30.0;
// Steps the escape march takes, doubling from half a metre: out to 32 m.
const PROBE_ESCAPE_STEPS: i32 = 7;
const PROBE_ESCAPE_BISECTIONS: i32 = 3;

// WHERE A RAY THAT LEFT THE ROOMS MEETS THE OUTDOORS, from what photograph
// `slot` saw through the opening. Marched outward from the doorway's outer
// face `e` in doubling steps until the ray passes behind the photographed
// surface -- the ground, the hill -- then bisected onto it.
//
// It used to be a fixed 30 m. The marble ceiling's reflection of the front
// door looks DOWN through it at grass a few metres outside, and read the
// photograph toward a point 30 m on and 13 m underground; the floor's looks up
// at the hill, and read it past the hill (headset, 2026-09-27: the door
// reflections showed neither terrain nor sky). Sky never stops the march,
// which then lands at `PROBE_ESCAPE_DISTANCE` as before.
fn probe_escape_hit(e: vec3<f32>, d: vec3<f32>, slot: i32) -> vec3<f32> {{
    let c = camera.probe_boxes[slot * 3].xyz;
    var t_lo = 0.0;
    var t_hi = PROBE_ESCAPE_DISTANCE;
    var t = 0.5;
    var met = false;
    for (var k = 0; k < PROBE_ESCAPE_STEPS; k = k + 1) {{
        let v = e + d * t - c;
        let seen = probe_seen_distance(slot, v);
        if (seen >= 0.0 && seen < length(v)) {{
            t_hi = t;
            met = true;
            break;
        }}
        t_lo = t;
        t = t * 2.0;
    }}
    if (met) {{
        for (var k = 0; k < PROBE_ESCAPE_BISECTIONS; k = k + 1) {{
            let tm = 0.5 * (t_lo + t_hi);
            let v = e + d * tm - c;
            let seen = probe_seen_distance(slot, v);
            if (seen >= 0.0 && seen < length(v)) {{
                t_hi = tm;
            }} else {{
                t_lo = tm;
            }}
        }}
    }}
    return e + d * t_hi;
}}

// THE COLOUR OF A REFLECTION THAT LEFT THE ROOMS through doorway `p` at `e`,
// heading `d`: what is out there is far, so it is read by direction, from a
// photograph that can SEE out along it.
//
// First choice, `room`'s own photographs whose view toward the far point
// passes through the same opening: they have the sky baked in, the panorama
// itself rather than its harmonics. Failing that, the outdoor volume's
// (`other`), with the sky filling what it left to the renderer -- unoccluded,
// because the trace has just proven the ray reaches it. `sky_dir` is `d` in the
// frame `environment_radiance` expects.
fn probe_escape_colour(e: vec3<f32>, d: vec3<f32>, room: f32, other: f32, p: i32, sky_dir: vec3<f32>, lod: f32) -> vec4<f32> {{
    let axis = i32(camera.probe_portals[p * 3].w);
    let plo = camera.probe_portals[p * 3].xyz - vec3<f32>(1e-3);
    let phi = camera.probe_portals[p * 3 + 1].xyz + vec3<f32>(1e-3);
    let far_point = e + d * PROBE_ESCAPE_DISTANCE;
    var best = -1;
    var best_d = 3.4e38;
    let count = i32(camera.probe_params.x);
    for (var i = 0; i < count; i = i + 1) {{
        if (camera.probe_boxes[i * 3 + 2].w != room) {{
            continue;
        }}
        let c = camera.probe_boxes[i * 3].xyz;
        let to = far_point - c;
        if (abs(to[axis]) < 1e-5) {{
            continue;
        }}
        // Where this photograph's own line of sight to the far point crosses
        // the doorway's plane -- inside the opening, or into the wall.
        let s = (e[axis] - c[axis]) / to[axis];
        if (s <= 0.0 || s >= 1.0) {{
            continue;
        }}
        let q = c + to * s;
        let within = (q >= plo) & (q <= phi);
        if (!((within.x || axis == 0) && (within.y || axis == 1) && (within.z || axis == 2))) {{
            continue;
        }}
        let v = c - e;
        if (dot(v, v) < best_d) {{
            best_d = dot(v, v);
            best = i;
        }}
    }}
    if (best >= 0) {{
        let c = camera.probe_boxes[best * 3].xyz;
        return textureSampleLevel(
            probe_cube, probe_samp, probe_escape_hit(e, d, best) - c, i32(camera.probe_boxes[best * 3].w), lod
        );
    }}
    // The outdoor photograph has no depth to march. A ray heading down meets
    // the ground near the building at about the doorway's own floor level.
    var t_out = PROBE_ESCAPE_DISTANCE;
    if (d.y < -1e-4) {{
        t_out = clamp((plo.y - e.y) / d.y, 0.0, PROBE_ESCAPE_DISTANCE);
    }}
    let oslot = probe_room_slot(other);
    let out = textureSampleLevel(
        probe_cube, probe_samp, e + d * t_out - camera.probe_boxes[oslot * 3].xyz, i32(camera.probe_boxes[oslot * 3].w), lod
    );
    let a = clamp(out.a, 0.0, 1.0);
    return vec4<f32>(out.rgb * a + environment_radiance(sky_dir) * (1.0 - a), 1.0);
}}

// THE COLOUR AT A TRACED HIT, from the photographs of its room that SAW it.
//
// The two resident photographs of the hit's room nearest the hit -- not the
// ones nearest the reflecting surface: the back wall's pool is best seen from
// the cell at the back, whichever cell the floor pixel reflecting it belongs
// to. Each counts by how well it saw the hit (its surface there within
// `PROBE_SEEN_TOLERANCE`) and by inverse square distance, so the two blend
// continuously as the hit moves between them. Neither seeing it -- the side
// of a pillar both photographed end-on -- falls to the one whose surface lies
// nearest the hit, which shows the thing standing closest to it.
//
// Each is read from its OWN capture point toward the hit, which is what makes
// the reflection parallax-correct everywhere rather than on a box.
fn probe_hit_colour(h: vec3<f32>, room: f32, other: f32, roughness: f32, t: f32) -> vec4<f32> {{
    var s0 = -1;
    var s1 = -1;
    var d0 = 3.4e38;
    var d1 = 3.4e38;
    let count = i32(camera.probe_params.x);
    for (var i = 0; i < count; i = i + 1) {{
        let r = camera.probe_boxes[i * 3 + 2].w;
        if (r != room && r != other) {{
            continue;
        }}
        let v = camera.probe_boxes[i * 3].xyz - h;
        let dd = dot(v, v);
        if (dd < d0) {{
            s1 = s0;
            d1 = d0;
            s0 = i;
            d0 = dd;
        }} else if (dd < d1) {{
            s1 = i;
            d1 = dd;
        }}
    }}
    let c0 = probe_clearance(s0, h);
    let tol0 = PROBE_SEEN_TOLERANCE + 0.01 * sqrt(d0);
    var w0 = (1.0 - smoothstep(tol0, 2.0 * tol0, abs(c0))) / (d0 + 1.0);
    var w1 = 0.0;
    var c1 = -3.4e38;
    if (s1 >= 0) {{
        c1 = probe_clearance(s1, h);
        let tol1 = PROBE_SEEN_TOLERANCE + 0.01 * sqrt(d1);
        w1 = (1.0 - smoothstep(tol1, 2.0 * tol1, abs(c1))) / (d1 + 1.0);
    }}
    if (w0 + w1 < 1e-6) {{
        w0 = select(1.0, 0.0, s1 >= 0 && abs(c1) < abs(c0));
        w1 = 1.0 - w0;
    }}
    // Each photograph read at the blur the hit's distance calls for, from its
    // own distance to the hit. See `probe_hit_lod`.
    var col = textureSampleLevel(
        probe_cube, probe_samp, h - camera.probe_boxes[s0 * 3].xyz, i32(camera.probe_boxes[s0 * 3].w),
        probe_hit_lod(roughness, t, sqrt(d0))
    ) * w0;
    if (w1 > 0.0) {{
        col += textureSampleLevel(
            probe_cube, probe_samp, h - camera.probe_boxes[s1 * 3].xyz, i32(camera.probe_boxes[s1 * 3].w),
            probe_hit_lod(roughness, t, sqrt(d1))
        ) * w1;
    }}
    return col / (w0 + w1);
}}

// THE DIRECTION TO LOOK UP PROBE `slot` IN for a reflection leaving `world_pos`
// along `d`: the reflected ray is run out to the probe's box and the cube is
// read towards that hit FROM THE PROBE'S OWN CAPTURE POINT. Per slot, because
// two blended photographs were taken from two different places.
fn probe_parallax_direction(world_pos: vec3<f32>, d: vec3<f32>, slot: i32) -> vec3<f32> {{
    let centre = camera.probe_boxes[slot * 3].xyz;
    let lo = camera.probe_boxes[slot * 3 + 1].xyz;
    let hi = camera.probe_boxes[slot * 3 + 2].xyz;
    // CLAMPED INTO THE BOX BEFORE THE PARALLAX, and this is load-bearing.
    //
    // The correction shoots the reflected ray from this fragment to the room
    // box and looks the cube up in that direction from the capture point. It
    // is only defined for a point INSIDE the box -- and a wall fragment sits
    // exactly ON the box surface, because a room's probe box IS its interior.
    //
    // With MSAA a pixel on a polygon edge is shaded at its centre, which can
    // lie outside the polygon, and the interpolated position is then
    // EXTRAPOLATED past the box. For the axis it escaped on, `t_hi` and `t_lo`
    // take the same sign, `dist` comes out NEGATIVE, the guard below fails,
    // and that one pixel falls back to the raw reflection direction while
    // every neighbour around it got a corrected one. A single-pixel
    // discontinuity along an edge, wherever the probe has something bright to
    // reflect -- which is why it showed up in the LIT part of the room and
    // nowhere else (headset, 2026-09-21).
    //
    // Clamping says the fragment is on the box surface, which is where the
    // wall actually is. It costs one clamp and removes the failure rather than
    // widening the window in which it does not happen -- two margin widenings
    // tried that and neither worked.
    //
    // NOTE this is the other half of the per-face fix above: selection now
    // uses the face centre so the whole face agrees on WHICH probe, and this
    // makes the SAMPLING of it defined for every fragment of that face.
    let parallax_pos = clamp(world_pos, lo, hi);
    // A zero component would divide by zero; "never reaches that pair of
    // planes" is the right answer and a huge number gives it.
    let inv = select(vec3<f32>(3.4e38), 1.0 / d, abs(d) > vec3<f32>(1e-6));
    let t_hi = (hi - parallax_pos) * inv;
    let t_lo = (lo - parallax_pos) * inv;
    let furthest = max(t_hi, t_lo);
    let dist = min(min(furthest.x, furthest.y), furthest.z);
    // `>= 0`, NOT `> 0`. The clamp above puts an edge fragment EXACTLY on a
    // wall plane, and when its reflection leaves through that wall the exit
    // distance is exactly zero -- the reflected point is the fragment itself,
    // a perfectly good answer. `> 0` rejected it, and that one pixel fell back
    // to the uncorrected direction while every neighbour, a hair inside the
    // box, used the corrected one: a single-pixel line of a DIFFERENT part of
    // the photograph along every junction the reflection runs out through.
    // That was the front-of-hall seam -- reproduced off the headset by
    // `quest_app::offline_frame`, gone with MSAA off, flat across every probe
    // factor, present in the probe colour alone (2026-09-23). With the
    // position clamped into the box `dist` cannot be negative; the guard now
    // only keeps a degenerate capture point from normalising zero.
    var sample_dir = d;
    if (dist >= 0.0) {{
        let to_hit = parallax_pos + d * dist - centre;
        if (dot(to_hit, to_hit) > 1e-12) {{
            sample_dir = normalize(to_hit);
        }}
    }}
    return sample_dir;
}}

// HOW FAR A DOORWAY'S BLEND REACHES, in metres: past each face of the opening
// along its axis, and past its edges across it. See `PROBE_PORTAL_FADE` on the
// Rust side, which these are generated from.
const PORTAL_FADE: f32 = {portal_fade:?};
const PORTAL_SIDE_FADE: f32 = {portal_side_fade:?};

// The brightness `probe_volume_sample` last read, for the normalisation.
var<private> volume_sample_brightness: f32 = 0.0;

// The photograph of room `room` nearest `select_world`, parallax corrected
// from its own capture point -- or alpha -1 when no resident slot belongs to
// that room. Alpha is coverage (0..1) everywhere else, so -1 cannot be a
// real sample.
fn probe_volume_sample(
    room: f32, select_world: vec3<f32>, world_pos: vec3<f32>, d: vec3<f32>, lod: f32,
) -> vec4<f32> {{
    let count = i32(camera.probe_params.x);
    var pick = -1;
    var pick_dist = 1e30;
    for (var i = 0; i < count; i = i + 1) {{
        if (camera.probe_boxes[i * 3 + 2].w != room) {{
            continue;
        }}
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

// THROUGH A DOORWAY, THE TWO ROOMS' PHOTOGRAPHS HAND OVER.
//
// A surface takes its room's photograph, and the room comes from the face
// (`probe_volume_pos`) -- so without this, the change from one room's
// reflection to the next is a hard step at the threshold, and the doorway's
// own jambs and floor, which lie in the wall and so in NO room, took the
// OUTDOOR photograph: sky reflected in an interior door frame.
//
// The doorways are not authored. The bake finds them as carves thin on one
// axis (`room_graph::doorways_from_scene`) and names the room either side.
// Across the opening the weight runs smoothly from the low side's room to the
// high side's, over its thickness plus `PORTAL_FADE` each way, and fades out
// `PORTAL_SIDE_FADE` past its edges -- so every surface near the door, on
// either side and in the jambs, reads one continuous function of position.
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
        // How far outside the opening's edges, ACROSS it.
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
            // One side is this surface's own room: blend toward the other.
            let other_room = select(low_room, high_room, own_room == low_room);
            let w_other = reach * select(1.0 - w_high, w_high, own_room == low_room);
            let other = probe_volume_sample(other_room, select_world, world_pos, d, lod);
            if (other.a < 0.0) {{
                return own;
            }}
            probe_brightness = mix(probe_brightness, volume_sample_brightness, w_other);
            return mix(own, other, w_other);
        }}
        // In the wall between them -- the jambs, the threshold -- neither side
        // is this surface's room, so the doorway answers outright.
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

// How much of the sky a ground plane bounces back up. Grass and bare earth sit
// around a quarter; it is an approximation standing in for a term the sky
// panorama does not carry.
const GROUND_ALBEDO: f32 = 0.28;

// The radiance arriving from a direction, for REFLECTIONS.
//
// `sky_irradiance` describes the SKY. Evaluated downward it returns almost
// nothing, because there is no sky down there -- and that is exactly where a
// wall's mirror direction points when you look along it. Used raw, an outdoor
// wall seen at a grazing angle reflects blackness, and once the shading became
// energy-conserving the diffuse gave up its light to pay for that blackness.
// The result was a dark band that slid along every wall as you moved, which
// reads as a shadow that is not there.
//
// What a wall actually sees looking down outdoors is GROUND, lit by the same
// sky. Approximating it as the sky's downward irradiance times a ground albedo
// costs one extra evaluation and removes the band.
//
// Only for the specular environment. The diffuse ambient is evaluated along the
// surface NORMAL, which points away from the ground, and adding a ground term
// there would light the undersides of things that see no ground at all.
fn environment_radiance(dir: vec3<f32>) -> vec3<f32> {{
    let sky = sky_irradiance(dir);
    // The ground term is the SAME for every fragment in the frame -- it depends
    // only on the sky. Evaluating nine harmonics per pixel to recompute a
    // constant was pure waste on a fill-bound GPU, so the CPU works it out once
    // and sends it in `probe_params.yzw`.
    let ground = camera.probe_params.yzw * GROUND_ALBEDO;
    // A soft horizon rather than a step: a hard switch puts a visible line
    // across every reflective surface at eye level.
    return mix(ground, sky, smoothstep(-0.15, 0.15, dir.y));
}}

// THE PROBE REFLECTION FOR THE HALF-RESOLUTION PASS TO STORE: exactly what
// `shade_material_env` reads from `probe_environment` for this surface, with
// its brightness normalisation already applied -- the pass has the lightmap
// that needs -- and premultiplied by its coverage, so the brush shader can
// filter it across texels. Same arguments as `shade_material_env`, minus what
// only the lights need. See `brush_pipeline::probe_pass`.
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
    let probe = probe_environment(world_pos, refl, roughness, probe_select_pos);
    let occ = clamp(ao, 0.0, 1.0) * clamp(sky_vis, 0.0, 1.0);
    let ambient_here = dot(env + sky_irradiance(n) * occ, vec3<f32>(0.2126, 0.7152, 0.0722));
    let probe_scale = select(
        1.0,
        clamp(ambient_here / max(probe_brightness, 1e-4), PROBE_NORMALISATION_FLOOR, 1.0),
        PROBE_NORMALISATION && probe_brightness > 0.0,
    );
    let a = clamp(probe.a, 0.0, 1.0);
    return vec4<f32>(probe.rgb * probe_scale * a, a);
}}

fn shade_material_env(
    world_pos: vec3<f32>,
    n: vec3<f32>,
    roughness: f32,
    ao: f32,
    sky_vis: f32,
    env: vec3<f32>,
    // The baked bounce DIRECTION: xyz is a unit vector encoded as `v * 2 - 1`,
    // w is how directional that light is. w = 0 means "from everywhere", and
    // then this behaves exactly as the flat term it replaced.
    env_dir: vec4<f32>,
    // The surface's own colour. Applied to the DIFFUSE terms only -- see
    // `LightSplit`. Pass white to get the old behaviour, where the caller
    // multiplies everything afterwards.
    albedo: vec3<f32>,
    // See `probe_environment`: the point the PROBE is chosen from. Pass the
    // fragment's own position for the previous behaviour.
    probe_select_pos: vec3<f32>,
    // THE GEOMETRIC NORMAL, for the Fresnel and the reflection vector only.
    //
    // Fresnel is `pow(1 - dot(n, view), 5)`, and that fifth power amplifies
    // whatever jitter is in `n` enormously near a grazing angle -- which is
    // where a distant surface is seen from. Fed the normal-MAPPED normal it
    // aliases: a minified normal map jitters per pixel, the fifth power turns
    // that into a large swing, and the swing shows up as a dotted line along
    // room seams. Measured in the lighting-sources view: the BAKED channel
    // drops and the PROBE channel rises at those pixels by matching amounts
    // while DIRECT does not move at all -- and Fresnel is the one term baked
    // and probe share and direct does not touch (headset, 2026-09-22).
    //
    // Fresnel is a low-frequency function of viewing angle; it does not want
    // per-texel normal detail and is not improved by it. Diffuse and direct
    // keep the mapped normal, which is where the detail belongs.
    //
    // The SSR path in the brush shader already does this -- it blends toward
    // the geometric normal as roughness rises -- so this is the same
    // correction applied to the environment term rather than a new idea.
    //
    // Callers with no separate geometric normal pass `n` and get the previous
    // behaviour exactly.
    geom_n: vec3<f32>,
) -> vec3<f32> {{
    let view_dir = normalize(cam_pos() - world_pos);
    let r = clamp(roughness, 0.04, 1.0);
    // Blinn-Phong has an exponent where a PBR model has a roughness, so the two
    // are bridged by the usual mapping: alpha = r^2, exponent = 2/alpha^2 - 2.
    // Exact enough for a preview and monotonic, which is what matters -- a
    // rougher material must never come out shinier.
    // Distance to the nearest punctual light, for the sphere-light widening.
    // Computed before the loop because the roughness it feeds is per surface,
    // not per light: one lobe, sized by whatever is actually lighting this spot.
    var light_dist = 1e9;
    // FIELDS, NOT THE WHOLE STRUCT.
    //
    // `let li = lights.lights[i]` copied all sixteen components of a Light
    // (four vec4s) to read exactly two of them. On a tile GPU that is register
    // pressure for nothing, and register pressure is occupancy: high GPR use
    // lowers how many waves stay in flight, which is what hides memory
    // latency. A shader that "should" be fast then stalls with no other wave
    // available to fill the gap -- and this frame is fill bound, so per-pixel
    // occupancy is the cost that counts.
    //
    // Indexing a UNIFORM buffer per field is free of the other trap here: the
    // documented Adreno cliff is a dynamically-indexed LOCAL array, which
    // spills to scratch memory. There are none of those in this shader, and a
    // uniform array is not one.
    for (var i: u32 = 0u; i < live_light_count(); i = i + 1u) {{
        if (lights.lights[i].params.z > 1.5) {{ continue; }}
        light_dist = min(light_dist, distance(lights.lights[i].position.xyz, world_pos));
    }}
    let alpha = r * r;
    // Widened by the solid angle the lamp subtends from here. `light_dist` is
    // the nearest light's distance, so a surface right under a fixture gets a
    // broad sheen and one across the room gets a tight one -- which is how a
    // real highlight behaves.
    let widened = clamp(alpha + LIGHT_SOURCE_RADIUS / (2.0 * max(light_dist, 0.05)), alpha, 1.0);
    let shininess = clamp(2.0 / (widened * widened) - 2.0, 1.0, MAX_SHININESS);
    // Rough surfaces spread the same energy over a wider lobe, so the peak is
    // dimmer. Without this, raising roughness only widens the highlight and a
    // matte wall still has a bright spot on it.
    let spec_strength = SPEC_STRENGTH * (1.0 - r);

    let occ = clamp(ao, 0.0, 1.0) * clamp(sky_vis, 0.0, 1.0);
    // Two accumulators from here on: what the surface's colour tints, and what
    // it does not.
    var diffuse = sky_irradiance(n) * occ;
    var specular = vec3<f32>(0.0);

    // ENVIRONMENT SPECULAR -- what actually makes a polished surface read as
    // polished.
    //
    // A punctual lamp can only ever put a small bright spot on a mirror; what
    // a real polished floor mostly shows is the ROOM AND SKY around it. Marble
    // here has a roughness of 0.048 -- near mirror -- and lighting it with
    // lamps alone left it looking like matte stone no matter how the highlight
    // was tuned, because the thing it should be reflecting was never sampled.
    //
    // The sky's spherical harmonics in the MIRROR direction, which is the same
    // approximation the water surface already uses. It is a blurry reflection
    // rather than a sharp one -- an L1 SH cannot resolve a window frame -- but
    // it carries the environment's colour and which side is bright, and that is
    // most of what a floor shows back.
    //
    // Multiplied by the same occlusion as the ambient: a surface sealed inside
    // a room cannot reflect a sky it cannot see, and without this every
    // polished thing indoors glows with outdoor light.
    // HOW MUCH NORMAL DETAIL THE ENVIRONMENT TERM SEES, by roughness.
    //
    // Fresnel is `pow(1 - dot(n, view), 5)`, and that fifth power amplifies
    // whatever jitter is in `n`. Fed a minified normal map at a grazing angle
    // -- which is how a distant surface is seen -- it aliases into a dotted
    // line along room seams.
    //
    // But the mapped normal cannot simply be dropped: at head-on incidence
    // Schlick gives 4% for ANY roughness, so a polished surface only reflects
    // more than a rough one because its normal map tilts it off head-on. Three
    // tests depend on exactly that, correctly.
    //
    // So blend, the way the brush SSR path already does: a polished surface
    // keeps its detail, a rough one takes the geometry's own normal, where the
    // detail was never going to survive minification anyway. `r` here is the
    // roughness that ALREADY carries the normal map's baked-in variance, so a
    // surface whose normals disagree at this distance is treated as rougher
    // and is blended further toward the smooth normal for free.
    let env_n = normalize(mix(n, geom_n, smoothstep(0.1, 0.4, r)));
    let refl = reflect(-view_dir, env_n);
    // Schlick, with the 0.04 normal-incidence reflectance of a dielectric.
    // Smooth surfaces get the full Fresnel sweep; rough ones barely any, which
    // is what stops brick from behaving like a mirror at a glancing angle.
    let cos_v = clamp(dot(env_n, view_dir), 0.0, 1.0);
    // SCHLICK, CAPPED BY ROUGHNESS.
    //
    // The plain form sweeps to 1.0 at a grazing angle whatever the surface is
    // made of, which is true of a MIRROR and not of rubble. Capping the sweep
    // at `1 - roughness` is the usual environment-map correction, and it is
    // what stops a rough stone wall developing a bright rim when you stand
    // beside it.
    //
    // The old form multiplied the uncapped sweep by `(1 - r)` afterwards, which
    // lands in a similar place at the extremes and is wrong in between -- and,
    // more importantly, left no single Fresnel value for the diffuse term below
    // to give up.
    let f0 = 0.04;
    let f_max = max(1.0 - r, f0);
    let fresnel = f0 + (f_max - f0) * pow(1.0 - cos_v, 5.0);
    // Sky and room, each occluded by what can actually reach this surface.
    // The sky term keeps its `occ`; the local term does NOT, because `env` is
    // already the light that got here -- occluding it again would darken a
    // reflection by the very geometry that produced it.
    // THE PROBE FIRST, THE SKY WHERE IT HAS NOTHING TO SAY.
    //
    // A probe knows what is actually around this point -- the far wall, the
    // doorway, the floor -- which the sky's harmonics cannot express and a
    // screen-space march can only find when it happens to be on screen. Its
    // alpha is coverage: 1 where the bake hit geometry, 0 where the ray reached
    // sky. So the two compose without either being baked twice, and an unbaked
    // level returns 0 everywhere and behaves exactly as it did before probes
    // existed.
    //
    // The sky keeps its `occ`: a surface sealed inside a room cannot reflect a
    // sky it cannot see. The probe does NOT, because what it captured is
    // already the light that got there.
    // From the half-resolution pass when this shader was built to read it;
    // `probe_brightness` then stays 0, because the pass has normalised it
    // already. See `probe_env_for_pass`.
    var probe = probe_env_given;
    if (!PROBE_ENV_FROM_PASS) {{
        probe = probe_environment(world_pos, refl, roughness, probe_select_pos);
    }}
    let sky_reflection = environment_radiance(refl) * occ;
    // SPECULAR OCCLUSION.
    //
    // A probe has NO visibility information. It photographed the room from its
    // capture point, so every surface that samples it sees the lamps whether or
    // not this particular spot can actually see them -- which is what puts a
    // crisp bright rectangle on a wall tucked round a corner from the fixture.
    //
    // The standard correction (Lagarde, *Moving Frostbite to PBR*, and the same
    // reasoning in Lagarde's parallax-corrected cubemap notes) reuses the
    // surface's own occlusion, scaled by roughness. At `r = 1` the specular
    // lobe covers the hemisphere, so it should be occluded exactly as much as
    // the diffuse ambient is; at `r = 0` a mirror shows one direction and AO
    // says nothing useful about it. The exponent interpolates between those.
    let a_o = clamp(ao, 0.0, 1.0);
    let spec_occ = clamp(
        pow(cos_v + a_o, exp2(-16.0 * r - 1.0)) - 1.0 + a_o,
        0.0,
        1.0,
    );
    // THE ROUGH END BELONGS TO THE LIGHTMAP, NOT THE PROBE.
    //
    // `env` is the baked indirect irradiance measured AT THIS TEXEL, so unlike
    // the probe it already knows what this spot can see. As roughness rises the
    // specular lobe widens until it is the same hemisphere the diffuse term
    // integrates, and the honest answer becomes the lightmap's own value --
    // `INV_PI` converting irradiance to mean radiance.
    //
    // This was previously ADDED to the probe rather than blended with it, which
    // double-counted: both terms describe the same bounced light, so every
    // rough surface got the room twice, once with local occlusion and once
    // without. That is why brick read as brighter than the light reaching it.
    let indirect_specular = env * INV_PI;
    // THE BASELINE, available at every roughness and on every surface: the
    // lightmap's own answer plus whatever sky this point can actually see.
    //
    // Not a rough-surface special case. A mirror in a room with no probe still
    // reflects SOMETHING, and the best estimate on hand is the light the bake
    // measured here -- which is what this term was already doing before the
    // probe existed. Keeping it as the base is what makes the probe an
    // improvement on the old behaviour rather than a replacement that goes
    // black wherever no probe was baked.
    let baseline = indirect_specular + sky_reflection;
    // The probe's sharper answer, where it has coverage, occluded by what this
    // spot can see.
    // NORMALISED TO THE LIGHT AT THIS PIXEL. See `PROBE_NORMALISATION`.
    //
    // A probe is one photograph for a whole room, so it reflects the room's
    // lamp pools and its bright doorway at the same strength into a corner the
    // lightmap says is nearly dark. Scaled by how much ambient light this
    // pixel actually receives against the average the probe saw, a dark corner
    // reflects darkly -- the soft glowing patches on unlit walls (headset,
    // 2026-09-17). Only ever DARKENS: a reflection is never brightened past
    // its photograph.
    let ambient_here = dot(env + diffuse, vec3<f32>(0.2126, 0.7152, 0.0722));
    let probe_scale = select(
        1.0,
        clamp(ambient_here / max(probe_brightness, 1e-4), PROBE_NORMALISATION_FLOOR, 1.0),
        PROBE_NORMALISATION && probe_brightness > 0.0,
    );
    let sharp = mix(baseline, probe.rgb * spec_occ * probe_scale, clamp(probe.a, 0.0, 1.0));
    // Marble here is 0.048 and takes the sharp answer; brick is near 1 and
    // falls back to the baseline, because at that roughness the probe's extra
    // directional detail is not information, it is the artefact.
    let lobe_is_hemispherical = smoothstep(0.25, 0.75, r);
    let environment = mix(sharp, baseline, lobe_is_hemispherical);
    // Recorded for the source diagnostic. See `BRUSH_SOURCE_DEBUG`. Set here
    // rather than straight off the probe sample, so the diagnostic shows what
    // the probe CONTRIBUTES after occlusion and the roughness blend, not what
    // it would have contributed if both were absent.
    // WEIGHTED AS THE PICTURE WEIGHTS IT -- times Fresnel and the
    // normalisation. The raw probe radiance made the sources view paint walls
    // blue that the probe barely touches: Fresnel is about 0.04 on a rough wall.
    dbg_probe = probe.rgb * clamp(probe.a, 0.0, 1.0) * spec_occ * probe_scale * (1.0 - lobe_is_hemispherical) * fresnel;
    dbg_probe_factors = vec3<f32>(
        fresnel / max(f_max, 1e-4),
        clamp(probe.a, 0.0, 1.0),
        clamp(spec_occ * probe_scale, 0.0, 1.0),
    );
    // No second `(1 - r)`: the Fresnel above is already capped by roughness,
    // and applying it twice was most of why rough stone read as polished.
    specular = specular + environment * fresnel;

    // BOUNCED LIGHT, SHAPED BY WHERE IT CAME FROM.
    //
    // Added here rather than by the caller because this is the only place that
    // knows the surface's roughness lobe, and the same direction has to drive
    // both halves: how the bounce falls across the normal, and the glossy
    // reflection of the room it implies.
    //
    // A flat irradiance added to every texel is identical on a surface facing
    // the lit wall and one facing away, so normal maps go dead wherever direct
    // light does not reach and corners lose their shape. The direction is what
    // gives bounced light somewhere to come FROM.
    let bd = env_dir.xyz * 2.0 - 1.0;
    let bd_len = length(bd);
    let directionality = clamp(env_dir.w, 0.0, MAX_BOUNCE_DIRECTIONALITY);
    var bounce = env;
    if (bd_len > MIN_BOUNCE_DIR_LENGTH) {{
        let bounce_dir = bd / bd_len;
        // Energy-preserving by construction: the shaping factor averages to
        // exactly 1 over the sphere, so this redistributes the bounce across
        // normals without inventing or destroying any. It ranges over
        // [1 - d, 1 + d] and cannot go negative.
        //
        // RELATIVE TO THE GEOMETRIC NORMAL. The atlas texel is the irradiance
        // at the face's own normal already; what varies within the texel is
        // the normal MAP, and that is all this may redistribute. The old form,
        // `(1 - d) + d * (dot(n, dir) + 1)`, averaged to one over the sphere
        // of normals but gave a flat face turned toward its light `1 + d`:
        // with lamps baked (2026-09-26) every spot pool on the marble came
        // out up to 50% brighter than the same lamp shaded live. A face with
        // no normal map now takes exactly its texel.
        let shaped = 1.0 + directionality * (dot(n, bounce_dir) - dot(geom_n, bounce_dir));
        bounce = env * max(shaped, 0.0);
        // The room, through the same lobe the lamps use. This is what puts a
        // highlight on a polished floor in a room with no lamp in view -- the
        // floor showing WHERE the light is, not merely how much of it there is.
        //
        // THROUGH A LOBE AS WIDE AS THE BAKED LIGHT'S SPREAD, not the
        // surface's own. The direction is an AVERAGE, and `directionality` is
        // how coherent it is (the mean resultant length, at most
        // `MAX_BOUNCE_DIRECTIONALITY`). Light arriving from a cone that wide,
        // through marble's near-mirror lobe, is a broad sheen -- not a point.
        // Using marble's shininess put a sharp highlight on a direction that
        // only changes every lightmap texel, and it came out as a square halo
        // with an aliased edge (headset, 2026-09-25). The sharp reflection of
        // a bright pool is the PROBE's to give, and the probe has the shape.
        //
        // Both as von Mises-Fisher lobes: the spread's concentration from its
        // mean resultant length (Banerjee et al.), the lobe's from the
        // Blinn-Phong exponent (about s / 4 in light-direction space), and the
        // two convolved by `k1 k2 / (k1 + k2)`. Renormalised with the
        // Blinn-Phong factor so a wider lobe spreads the energy rather than
        // adding to it.
        let spread = directionality * (3.0 - directionality * directionality)
            / max(1.0 - directionality * directionality, 1e-3);
        let lobe = shininess * 0.25;
        let combined = max(4.0 * lobe * spread / max(lobe + spread, 1e-4), 1.0);
        let h = normalize(bounce_dir + view_dir);
        let gloss = pow(max(dot(n, h), 0.0), combined) * spec_strength * directionality
            * (combined + 2.0) / (shininess + 2.0);
        specular = specular + env * gloss;
    }}
    diffuse = diffuse + bounce;
    // BAKED: the lightmap's bounce, before any runtime light is added.
    // Weighted as the return line weights diffuse light.
    dbg_baked = bounce * albedo * (1.0 - fresnel);
    for (var i: u32 = 0u; i < live_light_count(); i = i + 1u) {{
        // A LAMP THAT CANNOT REACH THIS PIXEL IS SKIPPED BEFORE ANY OF ITS
        // MATHS: past its range, where the window is exactly zero, or a
        // stationary lamp its baked mask says is hidden from here -- behind a
        // wall, in another room -- where the visibility is exactly zero. Both
        // multiply the whole contribution, so skipping changes no pixel. In a
        // level of several rooms it is the common case: most lamps are behind
        // a wall from most pixels, and each one skipped is a light's worth of
        // lighting the fill-bound frame no longer pays for.
        let marker = lights.lights[i].position.w;
        let seen = stationary_visibility_of(marker);
        if (light_culling()) {{
            if (seen <= 0.0) {{
                continue;
            }}
            if (lights.lights[i].params.z < 1.5) {{
                let to_light = lights.lights[i].position.xyz - world_pos;
                let reach = lights.lights[i].params.x;
                if (dot(to_light, to_light) >= reach * reach) {{
                    continue;
                }}
            }}
        }}
        let l = lights.lights[i];
        let c = light_contribution_split(l, world_pos, n, view_dir, shininess, spec_strength);
        // NOTHING ARRIVES, SO THERE IS NOTHING TO SHADOW.
        //
        // Outside a spot's cone, past its range, or facing away from it, the
        // contribution is exactly zero -- and the shadow test below is a 3x3
        // kernel, nine compare samples per light per fragment, spent multiplying
        // that zero. With four shadowed spots that was up to thirty-six samples
        // on every lit pixel of the level, most of them outside every cone. The
        // headset frame is fill-bound, so per-pixel work is the cost that counts.
        //
        // Lossless: skipping a light whose contribution is zero changes no
        // pixel. `textureSampleCompareLevel` takes an explicit level, so
        // sampling in non-uniform control flow is well defined.
        if (max(max(c.diffuse.r + c.specular.r, c.diffuse.g + c.specular.g), c.diffuse.b + c.specular.b) <= 0.0) {{
            continue;
        }}
        // ONE shadow factor for both halves: a surface in shadow receives no
        // light at all, and a highlight that survives its own shadow is the
        // classic tell of a renderer that shadows only the diffuse term.
        var shadow = seen;
        if (l.params.z > 1.5) {{
            shadow = sun_visibility(l, world_pos);
        }}
        let layer = i32(l.params.w);
        if (layer >= 0 && f32(layer) < camera.shadow_params.y) {{
            shadow = shadow * pcf_layer(spot_shadow_tex, layer, world_pos, camera.spot_view_proj[layer]);
        }}
        diffuse = diffuse + c.diffuse * shadow;
        specular = specular + c.specular * shadow;
        // DIRECT: runtime lights, after their shadow test.
        dbg_direct = dbg_direct + (c.diffuse * albedo * (1.0 - fresnel) + c.specular) * shadow;
    }}
    // ENERGY CONSERVATION. The albedo lands here, on the diffuse half only.
    //
    // `1 - fresnel` is the light that was NOT reflected off the surface, and so
    // is the only light available to enter it, scatter, and come back out as
    // diffuse. Without it the specular was added ON TOP of a full-strength
    // diffuse and the surface emitted more than arrived -- measured at 1.62x
    // for marble and 1.37x for rock at a grazing angle, which is every wall in
    // a room seen from anywhere but straight on. That surplus is what read as
    // "too reflective", and on the rougher materials as "wet" or "metallic":
    // a strong specular over a full diffuse is exactly how a coated surface
    // looks.
    //
    // Head-on this changes almost nothing -- a dielectric reflects 4% there, so
    // the diffuse keeps 96% of what it always had.
    return diffuse * albedo * (1.0 - fresnel) + specular;
}}

fn shade(world_pos: vec3<f32>, n: vec3<f32>) -> vec3<f32> {{
    return shade_with_sky(world_pos, n, 1.0);
}}

/// `shade`, with the ambient term scaled by baked sky visibility.
///
/// 1.0 is "sees the whole sky" and reproduces `shade` exactly, which is why
/// every caller that has no occlusion data can keep calling `shade` and get
/// what it always got.
fn shade_with_sky(world_pos: vec3<f32>, n: vec3<f32>, sky_vis: f32) -> vec3<f32> {{
    let view_dir = normalize(cam_pos() - world_pos);
    let flash_idx = u32(camera.shadow_params.z);
    var lit = sky_irradiance(n) * clamp(sky_vis, 0.0, 1.0);
    for (var i: u32 = 0u; i < live_light_count(); i = i + 1u) {{
        let l = lights.lights[i];
        var c = light_contribution(l, world_pos, n, view_dir);
        // As in `shade_material_env`: no light arriving, no shadow test.
        if (max(max(c.r, c.g), c.b) <= 0.0) {{
            continue;
        }}
        // Only the sun casts the orthographic map, and only the flashlight the
        // perspective one. Every other light is unshadowed, which is the whole
        // reason a scene may have eight of them.
        if (l.params.z > 1.5) {{
            c = c * sun_visibility(l, world_pos);
        }}
        c = c * stationary_visibility(l);
        // params.w is this light's own shadow layer, or -1 when it did not get
        // one. Asking the LIGHT beats the old "is this the flashlight index"
        // test, which by construction could only ever be true for one lamp.
        let layer = i32(l.params.w);
        if (layer >= 0 && f32(layer) < camera.shadow_params.y) {{
            c = c * pcf_layer(spot_shadow_tex, layer, world_pos, camera.spot_view_proj[layer]);
        }}
        lit = lit + c;
    }}
    return lit;
}}
"#
    )
}

#[cfg(test)]
mod shadow_slot_tests {
    use super::*;

    const MAX: usize = 4;

    /// THE SYMPTOM, REPRODUCED: rank afresh every frame and the set churns.
    ///
    /// Six spots, and the player walks past them so the scores shuffle. Without
    /// hysteresis the held set changes repeatedly, which on the headset is a
    /// shadow appearing and disappearing as you move (2026-09-18).
    #[test]
    fn re_ranking_every_frame_churns_and_hysteresis_does_not() {
        // SIX LAMPS AT NEARLY THE SAME DISTANCE, AND A HEAD THAT NEVER SITS
        // STILL. That is the case that churns, and it is the ordinary one in a
        // room: the lamps differ by centimetres and head tracking moves the
        // viewer by more than that every frame. Walking a straight line past
        // lamps metres apart does NOT reproduce it -- the first version of this
        // test did that, the best four changed twice in sixty frames, and it
        // measured nothing.
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

    /// A FREE SLOT IS FILLED AT ONCE. Hysteresis is about not churning, not
    /// about being slow -- a room with slots to spare must never withhold a
    /// shadow from a light that can have one.
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

    /// A light that leaves the scene frees its slot rather than holding it.
    #[test]
    fn a_light_that_is_gone_does_not_keep_its_slot() {
        let got = spot_shadow_slots(&[(5, 0.4)], &[0, 1, 2, 3], MAX, SHADOW_SLOT_MARGIN);
        assert_eq!(got, vec![5], "a departed incumbent still holds a slot: {got:?}");
    }
}

#[cfg(test)]
mod budget_tests {
    use super::*;
    use crate::renderer::Color3;

    fn light(kind: LightKind, pos: Vec3, intensity: f32) -> Light {
        Light {
            mask_channel: None,
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
    fn a_scene_inside_the_budget_is_passed_through_untouched() {
        // Reordering a level that already fits would change its look for no
        // reason, and could reshuffle between frames.
        let ls = vec![point(50.0, 1.0), point(1.0, 9.0), point(10.0, 3.0)];
        let ranked = rank_for_budget(&ls, MAX_LIGHTS);
        let positions: Vec<f32> = ranked.iter().map(|l| l.position.x).collect();
        assert_eq!(positions, vec![50.0, 1.0, 10.0]);
    }

    #[test]
    fn the_lamp_in_your_face_survives_and_the_distant_one_does_not() {
        // The bug this fixes. Filling the budget in scene order dropped
        // whichever light was authored last, however close it was.
        let mut ls: Vec<Light> = (0..MAX_LIGHTS).map(|i| point(80.0 + i as f32, 5.0)).collect();
        ls.push(point(2.0, 5.0)); // authored last, two metres away
        let ranked = rank_for_budget(&ls, MAX_LIGHTS);
        assert_eq!(ranked.len(), MAX_LIGHTS);
        assert!(
            ranked.iter().any(|l| l.position.x == 2.0),
            "the nearest light must not be the one dropped",
        );
    }

    #[test]
    fn the_sun_is_never_dropped() {
        // A directional light has no position to be far from, and losing it
        // changes every surface in the level at once.
        let mut ls: Vec<Light> = (0..MAX_LIGHTS).map(|_| point(0.5, 100.0)).collect();
        ls.push(light(LightKind::Directional, Vec3::ZERO, 0.1));
        let ranked = rank_for_budget(&ls, MAX_LIGHTS);
        assert!(ranked.iter().any(|l| l.kind == LightKind::Directional));
    }

    #[test]
    fn brightness_counts_as_well_as_distance() {
        // Two lights the same distance away are not equally important.
        let ls = vec![point(5.0, 0.1), point(5.0, 50.0), point(5.0, 1.0)];
        let ranked = rank_for_budget(&ls, 1);
        assert_eq!(ranked[0].intensity, 50.0);
    }

    #[test]
    fn ranking_is_stable_for_equal_lights() {
        // Identical lights must not swap places between frames: a light
        // flickering in and out of the budget is worse than a dim one staying.
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
    //! A surface may not emit more light than reaches it.
    //!
    //! `shade_material_env` is a WGSL string and cannot be called, so its
    //! weights are re-implemented here and the source pins below check the
    //! shader still spells them that way -- the same pairing used for the SSR
    //! blend. What this measures is the split of ONE unit of arriving
    //! environment light between the specular that bounces off the surface and
    //! the diffuse that enters it.
    use super::*;

    /// The shader's Fresnel and its two consumers.
    fn split(roughness: f32, view_deg: f32) -> (f32, f32) {
        let r = roughness.clamp(0.04, 1.0);
        let cos_v = view_deg.to_radians().cos().clamp(0.0, 1.0);
        let f0 = 0.04f32;
        let f_max = (1.0 - r).max(f0);
        let fresnel = f0 + (f_max - f0) * (1.0 - cos_v).powi(5);
        // specular, diffuse -- with a white environment and a white albedo, so
        // the pair is the whole energy budget.
        (fresnel, 1.0 - fresnel)
    }

    /// THE regression. Every material, every viewing angle, at most 1.
    ///
    /// Measured before the fix: 1.62x for marble and 1.37x for rock at 85
    /// degrees off normal, because the specular was added on top of a diffuse
    /// that had given up nothing. That surplus is what read as "too
    /// reflective", and on rough materials as wet or metallic.
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

    /// A dielectric reflects about 4% head-on, so the diffuse keeps 96%.
    /// Without this the fix would be a licence to darken everything.
    #[test]
    fn looking_straight_at_a_surface_barely_changes_it() {
        for r in [0.048f32, 0.437, 0.9] {
            let (spec, diff) = split(r, 0.0);
            assert!((spec - 0.04).abs() < 1e-3, "head-on reflectance is {spec}, not 4%");
            assert!(diff > 0.95, "head-on diffuse dropped to {diff}");
        }
    }

    /// Rough surfaces must not develop a mirror rim.
    ///
    /// The uncapped Schlick sweeps to 1.0 at grazing whatever the material, so
    /// rubble reflected as hard as polished marble. Capping at `1 - roughness`
    /// is what separates them.
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

    /// The shader spells it the way the split above assumes.
    #[test]
    fn the_shader_conserves_energy_and_caps_the_fresnel() {
        // Comments stripped: they quote the old forms to explain them.
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
            code.contains("return diffuse * albedo * (1.0 - fresnel) + specular;"),
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
    //! What a reflective surface sees when it looks DOWN.
    //!
    //! The shader is a WGSL string, so the blend is re-implemented here and the
    //! source pin keeps the two in step -- the pairing used throughout this
    //! module.
    use super::*;

    /// The shader's ground albedo, mirrored. Pinned below.
    ///
    /// The ground RADIANCE itself now arrives in `probe_params.yzw`, computed
    /// once on the CPU -- see `uniforms::sky_ground_irradiance` and the test
    /// that holds it to the shader's own evaluation.
    const GROUND_ALBEDO_F: f32 = 0.28;

    /// `environment_radiance`, in Rust, for a uniform sky of radiance `sky`.
    ///
    /// A flat sky makes `sky_irradiance` return the same value in every
    /// direction, which isolates the horizon blend from the harmonics.
    fn env(sky_up: f32, sky_at_dir: f32, dir_y: f32) -> f32 {
        let ground = sky_up * GROUND_ALBEDO_F;
        let t = ((dir_y + 0.15) / 0.30).clamp(0.0, 1.0);
        let smooth = t * t * (3.0 - 2.0 * t);
        ground * (1.0 - smooth) + sky_at_dir * smooth
    }

    /// THE regression. A real sky panorama has almost nothing below the
    /// horizon, so a downward reflection used to come back black -- and once
    /// the diffuse started paying for the specular, that blackness became a
    /// dark band sliding along every outdoor wall as the viewer moved.
    #[test]
    fn looking_down_does_not_reflect_blackness() {
        // Bright sky above, nothing below: the shape of a real panorama.
        let got = env(1.0, 0.0, -0.9);
        assert!(
            got > 0.2,
            "a downward reflection came back at {got} under a bright sky; that \
             is the dark band on the wall",
        );
    }

    /// And looking UP is still the sky itself, untouched.
    #[test]
    fn looking_up_is_still_the_sky() {
        assert!((env(1.0, 1.0, 0.9) - 1.0).abs() < 1e-5);
        assert!((env(1.0, 0.4, 0.9) - 0.4).abs() < 1e-5, "the sky was contaminated by the ground");
    }

    /// The horizon is a blend, not a step. A hard switch draws a visible line
    /// across every reflective surface at eye level.
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

    /// A dark sky still gives a dark ground -- the ground is lit BY the sky and
    /// must not invent light of its own.
    #[test]
    fn a_dark_sky_has_a_dark_ground() {
        assert!(env(0.0, 0.0, -1.0) < 1e-6);
    }

    /// Sampling in non-uniform control flow must use an EXPLICIT level.
    ///
    /// `probe_environment` searches for the covering probe with a loop that
    /// `continue`s, `break`s and returns early, so neighbouring fragments do
    /// not reach the sample together. WGSL only defines implicit derivatives in
    /// uniform control flow; on Adreno the plain `textureSample` there did not
    /// fail validation, it hung the GPU, and Android reported the app as not
    /// responding every few seconds.
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
            code.contains("let sky_reflection = environment_radiance(refl) * occ;"),
            "the specular environment is back to reading the raw sky harmonics",
        );
        // And the DIFFUSE ambient must not: it is evaluated along the surface
        // normal, and a ground term there would light undersides that see no
        // ground.
        assert!(
            code.contains("var diffuse = sky_irradiance(n) * occ;"),
            "the diffuse ambient is no longer the plain sky along the normal",
        );
        // The BLEND SHAPE, not just that a blend exists. `the_horizon_is_soft`
        // exercises the Rust mirror, so on its own it cannot notice the shader
        // switching to a hard step -- verified by making that change and
        // watching every test stay green.
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
    //! The shader and the baker must evaluate the same sky.
    //!
    //! The renderer lights a surface with the WGSL `sky_irradiance`; the baker
    //! predicts that value with `SkyIrradiance::evaluate` from the shared
    //! `space_soup_sky` crate, so a baked probe shows what the surface will
    //! actually look like. Nothing else in either crate can notice if those two
    //! drift -- and when they did, a reflected ceiling came out brighter than
    //! the ceiling.
    //!
    //! This test is the only place the two meet.
    use super::*;
    use space_soup_sky::SkyIrradiance;

    /// The WGSL `sky_irradiance`, transcribed. Kept beside the string it
    /// mirrors so a change to one is visibly a change to the other.
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
        // CLAMPED, matching the shader's own max against zero. A truncated harmonic
        // series rings below zero, and without this the sky would SUBTRACT
        // light from surfaces facing the ring. Omitting it here was the first
        // thing this test caught -- of its own transcription, which is the
        // point: a mirror that is not faithful proves nothing.
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

    /// THE guard: same coefficients, same direction, same answer.
    #[test]
    fn the_baker_and_the_shader_agree_in_every_direction() {
        let irr = asymmetric_sky();
        // UNIT vectors, exactly. `SkyIrradiance::evaluate` normalises its input
        // defensively; the shader does not, because every caller hands it a
        // normal that is already unit length. Feeding this test a direction of
        // length 0.99934 measures that difference rather than the harmonics --
        // which is what it did first time round, disagreeing by 4e-4 and
        // looking like a real drift.
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

    /// The transcription above must actually be the shader's.
    ///
    /// Without this the mirror is just a second implementation that agrees with
    /// the baker while the SHADER quietly differs -- verified by perturbing a
    /// basis coefficient in the WGSL and watching every test here stay green.
    #[test]
    fn the_transcription_matches_the_shader_text() {
        let src = wgsl_lights_block(0, 1);
        let code: String = src
            .lines()
            .filter(|l| !l.trim_start().starts_with("//"))
            .collect::<Vec<_>>()
            .join("\n");
        // The nine basis terms, exactly as the mirror spells them.
        for term in [
            "0.282095,",
            "0.488603 * y,",
            "0.488603 * z,",
            "0.488603 * x,",
            "1.092548 * x * y,",
            "1.092548 * y * z,",
            "0.315392 * (3.0 * z * z - 1.0),",
            "1.092548 * x * z,",
            "0.546274 * (x * x - y * y),",
        ] {
            assert!(
                code.contains(term),
                "the shader's harmonic basis no longer contains `{term}`, so the \
                 transcription in this module is measuring something else",
            );
        }
        // And the cosine-lobe weights, already divided by pi.
        assert!(code.contains("0.6666667, 0.6666667, 0.6666667,"));
        assert!(code.contains("0.25, 0.25, 0.25, 0.25, 0.25,"));
        assert!(code.contains("return max(e, vec3<f32>(0.0));"), "the shader stopped clamping");
    }

    /// And the physical sanity check the coefficients exist for: a sphere of
    /// uniform radiance L must light every normal to exactly L.
    ///
    /// This is what catches the classic double-normalisation -- the published
    /// constants already fold in the basis, so using both makes a flat 0.5 sky
    /// come back as 0.141.
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

/// How a fragment picks its probe when a room has many.
#[cfg(test)]
mod probe_selection_tests {
    use super::*;

    fn shader() -> String {
        wgsl_lights_block(0, 1)
    }

    /// SELECTION IS BY CAPTURE POINT, not by which box is smallest.
    ///
    /// Every cell of a room carries the ROOM's box, because that box is what
    /// reflected rays are projected onto and a cell-sized box projects them
    /// onto geometry that is not there. Containment therefore cannot separate
    /// two cells of the same room -- their boxes are identical -- so the
    /// nearest photograph wins instead.
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

    /// And a tighter volume still beats a looser one, so a room keeps winning
    /// over the outdoor volume that encloses it.
    #[test]
    fn a_tighter_volume_still_wins_over_one_that_encloses_it() {
        let code = shader();
        assert!(
            code.contains("let tighter = volume < best_volume * 0.999;"),
            "nesting no longer decides between volumes of different size; a room \
             inside the outdoor volume could reflect the sky",
        );
    }

    /// The box a ray is projected onto is read from the uniform, not derived
    /// from the cell -- the two are deliberately different things now.
    #[test]
    fn projection_uses_the_box_the_bake_supplied() {
        let code = shader();
        assert!(code.contains("let lo = camera.probe_boxes[slot * 3 + 1].xyz;"));
        assert!(code.contains("let hi = camera.probe_boxes[slot * 3 + 2].xyz;"));
    }
}

/// Blending between probe cells, which is what stops their boundaries showing.
#[cfg(test)]
mod probe_blend_tests {
    use super::*;

    fn shader() -> String {
        wgsl_lights_block(0, 1)
    }

    /// TWO PHOTOGRAPHS AT THE BOUNDARY, NOT ONE.
    ///
    /// Choosing exactly one cell per fragment makes every cell boundary a
    /// seam. Cells are axis-aligned boxes, so the seams are straight and meet
    /// at right angles -- light squares on a floor or wall -- and they move as
    /// residency changes, so they appear and disappear rather than sitting
    /// still. The blend now happens only in a band around the boundary (see
    /// `PROBE_BLEND_BAND` and the band tests below), but it must still happen.
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

    /// A room's photograph must never be blended with the outdoor one -- that
    /// would put sky on an interior wall. Only cells of the SAME ROOM mix, and
    /// "same room" is the volume id the bake wrote, not equal box size: two
    /// different rooms of one size used to count as one.
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
        // And a TIGHTER room drops the old runner-up, which belonged to the
        // room it just beat -- it used to survive and blend across rooms.
        let tighter = code.find("if (tighter) {").expect("the tighter branch is gone");
        let reset = code[tighter..].find("second = -1;").expect("tighter no longer resets the runner-up");
        assert!(reset < 400, "the runner-up reset is not in the tighter branch");
    }

    /// One probe in range is still valid and must not read the runner-up slot.
    #[test]
    fn a_single_probe_returns_without_a_second_sample() {
        let code = shader();
        assert!(
            code.contains("if (second >= 0) {") && code.contains("own = near;"),
            "with no runner-up the shader would index probe slot -1",
        );
    }

    /// NOTHING CONTAINING A SURFACE IS NOT THE END: a doorway's jambs lie in no
    /// room, and the portal pass is what gives them a photograph. The old early
    /// `return vec4(0.0)` skipped it.
    #[test]
    fn every_path_goes_through_the_portals() {
        let code = shader();
        // Up to `probe_trace`, whose own early returns are its caller's "no hit".
        let env = &code[code.find("fn probe_environment(").unwrap()..code.find("fn probe_trace(").unwrap()];
        assert!(!env.contains("return vec4<f32>(0.0);"), "an early return skips the doorways again");
        assert!(env.contains("return probe_through_portals(own, own_room, select_world, world_pos, d, probe_lod);"));
    }

    /// The doorway blend's reach is generated from the Rust constants.
    #[test]
    fn the_portal_fades_come_from_the_constants() {
        let code = shader();
        assert!(code.contains(&format!("const PORTAL_FADE: f32 = {:?};", PROBE_PORTAL_FADE)));
        assert!(code.contains(&format!("const PORTAL_SIDE_FADE: f32 = {:?};", PROBE_PORTAL_SIDE_FADE)));
    }

    /// The CPU twin of the doorway weight: how much of the HIGH side's room a
    /// point takes, `t` along the opening's axis, `side` metres past its edges.
    fn portal_high_weight(t: f32, lo: f32, hi: f32, side: f32) -> f32 {
        let smooth = |e0: f32, e1: f32, x: f32| {
            let u = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
            u * u * (3.0 - 2.0 * u)
        };
        let reach = 1.0 - smooth(0.0, PROBE_PORTAL_SIDE_FADE, side);
        reach * smooth(lo - PROBE_PORTAL_FADE, hi + PROBE_PORTAL_FADE, t)
    }

    /// Through a door the weight runs from one room to the other with no step:
    /// nothing at the far end of the reach on either side, an even share in
    /// the middle of the wall, and small increments between.
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
        // And past the opening's edge it fades to nothing, so the wall around
        // the frame does not show the door's outline.
        assert_eq!(portal_high_weight((lo + hi) * 0.5, lo, hi, PROBE_PORTAL_SIDE_FADE), 0.0);
    }

    /// The CPU twin of the probe blend weight: how much of the NEAREST probe a
    /// fragment takes, from the squared distances to the two capture points.
    fn probe_near_weight(best_dist_sq: f32, second_dist_sq: f32, band: f32) -> f32 {
        let gap = second_dist_sq.sqrt() - best_dist_sq.sqrt();
        let t = (gap / band).clamp(0.0, 1.0);
        0.5 + 0.5 * (t * t * (3.0 - 2.0 * t))
    }

    /// One reflection, not two, anywhere clear of a cell boundary.
    ///
    /// 1 m from one capture point and 2 m from the next. The inverse-distance
    /// weighting this replaced gave the nearer probe 0.8 of the result here and
    /// the other 0.2 -- a hanging lamp seen twice, the second copy at a fifth.
    #[test]
    fn a_fragment_clear_of_the_boundary_takes_one_probe_alone() {
        let w = probe_near_weight(1.0, 4.0, 0.6);
        assert!((w - 1.0).abs() < 1e-6, "the far probe still contributes {} here", 1.0 - w);
    }

    #[test]
    fn on_the_boundary_the_two_probes_share_equally() {
        assert!((probe_near_weight(2.25, 2.25, 0.6) - 0.5).abs() < 1e-6);
        // And approach it continuously from inside the band.
        let just_inside = probe_near_weight(1.0, 1.1f32.powi(2), 0.6);
        assert!(just_inside > 0.5 && just_inside < 0.75, "weight {just_inside} jumps near the boundary");
    }

    /// The shader must be doing what the twin above describes.
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

    /// EACH BLENDED PHOTOGRAPH IS READ FROM ITS OWN CAPTURE POINT.
    ///
    /// Reusing the near probe's corrected direction for the far one made both
    /// halves of the blend flip direction on the boundary, where "near" flips:
    /// a hard rectangle across the hall at every handover plane, whatever the
    /// band (offline_frame, 2026-09-23).
    #[test]
    fn the_far_probe_gets_its_own_parallax() {
        let code = wgsl_lights_block(0, 1);
        assert!(code.contains("let sample_dir = probe_parallax_direction(world_pos, d, best);"));
        // Its own parallax, or -- where the trace hit -- the same point seen
        // from its own capture point. Either way, never the near one's.
        assert!(code.contains("let far_dir = probe_parallax_direction(world_pos, d, second);"));
        assert!(code.contains("h - camera.probe_boxes[s1 * 3].xyz, i32(camera.probe_boxes[s1 * 3].w),"));
        assert!(code.contains("probe_hit_lod(roughness, t, sqrt(d1))"), "the far photograph's blur is not its own");
        assert!(
            code.contains("probe_cube, probe_samp, far_dir, i32(camera.probe_boxes[second * 3].w), probe_lod"),
            "the far photograph is read along the near one's direction again",
        );
    }
}

/// The probe's honesty about its own resolution.
#[cfg(test)]
mod probe_sharpness_tests {
    use super::*;

    /// A PROBE MAY NEVER BE SAMPLED AS A MIRROR.
    ///
    /// Level 0 of a 64px face is about 1.4 degrees a texel. Reflecting that
    /// sharply put crisp images of light fixtures on glossy walls, which is
    /// what the source diagnostic caught as blue blobs in otherwise green
    /// rooms -- light with no visible source.
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

    /// The floor still has to be below the ceiling, or roughness stops meaning
    /// anything and every surface reflects identically.
    #[test]
    fn roughness_still_has_room_to_vary() {
        let code = wgsl_lights_block(0, 1);
        assert!(code.contains("const PROBE_MAX_LOD: f32 = 7.0;"));
        // Rough stone must still land higher than polished marble.
        let lod = |r: f32| (r * 7.0).clamp(0.5, 7.0);
        assert!(
            lod(0.6) > lod(0.05) + 1.0,
            "brick and marble now sample the same level; roughness has stopped \
             changing the reflection at all",
        );
    }
}

/// Shadow tests are only paid for where light arrives.
#[cfg(test)]
mod shadow_skip_tests {
    use super::*;

    /// The guard must come BEFORE the shadow kernel it exists to skip, in both
    /// shading paths. After it, it would save nothing and every test of the
    /// picture would still pass.
    #[test]
    fn a_light_that_contributes_nothing_is_not_shadow_tested() {
        let code = wgsl_lights_block(0, 1);
        let env = code.find("fn shade_material_env(").expect("shade_material_env is gone");
        let guard = code[env..]
            .find("if (max(max(c.diffuse.r + c.specular.r, c.diffuse.g + c.specular.g), c.diffuse.b + c.specular.b) <= 0.0) {")
            .expect("the material path shadow-tests lights that contribute nothing");
        let kernel = code[env..].find("pcf_layer(spot_shadow_tex").expect("the spot shadow test is gone");
        assert!(guard < kernel, "the guard runs after the shadow kernel it should skip");

        let sky = code.find("fn shade_with_sky(").expect("shade_with_sky is gone");
        let guard = code[sky..]
            .find("if (max(max(c.r, c.g), c.b) <= 0.0) {")
            .expect("the sky path shadow-tests lights that contribute nothing");
        let kernel = code[sky..].find("pcf").expect("the sky path's shadow test is gone");
        assert!(guard < kernel, "the sky path's guard runs after its shadow kernel");
    }
}

/// Probe normalisation: the shader's rule, and a CPU twin of it.
#[cfg(test)]
mod probe_normalisation_tests {
    /// The shader's scale, transcribed. Pinned to the WGSL below.
    fn probe_scale(ambient_here: f32, probe_brightness: f32, floor: f32) -> f32 {
        if probe_brightness > 0.0 {
            (ambient_here / probe_brightness.max(1e-4)).clamp(floor, 1.0)
        } else {
            1.0
        }
    }

    #[test]
    fn a_dark_corner_reflects_darkly_and_a_lit_one_as_photographed() {
        // A probe that saw an average of 0.05.
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
            code.contains("let sharp = mix(baseline, probe.rgb * spec_occ * probe_scale, clamp(probe.a, 0.0, 1.0));"),
            "the scale is computed but not applied to the probe",
        );
        // The brightness comes from the chosen probe's slot, blended like the
        // photographs, and starts at zero (unknown) on every call.
        assert!(code.contains("probe_brightness = camera.probe_boxes[best * 3 + 1].w;"));
        assert!(code.contains("probe_brightness = 0.0;"));
    }
}

/// A surface lying ON its room's probe box must still select that box.
#[cfg(test)]
mod probe_box_margin_tests {
    /// The shader's inside test, transcribed.
    fn inside(p: [f32; 3], lo: [f32; 3], hi: [f32; 3], margin: f32) -> bool {
        (0..3).all(|i| p[i] >= lo[i] - margin && p[i] <= hi[i] + margin)
    }

    #[test]
    fn a_seam_pixel_just_past_the_wall_keeps_its_room() {
        // The hall interior, and an MSAA edge pixel shaded 1 cm past the
        // ceiling/wall corner.
        let (lo, hi) = ([-2.7, 0.0, -15.7], [2.7, 3.1, 3.7]);
        let seam = [2.71, 3.11, -5.0];
        assert!(!inside(seam, lo, hi, 0.0), "without a margin the seam falls out of its room");
        assert!(inside(seam, lo, hi, 0.05), "with the margin the seam keeps its room's probe");
        // The next room's surfaces are a wall's thickness (0.3 m) away and stay out.
        assert!(!inside([3.0, 1.0, -5.0], lo, hi, 0.05));
    }

    #[test]
    fn the_shader_tests_the_box_with_the_margin() {
        let code = super::wgsl_lights_block(0, 1);
        // The value the shader gets is the SCALED one -- see `probe_box_margin`.
        // Asserting the literal 0.05 would pass only at full resolution and
        // would have let the seam regression through unnoticed.
        assert!(
            code.contains(&format!(
                "const PROBE_BOX_MARGIN: f32 = {:?};",
                super::probe_box_margin()
            )),
            "the shader's probe margin is not the one `probe_box_margin` computes",
        );
        // And it must actually widen when the eye buffer shrinks.
        assert!(
            super::probe_box_margin() >= 0.05,
            "the margin got NARROWER than its full-resolution value",
        );
        // While staying well inside the thinnest wall, or it pulls the next
        // room's surfaces into this room's probe.
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
        // THE WHOLE FIX, stated once. Selection must read `select_world` and
        // parallax must read `world_pos`. Collapsing them back into one
        // variable is the regression, and it is invisible in a screenshot
        // until someone stands far enough away to see a room seam.
        let code = super::wgsl_lights_block(0, 1);
        let sel = code
            .find("if (any(volume_world < lo")
            .expect("the candidate loop no longer tests the selection point");
        let parallax = code
            .find("let t_hi = (hi - parallax_pos) * inv;")
            .expect("the parallax projection no longer uses the clamped position");
        // THREE DISTINCT POSITIONS, and each exists for its own reason:
        //   select_world  -- the face centre, so a whole face agrees on WHICH
        //                    probe and no seam opens between its pixels;
        //   parallax_pos  -- the fragment CLAMPED into that probe's box, so the
        //                    parallax is defined even for an MSAA edge pixel
        //                    whose interpolated position landed outside;
        //   world_pos     -- the raw fragment, which neither of the above may
        //                    be replaced by.
        // Collapsing any two of them back together is a regression, and each
        // one was a separate artifact on the headset.
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
        // And a ZERO exit distance is a hit. `> 0.0` sent every edge pixel
        // clamped onto a wall it reflects out through to the raw direction --
        // the front-of-hall seam (2026-09-23).
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
    /// The WGSL with comments stripped, so a rule about CODE is not matched
    /// against prose describing it.
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

    /// THE DOCUMENTED ADRENO CLIFF: a dynamically-indexed LOCAL array cannot be
    /// kept in registers, so the compiler spills it to scratch memory. A
    /// uniform or storage buffer indexed by a loop counter is fine and is what
    /// this shader does; a `var<function> x: array<...>` read at a runtime
    /// index is not.
    ///
    /// Asserted rather than assumed because it degrades silently: nothing
    /// errors, the shader just gets slower the day someone adds one.
    #[test]
    fn the_lights_block_declares_no_local_arrays() {
        let b = body();
        // A LOCAL array is a `var` DECLARATION. A struct FIELD of array type --
        // `view_proj: array<mat4x4<f32>, 2>` inside the camera uniform, or the
        // light array itself -- has no `var` and lives in a uniform buffer,
        // which is the safe case. Matching on `: array<` alone flagged those
        // and failed against correct code, which is how a guard gets deleted
        // instead of fixed.
        for (i, line) in b.lines().enumerate() {
            let t = line.trim();
            let local_array_decl =
                t.starts_with("var<function>") || (t.starts_with("var ") && t.contains(": array<"));
            assert!(
                !local_array_decl,
                "line {i} declares a LOCAL array, which spills to scratch memory \
                 on Adreno if it is ever indexed dynamically: {t}",
            );
        }
    }

    /// The nearest-light pass reads TWO fields; copying the whole 16-component
    /// Light to get them is register pressure for nothing, and register
    /// pressure is occupancy on a tile GPU.
    #[test]
    fn the_nearest_light_pass_does_not_copy_the_whole_light() {
        let b = body();
        let start = b.find("var light_dist = 1e9;").expect("the nearest-light pass is gone");
        let end = b[start..].find("let alpha = r * r;").expect("the pass lost its terminator") + start;
        let loop_body = &b[start..end];
        assert!(
            !loop_body.contains("let li = lights.lights[i];"),
            "the nearest-light loop copies the whole Light struct again",
        );
        assert!(
            loop_body.contains("lights.lights[i].position.xyz"),
            "the nearest-light loop no longer reads the position field directly",
        );
    }
}

#[cfg(test)]
mod fresnel_aliasing_tests {
    /// THE MEASUREMENT THIS EXISTS FOR (headset, 2026-09-22). Sampling across a
    /// room seam in the lighting-sources view: BAKED down ~31, PROBE up ~25,
    /// DIRECT flat at -0.3.
    ///
    /// That last number is the discriminator. Direct lighting uses the
    /// fragment's position and its normal, so if either were wrong at those
    /// pixels direct would move too -- it does not. What baked and probe share
    /// and direct does not touch is FRESNEL: baked is scaled by `1 - fresnel`
    /// and probe by `fresnel`, so a Fresnel spike pushes them apart by exactly
    /// the observed complementary amounts.
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

    /// The mapped normal must NOT simply be dropped. At head-on incidence
    /// Schlick returns f0 for every roughness, so a polished surface reflects
    /// more than a rough one only because its normal map tilts it off head-on.
    /// Replacing `n` with the geometric normal outright makes polished and
    /// rough identical -- which three existing tests caught, correctly.
    #[test]
    fn a_polished_surface_still_sees_its_own_normal_map() {
        // smoothstep(0.1, 0.4, r) is 0 at r <= 0.1, so a near-mirror keeps n.
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

    /// Fresnel must stay capped by roughness as well. The blend reduces the
    /// JITTER in the sweep; the cap bounds how far the sweep can go.
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
        // It travels AWAY from the sun: down, for a high one.
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

    /// THE STATIC MAP DOES NOT MOVE WITH THE PLAYER.
    ///
    /// A point on the level maps to the same shadow-map texel whichever way
    /// the player has walked or turned since the map was drawn -- which is
    /// what makes drawing it once correct. Without `player_to_world` the
    /// shadows would slide across the floor as the player moved.
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
                // And inside the map, with the margin to spare.
                assert!(now.x.abs() < 1.0 && now.y.abs() < 1.0 && now.z > 0.0 && now.z < 1.0, "{now}");
            }
        }
    }

    #[test]
    fn brushes_and_terrain_read_a_baked_sun_mask_and_everything_shades_the_sun_live() {
        let brush = crate::renderer::brush_pipeline::brush_shader_src();
        assert!(brush.contains("receiver_sun_mask = "), "brushes must read their baked mask");
        // The ground reads its own, beside its sky visibility, and falls back
        // to the static map when the bake predates it (alpha 1).
        let terrain = crate::renderer::terrain_pipeline::terrain_shader_src();
        assert!(
            terrain.contains("receiver_sun_mask = select(-1.0, smoothstep(-ground_sun_w, ground_sun_w, ground_sun_d), ground_map.a < 0.5);"),
            "terrain must read its baked sun mask, and only where one was baked",
        );
        // No surface skips the sun any more: the mask replaced the skip.
        assert!(!brush.contains("receiver_has_baked_sun") && !terrain.contains("receiver_has_baked_sun"));
    }
}

/// The glossy reflection of baked bounce is only as sharp as the bounce is.
#[cfg(test)]
mod baked_gloss_tests {
    use super::*;

    /// The CPU twin of the shader's combined exponent.
    fn combined(shininess: f32, directionality: f32) -> f32 {
        let spread = directionality * (3.0 - directionality * directionality)
            / (1.0 - directionality * directionality).max(1e-3);
        let lobe = shininess * 0.25;
        (4.0 * lobe * spread / (lobe + spread).max(1e-4)).max(1.0)
    }

    /// MARBLE UNDER HALF-COHERENT BOUNCE IS A SHEEN, NOT A POINT. With marble's
    /// own exponent the highlight rode the lightmap's texel grid as a square
    /// halo (headset, 2026-09-25). At the most coherent bounce the map allows,
    /// the lobe must be wider than a 10-exponent Blinn-Phong -- about 50
    /// degrees -- while the surface lobe alone would be a pinpoint.
    #[test]
    fn a_polished_surface_reflects_baked_bounce_as_broadly_as_it_arrives() {
        // The shader's ceiling on how coherent baked bounce may be.
        let code = wgsl_lights_block(0, 1);
        assert!(code.contains("const MAX_BOUNCE_DIRECTIONALITY: f32 = 0.5;"));
        let marble = 2.0 / (0.048f32 * 0.048) - 2.0;
        let c = combined(marble, 0.5);
        assert!(c < 10.0, "baked bounce gives marble an exponent of {c}: a highlight sharper than the light it came from");
        // And the WGSL uses the same formula.
        assert!(code.contains("let combined = max(4.0 * lobe * spread / max(lobe + spread, 1e-4), 1.0);"));
        assert!(code.contains("pow(max(dot(n, h), 0.0), combined)"));
    }

    /// A rough surface keeps its own lobe: the spread only ever narrows what
    /// the light allows, it never sharpens a matte wall.
    #[test]
    fn a_rough_surface_keeps_its_own_lobe() {
        let brick = 4.0;
        let c = combined(brick, 0.5);
        assert!(c <= brick + 1e-3, "the spread sharpened a rough lobe to {c}");
    }
}

/// Baked lamps ride behind the live ones, and lightmapped surfaces stop there.
#[cfg(test)]
mod baked_light_split_tests {
    use super::*;

    fn point(x: f32, i: f32) -> Light {
        Light {
            mask_channel: None,
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
        let baked: Vec<Light> = (0..10).map(|i| point(i as f32, 0.5)).collect();
        let out = append_baked(&live, &baked, MAX_LIGHTS);
        assert_eq!(out.len(), MAX_LIGHTS);
        assert_eq!(out[0].position, live[0].position);
        assert_eq!(out[1].position, live[1].position);
        // Live lights are never displaced by baked ones, however many there are.
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
        // A plain upload is all live.
        let gpu = pack_lights(&lights, 5, &[], false, true);
        assert_eq!(gpu.count[1], 5);
    }

    /// Every light loop honours the split, and only the surfaces that carry
    /// those lamps in a lightmap ask for it.
    #[test]
    fn lightmapped_surfaces_skip_the_baked_tail_and_nothing_else_does() {
        let code = wgsl_lights_block(0, 1);
        assert!(!code.contains("i < lights.count.x;"), "a light loop walks the baked tail");
        assert_eq!(code.matches("i < live_light_count();").count(), 4);
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

/// Baked light is redistributed by the normal map only, never by the face.
#[cfg(test)]
mod baked_shaping_tests {
    use super::*;

    /// A flat face takes exactly its texel: the shaping is one wherever the
    /// shading normal is the geometric normal, whatever the light direction.
    #[test]
    fn a_flat_face_keeps_its_texel_and_a_tilted_normal_redistributes() {
        let code = wgsl_lights_block(0, 1);
        assert!(
            code.contains("let shaped = 1.0 + directionality * (dot(n, bounce_dir) - dot(geom_n, bounce_dir));"),
            "the baked-light shaping is no longer relative to the geometric normal; a flat face \
             facing its lamp would be lit 1 + d times its own texel again",
        );
        // The CPU twin.
        let shaped = |n: Vec3, geom: Vec3, dir: Vec3, d: f32| 1.0 + d * (n.dot(dir) - geom.dot(dir));
        let up = Vec3::Y;
        assert!((shaped(up, up, up, 0.5) - 1.0).abs() < 1e-6, "a flat floor under its lamp");
        assert!((shaped(up, up, Vec3::X, 0.5) - 1.0).abs() < 1e-6, "a flat floor lit from the side");
        let tilted = Vec3::new(0.3, 0.95, 0.0).normalize();
        assert!(shaped(tilted, up, Vec3::X, 0.5) > 1.0, "a bump facing the light takes more");
        assert!(shaped(-tilted + 2.0 * up * up.dot(tilted), up, Vec3::X, 0.5) < 1.0, "a bump facing away takes less");
    }
}

/// THE REFLECTION TRACE, RUN ON THE GPU: the real WGSL `probe_trace`, called
/// from a compute shader against a synthetic hall, a hallway through a
/// doorway, a pillar and a rotated lamp box, with the hits read back.
///
/// String tests cannot say whether a slab test picks the right face or a
/// doorway hands a ray to the right room; the only honest check of shader
/// logic is running it. `probe_trace` reads nothing but the camera uniform,
/// so nothing else has to be bound.
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
        /// How much of the lobe passes through a doorway's rim, -1 for none.
        rim: f32,
        rim_went_through: bool,
        /// A solid proxy's outline within the footprint: its index or -1, how
        /// much of the footprint it covers, and whether the ray hit it.
        edge: i32,
        edge_cover: f32,
        edge_hit: bool,
    }

    /// Trace each `(origin, room, dir, roughness)` through `probe_trace`.
    fn trace(rays: &[(Vec3, f32, Vec3, f32)]) -> Option<Vec<Hit>> {
        let (device, queue) = crate::renderer::terrain_pipeline::tests::headless_gpu()?;
        let mut probes = ProbeUpload::default();
        probes.count = 4;
        probes.set(0, 0, Vec3::new(0.0, 1.55, -8.425), HALL_LO, HALL_HI);
        probes.set(1, 1, Vec3::new(0.0, 1.55, -3.575), HALL_LO, HALL_HI);
        probes.set(2, 2, Vec3::new(4.75, 1.3, -3.0), HALLWAY_LO, HALLWAY_HI);
        probes.set_volume(0, 0);
        probes.set_volume(1, 0);
        probes.set_volume(2, 1);
        // The outdoor volume: a box around everything, and no depth.
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
        };
        let lamp = ProbeProxy {
            centre: Vec3::new(0.0, 2.5, -12.0),
            half_size: Vec3::new(0.3, 0.1, 0.1),
            rotation: Quat::from_rotation_y(std::f32::consts::FRAC_PI_4),
            volume: 0,
            solid: true,
        };
        probes.set_proxies(&[pillar, lamp], Vec3::ZERO, &[0, 1]);
        let mut u: Uniforms = bytemuck::Zeroable::zeroed();
        u.probe_params = [probes.count as f32, 0.0, 0.0, 0.0];
        u.probe_boxes = probes.boxes;
        u.portal_params = [probes.portal_count as f32, 0.0, 0.0, 0.0];
        u.probe_portals = probes.portals;
        u.proxy_params = [probes.proxy_count as f32, 0.0, 0.0, 0.0];
        u.probe_proxies = probes.proxies;

        let code = format!(
            "{}\n{}",
            wgsl_lights_block(0, 1),
            r#"
@group(1) @binding(0) var<storage, read> rays: array<vec4<f32>>;
@group(1) @binding(1) var<storage, read_write> hits: array<vec4<f32>>;
@compute @workgroup_size(1)
fn trace_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let o = rays[id.x * 2u];
    let d = rays[id.x * 2u + 1u];
    let h = probe_trace(o.xyz, normalize(d.xyz), o.w, d.w);
    hits[id.x * 4u] = vec4<f32>(h.pos, select(0.0, 1.0, h.found));
    hits[id.x * 4u + 1u] = vec4<f32>(h.room, h.other, select(0.0, 1.0, h.escaped), f32(h.portal));
    hits[id.x * 4u + 2u] = vec4<f32>(h.rim, select(0.0, 1.0, h.rim_went_through), h.t, h.rim_t);
    hits[id.x * 4u + 3u] = vec4<f32>(f32(h.edge), h.edge_cover, select(0.0, 1.0, h.edge_hit), h.edge_t);
}
"#
        );
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
        // Depth: the three indoor photographs saw only sky (so they HAVE
        // depth), the outdoor one has none -- which is how the trace tells the
        // outdoor volume from a room.
        let depth_tex = device.create_texture(&crate::renderer::uniforms::probe_depth_descriptor(1, 4));
        // 0x3C00 is 1.0 as a half float: "sky", which is depth.
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
        let g0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: camera.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::TextureView(&depth_view) },
                wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::Sampler(&depth_samp) },
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
            // 0: the pillar's SIDE, which neither hall photograph ever saw.
            (floor_a, 0.0, toward(floor_a, side_face), 0.0),
            // 1: through the doorway to the far end of the hallway.
            (floor_b, 0.0, toward(floor_b, hallway_end), 0.0),
            // 2: into the opening and onto its jamb, inside the wall.
            (floor_c, 0.0, toward(floor_c, jamb), 0.0),
            // 3: the ROTATED lamp box, at a point its unrotated box misses.
            (floor_d, 0.0, Vec3::Y, 0.0),
            // 4: plain ceiling.
            (floor_e, 0.0, Vec3::new(0.0, 1.0, -0.2).normalize(), 0.0),
            // 5: too rough to trace.
            (floor_e, 0.0, Vec3::Y, 0.8),
            // 6: FROM a jamb (a surface in no room) across to the other jamb.
            (jamb, -1.0, Vec3::new(-0.05, 0.0, -1.0).normalize(), 0.0),
            // 7: from the same jamb out into the hallway, onto its side wall.
            (jamb, -1.0, Vec3::new(1.0, 0.0, -0.3).normalize(), 0.0),
            // 8: out through the FRONT door: escapes at the wall's far face.
            (floor_f, 0.0, toward(floor_f, Vec3::new(0.0, 1.5, 6.0)), 0.0),
            // 9: into the front door's opening and onto its jamb, inside the wall.
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

    /// A ROUGH REFLECTION OF A DOORWAY IS AS SOFT AS THE SURFACE.
    ///
    /// A floor at the hallway rock's roughness (0.43) reflects the hall's wall
    /// 2.5 m away through a lobe about 0.6 m across there. Aimed 10 cm beside
    /// the doorway, the ray itself meets the wall -- but a good part of its
    /// lobe passes through the opening, and the trace must say how much, and
    /// from which side, for the colour to blend. Aimed 10 cm INSIDE the
    /// opening, the ray goes through and the wall beside it takes its share.
    /// At the marble's 0.048 the lobe is under a centimetre: one ray is the
    /// whole answer, as it was, and a plain wall has no rim at all.
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

    /// THE PILLAR'S OUTLINE IN A REFLECTION IS BLENDED, NOT STEPPED.
    ///
    /// One ray per pixel decides hit or miss at the pillar's edge, so the
    /// edge crawls as the head moves -- the column rippling in the back wall's
    /// reflection (headset, 2026-09-27). Within the footprint the trace
    /// reports the outline and how much of the footprint the pillar covers:
    /// under half just outside the edge, over half just inside, and nothing at
    /// all for a ray well clear of it or a mirror whose footprint is a point.
    #[test]
    fn a_reflection_blends_across_the_pillars_outline() {
        let from = Vec3::new(-2.0, 1.0, -4.0);
        // Seen from here the pillar's left outline is its BACK-left corner,
        // x = -0.45, z = -7.45: a ray aimed beside its front corner still
        // clips its left face further on.
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
